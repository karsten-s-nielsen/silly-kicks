# HF Jobs training harness + re-fit campaign — design

**Date:** 2026-10-13
**Branch:** `feat/hf-jobs-training-harness`
**Status:** Owner-ratified 2026-10-13 (decisions below are locked, not author-proposed); revised for Reviewer A/B findings, awaiting re-review. Do not implement past the gate.

## Executive summary

The DGX is the only machine that can currently train or re-fit silly-kicks models. That is both a
throughput bottleneck (jobs serialize on one box) and a reproducibility problem (the DGX CPU is not
available to anyone else, so published results cannot be replicated on a common platform). This design
moves model training to **Hugging Face Jobs** — ephemeral, per-minute-billed batch containers that any
HF account can launch, run in parallel, and that self-clean on exit.

A spike (2026-10-13) validated the full loop end-to-end on HF: `silly-kicks==4.134.0` installs from
PyPI and imports; a real trainer (`train_win_probability`, public StatsBomb open data) runs from a
clean git checkout to retrievable weights; the `require_clean_tree` provenance gate passes on a shallow
clone; artifacts round-trip via a private HF dataset (spike; AS-BUILT moved run-artifacts to a bucket). All silly-kicks trainers are **CPU + RAM bound,
no GPU** — the DGX was never used for its GPU, only its cores and memory. A full CPU re-fit of every
trainer costs well under $50 on HF.

This cycle delivers (1) a reusable training harness under `scripts/hf_jobs/`, and (2) a full re-fit of
**9 of the 10 trainers** (gk_completion was retired upstream in 4.135.0 / ADR-121; gk_retention stays
blocked) on HF, with the new weights PR'd back into the repo. The harness is the
durable deliverable; the re-fit is the first campaign that exercises it.

## Goals

- A harness that launches any silly-kicks trainer as an ephemeral HF Job, fans out multiple trainers in
  parallel, and retrieves their artifacts — encoding the invocation details the spike discovered.
- Owner-tier corpora reach HF through a **pre-staged private HF dataset**, never by putting raw-data
  credentials on the HF runner.
- Re-fit the 9 Databricks-free trainers on HF and PR the new weights back into `silly_kicks/**/weights/`.
- Reproducibility on a common platform: pinned training environment, a no-regression-vs-committed
  quality floor (re-fit on the same full corpus), and the training environment stamped into each
  `metrics.json`.
- The DGX remains fully usable — as the materialization source and as a local pre-flight box.

## Non-goals (explicitly out of scope)

- **Databricks in any form.** Databricks is not a data source for this harness and is expected to be
  retired. The harness sources owner-tier data only through `_loader_pining` and public data through
  `_sb_open_data`/`statsbombpy`. `scripts/_loader_databricks.py` and the `calibrate_*` Databricks paths
  are untouched and never invoked.
- **Lakehouse re-materialization.** New bundled weights trigger downstream re-materialize obligations;
  those are coordinated by the owner/lakehouse session, not this cycle. This cycle only stamps what is
  owed.
- **`train_gk_retention`.** Its owner-run trainer is marts-native (Databricks gold marts), deliberately
  moved off tracking-frames; no pining data path exists today. It is flagged blocked-on-pining-source
  and excluded from the re-fit. The harness still registers it so a future pining-sourced corpus can
  drop in.
- **HF model-repo publishing (ADR-088).** Weights land in-repo via PR. The `publish_model_with_card`
  Hub flow is unchanged and not driven by this cycle unless a specific model already published to the
  Hub needs its Hub copy refreshed (flagged per-model, owner decides).
- **HF Spaces.** Spaces are persistent web apps, the wrong primitive for batch training; Jobs are used
  instead (see Compute model).

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Platform target | HF for all training; DGX retained as fallback + dev + materialization source |
| 2 | Compute primitive | Ephemeral HF **Jobs** (not Spaces), parallel fan-out bounded by account quota |
| 3 | Owner-tier data | Pre-staged **private HF dataset**; pining materialized on DGX/local, pushed once |
| 4 | Re-fit mode | Retrain / new weights, gated by each trainer's own metrics/non-vacuity gate |
| 5 | Reproducibility + quality bar | Not byte-identity (HF≠DGX). Re-fit on the SAME full owner corpus; **head-to-head in-job** the fresh fit and the currently-committed model are scored on the SAME held-out split, and the fresh primary metric must NOT regress beyond a pinned per-trainer δ — material regression BLOCKS PR-back pending owner investigation (no auto-accept) |
| 6 | Scope | Harness + re-fit 9 trainers; `gk_retention` flagged; `gk_completion` retired upstream (4.135.0, ADR-121) |
| 7 | Weights landing | Job → private dataset → launcher download → PR into `silly_kicks/**/weights/` |
| 8 | Commits | Minimal: (1) harness+docs+ADR, (2) all re-fit weights — NOT split per model |
| 9 | Lakehouse re-mat | Out of scope; owner coordinates |
| 10 | Databricks | Zero dependency anywhere in the harness |
| 11 | As-built source taxonomy (DGX-verified, reviewer B HJT-IMPL-01 — **owner sign-off**) | The 2-mode `public\|pining` sketch became 4 modes `SOURCES=(public, owner_flat, owner_pergame, owner_sb360)`: owner_pergame adds an in-job `repackage.py` (flat→per-game; the kinematics shards carry no shot labels); owner_sb360 adds an opt-in offline-manifest branch to the **shared** `scripts/_loader_pining.py` so receiver enumerates with no token/no network from the staged corpus (default off = unchanged; 306 loader/corpus tests green). Discovered-necessary + non-breaking, but EXCEEDS the approved plan — owner approves the expansion, especially the shared-loader edit, before Commit-1 |

## Architecture

### Compute model — Jobs, not Spaces

HF **Jobs** (`hf jobs run`) are ephemeral batch containers: each `--detach` invocation is independent,
runs to completion, bills per-minute, and terminates with nothing to delete. Parallelism is native —
fan out one Job per trainer. **[SUPERSEDED — AS-BUILT: one Job per committed VARIANT (15 jobs).]** HF
**Spaces** are persistent hosted web apps billed while alive and are the wrong primitive here. The only
persistent resource this design creates is a **private HF dataset repo** holding the staged corpus and
the run artifacts. **[SUPERSEDED — AS-BUILT: run-artifacts moved to a read-write HF Storage Bucket
(`/scratch`), the artifact-dataset dropped; the staged-corpus datasets remain.]**

Concurrency is bounded by the account's Jobs quota; the launcher fans out with a bounded worker pool and
tolerates partial fan-out (queue the remainder).

### Trainer compute profile

All 10 trainers are CPU + RAM bound. No GPU. The DGX was used for cores and memory, not acceleration.
Representative flavors (confirmed live, 2026-10-13):

| Flavor | vCPU / RAM | $/hr | Use |
|--------|-----------|------|-----|
| `cpu-upgrade` | 8 / 32 GB | 0.03 | small event trainers (win_prob, pass_completion, match_outcome, xsuccess) |
| `cpu-xl` | 16 / 124 GB | 1.00 | XGBoost tracking trainers (xshot, xcross) |
| `cpu-performance` | 32 / 256 GB | 1.90 | memory-bandwidth-bound RFCDE (ghost_gk ~887k rows, ghost_outfield) |

### Harness components — `scripts/hf_jobs/`

- **`registry.py`** — the single source of truth for every trainer's job config: entry script, required
  extra, data source (`public` | `pining`), flavor, timeout, invocation args, output weights path(s)
  including all variants, and a `databricks_blocked` flag for `gk_retention`. A registry gate test
  derives the population from `scripts/train_*.py` and asserts the registry covers exactly the trainable
  set (per the ADR-056 "derive the population, assert it exactly" convention), with `gk_retention`
  in an asserted-non-empty blocked set.
- **`launch.py`** — fans out one ephemeral Job per selected trainer. Encodes the invocation the spike
  proved: `hf jobs run <hf-opts> python:3.12 -- bash -lc "<cmd>"` where `<cmd>` git-clones the repo at a
  pinned tag/SHA, `uv pip install --system -e '.[train]' <trainer extras>`, runs the trainer against a
  job-local corpus, then uploads `<out>/weights` to the private dataset under a run key. **[SUPERSEDED — AS-BUILT: output goes to the mounted bucket `/scratch/...`; no in-job upload — retrieved locally via `hf cp`.]** HF options:
  `--flavor` (from registry), `--secrets HF_TOKEN=<token>`, `--timeout`; `--name` is NOT passed (it
  mis-parses in CLI v1.7.1 — the `--` end-of-options marker before `bash` is mandatory, otherwise `-lc`
  is eaten). Polls to terminal, then downloads artifacts to a local run dir. Dry-run mode prints the
  exact job commands without launching.
- **`stage.py`** — materializes owner-tier corpora on DGX/local via `_loader_pining` (reusing the
  existing `materialize_tc3_frames` driver for the 4 TC3 trainers, and the `pining_source` path for
  receiver_model), pushes the result once to the private HF dataset. Idempotent; records a
  corpus manifest for provenance.

### Data flow

- **Public trainers** (`win_probability`, `pass_completion` default, `match_outcome_dependence`,
  `xsuccess`): fetch StatsBomb open data in-job via `statsbombpy`. No staging.
- **Owner-tier trainers** (`ghost_gk`, `ghost_outfield`, `xshot_occurrence`, `xcross_attempt`,
  `receiver_model`): the job `hf download`s the pre-staged private dataset and points
  the trainer's `--data-dir`/`--providers` at the local copy. Pining credentials never reach HF.

**As-built launch contract (DGX-verified 2026-10-13, all 9 trainers).** The nine trainers' CLIs are
heterogeneous, so the launcher was built against what they actually accept, not an assumed uniform shape:

- **Module form, always.** Trainers run as `python -m scripts.train_<name>` — the file form
  `python scripts/train_<name>.py` raises `ModuleNotFoundError: No module named 'scripts'` (no
  `scripts/__init__.py`; `sys.path[0]` becomes `scripts/`).
- **Four source modes** in `registry.py` (`public` / `owner_flat` / `owner_pergame` / `owner_sb360`):
  - `owner_flat` (ghost_gk, ghost_outfield): the staged flat TC3 corpus consumed via `--data-dir <gen dir>`
    `--actions-dir` `--home-teams` (output `--output-dir`).
  - `owner_pergame` (xshot, xcross): the SAME flat corpus **repackaged in-job** (`scripts.hf_jobs.repackage`)
    to the per-game `{game}/frames.parquet`+`shots.parquet` layout `_iter_matches_from_dir` requires — the
    kinematics shards carry no shot labels, so the flat `--data-dir` yields "No usable training data".
  - `owner_sb360` (receiver): the staged SB360 cache + an **offline manifest**; the job sets
    `SILLY_KICKS_PINING_MANIFEST_DIR` so `_loader_pining` enumerates offline (zero token/network).
- **Per-variant runs.** ghost_gk (×4), ghost_outfield (×2), xshot/xcross (×2) are separate runs with pinned
  per-variant args (feature-set / grid ceiling / variant label), each to its own output dir.
- **Output layout per trainer.** Fresh weights land in `weights` (public, except match_outcome → the out dir
  itself), `ghost_gk_v1`, `{variant}` (ghost_outfield), `{model}_v1` (xshot/xcross), or `model` (receiver);
  the head-to-head's `--fresh-dir` is set accordingly, and the owner head-to-head eval always reads the flat
  TC3 corpus (`build_eval._owner_shards`). The floor gates the **default** variant; every variant is still
  produced for the PR-back (per-variant gating is a noted follow-up, not assumed).
- **One shared-library change:** `_loader_pining._list_matches` gains an offline-manifest branch (above);
  unset env = unchanged behaviour. Its own loader test covers it.

### DGX roles (retained)

1. **Materialization source** — owner-tier data already lives there; `stage.py` materializes via pining
   and pushes to the private dataset.
2. **Dev pre-flight** — smoke a trainer locally on a small `--data-dir` / `--max-matches` before
   committing the full HF retrain (Shift Left).

## Per-trainer registry (initial population)

| Trainer | Group | Source | Flavor | Weights path (incl. variants) |
|---------|-------|--------|--------|------------------------------|
| win_probability | public | SB-open | cpu-upgrade | `win_probability/weights/` |
| pass_completion | public | SB-open | cpu-upgrade | `expected_passing/weights/` |
| match_outcome_dependence | public | SB-open | cpu-upgrade | `match_outcome/weights/` |
| xsuccess | public | SB-open | cpu-upgrade | `xsuccess/weights/` |
| ghost_gk | owner | pining/TC3 | cpu-performance | `tracking/_ghost_gk_weights/{default,sweeper,position_only,sweeper_position_only}/` |
| ghost_outfield | owner | pining/TC3 | cpu-performance | `tracking/_ghost_outfield_weights/{default,position_only}/` |
| xshot_occurrence | owner | pining/TC3 | cpu-xl | `tracking/_xshot_weights/{default,position_only}/` |
| xcross_attempt | owner | pining/TC3 | cpu-xl | `tracking/_xcross_weights/{default,position_only}/` |
| receiver_model | owner | pining | cpu-xl | `tracking/_receiver_weights/default/` |
| gk_retention | BLOCKED | Databricks marts | — | not re-fit this cycle (flagged) |

Exact flavors, extras, and variant lists are confirmed and single-sourced in `registry.py` at impl; the
table is the design intent, the registry is the contract.

## Reproducibility

- **Pinned environment.** The job installs `.[train]`, which pins `scikit-learn>=1.9,<2` (marker-gated
  py>=3.11). HF runs py3.12, so it gets the correct sklearn range — this *fixes* the DGX py3.10 footgun
  where `.[train]` silently resolved an older sklearn and shipped mismatched weights. The clone is at a
  pinned tag/SHA.
- **Not byte-identity.** HF hardware differs from the DGX, so re-fits will not byte-reproduce committed
  weights; retrain regenerates weights + `SHA256SUMS` together. Byte-identity is not required.
- **No-regression-vs-committed quality floor (the guard that makes the bulk re-fit safe).** A trainer's
  own gate (ece/slope, non-vacuity, coherence) is ABSOLUTE — a fresh fit can pass it (ece 0.05 < 0.10)
  while being WORSE than the committed weights (ece 0.02), silently shipping a regression across the
  9-model bulk re-fit. So, separately from the own-gate: (a) re-fit on the SAME full owner corpus the
  committed weights used — never a subsample (the never-shrink-the-corpus rule); (b) **head-to-head**:
  in-job, score BOTH the currently-committed model (shipped in the cloned tag) AND the fresh fit on the
  SAME held-out split, computing the primary metric identically for each — needs no committed
  `metrics.json` (3 public trainers ship none) and removes the old-corpus-vs-new-corpus confound; (c) a
  regression of the fresh metric vs the committed metric beyond a pinned per-trainer δ BLOCKS PR-back and
  is handed to the owner to investigate — never auto-accepted. P1 (public) and P2 (owner-tier) each
  produce this head-to-head delta table. Scoring the committed model on a split it may have partly seen
  biases toward the committed model, making the guard STRICTER (fail-safe). **As-built:** `build_eval(name,
  corpus_dir)` assembles the real eval per trainer (owner = staged shards + the library/trainer extractor;
  public = statsbombpy via the trainer's loader) and is fail-loud (`EvalDataUnavailableError`, never
  synthetic). All 9 builders were EXECUTE-VERIFIED on the DGX (fresh venv + `~/tc3-kinematics` + statsbombpy)
  before Commit 1 — committed==fresh on real data (e.g. ghost_outfield 6.046m vs the committed 6.0m) — so
  the floor is live for every trainer, not gated to P1/P2.
- **Provenance stamp.** Each `metrics.json` already stamps `training_commit` and `run_tree_dirty`. The
  harness additionally stamps the training platform and key library versions (sklearn, xgboost, numpy,
  numba) so a weights artifact records the environment it was produced on.
- **numba note.** numba JIT output can vary by CPU SIMD across flavors; under the no-byte-identity bar
  this is immaterial. If any artifact later needs cross-run byte-identity, pin a single flavor or force
  the numpy backend (`SILLY_KICKS_GHOST_FORCE_NUMPY`).

## Artifact flow and PR-back

1. Job writes `<out>/weights/` (model + `SHA256SUMS` + `MODEL_CARD.md` + `metrics.json`) and uploads it
   to the private dataset under a run key. **[SUPERSEDED — AS-BUILT: written to the mounted bucket
   `/scratch/{run_key}/{name}/{variant}`; no in-job upload — retrieved locally via `retrieve_variant` (`hf cp`).]**
2. `launch.py` downloads each trainer's weights to a local run dir and verifies `SHA256SUMS`.
3. The new-vs-committed delta is computed from each `metrics.json`; a regression of the primary metric
   beyond the pinned per-trainer δ BLOCKS here and is handed to the owner — no auto-accept.
4. Weights that clear the floor are copied into `silly_kicks/**/weights/` per the registry path.
5. The full test suite runs (`test_bundled_*`, model-card-matches-bundle, chirality/feature-contract
   load gates) green before anything is proposed.

## Phasing (each phase ends at an owner approval gate)

- **P0 — Harness.** `scripts/hf_jobs/{registry,launch,stage}.py`, registry gate + launcher unit tests
  (mocked `hf`, dry-run), ADR, `docs/howto/hf-jobs-training.md`, C4 DSL regen. No weights change.
- **P1 — Validate on public trainers.** Re-fit the 4 public trainers on HF end-to-end on the full public
  corpus; produce the new-vs-committed delta table and apply the no-regression floor (regression beyond δ
  = investigate-block) before spending on owner data. No owner-tier staging needed.
- **P2 — Owner-tier staging + re-fit.** `stage.py` materializes the FULL owner corpus (never a subsample)
  and pushes the private dataset; re-fit the 5 owner-tier trainers (all variants) on HF; the same
  new-vs-committed floor applies.
- **P3 — PR weights back.** Copy all new weights in, suite green, stamp downstream re-mat owed, open PR.

## Commit plan (approval-gated; no commit without explicit owner yes)

- **Commit 1** — harness + tests + ADR + howto + C4 regen (end of P0, suite green).
- **Commit 2** — all re-fit weights in one coherent commit (end of P3, suite green). NOT split per
  model, consistent with existing practice.

Docs travel with their commit — no separate/early doc commit. This spec is reviewed uncommitted from the
working tree and lands with Commit 1.

### Branch and PR

One feature branch for the whole cycle — `feat/hf-jobs-training-harness` (the one-branch-per-cycle rule;
no worktrees). Both commits accumulate on it. After the Commit-1 gate the branch is PUSHED to origin
(owner-gated, with Commit 1); Commit 1 is pushed but does NOT MERGE early, and P1 cannot start until the
push lands. The re-fit jobs clone the pushed feature branch (tip = Commit 1 through P1-P2), which carries
`scripts/hf_jobs/**` + the committed weights the head-to-head scorer loads + the unchanged trainers — NOT
`v4.134.0` (it predates the harness → the in-job head-to-head would `ModuleNotFoundError`). A single PR
is opened from the branch after P3, carrying both commits.

## Testing

- Registry gate: population derived from `scripts/train_*.py`, asserted exactly; `gk_retention` in an
  asserted-non-empty blocked set.
- Launcher: unit tests with a mocked `hf` CLI asserting the exact argv (the `--` marker, no `--name`,
  `--secrets HF_TOKEN`, flavor from registry, clone-at-tag); dry-run golden.
- Staging: idempotence + corpus-manifest provenance.
- No-regression floor: a unit test over a synthetic committed-vs-new metrics pair asserts the δ gate
  blocks a regression and admits an improvement.
- Bundled-weights suite stays green after PR-back.

## Risks and open items

- **gk_retention blocked** on a non-Databricks data path (flagged; owner decides re-source separately).
- **Chirality / load gates — already cross-arch-stable; re-confirm on the first HF artifact (P1).**
  `silly_kicks/tracking/_chirality.py` keeps the behavioral fingerprint arch-tolerant (ATOL 1e-3 /
  RTOL 1e-2, tolerating ~1e-6 cross-platform float noise), and `_feature_contract.py:41-49` records a
  MEASURED cross-platform delta of max|delta| = 0.0 over all 69 bundled features (AMD64 vs aarch64, see
  `docs/research/pr5_platform_atol/`). Current bundled weights are save-aarch64 / load-x86 and pass in
  CI; HF re-fits are save-x86 / load-x86 (same-arch) — strictly easier. So this is NOT a blocker: simply
  re-confirm `verify_chirality` + `SHA256SUMS` + `_feature_contract` pass on the first HF-produced
  artifact in P1 (the measured evidence says they will).
- **receiver_model `--out`** writes `<out>/model`; the PR-back copies that into
  `tracking/_receiver_weights/default/`.
- **Account Jobs quota** caps parallelism; launcher degrades to bounded fan-out.
- **HF token scope** — needs write to the private dataset; passed via `--secrets`, never echoed.
- **Downstream re-mat** obligations created by new weights — stamped, handed to the owner.
- **Hub-published models** (e.g. ghost variants) — if their Hub copies must match the new in-repo
  weights, that is a flagged per-model follow-up via the existing `publish_model_with_card` seam.

## Docs and C4

ADR records the harness design and the Jobs-not-Spaces + no-Databricks decisions. `docs/howto/
hf-jobs-training.md` is the operator runbook (stage → launch → retrieve → PR-back, with the CLI-quirk
notes). The C4 model gains an `hf_jobs` harness container; DSL + `architecture.html` regenerated via the
pinned Graphviz `dot` pipeline.

## AS-BUILT (2026-10-14) — gold-standard storage + resume

The P1/P2 campaign required the owner's **gold standard** (DGX-style resume + storage decoupled from CPU),
which refined this design. Full rationale: the ADR-122 amendment. All changes are in the LOCAL
orchestrator `launch.py`; the in-job contract (clone ref, trainers, `headtohead`, `repackage`,
`_loader_pining`) is unchanged. Verified on real HF (public + all 3 owner source-modes + a
cancel→relaunch resume that skipped the already-written shards).

- **Storage model.** HF Jobs ephemeral disk is a fixed per-flavor cap (≤ 50 GB), not configurable — and
  `win_probability --all-competitions` (3961 matches) exhausts it. The harness mounts a private **HF
  Storage Bucket** read-write (`-v hf://buckets/<ns>/<bucket>:/scratch`); trainer output + the
  corpus-driver shards write to the bucket, not `/tmp`. The flat corpus is still `hf download`-ed to
  `/tmp` (fits).
- **Resume.** `hf jobs run --attempts N` auto-retries; the persistent bucket shard-root lets the
  corpus-driver `for_each` (ADR-052) skip shards already written and continue — the DGX OOM-continue
  behavior, on HF.
- **Per-variant jobs.** One `hf jobs run` per committed `VariantBuild` (15 jobs), each with its own
  head-to-head floor — superseding "the floor gates only the default variant" (now every variant is
  floor-gated in its own job) and bounding each job to a single fit.
- **No in-job upload.** Outputs persist in the bucket and are retrieved locally (`retrieve_variant`,
  `hf cp`); the artifact-dataset is dropped. `huggingface_hub` is installed in-job only for the corpus
  `hf download`.
- **Toolchain.** Requires **hf CLI ≥ v2.2** (for `--volume`; v1.7.1 lacks it). `--` before `bash` is still
  mandatory; `--secrets HF_TOKEN` (env form) keeps the token off the process listing; polling uses
  `inspect --format json`.
- **Corpora.** THREE staged datasets, each faithful to the committed weights' recorded provenance:
  `hf-refit-tc3` (flat 179; ghost — the committed `sportec` provider ≡ the current `idsse` relabel, built
  by `sportec_spadl`), `hf-refit-xshot` (the exact 17 `corpus_match_ids` from the committed
  `metrics.json`, gen `a34e992`), `hf-refit-sb360` (30-match cache + offline manifest). `win_probability`
  stays full `--all-competitions`.
