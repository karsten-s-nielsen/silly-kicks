# ADR-122: HF Jobs training harness + re-fit campaign

| Field | Value |
|---|---|
| **Date** | 2026-10-13 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner) |

## Context

Model training and re-fitting run only on the DGX. That is both a throughput bottleneck (jobs serialize
on one box) and a reproducibility problem: the DGX CPU is not available to anyone else, so published
results cannot be replicated on a common platform. A spike (2026-10-13) validated the full loop on
**Hugging Face Jobs** — ephemeral, per-minute-billed batch containers any HF account can launch, run in
parallel, and that self-clean on exit: `silly-kicks` installs from PyPI and imports; a real trainer
(`train_win_probability`, public StatsBomb open data) runs from a clean git checkout to retrievable
weights; the `require_clean_tree` provenance gate passes on a shallow clone; artifacts round-trip via a
private HF dataset. Every silly-kicks trainer is **CPU + RAM bound, no GPU** — the DGX was used for its
cores and memory, never its GPU. A full CPU re-fit of every trainer costs well under $50.

(ADR-121 is the upstream xT-GK v1 retirement, merged in 4.135.0; it deleted `gk_completion`, which this
cycle therefore drops. This ADR takes the next free number, 122.)

## Decision

Build a reusable harness under `scripts/hf_jobs/` and use it to re-fit **9 of the 10 trainers** on HF
(gk_retention is blocked on a non-Databricks data path; gk_completion was retired upstream), PR'ing the
new weights back into `silly_kicks/**/weights/`.

- **Ephemeral HF Jobs, not Spaces.** Jobs are batch containers that terminate on exit (nothing to
  delete, billed per-minute, natively parallel). Spaces are persistent web apps — the wrong primitive.
  The only persistent resource is a private HF dataset holding the staged corpus + run artifacts.
  **(Run-artifacts half SUPERSEDED 2026-10-14 — see the amendment: outputs now a read-write Storage
  Bucket, the artifact-dataset dropped; the staged-corpus datasets remain.)**
- **Owner-tier data via a pre-staged private HF dataset — zero pining token/network in-job.** Two staged
  shapes (DGX-verified against the real trainers): (1) the **flat TC3 corpus** (`shards/<gen>/*.parquet`
  + `_actions/` + `home_teams.json`) — consumed directly by ghost_gk/ghost_outfield, repackaged in-job to
  the per-game `{game}/frames.parquet`+`shots.parquet` layout xshot/xcross need, and read by every owner
  head-to-head eval (`build_eval._owner_shards`); (2) the **SB360 bundle** for receiver — the per-match
  pining cache + an **offline manifest** (`manifest/<provider>/matches.json`) written at stage time on the
  token-bearing host. The job sets `SILLY_KICKS_PINING_MANIFEST_DIR` so `_loader_pining` enumerates matches
  from the staged manifest, never the network — pining credentials never reach the HF runner.
- **Per-trainer launch contract, execution-verified (not assumed).** The nine trainers have heterogeneous
  CLIs, discovered only by running them on the DGX: all run as `python -m scripts.train_<name>` (the file
  form `python scripts/train_<name>.py` raises `ModuleNotFoundError: No module named 'scripts'`); the output
  flag is `--out` for the public trainers + receiver and `--output-dir` for ghost/xshot/xcross; the fresh
  weights land in a per-trainer subdir (`weights` / `` / `ghost_gk_v1` / `{variant}` / `model`); and each
  multi-variant trainer (ghost_gk ×4, ghost_outfield ×2, xshot/xcross ×2) is a separate run with pinned
  per-variant args. `registry.py` carries this as per-trainer `VariantBuild`s; `launch.py` emits it; the
  argv goldens assert the verified strings.
- **Zero Databricks.** The harness sources owner-tier data only through `_loader_pining` and public data
  through `_sb_open_data` / `statsbombpy`. `scripts/_loader_databricks.py` is never imported.
- **Retrain + a no-regression head-to-head quality floor.** HF hardware differs from the DGX, so a re-fit
  will not byte-reproduce the committed weights. Separately from each trainer's own acceptance gate, the
  harness scores BOTH the committed model (shipped in the cloned ref) and the fresh fit on the SAME
  held-out split in-job, and a regression of the primary metric beyond a pinned per-trainer δ BLOCKS
  PR-back pending owner investigation — never auto-accepted. This reads no committed `metrics.json`
  (three public trainers ship none) and removes the old-corpus-vs-new-corpus confound. The floor gates
  the **default** variant's primary metric (the shipped headline); a multi-variant trainer still produces
  every variant for the PR-back, but only the default is floor-gated (per-variant gating is a noted
  follow-up for the owner, not silently assumed).
- **Registry as the single contract.** `scripts/hf_jobs/registry.py` holds every trainer's job + floor
  config; an ADR-056 population gate derives the trainable set from `scripts/train_*.py` and asserts the
  registry covers it exactly (so the upstream `gk_completion` removal flows automatically), with
  gk_retention in an asserted-non-empty blocked set.
- **Clone the pushed Commit-1 ref.** Jobs clone the pushed feature branch (tip = the harness commit),
  which carries `scripts/hf_jobs/**` + the committed weights the scorer loads + the unchanged trainers —
  NOT `v4.134.0`, which predates the harness. The in-job stamp records platform + library versions, so a
  weights artifact records the environment it was produced on (zero trainer edits).

## Alternatives considered

| Option | Why rejected |
|---|---|
| HF Spaces for batch training | Persistent web-app primitive; billed while idle; "delete to clean up" vs Jobs' auto-terminate |
| Pining/Databricks creds as HF job secrets | Puts owner credentials on HF infra + couples the job to pining access; pre-staged dataset avoids both |
| Compare the fresh fit to the committed `metrics.json` | 3 public trainers ship none; the committed metric was computed on a possibly-different corpus (confounded) — head-to-head on one split is rigorous and uniform |
| Keep gk_completion | Retired upstream (ADR-121); the model no longer exists to re-fit |

## Consequences

- One reproducible, parallel training path any HF account can run; the DGX remains usable as the
  materialization source and a dev pre-flight box.
- New bundled weights trigger downstream lakehouse re-materialize obligations — stamped and handed to the
  owner, not run by this cycle.
- HF runs Python 3.12, so `.[train]` resolves the correct pinned `scikit-learn>=1.9,<2`; this fixes the
  DGX Python 3.10 footgun where an older sklearn was silently resolved and shipped mismatched weights.
- The trained-model load gates (`verify_chirality`, `SHA256SUMS`, `_feature_contract`) are already
  cross-arch-stable (measured max|delta| = 0.0, `docs/research/pr5_platform_atol/`); HF re-fits are
  save-x86 / load-x86 (same-arch), re-confirmed on the first HF artifact in P1.
- `_loader_pining` gains an **offline-manifest mode** (`SILLY_KICKS_PINING_MANIFEST_DIR`): `_list_matches`
  reads a staged `<dir>/<provider>/matches.json` instead of the live API when the env is set, unchanged
  otherwise. This is the one shared-library change the harness requires — it lets any pining-sourced
  trainer enumerate matches from a staged corpus with zero token/network (the receiver re-fit's enabler),
  and is covered by its own loader test.

## Amendment (2026-10-14): bucket-backed storage + resume, per-variant jobs, hf CLI ≥ v2.2

Running the P1/P2 campaign surfaced two failures the Commit-1 design did not cover, and the owner required
the **gold-standard** fix (DGX-style resume + storage decoupled from CPU), not a bandaid. All of the below
is `launch.py`-local (the local orchestrator; it never runs in-job), so the cloned ref still carries the
unchanged `headtohead`/`repackage`/trainers/weights. Verified end-to-end on HF (public + all 3 owner
source-modes + a cancel→relaunch resume).

- **Mini-wave findings (real HF, ~$0.05) that drove the change.** (a) The in-job upload step
  (`python -c "from huggingface_hub import HfApi …"`) failed `ModuleNotFoundError: No module named
  'huggingface_hub'` — `.[train]` does not pull it and the `python:3.12` base lacks it. (b)
  `train_win_probability --all-competitions` (3961 matches) died `OSError [Errno 28] No space left on
  device` writing shards under `--out` — HF Jobs ephemeral disk is a fixed per-flavor cap (≤ 50 GB), NOT
  configurable (`hf jobs run` has no `--disk`), shared by weights+dataset+checkpoints.
- **Storage is a read-write HF Storage Bucket, not the ephemeral disk (supersedes "/tmp output").** A
  private bucket (`silly-kicks/hf-refit-scratch`, `hf buckets create`) is mounted read-write at `/scratch`
  (`hf jobs run -v hf://buckets/<ns>/<bucket>:/scratch`). The trainer output + the corpus-driver shards
  write to `/scratch/{run_key}/{name}/{variant}`; the flat corpus is still `hf download`-ed to `/tmp`
  (fits). Storage is thus decoupled from the flavor's fixed disk. Rejected: a bigger-RAM/-disk flavor —
  it only raises the cap and gives no resume.
- **Resume via `--attempts` + the persistent bucket (the DGX OOM-continue behavior, on HF).** `hf jobs run
  --attempts N` auto-retries a crashed/timed-out job; because the shard-root persists in the bucket across
  attempts, the corpus-driver `for_each` (ADR-052) skips shards already written and continues. Verified: a
  re-run over a partially-populated bucket logs `[k/3961] … 0.0s skip (shard exists)` for every done shard.
- **One job per committed variant (supersedes "multi-variant trainer = one run").** `launch_all_variants`
  emits one `hf jobs run` per `VariantBuild` (→ 15 jobs): bounded per-variant timeout, partial-results
  safety (a crash never loses a finished variant), and a **per-variant** head-to-head floor — which closes
  the Commit-1 "default-only floor" follow-up (every variant is now floor-gated in its own job).
- **No in-job upload (supersedes the artifact-dataset upload).** Outputs persist in the bucket; they are
  retrieved locally with `retrieve_variant` (`hf cp` from the bucket). The planned `hf-refit-artifacts`
  dataset is dropped. `huggingface_hub` is still installed in-job, now only for the corpus `hf download`.
- **Requires hf CLI ≥ v2.2** (for `--volume`; v1.7.1 lacks it). `--` before `bash -lc` is still mandatory
  (v2.2 `--dry-run` confirms `-lc` is eaten without it); `--secrets HF_TOKEN` (no value, read from the
  launcher env) keeps the token off the process listing; polling reads `hf jobs inspect --format json`.
- **Three staged corpora (supersedes "two shapes"), each faithful to the committed weights' recorded
  provenance.** `hf-refit-tc3` (flat 179 = GS 64 + idsse 7 + SC 108; ghost_gk/ghost_outfield) — the
  committed ghost corpus recorded `providers = [gradientsports, skillcorner, sportec]`, and **sportec ≡
  idsse** (the re-materialization relabels the DFL matches; the `idsse__DFL-MAT-*` shards are built by
  `sportec_spadl.convert_to_actions`). `hf-refit-xshot` (the **exact 17** `corpus_match_ids` recorded in
  the committed xshot/xcross `metrics.json` = 7 idsse + 10 skillcorner, generation at the weights' own
  `run_commit a34e992`; a separate dataset because the in-job repackage globs the whole staged corpus).
  `hf-refit-sb360` (30-match pining cache + offline manifest + `match_ids.json`; receiver). win_probability
  stays full `--all-competitions` (3961 matches / ~80 competitions — faithful per committed metadata).
