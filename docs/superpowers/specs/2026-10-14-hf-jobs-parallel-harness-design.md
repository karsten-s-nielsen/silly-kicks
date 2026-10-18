# HF Jobs parallel training harness — design

- **Status:** DRAFT — awaiting owner review, then `writing-plans`.
- **Date:** 2026-10-14
- **Branch:** `feat/hf-jobs-training-harness` (continues the ADR-122 cycle; this is Commit-2, a new
  amendable harness-tune commit that does **not** touch the merged-ready Commit-1 `6f6508a`).
- **ADR:** ADR-123 (to be written during implementation; extends ADR-122).
- **Supersedes in part:** the `2026-10-13-hf-jobs-training-harness-design.md` sections on per-variant
  serial jobs, fixed per-trainer flavors, and the default-only / faithful-only head-to-head eval.

## Amendment A (2026-10-14, owner ruling — supersedes §4.1 xcross probe, §4.3 receiver, §7 staging)

**Principle: the code that runs on HF is DATASET-AGNOSTIC.** It must NOT send pining information to HF
(no token, no manifest, no `SILLY_KICKS_PINING_MANIFEST_DIR`, no pining-driving match-id list), and must
NOT import `_loader_pining` / use `--providers` pining mode in-job. Every owner-tier trainer consumes a
**staged neutral parquet dataset** via a data path; how that dataset was built (pining on the owner's
DGX, statsbombpy, kloppy, any user's own loader) is **off-HF** and not the repo's concern — "others 100%
need to be able to stage their own datasets." This is the ADR-123 headline decision.

Consequences (all implemented, all green):
- **xcross probe** is `--probe-data-dir` (a staged per-game probe dataset, `source_provider=gradientsports`),
  NOT the pining `--probe-match-ids-json`+`--providers`. The held-out TF-19 cohort is captured from staged
  DATA via `_iter_matches_from_dir` + `_extract(probe_providers=("gradientsports",))`. xcross stays
  `owner_pergame` for training; `TrainerSpec.probe=True`; launch downloads `probe_repo`→`/tmp/probe`.
- **receiver** gains a dataset-agnostic `--data-dir` source (`_data_dir_source`: flat shards + `_actions`,
  the head-to-head eval shape, yielding `(match_id, actions, frames)`). `owner_sb360` reads the staged flat
  corpus via `--data-dir /tmp/corpus --shard-root <bucket>` — **no manifest, no `--cache-dir`, no
  `--match-ids-json`**. The `sb360_repo` launch param is **removed** (receiver uses `corpus_repo`). This
  reworks the merged-ready Commit-1 receiver `owner_sb360` (it leaked pining info to HF).
- ghost (`owner_flat`) + xshot (`owner_pergame`) were already agnostic (plain parquet + `home_teams.json`).
  public = statsbombpy **open** data (universal, no owner-private dependency) — left as an in-job fetch.
- **Staging (§7, DGX)** produces ONLY neutral parquet: the GS `10502`/`10503` probe slice (per-game, for
  xcross) and the flat SB360 corpus (for receiver). Pining is used on the DGX to materialize; only the
  parquet crosses to HF.
- **Ghost `--build-cache` phase** added between extract and fold (race-free whole-corpus cache build), a
  gold-standard addition over the plan's 3 phases (no concurrent cache-build race among fold workers).

## 1. Context

The ADR-122 harness (Commit-1 `6f6508a`, PR #284, CI green, both reviewers APPROVE) re-fits the **four
public trainers** cleanly and cheaply on HF Jobs (wave-1: 4/4 head-to-head floor PASS, ~$0.058 total).
The **P1/P2 re-fit campaign** on 2026-10-14 then proved that the **owner-tier** path is systematically
broken under the as-built harness. The campaign was halted by the owner ("cancel the jobs, this is not
working") after four distinct failures:

1. **ghost_gk ×4 + ghost_outfield ×2 — perf wall.** Each ran on `cpu-performance` and was
   extraction-bound: `for_each` per-match extraction (~80 s/game × 179 games ≈ 4 h) never finished
   before even the 6 h timeout, so the single fit never started. Cancelled at ~4.74 h each; **~$54
   sunk** with zero weights. Bigger CPU does not fix this — the extraction is one long serial pass.
2. **xcross_attempt ×2 — TF-19 probe refusal.** Fresh extraction ran (91,999 rows, gates all True) but
   `assemble_studies` refused: `Feature cache present but _probe_sample/ absent -> ... Refusing to ship
   a spurious tf19_ready=False`. The committed xcross used a **separate held-out gradientsports probe
   pass** (`probe_sample_matches=[['gradientsports','10502'],['gradientsports','10503']]`, via
   `--probe-match-ids-json`). The staged `hf-refit-xshot` corpus dropped gradientsports, so the default
   `--probe-providers gradientsports` captured nothing and `_write_probe_sample` no-opped.
3. **xshot_occurrence position_only — floor crash.** `ValueError: feature_names mismatch`. The
   head-to-head eval (`_owner_eval_classifier`) extracts **faithful** (27-feature, includes `speed`)
   features, but the committed+fresh `position_only` booster is 26-feature (velocity-free). xgboost
   `predict_proba` is strict on `feature_names`. The per-variant floor refactor scores every variant;
   the original design floored default only.
4. **repackage gap.** `repackage.flat_to_pergame` writes only `frames.parquet` + `shots.parquet`;
   xcross's `_iter_matches_from_dir` also reads `{game}/actions.parquet`, patched in-job with a fragile
   symlink workaround.

The public tier works; the owner tier needs a proper, gold-standard rework. The owner has directed:
**cpu-upgrade only from now on**, path **A** (fix it properly, re-fit all nine), **tested e2e on the
DGX** (which stays a validated fallback executor), and **break the big jobs into smaller
parallelizable parts**. Scope and time are explicitly not a constraint.

### Decisions locked with the owner (2026-10-14)

- **Parallelism:** full fan-out — add the xshot/xcross study/fold seam to the three monolithic trainers;
  the single final model fit stays serial (splitting it would change the model and break faithfulness).
- **Probe fidelity:** faithful, including the TF-19 probe — stage the probe-provider matches and thread
  the exact probe config.
- **Executor:** HF Jobs (cpu-upgrade) primary; DGX is the e2e test bed and validated fallback. The
  harness must run identically in-job and on the DGX.
- **Flavor:** every trainer runs on `cpu-upgrade` (8 vCPU / 32 GB / $0.03/hr) until further notice.
- **Commit:** a new amendable harness-tune commit (not amending `6f6508a`); weights land after; PR #284
  merges non-squash so both commits stay reachable (weights' `run_commit` provenance). No commit, push,
  or merge without explicit per-action owner approval.
- **Reduce-phase RAM (owner HOLD, 2026-10-14):** fan-out does not shrink the serial reduce (final fit
  over the whole-corpus feature cache); `cpu-upgrade` is 32 GB vs `cpu-performance`'s 256 GB. The DGX
  e2e **measures peak RSS per phase vs 32 GB**; the owner declined to pre-authorize a fallback and will
  decide the reduce executor/flavor on the measured number. A reduce measured near/over 32 GB BLOCKS the
  HF wave until the owner rules (OOM is not `--attempts`-fixable).

## 2. Goals / non-goals

**Goals**

- Every heavy trainer decomposes into small, independently schedulable units that fan out across
  `cpu-upgrade` jobs, with a serial reduce that is **byte-identical** to a single-box serial run.
- Storage stays decoupled from compute (bucket mount), and every phase resumes (ADR-052 `for_each` +
  `--attempts`), so no single job must finish a multi-hour pass in one shot.
- Owner-tier re-fits are **faithful** to the committed artifacts: same corpus, same feature vintage per
  variant, same TF-19 probe.
- The head-to-head no-regression floor scores every variant correctly (feature-vintage-aware) and stays
  fail-closed.
- The whole pipeline is e2e-validated on the DGX against `~/tc3-kinematics` before any HF spend.

**Non-goals**

- No change to the trained models themselves, their features, or their metrics — this is a
  re-materialization harness, not a modeling change. The final fit is not parallelized.
- No new runtime dependency in `silly_kicks/` (the trainers already carry their deps).
- `gk_retention` stays BLOCKED (Databricks-marts corpus has no pining/staging path).
- No auto-accept of a regression; the floor BLOCKS and the owner investigates.

## 3. Target architecture — uniform three-phase map/reduce

Every heavy trainer exposes the same seam (xshot/xcross already do; ghost_gk, ghost_outfield, receiver
gain it):

| Phase | Work | Parallel | Unit | Output |
|---|---|---|---|---|
| **MAP-1 `extract`** | per-match feature extraction via `for_each` | yes | N workers, each a disjoint match subset | shards in the shared bucket `shard_root` |
| **MAP-2 `study`/`fold`** | HPO studies (xshot/xcross) · CV folds (ghost) | yes | M workers, each one study/fold | per-unit result files beside the shards |
| **REDUCE `assemble`** | ship-decision + **final fit (serial, faithful)** + TF-19 probe + CV aggregate + **permutation importance (serial)** | no | 1 worker | weights + `metrics.json` |
| **FLOOR** | per-variant head-to-head committed vs fresh, feature-vintage-correct | no | 1 worker | floor verdict into `metrics.json` |

**Determinism contract (the heart of the design).** The reduce over fanned units must equal the serial
run bit-for-bit. xcross/xshot already assert this (`assemble_studies` "byte-identical to the serial
run"). The new ghost seam adopts the same contract and the same guard: a unit test fits the toy corpus
serially and via the fanned `extract → study/fold → assemble` path and asserts identical shipped
weights + metrics.

**Subset selection (MAP-1).** A new `--shard-index I/N` flag (or `--only-keys <json>`) selects a
disjoint, deterministic subset of the match keys the trainer would otherwise walk. `for_each` already
resumes work rather than re-producing items, so N workers writing to one `shard_root` compose, and a
retried worker skips the shards it already wrote. The subset partition is by a stable hash of the
`for_each` key so the union is exactly the full corpus with no overlap and no gap (CI-asserted).

**Orchestration.** HF Jobs has no native DAG. `launch.py` grows a small staged orchestrator:
`run_stage(extract, N) → wait_all → run_stage(study/fold, M) → wait_all → run_stage(assemble+floor, 1)
→ wait`. Each job is `cpu-upgrade`, bucket-mounted at `/scratch`, with `--attempts` for in-stage
auto-resume. `wait_all` polls `hf jobs inspect --format json` (the existing `poll_until_terminal`),
and a stage fails loud if any unit is terminal-ERROR. The same orchestrator function drives a **DGX
run** by swapping the job runner for a local-subprocess runner (no HF), which is how the e2e gate works.

## 4. Per-trainer plan

### 4.1 xshot_occurrence, xcross_attempt — wire the existing seam

> **SUPERSEDED (probe mechanism) by Amendment A.** The xcross probe is NOT threaded via
> `--probe-match-ids-json`+`--providers` (pining to HF). Amendment A stages a neutral per-game probe
> dataset and threads `--probe-data-dir`; no pining reaches HF. The map/reduce seam below is unchanged.

The seam exists (`--list-studies`, `--study`, `--study-list`, `--assemble`, `--shard-root`). Work:

- Add MAP-1 `--shard-index I/N` to the extraction prep so extraction fans out (today the serial prep
  extracts all matches before studies).
- `launch.py` orchestrates extract-shards → `--list-studies` → fan `--study` workers → `--assemble`.
- **xcross probe (faithful):** stage the gradientsports probe matches `10502`/`10503` and thread
  `--probe-match-ids-json '{"gradientsports":["10502","10503"]}'` (+ `--providers idsse,skillcorner`
  as the pining training source) so `assemble_studies` reproduces the committed TF-19 probe. **xshot
  ships NO probe** (confirmed 2026-10-14: no `probe`/`tf19` keys in either `_xshot_weights` variant's
  metadata, no probe code in `train_xshot_occurrence.py`), so **only xcross** threads a probe config.
- Drop the in-job actions.parquet symlink workaround (see §4.4).

### 4.2 ghost_gk, ghost_outfield — add the seam (largest change)

Today: `for_each` extract → whole-corpus `_feature_cache` → single fit + per-fold CV
(`model.fit(X_train, y_train)` in a fold loop) + permutation importance. The fit is one model; the CV
folds and permutation-importance columns are embarrassingly parallel. New seam:

- **MAP-1 `extract`** via `--shard-index I/N` over the per-match `for_each` (shards already land under
  the bucket `--output-dir`). The whole-corpus `_feature_cache` is written by the reduce from the
  shards, not by each worker.
- **MAP-2** `--fold K` (one CV fold fit+score). Each reads the shared shards / feature cache.
  Permutation importance stays **serial in the reduce** (no `--importance-shard`): per-column
  permutation seeding across slices would be a byte-identity hazard the fan-out does not earn (importance
  runs on a bounded ~150k subsample, not the whole corpus, so it is not the memory/time driver).
- **REDUCE `--assemble`**: build the feature cache from shards, run the final fit, aggregate the CV
  folds, run permutation importance serially, write weights + `metrics.json`. Byte-identical to serial.

`ghost_outfield` mirrors `ghost_gk` (it already mirrors its extraction); the same seam is added once and
reused.

### 4.3 receiver_model — extraction fan-out + serial fit

> **SUPERSEDED (data source) by Amendment A.** Receiver no longer takes `sb360_repo`/manifest/
> `--match-ids-json` (pining to HF). Amendment A adds `_data_dir_source` and reads a staged neutral flat
> corpus via `--data-dir` through `corpus_repo`. The extraction fan-out + serial-fit seam below is unchanged.

`for_each` extraction already requires `--shard-root`; add `--shard-index I/N` so the per-provider
extraction fans out, then a serial reduce runs `ReceiverModel.fit`. The deployment-resolution
`for_each` pass folds into the reduce.

### 4.4 repackage — emit actions.parquet

`repackage.flat_to_pergame` writes `{game}/actions.parquet` (the full SPADL actions, symlinked like
`frames.parquet`) in addition to `shots.parquet`. The in-job symlink workaround in `launch._corpus_steps`
is removed. TDD: a unit test asserts all three files exist per game dir.

## 5. Head-to-head floor — feature-vintage-correct per variant

`headtohead.build_eval` / `_owner_eval_classifier` currently extract features at a fixed `feature_set`
(faithful). The scorer loads the committed+fresh model for a variant and calls `predict_proba`; a
velocity-free (26-feature) variant then mismatches the 27-feature eval frame.

Fix: the registry `VariantBuild` carries the variant's `feature_set` (already implied by its
`run_args`); `launch` passes `--feature-set <variant>` to `headtohead`; `build_eval` extracts the eval
features at that `feature_set`. The committed and fresh model for a variant are always the same vintage,
so a per-variant-vintage eval is correct and the floor comparison stays apples-to-apples. The ghost
scorers already reindex to the model's own feature names (`predict_mean`); only the xgboost classifiers
(xshot/xcross) need the vintage-correct extraction. CI: a test scores a 26-feature position_only model
against a position_only eval bundle with no mismatch, and asserts the faithful/position_only eval
frames carry 27/26 columns respectively.

## 6. Flavor policy

Every `TrainerSpec.flavor` → `cpu-upgrade`. Timeouts re-tuned for cpu-upgrade and for the new,
smaller per-phase jobs (each fanned unit is far shorter than the old monolith): extraction workers
sized to their match-subset, study/fold workers to one unit, assemble to the fit. A `DasCostWarning`-style
rough cost estimate per wave is printed before launch. The one possible exception to "cpu-upgrade only"
is the serial reduce phase if the DGX e2e measures its peak RSS near/over 32 GB — that is an owner
decision gated on the measurement (§8, §11), not a default.

## 7. Faithful staging (from the DGX)

> **SUPERSEDED (agnostic framing) by Amendment A.** Pining is the owner's OFF-HF staging tool only; it
> produces neutral parquet and never crosses to HF. Anyone can stage the documented neutral shapes with
> any loader — the DGX+pining path below is one example, not a requirement. Faithfulness targets below hold.

Owner-tier corpora are staged from the DGX (data + pining token live there) to private HF datasets.
The staging must preserve faithfulness:

- **ghost_gk / ghost_outfield:** full TC3 (`~/tc3-kinematics`, 179 games) — unchanged from the halted
  campaign's `hf-refit-tc3`.
- **xshot / xcross:** the exact recorded training providers (`idsse` + `skillcorner`, the 17 ids).
  **xcross additionally** needs the `gradientsports` probe matches `10502`/`10503`, staged in their own
  slice so the training glob does not pick them up (held-out by construction). **xshot needs no probe
  slice** (it ships no probe).
- **receiver:** SB360 licensed 30 (`hf-refit-sb360`, offline manifest) — unchanged.

Staging scripts live on the DGX (`~/hf_jobs_stage`); the recipe is documented in the howto. Dataset
names are confirmed with the owner before any push.

## 8. DGX e2e validation gate

Before any HF spend, each trainer's full `extract → study/fold → assemble → floor` pipeline runs **on
the DGX** against `~/tc3-kinematics` (and the SB360 cache for receiver), driven by the same orchestrator
with a local-subprocess runner. This proves: (a) the new seams produce byte-identical weights to a
serial run; (b) the faithful probe reproduces; (c) the floor scores every variant without mismatch;
(d) real wall-time per phase, to size timeouts and estimate HF cost; (e) **peak RSS per phase vs the
32 GB cpu-upgrade ceiling** (measured, e.g. `/usr/bin/time -v` or an RSS sampler per phase process) —
without this the serial reduce's OOM risk (§11) would surface only on paid HF, since the DGX's ample RAM
hides it. Only after a green DGX e2e — including every phase's peak RSS under 32 GB, or an owner ruling
on any phase that exceeds it — does the HF wave run. The DGX e2e is also the fallback executor if HF
Jobs misbehaves.

## 9. Testing strategy (TDD throughout)

- **Determinism + subset partition:** serial == fanned shipped weights + metrics, per heavy trainer, on
  a toy corpus spanning **≥2 shards + ≥2 CV folds** (an N=1 fan-out passes vacuously — the
  detector-must-fire principle); and `--shard-index I/N` over the keys is a disjoint cover (no overlap,
  no gap). The fixture must make a seeding/ordering divergence in the reduce actually fail byte-identity.
- **Floor vintage:** position_only eval is 26-feature and scores a 26-feature model with no mismatch;
  faithful eval is 27-feature.
- **repackage:** all three per-game files emitted; actions.parquet readable by the xcross iterator.
- **Orchestrator:** `run_stage`/`wait_all` sequence with an injected runner (no HF); a terminal-ERROR
  unit fails the stage loud; the DGX local-runner path is exercised.
- **Probe fidelity:** the threaded `--probe-match-ids-json` is rejected when a probe id overlaps
  training (the trainer's existing guard), and accepted for the held-out GS matches.
- Local lint/pyright scoped to changed files; the hour-long CI suite is authoritative (not run
  locally). DGX runs the real-data e2e.

## 10. Commit & provenance

- One new **harness-tune commit** on `feat/hf-jobs-training-harness`, amended during development until
  the DGX e2e is green and the local checks pass. It does **not** touch `6f6508a`. Docs (this spec, the
  plan, ADR-123, the howto update, C4 regen) travel **with** this commit (never a separate/early
  commit; reviewers read the uncommitted tree).
- The weights land in a **subsequent** commit after the HF (or DGX) re-fit clears the floor for every
  variant: cleared weights copied into `silly_kicks/**/weights/` (existing bundle shape, no new
  `metrics.json` where none existed) + `_version.py` bump + lakehouse re-mat stamp.
- PR #284 merges **non-squash** so Commit-1, the harness-tune commit, and the weights commit all stay
  reachable (the weights' `run_commit` provenance).
- No commit / push / merge / tag without explicit per-action owner approval.

## 11. Risks & open items

- **Reduce-phase wall — memory AND time.** The serial final fit + permutation importance is the one
  phase fan-out does not shrink. Moving it off `cpu-performance` (256 GB) onto `cpu-upgrade` (32 GB) is
  an 8× RAM cut; a 32 GB non-fit is an **OOM, which `--attempts` cannot fix** (it retries the same
  failing fit). The DGX e2e measures peak RSS per phase (§8); the owner **declined to pre-authorize** a
  fallback and decides on the measured number. A reduce measured near/over 32 GB BLOCKS the HF wave until
  the owner rules — candidate rulings are run-the-reduce-on-DGX (keeps HF cpu-upgrade-only) or a
  reduce-only larger-RAM HF flavor exception. A pure time overrun (RSS fits) is the benign case: a longer
  timeout with `--attempts` resume of the extraction/fold shards.
- **Probe-config recovery (RESOLVED — see §4.1/§7).** Confirmed 2026-10-14 that xshot ships no probe
  (no `probe`/`tf19` keys, no probe code), so only xcross reproduces its probe from the committed
  metadata. No open ambiguity remains.
- **ADR-123 number** assumes 122 is the current max (verified 2026-10-14); re-confirm at write time.
- **Staging cost/time** of the full TC3 + probe matches from the DGX; staged once, reused across waves.
- **Determinism under fan-out** depends on seeded studies/folds and order-independent reduce; the
  existing xcross/xshot guarantee is the precedent, extended to ghost.

## 12. Deliverables

1. `scripts/hf_jobs/registry.py` — flavor → cpu-upgrade; per-variant `feature_set`; probe config for
   xcross/xshot; re-tuned timeouts.
2. `scripts/hf_jobs/launch.py` — staged orchestrator (map/reduce DAG), `--shard-index` wiring, probe
   threading, symlink workaround removed, local-runner path for DGX.
3. `scripts/hf_jobs/headtohead.py` — feature-vintage-correct eval per variant.
4. `scripts/hf_jobs/repackage.py` — emit actions.parquet.
5. `scripts/train_ghost_gk.py`, `scripts/train_ghost_outfield.py` — add the study/fold seam.
6. `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` — `--shard-index` extraction
   fan-out (seam otherwise present).
7. `scripts/train_receiver_model.py` — `--shard-index` extraction fan-out.
8. Tests for every unit above (determinism, partition, floor vintage, repackage, orchestrator, probe).
9. `docs/superpowers/adrs/ADR-123-*.md`, `docs/howto/hf-jobs-training.md` update, C4 regen,
   `docs/superpowers/plans/2026-10-14-hf-jobs-parallel-harness-plan.md` (via `writing-plans`).
10. DGX staging + e2e driver (on the DGX; recipe documented in the howto) — runs every trainer's full
    map/reduce pipeline locally and reports **peak RSS per phase vs 32 GB** + wall-time, the gate before
    any HF spend.
