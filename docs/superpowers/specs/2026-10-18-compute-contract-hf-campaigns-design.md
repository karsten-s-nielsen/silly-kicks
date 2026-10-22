# Spec: Stage/unit compute contract + HF campaign orchestrator (Cycle 1)

**Status:** DRAFT r2 — addresses spec review r1 (change log in §13); awaiting re-review by reviewers A
and B
**Date:** 2026-10-18
**ADR:** ADR-128 (renumbered from ADR-127 on 2026-10-18: TF-46 took ADR-127 on `main` @ `cd2ebcb`;
ADR-128 was the next free number on `origin/main` and every remote branch then; re-confirm at the commit
gate, because ADR numbers are cross-branch volatile)
**Amends:** ADR-122 (launch shape), ADR-123 (per-trainer phase flags become the stage contract).
**Builds on:** ADR-052 (`scripts/_driver.for_each`), ADR-037 (clean-tree two-commit split), ADR-088
(`publish_model_with_card`), ADR-126 (trained-artifact integrity gate), the DGX parallel-launcher spec
`docs/superpowers/specs/2026-09-29-dgx-parallel-launcher-design.md` (`_parallel_launch`, `_mem_cap`,
`_thread_pin`).
**Branch:** `feat/compute-contract-hf-campaigns` (off `main` @ `7acf2e6`, 4.138.0). One feature branch
for the whole of Cycle 1.
**Authoring model:** Opus 5.5 (design only). Implementation runs on the owner's chosen model, inline,
with no subagents and no worktrees.

---

## 1. Executive summary (for reviewers)

The owner set three goals after the 4.138.0 re-fit campaign:

1. **Determinism.** The next retrain or re-fit must rediscover nothing.
2. **Fan-out.** Every job that takes more than one hour runs as many short jobs instead (for example
   6 × 1 h, not 1 × 6 h). Per-second billing makes the cost the same; the wall time drops.
3. **Born-ready.** New trainers and corpus drivers (for example TF-46 corners, in flight in the ORIGINAL
   session) are written on the fan-out contract from the first line. Nothing is retrofitted later.

The owner asked for the gold standard and set scope as no constraint. All jobs run on `cpu-upgrade`
(8 vCPU, 32 GB RAM, 50 GB disk, $0.03/h); no other flavor is ever used.

**Diagnosis (§2).** The 4.138.0 campaign did not run on committed code. The real orchestrator was two
uncommitted scratchpad scripts. They bypassed the committed ADR-123 map/reduce DAG, because that DAG had
untested real-HF paths, and ran every variant as one serial job. Those jobs used one of eight vCPUs and
took 5–9 hours each. The in-job environment floated, the `hf` CLI drifted under the harness, adoption
was manual, and no ledger recorded what ran.

**Design (§5).** One contract for every heavy pipeline. A pipeline is a static DAG of stages. A stage
is a set of independent, keyed, single-threaded units plus a reducer that reads results in canonical
key order. One executor places units as W memory-capped worker processes per job × N jobs. Outputs are
byte-identical for every (W, N); CI enforces it. Every unit persists atomically, so any kill at any point
resumes without redoing finished work. A committed campaign file drives a stateless orchestrator on the
typed `huggingface_hub` API, from preflight through adoption and publish. A ledger records every unit.
A planner sizes N and W from measured unit costs to meet a 60-minute wall target. A registry gate makes
a non-compliant new trainer or driver unmergeable.

**Cycle 1 is a vertical slice (owner ruling, decision 1 = C).** It ships ALL the plumbing plus a
representative job set chosen so that every framework part runs end-to-end on real HF: `ghost_gk`
(including the Hub-only `full` tier), `ghost_outfield`, `xshot_occurrence`, `xcross_attempt`,
`receiver_model`, `win_probability`, and the TF-54b territory census driver. The legacy launch path is
deleted in Cycle 1 (owner ruling R7): the three public trainers not yet migrated
(`match_outcome_dependence`, `pass_completion`, `xsuccess`) run through the new launcher as one-unit
adapter stages until Cycle 2 gives them real pipelines. Later cycles migrate the remaining jobs onto the
proven framework (§11).

**Cycle 1 acceptance (§7):** CI parity gates green; a real-HF campaign of the set runs from a committed
campaign file with zero scratchpad code; the re-fit reproduces the 4.138.0 weights byte-identically or
every difference is explained by a recorded environment delta; every job finishes in ≤ 60 minutes per
the ledger; adoption produces a commit-ready diff that passes the full `tests/` tree; and the publish
path runs end-to-end against private sandbox Hub repos.

---

## 2. Evidence (verified 2026-10-18)

### 2.1 Measured single-job wall times (HF Jobs API `inspect_job`, `cpu-upgrade`)

| Job (4.138.0 campaign) | Wall | Serial core of the job |
|---|---|---|
| `xshot_occurrence` sc_extended × 2 (115 matches) | 326–482 min | 50-trial TPE studies run inline; XGBoost `n_jobs=1` |
| `xcross_attempt` sc_extended × 2 (115) | 323–347 min | same |
| `ghost_gk` `full` (179) | 424 min | 179-match extraction + CV folds + final fit |
| `ghost_gk` × 4 variants (179) | > 480 min (timeout at 8 h), then ~76 min resumed | same |
| `ghost_outfield` × 2 | 153 min first attempt; ~57 min resumed | extraction + folds |
| `xshot_occurrence` bundled × 2 (17) | 57–61 min | inline TPE |
| `win_probability` (3961 open-data matches) | 74 min | open-data load |
| `match_outcome_dependence` | 39 min | — |

The ADR-123 DGX end-to-end run measured `ghost_gk` extraction at 6057 s for 179 matches and whole-phase
peak RSS of about 6 GB (`ghost_gk`), 8.07 GB (`xshot`), 7.33 GB (`xcross`), 0.59 GB (`receiver`). The
`profile` stage (§5.5) re-measures these per unit.

`hf jobs hardware` (2026-10-18): `cpu-upgrade` = 8 vCPU, 32 GB, 50 GB storage, $0.03/h. The trainers
use one core: `_xshot_occurrence.py:426` and `_xcross_attempt.py:627` set XGBoost `"n_jobs": 1`;
`train_xshot_occurrence._hpo_once` runs `OptunaStrategy(cfg, seed=42).run(obj,
backend=InProcessBackend())` with `sampler="tpe"`, `n_trials=50`; the only compute
backends that `ruthless` 0.7.0 exports are `ComputeBackend` (abstract) and `InProcessBackend` — it has no
distributed HPO backend. Only `train_ghost_gk` permutation importance uses
`n_jobs=-1`.

### 2.2 Root causes of rediscovery

1. **Uncommitted orchestration.** The waves ran from scratchpad `wave_driver.py` and `vclear.py`.
   Campaign configuration (corpus per trainer, run keys `RUN17`/`RUN115`/`RUNG`, tier routing, a
   uniform 24 h timeout) lived in their constants.
2. **The committed DAG was bypassed.** `wave_driver.solo`/`fan` launched each variant as one `full`
   job, because the ADR-123 DAG had defects that only real HF exposed: the xshot/xcross study
   enumeration returned zero studies for a single-candidate run (`launch.py:423` parses the study list
   from `--prep-only` stdout); `launch_variant_job` did not thread `--arms`; the xcross probe extractor
   read the `hf download` `.cache/` dotdir as a game; the owner head-to-head floor re-extracted the
   179-shard corpus and is vacuous for the owner tier (owner ruled CV-parity certification on
   2026-10-15).
3. **Floating environment.** The job runs `python:3.12` + `uv pip install --system -e '.[train]'`
   (`launch.py:165-169`), which resolves dependencies at job time. `uv.lock` pins
   `scikit-learn 1.9.0`, but the 4.138.0 `ghost_gk` `full` artifact was fit under `1.9.1`. Local
   verification ran on py3.10 while jobs ran on py3.12, which produced the feature-contract probe
   mismatch (`d86fc4ab` vs `9f17a54c`).
4. **CLI drift at spend time.** `hf` 1.7 → 2.2 changed `jobs inspect --format`, `cp --recursive`,
   bucket mount defaults (read-only unless `:rw`), and the job-id format. Each was found mid-campaign.
5. **Manual adoption.** Bundle shape, `corpus_match_ids` rehydration, the `_ANCHOR`/`_C1…_C7`/
   `_TF65_COMMIT7`/`_ADR126_COMMIT7` constants, model-card notes, and provenance keys were edited by
   hand across four or more files. Only the full test suite at CI time checked them; the first 4.138.0
   artifact commit broke 12 contract tests.
6. **No ledger.** Durations were reconstructed from the HF API afterwards. Timeouts were guesses
   (45 min → 2 h → 8 h → 24 h).

### 2.3 Prior art this design reuses (Chesterton's fence)

- `scripts/_driver.py` (ADR-052): `for_each(items, *, key, work, shard_root, token_inputs, …, load)`
  with token-versioned generation dirs, resume-before-load, `.excluded.json` markers, counters
  sidecars, `assert_conservation`, `_require_injective`; `shard_select(seq, index, n)` = round-robin
  `list(seq)[index::n]` (disjoint cover); `write_shard` writes atomically via `write_table_atomically`
  and writes an empty shard for "ran, produced nothing".
- `scripts/_parallel_launch.py` (2026-09-29 spec): `size_workers` = `max(1, min(nproc, usable //
  peak_rss))` with `RefusalError`; `should_launch` RAM backpressure; `run_parallel` relaunches a
  cap-killed worker and resumes from shards; `split_round_robin`.
- `scripts/_mem_cap.py` (`detect_backend`, `wrap`, `preflight` — systemd or rlimit caps) and
  `scripts/_thread_pin.py` (`thread_pin_env`).
- ADR-123 seams: `scripts/_corpus_identity` (visibility-blind helpers), `scripts/hf_jobs/arm_policy.py`,
  the I1 static import-closure gate (`tests/scripts/hf_jobs/test_visibility_blind_gate.py`).
- `scripts/_corpus.rehydrate_public_corpus_ids` (4.138.0) and `scripts/_hub_publish.py` (ADR-088).
- `huggingface_hub` 2.2 typed API: `HfApi.run_job(image, command, env, secrets, flavor, timeout,
  attempts, name, labels, volumes, …)`, `list_jobs(labels=…)`, `inspect_job` → `JobInfo` with
  `started_at`/`finished_at`/`status`, and a bucket API (`list_bucket_tree`, `download_bucket_files`,
  `batch_bucket_files`).

### 2.4 What new logic looks like (goal 3)

TF-46 (`origin/tf-46-corner-offence`, 53 files; verify with `git fetch origin tf-46-corner-offence &&
git diff --stat origin/main...origin/tf-46-corner-offence`) adds `silly_kicks/corners/` (zones, routines, windows,
a routine model, a run-zone model) and corpus drivers (`build_corner_routines`, `validate_corners`). The
contract therefore covers corpus drivers, not only `train_*` scripts.

---

## 3. Owner rulings (2026-10-18)

| # | Ruling |
|---|---|
| 1 | **Vertical slice.** Cycle 1 = all plumbing + the representative job set in §1; later cycles migrate the rest. |
| 2 | Approach A: a unified stage/unit contract (not incremental per-trainer work, not an external engine). |
| 3 | The contract covers trainers AND corpus-walking drivers. |
| 4 | HPO parallelism stays faithful: fold-parallel inside each seeded-TPE trial is the default; a non-adaptive sampler only per trainer with owner approval. |
| 5 | Same commit + same dataset on HF reproduces byte-identical weights; cross-platform within tolerance. |
| 6 | `uv sync --frozen` from `uv.lock` on a digest-pinned `python:3.12.x` image; local verification on the same python. |
| 7 | Vehicles: committed CLI + repo skills + howtos + an AGENTS.md bullet. No MCP. |
| 8 | One reviewed artifact manifest replaces the scattered constants; the cross-checks stay. |
| 9 | The planner targets ≤ 60 minutes per job. |
| 10 | Foundation and the born-ready gate land first; TF-46 merges as legacy and is the first burn-down item. |
| 11 | The DGX is an executor of the same contract, not a separate code path. |
| 12 | A committed `profile` stage + run ledger (no throwaway probes). The owner approved the rename from `calibrate` to `profile` (the word "calibration" already names model calibration in this repo). |
| — | `cpu-upgrade` only, always. cpu-upgrade spend for investigation is pre-approved; outward publish, commits, pushes and merges stay owner-gated. |
| — | Preserve and extend the current resilience: resume must never redo finished work. |
| R7 | `ghost_outfield` joins Cycle 1, and the legacy launch path is deleted in Cycle 1 (§5.6, §12 R7). |

**Terms used in this spec (one meaning each):** *unit* (one keyed piece of work); *stage* (a set of
units + a reducer); *pipeline* (a static DAG of stages); *worker* (one memory-capped process, W per
job); *job* (one HF Job, N per stage); *store* (the token-versioned generation dirs); *campaign file*
(the committed TOML that defines a campaign); *artifact manifest* (the committed TOML that declares
every shipped artifact); *ledger* (the per-unit run record); *job registry* (`scripts/hf_jobs/
registry.py`). The design conversation called the campaign file a "manifest"; this spec renames it so
that "manifest" has one meaning.

---

## 4. Invariants

- **INV-1 Determinism.** For a fixed commit, image, lockfile and dataset, every stage output and every
  shipped artifact is byte-identical for every worker count W, job count N, unit completion order, and
  restart history. (Mid-study HPO restart: §5.3.)
- **INV-2 Single-threaded units.** A unit uses one thread. Parallelism comes only from units.
- **INV-3 Resume.** A kill at any point loses at most the units in flight. A relaunch never redoes a
  finished unit and never trusts a unit whose checksum fails.
- **INV-4 Flavor.** Every job runs on `cpu-upgrade`. No code path selects another flavor.
- **INV-5 Data and policy agnosticism.** In-job code reads staged neutral datasets through a data path
  and receives any visibility policy as an explicit parameter (ADR-123 Amendment B). This now applies
  to drivers that run on HF.
- **INV-6 Pinned identity.** Every job and every local verification runs one recorded environment
  identity, and every artifact records it.
- **INV-7 No scratchpad orchestration.** Every campaign runs from a committed campaign file through the
  committed CLI.
- **INV-8 Born-ready.** Every trainer and corpus walker is contract-compliant or is on a legacy list
  whose every entry carries an explicit approval (§5.8).
- **INV-9 No self-certification.** No gate compares an artifact with a value derived from that same
  artifact. Expectations are declared before a campaign runs, or recorded at staging; `adopt` verifies
  each artifact against them and refuses on mismatch. `adopt` never writes an expectation field from an
  artifact; it copies `training_commit` from the campaign record only after the artifact is verified
  against that record (§5.7).

---

## 5. Design

### 5.1 Stage/unit contract (approved §1)

New module `scripts/_stages.py` (stdlib only at import time; pandas/numpy imported inside functions,
following `_driver.py`).

- **Unit.** A stable key (a tuple of strings: `("skillcorner", "1886347")`, `("fold", "3")`,
  `("public", "fold", "2")`) plus a pure, deterministic, single-threaded `work(unit, inputs) -> result`.
  A result is one tidy DataFrame, one `.npz`, or one JSON object, written to one file per key.
- **Stage.** A frozen dataclass: `name`; `units(inputs) -> Sequence[key]` (deterministic
  enumeration); `needs` (upstream stage names); `work`; `reduce(results) -> output`, where `results`
  arrive in sorted key order; `inner_parallel: bool` (§5.2); resource hints read from the ledger.
- **Store.** Each stage writes to an ADR-052 generation dir whose token is a digest of the code version,
  the stage parameters, and the upstream output tokens. Resume, exclusion markers, counters,
  conservation and injectivity are the ADR-052 mechanisms unchanged.
- **Pipeline.** Each trainer or driver module declares one static `PIPELINE` (a DAG of stages).
- **Uniform CLI** from `scripts._stages.main(PIPELINE)`:
  - `--plan` prints JSON: stages, dependencies, unit counts, tokens, and done/pending counts per stage.
  - `--stage S [--shard I/N] [--workers W] [--store URI]` runs one slice of one stage.
  - `--run-all [--workers W]` runs the whole DAG locally. This is the serial parity reference.
  It replaces the per-trainer `--shard-index`, `--fold`, `--study`, `--study-list`, `--list-studies`,
  `--assemble`, `--prep-only` and `--build-cache` flags for migrated modules. The launcher reads
  `--plan` and holds no per-trainer knowledge.
- **`for_each` compatibility.** `for_each` keeps its public signature (49 modules call it, §6.4). It
  becomes the implementation of an item-keyed stage. ADR-052 gates stay green unchanged.

### 5.2 Executor + determinism (approved §2, with one spec-time refinement)

- **One executor, two placements.** `LocalExecutor(W)` runs a stage's pending units on W worker
  processes. An HF job is only "run `--stage S --shard I/N --workers W`", which runs the same
  `LocalExecutor` inside the job. The DGX and a laptop run the same path (ruling 11).
- **Refinement of approved §2.** The design conversation named a stdlib `ProcessPoolExecutor`. The
  executor instead reuses the proven `_parallel_launch` model: W subprocess workers, each memory-capped
  by `_mem_cap.wrap` (rlimit inside HF containers), each started with `_thread_pin.thread_pin_env()`,
  sized by `size_workers`, gated by `should_launch` backpressure, and relaunched by `run_parallel` when
  its cap kills it. Reason: a pool worker killed by the OOM killer breaks the whole
  `ProcessPoolExecutor` (`BrokenProcessPool`); a capped subprocess dies alone and resumes from shards.
  Workers take units from a shared queue of pending keys, so a slow unit does not stall a static
  partition.
- **Why outputs never depend on (W, N):**
  1. Single-threaded units: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
     `NUMBA_NUM_THREADS` = 1 set before import (`_thread_pin`), XGBoost `n_jobs=1`.
  2. One atomic file per key (`write_table_atomically`); completion order is irrelevant.
  3. Canonical reduce: every reduction — a stage reducer AND every combination inside an
     `inner_parallel` unit (for example the fold losses of one TPE trial) — receives results keyed and
     combines them in sorted key order, never completion order. Floating-point addition is not
     associative, so this rule is load-bearing, not cosmetic.
  4. Input tokens: a stale or foreign shard is never reused.
- **Partition across jobs.** Default `shard_select` round-robin. When the ledger holds per-unit costs,
  the planner may use a longest-processing-time greedy partition. This is safe because outputs never
  depend on the partition.
- **`inner_parallel` stages.** Units of such a stage run one at a time at the top level of a job, and
  each unit gets the worker pool (for example a TPE study maps each trial's inner folds over the
  workers). There is no pool inside a pool. The unit maps inner keys (for example fold indices) over the
  pool and combines the returned values in sorted inner-key order (mechanism 3).
- **Memory bound.** W per stage = `size_workers(peak_rss_per_unit, ~30 GB usable, nproc=8, headroom)`.
  Workers record `ru_maxrss` per unit into the ledger. A unit that exceeds its cap fails with an OOM
  reason; a unit that cannot fit 32 GB alone is a `RefusalError` at preflight, not a flavor change.
- **Failure rule** (parallel-safe form of `for_each`'s consecutive-failure rule): a failed unit is
  recorded and skipped; the stage aborts when failures exceed a threshold, or when the first k
  completed units all failed. Both thresholds are parameters; the plan fixes their defaults.
  Exclusions and conservation are unchanged.
- **CI determinism gates:**
  - per migrated job: a toy corpus run serially (W=1, N=1) and fanned (W=2, N=2) produces
    byte-identical stage outputs and a byte-identical final artifact (SHA-256);
  - framework non-vacuity: an order-sensitive reduce, plus an assertion that completion order actually
    differed from key order in the fanned run;
  - worker thread-pin assertion;
  - fixture sizing that exercises every parallel path: each study fixture has ≥ 2 inner folds and runs
    at W ≥ 2; the `ghost_gk` `importance` unit and the census `bootstrap` unit run at W > 1; and each
    fanned run asserts that completion order differed from key order on that path, so no parity check
    is vacuous.
- **Campaign gate:** a same-commit HF re-run reproduces the artifact SHA-256 values in the prior ledger.

### 5.3 Resilience (approved §2b)

Preserved unchanged: per-key shards in token-versioned generation dirs; resume-before-load; exclusion
markers; counters sidecars replayed on skip; conservation and injectivity; `prune_stale_generations`;
atomic writes (absent = not run, empty = ran with no rows); HF `attempts` onto the bucket-mounted store;
the xshot/xcross study shards keyed by `objective_id` and `n_trials`; the `ghost_gk` feature cache and
its `cache_token`.

Extended:
1. Every stage resumes, not only extraction: folds, studies, importance units, the final fit, and
   reducer outputs each persist per key.
2. Campaign-level resume: `--plan` reports done/pending per stage from tokens; relaunching a campaign
   skips completed stages and units.
3. Resume-before-download: a job fetches only the units of its shard from the staged dataset (HF
   include patterns, or a read-only dataset volume if HF supports it — §12, R3).
4. Integrity on resume: each unit writes a checksum sidecar; resume verifies it before it skips. A
   truncated or poisoned shard is re-run.
5. The ledger records attempts, resumes, and per-unit outcomes.

**Mid-study HPO restart (decision rule).** Optuna's store resumes completed trials, but a seeded TPE
sampler restarted in a new process may suggest a different next trial than an uninterrupted run. The
implementation tests this on a toy study first. Rule: if a resumed study is byte-identical to an
uninterrupted one, keep resume; otherwise an interrupted study unit restarts from trial 0 and the ledger
records the restart. Persisting sampler state is not pursued (it couples the design to Optuna
internals). A study unit lasts minutes to tens of minutes once it is its own unit (§5.6), so a restart
is cheap.

### 5.4 Pinned environment + launcher API + preflight (approved §3)

**Environment.**
- In-job image `python:3.12.<patch>-slim@sha256:<digest>`, pinned in one constant (the plan fixes the
  exact patch version and digest), and
  `uv sync --frozen --extra train [extras]` from the committed `uv.lock`. Nothing resolves at job time.
- New dependency group `hfjobs` in `pyproject.toml` (launcher dependencies, `huggingface_hub >= 2.2`,
  locked in `uv.lock`). The scratchpad `hfcli` virtual environment retires.
- Local verification uses a separate environment built from the same lock on python 3.12:
  `UV_PROJECT_ENVIRONMENT=.venv-hfjobs uv sync --frozen --python 3.12 --extra train --group hfjobs`.
  It never touches `.venv`. `.venv-hfjobs/` is added to `.gitignore`.
- Fixed variables in every job and worker: the thread pins, `PYTHONHASHSEED=0`, `PYTHONUTF8=1`,
  `LC_ALL=C.UTF-8`, `TZ=UTC`.
- New helper `scripts._provenance.env_provenance()` returns the image digest, python version,
  `uv.lock` SHA-256, versions of numpy, scipy, pandas, scikit-learn, xgboost and numba, the platform,
  the commit, and the run key. Every artifact that the stage contract writes stamps it.
  `git_provenance()` keeps its signature and its output keys (57 consumers, §6.5).

**Acceptance criterion 3 (rebaseline rule).** 4.138.0 came from a floating environment. Cycle 1
re-fits under the pinned environment and compares each artifact with 4.138.0 byte for byte. Each
difference must be explained by a recorded environment delta, and the artifact must pass the ADR-126
integrity gate and the CV-metric δ of its certification. The pinned re-fit then becomes the baseline.
From then on every same-commit re-run must be byte-identical (INV-1).

**Launcher on the typed API.** `scripts/hf_jobs/launch.py` calls `HfApi.run_job(image, command,
volumes, labels, attempts, timeout, flavor)`; status comes from `list_jobs(labels=…)`; `JobInfo.
started_at` and `finished_at` feed the ledger; retrieval uses the bucket API. Every job carries labels
`campaign`, `run_key`, `job`, `variant`, `stage`, `shard`, `commit`. `flavor` is the constant
`"cpu-upgrade"` and a test asserts that no other value is reachable. The `hf` CLI remains for humans.

**Preflight** (no spend; `launch` refuses without a passing report for the same commit and the same
campaign-file hash):
1. Ref: the clone commit is pushed, equals local `HEAD`, and has a green CI run.
2. Lock: `uv lock --check` passes; the image digest resolves.
3. API contract: the pinned `run_job` and `list_jobs` expose the parameters above; the flavor catalog
   lists `cpu-upgrade` at 8 vCPU / 32 GB.
4. Dataset contract: every staged dataset exists and matches its declared shape (keys, counts,
   `home_teams.json`, probe and sb360 manifests).
5. Plan: every job's `--plan` and a toy `--run-all` pass in `.venv-hfjobs` (this catches empty study
   enumeration, missing arms, and dotdir crashes).
6. Estimate: wall time and cost per stage from the ledger; any stage predicted above 60 minutes is
   flagged.

### 5.5 Campaign orchestrator + ledger + planner + profile (approved §4)

- **Campaign file** `scripts/hf_jobs/campaigns/<name>.toml`: name, commit (`HEAD`, pushed), wall target
  (60 min), concurrency cap, and per job: job name, variants, dataset, expected tier. Datasets are
  declared once in `scripts/hf_jobs/datasets.toml`: HF repo, pinned revision, shape contract, generation
  id, and the expected corpus identity recorded at staging (§5.7). Preflight check 4 reads it.
- **Run keys are derived:** `{campaign}-{commit7}-{job}-{tier}`. Bucket collisions are impossible; no
  hand-picked run-key constants.
- **CLI** `python -m scripts.hf_jobs campaign <file> <command>` with `preflight`, `profile`, `plan`,
  `launch`, `status`, `retrieve`, `verify`, `adopt`, `publish`, `resume`, `cancel`, plus
  `python -m scripts.hf_jobs scaffold` (§5.8).
- **Stateless controller.** All state is in the bucket ledger and the job labels. The controller can
  stop at any time; `status` and `resume` rebuild the state from labels and tokens. It advances each
  job stage by stage (a stage barrier) and runs different jobs concurrently. A persistent stage failure
  marks that job failed; the other jobs continue; the report names the failure.
- **Ledger.** The full per-unit record is in the bucket: job id, attempts, resumes, start, end, wall,
  peak RSS, outcome, environment identity. A compact per-campaign summary is committed as
  `docs/research/hf_campaigns/<campaign>.json` with top-level `run_commit` and `run_tree_dirty`
  (it passes `test_artifact_provenance_output`). The planner learns from these summaries.
- **`profile` stage.** It runs a fixed small sample of units per stage at W=1 and at the RSS-bounded
  maximum W (≤ 8), and measures
  per-unit wall, peak RSS, worker-scaling efficiency (important for memory-bandwidth-bound RFCDE fits),
  and job startup overhead. It runs automatically when a stage token has no history.
- **Planner.** Inputs: unit counts (`--plan`), per-unit cost (p50, p95, peak RSS), and constraints
  (8 vCPU, ~30 GB usable, startup overhead, concurrency cap, 60-minute target). Output per stage: N,
  W, the partition, predicted wall and cost, and a derived timeout (predicted p99 × a safety factor,
  with a floor). A single unit longer than the target is reported as the irreducible wall floor.
- **Certification is single-sourced** as `TrainerSpec.certification` in the job registry:
  `public` = head-to-head floor + ADR-126 integrity gate; `owner` = CV-parity against the committed
  `metrics.json` + integrity gate (head-to-head is diagnostic only); `rebaseline` = criterion 3.
- **Replaced:** scratchpad `wave_driver.py`/`vclear.py`, their `_rewrite` CLI adaptations, the uniform
  24 h timeout, and the whole legacy `launch` DAG path (`run_pipeline`, `_phase_flags`, the
  `--prep-only` study-list parser), which is deleted in Cycle 1 (§5.6, §12 R7).

### 5.6 Migration of the representative set (approved §5)

**Rule:** only orchestration moves. Feature extraction, objectives, fits and seeds stay unchanged. Each
module keeps `--run-all` as its serial parity reference.

| Job | Pipeline (`[unit]`) | Notes |
|---|---|---|
| `ghost_gk` (`default`, `position_only`, `sweeper`, `sweeper_position_only`, `full`) | `extract[match]` → `cache` → `cv[fold]` → `fit` → `importance` → `assemble` | `fit` is the irreducible floor. `importance` stays one `inner_parallel` unit that calls scikit-learn with `n_jobs=W`, so scikit-learn's seed stream is unchanged. The extraction token excludes variant-only parameters, so the 179 matches are extracted once for all variants that share it, not five times. |
| `ghost_outfield` (`default`, `position_only`) | `extract[match]` → `cv[fold]` → `fit` → `assemble` | No whole-corpus cache stage (ADR-123 §1). Extraction is shared across the two variants when its token is variant-independent. |
| `xshot_occurrence` | `extract[match]` → `prep` → `study[arm × (fold ∪ final)]` → `cv[arm × fold]` → `select` → `fit[shipped arm]` → `assemble` | `_hpo_once` already runs one seeded-TPE study per `(arm, fold)` tag with a tag-keyed store; those studies become units. Spare workers run each trial's inner folds (`inner_parallel`). `select` applies the paired comparison and `ship_order[-1]`. `units()` enumerates studies, so the empty-study-list defect cannot occur. |
| `xcross_attempt` | as `xshot_occurrence`, plus `probe[match]` before `assemble` | Exercises the `owner_probe` source. The probe extractor ignores dotdirs. |
| `receiver_model` | `extract[match]` → `fit` → `assemble` | Exercises the `owner_sb360` offline-manifest source. |
| `win_probability` | `extract[match]` (3961, fetched per unit) → `fit` → `assemble` | Exercises the `public` source and large-N fan-out with resume-before-download. |
| `build_territory_ranking_census` | `prepass[match]` → `prefit` → `score[match]` → `bootstrap` → `report` | StatsBomb open data. `bootstrap` stays one `inner_parallel` unit (single RNG stream, faithful). Replicate-block units would change the numbers and need owner opt-in. |

**One-unit adapter stage (Cycle 1 to Cycle 2).** The new launcher runs an unmigrated module through one
generic stage with a single unit whose `work` invokes the module's existing CLI unchanged, inside the
same pinned image, labels, ledger and bucket store. The module's own `for_each` shards keep its resume
behaviour. This applies only to `match_outcome_dependence`, `pass_completion` and `xsuccess` (each
finishes in under 60 minutes as one job: 39 min, 84 s, 70 s). They stay in `_LEGACY_UNMIGRATED`; the
adapter is deleted when Cycle 2 migrates them. The adapter is one stage type inside the one launcher,
not a second launch path.

### 5.7 Adoption + publish + artifact manifest (approved §6)

**Artifact manifest** `docs/artifacts/manifest.toml` (repo-only; not in the wheel). One entry per
artifact, with two field classes (INV-9):
- **Expectation fields** — declared before the campaign runs; `adopt` reads and verifies them and never
  writes them from an artifact: destination (`wheel:silly_kicks/tracking/_xshot_weights/default` or
  `hub:silly-kicks/xshot-occurrence-v1`), job, variant, tier, certification, the corpus expectation (a
  reference to a `datasets.toml` entry), the re-fit event (version, ADR, reason), the card path, the
  sibling links (Hub card ↔ wheel sibling), and `training_commit`. `training_commit` is the campaign's
  commit from the preflight record. The independent check happens at adoption (artifact vs campaign
  record); the stored value then guards against later drift, exactly as `_ANCHOR` does today.
- **Observed fields** — written by `adopt` after verification and never used as an expectation by any
  gate: artifact SHA-256 values, the metric summary that the cards render, the environment provenance,
  the campaign id, and the adoption date.

**Corpus expectation (closes the 4.138.0 "uncheckable digest" gap).** At staging time the stage tool
computes the expected corpus identity with the same `_corpus_identity.corpus_identity` function over the
staged shards' own `source_provider`/`match_id` columns. It records in `datasets.toml`, next to the
dataset's pinned revision: the digest over the raw per-row names, the expected n, and — for a public
dataset — the canonical id list (for the wheel public arm, `BUNDLED_PUBLIC_ARM`). The trainer computes
its in-job digest with the same function over the same raw names, so `adopt` can require equality. The
job also records the dataset revision it read, and `adopt` requires that it equals the pinned revision.

It replaces: `_ANCHOR` and the `POLICY` values (`tests/test_bundled_weights_corpus_policy.py`);
`_C1`…`_C7` (`tests/tracking/test_position_only_bundled.py`); `_PROVENANCE_VERSION`, `_TF65_VERSION`,
`_TF65_COMMIT7`, `_ADR126_VERSION`, `_ADR126_COMMIT7` (`tests/test_model_cards_match_bundles.py`); and
`CARD_SOURCE`/`HUB_REGISTRY` (`scripts/_hub_publish.py`, §6.3). The tests compare each artifact's
`metadata.json`/`metrics.json` with the expectation fields and with `datasets.toml` — two independent
sources, because no expectation is derived from an artifact — and any disagreement fails. The generated
card regions (below) are a projection of expectation and observed fields, checked for staleness. Only
the duplication of constants goes.

**Generated card regions.** Provenance blocks (the wheel-sibling sentence, the re-fit paragraph, the
provenance line, the metric tables) sit between `<!-- generated:<name> -->` and
`<!-- /generated:<name> -->` markers and are rendered from the manifest and `metrics.json`. Prose
outside the markers is human-owned. A staleness test asserts that each region equals its re-render. The
manifest also declares each Hub artifact's own re-fit event, so a card states it truthfully, and
`publish` verifies that the Hub holds the declared artifact.

**`adopt`** (idempotent; produces a commit-ready diff; never commits):
1. Retrieve through the bucket API into a staging directory outside the repo.
2. Verify the load (`SHA256SUMS`, chirality, feature contract) in `.venv-hfjobs`.
3. Certify per `TrainerSpec.certification`, running `validate_trained_artifact_integrity` with the
   inputs recorded in the ledger.
4. Verify expectations, and refuse on any mismatch: the artifact's `training_commit` equals the
   campaign commit; its recorded dataset revision equals the pinned revision; its corpus digest and n
   equal the staging-time expectation; its shipped tier equals the expected tier.
5. Bundle the allowlisted files into wheel destinations in canonical shape. Public bundles go through
   `rehydrate_public_corpus_ids` after step 4, so the rehydrated ids rest on a verified digest; owner
   bundles keep the digest. Hub destinations are staged for `publish`.
6. Write the observed fields, and set `training_commit` from the campaign record (verified in step 4).
7. Re-render the generated card regions.
8. Write the audit memo and verdict JSONs with top-level `run_commit` provenance.
9. Draft the CHANGELOG section for a human edit.
10. Run the full `tests/` tree (`-m "not e2e and not slow" --continue-on-collection-errors`) and scoped
   ruff + pyright, report, and stop at the commit gate.

ADR-037 guard: `adopt` requires a clean tree at the campaign's commit, and its diff may touch only
artifacts, the manifest, cards, verdicts and docs.

**`publish`** (owner go every time): for each Hub artifact, the unchanged ADR-088
`publish_model_with_card`, then post-verification: a model-only file set, remote `SHA256SUMS` equal to
local, `README.md` equal to the rendered card, and a `from_hub` round-trip; results go to the ledger.
`--sandbox` publishes to private `silly-kicks/sandbox-<artifact>` repos (criterion 5).

### 5.8 Born-ready governance (approved §7)

1. **Compute registry gate** (ADR-056 derive-and-assert), new `tests/scripts/test_compute_contract.py`.
   The population is derived: every `scripts/train_*.py` plus every module that calls `for_each` (the
   ADR-052 corpus-walker population). Each member must declare `PIPELINE`, return a valid DAG from
   `--plan` on a toy input, register a W1N1-vs-W2N2 parity fixture, and have a job-registry entry
   (`TrainerSpec`, or a new `DriverSpec` for drivers). `_LEGACY_UNMIGRATED` lists non-compliant
   members as records `{reason, approval}`; it must be a subset of the population. Entries seeded on the
   day the gate lands (from `main`, including TF-46 if it merged first) carry the approval
   `"ADR-128 landing seed"`; any later addition must cite a dated owner ruling, and a test rejects an
   entry that lacks either field. No in-repo test can see history, so "shrink-only" is enforced as
   "every entry carries an explicit approval", and the PR diff is the review point for any growth. A
   red fixture (a toy module without `PIPELINE`) proves that the gate rejects a non-compliant member
   (the ADR-056 `test_the_rule_FLAGS_an_unenrolled_driver` pattern).
   `_NOT_A_PIPELINE` is separate: it names members that are legitimately not compute pipelines (for
   example the pining cache warmer `_loader_pining_to_cache` or the DGX staging tool
   `materialize_tc3_frames`), each with a written reason; a test asserts every entry names an existing
   file, and adding an entry is a reviewed decision (the ADR-056 `_NOT_A_DRIVER` pattern).
2. **Shared stage CLI** `scripts._stages.main(PIPELINE)` (§5.1). Migrated scripts no longer hand-write
   stage argparse, which also removes the "parserless `--help` runs `main()`" trap for them.
3. **Scaffolder** `python -m scripts.hf_jobs scaffold trainer|driver <name>` emits a compliant module
   with a minimal working toy `PIPELINE`, its parity test, a job-registry stub, a dataset-contract stub,
   and (for trainers) an artifact-manifest stub. The output passes the gate on day one.
4. **Dataset-agnostic driver loader.** Drivers read through the ADR-052 D14 refs/`load` port with three
   adapters: pining (owner, DGX only), staged-neutral (HF), and StatsBomb open (public). The I1
   import-closure gate extends to every driver that runs on HF. The census uses the open adapter; the
   trainers use staged-neutral.
5. **Repo skills** in `.claude/skills/`: new `compute-pipelines` (authoring on the contract: unit
   sizing, the single-thread rule, determinism, resilience, `profile`, the scaffolder) and new
   `hf-campaign` (the operator flow, with owner gates). `authoring-a-metric`, `new-metric` and
   `corpus-drivers` route trained models and drivers to `compute-pipelines`. Skills stay short and point
   to the howtos.
6. **Howtos.** New `docs/howto/compute-pipelines.md`; `docs/howto/hf-jobs-training.md` rewritten around
   the campaign CLI; `docs/howto/corpus-drivers-runbook.md` updated; `tests/test_howto_links_wired.py`
   extended.
7. **AGENTS.md.** One Key-conventions bullet (≤ 600 characters, ADR-128 reference) and the
   invariant-inventory fixture update (`tests/test_agents_md_budget.py`).
8. **ADR-128** records the contract and the orchestrator; ADR-122 and ADR-123 get supersede markers.
   The C4 model (`docs/c4/architecture.dsl`) gains the orchestrator, the stage executor and the
   artifact manifest, and `architecture.html` is regenerated with the pinned Graphviz `dot`.
9. No MCP server (ruling 7). TF-46 merges as legacy (ruling 10); the owner relays any heads-up to that
   session, and this session never writes the ORIGINAL folder.

---

## 6. Scope — every caller of every changed seam (house rule; sweep 2026-10-18)

### 6.1 Per-trainer phase flags (`--shard-index`, `--fold`, `--study`, `--study-list`, `--list-studies`, `--assemble`, `--prep-only`, `--build-cache`)

Callers outside `train_*.py`: `scripts/hf_jobs/launch.py` (`_phase_flags`, `launch.py:95-108`, and the
study-list parser `launch.py:423`), `scripts/_driver.py` (`shard_select` docstring), `scripts/
_parallel_launch.py`, `docs/howto/hf-jobs-training.md`, and the tests
`tests/scripts/hf_jobs/test_dgx_e2e.py`, `test_launch_orchestrator.py`,
`tests/scripts/test_expect_variant_wiring.py`, `test_train_ghost_outfield_mapreduce.py`,
`test_train_receiver_mapreduce.py`, `tests/tracking/test_train_ghost_gk_mapreduce.py`,
`test_xshot_xcross_shard_index.py`. Every trainer that uses these flags (`ghost_gk`, `ghost_outfield`,
`xshot_occurrence`, `xcross_attempt`, `receiver_model`) migrates in Cycle 1, so the flags and their
launcher path are deleted and the tests move to the stage CLI.

### 6.2 `scripts/hf_jobs/launch.py` symbols

| Symbol | Callers (excluding `launch.py`) |
|---|---|
| `run_pipeline`, `wait_all`, `StagePhaseError` | `tests/scripts/hf_jobs/test_launch_orchestrator.py`; docs (ADR-123, the 2026-10-14 plan, the howto) |
| `run_stage` | the name also exists independently in `scripts/calibrate_tracking_defaults.py` and `tests/calibration/*` (different symbol; untouched) |
| `phase_sequence`, `_phase_flags` | `scripts/hf_jobs/dgx_e2e.py`, `test_launch_orchestrator.py`, the howto |
| `build_phase_command` | `test_launch_orchestrator.py`; the 2026-10-15 spec/plan |
| `launch_variant_job`, `launch_all_variants`, `retrieve_variant`, `poll_until_terminal`, `build_argv`, `build_inner_command`, `arms_payload` | `tests/scripts/hf_jobs/test_launch.py`; `build_inner_command` also in `scripts/hf_jobs/registry.py`; docs |

The typed-API launcher replaces these. Their tests are rewritten against the new surface. Historic
specs, plans and ADRs are records and are not edited, except for the ADR-122/123 supersede markers.

### 6.3 Constants replaced by the artifact manifest

`_ANCHOR` (`tests/test_bundled_weights_corpus_policy.py`); `_TF65_COMMIT7`, `_ADR126_COMMIT7`,
`_PROVENANCE_VERSION` (`tests/test_model_cards_match_bundles.py`); `_C1…_C7`
(`tests/tracking/test_position_only_bundled.py`); `CARD_SOURCE` (`scripts/_hub_publish.py`,
`scripts/publish_model_card.py`, `scripts/validate_hub_variants.py`,
`tests/scripts/test_hub_variants_driver.py`, `tests/scripts/test_publish_model_card.py`);
`HUB_REGISTRY` (`scripts/_hub_publish.py`, `scripts/validate_hub_variants.py`,
`tests/scripts/test_hub_variants_driver.py`); `WEIGHTS_CLASSIFICATION`
(`tests/test_bundled_weights_classification.py`, `tests/test_bundled_weights_corpus_policy.py`) —
derived from the manifest, its assertions unchanged.

### 6.4 `for_each` (signature kept)

49 modules call it (the 50th grep match is `_driver.py`, which defines it): `_loader_pining_to_cache`,
`_xt_corpus`, 9 `build_*`, 3 `calibrate_*`, `check_stage1_argmax`, `derive_coordination_params`,
`derive_opengoal_range`, `evolve_xsuccess_features`, `materialize_tc3_frames`, 6 `measure_*`,
`run_signoff_power`, 9 `train_*`, `tune_structural_pass_sigma`, and 13 `validate_*`. No caller changes in
Cycle 1 except the migrated set. This list plus every `train_*.py` is the derived population of the §5.8
gate; non-migrated members seed `_LEGACY_UNMIGRATED` or, with a reason, `_NOT_A_PIPELINE`.

### 6.5 `_provenance.git_provenance` (signature and keys kept)

57 `scripts/*.py` files call it, and 57 call `require_clean_tree`. A text search for
`require_clean_tree` finds 59: the two extra files (`audit_velocity_fixtures.py:60`,
`render_sb360_matrix.py:3`) name it only in docstrings that explain a deliberate absence of the guard.
All counts exclude `_provenance.py` and `__pycache__`. `env_provenance()` is additive; only
stage-contract outputs stamp it in Cycle 1.

### 6.6 Job registry, floor, head-to-head

`TrainerSpec` and `VariantBuild` are used by `scripts/hf_jobs/launch.py` and
`scripts/hf_jobs/dgx_e2e.py`. `evaluate_spec` is used by `scripts/hf_jobs/floor.py`,
`tests/scripts/hf_jobs/test_floor.py`, `tests/scripts/hf_jobs/test_headtohead.py`. `TrainerSpec` gains
`certification` and a pipeline reference; `DriverSpec` is new.

---

## 7. Cycle 1 acceptance

> **Outcome (2026-10-21):** criterion 1 is met; criteria 2–5 are not met by Cycle 1, and by owner ruling they move
> to the campaign-throughput cycle (ADR-129). See ADR-128 "Cycle 1 outcome".

1. **CI gates:** the W1N1-vs-W2N2 parity gate per migrated job, framework non-vacuity, thread-pin, the
   compute registry gate with its approval-carrying legacy list and red fixture, the
   expectation-vs-artifact gates (INV-9), the generated-card staleness test, and the rewritten launcher
   tests. All green on the CI matrix.
2. **Real-HF campaign:** the representative set and the three adapter-launched public trainers run from
   `scripts/hf_jobs/campaigns/cycle1.toml` through the CLI, from `preflight` to `adopt`, with zero
   scratchpad code.
3. **Faithful migration:** byte-identical to 4.138.0, or every difference explained by a recorded
   environment delta under the rebaseline rule (§5.4). A second same-commit run is byte-identical.
4. **Wall time:** every job ≤ 60 minutes per the ledger, or reported as an irreducible floor with its
   measured unit time.
5. **Adoption + publish:** `adopt` produces a commit-ready diff that passes the full `tests/` tree;
   `publish --sandbox` runs end-to-end against private sandbox repos with all post-verification checks.
   Public Hub repos change only if weights change, and only on a separate owner go.

---

## 8. Testing (TDD, CI-gated)

- Every new module (`_stages`, the executor, the planner, the ledger, preflight, the typed launcher,
  `adopt`, `publish` post-verification, the scaffolder, the driver loader adapters) is written test
  first.
- Gates: §7.1, plus the existing ADR-052 driver gates, the I1 visibility-blind gate extended to HF
  drivers, `test_provenance_wiring`, `test_artifact_provenance_output`, and the AGENTS.md budget test.
- Side-effecting paths (`run_job`, bucket writes, Hub publish) are tested through injected fakes; the
  real-HF evidence is the §7.2 campaign.
- Local gate before any commit proposal: the full `tests/` tree with markers
  (`-m "not e2e and not slow" -p no:randomly --benchmark-skip --continue-on-collection-errors`), never
  a path subset, plus `ruff check silly_kicks/ tests/ scripts/`, `ruff format --check`, and pyright
  scoped to changed files.
- CI budget: any new `@pytest.mark.slow` test states its measured cost, and the plan places it so the
  `slow` job stays within its wall budget.

---

## 9. Deliverables (file map)

New: `scripts/_stages.py`; `scripts/hf_jobs/executor.py`; `scripts/hf_jobs/planner.py`;
`scripts/hf_jobs/ledger.py`; `scripts/hf_jobs/preflight.py`; `scripts/hf_jobs/campaign.py` (CLI,
`__main__`); `scripts/hf_jobs/adopt.py`; `scripts/hf_jobs/scaffold.py`; `scripts/hf_jobs/campaigns/
cycle1.toml`; `scripts/hf_jobs/datasets.toml`; `scripts/_driver_sources.py` (loader adapters);
`docs/artifacts/manifest.toml`; `docs/howto/compute-pipelines.md`; `.claude/skills/compute-pipelines/`;
`.claude/skills/hf-campaign/`; `docs/superpowers/adrs/ADR-128-*.md`; tests for each.

Changed: `scripts/hf_jobs/launch.py` (typed API), `registry.py` (`certification`, pipeline reference,
`DriverSpec`), `floor.py`, `headtohead.py` (certification policy), `dgx_e2e.py` (stage CLI);
`scripts/_provenance.py` (`env_provenance`); `scripts/_hub_publish.py` (manifest-derived card sources);
the seven migrated modules (§5.6); the tests in §6; `pyproject.toml` (`hfjobs` group); `uv.lock`; `.gitignore`;
the three existing skills; `docs/howto/hf-jobs-training.md`, `corpus-drivers-runbook.md`; `AGENTS.md`;
ADR-122/123 supersede markers; `docs/c4/architecture.{dsl,html}`; `CHANGELOG.md`; `TODO.md`.

Exact module boundaries are fixed in the plan.

---

## 10. Commit, review, release

- One feature branch for Cycle 1. No worktrees. No subagents.
- No `git commit` or `git push` without the owner's explicit approval for that specific commit. Each
  commit is a fully tested, coherent state; no micro-commits.
- Commit shape follows ADR-037: one code commit; an artifact commit only if the rebaseline changes
  weights, produced on the code commit's clean tree.
- Independent review by reviewers A and B (spec, plan, implementation) before any commit; the author
  does not run the reviews. The tree is frozen during each review.
- Version number and release are decided at the commit gate. This spec claims none.

---

## 11. Later cycles (owner ruling 1; not deferrals)

- **Cycle 2:** `match_outcome_dependence`, `pass_completion`, `xsuccess` get real pipelines (near-twins
  of `win_probability`). The one-unit adapter stage is deleted when the trainer half of
  `_LEGACY_UNMIGRATED` is empty.
- **Cycle 3+:** corpus-driver burn-down, TF-46 first (ruling 10), then by measured wall time.
- `gk_retention` stays blocked (no non-Databricks data path), as today.

---

## 12. Risks and open technical items (each with a decision rule)

- **R1 Mid-study HPO restart** — §5.3 rule.
- **R2 Pinned environment vs 4.138.0** — §5.4 rebaseline rule.
- **R3 Dataset volumes.** Verify in implementation whether HF Jobs mounts a staged dataset read-only.
  If yes, jobs read units lazily; if no, per-shard include-pattern downloads.
- **R4 RFCDE worker scaling.** RFCDE fits are memory-bandwidth-bound (`train_ghost_gk.py:285`, `:1051`).
  `profile` measures the scaling; the planner uses the measured efficiency, not the vCPU count.
- **R5 Account concurrency cap.** At least 10 concurrent jobs were observed. `profile` and the ledger
  record queueing; the planner treats the cap as a constraint.
- **R6 AGENTS.md byte budget.** The new bullet must fit `tests/test_agents_md_budget.py`; the plan
  measures it.
- **R7 Dual launch path — RESOLVED (owner ruling 2026-10-18).** `ghost_outfield` joins Cycle 1, and the
  legacy launch path is deleted in Cycle 1. The three unmigrated public trainers run through the
  one-unit adapter stage (§5.6) until Cycle 2.
- **R8 TF-46 timing.** If TF-46 merges after the gate lands, it must comply, or join the legacy list in
  that PR as a record that cites the owner's ruling (§5.8). The gate test enforces the record shape; the
  PR review controls the growth.

---

## 13. Change log

**r2 (2026-10-18)** — responses to spec review r1 (reports in `D:\Development\_reviews\`,
`2026-10-18-compute-contract-hf-campaigns-spec-{A,B}.md`). The owner approved every disposition,
including the two A/B disagreements.

| Finding | Change |
|---|---|
| A CCHF-SPEC-01 (`require_clean_tree` 59) — A/B disagreed (B: 57) | §6.5 states both numbers: 57 callers; 59 text matches, the two extra files name it only in docstrings that explain a deliberate absence. |
| B CCHC-SPEC-01 (manifest self-certifies) + A CCHF-SPEC-03 (three-source framing) — A/B disagreed on severity; B upheld | New INV-9. §5.7 splits the artifact manifest into expectation fields (never written from an artifact) and observed fields; `adopt` gains a verify-expectations step that refuses on mismatch; the staging-time corpus expectation makes the in-job digest checkable; the framing is now "artifact vs expectation (two independent sources) + card staleness". |
| B CCHC-SPEC-02 (inner reductions) + A CCHF-SPEC-04 (importance at W > 1) | §5.2 mechanism 3 covers reductions inside `inner_parallel` units; the CI fixtures use ≥ 2 inner folds at W ≥ 2, run `importance` and `bootstrap` at W > 1, and assert completion-order non-vacuity per path. |
| A CCHF-SPEC-02 (shrink-only vs R8) | §5.8 legacy entries are `{reason, approval}` records; INV-8 and R8 updated. |
| B CCHC-SPEC-03 (red fixture) | §5.8 adds a red fixture for a module without `PIPELINE`; §7 lists it. |
| B (ruthless wording) | §2.1 says "the only compute backends". |
| B (TF-46 unverifiable) | §2.4 gives the verification command. |
