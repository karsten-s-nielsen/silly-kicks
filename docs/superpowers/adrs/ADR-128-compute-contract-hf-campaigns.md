# ADR-128: Compute contract (stage/unit pipelines) + HF Jobs campaigns

| Field | Value |
|---|---|
| **Date** | 2026-10-18 |
| **Status** | Accepted (renumbered from ADR-127 on 2026-10-18: TF-46 took ADR-127 on `main` @ `cd2ebcb`; re-confirmed next-free at the commit gate). Cycle 1 outcome recorded 2026-10-21: the campaign's artifacts are cut by owner ruling (see "Cycle 1 outcome") |
| **Deciders** | Karsten Nielsen (owner rulings 2026-10-18); reviewers A and B (spec r2, plan r3) |

## Context

The 4.138.0 re-fit campaign did not run on committed code. Two uncommitted scratchpad scripts launched
every variant as ONE serial HF job, because the committed ADR-123 map/reduce DAG had real-HF defects (an
xshot/xcross study list parsed from `--prep-only` stdout came back empty for a single-candidate run;
`--arms` was not threaded; the xcross probe read the `hf download` `.cache/` dotdir as a game). Each job used
one of eight `cpu-upgrade` vCPUs and took 5-9 hours (`xshot_occurrence` sc_extended 326-482 min, `ghost_gk`
`full` 424 min, `ghost_gk` 4 variants > 480 min before a resume). The in-job environment floated
(`uv pip install -e '.[train]'` resolved at job time: `uv.lock` pinned scikit-learn 1.9.0, the `full`
artifact was fit under 1.9.1), the `hf` CLI changed under the harness (1.7 -> 2.2), adoption was manual
across four files (the first artifact commit broke 12 contract tests), and no ledger recorded what ran.

The owner set three goals: determinism (the next re-fit rediscovers nothing), fan-out (a job over one hour
runs as many short jobs; per-second billing keeps the cost), and born-ready (new trainers and corpus
drivers start on the contract). All jobs run on `cpu-upgrade` only. Spec:
`docs/superpowers/specs/2026-10-18-compute-contract-hf-campaigns-design.md` (r2); plan:
`docs/superpowers/plans/2026-10-18-compute-contract-hf-campaigns-plan.md` (r3).

## Decision

Every heavy pipeline (trainer or corpus walker) declares `PIPELINE` on ONE stage/unit contract
(`scripts/_stages.py`): a static DAG of stages, each a set of keyed single-threaded units plus a reduce that
reads results in sorted key order. One executor runs W memory-capped workers per job x N `cpu-upgrade` jobs,
and every output is byte-identical for every (W, N); a committed campaign file drives a stateless controller
on the typed `huggingface_hub` API from preflight to adoption and publish.

### The contract (spec 5.1-5.3)

- `Pipeline` / `Stage` / `ItemStage` (`ItemStage` runs one corpus item through `_driver.for_each`, ADR-052).
  Units are keyed (`int_key` zero-padded; keys never start with `_`), pure, single-threaded (INV-2) and
  persisted atomically with a checksum sidecar. A reduce combines `{key: result}` in `reduce_order` (default
  sorted keys). The uniform CLI (`--plan`, `--stages S --shard I/N`, `--shard-keys`, `--run-all --out`,
  `--workers`, `--inner-workers`) replaces every per-trainer phase flag (`--shard-index`, `--fold`, `--study`,
  `--study-list`, `--list-studies`, `--assemble`, `--prep-only`, `--build-cache`).
- Tokens: a stage generation = a digest of its declared inputs -- pipeline, stage, code identity, params,
  `declare(params)`, upstream tokens, and (for `uses_dataset`) the dataset identity (an optional second
  dataset parameter left unset has none). Code identity = the run
  commit; `cross_commit_version` opts an extraction stage into resume across commits (P9); a dirty tree uses
  `dirty-<git tree hash>`. Every reduced stage writes `_summary.json` and `_inputs.json` (its token inputs:
  the job's record of what it read).
- Executor refinement over a process pool: the four determinism mechanisms of `_stages.py` (keyed units,
  sorted-key reduce, `ordered_values` as the ONLY combination of inner-pool results, fixed thread pins), a
  memory cap per worker, the failure rule (`max_failures`, `abort_first_k`), and LPT partitioning of units
  over N jobs from profiled costs. An `inner_parallel` stage owns an `InnerPool` (`map_keyed`) for folds
  inside one unit (HPO stays faithful: fold-parallel inside each seeded-TPE trial, ruling 4).
- Resilience: every stage resumes; a kill loses at most the units in flight; a checksum-failed unit is
  redone (`_driver.write_shard` sidecars, P13); a campaign resumes from the store.
- The per-worker memory cap bounds RSS, the quantity the planner sizes it from (peak RSS x 1.3). On the
  DGX that is the cgroup `MemoryMax`. An HF container has no systemd, and `RLIMIT_AS` caps address space,
  not RSS: on real HF (Cycle 1, 2026-10-20) every planned job failed with `MemoryError` at 1.4-1.7 GB RSS
  under a 2.51 GB `RLIMIT_AS` (a probe measured `VmPeak` 1.80 GB against `VmHWM` 0.53 GB for numpy +
  pyarrow). So the executor runs the `rss` backend there (`_mem_cap.detect_backend(watched=True)`): each
  poll it reads the largest single RSS in every worker's process tree from `/proc` (the cap is per
  process, as the planner measures a unit's own peak) and kills a worker over its cap. The DGX launcher
  is unchanged. The cap is each process's FAIR SHARE of the usable RAM (usable / W; usable / (inner + 1)
  for an inner-parallel unit), never below 1.3 x the measured peak: 1.3 x the profile sample's peak
  killed every heavier unsampled unit on real HF (xshot's sportec J03WR9 at 2.21 GB against 2.20 GB;
  ghost_gk's gradientsports 10506/10517 above 8.11 GB). A unit over even its fair share is recorded as
  failed for memory (`_stages.record_memory_failure`; the attempt counts though no ledger was written),
  the job's other units go on, and the controller escalates it to a job of its own: W = 1, cap = the
  usable RAM (`--escalate`). Until it has failed `_stages.UNIT_ATTEMPTS` (2) times such a unit is DEFERRED
  (`deferred_memory_units`): it is no stage failure (the job completes, so it never counts toward the
  ERRORs-in-a-row bound), the stage does not reduce (a tolerant stage would drop it), and only an
  `--escalate` job runs it. A unit that fails there too exceeds one cpu-upgrade box and is reported.
- Worker sizing reads the cgroup, not the host: inside an HF cpu-upgrade container `os.cpu_count()` and
  the CPU affinity were 64 and `/proc/meminfo` showed 503 GB available, while the cgroup allowed 8 CPUs
  (`cpu.max` 800000 100000) and 32 GB (`memory.max`). `_parallel_launch.available_cpus` and
  `available_ram_bytes` read cgroup v2 first.

- Multi-provider stage outputs (real HF, Cycle 1): gradientsports stages int ids (`game_id` int64, frames
  `player_id` Int64 and `team_id` float64), idsse and skillcorner str ids (frames `team_id` a category of
  str). Each unit stores fine; the COMBINED output failed its parquet write in the reduce. Every reduce that
  concatenates unit frames goes through `_stages.combine_frames(ids=...)`: a mixed declared id column stores
  the canonical strings the legacy grouped on (`storable_id_column`), and any other mixed column refuses by
  name (`assert_storable`, also run by `_encode`). The defect reached HF because every toy corpus carried ONE
  id type; the toys now write each provider's measured id dtypes (`_toy_corpora.PROVIDER_ID_KIND`), and
  `datasets.lock.json` records each staged provider's id kind (`provider_id_kinds`), which a test holds the
  toy profile to, so the fixtures cannot drift from the data.

### Pinned environment, launcher, preflight (spec 5.4)

- One image, digest-pinned: `python:3.12.14-bookworm@sha256:dbbe4ceb...` (P1); `uv sync --frozen` from the
  committed `uv.lock` (group `hfjobs`); fixed thread variables (`DETERMINISTIC_VARS`); `env_provenance` in
  every artifact; local verification in `.venv-hfjobs` on the same Python.
- One flavor constant `launch.FLAVOR = "cpu-upgrade"` (INV-4). The typed launcher calls `HfApi.run_job`
  with labels (`campaign`, `run_key`, `job`, `variant`, `stage`, `shard`, `commit`), mounts the run's bucket
  prefix read-write at `/scratch` (the store) and each staged dataset read-only at `/data/<name>` at its
  pinned revision. The `hf` CLI stays a human tool.
- Dataset reads go through the dataset read gate (`scripts/_dataset_reads.py`). Real-HF evidence (Cycle 1,
  2026-10-20 probes on the tc3 mount, 360 files / 15.7 GB, one job alone): whole-file sequential reads were
  clean at 1, 2 and 4 concurrent readers (184 s, 90 s, 65 s), stalled about 183 s per file at 6, and failed
  every large file at 8 with `OSError: [Errno 5] Input/output error` after 180.6 s. Pyarrow reading parquet
  directly on the mount took 126-462 s per file at only 4 readers. Cycle 1's tc3 EIO came from 8 workers
  doing that. So every staged-dataset read is ONE whole-file read under a job-wide counting semaphore over
  every worker process (lock files; the kernel drops a dead holder's lock), parsed from memory
  (`_driver_sources.read_dataset_parquet` / `read_dataset_json`). 8 workers at a cap of 4 read all 360 tc3
  files clean in 69 s. Two reads stay metadata reads, each holding a slot: the footer of one shard
  (`read_dataset_schema`) and the footer statistics of every detection-aware shard (`validate_corpus_visibility`,
  whose rare no-statistics fallback reads the whole file). Cycle 1's ghost_gk `validate` read every skillcorner
  footer directly on this mount, one reader per job, and completed for three variants.
  `_driver_sources._first_provider` (a row-group read) runs only for a dataset with no lock record, so never
  in a job. The slot is re-entrant per thread, so a read inside a held slot never waits for a second one.
  A mounted dataset declares
  `read_concurrency` in `datasets.toml` (4 for every mount); `launch.job_env` sets the smallest cap of the
  job's mounts (`SILLY_KICKS_DATASET_READ_CAP`); unset (laptop, DGX, tests) there is no limit. The read-audit
  gate (`tests/scripts/test_dataset_read_audit.py`) runs every pipeline's dataset stages on its toy input and
  refuses a direct read and an undeclared read.
- `download` access (spec R3) fetches exactly the files the job's units will read (`datasets.fetch_patterns`):
  every stage of the job that reads that dataset; the first stage takes the job's shard (a profile job only
  its `_stages.profile_sample`, single-sourced with the executor); done and excluded units fetch nothing. A
  plain `uses_dataset` Stage declares `dataset_files` (`()` when it reads none; `Pipeline.validate` refuses a
  missing declaration); an ItemStage declares through its refs' `files` plus `home_teams.json`.
  `datasets.stage_files` is the one source for the fetch and for the read-audit gate. An LPT partition names
  its keys, so it needs mounted datasets (`campaign.lpt_allowed`).
- Preflight runs six no-spend checks (five until the 2026-11-06 amendment added `check_numba`) and uploads the report the launch gate requires: the commit is pushed
  and green (`check_ref`), the lockfile and the image (`check_lock`), the API contract -- the `run_job`
  signature and the flavor catalog (`check_api`), the staged datasets (`check_datasets`) and every instance's
  `--plan` (`check_plans`, in parallel). Amendment 2026-11-05 (owner): the per-module toy `--run-all` is removed
  from `check_plans` (~15 min per preflight, and CI's parity tests already run every trainer on toy data; a
  local toy run reproduces none of the failures that stopped real HF runs), and each check prints a timed line. The
  driver's `step` reads the instances' `--plan` in parallel too (it read them one after another, minutes per
  step and silent) and prints one timed `[driver] step` line (a FATAL step prints its `FATAL` line instead).

### Campaign controller, ledger, planner, profile (spec 5.5)

- A committed campaign file (`scripts/hf_jobs/campaigns/*.toml`, INV-7) names jobs, tiers, datasets and
  variants. Run keys are derived (`{campaign}-{commit7}-{job}-{tier}`, `-r<N>` for a fresh replica store,
  P18). The controller holds no state: each step rebuilds it from each instance's `--plan` JSON and the job
  labels, then launches the next stage per instance (the stage barrier). The plan JSON names each stage's
  `pending_units`; the controller schedules (and LPT-partitions) exactly those, and costs a pending unit with
  no measured wall at the stage p50. (Cycle 1: scheduling the first `pending` keys of `units` re-launched a
  job for an already-done unit of a resumed stage, in a loop.) A failed unit is scheduled again while its
  recorded attempts (`failed_units` in the plan JSON) stay below `Constraints.unit_attempts` (2), so a
  transient failure never drops a game from a tolerant stage's reduce. When only failures out of attempts
  are left, the stage's declared `max_failures` (in the plan JSON) decides: within it (`None` = unlimited,
  the legacy "record failures and fit the rest") the stage reduces without them, and the step report and
  `_summary.json` name them; over it the action is `halt`: the step reports the units and launches nothing.
  A stage whose latest job ended in ERROR is not halted while it has a retryable failed unit (a strict
  stage's job ends in ERROR on any failed unit) or after an HF job timeout with units pending
  (`campaign.resume_after_error`): finished units persist, so a fresh job resumes the rest. Cycle 1's xshot
  sc_extended job timed out with 83 units done. Every such resume is bounded by
  `Constraints.error_resumes` (2) ERROR jobs in a row per stage, a bound that needs no unit record: a
  timeout or a crash before any unit writes none, so the attempt count alone would resume forever.
- `profile` runs a measured sample of a multi-unit stage; the planner sizes N and W from measured unit
  costs (history keyed by `(pipeline, stage, declared-params digest)`, P14) to meet the 60-minute target; a
  stage whose single unit exceeds it is reported with its irreducible floor. The ledger records every unit
  (wall, peak RSS, worker, attempt); `summary` writes `docs/research/hf_campaigns/<campaign>.json`, with
  the measured workers, inner workers, job startup (first-unit start minus `JobInfo.started_at`, recorded
  on a grouped job's first stage only) and irreducible floor per stage. The planner takes its startup from the median of those committed summaries
  (`planning_constraints`; 300 s before any).
- The job registry (`scripts/hf_jobs/registry.py`) is the single source of how each trainer runs (its
  `pipeline`, or the one-unit `adapter`) and how it is certified (`certification`, `tier_certification`,
  `cv_metric`): `public` = head-to-head floor + ADR-126 integrity gate; `owner` = CV parity on `cv_metric` +
  integrity gate; `rebaseline` = the identity comparison (P3, P15).

### Artifact manifest, adopt, publish (spec 5.7, INV-9)

- `docs/artifacts/manifest.toml` declares every shipped artifact (19 wheel + 10 Hub, incl. the two TF-46 corners bundles) with its EXPECTATION
  fields: destination, job/variant/tier, certification, dataset + corpus rule, training commit + where it is
  recorded, re-fit event, card, sibling links, declared values. `docs/artifacts/observed.json` holds the
  OBSERVED fields (sha256, metrics summary, env, campaign, adopted_at), written only by `adopt` (P2). The
  manifest replaces the scattered test constants (`_ANCHOR`, `POLICY`, `_C1`...`_C7`, `_TF65_*`, `_ADR126_*`,
  `CARD_SOURCE`, `HUB_REGISTRY`, `WEIGHTS_CLASSIFICATION`); the cross-checks stay.
- Generated card regions (`<!-- generated:<name> -->`) are rendered from the manifest and the bundles; a
  staleness test guards them; `card_bytes` strips the markers so the Hub README bytes do not change (P7).
- `adopt` (never commits): retrieve into a staging dir outside the repo; load in `.venv-hfjobs`; certify;
  check exactly five expectations (training commit = campaign commit, the recorded dataset revision = the
  pin, corpus digest + n = the staging record, shipped tier = the expected tier); bundle the allowlisted
  files (a public bundle is rehydrated with the canonical public ids AFTER its digest is verified); write the
  observed fields; copy `training_commit` from the campaign record into the manifest (the ONE field a tool
  writes there, spec INV-9); re-render card regions; write verdict JSONs; draft the CHANGELOG; run the full
  local gate; stop at the commit gate. ADR-037 guards: a clean tree at the campaign commit; a diff limited to
  an allowlist.
- `publish` (an owner go every time) uses the unchanged ADR-088 `publish_model_with_card`, then verifies
  what the Hub holds (model-only file set, remote `SHA256SUMS` = local, README = `card_bytes(card)`, a
  `from_hub` round-trip). `--sandbox` targets private `silly-kicks/sandbox-<name>` repos, created private
  before the seam runs.

### Born-ready governance (spec 5.8)

`tests/scripts/test_compute_contract.py` derives its population (every `train_*.py` and every module that
calls `for_each` or declares an `ItemStage`) and asserts each member declares a valid `PIPELINE`, plans on its
toy input, and registers a W1N1-vs-W2N2 parity test and a job-registry entry -- or sits on
`_LEGACY_UNMIGRATED` with a reason and an approval (a later addition cites a dated owner ruling), or on
`_NOT_A_PIPELINE` with a reason. `python -m scripts.hf_jobs scaffold trainer|driver <name>` writes a module
that passes the gate on day one. The DGX is an executor of the same contract (ruling 11), not a second path.

## R1: mid-study HPO restart (spec 5.3, plan Task 14)

`tests/scripts/test_study_resume_r1.py` measured whether a seeded-TPE study resumed from its sqlite store
after a mid-study kill reaches the same best parameters as an uninterrupted study. It does not. Uninterrupted
(A): `learning_rate 0.09632829061814165, max_depth 3.216968971838151, min_child_weight 9.206955354200199,
n_estimators 114.19157844870183, reg_lambda 1.4561457009902097`. Killed and resumed (B, two runs, equal to
each other): `learning_rate 0.2678866941350096, max_depth 2.232334448672798, min_child_weight
12.421185223120967, n_estimators 104.59808211767093, reg_lambda 3.540362888980227`. So `STUDY_RESUME = False`:
a study unit keeps its store in a unit-local temp dir, and a killed study unit restarts from trial 0 (the
unit is the resume grain; its result is still byte-identical).

## Engineering resolutions P1-P18 (plan r3; reviewers A and B approved)

P1 and P2 correct approved spec TEXT.

| # | Resolution |
|---|---|
| P1 (spec-text correction) | The in-job image is the digest-pinned NON-slim `python:3.12.14-bookworm`: the spec's `-slim` image has no `git`, so the job could neither check out the exact commit nor record provenance. |
| P2 (spec-text correction) | Expectations and observations are two files each: `manifest.toml` (human-written) + `observed.json` (written by `adopt`); `datasets.toml` (declared) + `datasets.lock.json` (written by the staging tool). `tomllib` cannot write TOML, and the split makes INV-9's two field classes physical. The one tool write to `manifest.toml` is INV-9's own: `adopt` copies `training_commit` from the campaign record after verification. |
| P3 | "Byte-identical": same commit = every file; across commits = every payload file, plus `metadata.json` / `metrics.json` after removing `identity.PROVENANCE_KEYS` (`sklearn_version` is not removed); `SHA256SUMS` compares on its payload lines. |
| P4 | Migrated outputs write CORPUS-scoped counters (`n_attempted`, `n_failed`, `n_counters_unrecorded` = 0, `n_excluded`) and `generation` = the extract token. |
| P5 | INV-5 fixes: the receiver takes `--corpus-visibility` / `--corpus-n-matches` from `datasets.toml`; neutral `LoadedMatch` / `resolve_cache_dir` live in `scripts/_driver_sources.py`; the I1 gate covers every launched module. |
| P6 | StatsBomb open data is pinned through `SILLY_KICKS_OPEN_DATA_REVISION` (a no-op when unset). |
| P7 | `card_bytes` strips the generated-region markers. |
| P8 | Two AGENTS.md bullets were shortened (their WHY lives in `docs/context/tracking-features.md`) to fit this ADR's bullet. |
| P9 | Code identity = the run commit; `cross_commit_version` for declared extraction stages; `dirty-<git tree hash>` on a dirty tree. |
| P10 | The registry `ghost_gk` `default` build passes `--variant default`; a `full` build passes `--variant full`. |
| P11 | Non-vacuity for `ghost_gk` permutation importance: >= 2 worker PIDs at W = 2, byte-identical importances at W = 1 and W = 2. |
| P12 | The ADR-052 resilience gate recognises `ItemStage` as `for_each` adoption. |
| P13 | `for_each` gains checksum-verified resume (`<key>.sha256`; a `.checksummed` generation marker). |
| P14 | Profile history is keyed by `(pipeline, stage, declared-params digest)`. |
| P15 | The census certifies by `rebaseline` against the committed 4.120.0 (`dd7ed4f`) census: a numeric difference is an environment delta (adopted) or a code delta (`git log dd7ed4f..<commit>` over the `input_contract()` modules is non-empty), which stops for an owner ruling. |
| P16 | Each migration ran a one-off legacy-equivalence check (pre-migration CLI vs `--run-all`, byte-identical payload SHA-256 on a toy corpus in `.venv-hfjobs`). |
| P17 | Removed: the per-trainer phase flags; receiver `--owner-rows`; census `--list-matches`. The receiver's live-pining inputs became staged-dataset parameters. |
| P18 | `--replica N` gives a fresh store for the INV-1 same-commit re-run. |

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| B. Incremental per-trainer fan-out (fix the ADR-123 DAG trainer by trainer) | Smaller first change | Every trainer keeps its own phase flags and resume rules; the next trainer starts from zero; no single parity gate | Owner ruling 2 (2026-10-18): the rediscovery IS the per-trainer variation |
| C. An external engine (Ray, Dask, a workflow service) | Mature scheduling | A second runtime in every job; no byte-identity guarantee across worker counts; HF Jobs has no native cluster | Owner ruling 2; INV-1 and INV-4 |
| A. One stage/unit contract on HF Jobs (chosen) | One parity gate, one resume rule, one CLI, born-ready by gate | The migration of every pipeline (three cycles) | -- |

## Consequences

### Positive

- A campaign runs from a committed file with zero scratchpad code; each multi-unit stage fans out to meet
  the 60-minute target; any kill resumes without redoing finished units.
- Same commit + same data = byte-identical weights for every (W, N) (CI: W1N1-vs-W2N2 per migrated module).
- Adoption is a command that produces a commit-ready diff; the scattered constants are one manifest.
- A new trainer or driver cannot merge off the contract.

### Negative

- The legacy launch path is gone: `scripts/hf_jobs/launch.py`'s `hf`-CLI phases and every per-trainer phase
  flag are deleted (consumers: none outside the files this change rewrites).
- The owner-tier `pass_completion` pining cross-check runs from `scripts/_pass_completion_pining.py` (DGX
  only), not from `train_pass_completion.py --source pining`. The entry injects the pining source into the
  trainer's `main`. The HF module never imports the pining client, so the I1 gate covers it with no
  exemption (ADR-123 Amendment B).
- Three public trainers (`match_outcome_dependence`, `pass_completion`, `xsuccess`) ride the one-unit
  adapter until Cycle 2; 38 corpus drivers stay on `_LEGACY_UNMIGRATED` until the Cycle 3+ burn-down (TF-46's
  `build_corner_routines` / `validate_corners` first, ruling 10; TF-46 merged to `main` before this gate landed).
- Resume across commits applies only to declared extraction stages (P9); every other stage recomputes after
  a code change.
- A killed HPO study unit restarts from trial 0 (R1).

### Neutral

- The census runs on the contract and certifies by rebaseline (P15).
- `ghost_gk` permutation importance keeps scikit-learn's joblib parallelism (P11).

## Cycle 1 outcome (2026-10-21)

**What ran.** Campaign `cycle1` (`scripts/hf_jobs/campaigns/cycle1.toml`) ran on real HF at the code commit:
21 instances, 145 `cpu-upgrade` jobs, 38.1 job-hours. 20 instances completed their whole DAG. The 21st,
win_probability, completed all 3961 extract units, but its extract reduce never finished: it timed out at the
30-minute floor, then at 4 h in an owner-approved hand-launched copy; a 12 h copy was cancelled when the owner cut
the cycle. The campaign found and fixed, inside the code commit, the defects that the real HF runtime showed
(the dataset read gate, the RSS memory watchdog, cgroup-aware sizing, fair-share caps with memory deferral and
escalation, bounded error resumes, failed-unit retries, and the mixed-id-type combine with realistic toy fixtures).

**Defects still open, measured on real HF.** They move to the campaign-throughput cycle (ADR-129):
- The reduce is quadratic: `_stages.read_result` lists the generation directory once per unit (about 3961 listings
  of about 8000 entries over the bucket mount for win_probability).
- The controller's `live_history` reads every unit ledger record over the bucket API: 1978 s for the 3963 units
  of win_probability (about 0.5 s per record), against 10.9 s for `--plan`.
- The controller advances one stage at a time per instance (the ghost_gk `fit` waited 36 min for `cv`), the planner
  fills every job to the 60-minute target, and instances start in a fixed order. Measured: 4.6 h wall against a
  3.8 h lower bound at `concurrency_cap = 10`; the ghost_gk chain alone is about 3.5 h.
- Results are never reused across commits: the run key contains `commit7`, which defeats `cross_commit_version`.
- A halted stage has no supported resume (the win_probability reduce needed a scratchpad launcher, against INV-7),
  and a job that fails before its first unit (a transient mount `401`) is not resumed.
- Every job installs its environment (`pip install uv` + `uv sync`) before its first unit.

**Acceptance (spec section 7).** Criterion 1 (CI gates) is met. Criteria 2 to 5 are not met by Cycle 1: no
instance was adopted, the same-commit replica and the rebaseline did not run, win_probability did not finish, and
one scratchpad launcher was used.

**Owner ruling (2026-10-21).** The Cycle 1 artifact work is cut: plan Tasks 24–28 (the replica, the rebaseline,
adopt, the sandbox publish and the artifact commit) are superseded by the campaign-throughput cycle. That cycle
fixes the defects above and runs the re-fit with a concurrent replica on the fixed code. No artifact from
`cycle1` is adopted; the shipped weights stay as released in 4.139.0. The code commit merges WITHOUT a version
bump: the published package changes only in three private modules (`tracking/_xcross_attempt_objective.py`,
`_xcross_eval.py`, `_xshot_occurrence_objective.py`, additive per-fold seams with byte-identical outputs) and one
model-card wording, so a PyPI user would get an identical package. The `cycle1` bucket stores stay (no automatic
deletion); the throughput cycle reads their unit ledgers once to seed its cost history.

**Numbering.** The campaign-throughput cycle uses **ADR-129** (claimed here; next free on `origin/main` and every
remote branch on 2026-10-21). By owner ruling (2026-10-21) it is named **Cycle 1b**, so section 11's roadmap keeps
its numbers: Cycle 2 stays the three public trainers on real pipelines, and Cycle 3+ the corpus-driver burn-down.

## Amendment (2026-11-02): the minimal driver replaces the campaign controller

**Context.** The Cycle 1 controller (`campaign.step` / `next_actions` / `planner`) never completed a re-fit end
to end. The open defects above are operational: unbounded crash handling, a planner that fills jobs to a time
target, one stage at a time, and no clean resume past a transient fault. A first rewrite on the
full-utilization branch (PR #296, PR-S204) grew to about 24,000 lines; the owner closed it as abandoned
(2026-11-02). Its code stays on `feat/controller-step-reads` for reference.

**Decision.** A minimal, stateless driver (`scripts/hf_jobs/driver.py`) is the controller, on the ADR-128 engine
as it is on `main` (D1). The package entry (`python -m scripts.hf_jobs campaign ...`) routes to it. The engine
is unchanged in behaviour: `launch`, `executor`, the trainers and the wheel are untouched, and `_stages` changes
only in two docstrings that named the deleted planner and ledger function. `campaign.py` keeps only the model
(`load_campaign`, `instances`, `read_plan`) and the operator commands `preflight` / `status` / `cancel`;
`ledger.py` keeps only `job_record`.

**How it works.**
- **Launch gate.** Every step first calls `preflight.require_preflight`: a passing preflight report for THIS
  commit and THIS campaign-file hash, or refuse. No job is spent on an unpushed, CI-red or edited campaign.
- **One pure decision.** `decide` reads each instance's `--plan` and the live job labels, walks the stages in DAG
  order, and launches the pending units of the first ready, un-reduced stage (the stage barrier). An instance
  whose `--plan` fails is reported and skipped; the others proceed. `launch --dry-run` decides on the same state.
- **Jobs.** Up to `concurrency_cap` jobs in flight and `concurrency_cap` shards per stage. Each job is a full
  `cpu-upgrade` box: 8 unit workers, or one `inner_parallel` unit with `min(8, inner_width)` inner workers. Jobs
  match on the labels main's launcher writes, `(run_key, variant, stage)`; `run_key` encodes the tier, and
  `jobs_for` scopes to one commit, so there is one generation of each stage.
- **Per-unit coverage bound.** `step` writes a per-job manifest (the units a job holds, keyed by its HF job id) at
  launch. A unit that `attempts` terminal RUN jobs of this run covered without completing is SPENT and reported.
  One bound covers a recorded failure, a worker death and a pod crash. A crash charges only the units it held
  that are not yet spent, recorded failures first (a strict stage's abort never ran the others), then its other
  live units; jobs are counted in `created_at` order. So every crash moves a live unit toward its report, on the
  keyed and the round-robin path alike. A `Job timeout` is not counted, so a unit that needs more than the
  backstop keeps riding.
- **Known limits (owner awareness).** A job record never names the unit that crashed it, so a unit can be spent
  by crashes a co-sharded unit caused, and a stale recorded failure can absorb a charge; the bound stays finite.
  A round-robin (download) slice re-runs its spent units every wave, because the executor runs every pending or
  failed unit of its slice; that costs compute only.
- **Inline reduce.** The executor reduces a stage once every unit is terminal. When no unit is left to run, the
  driver triggers the reduce with a run over an EMPTY `--shard-keys` (it runs no unit). The trigger is bounded by
  `attempts` terminal empty-manifest jobs of the un-reduced stage, a COMPLETED one included (a declined
  `try_reduce`), so it cannot relaunch for ever.
- **Download datasets.** A `download` dataset is fetched in-job by `--shard I/N`, so a download instance's run
  shards are round-robin `--shard I/N` slices, not keyed; each manifest records the slice's candidate units.
- **Retries.** Every job gets ONE HF attempt (`run_job(attempts=1)`): the driver is the only retry authority.
- **Baseline-recoverable resume.** `run` seeds every job already terminal at start into `baseline_error_ids`.
  `decide` looks past them for the FATAL stop and `step` for coverage, so a resume after the operator frees a
  fault (a storage 403) re-attempts the work.
- **Failure classification.** `failures.classify` maps a job record to FATAL (stop) / RERUN (timeout, rides) /
  RETRY (crash, bounded) / terminal by the status MESSAGE, never `started_at`, verified against real captured HF
  `JobInfo` records (`tests/scripts/hf_jobs/fixtures/real_jobinfo.json`).

**Superseded in this ADR.** The section "Campaign controller, ledger, planner, profile (spec 5.5)" and the
preflight "planner's estimate" check: `planner.py`, the profile action, the cost-history ledger, the
`plan` / `profile` / `summary` commands and `target_minutes` are deleted. Preflight runs five checks (six since the 2026-11-06 amendment).

**Owner rulings (2026-11-02).**
0. D1 is the base: the minimal driver on the ADR-128 engine as it is on `main`; PR #296 is abandoned, and its
   full-utilization machinery and wheel changes are not part of this change.
1. Timeouts are bounded by the controller only; a unit timeout rides (HFU-TIMEOUT-05).
2. The crash bound is per unit, attributed through the per-job manifests, for the current generation.
3. A FATAL is recoverable: it blocks only current work, and a resume looks past the jobs terminal at its
   start (the baseline). The same baseline applies to crashes and to the reduce.
4. Reduce timeouts are bounded (a reduce is deterministic whole-stage work).
5. Delete the old controller from this branch; it stays on the abandoned branch for reference.
6. Support `download` datasets with round-robin `--shard`, as main did.
7. Accept no per-worker memory cap and no escalation job: a unit that runs out of memory is a RETRY that counts
   toward its `attempts`; a sizing fix is a new commit.
8. HF `run_job(attempts=1)`; the driver is the only retry authority.
9. At most three review rounds per artifact; only a BLOCKING finding reopens review.

**Version.** Scripts-only. No wheel change, so the package version stays 4.139.1.

## Amendment (2026-11-03): resource-sharing overhaul

**Context.** The first retrain on the minimal driver did not complete: ~1836 jobs, 0/21 instances reduced, the
owner tier crawling. Root cause — the driver fired too many small jobs without sharing rate-limited resources
("machine-gunning"): private-dataset mount-sidecar 401 flakes, HTTP 429 throttling of the bucket API, OOM churn
on `ghost_gk` extract, and O(N)-per-step coverage reads. Spec `2026-11-03-hfu-resource-sharing-design.md`, plan
`2026-11-03-hfu-resource-sharing-plan.md` (both A+B APPROVED).

**Decision.** Five driver-only fixes (the engine — `launch` / `executor` / `_stages` / trainers / wheel — is
unchanged; `JobSpec` already carried the memory/escalate fields), all tunable in `cycle1.toml`:

1. **Bounded job-sizing.** A stage launches `ceil(units / units_per_job)` jobs, capped at `max_shards` — few big
   jobs, not one-per-unit to the concurrency cap. This is the root fix; it shrinks mounts, coverage reads and
   OOM surface together.
2. **Per-dataset mount cap.** `decide` launches at most `mount_cap(dataset)` concurrently-mounting jobs per
   dataset, counting in-flight PLUS same-pass launches, over EVERY `access=="mount"` dataset of an instance
   (an `xcross` instance mounts two). The private `tc3` sidecar is no longer stormed.
3. **Coverage set+push.** Coverage / reduce-trigger failures / per-dataset mount faults are RECOMPUTED in
   `created_at` order from a per-`run_key` read-once cache (`_partitions/_coverage/state.json`, a
   reconstructable cache) instead of re-reading every manifest each step — O(active work), not O(all terminal);
   the HTTP 429s are gone. The recompute preserves the committed charging rule (COMPLETED credits; a crash RETRY
   charges only not-yet-spent units, recorded-first).
4. **Failure attribution by LOCUS (owner ruling 2026-11-03).** A `mount_auth_flake` is a per-DATASET fault — a
   dataset over `mount_fault_budget` is reported a mount outage — NEVER a per-unit `attempts` charge; a unit
   crash / OOM / disk-evict counts per unit; a timeout rides. This separates two faults that classify both calls
   RETRY, by their reason.
5. **Memory sizing + escalation, and launch pacing + backoff.** `cap_bytes` + the auto-detected `rss` backend +
   a static per-`job/stage` `peak_rss_bytes` hint size the workers and defer an over-cap unit to a solo
   whole-box (`--escalate`, W=1) job — no OOM churn. A step launches at most `launch_batch` jobs; a contention
   signal — a 429 from `list_jobs`/`launch`, or a rising `mount_auth_flake` rate — backs off the next poll
   (bounded by `backoff_max_s`) and resets on a clean step.

**Scope.** Driver-only (`driver.py` + `campaign.py` config + `cycle1.toml`). Lands as the single amend to the
minimal-driver commit; the version stays 4.139.1 (no wheel change). Reviewed spec and plan A/B, 0 blocking.

## Amendment (2026-11-04): two validation bugs — the gen depended on untracked files; a bare mount fault FATAL-stopped

The first cycle-1 retrain on the fixed driver PLATEAUED: ~660 jobs COMPLETED but `done_units` stuck and 0/21
instances reduced. Two root causes, both found during validation:

1. **The stage generation depended on the working tree, including untracked files (the wedge).** `code_identity`
   (decision P9) returned `dirty-<git_tree_hash()>` for ANY dirty tree, and `git_tree_hash()` `git add -A`s
   untracked files. The controller runs the per-instance `--plan` in its own cwd, which accumulates untracked
   files (its `_retrain.log`, `_preflight.log`, operator probes), so the gen it computed CHANGED between launch
   and status. The HF jobs run a clean HEAD clone (`code_identity` = the commit), so the controller read a gen
   the job never wrote → `done=0` → relaunch forever. Only `cross_commit_version`-pinned extract stages (gen
   tree-independent) survived. **Fix:** a tree dirty ONLY by untracked files yields the commit (the compute
   clones HEAD and never sees untracked files, so they do not change what runs); a TRACKED modification still
   yields the content tree-hash (a dev edit still invalidates a stale store). New `_provenance._has_tracked_changes`
   (`git diff --quiet HEAD`); `git_provenance`'s keys and the artifact-provenance tree-state are UNCHANGED (that
   correctly flags untracked files for provenance). Reproduced + fixed: an untracked file no longer moves the gen.
2. **A bare `volume mount failed` FATAL-stopped the campaign.** `failures.classify` sent any unmatched
   `volume mount failed` → FATAL `mount_failed`. That is the same TRANSIENT CSI mount-sidecar class as the 401
   (the specific storage-403 / init-retries faults are matched earlier and STAY FATAL). **Fix:** → RETRY
   `mount_failed`, routed per-DATASET in `_attribute` like `mount_auth_flake` (never charges a unit); a
   persistent mount problem exhausts `attempts` and is reported, never FATAL-stops the whole campaign. (In
   practice every observed mount fault carried a 401 and already RETRIED; this closes the latent branch.)

**Scope.** `scripts/_provenance.py` (engine; `code_identity` only — a genuine determinism bug) + `failures.py` +
`driver.py` `_attribute` routing. Folded into the same amend; version stays 4.139.1 (no wheel change). Impl A/B
reviewed (partial, these fixes).

## Amendment (2026-11-05): reduce reads, the job exit rule, match-outcome as a pipeline, job liveness

The cycle-1 retrain on `f03f4753` converged on 11 of 21 instances, then stalled on timeouts and ERROR emails.
The job logs and the store gave the real causes (spec `2026-11-04-hfu-reduce-reads-and-job-exit-design.md`
§1.1, E1-E12). **The 2026-11-04 hypothesis that jobs were sized too large for the 2 h timeout was wrong**: the
`Nof4` win_probability shards held 19 units each, extracted in about 2 s per unit, then died in the reduce.

1. **A reduce read the store quadratically (E3-E6).** `reduce_stage` called `read_result` once per done unit, and
   each call listed and stat'ed the whole generation directory (11,901 entries for 3,961 units: ~47 M stats on
   the HF bucket mount, which serves ~24 small files/s). Each done unit was also read 5 times. **Decision:** a
   reduce site lists its generation ONCE and reads each file ONCE: `read_listed(store, rel_dir, key, names)`
   takes the caller's listing; `done_results(ctx, stage)` serves a unit work that loops over another stage's
   keys (census, ghost_gk, the xcross probe loop); `reduce_stage` reads and verifies every unit (INV-3) before
   it writes `_summary.json`, so a failure leaves no orphan summary. `LocalStore.names` uses `os.scandir`, and
   `read_bytes` opens directly (no stat per read). The run-job reduce path costs a constant 5 listings and
   2 reads per unit plus 2, whatever the unit count (`tests/scripts/test_stage_reads.py`).
2. **A recorded unit failure failed its job (E7, E8).** `max_failures` was enforced at run time: the parent
   killed the sibling units and exited 3, so every recorded EIO fold became an HF ERROR job and an email.
   **Decision:** `max_failures` is a REDUCE-time rule only, through one predicate `exceeds_tolerance`
   (`_stages`), shared by `reduce_stage`, `LocalExecutor.try_reduce` (refuses over tolerance without starting
   a reduce subprocess; the job exits 0) and the driver's `_reduce_or_report`. The run-time check, the worker
   mirror and the `_FAILED_DIR` markers are removed. `abort_first_k` (the first k completed units all failed)
   stays a loud exit 3 by design. A local `--run-all` still fails at `require_reduced`, now with the counts.
   The driver's bound is unchanged: it counts every terminal reduce trigger, COMPLETED included.
3. **A failure record named no site (E9).** **Decision:** `.failed.json` gains an additive `traceback`
   (`_driver.format_failure_traceback`: the 20 frames nearest the raise, their last 6,000 characters, plus the
   message cut to 1,000), for plain `Stage` units and for `ItemStage` units (`CorpusPassResult.tracebacks`).
   The `error` string keeps its exact format.
4. **`match_outcome_dependence` ran as ONE adapter unit (E10: 5,543 s for 3,961 matches).** **Decision:** it
   declares `PIPELINE` (`refs` -> `extract` ItemStage per match -> `fit`), the `train_win_probability` shape;
   `fit_artifact` writes the legacy `model.json` bytes. The adapter's `_SHARD_ROOT_TARGETS` is removed.
5. **A cv job timed out silent (E12).** A plain `Stage` printed nothing per unit, so a stall and a slow fold
   looked the same. **Decision (diagnosis only, owner 2026-11-05):** every covered in-job Python process (stage
   main, executor worker, reduce subprocess, `datasets fetch`, each `InnerPool` child, the adapter's legacy
   trainer through `python -m scripts.hf_jobs.liveness`) prints an `[alive]` start line and a line every 300 s
   (unit, unit age, the main thread's innermost Python frame, its direct children from `/proc`), and arms a
   `faulthandler` dump 600 s or more before the job deadline (`SILLY_KICKS_JOB_DEADLINE_EPOCH`, exported by
   `job_command` at bash start), in a different second per process so the dumps do not interleave (owner
   decision 2026-11-05). Third-party pool children (loky under scikit-learn) are reported by their parent's
   `children=` field (owner decision 2026-11-05; a joblib hook only if a log shows a stall inside them).
   Nothing starts without the variable, so local runs are unchanged. A fix for E12 is a new owner scope
   question once the evidence exists.
6. **The HF bucket mount loses renames silently.** Task 8 and two 12-job probes measured ~0.3-0.4% of
   `os.replace` renames lost under load; the writer still saw the file and `fsync` did not help, but the
   container's listing showed the loss within 5 s and the `.tmp` part kept the bytes. **Decision (owner
   2026-11-05, 2c):** each process journals its atomic writes (`scripts/_write_confirm.py`) and, at its end
   (executor worker, reduce, stage main), settles 30 s, lists and repeats `os.replace` from the `.tmp` part, up
   to 3 rounds; a loss it cannot repair fails the worker, which the job relaunches on the now-pending units.

**Scope.** Scripts and engine only (`_stages`, `_driver`, `_write_confirm`, `hf_jobs/{executor,driver,launch,adapter,datasets,
registry,liveness}`, three trainers, `_toy_corpora`). Folded into the same amend of `f03f4753`; the version
stays 4.139.1 (no wheel change). Spec A/B approved (round 4; the Change 4 delta approved by the owner after
rounds 5-6); plan A/B approved (round 2; the Change 4 delta after rounds 3-4); implementation A/B review as it
lands. Measured on HF before the commit (spec §6 A-D).

## Amendment (2026-11-06): the retrain launch 2 failures

Retrain launch 2 on `e9301e04` ran 270 jobs (258 COMPLETED, 12 ERROR; 9 of 21 instances done) before the driver
hung. All job logs, the HF job records and py-spy stacks of the hung processes gave five faults (spec
`2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §13.1, F1-F5). The settle-and-repair of item 6 above worked
(85 lost renames repaired in 30 jobs, 0 `LOST`), and every timed-out job left its `[alive]` lines and dumps.

7. **numba was absent in every training job (F1).** `[train]` never carried numba, so ghost_gk `importance` and
   ghost_outfield `fit` ran the numpy leaf walk (11.8x slower, ADR-076) into the 2 h timeout. **Decision (owner
   2026-11-06):** numba is a core dependency, every kernel family dispatches through
   `silly_kicks._numba_dispatch.use_numba`, and a missing numba raises (ADR-076 amendment). Each `[alive]` start
   line names `numba=<version>`; preflight's `check_numba` checks that numba is in the locked job install set (a
   `uv export --extra train --group hfjobs`, markers evaluated for Linux CPython 3.12).
8. **Five processes read one 388 MB FileSet through the mount (F2).** One reader is past the read in < 300 s;
   five concurrent readers took over 6,600 s. **Decision:** `materialize_fileset` reads a FileSet from the store
   ONCE per job: `LocalExecutor.run_stage` owns a job-local cache (`SILLY_KICKS_FILESET_CACHE`), the first
   process fills it under a `filelock` (an OS lock, released when its holder dies) through a `.part-<pid>`
   rename, and the others copy it. Store reads of 8 MB or more print `[read] <rel> <MB> MB in <s> s (<MB/s>
   MB/s)`, with a progress line every 60 s.
9. **The driver hung 2 h 10 min on `--plan` reads (F3).** huggingface_hub 2.2.0 builds its client with
   `timeout=None`, and `paginate` passes `timeout=None` explicitly, which wins over any client default.
   **Decision:** `scripts/hf_jobs/hub_client.install_timeouts` adds a request event hook that fills each `None`
   timeout value (connect 10 s, read 60 s, write 60 s, pool 10 s), so the library's backoff retries a stalled
   read. `campaign.read_plan` kills a `--plan` read after 600 s (`PlanTimeoutError`, never a rate limit); the
   plan process arms a `faulthandler` dump 30 s before that deadline, so the report holds the stack.
10. **A timed-out unit was relaunched without limit (F4).** **Decision (owner 2026-11-06), replacing
    HFU-TIMEOUT-05 ("a unit timeout rides"):** a timeout charges only the units that STARTED in that job (from
    its `[unit]` log lines, fetched once per timed-out job); a unit charged by 2 timed-out jobs is STUCK and not
    relaunched. Coverage is unchanged (a timeout still spends none).
11. **No log trail (F5).** **Decision:** the driver prints one `[driver] job <id> ... <reason> after <s> s` line
    per job that ends ERROR or CANCELED (once per run) and one `[driver] launch ...` line per launch; each unit
    prints `[unit] <stage>/<key> start` and `[unit] <stage>/<key> <outcome> in <s> s`; each run exit prints
    `[job] <done>/<units> done, <failed> failed, exit <rc>, <s> s`.

**Scope.** `silly_kicks/_numba_dispatch.py` and the six kernel families (no kernel arithmetic changes; every
parity test stays green), `pyproject.toml` (numba core, `filelock` in `[train]`), `_stages`,
`hf_jobs/{executor,driver,campaign,preflight,liveness,hub_client,__main__}`. The wheel's dependencies change;
the version stays 4.139.1 in this amend (owner decision 2026-11-06: no version bump in this amend). Folded into the same cycle amend. Spec A/B approved (round 8), plan A/B approved (A r7, B r6).

## Amendment (2026-11-07): the launch 3 verify findings

Launch 3 on `c40ec7b3` retrieved all 21 instances; verify passed 15 (spec
`2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §14.1, F6-F19). Each decision below is the owner's.

12. **An instance ran a smaller recipe than the committed one (F6-F8, F10).** **Decision:** the registry pins each
    instance's corpus-scope and search-size args (`base_args`, also on a driver), and the campaign names a pool
    dataset (`pool_dataset`, mounted like the probe). A test pins those args for every cycle1 instance.
13. **Verify checked the score, not the recipe (F9, F13).** **Decision:** one corpus mechanism from one source.
    `datasets record` writes `sb_open.identity.open` (the `_corpus_identity` digest of the pinned refs) and the
    80 pinned competitions; each event-only bundle carries a `metrics.json` with that identity; verify compares
    the two strings with the manifest `digest` rule. Every certified artifact has a corpus-scope check (a gate
    derives the population and asserts it exactly). A rebaseline counts a code delta only where a module's AST,
    docstrings removed, differs between the two endpoints.
14. **Public certification (F6, F7, F11, F14).** **Decision:** public = the head-to-head, recorded as
    `in_sample_parity` (its rows lie inside both training sets), AND two-sided CV parity on the registry
    `cv_metric` within 5 % of the committed value (match-outcome `rho` within `1e-6`), plus the ADR-126 gate for
    xShot/xCross. A wheel with no `metrics.json` names its committed CV record (`cv_reference`). The tier policy
    is the caller's (`headtohead --certification`); an xShot/xCross public run is scored on its own extract
    rows, and the open-data builders read the locked revision.
15. **An HF job read open data from GitHub master (F14, F15).** **Decision:** a staged, read-only snapshot of
    the pinned commit (`silly-kicks/hf-refit-sb-open`) is the job's open-data cache; a miss in a job raises,
    never fetches. The lock records the snapshot and refuses one of another revision.
16. **Long single units were silent (F16-F18).** **Decision:** per-trial study lines, a timed census-bootstrap
    line on both paths, `[reduce]` and `[retrieve]`/`[verify]` lines, `instances done` on the driver step line;
    counters are read once per listing, in parallel, merged in key order.
17. **Two environment deltas (F19, I1, I2).** **Decision:** optuna is locked to 5.0.0 (seven of eight committed
    tuned models reproduce under it; xcross `public` `default` and `xsuccess` were tuned under 4.9.0 and certify
    without a parameter match). The back-line tie-break is stable at its source, so x86 and ARM agree.

**Scope.** `scripts/hf_jobs/{registry,campaign,datasets,adopt,floor,headtohead,launch,driver,executor,stage,
adapter,manifest}`, the four event-only trainers, `_sb_open_data`, `_stages`, `_crossed_icc`, `_study_shared`,
`silly_kicks/match_outcome/_dependence.py` (`load` takes `str`), and the stable sorts in
`silly_kicks/tracking/{_defensive_line,_collective,_ghost_outfield}.py`. The version stays 4.139.1 in this
amend. Spec A/B approved (A r15, B r15); plan A/B approved (A r13, B r12).

## Amendment (2026-11-08): the launch 4 findings

The owner stopped launch 4 on `22e0b65d` at 14/21 instances with no unit failed (spec
`2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §15, F20-F25). Each decision below is the owner's or was
made at the owner's direction.

18. **Converter noise filled the job logs (F20).** **Decision:** `launch.job_env` sets one `PYTHONWARNINGS`
    value that ignores the three known per-match `UserWarning`s (`Inferred xy_fidelity_version`, `Inferred
    shot_fidelity_version`, `StatsBomb`) from `scripts._sb_open_data` only. Local runs are unchanged.
19. **Long units printed only the 300 s beat (F21).** **Decision:** every optuna study passes the per-trial
    callback (`TrialPrinter.optuna_callback`), and the win_probability, pass_completion and xsuccess CV loops
    print `[progress] <trainer> fold <k>/<n> in <s> s`. An AST gate (`test_no_blind_study`) fails on any
    `.optimize(` call under `scripts/` without `callbacks=`.
20. **A 429 on a driver store write ended the driver (F22).** **Decision:** `_guarded_put` retries a contention
    error at 2, 4 and 8 s, then ends the step's launches, as a launch 429 does. A skipped shard-key write skips
    its launch; a lost per-job manifest is reported and looked up again; an unsaved cache does not lose the step.
21. **A long input read was silent (F23).** **Decision:** `done_results` reads over 8 threads, returns the
    results in sorted key order, and prints `[read] <stage> results <k>/<n> in <s> s`. The bootstrap always
    prints a final line.
22. **One xsuccess HPO unit could not fit the 2 h job limit (F24).** **Decision:** a stage may declare
    `timeout_s` (default 7,200 s); the xsuccess `hpo` stage runs each trial's 5 folds on the InnerPool (the
    xcross fold pattern, job-local memory-mapped arrays) and declares 14,400 s. The search is unchanged; the
    escalation rule is unchanged.
23. **Every retry reloaded 3,961 matches (F25).** **Decision:** `pass_completion` and `xsuccess` declare native
    pipelines on the win_probability pattern (`refs` → cached per-match `extract` → HPO/CV → `fit`; the shared
    open-data stage helpers are `scripts/_open_data_stages.py`). The trainer half of `_LEGACY_UNMIGRATED` is
    empty, so the one-unit adapter stage (`scripts/hf_jobs/adapter.py`), its liveness role and its tests are
    deleted. The pass_completion pining CLI (`pining_main`) stays and shares the fit (`fit_bundle`); xsuccess
    drops its legacy CLI and `--competitions`.

**Scope.** `scripts/hf_jobs/{launch,driver,registry,campaign,liveness}`, `scripts/_stages.py`,
`scripts/_crossed_icc.py`, `scripts/_study_shared.py`, `scripts/_open_data_stages.py`, the three trainers,
`scripts/_pass_completion_pining.py`, `silly_kicks/xsuccess/_objective.py` (fold split and combine; the serial
result is unchanged). The version stays 4.139.1 in this amend. Spec and plan A/B approved (A r18, B r18).

## Amendment (2026-11-09): the launch 5 findings

Launch 5 on `652dede9` finished 20 of 21 instances; xsuccess `fit` timed out twice (spec
`2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §16, F26-F29). Owner: amend the fixes into the cycle commit
and rerun all 21 instances; nothing from launch 5 is adopted.

24. **The xsuccess `fit` could not fit 2 h (F26).** One calibrated fit is 6 xgboost fits; the 5 CV folds and the
    final fit ran single-threaded for ~10,000-14,000 s. **Decision:** the `fit` stage runs them as InnerPool units
    (width 3: a measured 7.26 GiB peak per unit, 24 GiB budget; the final fit first), combined in key order so the
    result equals the serial one, with `timeout_s` 14,400 s.
25. **A large reduce was silent (F27).** **Decision:** `reduce_stage` reads through the `done_results` reader (8
    threads, the caller's order, the `[read]` line).
26. **The driver rate-limited itself (F28).** One `--plan` made 25-65 API requests, mostly repeated listings.
    **Decision:** a run re-plans only unreduced instances; a `BucketStore` lists its run key once, recursively (never
    with an empty prefix), and memoizes reads with write-through. Measured: 47/62/65/25 -> 16/16/17/4 requests.
27. **The visibility arms dropped the Bundesliga matches (F29).** `arm_policy` keyed them by the stem provider
    (`idsse`), the trainers by `source_provider` (`sportec`), so every xShot/xCross arm missed 7 matches since this
    ADR's migration. **Decision:** `compute_arms` takes the staged units and builds each key with the trainers' own
    `match_key`/`pergame_name`; a test compares it with the trainers' refs on every locked unit.
28. **The long pole started last.** **Decision:** xsuccess is the first instance of `cycle1.toml` (a step launches
    in instance order).

**Scope.** `scripts/hf_jobs/{arm_policy,campaign,driver,store}`, `scripts/_stages.py`, `scripts/train_xsuccess.py`,
`scripts/hf_jobs/campaigns/cycle1.toml`. The version stays 4.139.1 in this amend. Spec and plan A/B approved (A
r20/r18, B r20/r17).

## Amendment (2026-11-09): the launch 6 verify findings

Launch 6 on `0552f58e` finished 21 of 21 instances; verify passed 16 and failed 5 (spec
`2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §17, F30-F32). Owner: amend the fixes into the cycle commit and
rerun all 21 instances; nothing from launch 6 is adopted.

29. **A public trainer recorded ids, not the digest (F30).** Both trainers passed
    `all_public=bool(inp.is_public.all())`; once F29 made the public arm complete they recorded per-game names, which
    verify's digest check and `adopt`'s `rehydrate_public_corpus_ids` do not accept. **Decision (owner):** digest only.
    Both trainers pass `all_public=False`; an AST gate (`test_in_job_trainers_record_the_corpus_digest`) holds it;
    `adopt` rehydrates the canonical public ids off-HF.
30. **The in-sample gate failed a build without a parameter match (F31).** The xCross default build's committed
    params came from another tuning run (item 17), so the in-sample head-to-head favours them. **Decision (owner):**
    `VariantBuild.in_sample_gate` (default `True`); the xCross default build sets `False`: CV parity certifies, the
    head-to-head is recorded with `"gated": false`.
31. **The census rebaseline failed on platform float drift (F32).** Launch 6 differed by `tier2.icc` 3.1e-15
    relative, `tier2.lo` 4.5e-6, `n_boot_effective` 1 and 4,094 ranking values up to 1.27e-14 relative.
    **Decision (owner):** `DriverSpec.tolerance` (the census only); a rebaseline within it passes with
    `"within_tolerance": true` and the measured differences (`adopt._within_tolerance`; one note per parquet column):
    numbers 1e-9 relative, `tier2.lo` 1e-5 absolute, `n_boot_effective` 1, everything else exact; the P4 pass counters
    are not compared, and the fresh failure counters must be 0.
32. **Retrieve and verify ran one instance at a time.** **Decision:** both run 4 instances at once over threads;
    the result and the report stay in instance order; each statsbomb_open head-to-head gets its own open-data cache
    (`<staging>/_open_data/<instance>`). `cycle1.toml` `launch_batch` 16.

**Scope.** `scripts/train_{xshot_occurrence,xcross_attempt}.py`, `scripts/_corpus_identity.py`,
`scripts/hf_jobs/{adopt,floor,registry}.py`, `scripts/hf_jobs/campaigns/cycle1.toml`. The version stays 4.139.1 in
this amend. Spec and plan A/B approved (A r23/r21, B r23/r20).

## Amendment (2026-11-10): adopting launch 7

Launch 7 on `a1a3b52f` finished 21 of 21 instances, and verify passed all 21. `adopt` then refused (spec
`2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §18, F33-F34). Owner: fix `adopt`, then adopt launch 7 with the
fixed module; no rerun.

33. **`adopt` wrote a manifest commit for an artifact that records none (F33).** `match_outcome` and
    `expected_passing` have no `commit_field`, so their entries have no `training_commit`, and `adopt` refused.
    **Decision:** `adopt` writes `training_commit` only for an artifact with a `commit_field`; a manifest test pins
    "`training_commit` exactly when `commit_field`".
34. **A refusal left files behind (F34).** **Decision:** `adopt` refuses at entry when the root's `git status` is not
    empty (the provenance check reads the working directory, not the root), and runs its write phase under a
    rollback: on any `BaseException` it restores every changed path (a tracked path from HEAD, a new file removed),
    and a restore that fails raises `AdoptionRollbackError` naming the paths. The local gate runs after the write
    phase, outside the rollback.
35. **The adoption is a new commit.** An amend would replace `a1a3b52f`, the commit every adopted artifact names. The
    adoption and this fix are one new commit on top of `a1a3b52f`, the cycle's final commit; PR #299 merges without
    squashing, so `a1a3b52f` stays reachable from `main`. The fixed `adopt.py` runs from a copy outside the repo
    against the clean tree at `a1a3b52f`; the copy's sha256 is recorded in the cycle ledger.
36. **The xCross probe ids changed form (F35).** Since the migration the probe cohort recorded the staged per-game
    name (`gradientsports-10502`), not the canonical id (`10502`), in `probe_sample_matches` and the
    `probe_sample_in_training_folds` keys. **Decision (owner):** the trainer records canonical ids
    (`_corpus_identity.canonical_match_id`: a leading provider or identity prefix stripped, any other id unchanged),
    and adoption finalizes launch 7's records off HF (`canonical_probe_ids`) into ONE copy per instance,
    `<staging>/_final/<instance>/`, which the wheel bundle, a Hub artifact's `observed.json` entry and `publish` all
    read. The re-adoption loads exactly the two changed modules, `adopt.py` and `_corpus_identity.py`, from
    sha256-checked copies.
37. **The bundle's own CV record replaces `cv_reference` (F36).** Adoption adds a `metrics.json` to `expected_passing`,
    `match_outcome` and `xsuccess`, the three artifacts whose CV baseline came from a manifest `cv_reference`.
    **Decision (owner):** ship it; the bundle's record is the next campaign's baseline; the three `cv_reference`
    entries and `docs/artifacts/cv_reference/xsuccess.json` are removed (the field stays for a bundle without one).
    `match_outcome` and `expected_passing` record their training commit in the bundle, so the manifest declares it
    (`commit_field`) and checks it.

The launch also stopped once on the org's private storage limit: the scratch bucket held the run keys of 13
amended-away commits. They were deleted after the owner's go; the owner will not raise the limit for leftovers.

**Scope.** `scripts/hf_jobs/{adopt,manifest}.py`, `scripts/_corpus_identity.py`, `scripts/train_xcross_attempt.py`,
`docs/artifacts/manifest.toml` (comments, three `cv_reference` entries, two `commit_field` entries),
`docs/artifacts/cv_reference/xsuccess.json` (removed), the adopted artifacts. The adoption changes the wheel's
bundled weights, so the release is 4.140.0 (owner 2026-11-10). Spec and plan A/B approved (A r27/r25, B r27/r24; §18.6 A r29/r27, B r29/r26).

## Amendment (2026-11-10): xsuccess fit units, the ball-carrier pre-index, campaign hygiene

Spec `docs/superpowers/specs/2026-11-10-hfu-fit-units-carrier-index-hygiene-design.md` and its plan (A and B
approved both). The owner set the scope on 2026-11-10 after a read-only measurement of launch 7 (388 jobs, 1,810
job-minutes). `cpu-upgrade` costs $0.03 per hour, so that launch cost about $0.90. **The measure that decides:**
an HF change is ranked by wall time per launch, by failed launches and by storage; never by job-minutes.

38. **The xsuccess `fit` held 36 xgboost fits in one unit (63 min of the long pole).** One calibrated fit is 6
    xgboost fits, and the stage runs 6 calibrated fits. **Decision:** each xgboost fit is a stage unit. The grid
    is fixed (`FIT_UNITS`: `final` and `fold0` to `fold4`, each with `main` and `cal0` to `cal4`); a unit that the
    corpus does not have returns `{"absent": true}`. Every unit and the reduce work in the pooled row space
    (`_cv_splits(pooled)`), and the reduce assembles the bundle and never loads a booster. `XSuccessModel` composes
    the same private steps (`_model._training_arrays`, `_fit_booster`, `_calibration_splits`,
    `_calibration_fold_proba`, `_calibration_oof`, `_isotonic_thresholds`, `_raw_predict`, `_calibrate`,
    `_metadata_payload`, `_write_metadata_and_sums`), so the library and the trainer give the same bytes. The
    InnerPool path, `cross_val_metrics` and the two `[progress] xsuccess` fit lines are removed; each unit prints
    its `[unit]` lines.
39. **A stage can have its own job size.** `units_per_job_by_stage` (a `"job/stage"` map in the campaign file)
    replaces `units_per_job` for that stage. `cycle1.toml`: `"xsuccess/fit" = 3` with a peak-RSS hint of 8 GiB (the
    7.26 GiB of a whole calibrated fit bounds one unit), so 12 jobs of 3 units run all 36 fits at once.
40. **"Identical across commits" has one definition.** `identity.cross_commit_differences`: `metadata.json`,
    `metrics.json` and the file that the artifact's `commit_field` names are compared without `PROVENANCE_KEYS`;
    `SHA256SUMS` on the lines of the other files; `MODEL_CARD.md` after each 40-hex commit is replaced; every other
    file byte for byte. `artifact_size_bytes` is a provenance key (it counts the bytes of files that hold the
    commit and the job platform string). `campaign <file> compare --staging NEW --baseline OLD` applies the rule to
    two retrieved staging folders and prints the census verdict of NEW.
41. **A launch that changes no model is a parity proof, not a re-fit.** When `compare` finds every artifact
    identical to the shipped launch and the census rebaseline passes, nothing is adopted and nothing is published:
    the shipped artifacts keep their `training_commit`. A difference is a defect of the change; it is never
    adopted.
42. **Launch 7 stopped on the org storage limit.** **Decision:** preflight check 7, `storage` (`preflight.check_storage`): the org's
    buckets plus `usedStorage` of its private datasets and models must leave `storage_headroom_bytes` (20 GB) under
    `storage_limit_bytes` (100 GB); a count that cannot be made fails the check. `campaign <file> prune` lists the
    run keys of every commit other than HEAD; `prune --delete --i-have-owner-approval <commit7>` deletes the named
    commits' run keys. It refuses HEAD, a commit with a non-terminal job and a commit without a run key, and it
    touches only keys that are exactly a run key of a job of the campaign.
43. **A result kept its marker and lost its files.** Over the limit the small `.fileset.json` marker landed and
    the members did not, so the plan read the unit as done. **Decision:** `plan()` counts a `FileSet` result as
    present only with every member listed: such a unit is `pending` and named in `lost_units`; a lost stage output
    sets `lost_output`; either makes the stage not `reduced`. The driver reruns the unit (or the reduce) within its
    attempts and then reports `result lost` or `output lost`, never a crash. In-job code is unchanged.
44. **`post_verify` could not load an artifact without a Hub class.** **Decision:** an artifact with a `hub_class`
    round-trips with `from_hub`; one without (`hub_xsuccess`) is downloaded to `<staging>/_post_verify/` (emptied first) and
    loaded with the job loader, so only the files that the Hub holds are loaded. A manifest gate ties `hub_class` to the loader.
45. **`retrieve` made up to 32 concurrent reads.** **Decision:** one bound of 8 store reads for the whole call; an
    instance's `--plan` subprocess holds one slot of it.
46. **A repaired unit stays STUCK in the same driver run** (the baseline rule). **Decision:** every `STUCK` line
    ends with the operator path: fix the cause, stop the driver, then run `resume`.
47. **The deadline dump could end the process it reports on.** The job watch armed
    `faulthandler.dump_traceback_later` 600 s or more before the job deadline (the 2026-11-05 amendment). That
    watchdog is a C thread that reads the frames of a running thread without the GIL. Measured on HF
    (2026-11-10, jobs `6af366809737d9a9e0914f3a` and `6af3700bab34849891acb4a8`, Python 3.12.14, 40 processes
    per workload, a dump every 0.05 s for 3 s; production prints ONE dump per process): ended by a signal
    (SIGSEGV; one SIGBUS) were 18 of 40 processes during an xgboost fit, 38 of 40 during
    pandas work, 40 of 40 during `infer_ball_carrier`, 40 of 40 during a numba compile, 36 of 40 during the
    numba import; 0 of 40 while the main
    thread slept, ran pure Python calls or ran a compiled numba kernel. Launch 7 printed no dump (no job came
    within 600 s of its timeout), so no job was lost. **Decision:** a daemon Python thread prints the dump
    (`liveness.stack_dump`: every thread's Python stack, the main thread first, in the `faulthandler` line
    format); it holds the GIL while it reads the frames. The dump times of the processes of one job do not
    change. Every `[alive]` record and the dump are ONE write each, so a record of another thread cannot land
    inside it. **Proof** (HF job `6af378cb9737d9a9e0917f86`, the same 8 workloads, the new `stack_dump` printed
    every 0.05 s by a Python thread): 0 of 320 processes ended by a signal, over about 14,500 dumps. **Cost:** while a C extension holds the GIL in one long call, the dump waits until that call
    returns; the `[alive]` lines have the same limit. The plan-timeout dump of `scripts/_stages.py`
    (controller side, 30 s before the driver ends a `--plan` subprocess; the 2026-11-06 amendment) is the same
    dump now (`liveness.start_dump_thread`), and `campaign._plan_timeout_detail` reads its header; no
    `faulthandler` call is left in the campaign code.
48. **The baseline of a parity proof is the shipped form** (launch 8; amends item 40). `compare` against the
    retrieved files of launch 7 gave 4 DIFFERENT, all on the xCross `metrics.json`: the launch-7 jobs wrote
    the staged probe names (`gradientsports-10502`), and the trainer writes the canonical ids since the
    adoption of launch 7 (the amendment above; HFU spec 18.6). Launch 8 was the first launch with that
    trainer. The adopted copy of launch 7 (`<staging>/_final/`, the files that `adopt` finalized and `publish`
    uploaded; for xCross `public`, the bundled `metrics.json` holds `corpus_match_ids` where the adopted copy
    holds their digest and their count) holds the canonical ids. **Decision:** `compare` reads the baseline's adopted copy
    (`adopt._final_dir`, which `_finalize` writes) when the baseline has one for the instance, its retrieved
    files otherwise, and each line names the form. The new side is always its retrieved files: a job must
    write the shipped form itself, so a trainer that falls back to another form still shows as DIFFERENT.
    Launch 8 against launch 7 under this rule: 20 of 20 trainer artifacts identical, the census verdict ok.
49. **The driver measures its Hub requests.** The request count of a `--plan` was measured by hand in two
    cycles. The script of the first was not kept, and the bound that this cycle's spec wrote counted a store
    read as one request. Measured on launch 8 (huggingface_hub 2.2.0): a store read is two HTTP requests
    (`POST paths-info`, `HEAD resolve`; the bytes come over xet), and a listing is one request per 1,000
    files. One plan of a finished instance: win_probability 20, xsuccess 21, the census 21, ghost_gk/default
    12 requests (16, 16, 17 and 4 before item 43, which adds one marker read per `FileSet` result).
    **Decision:** (a) the bound of a plan is in store operations, which the code controls and a test binds:
    one recursive listing and one marker read per `FileSet` result; the HTTP counts are a measured fact of
    one library version. (b) The bounded client counts every request (`hub_client.count_request`); a
    `--plan` process of a bucket store prints `[plan] hub requests <n>` on stderr; `campaign.read_plan` adds
    it to the controller's tally; the step line prints `hub requests <total> (plans <p>)`, and a
    rate-limited step prints its count. A plan that fails prints its count before it ends, and `read_plan`
    adds it before it raises, so a 429 inside a plan is in the count; a plan that timed out was killed and
    reports none (the count of that step is a lower bound). (c) No request is removed: launch 8 had 208 steps and no
    rate-limited step. A reduction needs a measured need, and the step line now gives the measure.

**Launch 8** (`abcfda86`, 2026-11-10 to 2026-11-11; spec §17): 397 jobs (394 COMPLETED, 3 `mount_auth_flake`,
each relaunched), 21 of 21 instances reduced in 3 h 43 min. The xsuccess `fit` stage: 36 units in 12 jobs,
19 min 36 s from launch to reduced (63 min at launch 7), peak RSS 6.02 to 7.26 GiB per unit. `verify` 21 of 21
ok; `compare` as in item 48. Nothing was adopted or published: the shipped artifacts keep
`training_commit = a1a3b52f`.

**The launch-7 measurement that set the scope** (spec §1.2, the same ids; launch 7 on `a1a3b52f`, 388 jobs):

| Id | Measured |
|---|---|
| M1 | Wall time 4 h 35 min, with a storage stop. The long pole is xsuccess: `refs` + `extract` 25 min, the `hpo` job 143 min (one unit, 8,470 s for 30 trials), the `fit` job 63 min (one unit, 3,685 s). Every other instance ends inside that window. |
| M2 | Scheduling wait: median 4 s. Job setup (clone + `uv sync`): median 19 s, 152 job-min in total. |
| M3 | A job spends about 64 s outside its unit. Cause read from code, not probed: `_write_confirm.JOB_SETTLE_S = 30`, once in the worker and once in the reduce subprocess. |
| M4 | 15 stage generations were reduced by several jobs: 7,284 of 8,374 reduce seconds. All ended. |
| M5 | The scratch bucket held 23.3 GB (16.1 GB after the launch-6 run keys were deleted); one campaign is 7.3 GB. The private datasets held 28.4 GB, and 22.07 GB at a second measurement the same day. |
| M6 | The static import closure of every pipeline module holds the same 178 files (157 core library files, 21 runner files); each pipeline adds 2 to 13 files. |
| M7 | Launch 6 against launch 7 under the rule of item 40: 16 of 21 artifacts identical; the 5 differences are `metrics.json` of the four xShot/xCross `public` artifacts and `census.json`. |
| M8 | `_pre_index_frames` takes 13.5 to 58.0 s per match on HF (audit 2026-10-30): 58 s of a 115 s ghost_gk extract unit at 30 Hz. |

**The scope decisions** (owner 2026-11-10; spec §1.3; the item numbers are the handoff's):

| Handoff item | Decision | Reason |
|---|---|---|
| 4 xsuccess `fit` as stage units | IN (items 38, 39) | `fit` is 63 min of the long pole (M1). |
| 10a ball-carrier pre-index | IN (Scope, below) | It serves every `infer_ball_carrier` caller (M8). A wheel change is accepted. |
| 8 read bound in `retrieve` | IN (item 45) | HTTP 429s during retrieve. |
| 11 `post_verify` without a Hub class | IN (item 44) | `xsuccess-v1` was verified by hand. |
| 12 storage: preflight check and prune | IN (item 42) | Launch 7 stopped on the org storage limit (M5). |
| 13 FileSet member check | IN (item 43) | Over the limit, a result kept its marker and lost its members. |
| 14 STUCK operator path | IN (item 46) | A repaired unit stayed STUCK until the driver restarted. |
| 15 card citation gate, `.gitattributes` | IN | The open `TODO.md` row. |
| The HFU spec §14.10a record | IN | Owner-approved text. |
| 1 reuse of stage results across commits | DROPPED | M6: a sound derived closure reruns everything on any runner or core-library edit, where campaign fixes land. PR #290 and the ADR-129 stub stay untouched. |
| 2 job setup: prebuilt image, `uv` wheel cache, `xgboost-cpu` | DROPPED (the whole item) | M2: at most 11 s per job. |
| 3 stage chaining; the settle time | DROPPED | About 8 and 7 min, off the long pole (M1, M3). The settle guards against lost writes. |
| 5 hpo startup batch; 6a hpo startup over jobs | DROPPED | M1: about 8 min of 143; the TPE phase has no byte-identical speed-up. |
| 6b `fit` units over jobs | Delivered by item 4 | The driver shards stage units. |
| 7 reduce lease | DROPPED | M4: no wall time and no measured harm. The bucket mount gives no reliable lock. |
| 9 extraction and launch dedupe (all four parts) | DROPPED | M1: no part is on the long pole. |
| 10b per-frame feature loops | DROPPED | No wall time (M1); a float bit risk. |
| D1-IMPL-09, -11, -12 | Accepted limits | The "Known limits" of the 2026-11-02 amendment; no measured cost at launch 7. |

No `TODO.md` row is opened for a dropped item (owner).

**Scope.** `scripts/train_xsuccess.py`, `silly_kicks/xsuccess/_model.py`, `silly_kicks/tracking/_ball_carrier.py`
(the placement part of `_pre_index_frames`, pinned to the former function by an oracle test),
`scripts/_stages.py` (`plan`), `scripts/hf_jobs/{adopt,campaign,driver,identity,preflight,prune}.py`,
`scripts/hf_jobs/campaigns/cycle1.toml`, `scripts/hf_jobs/liveness.py` and `arm_plan_deadline` of `scripts/_stages.py` (item 47),
`scripts/hf_jobs/hub_client.py` and the `--plan` exit of `scripts/_stages.py` (item 49). Items 48 and 49 are a
second commit on top of the cycle commit, which the launch-8 jobs cloned and which is therefore not amended.
The second commit also holds one test-only fix: `tests/scripts/conftest.py` empties the `_write_confirm`
journal around every test (the cycle spec, §17.5).
The wheel changes (library code); no bundled artifact changes.

## Related

- **Specs:** `docs/superpowers/specs/2026-10-18-compute-contract-hf-campaigns-design.md`
- **Plans:** `docs/superpowers/plans/2026-10-18-compute-contract-hf-campaigns-plan.md`
- **ADRs:** supersedes in part ADR-122 and ADR-123 (the launch path and the per-trainer phase flags);
  builds on ADR-037, ADR-052, ADR-056, ADR-088, ADR-123 Amendment B, ADR-126.
