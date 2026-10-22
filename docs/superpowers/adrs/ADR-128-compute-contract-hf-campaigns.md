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
- Preflight runs six no-spend checks and uploads the report the launch gate requires: the commit is pushed
  and green (`check_ref`), the lockfile and the image (`check_lock`), the API contract -- the `run_job`
  signature and the flavor catalog (`check_api`), the staged datasets (`check_datasets`), every instance's
  `--plan` (`check_plans`), and the planner's estimate (stages predicted above the target are flagged).

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

## Amendment — Cycle 1b (ADR-129)

ADR-129 amends two decisions of this ADR:

- **P9 (closure identity).** The content-addressed main store (`{job}-{tier}`) names a generation by its token,
  so per-stage closure identity (`closures.lock.json`) replaces the campaign commit and the
  `cross_commit_version` as the reuse key.
- **P18 (same-commit replica).** A determinism attestation (a byte-identical main-vs-replica generation recorded
  in the main store) replaces re-running the full same-commit replica on every campaign; an attested generation
  is reused at a later clean HEAD behind a one-unit canary.

The controller gains attestation, canary, quarantine and content-addressed reuse (`campaign`/`adopt`/`store`); the
remaining Cycle 1 limitations above (quadratic reduce, per-unit ledger reads, one stage at a time, no reuse, no
halt/pre-first-unit resume, per-job install) are the ADR-129 scope. See
`docs/superpowers/adrs/ADR-129-content-addressed-reuse-attestation.md`.

## Related

- **Specs:** `docs/superpowers/specs/2026-10-18-compute-contract-hf-campaigns-design.md`
- **Plans:** `docs/superpowers/plans/2026-10-18-compute-contract-hf-campaigns-plan.md`
- **ADRs:** supersedes in part ADR-122 and ADR-123 (the launch path and the per-trainer phase flags);
  builds on ADR-037, ADR-052, ADR-056, ADR-088, ADR-123 Amendment B, ADR-126; amended by ADR-129.
