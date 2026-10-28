# How-to: write a compute pipeline (trainer or corpus driver)

Every heavy pipeline -- a trainer (`scripts/train_*.py`) or a corpus walker -- is a stage-contract
pipeline (ADR-128, spec `docs/superpowers/specs/2026-10-18-compute-contract-hf-campaigns-design.md`). The
contract lives in `scripts/_stages.py`; the executor in `scripts/hf_jobs/executor.py`. Start a new module
with the scaffolder; it passes the gate on day one:

```bash
python -m scripts.hf_jobs scaffold trainer <name>   # scripts/train_<name>.py
python -m scripts.hf_jobs scaffold driver <name>    # scripts/build_<name>.py
```

## When a module is a pipeline

The born-ready gate (`tests/scripts/test_compute_contract.py`) derives its population: every
`scripts/train_*.py` and every top-level module that calls `for_each` or declares an `ItemStage`. Each member
declares `PIPELINE = Pipeline(...)`, or sits on `_LEGACY_UNMIGRATED` (a reason and an approval; a later
addition cites a dated owner ruling) or on `_NOT_A_PIPELINE` (a reason). Running on HF also requires the
module to read only staged datasets (the I1 gate, below).

## Units

- One unit = one key, one result, one thread (INV-2). Parallelism comes only from units — UNLESS the stage
  opts in with `thread_safe=True` (ADR-128 amendments 2026-10-28 and 2026-10-29), behind a committed
  PER-STAGE cross-thread-count byte-identity arm (a `cv` stage opts in separately from its `fit`). A
  `thread_safe` worker runs at `vcpu // (workers·inner_workers)` threads via its declared
  `thread_lever`: `"omp"` raises `OMP_NUM_THREADS` (BLAS stays serial); `"n_jobs"` keeps the env serial and
  exports `SILLY_KICKS_RUNTIME_THREADS`, which the trainer reads trainer-side into the runtime `n_jobs` (never
  an env read inside the library `fit`). Every non-opted stage stays single-threaded.
- Keys are deterministic and sortable: zero-pad numeric components with `int_key`; a key never starts with
  `_` (store names such as `_summary.json`, `_inputs.json`, `_output` are reserved).
- A result is a DataFrame, a JSON-native value, or a `FileSet` (named files: a model artifact, or a
  persisted study-input directory). Results are written atomically with a checksum.

## Stages

- `Stage(name, units, work, needs=..., reduce=...)`: `units(ctx)` lists the keys; `work(key, ctx)` computes
  one result; `reduce(results, ctx)` gets `{key: result}` in `reduce_order` (default: sorted keys) and returns
  the stage output, which downstream stages read with `ctx.upstream(name)`.
- `ItemStage(name, refs, key, load, work, ...)`: units are corpus ITEMS run through `_driver.for_each`
  (ADR-052): `refs(ctx)` returns cheap references; `load(ref, ctx)` loads one item (resume-before-load);
  `work(item, ctx)` returns a tidy DataFrame or None; `counters(item, frame)` records per-item counters
  (sidecars, never in a token); `ItemExcluded` excludes an item with a reason.
- Declared inputs: `params` names the pipeline parameters the numbers depend on; `declare(params)` adds
  further declared inputs; `uses_dataset=True` keys the token on the dataset identity (`dataset_param` names
  a second dataset). A plain `Stage` with `uses_dataset=True` also declares `dataset_files(ctx)`: the
  dataset-relative files (or fnmatch patterns) its units read, `()` for none. An `ItemStage` declares through
  its refs' `files`. `cross_commit_version="<name>-<n>"` lets an extraction stage resume across commits
  (decision P9); every other stage recomputes after a code change.
- Failure rule: `max_failures` (default 0) and `abort_first_k` (the first k completed units all failed =
  a systematic bug; abort).
- `inner_parallel=True` + `inner_width`: the unit runs alone in its job and gets `ctx.pool`, an `InnerPool`
  whose `map_keyed(fn, {key: args})` runs inner tasks (e.g. CV folds) on `--inner-workers` processes.
- `Pipeline(name, module, stages, params, output)`: declaration order is the DAG order; `output` names the
  stage whose `FileSet` `--run-all --out` exports.

## Determinism rules

The four mechanisms of `scripts/_stages.py` make every output byte-identical for every (W, N): keyed units;
reduces in sorted key order; `ordered_values(results)` as the ONLY way to combine inner-pool results; fixed
thread pins in every worker (`scripts/_thread_pin.py`). In your code: no wall-clock values in a result, no
unseeded RNG, no iteration over a hash-ordered set into a result, and no thread pool inside a unit.

## Resilience

Every finished unit persists with a checksum; a kill loses at most the units in flight; a relaunch skips
every verified unit and redoes a unit whose checksum fails. A failed unit writes `<key>.failed.json` and
is retried on the next run; the campaign controller retries it until it has failed twice
(`Constraints.unit_attempts`). Then the stage's `max_failures` decides: within it the stage reduces
without the unit and names it; over it the controller halts the instance and names the unit. A job that
ends in ERROR is resumed while a retryable failed unit or (after a timeout) a pending unit remains, until
the stage's jobs have ended in ERROR twice in a row (`Constraints.error_resumes`). A
killed HPO study unit restarts from trial 0 (R1, ADR-128).

## Data

Read datasets only through `scripts/_driver_sources.py` (`staged_flat_refs`, `staged_pergame_refs`,
`dataset_dir`, `home_teams`, and the readers `read_dataset_parquet`, `read_dataset_json`,
`read_dataset_schema`). Never call `pd.read_parquet` or pyarrow on a dataset path: on an HF mount that
stalls and fails with EIO (ADR-128). The readers read each file whole under the job's read gate
(`scripts/_dataset_reads.py`; `read_dataset_schema` is a footer read that holds a slot); a mounted dataset declares its `read_concurrency` in `datasets.toml`. The
read-audit gate (`tests/scripts/test_dataset_read_audit.py`) refuses a direct read and a read that the stage
does not declare. Declare every dataset in `scripts/hf_jobs/datasets.toml` and record it in the
lock (`python -m scripts.hf_jobs datasets record <name>`). Never import a pining loader in a module a
campaign launches: the I1 gate (`tests/scripts/hf_jobs/test_visibility_blind_gate.py`) checks the static
import closure; a visibility policy arrives as an explicit parameter (ADR-123 Amendment B).

## Parity test

Register `tests/scripts/test_parity_<name>.py` on the `tests/scripts/_parity.py` harness: `run_serial`
(`--run-all --workers 1`) vs `run_fanned` (2 jobs x 2 workers per stage); assert `outputs` and `artifact` are
equal, and assert non-vacuity -- `completion_out_of_order` for a fanned ItemStage, `inner_out_of_order` for
an inner pool, `ran_concurrently` for a stage that must overlap. Mark it `@pytest.mark.slow` with its
measured cost (ADR-023). Build the toy input through `_toy_corpora`: it writes each provider's REAL id
dtypes (`PROVIDER_ID_KIND`, held to `datasets.lock.json` `provider_id_kinds` by a test), so a toy run meets
the int / str id mix a real multi-provider corpus has. A reduce that concatenates unit frames uses
`_stages.combine_frames(ids=...)`, naming its id columns.

## Sizing

The `profile` stage measures a sample of each multi-unit stage; the planner (`scripts/hf_jobs/planner.py`)
sizes N and W from the measured unit costs to meet the 60-minute target, bounded by RAM (W x peak RSS
within 32 GB). A stage whose single unit exceeds the target is reported with its irreducible floor. The
per-worker cap bounds RSS: a cgroup `MemoryMax` on the DGX, and in an HF container the executor's own RSS
watchdog (`rss` backend; `RLIMIT_AS` bounds address space and failed every unit there). The cap is each
worker's fair share of the usable RAM; a unit over it runs again alone, with the whole box. CPUs and RAM come
from the cgroup (`available_cpus`, `available_ram_bytes`), never the container's host view. The
job startup the planner subtracts from the target is the median that the committed campaign summaries
measured (`docs/research/hf_campaigns/*.json`, field `startup_s`); a committed bootstrap summary supplies the
HF-measured ~15 s startup before any campaign summary exists.

## Full utilization (ADR-128 amendment 2026-10-29, ADR-130)

The planner and controller drive the whole box; 30 minutes is a SIZING target, never a kill threshold. No
rebaseline — every artifact stays byte-identical.

- A `thread_safe` non-inner stage plans one worker (W=1), so its process gets the full vCPU thread budget;
  units fan across jobs. The `cv` stages opt in behind their own identity arms.
- An `inner_parallel` study's pool is sized to its fold width by the per-CHILD peak RSS the `InnerPool`
  children record (`UnitCost.inner_child_peak_rss_bytes`); before a child RSS is measured the plan surfaces
  `study-unsized-pending-measurement`, and a child too large for the fold width records an `inner-ram-floor`.
- The controller dispatches every ready stage in one step (a DAG, not a linear stage barrier).
- A run job's timeout is the measured `timeout_safety × wall_p99` plus the measured inline `reduce_wall` (or
  the generous profile ceiling on a first run); a job over 30 minutes is never failed.
- Declare `shardable_reduce=(ids…)` on a plain `combine_frames` concat reduce to let a heavy reduce
  (recorded p99 over the per-job budget) range-partition into `sub-reduce` jobs then a final ordered concat,
  byte-identical to the single reduce.
- Preflight caches the ~20–30 min `check_plans` toy train by its input key (ADR-130); the cheap checks and
  `check_ref` always re-observe.
- The campaign summary carries a per-stage utilization audit: each stage saturates the vCPU or records an
  irreducible floor.

## Scaffold and register

The scaffolder writes the module, its parity test, its job-registry entry (`TrainerSpec` / `DriverSpec` in
`scripts/hf_jobs/registry.py`), its toy input (`TOY_PARAMS` + `TOY_BUILDERS` in `scripts/_toy_corpora.py`)
and, for a trainer, a commented `[[artifacts]]` stub in `docs/artifacts/manifest.toml`. Replace the stage
bodies; keep the registrations. A shipped artifact needs its manifest entry and a card region set
(`tests/test_artifact_manifest.py`, `tests/test_card_regions.py`).

## Gate

`tests/scripts/test_compute_contract.py` asserts, per compliant member: a valid `PIPELINE` whose `module`
is its own; a plan on its toy input; a parity test and a registry entry. `_LEGACY_UNMIGRATED` shrinks by
migration; it grows only by a dated owner ruling.
