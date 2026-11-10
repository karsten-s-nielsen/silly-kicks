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

- One unit = one key, one result, one thread (INV-2). Parallelism comes only from units.
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
- Failure rule: `max_failures` (default 0) is a REDUCE-time tolerance (`_stages.exceeds_tolerance`): a run job
  records a failed unit, runs its other units and exits 0; the reduce, the reduce trigger and the driver refuse
  over it. `abort_first_k` (the first k completed units all failed = a systematic bug) aborts the job, exit 3.
- Reduce reads: a reduce site lists its generation ONCE and reads each file once. A unit work that loops over
  another stage's unit keys calls `_stages.done_results(ctx, stage)` (or `read_listed` on one listing it holds),
  never `read_result` per key: on the HF bucket mount a listing per key is quadratic (ADR-128 amendment
  2026-11-05). Both reads run over 8 threads in the caller's order (the reduce: its `reduce_order`) and print
  `[read] <stage> results <k>/<n>` on a long read.
- `timeout_s`: a stage whose one job needs more than the 7,200 s default declares its HF job limit (the
  xsuccess `hpo` and `fit`: 14,400 s); it reaches the job through the plan record.
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
killed HPO study unit restarts from trial 0 (R1, ADR-128). A failure record (`<key>.failed.json`) holds the
`error` string and a bounded `traceback`: the frames nearest the raise, then the message.

**Reading a job log.** In an HF job every covered process prints an `[alive]` start line, then one line every
300 s: `[alive] <job label> <role> pid=<pid> up=<s> unit=<stage>/<key> age=<s> at=<file>:<line> <function>
children=<pid:state:cpu+<s>:wchan,...>`. A slow unit shows its `at=` moving between lines; a stall shows the same
unit, a growing `age=` and the same `at=`. A child in state `D` with zero CPU waits in the kernel (its `wchan`
names where, e.g. a FUSE request). 600 s or more before the job timeout each process prints a `faulthandler`
dump (`Timeout (h:mm:ss)!` and every thread's stack); its start line names that time. Each process of a job
dumps in a different second (`main` at 600 s, `reduce` 601 s, worker `w<i>` at
600 + 10 × (i + 1) s, its pool child `w<i>.pool<j>` 1 + (j mod 9) s later), so the dumps of processes that start
before their dump time never share a second. A process that starts after its dump time (for example a worker
relaunched in the last minutes) dumps 1 s after it starts, so it shows its start-up stack and may share a second
with another late starter. The start line ends `numba=<version>` (`absent` would mean a broken install: numba
is a core dependency, and every kernel then raises `NumbaUnavailableError`).

Each unit prints `[unit] <stage>/<key> start` and `[unit] <stage>/<key> done|excluded|failed in <s> s`; the
driver's 2-timeout rule reads the start lines without an end line. Each run exit prints `[job] <done>/<units>
done, <failed> failed, exit <rc>, <s> s` (not for `--plan`; a kill by the HF timeout prints none). A store read
of 8 MB or more prints `[read] <rel> <MB> MB in <s> s (<MB/s> MB/s)`, and a progress line every 60 s while it
runs, so a slow read and a hung read look different. Inside an executor stage, `materialize_fileset` reads a
FileSet from the store ONCE per job (a job-local cache under a file lock; five concurrent readers of one 388 MB
FileSet through the mount took over 6,600 s where one took under 300 s): read a large upstream FileSet through it,
never per worker from the store.

The HF bucket mount can lose a rename silently (~0.3-0.4% under load; the writer still sees the file). Every
`_write_bytes_atomically` write is journaled per process (`scripts/_write_confirm.py`); at the end of each executor
worker, reduce and stage main, `confirm_job_writes` waits 30 s (0 s outside an HF job), lists, and repeats
`os.replace` from the `.tmp` part. A `[confirm] repaired <n> lost rename(s)` line in a job log is that repair.
Write store files only through `_write_bytes_atomically`/`LocalStore`, and delete them only through
`LocalStore.delete` (or call `_write_confirm.forget`), so the journal stays true. A pool child always shows
`unit=-` (the unit runs in its worker). Local runs print none of this (`scripts/hf_jobs/liveness.py`).

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

On HF the campaign driver (`scripts/hf_jobs/driver.py`, ADR-128 amendments 2026-11-02/03) sizes without a
profile: a stage is split into `ceil(units / units_per_job)` jobs capped at `max_shards` (few big jobs, not one
per unit), and each job is one full `cpu-upgrade` box (8 unit workers; an `inner_parallel` stage runs one unit
at a time with `min(8, inner_width)` inner workers). A memory-heavy stage with a `peak_rss_by_stage` hint is
sized down and capped (`--cap-bytes` + the auto-detected `rss` watchdog; `RLIMIT_AS` bounds address space and
failed every unit there); an over-cap unit is deferred and re-run solo on the whole box (`--escalate`, W=1).
CPUs and RAM come from the cgroup (`available_cpus`, `available_ram_bytes`), never the container's host view.
The Cycle 1 cost-based planner and its `profile` measurements were removed (ADR-128 amendment 2026-11-02).

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
