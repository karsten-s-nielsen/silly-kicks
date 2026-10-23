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
  its refs' `files`. Resume across commits is sound by CLOSURE identity (ADR-129): a clean-tree stage token
  keys on `stage_code_identity` (the stage's locked closure files, data members, `uv.lock`, the image digest,
  the Python version and the output-affecting environment values), not the commit, so an unrelated commit
  reuses the result and a change to a closure file recomputes exactly the stages that read it.
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
measured (`docs/research/hf_campaigns/*.json`, field `startup_s`); before any summary exists it is 300 s.

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

## Content-addressed reuse (ADR-129, Cycle 1b)

The main store is CONTENT-ADDRESSED: `campaign.run_key(c, commit, job, tier)` is `{job}-{tier}` (no campaign or
commit), so a generation named by its token is shared across commits and campaigns; a replica (`-r<N>`) and a
canary (`-canary`) stay fresh.

Prove determinism and reuse it:

- **Attest** with `campaign … resume --with-replica <N>` (steps the fresh replica beside the main run) then
  `campaign … verify --with-replica <N>`: a byte-identical main-vs-replica generation (`adopt.generation_diff`,
  run history excluded) writes `_attest/{stage}/{token}/{commit}.json` to the main store. An attested stage is
  skipped in the replica.
- **Canary** with `resume --canary` + `verify --canary`: for each attested stage the controller runs ONE unit
  (`canary_key`) in the `-canary` store, reading the reused upstream through `CanaryReadThroughStore` (allow file
  under `_campaign/.../canary/`), and `verify` compares it to main byte for byte. A mismatch quarantines the
  generation (`BucketStore.quarantine_generation`) and records an `unsound_reuse` finding.
- **Adopt** at a later clean HEAD: `adopt` reuses an attested generation whose token still equals HEAD's token
  (`adoption_refusals`), stamps each commit-free output from its oldest ancestor attestation, and records
  `adopted_at_commit`/`attested_by` in `observed.json`.

Every `verify --with-replica`/`--canary` call also records its byte-compares in
`_campaign/{campaign}/{commit7}/comparisons.json` (`adopt.comparisons_rel`). A key is `replica|{N}|{instance}|{stage}`
or `canary|{instance}|{stage}`, and a value is `true` (byte-identical) or `false` (mismatch). Each call MERGES its
results into the file, so a later call never erases an earlier mismatch. Only a stage that materialized on both
sides is recorded: an attested stage that the replica skipped is not a compare.

`campaign … summary` writes the S7.2 acceptance bars (`compute_bars`, `BAR_THRESHOLDS`) into the campaign summary.
The determinism bar reads `comparisons.json` (`adopt.read_comparisons`): no recorded compare is NOT a pass.

## Cycle 1b operations (ADR-129 quick reference)

- **Closure lock** — a stage's reuse key is its closure identity (`closures.lock.json`). Regenerate it after a
  stage's code or inputs change: `python -m scripts.hf_jobs closures record`; the gate `test_closure_lock`
  refuses a stale lock.
- **Stage inputs** — a stage declares the dataset files it reads (`data_files=`) and any Hub files
  (`hub_files=`, prefetched with network before the offline run line); the in-job environment is the registry's
  (`env_contract`), never an ad-hoc install.
- **Replica + canary** — prove determinism and reuse:
  `python -m scripts.hf_jobs campaign <file> resume --with-replica <N>` (fresh replica beside the main run) then
  `… verify --with-replica <N>` writes the attestations; `… resume --canary` + `… verify --canary` run and
  compare one canary unit per attested stage.
- **Clear a halt** — `… resume --clear-halt <instance>/<stage> --reason "<text>"` clears ONLY the error-resume
  band (never a deterministic unit-failure halt).
- **Build the image** — the owner pushes the tag `job-image/$(python -m scripts.hf_jobs image tag)` and approves
  the `job-image` environment run; then records `registry_ref`/`digest` and runs `image probe-startup` to fill
  `startup_s` in `image.lock.json`.
