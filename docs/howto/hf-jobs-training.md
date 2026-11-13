# How-to: train silly-kicks models on HF Jobs

The operator runbook for HF Jobs campaigns (ADR-128; it supersedes the ADR-122/ADR-123 launch path). A
campaign re-fits trainers and runs corpus drivers as stage-contract pipelines (`scripts/_stages.py`) on
`cpu-upgrade` HF Jobs: W workers per job x N jobs per stage, byte-identical for every (W, N), resumable at
any point. One committed CLI drives it: `python -m scripts.hf_jobs campaign <file> <command>`
(`scripts/hf_jobs/campaign.py`). Writing a new pipeline is `docs/howto/compute-pipelines.md`.

Three owner gates, every time:
- **Commit** -- `adopt` stops at the commit gate; nothing commits without an explicit owner "yes".
- **Push** -- a separate "yes".
- **Hub publish** -- `publish` needs an owner go per run (`--sandbox` first); a public repo also needs
  `--i-have-owner-approval <repo>` for every target.

`cpu-upgrade` only (8 vCPU / 32 GB / $0.03 per hour, INV-4): the flavor is one constant
(`scripts/hf_jobs/launch.py` `FLAVOR`); speed comes from W and N, never a bigger flavor.

## Prerequisites

- `.venv-hfjobs` (Python 3.12, the pinned in-job environment; the image is the digest-pinned
  `python:3.12.14-bookworm`, decision P1). Build it from the committed lock:

  ```bash
  UV_PROJECT_ENVIRONMENT=.venv-hfjobs uv sync --frozen --python 3.12.14 --extra train --extra xgboost --extra test --group hfjobs
  ```

  Run every campaign command and every local verification with `.venv-hfjobs/Scripts/python.exe` (Windows)
  or `.venv-hfjobs/bin/python`.
- An HF token: `hf auth login` (the `hf` CLI is for humans only; the controller uses the typed
  `huggingface_hub` API). The token rides `secrets`, never the command line.
- `gh auth login` (preflight checks that the campaign commit is pushed and its CI is green).
- The campaign commit is a CLEAN, PUSHED commit (ADR-037): jobs clone exactly that commit.

## Staged datasets

Every dataset a campaign reads is declared ONCE in `scripts/hf_jobs/datasets.toml` (name, repo, kind,
visibility, access). Kinds: `flat_tc3` (`shards/<gen>/<provider>__<game>.parquet` frames +
`_actions/<stem>.parquet` + `home_teams.json`), `pergame` (`<game>/frames.parquet` + `actions.parquet`), and
StatsBomb open data (fetched per unit at the pinned revision, decision P6). In-job code reads only these
neutral shapes (INV-5); how a dataset was built stays off HF.

After a dataset is staged, record it: `python -m scripts.hf_jobs datasets record <name>`. It MEASURES the
staged dataset and writes the pinned revision, the unit list and the expected corpus identity (digest + n;
for a public dataset the canonical id list) into `scripts/hf_jobs/datasets.lock.json`. `datasets show
<name>` prints the record. `adopt` checks every artifact against it (INV-9).

The open data (`sb_open`) is pinned to a GitHub commit and read in a job from a staged, read-only snapshot of
that commit (`snapshot_repo`, ADR-128 amendment 2026-11-07). Stage it only on an owner-approved pin change:
`stage_open_data(<revision>, <dir>)` (`scripts/hf_jobs/stage.py`) fills `<dir>` by calling the same fetch
functions with the pin set (each file up to 3 attempts; a file that never parses is listed as `unreadable`), then
the owner uploads `<dir>` to the snapshot repo and runs `datasets record sb_open`, which records the snapshot
(with its `unreadable` list) and refuses one of another revision. A job mounts the snapshot as its
open-data cache; a cache miss raises `OpenDataSnapshotMissError` and never fetches from GitHub.

## The campaign file

`scripts/hf_jobs/campaigns/<name>.toml` (committed, INV-7): the bucket, `concurrency_cap` (the in-flight
ceiling), `attempts` (the per-unit bound; each HF job itself gets one attempt), an optional `mode`
(`rematerialize`, the default: `verify` requires the fresh run to reproduce the committed model; `refit`: new
data replaces the model on purpose, so `verify` passes each arm on its own acceptance gates and reports the
comparison with the committed model; ADR-130), the resource-sharing knobs
(`units_per_job`/`max_shards` size a stage into few big jobs; `units_per_job_by_stage` gives one `"job/stage"` its own job size; `mount_cap` + `mount_cap_by_dataset` bound
concurrent mounts of a dataset; `mount_fault_budget`; `launch_batch` + `backoff_max_s` pace launches; a
`peak_rss_by_stage` hint sizes memory-heavy stages — ADR-128 amendment 2026-11-03), and per job the trainer or
driver, its tier, its dataset (plus
`probe_dataset` for xcross and `pool_dataset` for the receiver's Gradient Sports pool) and its variants. The
registry pins each instance's corpus-scope and search-size args (`base_args`). Run keys are derived: `{campaign}-{commit7}-{job}-{tier}`;
`--replica N` adds `-rN`, a FRESH store for the same-commit re-run (INV-1 gate, decision P18).

## The command sequence

Run long commands in the background. The controller is the minimal driver (`scripts/hf_jobs/driver.py`,
ADR-128 amendment 2026-11-02).

1. `preflight` -- seven no-spend checks (no tracked edit in the tree, commit pushed and green, lockfile and image, numba in the locked job
   install set, API contract, datasets, every instance's `--plan`, run in parallel, and the org's storage headroom). `check_numba` reads
   `uv export --frozen --no-hashes --extra train --group hfjobs` and evaluates each line's marker for Linux
   CPython 3.12: the job must install numba (a core dependency since 2026-11-06; without it the ghost leaf walk
   ran 12x slower into the 2 h timeout). Each check prints one `[preflight] <check> ok|FAIL <s>s` line as it
   ends, and each `--plan` prints its own line. It trains nothing: CI's parity tests already run every trainer on
   toy data. It uploads the report the launch gate requires: every `launch` / `resume` step
   refuses without a passing report for THIS commit and THIS campaign-file hash, so re-run `preflight` after
   any edit to the campaign file.
2. `launch --dry-run` -- prints what the driver would do now (FATAL stop, the launches, the stuck stages);
   launches nothing.
3. `launch` / `resume` -- the stateless driver loop (`--poll-s`, default 30 s). Each step rebuilds state from
   the store (`--plan` per instance) and the job labels, then launches the pending units of each instance's
   first ready, un-reduced stage: up to `concurrency_cap` jobs, each a full `cpu-upgrade` box (8 unit
   workers; an `inner_parallel` stage runs one unit with the box's threads inside) over its explicit keys (an
   instance with a `download` dataset runs round-robin `--shard I/N` slices instead, since that dataset is
   fetched by shard). A stage whose jobs still run gets more jobs: one for each group of the units that no
   running job of the stage holds, up to `max_shards` jobs of one stage at once. The driver knows the units of a
   running job from its own record of the jobs it launched, or else from the job's manifest in the store. A stage
   waits for its running jobs when a running job's units are not known, when a reduce trigger runs, or when the
   instance has a `download` dataset. Instances of one run key whose stage has one generation (the 5 ghost_gk
   variants at `validate`; `default` and `full` at `extract` to `importance`) are peers at that stage: the first
   one in a step decides the stage for all of them, with all their running jobs, attempts and timeouts, and the
   others wait (ADR-128 item 57). An instance whose `--plan` failed holds every same-named stage of its run key
   while its jobs run. A job that the driver launched and that the HF listing does not show yet
   counts as in flight and holds its units; after 5 listings that do not show it, the driver prints `[driver]
   job <id> <label> was launched but is not in the listing after 5 steps; its units are free again` (HFU launch
   top-up spec §3). Every keyed job has a tag: the first 12 hex characters of the sha256 of its keys file. The
   tag is in the keys file name (`<index>of<count>-<tag>.json`), at the end of the job name
   (`--<index>of<count>-<tag>`), in the `tag` label, and in the in-job label (`SILLY_KICKS_JOB_LABEL`, so every
   unit ledger record names one job). Each unit ledger record also holds `cpu_user_s` and `cpu_sys_s`, the CPU
   time of the unit's process (`cpu_scope` = `process`: an inner pool's children are not in it). An instance
   whose `--plan` fails is reported; the others go on (a `--plan` that fails with
   an HF rate limit, `Too Many Requests`, makes the step back off instead). The `--plan` reads run
   in parallel, and each step prints one line to stderr: `[driver] step: plans <n> in <s>s (<k> failed); hub
   requests <q> (plans <p>); in flight <i>; launched <l>; stuck <r>; instances done <d>/<n>` (done = the output
   stage is reduced). `hub requests` is the number of HTTP requests that the step sent to the Hub: the
   controller's own plus those of its `--plan` processes (the `plans` part). It is the load to hold against the
   HF rate limit. A `--plan` that fails reports its requests too (a 429 inside a plan is in the count). A
   `--plan` that timed out was killed and reports none, so the count of that step is a lower bound. One `--plan`
   costs one request per 1,000 files of its run key and two requests per file that it
   reads (measured with huggingface_hub 2.2.0); it reads one marker per `FileSet` result. A study
unit prints `[progress] study <tag> trial <k>/<n> best=<v> in <s> s` per trial, the census bootstrap
`[progress] bootstrap <k>/<n> convergent=<c>` about every 60 s and once at the end, a cross-validation loop
`[progress] <trainer> fold <k>/<n> in <s> s` per fold (the xsuccess `fit` is 36 stage units, one xgboost fit each: every unit prints its `[unit]`
lines), a long input read or reduce `[read] <stage> results <k>/<n>
in <s> s` at most every 60 s (once at the end when it read 100 files or more), and each reduce `[reduce]
<stage> done in <s> s`. A job hides the three per-match open-data converter warnings (`PYTHONWARNINGS`, scoped to
`scripts._sb_open_data`). A driver store write that stays rate-limited after 3 retries (2, 4 and 8 s) ends the
step's launches, as a launch 429 does; a lost per-job manifest prints `[driver] job <id> manifest not recorded
(rate-limited)`. A FATAL step prints its `FATAL` line instead, and a rate-limited step
   a `[driver] step: rate-limited ...; backing off` line. Each launch prints `[driver] launch
   <instance>/<stage> <i>of<n>-<tag> units=<k>` (an unkeyed launch of a `download` instance has no tag and
   prints `<i>of<n>`), and each job of this commit that ends ERROR or CANCELED prints one
   `[driver] job <id> <instance>/<stage> <i>of<n>-<tag> <reason> after <s> s` line (a job of a driver before the
   tag names `<i>of<n>`), once per run, so a failure shows
   in the driver log before the HF email arrives. Every Hub call is bounded (connect 10 s, read 60 s; a stalled
   read is retried), and a `--plan` read is killed after 600 s: the instance is reported `--plan failed
   (timeout after 600 s)` with the plan process's stack dump, and the step goes on (it is not a rate limit).
   The reduce is inline: the job that finds the stage complete reduces it, and the driver triggers a reduce
   with an empty key set when no unit is left to run. No per-worker memory cap is set, so a unit that runs
   out of memory kills its job (`oom`, a RETRY). A unit that `attempts` jobs covered without completing is
   reported, not retried. A job's limit is 7,200 s unless its stage declares `timeout_s` (the xsuccess `hpo` stage:
   14,400 s). A run re-plans only the instances whose output is not reduced yet, and each `--plan` lists its
   run key once.
   A `Job timeout` spends no coverage, but it charges the units that started in that job
   and did not finish (the driver reads them from the job log's `[unit]` lines); a unit charged by 2 timed-out
   jobs is reported `STUCK <instance>: <stage>/<key> timed out 2x (jobs <id>, <id>)` and not relaunched, while a
   held unit that never started is relaunched as before; a storage `403` or another FATAL stops the loop. Exit code: 0 done, 1 stuck, 2
   FATAL. After the operator fixes a FATAL, `resume` re-attempts the work (every job already terminal is
   the baseline). A killed job loses at most its in-flight units; `resume` never redoes a finished unit.
   Make NO tracked edit from `launch` until `adopt` ends: a tracked edit changes the code identity, so every
   controller command (`launch`, `resume`, `status`, `retrieve`, `verify`) computes generations the jobs never
   wrote, and finished instances read as not reduced (launch 5: retrieve found 2 of 20). Two guards (HFU
   launch top-up spec §19): check 1 of `preflight` fails on a tracked edit (it names the changed paths), and
   the driver stops before any launch with `FATAL code_identity: <instance> planned under <identity>, the jobs
   run <commit7>` when a `--plan` reports another identity. Then restore the tree and `resume`. `retrieve` has
   no guard of its own: keep the tree clean until `verify` and `compare` end, and draft the run record outside
   the repo.
4. `status` -- per instance, stage progress and the running jobs. A failed instance: read its job logs and
   the unit `*.failed.json`; a code defect is a new owner-approved commit, then `resume`.
5. `cancel` -- cancels every non-terminal job of the campaign commit.
6. `retrieve --staging DIR` -- copies each reduced output (and `<instance>.record.json`: what the job read,
   the extract counters) into DIR, OUTSIDE the repo; an xShot/xCross `public` instance also gets its extract
   rows (`<instance>.eval_rows.parquet`, beside the artifact). It prints `[retrieve] <i>/<n> <instance>:
   <files> files, <MB> MB, counters <k>/<units> in <s> s` per instance; the counters are read from one listing,
   in batches of 200 files, one `download_bucket_files` call each (run 9b: 13 calls for 1,436 files). 4
   instances run at once; a line prints when its instance ends, so `<i>` is not in order. At most 8 reads run at
   once. A read, listing or `--plan` call that hits the HF rate limit (2,500 requests in 5 min for the account) is
   retried after 30, 60, 120 and 240 s, with `[retrieve] rate-limited (<what>); retry <i>/4 in <w> s`, and holds
   no read slot while it waits; after the fourth wait the error stops `retrieve`.
7. `verify --staging DIR` -- loads every artifact in `.venv-hfjobs`, checks the five expectations (training
   commit = campaign commit; recorded dataset revision = the pin; corpus digest + n = the record; shipped
   tier = the expected tier), certifies per the registry (`public`: the head-to-head as in-sample parity AND two-sided CV parity on
   `cv_metric` within 5 %, from the committed `metrics.json` or its manifest `cv_reference`; `owner`: CV
   parity on `cv_metric`; `rebaseline`: identity comparison, decision P15, where a code delta is an AST change
   of an input module) and runs the ADR-126 integrity gate for xShot/xCross. A build with `in_sample_gate=False`
   (the xCross default: no parameter match) certifies on CV parity and records the head-to-head. A driver with a
   `DriverSpec.tolerance` (the census) passes a rebaseline whose outputs differ only within it
   (`"within_tolerance": true`). An `sb_open` artifact's corpus check is the lock digest (`identity.open`); each
   `sb_open` head-to-head caches its open data in `DIR/_open_data/<instance>`. 4 instances run at once. Prints
   `[verify] <i>/<n> <instance> ok|FAIL (<checks>) in <s> s` when each instance ends (an `sb_open` line ends with its
   coverage note) and writes `DIR/verify.json` (instances in campaign order in the returned report).
8. `adopt --staging DIR` -- on a clean tree at the campaign commit (the provenance check and the root's own
   `git status`): bundles the verified files into the
   wheel dirs (a public bundle is rehydrated with the canonical public ids), writes
   `docs/artifacts/observed.json`, copies the campaign commit into the `training_commit` of each adopted artifact
   with a `commit_field` (the one field a tool writes in `docs/artifacts/manifest.toml`), re-renders the generated card
   regions, writes the verdict JSONs and a CHANGELOG draft, runs the full local gate, and STOPS at the
   commit gate. Its diff touches only the allowlisted paths. A failure while it writes restores the tree (a
   restore that fails raises `AdoptionRollbackError` naming the paths); the local gate runs after the writes.
   Commit the adoption as a NEW commit on top of the campaign commit, never an amend: every adopted
   `training_commit` names that commit. adopt reads each instance from `DIR/_final/<instance>/`, the staging files
   with `metrics.json` finalized off HF (canonical xCross probe ids); `publish` uploads the same copy. A bundle's
   own `metrics.json` is its committed CV record; a manifest `cv_reference` is only for a bundle without one.
9. Owner gate (commit), then declare the re-fit event in the manifest (version, ADR) and re-render the
   cards (`tests/test_card_regions.py` flags a stale region).
10. `publish --sandbox --staging DIR` (owner go) -- private `silly-kicks/sandbox-<name>` repos, then the
    post-verification (model-only file set, remote `SHA256SUMS` = local, README = the rendered card, a load
    round trip: `from_hub` for an artifact with a Hub class; the job loader on the files downloaded to
    `DIR/_post_verify/<repo>/` for an artifact without one, such as `xsuccess-v1`). A public publish is a separate owner go after the artifact commit.

## The DGX as an executor

The DGX runs the SAME contract (ruling 11): `scripts/hf_jobs/dgx_e2e.py` `run_local_e2e` plans an instance
and runs each stage through the stage CLI with peak-RSS measurement against the 32 GB `cpu-upgrade` ceiling
(a stage over the ceiling, or a non-zero exit, is `blocked`).

## Cost

Per-second billing on `cpu-upgrade` ($0.03 per hour): N short jobs cost the same as one long job. The scratch
bucket adds a small per-GB-month cost.

## Storage and run keys

The org has a private storage limit of 100 GB (owner-verified 2026-11-10 on the HF billing page; the buckets,
the private datasets and the private models together). Launch 7 stopped on it (`403 Private storage limit
reached`). At the limit HF also refuses the bucket delete, so clean up BEFORE the limit is near.

- **Preflight check 7, `check_storage`.** It adds the size of the org's buckets and `usedStorage` of each private
  dataset and model, and fails when the sum plus `storage_headroom_bytes` (20 GB) is above `storage_limit_bytes`
  (100 GB); both values are in the campaign file. The line lists the used bytes by repo, largest first. An API
  error or a repo without a size fails the check.
- **`campaign <file> prune`** lists, for this campaign, the run keys of every commit other than HEAD (files and
  bytes) and deletes nothing. **`prune --delete --i-have-owner-approval <commit7>`** (the flag once per commit)
  deletes the run keys of the named commits. Deletion is irreversible: get the owner's yes for each run. Run it
  after an amend is pushed, and after a launch is retrieved and verified. Stop the driver first: between two job
  waves no job is non-terminal, so `prune` cannot see that a campaign is live.
  - It refuses HEAD's commit, a commit with a non-terminal job, and a commit without a run key.
  - It touches only top-level folders that are exactly `<campaign>-<commit7>-<job>-<tier>` (or with `-r<N>`) for a
    job of the campaign; never `_campaign`, `_staged`, `_canary` or another campaign's keys.
  - For a commit that the manifest names as a `training_commit` it prints a notice first: the run keys are run
    records; the artifacts are in the wheel and on the Hub.
  - It prints the files left for each run key and exits 1 when a file is left; a second run deletes the rest. A
    `403` stops the run with a message: delete a private dataset or model first.
- **A lost result.** Over the limit the bucket mount accepts a write locally and loses the upload; the small
  marker of a `FileSet` result can land without its files. The plan then reports the unit in `lost_units` (or the
  stage in `lost_output`), the stage is not reduced, and the driver reruns the unit or the reduce. When the
  attempts are spent the report is `<stage>/<key>: result lost (members missing); free storage, stop the driver,
  then run resume` (or `<stage>: output lost ...`).

## Compare two launches

`campaign <file> compare --staging NEW --baseline OLD [--baseline OLD2 ...]` compares the artifacts of a
launch with what shipped (local files only). It prints one line per instance and exits 0 only when every
artifact is identical and the census verdict of NEW (`NEW/verify.json`) is ok.

- NEW is always its retrieved files (`NEW/<run_key>/<variant>/`). A job must write the shipped form itself.
- What shipped can come from several launches, so `--baseline` can be given more than once. The baseline of an
  instance is the LAST given folder that holds a run key of its job and tier, under any campaign name
  (`<campaign>-<commit7>-<job>-<tier>`), with a folder of its variant. Two run keys of one job and tier in one
  baseline folder are refused.
- In that folder the baseline is its adopted copy (`OLD/_final/<run_key>/<variant>/`: the files that `adopt`
  finalized and `publish` uploaded) when it has one, its retrieved files otherwise. The baseline of a parity proof
  is what shipped. Each line names the form and the run key: `(baseline: adopted copy of <run key>)` or
  `(baseline: retrieved files of <run key>)`.
- The adopted copy is not always equal to the wheel bundle. For xCross `public`, the bundled `metrics.json`
  holds `corpus_match_ids`, and the adopted copy holds their digest and their count.

The rule for one artifact:

- `metadata.json`, `metrics.json` and the file that the artifact's `commit_field` names: equal without the
  provenance keys (`identity.PROVENANCE_KEYS`).
- `SHA256SUMS`: equal on the lines of the other files. `MODEL_CARD.md`: equal after each commit is replaced.
- Every other file: byte for byte.

A launch that changes no model is a parity proof: when `compare` passes against the shipped launch, adopt and
publish nothing. A difference is a defect of the change. Keep the staging folder of the shipped launch, with its
`_final/` folder: it is the baseline of every later parity proof.

A refit of some arms (a campaign in `refit` mode, for example `campaigns/tf46_refit.toml`) is proved in this
order: run the refit campaign; `retrieve`, `verify` and `adopt` it (a commit); then run the full campaign
(`cycle1.toml`, unchanged) at that commit and compare it with both launches that shipped:
`compare --staging <full run> --baseline <the shipped launch> --baseline <the refit run>`. The refit arms are
read from the refit run, every other arm from the shipped launch (`docs/superpowers/specs/2026-11-11-hfu-launch-topup-design.md` §14.3).

## Recover a STUCK unit

A `STUCK <instance>: <reason>` line ends with `fix the cause, stop the driver, then run resume`. The driver
counts the failed jobs it saw since it started; the jobs that were terminal at its start are its baseline. So a
unit that you repair while the driver runs stays STUCK in that run. Stop the driver (the jobs in flight go on),
then run `resume`: the new run has a new baseline and attempts the work again.
