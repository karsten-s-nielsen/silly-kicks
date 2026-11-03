# HFU — reduce reads, job exit semantics, match-outcome pipeline — design

Branch `feat/minimal-hf-driver` (PR #299); one amend per change, each amend and force-push owner-gated. Scope owner:
Karsten. Reviewers A and B (independent); reports in `D:\Development\_reviews\2026-11-0*-hfu-reduce-reads-*`.
The review rounds and their answers are in git history; this document states the standing decisions only.

| Change | Section | Scope set | Approved |
|---|---|---|---|
| 1–3 | §3–§10 | owner 2026-11-04 | A and B spec r4 |
| 4 (liveness) | §11 | owner 2026-11-05 | A and B spec r5–r6 |
| 5 (lost-rename repair) | §12 | owner 2026-11-05 | A and B |
| 6 (launch 2 failures) | §13 | owner 2026-11-06 | A and B spec r8 |
| 7 (launch 3 findings) | §14 | owner 2026-11-06 | A and B spec r11, r14 (I1/I2), r15 |
| 8 (launch 4 findings) | §15 | owner 2026-11-08 | DRAFT |

## 1. Why

The cycle-1 retrain on `f03f4753` reached 11/21 instances and was stopped. The job list and logs disproved the
job-sizing theory; this spec fixes the measured causes.

### 1.1 Evidence (cycle-1 run on `f03f4753`, 211 HF jobs: 198 COMPLETED, 10 ERROR, 3 CANCELED)

| # | Observation |
|---|---|
| E1–E2 | win_probability extract jobs finished in minutes (units 1.4–2.8 s); the `Nof4` shards held 19 units each, not ~990. |
| E3 | 3 jobs hit `Job timeout` in the INLINE REDUCE: `reduce_stage` → `read_result` → `store.names` → `OSError: [Errno 107]`. |
| E4–E5 | The generation dir held 11,901 entries; `read_result` listed and stat'ed the whole dir per unit (~47 M stats) and each done unit was read 5 times. |
| E6 | The HF bucket mount serves ~24 small files/s per container (~42 ms each): the cost is the number of files read. |
| E7–E9 | 3 xshot `sc_extended` cv jobs ended `exit code: 3` (`StageFailedError` after a recorded `OSError: [Errno 5]`, no path, no frame), inside the stat-storm window (correlation, not proof). |
| E10 | `match_outcome_dependence` ran as one adapter unit: 5,543 s of the 7,200 s wall. |
| E11 | 3 `Volume mount failed` errors are platform-side and already RETRY. |
| E12 | One cv job ended `Job timeout` with no unit output for 2 h (3× a healthy fold time); a plain `Stage` printed nothing per unit, so silence was no evidence. |

## 2. Goals, non-goals, anti-goals

A *reduce site* reads the per-unit results of one generation to build one value (`reduce_stage`, and every unit
work that loops over another stage's keys, §3.2). A *run job* runs units of one stage.

- G1. A reduce site lists its generation a constant number of times and reads each file it needs once.
- G2. A recorded unit failure never turns a run job into an HF ERROR (except the systematic `abort_first_k` stop).
- G3. A unit failure records where it happened.
- G4. `match_outcome_dependence` runs as a sharded ADR-128 pipeline.
- G5. The fix is proven on HF against the operating envelope (§6).
- G6. A running job shows it is alive at least every 5 minutes (unit, frame); a job at its timeout leaves every
  covered process's stacks and its children's kernel state in its log (§11).

Non-goals (at the time): job-sizing or `mount_cap` changes; a store-format change (options B/C/D); compute-speed
changes; a fix for E12 (Change 4 diagnoses it); a wheel change (the version stays 4.139.1). Change 8 (§15.5a)
later migrates `pass_completion` and `xsuccess` off the adapter.

Anti-goals: a second representation of unit results; a weaker checksum contract (INV-3); hiding a real failure (a
stage that cannot reduce is still reported; a local `--run-all` still fails); an in-job retry loop that masks a
dead mount.

## 3. Change 1 — reduce sites read the store once (option A)

### 3.1 Contract

One reader in `scripts/_stages.py` reads the done results of one generation:
- It takes the generation's listing (one `Store.names` set) from its caller and never lists.
- It reads each payload and its `.sha256` once, verifies in memory and decodes from the same bytes (a
  `.fileset.json` also reads each member once). Keys resolve by exact name, so `.tmp-*.part` orphans are ignored.
- An absent or bad-checksum result goes back to the caller, which applies INV-3; the reader never deletes.
- Order and values equal the old per-key `read_result` for the same store.
- `LocalStore.names` uses `os.scandir`; `LocalStore.read_bytes` opens directly and stats only on the error path
  (`None` exactly where `is_file()` was False; a real I/O error still raises).

### 3.2 Sites (census)

`reduce_stage` (one listing, `unit_states(verify=False)`, the reader), the census bootstrap/report reads, the
ghost_gk extract consumer and the xcross probe loop use the reader. `item_counters`, the `reduce_stage` excluded
loop, `adopt._counters` and `deferred_memory_units` already met G1 and stayed as they were.

### 3.3 Budget and guard

The run-job reduce path (`try_reduce` → `reduce` subprocess → `reduce_stage`) lists the store a constant number
of times. A test pins that count EXACTLY at two unit counts on an instrumented `Store`, run in one process (the
subprocess call replaced in-process), with `read_bytes` = 2 per done unit and the output unchanged; it fails on
the old per-unit loop. It is not in `SCALE_GUARDED` (that registry is the AST-derived `group_rows` caller set).

## 4. Change 2 — a recorded unit failure does not fail the job

### 4.1 Contract

`max_failures` is a REDUCE-TIME rule ("may this stage reduce without these units?"); the driver retries a failed
unit while attempts remain. `abort_first_k` stays a run-time rule: the first k completed units all failing is a
systematic defect and stops loud (exit 3).

- `_worker_should_stop` and `_check_failure_rule` apply only `abort_first_k`; the `_run` end check is removed.
- `try_reduce` over tolerance returns "not reduced" without starting the reduce subprocess.
- `reduce_stage` over tolerance still raises `StageIncompleteError`; `require_reduced` still raises, with the
  pending/failed/tolerance counts.
- One predicate `exceeds_tolerance` in `scripts/_stages.py`, used by `reduce_stage`, `try_reduce` and the driver.
- A reduce trigger (a keyed run over an empty key set) exits 0 whatever the refusal cause; the driver counts
  every terminal trigger of an un-reduced stage and reports after `attempts`.

Result: a run job with a recorded unit failure finishes its other units and exits 0; a local `--run-all` over
tolerance runs every unit, then fails at `require_reduced`.

### 4.2 Why the driver stays correct (verified)

A COMPLETED job credits all its units, so a recorded failure still spends one attempt and is a candidate again
below `attempts`; a spent unit reaches `_reduce_or_report` as before.

### 4.3 Failure locus

The failure record gains a bounded, additive `traceback` field, captured in `run_unit` (plain `Stage`) and passed
from `_driver.for_each`'s new per-key traceback map (`ItemStage`). The `error` string keeps its exact format; the
field never enters a result, a token or an output.

## 5. Change 3 — `match_outcome_dependence` as an ADR-128 pipeline

### 5.1 Shape

`refs` (one unit: the competitions and match refs via `open_data_source`) → `extract` (`ItemStage`, one unit per
match, the pure `extract_match_slice`, `max_failures=None`) → `fit` (`matches_from_shards` → `fit_rho` → a FileSet
with `model.json` + `SHA256SUMS`, the legacy fields). The pure kernels are unchanged. The `ItemStage` failure rule
(`abort_first_k=3`) replaces the legacy 3-consecutive rule; with zero failures the outputs are identical.

### 5.2 Wiring

The registry drops `adapter=True` and names the pipeline; `adapter._SHARD_ROOT_TARGETS` (this trainer's only use)
is removed; the compute-contract adapter allowlist and the `TODO.md` Cycle 2 row follow.

### 5.3 Parity oracle (before the commit)

1. Fit parity, offline on the legacy shards (read as the legacy fit did, sorted glob): `rho` and `n_matches` equal
   the legacy artifact exactly (`rho` 0.02161420866397881, 3,961 matches).
2. End-to-end parity on HF (`_canary` store, the pinned open-data revision): the `refs` key set equals the legacy
   shard set and `rho`/`n_matches` equal the legacy artifact exactly.

Any difference stops the cycle for the owner. The W1N1-vs-W2N2 byte-identity gate covers the pipeline.

## 6. Definition of done — measured on HF

Measured 2026-11-05 on patch-on-base canary jobs (21 jobs, 20 COMPLETED, 1 ERROR by design):
- C (fit parity, §5.3 part 1): `model.json` and `SHA256SUMS` byte-identical to legacy. PASS.
- A (the reader, 4 concurrent jobs over 3,961 units): 1 listing per job, 2 reads per unit, 702–725 s at ~11
  files/s. The ceiling is per container and set by the ~90 ms serial read latency. **Owner decision 2026-11-05
  (1a): accept the ~12 min reduce of 3,961 units.**
- B (match-outcome pipeline): longest job 913 s; `refs` key set and `model.json` equal legacy. PASS. The mount
  lost 55 of ~11,900 result renames (§12).
- D (liveness, a stalled job at `timeout_s=1500`): every `[alive]` field present; the dumps landed before the
  kill (offset −5 s). PASS. **Owner decision 2026-11-05 (3b): stagger the dump time per process (§11.4).**

## 7. Owner gates (in order)

A/B review of spec, plan and implementation; an owner yes for any HF launch; `/final-review`; the amend (an
explicit yes after the diff is shown); the force-push (its own yes); CI green; the retrain launch (its own yes).

## 8. Risks

- R1. E7 is a correlation and E12 had no diagnosis; Change 4 makes a hang leave its location in the log.
- R2. `d_type` on hf-mount is unverified; `scandir` is never worse.
- R3. An amend is a new run_key, and the store is per run_key: every instance and stage reruns (owner-accepted;
  `cross_commit_version` keeps a stage token stable across commits, not its store).
- R4. The match-outcome loader is network-bound; sharding spreads it, it does not speed one match.
- R5. The mount ceiling could be shared across containers; measurement A showed it per container.
- R6. The HF timeout clock may start before the bash script; the 600 s dump margin absorbs it (measured −5 s).

## 9. Docs that travel with the commit

ADR-128 amendment, `docs/howto/compute-pipelines.md` (the read rule, the exit rule, reading `[alive]` lines and a
deadline dump) and `CHANGELOG.md` `[Unreleased]`. Done.

## 10. Left to the builder (plan)

Names, the traceback cap, the listing constant, the match-outcome params and the test layout were the plan's;
they are in the code.

## 11. Change 4 — a running job shows it is alive; a timed-out job leaves its stacks (E12 diagnosis)

Owner scope 2026-11-05: diagnosis only, plus an alive signal at least every 5 minutes per job. Third-party pool
processes (joblib/loky) are reported from their parent; a joblib hook comes only if a log shows a stall in them.

### 11.1 Every process kind in a run job (verified in code)

Each covered process calls `start_job_watch(<role>)` explicitly at its entry point: the `datasets fetch`
pre-step (`fetch`), the stage main (`main`), each executor worker (`w<i>`), the reduce subprocess (`reduce`),
each `InnerPool` child (`w<i>.pool<j>`). Third-party pool children appear in their parent's
`children=` field.

### 11.2 The alive line

A start line at once (with the deadline-dump time and `numba=<version>`), then every 300 s one line per process:
label, role, pid, uptime, the current unit and its age (or `-`), the main thread's innermost Python frame, and
`children=`. `run_unit` sets and clears the unit slot that the thread reads.

### 11.3 Child processes in the line

`children=` lists each direct child from `/proc` as `pid:state:cpu+<Δs>:<wchan>` (`-` without `/proc`).

### 11.4 The deadline dump

`job_command` exports `SILLY_KICKS_JOB_DEADLINE_EPOCH` before git and uv. Each covered process arms
`faulthandler.dump_traceback_later` at the deadline minus a per-role margin, so no two dumps share a second:
`main`/`fetch` 600 s, `reduce` 601 s, `w<i>` `600 + 10 × (i + 1)` s, `w<i>.pool<j>` 1 + (j mod 9)
s after its worker. The C watchdog needs no GIL, so the dump fires even under a hung C extension. A malformed
value turns both off with one warning line.

### 11.5 Scope and safety

Only HF run jobs set the variable, so local runs and tests print nothing new. The thread touches no store, result
or token. At most `1 + W + W·P` lines per 5 minutes.

### 11.6 Tests and proof

One subprocess test per entry point, a stuck-child test (lines and dump name the function), the `run_unit`
slot, `/proc` parsing, the export line, the off and malformed cases, and distinct dump seconds per role. §6
measurement D proved it on HF.

## 12. Change 5 — a lost rename on the HF bucket mount is repaired in the job (2c)

Owner decision 2026-11-05: fix it in this cycle by settle-and-repair in the store.

### 12.1 Evidence

The mount lost 0.27–0.43 % of renames under paced load (probes r2, r3); `fsync` did not help; an immediate
`stat` saw the final, but a listing 5–120 s later did not. Every lost final left its `.tmp-*.part` with the full
bytes, and one `os.replace` from it repaired every loss. A final missing from one listing only, with no temp
part, is a listing flicker.

### 12.2 The contract

- `scripts/_write_confirm.py` keeps a per-process journal (final → temp) of every store rename
  (`_write_bytes_atomically`, so `LocalStore.write_bytes`, the `for_each` shards and `write_exclusion`); deletes
  forget their paths. Direct writes (the `.checksummed` marker, `.counters.json`) and the open-data cache are not
  journaled.
- `confirm_writes(*, settle_s, rounds=3)`: settle, list each journaled directory once, re-run `os.replace` for a
  missing final whose temp is listed, re-check a flicker; a final still missing after the last pass raises
  `LostWriteError` naming every path.
- `confirm_job_writes()` settles 30 s in an HF run job and 0 s elsewhere; it runs once at the end of every
  writing process (`worker_main`, `reduce_main`, `_stages.main`, `liveness._run_module`) and prints
  `[confirm] repaired <n> …` or `[confirm] LOST <path>`. A worker whose confirm raises exits non-zero and the
  executor relaunches its pending units.

### 12.3 Tests

No loss, a repaired loss, an unrepairable loss, a flicker, a deleted file, a lost exclusion,
the settle time per environment, and each entry point's call.

## 13. Change 6 — the retrain launch 2 failures (2026-11-06)

Owner decisions 2026-11-06: fix every fault in this cycle and give each a log trail; numba is a core dependency
(the numpy kernels run only behind a `SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1` switch; a missing numba raises); a unit
that times out twice is STUCK.

### 13.1 Evidence (retrain launch 2 on `e9301e04`: 270 jobs, 258 COMPLETED, 12 ERROR; 9 of 21 instances done)

- F1. numba was absent from `[train]`: ghost_gk importance ×5 and ghost_outfield fit ran the ~12× slower numpy
  leaf walk into `Job timeout`.
- F2. Five cv workers read one 388 MB FileSet through the mount at once (state `D`, CPU 0); one reader is fast.
- F3. The driver hung 2 h 10 min: 3 `--plan` subprocesses blocked in Hub reads with no timeout
  (huggingface_hub 2.2.0 builds its client with `timeout=None`).
- F4. A timed-out unit was relaunched without limit (a RERUN spent no coverage).
- F5. No log trail: no driver line per ended job or launch, no unit start/end lines, no `--plan` stack, no read
  size or rate, no numba proof.

The lost-rename repair (85 repairs, 0 `LOST`), the `[alive]` lines and dumps, and the mount-flake retry worked.

### 13.2 Change 6a — numba is a core dependency (F1)

- `numba>=0.59.0` is in `[project] dependencies`; `[numba]` stays an empty extra (Hyrum). Consumers inherit numba's
  numpy and llvmlite caps; the lock stays at numba 0.65.1.
- `silly_kicks/_numba_dispatch.py`: `use_numba(family)` is False only under that family's `FORCE_NUMPY` switch,
  else it imports numba once and raises `NumbaUnavailableError` (an `ImportError`) when absent; `require_numba()`
  serves an explicit numba request. Families: ghost, DAS, coordination, pitch control, ball carrier, xT-GK.
- An explicit engine argument wins: a numpy-side choice never imports numba; `engine="numba"` /
  `kde_backend="cpu-numba"` call `require_numba()` (no silent fallback).
- Each `[alive]` start line shows `numba=<version>`; preflight `check_numba` checks that the locked job install
  set (`uv export … --extra train --group hfjobs`) has a numba line for Linux x86_64 CPython 3.12.
- No artifact byte changes (ADR-076 parity). The version stays 4.139.1 (owner). ADR-076, ADR-008 D3 and ADR-013
  are amended.

### 13.3 Change 6b — a FileSet is read once per job (F2)

`materialize_fileset` materializes into a job-local cache keyed by generation and key under a `filelock`; the
first process reads the store, the others copy the cache. `LocalExecutor.run_stage` owns the cache directory
(`SILLY_KICKS_FILESET_CACHE`). Every store read of 8 MB or more prints `[read] <rel> <MB> MB in <s> s (<MB/s> MB/s)`.

### 13.4 Change 6c — every HF API call has a timeout; a plan read has a deadline (F3)

`hub_client.install_timeouts()` installs a client factory whose request hook replaces each `None` timeout with
`connect=10, read=60, write=60, pool=10` s (a client default is not enough: `paginate` passes `timeout=None`
explicitly). `campaign.read_plan` runs `--plan` with a 600 s deadline; the child dumps its stack 30 s before it,
and a timeout is a `--plan failed (timeout)` report with the stack, never a back-off.

### 13.5 Change 6d — a unit that times out twice is STUCK (F4)

A timeout charges only the units its log shows started and not finished (`[unit] … start` without an end line;
one log fetch per job, retried on a failed fetch). A unit charged by 2 timed-out jobs is reported
`STUCK <instance>: <stage>/<unit> timed out 2x (jobs …)` and never relaunched on any path; a round-robin shard
that holds one is held back and reported. A resume re-seeds the baseline.

### 13.6 Change 6e — the log trail (F5)

`[driver] job <id> … <reason> after <s> s` per newly ended job; `[driver] launch …` per launch; `[unit] <stage>/<key>
start` and `… done|excluded|failed in <s> s` per unit; `[job] <done>/<units> done, <failed> failed, exit <rc>, <s> s`
at every run exit; plus the `[read]` lines, the `--plan` dump and `numba=`.

### 13.7 Tests (each seen to fail first)

In `tests/` per change: the dispatch switches and the census registry, the explicit engines, `check_numba`, the
FileSet read-once and crash cases, the Hub timeouts against a silent server, the plan deadline, the 2-timeout
rule, and the log lines.

### 13.8 Definition of done

CI green; launch 3 of `cycle1` on the new commit (all 21 instances rerun under the new run_key). Measured in
launch 3 (§14.1): 21/21 reduced, 0 timeouts.

### 13.9 Docs that travel with the commit

ADR-076/008/013 amendments, ADR-128 items, `AGENTS.md`, `conventions-core.md`, the two howtos, `CHANGELOG.md`. Done.

## 14. Change 7 — the launch 3 verify findings, faithful recipes, the open-data snapshot, progress lines (2026-11-06)

Owner decisions 2026-11-06/07:
- Fix every finding in this cycle; launch 4 runs on the new commit. Every committed model is assumed valid: a
  fresh/committed difference is a campaign or verify defect until evidence shows otherwise.
- Any match-set comparison is one digest compare; re-runs read a staged open-data snapshot, not GitHub.
- Public certification = the head-to-head as an in-sample parity check plus two-sided CV parity within 5 %
  (relative, the ADR-126 `rel_tol`); a held-out set is impossible for models trained on every pinned match.
- Progress lines (14.8) and retrieve/verify lines with parallel counter reads (14.9); ghost `importance` keeps
  `[alive]` only.
- I1: lock optuna 5.0.0 (seven of the eight committed TPE models were tuned under it; xcross `public` `default`
  and `xsuccess` under 4.9.0 certify without an exact parameter match).
- I2: a stable back-line tie-break at the source, in `select_back_line_players` and `_collective.back_line_batch`.

### 14.1 Evidence (launch 3 on `c40ec7b3`: 275 COMPLETED, 1 ERROR; 21/21 reduced; verify 15 ok, 6 failed)

| Id | Finding |
|---|---|
| F6–F7 | `pass_completion` and `xsuccess` trained on 64 matches (committed: 3,961; xsuccess also 30 Optuna trials); verify passed them because the public floor scored in-sample rows. |
| F8 | The census ran 64 matches (committed: 3,961): `DriverSpec` passed no recipe args. |
| F9 | The census rebaseline counted a docstring-only commit as a code delta. |
| F10 | The receiver pooling gate did not run (no `--pool-dataset`); the shipped model was unaffected. |
| F11 | The xshot/xcross `public` head-to-head could not run (no public builder; it wanted a corpus folder). |
| F12 | `DependenceModel.load` failed on a `str` path. |
| F13 | Verify checked no corpus scope for the four event-only artifacts. |
| F14–F15 | The local head-to-head read GitHub `master`; every `sb_open` job fetched every match from GitHub (~20,000 fetches). |
| F16–F18 | Retrieve (50 min) and verify were silent; long single units (study, census bootstrap) printed only `[alive]`. |
| F19 | xshot `public` `default` failed the ADR-126 gate (`pr_auc` +6.5 %): the consequence of I1. |

Matched: ghost_gk weights byte-identical; win_probability ECE to 1e-8; match_outcome `rho` to 1e-10; receiver
`top1_cv` equal; xshot/xcross row counts and the `xshot17` digest equal.

### 14.2 Change 7a — each campaign instance runs the committed recipe (F6, F7, F8, F10)

`DriverSpec.base_args` reach every instance: `pass_completion`, `xsuccess` (also `--hpo-trials 30`) and the census
get `--all-competitions` (single-sourced in `all_open_competitions()`, 80 pairs at the pin); `receiver_model`
gets `pool_dataset = "tc3"` and `--pool-provider gradientsports`. A registry test pins every instance's
corpus-scope and search-size args.

### 14.3 Change 7b — verify checks the recipe, not only the score (F9, F13)

- `datasets.record` writes `sb_open.identity.open` (`corpus_match_ids_sha256`, `corpus_n_matches`) into the lock;
  each event-only bundle's `metrics.json` carries the same keys (`open_data_identity`), and verify applies the
  manifest `digest` rule to every fresh `statsbomb_open` artifact. `SHA256SUMS` does not list `metrics.json`
  (owner 2026-11-07, repo convention).
- A gate: every `public`/`owner` manifest artifact has a corpus-scope check (`public_ids`/`digest`, a match-count
  `expect`, or a `statsbomb_open` dataset); the receiver and five ghost artifacts gain count expectations.
- Rebaseline: a commit is a code delta only if a module's docstring-free AST differs.

### 14.4 Change 7c — public certification: in-sample parity plus CV parity (F6, F7, F11, F14)

Public certification = the head-to-head (`basis: "in_sample_parity"`), two-sided CV parity on the registry
`cv_metric` within 5 % (`match_outcome`: `rho` within 1e-6), the ADR-126 gate for xshot/xcross, and the 14.3
identity. A wheel artifact without `metrics.json` names a `cv_reference` (`expected_passing`: the research
metrics; `match_outcome`: its `model.json`; `xsuccess`: `docs/artifacts/cv_reference/xsuccess.json`, the card's
AUC 0.895 and Brier 0.084). `headtohead` takes `--certification` from the caller; `statsbomb_open` builders read
the snapshot; `flat_tc3` evaluates on the run's own `extract` rows, which retrieve copies beside the artifact.

### 14.5 Change 7d — `DependenceModel.load` takes `Path | str` (F12)

It wraps the argument in `pathlib.Path`, like its siblings.

### 14.6 Change 7e — the census output is compared like for like (F8, F9)

Verify compares the fresh `census.json`/`ranking.parquet` with the committed ones; a difference with no code delta
is an environment delta (reported), a real code delta stops for an owner ruling.

### 14.7 Change 7f — a staged, pinned open-data snapshot (F14, F15)

- The data pin stays the GitHub commit (`datasets.toml` `sb_open` `revision`); `snapshot_repo =
  "silly-kicks/hf-refit-sb-open"` holds a copy. The lock records `identity.open` and `snapshot = {revision,
  source_revision, n_files, bytes, unreadable}`; `record` refuses a snapshot of another `source_revision`.
- The snapshot stores the consumer cache form (`_cached_json`): `statsbomb_open/<id>.json`, `<id>.360.json`,
  `competitions.json`, `matches/<c>_<s>.json`, `lineups/<id>.json`, built by `stage_open_data` with the pin set.
- A job mounts it read-only as `$SILLY_KICKS_CORPUS_CACHE_DIR`; a miss raises `OpenDataSnapshotMissError`, never a
  GitHub fetch. The token inputs record `dataset.snapshot`, which verify checks against the lock. Verify's public
  builders fetch only their evaluation matches from the snapshot.
- Staged 2026-11-08 on the DGX: revision `a511dff0…`, 8,428 files, 12.46 GB; one upstream file
  (`three-sixty/3845506.json`, NUL bytes) is recorded `unreadable`.
- I5 probe (job `6af064750e0338d1bb84b12c`): mount p50 0.123 s vs GitHub 0.161 s per match (PASS); throughput 7.8
  files/s at 1, 29.5 at 4, no gain past 4, so `read_concurrency` = 4.

### 14.8 Change 7g — progress lines (F18)

`[progress] study <tag> trial <k>/<n> best=<v> in <s> s` per trial (the ruthless `observer` hook, which does not
change the search); `[progress] bootstrap <k>/<n> convergent=<c>` about every 60 s on both bootstrap paths;
`[reduce] <stage> done in <s> s`; the driver step line ends `; instances done <d>/<n>`.

### 14.9 Change 7h — retrieve and verify report progress; counters read in parallel (F16, F17)

`[retrieve] <i>/<n> <instance>: <files> files, <MB> MB, counters <k>/<units> in <s> s` (plus a 60 s heartbeat);
`_counters` lists once and reads the counter files over 8 threads, merging in sorted key order;
`[verify] <i>/<n> <instance> ok|FAIL (<reason>) in <s> s`.

### 14.10 Investigations (each before its related fix; results go into this section)

- I1 (F19). RESOLVED: the optuna version. The committed xshot/xcross models were tuned under optuna 5.0.0
  (xcross `public` `default` and `xsuccess` under 4.9.0); launch 3 ran the locked 4.8.0. The locked environment
  with optuna 5.0.0 alone reproduces each 5.0.0 model's best parameters within 3 ULP (relative ≤ 5e-16) at its
  committed best trial; extract rows and study code were already equal. xcross `public` `default` under 5.0.0
  differs (`pr_auc` −0.64 %), inside the 5 % CV parity.
- I2. RESOLVED: an unstable sort. The committed ghost_outfield extract differed from x86 in 325 of 20,696 rows
  (two defenders on exactly the same x swapped slots) because `select_back_line_players` used the default
  `np.argsort`; a stable sort makes the x86 extract equal the DGX shard in every slot. Direct callers:
  `_ghost_outfield.py:286`, `_gk_influence.py:458`, `_packing.py:185`; `compute_defensive_line` uses the batched
  `_collective.back_line_batch` with the same sort. At a tie the x-based outputs do not change; outputs that read
  the selected rows' other columns can, on x86 only.
- I3. Receiver pool: launch 4 must reproduce `pool_coverage` 51,440 / 50,719 and `keep_pool` false.
- I4. Census: the rebaseline must pass under 14.6.
- I5. The open-data mount read time (14.7).

### 14.10a Change 7i — optuna 5.0.0 (I1)

`optuna>=5.0.0,<6` beside every `ruthless-efficiency[optuna]` entry; `uv.lock` resolves 5.0.0. After launch 4 a
check confirms each of the seven 5.0.0 models' retrieved best parameters are within 3 ULP of the committed ones
(the plan's Task 27 Step 6 block); a failure stops for the owner.

### 14.10b Change 7j — a deterministic back-line tie-break at the source (I2)

`select_back_line_players` (`_defensive_line.py:96`/`:98`) and `_collective.back_line_batch`
(`_collective.py:256`/`:258`) use `kind="stable"`, as do their legacy oracles in the tests; `_ghost_outfield.py:231`
also. Defenders on exactly the same x keep their input order on every platform. The CHANGELOG records the x86
change at exact ties.

### 14.11 Tests (each seen to fail first)

In `tests/` per change: the recipe args, the corpus-identity gate and scope gate, the AST rebaseline, public
certification both ways and `cv_reference`, `DependenceModel.load(str)`, the snapshot cache per file kind and the
in-job miss, the progress and retrieve/verify lines, the optuna pin, and the tie tests.

### 14.12 Order and definition of done (HFURR-SPEC-35: one order)

I1 and I2 first; 7a–7j test-first; stage the snapshot and run the I5 probe (owner go); A/B implementation review,
`/final-review`, amend and force-push (owner gates), CI green; launch 4 (owner go); retrieve and verify 21/21 or
each difference explained or ruled, with I3, I4 and the 14.10a check. Status: CI green on `22e0b65d`; launch 4
stopped for Change 8 (§15).

### 14.13 Docs that travel with the commit

ADR-128 items, the `registry.py` head-to-head comment, `docs/howto/hf-jobs-training.md`, the `xsuccess` and
`pass_completion` card line `Corpus identity: <n> matches, sha256 <digest>.`, `CHANGELOG.md`. Done.

## 15. Change 8 — the launch 4 findings: feedback a person can monitor, a driver that survives a 429 (2026-11-08)

Status: APPROVED (A r18, B r18). Owner 2026-11-08: fix F20–F25 in this cycle, in the same amend, each in its smallest form; process
as Change 7. The version stays 4.139.1.

### 15.1 Evidence (launch 4 on `22e0b65d`, 2026-11-08)

The owner stopped launch 4 at 15:50 UTC with 14/21 instances done (291 COMPLETED, 7 CANCELED, 1 ERROR, a CSI mount
flake the driver relaunched). The Change 8 amend is a new commit, so a resume after it reruns all 21 instances
under new run_keys, every stage included (R3; about 5 h, as launch 3).
Every promised line printed (`[alive]`,
`[unit]`, `[job]`, `[reduce]`, `[read]`, one `[progress] study` per trial); no unit failed.
- F20. About 70 % of the open-data job log lines are per-match converter warnings and their source echo
  (`Inferred xy_fidelity_version`, `Inferred shot_fidelity_version`, `StatsBomb: <n> unrecognized event types
  dropped`).
- F21. xsuccess `run` (30 trials, a direct `study.optimize`) ran 81 min with no trial count; win_probability `fit`
  (1,130 s) and pass_completion `run` (1,934 s) showed only the 300 s beat.
- F22. A `429 Too Many Requests` from `cas-server.xethub.hf.co` on the per-job manifest upload (`driver.py:157`)
  ended the driver; the launched job ran on without its manifest. The shard-key write in `_job_spec` is
  unguarded too; a 429 in `_save_cache` already becomes `contention` (it runs inside `step`'s `try`), so it loses
  the step rather than the driver.
- F23. The census `bootstrap` unit spent ~6 min in a silent serial `done_results` read of 3,961 files, then
  printed no bootstrap line (it ran under 60 s).
- F24. One xsuccess trial took 1,208.6 s (5 folds of 224–241 s, single-threaded), so 30 serial trials cannot fit
  the fixed 7,200 s job limit (`driver.py:728`).
- F25. The adapter load of 3,961 matches (33–50 min) is inside the xsuccess and pass_completion units, so every
  retry, resume and probe pays it again.

### 15.2 Change 8a — the three known converter warnings are not shown in a job (F20)

`launch.job_env` sets `PYTHONWARNINGS` to three entries of the form
`ignore:<prefix>:UserWarning:scripts._sb_open_data` with the prefixes `Inferred xy_fidelity_version`, `Inferred
shot_fidelity_version` and `StatsBomb` (owner: acceptable to hide them). `:` separates the fields, so a prefix
cannot hold one; CPython matches the message case-insensitively at its start. The module field (an exact match on
the warning's attributed module, the open-data loader where the log shows them) keeps a `StatsBomb…` warning from
any other module visible. No job path resets the filters. Local runs are unchanged. Test: `job_env` carries the
string; in a job-mode subprocess the three raised from `scripts._sb_open_data` print nothing, and a `StatsBomb360 …`
warning from another module and one unrelated warning still print.

### 15.3 Change 8b — every long unit reports its progress (F21)

xsuccess's study passes `TrialPrinter.optuna_callback`, which prints the §14.8 line per trial (built and tested).
The CV loops of win_probability, the native pass_completion `fit` and the native xsuccess `fit` (its
`cross_val_metrics` at the best parameters: 5 serial folds plus the final fit, up to ~8 min a fold) print
`[progress] <trainer> fold <k>/<n> in <s> s`.
An AST test fails on any `.optimize(` call under `scripts/` without `callbacks=`.

### 15.4 Change 8c — a 429 on a driver store write never stops the driver (F22)

`_guarded_put` retries a contention error (`_is_contention`) 3 times at 2, 4 and 8 s, then returns `False`; any
other error raises. Each `False` sets `contention` and ends the step's launch loop, as a launch 429 does:
- the shard-key write in `_job_spec`: that launch is skipped, so nothing runs without its keys;
- the per-job manifest write in `step` (after `do_launch`): `[driver] job <id> manifest not recorded
  (rate-limited)`; that job is never charged (no coverage, no timeout charge) and its missing manifest is looked
  up again at each step (one read);
- `_save_cache` in `_attribute`: the cache is not saved and the step still launches (today it loses the step).

### 15.5 Change 8d — a long input read reports its progress (F23)

`done_results` prints `[read] <stage> results <k>/<n> in <s> s` at most every 60 s and once at the end when it read
100 files or more; the bootstrap always prints a final `[progress] bootstrap <n>/<n> convergent=<c>` line.
Owner 2026-11-08: `done_results` reads the listed results over 8 threads (the `_counters` pattern, §14.9) and
returns them in sorted key order, so the result is unchanged. Its callers: the census `prefit`, `bootstrap` and
`report`, the ghost_gk `extract` consumer and, after 8e, the xsuccess `hpo` and `fit` and the pass_completion
`fit` (each ~6 min serially for 3,961 results, inferred from F23; about 2 min at 8 threads, inferred from I5).

### 15.5a Change 8e — xsuccess and pass_completion become native pipelines (F25; F24 option B)

Both follow win_probability: `refs` → `extract` (one store-cached unit per match, sharded) → their HPO/CV stage →
`fit`; `adapter=True` goes. No trainer then uses the one-unit adapter, so `scripts/hf_jobs/adapter.py`, its stage
and its tests are deleted (owner 2026-11-08; ADR-128: delete the adapter stage when the trainer half of
`_LEGACY_UNMIGRATED` is empty). xsuccess keeps its exact search (TPE `seed=42`, the same objective and folds).
Tests: model bytes equal the legacy fit's except provenance; W1N1 = W2N2; the corpus identity unchanged.

### 15.6 Docs

ADR-128 items, `docs/howto/hf-jobs-training.md` (the fold, `[read]` results and `manifest not recorded` lines; the
per-stage limit), `CHANGELOG.md` `[Unreleased]`.

### 15.7 F24 measurement and decision

Probe 1 (`6af09aa5…`) lost its lines to a block-buffered pipe (author defect); probe 2 (`6af0b0d0…`) measured
trial 1 (287 trees, depth 7) at 1,208.6 s before the owner stopped it ("move along with a reasonable
suggestion"). Projection (inferred from that trial and the search space, 100–600 trees, depth 3–7): a
fold-parallel trial takes ~1–8.5 min, so 30 trials take ~2 h on average and ~4 h if every trial sits at the top
of the range.

Decision (author, at the owner's direction):
- The xsuccess `hpo` stage (`inner_parallel=True`, `inner_width=5`) runs each trial's 5 folds on the InnerPool and
  combines the fold results in fold order (`XSuccessObjective.evaluate_patch(..., fold_map=)`). The serial path is
  the same composition, so the search is unchanged. A per-fold result carries "skipped" (one training class) and
  "no PR-AUC" (one test class), and the combine keeps the `inf` result when no fold scores, as `_cv_logloss` does.
- The fold children get the data as xcross does (`train_xcross_attempt._study_fold_task`): the stage writes the
  invariant `X`/`y` (float64, `.npy`) and the fold plan (`.npz`) once to the job-local temp directory; each child
  loads them with `mmap_mode="r"` once and caches them per process. Nothing large is pickled. Expected peak
  (inferred): `X` ~2.1 GB in the shared page cache plus ~2.2 GB per child (the `X[tr]` copy and the xgboost
  matrix), ~13 GB for 5 children, under the 30 GiB cap; no `peak_rss_by_stage` hint. Escalation is unchanged
  (owner 2026-11-08): if the unit ever ran out of memory, its escalate job (`inner_workers=1`) would run the folds
  serially, time out at 4 h, and the 2-timeout rule would make it STUCK and reported (bounded).
- A stage may declare `timeout_s`; `plan()` writes it into the stage record (`_stages.py:1056`) and `_job_spec`
  uses it instead of 7,200 s. The xsuccess `hpo` stage declares 14,400 s: ~2× the projected average. The
  top-of-range worst case (~4 h, plus the `done_results` read) has no margin; the per-trial lines show it. A
  timeout costs one 4 h `cpu-upgrade` job (pennies); two make the unit STUCK and stop for the owner.
- Tests: the fold-parallel result equals the serial one bit for bit, including a skipped fold; `_job_spec` takes a
  declared `timeout_s` and defaults to 7,200 s.


## 16. Change 9 — the launch 5 findings (2026-11-09)

Status: APPROVED (A r20, B r20; CONSIDER notes applied). Owner 2026-11-09: amend the Change 9 fixes into the cycle commit and rerun all 21 instances
(launch 6); nothing from launch 5 is adopted. The version stays 4.139.1.

### 16.1 Evidence (launch 5 on `652dede9`)

- F26. xsuccess `fit` timed out twice (jobs `6af1444b…`, `6af160ba…`) and is STUCK. CV folds took 1,662 s (job 1)
  and ~2,250 s (job 2), single-threaded (`cpu+299.5` per 300 s beat). One `XSuccessModel.fit` is 6 xgboost fits
  (the model plus the 5 `cross_val_predict` fits that feed its one isotonic fit); a CV fold is one such fit on
  ~80 % of the rows, the final fit one on all rows: ~10,000–14,000 s against the 7,200 s default. Author defect:
  §15.3 estimated "up to ~8 min a fold". `hpo` finished (30 trials, ~9,800 s of 14,400 s).
- F27. `reduce_stage` (`_stages.py:894`) reads every unit result serially; only its closing `[reduce]` line prints.
  The win_probability `extract` reduce (3,961 results) showed only the 300 s `[alive]` beat for minutes.
- F28. One `--plan` makes 25–65 HF API requests (measured: win_probability 47, xsuccess 62, census 65,
  ghost_gk/default 25), 88–100 % of them `list_bucket_tree` pages: a plan lists the same stage directory several
  times. Every step re-plans all 21 instances, finished ones included; near the end the driver rate-limited itself.
- F29. Every xShot/xCross visibility arm drops the 7 Bundesliga matches. `arm_policy._injob_key`
  (`arm_policy.py:34`) builds `{provider}__{provider}-{game}` from the staged stem (`idsse`), but the trainers key a
  match `match_key(source_provider, pergame_name(provider, game))` with `source_provider` `sportec`
  (`train_xcross_attempt.py:510`, `train_xshot_occurrence.py:477`). `public` trained on 10/17 matches, `sc_extended`
  on 108/115, in all 8 instances. Evidence: the fresh xCross public params refit on the public-arm rows reproduce
  the fresh in-sample PR-AUC 0.0695 exactly; the committed params refit on all 17 reproduce the committed 0.1753.
  CV (all rows) could not see it; the §14.4 in-sample check did. It dates from the ADR-128 migration (`b2865a72`).

### 16.2 Change 9a — xsuccess `fit` on the InnerPool (F26)

The `fit` stage declares `inner_parallel=True`, `inner_width=3` (§16.7) and `timeout_s=14400`. Its
unit maps n_splits + 1 keyed inner units on `ctx.pool`: one per CV fold (`XSuccessModel().fit` on the fold's train
rows; returns the test-row predictions) and the final fit (all rows; returns the `model.save` file bytes and the
model's `feature_set`). The parent writes the pooled rows once to a job-local parquet file; a child reads it once
and caches it per process (the §15.7 pattern). The out-of-fold vector is assembled in fold order (`ordered_values`),
so the metrics and the model bytes equal the serial path, which stays the same composition. In the pool path
`on_done` prints `[progress] xsuccess fit <k>/<n> in <s> s` (n = n_splits + 1) in place of the serial fold line.
Escalation is unchanged: a memory failure reruns the unit at width 1 (`driver.py:736`), ~10,000–14,000 s against
14,400 s, so it can end STUCK and reported (bounded).

### 16.3 Change 9b — one reader for `reduce_stage` and `done_results` (F27)

The reader takes the caller's key order and returns the results in it (`reduce_stage`: the stage's
`reduce_order`, e.g. xShot/xCross `_mid_of`; `done_results`: sorted), reads over 8 threads, and prints
`[read] <stage> results <k>/<n> in <s> s` at most every 60 s and once at the end for 100 files or more. Every
reduce output is unchanged.

### 16.4 Change 9c — fewer HF API requests per step (F28)

- (A) The driver keeps, for its process life, the last plan of each instance whose output stage is reduced and
  does not re-plan it (a reduced output never changes under its run key). A restarted driver re-plans once.
- (B) As approved: `BucketStore` lists its run key ONCE, recursively, at its first `names()` call and serves every
  later `names(dir)` from that listing. Measured for one plan: xsuccess 62 requests today, 16 with one recursive
  listing (19 with a per-directory memo, the withdrawn author proposal); census 65 → 21.
- (C) `BucketStore.read_bytes` memoizes per path; `upload_bytes` writes through (the listing and the memo gain the
  file).
- Lifetime: one `BucketStore` per `--plan` process, per driver step (`driver.py:281` rebuilds the stores) and per
  `adopt.retrieve` instance; none outlives a snapshot it may serve.
- Tests: a counting fake API sees one listing per plan and an unchanged plan JSON; a `None` read is memoized; an
  upload is visible to the next `names()`/`read_bytes()`. After the change `count_plan.py` is rerun on the same 4
  instances and the figures are recorded. Pass rule: no instance above its count today (47, 62, 65, 25); a higher
  count (a recursive listing also pages through `FileSet` subdirectories) stops the change for the owner.
- A `BucketStore` with an empty prefix (`driver.py:243`, `campaign.py:310`) never lists recursively: its `names()`
  keeps today's one-directory listing, so it can never list the whole bucket.

### 16.5 Change 9d — arm keys from the staged unit's source provider (F29)

`arm_policy.compute_arms` takes the staged units (`staged_units(dataset)`: `stem`, `game`, `source_provider`) and
emits `arms.match_key(source_provider, datasets.pergame_name(provider, game))`, the key the trainers build;
`campaign.py:185` passes the units. Test: for every unit of every locked visibility-gated dataset the emitted key
equals the key the xShot/xCross ref builders compute from the same unit, and the public arm of the 17-match public
corpus has 17 matches.

### 16.6 Launch 6

Amend; owner gates (amend, force-push); CI green; preflight; `campaign scripts/hf_jobs/campaigns/cycle1.toml
launch` (all 21 instances under new run keys). xsuccess is the long pole: refs + extract, `hpo` ~2.7 h, `fit`
~1.5 h; owner 2026-11-09: its `[[jobs]]` block moves first in `cycle1.toml` (launches follow the file order), so it starts
at once (launch 5 started its extract ~1.5 h in). Then `retrieve`, `verify` and `adopt` on the clean amended HEAD, and Task 27 Step 6. No tracked edit from the
launch until adopt ends: a tracked edit changes `code_identity` and so every generation a controller command
computes (`_provenance.py:294`; the launch-5 retrieve read 2 of 20 instances that way).

### 16.7 The fit memory probe (owner go 2026-11-09)

Job `6af1b7b9…` at `652dede9`, on the launch-5 xsuccess extract (7,974,436 rows) at the hpo best parameters (568
trees, depth 7), measured each fit in its own process: one CV fold (6,379,767 rows) peaked at 6.59 GiB RSS (`ru_maxrss`) in
2,019 s, the final fit at 7.26 GiB in 2,561 s; loading the rows alone takes 1.21 GiB. (The job's `/proc/meminfo` shows the host,
so the node figure is not used.) W = 3: 3 × 7.26 GiB + ~1.5 GiB parent ≈ 23.3 GiB, within a 24 GiB budget (80 % of the 30 GiB box,
headroom for the OS and the mount sidecar; W = 4 needs ~30.5 GiB). The final fit is submitted first (longest first): two waves, ~4,000–4,600 s, up to ~6,200 s at the
slower launch-5 job speed, under the 14,400 s limit.

### 16.8 Docs

ADR-128 items, `docs/howto/hf-jobs-training.md`, `CHANGELOG.md` `[Unreleased]`.

## 17. Change 10 — the launch 6 verify findings (2026-11-09)

Status: DRAFT r23. Owner 2026-11-09: amend and rerun all 21 instances (launch 7); fold in parallel retrieve/verify
and `launch_batch` 16. The version stays 4.139.1.

### 17.1 Evidence (launch 6 on `0552f58e`: 21/21 done, verify 16 ok, 5 FAIL)

- F30. The 4 xShot/xCross `public` artifacts record raw per-game ids (`corpus_match_ids`, e.g. `[sportec,
  idsse-DFL-MAT-J03WMX]`) and no digest: both trainers call `corpus_identity(..., all_public=bool(inp.is_public.all()))`
  (`train_xcross_attempt.py:1165`, `train_xshot_occurrence.py:920`, since before ADR-128). The code holds two
  conflicting designs: "ids when all-public" (both trainer comments, the `tests/scripts/test_expect_variant_wiring.py`
  docstring and its slow test `test_public_run_records_ids_and_public_reproducibility`) and "digest only; `adopt`
  rehydrates the canonical ids" (`_corpus.rehydrate_public_corpus_ids`); ADR-123 Amendment B is silent on the id form.
  The ids a job records are per-game names, not the canonical pairs, so the first design never worked end to end; it
  stayed hidden until F29 made the public arm complete.
  Verify's `digest` mode fails, and adopt's rehydrate refuses (`corpus_n_matches=None != the pinned public arm (17)`,
  reproduced by B). Each raw list hashes to the lock's `xshot17` `pergame` identity (4/4).
- F31. xCross `public` `default`: in-sample parity 0.157 against the committed 0.175 (delta 0.01) while CV parity
  holds (0.0867 against 0.0873). The fresh params are re-tuned under optuna 5.0.0; the committed ones come from 4.9.0
  (ADR-128 item 17). The refit of the fresh params on all 17 matches gives the same 0.157. xsuccess, also in item 17,
  passed its gate (0.026591 against 0.026577).
- F32. The census rebaseline (a code delta: `scripts/_crossed_icc.py`) is not byte-identical: `tier2.icc` differs by
  4.2e-16, `tier2.lo` by 4.5e-6 because `tier2.n_boot_effective` is 400 against 399, the ranking by at most 1.1e-13;
  the committed census (4.120.0) predates the P4 pass counters. Same inputs and verdict.

### 17.2 Change 10a — the trainers record the digest in-job (F30)

Owner 2026-11-09 resolved the conflict: digest only. Both trainers pass `all_public=False`, so a job always records
`corpus_match_ids_sha256` + `corpus_n_matches`; verify's `digest` mode and adopt's `rehydrate_public_corpus_ids` are
unchanged. Inverted to the decision: `test_public_run_records_ids_and_public_reproducibility` (now: a public run
records the digest and no ids), the `test_expect_variant_wiring.py` docstring and both trainer comments. Tests: an AST gate
that every `corpus_identity` call in a `scripts/train_*.py` passes `all_public=False`; on launch-6-shaped metrics (the
digest of the raw `pergame` pairs, `corpus_n_matches` 17, `providers` `[idsse, skillcorner]`) verify's identity check
passes against `expected_identity("xshot17", "pergame")` and `rehydrate_public_corpus_ids` returns the canonical
pinned public pairs.

### 17.3 Change 10b — public certification without a parameter match (F31)

Owner 2026-11-09: xCross `default` only. `VariantBuild.in_sample_gate: bool = True`; the xCross `default` build sets
it `False`. `floor.certify(..., in_sample_gate=)`: with `False`, a `public` certification passes on CV parity alone and
records the head-to-head as `in_sample_parity` with `"gated": false`. `_certify` passes the build's flag. The flag is
per build: the `sc_extended` tier is owner-certified and never reads it. Test: the launch-6 xCross figures certify with
the flag off and fail with it on; a registry test pins xCross `default` as the only build with it off.

### 17.4 Change 10c — a census rebaseline within a stated tolerance (F32)

Owner 2026-11-09 ratified: `DriverSpec.tolerance` (the census only); a code-delta rebaseline whose outputs differ only
within it passes, with `"within_tolerance": true` and the measured differences. `census.json` after `PROVENANCE_KEYS`,
field by field: floats within 1e-9 relative (`|a-b| <= 1e-9 * max(|a|, |b|)`; zero only equals zero; NaN equals NaN
at the same field); `census.tier2.lo` within 1e-5 absolute; `census.tier2.n_boot_effective` within 1; the verdict
(all its fields, its `reason` text included) and `ranking_licensed` exact, as is every other value. The P4 pass
counters (`n_attempted`, `n_failed`, `n_counters_unrecorded`, `n_excluded`) describe the run's coverage and are not
compared with the committed census; the fresh `n_failed`, `n_counters_unrecorded` and `n_excluded` must be 0.
`ranking.parquet`: the same columns, rows and order; floats within 1e-9 relative, the rest
exact. Test: the launch-6 census differences pass; a `lo` change of 1e-4, an `n_boot_effective` change of 2, a changed
ranking row, a changed verdict or a fresh `n_failed` of 1 each fail.

### 17.5 Change 10d — faster retrieve and verify; launch batch 16

`retrieve` and `verify` work on up to 4 instances at once over threads; the returned report and the retrieve result
are in instance order (`verify.json` is written with sorted keys), and each `[retrieve]`/`[verify]` line prints when
its instance ends. Shared resources: each statsbomb_open head-to-head gets its own open-data cache folder
(`<staging>/_open_data/<instance>`), so no two processes download into one folder; every other file a worker writes is
per instance (the artifact folder, the `.headtohead.json`, the integrity verdict). Memory: the public head-to-heads
score 3 open matches (`headtohead._public_actions`), the xShot/xCross ones at most 92k extract rows; each launch-6
verify took 8-40 s, so 4 at once stays far below the box. `cycle1.toml` `launch_batch = 16`. The driver already saves
its coverage cache only when a new terminal job was attributed (`driver.py:973`; owner acknowledged).

### 17.6 Launch 7

Amend; owner gates; CI green; preflight; `campaign scripts/hf_jobs/campaigns/cycle1.toml launch` (all 21); no
tracked edit from launch until adopt ends; retrieve, verify, adopt; Task 27 Step 6.

### 17.7 Docs

ADR-128 items; the `rehydrate_public_corpus_ids` docstring keeps its contract and names the trainers' `all_public=False`;
`docs/howto/hf-jobs-training.md`; `CHANGELOG.md` `[Unreleased]`.
