# HFU — reduce reads, job exit semantics, match-outcome pipeline — design

Status: APPROVED — A and B round 4 (reports `D:\Development\_reviews\2026-11-04-hfu-reduce-reads-spec-r4-{A,B}.md`).
Post-approval edits: line ref `test_hf_driver.py:728` → `:730` (A-17) and the `_driver` import wording (B r4); no
substance changed.
**Change 4 delta APPROVED by the owner 2026-11-05 as fixed after A/B rounds 5 and 6 (reports
`D:\Development\_reviews\2026-11-05-hfu-reduce-reads-spec-r{5,6}-plan-r{3,4}-{A,B}.md`; answers below): Change 4 (§11), the E12
row, G6, the E12 non-goal, §6 measurement D, R1, R6, §9 and §10.** Every other line is approved and unchanged.
Branch: `feat/minimal-hf-driver` (PR #299). Lands as an **amend** to `f03f47536a671c20f066c7a8cf32b22836c9c7db`
(one commit per cycle; no intermediate commits). `f03f4753` is on `origin`, so the amend needs a **force-push**;
both steps are owner-gated (§7).
Scope owner: Karsten. Scope set by the owner on 2026-11-04: Change 1 (option A), Change 2, Change 3, nothing else.
Added by the owner on 2026-11-05: Change 4 (E12 diagnosis only, option (b), plus an alive signal at least every
5 minutes per job).
Added by the owner on 2026-11-05: Change 5 (§12, the lost-rename repair).
Added by the owner on 2026-11-06: Change 6 (§13, the retrain launch 2 failures). **Change 6 delta APPROVED: A spec r8, B spec r8 (`D:\Development\_reviews\2026-11-06-hfu-reduce-reads-spec-r8-{A,B}.md`).**
Reviewers: A and B (independent).

## Round-7 review answers (Change 6 delta)

Reports `D:\Development\_reviews\2026-11-06-hfu-reduce-reads-spec-r7-{A,B}.md`. Every finding is answered in
§13 itself (the `A-SPEC-*` / `HFURR-SPEC-*` tags mark each place); r8 approved by A and B.

## Round-5 review answers (Change 4 delta)

- **A-SPEC-18 / HFURR-SPEC-20 — fixed.** §11.1 now lists every process kind, verified in code: InnerPool children
  use SPAWN (`executor.py:99-101`), and the adapter's legacy-trainer subprocess and the scikit-learn loky
  children were missing. The covered kinds call `start_job_watch` explicitly at their entry point or
  initializer (inheriting the variable starts nothing). The loky children are reported by their parent
  (`children=`, §11.3), by owner decision 2026-11-05; a joblib hook comes only if a log shows a stall inside them.
- **A-SPEC-19 / HFURR-PLAN-13 — fixed.** The frame is the innermost PYTHON frame; a C builtin (`time.sleep`, a
  FUSE `read`) shows its caller. D names the probe function `_stall`, and `main` shows `unit=-` during dispatch.
- **A-SPEC-20 / HFURR-SPEC-21 — taken.** R6 measures the kill offset against the computed deadline directly in
  D, and states that one canary does not bound retrain queue time.
- **HFURR-SPEC-22 — taken.** The E12 row adds the 4 `position_only` sibling folds of `6aebcc54` and exact walls.
- **A-PLAN-13 (spec side) — taken.** D's row states that it ends `Job timeout` (one ERROR, one email).

## Round-6 review answers (Change 4 delta)

- **HFURR-SPEC-23 / A-SPEC-22 — fixed.** The kill offset comes from the probe's `[probe] utc=` lines, printed
  every 5 s (resolution ±5 s), not from the 300 s `[alive]` cadence.
- **A-SPEC-21 — fixed.** D expects the workers' dumps to name `_stall` and main's dump to name `_dispatch`.
- **HFURR-SPEC-24 / A-SPEC-23 — taken.** G6 names covered processes and child states; the volume bound adds the
  adapter runner; a pool child always shows `unit=-`.

## Round-3 review answers

A: APPROVE (2 wording CONSIDERs). B: REQUEST CHANGES (small, HFURR-SPEC-18). The owner permitted a delta-only
round 4 (2026-11-05).

- **HFURR-SPEC-18 — fixed by REVERTING the round-3 trigger change.** Its premise was false. The driver counts
  EVERY terminal reduce trigger in `reduce_errors`, COMPLETED included (`driver.py:832-834`), and
  `test_a_completed_reduce_trigger_that_left_the_stage_unreduced_is_bounded` (`test_hf_driver.py:730`) pins it.
  So an exit-0 refusal is already bounded: after `attempts` triggers the driver reports (`driver.py:595-597`).
  Making a refused trigger exit 3 bounded nothing; it only added ERROR jobs (emails), including the unnamed
  spent-memory-deferred case (`driver.py:519`, `:532`; `executor.py:255-256`). The `_stages.main` row and the
  "stays loud" paragraph are gone. A refused trigger keeps today's exit 0 for every refusal cause (pending,
  deferred, over tolerance). The one change: an over-tolerance refusal used to exit 3 through the subprocess
  (`executor.py:277-278`) and now exits 0, so it is one fewer email with the same bound and the same report.
  G2 is back to one loud exit. T3 and T4 are restated; T4 is a regression test.
- **HFURR-SPEC-19 — folded.** T3's fail-first is the exit code itself (today exit 3, after the change exit 0).
- **Outside round (B) — folded, no new scope.** The `_reduce_or_report` docstring (`driver.py:585-586`) says
  "ERROR jobs"; the code counts all terminal triggers. The plan edits that function to call the shared
  predicate, so it corrects the docstring in the same edit.
- **A-SPEC-15 — fixed.** The import-edge sentence now names `_stages`' module-level imports exactly.
- **A-SPEC-16 — fixed.** The round-2 answer below now says T1-T6.

## Round-2 review answers

A: APPROVE (1 SHOULD FIX). B: REQUEST CHANGES (small). Both raised the same trigger point.

- **A-SPEC-11 / HFURR-SPEC-14 — superseded in round 4.** Round 3 made a refused trigger exit 3 on the premise
  that the driver counted only ERROR triggers. That premise was false (see HFURR-SPEC-18 above); round 4
  reverts the change. The bound is the existing count of every terminal trigger.
- **HFURR-SPEC-01 (partial) / A-SPEC-14 — fixed.** The tests are now in the §4.1 body (T1-T6). T1 is the E8
  path: it asserts exit 0, siblings completed, no reduce output, and fails on today's code.
- **A-SPEC-12 / HFURR-SPEC-16 — fixed.** `read_bytes` returns `None` in exactly the cases today's `is_file()`
  pre-check does (missing, a directory, ENOTDIR/ELOOP/EBADF), and `stat`s only on the error path (§3.1).
- **A-SPEC-13 — fixed.** §3.3 names the guard's single-process seam across the reduce subprocess.
- **HFURR-SPEC-15 — fixed.** The §5.3 part-1 oracle reads the legacy shards as the legacy fit did (the
  `*.parquet` glob), not through INV-3: that dir holds 3,961 `.parquet` but only 3,956 `.sha256`.
- **HFURR-SPEC-17 — fixed.** §4.1 names the new `driver.py` → `scripts._stages` import edge.

## Round-1 review answers

Both reviewers blocked on the same defect; it is fixed in §4.

- **A-SPEC-01 / HFURR-SPEC-01 (BLOCKING) — fixed.** E8 now names the real raise site:
  `LocalExecutor._check_failure_rule` (`executor.py:341-348`), called during dispatch (`:416`) and after it
  (`:454`), which calls `_terminate(procs)` first. `run_stage` raises before `try_reduce` is reached
  (`_stages.py:1064` vs `:1065`). §4.1 now lists every tolerance site with its new behaviour, decides
  `abort_first_k`, single-sources the tolerance predicate across all of them, and names the fail-first test
  that reproduces the observed path (one failing unit, `max_failures=0`, multi-unit job).
- **HFURR-SPEC-02 (BLOCKING) — fixed.** §7 adds two explicit owner gates: the amend (diff + file list shown),
  then a separate yes for the force-push of `feat/minimal-hf-driver`.
- **A-SPEC-02 — fixed.** The reader takes the listing it is given, so `reduce_stage` lists once. §3.3 states
  the listing budget of the WHOLE run-job reduce path (`try_reduce` + subprocess) and the guard pins it.
- **A-SPEC-03 / HFURR-SPEC-04 — fixed, no scope change.** G1 now defines "reduce site" and the budget as "each
  file a site needs is read once". `item_counters` is IN G1 (it already reads each `.counters.json` once). The
  §3.2 census rule now includes per-key `read_bytes` loops.
- **A-SPEC-04 — fixed.** The parity oracle runs BEFORE the commit (§5.3), on the legacy run's own shards. The
  match set is held fixed by the pinned open-data revision (`launch.py:152`) and checked key-for-key against
  the legacy shard set; a difference stops the cycle and is reported, never accepted.
- **A-SPEC-05 — fixed.** §6 measures 4 concurrent readers (the E3 tail-job count).
- **HFURR-SPEC-03 — fixed.** §4.3 names both capture paths (`run_unit`, and `for_each` → `_run_item`), the
  additive `for_each` change, and one test per path.
- **HFURR-SPEC-05 — fixed.** The fourth cv failure is its own row (E12), with its own §6 metric; a fix is an
  owner scope question (§8 R1).
- **CONSIDER items — all folded:** line refs corrected (A-06/07, B-06); E7 wording tightened (A-08); the
  `require_reduced` needs-path stated (A-09, B-13); stats vs reads counted, and `read_bytes` drops its
  pre-read `stat` (A-10, B-09); verify before the summary write (B-07); `require_reduced` keeps the failure
  count (B-08); `abort_first_k` vs legacy `max_consecutive_failures` parity stated (B-10); the xcross site
  labelled a probe loop (B-11); `.tmp-*.part` orphans ignored (B-12).

## 1. Why

The cycle-1 retrain on `f03f4753` reached 11/21 instances and was stopped. The handoff blamed job sizing
(4 shards of ~990 games each over the flat 2 h `timeout_s`). The HF job list and the job logs disprove that.
This spec fixes the measured causes.

### 1.1 Evidence (cycle-1 run on `f03f4753`, 211 HF jobs: 198 COMPLETED, 10 ERROR, 3 CANCELED)

The 10 ERROR jobs: 3 `exit code: 3` (E8), 4 `Job timeout` (3 in E3, 1 in E12), 3 `Volume mount failed` (E11).

| # | Observation | Source |
|---|---|---|
| E1 | win_probability extract ran 8 waves of 3-8 shards (`Nof12`, `Nof10`; partition sizes 330 … 18-19 units). Each job finished in minutes; units take 1.4-2.8 s. | HF job list; `_partitions/.../extract/3c47d807efdf22a7/`; job logs |
| E2 | The 4 `Nof4` shards held **19 units each** (76 residual units), not ~990. | `default/{0..3}of4.json` |
| E3 | 3 of those 4 jobs hit `Job timeout` (wall 7,200.5 s). Job `0of4` extracted its 19 units in 1.8-2.4 s each, then spent the rest in the INLINE REDUCE and died: `reduce_stage` (`_stages.py:791`) → `read_result` (`:471`) → `store.names` (`:266`) → `is_file()` → `OSError: [Errno 107] Transport endpoint is not connected`. | log of `6aebcc550e0338d1bb804b69` |
| E4 | The generation dir holds **11,901 entries**: 3,961 `.parquet` + 3,961 `.sha256` + 3,953 `.ledger.json` + `_summary.json`/`_inputs.json` + `.tmp-*.part` orphans. `reduce_stage` calls `read_result` once per done unit, and each call lists and stats the WHOLE dir: ~47 M `stat` calls. | bucket listing; code |
| E5 | Each done unit is read **5 times** in a reduce: `unit_states(verify=True)` (`:518`, `_verified` at `:523`) reads payload + `.sha256`; `read_result` reads both again (`:472`), then the payload a third time (`:474`). `LocalStore.read_bytes` also stats before each read (`:257`). | code |
| E6 | The HF bucket mount serves ~24 small files/s (~42 ms each), whatever the reader pool width (measured 2026-10-30, one container). The cost driver is the NUMBER of files read. | memory `reference_hf_bucket_mount_small_file_ceiling` |
| E7 | 3 failing `xshot_occurrence/sc_extended` cv jobs (`6aebc755`, `6aebcc54`, `6aebdbee`) ran inside or across 00:23-02:22 UTC, the window when the 3 win_probability jobs stat-stormed the same bucket mount. The one cv success (`6aebe728`) started 02:17, near the window's end. A failing fold recorded `OSError: [Errno 5] Input/output error` after 4,065 s. **Time correlation, not proof.** | HF job list; `cv/31296fc9a0db767d/single__0003.failed.json` |
| E8 | Those 3 jobs ended `exit code: 3` with `STAGE FAILED: cv: 1 failed unit(s) > tolerance 0 (e.g. single__000N)`. That string is `LocalExecutor._check_failure_rule` (`executor.py:341-348`): during dispatch it calls `_terminate(procs)` on the running sibling folds, then raises `StageFailedError`; `_stages.main` maps it to exit 3 (`:1070-1072`). The unit failure was already recorded in `<key>.failed.json`. Each ERROR job emails the owner. | logs of `6aebc755…`, `6aebcc54…`, `6aebdbee…`; code |
| E9 | The failure record holds only `"OSError: [Errno 5] Input/output error"`: no path, no frame. The EIO site cannot be attributed. | `single__0003.failed.json` |
| E10 | `match_outcome_dependence` runs as ONE adapter unit (spec 2026-10-18 §5.6, "until Cycle 2"): 5,543 s for 3,961 matches, serial. That is 77% of the 2 h wall. | HF job `6aebbf3fc78863f9c6330884` |
| E11 | The 3 `Volume mount failed: MountVolume.SetUp failed` errors are platform-side at mount setup and already RETRY (`mount_failed`, ADR-128 am. 2026-11-04). | HF job list |
| E12 | The 4th cv ERROR, `6aebdcd5c78863f9c6331557` (`xshot_occurrence/sc_extended`), ran 01:33-03:33 and ended `Job timeout`. Its log ends at `Installed 55 packages`: no unit output for 2 h. Cause unknown; it is not a reduce and not a recorded unit failure. **Added 2026-11-05:** the job held all 5 `position_only` folds and wrote no `.ledger.json` and no `.failed.json` in 7,200 s. The `default` job `6aebe728` ran its 5 folds in parallel and finished each in 2,332.9-2,334.6 s. So E12 ran 3× a healthy fold time and completed nothing: a stall is likely, slow-but-alive is not excluded. The only `position_only` fold record is the 4,065.65 s EIO failure of job `6aebcc54`, whose 4 sibling folds ran at least that long without finishing before `_terminate` stopped them (E8). A plain `Stage` prints nothing per unit (`run_unit`, `worker_main`), so a HEALTHY cv log also ends at `Installed 55 packages` until exit: silence alone is no evidence. | `_partitions/xshot_occurrence/cv/31296fc9a0db767d/by_job/6aebdcd5c78863f9c6331557.json`; `cv/a956212bf3dc8a8a/single__000{0..4}.ledger.json`; code |

## 2. Goals, non-goals, anti-goals

**Definitions.** A *reduce site* is code that reads the per-unit results (or per-unit sidecars) of one
generation to build one value: `reduce_stage`, and every unit work that loops over another stage's unit keys
(§3.2 lists them). A *run job* is a driver-launched job that runs units of one stage.

**Goals.**
- G1. A reduce site lists its generation directory a constant number of times, independent of the unit count,
  and reads each file it needs once (result payload, `.sha256`, and any sidecar it consumes, such as
  `.counters.json` or `.excluded.json`).
- G2. A recorded unit failure never turns a run job into an HF ERROR. The one exception is the systematic
  stop (`abort_first_k`), which stays loud by design (§4.1).
- G3. A unit failure records where it happened, for plain `Stage` and `ItemStage` units.
- G4. `match_outcome_dependence` runs as a sharded ADR-128 pipeline, so no single job carries the whole corpus.
- G5. The fix is proven on HF against the operating envelope (§6), not by a green suite alone.
- G6. A running job shows it is alive at least every 5 minutes, with the unit it runs and where its code is; a
  job that reaches its timeout leaves, before HF kills it, the thread stacks of every covered process (§11.1) and
  the kernel state of every child process (§11.3) in its log.

**Non-goals.**
- Any job-sizing change (`units_per_job`, `max_shards`, `timeout_s`) or `mount_cap` change: no evidence (E1, E2).
- A store-format change (digest in the file name, per-job bundles, HTTP reads): options B/C/D, deferred by the
  owner until §6 measures option A.
- Migrating `pass_completion` or `xsuccess` (they stay on the adapter; they ran 91 s and 178 s).
- Any change to unit compute speed (cv fold fits, extract work), or a FIX for E12. Change 4 diagnoses E12; a fix
  is a new owner scope question once the evidence exists.
- Any `silly_kicks/` (wheel) change. The version stays 4.139.1.

**Anti-goals (a failure even if it works).**
- A second representation of unit results, or a cache that can disagree with the store.
- A weaker checksum contract (INV-3) to save reads.
- Hiding a real failure: a stage that cannot reduce must still be reported by the driver, and a local
  `--run-all` must still fail.
- A retry loop inside a job that masks a dead mount.

## 3. Change 1 — reduce sites read the store once (option A)

### 3.1 Contract

One reader in `scripts/_stages.py` owns "read the done results of one generation":

- It takes the generation's listing from its caller (a `set[str]` from one `Store.names` call). It never lists.
- For each requested key it reads the payload **once** and the `.sha256` sidecar **once**. It verifies the hash
  in memory and decodes from the same bytes. A `.fileset.json` result also reads each member once (unchanged rule).
- Key resolution is by exact name (`{key}{suffix}`), as `_present_suffix` does today, so `.tmp-*.part`
  orphans and other entries in the listing are ignored.
- A key whose result is absent or fails its checksum is returned to the caller as such. The caller applies INV-3
  (delete result + sidecar with the existing warning, treat as pending). The reader never deletes.
- Output order and values are identical to today's `{k: read_result(...)}` for the same store contents.

`LocalStore` changes (both are mount-operation savings, same results):
- `names` lists with `os.scandir` and `DirEntry.is_file()`. With `d_type` that needs no per-entry `stat`;
  without it, it falls back to `stat` and is never worse than today (§8 R2).
- `read_bytes` opens the file directly instead of `is_file()` then read. It returns `None` in exactly the cases
  where today's `is_file()` returns False (missing, a directory, ENOTDIR/ELOOP/EBADF): on an `OSError` from the
  open it returns `None` if `not p.is_file()` and re-raises otherwise. So it `stat`s only on the error path, and
  a real I/O error (for example EIO) still raises.

### 3.2 Sites (census)

| Site | Today | After |
|---|---|---|
| `reduce_stage` (`_stages.py:791`) | `unit_states(verify=True)` + per-key `read_result` | one listing → `unit_states(verify=False)` on it → the reader on the same listing |
| `build_territory_ranking_census.py:660` | per-key `read_result` | the reader |
| `train_ghost_gk.py:499` | per-key `read_result` | the reader |
| `train_xcross_attempt.py:677` (a probe loop bounded by `PROBE_KEEP`, not a corpus reduce) | per-key `read_result` | the reader (same contract, small N) |
| `item_counters` (`_stages.py:814-831`; called at `train_ghost_gk.py:808`, `train_receiver_model.py:472`, `:491`) | one listing + one `.counters.json` read per done unit | unchanged: already meets G1 |
| `reduce_stage` excluded loop (`:772-775`) | one `.excluded.json` read per excluded key | unchanged: already meets G1 |
| `adopt._counters` (`adopt.py:101-104`, controller side, added by the plan census 2026-11-05, owner-confirmed) | no listing; one `.counters.json` read per key, plus `.excluded.json` only when it is absent | unchanged: already meets G1 |
| `deferred_memory_units` → `_failure_record` (`_stages.py`, added by the plan census 2026-11-05, owner-confirmed) | one listing (`unit_states`) + one `.failed.json` read per failed key | unchanged: already meets G1 |

`read_result` stays for single-result reads (`_stages.py:303`, `:804`; `adopt.py:137`).

The plan proves the census complete: grep every `read_result(`, `read_bytes(` and `names(` call that runs
inside a loop over unit keys, across `scripts/`, with evidence for each hit.

**Order in `reduce_stage`.** Verification runs BEFORE `_summary.json` / `_inputs.json` are written. Today the
summary is written first (`_stages.py:787`, `:790`), which left an orphan `_summary.json` without `_output` in
`3c47d807efdf22a7`. After the change a checksum failure raises before any write.

### 3.3 Budget and guard

The run-job reduce path is `try_reduce` → `reduce` subprocess → `reduce_stage`. Its listing count is a
constant: `try_reduce` lists for `output_present`, `unit_states` and `deferred_memory_units`; `reduce_stage`
lists once; `try_reduce` lists once more for `output_present` after the subprocess. The plan states the exact
number from the code; it must not depend on the unit count.

A test asserts the budget EXACTLY on an instrumented `Store`, at two unit counts:
- the listing count of the full run-job reduce path equals the stated constant at both sizes;
- `read_bytes` calls = 2 per done unit (+ fileset members, + `.excluded.json` per excluded key);
- the reduce output equals today's output for the same store.

The instrumented store counts `Store` method calls. The path crosses a process boundary
(`LocalExecutor._subprocess`, `executor.py:266-278`), so the guard runs it in ONE process: it replaces the
subprocess call with an in-process call of the same `reduce` entry on the same instrumented store. The plan
names that seam; production code keeps the subprocess. The two `LocalStore` changes in §3.1 are covered by their
own unit tests (they reduce `stat`s per call, which a `Store`-level count cannot see).

The test proves it detects the defect: against today's per-unit `read_result` loop it fails. An exact count is
stronger than a growth exponent and cannot flake. It is NOT added to `SCALE_GUARDED`: that registry is pinned
to the AST-derived `group_rows` caller set (`tests/_scale_guarded.py`), and the reader does not call
`group_rows`.

## 4. Change 2 — a recorded unit failure does not fail the job

### 4.1 Contract

**Principle.** `max_failures` is a REDUCE-TIME rule: "may this stage reduce without these units?" ADR-128
(`:136-141`) already states it so for the driver: a failed unit is retried while attempts remain, and
`max_failures` decides only when no unit can run. The run-time copies of the rule date from the 2026-10-18
spec's "parallel-safe form of `for_each`'s consecutive-failure rule", before the driver retried units. Under
retries they do harm: they kill healthy sibling units (E8) and turn a recorded, retryable failure into an HF
ERROR. `abort_first_k` is a different rule: "the first k completed units all failed" means a systematic defect,
so stopping loud is correct.

**Every site, old and new behaviour:**

| Site | Rule today | After |
|---|---|---|
| `_worker_should_stop` (`executor.py:175-185`) | stop claiming when failures > `max_failures`, or first-k all failed | stop claiming only when first-k all failed |
| `_check_failure_rule` (`executor.py:341-352`, called `:416`, `:454`) | failures > `max_failures` → `_terminate(procs)` + raise; first-k all failed → terminate + raise | first-k only: terminate + raise (exit 3), unchanged |
| `_run` end check (`executor.py:334-338`) | failures > `max_failures` → raise | removed |
| `try_reduce` (`executor.py:243-258`) | no pending → start the reduce subprocess, which exits 3 over tolerance → `StageFailedError` | over tolerance → return "not reduced" without the subprocess |
| `reduce_stage` (`_stages.py:766-769`) | over tolerance → `StageIncompleteError` | unchanged (a direct reduce over tolerance still refuses) |
| `require_reduced` (`executor.py:260-264`, called by `--run-all` and by `run_stage` for each `needs` stage at `:291`) | raises "upstream stage … is incomplete" | still raises; the message keeps the counts (pending, failed, tolerance) |
| driver `_reduce_or_report` (`driver.py:592`) | spent > `max_failures` → report | unchanged behaviour; its docstring (`:585-586`, "ERROR jobs") is corrected to "every terminal trigger", which the code counts (`:832-834`) |

**One predicate.** "failed count exceeds tolerance" (`max_failures is not None and n > max_failures`) is one
function in `scripts/_stages.py`. `reduce_stage`, `try_reduce` and the driver (`driver.py:592`) call it. After
the table above, no other site applies the tolerance rule. This adds one import edge, `scripts/hf_jobs/driver.py`
→ `scripts._stages` (the driver does not import it today). `_stages`' module-level imports are the stdlib plus
`scripts._driver` (stdlib plus the leaf `scripts._item_outcome`) and `scripts._item_outcome` (stdlib only), so the
controller pays no heavy import; the
plan confirms there is no import cycle.

**A refused reduce trigger is already bounded.** A reduce trigger is a keyed run over an EMPTY key set
(`driver.py:43`, `:598`). `_stages.main` ignores `try_reduce`'s result (`_stages.py:1065`), so a trigger whose
stage does not reduce exits 0, for every refusal cause (pending, memory-deferred, over tolerance). The driver
counts every terminal trigger of an un-reduced stage in `reduce_errors`, COMPLETED included (`driver.py:832-834`,
pinned by `test_hf_driver.py:730`), and after `attempts` it reports (`:595-597`). This cycle keeps that
unchanged. The only trigger change is a consequence of the `try_reduce` row: an over-tolerance refusal now
exits 0 instead of 3, which is one fewer email with the same bound and the same report.

**Resulting job behaviour.**
- A driver-launched run job with a recorded unit failure finishes its other units and exits 0. Its healthy
  siblings are no longer killed, so they no longer spend an attempt without running (A-SPEC-01).
- A run job whose first `abort_first_k` completed units all failed still terminates and exits 3 (one email for a
  systematic defect). The driver classifies it RETRY `unit_crash` and charges only not-yet-spent units,
  recorded first (unchanged).
- A local `--run-all` run over tolerance runs every unit, then fails at `require_reduced` with the counts in
  the message. It no longer stops early on the first over-tolerance failure; the recorded results stay for
  resume.
- A `--stages a,b` job whose stage `a` does not reduce fails at `b`'s `require_reduced` (exit 3). The driver
  launches one stage per job and only when its `needs` are reduced, so G2 holds on the driver path.
- A reduce trigger whose stage does not reduce exits 0 (as today); the driver's trigger count bounds it.

**Tests (each fails on today's code unless marked "regression").**
- T1 (the E8 path). A multi-unit run job through `_stages.main` (`--stages <s> --shard-keys <file>`), one unit
  raises, `max_failures=0`, the other units healthy. Assert: exit 0; every other unit has a verified result
  (none terminated); the failing unit has `.failed.json`; no stage `_output`. Today: exit 3 and siblings killed.
- T2 (regression). The first `abort_first_k` completed units all fail. Assert: siblings terminated, exit 3.
- T3. A reduce trigger (empty `--shard-keys`) on a stage over tolerance. Assert: exit 0; no reduce subprocess
  started; no `_output`. Today: exit 3 (subprocess rc 3 → `StageFailedError`, `executor.py:277-278`).
- T4 (regression). The existing `test_a_completed_reduce_trigger_that_left_the_stage_unreduced_is_bounded`
  (`test_hf_driver.py:730`) stays green: COMPLETED triggers are counted and the driver reports after `attempts`.
- T5. `--run-all` with one failing unit over tolerance. Assert: every other unit ran; exit 3 at
  `require_reduced` with the counts in the message.
- T6 (driver, §4.2). A COMPLETED run job with a recorded failed unit charges that unit one attempt, and the unit
  is a launch candidate again while its coverage is below `attempts`.

### 4.2 Why the driver stays correct (verified)

The driver's coverage recompute credits every unit of a COMPLETED job (`driver.py:773`). So a run job that
exits 0 with a recorded failed unit still spends one of that unit's `attempts`, and the failed unit is a
candidate for the next launch (`driver.py:518-519`). Retries stay bounded. A unit spent at `attempts` reaches
`_reduce_or_report`, which reduces within `max_failures` or reports, as today.

### 4.3 Failure locus

The failure record gains a bounded, additive `traceback` field (formatted exception, last frames, capped
length). There are two capture paths:

- **Plain `Stage`:** `run_unit`'s `except Exception` (`_stages.py:719-720`) has the exception and records it.
- **`ItemStage`:** the exception is caught inside `_driver.for_each` (`_driver.py:810-811`), which keeps only a
  string. `for_each`'s result gains an additive per-key traceback map next to `failures`. `failures` keeps its
  exact string format, so legacy CLI callers see no change. `_run_item` (`_stages.py:688-690`) passes the
  traceback to `run_unit`, which records it.

The `error` string keeps its exact current format (Hyrum: `_failure_record`, the driver's `failed_units`,
`campaign status` read it). The new field never enters a unit result, a token or the stage output, so it cannot
change any generation or artifact byte. One test per capture path.

## 5. Change 3 — `match_outcome_dependence` as an ADR-128 pipeline

### 5.1 Shape

`scripts/train_match_outcome_dependence.py` declares `PIPELINE`, the same shape as `train_win_probability.py`:

- `refs` (one unit): the competition set + every match ref, through the same `_sb_open_data` walk the legacy CLI
  uses (`list_open_data_refs`, via `open_data_source`).
- `extract` (`ItemStage`, one unit per match): load through `load_open_data_match` (open data over HTTP with the
  resolved cache dir, the legacy loader; `dataset_files=()`), after `assert_statsbomb_open_data_mode()` as
  win_probability does. Work = the existing pure `extract_match_slice`. `max_failures=None` (the legacy
  `for_each` recorded failures and fit over the rest).
- `fit` (one unit): `matches_from_shards` → `fit_rho` → a FileSet with `model.json` + `SHA256SUMS`, the same
  fields and serialization as today (`rho`, `training_commit`, `run_tree_dirty`, `corpus`, `n_matches`).

The pure kernels (`fit_rho`, `_nll`, `extract_match_slice`, `matches_from_shards`, `_match_tuples`) do not change;
`validate_match_outcome_calibration.py` and the tests import them.

**Failure-rule parity.** Legacy `for_each` aborted after 3 consecutive failures anywhere
(`max_consecutive_failures=3`); the `ItemStage` aborts when its first `abort_first_k=3` completed units all
failed (`_stages.py:181`). The rules differ only on failure; with zero failures (the legacy run: 3,961 of 3,961)
the outputs are identical. The pipeline uses the `ItemStage` rule, as win_probability does.

### 5.2 Wiring

- `registry.py`: drop `adapter=True`; add `pipeline="scripts.train_match_outcome_dependence"`; keep the build
  destination `match_outcome/weights` with no subdir.
- `adapter.py`: `_SHARD_ROOT_TARGETS` exists only for this trainer. With it migrated, remove the constant and its
  branch (Chesterton's fence checked: `git log -S _SHARD_ROOT_TARGETS` shows one commit, `b2865a72`, which added
  it for this trainer only).
- `tests/scripts/test_compute_contract.py`: remove the trainer from the "Cycle 2 … adapter" allowlist.
- `TODO.md` "ADR-128 Cycle 2" row: `match_outcome_dependence` done; `pass_completion` and `xsuccess` remain.
- `adopt.py`, `headtohead.py`, `docs/artifacts/manifest.toml`: keyed by job name; the plan confirms each still
  resolves the artifact (the destination does not change).
- `test_match_outcome_train.py::test_train_main_writes_loadable_artifact` moves to the pipeline entry point.

### 5.3 Parity oracle (before the commit)

The legacy run on the f03f4753 store left both its artifact (`adapter/run/7186d1745ce95ed1/_output.files/
model.json`: `n_matches` 3961, `rho` 0.02161420866397881) and its 3,961 per-match shards
(`…/target_out/shards/aa3a3b14c7ff26e9/statsbomb__<id>.parquet`). The oracle runs before the commit, in two parts:

1. **Fit parity (offline, on the legacy shards).** Read the legacy shards exactly as the legacy fit did:
   `sorted(shard_dir.glob("*.parquet"))` with `pd.read_parquet`, file name = key (`train_match_outcome_dependence.py:151`).
   Do NOT read them through INV-3: that dir holds 3,961 `.parquet` but only 3,956 `.sha256` (plus 5 `.tmp` parts).
   Run the new `fit` stage's work over them. `rho` and `n_matches` must equal the legacy artifact EXACTLY (same
   kernel, same order).
2. **End-to-end parity (the §6 measurement campaign, HF, `_canary` store).** Run the new pipeline on HF. Its
   `refs` key set must equal the legacy shard key set exactly, and its `rho` / `n_matches` must equal the
   legacy artifact exactly. The open-data revision is pinned per job from the dataset lock
   (`launch.py:152`, `SILLY_KICKS_OPEN_DATA_REVISION`), so the match set cannot drift silently.

On any difference, the cycle stops and the difference (key-set diff or `rho` delta) is reported to the owner.
A difference is never accepted as drift.

The compute-contract W1N1-vs-W2N2 byte-identity test covers the new pipeline (the born-ready gate).

## 6. Definition of done — measured on HF

**Before the commit: a measurement campaign (no artifact, no push).** Patch-on-base jobs: check out `f03f4753`,
apply the working-tree diff (its sha256 in the job labels), write only to a `_canary/<run_id>` store, mount the
f03f4753 bucket `read_only=True` where they read it. The probe script lives outside the repo (it ships nothing).

| Measurement | Before (observed) | Target |
|---|---|---|
| A. 4 concurrent jobs, each running the new reader over win_probability extract gen `3c47d807efdf22a7` (3,961 units) | > 7,200 s, jobs died (E3) | each ≲ 6 min if the ~24 files/s ceiling is per container; the run states which it is |
| A. per job: listing wall time, `names` calls, reads per unit | 3,961 listings, 5 reads/unit | constant listings (§3.3), 2 reads/unit |
| B. the new match-outcome pipeline end to end | one 5,543 s job | no job near the 7,200 s wall; §5.3 part 2 parity holds |
| D. one job, `timeout_s=1500`, `--workers 2`, two units whose probe function `_stall` prints `[probe] utc=<epoch>` every 5 s and never returns (§11.6) | E12: a silent log | a start line at once and an `[alive]` line every 300 s ± one poll from `main`, `w0` and `w1`; each worker line names its unit, a growing `age=` and `at=…_stall`; the `main` line shows `unit=-` and `children=` listing both workers; the deadline dumps (`Timeout (`) before the kill: each worker's names `_stall`, main's names `_dispatch` (it waits in `time.sleep(poll_interval_s)` in `LocalExecutor._dispatch`); the measured kill offset (R6). The job ends `Job timeout`: one ERROR and one email by design |

**Measured 2026-11-05 (Task 8, patch sha256 `009859b3…`; 21 jobs, 20 COMPLETED, 1 ERROR = D by design).**
- C (fit parity, §5.3 part 1): `model.json` and `SHA256SUMS` byte-identical to legacy. PASS.
- A: 1 listing per job (1.0–2.2 s), 2 reads per unit, 0 missing. Each of the 4 concurrent jobs read for 702–725 s,
  ~11 files/s; the single C job read at ~8.5 files/s. So the ceiling is PER CONTAINER, and the serial read
  latency (~90 ms per file) sets it, not the ~24 files/s of E6. The ≲ 6 min target is missed. **Owner decision
  2026-11-05 (1a): accept the ~12 min reduce of 3,961 units; it is far inside the 7,200 s wall.** A reduce reads
  at ~11 files/s per job.
- B: longest job 913 s (the keyed rerun plus the extract reduce); refs 146 s; extract jobs 197–302 s; fit 47 s.
  Parity: `model.json` equal to legacy except `run_tree_dirty`; the refs key set equals the 3,961 legacy stems.
  Write scope unchanged (65,489 files outside `_canary/`). PASS. Finding: the mount lost 55 of ~11,900 result
  renames (§12).
- D: every listed item present; kill offset −5 s. PASS. Finding: the three dumps fired in the same second and
  interleaved character by character. **Owner decision 2026-11-05 (3b): stagger the dump time per process (§11.4).**

**After the commit: the full cycle-1 retrain.** Report from the HF job list:
- cumulative ERROR jobs (= emails), by reason. Targets: no `exit code: 3` from a recorded unit failure; no reduce
  timeout;
- cv-job timeouts, reported separately (E12 recurrence), each with its last `[alive]` lines and its deadline
  dump;
- whether the xshot cv `Errno 5` failures recur, with the traceback the record now holds (tests E7);
- the longest job wall-clock against 7,200 s;
- `match_outcome_dependence` stage wall-clocks;
- total wall-clock to 21/21 instances.

## 7. Owner gates (in order)

1. Spec, plan and implementation each pass A/B review (at most 3 rounds each without owner permission).
2. **Measurement campaign launch** (§6, HF spend): owner yes, with the job count and flavor shown.
3. `/final-review`, then **the amend**: the session shows the full diff and file list and waits for the owner's
   explicit yes.
4. **The force-push** of `feat/minimal-hf-driver` (it rewrites `f03f4753`, which is on `origin`): a separate
   explicit owner yes, after the amend.
5. CI green on PR #299, then **the retrain launch**: owner yes.

## 8. Risks

- R1. **E7 is a correlation, and E12 has no diagnosis.** If EIO or a silent cv hang recurs after this cycle,
  Change 2 still stops the emails for recorded unit failures, and the §4.3 traceback names any recorded site.
  A hang records no unit failure; Change 4 makes it leave its location in the log. A fix for it is a new scope
  question for the owner.
- R6. **The HF timeout clock may start before the bash script.** The deadline is taken at bash start, so image
  pull or scheduling time before it would move the real kill earlier than the computed deadline. The 600 s
  margin (§11.4) absorbs that. Measurement D measures it directly: the kill offset = (the last `[probe] utc=`
  epoch in the log, printed every 5 s, so ±5 s) − (the computed deadline = `main`'s start-line dump time + 600). If the job is killed 300 s or more (half the margin) before the
  computed deadline, the cycle stops and the offset goes to the owner before the commit. One unloaded canary
  does not bound queue time under the retrain load; the retrain report states, for every timed-out job,
  whether its dump landed.
- R2. **`d_type` on hf-mount is unverified.** `scandir` is never worse; measurement A records the listing time.
- R3. **Amend = new run_key.** All 21 instances rerun; only `cross_commit_version` extract stages carry over.
  Accepted by the owner (2026-11-04).
- R4. **The match-outcome loader is network-bound.** Sharding spreads it across jobs; it does not make one
  match faster. Measurement B and the retrain report the stage wall-clocks.
- R5. **The mount ceiling may be shared across containers.** Measurement A at 4 concurrent jobs shows it. If it
  is shared and the concurrent time is unacceptable, options B/C come back to the owner.

## 9. Docs that travel with the commit

- ADR-128: a new amendment for these four changes, with the corrected diagnosis (the 2026-11-04 sizing
  hypothesis was wrong), the reduce-time-only `max_failures` rule and the job liveness contract (§11).
- `docs/howto/compute-pipelines.md`: the reduce read rule (list once, read once), the job exit rule, and how to
  read `[alive]` lines and a deadline dump in a job log.
- `CHANGELOG.md`: extend the existing `[Unreleased] — minimal HF Jobs driver …` section (scripts-only, version
  stays 4.139.1).

## 10. Left to the builder (plan)

- The reader's name and signature; the predicate's name.
- The traceback format and cap; the `for_each` result field name.
- The exact listing constant of the run-job reduce path (§3.3), from the code.
- The exact `PIPELINE` params for match-outcome (mirror the legacy CLI flags: competitions, `max_matches`,
  `cache_dir`).
- Test layout, following the existing `tests/scripts/` modules.
- Change 4: the module and function names, the exact `[alive]` line layout (it holds the §11.2 fields), and the
  measurement-D probe stage (outside the repo, like the other probes).

## 11. Change 4 — a running job shows it is alive; a timed-out job leaves its stacks (E12 diagnosis)

Owner scope 2026-11-05: diagnosis only (option (b)), plus an alive signal at least every 5 minutes per job.
Owner decision 2026-11-05 (third-party pool processes): report them from their parent now (§11.3); hook joblib
so they report themselves only if a log later shows a stall inside them. The design keeps that extension a
one-call change (`start_job_watch` in a loky initializer); nothing else moves.

### 11.1 Every process kind in a run job (verified in code)

| Process | Started by | Covered by |
|---|---|---|
| `datasets fetch` pre-step (download datasets only) | `job_command` | `start_job_watch("fetch")`, first statement of `datasets.main` |
| stage main (`python -m <module>`, every trainer and the adapter) | `job_command` | `start_job_watch("main")`, first statement of `_stages.main` |
| executor worker | `_dispatch` `Popen` (its `launch` helper) | `start_job_watch(<worker id>)` in the `worker` branch of `executor.__main__` |
| reduce subprocess | `LocalExecutor._subprocess` | `start_job_watch("reduce")` in the `reduce` branch of `executor.__main__` |
| `InnerPool` child of an `inner_parallel` unit (spawn context, `InnerPool._executor`) | the worker's pool | `start_job_watch(f"{worker id}.pool{index}")` in the pool initializer `_inner_init`; `index` comes from a `multiprocessing.Value` counter the pool shares with its children (0, 1, …), so each child has its own role and dump time (§11.4) |
| the adapter's legacy-trainer subprocess (`pass_completion`, `xsuccess`) | `adapter._run_work` `subprocess.run` | started through a runner: `python -m scripts.hf_jobs.liveness adapter -m <target> <args…>` starts the watch, then `runpy.run_module(<target>, run_name="__main__", alter_sys=True)`; the target's code and arguments do not change |
| third-party pool children (`joblib`/`loky` under scikit-learn's `n_jobs`, ghost_gk permutation importance `train_ghost_gk.py:715-724`) | scikit-learn | NOT self-reporting; their parent's `[alive]` line reports them (`children=`, §11.3) |

Each covered process calls `start_job_watch` EXPLICITLY at its entry point or initializer. Inheriting the
environment variable alone starts nothing: a process without the call stays silent, so every row above has a
test that runs the real entry point with the variable set (§11.6).

### 11.2 The alive line

A covered process prints one `[alive]` start line at once (it states the UTC time its deadline dump fires,
§11.4), then starts one daemon thread. Every 300 s the thread prints ONE line to stderr with `flush=True`:

- the job label (`SILLY_KICKS_JOB_LABEL`), the role and the pid;
- the process uptime in seconds;
- the unit the process runs now (`<stage>/<key>`) and that unit's age in seconds, or `-` when it runs none
  (the stage main during dispatch; a pool child ALWAYS, because only `run_unit` sets the slot and it runs in
  the worker, so a pool child's line is its frame and uptime);
- the main thread's innermost PYTHON frame (`file:line function`, from `sys._current_frames()`). A C builtin
  has no frame: a process blocked in `time.sleep` or a FUSE `read` shows the Python line that called it;
- `children=`: §11.3.

A slow unit shows its frame moving between lines; a stall shows the same unit, a growing age and the same
frame. `run_unit` records the current unit (stage, key, start time) in a module-level slot in `_stages` and
clears it in a `finally`; the thread only reads it (the dependency stays `hf_jobs` → `_stages`).

**My call:** no separate per-unit start/end lines. The `[alive]` line names the unit and its age, and the
`.ledger.json` already records every outcome and wall time. A unit shorter than 300 s can pass between two
lines; the ledger covers it.

### 11.3 Child processes in the line

`children=` lists each DIRECT child of the process, read from `/proc` (Linux): `pid:state:cpu+<Δs>:<wchan>`.
`state` is the kernel state (`R` running, `S` sleeping, `D` uninterruptible, the FUSE-blocked case); `cpu+`
is the CPU seconds the child used since the previous line; `wchan` is `/proc/<pid>/wchan` (`-` when the kernel
hides it). Every process reports its own direct children, so the whole tree is covered one level at a time,
the third-party children included: a loky child that is computing shows `R` and growing CPU; a blocked one
shows `S`/`D`, zero CPU and its wait point. Without `/proc` (Windows, macOS) the field is `children=-`.

### 11.4 The deadline dump

`job_command` exports `SILLY_KICKS_JOB_DEADLINE_EPOCH=$(( $(date +%s) + timeout_s ))` as the first line after
`set -euo pipefail`, before the git and uv steps. Child processes inherit it (`deterministic_env` copies
`os.environ`). `start_job_watch` arms `faulthandler.dump_traceback_later(t, repeat=False)` with
`t = max(1, deadline − dump_margin_s(role) − now)`. One dump per covered process holds every thread's stack,
600 s or more before the kill.

The margin is different for each process of one job, so two dumps never share a second (Task 8 D: three
same-second dumps interleaved character by character; owner decision 2026-11-05). `main`, `fetch` and any
other role: 600 s. `reduce`: 601 s. `adapter`: 602 s. Worker `w<i>` owns a 10 s band and dumps at
`600 + 10 × (i + 1)` s; its pool child `w<i>.pool<j>` dumps at `600 + 10 × (i + 1) + 1 + j mod 9` s. With 8 workers of 8 pool children
each, the largest margin is 689 s. Only a worker with more than 9 pool children shares a
second between two of its children. A process that starts after its dump time gets `t = 1`: it dumps 1 s after
it starts, shows its start-up stack, and two such late starters (for example workers relaunched in the last
minutes, or a reduce started late) can share a second (impl review B, IMPL-02). `faulthandler` prints a fixed `Timeout (h:mm:ss)!` header; the process's `[alive]` start line
names that time, so a reader knows the dump is the deadline dump. A healthy job that runs past
`timeout_s − 600` also dumps, and nothing stops.

Why both mechanisms: the alive thread needs the GIL. A C extension that hangs while it holds the GIL stops the
alive lines; the `faulthandler` watchdog is a C thread that needs no GIL, so the dump still fires. A malformed
variable value turns both off with one `[alive]` warning line; it never fails the job.

### 11.5 Scope and safety

- Everything starts only when `SILLY_KICKS_JOB_DEADLINE_EPOCH` is set, so only in HF run jobs. Local runs,
  tests and the DGX controller print nothing new.
- The thread reads the slot, one frame and `/proc`, and prints one line. It touches no store, no result, no
  token and no generation, so no artifact or generation byte can change.
- Volume: one line per covered process every 5 minutes. With W workers and P pool children per worker, that
  is at most `1 + W + W·P` lines (an `inner_parallel` stage runs W = 1), plus 1 for the runner in an adapter job.
- The git and `uv sync` steps run before any Python process and print their own output; they are not covered.

### 11.6 Tests and proof

- One subprocess test per covered entry point (stage main, worker, reduce, fetch, pool initializer, adapter
  runner): the real entry runs with the variable set and prints its `[alive] … <role> … start` line at once.
  Each is seen to fail first.
- A subprocess test: a child stuck in a known Python function prints at least 2 `[alive]` lines (test-only
  short interval) whose frame names that function, then the deadline dump.
- A `run_unit` test: the slot names the unit during the work and is empty after it, for a success and a failure.
- `/proc` parsing tests on a fake `/proc` tree: a `comm` holding spaces and parentheses, the CPU delta, a hidden
  `wchan`, a child that exits between two lines, and no `/proc` at all.
- A `job_command` test: the export is the first line after `set -euo pipefail` and uses `spec.timeout_s`; the
  existing `bash -n` test covers its syntax.
- An off test: with the variable unset, no thread starts and no dump is armed. A malformed-value test: one
  warning line, no thread, no dump, no exception.
- §6 measurement D proves it on HF.
- A role test: `dump_margin_s` gives each role of a full job (main, reduce, adapter, 8 workers, 8 pool children
  each) a different second, and none less than 600 s; each pool child prints its own `w<i>.pool<j>` start line.

## 12. Change 5 — a lost rename on the HF bucket mount is repaired in the job (2c)

Owner decision 2026-11-05 (2c): fix it in this cycle. Owner decision 2026-11-05 (design): settle-and-repair
in the store.

### 12.1 Evidence

- Task 8 B: the mount lost 55 of ~11,900 result renames in the 12 extract jobs. 19 units lost the parquet, 19
  lost the `.sha256` (the 38 pending units) and 17 more lost `<key>.ledger.json`. Every lost final left its
  `.tmp-<name>.<uuid>.part`, which holds the full bytes. The losses came in bursts: consecutive units of one job.
  The base f03f4753 generation dirs also hold `.tmp` parts, so the loss is older than this cycle.
- Probe r1 (4 jobs, 12,000 files in a 1 s burst): 0 lost. A burst does not reproduce it.
- Probe r2 (12 jobs × 8 processes, paced unit triples for 300 s): 100 of ~37,180 lost (0.27%). `fsync` of the
  temp file before the rename lost as many (32 plain, 32 `fsync`, 36 `fsync` plus a directory `fsync`). The
  writer's own `stat` right after `os.replace` showed the final for all 100. The same container's listing 30 s
  after its last write showed all 100 missing.
- Probe r3 (same load, plain writes): 148 of 34,551 lost (0.43%). All 138 real losses (final missing, `.tmp`
  part present) were visible in the container's listing at +5 s and stayed so at +120 s. One `os.replace` from
  the `.tmp` part repaired every one; the remote listing then held every final and no `.tmp` part. Ten other
  finals were missing from one listing only, with no `.tmp` part, and came back by the next listing (a listing
  flicker, never a loss).

So an immediate check cannot see the loss, `fsync` does not stop it, and a check after a settle time can see
and repair it from the bytes the mount kept.

### 12.2 The contract

- `scripts/_write_confirm.py` keeps a per-process journal: final path → its temp path.
  `_driver._write_bytes_atomically` records every write there; a write to the same final replaces the entry.
  Every store rename in a run-job process goes through it: `LocalStore.write_bytes`, the `for_each` shard and
  sidecar, and `write_exclusion` (`<key>.excluded.json`; it had its own temp + `os.replace` before impl review B,
  HFURR-IMPL-01). `LocalStore.delete` and every `_driver` unlink of a written file forget the path.
- Not journaled, by design: the `.checksummed` generation marker and the `<key>.counters.json` sidecar are
  direct writes, not renames (the measured loss is a rename loss); the open-data events cache
  (`_sb_open_data`) is a cache, so a lost entry is downloaded again.
- `confirm_writes(*, settle_s, rounds=3)`: up to `rounds + 1` passes. Each pass sleeps `settle_s`, lists every
  journaled directory once and finds the journaled finals that are missing. None missing: the journal is cleared
  and the call returns the number of repairs. Otherwise, for each missing final whose temp name is listed, it
  runs `os.replace(temp, final)` again; a missing final without its temp is only re-checked (r3 flicker). After
  the last pass, a final that is still missing raises `LostWriteError`, which names every such path.
- `confirm_job_writes()` reads the settle time: 30 s in an HF run job (`SILLY_KICKS_JOB_DEADLINE_EPOCH` set;
  6× the observed ≤ 5 s), 0 s elsewhere, so local runs and tests do not wait. With an empty journal it returns
  at once.
- It runs once at the end of each process that writes to the store: `executor.worker_main` (after its units),
  `executor.reduce_main` (after the reduce), `_stages.main` (before it returns) and the adapter's legacy-trainer
  runner `liveness._run_module` (after the target exits 0; its `for_each` shards are on the store). It prints one line when it
  repairs: `[confirm] repaired <n> lost rename(s) in <dirs>`, and one `[confirm] LOST <path>` line per path
  before `LostWriteError` (a lost `ledger.json` alone leaves the unit done, so nothing is relaunched and only
  this line and the worker traceback show it; impl review B, IMPL-05).
- A worker whose confirm raises exits non-zero. The dead-worker branch of `LocalExecutor._dispatch` frees its units that
  are now pending and relaunches it, so the job redoes the unit itself (bounded by `max_relaunch`). A reduce whose
  confirm raises fails the stage, as any reduce error does.
- The repair moves the same bytes the unit wrote; no artifact byte changes. The cost is one settle wait per
  writing process at its end (the workers wait in parallel), plus one listing per journaled directory.

### 12.3 Tests

- A recorded write with nothing lost: confirm returns 0, sleeps `settle_s` once and clears the journal.
- A simulated loss (the final renamed back to its temp name): confirm repairs it, the final holds the written
  bytes and the temp is gone.
- A loss that does not repair (`os.replace` patched to do nothing): `LostWriteError` after `rounds + 1` passes,
  naming the path.
- A flicker (the final hidden from the first listing only, no temp): no error, no repair.
- A deleted file (`LocalStore.delete`) is forgotten: no error.
- A lost `<key>.excluded.json` rename (`write_exclusion`) is repaired.
- The adapter runner calls `confirm_job_writes` once, after the target wrote its output.
- The settle time: 30 s with the deadline variable set, 0 s without it.
- Each entry point (`worker_main`, `reduce_main`, `_stages.main`) calls `confirm_job_writes`, and a worker whose
  confirm raises exits non-zero.

## 13. Change 6 — the retrain launch 2 failures (2026-11-06)

Owner decisions 2026-11-06:
- Fix every fault below in this cycle, properly. Every fault without a log trail gets one.
- numba is a core dependency. The numpy kernels stay as parity references and run only when a
  `SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1` switch asks for them. A missing numba raises; it never falls back silently.
- A unit that hits the job timeout twice is STUCK. This replaces the "a unit timeout rides" rule (HFU-TIMEOUT-05).

### 13.1 Evidence (retrain launch 2 on `e9301e04`: 270 jobs, 258 COMPLETED, 12 ERROR; 9 of 21 instances done)

The session read all 270 job logs, the HF job records (`durations.running_secs`) and py-spy stacks of the hung
local processes.

| Fault | Jobs | Evidence |
|---|---|---|
| F1. numba is absent in training jobs | ghost_gk `importance` ×5 (`default`, `sweeper` ×2, `position_only`, `sweeper_position_only`) and ghost_outfield `default` `fit` ×1: `Job timeout` | ghost_outfield fit (`6aed7dce`): every `[alive]` line of the 2 h shows one core at 100 % in `_vectorized_leaf_values` (`silly_kicks/tracking/_ghost_gk.py:1504-1520`), the numpy body (observed). The 5 importance jobs: every w0 beat is at `joblib/parallel.py:1800 _retrieve` with 8 loky children at about 100 % CPU each (observed). Each child scores `predict_mean` (`train_ghost_gk.py:699-705`), which reaches the same numpy leaf walk (inferred from the code path; the children's frames are not in the log). `launch.py:204` installs `--extra train`. `[train]` holds no numba; `[test]` and `[calibration]` do, so CI never saw it. ADR-076 measured the numba leaf walk at 11.8×. The one importance job that finished (`full`) ran 5,242 s. |
| F2. 5 processes read one 388 MB FileSet through the mount | xshot_occurrence sc_extended `cv` ×5 (`default` ×3, `position_only` ×2): `Job timeout`. The third `position_only` try finished at 6,672 s. | Each of the 5 workers calls `materialize_fileset(ctx, "prep", "only", …)` (`train_xshot_occurrence.py:651`), which reads `only.files` (387.7 MB) through `LocalStore.read_bytes`. Every worker beat and dump is in `pathlib.read_bytes`, kernel state `D` (`folio_wait_bit_common`, `request_wait_answer`), CPU 0. The `study` stage reads the same file in ONE process and is past the read before its first beat (< 300 s). xcross sc_extended `cv` (155.8 MB × 5) took 124 s and 717 s (its two variants); public xshot (62 MB × 5) took 102 s. So one reader is fast, and five concurrent readers of one large file collapse. |
| F3. The driver hung for 2 h 10 min | the driver: last step line at 08:33 UTC | py-spy: 3 `--plan` subprocesses are blocked in `HfApi.list_bucket_tree` (`BucketStore.names`) or `get_bucket_paths_info` (`BucketStore.read_bytes`), in `_receive_response_headers`, each with one idle TLS connection to the HF CDN. The driver thread waits in `subprocess.run` (`campaign.read_plan`, no timeout). huggingface_hub 2.2.0 `default_client_factory` builds its client with `timeout=None`. Its `http_backoff` retries `httpx2.TimeoutException` (5 tries), so a read timeout turns the hang into a retry. |
| F4. A timed-out unit is relaunched without limit | the F2 unit: 3 tries of `default`, 2 h each | `_attribute` skips a RERUN (`driver.py:870`), so a timeout never spends coverage. |
| F5. No log trail | all of the above | The owner learned of the 12 failures from HF emails. The driver prints no line for a job that ends ERROR and no line for what it launched. A unit prints no start or end line, so a unit under 300 s leaves no trace and no unit has a measured duration. A `--plan` hang leaves no stack (py-spy was necessary). A large read prints no size or rate, so slow and hung look the same. No job states whether numba is active. |

Working as designed:
- The settle-and-repair (§12) repaired 85 lost renames in 30 jobs and reported 0 `LOST`.
- Every timed-out job printed its `[alive]` lines and its staggered deadline dumps (§11).
- No `Errno 5`.
- One `MountVolume.SetUp failed … DeadlineExceeded` (an HF mount flake) relaunched as designed.

The 12 instances not done:
- ghost_gk owner (5 variants) and ghost_outfield owner (2 variants): importance or fit onward.
- xshot_occurrence sc_extended (2 variants) and xcross_attempt sc_extended `default`: cv or assemble onward.
- win_probability public and match_outcome_dependence public: fit and reduce were next when the driver hung.

### 13.2 Change 6a — numba is a core dependency (F1)

- `pyproject.toml`: `numba>=0.59.0` moves into `[project] dependencies`. The `[numba]` extra stays as an empty
  list, so `pip install silly-kicks[numba]` still resolves (Hyrum). The duplicate numba lines in `[test]` and
  `[calibration]` go. `uv.lock` is regenerated. numba 0.65.1 has wheels for cp310–cp314 on Linux, macOS and
  Windows (verified in `uv.lock`).
- Inherited bounds (A-SPEC-27). Every numba release caps numpy and llvmlite: the locked 0.65.1 declares
  `numpy<2.5,>=1.22` and `llvmlite<0.48,>=0.47.0dev0`; numba 0.68.1 (with llvmlite 0.50) already installs with
  numpy 2.5.4 (the CI lint job log of run 42152733411). The floor stays `numba>=0.59.0`, so a consumer's resolver
  takes the newest numba its numpy allows. The consequence for consumers: a numpy upgrade waits until a numba
  release supports it. The lock stays at numba 0.65.1 (no change to the job environment in this cycle). The
  CHANGELOG and the `AGENTS.md` Dependencies line state this.
- One dispatch module, `silly_kicks/_numba_dispatch.py`. `use_numba(family) -> bool` returns False only when
  `SILLY_KICKS_<FAMILY>_FORCE_NUMPY == "1"`. Otherwise it imports numba (lazily, once) and returns True. An
  `ImportError` raises `NumbaUnavailableError` (a subclass of `ImportError`) that names the install fix. The
  error rises on the first accelerated call, not at `import silly_kicks`. `require_numba() -> None` does the same
  import for an explicit numba request and ignores the switches.
- Every dispatch site uses it (census 2026-11-06):

| Family | Site today | Behaviour today | Switch |
|---|---|---|---|
| ghost leaf walk and KDE | `_ghost_gk._use_ghost_numba` (also serves `_ghost_outfield.predict_mean`) | lazy import, silent fallback | `SILLY_KICKS_GHOST_FORCE_NUMPY` (exists) |
| DAS | `_das_engine._numba_available` (and its caller in `_das`) | silent fallback | `SILLY_KICKS_DAS_FORCE_NUMPY` (exists) |
| coordination | `coordination/_kernels/_numba.HAVE_NUMBA` | module-level try/except | `SILLY_KICKS_COORDINATION_FORCE_NUMPY` (exists) |
| pitch control | `_fernandez_bornn._HAS_NUMBA`, `_spearman._HAS_NUMBA`, `_numba_kernels` | module-level try/except | `SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY` (new) |
| ball carrier | `_ball_carrier._HAS_NUMBA`, `_ball_carrier_numba` | module-level try/except | `SILLY_KICKS_BALL_CARRIER_FORCE_NUMPY` (new) |
| xT-GK turnover scan | `xtgk/_turnover._NUMBA_TURNOVER` | module-level try/except | `SILLY_KICKS_XTGK_FORCE_NUMPY` (new) |

- Explicit engine arguments (HFURR-SPEC-28). Some public calls take an engine choice as well as the `auto` path:
  `compute_das` and `compute_das_paired` (`engine="auto"|"numpy"|"numba"`, `_das_engine.py:257, 280`) and
  `GhostGkModel.predict_density` (`kde_backend="auto"|"vectorized"|"scipy"|"cpu-numba"|"fft"|"fft-cic"`,
  `_ghost_gk.py:2074`). The rule:
  - `auto` asks `use_numba(<family>)`.
  - An explicit numpy-side choice (`engine="numpy"`; `kde_backend` `vectorized`, `scipy`, `fft`, `fft-cic`) runs as
    asked and never imports numba.
  - An explicit numba choice (`engine="numba"`, `kde_backend="cpu-numba"`) calls `require_numba()`: it runs the
    numba kernel, or raises `NumbaUnavailableError`. The explicit argument wins over the env switch, because it
    is the more specific request.
  - The silent `engine="numba"` fallback through `_das_numba.AVAILABLE` (`_das_engine.py:264-268`) goes: a
    `_das_numba` without numba raises `NumbaUnavailableError`.
- `scripts/validate_das_native_parity._numba_available` uses the DAS family. The numba on-disk cache stays
  opt-in (`tests/tracking/test_numba_cache_gating.py`). Before the change, `docs/PRIVATE_CONSUMERS.md` is checked
  for a pin on any renamed private name.
- Visible everywhere: each `[alive] … start` line carries `numba=<version>` (the runtime proof in the job).
  Preflight gets a sixth check, `check_numba`: it runs `uv export --frozen --no-hashes --extra train --group
  hfjobs` (the job's locked install set, `launch.py:204`), evaluates each line's environment marker for the job
  platform (Linux x86_64, CPython 3.12) and passes only when a `numba==` line applies there. It fails loud
  otherwise. A static check of the locked set is used because preflight runs on the Windows controller, where an
  `import numba` would prove the wrong wheel (plan ruling, accepted by A in plan r5).
- No artifact byte changes from the switch (bit-identical by ADR-076 and the existing parity tests).
- The wheel dependencies change, but the version stays 4.139.1 in this amend (owner decision 2026-11-06: no version bump in this amend; the bump comes
  with the release). ADR-076 is amended: option B
  ("make numba a hard dependency") is now chosen by owner decision, because the optional extra hid a 12×
  slowdown in production twice. ADR-008 D3 ("numba is not a runtime dependency") and ADR-013 (the precedent
  ADR-076 cited) get the same amendment (A-SPEC-30).

### 13.3 Change 6b — a FileSet is read once per job (F2)

- `materialize_fileset` keeps its signature and result. It materializes into a job-local cache directory keyed
  by the result's generation and key, under a `filelock.FileLock` (an OS lock that the OS releases when its
  holder dies). The first process reads from the store, writes the files to a temp directory, then renames it
  into place. Every other process waits on the lock and copies from the local cache, never from the mount.
  `filelock` becomes an explicit `[train]` dependency (3.29.0 is already in `uv.lock`).
- The bytes are verified once (the existing `read_result` sha256). The local copy holds the verified bytes, so a
  unit sees the same input as before. `LocalExecutor.run_stage` owns the cache: it creates the directory under
  the job's temp directory before it starts the workers, passes it to them in `SILLY_KICKS_FILESET_CACHE`, and
  removes it after the stage, when no worker is left (A-SPEC-29). Without that variable (a direct call outside
  the executor) `materialize_fileset` reads from the store as today.
- Callers (census 2026-11-06): `train_xshot_occurrence.py:651`, `train_xcross_attempt.py:813` and `:1084`,
  `train_ghost_gk.py:688`, `train_receiver_model.py:559`. Each keeps its call.
- Every store read of 8 MB or more prints one line: `[read] <rel> <MB> MB in <s> s (<MB/s> MB/s)`. A read that
  takes over 60 s also prints a progress line every 60 s.

### 13.4 Change 6c — every HF API call has a timeout; a plan read has a deadline (F3)

- A client-level timeout is not enough (A-SPEC-24, verified): `list_bucket_tree` calls `paginate`, which passes
  `timeout=None` EXPLICITLY to `http_backoff` (`huggingface_hub/utils/_pagination.py:27, 35, 44`), and `httpx2`
  turns an explicit `None` into `Timeout(None)` (`httpx2/_client.py:358-359`), which wins over the client default.
  Two of the three observed hangs sat in that call.
- So `scripts/hf_jobs/hub_client.py`: `install_timeouts()` calls `huggingface_hub.set_client_factory` with a
  factory that builds the library default client (`default_client_factory`) and appends one request event hook.
  The hook runs after the per-request timeout is set and before the send (`httpx2/_client.py:1040-1043`; the
  transport reads `request.extensions["timeout"]`, `httpcore2/_sync/http11.py:130-184`). It replaces each `None`
  value of that dict with `connect=10 s, read=60 s, write=60 s, pool=10 s` and keeps any finite value a caller set.
  So every Hub call, `paginate` included, is bounded. A stalled read then raises `httpx2.ReadTimeout`, which
  `http_backoff` retries (5 tries).
- `install_timeouts()` runs once in each entry point that talks to the Hub: `scripts/hf_jobs/__main__.main` and
  the `--plan` path of `_stages.main` when its store is `hf://`.
- `campaign.read_plan` passes `timeout=PLAN_TIMEOUT_S` (600 s) to `subprocess.run` and sets
  `SILLY_KICKS_PLAN_DEADLINE_S` for the child. The `--plan` path arms `faulthandler.dump_traceback_later` 30 s
  before that deadline. On `TimeoutExpired` the child is killed and the result is `--plan failed (timeout)`,
  with the dumped stack in its detail. `TimeoutExpired.stderr` is `bytes` on POSIX even with `text=True` and `str`
  on Windows, so the detail decodes either (UTF-8, `errors="replace"`) (HFURR-SPEC-32). The driver reports it and goes on. It is not a rate limit, so there is no
  back-off.

### 13.5 Change 6d — a unit that times out twice is STUCK (F4)

- A timeout charges only the units that STARTED in that job and did not finish (A-SPEC-26). A job holds up to
  `units_per_job` (20) units but runs 8 workers, so most held units of a timed-out job may never start.
- The started set comes from the job's own log: the `[unit] <stage>/<key> start` lines (§13.6) without a matching
  end line. `_attribute` fetches the log once for each new RERUN job and stores the started set in the same
  per-run_key cache as the manifest (no extra store write per unit). If the log fetch fails, that job is not
  cached and is tried again on the next step, so a unit is never charged on a guess.
- A unit charged by 2 timed-out jobs is not relaunched, and the step reports
  `STUCK <instance>: <stage>/<unit> timed out 2× (jobs <id>, <id>)`. The other units and instances go on. A
  held unit that never started is relaunched as today. The coverage rule for crashes and recorded failures does
  not change.
- A download instance runs unkeyed round-robin shards, and the executor runs a shard's whole slice. So a shard
  that holds a STUCK unit is held back and reported, and its other units wait for the resume after the fix
  (owner decision 2026-11-06, impl review B HFURR-IMPL-17; cycle1 has no download dataset).
- A resume seeds its baseline as today, so after a fix the owner resumes and the unit runs again.

### 13.6 Change 6e — the log trail (F5)

- Driver: one line for each job that newly ends ERROR or CANCELED:
  `[driver] job <id> <instance>/<stage> <shard> <reason> after <running_secs> s`. One line per launch:
  `[driver] launch <instance>/<stage> <shard> units=<n>`.
- Executor: per unit, `[unit] <stage>/<key> start` and `[unit] <stage>/<key> done|excluded|failed in <s> s`.
  Per job, one summary line at every exit of a run (0, 2 `REFUSED` and 3 `STAGE FAILED`; not `--plan`):
  `[job] <done>/<units> done, <failed> failed, exit <rc>, <s> s` (HFURR-PLAN-20). A kill by the HF timeout prints
  none; the `[unit]` lines and the deadline dump cover that case.
- The `[read]` lines (§13.3), the `--plan` dump (§13.4) and `numba=` on the start line (§13.2).

### 13.7 Tests (each seen to fail first)

- 6a:
  - `use_numba` with the switch set returns False and does not import numba.
  - With numba hidden (`sys.modules["numba"] = None`), `use_numba` raises `NumbaUnavailableError`.
  - A registry test: no `import numba` / `from numba` outside the kernel modules and the dispatch module, and
    every family of the census table routes through `use_numba`.
  - A `pyproject` test: numba is in `dependencies`, `[numba]` is empty, and numba is absent from `[test]` and
    `[calibration]`.
  - Explicit engines: `compute_das(..., engine="numba")` and `predict_density(..., kde_backend="cpu-numba")`
    with numba hidden raise `NumbaUnavailableError`; `engine="numpy"` and `kde_backend="vectorized"` with numba
    hidden still run.
  - Preflight `check_numba`: an export without numba fails; an export whose numba line carries a marker that
    does not hold on Linux x86_64 CPython 3.12 (for example `; sys_platform == 'win32'`) fails; a plain
    `numba==0.65.1` passes. The start line holds `numba=`.
- 6b:
  - 3 processes materialize one FileSet: the store is read once (a counting store), and every process gets
    identical bytes.
  - A holder killed mid-write leaves no partial cache, and the next process reads again.
  - A read of 8 MB or more prints its `[read]` line.
- 6c:
  - The installed factory's client has the 4 timeouts.
  - A local server that accepts and never answers: `BucketStore.names` (the `list_bucket_tree` / `paginate`
    path with its explicit `timeout=None`) and `BucketStore.read_bytes` (`get_bucket_paths_info`) each raise a
    timeout within the retries, not a hang. The test lowers the read timeout so it runs in seconds.
  - `read_plan` against a child that sleeps past the deadline returns a timeout failure whose detail holds a
    `File "` frame of the child.
- 6d: 2 RERUN jobs in whose logs one pending unit started give one STUCK report and no launch for it. 1 RERUN
  still relaunches. A unit done after 1 timeout is never reported. A unit held but never started in 2 RERUN jobs
  is relaunched and never reported. A RERUN job whose log fetch fails charges nothing and is fetched again on the
  next step. Timeouts of baseline jobs do not count.
- 6e: the driver prints one line per newly ERROR job (not again on the next step) and one per launch. A unit
  prints its start and end lines with a duration. The job summary line counts done and failed units and prints
  on exit 0, 2 and 3.

### 13.8 Definition of done

- Before the commit: CI green on the full suite (CI now installs numba through the core dependencies).
- After the commit (owner yes): a new launch of `cycle1` on the new commit. The amend changes the commit, and
  `run_key` holds `commit[:7]` (`campaign.py:120-123`), so all 21 instances run again, the 9 that finished in
  launch 2 included (A-SPEC-25). Only the `cross_commit_version` stages carry over (ghost_gk validate and
  extract, ghost_outfield extract, match_outcome and win_probability shards, receiver rows, xcross extract and
  probe, xshot extract). The owner gate states the spend: launch 2 used 60.3 compute-hours (about $1.81 at the
  `cpu-upgrade` rate of $0.0005 per minute that preflight reads), of which about 22 h were the 11 timeouts and
  2.7 h were the 9 finished instances. Report from the HF job list and the new log lines:
  - ERROR jobs by reason (target: 0 timeouts);
  - ghost importance and fit wall-clocks with numba;
  - the xshot sc_extended cv read time from its `[read]` lines (one read per job) and the job wall-clock;
  - the longest job against 7,200 s;
  - total wall-clock to 21/21;
  - `[confirm]` repairs and any `LOST`.

### 13.9 Docs that travel with the commit

- ADR-076 amendment (numba is core; the numpy switches are explicit), with the same amendment to ADR-008 D3 and
  ADR-013.
- ADR-128 amendment items: one read per FileSet per job; Hub timeouts and the plan deadline; the 2-timeout
  STUCK rule that replaces HFU-TIMEOUT-05; the log lines.
- `AGENTS.md` Dependencies line (numba moves from Optional to Runtime) and `docs/context/conventions-core.md`.
- `docs/howto/compute-pipelines.md` and `docs/howto/hf-jobs-training.md`: the new lines, the STUCK rule,
  `check_numba`.
- Every live "five checks" text becomes six: ADR-128 (`:122`, `:360`) and the `preflight.py` module docstring
  (HFURR-SPEC-31).
- `CHANGELOG.md`: the section states the dependency change and that the version stays 4.139.1 in this amend.
