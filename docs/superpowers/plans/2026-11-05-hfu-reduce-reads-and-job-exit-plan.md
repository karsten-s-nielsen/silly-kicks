# HFU reduce reads, job exit semantics, match-outcome pipeline — Implementation Plan

Status: APPROVED — A and B round 2 (`D:\Development\_reviews\2026-11-05-hfu-reduce-reads-plan-r2-{A,B}.md`).
Post-approval edits (CONSIDERs only): A-09 T5 note wording; B-09 measurement C prints CPU flags and drops the
"no platform excuse" claim; B-10 the fake corpus includes a non-two-team match.
**Change 4 delta APPROVED by the owner 2026-11-05 as fixed after A/B rounds 3 and 4 (answered below): spec Change 4 (§11).**
Changed: Goal, Architecture, Global Constraints (E12 line), Review Focus 6, Task 4A, Task 6 Steps 1-2, Task 8
Steps 1 and 5A-6. Every other line is approved and unchanged.
**Change 6 delta (spec §13, owner 2026-11-06): APPROVED — A plan r7, B plan r6 (`D:\Development\_reviews\2026-11-06-hfu-reduce-reads-plan-r{7-A,6-B}.md`); B's HFURR-PLAN-22 is A-PLAN-20, answered in Task 14.** — the section "Change 6 delta" (Tasks 10–17) at the end.
Round-3 answers: HFURR-PLAN-11 (one RED-first subprocess test per entry point, six of them) and A-PLAN-12 (an
autouse reset fixture; patched thread and dump in-process; an integer deadline, and `"1.5"` is a malformed case)
are in Task 4A. A-PLAN-10 / HFURR-PLAN-13 (the frame is the Python caller `_stall`, never `sleep`; `main` shows
`unit=-`), A-PLAN-11 / HFURR-PLAN-12 (the exact D script composition and the export byte check) and A-PLAN-13
(D ends `Job timeout`; the ADR line) are in Task 8 Step 5A, Step 1 and Task 6. Owner decision 2026-11-05: loky
children are reported by their parent (`children=`); a joblib hook comes only if a log shows a stall inside them.
Round-4 answers: HFURR-SPEC-23 / A-SPEC-22 (kill offset from the probe's 5 s `[probe] utc=` lines) and A-PLAN-14
(main's dump names `_dispatch`) are in Step 5A; HFURR-PLAN-14 (the runner imports the real module; one start
line asserted), HFURR-PLAN-15 (the two fixed RED reasons) and A-PLAN-15 (adapter line refs) are in Task 4A;
HFURR-SPEC-24 (ADR liveness wording) is in Task 6 Step 1.

Round 2 answered the round-1 A and B findings:
- **A-PLAN-01 / HFURR-PLAN-01 (traceback cap):** Task 4 caps the frames and the message SEPARATELY, so the frames
  survive a 1 MB message; the cap test asserts the raising function's name survives.
- **A-PLAN-02 (fit shape) + HFURR-PLAN-CONSIDER-08:** Task 5 follows `train_win_probability.py` exactly: extract
  reduces with `combine_frames` (sorted key order, through `reduce_stage` and INV-3), fit reads
  `ctx.upstream("extract")`. No new reduce site; no `done_results` use in fit.
- **A-PLAN-03 / HFURR-PLAN-02 (oracle):** the fit-parity oracle runs the NEW pure fit function over the legacy
  shards and compares `model.json` BYTES; it runs on HF `cpu-upgrade` (the legacy platform) in the patched
  f03f4753 uv env and records numpy/scipy versions. It moved from Task 7 to Task 8 (measurement C, owner-gated).
- **CONSIDERs:** the budget escape is dropped (A-07, B-03); T5 pins `--workers 1` and asserts the
  `require_reduced` text (B-04); compute-contract refs `:204-213` (A-04, B-05); T6 is a regression, expect PASS
  (A-05); `manifest.toml` check added (A-06, B-06); measurement B's write scope is guarded (A-08); a fallback
  reduce trigger before fit (B-07).

> **For the implementer:** execute inline with `superpowers:executing-plans`, one task at a time, in this
> session. Subagents, worktrees and per-task commits are banned for this owner. There is ONE commit for the whole
> cycle: an amend of `f03f4753` after the owner gate in Task 9.

**Goal:** Make a stage reduce read the store once, stop a recorded unit failure from failing its HF job, record
where a unit failed, run `match_outcome_dependence` as a sharded ADR-128 pipeline, and make every HF job show it
is alive every 5 minutes and leave its stacks before a timeout.

**Architecture:** `scripts/_stages.py` gains one listed-read primitive (`read_listed`), one done-results helper
(`done_results`) and one tolerance predicate (`exceeds_tolerance`). The executor drops its run-time copies of the
tolerance rule. `_driver.for_each` returns per-key tracebacks additively. The match-outcome trainer declares
`PIPELINE` in the `train_win_probability.py` shape. A new `scripts/hf_jobs/liveness.py` starts the `[alive]`
thread and arms the `faulthandler` deadline dump in every in-job Python process; `_stages.run_unit` exposes the
current unit through one read-only slot.

**Tech Stack:** Python 3.12, pandas, pytest, the ADR-128 stage engine (`scripts/_stages.py`,
`scripts/hf_jobs/executor.py`, `scripts/hf_jobs/driver.py`), HF Jobs (`cpu-upgrade`), the HF bucket mount.

**Spec:** `docs/superpowers/specs/2026-11-04-hfu-reduce-reads-and-job-exit-design.md` (APPROVED, A and B r4).
Read it first; this plan argues from it and cites its sections as "spec §N".

## Global Constraints

- Branch `feat/minimal-hf-driver`, PR #299. Base `f03f47536a671c20f066c7a8cf32b22836c9c7db`. Lands as ONE amend.
- No `silly_kicks/` change. Version stays `4.139.1` (`silly_kicks/_version.py`).
- Out of scope (owner): `units_per_job`, `max_shards`, `timeout_s`, `mount_cap`, store-format options B/C/D,
  `pass_completion`, `xsuccess`, unit compute speed, a FIX for E12 (Change 4 diagnoses it only).
- Lint pins (`.github/workflows/ci.yml:31`): `ruff==0.15.7`, `pyright==1.1.409`, `pandas-stubs==2.3.3.260113`.
- Lint at CI scope: `python -m ruff check silly_kicks/ tests/ scripts/`,
  `python -m ruff format --check silly_kicks/ tests/ scripts/`, bare `pyright`.
- Tests: `python -m pytest <paths> -m "not e2e" --tb=short`. Never co-run the `tests/scripts/hf_jobs` tree with
  pyright or the C4 render (memory `feedback_hf_jobs_suite_executor_flake_scope`). CI is the full-suite gate.
- The `error` string of every failure record keeps its exact current format (spec §4.3, Hyrum).
- Every HF launch, the amend, the force-push and the retrain each need their own explicit owner yes (spec §7).

## Review Focus

1. A generation dir holding crash leftovers (`.tmp-*.part` orphans, an orphan `_summary.json` without `_output`)
   must reduce normally. Test in Task 1: `test_reduce_ignores_crash_leftovers`.
2. A `.fileset.json` result whose member bytes are corrupt must be treated as not done (INV-3: delete, refuse),
   not decoded. Test in Task 1: `test_reduce_refuses_a_corrupt_fileset_member`.
3. `read_bytes` on a path that is a directory (`<key>.files/`) returns `None`, as `is_file()` made it today; a
   real I/O error still raises. Test in Task 1: `test_read_bytes_matches_is_file_semantics`.
4. A stage whose units are all excluded still reduces (empty done set, summary written). Test in Task 1:
   `test_reduce_with_no_done_units_writes_summary`.
5. A huge exception message or a deep stack must not bloat the failure record. Test in Task 4:
   `test_traceback_is_capped`.
6. A malformed `SILLY_KICKS_JOB_DEADLINE_EPOCH` (empty, non-integer) must never fail a job: liveness turns off
   with one warning line. Test in Task 4A: `test_malformed_deadline_turns_liveness_off`.

---

### Task 1: Listed reads and the single-pass reduce (spec §3)

**Files:**
- Modify: `scripts/_stages.py` (`LocalStore.read_bytes` `:255-257`, `LocalStore.names` `:264-266`,
  `_verified` `:455-465`, `read_result` `:468-480`, `unit_states` `:513-536`, `reduce_stage` `:755-793`)
- Create: `tests/scripts/test_stage_reads.py`
- Modify: `tests/scripts/test_stage_contract.py` (only if an existing assertion pins the old read order)

**Interfaces:**
- Produces: `read_listed(store: Store, rel_dir: str, key: str, names: Collection[str]) -> Any` — the verified
  result of `key` in `rel_dir`, or `MISSING`. Reads the payload once and `.sha256` once (plus each fileset member
  once), verifies in memory, decodes from the same bytes. Never lists, never deletes.
- Produces: `unit_states(ctx, st, keys, *, verify: bool, names: Collection[str] | None = None) -> dict[str, str]`
  — `names` given ⇒ no listing.
- Produces: `done_results(ctx: RunContext, stage: str) -> dict[str, Any]` — done units of `stage` in sorted key
  order, one listing, values via `read_listed` (may be `MISSING`, as today's `read_result`).
- Unchanged signature: `read_result(store, base) -> Any` becomes `read_listed(store, rel_dir, stem, store.names(rel_dir))`.

- [ ] **Step 1: Write the failing tests in `tests/scripts/test_stage_reads.py`**

  A `_CountingStore(LocalStore)` counts `names` and `read_bytes` calls. Build a one-stage toy generation of N done
  units with `tests/scripts/_stage_harness.run_in_process`. Tests:
  - `test_run_job_reduce_path_budget_is_constant(tmp_path, n)` parametrized `n in (8, 32)`: run
    `LocalExecutor.try_reduce` on the counting store with `LocalExecutor._subprocess` monkeypatched to
    `lambda self, ctx, args: reduce_stage(ctx, args[args.index("--stage") + 1])` (the single-process seam, spec
    §3.3). Assert `names_calls == 5` (`output_present`, `unit_states`, `deferred_memory_units` → `unit_states`
    `:594`, `reduce_stage`, `output_present`) and `read_bytes_calls == 2 * n + 2` (2 per unit + the final
    `output_present` reading `_output` and its `.sha256`) at both sizes, and the `_output` bytes equal those of a
    reference reduce through today's `read_result` loop. These constants are derived from the code; if the
    implementation gives others, stop and re-derive before changing the test.
  - `test_budget_guard_detects_the_rescan(tmp_path)`: monkeypatch `scripts._stages.read_listed` with a wrapper
    that ignores its `names` argument and calls `store.names(rel_dir)` itself (today's per-key rescan); run the
    same path and assert the budget check FAILS (`names_calls > 5` at `n=32`, growing with `n`).
  - `test_reduce_ignores_crash_leftovers`, `test_reduce_refuses_a_corrupt_fileset_member`,
    `test_read_bytes_matches_is_file_semantics`, `test_reduce_with_no_done_units_writes_summary` (Review Focus 1-4).
  - `test_reduce_verifies_before_writing_summary`: corrupt one payload; assert `StageIncompleteError`, the bad
    payload + sidecar deleted, and NO `_summary.json` / `_inputs.json` written.
  - `test_names_uses_scandir_and_skips_dirs`: a `<key>.files/` directory is not in `names()`.

- [ ] **Step 2: Run them; expect the budget, verify-order and leftovers tests to FAIL on today's code**

  Run: `python -m pytest tests/scripts/test_stage_reads.py -v --tb=short`

- [ ] **Step 3: Implement in `scripts/_stages.py`**
  - `LocalStore.names`: `os.scandir` + `DirEntry.is_file()`; missing dir → `set()`.
  - `LocalStore.read_bytes`: open directly; on `OSError` return `None` iff `not p.is_file()`, else re-raise.
  - `read_listed` as in Interfaces; `_verified` keeps its signature for `output_present` and the in-job resume
    check and shares the in-memory check with `read_listed` (no second read path).
  - `unit_states(..., names=None)`.
  - `reduce_stage`: list once → `unit_states(verify=False, names=…)` → pending/tolerance checks →
    `read_listed` for each done key in `reduce_order` → on any `MISSING`: delete payload + `.sha256` with the
    existing warning text, raise `StageIncompleteError` → only then write `_summary.json`, `_inputs.json`, `_output`.
  - `done_results` as in Interfaces.

- [ ] **Step 4: Run Task 1 tests plus the stage contract**

  Run: `python -m pytest tests/scripts/test_stage_reads.py tests/scripts/test_stage_contract.py tests/scripts/test_compute_contract.py -m "not e2e" --tb=short`
  Expected: all PASS.

### Task 2: Reduce sites use the listed read (spec §3.2)

**Files:**
- Modify: `scripts/build_territory_ranking_census.py:653-660` (`_done_results` → `done_results`)
- Modify: `scripts/train_ghost_gk.py:490-500` (`_cache_work`)
- Modify: `scripts/train_xcross_attempt.py:668-680` (probe loop)
- Test: `tests/scripts/test_stage_reads.py`

**Interfaces:** Consumes `done_results`, `read_listed` (Task 1).

- [ ] **Step 1: Census proof.** Run `rg -n "read_result\(|read_bytes\(|\.names\(" scripts/` and record, in the
  task notes, every hit that runs inside a loop over unit keys, with its disposition (spec §3.2 table). Expected
  set: `reduce_stage`, census `:660`, ghost_gk `:499`, xcross `:677`, `item_counters` `:814-831` (unchanged),
  `reduce_stage` excluded loop (unchanged). Any other hit: stop and raise it with the owner.
- [ ] **Step 2: Write `test_trainer_reduce_sites_list_once`**: for census `_done_results`, ghost_gk `_cache_work`
  and the xcross probe loop on a counting store with n units, assert one `names` call for the generation and
  2 reads per unit read. Expect FAIL.
- [ ] **Step 3: Implement.** Census and ghost_gk call `done_results(ctx, "extract")` (ghost_gk keeps its `len(f)`
  filter). xcross lists `ctx.store.names(gen)` once before its loop and calls `read_listed` per key (it keeps its
  lazy early stop at `PROBE_KEEP`).
- [ ] **Step 4: Run** `python -m pytest tests/scripts/test_stage_reads.py tests/scripts/test_build_territory_ranking_census.py tests/scripts/test_parity_win_probability.py -m "not e2e" --tb=short`
  plus the ghost_gk and xcross trainer test modules (`rg -l "train_ghost_gk|train_xcross_attempt" tests/scripts`).
  Expected: all PASS, outputs unchanged (the parity/W1N1 tests are the byte oracle).

### Task 3: `max_failures` is a reduce-time rule (spec §4.1, §4.2)

**Files:**
- Modify: `scripts/_stages.py` (add `exceeds_tolerance`; `reduce_stage` `:766` uses it)
- Modify: `scripts/hf_jobs/executor.py` (`_worker_should_stop` `:174-191`, `_FAILED_DIR` `:171` + its writes in
  `worker_main` `:578-579`, `_run` end check `:334-338`, `_check_failure_rule` `:341-352`, `try_reduce`
  `:243-258`, `require_reduced` `:260-264`)
- Modify: `scripts/hf_jobs/driver.py` (`_reduce_or_report` `:592` uses the predicate; docstring `:585-586`)
- Test: `tests/scripts/hf_jobs/test_executor.py`, `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:**
- Produces: `exceeds_tolerance(max_failures: int | None, n_failed: int) -> bool` in `scripts/_stages.py`
  (`max_failures is not None and n_failed > max_failures`). The only tolerance rule after this task; the driver
  imports it at module level (new edge `scripts/hf_jobs/driver.py` → `scripts._stages`; check no cycle with
  `python -c "import scripts.hf_jobs.driver"`).

- [ ] **Step 1: Write the tests (toy pipeline `tests.scripts._toy_pipeline`, stage `strict` = 6 units,
  `max_failures=0`; stage `sloppy` = tolerant).**
  - T1 — replace `test_zero_tolerance_stage_aborts_on_a_failure` with
    `test_a_recorded_failure_does_not_fail_a_run_job`: `_cli(store, "--stages", "strict", "--shard-keys", <file
    with all 6 keys>, "--workers", "2", "--fail-keys", "0002")`. Assert `returncode == 0`; the 5 other keys have a
    verified result; `0002.failed.json` exists; no `_output` in the strict generation.
  - T2 — keep `test_first_k_all_failed_aborts_early` unchanged (regression: exit 3, `"first 3"` in stderr).
  - T3 — `test_a_reduce_trigger_over_tolerance_exits_0`: after T1's run, `_cli(..., "--stages", "strict",
    "--shard-keys", <file "[]">, "--fail-keys", "0002")`. Assert `returncode == 0`, no `_output`, and stderr does
    not contain `reduce --stage` (no subprocess started).
  - T5 — `test_run_all_over_tolerance_fails_with_counts`: `--run-all --workers 1 --fail-keys 0002` (no `--out`).
    Assert `returncode == 3`; stderr contains `upstream stage 'strict' is incomplete: 0 pending, 1 failed
    (tolerance 0)`; the 5 other strict units have results. Today it fails on the message (today the run-time rule raises
    first, most likely `_check_failure_rule` with `strict: 1 failed unit(s) > tolerance 0 (e.g. 0002)`).
  - T4 — regression: `test_a_completed_reduce_trigger_that_left_the_stage_unreduced_is_bounded`
    (`test_hf_driver.py:730`) unchanged.
  - T6 — regression, expect PASS (the driver's charging rule does not change; this pins it for the new exit-0
    jobs): `test_a_completed_run_job_charges_its_recorded_failure` in `test_hf_driver.py`, modelled on
    `test_a_strict_abort_charges_only_the_recorded_failure` (`:879`): job `stage="COMPLETED"`, manifest
    `["a", "b"]`, `a` recorded failed. Assert `_cover(...) == {(inst.id, "fit", "a"): 1, (inst.id, "fit", "b"): 1}`
    and that `decide` lists `a` as a launch candidate while coverage `< attempts`.
- [ ] **Step 2: Run** `python -m pytest tests/scripts/hf_jobs/test_executor.py tests/scripts/hf_jobs/test_hf_driver.py -m "not e2e" --tb=short -k "recorded_failure or reduce_trigger or run_all_over or first_k or completed_run_job"`.
  Expected: T1, T3, T5 FAIL; T2, T4, T6 PASS.
- [ ] **Step 3: Implement** per the spec §4.1 table:
  - `_worker_should_stop`: first-k branch only; delete `_FAILED_DIR`, its marker writes and its docstring line.
  - `_check_failure_rule`: first-k branch only (terminate + raise, unchanged text).
  - `_run`: delete the end-of-run tolerance raise.
  - `try_reduce`: order pending → deferred → `exceeds_tolerance(st.max_failures, n_failed)` → return `False`
    before `_subprocess`.
  - `require_reduced`: message `f"{ctx.pipeline.name}: upstream stage {name!r} is incomplete: {n_pending}
    pending, {n_failed} failed (tolerance {st.max_failures})"`.
  - `reduce_stage` and driver `:592` call `exceeds_tolerance`; driver docstring `:585-586` says "every terminal
    trigger" (it counts COMPLETED too, `:832-834`).
- [ ] **Step 4: Find every other test that pins the old exit.** `rg -n "returncode == 3|tolerance" tests/scripts`;
  update only tests whose premise is the removed run-time rule, and list each in the task notes.
- [ ] **Step 5: Run** `python -m pytest tests/scripts/hf_jobs/test_executor.py tests/scripts/hf_jobs/test_hf_driver.py tests/scripts/test_stage_contract.py -m "not e2e" --tb=short`.
  Expected: all PASS.

### Task 4: Failure locus for both unit kinds (spec §4.3)

**Files:**
- Modify: `scripts/_driver.py` (`CorpusPassResult` `:663-680`; the `except Exception` at `:810-811`)
- Modify: `scripts/_stages.py` (`_run_item` `:666-692`, `run_unit` `:695-730`)
- Test: `tests/scripts/test_stage_reads.py` (or the closest existing `run_unit` test module), `tests/scripts/test_driver.py`

**Interfaces:**
- Produces: `CorpusPassResult.tracebacks: dict[str, str]` (default empty; additive; `failures` unchanged).
- Produces: `format_failure_traceback(exc: BaseException) -> str` in `scripts/_driver.py`. Two parts, each capped
  on its own so a long message can never push the frames out:
  - frames: `"".join(traceback.format_tb(exc.__traceback__, limit=-20))` (the 20 frames nearest the raise), keeping
    its LAST 6,000 characters;
  - message: `f"{type(exc).__name__}: {exc}"`, keeping its FIRST 1,000 characters (the full message is already in
    `error`).
  Result = frames + message; at most 7,000 characters plus the joining newline.
- Changes: `_run_item(...) -> tuple[str, str | None, str | None]` (outcome, error, traceback).
- Record shape: `.failed.json` = `{"error": <unchanged>, "attempt": n, "traceback": <str>}`; the memory-cap record
  (`_stages.py:584`) is unchanged.

- [ ] **Step 1: Write the tests.** `test_plain_stage_failure_records_traceback` (a `Stage` work raising
  `OSError(5, "Input/output error")`; record has `error == "OSError: [Errno 5] Input/output error"` and a
  `traceback` naming the raising function); `test_item_stage_failure_records_traceback` (same via an
  `ItemStage` load); `test_for_each_failures_strings_unchanged` (legacy `failures[k]` equals today's string);
  `test_traceback_is_capped` (Review Focus 5): (a) a work function `_raise_big` raising `OSError` with a 1 MB
  message → `len(tb) <= 7001`, `"_raise_big"` in `tb` and `'File "'` in `tb`; (b) a 200-deep recursion ending in
  `_raise_deep` → `len(tb) <= 7001` and `"_raise_deep"` in `tb`.
- [ ] **Step 2: Run them; expect FAIL** (`python -m pytest <those tests> -v`).
- [ ] **Step 3: Implement** per Interfaces. The traceback never enters a unit result, token or output.
- [ ] **Step 4: Run** the Task 4 tests plus `tests/scripts/test_driver*.py tests/scripts/test_corpus_driver_resilience.py -m "not e2e"`. Expected: PASS.

### Task 4A: Job liveness — `[alive]` lines, child report and the deadline dump (spec §11)

**Files:**
- Create: `scripts/hf_jobs/liveness.py`
- Modify: `scripts/_stages.py` (`run_unit` `:695-745`: set and clear the slot; `main` `:1019`: first statement)
- Modify: `scripts/hf_jobs/executor.py` (`_inner_init` `:75-79` and `InnerPool.__init__`/`_executor` `:91-104`:
  a `role` argument passed to the initializer; `worker_main` `:547`: the pool gets `role=f"{a.worker_id}.pool"`;
  the `__main__` block `:604-608`: start the watch with the role before dispatching the command)
- Modify: `scripts/hf_jobs/adapter.py` (`_run_work` `:45`: the argv at `:59` goes through the runner; the run at `:62` is unchanged)
- Modify: `scripts/hf_jobs/datasets.py` (`main` `:363`: first statement, role `fetch`)
- Modify: `scripts/hf_jobs/launch.py` (`job_command` `:188-200`: the export line)
- Create: `tests/scripts/hf_jobs/test_job_liveness.py`
- Modify: `tests/scripts/hf_jobs/test_launch_typed.py` (new test next to `:134`)

**Interfaces:**
- Produces in `scripts/_stages.py`: `current_unit() -> tuple[str, str, float] | None` = (stage, key,
  `time.monotonic()` at start). `run_unit` sets the slot before its `try` and clears it in a `finally` that wraps
  the whole body (success, recorded failure, `ItemExcluded`, and a raised exception alike).
- Produces in `scripts/hf_jobs/liveness.py`:
  - `DEADLINE_ENV: Final = "SILLY_KICKS_JOB_DEADLINE_EPOCH"`, `ALIVE_INTERVAL_S: Final = 300`,
    `DUMP_MARGIN_S: Final = 600`.
  - `parse_proc_stat(text: str) -> tuple[int, str, int]` — (ppid, state, utime + stime ticks) from one
    `/proc/<pid>/stat` line; the `comm` field is cut at the LAST `)` (it may hold spaces and parentheses).
  - `direct_children(pid: int, *, proc: pathlib.Path = pathlib.Path("/proc")) -> dict[int, tuple[str, int, str]]`
    — {child pid: (state, cpu ticks, wchan or `-`)} by scanning `proc/[0-9]*/stat`; a pid that vanishes during
    the scan is skipped; `proc` missing → `{}`.
  - `children_field(now: dict, before: dict, clk_tck: int) -> str` — `"pid:state:cpu+<Δs>:wchan"` joined by `,`
    in pid order; `Δs` = (ticks − previous ticks, or 0 for a new pid) / `clk_tck`, one decimal; `-` when empty.
  - `alive_line(role: str, *, started: float, now: float, children: str) -> str` — `"[alive] {label} {role}
    pid={pid} up={s}s unit={stage}/{key} age={s}s at={file}:{line} {func} children={children}"`;
    `unit=- age=-` with no current unit; `label` = `SILLY_KICKS_JOB_LABEL` or `-`; the frame = the main
    thread's innermost Python frame from `sys._current_frames()[threading.main_thread().ident]`.
  - `start_job_watch(role: str, *, env: Mapping[str, str] = os.environ, interval_s: float = ALIVE_INTERVAL_S) -> bool`
    — returns False and does nothing when `DEADLINE_ENV` is unset; on a value that is not an integer prints one
    `[alive] {role}: liveness off: bad {DEADLINE_ENV}={value!r}` line and returns False; otherwise prints the
    start line `"[alive] {label} {role} pid={pid} start; deadline dump at {UTC ISO time}"`, calls
    `faulthandler.dump_traceback_later(max(1, deadline - DUMP_MARGIN_S - time.time()), repeat=False)`, starts ONE
    daemon thread (`name="sk-alive"`) that prints `alive_line` every `interval_s` (`Event.wait`), and returns
    True. It never raises. A second call in the same process is a no-op returning False (a module flag).
  - `__main__`: `python -m scripts.hf_jobs.liveness <role> -m <module> [args…]` → `from scripts.hf_jobs.liveness
    import start_job_watch` (the IMPORTED module, so a later import of it shares the one-start flag), `start_job_watch(role)`, then
    `sys.argv = [<module>, *args]` and `runpy.run_module(<module>, run_name="__main__", alter_sys=True)`; its
    exit status is the target's (`SystemExit` propagates).
- Roles: `_stages.main` → `"main"`; executor `worker` → the `--worker-id` value; executor `reduce` → `"reduce"`;
  `datasets.main` → `"fetch"`; `_inner_init` → `f"{worker id}.pool"` (`InnerPool(..., role="pool")` default);
  the adapter runner → `"adapter"`. Every print: `file=sys.stderr, flush=True`.
- `adapter._run_work` argv: `[sys.executable, "-m", "scripts.hf_jobs.liveness", "adapter", "-m", target, *args,
  "--out", …]`.
- `job_command`: the line after `"set -euo pipefail"` is
  `f"export {DEADLINE_ENV}=$(( $(date +%s) + {int(spec.timeout_s)} ))"` (import the constant; no second literal).

- [ ] **Step 1: Write the tests** in `tests/scripts/hf_jobs/test_job_liveness.py`. An autouse fixture resets the
  module flag and cancels any armed dump (`faulthandler.cancel_dump_traceback_later`) after each test; in-process
  tests patch `threading.Thread.start` and `faulthandler.dump_traceback_later` so no real thread or dump leaks.
  Child processes get `env = os.environ | {DEADLINE_ENV: str(int(time.time()) + DUMP_MARGIN_S + 3),
  "SILLY_KICKS_JOB_LABEL": "t"}`, `cwd=REPO_ROOT`, `capture_output=True`, `timeout=60`.
  - `test_alive_lines_name_the_stuck_function`: a child `python -c` calls `start_job_watch("main",
    interval_s=0.5)`, then `def _stall(): time.sleep(6)`. Assert: the start line; at least 2 `[alive] t main`
    lines with `at=` ending in ` _stall`; the dump (`"Timeout ("` and `_stall` in stderr); return code 0.
  - Entry-point wiring, one test each, each run against today's code first (RED): `test_stage_main_starts_the_watch`
    (`python -m tests.scripts._toy_pipeline --plan --store <tmp>` → `[alive] t main ` … ` start`);
    `test_worker_starts_the_watch` and `test_reduce_starts_the_watch` (`python -m scripts.hf_jobs.executor
    worker|reduce` with a missing `--ctx` file → the start line is printed BEFORE the argument error; role `w9`
    / `reduce`); `test_fetch_starts_the_watch` (`python -m scripts.hf_jobs.datasets fetch` with no dataset name →
    the start line with role `fetch` before the usage error); `test_pool_children_start_the_watch` (an
    `InnerPool(2, role="w0.pool")` in a child process maps a module-level function → 2 start lines with role
    `w0.pool`); `test_adapter_runner_starts_the_watch_and_runs_the_target` (`python -m scripts.hf_jobs.liveness
    adapter -m tests.scripts._toy_legacy_trainer <its minimal args>` → the start line with role `adapter`, and
    the target's exit code and output are those of a direct `python -m` run, and exactly ONE start line).
  - `test_run_unit_exposes_the_current_unit`: a toy `Stage` whose work returns `current_unit()[:2]`; the stored
    result is `(stage, key)`, and `current_unit() is None` after `run_unit`; the same for a work that raises
    (recorded failure).
  - `/proc` tests on a fake tree under `tmp_path`: `test_parse_proc_stat_handles_odd_comm` (`comm` =
    `"(a) b) (c"`), `test_direct_children_skips_vanished_and_others` (two children, one unrelated pid, and one
    pid directory without `stat`, as a process that exits mid-scan leaves), `test_children_field_reports_cpu_delta`
    (Δ = 150 ticks at 100 Hz → `cpu+1.5`; a new pid → `cpu+0.0`; a hidden wchan `0` → `-`), and
    `test_no_proc_gives_dash` (`proc` missing → `children=-`).
  - `test_watch_is_off_without_the_deadline`: `start_job_watch("main", env={})` returns False; nothing armed,
    no thread, nothing printed (`capsys`).
  - `test_malformed_deadline_turns_liveness_off` (Review Focus 6): `"x"`, `""` and `"1.5"` → False, exactly one
    stderr line containing `liveness off`, no dump armed, no thread.
  - `test_dump_delay_is_the_deadline_minus_the_margin`: deadline = `int(time.time()) + 1000` → the patched
    `dump_traceback_later` gets `timeout` in `[398, 401]`, `repeat=False`; a past deadline → `timeout == 1`.
  - In `test_launch_typed.py`: `test_job_command_exports_the_deadline_first` — `job_command(_spec())[2].splitlines()[1]`
    equals `f"export SILLY_KICKS_JOB_DEADLINE_EPOCH=$(( $(date +%s) + {_spec().timeout_s} ))"`.
- [ ] **Step 2: Run them; expect FAIL** (`python -m pytest tests/scripts/hf_jobs/test_job_liveness.py tests/scripts/hf_jobs/test_launch_typed.py -v`).
  Every entry-point test must fail on the missing start line, except two whose RED reason is fixed in advance:
  the adapter-runner test fails with `No module named scripts.hf_jobs.liveness` (non-zero exit, no start line),
  and the pool test with `TypeError: … unexpected keyword argument 'role'`.
- [ ] **Step 3: Implement** per Interfaces. The watch reads only the slot, one frame and `/proc`; it touches no
  store, result, token or generation (spec §11.5).
- [ ] **Step 4: Run** the Step 2 command plus `tests/scripts/hf_jobs/test_executor.py tests/scripts/hf_jobs/test_adapter.py tests/scripts/test_stage_contract.py tests/scripts/test_stage_reads.py -m "not e2e" --tb=short`
  (the `hf_jobs` tree on its own, not beside pyright). Expected: PASS.

### Task 5: `match_outcome_dependence` as an ADR-128 pipeline (spec §5)

**Files:**
- Modify: `scripts/train_match_outcome_dependence.py` (replace the argparse `main` `:110-171`; keep the kernels)
- Modify: `scripts/hf_jobs/registry.py:129-140`, `scripts/hf_jobs/adapter.py:41-42,60-61`
- Modify: `tests/scripts/test_compute_contract.py:82-87` and `:204-213`, `TODO.md:57`
- Check: `docs/artifacts/manifest.toml:309-316` (`id = "match_outcome"`, `job = "match_outcome_dependence"`,
  destination `wheel:silly_kicks/match_outcome/weights`) still resolves (spec §5.2)
- Modify: `tests/scripts/test_match_outcome_train.py:87-111`
- Create: `tests/scripts/test_parity_match_outcome.py`
- Check: `tests/scripts/hf_jobs/test_adapter.py`, `test_registry.py`, `test_campaign_cycle1.py`, `test_adopt.py`,
  `test_headtohead.py` (`rg -n "match_outcome|_SHARD_ROOT_TARGETS" tests/scripts`)

**Interfaces:**
- Consumes: `combine_frames` (`scripts/_stages.py:649`); `list_open_data_refs`, `load_open_data_match`, `assert_statsbomb_open_data_mode`,
  `all_open_competitions`, `OpenDataRef` (`scripts/_sb_open_data.py`); `resolve_cache_dir` (`scripts/_driver_sources.py`).
- Produces: `PIPELINE = Pipeline(name="match_outcome_dependence", module="scripts.train_match_outcome_dependence", …, output="fit")`.
  Params: `dataset` (`str`, `"sb_open"`), `competitions_json` (`str | None`: a PATH to `[[competition_id,
  season_id], …]`, the legacy flag; its CONTENT enters the token through the refs stage `declare`, as census
  `_corpus_declare` does; `None` ⇒ `all_open_competitions()`), `max_matches` (`int | None`), `cache_dir` (`str | None`).
- Stages: `refs` (one unit `"only"`; `{"competitions": [...], "refs": [asdict(r) …]}`; `uses_dataset=True`,
  `dataset_files=lambda ctx: ()`); `extract` (`ItemStage`, key `ref.key`, load = assert mode +
  `load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))`, work =
  `extract_match_slice(lm.actions, "xg")`, `needs=("refs",)`,
  `reduce=_extract_reduce` (`combine_frames(list(results.values()), ids=())` — results arrive in sorted key order,
  the legacy sorted shard glob; no part → an empty frame with the `extract_match_slice` columns), `max_failures=None`,
  `cross_commit_version="match-outcome-shard-1"`, `uses_dataset=True`); `fit` (one unit; `needs=("refs",
  "extract")`; work = `fit_artifact(ctx.upstream("extract"), ctx.prov)`).
- Produces: `fit_artifact(pooled: pd.DataFrame, prov: Mapping[str, Any]) -> FileSet` — pure:
  `matches_from_shards([pooled])` → `fit_rho` → `FileSet({"model.json": …, "SHA256SUMS": …})`. The Task 8
  oracle calls exactly this function.
- Artifact bytes: `model.json` = `json.dumps({"rho", "training_commit", "run_tree_dirty", "corpus":
  "statsbomb-open", "n_matches"}, indent=2)` in that key order, no trailing newline (legacy `:157-164`);
  `SHA256SUMS` = `f"{sha}  model.json\n"`.

- [ ] **Step 1: Write the tests.**
  - `tests/scripts/test_parity_match_outcome.py::test_w1n1_equals_w2n2` — copy the shape of
    `tests/scripts/test_parity_win_probability.py` (`@pytest.mark.slow`, `importorskip("statsbombpy")`,
    `build_open_data` + `serve_open_data`), `MODULE = "scripts.train_match_outcome_dependence"`,
    `ARGS = ["--competitions-json", <tmp file "[[43, 106]]">]`. Assert equal outputs and artifacts,
    `set(artifact) == {"model.json", "SHA256SUMS"}`, and `completion_out_of_order` on the extract generation.
  - Rewrite `test_train_main_writes_loadable_artifact` to run `PIPELINE` via
    `tests/scripts/_stage_harness.run_in_process(PIPELINE, ["--run-all", "--allow-dirty", …])` on the fake
    corpus (patch `list_open_data_refs` / `load_open_data_match`), then `DependenceModel.load(out)`.
  - `test_the_compliant_set_is_the_cycle1_migration`: add `"train_match_outcome_dependence"`.
  - `test_fit_artifact_matches_the_legacy_main`: on the 8-game fake corpus (one game made non-two-team, so its
    shard is the empty-frame concat edge), the legacy computation
    (`fit_rho(matches_from_shards(per-match shards))`, today's `main` body) and
    `fit_artifact(_extract_reduce(<per-match shards keyed by match key>, None), prov)` give byte-identical
    `model.json`.
- [ ] **Step 2: Run them; expect FAIL.**
- [ ] **Step 3: Implement** per Interfaces. `if __name__ == "__main__": raise SystemExit(main(PIPELINE))`
  (`from scripts._stages import main`). Update the module docstring usage line. Registry: remove `adapter=True`,
  add `pipeline="scripts.train_match_outcome_dependence"`, keep `VariantBuild("default", (), "match_outcome/weights", "")`.
  Adapter: delete `_SHARD_ROOT_TARGETS` and its branch. Contract allowlist: remove the trainer. `TODO.md:57`:
  mark `match_outcome_dependence` done, keep `pass_completion` / `xsuccess`.
- [ ] **Step 4: Run** `python -m pytest tests/scripts/test_match_outcome_train.py tests/scripts/test_parity_match_outcome.py tests/scripts/test_compute_contract.py tests/scripts/test_match_outcome_calibration.py tests/scripts/hf_jobs/test_adapter.py tests/scripts/hf_jobs/test_registry.py tests/scripts/hf_jobs/test_campaign_cycle1.py tests/scripts/hf_jobs/test_adopt.py tests/scripts/hf_jobs/test_headtohead.py --tb=short`
  (include `slow`). Expected: all PASS. Then confirm the `manifest.toml` entry above still names this job, tier and
  destination (no edit expected).

### Task 6: Docs (spec §9)

**Files:** `docs/superpowers/adrs/ADR-128-compute-contract-hf-campaigns.md` (new section after `:412-437`, before
`## Related`), `docs/howto/compute-pipelines.md`, `CHANGELOG.md` (the `[Unreleased] — minimal HF Jobs driver …`
section at `:8`).

- [ ] **Step 1: ADR-128 amendment (2026-11-05).** Context: the corrected diagnosis (E1-E12; the 2026-11-04 sizing
  hypothesis was wrong). Decision: the four changes; `max_failures` is reduce-time only; `abort_first_k` stays
  loud; a refused trigger exits 0 and is bounded by the trigger count; every in-job Python process prints
  `[alive]` every 300 s and arms a `faulthandler` dump at deadline − 600 s (spec §11). Scope: scripts-only, folded into the amend,
  4.139.1. Reviews: spec A/B r4 APPROVED and the Change 4 delta rounds; plan and impl rounds as they land.
  Liveness wording: every COVERED in-job Python process prints `[alive]` and arms the dump; third-party pool
  children (loky) are reported by their parent's `children=` field, by owner decision 2026-11-05.
- [ ] **Step 2: `compute-pipelines.md`:** a reduce site lists once and reads each file once (`read_listed`,
  `done_results`); a run job exits 0 on recorded unit failures; failure records carry `traceback`; how to read a
  job log: the `[alive]` fields, a stall (same unit, growing age, same `at=`) versus slow progress, and the
  `Timeout (h:mm:ss)!` deadline dump.
- [ ] **Step 3: CHANGELOG:** extend the existing `[Unreleased]` section (Changed / Fixed bullets; no version bump).
- [ ] **Step 4:** grep the docs for the removed adapter path of match-outcome and the old exit rule
  (`rg -n "one-unit adapter|_SHARD_ROOT_TARGETS|tolerance.*abort" docs/`), and fix live (non-historical) text only.

### Task 7: Local gates

- [ ] **Step 1: Lint at CI scope** with the pinned versions (Global Constraints). Expected: exit 0 for all three.
- [ ] **Step 2: Targeted tests** for every module touched in Tasks 1-5 (run the `tests/scripts/hf_jobs` subset on
  its own, not beside pyright). Show the real exit codes.

### Task 8: Measurement campaign on HF (spec §6 "before the commit") — OWNER GATE FIRST

- [ ] **Step 1: Owner gate.** Show the owner: job count (A: 4; B: 1 refs + 12 extract + 1 fit = 14, plus at most
  1 fallback trigger; C: 1; D: 1 job, `timeout=1500` s, about 25 min, ends `Job timeout` = one ERROR and one
  email by design), flavor
  `cpu-upgrade`, store paths, and the diff sha256. Wait for an explicit yes. No launch before it.
- [ ] **Step 2: Patch-on-base mechanics** (probe scripts in the scratchpad; nothing enters the repo). Upload the
  working-tree diff (`git diff f03f4753` incl. new files) and the probe script to
  `silly-kicks/hf-refit-scratch/_canary/<run_id>/`. Each job: clone + `git checkout f03f4753` as in `_job_spec`,
  verify the diff sha256, `git apply`, `uv sync` as in `_job_spec`, then run. Labels: `canary=<run_id>`,
  `patch_sha256=<sha>`. Set `SILLY_KICKS_OPEN_DATA_REVISION` from `scripts/hf_jobs/datasets.lock.json` `sb_open`
  as `launch.py:152` does.
- [ ] **Step 3: Measurement A.** 4 jobs launched together; bucket mounted `read_only=True`. Each lists
  `cycle1-f03f475-win_probability-public/win_probability/extract/3c47d807efdf22a7/` once and runs `read_listed`
  over every done key; prints JSON `{listing_s, names_calls, reads, read_s, n_units}` to the job log.
- [ ] **Step 4: Measurement C — fit parity (spec §5.3 part 1), run first.** One job, bucket mounted
  `read_only=True`, in the patched f03f4753 uv env (the legacy fit's platform: HF `cpu-upgrade`, x86). It reads every
  `*.parquet` under `cycle1-f03f475-match_outcome_dependence-public/adapter/run/7186d1745ce95ed1/target_out/shards/aa3a3b14c7ff26e9/`
  as `{stem: pd.read_parquet(p) for p in sorted(glob("*.parquet"))}` (the legacy glob; NOT INV-3: 3,961 parquet,
  3,956 sha256), calls the NEW `fit_artifact(_extract_reduce(<that dict>, None), {"commit": <legacy
  training_commit>, "dirty": <legacy run_tree_dirty>})`, and compares its `model.json` BYTES with
  `…/7186d1745ce95ed1/_output.files/model.json`. It prints `numpy`/`scipy` versions and the `lscpu` flags line.
  Pass = bytes identical (`rho` 0.02161420866397881, `n_matches` 3961). On any difference: stop and report the
  `rho` delta with the versions and CPU flags; the owner decides (spec §5.3: a difference is never accepted
  silently).
- [ ] **Step 5: Measurement B.** Store `/scratch/_canary/<run_id>/mo` (rw mount). `--stages refs`; then 12 jobs
  `--stages extract --shard i/12 --workers 8`; if the extract stage is not reduced when the last of them ends,
  one empty `--shard-keys` trigger; then `--stages fit`; all `--allow-dirty`, no `--competitions-json`.
  Write scope: every job's only `--store` is `/scratch/_canary/<run_id>/mo`; the probe lists the bucket (path +
  size + last-modified) outside `_canary/<run_id>/` before and after B and asserts no change.
  Pass = refs key set == legacy shard key set (the Step 4 stems) AND `model.json` bytes == legacy except
  `training_commit`/`run_tree_dirty`; record every job's wall-clock.
- [ ] **Step 5A: Measurement D — liveness (spec §6 D, §11.6).** One job, `timeout=1500`, `--workers 2`, store
  `/scratch/_canary/<run_id>/live`. The probe module `_probe_live.py` (scratchpad, uploaded beside the diff)
  declares one `Stage` `stall` with units `a` and `b` whose work calls a module-level `_stall()` that loops
  forever: `print(f"[probe] utc={int(time.time())}", flush=True)`, then `time.sleep(5)`; the module ends `raise SystemExit(main(PIPELINE))` (so the real `_stages.main` and executor
  entry points run). The job script is composed as follows, and the launcher checks it before the launch:
  1. take `job_command(spec)[2]` for the spec (`timeout_s=1500`) unchanged up to and including the
     `test "$(git rev-parse HEAD)" = …` line;
  2. insert the Step 2 lines: verify the diff sha256, `git apply` it, copy `_probe_live.py` to `/work/`;
  3. keep the `pip install` and `uv sync` lines unchanged;
  4. replace the last (run) line with `.venv/bin/python -m _probe_live --store /scratch/_canary/<run_id>/live
     --data-root <as _run_argv> --stages stall --shard 0/1 --workers 2 --inner-workers 1 --allow-dirty`.
  Byte check: line 2 of the composed script equals `job_command(spec)[2].splitlines()[1]` (the real export
  line). Pass = the log shows a start line from `main`, `w0` and `w1`; `[alive]` lines from each every 300 s
  (± one poll); each worker line names its unit, a growing `age=` and `at=…_stall`; the `main` line shows
  `unit=-` and `children=` listing both workers; the `Timeout (` dumps before the kill: each worker's names
  `_stall`, main's names `_dispatch`. Record the kill offset = (the last `[probe] utc=` epoch, ±5 s) − (the start
  line's dump time + 600) (spec R6). If the job is killed 300 s or more before the computed deadline, stop and report.
  D ends `Job timeout` by design: one ERROR job and one owner email.
- [ ] **Step 6: Report** to the owner: the spec §6 table filled with measured numbers, the per-container vs shared
  ceiling reading (A's 4 times), C's and B's parity verdicts, B's max wall-clock, and D's log excerpt and offset.
  A miss on any target → stop and report.

### Task 9: Review, amend, force-push, retrain — OWNER GATES

- [ ] **Step 1:** implementation A/B review (`(Ragna): Review A-Deux`, `(D32): Review B-Deux`; ≤ 3 rounds, delta-only after round 1). Fix or answer every finding.
- [ ] **Step 2:** `/final-review`.
- [ ] **Step 3: Amend gate.** Show `git status`, the full `git diff --stat` and diff, and the proposed commit
  message (the existing f03f4753 message extended for this amendment, Conventional Commits, `Co-Authored-By:
  Claude Opus 5.5 <noreply@anthropic.com>`). Wait for an explicit yes, then `git commit --amend`.
- [ ] **Step 4: Force-push gate.** Separate explicit yes, then `git push --force-with-lease origin feat/minimal-hf-driver`.
- [ ] **Step 5:** watch PR #299 CI to green (`gh pr checks 299`); a red check stops here.
- [ ] **Step 6: Retrain gate.** On the DGX, `git fetch && git checkout` the new HEAD; `campaign … preflight` then
  `launch` per the `hf-campaign` skill, after an explicit owner yes.
- [ ] **Step 7: Report** the spec §6 post-commit metrics from the HF job list: cumulative ERROR jobs by reason,
  cv timeouts separately, `Errno 5` recurrences with their tracebacks, longest job vs 7,200 s, match-outcome stage
  wall-clocks, total wall-clock to 21/21.

---

## Change 6 delta (spec §13, owner 2026-11-06) — APPROVED

Tasks 1–9 shipped in `e9301e04`; retrain launch 2 ran on it. Tasks 10–17 fix its five faults (spec §13.1) and land
in the same cycle commit (an amend of `e9301e04`, owner-gated as Task 9 Steps 3–4). Task 17 replaces Task 9
Steps 6–7 with a new launch (all 21 instances; spec §13.8). Task 16 runs before Task 15: the 2-timeout rule reads
the `[unit] … start` lines that Task 16 adds.

Round-5 and round-6 findings (`D:\Development\_reviews\2026-11-06-hfu-reduce-reads-plan-r{5,6,7}-*.md`) are
answered in the tasks below; HFURR-PLAN-21 (implementation began before approval) was stopped until approval.

**Global Constraints for the delta** (these replace the two lines above that they contradict):
- `silly_kicks/` changes are in scope for Change 6a only: `silly_kicks/_numba_dispatch.py` and the six dispatch
  families of spec §13.2. No kernel arithmetic changes; every parity test stays green.
- The wheel's dependencies change, but `silly_kicks/_version.py` stays 4.139.1 in this amend (owner decision
  2026-11-06; the bump comes with the release, ADR-079).
- The existing switch names stay exact (Hyrum): `SILLY_KICKS_GHOST_FORCE_NUMPY`, `SILLY_KICKS_DAS_FORCE_NUMPY`,
  `SILLY_KICKS_COORDINATION_FORCE_NUMPY`. The numba on-disk cache stays opt-in (`SILLY_KICKS_NUMBA_CACHE`,
  `NUMBA_CACHE_DIR`).
- New test files have unique basenames (the test dirs have no `__init__.py`).

**Review Focus for the delta:**
1. A kernel module that imports numba and then raises `RuntimeError` (the cache-locator trap, `_numba_kernels.py`
   comment) must propagate, never become "numba unavailable". Test in Task 10:
   `test_kernel_runtime_error_is_not_swallowed`.
2. A new FileSet generation (new token) must get a new cache entry, never the old bytes. Test in Task 13:
   `test_new_generation_gets_new_cache_entry`.
3. A `--plan` timeout must never be read as a rate limit (no back-off loop). Test in Task 14:
   `test_plan_timeout_is_not_rate_limited`.
4. Timeouts of jobs in the resume baseline must not count toward the 2-timeout rule. Test in Task 15:
   `test_baseline_timeouts_do_not_count`.
5. A read under 8 MB, and a missing file, print no `[read]` line. Test in Task 13:
   `test_small_and_missing_reads_print_nothing`.

### Task 10: The numba dispatch module and the core dependency (spec §13.2)

**Files:** Create `silly_kicks/_numba_dispatch.py`, `tests/test_numba_dispatch.py`,
`tests/test_numba_core_dependency.py`. Modify `pyproject.toml` (`[project] dependencies` `:33`, `[numba]` `:47`,
`[calibration]`, `[test]`), `uv.lock` (`uv lock`).

**Interfaces — Produces:**
- `FAMILIES: tuple[str, ...] = ("GHOST", "DAS", "COORDINATION", "PITCH_CONTROL", "BALL_CARRIER", "XTGK")`.
- `class NumbaUnavailableError(ImportError)`; its message names `pip install --force-reinstall silly-kicks`.
- `use_numba(family: str) -> bool`: `ValueError` for a family not in `FAMILIES`; False when
  `os.environ.get(f"SILLY_KICKS_{family}_FORCE_NUMPY") == "1"` (numba is not imported); else imports numba once
  (module-level cache) and returns True; an `ImportError` from `import numba` raises `NumbaUnavailableError` from
  it. Any other exception propagates unchanged.
- `require_numba() -> None`: the same import for an explicit numba request; ignores the switches; raises
  `NumbaUnavailableError` when numba cannot be imported.
- `numba_version() -> str`: `numba.__version__`, or `"absent"` (never raises; for log lines only).

- [ ] **Step 1: RED.** `test_numba_dispatch.py`: `test_force_switch_skips_numba` (switch set; `sys.modules`
  shows no new `numba` import via a `monkeypatch.setitem(sys.modules, "numba", None)` that would raise if
  touched), `test_missing_numba_raises` (`sys.modules["numba"] = None` -> `NumbaUnavailableError`, and
  `isinstance(err, ImportError)`), `test_unknown_family_raises`, `test_kernel_runtime_error_is_not_swallowed`
  (a fake `numba` module whose import raises `RuntimeError` -> `RuntimeError`), `test_numba_version_absent`.
  `test_numba_core_dependency.py`: parse `pyproject.toml` with `tomllib`: a `numba>=0.59.0` entry in
  `project.dependencies`; `optional-dependencies.numba == []`; no numba entry in `test` or `calibration`; the
  `uv.lock` root package lists `numba` among its non-optional dependencies.
  Run: `python -m pytest tests/test_numba_dispatch.py tests/test_numba_core_dependency.py -q`. Expected: FAIL
  (module missing; numba still optional).
- [ ] **Step 2: GREEN.** Write the module; move numba in `pyproject.toml`; `uv lock`. Expected: PASS.

### Task 11: Route every dispatch site through `use_numba` (spec §13.2 census)

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py` (`_use_ghost_numba` `:62-85`),
`silly_kicks/tracking/_das_engine.py` (`_numba_available` `:52-60`), `silly_kicks/tracking/_das.py:130`,
`silly_kicks/coordination/_kernels/_numba.py` (`use_numba` `:33-35`),
`silly_kicks/tracking/pitch_control/_fernandez_bornn.py` (`:21-26`, `:52`),
`silly_kicks/tracking/pitch_control/_spearman.py` (`:23-28`, `:60`, `:111`), `silly_kicks/tracking/_ball_carrier.py`
(`:21-26`, `:443`), `silly_kicks/xtgk/_turnover.py` (`:96-106`), `scripts/validate_das_native_parity.py`
(`_numba_available`). The kernel modules (`_ghost_gk_numba`, `_das_numba`, `_numba_kernels`, `_ball_carrier_numba`,
`coordination/_kernels/_numba`) keep their `from numba import njit` and their import-time `ImportError`, except
`_das_numba.AVAILABLE` (`:21-27`): its silent fallback in `compute_das` (`_das_engine.py:264-268`) goes.
`_ghost_gk.predict_density` (`kde_backend`, `:2132-2174`) and `compute_das` / `compute_das_paired` (`engine=`,
`:257`, `:280`) follow the explicit-engine rule of spec §13.2. Tests updated where they patch the old flags (the
census grep `_HAS_NUMBA|HAVE_NUMBA|_NUMBA_TURNOVER|_numba_available|_use_ghost_numba|_HAS_GHOST_NUMBA|FORCE_NUMPY`):
`tests/coordination/kernels/test_cluster.py`, `test_surrogates.py`, `test_numba_parity.py`;
`tests/xtgk/test_turnover_perf.py`; `tests/scripts/test_das_native_parity_driver.py`,
`tests/scripts/test_reduce_split.py`; `tests/tracking/test_ball_carrier_numba_parity.py`, `test_das.py`,
`test_das_cost_guardrail.py`, `test_das_engine_parity.py`, `test_das_kernel.py`,
`test_ghost_gk_kde_auto_default.py`, `test_ghost_gk_leaf_numba.py`, `test_ghost_gk_kde_vectorized.py`;
`tests/tracking/pitch_control/test_numba_parity.py` (the pitch-control parity test). The two fixture files the grep
also hits (`tests/fixtures/claude_md_at_77286f4.md`, `agents_md_invariant_inventory.json`) are historical
snapshots and do not change. Also the 9 `importorskip("numba")` / numba `skipif` sites (`test_surrogates.py`,
`test_cluster.py`, coordination `test_numba_parity.py`, `test_turnover_perf.py`, `test_das_benchmark.py`,
`test_das_kernel.py`, `test_das_engine_parity.py`, `test_ghost_gk_kde_auto_default.py`,
`test_ghost_gk_leaf_numba.py`): numba is core, so each skip goes and the test always runs. Create
`tests/test_numba_dispatch_registry.py`.

**Interfaces — Consumes:** `use_numba`, `NumbaUnavailableError` (Task 10).

**Rules:**
- Each family decides once per call: `if use_numba("<FAMILY>"): <numba kernel> else: <numpy reference>`. The
  numba kernel module is imported lazily inside the True branch (bound once, as `_use_ghost_numba` does today,
  so the "bare import is numba-light" pins `test_ghost_gk_does_not_eagerly_import_numba` stay green). An
  `ImportError` from a kernel module after `use_numba` returned True re-raises as `NumbaUnavailableError`.
- `coordination/_kernels/_numba.use_numba()` keeps its name and returns `use_numba("COORDINATION")`; its
  module-level `HAVE_NUMBA` stays only to gate the eager `@njit` compile.
- `_turnover`: `_opp_first_shot_scan_fast` is bound on first call when `use_numba("XTGK")`; else the pure-Python
  scan.
- Check `docs/PRIVATE_CONSUMERS.md` for pins on `_HAS_NUMBA`, `_numba_available`, `_use_ghost_numba`,
  `_NUMBA_TURNOVER`, `HAVE_NUMBA` before any rename; keep a pinned name.

- [ ] **Step 1: RED.** `test_numba_dispatch_registry.py`: `test_no_numba_import_outside_kernels` (AST scan of
  `silly_kicks/`: `import numba` / `from numba` appears only in `_numba_dispatch.py`, the 5 kernel modules and
  `xtgk/_turnover.py`), `test_every_family_routes_through_dispatch` (for each family, monkeypatch its
  `SILLY_KICKS_<F>_FORCE_NUMPY=1`, spy the numba kernel, call the public entry — `compute_ghost_gk`/
  `_vectorized_leaf_values`, `get_das`, `compute_team_coordination` kernel path, `pitch_control_at_target`
  (both models), `infer_ball_carrier`, `compute_xt_gk_v2` turnover scan — and assert the kernel is not called;
  then hide numba and assert `NumbaUnavailableError`), `test_explicit_numba_engine_requires_numba`
  (`compute_das(..., engine="numba")` and `predict_density(..., kde_backend="cpu-numba")` with numba hidden raise
  `NumbaUnavailableError`, also with the DAS/GHOST switch set), `test_explicit_numpy_engine_never_imports_numba`
  (`engine="numpy"`, `kde_backend="vectorized"` with numba hidden return results). Expected: FAIL (four families
  have no switch; all fall back silently; `engine="numba"` falls back through `AVAILABLE`).
- [ ] **Step 2: GREEN.** Route the sites. Then run every parity test of the census:
  `python -m pytest tests/tracking/test_ghost_gk_leaf_numba.py tests/tracking/test_ball_carrier_numba_parity.py
  tests/tracking/test_das_engine_parity.py tests/tracking/test_das_kernel.py tests/coordination/kernels/
  tests/xtgk/test_turnover_perf.py tests/tracking/test_numba_cache_gating.py tests/test_numba_dispatch_registry.py
  tests/tracking/pitch_control/test_numba_parity.py -q`. Expected: PASS, no skip for "numba not installed".

### Task 12: numba is visible in jobs and in preflight (spec §13.2)

**Files:** Modify `scripts/hf_jobs/liveness.py` (start line `:164`), `scripts/hf_jobs/preflight.py` (new
`check_numba`), `scripts/hf_jobs/campaign.py` (`cmd_preflight` check list). Tests:
`tests/scripts/hf_jobs/test_job_liveness.py`, `tests/scripts/hf_jobs/test_preflight.py`.

**Interfaces — Produces:** `check_numba(*, run=subprocess.run) -> Check`. It runs
`uv export --frozen --no-hashes --extra train --group hfjobs` (the job's install set, `launch.py:204`), parses each
requirement line with `packaging.requirements.Requirement`, evaluates its marker with the job environment
(`sys_platform="linux"`, `platform_system="Linux"`, `platform_machine="x86_64"`, `python_version="3.12"`,
`python_full_version="3.12.14"`, `implementation_name="cpython"`, `os_name="posix"`), and passes only when a `numba`
requirement applies there; the detail names its version.
Ruling (plan): a static export of the locked install set replaces the spec's `import numba` in a synced env — it
checks exactly what the job installs, takes about 1 s, and does not mutate `.venv`. Spec §13.2 wording follows
after review.

- [ ] **Step 1: RED.** `test_start_line_names_numba` (the start line contains `numba=`);
  `test_check_numba_fails_without_numba` (a fake `run` whose export lacks numba -> `ok=False`);
  `test_check_numba_passes` (fake export with `numba==0.65.1`); `test_check_numba_ignores_other_platform_line`
  (`numba==0.65.1 ; sys_platform == 'win32'` alone -> `ok=False`); `test_preflight_runs_six_checks` (the timed
  `[preflight]` lines name `check_numba`). Expected: FAIL.
- [ ] **Step 2: GREEN.** Implement. Expected: PASS.

### Task 13: A FileSet is read once per job; large reads log their rate (spec §13.3)

**Files:** Modify `scripts/_stages.py` (`LocalStore.read_bytes` `:255-262`, `materialize_fileset` `:890-899`),
`scripts/hf_jobs/executor.py` (`LocalExecutor.run_stage` `:302`: own the cache dir), `pyproject.toml` (`[train]`
gains `filelock>=3.12`). Test: create `tests/scripts/test_fileset_cache.py`.

**Interfaces — Produces:**
- `FILESET_CACHE_ENV = "SILLY_KICKS_FILESET_CACHE"`. `LocalExecutor.run_stage` creates a temp dir, sets the env for
  its workers, and removes the dir after the stage (it is the cache's creator).
- `materialize_fileset(ctx, stage, key, dest)` (signature unchanged): with the env set, it materializes into
  `<cache>/<generation-token>/<key>/` under `filelock.FileLock(<that dir>.lock)`: if the dir exists, copy it to
  `dest`; else read through `read_result` (sha256-verified), write to `<dir>.part-<pid>`, `os.replace` it to the
  dir, then copy. Without the env: today's behaviour.
- `LocalStore.read_bytes`: for a file of 8 MB or more (size from `os.fstat` on the open fd, no extra `stat`), read in
  8 MB chunks; print `[read] <rel> <MB> MB in <s> s (<MB/s> MB/s)` at the end and a `[read] <rel> <done>/<total> MB
  after <s> s` line every 60 s. Smaller files and missing files: unchanged, silent.

- [ ] **Step 1: RED.** `test_fileset_read_once_across_processes` (3 spawn processes, a store whose `read_bytes`
  appends to a counter file: one store read, identical bytes in all 3 dests), `test_killed_holder_leaves_no_cache`
  (a holder killed after writing `.part-<pid>`: the next process reads from the store and the cache dir is
  complete), `test_new_generation_gets_new_cache_entry`, `test_large_read_prints_rate`,
  `test_small_and_missing_reads_print_nothing`, `test_no_env_means_direct_read`. Expected: FAIL.
- [ ] **Step 2: GREEN.** Implement; `uv lock`. Then `python -m pytest tests/scripts/test_fileset_cache.py
  tests/scripts/test_stages*.py -q`. Expected: PASS.

### Task 14: Hub timeouts and the `--plan` deadline (spec §13.4)

**Files:** Create `scripts/hf_jobs/hub_client.py`, `tests/scripts/hf_jobs/test_hub_client.py`. Modify
`scripts/hf_jobs/__main__.py` (`main`), `scripts/_stages.py` (`main` `--plan` path `:1122`),
`scripts/hf_jobs/campaign.py` (`read_plan` `:189-195`), `scripts/hf_jobs/driver.py` (`_gather` `read_one`).

**Interfaces — Produces:**
- `HUB_TIMEOUT: dict[str, float] = {"connect": 10.0, "read": 60.0, "write": 60.0, "pool": 10.0}`;
  `bound_request_timeout(request) -> None`: a request event hook that replaces each `None` value of
  `request.extensions["timeout"]` with the `HUB_TIMEOUT` value and keeps a finite caller value (it runs after the
  per-request timeout is set, `httpx2/_client.py:1040-1043`, so it wins over `paginate`'s explicit `timeout=None`);
  `install_timeouts() -> None`: `set_client_factory` with a factory that builds the library default client
  (`huggingface_hub.utils._http.default_client_factory`, verified in 2.2.0) and appends the hook to
  `client.event_hooks["request"]`. Idempotent.
- `PLAN_TIMEOUT_S = 600`; `PLAN_DEADLINE_ENV = "SILLY_KICKS_PLAN_DEADLINE_S"`. `read_plan` passes `timeout=` and
  the env; on `subprocess.TimeoutExpired` it raises `PlanTimeoutError(RuntimeError)` whose message holds
  `--plan failed (timeout after 600 s)` and the last 4,000 chars of the child's stderr (the dump), decoded as UTF-8
  with `errors="replace"` when it is `bytes` (POSIX) and used as is when it is `str` (Windows).
- `_stages.main --plan`: when the env is set, `faulthandler.dump_traceback_later(deadline - 30, exit=False)`;
  when the store is `hf://`, `install_timeouts()` first.

- [ ] **Step 1: RED.** `test_hook_fills_none_and_keeps_finite` (a request with `Timeout(None)` gets the 4 values;
  a request with `read=5` keeps 5); `test_bucket_listing_times_out` and `test_bucket_read_times_out` (a local socket
  server on `127.0.0.1` that accepts and never answers. The endpoint goes in through the store, not the env
  (`HF_ENDPOINT` is read once at import, `huggingface_hub/constants.py:69`): `BucketStore("ns/b", "rk",
  api=HfApi(endpoint=f"http://127.0.0.1:{port}"))`. `BucketStore.names` — the `list_bucket_tree` / `paginate` path —
  and `BucketStore.read_bytes` each raise a timeout-class error (`httpx2.TimeoutException`, directly or as the
  `__cause__` of the library's wrapper) within the retries; the server records at least one accepted connection,
  so the call reached it. The test patches `HUB_TIMEOUT["read"]` to 0.5 s and the backoff waits to 0. The call runs
  in a daemon thread joined with a 30 s limit, so a hang fails the test instead of blocking the suite. RED, before
  the hook is installed: the listing call does not return within 30 s (A-PLAN-20). `install_timeouts` replaces the
  process-wide client factory, so an autouse fixture in `test_hub_client.py` restores
  `set_client_factory(default_client_factory)` and closes the session after each test (A plan-r7 note)); `test_read_plan_timeout_carries_stack` (a child module that
  sleeps in `_hang()`; `PLAN_TIMEOUT_S` patched to 3; the error text holds `File "` and `_hang`);
  `test_plan_timeout_is_not_rate_limited` (`_plan_rate_limited(PlanTimeoutError(...))` is False and `step`
  reports the instance as `--plan failed`). Expected: FAIL.
- [ ] **Step 2: GREEN.** Implement. Expected: PASS.

### Task 15: A unit that times out twice is STUCK (spec §13.5)

**Files:** Modify `scripts/hf_jobs/driver.py` (`_attribute` `:802-880`, `decide` candidate filter `:551-554`,
`_gather` return). Test: `tests/scripts/hf_jobs/test_hf_driver.py` (update
`test_coverage_excludes_timeouts` `:575` to the new rule).

**Interfaces — Consumes:** the `[unit] <stage>/<key> start` and end lines (Task 16).
**Interfaces — Produces:**
- `started_unfinished(log_lines: Iterable[str], stage: str) -> frozenset[str]`: the keys with a `[unit]
  <stage>/<key> start` line and no `[unit] <stage>/<key> <outcome> in` line.
- `_attribute(..., fetch_logs: Callable[[str], Iterable[str]])`: for each new RERUN job it fetches the log once
  (`api.fetch_job_logs`), stores `started` (sorted keys) beside the manifest in the per-run_key cache, and charges
  only those keys. A fetch error: the job is not cached this step (it is retried next step) and one
  `[driver] job <id>: log fetch failed (<error>); timeout not charged yet` line prints.
- It also returns `timeouts: dict[tuple[str, str, str], tuple[str, ...]]` (unit -> the ids of the RERUN jobs that
  charged it, `created_at` order, baseline excluded). `TIMEOUT_LIMIT = 2`. `decide` drops a unit with
  `len(timeouts[unit]) >= TIMEOUT_LIMIT` from `to_run` and adds one report per such unit:
  `(<instance>, "<stage>/<unit> timed out 2x (jobs <id>, <id>)")`.

- [ ] **Step 1: RED.** `test_two_timeouts_make_a_unit_stuck` (the unit started in both logs),
  `test_one_timeout_still_relaunches`, `test_unit_done_after_a_timeout_is_not_reported`,
  `test_queued_unit_is_never_charged` (held by 2 RERUN jobs, no start line in either: relaunched, no report),
  `test_log_fetch_failure_charges_nothing_and_retries`, `test_baseline_timeouts_do_not_count`,
  `test_started_unfinished_parses_unit_lines`. Expected: FAIL.
- [ ] **Step 2: GREEN.** Implement. Then `python -m pytest tests/scripts/hf_jobs/test_hf_driver.py -q`.
  Expected: PASS.

### Task 16: The log trail (spec §13.6)

**Files:** Modify `scripts/hf_jobs/driver.py` (`step`, `run`), `scripts/_stages.py` (`run_unit` `:744`, `main`).
Tests: `tests/scripts/hf_jobs/test_hf_driver.py`, create `tests/scripts/test_unit_log_lines.py`.

**Interfaces:**
- `step(..., reported: set[str] | None = None)`: prints `[driver] job <id> <instance>/<stage> <shard> <reason>
  after <running_secs> s` for each ERROR or CANCELED job of this commit not in `baseline_error_ids` and not in
  `reported`, then adds its id. `run` owns one `reported` set for its life. `running_secs` comes from the job
  record (`durations.running_secs`; `-` when absent). After each successful launch:
  `[driver] launch <instance>/<stage> <shard> units=<n>`.
- `run_unit` prints `[unit] <stage>/<key> start` before the work and `[unit] <stage>/<key> <outcome> in <wall_s:.1f>
  s` after the ledger write (outcome = done, excluded or failed).
- `_stages.main` prints `[job] <done>/<units> done, <failed> failed, exit <rc>, <s> s` on every exit of a run
  (`return 0`, `return 2` REFUSED, `return 3` STAGE FAILED; not `--plan`), through one `try`/`finally`, with the
  counts from the executor `StageReport`s gathered so far.

- [ ] **Step 1: RED.** `test_error_job_line_once`, `test_launch_line_per_launch`, `test_unit_lines_with_duration`,
  `test_job_summary_line` (parametrized over exits 0, 2 and 3). Expected: FAIL.
- [ ] **Step 2: GREEN.** Implement. Expected: PASS.

### Task 17: Docs, gates, resume

- [ ] **Step 1: Docs (spec §13.9).** ADR-076 amendment, and the same amendment to ADR-008 D3 and ADR-013; ADR-128
  amendment items, and "five checks" -> six at ADR-128 `:122`, `:360` and the `preflight.py` docstring; `AGENTS.md` Dependencies line
  (keep `tests/test_agents_md_budget.py` green); `docs/context/conventions-core.md`; `docs/howto/compute-pipelines.md`
  and `docs/howto/hf-jobs-training.md`; `CHANGELOG.md` (the `[Unreleased]` section: Changed — numba core,
  Added — the log lines, Fixed — F2–F4). Grep live docs for "optional `[numba]`" / "pip install
  silly-kicks[numba]" and fix live text.
- [ ] **Step 2: Local gates.** Lint at CI scope with the pinned versions (scratch venv). Targeted tests for every
  touched module, `tests/scripts/hf_jobs` on its own. Show real exit codes. CI runs the full suite.
- [ ] **Step 3: Review and commit gates.** Implementation A/B review of the delta; `/final-review`; no version
  bump (owner 2026-11-06); Task 9 Steps 3–5 (amend yes, force-push yes, CI green).
- [ ] **Step 4: Launch gate (replaces Task 9 Step 6).** Show the owner the spend first (spec §13.8): the amend is a
  new run_key, so all 21 instances run, the 9 finished in launch 2 included; only the `cross_commit_version`
  stages carry over. Launch 2 used 60.3 compute-hours (about $1.81), about 22 h of it the 11 timeouts and 2.7 h the
  9 finished instances. Owner yes, then `campaign … preflight` and `campaign … launch` on the new commit.
- [ ] **Step 5: Report** spec §13.8.
