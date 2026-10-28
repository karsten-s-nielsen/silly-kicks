# HF full-utilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (owner runs inline; no subagents
> per CLAUDE.md). Steps use `- [ ]`. TDD each step.

**Goal:** Make every campaign job use all 8 vCPU or record why it cannot (a sequential search's idle cores),
split every splittable stage, and make no job's timeout a function of the 30-minute target — byte-identical
to the single-thread artifact — by threading `cv`/`study`, overlapping independent stages (DAG), sizing
timeouts from measured walls, slot-filling free dispatch slots, and reusing unchanged preflight work.

**Architecture:** This cycle AMENDS `cccb702` (PR #296), which already ships the `thread_safe`/`thread_lever`
opt-in, `thread_budget`, executor per-lever injection, the xgb `runtime_n_jobs` kwarg, the
`dispatch_cap`/`shard_cap` split, dispatch-overflow handling, and the fit-stage byte-identity gate. This
cycle extends the opt-in to `cv` (and corrects `study` inner width), replaces the linear stage barrier with a
DAG schedule, decouples every job timeout from `target_s`, adds slot-filling with a `budget_floor`,
re-profiles `startup_s`, profiles reduces, shards a heavy reduce byte-identically, and makes preflight
incremental.

**Tech Stack:** Python; `scripts/hf_jobs/` (planner, campaign controller, executor, preflight, launch);
`scripts/_stages.py`; the four trainers; `scripts/train_*`; pandas/numpy/scikit-learn/xgboost; HF Jobs
`cpu-upgrade`; pytest; `.venv-hfjobs` (`hf_hub>=2.2`).

**Spec:** `docs/superpowers/specs/2026-10-29-hf-full-utilization-design.md` (r5, A+B APPROVED).
**Branch:** `feat/controller-step-reads` (amends `cccb702`; no new branch).

## Global Constraints

- **ONE commit per cycle (CLAUDE.md).** All tasks build on one working tree; there are **no per-task
  commits**. The single commit is an **amend of `cccb702`** and happens once, at the cycle gate, after the
  whole-branch A/B impl review AND an explicit owner YES. The amend and the force-push of
  `feat/controller-step-reads` are each a separate explicit owner gate showing the exact state (spec
  "Interaction and the owner gates"). No merge/adopt/publish without a per-action owner gate.
- **No rebaseline.** Every artifact stays bit-for-bit identical to the single-thread, single-process build,
  proven by `tests/tracking/test_thread_identity.py` bound by `tests/scripts/test_thread_safe_registry.py`.
  A stage without a passing identity test stays pinned to one thread.
- **R3: `target_s` is a SIZING target only.** No job timeout is a function of `target_s`. Every timeout
  derives from a measured `p99` or the fixed `profile_timeout_s`. No job is failed for exceeding 30 minutes.
- **Default/shared/reduce/adapter worker env stays `deterministic_env()`** (all thread vars `"1"`); the
  thread budget is injected only on the `thread_safe` worker-launch path, per lever (`omp` raises
  `OMP_NUM_THREADS` only; `n_jobs` exports `SILLY_KICKS_RUNTIME_THREADS`); `OPENBLAS/MKL/VECLIB` and
  `NUMBA_NUM_THREADS` stay `"1"` on every path.
- **No owner-tier raw data committed** — synthetic fixtures at real shape and scale only, provenance
  recorded.
- **Task order (deps):** **T0 (Step 0 — measure first, ON HF as measurement jobs, no push; owner ruling
  2026-10-30: the DGX is for debugging only) PRECEDES everything** — the
  approved spec requires "Step 0 before building the rest / no stage opted in on a guess". Its measured walls
  set `profile_timeout_s`/`budget_floor` (T2), the per-stage opt-in set (T11), and the `win_probability`
  reduce shard decision (T8). Then T1 → T2 → {T3, T4}; T5 → {T6, T7}; T4 → T8; T2 → T8; T9, T10 independent;
  T11 ← (T0 + T3 + shipped thread infra); T12 ← T5 + T6 + T11; T13 after all code; T14 at the gate.

## Verification (how to run, per step)

- Unit/local: `.venv/Scripts/python.exe -m pytest <path> -q`. `scripts/hf_jobs` tests need `.venv-hfjobs`
  (`hf_hub>=2.2`): `.venv-hfjobs/Scripts/python.exe -m pytest <path> -q`.
- Lint at CI scope: `uvx ruff@0.15.7 check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright`.
- Identity + parity thread arms run on x86 in CI (the HF `cpu-upgrade` production arch); no aarch64/DGX leg
  (DGX is a backup, never the byte-identity benchmark).

## Review Focus (input classes the tests must exercise)

- A `thread_safe` stage on its FIRST run (no committed wall): the profile job must not die at
  `profile_timeout_s` (bounded `k`), and the inline reduce must not die at the 1800 floor (first-run
  `max(plan_stage timeout, profile_timeout_s)` fallback). [T3, T4]
- A halt in one tier (e.g. `xshot_occurrence` public) must NOT cancel a same-`job`/`variant` sibling in
  another tier (`sc_extended`): the cancel filter is `(run_key, variant)`, not `(job, variant)`. [T6]
- Two active instances must split free dispatch slots fairly — the first instance must not slot-fill into
  every free slot. [T7]
- A sharded reduce must equal the single reduce byte-for-byte, including id-column idempotence
  (`storable_id_column`) and a lossless parquet round-trip of each partial. [T8]
- `{gen}/_reduce.json` (a wall-time file) must be VOLATILE (`_parity.py`) and a controller record
  (`store._is_record`) or every reduce-stage parity arm goes red and each step adds a per-stage download.
  [T4]
- `cv` byte-identity compares the per-fold result dicts AND the reduced table (`cv` saves NO model files).
  [T11]
- `target_s` lowered to 1800 must NOT shrink any job timeout (the four non-`plan_stage` action sites). [T2]

---

## Task 0: Step 0 — per-stage utilization audit (measure FIRST; local/DGX, no push)

> **Status (2026-10-31, review S3B-15).** Tasks 0-13 shipped in `2474b73` (PR-S204). Their step boxes below are
> the original TDD sub-steps, kept unticked as written; the tests exist. Later revisions changed four of them:
> Task 0's local/DGX audit was re-done on HF as R3-8 Step 0 (owner ruling: HF is the measuring platform); Task 1's
> W=1 rule for a `thread_safe` stage of 7 units or fewer became the THREADS shape (R3-8 S3 item 2; test
> `test_a_thread_safe_stage_of_seven_units_or_fewer_runs_all_at_once_in_one_job`); Task 6's test is
> `test_halt_cancels_only_this_instance_by_run_key_tier_variant` (r4 D18 adds the tier); Task 11's `cv` opt-in was
> reversed by r3 D16 (owner ruling) and reopened per stage by R3-8 S3 items 1-3 behind the thread-identity gate.
> Task 7's test is `test_slot_filling_fans_toward_free_slots_bounded_by_budget_floor`. Task 14 (release prep at the
> commit gate) is this amend's gate and stays open.

**Files:**
- Create: a throwaway measurement script (scratchpad; NOT committed).
- Create: the committed audit summary (`docs/research/hf_campaigns/<campaign>-audit.json` or the campaign
  summary audit field).
- Modify: none of the production paths yet (no opt-in, no HF launch, no push — spec "Step 0 before building
  the rest").

**Interfaces:**
- Produces LOCAL/DGX (host-INDEPENDENT, chosen from the local measurement): the `profile_timeout_s` and
  `budget_floor` values; the per-stage SHAPE decision by spec §B; the study per-CHILD (per-fold) RSS, and
  the verdict whether fold-width (5) children fit the usable RAM (for the T11 sizing correction, B-14);
  whether `win_probability`'s reduce is heavy (decides T8).
- Produces HF (host-CORRECT, feeds the PLANNER channel): the per-stage walls + median `startup_s` come from
  the first HF profiling run's committed summary, with `history_key` + host, read by `load_history`
  (`ledger.py:168-175`) and `median_startup_s` (`:121-134`). The local run has NO `history_key` (dropped by
  `load_history`) and the wrong host, so it does NOT feed the planner cost channel — it only CHOOSES values
  and shapes (B-15).

- [ ] S1: run each stage's work at 1 and 8 threads locally (or on the DGX), timing one part: the 8-thread
  lone fit per model (ghost_gk, ghost_outfield HistGB; xshot, xcross xgb), per-fold `cv`, and the
  `win_probability` `extract` reduce at real scale; and measure the study per-CHILD (per-fold) RSS against
  the usable RAM (B-14). Run the stage work directly with the env thread count set — no controller, no
  `plan()`, no opt-in needed to MEASURE.
- [ ] S2: choose `profile_timeout_s` (≈ `k ×` the longest 1-thread part wall + margin) and `budget_floor`
  (startup ≈ 10% of the minimum per-job work); decide each stage's §B shape; record whether the study
  per-child RSS lets fold-width (5) children fit. Write all of this to the committed audit summary.
- [ ] S3: the planner-channel per-stage walls + bootstrap `startup_s` are NOT taken from this local run —
  they come from the first HF profiling run (T10 commits the HF `startup_s`; the profile→summary mechanism
  commits the HF walls with `history_key`). This local audit chooses values/shapes/study-fit only (B-15).
- [ ] No TDD cycle: the deliverable is the committed audit (values + shapes + study-fit verdict), not code.
  T12 adds the audit ASSERTION (saturate-or-irreducible) in code.

## Task 1: Planner sizes a thread_safe non-inner stage at W=1; target from the TOML; plan() emits thread_safe

**Files:**
- Modify: `scripts/_stages.py` (`plan()` emits `thread_safe`/`thread_lever`, `:919-942`)
- Modify: `scripts/hf_jobs/planner.py` (`plan_stage`, `:108`)
- Modify: `scripts/hf_jobs/campaigns/cycle1.toml` (`target_minutes`)
- Test: `tests/scripts/hf_jobs/test_planner.py`, `tests/scripts/hf_jobs/test_campaign.py:109`,
  `tests/scripts/test_compute_contract.py` (plan() shape)

**Interfaces:**
- Consumes: `st.thread_safe`/`st.thread_lever` (shipped on the Stage objects at `cccb702`); `thread_budget`
  (shipped).
- Produces: `_stages.plan()` emits `thread_safe` and `thread_lever` per stage into the `--plan` JSON
  (`:919-942`, which today emits `inner_parallel`/`inner_width` but NOT these — B-01); `plan_stage` returns
  `workers == 1` for a `thread_safe` non-inner stage, `n_jobs` sized so each one-unit job's predicted wall
  targets `target_s`; `cycle1.toml` `target_minutes = 30`.

- [ ] S1: failing test `test_plan_emits_thread_safe` — `_stages.plan()` output carries `thread_safe` and
  `thread_lever` for a thread_safe stage (RED today: `plan()` omits them, so the controller's `plan_stage`
  never sees them and the W=1 rule is dead in production while a hand-built-dict unit test would pass — B-01).
- [ ] S2: failing test `test_thread_safe_non_inner_plans_workers_1`, driven from REAL `plan()` output (not a
  hand-built stage dict) — a `thread_safe` non-inner stage with N heavy fold units plans `workers == 1`,
  `inner_workers == 1`, and `n_jobs` fans the units so each job's `predicted_job_wall_s ≈ per_unit_wall +
  startup` (not `vcpu` workers at `budget == 1`).
- [ ] S3: failing test `test_cycle1_target_is_30_minutes` — the committed `cycle1.toml` has
  `target_minutes == 30`; update `test_campaign.py:109` (currently asserts `3600`) to assert
  `constraints().target_s == 1800`.
- [ ] S4: run → fail.
- [ ] S5: emit `thread_safe`/`thread_lever` in `_stages.plan()`; in `plan_stage`, set `workers = 1` on the
  `thread_safe and not inner_parallel` branch (the worker gets `thread_budget(vcpu, 1, 1) = 8`); set
  `cycle1.toml` `target_minutes = 30`. Note: `campaign_sha256` changes, so a fresh preflight is required (T9
  makes it fast). Run → green.

## Task 2: Decouple every job timeout from `target_s`

**Files:**
- Modify: `scripts/hf_jobs/planner.py` (`Constraints`: add `profile_timeout_s`, `budget_floor`)
- Modify: `scripts/hf_jobs/campaign.py` (`next_actions`, the four `int(2 * c.target_s)` sites around `:253`)
- Test: `tests/scripts/hf_jobs/test_campaign.py`, `tests/scripts/hf_jobs/test_planner.py`

**Interfaces:**
- Produces: `Constraints.profile_timeout_s: float` (a generous fixed ceiling, independent of `target_s`) and
  `Constraints.budget_floor: float` (min work per job, used by T7); the escalate/reduce/`n_units≤2`/profile
  actions take a timeout from a measured `p99` or `profile_timeout_s`, never `2 * target_s`. The VALUES of
  `profile_timeout_s`/`budget_floor` come from T0's measurement.

- [ ] S1: failing test `test_tiny_stage_timeout_independent_of_target` — the `n_units≤2` run action's timeout
  is `timeout_safety × p99` from the stage's committed HF-measured wall (via its `history_key`; host-correct,
  NOT the T0 local audit — B-15), floored by `timeout_floor_s`; with no committed wall it is
  `profile_timeout_s`; lowering `target_s` from 3600 to 1800 does NOT change it.
- [ ] S2: failing test `test_profile_action_timeout_is_profile_timeout_s` — the profile action's timeout is
  `profile_timeout_s`, independent of `target_s`.
- [ ] S3: failing test `test_escalate_timeout_independent_of_target` — the escalate action's timeout is
  `timeout_safety × p99` of the unit's measured wall if known, else `profile_timeout_s`; independent of
  `target_s`.
- [ ] S4: failing test `test_reduce_action_timeout_independent_of_target` — the dedicated reduce action
  (`:283`/`:287`) timeout is `timeout_safety × reduce_p99` (or `profile_timeout_s` with no measurement),
  independent of `target_s`. (The spec requires a test for ALL FOUR sites — B-05; the INLINE run-job reduce
  timeout is T4.)
- [ ] S5: run → fail.
- [ ] S6: add the two `Constraints` fields; replace each `target` timeout at the four action sites
  (`:274`, `:283`/`:287`, `:295`, `:301`) with the measured-or-`profile_timeout_s` value (the inline run-job
  reduce timeout + the `reduce_wall` source complete in T4). Run → green.

## Task 3: Profile a thread_safe stage in one bounded job

**Files:**
- Modify: `scripts/hf_jobs/executor.py` (`_profile`, `:513`)
- Test: `tests/scripts/hf_jobs/test_executor.py`

**Interfaces:**
- Consumes: `profile_timeout_s` (T2).
- Produces: for a `thread_safe` non-inner stage, `_profile` runs only the `first` sample (`k = 2` units) at
  `W=1`, `inner=1`, SKIPS the `rest`/`wmax` phase, and writes the single `{gen}/_profile.json`.

- [ ] S1: failing test `test_thread_safe_profile_is_one_bounded_job` — profiling a `thread_safe` non-inner
  stage runs `k == 2` units sequentially at `workers == 1` with the 8-thread budget, runs NO `wmax` phase,
  and writes exactly one `_profile.json`; the job timeout is `profile_timeout_s`.
- [ ] S2: run → fail.
- [ ] S3: branch `_profile` on `st.thread_safe and not st.inner_parallel`: profile `sample[:k]` at `W=1`
  only; leave the existing path for other stages unchanged. Run → green.

## Task 4: Profile reduces; inline-reduce timeout; `{gen}/_reduce.json` record

**Files:**
- Modify: `scripts/hf_jobs/executor.py` (`try_reduce`, `:244-259` — record the reduce wall)
- Modify: `scripts/hf_jobs/campaign.py` (`stage_summary`, `:993-1012` — a `reduce_wall` field;
  `committed_history` `:351-358` + `live_history` `:372-381` + the caller `:1045` — carry `reduce_wall` to
  `next_actions`; `next_actions` reduce + run-job timeouts)
- Modify: `scripts/hf_jobs/store.py` (`_is_record`, `:78-82`; note `_is_record` takes a BASENAME, `:161-165`)
- Modify: `tests/scripts/_parity.py` (`VOLATILE`, `:19`)
- Test: `tests/scripts/hf_jobs/test_executor.py`, `tests/scripts/hf_jobs/test_campaign.py`

**Interfaces:**
- Consumes: `profile_timeout_s` (T2); `work_p99` = `plan_stage`'s job-level `wall_p99` (`planner.py:134`/`:152`).
- Produces: `try_reduce` writes `{gen}/_reduce.json` (the reduce wall); `stage_summary` reads it into
  `reduce_wall`, and `committed_history`/`live_history` carry `reduce_wall` into `next_actions` (today they
  build only cost/efficiency/per_unit — B-04); a run job's timeout on a stage with a recorded reduce is
  `timeout_safety × (work_p99 + reduce_p99)`; a first-run (no `reduce_wall`) run job uses
  `max(plan_stage timeout, profile_timeout_s)`; `{gen}/_reduce.json` is VOLATILE and an `_is_record` record.

- [ ] S1: failing test `test_try_reduce_records_reduce_wall` — after a reduce, `{gen}/_reduce.json` holds a
  positive wall; `stage_summary` surfaces it as `reduce_wall`.
- [ ] S2: failing test `test_reduce_wall_reaches_next_actions` — END-TO-END from the STORE record (not a
  hand-built history dict): a `{gen}/_reduce.json` in the store flows through `committed_history`/
  `live_history` so a run job's timeout is `int(timeout_safety × (work_p99 + reduce_p99))`; with no
  `{gen}/_reduce.json` (first run) the run job gets `max(plan_stage timeout, profile_timeout_s)`, never the
  bare 1800 floor (B-04).
- [ ] S3: failing test `test_reduce_json_is_volatile_and_record` — `"_reduce.json"` is in `_parity.VOLATILE`;
  `store._is_record("_reduce.json")` is True (tested with the BASENAME `_is_record` consumes, `:161-165` —
  B-09).
- [ ] S4: run → fail.
- [ ] S5: time the reduce in `try_reduce`, write `{gen}/_reduce.json`; add `reduce_wall` to `stage_summary`
  and carry it through `committed_history`/`live_history`/the `:1045` caller; inflate the run-job timeout;
  add `"_reduce.json"` to `VOLATILE` and `_is_record`. Run → green.

## Task 5: DAG schedule — overlap independent stages

**Files:**
- Modify: `scripts/hf_jobs/campaign.py` (`next_actions` `:233` barrier `:257-258`; `step` `:806` `acts[0]`;
  per-stage resume cursor `~:813`; `_group` grouped `"a-b"` labels `:217-230`; the `--watch` settle rule
  `:1129`)
- Test: `tests/scripts/hf_jobs/test_campaign.py`, `tests/scripts/hf_jobs/_step_world.py`

**Interfaces:**
- Produces: `next_actions` returns an action for EVERY ready stage (dependencies in `needs` met, units not
  running), not only the first in declaration order; `step` dispatches every returned action; resume is
  keyed per stage (handling `_group`'s `"a-b"` grouped labels, `:217-230`), not by one `"-".join(a.stages)`
  cursor; the per-instance step report gains per-stage entries, and the `--watch` settle rule (`:1129`)
  settles only when EVERY stage of the instance is done or failed.

- [ ] S1: failing test `test_two_independent_ready_stages_both_launch` — a plan where `cv` and `fit` both
  need `extract` (reduced) returns actions for BOTH; `step` launches both in one step.
- [ ] S2: failing test `test_unready_stage_not_launched_running_not_relaunched` — a stage whose `needs` are
  unmet is not launched; a running stage is not relaunched.
- [ ] S3: failing test `test_per_stage_resume_records_siblings_independently` — two concurrent stages record
  ERROR/resume state independently (no shared `"-".join` cursor collision), including a grouped `"a-b"` stage
  (B-12).
- [ ] S4: failing test `test_watch_settles_on_all_stages_done` — the updated `--watch` settle rule (`:1129`)
  settles a multi-stage instance only when every stage is done/failed, not when the first reported action is
  (B-07).
- [ ] S5: run → fail.
- [ ] S6: rewrite `next_actions` to walk `needs` edges and return all ready actions; change `step` to iterate
  the returned actions (spend the dispatch budget across them); key resume per stage (grouped labels
  included); update the `--watch` settle rule to the new report shape. (Largest step — guarded by the four
  tests above; A-HFU-PLAN-02.) Run → green.

## Task 6: Per-instance halt cancel

**Files:**
- Modify: `scripts/hf_jobs/campaign.py` (`step` halt path `:807-808`; a new per-instance cancel helper)
- Test: `tests/scripts/hf_jobs/test_campaign.py`

**Interfaces:**
- Consumes: `launcher.jobs_for(api, campaign, commit)` (`launch.py:260`), `launcher.TERMINAL` (`launch.py:32`),
  `api.cancel_job`; job labels `run_key`, `variant` (`launch.py:98-110`); `step`'s per-action dispatch (T5).
- Produces: on a `halt`, `step` cancels the halting INSTANCE's in-flight jobs filtered by `(run_key,
  variant)`, each not in `TERMINAL`; a failed cancel is retried the next stateless step, with the exception
  NARROWED to the cancel-API error and the cancel outcome recorded in the instance report (B-11).

- [ ] S1: failing test `test_halt_cancels_only_this_instance_by_run_key_variant` — an instance halts with a
  concurrent sibling stage in flight; its in-flight jobs are cancelled by `(run_key, variant)`; a SECOND
  instance sharing `job`+`variant` on another tier (different `run_key`) keeps its jobs (NOT cancelled).
- [ ] S2: failing test `test_failed_cancel_retried_next_step` — a cancel that raises leaves the instance
  halted and the job is targeted again on the next step (re-derived from live state).
- [ ] S3: run → fail.
- [ ] S4: add the per-instance cancel in the `step` halt path (filter `jobs_for` by the instance's
  `(run_key, variant)` labels; `api.cancel_job` each non-`TERMINAL`; catch ONLY the cancel-API exception,
  record the outcome in the report, retry next step). Run → green.

## Task 7: Slot-filling with a `budget_floor` and a per-instance fair share

**Files:**
- Modify: `scripts/hf_jobs/planner.py` (`plan_stage` — accept a free-slot share; both branches)
- Modify: `scripts/hf_jobs/campaign.py` (`step` `:778-782` computes the share; `next_actions` threads it)
- Test: `tests/scripts/hf_jobs/test_planner.py`, `tests/scripts/hf_jobs/test_campaign.py`

**Interfaces:**
- Consumes: `budget_floor` (T2); the dispatch budget in `step` (`:780-782`).
- Produces: `plan_stage(..., free_slots: int)` fans `n_jobs` toward `free_slots` to shorten each job's wall,
  never below `budget_floor` work per job, in BOTH branches; `step` computes `n_active_instances` = the
  non-done instances at step start (known from the instance list BEFORE `next_actions`, so no circular
  dependency — B-08) and passes each instance a share `max(1, budget // n_active_instances)`; the share caps
  that instance's TOTAL new jobs this step across ALL its ready stages, not per stage.

- [ ] S1: failing test `test_slot_filling_exceeds_min_jobs_within_share` — given a free-slot share and work
  above `budget_floor`, `n_jobs` exceeds `ceil(work / budget)` but never the share.
- [ ] S2: failing test `test_slot_filling_respects_budget_floor` — no planned job's work is below
  `budget_floor` (no over-shard).
- [ ] S3: failing test `test_two_instances_split_free_slots_fairly` — with two active instances and a budget
  of B, neither instance's plan fans beyond `max(1, B // 2)` via slot-filling.
- [ ] S4: failing test `test_two_ready_stages_share_one_instance_allowance` — an instance with two ready
  stages does not let each stage fan into the full share; the two stages together stay within the instance's
  share (B-08).
- [ ] S5: failing test `test_slot_filling_both_branches` — the rule holds for `inner_parallel` and non-inner.
- [ ] S6: run → fail.
- [ ] S7: compute `n_active_instances` in `step` from the instance list; thread each instance's share from
  `step` through `next_actions` into `plan_stage`; raise `n_jobs` toward the share bounded by `budget_floor`
  in both branches; cap the instance's total new jobs this step across its ready stages at the share. Run →
  green.

## Task 8: Byte-identical sharded reduce

**Files:**
- Modify: `scripts/_stages.py` (a `shardable_reduce` declaration on `Stage`: the `ids` tuple + the empty
  policy); the trainers that declare it ONLY on a plain-concat reduce (`train_ghost_outfield.py:352-358`
  `ids=("game_id","player_id","team_id")`; `win_probability` `:308-312` `ids=("game_id",)`, typed-empty;
  xshot/xcross `_extract_reduce` `:533-539`/`:591-597` `ids=("_group",)`, raise-on-empty).
- Modify: `scripts/hf_jobs/campaign.py` (`next_actions` — a `sub-reduce` action + a final `reduce`; disable
  inline `try_reduce` for a sharded-reduce stage).
- Modify: `scripts/hf_jobs/executor.py` / `scripts/_stages.py` (sub-reduce over a sorted-key range via the
  declared `combine_frames` ids `:670-684`; write `{gen}/_subreduce/{range}.parquet`; final ordered concat).
- Test: `tests/scripts/_parity.py` (a sharded-reduce arm), `tests/scripts/hf_jobs/test_campaign.py`.

**Interfaces:**
- Consumes: the recorded `reduce_wall` + the budget rule (T4); `combine_frames` (ordered `pd.concat`);
  `storable_id_column` (`_stages.py:646-655`).
- Produces: sharding is OPT-IN per stage — only a stage that DECLARES `shardable_reduce` (a plain
  `combine_frames` concat with its `ids` and its empty policy) is sharded. A stage whose reduce is not a
  plain concat (`_cv_reduce`, `_probe_reduce`, `_prep_reduce`) or whose empty handling differs is NOT
  shardable; a heavy such reduce is reported as an irreducible floor under R3, never silently replaced
  (B-03). For a shardable stage with recorded `p99 > target_s − startup_s`, the reduce is range-partitioned
  into `sub-reduce` jobs writing `{gen}/_subreduce/{range}.parquet` (combining via the declared `ids`), then
  one final `reduce` concatenates the partials in sorted range order; the inline `try_reduce` is disabled.

- [ ] S1: failing test `test_sharded_reduce_equals_single_reduce` — a shardable stage whose reduce profiles
  over budget produces, via `≥ 2` sub-reduces + a final concat, a frame byte-identical to the single-reduce
  frame (proves `combine_frames` associativity + `storable_id_column` idempotence + a parquet round-trip).
- [ ] S2: failing test `test_sharded_reduce_empty_range_writes_empty_partial` — the declared empty policy
  applies ONLY at the FINAL combine, not per range: a single empty key range among non-empty ranges writes
  an empty partial and the final frame is byte-identical (NO spurious raise); a genuinely all-empty stage
  raises at the final combine, matching the single reduce (which raises only when EVERY part is empty,
  xshot `:537-539`) (B-16).
- [ ] S3: failing test `test_sharded_reduce_mixed_id_types` — a range mixing int and str ids reduces to the
  same canonical ids as the single reduce (B-03).
- [ ] S4: failing test `test_non_concat_reduce_not_shardable` — a stage without `shardable_reduce` is never
  sharded; a heavy such reduce is reported as an irreducible floor.
- [ ] S5: failing test `test_sharded_reduce_disables_inline_try_reduce` — a sharded-reduce stage does not run
  the inline `try_reduce`.
- [ ] S6: failing test `test_final_concat_over_budget_is_irreducible_floor` — when the final ordered concat
  alone exceeds the budget, it is reported as an irreducible floor (surfaced), not failed.
- [ ] S7: run → fail.
- [ ] S8: add the `shardable_reduce` declaration + the sub-reduce/final-reduce actions + the range partition;
  gate the inline `try_reduce`; add the `_parity.py` sharded-reduce arm with non-vacuity (`≥ 2` sub-reduces).
  Run → green.

## Task 9: Incremental preflight

**Files:**
- Modify: `scripts/hf_jobs/preflight.py` (per-check digest store; `check_ref`/`check_lock`/`check_datasets`/
  `check_plans` input keys; `require_preflight` unchanged in contract `:195`)
- Test: `tests/scripts/hf_jobs/test_preflight.py`

**Interfaces:**
- Produces: a persistent per-check cache (a single file in the preflight report store, keyed by check name
  and input digest, NOT by commit; owned by `preflight.py`; a missing/unreadable cache forces full
  recompute). The CHEAP checks (`check_api`, `check_datasets`, and `check_lock`'s observe) ALWAYS re-observe
  (spec §F); only `check_plans`'s ~20–30-min toy train is cached. `check_plans`'s key is `campaign_sha256` +
  `uv.lock` digest + resolved image digest + `datasets.lock` digest + a tree digest of `scripts/**`,
  `silly_kicks/**` and `pyproject.toml` (B-13). `check_ref` re-observes `git branch -r --contains` + CI
  status (a force-push can un-green).

- [ ] S1: failing test `test_unchanged_tree_reuses_check_plans` — a second preflight with no changed input
  skips the `check_plans` toy train and reuses its stored pass; the cheap checks still re-observe (B-10).
- [ ] S2: failing test `test_library_change_invalidates_check_plans` — a `silly_kicks/**`-only change
  changes the `check_plans` key, so the toy train re-runs.
- [ ] S3: failing test `test_changed_lock_or_image_reruns_check_plans` — a changed `uv.lock` or image digest
  changes the `check_plans` key, so its toy train re-runs; the cheap checks (`check_api`, `check_datasets`,
  `check_lock`'s observe) always re-observe regardless (spec §F); `check_plans` reuses its stored pass ONLY
  when its key is unchanged (B-10, B-18).
- [ ] S4: failing test `test_force_pushed_ref_rechecks` — a not-yet-green or force-pushed `check_ref` always
  re-checks.
- [ ] S5: run → fail.
- [ ] S6: implement the per-check digest store and the four keys; keep `require_preflight`'s contract. Run →
  green.

## Task 10: Re-profile `startup_s` and bootstrap the first run

**Files:**
- Modify: `scripts/hf_jobs/campaign.py` / `scripts/hf_jobs/ledger.py` (bootstrap summary write;
  `planning_constraints` `:362-367` already reads `median_startup_s`)
- Test: `tests/scripts/hf_jobs/test_campaign.py`, `tests/scripts/hf_jobs/test_ledger.py`

**Interfaces:**
- Produces: a bootstrap summary carrying the HF-MEASURED median startup (~15 s: git clone + venv + `uv sync`
  on the `cpu-upgrade` box — NOT a local/DGX startup, B-15), read by `median_startup_s` (`ledger.py:121-134`)
  from the committed summary, so the FIRST planning run uses the real HF startup, not the 300 s
  `Constraints.startup_s` default.

- [ ] S1: failing test `test_planner_uses_committed_median_startup` — with a committed summary present, the
  planner's `startup_s` is the committed median, not 300.
- [ ] S2: failing test `test_bootstrap_summary_supplies_first_run_startup` — the bootstrap summary (carrying
  the HF-measured startup) supplies the median on a run with no prior summary.
- [ ] S3: run → fail.
- [ ] S4: add the bootstrap-summary writer (a committed summary carrying `median_startup_s`); confirm
  `planning_constraints` reads it. Run → green.

## Task 11: Opt in `cv` (ghost HistGB + xgb); wire xgb cv runtime threads; size the study pool to fold width

**Files:**
- Modify: `scripts/train_ghost_outfield.py` (`cv` stage `:510`), `scripts/train_ghost_gk.py` (`cv`),
  `scripts/train_xshot_occurrence.py` (`cv` `:976`; the cv fold fit reads `runtime_n_jobs_from_env()`),
  `scripts/train_xcross_attempt.py` (`cv`; cv fold fit)
- Modify (study per-child RSS data path, B-19): `scripts/hf_jobs/executor.py` (`InnerPool` children `:82`/
  `:572` record their per-child peak RSS into the unit ledger), `scripts/hf_jobs/ledger.py` (`unit_cost`
  `:34-42` surfaces it), `scripts/hf_jobs/planner.py` (`UnitCost` `:55-61` gains
  `inner_child_peak_rss_bytes: int | None = None`; `plan_stage` inner branch `:127-130` sizes the pool AND
  its cap by it),
  `scripts/hf_jobs/campaign.py` (`committed_history` `:351-358` + `live_history` `:372-381` carry the field)
- Create/Modify: `tests/tracking/test_thread_identity.py` (cv arms), `tests/scripts/test_thread_safe_registry.py`
- Test: as above

**Interfaces:**
- Consumes: shipped `thread_safe`/`thread_lever`, `thread_budget`, `runtime_n_jobs_from_env()` (`_thread_pin`;
  the final fit already uses it, `:858`/`:892`), the registry gate; T0 shape decision; T3 profile shape.
- Produces: `cv` declares `thread_safe=True` with its lever (`omp` for ghost HistGB, `n_jobs` for xgb) behind
  a committed identity arm; the xgb `cv` fold fit reads `runtime_n_jobs_from_env()` so the budget actually
  reaches the cv fit; the registry gate is PER-STAGE (each thread_safe stage needs its own identity arm, not
  a one-off string grep the final `fit` already satisfies). The study inner pool is sized by the measured
  per-CHILD RSS (T0), NOT the whole-unit peak (`planner.py:129` over-counts), so study runs fold-width (5)
  when the children fit — spec §B "corrected to the fold width here" (owner ruling 2026-10-29: follow the
  spec, measure + correct — B-14). `inner_width` is already 5 (`:974`); the fix is the SIZING clamp. The
  per-child RSS reaches `plan_stage` through a real data path (B-19): `InnerPool` children record their own
  peak into the unit ledger, `unit_cost` surfaces it, a new `UnitCost.inner_child_peak_rss_bytes` carries it
  through `committed_history`/`live_history` (HF-measured, with `history_key` — NOT `_profile`, which runs
  the first sample at `inner=1` and never spawns the pool; NOT the T0 local value, which `load_history`
  drops). Before any child RSS is recorded (first run), `plan_stage` falls back to the whole-unit-peak clamp
  and SURFACES that study is unsized-pending-measurement. T0's local measurement only DECIDES whether 5 fits;
  if it shows 5 genuinely cannot fit the usable RAM, that is a real RAM floor surfaced to the owner, never
  silently left at 1. `UnitCost.inner_child_peak_rss_bytes` is `int | None = None` (DEFAULTED, so
  `committed_history`'s `UnitCost(**cost)` on older summaries `:358` and the positional `UnitCost`
  constructors in tests do not break — B-20). The inner-branch cap (`planner.py:127`/`:130`) sizes the pool
  children by the CHILD peak, not the over-counted unit peak, so `unit_peak + inner × child_peak ×
  cap_factor` stays within the usable RAM — the cap keeps protecting the box at `inner=5` (B-21).

- [ ] S1: failing identity test `test_cv_thread_identity_ghost` — the ghost `cv` artifact is byte-identical
  across threads `{1,2,4,8}`; the "artifact" is the serialized per-fold result dicts (`_cv_work`,
  `train_ghost_outfield.py:385-405`) AND the reduced `[cv_mae, per_slot, per_provider, per_possession]`
  (`_cv_reduce`, `:408-415`) — NO model files; a precondition test proves the fixture engages every thread.
- [ ] S2: failing identity test `test_cv_thread_identity_xgb` — the xgb `cv` artifact byte-identical across
  threads via the cv fold fit reading `runtime_n_jobs`, with a per-axis non-vacuity precondition.
- [ ] S3: failing test `test_xgb_cv_fit_reads_runtime_n_jobs` — the xgb cv fold fit passes
  `runtime_n_jobs_from_env()` into the fit (not the constant 1); without this, cv stays 1-thread while S2's
  identity passes trivially (B-06).
- [ ] S4: failing registry test `test_registry_is_per_stage` — a `thread_safe` `cv` stage without its OWN
  committed identity arm fails `test_thread_safe_registry` even though the final `fit` has one (the gate is
  per-stage, not a one-off string grep — B-06).
- [ ] S5: failing test `test_study_pool_sizes_to_fold_width_from_store_record` — driven from a committed
  summary carrying `inner_child_peak_rss_bytes` (NOT a hand-built value, B-19): `plan_stage` sizes the study
  inner pool to the fold width (5), clamping by the per-child RSS, not the whole-unit peak. Companion tests:
  with no child field yet (first run), `plan_stage` falls back to the whole-unit-peak clamp and the plan
  marks study unsized-pending-measurement; if the recorded child RSS does NOT fit 5, the plan records a RAM
  floor (surfaced), not a silent 1.
- [ ] S5b: failing test `test_inner_pool_child_records_peak` — an `InnerPool` child writes its per-child
  peak to the unit ledger, and `unit_cost` surfaces it as `UnitCost.inner_child_peak_rss_bytes` (defaulted
  `int | None = None`, so `UnitCost(**cost)` on an older summary and positional constructors still work —
  B-20) carried by `committed_history`/`live_history` (B-19).
- [ ] S5c: failing test `test_child_sized_pool_cap_protects_box` — with the study pool sized to 5 by the
  child RSS, the planned cap keeps `unit_peak + 5 × child_peak × cap_factor` within the usable RAM (the
  inner-branch cap uses the child peak, not the over-counted unit peak — B-21).
- [ ] S6: run → fail.
- [ ] S7: declare `thread_safe`+lever on each `cv` stage; wire the xgb cv fold fit to
  `runtime_n_jobs_from_env()`; make the registry gate per-stage; add the identity arms. For the study pool:
  record each `InnerPool` child's peak into the unit ledger, surface it via `unit_cost` into a new
  `UnitCost.inner_child_peak_rss_bytes` (`int | None = None`, defaulted), carry it through
  `committed_history`/`live_history`, and size the `plan_stage` inner_parallel pool AND its cap by the child
  peak (fallback: whole-unit-peak clamp, surfaced). Run → green.

## Task 12: (W, N, thread) parity at production shape + the audit artifact

**Files:**
- Modify: `tests/scripts/_parity.py` (thread axis at production `W=1`)
- Modify: `scripts/hf_jobs/campaign.py` (`stage_summary` / summary — the per-stage utilization audit field +
  its assertion)
- Test: `tests/scripts/_parity.py`, `tests/scripts/hf_jobs/test_campaign.py`

**Interfaces:**
- Consumes: the thread-safe stages (T11); the DAG schedule (T5) and the per-instance halt cancel (T6) —
  S2's cross-stage overlap arm and the halt arm need both behaviours to exist first (A-HFU-PLAN-01);
  `fanned_jobs` (`_parity.py:152`).
- Produces: a thread-axis parity arm that runs at the PRODUCTION `W=1` (one unit per job at 8 threads),
  proving parallelism by `fanned_jobs ≥ 2` + the identity gate (NOT `ran_concurrently`, which is intra-job
  W≥2); a cross-stage wall-interval-overlap assertion for the DAG with a non-vacuity (red-when-serialized)
  arm and a halt arm (T6); a per-stage utilization audit written to the campaign summary asserting each
  stage saturates 8 cores OR records an irreducible floor.

- [ ] S1: failing parity test `test_thread_axis_parity_at_w1` — `run_serial` (one unit, 1 thread) equals
  `run_fanned` (≥ 2 jobs × 8 threads) for `outputs` and `artifact`, with `fanned_jobs ≥ 2`; run at `W=1`.
- [ ] S2: failing DAG test `test_cross_stage_overlap_asserted_and_red_when_serial` — the overlap assertion is
  True when two stages overlap and goes RED when they run serially.
- [ ] S3: failing audit test `test_audit_artifact_saturates_or_irreducible` — the summary's audit field
  asserts each stage either saturates 8 cores or records an irreducible floor (a study at ~5-fold width
  records its idle cores).
- [ ] S4: run → fail.
- [ ] S5: add the W=1 thread parity arm, the cross-stage overlap assertion + non-vacuity + halt arm, and the
  audit field + assertion. Run → green.

## Task 13: ADR-128 amendment #2 + surface sweep + preflight-reuse ADR

**Files:**
- Modify: `docs/superpowers/adrs/ADR-128-compute-contract-hf-campaigns.md` (amend INV-2 to this cycle's
  rules); `AGENTS.md`; `docs/howto/compute-pipelines.md`; the `compute-pipelines` skill; comments in the
  touched modules.
- Create: a short ADR for the preflight-reuse contract.

- [ ] S1: amend INV-2 to the full-utilization wording (W=1 thread_safe sizing, DAG schedule, timeout from
  measured wall only, committed `startup_s`, slot-filling with `budget_floor`); update every listed surface;
  keep the four default-env `"1"` assertions. Doc/contract only.
- [ ] S2: write the preflight-reuse ADR (the per-check digest store + the four check keys). Doc only.

## Task 14: Release prep (at the commit gate)

- [ ] S1: CHANGELOG entry (DAG schedule, timeout decoupling, slot-filling + `startup_s`, sharded reduce,
  preflight reuse, `cv`/`study` opt-in). Version bump IF a `silly_kicks/` wheel surface changed (decide at
  the gate; `cv` thread wiring touches trainers under `scripts/`, not the wheel — confirm). `TODO.md`.

## Revision r3 tasks (spec § "r3 — the real campaign on `2474b73`", 2026-10-30; A+B APPROVED)

The real campaign on `2474b73` showed that T0 was not executed and that several shipped mechanisms fail at
production scale. These tasks implement the r3 designs D1–D15 in the SAME working tree; the Global Constraints
above apply unchanged (one amend at the cycle gate, no rebaseline, owner gates). Temp files for every run go
under `D:\Temp\hfu` (owner instruction): `TEMP`/`TMP` set there and `--basetemp=D:\Temp\hfu\pytest-<run>`.

Order (the ONE order line for r3 and r4; B r4-spec review R4S-08): R3-1 → {R3-2, R3-3, R3-4} → R3-5 →
{R3-6, R3-7} → R3-12 → R3-14 → {R3-9, R3-10} → R4-1 … R4-10 (below) → R3-8 (Step 0 measures the r4 stage
shapes) → R3-11 → R3-13. The owner approved D2/D11/D13 on 2026-10-30. Step 0 (R3-8) and rung 2 run ON HF as measurement jobs
(the pushed base plus the working-tree diff; no push needed). Rungs 3 and 5 run the campaign controller, which
needs a pushed, CI-green commit, so they run only after the spec's intermediate amend and force-push, each an
explicit owner gate.

### R3-1 (D1): a linear reduce and a `read_result` that never lists — IMPLEMENTED, in review

**Files:** `scripts/_stages.py` (`read_result`, `_read_units`, `_reduce_states`, `_write_reduce_metadata`,
`reduce_stage`, `reduce_subrange`, `reduce_from_partials`, `reduce_done_order`); `tests/scripts/test_reduce_linear.py`.
**Interfaces:** Produces `read_result(store, base, *, names=None)`; `REDUCE_READ_WORKERS = 16`;
`REDUCE_PROGRESS_S = 30.0`; `_read_units(ctx, name, keys, *, label=None) -> dict[str, Any]` (keys order).
- [x] RED: `test_reduce_lists_the_generation_a_constant_number_of_times_and_reads_each_result_once[stage|item_stage]`
  (42 listings for 40 units), `test_reduce_directory_entries_visited_grow_subquadratically[single|sharded]`
  (exponent 1.973), `test_reduce_reports_progress`; the `read_result` gate's RED proven against
  `git show HEAD:scripts/_stages.py` (20 listings for 20 reads, 2 reads per result).
- [x] GREEN: the D1 tests plus `test_sharded_reduce.py`, `test_stage_contract.py`, `test_extract_reduce_mixed_ids.py`,
  `test_combine_frames.py`, `hf_jobs/test_resume_repair.py`: 67 passed.
- [x] RED then GREEN (spec D1 test list; B plan review R3-05): `test_cache_work_lists_a_constant_number_of_times`,
  `test_probe_comparison_lists_a_constant_number_of_times`, `test_census_done_results_lists_a_constant_number_of_times`
  (a counting store under each module-level loop). RED proven with the HEAD `read_result` patched in (41, 41 and
  40 listings for 40 units); GREEN with the fix.
- [ ] The full `tests/scripts` suite and the CI lint trio green; A/B implementation review of R3-1.

### R3-2 (D3, D15): resume into a reduce; a watch that ends

**Files:** `scripts/hf_jobs/campaign.py` (`resume_after_error`, the `--watch` loop in the command entry);
`tests/scripts/hf_jobs/test_campaign.py`.
- [ ] RED: `test_a_timed_out_run_job_with_nothing_pending_resumes_into_its_reduce` (latest stage job ERROR
  `Job timeout`, no pending and no retryable failed unit → the next action is `reduce`, not a halt);
  `test_a_reduce_that_times_out_again_halts_with_a_named_reason` (bounded by `error_resumes`);
  `test_watch_exit_code_is_nonzero_when_an_instance_halted` and `..._zero_when_every_instance_is_done`.
- [x] RED (resume halted with `job j1 ERROR: Job timeout`; `watch_exit_code` missing) then GREEN:
  `resume_after_error` returns `None` when nothing is pending; `watch_exit_code(rep)` decides the settled `--watch`
  exit (`all_settled` already decides WHEN it exits); `test_campaign.py` 79 passed.
- Interim reduce timeout (B plan review PLAN-R3-03): until the owner's D2 decision, the resumed reduce keeps
  `_reduce_timeout(None)` = `profile_timeout_s` (7200 s). This is sound because D1 makes the reduce linear: the
  3,961-unit extract reduce reads 0.36 s per key sequentially (measured), well under 7200 s with 16 readers.
  R3-12 re-runs R3-2's tests after D2 lands.

### R3-3 (D4): fresh controller history

**Files:** `scripts/hf_jobs/campaign.py` (`live_history`, `instance_views`); `tests/scripts/hf_jobs/test_campaign.py`.
- [ ] RED: `test_live_history_rereads_a_stage_whose_ledger_lands_after_its_result` (the first step sees the result
  and `.sha256` but no ledger; the next step gets a cost); `test_live_history_refreshes_a_cached_cost_when_a_new_ledger_lands`.
- [x] RED (`KeyError: 'x'` from the frozen None; a stale `n_samples` 1 after a second ledger) then GREEN: the cache
  maps a generation to `(frozenset(ledger names), cost)` and serves it only while the ledger set is unchanged;
  `None` is never cached; `instance_views` uses the same freshness test. `test_campaign.py` 81 passed.

### R3-4 (D5): a profile that always makes progress; the zero-progress guard

**Files:** `scripts/hf_jobs/executor.py` (`run_stage`, `_profile`); `scripts/hf_jobs/campaign.py` (`step`, the guard);
`tests/scripts/hf_jobs/test_executor.py`, `test_campaign.py`.
- [ ] RED: `test_a_repeat_profile_runs_pending_units_only`; `test_profile_never_overwrites_an_existing_record`;
  `test_zero_progress_guard_halts_after_two_jobs_with_no_new_committed_record`;
  `test_sub_reduce_jobs_never_trip_the_zero_progress_guard` (N sub-reduces plus the final reduce).
- [ ] GREEN: `_profile` samples `mine` minus done units and writes `_profile.json` with `write_if_absent`. The
  controller is stateless (`campaign.py` module docstring, `step`), so the guard's baseline lives in the HF job
  LABELS (B plan review PLAN-R3-01): every launched job carries `records=<mark>`, the stage's PROGRESS MARK
  (`progress_mark_of`: a fingerprint of the committed record names -- unit results, exclusion markers, failure
  records, sub-reduce partials, stage output -- and each failed unit's attempt number) from the step's listing.
  The guard halts the instance with a named reason when the stage's 2 most recent SETTLED completed jobs (HF's
  `finished_at` at least `PROGRESS_SETTLE_S` = 600 s ago) both carry the stage's current mark; jobs inside the
  settle window are ignored, never waited for. Corrected in round 4: the first rule required the 2 NEWEST jobs
  to be settled, so a loop that relaunches every step never halted (B review R4-01); and a record COUNT stood
  still across a retry that failed again or succeeded (B review R4-02), so the measure became the mark. Tests:
  `test_zero_progress_guard_survives_a_controller_restart`, `test_zero_progress_guard_waits_for_late_records`,
  `test_a_fast_no_op_loop_halts_within_the_settle_window_plus_two_jobs` (RED on the first rule),
  `test_a_settled_job_whose_records_landed_is_progress`,
  `test_a_retry_changes_the_progress_mark_whether_it_fails_again_or_succeeds`,
  `test_a_tolerant_stage_whose_retries_fail_again_reduces_instead_of_halting` and
  `test_a_stage_whose_retries_succeed_is_progress_not_a_stall` (both RED under a count, proven).

Status R3-4: [x] RED then GREEN. Executor: `test_a_repeat_profile_runs_pending_units_only` (`assert 4 > 4` RED),
`test_profile_never_overwrites_an_existing_record` (bytes differed RED); `test_executor.py` 30 passed. Guard: the
`records` and `kind` labels (`JobSpec.records`/`JobSpec.kind`), `_committed_records`, `_zero_progress` with
`PROGRESS_SETTLE_S = 600`; the two halt tests RED (`profile` launched) then GREEN; `test_campaign.py` 85 passed.

### R3-5 (D6): survive a lossy mount

Status: [x] (b) sweep, (c) loss report + `kind` label, (d) `UnitsInvalidated` (spec D6(d)), and D1(b) (pooled INV-3
re-hash) each RED then GREEN; (a) guard `test_cost_statistics_exclude_null_measurements` (existing behaviour,
passes); the integration test `tests/scripts/hf_jobs/test_lossy_mount.py` passes (10 % of each run's unit writes
lost BEFORE the reduce; the real controller loop finishes byte-identical to a loss-free run and names the loss).
Its first two runs failed on HARNESS bugs (loss injected after a stage's own inline reduce, the harmless case);
fixed by injecting per stage, before the reduce.

**Files:** `scripts/_stages.py` (`ORPHAN_TEMP_AGE_S`, the sweep in the reduce path); `scripts/hf_jobs/ledger.py`
(null-measurement exclusion); `scripts/hf_jobs/campaign.py` (the loss report); tests.
- [ ] RED: `test_a_lossy_store_completes_every_unit_through_plan_launch_resume` (a store that drops 1 % of writes at
  random; every instance completes and the report names the loss); `test_orphan_sweep_spares_an_in_flight_temp`
  (sweep during another writer's mid-write: both writes complete; an older orphan is removed; a placeholder mtime
  earlier than the generation marker is never swept); `test_cost_statistics_exclude_null_measurements`.
- [ ] GREEN: as designed in spec D6 (a)–(c).

### R3-6 (D7): a fast, correct summary

**Files:** `scripts/hf_jobs/campaign.py` (`cmd_summary`, the job attribution); tests.
- [ ] RED: `test_summary_downloads_scale_with_stages_not_units` (download calls and bytes bounded by record files);
  `test_summary_attributes_jobs_by_run_key_variant_and_stage`.
- [ ] GREEN: `cmd_summary` reads through `StoreSnapshot` with `STEP_WORKERS`; jobs keyed by `(run_key, variant, stage)`.

### R3-7 (D8): measured R1 at the job level

**Files:** `scripts/_stages.py` (`run_unit` records `cpu_s`; the job record written once at the end of `main`, at
the unique per-job path `_jobs/<job label>.<host>.<pid>.json` under the run key); `scripts/hf_jobs/campaign.py`
(`audit_utilization`, `stage_summary`); `scripts/hf_jobs/ledger.py`; `tests/scripts/_parity.py` (the job record is
VOLATILE: it holds timing); `scripts/hf_jobs/store.py` (`_is_record` covers the job record, so the summary reads
it in bulk); a serial-reason registry gate; tests. Every ledger-schema assertion is updated for the new field
(B plan review PLAN-R3-02).
- [ ] RED: `test_a_w8_job_of_single_thread_units_audits_saturated`; `test_a_lone_single_thread_unit_audits_idle`;
  `test_a_missing_job_record_audits_unmeasured`; `test_a_bundled_job_is_judged_once_and_shown_on_each_stage`;
  `test_every_serial_stage_declares_a_reviewed_reason`.
- [ ] Also RED: `test_job_record_is_volatile_for_parity` and `test_job_record_is_read_in_bulk_by_the_summary`.
- [ ] GREEN: the job record holds `cpu_s` (`getrusage` self + reaped children), `wall_s` from process start
  (`/proc/self/stat` start time on Linux), the stages run and the job label; the summary joins HF `started_at`
  and reports the prelude separately; `audit_utilization` judges measured job utilization; floors only as spec D8.

Status R3-6: [x] RED (84 download calls for 2x40 units; 4 jobs counted for a 3-job variant) then GREEN;
`test_campaign.py` 89 passed.

Status R3-7: [x] RED then GREEN. `write_job_record` (end of the stage CLI when `SILLY_KICKS_JOB_LABEL` is set;
`os.times` self + reaped children; wall from `/proc/self/stat`), unit `cpu_s` (self delta + live `InnerPool.child_pids`
via `/proc`), `.job.json` VOLATILE and `_is_record`, `stage_measured_cores`, measured `audit_utilization`
(`SATURATION_FRACTION = 0.75`, PROVISIONAL until R3-8 S0 calibrates it on HF; lone unit a floor only at the full
thread budget; `serial_reason` read from the plan stage). Round 4 (A review R4-IMPL-01): the `serial_reason` Stage
field (both stage types; blank rejected; in the plan JSON) and its EXACT registry gate
(`scripts/hf_jobs/serial_reasons.py`, empty until the R3-8 table adds reviewed entries;
`tests/scripts/test_serial_reasons.py` with an anti-rot test) now land with D8, as the spec places them, not in
R3-8. Also (B review R4-11): a unit's `cpu_s` subtracts the pre-unit CPU of an inner-pool child reaped during the
unit (`_unit_cpu_s`). `test_campaign.py` 93 passed; `test_executor.py` 36 passed.

### R3-8 (D9): Step 0 executed for real; every stage shaped by measurement

- [ ] S0 (B review R4-04, A review R4-IMPL-02): calibrate `SATURATION_FRACTION` (the R1 pass bar, now a
  provisional 0.75) from the S1 HF measurements of the stages that are saturated by construction (8 concurrent
  single-thread units; a full-budget HistGB fit). The chosen value and its evidence go to the OWNER before the
  rung-5 canary campaign, since it sets how strict R1 is.
  - OWNER RULING 2026-10-30: `SATURATION_FRACTION = 0.85`. Evidence (S1 round 2, D8 job records): the 7 stages
    saturated by construction measured 0.883, 0.937 (ghost_gk extract), 0.910 (ghost_outfield extract), 0.891
    (xshot extract), 0.889 (xcross extract), 0.912, 0.919 (ghost_gk importance): minimum 0.883, median 0.910. The
    single-unit `fit` STAGES (0.22-0.36) are not calibration points: their unit is the HistGB fit plus a serial
    leaf walk in the same process (S2 review S2B-01; S1g below), not an 8-thread fit alone. The HistGB fit itself
    ran at 7.33 cores on HF (spec r3 D16). 0.85 separates a full W=8 job (0.88-0.94) from a W=7 job (about 0.83).
    The constant and its test change in S3 (`scripts/hf_jobs/campaign.py` is frozen during the r4 impl review
    round).
  - Repeat measurements (S2 review S2B-C04; the same stages, other read-only HF jobs, 2026-10-30/31): ghost_gk
    `extract` 0.933 and 0.896 (the S1g jobs' upstream records); ghost_outfield `extract` 0.939 (job
    `6ae565e80e0338d1bb7d0af6`); xshot `extract` 0.892 and 0.920, xcross `extract`
    0.893 and 0.923 (S1e2 jobs `6ae565f10e0338d1bb7d0af8`, `6ae565f30e0338d1bb7d0afa`). Across two to three runs
    each, the W=8 stages measured 0.883-0.937; the lowest is 0.033 above the bar. The rung-5 audit reads one run;
    a W=8 stage under 0.85 there is a finding to report, not a flake to absorb.
- [ ] S0b (B review R4-12): the measurement-job launcher moves from `D:\Temp` into the repo
  (`scripts/hf_jobs/measure.py`) with tests: the pushed base commit plus the working-tree diff, its sha256
  verified in the job before `git apply`; the campaign bucket mounted READ-ONLY. A measurement job that must
  write (rung 2, rung 3) writes only to a separate canary bucket, `silly-kicks/hf-refit-canary` (private;
  created 2026-10-30 by owner ruling, "You create it"; no decision about it remains open), so it cannot reach
  the campaign bucket by mechanism, not by convention.
  - [x] Done 2026-10-30. `scripts/hf_jobs/measure.py` (`python -m scripts.hf_jobs measure`): `worktree_patch`
    (the binary diff from the base to the working tree, untracked files included, built in a temporary index);
    `patch_env` (gzip + base64 in environment chunks of at most 96 KiB, because Linux caps one `bash -c` argument at
    128 KiB; the full patch is 328 KB encoded); `patch_lines` (rebuild, `sha256sum` check under `set -e`, then
    `git apply --binary`); `measure_volumes` (every mount read-only, asserted where built); `launch_measurement`
    (no `secrets`). `tests/scripts/hf_jobs/test_measure.py` 9 passed, including the job's own shell lines run in
    bash on a clone (a wrong sha256 stops before `git apply`). Ruling: the module came before its tests; each
    test was then shown to bite by mutation (no sha check: the tamper test fails; a read-write mount: 2 fail; a
    secret: 1 fails), module restored byte-identical. Cost if wrong: none found; the mutations cover the three
    invariants.
- [ ] S1 (HF, no push; owner ruling 2026-10-30: HF is the primary platform, so every number that chooses a shape
  or a value is measured on HF hardware, never on the DGX): T0's per-stage measurements for every stage of every
  campaign module, at production shape (the campaign image, `cpu-upgrade` flavor and bucket mount). Code not yet
  pushed runs as a measurement job: it checks out the pushed base `2474b73` and applies the working-tree patch,
  carried INLINE in its command (its sha256 verified before `git apply`; spec r4.4). A read-only measurement mounts
  the campaign bucket read-only and writes nothing outside the container. A measurement that must write writes
  only to the separate canary bucket (S0b; `silly-kicks/hf-refit-canary`, created 2026-10-30 by owner ruling);
  it never writes to a campaign store (B round-5 review R5-02). So no measurement job can produce an artifact. Per stage: part count, one-thread and eight-thread part
  walls, peak RSS, measured cores (`cpu_s`, D8). `ghost_gk` `cache` and `xcross` `probe` are re-measured with D1
  (spec D9: probably listing-bound). The DGX is for debugging only (aarch64, local disk; T0 B-15; B plan review
  PLAN-R3-04).
  - [x] Done 2026-10-30 (probe `D:\Temp\hfu\r38\step0_probe.py`, one read-only job per module and variant; the
    datasets read-only; every stage at production shape into container-local stores). Round 1 (jobs
    `6ae53da2…`–`6ae53db4…`, `6ae539c8…`) measured the units; round 2 (jobs `6ae53fcb…`–`6ae53fdd…`) added the D8
    job record of every stage process (`SILLY_KICKS_JOB_LABEL` set: the CPU of the process and all reaped children
    over its wall from the process start, the measure the R1 audit uses) and a one-unit run at eight threads for
    every multi-unit `thread_safe` stage. Ruling: the probe's "one_thread" run is NOT a one-thread wall (the
    executor sizes the budget from the workers it launches, one for one unit, so it ran at 8); S1d measures the
    true one-thread and two-thread walls. Cost if wrong: none; the numbers are labelled by their real budget.
    Ruling: the round-1/round-2 xShot/xCross runs predate the R4-10a fix, so their `prep`, `study`, `cv` and `fit`
    ran without the 7 DFL matches (8.4 % / 7.6 % fewer rows on sc_extended). Cost if wrong: the study walls are
    about 8 % low; no shape decision turns on that margin. Job-level utilization (`cpu_s / (wall_s x 8)`):

    | Stage (module) | Units, W x threads | Unit wall p50 / max | Job wall | Utilization |
    |---|---|---|---|---|
    | extract (ghost_gk) | 179, 8 x 1 | 16.7 / 75.3 s | 494 s | 0.883 (0.937 for `full`) |
    | extract (ghost_outfield) | 179, 8 x 1 | 34.2 / 82.8 s | 827 s | 0.910 |
    | extract (xshot / xcross) | 115, 8 x 1 | 10.9 / 21.8 s; 10.5 / 23.8 s | 176 / 173 s | 0.891 / 0.889 |
    | importance (ghost_gk) | 27, 8 x 1 | 92.7 / 117.1 s | 344 s | 0.912 (0.919) |
    | extract (receiver) | 30, 8 x 1 | 14.7 / 20.6 s | 64 s | 0.784 |
    | extract (win_probability / match_outcome) | 3,961, 8 x 1 | 0.4 / 4.8 s | 249 / 244 s | 0.655 / 0.579 |
    | cv (ghost_gk / ghost_outfield) | 5, 5 x 1 | 128.6 / 280.1 s | 149 / 292 s | 0.564 / 0.602 |
    | cv, ONE unit at 8 threads (ghost_gk / ghost_outfield) | 1, 1 x 8 | 70.4 / 118.1 s | 75 / 121 s | 0.318 / 0.527 |
    | fit (ghost_gk / ghost_outfield) | 1, 1 x 8 | 74.5 / 512.5 s | 79 / 516 s | 0.356 / 0.219 |
    | study (xshot / xcross, inner pool 5) | 1, inner 5 | 1212 / 519 s | 1218 / 524 s | 0.593 / 0.535 |
    | cv (xshot / xcross) | 5, 5 x 1 | 25.6 / 12.9 s | 31 / 21 s | 0.542 / 0.446 |
    | cv, ONE unit at 8 threads (xshot / xcross) | 1, 1 x 8 | 21.9 / 11.2 s | 25 / 15 s | 0.120 / 0.117 |
    | fit (xshot / xcross) | 1, 1 x 8 | 25.8 / 15.5 s | 29 / 20 s | 0.121 / 0.119 |
    | assemble (xcross) | 1, 1 x 1 | 755.9 s | 761 s | 0.125 |
    | fit (win_probability) | 1, 1 x 1 | 1097.5 s | 1104 s | 0.125 |
    | every other stage (validate, select, cache, prep, assemble, refs, prefit, report, pool, deploy, probe, probe_comparison, the small cv/fit/study, the census item stages, pass_completion / xsuccess extract) | 1-179 | under 25 s each | 4-41 s | 0.02-0.53 |

    The probe passed no `--peak-rss-bytes` (`step0_probe.py`, `s1e_probe.py`), so every S1 shape ran WITHOUT the
    production RSS clamps: no `size_workers` clamp in the executor and no per-worker cap (S2 review S2B-04). The
    ghost_gk `extract` at W=8 and the `win_probability` `extract` at W=12/16 are therefore shapes the production
    executor would not run today; S3 items 5 and 6 change the clamps.
    Findings for S2: (1) W=8 stages of long single-thread units saturate (0.88-0.94); (2) a single-unit HistGB
    `fit` STAGE at an 8-thread budget uses 1.75-3.05 of 8 cores, because the unit also runs a serial leaf walk
    (S1g: ghost_gk `fit` 76.5 s, of which the walk is 49.7 s at one core and the HistGB fits 23.3 s); (3) the xgb
    `n_jobs` lever is dead in production: `thread_safe_env("n_jobs", b)`
    keeps `OMP_NUM_THREADS=1`, and XGBoost caps `nthread` at `omp_get_max_threads()`, so the xShot/xCross `cv`
    and `fit` run on ONE core at an 8-thread budget (measured 1.00 core); the model-level identity arm passes
    `n_jobs` in a test process where OpenMP is not capped, so it never saw production's cap; (4) a 5-fold
    stage at W=5 uses 5 of 8 cores; (5) the many-short-unit extracts lose about 29 % of each unit to I/O wait
    (0.32 s CPU of 0.45 s wall).
- [x] S1b (HF, same measurement jobs): the per-file read cost through the mount and the `win_probability`
  `extract` reduce at real scale (spec Step 0).
  - [x] Done 2026-10-30 (job `6ae53aa4c78863f9c62ede70`, generation `win_probability/extract/1d6a7d8a77ce8356`,
    3,961 units of 0.038 MB): listing 0.7 s; one file read serially 0.183 s mean (fetch 0.173 s, verify 0.000 s,
    decode 0.009 s); 200 files through an 8-thread pool in 8.6 s. The reduce at real scale: pooled read 154.5 s,
    combine 1.1 s, encode 1.8 s (71.7 MB, 7,974,436 rows), 157.3 s in all. S1c (job `6ae53f15c78863f9c62ee0e5`,
    four disjoint cold slices of 990 files): 41.2 / 42.3 / 42.6 / 41.1 ms per file at pool widths 8 / 16 / 32 /
    64. The mount serves about 24 small files per second whatever the width, so `REDUCE_READ_WORKERS = 16` neither
    helps nor hurts; the reduce of a many-unit stage costs about 42 ms per unit file.
- [ ] S2: a per-stage decision table for every row the campaign audit failed (65) or passed only as a floor (9):
  spec §B shape, inner pool, split, or a declared serial reason. The table is added to this plan BEFORE any opt-in
  and goes to a partial A/B review.
  - Supplementary HF measurements (read-only jobs; probes in `D:\Temp\hfu\r38\`):
    - S1d (`s1d_probe.py`, patches `scripts/_thread_pin.py` inside its own container only: `S1D_TB` forces the
      per-worker thread budget; `S1D_NJOBS_OMP=1` makes the `n_jobs` lever raise `OMP_NUM_THREADS` too). ghost_gk
      `cv` (job `6ae54a000e0338d1bb7d039b`): one fold at 1 / 2 / 8 threads 124.4 / 94.9 / 70.4 s; all 5 folds at
      W=5 x 2 threads 111.3 s per fold, job 118 s, utilization 0.765 (W=5 x 1: 149 s, 0.564). xcross (job
      `6ae54a070e0338d1bb7d03a2`, lever fixed): one `cv` fold at 8 threads 5.5 s at 4.87 cores (11.2 s at 1.00 core
      with the dead lever); `fit` at 8 threads 5.5 s at 5.85 cores (15.5 s at 1.00 core); `cv` at W=5 x 2 threads
      12.1 s per fold, 7.18 cores over the unit span, job 17.9 s (0.674: start-up is a third of an 18 s job).
      ghost_outfield `cv` (job `6ae54a020e0338d1bb7d039f`): one fold at 1 / 2 threads 245.0 / 170.1 s; W=5 x 2
      threads 208.2 s per fold, job 214.7 s, utilization 0.869 (W=5 x 1: 292 s, 0.602). xshot (job
      `6ae54a05c78863f9c62ee6c0`, lever fixed): one `cv` fold at 8 / 2 threads 7.4 / 13.7 s (21.9 s on one core
      with the dead lever); `fit` at 8 threads 7.4 s at 5.84 cores (25.8 s at 1.00 core); `cv` at W=5 x 2 threads
      16.6 s per fold, job 21.7 s (0.753, start-up bound); its `study` (upstream, inner pool 5, one thread per
      child) 1,046 s.
    - S1e (`s1e_probe.py`, a worker-count sweep): `win_probability` `extract` (3,961 units) W=8 / 12 / 16:
      utilization 0.686 / 0.946 / 0.946, job wall 327 / 263 / 228 s (job `6ae54a820e0338d1bb7d03e1`);
      `match_outcome_dependence` `extract` 0.713 / 0.965 / 0.963, 326 / 263 / 229 s (`6ae54a84c78863f9c62ee71c`);
      ghost_gk `select` W=8 / 16: 0.443 / 0.609 but 9.4 / 11.7 s (`6ae54a86c78863f9c62ee71e`); `pass_completion`
      `extract` W=8 / 16: 0.465 / 0.661 but 13.3 / 14.8 s and 58 % more CPU (`6ae54a880e0338d1bb7d03e3`). A unit
      that waits on I/O for part of its wall (0.32 s CPU of 0.45 s) saturates the box only above W=8; a stage of
      sub-second units gains utilization from extra workers only by spending CPU on their start-up.
    - S1f (`s1f_probe.py`, cProfile of one long unit in-process, with the stage CLI's own provenance so the
      tokens match; the first pair of jobs stopped on a probe defect, fixed). `win_probability` `fit` (job
      `6ae550c7c78863f9c62eea74`; 2,294 s under the profiler, 1,098 s without): the 5-fold OOF calibration takes
      2,236 s, of which the 5 `compute_win_probability` calls take 2,060 s, in two per-row `iterrows` loops over
      about 16 M rows (`win_probability/_state.py` `derive_match_states` 1,009 s, `_compute.py` `_score_states`
      1,024 s; pandas builds one Series per row, 965 s); the 6 GLM fits take 192 s. xcross `assemble` (job
      `6ae550c90e0338d1bb7d0667`, with the R4-10a rows; 995 s): `gk_substitution_probe` 456 s, of which
      `_model_eval._eligible_groups` takes 390 s (a per-frame `groupby` walk with 1.64 M `DataFrame.__getitem__`
      calls); `permutation_importance_report` 443 s (805 xgb predictions at 0.46 s each on one thread, 378 s); 15
      xgb fits about 131 s.
    - S1g (S2 review S2B-01; `s1g_probe.py`: the S1f profile with the unit under an OpenMP limit of n threads,
      BLAS at 1, and every upstream stage run with its D8 job record). ghost_gk `cv` fold 0 at 2 threads (job
      `6ae565ebc78863f9c62ef1c9`): 101.3 s wall, 136.5 s CPU. `_walk_rows` (the serial numpy leaf walk) takes
      60.8 s: 44.6 s for `fit`'s `_training_leaves` over every training row and 16.8 s for `predict_mean` over the
      held-out fold. The two HistGB fits take 36.8 s. ghost_gk `fit` at 8 threads (job
      `6ae565e9c78863f9c62ef1c7`): 76.5 s wall, 233.5 s CPU (3.05 cores); the walk 49.7 s (mostly
      `_training_leaves`), the HistGB fits 23.3 s, the parity gate, save and round trip the rest. Finding: the
      trainers never read `_training_leaves`. `_cv_work` scores only `predict_mean`; `_fit_work` runs the parity
      gate (`predict_mean`), `save` and `load`; `save` never persists the training arrays ("STRUCTURALLY never
      persisted", `GhostGkModel.save`). Only `predict_density` on a locally fit model reads them. So about 45 s of
      each ghost_gk `cv` fold and of each final fit is a walk whose result no trainer uses, and the (rows x 500)
      leaf matrix is held in memory for the whole unit.
    - S1h (S2 review S2B-08; `s1h_probe.py` patches the executor's claim order inside its own container only):
      receiver_model `extract` (30 units, W=8, job `6ae565eec78863f9c62ef1cb`), sorted / longest-first / sorted /
      longest-first: utilization 0.843 / 0.850 / 0.838 / 0.830, job wall 69.4 / 70.7 / 66.2 / 75.1 s.
      Longest-first claim order gives no gain over the run-to-run spread; S3 item 11 (longest-first) is withdrawn
      and row 29 is decided by the S1i worker sweep.
    - S1i (`s1e_probe.py`): receiver_model `extract` W=8 / 10 / 12 / 16, twice each (job
      `6ae5694b0e0338d1bb7d0bd0`). Utilization 0.844 / 0.868 / 0.870 / 0.900, then 0.845 / 0.855 / 0.854 / 0.902;
      job wall 52.7 / 54.4 / 56.4 / 56.7 s, then 54.1 / 85.2 / 88.4 / 74.5 s; job CPU 356 / 378 / 393 / 409 s, then
      366 / 583 / 604 / 538 s. W=8 has the shortest wall in both runs. More workers raise utilization only by
      spending more CPU on the same work (up to 65 % more), so W>8 is rejected for this CPU-bound stage. At W=8 the
      units' span is 47-48 s at 7.26-7.33 cores (0.90 of the box); 5.7-6.1 s of the 53-54 s job is outside the
      unit span (process start-up and the reduce), and that is most of the gap to 0.85.
    - S1e2 (C03; `s1e_probe.py`, job `6ae565efc78863f9c62ef1cd`): `win_probability` `extract` W=8 / 12 / 16:
      utilization 0.617 / 0.926 / 0.891, job wall 352 / 212 / 336 s, job CPU 1,739 / 1,569 / 2,392 s. HF evicted
      the job during its fourth run (container ephemeral storage over the 50 GB limit; the probe keeps every store
      copy in one container); the three runs before it are complete. With S1e: W=12 measured 0.946 and 0.926 and
      W=16 0.946 and 0.891; W=16 also spent up to 52 % more CPU and its wall moved from 228 to 336 s. W=12 (the
      formula's value for a 0.71 CPU share) is the shape: above the bar in both runs, with the lower CPU.
  - OWNER RULINGS 2026-10-30 (asked directly after the S2 A/B reports; recorded in spec §A, §B and D8):
    - WIO RATIFIED: a third stage shape beside spec §B's two (S2-PLAN-01).
    - The generalized FLOOR RATIFIED (S2-PLAN-01, S2B-02 item 1). Every configuration-based floor also needs
      MEASURED use (S2B-02 item 2): see FLOOR below.
    - NUMPY rewrites THIS CYCLE: `win_probability` `derive_match_states` / `_score_states` and xCross
      `_model_eval._eligible_groups` (S2-PLAN-04).
    - "Too small to pay" RATIFIED (S2B-03): a stage whose summed unit CPU (the unit ledger records' `cpu_s`,
      without the process start-up) plus its reduce CPU (S2B-r2-04; S3 item 3) is under 60 s takes a reviewed serial reason of that class, with its measured
      time. Over 60 s, a stage must be shaped (split, threads, or a structural floor).
    - The kind (c) reference variant RATIFIED (S2B-07): where no stored `2474b73` unit can exist, the old bodies
      and the new code are compared byte for byte at full scale in one HF job on the same inputs (the D16 canary
      method), plus the verbatim oracle tests.
  - OWNER RULINGS 2026-10-31 (asked directly with the S1g/S1h/S1i numbers; S2B-01, S2B-08):
    - Rows 5 and 6: make `GhostGkModel._training_leaves` lazy (DEAD, S3 item 14).
    - Row 11: `ghost_rearguard_coherence` over arrays, and the coherence predict as its own stage of row-chunk
      units (S3 item 15).
    - Row 29: measure a per-(match, period) split on HF first; adopt only at >= 0.85 with a byte-identical
      reduce, else the row returns to the owner (S3 item 16).
  - OWNER RULINGS 2026-10-31, second set (asked directly after the S2 r2 reviews; recorded in spec §A):
    - The floor bar (S2B-r2-03, "half the budget"): one measure, the D8 job record; a floor passes only when its
      cores per worker are more than half its thread budget. Shown to the owner with the row-11 example (1.75 cores
      at budget 8 passes "> workers" and fails this bar). The boundary is STRICT by a separate owner ruling the same
      day (S2B-r3-01, A S2-PLAN-07): the owner's first answer read ">= budget / 2"; asked again with the reason (at
      budget 2 a dead lever measures exactly 1.0 core per worker, which ">=" would pass), the owner ruled "more
      than half".
    - Inner-pool divisor (S2B-r3-03): for an inner-pool stage the cores are divided by `workers x inner_workers`
      (the processes that share the budget), so a dead lever in the 5 study children (4.7 job cores / 5 = 0.94,
      under the half of 2) never audits as a floor.
    - The `ceil(vcpu / n)` budget covers inner pools of n <= 5 children (S2B-r2-02, "Yes, with proof"): S3 items
      0 (c) and 9.
  - Decision vocabulary. SAT: the stage saturates as it is. WIO: an item stage whose units wait on I/O runs
    `W = ceil(vcpu / cpu_share)` workers (`cpu_share` = the measured unit CPU over unit wall), above the vCPU, capped
    by RAM. THREADS: a `thread_safe` stage with n < vcpu units runs all n at once in ONE job with
    `ceil(vcpu / n)` threads each (today the planner gives it W=1 x 8 threads in n jobs, `planner.py:215`); only
    for n <= 5, the measured shapes (S2B-06); n = 6 or 7 keeps today's rule. LEVER: the `n_jobs` lever also raises
    `OMP_NUM_THREADS` to the budget, so XGBoost uses it (XGBoost caps `nthread` at `omp_get_max_threads()`). MEM:
    W=8 kept by a memory admission gate against measured resident memory (S3 item 6) instead of the planner's
    `1.3 x max peak` worker cap. SPLIT: new units, each byte-identical to the serial result. NUMPY: a per-row or
    per-group pandas loop rewritten over arrays by the D20 method (no-rebaseline kind (c): the old body verbatim as
    the test oracle, then the HF identity run, or the owner-ruled reference variant). DEAD: work whose result no
    caller reads, removed (S1g `_training_leaves`). FLOOR: a spec §A structural floor (a single final fit at the
    full thread budget; a sequential study at its fold width; a stage whose every unit runs at once with the whole
    thread budget, `W >= n` and `W x threads >= vcpu`). Each floor also needs measured use when its budget is above
    one thread (S2B-02 item 2; owner ruling 2026-10-31 on S2B-r2-03): ONE measure, the stage's D8 job record, whose
    cores per process (`cpu_s / wall_s / (workers x inner_workers)`) must be MORE than half the thread budget (the
    strict boundary is an owner ruling, 2026-10-31). A dead lever (one core
    per worker at a budget of 2 or 8) and a mostly-serial fit (row 11 as measured in S1: 1.75 cores at budget 8)
    never audit as a floor. SERIAL: a reviewed
    `serial_reason` (`scripts/hf_jobs/serial_reasons.py`), either the measured cause why the stage cannot use the
    box or the "too small to pay" class with its measured time. The exact texts are in the annex after the table.

    | # | Module: stage (`2474b73` audit rows) | r4 shape | HF measure | Decision | S3 change and its gate |
    |---|---|---|---|---|---|
    | 1 | ghost_gk: validate (5 FAIL) | 1 unit | 19-22 s wall, 1 s CPU: the serial footer loop (`_corpus_visibility.py:32-35`), not the read cap | SERIAL (too small to pay) | annex text |
    | 2 | ghost_gk: extract (5 FAIL: 3 cores, 21.6 min) | 179 units | W=8: 0.883 / 0.937, 494 s (repeats 0.933, 0.896); unit peak RSS p50 2.08, p95 4.32, max 5.51 GB (52 units above 4 GB); measured without the production RSS clamps | SAT + MEM | S3 item 6: the planner's `1.3 x max` cap and the executor's `size_workers` clamp give W=4; the admission gate keeps W=8 safe |
    | 3 | ghost_gk: select (new) | 179 units of 0.0-2.1 s | W=8: 0.443, 9.4 s; W=16: 0.609, 11.7 s | SERIAL (too small to pay) | annex text |
    | 4 | ghost_gk: cache (5 FAIL: 1 core, 28.5 min) | 1 unit | 3.6 s (r3 D1 fixed the listing) | SERIAL (too small to pay) | annex text |
    | 5 | ghost_gk: cv | 5 folds, omp | as S1d above; S1g: a fold at 2 threads 101.3 s, of which the dead `_training_leaves` walk is 44.6 s and the held-out `predict_mean` 16.8 s; fold peak RSS 5.12 GB | DEAD + THREADS + FLOOR | S3 item 14 (owner ruling 2026-10-31) removes the dead walk and the leaf matrix; then items 2-3; identity: `test_cv_thread_identity_ghost_gk` covers {1,2,4,8} |
    | 6 | ghost_gk: fit (5 SAT, configuration-based, in the `2474b73` audit) | 1 unit, omp x 8 | 74.5 s at 2.85 cores (0.356); S1g: 76.5 s at 3.05 cores, the walk 49.7 s (mostly the dead `_training_leaves`), the HistGB fits 23.3 s | DEAD + FLOOR | S3 item 14; the floor with measured use (item 3) |
    | 7 | ghost_gk: importance (5 FLOOR + SAT in the `2474b73` audit) | 27 column units (r3 D10) | W=8: 0.912 / 0.919 | SAT | none |
    | 8 | ghost_gk: assemble (5 FAIL) | 1 unit | 2.7 s | SERIAL (too small to pay) | annex text |
    | 9 | ghost_outfield: extract | 179 units | W=8: 0.910 (repeat 0.939); peak RSS max 2.52 GB | SAT | none |
    | 10 | ghost_outfield: cv | 5 folds, omp | W=5 x 2: 214.7 s, 0.869 (W=5 x 1: 292 s, 0.602); one fold x 8: 118.1 s; fold peak RSS 4.56-4.71 GB | THREADS (saturates) | as row 5 (`test_cv_thread_identity_ghost`); the RAM clamp fits W=5 (item 2) |
    | 11 | ghost_outfield: fit (2 FLOOR + SAT in the `2474b73` audit) | 1 unit, omp x 8 | 512.5 s at 1.75 cores (0.219); S1g (job `6ae565e80e0338d1bb7d0af6`, profiled 962.9 s): the coherence `predict_mean` over every training row is a serial walk of 259 s; `ghost_rearguard_coherence` 416 s profiled (a pandas loop over 1,042,736 groups); the HistGB fits 280 s at one thread in the probe (the probe's OpenMP limit missed the lazily imported sklearn). From the production CPU (904.7 s): HistGB about 65 s at about 7 cores, serial about 445 s | NUMPY + SPLIT + FLOOR | S3 item 15 (owner ruling 2026-10-31): `ghost_rearguard_coherence` over arrays (kind (c)); the coherence predict as its own stage of row-chunk units; the fit unit then is the HistGB fit, a floor with measured use |
    | 12 | ghost_outfield: assemble (2 FAIL) | 1 unit | 3.5 s | SERIAL (too small to pay) | annex text |
    | 13 | xshot: extract | 115 units | W=8: 0.891 (repeats 0.892, 0.920) | SAT | none |
    | 14 | xshot: prep (4 FAIL) | 1 unit | 5.8 s | SERIAL (too small to pay) | annex text |
    | 15 | xshot: study (4 FLOOR) | inner pool 5 (one trial's folds) | 1212 s at 4.74 cores (0.593) | FLOOR + LEVER in the children | S3 item 9: each inner child fits xgb with `ceil(vcpu / 5)` = 2 threads (S1d: a fold at 2 threads 13.7 s vs 21.9 s at 1); identity: the study's best params and trial values byte-identical across child budgets {1, 2} in the production env; the floor needs measured use (item 3) |
    | 16 | xshot: cv | 5 folds, n_jobs | W=5 x 1: 31 s, 0.542; one fold "x 8": 1.00 core (dead lever); 128 s of unit CPU | LEVER + THREADS + FLOOR | S3 items 1-3: the LEVER test in the PRODUCTION env, the whole-stage identity across {1,2,4,8} in a subprocess, the floor with measured use |
    | 17 | xshot: select (4 FAIL) | 1 unit | 0.9 s | SERIAL (too small to pay) | annex text |
    | 18 | xshot: fit | 1 unit, n_jobs x 8 | 25.8 s at 1.00 core (dead lever); 26 s of unit CPU; with the lever fixed the unit runs 7.4 s at 5.84 cores (unit span), but the job record of the same shape on xCross is 3.77 cores (start-up is half of a 9 s job) | LEVER + SERIAL (too small to pay) | the LEVER fix (item 1) applies to it; the audit passes it by the class (annex text), because the floor bar (job record, more than 4.0 at budget 8) cannot hold for a start-up-bound job (S2B-r3-02) |
    | 19 | xshot: assemble (4 FAIL) | 1 unit | 1.2 s | SERIAL (too small to pay) | annex text |
    | 20 | xcross: extract | 115 units | W=8: 0.889 (repeats 0.893, 0.923) | SAT | none |
    | 21 | xcross: probe (4 FAIL: 1 core, 10.9 min) | 2 units, one probe match each | W=2: 41 s, 1.51 cores (r3 D1 fixed the listing); 45 s of unit CPU | SERIAL (too small to pay) | annex text |
    | 22 | xcross: probe_comparison (4 FAIL) | 1 unit | 4.0 s | SERIAL (too small to pay) | annex text |
    | 23 | xcross: prep (4 FAIL) | 1 unit | 4.1 s | SERIAL (too small to pay) | annex text |
    | 24 | xcross: study (4 FLOOR) | inner pool 5 | 518 s at 4.28 cores (0.535) | FLOOR + LEVER in the children | as row 15 |
    | 25 | xcross: cv | 5 folds, n_jobs | as S1d above; 67 s of unit CPU | LEVER + THREADS + FLOOR | as row 16 |
    | 26 | xcross: select (4 FAIL) | 1 unit | 0.7 s | SERIAL (too small to pay) | annex text |
    | 27 | xcross: fit | 1 unit, n_jobs x 8 | as S1d above; 15 s of unit CPU; job record with the lever fixed 35.2 s CPU in 9.3 s, 3.77 cores | LEVER + SERIAL (too small to pay) | as row 18 |
    | 28 | xcross: assemble (4 FAIL: 886 s) | 1 unit: the TF-19 block (GK-block ablation CV; CV-held-out permutation importance, 10 repeats; the substitution probe) | 755.9 s at 1.00 core (0.125); S1f: the probe 456 s (`_eligible_groups` 390 s), permutation 443 s, fits 131 s | NUMPY + SPLIT + LEVER | S3 item 8 (a)-(g): `_eligible_groups` as arrays (oracle verbatim, both arms); the block as units; the combine byte-identical to `_tf19_block`; identity across {1,2,4,8}; the HF reference by the owner-ruled kind (c) variant |
    | 29 | receiver_model: extract | 30 units of 14.7 s; 385 s of unit CPU | W=8: 0.784 in S1; S1h longest-first 0.850 / 0.830 vs sorted 0.843 / 0.838; S1i W=8 / 10 / 12 / 16: 0.844 / 0.868 / 0.870 / 0.900 and 0.845 / 0.855 / 0.854 / 0.902, W=8 the shortest wall | SPLIT, measured first | S3 item 16 (owner ruling 2026-10-31): per-(match, period) units prototyped as an HF measurement; adopted only at >= 0.85 with a byte-identical reduce, else the row returns to the owner |
    | 30 | receiver_model: pool, deploy (2 FAIL) | 0 units (a reduce only) | 6.4 / 5.4 s | SERIAL (too small to pay) | annex text |
    | 31 | receiver_model: fit, assemble (2 FAIL) | 1 unit each | 1.7 s / 0.0 s | SERIAL (too small to pay) | annex text |
    | 32 | win_probability: refs (1 FAIL) | 1 unit | 5.5 s | SERIAL (too small to pay) | annex text |
    | 33 | win_probability: extract | 3,961 units (0.32 s CPU of 0.45 s wall) | S1e and S1e2: W=12 0.946 / 0.926; W=16 0.946 / 0.891 with up to 52 % more CPU | WIO (W=12) | S3 item 5, with the executor bound changed and an executor-level test |
    | 34 | win_probability: fit (1 FAIL) | 1 unit: a 5-fold GroupKFold OOF calibration (a GLM fit and a Markov-chain serve per fold) + the final fit | 1097.5 s at 1.00 core; S1f: the serves 2,060 of 2,294 profiled s, in two per-row `iterrows` loops | NUMPY + SPLIT | S3 item 7: the oracle on every caller path; the `cv` stage; the HF reference by the owner-ruled kind (c) variant (no stored `2474b73` `fit` output exists) |
    | 35 | census: refs, prefit, report (3 FAIL) | 1 unit each | 0.5-0.6 s each | SERIAL (too small to pay) | annex text |
    | 36 | census: fit_prepass, score | 32 units of 0.6-0.9 s | W=8: 0.430 / 0.445, 9-10 s jobs | SERIAL (too small to pay) | annex text (S1e: more workers only add start-up CPU) |
    | 37 | census: bootstrap (1 FLOOR + SAT in the `2474b73` audit) | inner pool 8 | 0.1 s of work in a 6 s job (0.106) | SERIAL (too small to pay; C02) | annex text; it no longer rests on a configuration-only floor |
    | 38 | match_outcome_dependence: run (1 FLOOR, 2565 s) | r3 D12: refs, extract (3,961 units), fit | refs 5 s; extract as row 33 (S1e: 0.713 / 0.965 / 0.963 at W=8 / 12 / 16); fit 3.8 s | refs, fit SERIAL (too small to pay); extract WIO | as rows 32 / 33 |
    | 39 | pass_completion: run (1 FAIL, 56 s) | refs, extract (64 units), cv (5 x 0.3 s), fit | every stage under 30 s of unit CPU | SERIAL (too small to pay, every stage) | annex text |
    | 40 | xsuccess: run (1 FAIL, 106 s) | refs, extract (64 units), study (no-op), cv (5 x 9.7 s), fit (13.5 s, xgb) | unit CPU: extract 20 s, cv 36 s, fit 9 s | SERIAL (too small to pay: refs, extract, cv, fit); study keeps its reviewed reason | annex text; S3 item 10 withdrawn |
  - SERIAL annex (S2 review S2B-03): the exact `serial_reason` text of every SERIAL stage, keyed `(module, stage)`
    as in `scripts/hf_jobs/serial_reasons.py`. The measure is the S1 round-2 unit ledgers (`cpu_s` summed over the
    stage's units; `2474b73` plus the r4 patch, `cpu-upgrade`, read-only jobs `6ae53fcb…`–`6ae53fdd…`). Every
    text below is the "too small to pay" class (owner ruling 2026-10-30): under 60 s of unit CPU. S3 item 12 copies
    each text verbatim into the stage declaration and the registry.

    | Module | Stage | `serial_reason` (verbatim) |
    |---|---|---|
    | `scripts.train_ghost_gk` | `validate` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU in a 19-22 s wall on HF; the wall is the visibility check reading each corpus shard's parquet footer in turn through the bucket mount (about 0.18 s a file). |
    | `scripts.train_ghost_gk` | `select` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 179 column projections, 21-24 s of CPU in all; a W=8 job is 8.5-9.3 s, mostly worker start-up, and W=16 measured slower (11.7 s). |
    | `scripts.train_ghost_gk` | `cache` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU, combining the select outputs into the feature cache. |
    | `scripts.train_ghost_gk` | `assemble` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU, building the metrics block and the bundle from the cv, fit and importance results. |
    | `scripts.train_ghost_outfield` | `assemble` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit of 3.5 s, building the metrics block and the bundle from the cv and fit results. |
    | `scripts.train_xshot_occurrence` | `prep` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 6 s of CPU, the arm-restricted design matrix from the extract results. |
    | `scripts.train_xshot_occurrence` | `fit` | Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): one xgb fit, 26 s of CPU at one core with the dead lever (S1), 7.4 s at 5.84 cores with the lever fixed (S1d); a start-up-bound job of under 30 s. |
    | `scripts.train_xshot_occurrence` | `select` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU, choosing the parameters from the study and cv results. |
    | `scripts.train_xshot_occurrence` | `assemble` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU, the metrics block and the bundle. |
    | `scripts.train_xcross_attempt` | `probe` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): two units, one probe-match extraction each, 45 s of CPU in all and a 24 s span at W=2. |
    | `scripts.train_xcross_attempt` | `probe_comparison` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU, comparing the two probe extractions. |
    | `scripts.train_xcross_attempt` | `prep` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU, the arm-restricted design matrix from the extract results. |
    | `scripts.train_xcross_attempt` | `fit` | Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): one xgb fit, 15 s of CPU at one core with the dead lever (S1); with the lever fixed the job is 9.3 s at 3.77 cores, half of it start-up (S1d). |
    | `scripts.train_xcross_attempt` | `select` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU, choosing the parameters from the study and cv results. |
    | `scripts.train_receiver_model` | `pool` | Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): in the campaign configuration no pool dataset is passed, so the stage has no units and only an empty reduce; the job is 6.4 s. With a pool dataset it has units and is measured like `extract` (S3 item 16). |
    | `scripts.train_receiver_model` | `fit` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 2 s of CPU. |
    | `scripts.train_receiver_model` | `deploy` | Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): in the campaign configuration no deployment dataset is passed, so the stage has no units and only an empty reduce; the job is 5.4 s. |
    | `scripts.train_receiver_model` | `assemble` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU. |
    | `scripts.train_win_probability` | `refs` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU in a 6 s wall, listing the match references. |
    | `scripts.train_match_outcome_dependence` | `refs` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU in a 7 s wall, listing the match references. |
    | `scripts.train_match_outcome_dependence` | `fit` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU. |
    | `scripts.build_territory_ranking_census` | `refs` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU. |
    | `scripts.build_territory_ranking_census` | `fit_prepass` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 32 units of 0.6-0.9 s, 15 s of CPU in all; a W=8 job is 9.4 s, mostly worker start-up. |
    | `scripts.build_territory_ranking_census` | `prefit` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU. |
    | `scripts.build_territory_ranking_census` | `score` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 32 units of 0.6-0.9 s, 20 s of CPU in all; a W=8 job is 10.4 s, mostly worker start-up. |
    | `scripts.build_territory_ranking_census` | `bootstrap` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 0.1 s of work in a 6 s job; its idle cores are start-up, not structure. |
    | `scripts.build_territory_ranking_census` | `report` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU. |
    | `scripts.train_pass_completion` | `refs` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU. |
    | `scripts.train_pass_completion` | `extract` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 64 units, 28 s of CPU in all; W=16 measured slower than W=8 (14.8 s against 13.3 s) and spent 58 % more CPU. |
    | `scripts.train_pass_completion` | `cv` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 5 folds of 0.3 s, 1 s of CPU in all. |
    | `scripts.train_pass_completion` | `fit` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU. |
    | `scripts.train_xsuccess` | `refs` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU. |
    | `scripts.train_xsuccess` | `extract` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 64 units, 20 s of CPU in all; a W=8 job is 9.8 s, mostly worker start-up. |
    | `scripts.train_xsuccess` | `cv` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 5 folds, 36 s of CPU in all; at W=5 the job is 13.4 s. |
    | `scripts.train_xsuccess` | `fit` | Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one xgb fit, 9 s of CPU. |
    | `scripts.train_xsuccess` | `study` | (unchanged, R3-10) Optuna TPE: each trial reads every earlier trial, so trials cannot run in parallel without changing the search; with hpo_trials=0 (the campaign default) the unit is a no-op |

    Reviewed floors added in S3 (owner rulings 2026-10-31: row 29, "Per-match + reason"; S3B-04), each measured on HF,
    verbatim as in `scripts/hf_jobs/serial_reasons.py` (review S3B-13):

    | Module | Stage | `serial_reason` (verbatim) |
    |---|---|---|
    | `scripts.train_receiver_model` | `extract` | Reviewed floor (owner ruling 2026-10-31, row 29): 30 per-match units at W=8 measured 0.844 and 0.845 of the vCPU on HF (S1i), just under the 0.85 bar. A per-(match, period) split (90 units) gave the same bytes but 0.797 and 0.829 (S3 item 16 (c), job 6ae611200e0338d1bb7d327a), so the stage keeps per-match units. |
    | `scripts.train_xcross_attempt` | `tf19_ablation` | Reviewed floor (owner ruling 2026-10-31, S3B-04): 10 single-thread xgb fits of 8-19 s; the longest unit bounds the wall. On HF every parallel shape took 30.6-32.3 s (W=8 x 1: 32.3 s, 162 CPU-s, 0.66; W=10 x 1: 0.78-0.81; W=5 x 2: 0.79; job 6ae62901c78863f9c62f3a63), so the stage runs pinned at W=8, the least CPU. The n_jobs-lever shape (W=1 x 8) took 58.8 s and 327 CPU-s (job 6ae6254b0e0338d1bb7d3760). |
    | `scripts.train_xcross_attempt` | `tf19_probe` | Reviewed floor (owner ruling 2026-10-31, S3B-04): one single-thread unit, the TF-19 substitution probe, 68-81 s of CPU on HF (job 6ae629e00e0338d1bb7d3825). It cannot split inside one stage: one sequential RNG draws the frame sample and then each frame's placebo picks, so a split needs a second stage and its job start-up for about 60 s of gain. |
    | `scripts.train_win_probability` | `cv` | Reviewed floor (owner ruling 2026-10-31, S3B-04): the 5 GroupKFold folds of the calibration (fixed at 5), each a single-threaded GLM fit on its training matches and a serve of its held-out matches. On HF the stage took 85 s, 395 CPU-s and 4.64 cores (0.58) at W=5 (job 6ae624e7c78863f9c62f3947); a finer split re-fits a fold model per chunk or adds a second stage. |

    Row 1 (B S2B-03): the cause is the serial footer loop (`_corpus_visibility.py:32-35`), not the read cap. A pooled
    read at the cap of 4 would save about 15-19 s of wall per `ghost_gk` instance; the stage's 1 s of CPU puts it in
    the "too small to pay" class, so it stays one unit by the owner's ruling, with this cause recorded.
- [ ] S3: implement each decision behind its identity gate (`test_thread_identity` + `test_thread_safe_registry`),
  TDD per stage.
  - S2 table APPROVED by A and B (r4, 2026-10-31). S3 progress (local, tests GREEN on pandas 2.3.3 / 3.0.3 / 3.1.0
    unless noted; HF identity runs still owed per item):
    - item 0 written (ADR-128 Amendment 2026-10-31).
    - items 1 to 4 code: the `n_jobs` lever raises OMP (env runner `assert_thread_axis_identical_in_env`, the four
      xshot/xcross arms byte-identical across {1, 2, 4, 8}); `thread_budget` ceil for n <= 5 processes; THREADS
      (`THREADS_MAX_UNITS = 7`, the RAM clamp surfaced as `threads-ram-floor`); the three floors with measured use,
      the "too small to pay" class, the reduce CPU record; `SATURATION_FRACTION = 0.85`.
    - item 5 code: WIO. `UnitCost.cpu_share` (p50 of `cpu_s / wall_s`), `planner.is_wio`, `StagePlan.wio`, and
      `--wio` through the action, the job command and `ExecutorConfig.wio`. Author ruling for the S3 review: the
      planner marks a stage WIO when its share is under `SATURATION_FRACTION`. The plan names the formula but no
      criterion, and the formula alone gives W = 9 for a 0.89-0.99 share, which S1i measured as wasted CPU. Tests:
      0.71 plans W = 12; 1.0, 0.96 and 0.85 plan W = 8; the RAM cap binds (10); the executor launches 12 for a WIO
      job planned at 12 and 8 for one that is not WIO.
    - item 6 code: MEM. `scripts/hf_jobs/mem_gate.py` (`MemGate.decide`: grant, wait, or stop the idle workers
      whose retained heap blocks the claim, rule (i); rule (ii) admits a unit alone; `box_victim`), the parent-granted
      claim protocol in `executor._dispatch` (`--gate`), W = vcpu for a gated stage (its WIO W when it is also WIO,
      owner ruling 2026-10-31) in the planner and in `_run`,
      `campaign.bucket_unit_peaks` (since review round 1, owner ruling "Bucket search": the newest REDUCED
      generation of the same `history_key` in any run key of the campaign bucket; the run keys listed once per
      controller, a snapshotted run key read from its step snapshot) staged as `--unit-peaks`. Tests: `test_mem_gate.py` (12) and the planner, controller, CLI and executor tests; with real
      workers, seven at 4 GB of retained heap and a 7.2 GB unit of a 30 GB budget: the idle workers are stopped and
      every unit completes.
    - item 7 (b) code: the win_probability `cv` stage, one unit per GroupKFold fold (`calibration_fold`); `fit`
      reads `metrics_from_cv` over the folds in fold order. `calibration_metrics` stays as the verbatim oracle:
      the metrics JSON is byte-identical for 6, 3 and 1 matches.
    - item 8 (b)-(f) code: the xcross TF-19 block as units. `tf19_ablation` (10 units: arm x 5 fold slots),
      `tf19_perm_fit` (5 fold models), `tf19_perm` (80 (fold, column) units through the new shared
      `scripts/_permutation_units.py`, which ghost_gk `importance` now also uses), `tf19_probe` (the serial probe
      part, unchanged); `assemble` combines them in the serial key order. A fold slot past the shipped arm's fold
      count returns an absent marker, so the units need no corpus read at plan time. `tf19_perm_fit` takes the
      n_jobs LEVER; `tf19_ablation` runs pinned and `tf19_probe` is one unit, each with a reviewed reason (owner
      rulings 2026-10-31, S3B-04, below). Tests: `tests/scripts/test_xcross_tf19_units.py` (the `2474b73`
      `gk_block_ablation`, `permutation_importance_report` and `_tf19_block` verbatim as the oracles; 12 and 3
      games; a one-class fold; the toy-corpus pipeline against the serial block); the identity arm
      `test_tf19_perm_thread_identity_xcross` across {1, 2, 4, 8} in the production env (the `tf19_ablation` arm
      left with its `thread_safe` opt-in). Item 8 (g), the HF reference: done (below).
    - item 9 code: `eval_fold`, `_cv_logloss` and both study objectives take `runtime_n_jobs`; the trainers' InnerPool
      fold task reads `runtime_n_jobs_from_env()`; the xShot `study` stage opts into `thread_safe` with the n_jobs
      lever, so `thread_budget(vcpu, 1, 5)` gives each child ceil(8 / 5) = 2 threads; the xCross `study` stays at
      one thread (owner ruling 2026-10-31, "xShot only", below). Tests: the fold fit and the in-process objective
      path read the budget; the child task reads the env budget; the identity arm
      `test_study_children_thread_identity_xshot` (four trials' values through real InnerPool children) at child
      budgets 1 and 2. The HF measurement: done (below).
    - item 12 code: the 35 SERIAL annex texts, copied by a script from the annex table above (no hand
      transcription), into each module's `_SERIAL_REASONS` (its stages' `serial_reason`) and into
      `scripts/hf_jobs/serial_reasons.py`; `test_serial_reasons` holds them EXACTLY equal (RED with the registry
      alone, then GREEN).
    - item 13 code: `measure --canary` adds ONE read-write mount, `silly-kicks/hf-refit-canary` at `/canary`;
      `measure.check_mounts` refuses any other writable mount where the mounts are built. The committed canary
      `scripts/hf_jobs/canary.py`: `writer` (the toy pipeline through the production executor and store, 64 probe
      files, one real-scale output copied through `LocalStore.write_bytes`), `reader-mount` (a fresh container's
      mount view with the mtime check) and `reader-api` (the bucket API through `StoreSnapshot` over
      `BucketStore`, run on this machine; jobs get no secrets); one shared `verify_units` over either view. Tests:
      `test_measure.py` (the canary mount, the refusals, no secrets) and `test_canary.py` (a local writer, both
      readers over the same files, a corrupt unit and a corrupt large output reported). The HF run is rung 3.
    - item 15 (b) code: the ghost_outfield `coherence` stage (8 row-chunk units after `fit`, the saved model, the
      reduce runs `ghost_rearguard_coherence` on the chunks in key order); `fit` no longer predicts the training
      rows; `assemble` reads the coherence from the stage. Tests: `test_ghost_outfield_coherence_stage.py` (the
      `2474b73` fit-time lines verbatim as the oracle, the chunks through the store's parquet; the toy-corpus pipeline
      against the oracle on the same rows and model). Item 15 (c) needs no code (the planner sizes the stage from its
      history, with the MEM gate); (d) is the HF identity.
    - item 16 (b) code: the receiver `extract` and `pool` take one unit per (match, period slot): periods 1, 2, and
      one slot for every later period. The reduce orders the units by (match stem, slot), never by the raw key
      ("m10__p1" sorts before "m1__p1"); `n_failed_matches` counts matches, not units. Tests:
      `test_receiver_period_units.py` (two-period fixtures, one with a pass at the end of period 1 whose touch
      opens period 2, and one with extra time: the slots' rows and coverage counters equal the per-match ones; the
      reduce order; the failed-match count; the unit keys). NOT adopted: the HF measurement (c) gave 0.797 and
      0.829, so the change and its test file are reverted and row 29 keeps per-match units with a reviewed reason
      (owner ruling 2026-10-31, "Per-match + reason", below).
    - item 1 (b) result (HF job `6ae60f4bc78863f9c62f312b`, production env, `thread_safe_env("n_jobs", 4)`): in
      each n_jobs-lever process (xShot and xCross `cv` fold and `fit`, the study children, the xCross `tf19_ablation`
      and `tf19_perm` units) `threadpool_info()` lists ONE OpenMP library, `libgomp`, at 4 threads, loaded at the
      sklearn import (xgboost's OpenMP resolves to the same `libgomp`); the two BLAS pools (numpy's and scipy's
      `libscipy_openblas`) stay at 1. Only xgboost's fit and predict call OpenMP there; the sklearn calls in these
      processes (metrics, `StratifiedGroupKFold`, the permutation scoring) are not OpenMP code.
    - S3 review round 1 (A and B, both REQUEST CHANGES; reports `D:\Development\_reviews\2026-10-31-hfu-s3-impl-A.md`
      and `-B.md`; author response `-r2-author.md`). Owner rulings 2026-10-31 (third set, AskUserQuestion):
      - row 29, "Per-match + reason": item 16 is NOT adopted. The per-(match, period) units gave byte-identical
        output (two split runs, the base per-match run and the stored `2474b73` output all hash `47014a41…`), but
        utilization 0.797 and 0.829 at W=8 (job `6ae611200e0338d1bb7d327a`). The stage reverts to per-match units
        with a reviewed serial reason that records both measurements (S1i per-match 0.844 / 0.845).
      - item 9, "xShot only": the xShot study keeps child budget 2 (230.5 s against 344.6 s at budget 1; utilization
        0.761 against 0.352; 1.22 cores a child, above the floor bar; output byte-identical; job
        `6ae611e40e0338d1bb7d32b3`). The xCross study stays at budget 1: at budget 2 it took 90.6 s against 74.5 s,
        406 against 262 CPU-s, 0.9 cores a child (below the bar); output byte-identical (job
        `6ae611e60e0338d1bb7d32b5`).
      - item 6 history source, "Bucket search" (B S3B-02, S3B-03): a new commit's campaign starts in a fresh store, so
        "an earlier generation in the same store" never gated row 2 in production. `campaign.bucket_unit_peaks`
        reads the newest REDUCED generation (with `_inputs.json` and `_summary.json`) of the same `history_key` in
        any run key of the campaign bucket that runs the same JOB (round 2, S3B-11: the run key names the job, not
        the pipeline); never the current generation, a sibling tier's generation (another `history_key`) or a running
        one. The first campaign of a new computation runs ungated, as before. Its API cost per controller (S3B-12):
        one root listing; per earlier run key of the job and stage, one listing and one `_inputs.json` download; per
        matching generation, one ledger download; every read is shared by all instances.
      - gated + WIO, "Keep code, fix text" (A S3-IMPL-02): a gated stage plans W = vcpu, or the WIO W when it is
        also WIO; the gate bounds memory, WIO adds the workers that wait on I/O.
    - Owner rulings 2026-10-31 (fourth set, S3B-04: the new stages, from clean HF measurements; every output
      byte-identical):
      - `tf19_ablation`, "Pinned W8 + reason": 10 xgb fits of 8-19 s; every parallel shape takes 30.6-32.3 s
        (W=8 x 1: 32.3 s, 162 CPU-s, 0.66; W=10 x 1: 0.78-0.81; W=5 x 2: 0.79; job `6ae62901c78863f9c62f3a63`);
        the lever shape W=1 x 8 took 58.8 s and 327 CPU-s (job `6ae6254b0e0338d1bb7d3760`). The stage drops
        `thread_safe` (its identity arm and registry entry go) and carries a reviewed reason.
      - `tf19_probe`, "Reviewed reason": one single-thread unit, 68.3 and 80.5 s of CPU (job
        `6ae629e00e0338d1bb7d3825`); one sequential RNG draws the frame sample and then each frame's placebo picks.
      - win_probability `cv`, "Reviewed reason": the 5 GroupKFold folds, single-threaded; 85 s, 395 CPU-s, 4.64
        cores (0.58) at W=5 (job `6ae624e7c78863f9c62f3947`).
      - No ruling needed (the audit passes as coded): `tf19_perm_fit` THREADS (W=5 x 2: 19.5 and 22.3 s, 5.92 and
        5.62 cores, all-at-once floor; W=5 x 1 22.5 and 25.3 s, 3.94 cores; job `6ae6276b0e0338d1bb7d37bb`);
        `tf19_perm` SAT (W=8: 104.1 s, 7.16 cores, 0.896, same job); win_probability `fit` "too small to pay" (44
        CPU-s, job `6ae624e7c78863f9c62f3947`); ghost_outfield `coherence` SAT (58 s, 7.05 cores, 0.88).
    - HF evidence so far (read-only measurement jobs, stored `2474b73` cycle-1 units at /scratch, no secrets):
      - item 8 (g): the TF-19 block JSON is byte-identical, serial `_tf19_block` against the units plus the combine,
        all 8 keys in order, on the public tier (job `6ae611aec78863f9c62f320a`; serial 84 s) and on `sc_extended`
        (job `6ae613f50e0338d1bb7d3330`; serial 931 s).
      - item 15 (d), ghost_outfield default: the `fit` model files are byte-identical to the stored fit's; the 8
        coherence chunks (W=8, 54-70 s, 6.9-7.3 cores) give the stored fit's `coherence.json` byte for byte; the
        `2474b73` coherence body gives the same result on the same served frame (259-343 s). `assemble`: the model
        files are byte-identical and `metrics.json` equals the stored one except `run_tree_dirty` (the probe tree
        carries the patch; job `6ae61d52c78863f9c62f3605`). The fit's measured use in seven clean runs: 88-118 s at
        6.72-7.29 cores (jobs `6ae611ac0e0338d1bb7d32a4`, `6ae623d60e0338d1bb7d3717`, `6ae623d8c78863f9c62f38fd`,
        `6ae62549c78863f9c62f3954`), above the floor bar. The stored `position_only` extract predates r4 D18 (its
        own extraction, no faithful columns), so `position_only` is not comparable with the stored units; the
        default variant carries the identity.
      - item 14, ghost_gk (jobs `6ae61bb5c78863f9c62f354a`, `6ae6255fc78863f9c62f395a`): every `cv` fold and the
        `fit` are byte-identical to the stored units for `default` and `position_only`; the `fit` used 4.68-6.85
        cores (above the bar).
      - item 7, win_probability (jobs `6ae61219c78863f9c62f3228`, `6ae624e7c78863f9c62f3947`): the `extract` output
        and the `fit` model are byte-identical to the base side (`metrics.json` differs only in the fields that name
        each side's tree); `derive_match_states` and `compute_win_probability` give byte-identical states and
        scores on the full pooled extract with the new and the `2474b73` library.
      - Slow windows (jobs started at 16:10 and 16:46-16:54 UTC): OpenMP stages ran 7-14 times slower at 3-6 times
        the CPU (ghost_outfield `fit` 1,330 and 1,650 s; xcross `tf19_ablation` at W=1 x 8 1,533 s), every output
        byte-identical. Clean re-runs gave the numbers above. In-container contention did not reproduce it, and
        `OMP_WAIT_POLICY=PASSIVE` costs about 33 % of the wall with no gain under contention, so the policy is not
        changed (ADR-128 Amendment 2026-10-31).
      - The bucket mount finding: five processes reading one cold 156-215 MB file at once took 546 s and 4,161 s,
        where one reader takes 1.4-1.6 s and two take 2.4 s (jobs `6ae62805c78863f9c62f3a2c`,
        `6ae633d5c78863f9c62f3e09`, `6ae634a00e0338d1bb7d3a15`); the five ghost_gk `cv` folds of a THREADS job
        spent 455 of 527 s in `read_bytes` (job `6ae634ccc78863f9c62f3e3e`). Owner ruling 2026-10-31 (fifth set),
        "Per-file read lock": `LocalStore.read_bytes` holds a container-local `fcntl` lock per file
        (`_stages._file_read_lock`; `tests/scripts/test_store_read_lock.py`, RED then GREEN on Linux). With it the
        five folds took 70-75 s and their reads 5 s or less (job `6ae63a7cc78863f9c62f4051`).
    - two stale tests left by the r4 work, fixed: `test_reduce_linear` (the ghost_gk cache reads the reduced `select`
      units, r4 D19; the xcross probe cohort reads `arms_json`, R4-10) and `test_tf54b_drivers` (R3-10 moved
      `train_pass_completion` onto the stage CLI; the gate now resolves the CLI as `test_provenance_wiring` does).
    - item 14 code: `GhostGkModel._training_leaves` lazy; `tests/tracking/test_ghost_gk_lazy_leaves.py` (RED 3 of 6, then
      6 passed); the two numba spy tests of `test_ghost_gk_leaf_numba.py` bind numba and read the leaves first (RED alone,
      then GREEN). The full ghost_gk tracking suite ran once on 2.3.3 (365 passed, the 1 spy failure fixed since).
    - item 15 (a): `ghost_rearguard_coherence` over arrays; `tests/tracking/test_ghost_rearguard_coherence_arrays.py` 20
      passed; at 100,000 groups the old body 17.4-24.8 s, the new 0.15-0.17 s, results identical; 1,000,000 groups 2.3 s.
    - item 7 (a): `derive_match_states` (per game over arrays, an exact array walk for a null-team scorer or card) and
      `_score_states` (per group over arrays); `tests/win_probability/test_state_arrays_identity.py` with both 2474b73
      bodies verbatim; the win_probability suite 90 passed on each pandas version.
    - item 8 (a): `_eligible_groups` over arrays, group frames built on read (`_EligibleGroups`);
      `tests/tracking/test_eligible_groups_arrays.py` 28 passed; the probe suites (xcross_eval, xs_placebo_v2,
      xs_probe_wiring, probe_discriminating_power, xcross integration) 77 passed on each pandas version.
  - The S3 work list (from the S2 table; each item test-first; files frozen by the r4 impl review round wait for it
    to close):
    0. ADR-128 INV-2 amendment (S2B-06), written FIRST because items 1, 2 and 9 depend on it. INV-2 today:
       `thread_budget = max(1, vcpu // (workers·inner_workers))`, so `threads x units_in_flight <= vcpu`, and the
       `n_jobs` lever keeps the worker env fully serial. Amended: (a) the THREADS shape takes
       `ceil(vcpu / n)` threads for n <= 5 units, an oversubscription of at most `n x ceil(vcpu / n) - vcpu`
       threads (2 at n = 5, 1 at n = 3, 0 at n = 1, 2 and 4), measured on HF at n = 5 (S1d); n = 6 and 7 keep the
       floor rule (`vcpu // n`), because no HF measurement covers 12 or 14 threads on 8 vCPU; (b) the `n_jobs`
       lever raises `OMP_NUM_THREADS` to the budget (item 1), and the BLAS and numba variables stay "1"; (c) an
       inner pool of n <= 5 children gets `ceil(vcpu / n)` threads per child (owner ruling 2026-10-31 on
       S2B-r2-02; today `thread_budget(8, 1, 5) = 8 // 5 = 1`, `_thread_pin.py:66-69`), the same bounded
       oversubscription as (a), proven by item 9's HF measurement and identity. The ADR text names all three.
    1. LEVER (rows 15-18, 24-28). `_thread_pin.thread_safe_env("n_jobs", b)` also sets `OMP_NUM_THREADS=b`;
       `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMBA_NUM_THREADS` stay "1"; the
       explicit runtime `n_jobs` stays. Chesterton (B38, `scripts/_thread_pin.py:78-82`; S2-PLAN-02, S2B-06): the
       lever kept EVERY thread variable at "1" and passed the budget trainer-side because the xgb determinism
       evidence varied OpenMP only and never covered BLAS or numba ("so the xgb worker's numpy/BLAS prep -- an
       axis the xgb determinism evidence never covered -- never multi-threads"). This item raises ONLY the OpenMP
       axis, which the identity arm varies; BLAS and numba stay "1", so nobody re-raises them by the same logic.
       XGBoost caps `nthread` at `omp_get_max_threads()`, so the env must carry the budget. Raising OpenMP reaches
       every OpenMP consumer in the process, not only xgb. Steps, each RED first:
       - (a) a subprocess under `thread_safe_env("n_jobs", 4)` reports `num_threads` 4 for every OpenMP library
         `threadpoolctl.threadpool_info()` lists, and 1 for every BLAS library;
       - (b) the S3 record lists, per lever process (xshot/xcross `cv` and `fit`, the study children, the xcross
         block units of item 8), the OpenMP libraries `threadpool_info()` reports in the production env and the
         code in that process that calls into each (expected: xgboost's libgomp; sklearn's vendored libgomp where
         sklearn is imported, and which sklearn calls in that process use it);
       - (c) the identity arm `assert_thread_axis_identical(..., lever="n_jobs")` runs each thread count in a
         SUBPROCESS under the production env (today it runs in the test process, where OpenMP is not capped) and
         compares the WHOLE stage output (the stage's encoded unit results, byte for byte) across {1, 2, 4, 8}.
    2. THREADS (rows 5, 10, 16, 25). The planner gives a `thread_safe`, non-inner stage of n <= 5 units ONE job with
       W = n and `ceil(vcpu / n)` threads per worker; n = 6 or 7 keeps `vcpu // n`; more units keep today's rule.
       Executor clamps (S2B-04): `_run` (`executor.py:375-376`) still clamps W by `size_workers` against the RSS
       hint, and `should_launch` (`executor.py:492-494`) still gates each launch; the per-worker cap stays
       `max(floor_cap, usable // W)` (`planner.py:212`). With item DEAD, the ghost_gk fold peak loses the leaf
       matrix; ghost_outfield's measured fold peak (4.56-4.71 GB) gives 5 x 1.3 x 4.71 = 30.6 GB, inside the
       usable 30 GiB (32.2 GB). If the RAM clamp still lowers W below n, the job runs at the clamped W with the
       budget of that W, the plan carries a note, and the FLOOR check fails honestly (W < n). Tests: n = 5 plans
       one job, W = 5, budget 2; n = 7 plans budget 1; EXECUTOR level: a job planned W = 5, budget 2 launches 5
       workers and each worker's env carries budget 2; a RAM hint that fits 4 launches 4 with the note.
    3. FLOOR (rows 5, 6, 10, 11, 16, 25 and the studies 15, 24; S2B-02, S2B-r4-01, S2B-r4-02).
       `campaign.audit_utilization` passes each of the three floors of spec §A, (a) the generalized floor
       (`workers >= n_units` and `workers x thread_budget >= vcpu`), (b) the existing final-fit floor
       (`campaign.py:1646`, `n_units <= 1 and thread_budget >= vcpu`) and (c) the existing study floor at its fold
       width (`at_fold_width`, `campaign.py:1645`), ONLY with measured use when the budget is
       above one thread: from the stage's D8 job record, `cpu_s / wall_s / (workers x inner_workers) >
       thread_budget / 2`, strictly (owner rulings 2026-10-31; ONE measure, single-sourced in one function that
       the vocabulary and spec §A name). Every test number is a D8 job-record value or labelled synthetic. Tests,
       RED first for (b): a W = 1, budget 8 fit at 0.97 cores (the dead lever, S1 job record) is NOT a floor (it
       passes today); a W = 1, budget 8 fit at 1.75 cores (row 11, S1 job record) is NOT; a W = 1, budget 8 fit
       at 3.77 cores (xCross fit with the lever fixed, S1d job record) is NOT; a W = 1, budget 8 fit at exactly
       4.0 (synthetic, the boundary) is NOT; a W = 1, budget 8 fit at 6.0 (synthetic) is; W = 5, budget 2 at 5.0
       cores (one per worker, a dead lever; synthetic) is NOT; W = 5, budget 2 at 6.12 (S1d ghost_gk `cv` job
       record) is; W = 5, budget 2 at 5.39 (S1d xCross `cv` job record, 1.078 per worker) is; an inner-pool
       study, W = 1, inner 5, budget 2 at 4.7 cores (synthetic dead lever in the children) is NOT; W = 5, budget
       1 is not. The "too
       small to pay" class is also checked against the measurement: a stage that declares it fails the audit when
       its summed unit CPU plus its reduce CPU is 60 s or more, so the class cannot hide a stage that grew. The
       reduce CPU (S2B-r2-04): the `{gen}/_reduce.json` record (HFU T4) gains `cpu_s`, the process CPU over the
       reduce; a reduce-only stage then measures its reduce, not 0 s. Test: a reduce-only stage whose reduce
       record shows 157 s of CPU fails the class.
       Variant and tier inheritance (C06): every decision is per (module, stage) and holds for every variant and
       tier, because the planner sizes each instance from that instance's own measured history; a smaller tier's
       instance of a shaped stage whose measured unit CPU is under 60 s passes as "too small to pay" by the same
       measured rule, and the audit artifact records its measured CPU. The S1 shapes were measured on ghost_gk
       `default` and `full`, ghost_outfield `default` and xshot/xcross `sc_extended`; the rung-5 audit covers the
       rest.
    4. `SATURATION_FRACTION = 0.85` (owner ruling S0). Test: the constant, and the audit verdict at 0.849 / 0.851.
    5. WIO (rows 33, 38). The planner sizes an item stage's W as `min(RAM cap, ceil(vcpu / cpu_share))`, where
       `cpu_share` is the p50 of the measured unit `cpu_s / wall_s` (both in every ledger record); W=12 for the
       measured 0.71 share (S1e and S1e2, C03). Executor clamps (S2B-04): today `size_workers` clamps W to
       `[1, nproc]` (`executor.py:376`, `_parallel_launch.py:41`), so a planned W=12 runs at 8. The planner marks a
       WIO stage, and for it the executor bounds W by the planned W instead of `nproc` (the RAM part of
       `size_workers`, `should_launch` and the per-worker cap stay; the units peak at 0.36 GB, so 12 x 1.3 x 0.36 =
       5.6 GB). A stage that is not WIO keeps the `nproc` bound. Tests: a share of 0.71 plans W=12; a share of 1.0
       plans W=8; the RAM cap still binds; EXECUTOR level: a job planned WIO W=12 launches 12 workers, and a job
       planned W=12 for a stage that is not WIO launches 8.
    6. MEM (row 2; S2B-04, S2B-05, S2-PLAN-03). A memory admission gate in the executor replaces the planner's
       `1.3 x max peak` worker cap for a stage with per-unit history. Chesterton (`planner.py:173-175`): the 1.3x
       cap exists because "a cap of 1.3 x the profile SAMPLE's peak killed every heavier unsampled unit on real HF
       (2026-10-20)"; the gate keeps that protection per unit and never relaxes a stage without per-unit history.
       - History source: the unit ledger records' `peak_rss_bytes` (`_stages.py:892`) of the newest EARLIER
         generation of the same `(run_key, stage)` in the campaign store, read by the controller once per stage
         and cached by `(run_key, stage)`, so it survives a new generation; the controller passes each key's
         predicted peak to the job with its shard keys. A unit with no earlier record takes the earlier
         generation's stage MAXIMUM, never the p95. A stage with no earlier generation keeps today's rule (W from
         1.3 x the profile's maximum): no first-run relaxation.
       - Reservation: `1.3 x` the predicted peak (`cap_factor`).
       - Admission against measured resident memory: a worker claims its next unit only while the sum, over the
         other live workers, of `max(measured RSS, reservation of the unit in flight)` (an idle live worker counts
         its measured RSS, so heap retention after a heavy unit is counted), plus `max(own measured RSS,
         reservation of the next unit)`, fits the usable RAM (`Constraints.usable_bytes` = 30 GiB on
         `cpu-upgrade`).
       - Clamps: for a gated stage, `_run` skips the RSS part of `size_workers` (`executor.py:376`; it would clamp
         ghost_gk `extract` to W=4) and the gate replaces `should_launch`; the per-worker cap stays `floor_cap`
         (1.3 x the stage maximum, 7.2 GB) against one runaway unit; a box guard kills the worker with the largest
         excess over its reservation when the summed RSS passes the usable RAM, and its unit goes to the existing
         memory escalation (`deferred_memory_units`).
       - First-run safety (A S2-PLAN-03): the gate admits a set only while its reservations fit 30 GiB, whatever
         the mix, so the worst admitted set is bounded by the box: 8 x 1.3 x 5.51 = 57 GB is never admitted; at
         most four of the largest units run at once, and eight of the p50 units (8 x 1.3 x 2.08 = 21.6 GB) do.
       - Progress rule (S2B-r2-01): retained heap must never stall the stage. (i) An idle worker whose measured
         RSS is part of the reason a pending claim is refused exits, which returns its heap to the box, and a fresh
         worker launches in its slot once the gate admits it. (ii) Only after (i) has run, so that no idle worker still
         holds retained heap, and when no unit is in flight in the job, the next claim is admitted whatever its
         reservation (S2B-r3-04) (one unit alone always fits: the per-worker cap and today's
         "the first worker always starts" sizing already prove it).
       - Tests (RED first): every worker idle with retained RSS (seven at 4 GB) and a pending unit with a 7.2 GB
         reservation that does not fit: the idle workers exit and the stage completes; a worker that retains 4 GB
         RSS after a 5 GB unit blocks a claim that would overfill;
         a first run with no earlier generation plans today's W; 8 units of 5 GB under a 16 GB budget run at most
         2 at once; a unit with no record reserves the stage maximum; EXECUTOR level: a gated job planned W=8
         launches 8 workers (today `size_workers` clamps it to 4).
    7. Row 34 (`win_probability`; owner ruling 2026-10-30: this cycle). (a) `derive_match_states`
       (`silly_kicks/win_probability/_state.py`) and `_score_states` (`_compute.py`) walk arrays instead of
       `iterrows`, by the D20 method: the `2474b73` bodies verbatim as the test oracle, identity on fixtures and
       randomized inputs, through every caller path (S2B-07): the targets mode, the public
       `compute_win_probability` call (`silly_kicks/win_probability/__init__.py:14`) and the head-to-head consumer
       (`scripts/hf_jobs/headtohead.py:159`). (b) A `cv` stage with one unit per calibration fold that returns the
       fold's OOF predictions and targets; `fit` keeps the final model; the metrics come from the folds'
       predictions concatenated in fold order, byte-identical to `calibration_metrics`. HF reference (the
       owner-ruled kind (c) variant; no stored `2474b73` `fit` output exists, because that instance halted at the
       `extract` reduce): one measurement job runs the old bodies and the new code on the same full-scale inputs
       and compares the states, the scores and the `fit` output byte for byte.
    8. Row 28 (xcross; owner ruling 2026-10-30: this cycle), in ordered sub-steps, each RED first (S2-PLAN-05):
       - (a) `_model_eval._eligible_groups` walks arrays (the D20 method, the old body verbatim as the oracle);
         the oracle tests cover the xcross arm and `arm="xs"` (`xs_substitution_probe`, `_model_eval.py:163`);
       - (b) the 10 GK-block ablation fold fits as units; test: each unit equals the serial block's fold value;
       - (c) the 5 permutation fold fits, then the 80 (fold, column) permutation units that score with the fold
         model (sklearn's permutation seed rule, as r3 D10); test: the scores equal the serial loop's;
       - (d) the substitution probe as one unit;
       - (e) the combine, which writes the block JSON byte-identical to `_tf19_block`; test on a toy corpus;
       - (f) the block's xgb fits and predictions take the LEVER budget; identity of the whole block JSON across
         {1, 2, 4, 8} in a subprocess under the production env (S2B-06);
       - (g) HF reference (the owner-ruled kind (c) variant; R4-10a changed the block's inputs, so no stored
         `2474b73` unit can match): one measurement job runs the serial `_tf19_block` and the new units plus the
         combine on the same full-scale inputs; the block JSON is byte-identical.
    9. The xgb study children (rows 15, 24): each inner child runs under `thread_safe_env("n_jobs",
       ceil(vcpu / inner))` (5 x 2 = 10 threads on 8 vCPU, item 0 (c)) and its fold fit reads the runtime
       `n_jobs`. Identity: the study's best params and every trial value byte-identical at child budgets 1 and 2,
       in the production env. HF measurement: the xShot and xCross `study` at child budget 2 against 1 (job
       utilization and wall); the floor's measured use (item 3) applies to the result.
    10. Withdrawn (owner ruling 2026-10-30, "too small to pay"): xsuccess `cv` (36 s of unit CPU) and `fit` (9 s)
        take the SERIAL class (annex) instead of the LEVER.
    11. Withdrawn: longest-first claim order gave no gain on HF (S1h).
    12. SERIAL (rows 1, 3, 4, 8, 12, 14, 17-19, 21-23, 26, 27, 30-32, 35-40 as the annex lists): the annex text
        verbatim on each stage and its registry entry; `test_serial_reasons` holds them equal.
    13. The rung-3 canary in the repo (spec r3.4 item 3 says "committed"): the writer/reader probe (today
        `D:\Temp\hfu\r38\rung3_canary.py`) and a `measure.py` canary mode whose ONLY read-write mount is the canary
        bucket `silly-kicks/hf-refit-canary` (owner ruling 2026-10-30), asserted where the mounts are built, so a
        measurement job cannot write the campaign bucket by mechanism (S0b). Tests: the canary mode mounts exactly
        that bucket read-write and nothing else; a campaign bucket refuses. C05: the committed reader lists and
        reads through the bucket API (`HfApi`), not the mount; the writer and the reader run on ONE tree; the
        writer also writes one real-scale stage output through the mount (a `ghost_gk` `extract` reduce
        `_output`).
    14. DEAD (rows 5, 6; owner ruling 2026-10-31, "Make lazy"). `GhostGkModel.fit`
        (`silly_kicks/tracking/_ghost_gk.py:2126`) stops walking every training row: it keeps a reference to the
        training feature matrix and `_training_leaves` becomes a property that computes the leaves on first read,
        with the same function on the same matrix, then drops the matrix. Its setter keeps `__init__` and `load()`
        working (`load()` assigns the stored value, today always None). `predict_density` reads it as today. No
        trainer reads the leaves and `save()` never persists them, so the `cv` and `fit` unit outputs are
        byte-identical by construction. Tests, RED first: (a) `fit` runs no leaf walk over the training rows (a
        spy on `_vectorized_leaf_indices`); (b) the lazy leaves `np.array_equal` the eager ones from the old body
        kept verbatim as the oracle, on random and fixture data; (c) `predict_density` gives byte-identical grids
        eager versus lazy; (d) a loaded model still refuses `predict_density` as today. HF: the ghost_gk `cv` and
        `fit` units byte-identical to the stored `2474b73` units (no-rebaseline kind (c): a restructure of the code
        that computes the result, with the eager body as the verbatim oracle and the HF identity on stored units;
        S2B-r2-05), and the new fold wall and fold peak RSS measured for the THREADS sizing; the job also reports the
        `fit` stage's D8 job record against the floor bar of item 3 (more than 4.0 cores at budget 8; S2B-r4-02). The existing
        `test_ghost_gk_leaf_numba.py::test_predict_density_uses_the_indices_numba_kernel_on_the_default_path`
        counts kernel calls in the first `predict_density`; with lazy leaves that call also walks the training
        rows (2 calls, not 1). The test binds numba and reads the training leaves before it installs its spy, so it
        still counts only the query leaf-match. Process (S2B-r2-06): the item 14 code started inside the frozen
        r4 round; it was reverted at once and restarts after the round closed (2026-10-31), with its own A/B
        review in S3 and the identity suite rerun on the three pandas versions after it lands.
    15. Row 11 (owner ruling 2026-10-31, "NUMPY + split"), in ordered sub-steps, each RED first:
        - (a) `ghost_rearguard_coherence` (`silly_kicks/tracking/_ghost_outfield.py:1165`) over arrays by the D20
          method: the `2474b73` body verbatim as the test oracle; the same groups (`groupby` on game, period,
          frame and team with `sort=False`, `observed=True` and the default `dropna`), the stable `slot_index`
          order inside each group, the ordering test on consecutive `ghost_gr_y` differences, and the minimum
          pairwise distance with NumPy's NaN propagation; identity on fixtures, randomized frames (missing keys,
          ties in `slot_index`, single-slot groups, infinite coordinates) and the public examples
          (`tests/test_public_api_examples.py`);
        - (b) a `coherence` stage after `fit` with 8 row-chunk units (`int_key` 0-7, equal chunks in row order):
          each unit loads the fitted model from the `fit` FileSet (`materialize_fileset` into a worker-local dir,
          `GhostOutfieldModel.load`) and the rows once per worker (`_data`), predicts its chunk with
          `predict_mean` and returns the served slice; the reduce concatenates the slices in key order and writes
          `coherence.json` with (a). `fit` no longer computes coherence; `assemble` reads it from `coherence`. The
          walk is row-independent and the loaded model's trees and baselines equal the fitted ones (the round
          trip `_fit_work` already asserts), so the served frame is byte-identical by construction. Test: on the
          toy corpus, the new pipeline's `assemble` output equals the old pipeline's byte for byte;
        - (c) the stage sizing comes from its measured history (the RAM clamp and S3 item 6 apply);
        - (d) HF identity (kind (c) with stored units): the `assemble` output (`metrics.json` and the bundle) is
          byte-identical to the stored `2474b73` `assemble` output, and the old and new coherence bodies agree
          on the real served frame; the job reports the `fit` stage's D8 job record against the floor bar of item 3
          (more than 4.0 cores at budget 8; S2B-r4-02).
    16. Row 29 (owner ruling 2026-10-31, "Measure a split"). (a) RED: on fixture matches, the per-(match, period)
        units' reduce equals the per-match reduce byte for byte (this also tests that no extracted row reads
        across a period boundary); (b) the receiver `extract` and `pool` ItemStages take one unit per (match,
        period); each unit loads its match shard and extracts that period; (c) an HF measurement (W=8, twice):
        job utilization, and the reduce output hash against the per-match reduce on the same inputs; (d) adopt only
        if both runs reach 0.85 and the hashes match; otherwise revert the stage to per-match units and return the
        row to the owner with the numbers.

### R3-9 (D10): split `ghost_gk` `importance` by column

**Files:** `scripts/train_ghost_gk.py` (`importance` units = column groups; per-column scores via the private
`_calculate_permutation_scores` with the seed derived as the public function derives it); tests.
- [x] RED then GREEN (`tests/scripts/test_ghost_importance_split.py`, 7 tests; RED was the missing helpers):
  `test_importance_split_table_is_byte_identical_to_the_single_unit_table` (against sklearn's public call),
  `test_permutation_private_api_signature_is_pinned`, the reduce over JSON-persisted units, the per-feature-set
  units (a `baseline` unit plus one per corpus column: 26 faithful, 21 position_only), the skipped run, and the
  corpus-width guard. `test_ghost_importance_parallel.py` (P11, `@slow`) now compares the stage OUTPUT at W=1 and
  W=2 and proves >= 2 worker processes. The table rows gain the std column (the importance output feeds no
  artifact: `assemble` only orders after it).
- Maintenance cost, accepted consciously (A plan review PLAN-R3-01): the split depends on sklearn's PRIVATE
  `_calculate_permutation_scores`, the only way to hold byte-identity per column. A sklearn bump can break it;
  `test_permutation_private_api_signature_is_pinned` is the tripwire, and the identity gate catches any numeric
  change. A future sklearn upgrade re-checks both before it lands.

### R3-10 (D12): migrate the three adapter trainers

**Files:** `scripts/train_pass_completion.py`, `scripts/train_xsuccess.py`, `scripts/train_match_outcome_dependence.py`
(each gains a `PIPELINE`: `refs` → `extract` `ItemStage` → its fit stages, listed under "Design per trainer");
`scripts/hf_jobs/adapter.py` and its tests and registry entries (deleted at the end); `scripts/hf_jobs/campaigns/cycle1.toml`;
the ADR-056 population gate and `test_compute_contract` registries.
- [x] Per trainer, RED then GREEN: `test_pipeline_artifact_is_byte_identical_to_the_legacy_cli[<trainer>]`
  (`tests/scripts/test_adapter_migration_identity.py`). Each trainer also gains a `_parity` (W, N) arm, and every
  newly thread-enabled stage its `test_thread_identity` arm and registry entry (B plan review R3-06).
  - OWNER RULING (2026-10-30): no committed golden digest. The legacy CLIs give DIFFERENT numbers across environments
    (on the same toy tree: x86_64 with the `uv.lock` versions against aarch64 with newer ones, the pass-completion
    coefficients and the xsuccess booster differ; evidence `D:\Temp\hfu\evidence\r310\`). So each legacy trainer
    is kept VERBATIM as a test oracle (`tests/scripts/_legacy_trainers/`, git blob ids recorded), and the oracle
    and the pipeline run in the SAME environment on the committed offline open-data tree. Text files the legacy
    CLI wrote in text mode are compared with CRLF folded to LF; `SHA256SUMS` is checked for self-consistency on
    each side. Cost if wrong: none for identity (the oracle runs on every `.venv-hfjobs` leg).
  - Ruling: a pipeline keeps its legacy CLI options and defaults (match_outcome: every open competition unless
    `--competitions-json`, declared by file CONTENT; pass_completion and xsuccess: WC2022), so the registry
    build args do not change.
  - OWNER RULING (2026-10-30): the bundle is FLAT (`adopt.retrieve` refuses a nested path: "an artifact is flat",
    ADR-088) and ships `metrics.json` beside the weights and the card, as the six other campaign-trained bundles
    do. pass_completion: `model.json`, `SHA256SUMS`, `MODEL_CARD.md`, `metrics.json`; xsuccess adds the stamped
    `metadata.json`; `fresh_subdir` becomes `""`. The weights bytes do not change; the wheel folders gain
    `metrics.json` at the next adopt (one added, documented file each). match_outcome is unchanged (no evaluation
    metrics; its `model.json` carries rho and provenance).
  - OWNER RULING (2026-10-30): pass_completion's MODEL_CARD.md names the PIPELINE command in its open-data
    "Reproduce with" line (the legacy CLI is gone); every other byte of the card is identical.
  - OWNER RULING (2026-10-30): the pining cross-check keeps its own `for_each` flow in
    `scripts/_pass_completion_pining.py` (the legacy `main` moved there, pining source only), registered as a
    non-pipeline driver in `test_compute_contract` (I1 forbids the pining client in any HF job).
  - OWNER RULING (2026-10-30): the open-data identity and parity tests run in CI. CI installs statsbombpy at the
    `uv.lock` version in the fast test job AND the slow job, so every open-data test runs there against the
    offline toy tree. Both jobs need it: modules that `importorskip("statsbombpy")` collect differently without
    it, and the slow-shard conservation proof compares the two jobs' node-ID sets (B r3 R4S-r3-03).
  - OWNER RULING (2026-10-30): keep the test-only open-data pin isolation fix (below).
  - Result: `test_adapter_migration_identity.py` 3/3 GREEN in `.venv-hfjobs` (the legacy oracle runs under
    `deterministic_env`, exactly as the one-unit adapter ran it: xgboost's `hist` sums depend on the thread count,
    so an oracle on default threads was not the campaign baseline -- the xsuccess card differed in one digit until
    then); the three `_parity` (W, N) arms GREEN.
- [x] Delete the adapter once no target remains; the population gates stay EXACT. Deleted `scripts/hf_jobs/adapter.py`,
  `tests/scripts/hf_jobs/test_adapter.py`, `tests/scripts/_toy_legacy_trainer.py`, `campaign.ADAPTER_MODULE` and its
  `instances` branch, and the registry `adapter` field.
  - Ruling: `VariantBuild.fresh_subdir` is deleted with the adapter -- its one consumer was the adapter (it named the
    target's artifact dir inside `--out`); every launched trainer now writes a flat bundle, `adopt` passes the
    retrieved dir as the head-to-head `--fresh-dir`, and the `BLOCKED` `gk_retention` entry is not launched. Cost if
    wrong: a future non-flat trainer re-adds one field.
- Design per trainer (the `win_probability` pattern; every stage function wraps the trainer's EXISTING pure
  functions, so the numbers come from the same code):
  - `match_outcome_dependence`: `refs` → `extract` (`extract_match_slice`) → `fit` (`matches_from_shards` in key
    order → `fit_rho` → `model.json` + `SHA256SUMS`).
  - `pass_completion`: `refs` → `extract` (`pass_rows_for_match`) → `cv` (one unit per GroupKFold fold: its
    out-of-fold predictions) → `fit` (the reduce assembles the OOF array in row order, `cross_val_metrics`'
    arithmetic, the final fit, the flat bundle: the `save` files + `MODEL_CARD.md` + `metrics.json`).
  - `xsuccess`: `refs` → `extract` (`onball_rows_for_match`) → `study` (the Optuna TPE study, one unit; a reviewed
    `serial_reason` because each TPE trial reads every earlier trial; a no-op unit when `hpo_trials == 0`, the
    campaign default) → `cv` (folds) → `fit`.
- The corpus of every parity test is the committed offline open-data tree (`scripts/_toy_corpora.build_open_data`:
  real public match events 3754058, 7298, 7584 at their committed bytes, served by
  `tests/scripts/_open_data_server`), so the tests need no network. They need statsbombpy: CI installs it in the
  fast and the slow job (owner ruling above), so these `@slow` tests run in CI as well as in the `.venv-hfjobs`
  legs. The compared artifact is the flat bundle, byte for byte (the legacy `weights/` names flattened; text files
  with CRLF folded to LF; the one card line the owner ruling changes); `metrics.json` is compared field by field
  except the generation token, which is a different hash by design (the `win_probability` migration precedent).
- Rung-1 hygiene found while running the `.venv-hfjobs` leg (fixed test-first): `test_dataset_read_audit`'s
  in-process open-data pin leaked into later tests (it edits statsbombpy's `OPEN_DATA_PATHS` in place, process
  wide), so `test_open_data_pin` failed after it in one session. RED 2 failed / 12 passed in that order; GREEN
  14 passed after an autouse fixture in `tests/scripts/conftest.py` restores the paths after every test.
- Rung-1 hygiene, second find (2026-10-30, the post-deletion gate run: 509 passed, 1 failed): the r3 D11 test
  `test_a_loading_unit_records_its_upstream_load_times` compared a `perf_counter` load with a `time.time_ns()` unit
  wall; on Windows Python 3.12 that clock ticks every 15.6 ms, so a fast unit recorded wall 0.0 under its 5.5 ms
  load. The bound now allows one tick of the `time` clock (`time.get_clock_info("time").resolution`); 3/3 GREEN.
  The ledger is unchanged (`compute_wall_s` already clamps at 0; HF's Linux clock resolves nanoseconds).
- The owner-tier pining cross-check (`scripts/_pass_completion_pining.py`, DGX only) cannot become an HF pipeline:
  the I1 gate forbids the pining client in an HF job's import closure. It keeps its own `for_each` flow over the
  same pure functions, so retiring the legacy HF path does not break it (a test pins it).

### R3-11 (D13 + documentation)

- [x] `cycle1.toml` comment states the real role of `shard_cap` (value 20, owner approved 2026-10-30; SUPERSEDED 2026-11-01 — owner ruling `shard_cap=25`, full-utilization sharding this cycle; see the cycle1-launch record).
- [x] ADR-128 amendment for r3; CHANGELOG `Fixed` and `Changed` entries; the runbooks
  (`docs/howto/hf-jobs-training.md`, `docs/howto/compute-pipelines.md`) for the job record, the measured audit and
  the validation ladder. Done 2026-10-30 (the Step 0 outcome, S3, is added when it lands): ADR-128 amendment
  2026-10-30 (r3 and r4); CHANGELOG 4.140.0: the wheel note, `Added` (measurement jobs), `Changed` (D2, D8,
  D10-D12, D16-D20), `Removed` (the adapter), `Fixed` (r3 D1, D3-D7); `docs/howto/hf-jobs-training.md` (run keys
  and shared tier stores, job labels, the job record, measurement jobs, the validation ladder);
  `docs/howto/compute-pipelines.md` (reading an upstream's units, one extraction per source).

### R3-12 (D2, D11): owner approved 2026-10-30

- [x] D2 (spec D2 (b)): the profile times its sample's read-back and combine; the planner estimates the reduce
  wall as units × per-unit cost ÷ read workers, shards when the estimate exceeds the per-job budget, and sizes the
  reduce timeout from it. Test: a fresh store with a heavy shardable reduce plans the sharded path.
- [x] D11 (spec D11 (a)+(b)): `RunContext.upstream` times fetch, verify and decode; `run_unit` records them in the
  unit's ledger record; the unit cost excludes them; the §G-2 bound charges each extra job the measured overhead.
  Tests: `test_a_loading_unit_records_its_upstream_load_times`,
  `test_a_unit_reusing_the_loaded_upstream_records_no_load`, `test_the_unit_cost_excludes_the_upstream_load`,
  `test_the_split_cost_bound_charges_the_measured_overhead_per_extra_job`.

### R3-14 (D16): the `cv` thread opt-in that gives no speedup

- [x] Measure on HF hardware (read-only canary job `6ae4a0dfc78863f9c62e806e`): the 8-thread fold is 1,075 s, of
  which `predict_mean` is 1,012 s at 1.00 core (no numba in the HF env: the numpy fallback leaf walk); the HistGB
  fit is 50 s at 7.33 cores; the upstream load is 2.75 s.
- [x] RED then GREEN: `tests/tracking/test_ghost_leaf_walk.py` (the old numpy body kept verbatim in
  `tests/tracking/_ghost_leaf_reference.py` as the oracle; values and indices `np.array_equal` at chunkings that cut
  rows unevenly; the asymmetric depth-cap contract; both models end to end, training leaves included). Also checked
  locally on the REAL committed bundles' trees (500 trees, depth 8, ties, NaNs): identical.
- [x] HF measurement job `6ae4ab410e0338d1bb7cb791` (pushed base + working-tree diff): the predict of a real
  `ghost_outfield` fold fell from 1,012 s to 52 s serial. A thread pool over the chunks measured SLOWER at 8 threads
  for `ghost_outfield` (66 s; GIL-bound) and 2.2 s faster for `ghost_gk` (11.9 s against 14.1 s); it was removed
  with its `n_threads` keyword (OWNER RULING 2026-10-30, after the round-5 A/B disagreement: serial; the `ghost_gk`
  gain is given up for one code path). The `cv` shape is decided in the R3-8 decision table from
  these numbers (fold 300 s at 1 thread, 160 s at 8 threads: folds in parallel win).
- [x] HF identity: canary job `6ae4aee00e0338d1bb7cb8a7` compares, on the real fold 0, the old body vs the flat walk
  on the same fit, and both trainers' `_cv_work` results (encoded by `_encode`) with the 2474b73 campaign's stored
  bytes. (Canary 2 reported a mismatch that was its own bug: it compared `_canon_json`, which sorts keys, with the
  store's insertion-ordered `_encode` bytes.)
- [x] ADR-076 amendment (the serial flat walk; numba stays out of the training env, and why).

### R3-13 (D14): the validation ladder (spec r3.4)

- [ ] Rung 1: the full test suite and the CI lint trio, green.
- [x] Rung 2 (HF measurement job, no push; owner ruling 2026-10-30): `win_probability` `extract` with all 3,961
  units and its reduce, timed against the per-job budget on the mount.
  - Done 2026-10-30 (R3-8 S1 / S1b / S1e). The extract at W=8: job wall 249 s (round 2) and 327 s (S1e); at W=12:
    263 s; at W=16: 228 s. The reduce over the mount: 157.3 s (the pooled read 154.5 s, the combine 1.1 s, the encode
    1.8 s; 71.7 MB, 7,974,436 rows). Both are far under the per-job budget (30 min target, 6 h orphan sweep age).
- [ ] Rungs 3 and 5 need code on HF: an A/B implementation review of the whole tree (B plan review R3-07) →
  OWNER GATE (intermediate amend) → OWNER GATE (force-push) → CI green.
- [ ] Rung 3: the committed real-mount canary (durability through the bucket API, read cost per file, the reduce
  at real scale with one real-scale write, cross-container mtime, writer and reader on one tree).
  - Run ahead as a measurement job, 2026-10-30 (the uncommitted tree; owner ruling 2026-10-30: the canary bucket
    `silly-kicks/hf-refit-canary`, private, created for it, is the job's ONLY mount; no secrets). Writer (job
    `6ae54e8d0e0338d1bb7d058d`, read-write): the toy pipeline's `gen`, 2,000 units through the production executor
    at W=8 and the production store on the mount (47.8 s), then `combined` (reads every unit back, 22.7 s); in the
    same container 2,000 of 2,000 units have their result, a sha256 sidecar equal to the result and a ledger; 0
    orphan `.tmp-*` files; 64 mtime probe files written through `LocalStore.write_bytes`. Reader (a FRESH
    container, read-only, job `6ae552100e0338d1bb7d06df`): the same 2,000 of 2,000 units and `combined`, 0 orphan
    temps, both `_output` files present; all 64 probe files present with the right sha256, and the mtime it reports
    is 0.14 to 15.96 s after the writer's write call returned (median 8.05 s; 64 distinct values; none before the
    write, none a placeholder). The D6(b) sweep age is 6 h and it ignores an mtime before 2020, so a 16 s lag is
    safe. Read cost per file and the reduce at real scale: R3-8 S1b/S1c (0.183 s per file serially, about 42 ms
    per file pooled, the 3,961-unit reduce in 157.3 s). Ruling: the first reader (job `6ae55186c78863f9c62eeaea`)
    recomputed the generation tokens from its own tree, which differed from the writer's by one annotation in
    `scripts/_toy_corpora.py`, so it looked at an empty generation; the reader now takes the writer's generations
    by listing the store. Cost if wrong: none; the second reader is the evidence. The committed rerun of this canary
    is S3 item 13 and stays at its place in this ladder.
  - Scope of this run-ahead (S2 review S2B-C05): both readers listed and read through a FRESH container's mount
    (`rung3_canary.py` `verify_units`, `os.listdir`), not through the bucket API; the writer ran the writer patch
    (`8627e143…`) and both readers the reader patch (`98007257…`); the only output written through the mount was
    the toy `combined`, and S1b/S1c (read-only jobs) measured reads, not a large `_output` write. So durability
    through the bucket API, one tree for writer and reader, and a real-scale write are NOT yet shown; S3 item 13's
    committed canary covers all three.
  - Done 2026-10-31 on `93c572c` (CI green): writer job `6ae685860e0338d1bb7d4d7a` (canary mode, the canary bucket
    the only writable mount, no secrets): `gen` 2,000 units at W=8 46.5 s, `combined` 80.9 s, 2,000 of 2,000
    verified, 0 temp files, 64 probe files, the 106.8 MB `ghost_gk` `cache` `_output.parquet` copied
    (sha256 `e428f8ba…`); fresh-container mount reader `6ae6898d0e0338d1bb7d4e73`: 2,000 of 2,000, 64 of 64 probe
    files with mtime 2.36-18.39 s after the write (median 10.3 s), the real file sha-equal; bucket-API reader (this
    machine): 2,000 of 2,000, 64 of 64, the real file sha-equal. Durability through the bucket API, one tree for
    writer and readers, and a real-scale write are shown.
- [ ] Rung 4: the planner history committed where `committed_history` reads it, taken from the HF canary
  campaign's summary (rung 5) with `history_key` and host, never from the DGX (PLAN-R3-04). It is committed with a
  SECOND owner-gated amend (OWNER GATE amend, OWNER GATE force-push, CI green; review R5B-07).
- [ ] Rung 5: the committed canary campaign (one instance per module, the 3,961-unit extract and its reduce, a
  variant pair, a long-lived `--watch` multi-unit `cv`, a forced-halt arm).
  - Design `docs/superpowers/specs/2026-10-31-hfu-rung5-canary-design.md` (A and B APPROVE, round 3; owner rulings
    2026-10-31: "Design, review, build, run"; 13 instances; the ERROR-in-a-row stop is an instance halt). Built
    test-first: `scripts/hf_jobs/campaigns/canary.toml`, `registry.CanarySpec` / `CANARY` / `canary_spec`, the toy
    `hold` stage, the canary-only guards, `--reports`, `canary-check`, `campaign.errors_exhausted` and the halt in
    `step`, `summary_data` (the canary rows left out); `tests/scripts/hf_jobs/test_canary_campaign.py` (24, RED then
    GREEN). Owner gates before the run: the implementation A/B review, amend, force-push, CI green, launch.
  - Run 1 (2026-11-01, commit `a4aa128`; preflight 6 of 6 ok): the controller stopped at its third `--watch` step
    with `PermissionError: controller uploads are limited to _campaign/ and _partitions/, got
    '_unit_peaks/xshot_occurrence/extract/80b8c176a275ff76.json'`. The first gated (MEM) run stages its per-unit
    peaks through `BucketStore.upload_bytes`, whose prefix guard did not list `_unit_peaks/`; the unit test used a
    fake store with no guard. All 22 launched jobs ended `COMPLETED`; no job ran on. Fix, test-first: the guard
    reads `store.CONTROLLER_UPLOAD_PREFIXES` (`_campaign/`, `_partitions/`, `_unit_peaks/`), and
    `test_the_gated_run_stages_its_peaks_through_the_real_bucket_store` drives the gated staging through the real
    `BucketStore` behind its `StoreSnapshot` (RED: the HF error; GREEN). Review R5-FIX-01:
    `test_every_other_controller_upload_passes_the_real_bucket_store_guard` sends the other three controller
    uploads (the LPT partition, the preflight report, the preflight cache) through the real guard; dropping any one
    prefix fails exactly the test that covers it. The canary ran again at the new commit `3244962`.
  - Run 2 (2026-11-01, commit `3244962`; preflight 6 of 6 ok): the jobs launched, the forced-halt arm's `strict`
    job ended ERROR as designed, and no real instance failed. The controller's first `--watch` step took ~15 min;
    a read-only timing on MEDIA-PC (`unit_peak_views` 960 s wall) put 98 % in the per-unit-peak bucket search
    (`bucket_unit_peaks`), 591 s of it in the `win_probability` `extract` call alone. Root cause: the search read
    the unit ledgers of EVERY matching reduced generation (13 `win_probability` run keys x 3,961 ledgers) only to
    pick the newest by `finished_ns`. Fix (owner ruling 2026-11-01 "Fix now, restart canary"), test-first:
    - **Newest-first.** `_stage_generations`/`_reduced_generations` carry each generation's newest ledger upload
      time; `bucket_unit_peaks` picks the newest candidate by that time and downloads ONLY its ledgers, never the
      losers'. This assumes upload order == finish order, which holds for distinct earlier campaigns' run keys (the
      usual candidates); concurrent generations of one computation (INV-1 replica run keys `-r<N>`, review
      R5-OPT-02) can differ, but the peaks size only the memory gate (W) and the output is byte-identical per
      (W, N), so either replica's sample is sound. The exact `finished_ns` scan stays as the fallback when a
      snapshotted candidate carries no upload time. A scale equivalence run (MEDIA-PC, real bucket) matched the two
      over 74 gate-candidate stages, 0 mismatches.
    - **Parallel.** `unit_peak_views` searches the misses in a `STEP_WORKERS` thread pool; a single-flight lock on
      the shared `_reads` keeps each store listed and downloaded once (`_reads_cached`, review S3B-12).
    - Measured on MEDIA-PC (empty cache, real bucket): `unit_peak_views` 941 s -> 349 s (newest-first) -> 149 s
      (+ parallel); `win_probability` `extract` 591 s -> 85 s. The `win_probability` extract keeps 3,934 peaks
      (its newest generation's done units), unchanged by the fix. Tests: `test_step_snapshot.py` gains the
      newest-first download-pruning, the fallback, and a controlled-race single-flight test (RED proved by
      disabling single-flight: two concurrent searches list the store twice). The canary runs again at the next
      commit.
  - Run 3 (2026-11-01, commit `88a3dd88`; controller on DGX, preflight 6 of 6 ok): 12 of the 13 real instances
    reached `done`; the forced-halt arm's `strict` job ended ERROR as designed, but its sibling `hold` job (a
    3,600 s sleeper, C3) was NEVER cancelled and ran ~47 min while `--watch` looped on an identical report for 40
    steps. Root cause: `strict` has 6 units, so it is PROFILED; the profile sample includes the failing unit
    `0000`, so the profile JOB crashed (exit 3) before `strict` could run and record `0000` as a failed unit — the
    designed clean failed-units halt (`max_failures=0` -> `Action("halt")` -> cancel `hold`) was never reached.
    `step` then treated the bare profile ERROR per the per-stage-independent design (T5 S3 / B-12): report the
    stage failed, do not halt, do not cancel, so `hold` was orphaned (uncancelled); `all_settled` treats a
    `failed` entry as terminal, so the arm itself would have exited 1 with `hold` still running once the other
    instances finished — the 40-step loop observed on the launch was `match_outcome_dependence` still running, not
    the halt arm (review HALT-B-04). The first fix
    attempt (halt on any non-resumable ERROR) broke `test_concurrent_stage_resume_is_per_stage_independent`;
    designing the reconciliation showed that "cancel live siblings AND keep dispatching pending ones" ping-pongs
    (a dispatched pending stage becomes a live sibling next step, so it is cancelled, relaunched, cancelled). Fix
    (owner ruling 2026-11-01, two AskUserQuestion rounds -> "Halt instance on non-resumable error"), test-first: a
    stage whose latest job ended in ERROR with nothing left to resume (`resume_after_error` returns its halt
    default) now halts the INSTANCE, named, and its live siblings are cancelled through the failed-units cancel
    path, the same contract `errors_exhausted` already states (`campaign.step`; RED: `r["action"]` was `run`, not
    `halt`, and the live `hold` was not cancelled — GREEN:
    `test_a_nonresumable_error_halts_the_instance_and_cancels_its_live_sibling`,
    `test_a_stage_whose_latest_job_ended_in_error_with_nothing_to_resume_halts`,
    `test_an_error_with_nothing_retryable_still_halts`). T5 S3's intent (per-stage cursor independence of RESUME)
    is kept by making its test exercise a RESUMABLE error (a retryable failed unit), which resumes per stage
    without halting while its independent siblings still dispatch. A/B round 1: A APPROVE (2 CONSIDER); B REQUEST
    CHANGES, HALT-B-01 BLOCKING — the halt reason `job <id> ERROR` was not an approved `_CANARY_HALT_FORMS` form,
    so `canary-check` `halt_arm` would still fail. Owner ruling 2026-11-01 (Route B): make the non-resumable halt
    reason stage-named (`strict: latest job <id> ended in ERROR, nothing to resume`), add it as a third accepted
    `_CANARY_HALT_FORMS` form (spec C4), and add the reachable single-crash canary test
    (`…[single crash, no record]`); the other findings (profile-ERROR-under-`only` test, the stale
    `test_watch_not_settled…` docstring note, this wording) were fixed in the same round. The canary runs again at
    the next commit.
  - cycle1 launch (2026-11-01, commit `0f792151`): the full retrain ran on HF (preflight cache-hit; clean
    provenance -- every job pinned + asserted `0f792151`, clean tree, run keys embed the commit). 130 jobs
    COMPLETED, 8 instances `done`, before `xcross_attempt-xshot115` halted on a transient HF
    `Volume mount failed` (hf-vol-2 rpc error): the halt fix treated the mount failure as non-resumable and killed
    the instance. Owner killed the run ("kill everything"). Two defects surfaced. (1) The campaign shipped
    `shard_cap=20` -- owner-approved 2026-10-30 (R3-11/D13, "sharding is Cycle 2") -- which the owner SUPERSEDED
    2026-11-01: `shard_cap=25` (full-utilization sharding this cycle). Set to 25 (`cycle1.toml`, `canary.toml`,
    `test_cycle1_caps` gate, R3-11, the design docs). With `dispatch_cap=25` the back-off floor rises to 25 (no
    range: the campaign halts at 2 never-started timeouts, owner-accepted and pinned by a test, INF-B-01). (2)
    A transient infra fault (container NEVER started, `started_at` None) must RETRY, not halt. Fix (owner ruling
    2026-11-01 "make the fix properly"), test-first: `job_never_started(info)` (started_at present-and-None);
    `job_never_started_timeout = _timed_out and job_never_started` (dispatch back-off unchanged); and
    `resume_after_error` resumes a never-started latest job with pending units, bounded by `error_resumes` (a
    mount failure is COUNTED, so a persistent one halts after 2; a deterministic crash's `started_at` is set, so
    it still halts). RED: `test_a_never_started_infra_error_resumes_not_halts` halted;
    `test_repeated_never_started_infra_errors_halt_via_errors_exhausted` guards the bound. Deferred per the owner
    ("even if not optimized, does it work"): the R1 audit ONLY (~19 stages under 8 vCPU, largely IO-bound or
    single-threaded -- inherent). The orphan-temp sweep is NOT deferred -- it is the specified D6 behavior
    (`scripts/_stages.py` `sweep_orphan_temps`), which runs after a reduce of a generation. The 178 `.tmp-`
    orphans were the KILLED run's in-flight temps (spared by design); run keys embed `commit[:7]`, so the
    relaunch (a new commit) never reduces the `…-0f79215-…` generations and these specific files PERSIST --
    harmless inert `.part` files, never re-read (reviews INF-B-03, INF-B-06). `shard_cap=25` splits more, which
    helps R1 for splittable stages.
  - Run 3 (2026-11-01, commit `bc6a246`): preflight re-ran the ~30-min toy `check_plans` train AGAIN -- as on
    `a4aa128` and `3244962`. Root cause (owner flagged the repeat as unacceptable): `check_plans_key` hashed
    `scripts/**/*.py`, so every amend to a controller-only file (`campaign.py`, `store.py`) busted the cache key
    although the toy train was byte-identical (the trainers, `silly_kicks/` and `uv.lock` never changed). Fix (owner
    ruling 2026-11-01), test-first: the key hashes ALL of `scripts/**/*.py` + `silly_kicks/**/*.py` + the canary
    job modules EXCEPT an explicit controller-only list (`_CONTROLLER_ONLY_FILES`: `campaign`, `launch`, `ledger`,
    `adopt`, `measure`, `dgx_e2e`, `scaffold`, `headtohead`, `__main__`). The whole library and every
    trainer-reachable file stay hashed, so the key can NEVER under-hash code the train loads; a controller-only edit
    does not change it. The exclusion is gated by the static import closure of the registry + canary entry modules
    (`_repo_import_closure`, following every import incl. in-function, relative, ancestor packages and bare
    `sys.path` siblings), which over-approximates what a `--plan`/`--run-all` loads and must reach none of the
    excluded files. A/B round 1 BLOCKED the first attempt (closure-AS-key under-hashed the `vaep`/`xtgk` families,
    32 files, because it missed ancestor `__init__.py`); inverting to hash-all-minus-exclude removes that class.
    Round 2 (B, CK-B-06): the key also digests the checked instance `(module, args)` (built by the excluded
    `campaign.instances()`), so a changed `--plan` argv re-runs the train though `campaign.py` stays excluded; the
    gate's bare `sys.path` sibling resolution was fixed (CK-B-07). Costs one more train on the fix commit, then
    stable across controller amends.
- [ ] Rung 6 (cycle gate): whole-branch A/B implementation review → OWNER GATE (amend) → OWNER GATE (push) → CI
  green → the full campaign → `verify` (R4 identity, measured R1 audit) → OWNER GATE (adopt).

## Revision r4 tasks (spec § "r4 — the optimization audit", 2026-10-30; owner-approved, A/B review pending)

The audit's four items, D17–D20, in the SAME working tree under the same Global Constraints. Every task is
test-first (RED shown, then GREEN). The spec's no-rebaseline kind (c) applies to R4-3…R4-9: the old code is kept
verbatim in the tests as the oracle, and R4-10 reproduces every stored `2474b73` unit on HF.

Order: R4-1 → R4-2 → R4-3 → R4-4 → R4-5 → R4-6 → R4-7 → R4-8 → R4-9 → R4-10, inside the one order line of the r3
section above (then R3-8: Step 0 measures the NEW stage shapes). R3-10 is independent of r4.

### R4-1 (D17): work keyed by `(run_key, generation)`

- Files: `scripts/hf_jobs/launch.py` (the `gens` label: 8-hex generation prefixes in stage order, refused over
  `LABEL_MAX`; the planner refuses a prefix collision within one store),
  `scripts/hf_jobs/campaign.py` (`_running`, `_error_history`, `_job_records`, the D5 key, `step`,
  `_cancel_instance`, `cmd_status`, `cmd_summary` / `stage_summary`, the R1 audit attribution);
  `tests/scripts/hf_jobs/test_campaign.py`, `tests/scripts/hf_jobs/test_launch_typed.py`.
- Tests (RED first), two instances sharing a generation, one per launcher state:
  `test_a_shared_generation_launches_once_per_step`; `test_a_shared_stage_in_flight_is_not_relaunched`;
  `test_a_shared_stage_is_relaunched_when_its_launcher_halts_on_its_own_stage` (the launcher's other stage halts
  while the shared stage is in flight; its shared jobs are NOT cancelled while the other instance needs them);
  `test_a_plan_error_in_one_instance_does_not_duplicate_a_shared_launch`;
  `test_a_refused_resume_on_a_shared_stage_halts_every_instance_that_needs_it`;
  `test_a_shared_stage_out_of_attempts_halts_every_instance_that_needs_it`;
  `test_every_case_settles_the_watch` (`all_settled` reaches true); `test_a_waiting_instance_takes_no_budget_share`;
  `test_a_shared_stage_is_audited_once_under_its_launcher`; `test_the_job_labels_carry_every_stage_generation`;
  `test_a_gens_label_over_label_max_is_refused` and `test_a_generation_prefix_collision_is_refused`;
  `test_a_halting_instance_defers_a_possibly_shared_cancel_while_another_plan_errors` (B r4-spec-r2 CONSIDER).
- [x] Done 2026-10-30. RED: 15 failed for the missing feature (a shared stage launched twice, a shared job
  cancelled, a share of 4 where 9 was free, no collision refusal, no `gens`); 5 passed, among them
  `test_a_shared_stage_out_of_attempts_halts_every_instance_that_needs_it`, which holds before the change by design
  (unit attempts are store-derived, as the spec says) and stays as a regression pin. GREEN: the controller suites
  190 passed (`test_campaign`, `test_launch_typed`, `test_ledger`, `test_step_snapshot`, `test_dispatch_overflow`).
  Three tests beyond the list: `test_a_zero_progress_stall_on_a_shared_head_halts_every_instance_that_needs_it`,
  `test_a_shared_row_never_masks_its_launchers_history`, and the shared-row summary case.
  - Ruling: a job counts in the history of EVERY generation it carries; an action reads the history of its FIRST
    stage's generation (the stage `resume_after_error` already judged), and the D5 progress mark -- the job's
    `records` label and the step's check -- covers that generation only. A mark over a whole grouped action differs
    between instances whose groups share only their head, so they would not halt in the same step. Cost if wrong:
    a grouped job whose head stalls while a tail stage commits halts after its two settled jobs, as a lone head would.
  - Ruling: the `gens` label is required. `labels_for` refuses a job without one generation per stage, and
    `launch.gens_of` refuses a listed campaign job without the label (a job the controller cannot key could have
    its work launched twice). Cost if wrong: a job launched by hand with campaign labels stops the step, named.
  - Ruling: an instance takes a share of the step budget only if it has a ready, unreduced stage that no live job
    and no instance before it holds (the D17 budget rule, which also covers an instance with no ready stage).
  - Ruling: a shared summary row keeps its store-derived fields, gains `shared_with` and `audit: None`, and
    `ledger.load_history` skips it, so a later shared row never replaces the launcher's row (or its startup).
  - Ruling: the TF-85 step-world fixture gave two generations of one store the same 8-character prefix
    (`t_extract`, `t_extract_po`); the second is now `po_extract`. Its assertions are about reads, not tokens.
  - OWNER RULING (2026-10-30): the step budget subtracts every LIVE JOB of the campaign, as spec §C states
    (`dispatch_cap` caps total jobs in flight) and the `Campaign.dispatch_cap` field says. Since `b2865a7` it
    subtracted one per running stage, so one stage running 20 jobs held 1 of 25 slots.
    `test_the_step_budget_subtracts_every_live_job_not_every_live_stage`: RED `9 == 4`, then GREEN.
  - Verification: the R3-10 + R4-1 leg in `.venv-hfjobs` (`tests/scripts/hf_jobs`, `test_compute_contract`, the
    three trainers' unit, identity and parity tests, `test_match_outcome_calibration`): 543 passed, 2 skipped;
    the CI-pinned lint trio (ruff 0.15.7, pyright 1.1.409, pandas-stubs 2.3.3.260113, numpy 2.5.2): clean.
  - Two findings of that run, fixed: R3-10 had deleted `_patch_open_corpus` from `test_match_outcome_train`, but
    `test_match_outcome_calibration` imports it (restored, marked shared); whole-repo pyright with scipy 1.18
    inferred `minimize_scalar`'s result in `fit_rho` as `object` (a single-file run did not), so the result type is
    declared with `cast(OptimizeResult, ...)` (runtime unchanged; the identity test is GREEN).

### R4-2 (D19): `ghost_gk` `cache` reads in a pool

- Files: `scripts/train_ghost_gk.py` (`_cache_work`); a new test in `tests/scripts/test_ghost_importance_split.py`
  or its own module.
- Test (RED first): the cache result equals the serial reference result byte for byte (`_stages._encode`), and
  the unit reads go through `_read_units` (a spy), never `read_result` one key at a time.
- Test (RED first; B r4-spec reviews R4S-05, R4S-r2-01), through the real readiness logic (`plan`,
  `next_actions`): a corrupt upstream unit, and an upstream unit lost after the reduce, each reopen the upstream
  stage (the unit and the upstream `_output` deleted); the unit is redone; the upstream `_output` is
  byte-identical to the clean run's; `cache` records no failure and keeps its attempts (`run_unit` re-raises
  `UnitsInvalidatedError`); its result equals the clean run's; `cache` never builds a corpus from fewer units than
  the upstream's reduce combined (its sorted units minus the `excluded` and `failed_keys` of its `_summary.json`;
  B r3 R4S-r3-01), and a failure the upstream tolerated (`max_failures > 0`) reopens nothing (both sides).
- [x] Done 2026-10-30. `_stages.combined_units` / `read_combined_units` (the units the `_summary.json` says the reduce
  combined, read through `_read_units`); `_read_units` deletes the stage's `_output`, its sidecar and `_reduce.json`
  when it finds a broken unit of a REDUCED stage; `run_unit` re-raises `UnitsInvalidatedError`; an executor worker
  child exits with `_RC_UNITS_INVALIDATED` on it and the parent ends the stage without a relaunch (the job exits 0, as
  D6(d)). RED 5 failed (the reader recorded a failure; `cache` read no pool); GREEN
  `tests/scripts/test_reopen_on_breach.py` 4 passed and `tests/scripts/test_ghost_gk_cache_pool.py` 1 passed (the
  `2474b73` `_cache_work` kept verbatim as the oracle, the same `_encode` bytes on the toy ghost_gk tree). The toy
  pipeline gained a `combined` stage that reads `gen`'s combined units.
  - Ruling: the executor path (a worker child, then the parent) is part of D19, though the spec names only `run_unit`:
    without it the parent relaunches the dead worker until `max_relaunch` and the job ends in ERROR, which halts the
    instance. Cost if wrong: none -- the D6(d) exit code already existed for the reduce child.
  - Ruling: the two subprocess-heavy tests are `@slow` (12 s and 18 s measured; ADR-023).

### R4-3 (D18 a): the `ghost_gk` superset `extract` + a per-variant `select`

- Files: `scripts/train_ghost_gk.py` (`extract` work and declare; new `select` `ItemStage`; `cache` and `assemble`
  read `select`); `tests/scripts/_ghost_gk_extract_reference.py` (the old `_extract_work` verbatim, the oracle);
  `tests/scripts/test_ghost_gk_superset_select.py`; the population gates (`test_compute_contract`, parity arms).
- Tests (RED first): for every variant (`default`, `position_only`, `sweeper`, `sweeper_position_only`, `full`) and
  every fixture unit, `select` over the superset unit READ BACK FROM THE STORE (`write_result` → `read_result`)
  equals the oracle's unit frame (`DataFrame.equals`, dtypes, column order) and its counters dict. Fixture units
  (A R4-SPEC-01, B R4S-03): an empty grid-filtered game, an all-null visibility (`select` fails closed, with
  `max_failures=0`), an unmapped unit (excluded by `extract` and by `select`, same reason and details), and a unit
  whose games all lose their keeper frames (no result). Also: the `extract` generation is identical across the
  five variants; a missing or corrupt superset unit reopens `extract` (the D19 rule; B r3 R4S-r3-02) and no
  variant selects from a smaller corpus; the `select` generation differs by `feature_set` and by `grid_x_max`; a variant grid outside
  the superset grid refuses by name; `adopt`'s `retrieve` reads the counters of the one counters-bearing stage;
  the end-to-end toy run's `assemble` bundle is byte-identical to the pre-change bundle.
- [x] Done 2026-10-30. `extract` is the superset (`_SUPERSET_FEATURE_SET`, `_SUPERSET_GRID_X_MAX = 52.5`, the
  `ghost-gk-superset-1` schema marker, `cross_commit_version` `ghost-gk-extract-2`, no counters); `select` is a new
  `ItemStage` (`_select_load` reads the superset unit back through `_stages.read_upstream_unit`, or replays
  `extract`'s exclusion; `_select_work` is the old tail; counters; `max_failures=0`; `ghost-gk-select-1`); `cache`
  and `assemble` read `select`; `check_variant_grid` refuses a grid outside the superset. The plan JSON gains a
  per-stage `counters` flag. Oracle: `tests/scripts/_ghost_gk_extract_reference.py` (the `2474b73` `_extract_work`,
  git blob `eb0cb293`). Tests: `tests/scripts/test_ghost_gk_superset_select.py` (all five variants x every unit,
  `DataFrame.equals` + dtypes + column order + `_encode` bytes + the bias counters; the four fixture units; the
  shared/per-variant generations; the grid guard; the replayed exclusion; and, `@slow`, the cache bytes plus the bias
  counters against the old path on the toy tree), two `test_adopt` tests. GREEN: 38 passed; the wider ghost_gk and
  contract gates 585 passed after two test updates (below).
  - Ruling (process): the implementation was written before its RED was watched. Corrected the same day: the new
    tests ran against the `HEAD` trainer and `adopt` swapped in from git (10 failed, all for the missing feature;
    the working files restored and compared byte for byte), and a mutation run (bias depth from `gk_y`, no
    `position_only` projection) failed all five variant cases.
  - Ruling: the superset stores the keeper visibility as a nullable `boolean` column. `fillna(False).astype(bool)`
    and `isna().all()` give the same mask and the same fail-closed verdict for every value the frames carry, and a
    raw object column of bools, `None` and `NaN` is not always storable. Cost if wrong: a fail-loud `TypeError` at
    extract on an unexpected visibility value, never a silent change.
  - Ruling: `adopt` records the counters of EVERY counters-bearing ItemStage, by stage name. The spec says "the
    ONE counters-bearing ItemStage", but `receiver_model` carries counters on two (`extract`, `pool`); the ADR-126
    integrity gate reads its families' `extract` by name. Cost if wrong: none for ghost_gk (its record counters feed
    no gate).
  - Ruling: the end-to-end proof is the `cache` output bytes plus the `select` bias counters equal to the old path's;
    `cv`, `fit`, `importance` and `assemble` read nothing else of the extraction, so the bundle is identical. The
    full new pipeline runs end to end in `test_parity_ghost_gk` (W1N1 = W2N2). Cost if wrong: R4-10 compares every
    stored `2474b73` unit on HF.
  - Test updates: `test_train_ghost_gk_sweeper.test_grid_x_max_keys_the_shard_generation` (the 4.77.1 rule now
    keys `select`; the superset grid holds every variant's rows) and
    `test_trainer_cache_and_providers.test_main_calls_validate_corpus_visibility_before_extraction` (the keeper filter
    is `select`'s; `extract` and `select` both wait for `validate`).

### R4-4 (D18 b, d): `position_only` as a column subset; one `probe`

- Files: `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py`, `scripts/train_ghost_outfield.py`
  (extract declares; the one consumer projects); their parity tests.
- Tests (RED first): the `extract` (and `probe`) generation is identical for `default` and `position_only`; the
  projected frame equals the old `position_only` extract output (oracle: the old extract run with
  `feature_set="position_only"` on the fixtures); the toy bundles of both variants are byte-identical to the
  pre-change bundles.
- [x] Done 2026-10-30. Each trainer's `extract` (and the xcross `probe`) passes the faithful feature set; each gains
  `_extract_for(ctx)`, the extract output as the variant's consumer reads it (`prep`; ghost_outfield's `_data`, now
  keyed by store, extract token and feature set). Oracles: `tests/scripts/_{xshot,xcross,ghost_outfield}_extract_reference.py`
  (the `2474b73` functions, git blobs recorded). Tests: `tests/scripts/test_position_only_projection.py` (the shared
  generations; for both variants of each trainer, `_extract_for` equals the oracle's extract output -- `DataFrame.equals`,
  dtypes, column order, `_encode` bytes -- on the toy trees; the xcross probe cohort bytes). RED 6 failed for the missing
  feature (the probe test passed pre-change by construction: both sides then ran `position_only`); GREEN 49 passed with
  the three trainers' unit and parity suites.
  - Ruling: the extract still declares `feature_set`, as the shards' constant `"faithful"` (not removed): the token then
    equals the `2474b73` default's, and a change of the extracted columns would still change the generation (4.77.1).
  - Ruling: the bundle proof is the consumer's input, byte-identical for both variants; `prep` / `cv` / `fit` code is
    unchanged and reads nothing else of the extraction. Cost if wrong: R4-10 compares the stored units on HF.
  - Test update: `test_train_ghost_outfield` (`extraction_inputs` takes no variant; every later stage declares `variant`).

### R4-5 (D18 c): tiers that read the same dataset share one store

- Files: `scripts/hf_jobs/campaign.py` (`run_key`, `instances`, `Instance.id`, `_cancel_instance` at `:1424`,
  status, `cmd_summary` at `:1631`, `stage_summary`), `scripts/hf_jobs/launch.py` (the `tier` label; `job_name` at
  `:148`), `scripts/hf_jobs/adopt.py` (`retrieve` staging at `:121`; the arm-restricted corpus identity in
  `verify`), `scripts/_provenance.py` (`env_provenance()["run_key"]` at `:262`: trace whether it reaches a shipped
  file), `scripts/hf_jobs/arm_policy.py` (the per-tier arm restriction), `scripts/train_xshot_occurrence.py` /
  `scripts/train_xcross_attempt.py` (`prep` and xcross `probe_comparison` keep the union of the arm keys and
  declare `arms_json`), `scripts/hf_jobs/campaigns/cycle1.toml` (the public tiers read `xshot115`), and their tests.
- Tests (RED first): the run key is per `(job, dataset)`; instance ids, job labels, job names and staging paths
  carry the tier; one tier's halt cancels none of the other tier's jobs; two tiers on one dataset share the
  `extract` generation (and, with R4-1, launch it once); each tier receives only its arms; `prep` on the shared
  corpus with the public arms equals `prep` on the public-only corpus (toy corpora, byte-identical study inputs);
  the public tier's `probe_comparison` cohort equals the public-only one; the public toy bundle is byte-identical
  to the one built from the public-only dataset.
- [x] Done 2026-10-30. Facts first (offline, from the dataset lock): `xshot17` is a subset of `xshot115`; `xshot115`'s
  arms are public (17) and sc_extended (115), with no `full` arm, so restricting `sc_extended` to {public,
  sc_extended} changes nothing; the two public arms are the same 17 keys. Changes: `run_key(c, commit, job,
  dataset)`; `Instance.id` `{run_key}/{tier}/{variant}`; `JobSpec.tier` and the `tier` label (refused when empty),
  the tier in `job_name` and in the in-job label `label_of` (job records are keyed by it: two tiers run the same
  variant, stage and shard); `_cancel_instance` and `cmd_summary` key by (run_key, tier, variant); the summary
  staging path is `adopt`'s `<run_key>/<tier>/<variant>`; `arm_policy.arms_for_tier` (`TIER_ARMS`; an unknown
  tier refuses); the neutral in-job `arms.restrict_to_arms` / `arm_keys`; `prep` (both trainers) keeps the
  tier's arm rows first; the xcross `probe_comparison` keeps the tier's arm units and declares `arms_json`;
  `cycle1.toml` public tiers read `xshot115`; the manifest's public artifacts gain `read_dataset = "xshot115"`;
  `adopt` checks the revision of the dataset the job READ and sums each counters-bearing stage's counters over
  the tier's arm units only (the ADR-126 integrity inputs). The run key does NOT reach a shipped file: the
  trainers write only `commit`, `dirty` and `tree_state` of the provenance; the env block that holds it rides the
  worker context only. RED 48 failed for the missing feature; GREEN: the R4-5 set 198 passed (incl.
  `tests/scripts/test_tier_arm_restriction.py`: prep and the comparison cohort byte-identical to the public-only
  dataset's, and the WHOLE public toy bundle of xshot and xcross byte-identical); the `hf_jobs` folder, the
  compute contract and the manifest tests 626 passed after two pinned-id updates (`test_step_snapshot`).
  - Ruling: the manifest keeps `dataset` as the corpus identity an artifact records and gains `read_dataset` for
    the dataset its job reads; `test_every_read_dataset_holds_its_artifacts_corpus_as_its_tier_arm` guards that the
    tier's arm of `read_dataset` IS that corpus, match for match. Cost if wrong: an adopt refusal, named.
  - Ruling: the in-job label gains the tier. Without it, job records of two tiers' same-variant jobs share a key and
    the measured-cores audit mixes them. Cost if wrong: committed summaries keep their old labels (no reader parses
    them).

### R4-6 (D20): `_pre_index_frames` without per-frame DataFrames

- Files: `silly_kicks/tracking/_ball_carrier.py`; `tests/tracking/_extract_reference.py` (the old body verbatim);
  `tests/tracking/test_extract_numpy_identity.py`.
- Tests (RED first): every key of the returned dict equal (`np.array_equal` with NaN equality, same dtype, same
  lists/dicts) against the oracle on the committed tracking fixture of EVERY provider and on randomized synthetic
  matches (NaN positions, a missing ball, dead-ball frames, int and str ids, uneven players per frame, two
  periods, float32 storage); `infer_ball_carrier` output `DataFrame.equals` the oracle path.
- [x] Done 2026-10-30. `_pre_index_frames` places the player rows without a DataFrame per frame: rows with a NaN
  key are dropped (as the old `groupby` dropped them), each row maps to its frame by `MultiIndex.get_indexer`,
  one stable `argsort` orders the rows by frame, `bincount` gives the per-frame counts, and the slots fill in one
  vectorized step. Each element keeps its old conversion (`to_numpy(float64)` at the same boundary,
  `pid_to_slot.get` per player id). The oracle is the `2474b73` body verbatim in
  `tests/tracking/_extract_reference.py` (git blob `351cbcef`). RED: the structural test failed on the old body
  ("a DataFrame per frame was built (groupby iteration)"). GREEN: `tests/tracking/test_extract_numpy_identity.py`
  53 passed (the structural test; the identity of every dict key over every provider fixture with and without
  derived velocities, the IDSSE half and 17 synthetic matches; `infer_ball_carrier` end to end). Regression: the
  32 test files that use `_pre_index_frames` or `infer_ball_carrier`, 933 passed, 3 skipped.

### R4-7 (D20): `ghost_gk` features from arrays

- Files: `silly_kicks/tracking/_ghost_gk.py` (`_extract_all_ghost_gk_features`, `extract_ghost_gk_features` as a
  wrapper over the array core); the reference module and the identity test above.
- Tests (RED first): `prepare_ghost_gk_training_data` (both feature sets, both grids, `return_meta`) and the
  serving path equal the oracle (`DataFrame.equals`, dtypes) on the same input classes; the public per-frame
  function's output is unchanged for its callers.
- [x] Done 2026-10-30. `_extract_all_ghost_gk_features` walks `silly_kicks._frame_index.ordered_group_rows` (the
  groupby's own walk as row positions: the iteration's keys from `indices`, its order from `ngroup()`; NA-bearing
  keys from the iteration itself) and reads each column once through `FrameArrays`
  (`silly_kicks/tracking/_frame_arrays.py`, shared with R4-8/R4-9). The keeper cells it hands on (goal-map key,
  velocity-state key, callback argument, meta cell) come from `_KeeperCells`, which reproduces the `iterrows` cells
  (`DataFrame.values` converts to the frame's common dtype first, object for tracking frames). One whole-column
  `ids_match` per keeper team replaces the per-frame calls; the carrier lookup is a dict when the carrier index is
  unique (a duplicated key keeps the old MultiIndex lookup, which returned rows). `extract_ghost_gk_features` is a
  wrapper over the array core `_ghost_gk_feature_row`: `_to_gr_*` per element on the Python floats `Series.apply`
  passed, the back-line width from the same `Series.max`/`min`, the same one-row DataFrames and `pd.concat`. The
  oracle is the `2474b73` bodies verbatim (git blob `ad765b0f`). RED: the structural test failed ("per-frame pandas
  work: a DataFrame per frame (groupby iteration)"); the keeper-cells test failed (no `_KeeperCells`). GREEN: the
  identity module plus `tests/test_frame_index.py`, 157 passed on pandas 2.3.3 and 157 passed on pandas 3.1.0
  (`_extract_all` with recorded callback arguments, `prepare` on both feature sets and both grids with
  `return_meta`, `compute_ghost_gk` and `serve_ghost_gk_positions`, the per-frame function; floats compared bit for
  bit, other cells by type and value). Mutation: the width upcast to float64 fails all 6 float32 inputs; restored.
  - Ruling: `ordered_group_rows` refuses a one-column key (the iteration yields Python scalars there, `indices`
    numpy scalars; no consumer). The pin test caught it, and the pandas 3 `nan`/`<NA>` key spelling. Cost if
    wrong: none (no caller).
  - Ruling: the default suite runs the first 75 frames per period of each provider fixture; the whole fixtures and
    the IDSSE half run in `@slow` (the oracle's per-frame pandas cost is about 6 minutes per identity test on a
    whole fixture). Cost if wrong: a provider-specific case past frame 75 is seen only in the slow job.
  - Ruling: a nullable `is_ball`/`is_goalkeeper` holding a missing value fails the batch call on the first
    whole-column read (the old code failed on the first frame read that held it). On the schema's bool columns every
    read equals the old read. Cost if wrong: an out-of-schema frame set fails earlier or where it once passed.
  - Observation for the owner (not D20's call, unchanged): the `2474b73` ghost extract raises `ValueError: Points
    cannot contain NaN` when a back line of three or more holds a NaN position (`ConvexHull` raises `ValueError`, not
    the `QhullError` it catches). Pinned as-is by `test_ghost_gk_nan_back_line_raises_as_before`.

### R4-8 (D20): `xcross` frame loop and features from arrays

- Files: `silly_kicks/tracking/_xcross_attempt.py` (`prepare_xcross_training_data`, `compute_xcross_attempt`,
  `extract_xcross_features` as a wrapper); the reference module and the identity test.
- Tests (RED first): as R4-7, including `return_meta=True` and `compute_xcross_attempt`.
- [x] Done 2026-10-30. `prepare_xcross_training_data` walks `ordered_group_rows(poss, keys, sort=False)` and
  `compute_xcross_attempt` walks `ordered_group_rows(poss, keys, sort=True, dropna=False)`: each the walk of its old
  groupby. Columns come through `FrameArrays` (`cells` for the cells a loop hands on: the possession team, the
  carrier id, `ball_state`, `time_seconds`). The defending team is `first_other`, the first non-ball team cell that
  differs from the possessing team, which is the head of the old `dropna().unique()` filter.
  `extract_xcross_features` wraps the array core `_xcross_feature_row` over `_XCrossFrames`, which applies the
  row's per-frame reads (bool flags, canonical ids, float64 coordinates) once to the whole columns; each read is
  element-wise (`test_canonical_id_series_answers_element_by_element`, with R4-9). The oracle is the `2474b73`
  bodies verbatim with their `@overload` stubs (git blob `0b4bcec2`). RED: the structural test failed ("a per-frame
  groupby walk: [1, 80, 2, 1, 80] groups, 80 frames"). GREEN: the identity module plus `tests/test_frame_index.py`,
  204 passed on pandas 2.3.3 and 204 passed on pandas 3.1.0 (`prepare` on both feature sets, the wide-area domain on and off, with and without
  `return_meta`, warnings compared; `compute_xcross_attempt` on both feature sets, linked and not; the per-frame
  function with every team and carrier id, a missing and an unknown one, with and without the box detail).
  - Ruling: `prepare_xcross_training_data` refuses a missing `is_ball` up front. The old code masked each frame
    with the raw column (`grp[grp["is_ball"]]`), which pandas refuses for a missing value. Cost if wrong: an
    out-of-schema frame set fails earlier, or where it once passed because its missing values sat in frames the
    filters skipped.
  - Ruling: the serving loop reads its ball flags after the possession check, where the old code first read them.
  - Ruling: the tests compare refusals like values (exception type and message, `_outcome`), because the
    `2474b73` bodies refuse some inputs and the array core must refuse them alike.
  - Observation for the owner (not D20's call, unchanged): `extract_xcross_features(gk_team_id=None)` raises
    `TypeError: boolean value of NA is ambiguous`; `compute_xcross_attempt(link_frame_ids=...)` raises
    `ValueError: invalid literal for int() with base 10: '312.0'` on float frame ids (`int(str(frame_id))`), and
    the committed metrica, skillcorner and sportec fixtures carry a float `frame_id`. The same `int(str(...))`
    sits in `compute_xshot_occurrence`.

### R4-9 (D20): `xshot` frame loop and features from arrays

- Files: `silly_kicks/tracking/_xshot_occurrence.py` (`prepare_xshot_training_data`, `compute_xshot_occurrence`,
  `extract_xshot_features` as a wrapper); the reference module and the identity test.
- Tests (RED first): as R4-7, including `return_keys=True` and `compute_xshot_occurrence`.
- [x] Done 2026-10-30. Both batches walk `ordered_group_rows(poss, keys, sort=True, dropna=False)` (the walk of
  their old groupby) over `_XShotFrames`: the columns read once (`FrameArrays`), the `astype(bool)` flags read once
  on first use, ONE whole-column `ids_match` per defending team. The defending team is `first_other`.
  `extract_xshot_features` wraps the array core `_xshot_feature_row`, which keeps the old order of reads:
  `_geo.to_goal_relative_*` per element on the Python floats `Series.map` passed, then the flags, the ball, the
  team mask. The oracle is the `2474b73` bodies verbatim with their `@overload` stubs (git blob `2c1347c7`). RED:
  the structural test failed ("a per-frame groupby walk: [71, 1, 80] groups, 80 frames"). GREEN: the identity
  module, `tests/test_frame_index.py` and `tests/scripts/hf_jobs/test_measure.py`, 261 passed on pandas 2.3.3 and
  261 passed on pandas 3.1.0 (`prepare` on both feature sets, the attacking-third domain on and off, with and without `return_keys`, warnings
  compared; `compute_xshot_occurrence` on both feature sets, linked and not, warnings compared; the per-frame
  function). `test_canonical_id_series_answers_element_by_element` pins the R4-8 whole-column read.
  - Ruling: the xShot inputs read a missing `ball_state` as `alive`. The metrica and skillcorner fixtures carry
    no ball state, and xShot's training domain keeps only `alive` frames, so without it both paths return nothing
    and the identity is vacuous. Cost if wrong: none to the code; the vacuous case stays equal.

### R4-10: the HF identity run and the r4 measurements

- A read-only HF measurement job (the pushed base plus the working-tree patch carried INLINE in the command,
  gzip + base64, sha256-verified before `git apply`; nothing staged in any bucket). Byte for byte as encoded by
  `_stages._encode` (B r4-spec review R4S-04), the new code reproduces every stored `2474b73` unit of:
  the four trainers' extract generations (the `select` / projected outputs against the stored per-variant
  units); `ghost_gk` `cache`; xcross `probe` and `probe_comparison`; the public-tier xshot/xcross `prep`; and,
  on real units in the same job, the D20 serving paths against their test-only oracles. It records per-unit wall
  and peak RSS for the new extract stages.
- The numbers go into the R3-8 decision table (the extract W comes from the measured peak RSS).
- [x] Done 2026-10-30 for all four families; xShot/xCross with the R4-10a fix, below (probe
  `D:\Temp\hfu\r410\identity_probe.py`; every job
  read-only, no secrets, the patch inline and sha256-checked):
  - `ghost_gk` (job `6ae53f590e0338d1bb7cff81`): 1,805 of 1,805 OK. Per variant (5): the 179 stored extract units
    and their 179 counter sidecars against the new `select` units, byte for byte; the `cache` output; the extract
    `_output` (a reduce SUMMARY) equal except its `stage` and `token` fields, which name the new stage by
    construction. That job printed no SUMMARY line: after the identity phase its serving step hit a probe defect
    (a `ZeroDivisionError` with 0 serving matches; the probe now skips serving at 0), so the count is its 1,805
    `IDENTITY ... OK` lines, 5 variants x 361 comparisons (179 units, 179 counter sidecars, the extract `_output`,
    the `cache` unit and the `cache` `_output`), and 0 MISMATCH lines (B review R4I-02). Serving (job
    `6ae536d4c78863f9c62edc96`): `gradientsports__10502` (175,969 frames) new 23.7 s,
    oracle 481.8 s; `skillcorner__1287842` new 7.9 s, oracle 126.7 s; both OK. New extract: wall p50 13.7 s, max
    31.4 s (the `2474b73` unit was about 197 s); peak RSS p50 2.15 GB, max 5.56 GB.
  - `ghost_outfield` (job `6ae5396b0e0338d1bb7cfd73`): 360 of 360 OK. Extract wall p50 27.8 s, max 71.3 s; peak
    RSS p50 1.55 GB, max 2.53 GB. No serving path (D20 did not touch it).
  - Ruling (harness): the first outfield job (`6ae5367b0e0338d1bb7cfc58`) stopped with "the first 3 completed
    units all failed (systematic)". Its `measure.py` environment lacked `SILLY_KICKS_DATASET_READ_CAP`, so 8
    workers read whole files from the tc3 mount at once: the 2026-10-20 failure mode (EIO at 8 readers). The
    `ghost_gk` job ran 4 workers and passed; the rerun ran 8 workers with the cap at 4 and passed. The fix
    (`measure_env` takes `launch.dataset_env`) and its test (`test_open_data_rides_the_env_and_mounts_nothing`)
    came before the rerun. Production `job_env` always set the cap. Cost if wrong: none; the rerun is the proof.
  - `xshot_occurrence` / `xcross_attempt`: every extract unit, the xCross `probe` and `probe_comparison`, and the
    serving paths are OK (jobs `6ae53ba0c78863f9c62edefa`, `6ae53ba2c78863f9c62edf00`; serving on 2 SkillCorner
    matches each: xShot new 3.8 / 3.5 s, oracle 6.4 / 6.5 s; xCross new 3.8 / 3.6 s, oracle 5.5 / 5.3 s). The
    `prep` comparison found a defect; see R4-10a.

### R4-10a: the xShot/xCross arm keys (owner ruling 2026-10-30: fix it, as a recorded correctness change)

- Finding (HF, job `6ae53e42c78863f9c62ee071`, then the stored-unit probe `6ae54142c78863f9c62ee234`): the new
  `prep` lost the 7 public DFL matches. The locks stage them as `idsse__<game>` with `source_provider` `sportec`;
  the job keys a row `sportec__idsse-<game>`, but `arm_policy.compute_arms` emitted `idsse__idsse-<game>` (built
  from the stem). `restrict_to_arms` (spec r4 D18 (c)) then dropped those rows. The same mismatch was already in
  `2474b73`: there the masks marked the DFL rows False in every arm, so every xShot/xCross final fit
  (`X[arms[shipped]]`) omitted them (public xShot 75,310 of 156,106 rows; public xCross 47,944 of 91,999;
  sc_extended xShot 883,468 of 964,264; sc_extended xCross 539,273 of 583,328), while CV, `n_rows`, `providers`
  and `corpus_n_matches` counted all matches. The wheel and Hub artifacts (4.138.0 run `4095251`) list all 17
  public ids explicitly (`all_public`), so their fits had the rows.
- Owner ruling (2026-10-30): fix it; record it as a correctness change (not byte-identical to `2474b73`).
- [x] Done 2026-10-30 (TDD). RED: `test_every_arm_key_is_the_injob_key_of_a_locked_unit` (the campaign's own
  `--arms-json` against the real locks) failed on the 7 `idsse__idsse-DFL-…` keys; the guard tests failed first on
  the missing import, then, with the guard in `arms.py`, on "DID NOT RAISE" in all three readers. GREEN:
  `compute_arms` takes the locked units and keys each by `match_key(source_provider, pergame_name(provider,
  game))`; `arms.require_arm_keys_in` (`ArmKeyMismatchError`) refuses an arm key that names no staged unit, in
  both trainers' `prep` and the xCross `probe_comparison`, before any row is cut. 48 passed
  (`test_arm_policy`, `test_hf_arms`, `test_campaign_cycle1`, `test_adopt`, `test_tier_arm_restriction` with its
  slow tests). ADR-123 Amendment B.2; CHANGELOG 4.140.0 Fixed.
- HF identity with the fix (jobs `6ae549fcc78863f9c62ee6af` xShot and `6ae549fec78863f9c62ee6b2` xCross, all 4
  instances of each): the `prep` members `X` (`X.parquet` for xCross), `y`, `groups`, `match_ids`, `providers`
  and `design_matrix` are byte-identical to `2474b73`; `arms/*.npy` and `is_public.npy` differ exactly on the DFL
  rows (xShot 80,796 rows, xCross 44,055 rows, False -> True; asserted: new = old OR DFL, and every DFL row was
  False). `study_config.json` and the `prep` `_output` differ only by the run provenance (a measurement run is
  dirty) and by the `arms_json` text they record (the same arms with the DFL keys renamed). The xCross `probe` and
  `probe_comparison` units and outputs are byte-identical. The study is seeded (`seed=42`) and reads all rows, so
  its trials, `cv` and `select` do not change; only the final fit and `metrics.json` change.
- Final classification runs (the probe's `_expected_json_drift` accepts only the run provenance and the renamed
  `arms_json`; `_arm_mask_fix` asserts the mask rule; any other difference is a mismatch): xShot job
  `6ae54ba30e0338d1bb7d0433` `SUMMARY {"ok": 538, "mismatch": 0, "missing_new": 0, "skipped": 2}`; xCross job
  `6ae54ba60e0338d1bb7d0435` `SUMMARY {"ok": 558, "mismatch": 0, "missing_new": 0, "skipped": 2}`. The 2 skipped
  per family are the public tier's extract `_output` (that tier now reads the superset store, so no combined
  extract equals its old one; its `prep` is compared instead, per spec r4 D18 (c)).

## Pre-commit prerequisites

- T0 (Step 0, measured ON HF as measurement jobs, no push; owner ruling 2026-10-30) must be DONE before
  T1/T2/T7/T8/T11 consume its per-stage walls and the chosen `profile_timeout_s`/`budget_floor`. The ON-HF
  campaign confirmation (the `cycle1` refit in the
  execution handoff) needs a pushed, CI-green commit (`check_ref`); the amend + force-push that enable it are
  behind the owner's amend/push gates — the builder does not force-push to unblock itself.

## Execution handoff

All tasks on one working tree, no intermediate commits. After the last task is green: run the FULL
CI-faithful suite once (`pytest -m "not e2e"` + the `.venv-hfjobs` legs + the lint trio) and record the
result → whole-branch A/B impl review (same two reviewer sessions) → fix findings (TDD) → **then** present
the single amend of `cccb702` at the owner's gate for explicit approval, and the force-push as a separate
yes. Then re-run `cycle1` at full utilization for the on-HF (W, N, thread) parity confirmation; adopt
artifacts and merge only behind their own owner gates.
