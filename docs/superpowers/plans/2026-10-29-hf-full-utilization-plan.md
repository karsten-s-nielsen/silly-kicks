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
- **Task order (deps):** **T0 (Step 0 — measure first, local/DGX, no push) PRECEDES everything** — the
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

## Pre-commit prerequisites

- T0 (Step 0, measured locally/DGX, no push) must be DONE before T1/T2/T7/T8/T11 consume its per-stage walls
  and the chosen `profile_timeout_s`/`budget_floor`. The ON-HF confirmation (the `cycle1` refit in the
  execution handoff) needs a pushed, CI-green commit (`check_ref`); the amend + force-push that enable it are
  behind the owner's amend/push gates — the builder does not force-push to unblock itself.

## Execution handoff

All tasks on one working tree, no intermediate commits. After the last task is green: run the FULL
CI-faithful suite once (`pytest -m "not e2e"` + the `.venv-hfjobs` legs + the lint trio) and record the
result → whole-branch A/B impl review (same two reviewer sessions) → fix findings (TDD) → **then** present
the single amend of `cccb702` at the owner's gate for explicit approval, and the force-push as a separate
yes. Then re-run `cycle1` at full utilization for the on-HF (W, N, thread) parity confirmation; adopt
artifacts and merge only behind their own owner gates.
