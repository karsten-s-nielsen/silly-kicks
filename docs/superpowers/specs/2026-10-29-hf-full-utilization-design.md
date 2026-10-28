# Design: HF full-utilization — every job uses 8 vCPU, every splittable stage split

Author session. Branches off `feat/controller-step-reads` (`cccb702`, PR #296, unmerged). This work
**amends into `cccb702`** (owner ruling 2026-10-29), folding a third cycle into the one commit, as the prior
two folded.

Revision r2 (2026-10-29) answers the A and B spec reviews and two owner rulings:

- **Owner ruling 1 (study):** no-rebaseline wins over R1 for a sequential search. A stage that cannot reach
  8 cores without a rebaseline uses the maximum its structure allows and records the idle remainder as
  irreducible (§A, §B, R1 caveat).
- **Owner ruling 2 (scope):** slot-filling sharding AND the `startup_s` re-profile are IN this cycle (§G),
  not deferred. Only image bake is excluded, by a measured decision.

It supersedes the "Left to the builder / Cycle 2" deferrals of
`2026-10-28-hf-compute-parallelism-design.md`. That cycle threaded ONLY the final `fit` stage and parked
three items in a future "Cycle 2" (its L11–12, L168): per-fold and per-trial sizing, a re-profiled
`startup_s`, and slot-filling sharding with the `(n_jobs−1)·startup_s` cost bound. A refit on `cccb702` then
ran `ghost_outfield` `cv` and `ghost_gk` `cv` for over 70 minutes at one thread per fold. This cycle removes
that gap and picks up all three parked items. It defers nothing. Image bake is excluded by an explicit owner
decision with a measured rationale (§ "Image bake is excluded"), not parked.

Revision r3 (2026-10-30) answers the first real campaign on `2474b73` (534 jobs; `win_probability` failed;
`ghost_outfield/position_only` looped through 354 no-op jobs; billed compute flat against the last full run) and
a third owner ruling:

- **Owner ruling 3 (scope, 2026-10-30):** there is no "Cycle 2". The three adapter-launched trainers
  (`train_pass_completion`, `train_xsuccess`, `train_match_outcome_dependence`) migrate to stage-contract
  pipelines in THIS cycle, and the whole cycle lands as one force-amended commit. Gold standard; no partial
  fixes.

The findings, their root causes and their designs are in § "r3 — the real campaign on `2474b73`". Where r3
and an earlier section disagree, r3 wins.

## Problem (measured 2026-10-29 on the `cccb702` refit)

1. **`cv` runs one thread per fold, and the overrun job was the PROFILE action.** `train_ghost_outfield.py:510`
   and `train_ghost_gk.py` declare `cv` as a plain `Stage` with 5 fold units and no `thread_safe` opt-in.
   Because `cv` had no cost history, `next_actions` (`campaign.py:297-301`) issued a `profile` action, which
   `executor._profile` (`executor.py:513`) runs as two units at `--workers 1` then the rest at the
   RAM-bounded maximum `W`. All worker processes are pinned to one thread by `deterministic_env`, so each
   fold ran at its full single-core wall (~70+ minutes observed, still running when cancelled). The
   eight-thread `fit` the prior cycle shipped never helped, because `cv` precedes `fit`.
2. **The stage barrier serializes independent stages.** `next_actions` (`campaign.py:233`, barrier at
   `:257-258`) walks stages in declaration order and stops at the first running stage. `fit` needs only
   `extract`/`cache`, yet it waits for `cv` to finish. Independent stages do not overlap.
3. **A reduce can time out unmeasured, on the INLINE path.** `win_probability` `extract` reduce ended
   `Job timeout` twice (jobs `6ae2ad16…`, `6ae2b47f…`, ~31 min apart), halting the instance
   (`error_resumes=2`). The reduce runs INLINE inside a run job: every run job calls `run_stage` then
   `try_reduce` (`_stages.py:1131-1132`), so the reduce executes under that run job's `plan_stage` timeout
   `max(timeout_floor_s=1800, timeout_safety × wall_p99)` (`planner.py:154`). `wall_p99` is the per-UNIT
   work and excludes the reduce, so a heavy reduce adds wall the timeout never budgeted and the job dies at
   the 1800 floor (~31 min). This is the inline path, NOT the dedicated reduce Action's `int(2 * c.target_s)`
   (`campaign.py:253`), which at `target_minutes=60` is 7200 s and would not have fired at ~31 min.
4. **`preflight` recomputes identical work.** `preflight.py` `check_plans` (`:149`) runs a per-instance
   `--plan` subprocess plus a once-per-module toy `--run-all` that trains a real mini-model. For an unchanged
   commit and campaign file, every check is deterministic, yet a re-run redoes all of it (~20–30 minutes
   measured). `check_datasets` (`:116`) is metadata-only, so no data file is even read.

## Owner requirements (2026-10-29)

- **R1. Every job uses all 8 vCPU.** No stage runs pinned to one thread when work remains to parallelize
  WITHOUT a rebaseline. Where byte-identity forbids 8 cores — a sequential search whose only parallelism is a
  small inner pool — the stage uses the maximum its structure allows and records the idle remainder as an
  irreducible floor (owner ruling 1; §A, §B).
- **R2. Everything splittable is split.** Independent parts (folds, trials, shards, variants) become separate
  units or inner-pool tasks; nothing bundles independent work into one single-threaded run. Within a job,
  units pack to fill the box (slot-filling sharding, §G).
- **R3. 30 minutes is a sizing TARGET, not a kill threshold.** The planner sizes work to aim under 30 minutes
  per job. A job that legitimately runs longer is NOT failed for it. Every timeout is generous and tracks a
  measured wall; no timeout is a function of the 30-minute target (§D timeouts).
- **No rebaseline.** No model, hyperparameter, or numeric path changes. Every speed-up is bit-for-bit
  identical to the single-thread, single-process artifact, proven by a committed gate.

## No-rebaseline guarantee (the central constraint)

Every change here either (a) runs independent units on separate jobs — already byte-identical by the
ADR-128 sorted-key reduce — or (b) raises a worker's thread count on a stage proven thread-identical by a
committed gate. The gate is `tests/tracking/test_thread_identity.py` (artifact byte-identical across thread
counts `{1,2,4,8}`) bound to its stage by `tests/scripts/test_thread_safe_registry.py`. A stage without a
passing identity test stays pinned to one thread. This is how "no rebaseline" holds mechanically: a
thread count that changed any byte would fail the gate and the stage could not opt in.

A third kind (r4, D18–D20): (c) a restructure of the code that computes a result — one shared extraction
in place of several identical ones, numpy arrays in place of per-frame pandas objects. It is allowed only
with two proofs: a committed oracle gate (the old code kept verbatim in the tests, compared byte for byte
on fixtures and randomized inputs), and an HF measurement job that reproduces every stored `2474b73` unit
of the affected stages byte for byte. Owner ruling 2026-10-30: where no stored `2474b73` unit can exist (the
`win_probability` `fit` never ran in `2474b73`; R4-10a changed the xCross TF-19 block's inputs), the second proof is
an HF job that runs the old bodies and the new code on the SAME full-scale inputs in one container and compares
them byte for byte (the method of the D16 canary).

**The one owner-approved exception (2026-10-30, plan R4-10a, ADR-123 Amendment B.2).** The R4-10 identity run
found that the `2474b73` baseline itself was wrong for xShot/xCross: the arm keys missed the 7 public DFL
matches (keyed by the shard stem, not by the lock's `source_provider`), so every xShot/xCross final fit
omitted them. The owner ruled to fix it as a recorded correctness change, NOT byte-identical. Its exact
extent: the arm masks (`arms/*.npy`, `is_public.npy`) mark the DFL rows (new = old OR DFL); the 8 xShot/xCross
final fits train on them, so each model and its `metrics.json` (`artifact_size_bytes`; for the all-public
tier, the exact match ids instead of a digest) change. Everything else keeps this guarantee, and the HF
identity run asserts that boundary exactly.

## The irreducible units

Two shapes are irreducible under no-rebaseline:

- **A final `fit`** trains ONE model (`GhostOutfieldModel`/ghost-GK = sklearn `HistGradientBoostingRegressor`;
  xShot/xCross = one xgb booster; `win_probability` `fit` is also one model and is NOT thread_safe today).
  Boosting is sequential across rounds; the only parallelism is within a round (split-finding), which is
  shared-memory threading in one process. Splitting one fit across jobs needs a distributed-training
  algorithm — a different numeric path — which is a rebaseline and is refused.
  `HistGradientBoostingRegressor` has no distributed mode at all. So the final fit per variant is one job
  that uses all 8 cores via threads (where a thread-identity gate passes).
- **A sequential HPO `study`** (xShot/xCross) is a seeded TPE search: trial N depends on trials `1..N-1`.
  Running trials in parallel changes the search sequence — a rebaseline, refused. Its only no-rebaseline
  parallelism is one trial's ~5 CV folds on the inner pool (`train_xshot_occurrence.py:71-72`). So a study
  saturates at most ~5 of 8 cores; the remaining cores are an irreducible floor (owner ruling 1).

Under R3, neither is a blocker: if an 8-thread fit or a fold-width study runs over 30 minutes, the job is not
failed; the number is reported.

## Design

### A. A per-stage utilization audit is implementation step 0

Before any opt-in, profile EVERY stage of EVERY campaign module (`ghost_gk`, `ghost_outfield`,
`xshot_occurrence`, `xcross_attempt`, and every public driver the campaign runs, `win_probability`
included) at its production parallelism. Record, per stage: the number of independent parts, the per-part
wall at one thread, the per-part wall at eight threads, the peak RSS, and the measured startup. The audit
decides each stage's shape by the rule in §B, and it is written to the campaign summary so the planner and
the reviewers see it. No stage is opted in or re-sized on a guess.

The audit artifact asserts, per stage, ONE of:

- the stage saturates all 8 cores (an 8-thread fit, or an inner pool that fills the box), OR
- the stage uses the maximum its no-rebaseline structure allows and records the idle remainder as an
  irreducible floor (a sequential study at its ~5-fold inner width; a single final fit).

This replaces the prior wording "saturates 8 or is a single irreducible unit", which a study could not
satisfy (B review D1-SPEC-03).

**Owner rulings 2026-10-30 (Step 0 decision table, plan R3-8 S2; S2 reviews A S2-PLAN-01, B S2B-02/03).**
- The structural floors are exactly three: a single final fit at the full thread budget; a sequential study at
  its fold width; and a stage whose every unit runs at once with the whole thread budget (`W >= n_units` and
  `W x threads >= vcpu`), which generalizes the first. Each floor also requires MEASURED use when its thread
  budget is above 1 (owner rulings 2026-10-31: "half the budget", and the boundary strict): from the stage's D8
  job record, the cores per process (`cpu_s / wall_s / (workers x inner_workers)`) must be MORE than half the
  thread budget. The owner's first answer read ">= budget / 2"; asked again with the reason (at budget 2 a dead
  lever measures exactly 1.0 core per worker), the owner ruled "more than half". So a dead thread lever (one core
  per worker at a budget of 2 or 8) and a mostly-serial fit (the `ghost_outfield` `fit` as measured in S1, 1.75
  cores at a budget of 8) never audit as a floor.
- "Too small to pay": a stage whose summed unit CPU plus its reduce CPU is under 60 s takes a reviewed serial
  reason of this class, each with its measured time. Any stage over 60 s is shaped (split, threads, or a
  structural floor).

**Owner rulings 2026-10-31 (S2 r2 review B S2B-r2-02, S2B-r2-03).** The floor bar above. The `ceil(vcpu / n)`
thread budget of §B covers an inner pool of n <= 5 children too (the xShot/xCross study's 5 fold children get 2
threads each), with an HF measurement and the identity at child budgets 1 and 2.

### B. The shape rule for every multi-part stage

Each stage takes exactly one of three shapes, chosen by its audit (the third, WIO, by owner ruling 2026-10-30):

- **Units across jobs, 8 threads each** — for a stage whose parts are heavy model fits (`cv` folds, and any
  stage that trains or scores a model per part). The stage declares `thread_safe=True` with its lever
  (`omp` for HistGB, `n_jobs` for xgb). The planner runs it at `W=1`, `inner=1`, so each unit gets the full
  `thread_budget(vcpu, 1, 1) = 8` (§D). Units fan across N jobs (§D, §G). Byte-identity: the thread-identity
  gate plus the sorted-key reduce. Owner ruling 2026-10-30 (the generalized floor): a stage of n <= 5 such units
  runs all n at once in ONE job with `ceil(vcpu / n)` threads each (at most 25 % oversubscription). Step 0
  measured it better than n jobs of 8 threads: ghost_gk `cv` 118 s at 0.765, ghost_outfield `cv` 215 s at 0.869.
- **Inner pool across processes** — for a stage with many independent light parts whose process-pool
  parallelism fills the box WITHOUT changing a numeric sequence. The stage stays `inner_parallel` with
  `inner_width` sized to the part count, capped by RAM and by 8. A sequential `study`'s inner pool
  parallelizes ONE trial's ~5 CV folds, so its `inner_width` is at most the fold count (~5); the audit
  records the ~3 idle cores as an irreducible floor, it does NOT size the study to 8 (that would need
  parallel trials = a rebaseline). The audit must show the pool genuinely runs its sized width; a `study`
  observed at `--inner-workers 1` is mis-sized and is corrected to the fold width here.
- **Units across MORE workers than vCPUs (WIO)** — owner ruling 2026-10-30. An item stage whose single-thread
  units wait on I/O for part of their wall runs `W = ceil(vcpu / cpu_share)` workers, where `cpu_share` is the
  measured unit CPU over unit wall, capped by RAM. Step 0: `win_probability` `extract` (3,961 units, 0.32 s CPU of
  0.45 s wall) at W=8 / 12 / 16: utilization 0.686 / 0.946 / 0.946, wall 327 / 263 / 228 s. Byte-identity: the
  units are single-threaded and the reduce is sorted-key, so W does not touch a byte.

The audit's one-thread vs eight-thread walls decide which shape minimizes wall for each stage. A stage whose
single part is a heavy fit AND whose parts are few (`cv`, 5 folds) takes the first shape; a stage with many
short independent parts takes the second. A stage that is neither single-threaded nor splittable today and
already saturates the box (a many-shard `extract`) is left unchanged, and the audit records that it already
meets R1.

#### B.1 The pipeline CLI entry survives a native teardown abort

Every stage now runs as a short `python -m <module>` process, and the sharded reduce adds more of them (one
per `--subreduce` shard plus the final `--from-partials`). A short process that uses Arrow (every parquet
read/write) can abort at interpreter shutdown with `terminate called without an active exception` (SIGABRT)
when Arrow tears down its C++ runtime — AFTER the job has already written its correct output. The controller
marks any nonzero rc as a failed job, so a correct `--subreduce` would be failed spuriously (observed once on
Linux CI, shard 2 of 3; the two other shards and the Windows box did not hit it — it is a shutdown race, not a
logic path). The fix is a single entry helper, `scripts/_stages.py` `cli_main`, that every migrated module's
`__main__` calls in place of `raise SystemExit(main(...))`: it runs `main`, then runs the registered Python
atexit handlers (so the worker-input temp-dir rmtree still happens), flushes stdout/stderr (so the `--plan`
JSON is not lost), then `os._exit(rc)` — a hard exit that skips interpreter finalization (the C++ static
destructors that raise, and `threading._shutdown`; no pool or thread join runs through this exit, since an
`InnerPool` is joined by `worker_main`'s `finally` before it returns, so that join completes before this exit
is reached). Results are already persisted by atomic writes before `main` returns, so the hard exit loses
nothing, and `main`'s rc reaches the exit unchanged (a raised acceptance gate still exits nonzero as before).
The hardened exit is the shared `hard_exit`, which the executor's reduce/worker child (`scripts/hf_jobs/executor.py`
`__main__`) also uses: that child does Arrow I/O too, and `_subprocess` fails the inline reduce on any nonzero
rc, so a correct inline reduce that aborted at teardown would otherwise be failed.

### C. Independent stages overlap (DAG schedule)

Replace the linear barrier in `next_actions` (`campaign.py:233`, barrier at `:257-258`). The controller
launches EVERY stage whose dependencies (`needs`) are satisfied and whose units are not yet running, not
only the first in declaration order. For `ghost_outfield`, `cv` and `fit` both need `extract`; once
`extract` has reduced, both become ready and run concurrently (the §C example in r1 wrongly named a `cache`
stage, which only `ghost_gk` has — A review HFU-SPEC-06). The change preserves the existing per-stage
guards: a stage still launches only when ready, a running stage is not relaunched, and the dispatch budget
(`dispatch_cap`) still caps total jobs in flight.

The controller returns actions for more than one stage per step. Three framework points change with it
(B review D1-SPEC-06):

1. **`step()` consumes every returned action, not `acts[0]`** (`campaign.py:806`). It dispatches each ready
   stage's action in the step.
2. **The instance report carries per-stage state.** Resume is keyed per stage, not by one
   `"-".join(a.stages)` cursor (`campaign.py:~813`), so a concurrent sibling's progress is recorded
   independently.
3. **`error_resumes` and `halt` are per stage, and halt cancels the instance's siblings.** A stage that
   halts (over its `max_failures`, or `error_resumes` ERRORs in a row) halts the INSTANCE and names itself.
   There is no automatic stop path today: `halt_report` (`campaign.py:656-658`) launches and cancels
   nothing, and `cmd_cancel` (`:964-973`) is a manual whole-campaign cancel (B review 25). Design the
   per-instance cancel: on halt, `step()` cancels THAT INSTANCE's in-flight jobs — filter
   `launcher.jobs_for(api, campaign, commit)` to the instance key `(run_key, variant)`, NOT `(job, variant)`
   (`run_key` encodes the tier, and `cycle1.toml` runs `xshot_occurrence`/`xcross_attempt` in two tiers,
   public and sc_extended, sharing `job`+`variant`, so a `(job, variant)` filter would cancel both tiers —
   B review 30), and `api.cancel_job` each one not in `launcher.TERMINAL`, the same primitive `cmd_cancel`
   uses, scoped to the instance. A cancel that fails is retried the next stateless step, because the
   controller re-derives from live job state and the instance stays halted until no job of it is live. Only
   the halting instance is cancelled; other instances, including a same-`job`/`variant` sibling on another
   tier, are untouched. The halt bookkeeping becomes per stage; the halt threshold
   semantics do not change.

### D. The planner sizes a thread_safe stage at W=1; the target is a sizing lever; every timeout is measured

`plan_stage` (`planner.py:108`) changes:

1. **Thread_safe non-inner stages run at `W=1`.** Today `workers = _workers(peak, c)` can be up to `vcpu`,
   so `thread_budget(vcpu, W, 1) = vcpu // W` collapses to 1 when W fills the box. For a `thread_safe`
   non-inner stage, set `workers = 1` so each unit's worker gets the full `vcpu` threads, and fan the units
   across `n_jobs` so each job's predicted wall is within the target. RAM still bounds it: one 8-thread
   worker must fit the usable RAM (it does — one worker, not W). (`executor.py:318` already clamps the
   worker count to the pending-unit count, so a job with one pending unit already runs one worker; the W=1
   planner rule makes the sizing explicit and drives N, and matters for any multi-unit job — A/B CONSIDER.)

2. **The target is set in the campaign TOML, not the planner default.** The live target is
   `cycle1.toml` `target_minutes` fed through `Campaign.constraints` (`campaign.py:63-66`); the
   `Constraints.target_s` default (3600) is never used for the campaign (B review D1-SPEC-08). Set
   `cycle1.toml` `target_minutes = 30`. Update the gate that pins it (`test_campaign.py:109`, currently
   `3600`). Note: changing the TOML changes `campaign_sha256`, so a fresh preflight is required for the new
   hash (§F makes that fast).

**Timeouts are decoupled from the target (B review D1-SPEC-01, the R3 contradiction).** Today four actions
take `int(2 * c.target_s)` (`campaign.py:253`): the escalate run (`:274`), the reduce (`:283`, `:287`), the
tiny-stage run `n_units ≤ 2` including the lone fit (`:295`), and the profile (`:301`). Lowering `target_s`
to 1800 would silently halve all four to 3600 s = a 60-minute kill threshold, which violates R3 for a fit or
reduce that legitimately runs longer. Replace the `2 * target_s` timeout at each of the four sites:

- **Lone fit / tiny stage (`n_units ≤ 2`):** this branch (`campaign.py:293-295`) runs BEFORE the profile
  branch (`:297`), so a tiny stage is never auto-profiled (B review 17). A tiny/grouped job runs its bundled
  stages' units one after another plus an inline reduce, so its timeout is sized to that ACTUAL sequential
  work. Each bundled stage's p99 comes from a committed per-stage wall the §A audit (Step 0) writes to the
  campaign summary, read at plan time by the stage's history key — the same channel `median_startup_s` uses.
  Timeout = `timeout_safety × (Σ n_units·p99 over the bundled stages + their inline reduce walls) + startup_s`,
  floored by `timeout_floor_s`. A single flat `profile_timeout_s` ceiling kills a heavy grouped chain whose
  wall exceeds it, and `timeout_safety × p99` of ONE unit kills a 2-unit tiny job that runs ~`2 × p99`
  sequentially; summing the work does neither (owner-ratified gold standard 2026-10-29; reviewer A D-IMPL-07,
  partial review B D1-IMPL-02). No committed measurement yet (a first run, any bundled stage unmeasured): the
  generous fixed `profile_timeout_s` fallback below, never `2 × target_s`.
- **Reduce (dedicated Action and inline, §E):** timeout = `timeout_safety × profiled_reduce_p99`, the reduce
  wall measured in §E; the carrying run job's timeout includes that reduce p99 (§E). No measurement yet: the
  `profile_timeout_s` fallback.
- **Escalate (a memory-bound unit alone):** timeout = `timeout_safety × p99` of that unit's measured wall if
  known, else `profile_timeout_s`.
- **Profile:** a measuring action cannot use its own measurement. Add a dedicated, generous
  `profile_timeout_s` to `Constraints`, independent of `target_s`, sized for the bounded `k`-unit
  thread_safe profile job (§E): about `k ×` the longest expected one-thread part wall plus margin (the
  70-minute `cv` observation sets the per-part scale). The profile is a one-time measurement; its ceiling
  protects against a hang, not against exceeding 30 minutes.

Each of the four timeout sources gets a test asserting it does not shrink when `target_s` falls, and that it
derives from a measured wall (or the fixed profile ceiling). `plan_stage`'s own timeout
(`planner.py:154`, `timeout_safety × wall_p99`, floored) already satisfies R3 and is unchanged.

The `irreducible_floor_s` report (`planner.py:100`) stays: a stage whose single part exceeds the per-job
budget is reported with its floor (the lone fit, if over 30 minutes), surfaced to the owner, never silently
capped and never auto-cut.

### E. Every stage that can run long is profiled, reduces and 8-thread stages included

**Profiling a thread_safe stage (fix the clobber, B review 18).** The `profile` stage
(`executor._profile`, `executor.py:513`) today runs two units at `W=1` then the rest at the RAM-bounded
maximum `W`, which for a `thread_safe` non-inner stage mixes an 8-thread per-unit wall (the `W=1` sample)
with a budget-collapsed wall (the `wmax` sample), so the measured per-unit wall is wrong
(B review D1-SPEC-02). It also writes ONE `{gen}/_profile.json` per generation (`executor.py:550`), read by
`live_history` (`campaign.py:381`), so N separate profile jobs would overwrite each other and the last
writer's `peak_rss`/`efficiency` would win (B review 18). Fix: a `thread_safe` non-inner stage profiles only
the bounded `first` sample (`k = 2` units, the count `_profile` already runs at `W=1`) SEQUENTIALLY at
`W=1`, `inner=1` — each unit at its full 8-thread budget, the shape the planner will use — and SKIPS the
`rest`/`wmax` phase (there is no multi-worker phase at `W=1`, which otherwise ran up to
`min(n_units, 2·vcpu)` = 16 units, `campaign.py:300`). One job, one `_profile.json`, one writer, no clobber;
the bounded `k` keeps the profile job short (B review 24). The job's timeout is `profile_timeout_s`, sized
for `k` one-thread parts plus margin (`k` units at 8 threads never exceed `k` units at one thread),
independent of `target_s`. The planner seeds from these `k` samples and refines each per-unit cost from
completed units via `live_history`. A test pins the profile's `W`, thread count, bounded sample `k`,
single-job shape and timeout for a thread_safe stage.

**Profiling reduces, inline and dedicated (fix the real `win_probability` cause, B review 19, 17).** Every
reduce whose wall can exceed the target is profiled at real scale, the `extract` REDUCE among them. Today
nothing records a reduce wall: `_profile` times units only (`executor.py:520-536`), `try_reduce` records no
wall (`executor.py:244-259`), and `stage_summary` has no reduce field (`campaign.py:993-1012`). Add the
timing: `try_reduce` records the reduce wall to a store record (`{gen}/_reduce.json`, beside
`_profile.json`), and `stage_summary` (`campaign.py:993-1012`) reads it into a `reduce_wall` field at plan
time like a unit cost (B review 17, 33). `{gen}/_reduce.json` is declared VOLATILE (`_parity.py:19`) and a
controller record (`_is_record`, `store.py:78-82`), exactly as `_profile.json` is, so a wall-time file never
enters the parity `outputs` (it would make every reduce-stage parity arm red) and it batches in the one
recursive store listing instead of a download per stage per step (B review 34). The reduce runs INLINE in whichever run job completes the stage's final
unit (`try_reduce`, `_stages.py:1131-1132`); the planner cannot know at plan time which job that is, so EVERY
run job of a stage whose `reduce_wall` is recorded takes its timeout as
`timeout_safety × (work_p99 + reduce_p99)`, where `work_p99` is `plan_stage`'s job-level `wall_p99`
(`planner.py:134`, `:152`) and `reduce_p99` is the recorded reduce wall (B review 28). A stage whose reduce
is NOT yet measured — a first run, EVERY stage, not only `win_probability` — takes
`max(plan_stage timeout, profile_timeout_s)` on each run job, so a larger planned timeout is never lowered
(B review 32), never the bare 1800 floor. A regression test proves a long inline reduce is not killed at
1800 on the first run. The sharded-vs-timeout decision is a fixed rule, not a
build-cost choice (A review HFU-SPEC-03): **if the recorded reduce p99 exceeds the per-job budget
(`target_s − startup_s`), the reduce is sharded (§G); otherwise every run job carries the inflated inline
timeout above.** A sharded reduce changes the ADR-128 reduce contract, so its design is in this spec (§G),
not left open.

### F. Preflight reuses unchanged results

`preflight` becomes incremental. Store, per check, the digest of that check's DECLARED inputs in a persistent
per-check store (NOT nested under a commit-keyed report, which a new commit would wholly invalidate —
B review D1-SPEC-10). On a re-run, recompute ONLY a check whose input digest changed; reuse the stored
passing result otherwise. Per-check input keys:

- `check_ref`: the commit AND its remote push/green state. A not-yet-green result is always re-checked; a
  force-push can make a previously green commit un-green, so the check re-observes
  `git branch -r --contains` and the CI status rather than trusting a cached pass for a rewritten ref.
- `check_api`: the live flavor catalog — cheap, always run.
- `check_datasets`: metadata-only (`list_repo_files`) — cheap, always run; its observed file-list digest is
  recorded so a downstream check can see it changed.
- `check_lock`: the `uv.lock` digest and the resolved image digest (both re-observed cheaply; the resolve is
  the only remote call).
- `check_plans` (the ~20–30-minute toy train): the EXPENSIVE part. The toy `--run-all` and `--plan`
  subprocesses import `scripts/_stages`, `scripts/hf_jobs/**`, `silly_kicks/**` and read `datasets.lock`, so
  a library-only edit changes their behaviour (B review 20). Its key is the `campaign_sha256`, the `uv.lock`
  digest, the resolved image digest, the `datasets.lock` digest, AND a tree digest of every IMPORTABLE source
  path the subprocesses load: `scripts/**/*.py` and `silly_kicks/**/*.py` (owner-ratified 2026-10-29; reviewer
  B D1-IMPL-14), MINUS the controller-only files (`_CONTROLLER_ONLY_FILES`) that run only on the orchestrator host
  and that no toy `--plan`/`--run-all` imports (HFU canary run 3, 2026-11-01: the unqualified glob re-ran the
  30-min train on every controller amend, e.g. `campaign.py`). The whole library and every trainer-reachable file
  stay hashed, so the key can never UNDER-hash code the train loads; the exclusion is gated by the static import
  closure of the registry + canary entry modules (`_repo_import_closure`), which over-approximates what the
  subprocesses load and must reach none of the excluded files (reviews R5-PF-01 / CK-B-01/02/07). The key also
  digests the checked instance `(module, args)` (built by the excluded `campaign.instances()`), so a changed
  `--plan` argv re-runs the train though `campaign.py` is excluded (review CK-B-06). Trained weights
  (`.npz`/`.json`) and other data under those trees are loaded at RUNTIME by a fit, not imported, and do NOT change
  the PLAN the toy train validates, so they are excluded by design (and the `datasets.lock` digest already covers
  the data pins). A dep change, a trainer change, OR a library `.py` change re-runs the toy train; it reuses its
  stored pass only when all are unchanged. A test shows a library submodule edit invalidates the key, a controller
  edit does not, and another shows the key is independent of the process cwd.

The win the owner asked for — "not redoing the exact same work when not a single data file has changed" — is
the same-commit, same-TOML re-run: every check's input digest matches, so `check_plans` reuses its pass and
preflight returns in seconds. When a file DID change, only the checks whose declared inputs changed re-run.

The launch gate `require_preflight` (`preflight.py:195`) is unchanged in contract: it still requires a
passing report for this commit and this `campaign_sha256`. The incremental store feeds it a fresh report
quickly; it does not weaken the gate. The per-check cache is a single file in the preflight report store
keyed by check name and input digest (not by commit), owned by `preflight.py`; a missing or unreadable cache
forces a full recompute (fail-safe).

### G. Slot-filling sharding and the re-profiled `startup_s` (owner ruling 2)

The prior cycle parked these; this cycle designs them. (The prior record defined "slot-filling sharding" and
required a `budget_floor` plus a cost bound; r2 of this spec inverted it — B review 16.)

1. **Re-profile `startup_s` and bootstrap the first run.** The planner ALREADY reads the measured median
   startup: `planning_constraints` (`campaign.py:362-367`) calls `ledger.median_startup_s` over the
   committed summaries, and the 300 s `Constraints.startup_s` default (`planner.py:23`) applies ONLY before
   any summary is committed (B review 16). So the gap is the FIRST run: no summary exists
   (`docs/research/hf_campaigns` is absent), so the first plan still uses 300 s. Fix: Step 0 measures
   startup (about 15 s on `cccb702`: `uv` prepared 55 packages in 7.52 s inside a 24 s job) and commits a
   bootstrap summary carrying that median, so the first planning run uses the real startup, not 300 s. A
   test asserts the planner uses the committed median when a summary exists, and that the bootstrap summary
   supplies it on the first run.

2. **Slot-filling with a `budget_floor` (the approved item, B review 16).** The current formula
   `n_jobs = ceil(work / budget)` (`planner.py:131`, `:142`) picks the MINIMUM jobs to get each under the
   target — it UNDER-fills free dispatch slots (prior review HCP-SPEC-01/05). Slot-filling is the opposite:
   when dispatch slots are free, raise `n_jobs` toward filling them so each job's wall is SHORTER than the
   target, bounded by a new `Constraints.budget_floor` — the minimum work per job, so a job never shrinks to
   startup-sized. Apply it to BOTH planner branches (inner_parallel and non-inner; HCP-SPEC-05).
   Slot-filling needs a free-slot INPUT the planner does not have today: `plan_stage` takes no slot count
   (`planner.py:108-118`), `next_actions` passes none (`campaign.py:305-313`), and the dispatch budget lives
   in `step()` (`:780-782`) and is spent instance-by-instance (`:795-855`), so the first instance could take
   every free slot (B review 26). Fix: `step()` divides the free-slot budget into a per-instance fair share
   `max(1, budget // n_active_instances)` and passes it through `next_actions` into `plan_stage`;
   slot-filling fans an instance up to its share, never beyond, so no instance starves the others.
   `budget_floor` still bounds how small a job may get, and is set so the re-profiled startup is a small
   fraction of the minimum per-job work (about 10 %, so a slot-filled job is at most ~10 % startup overhead
   — the owner-facing cost knob, B review 29). The cost bound `(n_jobs − 1)·startup_s` is an identity of the
   planner model, not a gate (prior review B03), so it is stated, not tested. The TESTABLE rules, each able
   to go red against today's min-N code: (a) given a free-slot share and work above `budget_floor`, `n_jobs`
   exceeds `ceil(work / budget)` but never the share; (b) no planned job's work is below `budget_floor` (no
   over-shard); (c) slot-filling applies in both branches; (d) two active instances split the free slots
   fairly. Cost note for the owner: at the re-profiled ~15 s startup, filling slots adds cents, not the ~2×
   bill the prior review feared at the 300 s placeholder (that concern assumed `work_per_job ≈ 300 s`);
   §G-1 removes it.

3. **The sharded reduce (execution design, B review 21).** When §E sends a heavy reduce to sharding (its
   profiled p99 exceeds the per-job budget), the reduce is range-partitioned on sorted keys:
   - The inline `try_reduce` is DISABLED for that stage; the reduce is dedicated, not inline.
   - A `sub-reduce` Action per contiguous sorted-key range concatenates that range via `combine_frames`
     (`_stages.py:670-684`, an ordered `pd.concat` then storable id columns) and writes a partial to
     `{gen}/_subreduce/{range}.parquet`. Sub-reduces fan across jobs like any unit stage, under the §D
     timeout.
   - A final `reduce` Action concatenates the partials in sorted range order via `combine_frames`.
   - Byte-identity: `combine_frames` is an ordered `pd.concat` (`ignore_index` renumbers a non-payload row
     index only), so a concat of range-partitioned sub-concats in sorted order equals the single
     sorted-order concat byte-for-byte. It also rests on `storable_id_column`/`canonical_id_series` being
     idempotent (`_stages.py:646-655`, a second pass over an already-canonical id column is a no-op) and on
     a lossless parquet round-trip of each partial; the parity harness gains a sharded-reduce arm proving
     equality end-to-end, with non-vacuity (`≥ 2` sub-reduces) (B review 27).
   - If the final ordered concat alone still exceeds the budget, it is an irreducible floor (a single ordered
     concat is O(rows) and cannot be split without reordering): reported under R3, surfaced, never failed.

## Image bake is excluded (owner decision, with the measured reason)

Per-job startup was measured on the `cccb702` refit: `uv` prepared 55 packages in 7.52 s and installed in
155 ms inside a 24 s `extract` job, so startup (git clone + venv + `uv sync`) is about 15 s, not the 300 s
`startup_s` placeholder. At the campaign scale (about 240 jobs at concurrency 25), baking the image saves
about 15 s per job — roughly 2.5 minutes of total wall and about three cents. The large ADR-128 P1 change
(baked image with provenance fail-closed when the installed environment differs from `uv.lock`) is not
justified by that saving. It is excluded from this cycle, is fully reversible, and can be its own later cycle
without touching any artifact. This is an exclusion decision, not a deferral of required scope. (The
`startup_s` re-profile in §G is the part of the old "Cycle 2" that IS valuable and is done here; it needs no
bake.)

## ADR-128 amendment (full sweep)

Amend INV-2 again. The prior cycle's amendment allowed `thread_safe` on the final fit; this cycle states the
general rules:

> Every unit is single-threaded UNLESS its stage is `thread_safe` with a committed cross-thread-count
> byte-identity test. A `thread_safe` non-inner stage is sized at `W=1` so its worker gets
> `thread_budget(vcpu, 1, 1)` threads via its declared lever; a `thread_safe` inner_parallel stage's pool
> children get the per-child budget, and the pool width is the stage's independent-part count, never a width
> that reorders a sequential search. The omp lever raises `OMP_NUM_THREADS`; the n_jobs lever a runtime arg;
> `OPENBLAS/MKL/VECLIB` and `NUMBA` stay `"1"` on every path. The controller schedules every stage whose
> `needs` are met, not only the first in declaration order. The planner's `target_s` is a sizing target
> ONLY; every job timeout derives from a profiled wall (or a fixed profile ceiling) and never from the
> target, so no job is failed for exceeding the target. The planner uses the committed median `startup_s`
> and, when dispatch slots are free, fills them to shorten each job's wall, never below `budget_floor` work
> per job.

Update every surface the prior amendment listed, plus `campaign.py` `next_actions` and `step` (the DAG
change and per-stage resume), `planner.py` `plan_stage` (W=1, the target, slot-filling), the four
`campaign.py` timeout sites, `preflight.py` (incremental checks), `AGENTS.md`,
`docs/howto/compute-pipelines.md`, and the compute-pipelines skill. A separate short ADR records the
preflight-reuse contract.

## Validation

- **No-rebaseline gate.** Each newly thread-enabled stage (`cv` for ghost HistGB and for xgb; any other the
  audit opts in) gains a `test_thread_identity` arm proving its artifact byte-identical across `{1,2,4,8}`
  threads at a non-vacuous fixture (a precondition test proves the fixture large enough that the
  multi-thread path engages, per AGENTS.md), and a `test_thread_safe_registry` entry binding the stage to
  its test. For `cv`, the "artifact" the gate compares is DEFINED (B review D1-SPEC-11): `cv` saves NO model
  files — `_cv_work` returns a result dict per fold (`train_ghost_outfield.py:385-405`) and `_cv_reduce`
  returns the aggregated `[cv_mae, per_slot, per_provider, per_possession]` (`:408-415`). The gate compares
  the serialized per-fold result dicts AND the reduced table, byte-identical across thread counts; the
  registry binds `cv` to those, as it binds a `fit` to its model class. Synthetic fixtures at real shape and
  scale, never owner-tier raw, provenance recorded.
- **(W, N, thread) parity, at the production shape.** The ADR-128 parity harness gains the thread axis for
  every thread_safe stage: `run_serial` (one unit, 1 thread) equals `run_fanned` (≥2 jobs × the production
  thread count) for `outputs` and `artifact`, with non-vacuity (`fanned_jobs ≥ 2`). The thread-safe parity
  runs at the PRODUCTION `W=1` — one unit per job at 8 threads — not a forced `W=2`
  (B review D1-SPEC-07); `ran_concurrently` (intra-job, W≥2) is the WRONG primitive for a W=1 stage and is
  not used to prove its parallelism. Its parallelism is proven by `fanned_jobs ≥ 2` plus the thread-identity
  gate.
- **Planner.** A thread_safe non-inner stage plans at `W=1` with the full thread budget; `target_s=1800`
  splits a heavy multi-fold stage into one job per unit; every timeout equals `timeout_safety × a measured
  p99` (or the fixed profile ceiling) and does NOT shrink when `target_s` falls; a single part over the
  target is reported as an irreducible floor, not cut; the planner uses the committed `median_startup_s`,
  not 300. Slot-filling (§G-2): given a free-slot share and work above `budget_floor`, the planned `n_jobs`
  exceeds `ceil(work / budget)` but never the per-instance share; no planned job's work is below
  `budget_floor`; the rule holds in both planner branches; two active instances split the free slots fairly.
- **DAG schedule.** A unit test builds a plan with two independent ready stages and asserts `step` launches
  BOTH in one step; a stage whose `needs` are unmet is not launched; a running stage is not relaunched;
  per-stage resume records each sibling independently. A cross-stage concurrency assertion proves two
  stages' job wall-intervals overlapped (NOT `ran_concurrently`, which only proves intra-job W≥2 overlap —
  B review D1-SPEC-07); a non-vacuity arm proves the SAME assertion goes RED when the two stages run
  serially. A halt arm proves a stage halting while a concurrent sibling has in-flight jobs cancels that
  instance's sibling jobs by its `(run_key, variant)` key (the per-instance cancel designed in §C) with
  nothing orphaned; the arm adds a second instance sharing `job`+`variant` on another tier and asserts its
  jobs SURVIVE (A review HFU-SPEC-07, B review 25, 30).
- **Reduce profiling and sharding.** `try_reduce` records a reduce wall and `stage_summary` carries it
  (`reduce_wall`); a run job's timeout then covers the inline reduce (`work_p99 + reduce_p99`). A regression
  test proves a long inline reduce is not killed at 1800 on the FIRST run (before any `reduce_wall` exists),
  via the `max(plan_stage timeout, profile_timeout_s)` fallback. When a reduce's recorded p99 exceeds the per-job budget, the sharded
  reduce equals the single reduce byte-for-byte (a parity arm).
- **Preflight reuse.** A re-run with no changed input recomputes nothing (above all skips the `check_plans`
  toy train) and reuses the stored per-check passes; a changed `uv.lock`, image digest, importable-source
  tree digest (including a `silly_kicks/**`-only change), `datasets.lock`, commit, or `campaign_sha256`
  re-runs exactly the affected checks;
  `require_preflight` still gates launch on a passing report for the commit and campaign hash; a force-pushed
  or not-yet-green `check_ref` always re-checks.
- **Audit artifact.** The per-stage utilization audit (§A) is written to the campaign summary and asserts
  every campaign stage either saturates 8 cores OR uses the maximum its no-rebaseline structure allows with
  the idle remainder recorded as an irreducible floor.

## Step 0 before building the rest

Profile, at production shape: an 8-thread lone `fit` for each model; the per-fold `cv` wall at 8 threads and
at 1 thread; the `study` at its ~5-fold inner width; and the `win_probability` `extract` reduce at real
scale. Report the numbers and the measured median startup. The rest of the build assumes the audited walls;
it does not proceed on an estimate. If an 8-thread lone fit or a fold-width study exceeds 30 minutes, that is
reported under R3 (not a failure, not a cut).

## Left to the builder

- The exact set of stages that opt in, decided by the §A audit (the rule in §B is fixed; the per-stage
  outcome is measured).
- The synthetic-fixture generators at real shape and scale for each newly thread-enabled stage.
- The `profile_timeout_s` value in `Constraints`, chosen for the bounded `k`-unit profile job (§E): about
  `k ×` the longest one-thread part wall plus margin.
- The `budget_floor` value in `Constraints` (§G-2), chosen so the re-profiled startup is a small fraction
  (about 10 %) of the minimum per-job work.

(The sharded-vs-timeout reduce choice is NO LONGER open: §E fixes the rule — shard when the profiled reduce
p99 exceeds the per-job budget, else measured timeout — and §G designs the sharded reduce.)

## r3 — the real campaign on `2474b73`

### r3.1 What the campaign showed

The first full campaign on `2474b73` (`cycle1.toml`, 21 instances) ran 534 jobs: 533 COMPLETED, 1 ERROR.
Evidence: the author diagnosis and its A/B partial reviews
(`D:\Development\_reviews\2026-10-30-hfu-campaign-diagnosis-{author,A,B,author-r1}.md`), the campaign summary
`summary-cycle1-2474b73.json` (kept outside the repo), and the canary jobs labelled `campaign=canary`.

- `win_probability` failed: the `extract` run job that completed the stage ran the inline reduce of 3,961 shards,
  silently, until the 7200 s timeout. The controller then halted the instance for 5 h.
- `ghost_outfield/position_only` `cv` launched 354 jobs, one per controller step, each finishing in about 18 s
  with no unit done.
- Against the last FULL run (`74d57ba`), billed job-minutes are flat (2,388 vs 2,348). `ghost_gk` x5 improved
  8–13 %; `ghost_outfield/position_only` is 2.4x slower; only small instances improved 1.8–4.3x.
- The campaign's own R1 audit fails 65 of 124 stage rows, and 9 more pass only as a "floor" with idle cores.
- Step 0 and the §A audit (both required "before building the rest") were not executed before the campaign.

### r3.2 Defects and root causes

| Id | Defect | Root cause (evidence) |
|---|---|---|
| D1 | The `extract` reduce of 3,961 shards runs for hours | `reduce_stage` calls `read_result` per done key (`_stages.py:912`); `read_result` calls `store.names(rel_dir)` per call (`:501`); `LocalStore.names` lists the whole generation with one `stat` per entry (`:294-296`), uncached. Through `hf-mount` that is about 3,961 listings of about 12,000 entries. Each shard is also read twice (`_verified`, then `read_bytes`). `reduce_subrange` (`:930`) has the same per-key listing, and `reduce_done_order` re-hashes ALL units in every sub-reduce. Three module-level loops call `read_result` per key with the same listing: `ghost_gk` `_cache_work` (`train_ghost_gk.py:474-482`), `xcross` `_probe_comparison_work` (`train_xcross_attempt.py:661-677`) and the census `_done_results` (`:653-658`) (B review R3-02). This also corrects Problem 3 above: the reduce is not inherently heavy; its algorithm is quadratic. Measured on the real generation through `hf-mount` (read-only job `6ae4884ac78863f9c62e7481`): one cold listing of the 11,987 entries 1.36 s, but `read_result` 85.3 s per key (repeated listings fetch every entry's attributes), so the full reduce projects to 338,059 s (about 94 h); the same reads without the listing cost 0.36 s per key. The 7200 s timeout fired about 85 keys in. The same failure is in `74d57ba` (622 min, 3 errors). |
| D2 | The sharded reduce (§G-3) cannot engage on a first run | The shard decision needs a measured `reduce_wall` (`campaign.py:355`). It is recorded only after an inline reduce completes, and no committed summary carries one. |
| D3 | The controller halts instead of reducing | `resume_after_error` (`campaign.py:784-801`) resumes after a timeout only with pending units. A job that timed out in the inline reduce leaves none, so the instance halts. A resume that only re-launches the same reduce under the same timeout would loop (B DIAG-05). |
| D4 | A long-lived controller re-profiles a stage forever | `live_history` (`campaign.py:512-539`) reads a stage when its `done` COUNT is non-zero and caches the cost per generation for the life of the process. The plan counts a unit done when its result and `.sha256` exist, but `run_unit` writes the ledger last (`_stages.py:753`, `:794`). A step inside that window caches `None` permanently. A non-`None` cost never refreshes either. |
| D5 | A repeat profile can never make progress | `run_stage` passes all shard units, done ones included, to `_profile` (`executor.py:339-346`); `profile_sample` picks from them; `_run` skips the done ones; `_profile` then overwrites `_profile.json` with the empty measurement. No guard stops a stage whose jobs complete with no new done unit. |
| D6 | One job's mount silently lost about 0.85 % of its renames | Run job `6ae415040e0338d1bb7c7cd3` left 100 orphan `.tmp-*.part` files, one per lost target (48 sidecars, 25 shards, 27 ledgers), uploaded at a steady 2–7 per minute over the whole job (`uploaded_at`); `os.replace` raised nothing. The fault was confined to that one container's `hf-mount`: across all 258 generations (28,073 files) of the campaign, orphans exist in that generation only, and a uniform 0.85 % would have left about 137 elsewhere. It does not reproduce: 4 synthetic canaries (about 60,000 files, including 3 concurrent CPU-saturated soaks) and 2 production-identical reruns of the same `extract` (3,961 units each; production rename and direct write) lost nothing through the bucket API. Conclusion: a platform transient the pipeline must SURVIVE, not a write protocol to replace. |
| D7 | `summary` takes about 70 min | `cmd_summary` reads through a plain `BucketStore`, not the TF-85 `StoreSnapshot`, so it downloads every ledger with its own call. It also attributes jobs by `run_key` + stage, not variant, so the two `ghost_outfield` variants both show 333 cv jobs. |
| D8 | R1 is not measured, and the audit blesses idle cores | The ledger has no CPU time. `audit_utilization` (`campaign.py:1187-1209`) computes `cores_used` from `workers x inner_workers x thread_budget` (configuration) and counts a lone unit over the budget as a floor at any thread count (`match_outcome_dependence`: 1 core, 7 idle, `ok: true`). |
| D9 | 65 stage rows under-use the box | The §A audit was not executed, so no stage was shaped by measurement. Largest: `ghost_gk` `cache` (1 core, 28.5 min), `ghost_gk` `extract` (3 cores, 21.6 min), `xshot` `select` (1 core, 14.8 min), `xcross` `probe` (1 core, 10.9 min), the `study` floors (5 of 8 cores). `cache` and `probe` run the per-key `read_result` loops of D1, so their walls are probably listing-bound, not compute-bound; they are re-measured after D1 before any shape decision (B review R3-02). |
| D10 | Critical-path single units | `ghost_gk` `importance` is one unit (71–79 min) although permutation importance is per column. `ghost_outfield` `fit` is one 8-thread HistGB fit (89–90 min): an irreducible unit under R3. |
| D11 | The split cost model counts only `startup_s` | The §G-2 cost bound charges each extra job `startup_s` only. It does not count the prelude (checkout, `uv sync`) or the upstream load. The load runs INSIDE the first unit of each worker (`_data(ctx)` calls `ctx.upstream`), so "HF `started_at` to the first unit start" alone also misses it. CORRECTION (2026-10-30, from the unit ledgers): an earlier reading blamed the reload for `ghost_gk` `cv` 36 to 60 and `ghost_outfield` `cv` 67 to 95 job-min. That is wrong. The `ghost_outfield` upstream is one parquet of 156 MB (`default`) or 116 MB (`position_only`); 4.09 GB is the fold's peak RSS, not data read. Job wall minus unit walls is 56 s for the 3 `ghost_outfield` `cv` jobs. The real cause is D16. |
| D12 | Adapter trainers run as one thread-pinned unit | `adapter.py` runs the legacy CLI as one unit under `deterministic_env()`. `match_outcome_dependence`: 43 min, 1 core. |
| D13 | Stale campaign text | RESOLVED (HFU, owner ruling 2026-11-01): `shard_cap = 25` THIS cycle — the "(sharding is Cycle 2)" deferral is removed and `test_cycle1_caps` pins 25. `shard_cap` is the per-stage fan-out ceiling. |
| D14 | No real-mount validation before a full campaign | The local executor gate runs on local disk, where D1 costs milliseconds and D6 cannot occur. |
| D15 | `--watch` hides a halt | `--watch` exits once every instance is done or failed (`all_settled`, `campaign.py:280-293`), but it returns 0 even when an instance halted, so a wrapper cannot tell a halted campaign from a finished one. (In this campaign it kept running only because D5's loop kept an instance active; an earlier "never ends" reading was wrong.) |
| D16 | The `cv` thread opt-in (HFU T11) gives no speedup | Per-fold wall from the unit ledgers. `ghost_outfield`: 1,148–1,272 s at 1 thread (`74d57ba`, 3 folds in parallel) and 1,036–1,290 s at 8 threads (`2474b73`, one fold at a time). `ghost_gk`: 719–782 s and 689–770 s. So the folds now run one after another at the same per-fold wall: more job-minutes, and a longer wall (`ghost_outfield` `cv` 67 to 96 min). The identity gate proved the 8-thread output byte-identical; no gate checked a speedup. Root cause, measured on HF hardware (read-only canary job `6ae4a0dfc78863f9c62e806e`, AMD EPYC 7R13, one real `ghost_outfield` fold at `OMP_NUM_THREADS=8`): fold 1,075 s = load 0.6 s + `_cv_splits` 10 s + HistGB fit 50 s at 7.33 cores + **`predict_mean` 1,012 s at 1.00 core** + metrics 0.1 s. The HF job env (`uv sync --extra train --group hfjobs`, 55 packages) has NO numba, so the leaf walk runs the numpy fallback body of `_vectorized_leaf_values`, which fancy-indexes the structured node arrays per tree and per level, on one core. The DGX controller env has numba, so a DGX measurement would have hidden this. `ghost_gk` `cv`, `fit` and `importance` and `ghost_outfield` `fit` (its coherence block predicts every training row) run the same walk. |

### r3.3 Designs

Each design names the test that proves it. Tests come first (TDD). "R4" below names the no-rebaseline
requirement of § "Owner requirements": every output is byte-identical to the single-thread, single-process
artifact, proven by a committed gate (B review R3-05).

- **D1 — a linear reduce, and a `read_result` that never lists.** `read_result` finds a result by probing its
  candidate files in `_RESULT_SUFFIXES` order (each probe is one read attempt; `read_bytes` returns `None` for
  an absent file) instead of listing the directory, so EVERY caller is linear, the three module-level loops
  included. A caller that already holds the listing passes it as `names`. Each result's bytes are read once and
  verified against the sidecar on those same bytes. The reduce selects done units from ONE listing (presence),
  checks completeness before any read, reads each unit once through a bounded thread pool (I/O-bound) in reduce
  order, and writes `_summary.json`/`_inputs.json` only after every read verified, so the output is
  byte-identical and a poisoned unit never leaves stale metadata. A unit that fails its checksum is deleted and
  the reduce refuses (INV-3, unchanged). `reduce_done_order` selects by presence, and each sub-reduce verifies
  only its own range. The reduce logs its unit count, periodic progress and its end. D1(b), found while
  implementing: every run job starts with `unit_states(verify=True)` (INV-3: re-hash before skipping), which
  re-hashed its units ONE AT A TIME; a reduce-only job (the D3 resume) re-reads all 3,961 extract shards first
  (about 24 min at the measured 0.36 s per key). The re-hash runs through the same bounded pool, semantics
  unchanged (`test_unit_states_verifies_through_a_bounded_pool`). Tests: a
  counting store proves `read_result` without `names` never calls `names()`, that a reduce lists a constant
  number of times and reads every unit result once, and that the three module-level sites list a constant
  number of times;
  a scale test drives `reduce_stage` and `reduce_subrange` through `tests/_perf_structural.assert_subquadratic_growth`
  on the number of directory entries visited (sizes 256/1024/4096, exponent at most 1.5; the current code is
  quadratic and fails it); the parity harness proves the output is byte-identical to the current reduce.
- **D2 — a reduce that can finish on the first run.** OWNER APPROVED (2026-10-30). (a) D1; (b) estimate the
  reduce wall before any reduce has run: the profile reads its sample back and combines it, timed, so the
  per-unit reduce cost is measured; the estimate is units × per-unit cost ÷ the reduce's read workers; the stage
  takes the sharded reduce when the estimate exceeds the per-job budget, and the reduce timeout is sized from the
  estimate; the first real reduce wall replaces the estimate; (c) execute Step 0 and commit its measurements where
  `committed_history` reads them. The profile reads back units its OWN container just wrote, which a local write
  cache may serve, while the real reduce reads cold in a fresh container, so the per-unit read the estimate uses
  is never below the HF-measured cold read for the unit's size: 0.3628 s per unit (job
  `6ae4884ac78863f9c62e7481`, the real 3,961-shard generation, fresh read-only container) plus its size over
  214 MB/s (job `6ae4a0dfc78863f9c62e806e`) (B review R4-03); rung 3 records the estimate beside the first real
  reduce wall and states the ratio. Tests: a fresh store with a heavy shardable reduce plans the sharded path; a
  read faster than a cold HF read is floored.
- **D3 — resume into a reduce that can finish.** After an ERROR, a stage with no pending and no retryable failed
  unit proceeds to its reduce, with a timeout from the D2 estimate, still bounded by `error_resumes`. Tests: a run
  job that timed out with nothing pending yields a reduce action, not a halt; a second timeout of that reduce
  halts with a named reason.
- **D4 — fresh history.** The `live_history` cache keys on the generation AND the set of records it reads (the
  unit ledgers, `_profile.json` and `_reduce.json`; B review R4-10), and never stores `None`. Tests: a stage whose
  ledger lands one step after its result is re-read on the next step and gets a cost; a cached non-`None` cost
  is replaced when a new ledger lands (B review R3-08); a profile record that lands after the ledgers refreshes
  the cached entry.
- **D5 — a profile that always makes progress.** The profile samples PENDING units only, and never overwrites an
  existing `_profile.json`. A zero-progress guard halts the instance with a named reason when the stage's 2 most
  recent SETTLED completed jobs (finished at least `PROGRESS_SETTLE_S` = 600 s ago, so every record they wrote
  has landed) both launched at the stage's CURRENT progress mark: they and everything after them committed
  nothing. A job inside the settle window is ignored, never waited for, so a loop that relaunches every step
  halts about one settle window plus two jobs after it starts (B review R4-01: requiring the NEWEST job to be
  settled let a 60 s loop run forever). The progress mark is a fingerprint (each job's `records` label) of the
  stage's committed records -- every unit result, exclusion marker, failure record WITH its attempt number,
  sub-reduce partial and stage output -- so the sub-reduce jobs of a sharded reduce, which write only partials,
  are progress (B review SPEC-R3-01), and so is a retry: one that fails again only moves its attempt, one that
  succeeds swaps its failure record for a result, and a record COUNT stands still in both (B review R4-02).
  Tests: a repeat profile runs pending units; the guard halts two settled no-op jobs; a 354-job loop at a 66 s
  cadence halts while 9 jobs inside the settle window do not; a settled job whose records landed is progress; N
  sub-reduce jobs plus the final reduce never trip it; two retry shards that fail again, or succeed, are
  progress (the count-based measure halts both, proven).
- **D6 — survive a lossy mount.** The pipeline already detects a lost or partial result at read time (absent =
  pending; a sidecar mismatch = untrusted, deleted and redone, INV-3). With D1 and D3, the 2026-10-30 event
  costs a small re-run of the lost units and a fast reduce instead of a timeout and a halt. Added: (a) a lost
  ledger beside a verified result is repaired by the D7-repair record (`repair_missing_ledgers`) with null
  measurements, and every cost or utilization statistic excludes null measurements instead of failing (B review
  R3-04); (b) each reduce sweeps the generation's orphan `.tmp-*` files whose mtime is older than
  `ORPHAN_TEMP_AGE_S` (6 h, far above the longest single write), so a concurrent reducer's or straggler's
  in-flight temp is never touched (the contract allows two jobs to reduce at once; B review SPEC-R3-10). An
  mtime earlier than the generation's `.checksummed` marker is implausible (a placeholder such as 0 or the
  epoch), reads as unknown, and is never swept (B review SPEC-R3-12); (c) the
  controller report names a
  completed run job whose units were later missing, so a storage fault is visible, not silently absorbed (a
  `kind` job label tells a run job from a profile); (d), found while implementing: a reduce that finds a corrupt
  unit (for example a truncated upload) deletes it (INV-3) and raises `UnitsInvalidatedError`; the job ends cleanly
  with the stage incomplete (`reduce` child exit 4 → `try_reduce` returns False; a controller-launched stage CLI
  exits 0), never in ERROR, because the controller halts on a job that errors with units pending
  (`test_a_reduce_that_finds_a_corrupt_unit_leaves_the_stage_incomplete_not_failed`). A `--run-all` caller has no
  controller behind it, so its incomplete pipeline exits 3 like any incomplete stage (B review R4-07). The
  write protocol (temp + `os.replace`, sidecar first) is unchanged: the evidence does not implicate it, so the
  direct-write ordering question (A review R3-02) does not arise. Tests: a store that drops 1 % of writes at
  random, driven through the controller's plan-launch-resume loop, completes every instance with every unit
  done and the loss reported; a sweep that runs while another writer is mid-write leaves that writer's temp
  alone, both writes complete, and an older orphan is removed.
- **D7 — a fast, correct summary.** `cmd_summary` reads through `StoreSnapshot` (bulk record reads) with the
  step's parallel workers, and attributes jobs by `(run_key, variant, stage)`. Tests: a summary over N instances
  issues O(stages) downloads, not O(units), and its bytes read are bounded by the record files (B review R3-06);
  two variants sharing a run key get their own job counts.
- **D8 — measured R1, at the JOB level.** R1 is a property of a job, so it is measured per job: every job's main
  process writes a job record at exit with its total CPU (`getrusage` of itself plus all reaped children: every
  worker, reduce child and inner-pool process) and its wall from process start. Job utilization is
  `cpu_s / (wall_s × vcpu)`, so 8 concurrent single-thread units read saturated and a lone single-thread unit
  reads one-eighth (A review SPEC-R3-01, B review R3-03). The job record names every stage the job ran; a
  bundled multi-stage job is judged as ONE job (R1 is a job property), and each of its stages shows that job's
  utilization, so a stage's utilization aggregates the jobs that ran it (B review R3-11). The wall base is the
  process start; the summary joins the record with HF's `started_at` and reports the prelude before the process
  (checkout, `uv sync`) separately, as startup overhead (D11). `run_unit`
  also records the unit's `cpu_s` for stage diagnostics (a ledger schema change: every ledger-schema assertion is
  updated; provenance only, no R4 impact, A review R3-03). A floor is only one of the spec's structural shapes (a
  final fit at the full thread budget; a sequential study at its fold width; and, by owner ruling 2026-10-30, a
  stage whose every unit runs at once with the whole thread budget; each with MEASURED use, see §A) or a stage
  that declares a reviewed serial reason (including the owner-ruled "too small to pay" class: summed unit CPU
  under 60 s, §A), enforced by a registry gate: `Stage.serial_reason` / `ItemStage.serial_reason` (blank is
  rejected), reported in the plan JSON, and every declaration EXACTLY equal to its entry in
  `scripts/hf_jobs/serial_reasons.py` (empty until the R3-8 decision table adds reviewed entries). A unit's
  `cpu_s` subtracts the pre-unit CPU of an inner-pool child reaped during the unit (B review R4-11). A lone
  single-threaded unit is never a floor by itself. Tests: a W=8 job of single-thread units audits saturated; a
  lone 1-core unit audits idle; a missing job record audits as unmeasured, never as passing; the registry gate
  fails an unreviewed declaration and a stale entry (`test_every_serial_stage_declares_a_reviewed_reason` and
  its anti-rot test).
- **D9 — every stage shaped by measurement.** Step 0 runs every stage of every campaign module at production
  shape and records, per stage, the part count, the one-thread and eight-thread walls, peak RSS and measured
  cores. The plan lists the per-stage decision (§B shapes; inner pool; split; or a declared floor) for all 65
  failing rows before any opt-in. Every opt-in keeps the thread-identity gate.
- **D10 — split the importance stage.** sklearn's `permutation_importance` draws ONE `random_seed` and permutes
  each column with a fresh `RandomState(random_seed)` (verified in the locked sklearn 1.9.0 and in 1.7.2:
  `_permutation_importance.py:43,277-278,286,300`), so its columns split into units across jobs and the
  per-column scores assemble in column order. Each unit calls the private `_calculate_permutation_scores` per
  column with the seed derived exactly as the public function derives it; a guard test pins that private
  function's signature and fails loudly on an sklearn change (B review R3-07). The compared artifact is the
  serialized importance table the stage prints (mean and standard deviation per feature). Test: the identity gate
  proves the split table byte-identical to the single-unit table. `ghost_outfield` `fit` stays one 8-thread
  unit, reported under R3 with its measured utilization.
- **D11 — the split cost model.** OWNER APPROVED (2026-10-30). Each job's overhead is measured, not assumed:
  (a) the prelude and the environment, from HF's `started_at` to the first unit start (process start would miss
  the prelude, B review R3-11); (b) the upstream load, which runs inside the first unit of each worker:
  `RunContext.upstream` records its fetch (the store read), verify (the sha256) and decode times, and `run_unit`
  writes them into that unit's ledger record, so the load is split out of the unit's compute cost and every job
  reports where its load time goes. The §G-2 cost bound charges each extra job the stage's measured overhead
  (a) + (b) instead of `startup_s` alone; a reduce job's budget charges (a) alone, since it loads no upstream (B
  review R4-09). Tests: a unit that loads its upstream records the three load times and
  a unit that reuses the loaded upstream records none; the cost bound of a stage with a measured overhead
  charges it per extra job; the unit cost excludes the load time.
- **D12 — migrate the adapter trainers.** Each becomes a stage-contract pipeline on the `win_probability`
  pattern: `refs` -> `extract` (`ItemStage`, one unit per match) -> its fit stages (`cv` folds and the final
  `fit`, thread-enabled only behind the identity gate; `xsuccess` HPO trials as a sequential study at its fold
  width) -> `assemble`. Their artifacts stay byte-identical to the committed bundles. That is proven BEFORE any
  HF run by a committed parity test per trainer: the legacy CLI and the new pipeline, on the same pinned small
  corpus, produce byte-identical artifacts; the legacy path is retired only after the test passes, with its
  golden digest committed (B review R3-07). The campaign `verify` then proves it at full scale. `adapter.py`, its
  tests and its registry entries are deleted when no target remains.
- **D13 — campaign text.** The `cycle1.toml` comment states the real role of `shard_cap`. Its value: 20
  (OWNER APPROVED, 2026-10-30).
- **D14 — the validation ladder** (below) is mandatory.
- **D15 — a watch that reports a halt.** When `--watch` settles, it returns non-zero if any instance halted (a
  `failed` report), and 0 only when every instance is done. Tests: a settled report with a halted instance
  gives a non-zero exit; an all-done report gives 0.
- **D16 — a fast, bit-identical leaf walk.** Replace the numpy fallback bodies of
  `_vectorized_leaf_values` and `_vectorized_leaf_indices` (`silly_kicks/tracking/_ghost_gk.py`) with a walk over
  the flat arrays `_FlatTrees` already holds: rows in chunks; per tree, level by level, only the rows not yet at
  a leaf advance; the same comparisons (`x <= num_threshold`; NaN goes to `missing_go_to_left`); the same
  depth-cap guard (`RuntimeError`). Each row still sums its leaf values tree by tree in tree order, so the output
  is bit-identical to the old body BY CONSTRUCTION (the identical float additions in the identical order); the
  chunks write disjoint slices. The walk is SERIAL, and the public API is unchanged; with numba importable the
  existing serial `@njit` kernel still serves. A thread pool over the chunks (a keyword-only `n_threads`, the
  approved r3 design) was built and measured on HF hardware (canary job `6ae4ab410e0338d1bb7cb791`): the walk
  is GIL-bound there (8 threads used 2.8–3.2 cores). The `ghost_outfield` fold predict took 66.0 s at 8
  threads against 52.3 s serial (slower); `ghost_gk` took 11.9 s against 14.1 s (2.2 s, 16 % faster). OWNER
  RULING (2026-10-30, after the round-5 A/B disagreement, B R5-01): the walk ships SERIAL. The `ghost_gk` gain
  is given up for one code path with no thread knob; parallelism comes from the stage units instead. Training
  parallelism comes from the units instead (folds, columns), shaped per stage in the D9 decision table: with the
  predict fixed, a `ghost_outfield` fold is 300 s at 1 thread and 160 s at 8 threads, so five folds in
  parallel (one job, W=5, about 24 GB) beat five 8-thread folds in a row. Rejected: add numba to the HF training
  env. The leaf walk would be bit-identical (ADR-076 gate), but every path that switches backend when numba is
  importable would switch too (ghost-GK KDE `auto` becomes `cpu-numba`, which differs by up to 1e-9 per ADR-076;
  the DAS, pitch-control, ball-carrier, coordination and xT-GK kernels), a rebaseline risk across every campaign
  trainer (R4). Evidence: on HF, the real `ghost_outfield` fold predict took 1,012 s with the old body and 52 s
  with the flat walk, both serial (19×). The local algorithm benchmark (200 trees × 200,000 rows; never a
  planning number) agreed: old body 83.5 s, flat walk 12.7 s, serial numba 6.1 s. Identity: the old numpy body
  is kept verbatim in the tests as the oracle and compared `np.array_equal` for values and indices on random
  trees with NaNs at chunkings that cut the rows unevenly, end to end through both models' `predict_mean` and the
  stored training leaves, and on the REAL committed bundles' trees (500 trees, depth 8, threshold ties, NaNs);
  the depth-cap guard keeps its asymmetric contract. On HF, the real fold-0 predictions of the old body and the
  flat walk on the same fit, and the trainers' `_cv_work` fold-0 results encoded by the store's own `_encode`,
  are compared byte for byte with the 2474b73 campaign's stored results (canary job `6ae4aee00e0338d1bb7cb8a7`):
  822,176 rows, 0 differ, maximum difference 0.0; the fold JSON is byte-identical to the stored one from both
  walks; both trainers' `_cv_work` at OMP 8 and 1 are byte-identical to stored (4 of 4). (Canary job
  `6ae4ab410e0338d1bb7cb791` first reported a mismatch; that was its own bug: it serialized with `_canon_json`,
  which sorts keys, while the store's `_encode` keeps insertion order.)

### r3.4 Validation ladder (mandatory, in order; no full campaign until every rung passes)

1. Unit and integration tests for every design above, written first.
2. The heaviest `ItemStage` at real scale ON HF (owner ruling 2026-10-30: HF is the primary platform, so every
   number that chooses a shape or a value is measured on HF hardware; the DGX is for debugging only):
   `win_probability` `extract` with all 3,961 units, its reduce included, timed against the per-job budget. Code
   not yet pushed runs as a measurement job (owner-gate item 3), which cannot produce an artifact.
3. The committed real-mount canary on HF: write durability (zero lost targets through the bucket API), read
   cost per file, the `extract` reduce at real scale on the mount, and the mtime a FRESH container reports for
   a file another container wrote (current, never a placeholder; the D6(b) sweep relies on it).
4. Step 0: the per-stage audit at production shape, committed where the planner reads it.
5. A canary campaign: a committed small campaign with one instance per module, including the 3,961-unit
   `win_probability` `extract` and its reduce, a variant pair sharing one run key, and a multi-unit `cv` under a
   long-lived `--watch` (B review R3-09). It passes when every instance completes, the measured R1 audit passes,
   no orphan temp or lost output exists, and no zero-progress halt fires. A forced-halt arm (an instance made to
   fail) proves the halt is named, its sibling jobs are cancelled, and `--watch` exits non-zero.
6. The full campaign, then `verify` (R4 identity and the measured R1 audit), then the owner's adopt gate.

## r4 — the optimization audit (2026-10-30)

The owner asked whether an optimization audit of the campaign hot paths would find more, and on its report
approved four items into this cycle (2026-10-30): D17–D20. The report and its evidence are outside the repo
(`D:\Temp\hfu\audit\`). Every number is an HF number unless marked: the `2474b73` campaign summary (510 jobs,
2,426 job-minutes), the read-only HF profile job `6ae4bd0f0e0338d1bb7cbdf2` at the pushed `2474b73`, and
read-only comparisons of the stored `2474b73` results.

### r4.1 What the audit measured

| Unit (HF, `has_numba=false` as in every campaign job) | Frames | Extract | Ball carrier: `_pre_index_frames` / numba-able loop | Peak RSS |
|---|---|---|---|---|
| ghost_gk gradientsports 10502 (30 Hz) | 175,969 | 115.4 s | 58.0 s / 3.5 s | 5.84 GB |
| ghost_gk sportec J03WMX (25 Hz) | 145,967 | 101.1 s | 42.8 s / 2.4 s | 4.74 GB |
| ghost_gk skillcorner 1021404 (10 Hz) | 45,705 | 59.6 s | 13.5 s / 1.4 s | 1.63 GB |
| xcross sportec J03WMX | 145,967 | 179.3 s | 44.8 s / 2.5 s | 3.91 GB |
| xshot sportec J03WMX | 145,967 | 195.1 s | (alive frames) | 4.34 GB |

- The extract stages are 820 of the 2,426 job-minutes (34%). The numba-able carrier loop is 2–3% of a unit;
  the cost is pandas object work per frame (cProfile: `DataFrame.__getitem__` with boolean masks is about half
  of the profiled time; a Series or DataFrame is built per frame; `ids_match` / `canonical_id_series` and
  `Series.apply` run per frame). `_pre_index_frames` builds one DataFrame per frame on every backend.
- Memory caps the extract stages: 4.7–5.8 GB per 25–30 Hz unit, so `ghost_gk` `extract` ran at W=3 on 8 vCPU
  (1.3–1.9 cores used) and some `xcross` units escalated to a whole box.
- The same extraction runs many times (stored results, `DataFrame.equals`, 2 units per family): the 4
  distinct `ghost_gk` extract generations are exact column projections or row filters of one superset
  (faithful features, the sweeper grid); `position_only` = faithful minus 1–4 velocity columns in xshot,
  xcross and ghost_outfield; every public-tier (`xshot17`) xshot and xcross unit equals the same match's
  `sc_extended` (`xshot115`) unit; the 4 `xcross` `probe` runs are identical.
- The controller ran each stage that `default` and `full` share (E11) twice, in the same second: 279
  job-minutes. `_running` keys in-flight jobs by `(run_key, variant)` (`campaign.py:1089`).
- `ghost_gk` `cache` reads its 179 results one by one on one core, 26.5–28.6 min per variant. At `2474b73`
  a read cost 6.07 s (1.39 s/unit with 8 threads); most of that is the per-read listing that D1 removes.

### r4.2 Findings (owner-approved 2026-10-30)

| # | Audit | Problem | Job-min at `2474b73` |
|---|---|---|---|
| D17 | F2 | A generation that two instances share launches once per instance, not once. | 279 |
| D18 | F1 | One extraction runs once per variant and once per tier, not once per source. | ≈590 saveable (ghost_gk 420, xcross 74 + probe 32, xshot 48, ghost_outfield 17) |
| D19 | F5 | `ghost_gk` `cache` reads its unit results serially. | inside 137 |
| D20 | F3+F4 | The extract stages do pandas object work per frame, at the full frame rate, and their memory caps W. | 820 touched |

### r4.3 Designs

- **D17 — work is keyed by `(run_key, generation)`, not by instance (r4 review round 1: B R4S-01, R4S-07;
  A R4-SPEC-03).** A stage's work is its generation in its store. So every job carries the generation of
  each stage it runs (a `gens` label: the 8-hex prefix of each generation, in stage order), and the
  controller keys everything that decides a launch by `(run_key, generation)`:
  - In flight: a stage is in flight when ANY live job carries its generation, whoever launched it. Within one
    step, a launched generation joins the in-flight set at once, so a later instance in input order cannot
    launch it again. So a shared stage launches once, and no instance waits on a stage that no live job
    runs: if the instance that launched it halts and its jobs are cancelled, the next step relaunches the
    stage for the instances that still need it.
  - History: `_error_history` (resume after an ERROR, the D5 zero-progress guard, `_lost_after_completed`)
    and the job-derived counters (`error_resumes` via `_trailing_errors`) read the jobs of a generation, not of
    an instance. Unit attempts are already store-derived (`failed_units` with `attempt`), so they cannot
    double whoever launches.
  - Outcome: a shared stage's failure (attempts exhausted, resume refused, zero-progress) is derived from that
    shared history and the store, so EVERY instance that needs the generation reaches the same halt, with the
    same reason, in the same step. A plan error of one instance changes nothing for the others: the in-flight
    and history sets come from job labels, not from plans.
  - Cancel: a halting instance cancels its live jobs EXCEPT those that carry a generation another
    non-halted instance still needs. When another instance of the same run key has a plan error in this step,
    the halting instance cancels none of its jobs that carry a generation the errored instance held in its
    last plan, and retries the cancel next step (r4 review round 2: B CONSIDER).
  - Labels: `gens` holds the 8-hex prefix of each generation in stage order (8 stages: 71 characters, under
    `LABEL_MAX` = 100). The planner refuses two generations of one store with the same prefix, and the launcher
    refuses a label over `LABEL_MAX` (r4 review round 2: B CONSIDER).
  - Budget: an instance whose ready stages are all in flight takes no share of the step budget.
  - Audit and summary: a job is attributed to the instance that launched it (its `variant` and `tier`
    labels); a stage whose jobs another instance launched is audited once, under the launcher, and is listed
    for the others as `shared with <id>` (never as unmeasured).
  - Outputs do not change: the same jobs run, once. Tests, one per state: two instances that share
    generations with the launcher idle, running, failed, halted on the shared stage, halted on its OWN stage
    while the shared stage is in flight, with a plan error for one step, and with its resume refused; the
    other instance launches, waits, takes over or halts accordingly, and `--watch` settles in every case.
- **D18 — one extraction per source.** The extract tokens declare `feature_set` (and `ghost_gk`'s
  `grid_x_max`) on purpose: the 4.77.1 stale-shard rule says shards with different columns or rows must never
  be interchangeable. D18 keeps that rule. A shared extract emits ONE fixed schema (the superset, with a new
  schema marker in its token, so no older shard is reused), and the per-variant stage that projects or
  filters declares `feature_set` / `grid_x_max`.
  - (a) `ghost_gk`. `extract` (shared by all five variants) runs `prepare_ghost_gk_training_data` with
    `feature_set="faithful"` and the widest variant grid (`grid_spec_from_args(52.5)`). Per row it stores the
    26 faithful features, the labels, `_game_id`, `_provider`, `_keeper` and the keeper visibility, BEFORE the
    keeper-detection filter. Its token declares the extraction inputs without `feature_set` and `grid_x_max`. A
    guard refuses a variant grid that the superset grid does not contain. A new per-variant `ItemStage`
    `select` (the same unit keys) takes over the extract stage's WHOLE contract for its variant (r4 review
    round 1: A R4-SPEC-01, B R4S-03):
    - Work: per unit and per game in sorted game order, exactly the old `_extract_work` tail: the variant grid
      filter; a game whose grid-filtered frame is empty is skipped BEFORE the bias sums and the keeper mask;
      the SkillCorner bias sums (guarded by `prov == "skillcorner" and len(meta)`); the keeper-detection
      filter (fail-closed on an all-null visibility, as before); a game with no detected-keeper frame is
      skipped; the `position_only` projection; `_pack`. A unit with no packed game returns no result, as the
      old extract did.
    - Failures and exclusions: `select` has `max_failures=0`, as `extract` had. The superset `extract` owns
      the exclusion of a unit with no mapped game (`ItemExcluded`, unchanged); `select` excludes the same unit
      with the same reason and details, so every variant's summary reports it as before.
    - Counters: `select` carries the old extract counters (`_UNIT_BIAS`), so `assemble` reads
      `item_counters(ctx, "select")`, and `adopt` reads the counters of the pipeline's ONE counters-bearing
      `ItemStage` (`retrieve`, `adopt.py:160`; `_integrity`, `adopt.py:240-242`) instead of the stage named
      `extract`.
    - `cache` combines the `select` results. Nothing else downstream changes.
    - Tests go through the store round trip (`write_result` → `read_result`, where the visibility column and
      `_game_id` can change type), with fixture units for: an empty grid-filtered game, an all-null visibility,
      an unmapped (excluded) unit, and a unit whose games all lose their keeper frames.
  - (b) xshot, xcross, ghost_outfield. `extract` always computes the faithful set and no longer declares
    `feature_set`. The one consumer of the extract output (`prep` in xshot/xcross, `_data` in ghost_outfield)
    drops the faithful-only columns for `position_only` (`speed`; `ball_speed`; `ball_speed`, `ball_vx`,
    `ball_vy`, `opp_forward_centroid_vx`). The result equals the old `position_only` output column for column.
  - (c) Tiers that read the same dataset share one store. The xshot/xcross `public` tier reads `xshot115`, as
    `sc_extended` does. The run key becomes `{campaign}-{commit7}-{job}-{dataset}` (+ `-r<N>`); the instance
    id becomes `{run_key}/{tier}/{variant}` and every job label carries `tier`. The launcher gives each tier
    only the arms it ships at most (`public` → `public`; `sc_extended` → `public`, `sc_extended`), computed
    off-HF (ADR-123 I2). Every reader of the extract output keeps only the rows (or units) of the union of its
    arm keys before anything else, a no-op when the arms cover the dataset (r4 review round 1: B R4S-02):
    `prep`, and xcross `probe_comparison` (`train_xcross_attempt.py:661-680`, which takes the first
    `PROBE_KEEP` units of a provider: without the restriction the public tier's cohort would become a
    non-public match). Both then declare `arms_json`, so the two tiers keep separate results there.
    Visibility stays a per-run parameter that stamps provenance (ADR-123 I3); the public bundle records the
    same 17 match ids and `reproducibility: "public"`.
    Every keyer of `(run_key, variant[, stage])` moves to the tier-bearing identity (A R4-SPEC-02, B R4S-02),
    with evidence for each site:
    - launch decisions: `_running`, `_error_history`, `_job_records` / `by_stage`, the D5 zero-progress key
      (`campaign.py:802-810, 1089-1106, 1220, 1245, 1608`) are generation-keyed by D17;
    - attribution and cancel: job labels gain `tier`; `_cancel_instance` filters `(run_key, tier, variant)`
      (`campaign.py:1424`), with D17's exception for a generation another instance needs;
    - names and paths: `launch.job_name` (`launch.py:148`) gains the tier; the staging directories become
      `<staging>/<run_key>/<tier>/<variant>/` in `retrieve` (`adopt.py:121`) and `cmd_summary`
      (`campaign.py:1631`);
    - provenance: `env_provenance()["run_key"]` (`_provenance.py:262`) records the new run key; the builder
      traces whether it reaches a shipped file and, if it does, states it in the bundle comparison (a
      provenance field, never a number);
    - `run_key`, `instances`, `Instance.id`, `cmd_status`, `cmd_summary`, `stage_summary`,
      `retrieve`/`verify`/`adopt` (`verify` checks the arm-restricted corpus identity), and their tests.
  - (d) `xcross` `probe`: its output does not depend on the feature set (identical across variants). `probe`
    always calls the extractor with `faithful` and no longer declares `feature_set`. With (c) and D17 it runs
    once per campaign.
- **D19 — the cache reads in a pool, and a broken upstream unit reopens its stage.** `ghost_gk` `_cache_work`
  reads EVERY unit its upstream's reduce combined (the sorted units minus the `excluded` and `failed_keys` that
  the upstream's `_summary.json` records; B r3 R4S-r3-01: units minus exclusions alone would also count a failure
  the stage tolerated under `max_failures > 0` and livelock it), not only the units marked done now, through
  `_read_units` (one listing, `REDUCE_READ_WORKERS` threads, each result verified once), in the same key order.
  Same bytes. The same rule covers `select`'s per-unit read of a superset `extract` unit (B r3 R4S-r3-02): a
  missing or corrupt superset unit reopens `extract` the same way, so no variant can select from a smaller
  corpus. A missing or corrupt unit of a stage that has already reduced breaks INV-3, and
  it must never become a smaller corpus (r4 review rounds 1–2: B R4S-05, R4S-r2-01). Readiness and `reduced` are
  both "`_output` present" (`_stages.py:1310-1311, 1344`), so deleting only the unit would leave the stage
  reduced, nothing would rerun it, and the old `done`-only read would build a corpus without that game,
  silently. So `_read_units` reopens the upstream stage: it deletes the corrupt unit with its sidecar, deletes
  the upstream `_output` (and `_reduce.json`), and raises `UnitsInvalidatedError`. Inside a unit's work,
  `run_unit` re-raises that error instead of recording a unit failure, so `cache` uses none of its attempts;
  the job ends as an invalidated reduce does (D6(d)). The next controller step sees the upstream stage
  unreduced with one pending unit, reruns it, reduces it again (the same `_output` bytes: the reduce is
  deterministic), and then runs `cache`. Tests, through the real readiness logic (`plan` and `next_actions`):
  a corrupt upstream unit, and an upstream unit lost after the reduce; in both the upstream reopens, the unit
  is redone, the upstream `_output` is byte-identical to the clean run's, `cache` keeps its attempts, and its
  result equals the clean run's.
- **D20 — numpy extract (the D16 pattern).** Scope: `_ball_carrier._pre_index_frames`; `ghost_gk`
  `_extract_all_ghost_gk_features` and `extract_ghost_gk_features`; the `prepare_xcross_training_data` frame
  loop and `extract_xcross_features`; the `prepare_xshot_training_data` frame loop and
  `extract_xshot_features`; and the serving paths that share them (`compute_xcross_attempt`,
  `compute_xshot_occurrence`, ghost-GK serving) through the same core. Rules:
  - Bit-identical by construction: the columns become numpy arrays ONCE per match (frame coordinates upcast
    from float32 storage to float64 at the same boundary as the old path, ADR-106); each frame is a slice by
    precomputed boundaries in the old iteration order; each frame calls the SAME scalar functions in the SAME
    order on the SAME argument types (`math.hypot`, `_geo.to_goal_relative_x`; a numpy function applied to a
    pandas object dispatches to the pandas method, e.g. `np.mean(series)` is the NaN-skipping `Series.mean`,
    so the core calls that method or proves equality). Not allowed: `math` → numpy ufunc swaps, re-associated
    sums, a pandas `groupby().mean()` replaced by a manual mean.
  - Output dtypes (r4 review round 1: B R4S-06): the per-frame output rows are still built as before (the same
    one-row DataFrames, the same `pd.concat`), so each column's dtype resolves exactly as before, including
    the data-dependent cases (a `bool`/`float` mix, a `None`). D20 removes the per-frame INDEXING work only:
    boolean masks, per-frame Series, `dropna`, per-frame id comparisons, `Series.apply`.
  - Every public per-frame function keeps its signature and output; it becomes a thin wrapper over the array
    core, so train and serve still share one implementation. Callers (enumerated, B R4S-09):
    `infer_ball_carrier` (27 files, including library consumers that no campaign stage runs:
    `coordination/_windows.py`, `gkdv/_engine.py`, `positioning/_compute.py`, `restdefense/*`,
    `providers/sportec/parse.py`, `id_compat.py`); `_pre_index_frames` (one direct caller outside
    `_ball_carrier`: `calibration/_carrier_objective.py:95`; `tracking/features.py:276` passes `pre=` through
    and `scripts/_coordination_hypotheses.py:79` passes `carrier=`; the dict contract does not change);
    `extract_xcross_features` / `extract_xshot_features` (`causal/opportunities.py`, `tracking/_model_eval.py`,
    2 scripts each).
  - Oracle gates: the old bodies verbatim in the tests, compared with `DataFrame.equals` and dtypes on the
    committed tracking fixture of EVERY provider and on randomized synthetic matches (NaN positions, a missing
    ball, dead-ball frames, a missing keeper, two periods, tied distances, int and str ids, and columns that mix
    `None`, NaN, `bool` and `int` across frames).
  - Memory: the restructure removes the per-frame copies; the peak RSS and the W it allows are measured on HF.
    The exit numbers (per-unit wall and peak RSS) are measured on HF and recorded; no target number is
    decided in advance.

### r4.4 Ladder additions

- Rung 1 includes every D17–D20 test.
- A rung before rung 3, the HF identity run (a read-only measurement job; r4 review round 1: B R4S-04). The
  new code reproduces, byte for byte as encoded by the store's own `_encode`, every stored `2474b73` unit of
  every affected stage:
  - the extract units of the four trainers (the D18 `select` / projection outputs against the stored
    per-variant units; the D20 functions inside them);
  - `ghost_gk` `cache` (D19; it now reads `select`);
  - xcross `probe` and `probe_comparison` (D18(c), (d));
  - the public-tier `prep` of xshot and xcross (D18(c)): byte for byte, except the owner-approved R4-10a
    difference (the arm masks gain the DFL rows; see "No-rebaseline guarantee"), which the run asserts exactly;
  - the D20 serving paths (`compute_xcross_attempt`, `compute_xshot_occurrence`, ghost-GK serving): the
    test-only oracles against the new code on real units in the same job.
  It also records the per-unit wall and peak RSS. A read-only measurement job carries the working-tree patch
  INLINE in its command (gzip + base64, its sha256 verified before `git apply`), so it stages nothing in any
  bucket.

## Interaction and the owner gates

This work amends `cccb702` (PR #296), which rewrites the open PR's head and needs a force-push of
`feat/controller-step-reads`. The owner's approval of this APPROACH is NOT approval of the amend commit or
the push (standing commit rule; A review HFU-SPEC-01, B review D1-SPEC-05). Explicit, separate owner gates,
each showing the exact state:

1. Before the amend: `git commit --amend` happens only after the owner sees the final diff and says yes.
2. Before the push: the force-push of `feat/controller-step-reads` is a separate explicit yes.
3. A campaign launch needs a pushed, CI-green commit for `check_ref`. Step 0 does not wait for one: it runs on
   HF as measurement jobs (the pushed base commit plus the working-tree diff, its sha256 verified in the job
   before `git apply`), launched by the committed `scripts/hf_jobs/measure.py`, which need no push (owner ruling
   2026-10-30: HF is the primary platform; the DGX is for debugging only). They cannot produce an artifact by
   MECHANISM, not convention (B review R4-12): the campaign bucket is mounted read-only, and a measurement job
   that must write (rungs 2 and 3) writes only to a separate canary bucket, `silly-kicks/hf-refit-canary`
   (private; created 2026-10-30 by owner ruling).
   The builder does not force-push to unblock itself.
4. No merge, no adopt, no publish without the owner's per-action gate.

Unblocks the `ghost_gk`/`ghost_outfield` refit at full utilization and the wheel-excluded
`full`/owner-tier bundles.
