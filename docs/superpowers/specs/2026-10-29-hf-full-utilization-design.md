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

### B. The shape rule for every multi-part stage

Each stage takes exactly one of two shapes, chosen by its audit:

- **Units across jobs, 8 threads each** — for a stage whose parts are heavy model fits (`cv` folds, and any
  stage that trains or scores a model per part). The stage declares `thread_safe=True` with its lever
  (`omp` for HistGB, `n_jobs` for xgb). The planner runs it at `W=1`, `inner=1`, so each unit gets the full
  `thread_budget(vcpu, 1, 1) = 8` (§D). Units fan across N jobs (§D, §G). Byte-identity: the thread-identity
  gate plus the sorted-key reduce.
- **Inner pool across processes** — for a stage with many independent light parts whose process-pool
  parallelism fills the box WITHOUT changing a numeric sequence. The stage stays `inner_parallel` with
  `inner_width` sized to the part count, capped by RAM and by 8. A sequential `study`'s inner pool
  parallelizes ONE trial's ~5 CV folds, so its `inner_width` is at most the fold count (~5); the audit
  records the ~3 idle cores as an irreducible floor, it does NOT size the study to 8 (that would need
  parallel trials = a rebaseline). The audit must show the pool genuinely runs its sized width; a `study`
  observed at `--inner-workers 1` is mis-sized and is corrected to the fold width here.

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
  B D1-IMPL-14). Trained weights (`.npz`/`.json`) and other data under those trees are loaded at RUNTIME by a
  fit, not imported, and do NOT change the PLAN the toy train validates, so they are excluded by design (and the
  `datasets.lock` digest already covers the data pins). A dep change, a trainer change, OR a library `.py`
  change re-runs the toy train; it reuses its stored pass only when all are unchanged. A test shows a
  `silly_kicks/**`-only change invalidates the key, and another shows the key is independent of the process cwd.

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

## Interaction and the owner gates

This work amends `cccb702` (PR #296), which rewrites the open PR's head and needs a force-push of
`feat/controller-step-reads`. The owner's approval of this APPROACH is NOT approval of the amend commit or
the push (standing commit rule; A review HFU-SPEC-01, B review D1-SPEC-05). Explicit, separate owner gates,
each showing the exact state:

1. Before the amend: `git commit --amend` happens only after the owner sees the final diff and says yes.
2. Before the push: the force-push of `feat/controller-step-reads` is a separate explicit yes.
3. Step 0 needs a pushed, CI-green commit for `check_ref` before an HF launch. If Step 0 must run on HF
   before the full cycle is done, the mid-cycle amend+force-push that enables it is itself gated by (1) and
   (2) — the owner approves that intermediate state explicitly, or Step 0 profiling runs locally/on the DGX
   without a push. The builder does not force-push to unblock itself.
4. No merge, no adopt, no publish without the owner's per-action gate.

Unblocks the `ghost_gk`/`ghost_outfield` refit at full utilization and the wheel-excluded
`full`/owner-tier bundles.
