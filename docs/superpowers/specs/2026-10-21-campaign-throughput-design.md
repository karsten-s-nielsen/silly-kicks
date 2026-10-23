# Spec: Campaign throughput, reduce I/O and content-addressed reuse (Cycle 1b)

**Status:** r13 — owner ruling 13 (Cycle 1 cut, branch off `main`); r12 — owner APPROVED r10 (2026-10-21); r11 and r12 record owner rulings 11 (§5.12 build, §7.1 item 3) and 12 (§5.4 job wall model)
**Date:** 2026-10-21
**ADR:** ADR-129 (new: the reuse, attestation and adoption rule). No local ref and no fetched remote-tracking ref
held an ADR-129 file on 2026-10-21. Re-confirm on `origin` at the commit gate, because ADR numbers are
cross-branch volatile.
**Amends:** ADR-128 (the stage/unit contract, the controller, the planner, P9 code identity, P18 the same-commit
replica), ADR-037 (the adoption rule "HEAD == campaign commit").
**Builds on:** ADR-128, ADR-068 (`group_rows`, no rescan in a loop), ADR-073 (`assert_subquadratic_growth`),
ADR-052 (`for_each`, declared inputs), ADR-037 (clean-tree split), ADR-088 (`publish_model_with_card`).
**Branch:** `feat/compute-campaign-throughput`, off `main` after PR #288 (Cycle 1's code commit) is squash-merged.
Owner ruling 13 (2026-10-21) cut Cycle 1's artifact work, so the branch is created once that merge lands (§10). Until then this spec and its
plan live outside the repo, in `D:\Development\_drafts\`. The reason: the Cycle 1 controller needs a clean tree at
`74d57ba`. On a clean tree the code identity is the commit; one new file makes the tree dirty, and then the code
identity becomes `dirty-<git tree hash>` (`scripts/_provenance.py:264-276`), which changes every token the
controller computes.
**Authoring model:** Opus 5.5. Implementation runs inline, with no subagents and no worktrees.

---

## 1. Executive summary (for reviewers)

The Cycle 1 campaign on `74d57ba` ran 21 instances on real HF. Twenty finished the whole DAG. The 21st
(win_probability) finished all 3961 extract units, but its reduce timed out twice and the controller halted it.

Measurement of the campaign (§2) shows three classes of waste:

1. **A quadratic reduce.** `read_result` lists the generation directory once for every unit. Over the bucket
   FUSE mount that is about 3961 listings of about 8000 entries for one reduce.
2. **A serial critical path.** The controller launches one stage at a time per instance, even when the DAG
   allows two. The planner fills each job to a fixed 60-minute target, so a chain of five stages takes about
   five hours whatever the concurrency cap is. Instances start in a fixed order, so late instances wait.
3. **No reuse across commits.** Every commit gets an empty store, because the run key contains `commit7`. This
   defeats the `cross_commit_version` reuse that 7 modules already declare on 11 stages. A one-line orchestrator
   fix therefore costs a full re-run of every unit. `cross_commit_version` itself is a hand-written string with no
   soundness guard.

This cycle removes all three. Every stage output stays byte-identical for any W, N, completion order and restart
history (INV-1). Reuse across commits is made sound by a closure-based code identity (INV-10) and proven by
attestation: a fresh replica on HEAD or on an ancestor of HEAD, plus a sampled re-run at HEAD (§5.10).

| Area | Change | Section |
|---|---|---|
| Reduce I/O | One listing and one verified read per result, before anything is written; ordered concurrent reads behind a store read gate | 5.1 |
| Reduce timeout | A measured reduce cost model; the planner sizes the reduce job | 5.2 |
| DAG scheduling | Every ready stage of an instance launches; a stage with jobs running can get more jobs for unassigned units | 5.3 |
| Planner | Minimize the makespan within the concurrency cap, with job startup and per-unit store overhead | 5.4 |
| Scheduler | Critical-path-first across instances; `concurrency_cap = 20`; the replica runs beside the main run | 5.5 |
| Fan-out | Short jobs for critical-path stages; the ghost_gk `importance` split rule | 5.6 |
| History | Plan from the committed campaign summary (per-unit costs); no profile job when history exists | 5.7 |
| Errors | A bounded resume after an infra failure; an operator command that clears a halt | 5.8 |
| Reuse | Content-addressed stores; code identity = the stage's locked closure (code + data); fail-closed runtime checks | 5.9 |
| Provenance | ADR-129: commit-free stage outputs; attestation; adoption by token equality at a clean HEAD | 5.10 |
| Controller reads | `--plan` and live history read one listing and per-job aggregates, not every unit record | 5.11 |
| Store writers | One named writer per main-store path; the controller guard stays | 5.13 |
| Job image | A prebuilt, digest-pinned image with the locked environment; no install in the job | 5.12 |

**Predicted effect (an estimate, not a measurement).** The Cycle 1 main run took 4.6 h of wall time, and the
replica was a second full run. With §5.1–5.7, the main run and its replica together take about 2 to 2.5 h. After
this cycle, a commit that changes no closure file of a stage runs that stage only as a 1-unit canary (§5.10). §7
measures the result against declared, model-based bars.

---

## 2. Evidence (measured 2026-10-21, HF Jobs API, campaign `cycle1` on `74d57ba`)

### 2.1 Campaign timeline

| Measure | Value |
|---|---|
| Jobs | 145 |
| Sum of job walls | 38.1 job-hours |
| Lower bound at `concurrency_cap = 10` | 3.8 h |
| Wall to the last of the 20 finished instances | 4.6 h |
| Highest concurrency HF accepted | 18 jobs |
| Hours at concurrency 0 (waiting on an owner decision) | 5.0 h |
| Hours at concurrency 1 (the win_probability reduce tail) | 2.9 h |

### 2.2 The critical path (ghost_gk `default`, minutes from the first job)

| Stage | Jobs | Start | Run (min) | Needs |
|---|---|---|---|---|
| validate | 1 | 0 | 0.9 | — |
| extract (profile) | 1 | 3 | 13.0 | validate |
| extract | 3 | 18 | 54.9 / 26.6 / 27.9 | validate |
| cache | 1 | 73 | 38.5 | extract |
| cv | 1 | 108 | 36.4 | cache |
| fit | 1 | 148 | 11.0 | cache |
| importance | 1 | 163 | 65.1 | cache, fit |
| assemble | 1 | 226 | 0.7 | all |

Observations:
- `fit` waited 36 minutes for `cv`. The DAG does not need that: both need only `cache`.
- The extract partition is unbalanced (54.9 vs 27 min), because LPT costed the unsampled units at the profile p50.
- `cv` has 5 units but ran as 1 job, because the 60-minute target fits all 5 folds in one job.
- `importance` is one unit of 65 minutes.

### 2.3 The quadratic reduce (code reading, `74d57ba`)

`reduce_stage` (`scripts/_stages.py:755`) first calls `unit_states(..., verify=True)` (`:760`). That call lists the
generation once and reads each result and its `.sha256` file. It then calls `read_result` for each done unit
(`:790`). `read_result` (`:467`) calls `store.names(rel_dir)` again for each unit (`:470`) and verifies again
(`:471`). `LocalStore.names` (`:263`) runs `iterdir()` and `is_file()` over every entry. `reduce_stage` writes
`_summary.json` and `_inputs.json` (`:786`, `:789`) before it reads the results.

For win_probability that is about 3961 listings of about 8000 entries over the FUSE mount, plus 5 reads per unit.
The two jobs that ended in `Job timeout` were `6ad844710a35baab53a19998` (it ran 3945 of the 3961 units, then timed
out) and `6ad84c855275124786e21728` (it ran the last 16 units, then timed out in the reduce at the 30-minute timeout
floor). Two owner-approved copies with 4 h and 12 h timeouts (`6ad89c3b0a35baab53a1c7db`,
`6ad8ccde0a35baab53a1df1f`) give the measured baseline for §7.

### 2.4 No reuse across commits

The run key is `{campaign}-{commit7}-{job}-{tier}` (`scripts/hf_jobs/campaign.py:129`). The stage token already
addresses its generation by content (`compute_tokens`, `scripts/_stages.py:322`). 7 modules declare
`cross_commit_version` on 11 stages: ghost_gk `validate` and `extract`; ghost_outfield `extract`; receiver
`extract` and `pool`; win_probability `extract`; xcross `extract` and `probe`; xshot `extract`; census
`fit_prepass` and `score`. But a new commit gets a new store root, so the token never finds the earlier generation.
`adopt` also refuses unless HEAD is the campaign commit (`scripts/hf_jobs/adopt.py:571`).

`cross_commit_version` has one test (`tests/scripts/test_stage_contract.py:63`). It proves that the string replaces
the commit in the token. Nothing fails when the stage code changes and the string stays the same.

### 2.5 Errors the controller cannot resume

- A job that fails before any unit runs (Cycle 1: a transient `Volume mount failed ... 401 Invalid username or
  password` on xcross) leaves no unit record. `resume_after_error` then returns the halt reason at once.
- After `error_resumes` ERRORs in a row, a stage halts. No supported command clears that halt. Cycle 1 needed a
  scratchpad launcher to continue win_probability. That breaks INV-7 (no scratchpad orchestration).

### 2.6 Job startup

Each job runs `git fetch`, a pinned `pip install` of `uv` and a frozen `uv sync` before its first unit
(`scripts/hf_jobs/launch.py:188`). The planner uses the committed median startup (`campaign.planning_constraints`).
Store I/O per unit (the claim, the outcome, the result and its checksum) is not in any cost the planner sees.

### 2.7 A slow controller read

Three runs of a scratchpad launcher (the owner-approved win_probability reduce copies) each took about 32 minutes on
the laptop before the launch call, for ONE instance. Almost all of it is `read_plan` (the module's `--plan` over the
bucket store) and `live_history` (the unit ledger records). The 60-second `--watch` interval is therefore not the
real step time for a large instance. Which of the two reads dominates is not yet measured; the plan measures it
first (§5.11).

### 2.8 Stage outputs that embed the commit

The ghost_gk `fit` and `assemble` outputs write `ctx.prov["commit"]` into their bytes (`scripts/train_ghost_gk.py:646`,
`:749`, `:772`: `training_commit`, `run_commit`, `run_tree_state`). `scripts/hf_jobs/identity.py` already treats
these keys as provenance (`PROVENANCE_KEYS`): they change on every commit even when the weights do not. So today a
generation is not a pure function of its token.

### 2.9 What a unit reads besides imported code

Library modules read package data by a path next to their own file: `silly_kicks/tracking/_ghost_gk.py:267`,
`_ghost_outfield.py:410`, `_receiver.py:30`, `_xcross_attempt.py:98`, `_xshot_occurrence.py:347`,
`silly_kicks/win_probability/_model.py:36`, `silly_kicks/match_outcome/_dependence.py:22`,
`silly_kicks/xtgk/_retention.py:21`. `scripts/_events_admission.py:28` reads
`docs/research/skillcorner_s1_event_validity/verdicts.json`. Environment variables change behaviour:
`SILLY_KICKS_GHOST_GK_PATH`, `SILLY_KICKS_GHOST_OUTFIELD_PATH`, the `SILLY_KICKS_*_FORCE_NUMPY` switches and
`SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS`. An import closure sees none of these.

---

## 3. Owner rulings (2026-10-21)

1. Cycle 1 finishes and releases on `74d57ba` first, because other sessions wait for it. This cycle starts now as a
   spec and a plan outside the repo, then moves to the sub-branch.
2. Scope is all of §5. Nothing in §5 is deferred.
3. Code identity uses approach A: a locked static closure, a CI gate and a runtime fail-closed check (§5.9).
   Approach B (a runtime dependency manifest per result) and approach C (keep the manual version with a gate) are
   rejected.
4. The adoption rule becomes a new ADR (ADR-129), not an ADR-128 amendment.
5. The planner minimizes the makespan. `concurrency_cap = 20`. The replica runs beside the main run.
6. The ghost_gk `importance` stage splits only if the output stays byte-identical.
7. The controller reads (§5.11) and a prebuilt job image (§5.12) are in scope.
8. (Spec review r1, D1) **The attestation model.** Stage outputs are commit-free. A generation is adoptable at HEAD
   only when a fresh replica attested its token on HEAD or on an ancestor of HEAD. Each campaign also re-runs one
   sampled unit per reused stage at HEAD and requires identical bytes. This replaces the ADR-128 P18 rule "a
   same-commit replica of every stage" (§5.10).
9. (Spec review r1, D2) **Store retention.** No automatic deletion of store generations in this cycle.
   `campaign status` reports the store size per job.
10. (Spec review r1, D3) **Acceptance bars** are model-based and declared before the run (§7.2): schedule efficiency,
    planner calibration, the reduce, startup, cost and determinism. A missed bar is a finding for the owner. It is
    never a silent pass, and a bar is never relaxed after the run.
11. (Plan review, 2026-10-21) **(a)** The image build follows the house publish pattern (§5.12): a `job-image/env-*`
    tag trigger, a protected `job-image` environment with the owner as required reviewer, a tag ruleset, and a SLSA
    build-provenance attestation that preflight verifies. **(b)** The reuse demonstration (§7.1 item 3) runs at the
    Cycle 1b artifact commit: the plan table before the push, then a real canary run after the push.
12. (Plan review r3, 2026-10-21) **The job wall model** is the LPT makespan of a job's units on its W workers, not a
    fluid `work / W` (§5.4).
13. (2026-10-21) **Cycle 1's artifact work is cut.** Its replica, rebaseline, adopt, publish and artifact commit move
    into this cycle; PR #288 merges without a version bump (ADR-128 "Cycle 1 outcome"). No Cycle 1 summary is
    committed, so the Cycle 1 cost history comes from a one-time export of the Cycle 1 stores (plan Task 22).

---

## 4. Invariants

INV-2 to INV-9 of ADR-128 apply unchanged. INV-1 keeps its byte-identity statement; its proof changes (§5.10). This
cycle adds two invariants:

- **INV-1 (proof, amended).** INV-1 still says: for a fixed code identity, image, lockfile and dataset, every stage
  output is byte-identical for every W, N, unit completion order and restart history. ADR-128 proved it with a
  replica at the same commit (P18). From this cycle, a token is proven by an **attestation**: a fresh replica that
  produced byte-identical outputs for that token, on HEAD or on an ancestor of HEAD. A reused token is also
  re-checked by a sampled unit at HEAD on every campaign (§5.10).
- **INV-10 Sound reuse.** A stored result is reused only when its token equals the token that HEAD computes. The
  token covers two input classes, each through a named mechanism:
  - **code**: every repo module file in the stage's locked code closure (§5.9);
  - **data and configuration**: every repo data file in the stage's locked data membership, every declared input
    (`declare=`), every dataset identity and every output-affecting environment variable on the registry (§5.9).
  A unit that loads a repo module, opens a tracked repo file, or sees a `SILLY_KICKS_*` variable outside these sets
  fails with `ClosureViolationError` and writes no result. A stage callable reads only the parameters it declares.
- **INV-11 Linear store access.** A stage reduce lists each generation directory at most once and reads each result
  file (including each FileSet member) at most once. `--plan` and `live_history` are linear in the units too. A test
  with a counting store enforces this (ADR-073 style).

---

## 5. Design

### 5.1 Reduce I/O

- **One listing.** `unit_states` returns the listing it read with the states (a result object; §6.2 keeps every
  caller compatible). `read_result(store, base, *, names=None)` uses `names` when the caller gives it, and lists
  the directory only when not.
- **One verified read, before anything is written.** The reduce runs one pass over the done units. It reads each
  result's data bytes and its `.sha256` side file once, checks the digest on those bytes, then decodes them. A
  FileSet result reads its manifest and each member once, and checks each member's digest on those bytes. The
  separate `verify=True` pass in the reduce goes, because this pass does the same work.
- **Order.** The reduce reads and verifies every result first. Only when all results pass does it write
  `_summary.json`, `_inputs.json` and the output. Today it writes the summary first.
- **Corrupt result (INV-3).** A result whose digest fails moves to `<gen>/.corrupt/<utc>/<key>.*` (copy, then
  delete the original files). The unit is then `pending` in the store, and the reduce raises
  `StageIncompleteError` that names it. The next `step` runs the unit again; nothing waits for an operator.
- **Ordered concurrent reads.** The pass reads through a bounded thread pool. It writes into a dict keyed by unit
  key and builds `results` in `reduce_order`. The order never depends on completion order, so INV-1 holds.
- **A store read gate.** Concurrent store reads go through the read gate in `scripts/_dataset_reads.py`, with a
  separate cap, `SILLY_KICKS_STORE_READ_CAP`. The value comes from a probe on real HF (§7.1), recorded in
  `scripts/hf_jobs/datasets.toml` under a `[store]` table. The default before the probe is 4, the measured safe
  value for the dataset mount.
- **Guard.** `tests/scripts/test_reduce_store_access.py` runs a reduce over a counting store for n = 50, 100 and
  200 units, with both a parquet result type that carries real mixed id types (the
  `test_extract_reduce_mixed_ids.py` pattern) and a FileSet result type. It asserts one `names` call per
  generation and one read per file, and it registers `assert_subquadratic_growth` over `reduce_stage` in
  `SCALE_GUARDED`. A second case corrupts one result and asserts: nothing is written, the unit is `pending`, the
  corrupt bytes are under `.corrupt/`, and a second reduce after a re-run succeeds.

### 5.2 Reduce cost and timeout

- The reduce writes a ledger record: wall, results read, bytes read, the read cap, and `in_job` (true when it ran at
  the end of a unit job, `Executor.try_reduce`, `executor.py:243-258`; false for a separate reduce job).
- The planner gets `plan_reduce(n_results, cost, c)`. Before any history, the cost is `n_results × store_read_s`
  with `store_read_s` from the §7.1 probe. With history, it uses the measured seconds per result.
- The reduce job timeout is `max(timeout_floor_s, timeout_safety × predicted_wall)`, the same rule as a unit stage.
  The fixed `2 × target_s` for a reduce (`campaign.py:237`) goes.
- Test: `tests/scripts/hf_jobs/test_planner.py` asserts `plan_reduce` for 0, 1 and 3961 results, with and without
  history, and the timeout floor at both sides.

### 5.3 DAG-parallel stage actions

- `next_actions` returns an action for EVERY stage whose `needs` are all reduced and which has pending units that
  no running job owns. Today it returns at the first unreduced stage, and a running stage stops the walk
  (`campaign.py:241-242`).
- **A stage with jobs running.** Every action records what it owns, in
  `_actions/<pipeline>/<stage>/<token>/<action id>.json` in the instance store, before its jobs launch. All variants of
  a job share one store (`campaign.py:155-161`) and can share unit keys (ghost_gk folds are `int_key` 0..4,
  `train_ghost_gk.py:882`), so the record is keyed by pipeline and stage token, as the partition path is. The
  controller writes it (§5.13). Every job of an action carries the action id as a new HF job label, `action`
  (`labels_for`, `launch.py:101-107`), so the controller maps each running job to its action from the job list alone.
  The action id is unique: `<UTC as %Y%m%dT%H%M%SZ>_<first 8 hex of sha256(instance id, stage names, stage tokens,
  partition)>`, for example `20261021T143000Z_1a2b3c4d`. The hash input names the action itself, so two actions of
  one instance launched in the same second still get different ids. The id contains no `-`, because readers split the
  `stage` label on `-` (`campaign.py:489`, `:716`, `:760`). It matches `LABEL_PATTERN` (`[a-zA-Z0-9_-]*`,
  `launch.py:37`) and the label-value length limit that `labels_for` already checks; a test asserts both, and a test
  launches two actions of one instance in one step and asserts distinct ids. Every partition
  file moves under it: `_partitions/<pipeline>/<stage>/<token>/<action id>/<i>of<n>.json`, so a second grant with the
  same `n` never overwrites the first.
- **Per-attempt identity in the job.** Today the in-job label is `job.variant.stage.shard` (`label_of`,
  `launch.py:88-91`), so job `i` of two actions with the same `n`, or two attempts of one retried job, share a label.
  From this cycle:
  - `label_of` appends the `action` label: `job.variant.stage.shard.action` (through `parse_stage_label`, so the
    R8 fallback gives the same value). Jobs of different actions never share a label. A job record with no `action`
    label (every Cycle 1 job, and any job launched before this cycle) keeps today's value `job.variant.stage.shard`;
    `label_of` never raises on a missing `action`. Cycle 1 startup for bar 4's report comes from the committed
    `startup_s` of the Cycle 1 summary (`ledger.py:119-130`), not from re-joining old job records. A test calls
    `label_of` and `job_startups` on a Cycle 1 job record with no `action` label.
  - The job script makes an **attempt id** at script start, `<UTC as %Y%m%dT%H%M%SZ>_<4 random hex>`, and exports it
    as `SILLY_KICKS_ATTEMPT_ID` (`neutral` in `env_contract.py`). Every per-job file name carries it:
    `<in-job label>.<attempt id>.json`. An HF retry of the same job (`attempts`) starts a new script, so it gets a new
    attempt id and a new file. No per-job file has two writers.
  - Readers join per-job files to HF jobs by the in-job label, and keep every attempt: `job_startups`
    (`ledger.py:82-104`) takes the startup of each attempt record; `live_history` merges unit costs by unit key and
    keeps the record of the attempt that finished the unit.
  - Test (`test_campaign.py`, `test_ledger.py`): a same-`n` second grant and a retried job write distinct cost and
    phase files, nothing is overwritten, and the merge gives one cost per unit.

  Ownership by action type:

| Action | Owns while any of its jobs runs |
|---|---|
| LPT run | the keys of its partitions |
| profile | every unit of the stage (the stage waits for its history) |
| single or grouped job (`n_units <= 2`) | every unit of every stage in the group |
| memory escalation | the keys it names (one per job) |
| round-robin run (`partition is None`, including every `download` instance) | every unit of the stage |
| reduce | the stage |

  The new action takes only the pending units that no running action owns, and partitions them with LPT. Each new
  job keeps the job length limit (§5.4). An action's ownership ends when all its jobs are terminal. Because the
  main store is shared across campaigns and commits (§5.9), the controller reads job states for ownership by the
  store's run key label over ALL jobs on the account, not only the jobs of its own campaign and commit
  (`launch.jobs_for`, `launch.py:261`, filters by both today). So two concurrent campaigns never run the same
  unowned units twice. A test runs two campaigns on one store.
- `step` launches every action of an instance, subject to the global budget (§5.5).
- Grouping (`_group`, `campaign.py:199`) keeps its rule: consecutive single-unit stages join one job when each one's
  `needs` are met by reduced stages or by stages in the same group. A stage in a group is not also launched as its
  own action.
- Tests (`tests/scripts/hf_jobs/test_campaign.py`): a DAG `a → {b, c}`, `{a, c} → d` launches `b` and `c` together,
  and `d` only after both reduce; the real ghost_gk stage graph launches `cv` and `fit` in the same step; a grant of
  k < wanted jobs launches k, and the next step launches jobs for exactly the unowned remainder; a second grant with
  the same `n` writes partition files under a new action id and leaves the first intact; while a profile job runs,
  no second action launches for its stage; a `download` instance with a partial grant launches nothing; two variants
  in one store with equal unit keys do not block each other's actions; a running job maps to its action through its
  `action` label.

### 5.4 Planner: minimize the makespan

- The planner input gains per-job startup (`startup_s`, the committed median) and per-unit store overhead
  (`unit_overhead_s`, measured: claim, outcome, result write, checksum).
- For a unit stage, the planner picks `n_jobs` to minimize the stage's finish time: the longest job wall, where a
  job's wall is `startup_s + max(job_makespan / efficiency, unit p95)` and `job_makespan` is the LPT list-scheduling
  makespan of that job's units on its W workers, each unit costed at its wall plus `unit_overhead_s` (owner ruling 12).
  Units run whole, in waves, so a fluid `work / W` model underestimates a job of long units (13 units of 2000 s on 8
  workers take 2 waves, 4000 s, not 3250 s). The controller writes each partition in LPT order and the executor runs
  a `--shard-keys` list in that order, so the model is the schedule the job really runs. `n_jobs` is at most the jobs the scheduler grants the stage (§5.5). Among plans whose finish
  times are within 5 % of the best, it takes the one with the fewest jobs, because each job costs `startup_s`.
- `target_minutes` stays in the campaign file. Its meaning changes: it is the longest a job may run, not the length
  the planner fills. A stage whose single unit exceeds it is still reported as the irreducible floor.
- The cost prediction stays: `n_jobs × wall × usd_per_hour`. Preflight prints the campaign total and the predicted
  makespan (§7.2 bar 2).
- Tests (`test_planner.py`): the makespan choice for 1, 5 and 3961 units; the 5 % fewest-jobs rule from both sides;
  the job length limit; the grant ceiling.

### 5.5 Scheduler: critical path first

- `step` computes, for every instance, the **remaining critical path**: the longest chain of not-yet-reduced stages,
  each costed at its planned wall (§5.4) or its history.
- It grants the launch budget (`concurrency_cap` minus running jobs) to actions in order of remaining critical path,
  longest first. Ties break by instance id, so the order is a pure function of the state.
- `concurrency_cap = 20` in `cycle1b.toml`. A new preflight check, `check_concurrency`, reads the account's
  concurrent-job limit from HF if the API exposes it. Otherwise it reads the highest concurrency the account accepted
  in the committed summaries (Cycle 1: 18) and warns when the cap is above it.
- **The replica runs beside the main run.** `campaign launch --with-replica 1` steps the main instances, their
  replica instances and their canary instances (§5.10) in one controller loop, under one budget.
- Tests (`test_campaign.py`): the grant order for three instances with known chains; the tie-break by instance id
  (the same state gives the same order across runs); a budget of 0 launches nothing.

### 5.6 Fan-out of critical-path stages

- With §5.4, `cv` (5 folds) becomes up to 5 jobs, because a shorter makespan wins over the filled 60-minute job.
- **ghost_gk `importance` split rule.** The stage is one unit with inner parallelism. The plan reads
  `train_ghost_gk.py` and decides one of two outcomes:
  - **Split:** each permutation-importance group becomes a unit, and the reduce combines the groups in sorted key
    order. Gate 1: `tests/scripts/test_train_ghost_gk_importance_split.py` proves on the toy that the reduced
    output is byte-identical to the one-unit output of `74d57ba`'s code path. Gate 2: an acceptance parity run on
    real HF (§7.1 item 4).
  - **No split:** if any per-group result depends on shared random state, on the evaluation order, or on a
    whole-set quantity, the stage stays one unit. The plan records the reason, and ADR-129 cites it.
- `cache` stays one unit. It builds one feature matrix from all extracts; splitting it changes no wall, because `cv`,
  `fit` and `importance` need the whole matrix.

### 5.7 Plan from history

- `committed_history` reads per-unit costs as well as the stage cost model from the committed summaries. LPT then
  uses the measured wall of every unit, not the profile p50. A new unit (not in any summary) costs the stage p50.
- A stage whose history key has committed history runs no profile job. The Cycle 1 summary
  (`docs/research/hf_campaigns/cycle1.json`, committed in the Cycle 1 artifact commit) is the first history.
- A summary measured on another code identity still costs a stage. Cost history affects only the plan, never an
  output (INV-1), so a stale cost can cost time but never correctness.
- Tests (`test_campaign.py`): a stage with committed history gets a `run` action and no `profile` action; LPT uses
  per-unit history; an unknown unit costs the p50.

### 5.8 Error handling

- **Infra failure resume.** A job ERROR with no unit record in the ledger for its job label is an infra failure (a
  mount error, an image pull, an environment check, a crash before the first unit). The controller resumes it,
  bounded by the same `error_resumes` count. The halt report names the cause class (`infra`, `image`, `timeout`,
  `unit`).
- **Operator resume.** `campaign <file> resume --clear-halt <instance>/<stage> --reason "<text>"` writes an operator
  record (`_operator/<pipeline>/<stage>/<token>/<utc>.json`: who, when, the reason) to the instance store. `_trailing_errors` counts
  ERROR jobs only after the newest operator record. The controller stays stateless: the record is in the bucket.
  The command refuses when the stage is not halted, and it never launches a job itself; the next `step` does.
- This removes the need for any scratchpad launcher (INV-7).
- Tests (`test_campaign.py`): an ERROR job with no unit record resumes, and the third one in a row halts with cause
  `infra`; `--clear-halt` on a halted stage writes the record and the next step launches; `--clear-halt` on a stage
  that is not halted refuses and writes nothing.

### 5.9 Content-addressed reuse

**Stores.** The main run key becomes `{job}-{tier}`: one content-addressed store per job and tier, shared across
commits and campaigns. A generation directory is still named by its token, so two commits with equal tokens share
one generation, and two commits with different tokens never meet. Replica and canary stores stay fresh per campaign
and commit: `{campaign}-{commit7}-{job}-{tier}-r<N>` and `{campaign}-{commit7}-{job}-{tier}-canary`.

**Where a stage is declared.** Every callable a stage declares (`work`, `load`, `reduce`, `reduce_order`, `units`,
`refs`, `key`, `counters`, `declare`; `scripts/_stages.py:131-140`, `:167-178`) is part of the stage's code.
A stage is declared, with all its callables, in ONE **stage module**. A trainer keeps its fit-side stages in the
trainer module and imports its reusable stages from their own stage modules, for example
`scripts/_ghost_gk_extract.py`. The **reusable set** is the 11 stages of §2.4 plus every stage in their upstream
closure: an upstream token enters each downstream token (`_stages.py:344`), so an upstream left in the trainer would
re-run its downstream on every trainer edit. At `74d57ba` that adds win_probability `refs` and census `refs` and
`prefit`; the plan derives the full set from the DAG and the gate asserts it. A CI gate
(`tests/scripts/test_stage_modules.py`) fails when any declared callable of a stage, including a lambda, has a
`__module__` other than the stage module, and when a reusable stage has an upstream outside a stage module.

**Declared parameters.** Every parameter a stage callable reads must reach the stage token, through `params=` or
through the output of `declare=` (ghost_gk `extract` keys its parameters through `declare=`,
`train_ghost_gk.py:303-310`). This is an ADR-128 rule today with no gate; reuse across commits makes a miss unsound,
so this cycle gates it with two parts in `test_stage_modules.py`:
- **The read set (AST).** For each stage module, the gate finds every parameter name a stage callable reads: a direct
  `ctx.params[...]` / `ctx.params.get(...)`; a read through an alias (`p = ctx.params`, then `p["x"]`, as at
  `train_ghost_gk.py:426`); and a read inside a helper that receives `ctx.params` or `ctx`, when the helper is in a
  repo module (the gate follows it). A key it cannot resolve to a literal, or a helper it cannot follow (outside the
  repo, or a dynamic call), fails the gate.
- **The token test (behavioural).** For every stage and every name in its read set, the gate builds the toy context,
  changes that one parameter to a second valid value, and asserts that the stage token changes. A parameter keyed
  through `declare=` passes this test; a parameter the stage reads but never keys fails it.
- Gate tests: a `declare=`-keyed read passes; an aliased read of an unkeyed parameter fails; a whole-dict pass to a
  helper that reads an unkeyed parameter fails (`_competitions(ctx.params)`, `train_win_probability.py:282`, is the
  real-code case the gate must follow).

**The code closure.** The code closure of a stage is the static import closure of its stage module plus the runner
files that serialize results (`scripts/_stages.py`, `scripts/_driver.py`), restricted to repo files
(`silly_kicks/**`, `scripts/**`). It follows every `import` and `from … import` in every code object, including
imports inside functions (`modulefinder` semantics). It resolves imports with the job's `sys.path` (the repo root and
`scripts/`), so the census bare import `from _crossed_icc import …` (`build_territory_ranking_census.py:48-51`) and
`scripts._crossed_icc` resolve to the same file; files are de-duplicated by resolved path. Third-party code is
covered by `uv.lock` and the image digest.

**The data membership.** The data membership of a stage is EXPLICIT. A stage declares it with a new
`data_files=` field: a tuple of repo-relative paths or directory prefixes (a prefix covers its whole subtree,
recursively), for example `("silly_kicks/tracking/_ghost_gk_weights/default",
"docs/research/skillcorner_s1_event_validity/verdicts.json")`. No heuristic derives it: `pyproject.toml` defines no
package-data list (`:236-249`), and a "files near a closure module" rule would put every bundled model into every
tracking token. `closures record` expands each prefix to its sorted tracked files with `git ls-files`, so the lock
is deterministic.
- The file audit hook (below) enforces the declaration: a stage that opens a tracked repo file outside its closure
  and its data membership fails closed. Every CI toy run exercises each stage, so a missing declaration fails in CI
  first; a code path that the toy does not reach still fails closed on HF, never silently.
- **Adopt destinations.** `adopt` writes trained weights into `silly_kicks/tracking/_*_weights/…`
  (`docs/artifacts/manifest.toml:108-249`). Those files enter the token only of the stages that declare them. So an
  artifact commit that changes one model's weights re-runs only the stages that read that model.
  `tests/scripts/test_closure_lock.py` asserts this on the lock: a change to the files under one model's weights
  prefix changes the tokens of exactly the stages that declare that prefix.

**The token.** The token's `code` field is no longer the commit. It is the digest of the source bytes of every
closure file, the bytes of every data-membership file, `uv.lock`, the image digest and the Python version (INV-6),
plus the values of the output-affecting environment variables (below). The existing `declare=` inputs and dataset
identities stay in the token as today.

**The lock.** `scripts/hf_jobs/closures.lock.json` records, for every pipeline stage, the closure **membership** and
the data membership: sorted lists of repo paths. It records no digests: the digests are computed from HEAD at plan
time, so an edit to a member changes the token without a lock update. The lock changes only when the import
structure or the data membership changes, so the lock diff is a review signal.
- `python -m scripts.hf_jobs closures record` rewrites the lock.
- The CI gate `tests/scripts/test_closure_lock.py` recomputes every membership and fails on any difference, with the
  record command in the message.

**Environment variables.** `scripts/hf_jobs/env_contract.py` registers every `SILLY_KICKS_*` variable in one of
three classes. The 19 names at `74d57ba` (`git grep -ohE "SILLY_KICKS_[A-Z0-9_]+"` over `silly_kicks/` and
`scripts/`) are classed here; the plan re-runs the grep and the gate keeps the list complete.

| Class | Rule | Names |
|---|---|---|
| `neutral` | orchestration, provenance, caches or diagnostics; never changes a result byte | `RUN_KEY`, `JOB_LABEL`, `ATTEMPT_ID` (new), `IMAGE`, `IN_HF_JOB`, `DATASET_READ_CAP`, `DATASET_READ_GATE_DIR`, `STORE_READ_CAP` (new), `NUMBA_CACHE`, `CORPUS_CACHE_DIR`, `IMPORTANCE_PID_LOG` |
| `output` | can change bytes; its value (or absence) enters the token | `GHOST_FORCE_NUMPY`, `DAS_FORCE_NUMPY`, `COORDINATION_FORCE_NUMPY`, `COORDINATION_REFERENCE_NUMERICS`, `ASSERT_INVARIANTS`, `OPEN_DATA_REVISION` |
| `forbidden_in_job` | selects a data source, a credential or a model outside the token; set in a job, it fails the job before any unit runs (cause `infra`) | `OPEN_DATA_BASE`, `PINING_MANIFEST_DIR`, `GHOST_GK_PATH`, `GHOST_OUTFIELD_PATH`; and the non-prefixed `PINING_API_URL` (`scripts/_loader_pining.py:74`), `PINING_FOR_THE_DATA_TOKEN` (`:79`), `SB_USERNAME`, `SB_PASSWORD` (`scripts/_sb_open_data.py:76`), every `DATABRICKS_*` name (`scripts/_loader_databricks.py:43-86` and the fixture builders), and the `SK_DAS_*_PYTHON` reference-oracle paths |

(Names are shown without the `SILLY_KICKS_` prefix.) `OPEN_DATA_REVISION` is also in the dataset identity; it is
`output` so that a run outside a dataset record still keys on it. `CORPUS_CACHE_DIR` is `neutral` because a cache
entry is keyed by its content, and the dataset identity covers what the cache holds.
- In a job, any `SILLY_KICKS_*` variable that is not registered fails the job before any unit runs (cause `infra`).
- The CI gate `tests/scripts/test_env_contract.py` scans `silly_kicks/` and `scripts/` with the AST for every read
  through `os.environ[...]`, `os.environ.get(...)` and `os.getenv(...)`. It resolves the key expression: a string
  literal, a module-level name constant, and a constant imported from another repo module (for example
  `_ENV_VAR` in `silly_kicks/tracking/_ghost_gk.py:314`, `OPEN_DATA_REVISION_ENV` in `scripts/_sb_open_data.py:54`).
  A key it cannot resolve to a literal fails the gate. It fails on any unregistered name. A gate test includes an
  indirect read through an imported constant and asserts that the gate finds it.
- The registry and the gate cover EVERY environment read in `silly_kicks/` and `scripts/`, not only
  `SILLY_KICKS_*` names. A fixed list of platform names needs no class (`PATH`, `HOME`, `TMPDIR`, `HF_TOKEN`,
  `HF_HOME`, `HF_HUB_DISABLE_IMPLICIT_TOKEN`, `HF_HUB_OFFLINE`, `NUMBA_CACHE_DIR`, `NUMBA_NUM_THREADS`, and the
  `DETERMINISTIC_VARS` of `scripts/_thread_pin.py:29`, which `launch.py` imports); any other read name must be
  registered. The plan re-runs the full inventory grep and classes every name it finds.

**Hub downloads.** Three library modules call `huggingface_hub.snapshot_download` without a pinned revision
(`silly_kicks/tracking/_ghost_gk.py:2559`, `_xcross_attempt.py:847`, `_xshot_occurrence.py:689`). A file fetched that
way lands in the untracked HF cache, outside the audit hook and outside the token. Rule: every stage process runs
with `HF_HUB_OFFLINE=1`, so such a call fails closed in a job (its cache is empty). A stage that needs a Hub file
declares it with a new `hub_files=` field (repo, pinned revision, path). The job fetches the declared files before
`install_stage_checks` runs, and each `(repo, revision, path)` enters the token. The plan measures whether any
reusable stage reaches these calls today; the rule holds either way. Test: a stage that calls `snapshot_download`
without a declaration fails; a declared file loads, and a revision change changes the token.

**Runtime checks (fail closed).** They run in EVERY process that executes a stage callable: the unit worker
(`executor worker`), the reduce command (`executor reduce`, `executor.py:593`) and the inner-pool children
(`InnerPool`, `spawn` children, `executor.py:97-101`). Each is a fresh interpreter, on HF and in every CI toy run.
The checks are installed by one function, `install_stage_checks(ctx, stage)`, which the worker and reduce entry
points call at start and which the inner pool passes as its process `initializer`.

- **Loading a stage without the trainer.** Today `context_from_json` imports `d["module"].PIPELINE`
  (`_stages.py:945-952`) for the pipeline name, the parameter types and the stage. From this cycle:
  - every stage module exports `STAGES: Mapping[str, Stage | ItemStage]` for the stages it declares;
  - `context_to_json` also writes the pipeline name, every declared parameter's type tag, the stage module of each
    stage, and the tokens (already there, `:939`);
  - a new `stage_context_from_json(text, stage)` builds the `RunContext` from that JSON alone: a `PipelineView` with
    the pipeline name, the restored params and the ONE stage, imported from its stage module. `ctx.upstream` needs
    only the pipeline name and the upstream tokens (`_stages.py:295-305`), both in the JSON.
  - the worker, the reduce command and the inner-pool initializer use `stage_context_from_json`. The full
    `context_from_json` stays for the parent process and the local CLI.
- **Modules.** After the process imports the runner and the stage module, and again after every unit or inner task,
  every loaded module whose file is under the repo root must be in the code closure or in the declared harness set
  (`scripts/hf_jobs/executor.py`, which also holds `InnerPool`, and its imports; none defines a stage callable, and
  `test_stage_modules.py` proves that). Because the process imports nothing else first, the check does not depend
  on load order.
- **Files.** A `sys.addaudithook` hook watches `open` events for TRACKED repo files only. The process reads the
  tracked-file set once at start (`git ls-files` in the job checkout; the same in CI). The rules:

| An `open` of … | Rule |
|---|---|
| a tracked closure module (`.py`) | allowed |
| a tracked data member | allowed |
| `uv.lock` | allowed (it is in every token) |
| a tracked file under a declared dataset root, including a toy corpus root under `tests/` | allowed (the dataset identity covers it) |
| any other tracked file | `ClosureViolationError` |
| an untracked path (`__pycache__/*.pyc`, numba caches, the venv, temp files, the store, dataset mounts) | not checked by this hook |

  A `.pyc` is not a tracked file: it is the compiled form of a module, and loading it means importing that module,
  which the module check validates. The numba caches hold compiled forms of closure modules (ADR-076: bit-identical).
  The venv holds third-party code, which `uv.lock` and the image digest cover.
- **Environment.** The process checks its `SILLY_KICKS_*` variables against `env_contract.py` at start.
- A violation raises `ClosureViolationError`; the unit writes no result.
- Tests (`tests/scripts/test_closure_runtime_check.py`):
  - a dynamic `importlib.import_module` of a repo module outside the closure fails;
  - a module preloaded outside the closure fails at the first check;
  - a lazy import of a closure module inside a unit passes;
  - importing a non-closure module from its `.pyc` fails (by the module check);
  - an `open` of an undeclared tracked data file fails; an `open` of a data member, of `uv.lock`, of a third-party
    file in the venv and of a toy-corpus file passes;
  - an unregistered `SILLY_KICKS_*` variable and a `forbidden_in_job` variable each fail the job;
  - the worker and the reduce command for a reusable stage leave no trainer module in `sys.modules`;
  - an inner task that opens an undeclared tracked file fails.
- Token tests: an edit to a `load` callable and, separately, to a `reduce` callable in a stage module changes the
  token; an edit to a trainer-only module leaves every reusable stage's token unchanged.

**`cross_commit_version` is deleted** from all 11 stages in 7 modules (§6.4), from `Stage`/`ItemStage`, and from the
docs. The stage modules give the reuse the string tried to give, with a soundness guarantee.

**A dirty tree** keeps today's rule: the code identity is `dirty-<git tree hash>` for every stage, so local
development never reuses or poisons a store result. Stores are content-addressed only for a clean tree.

### 5.10 Provenance: ADR-129 (the attestation model)

- **Commit-free stage outputs.** `PROVENANCE_KEYS` (`scripts/hf_jobs/identity.py:19-27`) holds 7 keys. Six of them
  depend on the producing run: `training_commit`, `training_platform`, `run_commit`, `run_tree_dirty`,
  `run_tree_state`, `reproducibility_note`. No stage output contains one of these six, and no stage output holds a
  file whose bytes depend on one (`SHA256SUMS` lists the digest of `metadata.json`). The seventh, `generation`, stays
  in the stage output where a stage writes it today (for example the census writes `ctx.tokens["score"]`,
  `build_territory_ranking_census.py:798`): a token is commit-free from this cycle (§5.9), so `generation` is a pure
  function of the token, and keeping it preserves today's value exactly. The ghost_gk `fit` and `assemble` stages (§2.8, `train_ghost_gk.py:646-648`, `:749`, `:760`,
  `:772-777`) stop writing them and stop writing `SHA256SUMS`. Gate
  (`tests/scripts/test_stage_outputs_commit_free.py`): the AST forbids any read of `ctx.prov` in a stage module, direct
  or by passing `ctx.prov` (or `ctx`'s `prov` attribute) to a helper; every toy run's outputs hold none of the six
  run-dependent keys and no `SHA256SUMS`; a `generation` value in an output equals a stage token of that run.
- **Stamping at `adopt`.** `adopt` writes the provenance keys into the shipped `metadata.json` / `metrics.json`, then
  regenerates `SHA256SUMS` over the final shipped files with the same writer the trainer uses today. The source of
  each key:

| Key | Source at stamping |
|---|---|
| `training_commit`, `run_commit` | the oldest attesting commit that is an ancestor of HEAD (below) |
| `run_tree_state`, `run_tree_dirty` | that attesting job's recorded provenance (always `clean` / `false`: a dirty tree never reaches a shared store) |
| `training_platform` | `resolve_training_platform(<the stage's declared param>, <that job's recorded provenance>)` (`train_ghost_gk.py:229`) |
| `reproducibility_note` | the same `reproducibility(...)` function, called with the stamped `training_commit` |

  The attestation record therefore holds the full `env_provenance` of the main and the replica job (commit, tree
  state, image, platform), not only job ids. Every shipped file keeps the same keys with the same meaning (Hyrum's
  law), and `manifest.py:168` still checks the embedded commit against the manifest.
- **Attestation.** `campaign verify` compares each main generation with the replica generation of the same token,
  byte for byte. Both are commit-free stage outputs, so the comparison is exact; stamping happens only later, in
  `adopt`. When they are identical, `verify` writes `_attest/<stage>/<token>/<commit>.json` to the main store, with
  both jobs' recorded provenance, the job ids and the time. Only `verify` writes an attestation.
- **Plan at HEAD.** For each stage token at HEAD:
  - an attestation on HEAD or on an ancestor of HEAD exists: the main run reuses the generation, and the replica
    skips the stage. The canary re-runs one sampled unit (below);
  - no such attestation exists: the main run computes the stage (or reuses an unattested generation, for example one
    computed on another branch), and the replica computes it fresh. `verify` then attests it on HEAD.
  The controller runs the ancestry test locally with full history (`git merge-base --is-ancestor`); a job never
  needs it.
- **The canary.** For every reused stage with an attestation, the canary instance re-runs ONE unit at HEAD in its
  fresh canary store. The unit is `min(units, key=lambda k: sha256(token + head_commit + k))`: deterministic for a
  (token, HEAD) pair, and a new sample on each commit, so coverage grows over campaigns. For a single-unit stage the
  canary re-runs the whole stage.
  - **Upstream.** A canary unit reads its upstream outputs from the main store, never from the canary store: the
    canary store is opened with a read-through to the main store for upstream generations whose tokens are attested
    and not quarantined. It writes only into the canary store. So the census chain `fit_prepass → prefit → score`
    (`build_territory_ranking_census.py:840-870`) runs one canary unit per stage, each on the attested upstream.
  - **Compare.** Its result bytes must equal the stored bytes for that unit (or, for a single-unit stage, the stored
    stage output).
- **Quarantine.** A canary mismatch moves the whole generation directory to `_quarantine/<stage>/<token>/` in the main
  store, with a record of the canary job, both digests and the time, and removes its attestations. The token then has
  no generation and no attestation, so the next `step` recomputes the stage in the main store and in a fresh replica at
  HEAD, and `verify` decides again. The controller reports every quarantine to the owner as an unsound-reuse finding,
  whatever the recompute shows. `adopt` refuses any generation of a token that has a quarantine record, until the
  recompute and its replica attest it on HEAD.
- **Adoption.** `adopt` requires a clean tree. For each instance it recomputes the tokens at HEAD (`--plan`) and
  requires: every adopted generation's token equals HEAD's token; that token has an attestation on HEAD or an
  ancestor of HEAD; its canary passed (or the token was attested on HEAD by this campaign's replica); it is not
  quarantined. "HEAD == campaign commit" goes.
- **`training_commit`** keeps its meaning: a commit whose code produced these bytes. `adopt` stamps the oldest
  attesting commit that is an ancestor of HEAD, in HEAD's history order, so the citation always resolves in the
  released history. The manifest adds `adopted_at_commit` (the HEAD that adopted it) and `attested_by` (the attesting
  commits). The plan sweeps every reader of `training_commit` first (R5).
- ADR-037 holds in substance: every adopted byte, except the provenance keys that `adopt` stamps, is what the code at
  HEAD produces, proven by the token, an attestation and the canary.
- Tests (`tests/scripts/hf_jobs/test_adopt.py`, `test_campaign.py`): adoption refuses a token with no attestation; it
  refuses an attestation by a commit that is not an ancestor of HEAD; it refuses a quarantined token; it accepts an
  ancestor attestation with a passing canary and stamps the oldest ancestor attester; the plan skips the replica for
  an attested token and schedules exactly one canary unit; a canary mismatch moves the generation to quarantine and
  the next plan recomputes it; a stamped toy artifact passes `SHA256SUMS` and `manifest.py:168`; the canary runs over
  the real census graph (`test_reuse_plan.py`).

### 5.11 Controller reads (the same defect class as §5.1)

- The plan first times `read_plan` and `live_history` separately on the win_probability store, and records both.
- `--plan` over a store lists each generation once (the §5.1 listing) and never reads result bytes.
- The ledger keeps one append-only record per unit. Each job also writes ONE cost aggregate per stage it ran:
  `_ledger/<pipeline>/<stage>/<token>/_costs/<in-job label>.<attempt id>.json` (keyed like the action records, §5.3;
  the label and attempt id are defined in §5.3), written once at
  the end of that job's stage run. No file has two writers.
  `live_history` merges every job's aggregate file. A unit with a record but no aggregate (a job that crashed before
  it wrote its aggregate) is read from its unit record, so a crash never loses history.
- Guard: the INV-11 counting-store test covers `--plan` and `live_history`, with a stage of 5 jobs × 200 units, one
  job missing its aggregate. It asserts one listing per directory and one read per aggregate file, plus unit-record
  reads only for the units of the job that crashed.

### 5.12 Prebuilt job image

- **What it holds.** The base image `python:3.12.14-bookworm@sha256:dbbe4ceb…` (`launch.IMAGE` today), `uv` at the
  pinned `UV_REQUIREMENT`, and a virtual environment made by
  `uv sync --frozen --no-install-project --extra train --extra statsbombpy --group hfjobs` from the committed
  `uv.lock`. The extras are the union of every extra the job registry declares (today only `statsbombpy`,
  `scripts/hf_jobs/registry.py:106`, `:119`, `:131`, `:144`), so one image serves every job spec. It holds only
  public third-party packages, no repo code and no data. The job still fetches the code by commit
  (`git fetch --depth 1`).
- **Definition.** A committed `scripts/hf_jobs/image/Dockerfile`. Its tag is derived:
  `env-<first 12 hex of sha256(uv.lock + Dockerfile + base digest + extras)>`. The launcher pins the image by its
  pushed DIGEST, recorded in `scripts/hf_jobs/image.lock.json` with the tag, the inputs digest, the build time and the
  measured `startup_s` (below).
- **Build (owner ruling 11a).** A GitHub Actions workflow, `.github/workflows/job-image.yml`, mirrors the PyPI
  publish (`publish.yml`: a tag trigger and a protected environment). A `workflow_dispatch` workflow cannot be used:
  it only starts when the workflow exists on the default branch, and this branch does not reach `main` before the
  campaign.
  - The trigger is the push of a tag `job-image/env-<12 hex>` (the derived tag). A repository ruleset lets only the
    owner create `job-image/*` tags. The prefix never starts with `v`, so it cannot trigger the PyPI publish.
  - The publish job runs in `environment: job-image`, with the owner as required reviewer: nothing publishes until
    the owner approves. That recorded approval is the publish yes.
  - The workflow refuses a tag that differs from the derived tag, and an image tag that exists, so the same inputs
    always map to the one recorded digest; a rebuild cannot change every token without an input change.
  - `actions/attest-build-provenance` signs a SLSA build-provenance attestation for the pushed digest.
  - The owner sets the GHCR package public once; it holds only public third-party packages.
- **Registry.** HF Jobs must pull it. The plan probes, on one `cpu-upgrade` job each: (a) a public GHCR image, (b) a
  Docker Hub image, (c) an HF Space image. Decision rule: use the first that pulls with no credential in the job; if
  a registry needs a credential, the token rides `secrets`, never the command line. If none pulls, implementation
  stops at that point and the owner rules.
- **Startup probe (the bar 4 baseline).** `python -m scripts.hf_jobs image probe-startup` launches, on the chosen
  registry, 5 `cpu-upgrade` jobs together. Each runs the real `job_command` for a REAL reusable stage, so the import
  cost matches a campaign job: the win_probability `refs` and `extract` stage modules (pipeline `win_probability`,
  dataset `sb_open`, `--stages refs,extract --profile 1`). `extract` needs `refs` (`train_win_probability.py:405-415`)
  and the executor requires every need reduced (`executor.py:289-291`), so the probe job runs `refs` first and reduces
  it in-job; startup ends at the first unit of either stage. Each job writes into its own fresh probe store
  `image-probe-<tag>-<i>-win_probability-public` (i = 1..5), so the 5 jobs share nothing. A test asserts that the
  probe's stage list is closed: every `needs` entry of each listed stage is in the list. Each job writes the
  job phase record (§7.2). Their median **startup** (defined in §7.2) is recorded as `startup_s` in
  `image.lock.json`, with the 5 values and the job ids, before any campaign uses the image. `preflight` refuses an
  image lock without `startup_s`. The planner uses it as its `startup_s` input until a committed summary measures a
  median on that image.
- **In-job check (fail closed).** `job_command` runs, before the first unit,
  `uv sync --frozen --check --no-install-project --extra train --extra statsbombpy --group hfjobs`: the same flags as
  the image build, so it checks the same environment. The plan confirms the flag spelling at `uv==0.9.28`. A
  mismatch fails the job before any unit runs, as an infra failure with the cause `image` (§5.8). A job never repairs
  the environment on its own.
- **Preflight.** A new check, `check_image`: `image.lock.json` matches the derived tag of the current inputs, the
  registry serves the recorded digest, and the digest's build-provenance attestation verifies
  (`gh attestation verify`). A stale or unattested image refuses the campaign.
- **Identity.** The image digest is part of every stage token (§5.9) and of the provenance record (INV-6).
- **Gate.** `tests/scripts/hf_jobs/test_image_lock.py`: the derived tag equals the lock's tag; `job_command` contains
  the check command above exactly once; and it contains no `pip install` and no `uv sync` without `--check`.
  `test_launch_typed.py` covers a registry spec with the `statsbombpy` extra and one without.

### 5.13 Who writes what in the main store

The controller's store handle today accepts `upload_bytes` under `_campaign/` and `_partitions/` only, and refuses
`write_bytes` and `delete` (`scripts/hf_jobs/store.py:53-64`). This cycle keeps that guard and names every new
writer:

| Path | Writer | Operation |
|---|---|---|
| `_campaign/…` (incl. `predictions.jsonl`) | the controller (`step`) | `upload_bytes` (existing prefix) |
| `_partitions/…` | the controller (`step`) | `upload_bytes` (existing prefix) |
| `_actions/…` | the controller (`step`) | `upload_bytes` (new prefix) |
| `_operator/…` | the controller (`resume --clear-halt`) | `upload_bytes` (new prefix) |
| `_attest/…` | the controller (`verify`) | `upload_bytes` (new prefix) |
| `_quarantine/…`, the move of one generation directory and the removal of that token's attestations | the controller, through ONE method `quarantine_generation(pipeline, stage, token, record)` | copy every generation file and every `_attest/<stage>/<token>/…` file to `_quarantine/…`, write the record, then delete the originals of both; the only controller path that deletes |
| unit results, `.sha256` files, `.corrupt/`, `_summary.json`, `_inputs.json`, stage outputs, `_ledger/…` | the job processes (worker, reduce) | `write_bytes` / `delete` on the job's bucket mount, as today |

Everything else stays refused for the controller: it never writes or deletes a unit result except through
`quarantine_generation`. Tests (`tests/scripts/hf_jobs/test_bucket_store.py`): the controller handle refuses a write to
a unit result path and to a stage output path; each new prefix is accepted; `quarantine_generation` moves every
generation file and every attestation of the token and leaves no original; after it, the plan schedules the stage in
the main run and in the replica.

---

## 6. Scope — every caller of every changed seam (sweep 2026-10-21, `74d57ba`)

### 6.1 `read_result(`
`scripts/_stages.py`, `scripts/build_territory_ranking_census.py`, `scripts/train_ghost_gk.py`,
`scripts/train_xcross_attempt.py`, `scripts/hf_jobs/adopt.py`; tests `tests/scripts/test_stage_contract.py`,
`test_combine_frames.py`, `test_extract_reduce_mixed_ids.py`, `test_train_ghost_gk_sweeper.py`,
`test_train_ghost_outfield.py`, `tests/scripts/hf_jobs/test_executor.py`. The `names=` keyword is optional, so every
caller keeps working. The plan lists which callers loop over units and must pass `names`.

### 6.2 `unit_states(`
`scripts/_stages.py`, `scripts/build_territory_ranking_census.py`, `scripts/train_ghost_gk.py`,
`scripts/hf_jobs/executor.py` (7 call sites), `scripts/hf_jobs/datasets.py`; tests `test_stage_contract.py`,
`test_executor.py`. The return value becomes a result object that still behaves as the `{key: state}` mapping, so no
caller breaks silently; its `names` attribute carries the listing.

### 6.3 Controller and planner
`next_actions(`, `run_key(`, `Action(`, `_job_specs(` and `resume_after_error(` are in
`scripts/hf_jobs/campaign.py`. Of these, `next_actions(` and `run_key(` are also called in
`tests/scripts/hf_jobs/test_campaign.py`. `plan_stage(` is in `campaign.py`, `planner.py` and
`tests/scripts/hf_jobs/test_planner.py`.

### 6.4 `cross_commit_version=` (deleted)
11 call sites in 7 modules: `scripts/build_territory_ranking_census.py` (2: `fit_prepass`, `score`),
`scripts/train_ghost_gk.py` (2: `validate`, `extract`), `scripts/train_ghost_outfield.py` (1: `extract`),
`scripts/train_receiver_model.py` (2: `extract`, `pool`), `scripts/train_win_probability.py` (1: `extract`),
`scripts/train_xcross_attempt.py` (2: `extract`, `probe`), `scripts/train_xshot_occurrence.py` (1: `extract`); test
`tests/scripts/test_stage_contract.py:63`. The field is also in `Stage` and `ItemStage` (`scripts/_stages.py:138`,
`:177`), ADR-128 (P9 row, `:217`) and `docs/howto/compute-pipelines.md:42`.

### 6.5 `code_identity(` and `compute_tokens(`
`code_identity(` is in `scripts/_provenance.py`, `scripts/_stages.py`, `scripts/hf_jobs/datasets.py`; tests
`test_env_provenance.py`, `test_stage_contract.py`. `compute_tokens(` is in `scripts/_stages.py`; tests
`test_stage_contract.py`, `test_trainer_cache_and_providers.py`. `code_identity` keeps its signature for the
dirty-tree rule. The per-stage closure digest is a new function, `stage_code_identity`.

### 6.6 `job_command(` and `launch.IMAGE`
`job_command(` is in `scripts/hf_jobs/launch.py`; tests `tests/scripts/hf_jobs/test_campaign.py`,
`tests/scripts/hf_jobs/test_launch_typed.py`. `IMAGE` is read by `scripts/hf_jobs/preflight.py:82-91`. `IMAGE_ENV`
(`scripts/_provenance.py:227`) keeps its name; its value becomes the prebuilt image digest.

### 6.7 Stage outputs and `training_commit`
The plan sweeps every writer of a `PROVENANCE_KEYS` key in a stage output (at least `scripts/train_ghost_gk.py:646`,
`:749`, `:772`) and every reader of `training_commit` (`scripts/hf_jobs/manifest.py:168`, `:282-295`, the model
cards, `identity.py`) before §5.10 changes them.

### 6.8 Downstream path pins
The plan checks `docs/PRIVATE_CONSUMERS.md` for every private module this cycle renames or moves (the stage modules
of §5.9).

---

## 7. Cycle 1b acceptance

### 7.1 Required outcomes

1. **Store probe.** One `cpu-upgrade` job reads 8000 small files from a bucket store at read caps 1, 2, 4, 8 and 16,
   and records the wall and any EIO. The safe cap is the largest cap with no error and a wall within 10 % of the
   best. `[store] read_concurrency` is recorded from it.
2. **Campaign `cycle1b`** runs every Cycle 1 instance with its replica and canary instances, in one controller loop,
   from the committed `cycle1b.toml`. Every instance finishes. No scratchpad launcher is used.
3. **Reuse demonstration (owner ruling 11b).** It runs at the Cycle 1b artifact commit, so it needs no extra commit.
   That commit changes only `docs/**` files and adopted weight files; for every stage that does not declare a changed
   weight file, the commit changes only files outside the stage's closure.
   - Before the push, `campaign plan --with-replica 1` shows, per stage, zero main-run units, no replica action and
     one canary unit; or a re-run, for a stage that declares a changed weight file, with its predicted cost.
   - After the owner-approved push and a green CI, `campaign launch --with-replica 1` runs the canary units on real
     HF. Every canary unit is byte-identical to its stored unit, the main run takes zero units, and the replica
     skips every attested stage. A re-run launches only after the owner sees its cost. A mismatch quarantines the
     token and is an owner finding before any merge.
   - The CI version runs on toys (`tests/scripts/hf_jobs/test_reuse_plan.py`), with a file the lock proves is in no
     closure (for example `scripts/hf_jobs/campaign.py`; the test asserts its absence from every closure).
4. **The importance split** (if §5.6 splits it): the split output on the real corpus is byte-identical to the
   one-unit output.
5. **All six bars of §7.2 pass,** or each miss is reported to the owner as a finding with its numbers.

### 7.2 Declared bars (owner ruling 10; declared before the run, INV-9)

The `cycle1b` summary (`campaign summary`) computes each bar from the records below and records pass or fail.

| # | Bar | Threshold |
|---|---|---|
| 1 | **Schedule efficiency:** measured makespan ÷ LB. Makespan = last terminal job `finished_at` − first job `created_at`, over every job of the campaign. LB = max(the measured critical path, total job-hours ÷ `concurrency_cap`), where the critical path is the longest chain over each instance's stage DAG with each stage costed at its **ideal wall**: the startup median plus `plan_stage` run on the stage's MEASURED unit walls and overheads with no grant ceiling (the jobs the scheduler would launch with an unlimited budget), plus the stage's measured reduce wall: the SHORTEST wall among the stage's successful §5.2 reduce records for that token (a stage can reduce more than once: two jobs that both see a complete stage, `_stages.py:756-757`, or a re-reduce after a `.corrupt/` re-run; a failed reduce counts in the measured makespan, never in the ideal). A reduce runs either in-job at the end of a unit job (`Executor.try_reduce`, `executor.py:243-258`) or in a separate reduce job, plus the startup median only when the reduce ran as a separate job. The §5.2 record states which (`in_job` true or false). One reduce, so no grant can stretch it. The scheduler cannot stretch that quantity: a partial grant or a late launch changes the measured walls, not the ideal. All inputs come from the ledger and the HF job records. | ≤ 1.25 |
| 2 | **Planner calibration:** measured makespan ÷ the predicted makespan in the FIRST `_campaign/…/predictions.jsonl` record (written at launch); per ACTION, its measured wall (first job start to last job end of that action) ÷ its predicted wall in `predictions.jsonl`, for every action whose measured wall exceeds 10 minutes; a stage with several actions gets one ratio per action | within 0.75–1.25 |
| 3 | **Reduce:** measured reduce wall (its §5.2 ledger record) ÷ the `plan_reduce` prediction in that action's record, for every reduce over 1000 results; INV-11 is proven by the §5.1 test. The `6ad89c3b`/`6ad8ccde` walls are reported beside it | ≤ 1.5 |
| 4 | **Startup:** from the job phase record, no `install` phase exists; median **startup** (defined under the records below: first unit start − script start, minus any dataset fetch) ÷ the **declared startup**: `startup_s` in `image.lock.json`, the median startup over the 5 jobs of the §5.12 startup probe on the image that the campaign uses, recorded before the campaign. The Cycle 1 median (with `uv sync`) is reported beside it, not used as the denominator. The HF pull and queue time (`started_at − created_at`) is reported, not barred, because the job cannot time it | ≤ 1.5 |
| 5 | **Cost:** main-run job-hours of `cycle1b` ÷ main-run job-hours of Cycle 1. "Main run" excludes replica, canary and profile jobs on both sides, and excludes the hand-launched win_probability copies (`6ad89c3b`, `6ad8ccde`) from Cycle 1. Units that `cycle1b` reused are excluded on both sides, by unit key | ≤ 1.1 |
| 6 | **Determinism:** every replica output and every canary unit is byte-identical to its main output | all |

**Records the bars need (each with one writer):**
- **Predictions:** `_campaign/<campaign>/<commit7>/predictions.jsonl`, appended by `step` for every action it
  launches (action id, stage, predicted wall, predicted reduce wall) and, at the first launch, the predicted campaign
  makespan. Append-only; one writer, the controller.
- **Job phases:** `_ledger/_jobs/<in-job label>.<attempt id>.json` (§5.3), one file per attempt, written by the job
  script itself: UTC timestamps at script start,
  after `git fetch`, after the environment check, after the dataset fetch (only for a `download` dataset,
  `launch.py:205-228`), and at the first unit start. An `install` phase exists only if the job installed packages; with
  §5.12 it never does. **Startup** = (first unit start − script start) − (the dataset fetch duration, if any). The
  dataset fetch is data movement, not environment startup; it is reported per instance beside bar 4, not barred.
- **Reduce records:** the §5.2 ledger record. **Unit and job walls:** the existing ledger records
  (`ledger.py:58-104`) and the HF job records (`created_at`, `started_at`, `finished_at`).

---

## 8. Testing (TDD, CI-gated)

- Every behaviour in §5 gets a failing test first. Each §5 subsection names its test file and boundary cases.
- Toys reproduce production properties (the owner rule: production is not the integration test). The reduce test
  uses 200 units, real mixed id types and a FileSet. The DAG test uses the real ghost_gk stage graph. The closure,
  stage-module and environment tests use the real pipeline modules.
- CI is the full-suite gate. Locally only targeted tests, ruff (`uvx ruff@0.15.7`) and scoped pyright run.
- New test files: `test_campaign_cycle1b.py` (the committed campaign file, as `test_campaign_cycle1.py` does),
  `test_reduce_store_access.py`, `test_closure_lock.py`, `test_closure_runtime_check.py`, `test_stage_modules.py`,
  `test_env_contract.py`, `test_stage_outputs_commit_free.py`, `test_train_ghost_gk_importance_split.py` (if split),
  `tests/scripts/hf_jobs/test_image_lock.py`, `tests/scripts/hf_jobs/test_reuse_plan.py`. Extended:
  `test_campaign.py`, `test_planner.py`, `test_adopt.py`, `test_launch_typed.py`, `test_executor.py`,
  `test_stage_contract.py`.
- `test_compute_contract` keeps its population rules. Its `cross_commit_version` checks go.

---

## 9. Deliverables (file map)

| File | Change |
|---|---|
| `scripts/_stages.py` | listing reuse, one verified read before writes, `.corrupt/`, ordered concurrent reads, reduce ledger record, `stage_code_identity`, `data_files=`, `STAGES` export, richer `context_to_json`, `stage_context_from_json` + `PipelineView`, `install_stage_checks`, `cross_commit_version` removed |
| `scripts/_dataset_reads.py` | a second gate for store reads |
| `scripts/hf_jobs/executor.py` | worker, reduce command and inner-pool initializer load the stage without the trainer and install the module, file and environment checks; the job phase record |
| `scripts/hf_jobs/campaign.py` | DAG and partial actions, critical-path scheduler, `{job}-{tier}` run key, replica and canary instances, infra resume, `--clear-halt`, `--with-replica`, attestation-aware plan |
| `scripts/hf_jobs/planner.py` | makespan objective, `plan_reduce`, overheads |
| `scripts/hf_jobs/ledger.py` | per-unit history, per-job cost aggregates, reduce records, operator records |
| `scripts/hf_jobs/store.py` | the controller write allowlist and `quarantine_generation` (§5.13) |
| `scripts/hf_jobs/adopt.py`, `manifest.py` | adoption by token + attestation + canary, provenance stamping, `adopted_at_commit`, `attested_by` |
| `scripts/hf_jobs/closures.py`, `closures.lock.json`, `env_contract.py` | closure and data membership, `closures record` CLI, the lock, the environment registry |
| `scripts/hf_jobs/datasets.toml` | `[store] read_concurrency` |
| `scripts/hf_jobs/image/Dockerfile`, `image.lock.json`, `.github/workflows/job-image.yml` | the prebuilt image |
| `scripts/hf_jobs/launch.py`, `preflight.py` | image digest from the lock, the in-job check, the `action` job label, the job phase record, `check_image`, `check_concurrency` |
| `scripts/hf_jobs/image.py` (`image probe-startup`) | the startup probe and the `startup_s` write to `image.lock.json` |
| `scripts/hf_jobs/campaigns/cycle1b.toml` | the Cycle 1b campaign, `concurrency_cap = 20` |
| 7 pipeline modules + new stage modules | stage declarations moved, `cross_commit_version` removed, ghost_gk outputs commit-free |
| `docs/superpowers/adrs/ADR-129-…md` | the reuse, attestation and adoption rule |
| ADR-128, ADR-037 | amendment notes pointing to ADR-129 (P9, P18, the adoption rule) |
| `docs/howto/compute-pipelines.md`, `AGENTS.md` compute bullet, `CHANGELOG.md`, C4 | updated |

---

## 10. Commit, review, release

- Until the Cycle 1 controller work on `74d57ba` ends (the win_probability fit, the Task 25 replica, retrieve,
  verify, adopt), this spec and its plan stay in `D:\Development\_drafts\`. Reviewers A and B read them there.
- After that, the session creates `feat/compute-campaign-throughput` off the Cycle 1 branch and moves the drafts in.
  When the Cycle 1 artifact commit lands, the sub-branch merges it (artifacts and docs only).
- Commits follow the house rules: one fully-tested code commit, then the Cycle 1b artifact commit (ADR-037), each
  behind an explicit owner yes. No micro-commits.
- The version bump and the release follow Cycle 1's release. This cycle does not bump the version in its code commit.

---

## 11. Risks and open technical items (each with a decision rule)

- **R1 Closure too coarse.** `silly_kicks/__init__.py` may import most of the library, so most stages may share one
  large closure. Rule: measure the closures in the plan. If a stage closure includes modules it does not use only
  because of a package `__init__`, the plan reports it. It does not change any `__init__` in this cycle without an
  owner ruling, because import-time behaviour is public (Hyrum's law).
- **R2 FUSE store concurrency.** If the §7.1 probe shows EIO at every cap above 1, the reduce stays serial but linear
  (§5.1 still removes the quadratic term). Rule: record the probe and continue.
- **R3 HF concurrency limit.** If HF refuses jobs at a cap of 20, the launcher already reports the refusal. Rule:
  lower the cap in `cycle1b.toml` to the highest accepted value, with the evidence in the summary.
- **R4 Shared stores grow** (owner ruling 9). No automatic deletion in this cycle. `campaign status` reports the store
  size per job.
- **R5 `training_commit` semantics.** If any consumer reads `training_commit` as "the commit of the adopting HEAD", the
  new stamping can mislead it. Rule: the §6.7 sweep runs before the change; any such reader is reported to the owner.
- **R6 Audit-hook cost.** An `open` audit hook runs on every file open in a unit. Rule: the plan measures it on the
  win_probability toy; if it adds more than 2 % to unit wall, the hook filters by path prefix first, and the plan
  records the measurement.
- **R8 HF label count.** Every job carries 8 labels from this cycle (7 today plus `action`). `labels_for` checks each
  value, not the count, and the HF per-job label-count limit is not documented in the repo. Rule: the plan launches one
  probe job with the 8 labels before any other change depends on them. If HF refuses it, the fallback folds the action
  id into the `stage` label as `<stages joined by "-">__<action id>`. Both parts use only `LABEL_PATTERN` characters,
  and the action id has no `-`, so the value stays parse-safe. The fallback then also requires: one helper
  `parse_stage_label(value) -> (stages, action_id | None)` in `launch.py`, used by every reader of the `stage` label
  (`campaign.py:489`, `:716`, `:760`, and `label_of`); and a test that the longest real grouped stage label plus the
  27-character suffix stays within `LABEL_MAX` (100, `launch.py:36`). The single-sourced parser is written in this
  cycle in either case, so the fallback changes one function, not its readers.
- **R7 Data membership too wide.** A `data_files=` prefix may cover files a stage never opens. That costs only token
  sensitivity (an unrelated weights change re-runs a stage), never soundness. Rule: the plan reports, per reusable
  stage, the declared data members that it never opens in the toy runs, and narrows a prefix to the opened files when
  the toy run proves it.

### 11.1 Open owner questions
- None open.

---

## 12. Change log

- r1 (2026-10-21): first draft from the owner rulings of 2026-10-21. Owner ruled §5.11 (controller reads) and §5.12
  (prebuilt image) into scope the same day.
- r2 (2026-10-21): addresses spec review r1. Owner rulings 8–10 (D1 attestation, D2 retention, D3 bars).
  - A CCTP-SPEC-01 / B -04: INV-10 restated to two named mechanisms (code closure; data membership, `declare=`,
    datasets, the environment registry); `data_files=`; the `open` audit hook; `env_contract.py` (§4, §5.9).
  - B -01: stage modules hold every callable of a stage; the closure is the stage module plus the runner files;
    `test_stage_modules.py`; load/reduce token tests (§5.9).
  - B -02: INV-1 proof amended; P18 listed in Amends; the attestation model and canary (owner ruling 8; §4, §5.10).
  - B -03: commit-free stage outputs; `adopt` stamps provenance; ancestor-only attestation; `attested_by` (§5.10, §6.7).
  - B -05: the worker imports the stage module only, so the module check does not depend on load order (§5.9).
  - B -06: one verified read before any write; `.corrupt/`; FileSet members once; corrupt-result test (§5.1).
  - B -07: partial actions for unowned pending units; the `download` rule; grant tests (§5.3).
  - B -08: the in-job check runs in `job_command` with the image build flags; one image with the registry's extras;
    the gate changed to match (§5.12).
  - B -09: one aggregate file per job label; many-job guard (§5.11).
  - B -10 / A -07: a named test file and boundary cases per §5 subsection; the importance-split gates; declared bars
    (§5, §7, §8).
  - A -02: retention ruled by the owner (R4). A -03 / B -11: the exact 11 stages (§2.4, §6.4). A -04: 11 sites /
    7 modules. A -05: named preflight checks. A -06 / B -16: header mechanism sentence. B -12: `sys.path`
    resolution and de-duplication. B -13: the no-op file is proven outside every closure. B -14: §6.3 / §6.6 caller
    lists. B -15: the tag-exists refusal, the no-registry rule, the trigger wording.
  - New risks R6 (audit-hook cost) and R7 (package-data width).
- r3 (2026-10-21): addresses spec re-review r2.
  - A R2-01 / B R2-02: the file audit hook checks TRACKED repo files only, with a rule per open class (closure
    module, data member, `uv.lock`, dataset and toy roots; `.pyc`, numba caches and the venv are untracked and left to
    the module check, ADR-076 and `uv.lock`); new pass and fail tests (§5.9).
  - A R2-02 / B R2-01: `STAGES` per stage module, a richer `context_to_json`, `stage_context_from_json` with a
    `PipelineView`; the checks run in the worker, the reduce command and the inner-pool children through
    `install_stage_checks`; no-trainer and inner-task tests (§5.9).
  - A R2-03 / B R2-08: the bar records (`predictions.jsonl`, the job phase record, reduce records), exact bar
    formulas, the pull time reported not barred, and "main run" defined for bar 5 (§7.2).
  - A R2-04 / B R2-03: all 19 `SILLY_KICKS_*` names classed (`neutral`, `output`, `forbidden_in_job`); the AST gate
    resolves literal, module and imported constants (§5.9).
  - B R2-04: `data_files=` is explicit (prefixes, recursive, expanded by `git ls-files`); adopt destinations enter only
    the tokens of the stages that declare them; the lock test for a one-model weights change (§5.9). R7 reworded.
  - B R2-05: the canary reads upstream through the main store's attested generations; quarantine moves the generation
    away, so the stage recomputes, and every quarantine is an owner finding (§5.10).
  - B R2-06: the stamping source of every `PROVENANCE_KEYS` key; `SHA256SUMS` leaves the stage output and `adopt`
    regenerates it; `verify` compares commit-free outputs; the gate forbids any `ctx.prov` read in a stage module
    (§5.10).
  - B R2-07: an ownership record per action, unique action ids in partition paths, ownership per action type (§5.3).
  - B R2-09: the reusable set includes every upstream of the 11 stages (win_probability `refs`, census `refs` and
    `prefit`); the gate checks it (§5.9).
  - B R2-10: the canary key includes the HEAD commit (§5.10).
  - B "outside this round": an AST gate forbids a stage callable from reading a parameter it does not declare in
    `params=` (§5.9).
- r4 (2026-10-21): addresses B's re-review r3 (A approved r3 with no finding).
  - B R3-01: the parameter rule accepts `params=` and `declare=`; the gate finds direct, aliased and helper reads (or
    fails closed) and a behavioural test changes each read parameter and asserts the token moves (§5.9).
  - B R3-02: §5.13 names one writer per main-store path, the new controller prefixes, and `quarantine_generation` as
    the only controller delete; `store.py` in §9; guard tests.
  - B R3-03 and "outside this round": action, cost-aggregate and operator records are keyed by pipeline, stage and
    token; a two-variant test (§5.3, §5.8, §5.11).
  - B R3-04: `generation` stays in the stage output (a commit-free token), so today's value is kept; the stamping
    table no longer lists it (§5.10).
  - B R3-05: bar 1 costs each critical-path stage at its ideal wall from measured unit walls with no grant ceiling;
    bar 2 is per action (§7.2).
  - B R3-06: `HF_HUB_OFFLINE=1` in stage processes; `hub_files=` declares pinned Hub files into the token (§5.9).
  - B R3-07: bar 4's denominator is the startup measured on the image probe job, declared before the run (§7.2).
  - B R3-08: the environment registry and gate cover every environment read, including `PINING_API_URL`,
    `SB_USERNAME`, `SB_PASSWORD` (§5.9).
- r5 (2026-10-21): addresses re-review r4 (A APPROVE with 1 CONSIDER; B APPROVE WITH FINDINGS).
  - A R4-01 / B R4-05: §5.12 adds a startup probe (5 jobs, the median) that writes `startup_s` into `image.lock.json`;
    `preflight` refuses a lock without it; bar 4 cites it.
  - B R4-01: bar 1's ideal wall adds the reduce job (startup plus its measured wall).
  - B R4-02: `quarantine_generation` also moves the token's attestations; the test asserts the stage is re-planned.
  - B R4-03: `PINING_FOR_THE_DATA_TOKEN`, `DATABRICKS_*` and `SK_DAS_*_PYTHON` are `forbidden_in_job`;
    `HF_HUB_DISABLE_IMPLICIT_TOKEN` and `HF_HUB_OFFLINE` join the platform list; `DETERMINISTIC_VARS` cited at
    `scripts/_thread_pin.py:29`.
  - B R4-04: ownership reads job states by the store's run key over all jobs on the account, so concurrent campaigns
    on one store do not double-run; a two-campaign test.
- r6 (2026-10-21): addresses re-review r5 (A APPROVE with 1 CONSIDER; B APPROVE WITH FINDINGS).
  - A R5-01: the job phase record gains an "after dataset fetch" timestamp; startup excludes the dataset fetch, which
    is reported beside bar 4 (§7.2).
  - B R5-01: bar 1 adds every stage's measured reduce wall (in-job or separate), plus a startup only for a separate
    reduce job; the §5.2 record gains `in_job`.
  - B R5-02: the startup probe runs a real reusable stage (win_probability `extract`, `--profile 1`) into a named
    probe store, through `image probe-startup` (§5.12, §9).
  - B "outside this round": every job carries its action id as an `action` label, so a running job maps to its action
    (§5.3).
- r7 (2026-10-21): addresses re-review r6 (A APPROVE with 0 findings, one could-not-verify item; B APPROVE WITH
  FINDINGS).
  - B R6-01: the startup probe runs `--stages refs,extract --profile 1`, one fresh store per job; a closed-stage-list
    test (§5.12).
  - B R6-02: the action id format `%Y%m%dT%H%M%SZ-<8 hex>` matches `LABEL_PATTERN`; a test (§5.3).
  - B R6-03: bar 1 uses the shortest successful reduce record of the token; failed reduces stay in the measured
    makespan (§7.2).
  - A could-not-verify (label count): new risk R8 with a probe job and a fallback (§11).
- r8 (2026-10-21): addresses re-review r7 (A APPROVE with 1 CONSIDER; B APPROVE WITH FINDINGS).
  - A R7-01 / B R7-01: the R8 fallback is `<stages>__<action id>`, which is pattern-safe and parse-safe; the action
    id uses `_`, not `-`; one `parse_stage_label` helper serves every `stage`-label reader (written in this cycle in
    either case); a length test against `LABEL_MAX` (§5.3, §11 R8).
  - B R7-02: the action-id hash covers the instance id, the stage names, the stage tokens and the partition, so
    same-second actions differ; a test (§5.3).
- r9 (2026-10-21): addresses re-review r8 (A APPROVE, 0 findings; B APPROVE WITH FINDINGS).
  - B R8-01: the in-job label gains the action id; a per-attempt id (`SILLY_KICKS_ATTEMPT_ID`, `neutral`) names every
    per-job cost and phase file, so a same-`n` second grant or an HF retry never overwrites a file; readers keep every
    attempt; a test (§5.3, §5.9 table, §5.11, §7.2).
- r10 (2026-10-21): A APPROVE r9 (0 findings); B APPROVE r9 (1 CONSIDER). B R9-01: `label_of` accepts a job record
  with no `action` label (today's value), and Cycle 1 startup comes from the committed `startup_s`; a test (§5.3).
- r11 (2026-10-21): owner APPROVED r10. Owner ruling 11 from the plan: (a) the image build is tag-triggered with a
  protected `job-image` environment, a tag ruleset and a verified build-provenance attestation (§5.12, replacing
  `workflow_dispatch`, which needs the workflow on the default branch); (b) the reuse demonstration runs at the
  artifact commit, the plan table before the push and a real canary run after it (§7.1 item 3).
- r12 (2026-10-21): owner ruling 12 from plan review r3 (B, outside the round): the §5.4 job wall is the LPT makespan of a
  job's units on its W workers; partitions keep LPT order and the executor runs them in that order.
- r13 (2026-10-21): owner ruling 13 — Cycle 1's artifact work is cut; the branch comes off `main` after PR #288 merges;
  the Cycle 1 cost history comes from an export of its stores.

- (2026-10-21) Owner ruling: this cycle is named **Cycle 1b** (`cycle1b.toml`, `cycle1b.json`), so ADR-128 section 11's
  roadmap keeps its numbers (Cycle 2 = the three public trainers; Cycle 3+ = the corpus drivers). Every "Cycle 2" /
  `cycle2` of earlier revisions now reads "Cycle 1b" / `cycle1b`.
