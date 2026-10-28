# HFU rung 5: the canary campaign (design, revision 3)

**Status:** revision 3 after the A/B design review round 2 (A APPROVE: `…-rung5-spec-r2-A.md`; B REQUEST CHANGES,
R5B-06 BLOCKING, R5B-07 CONSIDER: `…-rung5-spec-r2-B.md`; R5B-06 settled by an HF measurement, C3). Round 1 (A APPROVE with 3 CONSIDER:
`D:\Development\_reviews\2026-10-31-hfu-rung5-spec-A.md`; B REQUEST CHANGES, 2 BLOCKING, 2 SHOULD FIX, 1 CONSIDER:
`…-rung5-spec-B.md`). Owner instruction 2026-10-31: "Design, review, build, run". Owner ruling 2026-10-31
(AskUserQuestion): the canary runs 13 instances ("11 + public tiers").
**Parent:** spec `2026-10-29-hf-full-utilization-design.md` § r3.4 (the validation ladder), rungs 4 and 5; plan
`2026-10-29-hf-full-utilization-plan.md` § R3-13. **Code base:** `93c572c` (CI green, run 41254432929).

## 1. What rung 5 must prove (spec r3.4)

A committed small campaign with one instance per module, including the 3,961-unit `win_probability` `extract` and
its reduce, a variant pair sharing one run key, and a multi-unit `cv` under a long-lived `--watch`. It passes when:

1. every instance completes;
2. the measured R1 audit passes;
3. no orphan temp file or lost output exists;
4. no zero-progress halt fires.

A forced-halt arm (an instance made to fail) proves that the halt is named, that its sibling jobs are cancelled,
and that `--watch` exits non-zero. Items 1-4 apply to every instance except the forced-halt arm (review R5B-03).

Rung 4 then commits the planner history where `committed_history` reads it, taken from this campaign's summary.

## 2. Decisions

### C1. One campaign file in the cycle-1 bucket

`scripts/hf_jobs/campaigns/canary.toml`, `name = "canary"`, `bucket = "silly-kicks/hf-refit-scratch"` (the bucket
of `cycle1.toml`), with the `cycle1` caps (`target_minutes = 30`, `dispatch_cap = 25`, `shard_cap = 25`,
`attempts = 3`; `attempts` is the HF job-level retry, `launch.py`; the controller's own limits are
`unit_attempts = 2` and `error_resumes = 2`, `planner.Constraints`).

- The MEM gate reads the per-unit peak history only within the campaign bucket (`bucket_unit_peaks` filters run
  keys by `-{job}-`; there is no cross-bucket path), so rung 6 finds the canary's peaks only here (both
  reviewers: required, not just acceptable).
- Rung 4 takes the planner history from this campaign's summary.
- Run keys are `canary-<commit7>-<job>-<dataset>`, so they never collide with `cycle1` run keys.
- The canary bucket `silly-kicks/hf-refit-canary` stays the rung-3 write sandbox.

### C2. Thirteen instances at the cycle-1 computation, plus the halt arm (owner ruling 2026-10-31)

Each instance uses the same job, tier, dataset and variant arguments as its `cycle1.toml` entry, so every stage's
`history_key` equals the full campaign's. "Small" means fewer instances, not smaller data.

| Job | Tier | Dataset | Variants | What it exercises |
|---|---|---|---|---|
| `ghost_gk` | owner | `tc3` | `default`, `position_only` | the variant pair sharing one run key; the 5-fold THREADS `cv` under the long-lived `--watch`; the MEM-gated `extract`; the per-file read lock |
| `ghost_outfield` | owner | `tc3` | `default` | the `coherence` stage; the full-budget HistGB `fit` |
| `xshot_occurrence` | `sc_extended` | `xshot115` | `default` | the n_jobs lever; the THREADS `cv`; the xShot `study` children at budget 2 |
| `xshot_occurrence` | public | `xshot115` | `default` | two tiers of one module on one shared run key (r4 D18 (c)), and its per-instance cancel scope |
| `xcross_attempt` | `sc_extended` | `xshot115` (+ `xcross_probe`) | `default` | the TF-19 stages (`tf19_ablation` pinned, `tf19_perm_fit` THREADS, `tf19_perm` SAT, `tf19_probe`) |
| `xcross_attempt` | public | `xshot115` (+ `xcross_probe`) | `default` | as the public xShot row |
| `receiver_model` | owner | `sb360` | `default` | the per-match `extract` (reviewed floor) |
| `win_probability` | public | `sb_open` | `default` | the 3,961-unit WIO `extract` and its reduce; the `cv` stage |
| `build_territory_ranking_census` | public | `sb_open` | `default` | the census driver |
| `match_outcome_dependence` | public | `sb_open` | `default` | the migrated public trainer |
| `pass_completion` | public | `sb_open` | `default` | the migrated public trainer |
| `xsuccess` | public | `sb_open` | `default` | the migrated public trainer |
| `canary_halt` | canary | none | `default` | the forced-halt arm (C3) |

Coverage (review R5B-04): 13 of the 21 `cycle1` instances. The 8 not run are the other ghost_gk (`sweeper`,
`sweeper_position_only`, `full`), ghost_outfield (`position_only`) and xShot/xCross (`position_only`, both tiers)
variants. Their shared stages (the extracts and the stages a variant shares, r4 D17) are covered; their
variant-only stages get no rung-4 history and are profiled on their first rung-6 run, as before this cycle.

### C3. The forced-halt arm: the toy pipeline, canary-only

- A third registry kind, `CanarySpec` (`name`, `module`, `base_args`), with one entry, `canary_halt`: module
  `tests.scripts._toy_pipeline` (kept under `tests/`, both reviewers), base arguments
  `--n-items 16 --fail-keys 0000 --hold-s 3600`.
- `--fail-keys 0000` fails only unit `0000` of `strict` (zero tolerance) and `sloppy` (tolerant: its reduce drops
  the unit); `gen` keys are joined (`toy__…`), so `gen` and every other stage succeed (review R5B-02: `"*"` also
  failed `sloppy`).
- The halt path, from the code and an HF measurement (review R5B-06; this corrects revision 2). HF retries a
  failed job INSIDE the same job id: with `attempts = 3` and a 60 s command that exits 3, the job ran 215 s of
  running time (three attempts; `JobInfo.retry` is only the configured `attempts - 1`, review R5B-08), against 7 s
  for one 5 s attempt with `attempts = 1`; the logs keep the last attempt only (jobs
  `6ae6a9f3c78863f9c62f6cbf`, `6ae6a9c9c78863f9c62f6cb9`, `6ae6a961c78863f9c62f6ca8`; no secrets, no mounts).
  The store persists across the retries and the executor reruns a failed unit (`_run`: `pending` includes
  `failed`), so unit `0000` fails three times inside ONE job, which ends ERROR. At the next step
  `resume_after_error` returns None (one ERROR in a row, no retryable unit, nothing pending) and `next_actions`
  halts the instance: `strict: 1 unit(s) failed 2 time(s), over the stage's max_failures=0: {...}` (the tail is
  the spent units' failure counts, a dict). Without
  in-job retries the result is the same: one failure per job, so after the second job the unit has failed
  `unit_attempts = 2` times, and `step` computes `next_actions` (which halts on the spent unit) BEFORE it checks
  the ERROR history (found in the build). A SINGLE crash before any unit writes a record — the real run-3 shape,
  the `strict` profile job exits 3 — halts the instance on the FIRST ERROR as a non-resumable error
  (`strict: latest job <id> ended in ERROR, nothing to resume`, HFU rung 5); two ERROR jobs in a row with no
  record (e.g. repeated timeouts) halt via the ERROR-in-a-row stop (`strict: 2 job(s) in a row ended in ERROR`).
  The check accepts exactly these THREE controller halt reasons when they name `strict`, matched by structure (the
  `strict: ` prefix and the reason's fixed words, never its literal tail; review R5-SPEC-04), so the arm does not
  depend on the HF retry semantics, and records which one fired; a step-level fake-API test drives all four cases
  (C4, C7).
- A new toy stage `hold` (one unit, independent of every other stage, declares only `hold_s`) sleeps `hold_s`
  seconds. Under the DAG schedule its job runs while `strict` fails, so the halt must cancel it. A single-unit
  stage may take the tiny (grouped) path rather than a profile job; both use `profile_timeout_s = 7200` (review
  R5B-07). `hold_s = 3600` is longer than the expected halt (one `strict` job with its in-job retries plus a step,
  about 10-20 minutes) and shorter than that timeout; the check finds the job whose `stage` label (the job's
  stages joined by `-`) contains `hold`, and fails the run unless it ended `CANCELED`, so a mistimed arm cannot
  pass vacuously.
- The touch points (review R5B-05): `registry.spec_for` and the new `CANARY` table; `campaign.instances` (no
  dataset, no mounts, the canary args); `run_key` with dataset `none`; `load_campaign` refuses a `CanarySpec` job
  unless the campaign name starts with `canary`; `adopt`, `verify` and `publish` refuse a canary instance; the
  committed campaign summary omits the canary rows (`write_summary`); the registry and compute-contract gates
  (`test_compute_contract` derives its population from `scripts/`, so the toy is outside it; a new guard test
  pins the canary path).
- A committed test (review R5-SPEC-03) runs `python -m tests.scripts._toy_pipeline ... --plan` in a subprocess
  from the repo root and asserts every canary-only guard fires, so a future packaging change cannot turn the halt
  arm into an import-error halt.

### C4. The pass check (scripted)

Commands (review R5B-03: there is no `step` command; `--watch` belongs to `launch` and `resume`):

1. `python -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/canary.toml launch --watch --reports PATH`. The
   new `--reports PATH` option appends each step's report to `PATH` as one JSON line and records the settled
   exit code as the last line (`--watch` today saves neither).
2. `python -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/canary.toml canary-check --reports PATH`. It reads
   the reports, the campaign summary and one bucket listing of every canary run key, and asserts:
   1. every instance except `canary_halt` is done;
   2. every stage's R1 verdict is `ok`, read from `audit_utilization` and the `serial_reasons` registry (the single
      sources; the check adds no rule of its own; review R5-SPEC-01);
   3. no step report carries a `lost` note (spec r3 D6(c)), and the listing shows no `.tmp-*` file under any
      canary run key (the reduce prints its orphan sweep to stderr only);
   4. no instance halted for zero progress;
   5. the `win_probability` `extract` planned 3,961 units, its instance settled `done`, and the `_summary.json`
      that its reduce wrote shows `n_failed` = 0 and `n_done` + `n_excluded` = 3,961. The extract tolerates failed
      units (`max_failures=None`), so `done` alone does not prove that the reduce took every unit. The check
      result carries the counts (reviews R5-SPEC-02, R5B-10 (a), R5B-11);
   6. `canary_halt` halted with one of the three controller reasons that name `strict` (C3; the check records
      which), its `hold` job ended `CANCELED` (the HF status value), and the recorded `--watch` exit code is
      non-zero.

   `PATH` lies outside the repository (`D:\Temp\hfu\…`; review R5B-07); `launch` refuses an existing `PATH` (one
   run per record; `resume` appends; review R5B-10 (c)). Preflight skips the halt arm's dataset (`none`; review
   R5B-09), and its `check_plans` cache key covers the toy module (review R5B-10 (b)). A step-level fake-API test drives the arm
   through the three halt forms (in-job retries -> the failed-units halt; a single crash before any unit record ->
   the non-resumable-error halt, HFU rung 5; two no-record errors in a row -> the ERROR-in-a-row halt) and asserts
   each is named, cancels the `hold` job and makes a settled `--watch` exit non-zero.

### C5. Rung 4 from the rung-5 summary

After a passing rung 5, the planner history (`committed_history`) is written from the canary summary with each
stage's `history_key` and host, and committed with a second owner-approved amend (OWNER GATE amend, OWNER GATE
force-push, CI green; plan R3-13 carries this gate). The rung-3 and rung-5 records go into the plan record in the
same commit.

### C6. Owner gates (review R5B-01)

The canary needs new committed code (`CanarySpec`, the toy `hold` stage, `--reports`, `canary-check`,
`canary.toml`, their tests), so the order is:

1. build test-first, then an A/B implementation review;
2. OWNER GATE: amend PR #296's commit; OWNER GATE: force-push; CI green;
3. OWNER GATE: launch. The canary is a production-like campaign: its jobs receive the HF token from this
   machine's login as a job secret (`launch.py`), unlike the rung-3 measurement jobs, and they write the
   `canary-*` run keys in the cycle-1 bucket;
4. run, `canary-check`, report.

The canary runs the stages as committed at that run commit, not `93c572c` (this corrects revision 1 § 4).

### C7. The ERROR-in-a-row stop is an instance halt (owner ruling 2026-10-31)

Found in the build: the failed-units halt cancels the instance's live jobs (`step` -> `halting` -> `_cancel_scope`),
but a bare ERROR stop only reported the failure and let the instance's other jobs run to the end. Owner
ruling: it is an instance halt, named as before, whose live jobs are cancelled through the same path. Found on the
live run-3 launch (HFU rung 5): a SINGLE non-resumable crash halts the instance on the first ERROR and cancels its
live `hold` sibling, named `strict: latest job <id> ended in ERROR, nothing to resume` (a third accepted form).
Test: `test_the_halt_arm_halts_named_cancels_hold_and_a_settled_watch_exits_1[single crash, no record]` (the `hold`
job is cancelled; RED before the change).

## 3. Cost and time (estimate)

The `2474b73` campaign used 2,426 job-minutes over 510 jobs for every variant and tier. The canary runs 13 of
the 21 instances; the shared extract generations dominate, and the two public tiers reuse the `sc_extended`
store, so the estimate is 45-65 % of that: about 1,100-1,600 job-minutes, about 1.5-3 hours of wall at
`dispatch_cap = 25`. The halt arm adds one `hold` job of at most 60 minutes and two failed `strict` jobs.

## 4. Out of scope

- Any change to the production stage code.
- The full campaign (rung 6) and the adopt gate.
