# HFU launch top-up: a stage gets more jobs while its jobs run (design)

**Status:** revision 6d (2026-11-12): §22.2a adds F3, batched counter reads in `retrieve` (owner: "3: fold it
in"), with the review round 11 and 12 answers (§23) and the field check of §22.4a. Revision 6a (approved, "1: approved"): §22 adds Part F, after run 9b: one decision per shared generation,
and a `retrieve` that waits out a rate limit (in C2); §23 answers review round 9. Revision 5a (approved): §19 adds Part E, no launch from a tree that the jobs do not run. Revision 4a (approved): the
refit leaves this cycle, and run 9b is a parity run only (§18: the review round 5 answers). §17 replaces
the parts of §14 that it names, and it is the current text where the two differ. The text below this line is
revision 3b (2026-11-11), after spec review rounds 1 to 4 (§12, §13, §15, §16; A and B approve
revision 3a) and the merge of the TF-46 stoppage-gate PR. §14 is new: the base, Part D (`compare`), launch 9 in
two runs, the release. §14.3 replaces §8 and §14.4 replaces §9; §3 to §7 are the text that A and B approved in
round 2. **Base now:** `main` at
`05f3a692` (4.141.0). **Branch:** `feat/hfu-launch-topup`, created off `main` at `2a0cb92b` and moved to
`05f3a692` (it holds no commit of its own yet). **Owner approval of the scope:** 2026-11-11 ("approved, 5.1: as
recommend, prove it with launch 9"), with the later rulings that §2 and §14.1 list. **Version:** the owner said
"we will do a patch version to signal the change in general" before the refit was part of this cycle. With the
refit the bundled weights change. The number is his to set at the release commit (§14.6 item 13). **Review:** two
independent reviewers, A and B.

This cycle answers the `TODO.md` row "Find the cause of four launch-8 timings before the next HF launch". It has
one change of substance (Part A), three small ones (Parts B, C and D), and launch 9 in two runs: the xShot/xCross
refit of the TF-46 cycle and a full parity run (§14).

## 1. What launch 8 showed (the investigation of 2026-11-11)

Read-only work on the job records and the job logs of launch 7 (`a1a3b52`) and launch 8 (`abcfda8`), the launch-8
driver log, the 36 `fit` unit ledgers and one measurement job. Files: `D:\Temp\hfu-next\timing\` (the record is
`FINDINGS.md`) and `D:\Temp\hfu-next\fit_probe\`. No tracked file changed during the investigation.

| Timing of the `TODO.md` row | Cause | Evidence |
|---|---|---|
| xsuccess `refs` + `extract`: 40 min 28 s (launch 7: 25 min) | The driver's launch rule (§1.1). Not slower compute. | The `extract` units took 149.2 unit-min at launch 8 and 192.2 at launch 7. |
| xsuccess `hpo`: 160.9 min (launch 7: 141.2 min) | Host speed. No code cause. | §1.2 |
| One xsuccess `fit` unit: 655 to 995 s (expected about 430 s) | Every fit ran at about half speed while 12 `fit` jobs ran at once. The unit code is not the cause. | §1.3 |
| xsuccess `fit` stage: 19 min 36 s (expected about 12 min) | It follows from the unit time. | The slowest unit (995 s), the reduce (65 s) and the job overhead. |

### 1.1 The launch rule

`driver.decide` splits ALL the units that a stage can run into at most `max_shards` (12) jobs. `driver.step`
launches only `min(concurrency_cap - in_flight, launch_batch)` launches of the step, in instance order. `decide`
adds no job to a stage that has a job in flight (`_stage_in_flight`: "this stage already has jobs in flight: wait
for them"). So the units of the jobs that were not launched wait until every launched job has ended. Then they are
split into 12 jobs again.

| xsuccess `extract` | Launch 7 | Launch 8 |
|---|---|---|
| Waves | 6 | 7 |
| Jobs per wave | 1, 1, 6, 7, 7, 12 | 6, 7, 4, 4, 5, 6, 5 |
| Units per wave | 329, 302, 1664, 971, 404, 290 | 1980, 1155, 275, 183, 153, 117, 98 |
| Stage wall time | 22.9 min | 36.7 min |

At launch 8 the campaign was at its cap of 25 jobs in 41 of the first 45 steps, with 6.8 free slots on average at
a step. One wave waited 494 s for one job: its package install took 3 min 21 s and its scheduling 106 s. The five
jobs of the last wave each ran the `extract` reduce (144 to 153 s). The necessary work, with all 12 jobs at once,
is about 1.6 min of unit time (149 unit-min on 96 workers), about 1 min of job setup and one reduce of about
2.5 min. So the rule cost about 30 min on the long pole of launch 8, and about 17 min at launch 7: the 25 min
baseline already held this cost. The rule applies to every stage of every instance.

### 1.2 `hpo`

The 30 trials have the same best values in both launches, trial by trial: the same parameters in the same order.
The ratio of the trial times, launch 8 to launch 7, is 0.80 to 1.47; the total is 1.14. Over the stages with equal
unit counts the two launches took 2,060 and 2,137 unit-min (ratio 1.04); single stages range from 0.60 to 1.70.
One job on one host at 1.14 is inside that range.

**Correction to the fit-units spec §17.1.** That text says that the late trials were slower at launch 8 (367 to
670 s against about 285 s). The late trials are slower in BOTH launches (trial 23: 527 s and 671 s). It is the
trial profile, not a slow-down.

### 1.3 `fit`

The 36 fits used 28,741 core-seconds at launch 8 and 10,398 at launch 7 (6 tasks of 6 fits on 3 pool processes),
for byte-identical bundles. The logs rule out: the input read (75.4 MB in 0.1 s), the thread count (one thread per
fit in both launches), the packages (the 57 installed versions are equal), the memory cap (the `rss` backend is a
watchdog), a preparation cost (all 36 units are inside xgboost boosting at 300 s and at 600 s), CPU starvation
(each process had 97 % of a core) and a slower fleet (other `fit` stages ran at launch-7 speed).

The measurement job (`6af4d4e79737d9a9e093649c`, `cpu-upgrade`, commit `2a0cb92b`, read-only, owner go 2026-11-11)
ran the launch-8 units again. Each result equals the launch-8 result (five of five).

| Stage of the job | Unit | Wall time | Launch 8, the same unit |
|---|---|---|---|
| One unit alone | `fold0-cal0` | 336.5 s (boosting 305.7 s, preparation 12 s, prediction 10 s) | 654.9 s |
| Three units at once | `fold0-cal0`, `-cal2`, `-cal1` | 354.0, 355.3, 370.8 s | 654.9, 691.8, 694.0 s |
| One process, two fits | boosting only | 289.0 s, then 314.7 s | - |

So a unit that nothing slows takes about 340 s, and three at once in one job cost 5 to 10 %. The 430 s estimate
of the fit-units spec §3.4 was sound. It is NOT proven that our own 12 jobs slowed each other: the pattern of 12
jobs at once was not run again, and a busy host at that hour is not ruled out. Launch 9 shows it (§5, §8).

Host facts from the same job: AMD EPYC 7R13; the container sees the 64 logical CPUs and the 517 GB of the host;
cgroup `cpu.max` is `800000 100000` (8 CPUs); cgroup `memory.max` is 32,000,000,000 bytes (29.80 GiB);
`_parallel_launch.available_ram_bytes()` read 31,889,846,272 at the job start.

## 2. Scope

| Item | Decision | Source |
|---|---|---|
| A. The launch rule: a stage with jobs in flight gets more jobs | IN | Owner 2026-11-11: "the launch rule is obviously part of this cycle". |
| A. Every keyed launch has a unique tag; this closes a collision that today's code has (§3.4) | IN | Owner 2026-11-11 ("1: yes"), after both reviewers found the collision in round 1. |
| B. `driver._BOX_RAM` equals the real memory limit of the box | IN | Owner 2026-11-11 (the approved proposal, item 2). |
| C. CPU time in the unit ledger | IN | Owner 2026-11-11 (the approved proposal, item 2). |
| Launch 9 as a parity launch | IN | Owner 2026-11-11: "prove it with launch 9". The launch needs its own go. |
| Launch 9 carries the xShot/xCross refit of the TF-46 cycle; the adoption and the Hub publish are this cycle's | IN | Owner 2026-11-11 ("confirmed"). §14.3. |
| `cycle1.toml` stays whole: the refit run first, its adoption, then the full parity run | IN | Owner 2026-11-11 ("Option 1"). §14.3. |
| D. `compare` reads the baseline from several launches | IN | Owner 2026-11-11 ("1 as recommended"). §14.2. |
| A release | IN | Owner 2026-11-11: "we will do a patch version to signal the change in general"; with the refit the bundled weights change. The number is his, and it goes into C3 (§14.4, §14.6 item 13). |
| The number of `fit` jobs at once | OUT | The cause is not proven; at most about 8 min. Launch 9 gives the data. |
| `hpo` (items 5 and 6a of the fit-units cycle) | OUT | Dropped by the owner 2026-11-10; no new evidence. |
| A prebuilt image or a package cache (item 2) | OUT | Dropped by the owner 2026-11-10. Part A removes most of the cost of one slow job. |
| The reduce lease (item 7) | OUT | Dropped by the owner 2026-11-10; the duplicate reduces ran in parallel. |
| The `TODO.md` row "ADR-128 Cycle 3+: corpus-driver burn-down" | OUT | Owner 2026-11-11: "ignore things from the TF-46 related row". |

No `TODO.md` row is opened for an OUT item. The row "Find the cause of four launch-8 timings" leaves `TODO.md`
with this cycle.

## 3. Part A: the top-up rule

### 3.1 Terms

- **Listing:** the job records of the campaign and the commit (`launch.jobs_for`, the HF `list_jobs` call).
- **Flight** of a stage: its jobs that `classify` reports as in flight (running or scheduling), matched by
  `(run_key, variant, stage)`, as `_stage_in_flight` matches them today, plus the jobs that this run launched and
  that the listing does not show yet (§3.3).
- **Manifest** of a job: the units it was launched to run. `step` writes it after the launch
  (`by_job_rel`: `_partitions/<pipeline>/<stage>/<token>/by_job/<job id>.json`). An empty list is a reduce trigger.
- **Held** units of a stage: the union of the unit lists of its flight.
- **Keyed launch:** a launch that names its units in a keys file (`--shard-keys`). Every launch of an instance
  without a `download` dataset is keyed, a reduce trigger included (its keys file is `[]`).

### 3.2 The rule (`driver.decide`)

For the first un-reduced, ready stage of an instance:

1. **No flight:** as today. `decide` computes `to_run`, `memory_units` and `stuck`, makes the STUCK reports, and
   returns the escalation launches of `memory_units` and the launches of `to_run` (each split with `max_shards`),
   or a reduce, or a report.
2. **A flight, and the stage cannot be topped up:** `decide` launches nothing for the stage and makes no report,
   as today. The stage cannot be topped up when one of these is true:
   - the instance is not keyed (it has a `download` dataset and runs `--shard I/N` slices);
   - the unit list of a job of the flight is not known (§3.3);
   - a job of the flight has an empty unit list (a reduce trigger is running).
3. **A flight, and the stage can be topped up:** `decide` computes `to_run`, `memory_units` and `stuck` as in
   case 1, and makes NO report (a stage with a flight makes no STUCK report today; the reports come when the
   flight has ended). `to_run` and `memory_units` hold no stuck unit, as today. Then
   `rest = to_run - held` and `rest_memory = memory_units - held`.
   - Both empty: nothing is launched. A reduce is never triggered while a job of the stage is in flight.
   - `room = max_shards - len(flight)`. No room: nothing is launched.
   - Otherwise the launches are, in this order:
     `_keyed_launches(inst, stage, rest_memory, units_per_job, room, workers, escalate=True)`, and then, when
     `room2 = room - (the number of those escalation launches)` is at least 1,
     `_keyed_launches(inst, stage, rest, units_per_job, room2, workers)`. The fifth argument is the `max_shards`
     of the split, so `rest` is split into at most `room2` jobs: the contiguous near-even split of the sorted
     keys, unchanged.

A top-up job can hold more units than `units_per_job` when the room is small (36 `fit` units of 3 per job: with
11 jobs in flight, one top-up job holds every unit that is left). This is today's `max_shards` rule; the job
sizes its workers itself.

The mount headroom (`mount_cap_for`), the `launch_batch` and the `concurrency_cap` cut the launches as they do
today. The stage barrier (one stage per instance per step; the next stage only after the reduce) does not change.

**Example, launch 8:** step k: 3,961 units, 12 jobs of 330 or 331, the budget is 6, so 6 jobs start (1,980 units
held). Step k+1: `rest` is 1,981 units, `room` is 6, so 6 jobs of 330 or 331. xsuccess is the first instance, so
it takes the free slots of each step until its 12 jobs run.

### 3.3 Where the flight and the held units come from

`run` owns a registry of the jobs that it launched (as it owns `plan_cache`): for each job id, the labels of the
job and its units. `step` adds a job when `do_launch` returns its id, before the manifest write. So the run
knows the units of every job that it launched, also when the manifest write got a 429.

`_gather` builds two things from the listing, the registry and the store:

1. **The jobs of the run that the listing does not show.** A registry job whose id is not in the listing is added
   to the job records of the step as a scheduling job with its labels. So it counts as in flight for the flight
   of its stage, for the `concurrency_cap` budget, for the mount count and for the stall rule of `run`. A lag of
   the listing therefore cannot free the units of a job. The registry keeps, for each job, whether a listing has
   shown it ENDED (a terminal state). A job that a listing showed ended is never added again, whatever a later
   listing shows. Every other registry job that the listing of the step does not show is added: a job that no
   listing has shown yet, and a job that a listing showed in flight and that this listing leaves out. Such a job
   is removed from the registry when `UNLISTED_STEPS` (5) listings in a row do not show it, with one line,
   `[driver] job <id> <label> was launched but is not in the listing after 5 steps; its units are free again`.
   Only a step that read a listing counts; a step that ended in a rate limit before or in the listing does not.
   A job that ended and left the listing before any listing showed it ended is held for those 5 steps too; that
   costs time, not correctness.
   **The limit of this rule:** a job that was removed can still exist. Its units are then free, so a second job
   can run them at the same time, with the same tag, keys file content and name. This needs a listing that does
   not show a launched job for 5 steps; the other choice, to hold the units for ever, stops a stage until an
   operator acts.
2. **`flight_units`**, a new argument of `decide`: for each in-flight job id, its unit list, or `None` when it is
   not known. The source is the registry; for a job that is not in the registry (a job of an earlier run, after a
   `resume`; every job at a `--dry-run`), the store (`_manifest_for`). The answer of the store is kept in the
   registry for the run, the answer "no manifest" too, so each such job costs one store read (two HTTP requests)
   in a run. Without the argument, every unit list is unknown, so `decide` behaves as it does today.

A registry entry of an ended job is not read again; it stays until the run ends (a launch has about 400 jobs).

### 3.4 Every keyed launch has a unique tag

**Today's code has a collision** (found by both reviewers in round 1, reproduced by each at `2a0cb92b`). In one
step, for a stage without a flight, `decide` returns the escalation launches of `memory_units` and the launches of
`to_run`. Each list comes from its own `_keyed_launches` call and is numbered from 0. When both have the same
count, two launches of one step have the same `(index, count)`. `_job_spec` names the keys file by that pair only
(`_partitions/.../<variant>/<index>of<count>.json`), and `launch.job_name` uses the same pair. `step` writes the
keys file of the second launch over the first while the first job is still scheduling. So the escalation job runs
the units of the other job, the memory unit is not run, and `_attribute` still counts an attempt for it (the
manifest of the escalation job names it). The reproduction: a `fit` stage with `a` done, `b` deferred for memory
and `c` to run gives the launches `(escalate, (0, 1), ('b',))` and `(run, (0, 1), ('c',))`, two writes to
`0of1.json` (`["b"]`, then `["c"]`) and two jobs named `rk--default--fit--0of1`. No launch has shown it (launch 7
and launch 8 had no escalation job). The top-up rule would add more such pairs.

**The rule:** every keyed launch carries a `tag`: the first 12 hex characters of the sha256 of the JSON list of
its sorted keys (the bytes that the keys file holds). The tag is in:

- the keys file name: `<index>of<count>-<tag>.json`;
- the HF job name: it ends with `--<index>of<count>-<tag>` (`launch.job_name` applies its length rule to the
  name with the tag);
- a new HF job label `tag`;
- the in-job label (`SILLY_KICKS_JOB_LABEL`, `launch.label_of`): `<job>.<variant>.<stage>.<index>of<count>-<tag>`.
  The unit ledger records (`job_label`) and the `[alive]` lines carry it, so a record or a line names one job;
- the driver lines that name the shard of a job (`[driver] launch …`, the ended-job line): `<index>of<count>-<tag>`.

The HF `shard` label and the `--shard` argument do not change. `driver.Launch` and `launch.JobSpec` get the field
`tag` (default empty). An unkeyed launch (a `download` instance) has no tag, and its name and labels are as today.

**Why the tag is unique among the jobs that can be alive at once.** The keyed jobs of one stage that are in
flight at one time hold unit sets that are not empty and that share no unit: `to_run` and `memory_units` share
no unit, a split gives disjoint groups, and a top-up takes only units that no job of the flight holds. Different
sets give different tags. A reduce trigger (the tag of `[]`) is launched only when the stage has no flight and
nothing to run, so it is alone. Two jobs with the same units, one after the other (a second attempt after the
first job ended), have the same tag, the same keys file content and the same name, as two attempts have today.
The names of different stages, variants and run keys differ before the tag. The argument holds for the jobs that
the driver knows; its one exception is a job that the registry removed after 5 steps (§3.3).

### 3.5 What does not change

- `step`'s budget rule, the instance order, the mount cap, the FATAL rule, the timeout rule (`TIMEOUT_LIMIT`), the
  STUCK reports (their text and the step at which they come), the attempts count (`_attribute` reads the same
  manifests) and the lost-result rule.
- A unit is never in two jobs of one flight. With the tag this holds for the first launches of a stage too. The
  exceptions are the two limits of the registry (§3.3, §11).
- The reduce: each job tries the reduce after its units, and it succeeds only when every unit is done. The job
  that ends last reduces. When no job reduces, `decide` triggers a reduce after the flight has ended, as today.
- An instance with a `download` dataset keeps the one-wave rule. No instance of `cycle1.toml` has one (every
  dataset is `mount`, or `open` with a snapshot).
- No unit code, no trainer and no library file changes. The artifacts of a launch do not change.

What does change for a reader: the name, the keys file name and the in-job label of every keyed job (§3.4).

### 3.6 Cases

| Case | Result |
|---|---|
| The budget cuts the first split | The next step launches jobs for the units that no job of the flight holds. |
| An escalation launch and a run launch in one step (today's collision) | Two keys files, two job names, two labels. |
| A job that the run launched is not in the next listing | It counts as in flight and holds its units. After 5 such steps it is removed with a line. |
| A job of the flight has no known unit list (an earlier run's job without a manifest) | The stage waits for the flight (today's rule). |
| A reduce trigger is in flight | The stage waits. |
| A job of the flight failed a unit and still runs | The unit is in that job's unit list, so it is held. It is a candidate again after the job ends, with its attempt counted. |
| A job of the flight crashes while the others run | Its units are no longer held. The next step can launch them in a top-up; `attempts` bounds them as today. |
| A memory unit is deferred while other jobs of the stage run | Its escalation job starts in the top-up (it counts toward `room`). |
| The flight has `max_shards` jobs | Nothing is launched for the stage. |
| A unit timed out twice | It is never in a top-up. Its STUCK report comes when the stage has no flight, as today. |
| `--dry-run` | It shows the top-up launches (it reads the manifests from the store). |

### 3.7 Expected result

The xsuccess `extract` stage takes about 7 to 9 min in place of 36.7 min (launch 8), so a complete launch takes
about 3 h 13 min in place of 3 h 43 min. One slow job delays only its own units. The other fan-out stages gain
too; they are not on the long pole.

## 4. Part B: the memory limit of the box

`driver._BOX_RAM` is `32 * 1024**3` (34,359,738,368 bytes). The cgroup limit of a `cpu-upgrade` job is
32,000,000,000 bytes (`memory.max`, measured 2026-11-11; `_parallel_launch.py` records "32 GB" from a job of
2026-10-20). An escalation job gets the cap `_BOX_RAM - _HEADROOM` = 32,212,254,720 bytes, which is above the
limit. So the kernel can end the job before the watchdog of the `rss` backend records the unit as a memory
failure. No launch has shown this; it is from a code read and the measured limit.

Change: `_BOX_RAM = 32_000_000_000`, with the measurement in its comment. The cap of an escalation job becomes
29,852,516,352 bytes, 2 GiB under the limit.

## 5. Part C: CPU time in the unit ledger

The unit ledger record (`_stages._run_unit`) has the wall time and the peak RSS. It gets three keys:

- `cpu_user_s` and `cpu_sys_s`: the user and system CPU time of the unit's process between the start and the end
  of the unit (`os.times()`).
- `cpu_scope`: `"process"`. The pool children of an inner-parallel unit are not in these numbers.

With them, "a full core, half the speed" (§1.3) shows in the ledgers of a launch. The record is a sidecar: no
result, no checksum and no artifact holds it. No reader of the record changes.

## 6. Tests (each seen to fail first)

Part A, `tests/scripts/hf_jobs/test_hf_driver.py` (the rule) and `tests/scripts/hf_jobs/test_launch_typed.py`
(the name and the labels):

1. **Today's collision:** one step, no flight, an escalation launch and a run launch with the same
   `(index, count)`: two keys files with different names and contents, two job names, two in-job labels. This
   test fails on the code of `2a0cb92b`.
2. A flight with known unit lists and units that no job holds: `decide` launches jobs for exactly those units.
3. The launch-8 example: 3,961 units, 6 of 12 jobs in flight: 6 launches, 1,981 units, none of a held unit.
4. `room`: 11 jobs in flight give at most one launch (it holds every unit that is left); 12 give none.
5. `room` with an escalation: `room` 2, one memory unit and other units: one escalation launch and one run launch.
6. A job of the flight with an unknown unit list: no launch and no report. `decide` without `flight_units`: no
   launch (`test_no_launch_while_stage_in_flight`, unchanged).
7. A reduce trigger in flight: no launch. No unit left and a job in flight: no launch and no reduce.
8. An instance with a `download` dataset and a flight: no launch; its launches have no tag.
9. A memory unit that no job holds, with a flight: one escalation launch.
10. A unit that timed out twice, with a flight: it is in no launch, and `decide` makes no report; without a
    flight the STUCK report comes (today's test, unchanged).
11. A failed unit in the unit list of a running job: not launched. After that job ended COMPLETED: launched, and
    its coverage is 1.
12. A job of the flight that ended in a crash while two others run: its units are in the launches of the step.
13. The tag: equal keys give equal tags; different keys give different tags; a reduce trigger has the tag of
    `[]`; `job_name`, `labels_for` (`tag`) and `label_of` carry it; a name over the length limit still ends with
    a digest and is unique.
14. `step` and the registry: a launched job is in the registry when the manifest write fails; `_gather` reads the
    store once for an in-flight job that is not in the registry, also when the store has no manifest.
15. A launched job that the listing does not show: it counts as in flight (the budget falls by one, its units
    are held, `run` does not report a stall). After 5 steps with a listing it is removed, the line is printed,
    and its units are launched; a step without a listing (a rate limit) does not count. A job that a listing
    showed in flight and that the next listing leaves out is held in the same way. A job that a listing showed
    ended is not added when a later listing leaves it out.
16. Two steps through `step` with a budget of 6: 12 jobs in flight after the second step, each unit in one job.
17. `--dry-run` with a flight prints the top-up launches.
18. The driver lines: `[driver] launch` and the ended-job line name the shard with the tag.

Part B: the cap of an escalation job is below 32,000,000,000 bytes by `_HEADROOM`.

Part C (the test file that holds the unit ledger tests; the plan names it): a unit ledger has `cpu_user_s`,
`cpu_sys_s` (numbers, not negative) and `cpu_scope == "process"`; a unit that burns CPU has `cpu_user_s` above
zero.

Mutation checks after the build: the top-up runs a held unit; the tag is dropped from the keys file name; `room`
is `max_shards`; the unknown case launches; an unlisted job holds nothing.

## 7. Docs that travel with the commit

- `docs/howto/hf-jobs-training.md`: the launch rule, the tag (keys file, job name, labels), the unlisted-job rule,
  the unit ledger keys.
- ADR-128: an amendment of 2026-11-11 (the launch rule replaces the one-wave rule; the tag and the collision it
  closes; the box limit; the ledger keys), with the investigation result of §1 and the launch-9 record.
- `CHANGELOG.md`, `silly_kicks/_version.py`, `uv.lock` and the `TODO.md` "Current" line: §14.4 names the commit
  that holds each text.
- `TODO.md`: the row "Find the cause of four launch-8 timings" is removed.
- The fit-units spec §17.1: one line that points to §1 of this spec.

## 8. Launch 9: the proof

**Replaced by §14.3 (revision 3).** The text below is the plan before the owner's rulings on the refit; its
acceptance items 2, 3 and 4 still apply, to the parity run.

Launch 9 runs `cycle1.toml` at the cycle commit. It is a parity launch: no model changes. Every gate needs the
owner's go at that time: preflight (seven checks), the launch, and each `prune --delete`.

Acceptance:

1. `retrieve` and `verify`: 21 of 21. `compare --staging <launch 9> --baseline D:\Development\_staging\cycle1-a1a3b52`
   exits 0: 20 of 20 trainer artifacts identical, the census verdict ok. Nothing is adopted or published.
2. The xsuccess `extract` stage takes less than 15 min from its first job to its reduce (launch 8: 36.7 min).
3. From the job logs and the manifests of the launch: every unit that a job started (`[unit] <stage>/<key> start`)
   is in the manifest of that job, and no unit was started by two jobs whose run times overlap.
4. The record states: the wall time of the launch; the step lines (the Hub request counts of §17.4 of the
   fit-units spec run in a launch for the first time); the `fit` unit times with their CPU time, against §1.3.

A miss of 2 or 3 is a defect, reported to the owner before anything else. The record goes into §9 of this spec
and into the ADR amendment, in a second commit when the launch has cloned the first (the rule of the fit-units
cycle).

Storage: the run keys of `a1a3b52` (7.28 GB) and `abcfda8` (7.47 GB) are in the scratch bucket. Their job logs
and the `fit` ledgers are saved locally. Preflight check 7 decides whether launch 9 fits; a `prune --delete` of
either commit needs the owner's yes.

## 9. Order and definition of done

**Replaced by §14.4 (revision 3).**

1. This spec: reviews by A and B, then the owner's approval.
2. The plan: reviews by A and B, then the owner's approval.
3. The build, tests first, in this session (no subagents). Lint, format, targeted pyright and the changed test
   files on this PC (nothing heavy here); the full suite is CI's.
4. The implementation review by A and B, `/final-review`, the version at commit time. Then the commit gate (one
   commit; the owner's yes for commit and push).
5. CI green. Then launch 9 on the owner's go, and its record (§8).
6. The PR, the merge, the tag (after green CI on `main`; it publishes the patch release to PyPI) and each
   `prune`: each on the owner's yes.

Done: the acceptance of §8 holds, CI is green on `main`, the patch release is on PyPI, and the `TODO.md` row is
gone.

## 10. Decisions for the owner to check

1. The top-up needs the unit lists of the running jobs, and a stage whose flight has an unknown unit list keeps
   the one-wave rule (§3.2). The other choice was to guess the held units from the shard labels; a guess can run
   a unit twice.
2. **Owner ruling (2026-11-11, "1: yes"):** every keyed launch has the tag, so the collision of today's code is
   closed in this cycle (§3.4). The name, the keys file name and the in-job label of every keyed job change.
3. `room = max_shards - len(flight)`: the number of jobs of one stage at once stays at most `max_shards` in a
   top-up. The first launches of a stage can still be more than `max_shards` (the escalation launches and the run
   launches are each split with `max_shards`), as today.
4. `download` instances keep the one-wave rule (no campaign has one).
5. The 15 min bound of §8 item 2. It is twice the expected 7 to 9 min.
6. The fit-units spec gets one pointer line only; its text of §17.1 stays as the record of that day.
7. A job that the run launched and that the listing does not show counts as in flight for 5 steps (§3.3). The
   other choice was to trust the listing; no evidence shows whether it lags.
8. A stage with a flight makes no STUCK report, as today (§3.2, case 3).
9. **Owner (2026-11-11):** "we will do a patch version to signal the change in general". That was said before
   the refit was part of this cycle; §14.6 item 13 puts the number to him again.

## 11. Not verified

- Whether the HF listing lags behind a launch. The rule of §3.3 does not depend on a short lag. Two limits need
  a lag: (a) a job that the registry removed after 5 steps can still exist, and its units can then run in a
  second job (§3.3); (b) the registry lives in one `run`, so a `resume` that starts inside the lag begins with an
  empty registry and does not know a job that the listing does not show yet.
- Whether HF ever drops a job from the listing. A job that a listing showed ended stays ended; any other job that
  a listing leaves out is held for 5 listings (§3.3). `_attribute` needs an ended job to stay in the listing to
  count its attempt, as today.
- Whether HF accepts two running jobs with one name. Today's code can ask for it (§3.4); the tag ends that.
- The 7 to 9 min estimate of §3.7 and the 3 h 13 min that follows from it: arithmetic on the launch-8 numbers.
- That our own 12 `fit` jobs caused the half speed of §1.3.
- That a job of an escalation ever reached the cap of Part B.
- That no two jobs with one name ran at one time in launch 7: two name pairs of launch 7 have a first job with no
  end time in the record. Launch 8 had none (0 pairs).

## 12. Review round 1 answers

Reports: `D:\Development\_reviews\2026-11-11-hfu-launch-topup-spec-{A,B}.md`. Both: REQUEST CHANGES; the approach
is sound; no disagreement between A and B.

| Finding | Answer |
|---|---|
| A TOPUP-SPEC-01, B HFU2-SPEC-01 (blocking): two launches of one step can share `(index, count)`, the keys file and the job name in today's code; revision 1 said the opposite and kept that path | §3.4 states the fact and the reproduction. Owner ruling: the tag is on every keyed launch (§2, §10 item 2). Test 1 fails on today's code. Acceptance 3 reads the job logs, so it sees a job that ran other units than its manifest lists. |
| A TOPUP-SPEC-02, B HFU2-SPEC-02: a launched job that the listing does not show holds nothing | §3.3: the run's registry; such a job counts as in flight and holds its units, for at most 5 steps. Tests 14 and 15. §11 states that the lag is not verified. |
| A TOPUP-SPEC-03: the STUCK reports during a flight | §3.2 cases 2 and 3: no report while the stage has a flight, as today. Test 10. |
| A TOPUP-SPEC-04, B HFU2-SPEC-04: two running jobs with one in-job label | §3.4: the tag is in the in-job label, a new HF label and the driver lines. Tests 13 and 18. |
| A TOPUP-SPEC-05: no test for three behaviours | Tests 10, 11 and 12. |
| A TOPUP-SPEC-06: is an absent manifest read at each step | §3.3: once in a run; the answer is kept. |
| A TOPUP-SPEC-07: a top-up job can hold more than `units_per_job` | §3.2 says so, with the `fit` example. |
| A TOPUP-SPEC-08: "no version" for the owner to check | Owner ruling: a patch release (§10 item 9). |
| B HFU2-SPEC-03: the `max_shards` argument of each split | §3.2 case 3 names both calls and their fifth argument. |
| B: 3,960 units in the example; the 3 h 13 min not listed as unverified | §3.2 uses 3,961; §11 lists the estimate. |
| A: the launch-7 name pairs | §11, last item. |

## 13. Review round 2 answers, and what waits for the TF-46 merge

Reports: `D:\Development\_reviews\2026-11-11-hfu-launch-topup-spec-r2-{A,B}.md` (revision 2, sha256
`7ee31c2a…3510`). Both: APPROVE, no blocking finding, no should-fix finding; every round-1 finding RESOLVED.
Both checked the uniqueness argument of §3.4 and the registry rule of §3.3 against the code.

| Finding (all CONSIDER) | Answer |
|---|---|
| A TOPUP-SPEC-09 (a), B HFU2-SPEC-05: a job that the registry removed after 5 steps can still exist, and its units can then run twice with one tag and one name | Stated as the limit of the rule in §3.3, as the exception in §3.4 and §3.5, and in §11. The other choice (to hold the units for ever) is named there. |
| A TOPUP-SPEC-09 (b): a `resume` inside the lag starts with an empty registry | §11. |
| A TOPUP-SPEC-10: what a step without a listing is | §3.3: only a step that read a listing counts. Test 15. |
| B HFU2-SPEC-06: "not listed yet" against "listed before, not listed now" | §3.3: the rule applies only to a job that no listing of the run has shown. Test 15. §11 names what is not verified. |
| B, a note for the plan: `launch.label_of` reads `labels["shard"]` only | The plan makes it read the `tag` label (§3.4 already states the form of the in-job label). |

**What waits (owner ruling 2026-11-11, "confirmed").** This cycle builds on `main` after the merge of the TF-46
PR (the stoppage gate and the refit preparation). Launch 9 then is two driver runs: `cycle1.toml` without its
xShot and xCross entries (that PR removes them), and `campaigns/tf46_refit.toml` in `refit` mode. The retrieve,
the verify, the adoption of the refit weights and the Hub publish of the `sc_extended` models are this cycle's,
each on the owner's yes. So revision 3 of this spec restates §8 (the acceptance of each of the two runs), §9
(the order and the gates of the adoption and the publish) and the release level (the bundled xShot and xCross
weights change, so the wheel changes by more than its version). Parts A, B and C (§3 to §6) do not depend on it.
Revision 3 goes to A and B again, and then to the owner.

## 14. Revision 3: the base, Part D, launch 9 in two runs, the release

### 14.1 The base and the rulings

`main` is `05f3a692` (silly-kicks 4.141.0, tag `v4.141.0`): the merge of PR #301, the stoppage gate of the TF-46
cycle (ADR-130). The branch of this cycle is at that commit; it holds no commit of its own yet.

Checked by the author at `05f3a692`:

- `scripts/hf_jobs/driver.py`, `scripts/hf_jobs/launch.py` and `scripts/_stages.py` are equal to `2a0cb92b`, so §3
  to §6 describe the code as it is.
- `scripts/hf_jobs/campaigns/tf46_refit.toml`: `mode = "refit"`, `concurrency_cap = 8`, four job blocks (8
  instances: xShot and xCross, tiers `public` and `sc_extended`, variants `default` and `position_only`). The four
  blocks are equal to the xShot and xCross blocks of `cycle1.toml` (compared with `tomllib`).
- `adopt.py`: in `refit` mode `verify` passes an arm on its own `acceptance` block and skips `_integrity`
  (lines 782 to 796); the comparison with the committed model is in the row, not gated.
- The acceptance gates of both trainers: `enough_usable_folds`, `pr_auc_gt_base_rate`,
  `brier_lt_base_rate_brier`, `log_loss_lt_uniform`.
- The two trainers load only the artifact that they just wrote (`XShotOccurrenceModel.load(art)`,
  `XCrossAttemptModel.load(art)`); neither loads the bundled weights.
- The dataset lock pins the re-staged `xshot115`, `xshot17` and `xcross_probe`.

The input from the TF-46 cycle is its handover note, `D:\Temp\tf46-recut\records\handover_partdeux.md`.

Owner rulings of 2026-11-11 after revision 2a:

1. "confirmed": this cycle waits for that merge, and launch 9 carries the xShot/xCross refit. The retrieve, the
   verify, the adoption and the Hub publish of the refit are this cycle's, each on his yes.
2. "Option 1": `cycle1.toml` stays whole. The refit run comes first, then its adoption in a commit, then
   `cycle1.toml` unchanged at that commit as a full parity run.
3. "1 as recommended": `compare` is extended so that one command proves the full parity run (Part D).
A fact, not a ruling: `v4.141.0` is tagged on `05f3a692` (the owner: "Getting tagged now"), so the version of
this cycle is a later one. The owner has not ruled on its level since the refit joined the cycle (§14.6 item 13).

### 14.2 Part D: `compare` reads what shipped from several launches

**The problem.** `compare` finds the run keys of the baseline folder by the campaign's own name
(`adopt._staging_run_key`: `<campaign>-<commit7>-<job>-<tier>`). The run keys of the refit run are named
`tf46_refit-…`, so a `cycle1` launch cannot be compared with them. Against launch 7 the 8 xShot and xCross
artifacts differ by design (new data). After this cycle, what shipped comes from two launches: launch 7 for 12
trainer artifacts, the refit run for 8.

**The rule.**

- `compare --staging NEW --baseline OLD [--baseline OLD2 …]`: `--baseline` can be given more than once.
- The NEW side does not change: the run keys of the campaign's own name, and their retrieved files.
- The baseline of an instance (`<job>`, `<tier>`, `<variant>`) is the LAST given folder that holds a run key
  `<any campaign name>-<commit7>-<job>-<tier>` with a folder of the variant. In that folder the rule of the
  fit-units spec §17.3 holds: the adopted copy (`_final/`) when it has one, the retrieved files otherwise.
- Two run keys of one job and tier in ONE baseline folder are refused. Today that means two commits of one
  campaign; with the lookup under any campaign name it also means two campaigns in one folder.
- An instance that no baseline folder holds is a difference ("present in only one staging folder"), as today.
- The census row does not change (it reads `verify.json` of NEW).
- Each trainer row gets `baseline_run_key`: the name of the run key folder that was read. The line is
  `[compare] <key> IDENTICAL (baseline: adopted copy of <run key>)` (or `retrieved files of <run key>`), so the
  record shows which launch each artifact was compared with.
- `adopt.compare_staging(c, *, staging, baselines)`. With one baseline folder of the same campaign the verdict is
  the one of today.

**Why the campaign name does not matter on the baseline side.** An artifact is named by its job, its tier and its
variant. The refit campaign and `cycle1` train the xShot and xCross arms from the same registry entry, with the
same arguments and the same dataset lock (§14.1), so `tf46_refit-<c7>-xcross_attempt-public/default` and
`cycle1-<c7>-xcross_attempt-public/default` are two runs of one artifact.

**Tests (each seen to fail first), `tests/scripts/hf_jobs/test_adopt.py` and `test_hf_driver.py`:**

- D1. One baseline of the same campaign: the rows of today, each with `baseline_run_key`.
- D2. A baseline whose run keys have another campaign name: the instance is found and compared.
- D3. Two baselines: an instance that both hold is read from the last one; an instance that only the first holds
  is read from the first.
- D4. An instance in no baseline: a difference, with no form and no run key.
- D5. Two run keys of one job and tier in one baseline folder: refused.
- D6. The command line: `--baseline` twice; the lines name the run key; exit 0 and exit 1.

Mutation checks: the first folder wins; the lookup uses the campaign's own name.

### 14.3 Launch 9 in two runs (this replaces §8)

Every step below needs the owner's go at that time: each preflight, each launch, the adoption, each commit, each
publish run and each `prune --delete`.

**Run 9a: the refit** (`scripts/hf_jobs/campaigns/tf46_refit.toml`, at the build commit C1).

1. Preflight (seven checks), the launch, `retrieve`, `verify` in `refit` mode.
2. Acceptance: 8 of 8 arms pass their four gates. The author reports the gates and the comparison with the
   committed models (`certification.comparison`) to the owner. A failed gate stops at the owner; no other path
   is built.
3. `adopt` on the owner's yes. It writes: the bundled weights
   `silly_kicks/tracking/_xshot_weights/{default,position_only}` and
   `silly_kicks/tracking/_xcross_weights/{default,position_only}` (the `public` arms) with their bundled cards;
   `docs/artifacts/observed.json` (the record of all 8 arms); the manifest `training_commit` of the 8 artifacts;
   the four Hub model cards of the `sc_extended` arms under `docs/huggingface/model-cards/`; and the adopted copy
   of each of the 8 arms in `<9a staging>/_final/`. `adopt` stops at the commit gate: commit C2, push, CI green.
4. Nothing is published yet. So from C2 until the publish the repo names `sc_extended` weights that the four Hub
   repos do not hold: the Hub still holds the weights of launch 7.

**Run 9b: the full parity run** (`scripts/hf_jobs/campaigns/cycle1.toml`, unchanged, at C2).

1. Preflight, the launch, `retrieve`, `verify`: 21 of 21. The xShot and xCross instances run in `rematerialize`
   mode, so their `_integrity` check compares their fresh `metrics.json` with what C2 holds (the bundle for a
   `public` arm, `observed.json` for an `sc_extended` arm). That is a check of the metrics within a tolerance.
   The byte proof is item 2.
2. `compare --staging <9b> --baseline D:\Development\_staging\cycle1-a1a3b52 --baseline <9a staging>` exits 0:
   20 of 20 trainer artifacts identical (12 against the adopted copy of launch 7, 8 against the adopted copy of
   run 9a), the census verdict ok. Nothing is adopted from run 9b.
3. The xsuccess `extract` stage takes less than 15 min from its first job to its reduce (launch 8: 36.7 min).
   Run 9b runs alone, as launch 8 did, so the two times compare.
4. For both runs, from the job logs and the manifests: every unit that a job started is in the manifest of that
   job, and no unit was started by two jobs whose run times overlap.
5. The record states: the wall time of each run; the step lines (the Hub request counts run in a launch for the
   first time); the `fit` unit times of run 9b with their CPU time, against §1.3.

A miss of 2, 3 or 4 is a defect and stops at the owner. If an xShot or xCross artifact of run 9b differs from
the adopted copy of run 9a, the refit does not reproduce: that stops at the owner, before any publish and before
the release.

**The Hub publish** comes after run 9b has passed: the four `sc_extended` arms from the staging folder of run 9a,
to `silly-kicks/xshot-occurrence-v1`, `silly-kicks/xshot-occurrence-position-only-v1`,
`silly-kicks/xcross-attempt-v1` and `silly-kicks/xcross-attempt-position-only-v1`: `--sandbox` first, then the
public repos (`--i-have-owner-approval`), each run on the owner's go.

**Storage.** The scratch bucket holds the run keys of `a1a3b52` (7.28 GB) and `abcfda8` (7.47 GB). Run 9a adds
about 4.5 GB (the xShot and xCross run keys of launch 8 hold 4.49 GB) and run 9b about 7.5 GB. Preflight check 7
counts the bucket before each run. The job logs and the `fit` ledgers of both old launches are saved locally, and
the baseline of `compare` is the local staging folder of launch 7, so a `prune --delete` of either old commit
loses no evidence; each needs the owner's yes.

### 14.4 Order, commits and definition of done (this replaces §9)

1. This spec (revision 3): reviews by A and B, then the owner's approval.
2. The plan: reviews by A and B, then the owner's approval.
3. **C1, the build:** Parts A to D, tests first, in this session (no subagents). Lint, format, targeted pyright
   and the changed test files on this PC (nothing heavy here); the full suite is CI's.
4. **Run 9a** and its adoption: **C2**.
5. **Run 9b** at C2.
6. The Hub publish.
7. **C3:** the record and the release state.
8. The PR, the merge, the tag (after green CI on `main`; it publishes the release to PyPI) and each `prune`:
   each on the owner's yes.

Three commits, each for provenance: run 9a clones C1, run 9b clones C2, and C3 holds the record of both. No
commit states a result before the run that gives it:

| Commit | It holds | It does not hold |
|---|---|---|
| C1 | The code and the tests of Parts A to D. The how-to (the launch rule, the tag, the unlisted-job rule, the ledger keys, `compare` with several baselines, and the order "refit run, adoption, full parity run" as a procedure). The ADR-128 amendment: the rules of Parts A to D and the investigation of §1. `CHANGELOG.md`: an `[Unreleased]` section with the scripts-only entries. `TODO.md`: the timing row is removed (§1 answers it). The pointer line in the fit-units spec. | A version. Any word on the refit or on a launch result. |
| C2 | What `adopt` writes for run 9a (§14.3 item 3). `CHANGELOG.md` `[Unreleased]`: the refit of the bundled xShot and xCross weights. ADR-130: the result of the refit (the gates, the comparison). | A version. The record of run 9b. |
| C3 | The record of both runs (a new section of this spec; the ADR-128 amendment). The version in `silly_kicks/_version.py`, the `CHANGELOG.md` heading of that version, `uv.lock`, the `TODO.md` "Current" line. | - |

Each commit has its own gate, in this order: the implementation review by A and B of that commit's content, the
answers to their findings, `/final-review`, the diff shown to the owner, his yes for commit and push, green CI.

If run 9a does not pass (a failed gate, or the owner declines the adoption): C1 stays true as it is written,
nothing is adopted, and the next step is the owner's. With the new dataset lock on `main` the xShot and xCross
instances of `cycle1.toml` cannot reproduce the committed models, so run 9b is not possible as written either.
This spec builds no path for that case.

Done: the acceptance of §14.3 holds for both runs, the four Hub repos hold the refit, CI is green on `main`, the
release is on PyPI, and the `TODO.md` row "Find the cause of four launch-8 timings" is gone.

### 14.5 Docs (this adds to §7)

- `docs/howto/hf-jobs-training.md`: `compare` with several `--baseline` folders; the order "refit run, adoption,
  full parity run" for a refit of some arms.
- ADR-128 amendment of 2026-11-11: Part D and the record of both runs. ADR-130: the result of the refit (the
  gates, the comparison, the adoption).
- `CHANGELOG.md`: the section of the new version: the refit of the bundled xShot and xCross weights (a change of
  the wheel), and the scripts-only items of Parts A to D.

### 14.6 Decisions for the owner to check (this adds to §10)

10. The Hub publish comes after run 9b, so only weights that were reproduced are published.
11. Two baselines in the proof of run 9b: launch 7 for 12 artifacts, run 9a for 8. After this cycle the baseline
    of a later parity launch is these two folders (or the folder of run 9b alone, once it is the shipped proof).
12. The `compare` row gets `baseline_run_key`, and its line names the run key.
13. **Open, the owner's:** the version. He said "a patch version" before the refit joined the cycle; the bundled
    xShot and xCross weights now change. For comparison: 4.140.0 was a minor release for two reasons, a new
    runtime dependency (numba) and the launch-7 re-fit of the bundled weights. The number goes into C3. No bundle holds the package version (launch 8 at 4.140.1 equals launch 7), so the number does not
    enter an artifact, and the runs do not need it.
14. Three commits in this cycle, each with an A and B review, `/final-review` and its own commit gate (§14.4).
15. No path is built for a refit that fails its gates or that does not reproduce (§14.3, §14.4).

### 14.7 Not verified (this adds to §11)

- That the xShot and xCross fits of run 9b reproduce the refit byte for byte. Run 9b is that test.
- That the refit passes its acceptance gates. The TF-46 handover expects fewer training rows and moved metrics.
- That the 12 trainer artifacts outside the refit still equal launch 7 after PR #301, which changed shared files
  (`_provider_visibility.py`, `_loader_pining.py`, `_study_shared.py`, `_toy_corpora.py`). Run 9b is that test.
- That `adopt` works on a refit staging folder: it is code of PR #301 that no launch has run.
- The storage that runs 9a and 9b need; the 4.5 GB and 7.5 GB are the sizes of launch 8.
- The claims of the handover note that this section does not list as checked (the per-match share of `dead` and
  `unknown` frames; that `stoppage_provenance` is equal in two runs).

## 15. Review round 3 answers

Reports: `D:\Development\_reviews\2026-11-11-hfu-launch-topup-spec-r3-{A,B}.md` (revision 3, sha256
`ca3d23f6…dc3a`). Both: APPROVE, no blocking finding. Both checked the facts of §14.1, the baseline lookup of
§14.2 (a wrong pick fails closed, exit 1) and that run 9b proves the reproduction before anything is published.

| Finding | Answer |
|---|---|
| A TOPUP-SPEC-11, B HFU2-SPEC-08 (should fix): the docs and the version of C1 state results of later runs | §14.4: a table names what each commit holds. C1 holds no version and no word on the refit; the refit text is in C2; the record and the version are in C3. The case "run 9a does not pass" is stated. |
| A TOPUP-SPEC-12, B HFU2-SPEC-07 (should fix): no A and B review before C2 and C3 | §14.4: each of the three commits has an A and B review and its own gate; `/final-review` before C1 and before C3. |
| A TOPUP-SPEC-13: the 12 other artifacts after PR #301 | §14.7. |
| A TOPUP-SPEC-14: one listing that leaves out a running job freed its units | §3.3: a job is ended only when a listing showed it ended; any other absent job is held for 5 listings. Test 15, §11. |
| A TOPUP-SPEC-15, B HFU2-SPEC-10: stale words on the version and the base; "a ruling" that the owner did not make | The header, §7, §10 item 9 and §14.1 are corrected. The version is an open decision of the owner (§14.6 item 13). |
| B HFU2-SPEC-09: what `adopt` writes; the repo names weights that the Hub does not hold between C2 and the publish | §14.3, run 9a, items 3 and 4. |
| B: `verify` is a tolerance check, `compare` is the byte proof | §14.3, run 9b, item 1. |
| B: the refusal also fires for two campaigns in one folder | §14.2. |

## 16. Review round 4 (the confirmation of revision 3a)

Reports: `D:\Development\_reviews\2026-11-11-hfu-launch-topup-spec-r4-{A,B}.md` (revision 3a, sha256
`73d30d4e…a12a`). Both: APPROVE; every round-3 finding RESOLVED; no blocking and no should-fix finding. Reviewer B
reversed the 13 edits of revision 3a and got revision 3 exactly.

| Note | Answer (revision 3b) |
|---|---|
| A TOPUP-SPEC-16 (consider): no `/final-review` before C2 | §14.4 and §14.6 item 14: `/final-review` runs before each of the three commits. |
| A: the scope row "A release" named the wrong commit for the version | §2: the number goes into C3. |
| B: the how-to item on the order of the two runs was in no commit | §14.4, the C1 row. |
| B: 4.140.0 was a minor release for two reasons | §14.6 item 13 names both. |
| B: one long line in the header | Wrapped. |

## 17. Revision 4: the refit leaves this cycle (2026-11-12)

### 17.1 What happened, and the rulings

- **Run 9a ran** (`tf46_refit.toml` at C1 `aadd4cf7`, 2026-11-12 05:08:30Z to 06:13:29Z): 85 launches, 80 steps,
  no unlisted-job line, and no driver line for a crashed or failed job; the only failed unit is the gate failure
  below (its three jobs ended COMPLETED). 7 of 8 arms passed their four gates. `xcross_attempt-public` /
  `position_only` failed `brier_lt_base_rate_brier` in 3 attempts (Brier 0.03728 against 0.03690). The driver
  stopped with that instance STUCK (exit 1). Nothing was retrieved or adopted.
- The owner gave the failed arm to the TF-46 session ("1"). That session found the Brier bar biased low: a
  constant forecast at the training-fold rate also failed it. The owner approved new gates: ADR-131, PR #303,
  4.142.0, merged into `main` as `e5573a40`.
- The owner then ruled "Widen before refit": the refit of all xShot/xCross arms follows a corpus-widening cycle.
  So the refit leaves this cycle.
- **Decoupling.** Owner, 2026-11-12: "option 2 is fine, if we can do it cleanly and coordinated with the original
  silly-kicks session (doing the TF-46 work)". The TF-46 session agreed to this order: PR #303 merges first; then
  this PR is rebased, runs run 9b and merges; then the widening code starts.
- **The rebase.** On the owner's yes: C1 is rebased onto `e5573a40` and is now `57a17a42`. The only conflict was
  `CHANGELOG.md` (the `[Unreleased]` section now stands above `[4.142.0]`). The commit message says PR-S207,
  because PR #303 took PR-S206. The five changed test files pass after the rebase, and ruff exits 0.
- **Run 9b scope.** Owner: run 9b launches `cycle1.toml` whole.
- **The first run 9b, stopped; the amend.** Run 9b started at 19:43:11Z on `57a17a42` while this revision was
  uncommitted in the tree. So the driver computed its stage tokens from `dirty-<tree hash>`
  (`scripts/_provenance.code_identity`), and the jobs computed them from the commit. The driver did not see the
  records of the jobs and launched done work again (xsuccess `refs` three times, then STUCK). This was an
  operator error, not a defect of the C1 code. The driver was stopped after 46 launches, and the run was
  cancelled (38 jobs completed, 8 cancelled). Owner, 2026-11-12: "stop the run, given that the previous commit is
  wasted this will an amend". So C1 is amended: it also holds revision 4a of this spec and revision 3a of the
  plan. Run 9b runs again on the amended commit, with a clean tree. The run keys of `57a17a42` are pruned.
- **Storage.** On the owner's "prune as makes sense": `abcfda8` (7.47 GB) and the `tf46_refit` keys of `7561f77`
  (4.06 GB) are deleted. The run keys of `aadd4cf` (run 9a, 3,027 files) are deleted after their 704 per-job
  manifests and ledgers were saved locally.
  The run keys of `a1a3b52` stay: they are the run record of 25 shipped artifacts.

### 17.2 What run 9a still proves

Run 9a is not an adoption source any more. It stays as evidence for the top-up rule: the driver log, the gates of
the 8 arms and the per-job manifests are saved under `D:\Temp\hfu-topup\run9a\`. §14.3 item 4 (no unit started by
two jobs whose run times overlap) is checked on these saved manifests. Result: of 619 units in 85 job manifests,
one unit is in three jobs: the failed `xcross_attempt-public` `fit` unit (`only`, tag `47e912513aa2`). The three
jobs are its retries, created at 05:35:47Z, 05:37:26Z and 05:39:05Z. The driver launched them in three different
steps (`driver.log` lines 3937, 4241 and 4521), and the unit's ledger counts
attempt 3: each attempt read the ledger that the attempt before wrote. So the three runs did not overlap. The
ledger holds only the last attempt (2.55 s); the run times of the first two jobs are not available any more
(the job API returns 404).

### 17.3 Run 9b (this replaces §14.3)

`scripts/hf_jobs/campaigns/cycle1.toml`, whole, at the amended C1 (§17.1), from a clean tree. Every step needs
the owner's go at that time.

1. Precondition: the PR CI of the amended C1 is green; a red CI stops before run 9b (the green CI of `57a17a42`,
   run 43288396560, does not count for the amended commit). `git status --porcelain --untracked-files=no` is
   empty. Then preflight (seven checks), the launch, `retrieve`, `verify`.
2. `compare --staging <9b> --baseline D:\Development\_staging\cycle1-a1a3b52`. Acceptance: the 12 trainer
   artifacts outside xShot and xCross are identical to launch 7, and the census verdict is ok. The 8 xShot/xCross
   artifacts are expected to differ: ADR-130 changed their data and ADR-131 their gates. So `compare` exits
   non-zero, and the acceptance reads its lines, not its exit code. Nothing is adopted from run 9b.
3. The xsuccess `extract` stage takes less than 15 min from its first job to its reduce (launch 8: 36.7 min).
   Run 9b runs alone, as launch 8 did. But on 4.142.0 each xShot/xCross cv arm also runs 5 nested studies, so the
   load on the concurrency cap is higher than at launch 8. The record states this load next to the time.
4. For runs 9a and 9b: every unit that a job started is in the manifest of that job, and no unit was started by
   two jobs whose run times overlap.
5. The record states the wall time of each run, the step lines, and the `fit` unit times of run 9b with their CPU
   time, against §1.3.
6. **The 8 xShot/xCross instances** (4 job/tier pairs, 2 variants each; an instance is job/tier/variant) are
   outside the parity check. Their `verify` result is recorded, not accepted: they run in `rematerialize` mode
   against bundles that the old data and the old gates made.
   `xcross_attempt-public` / `position_only` is expected to fail an ADR-131 gate (the TF-46 session measured a
   Brier lower bound of −0.0000139 on the run 9a data), so the driver is expected to end with that instance STUCK
   (exit 1). This is not a defect. The ADR-131 retirement rule (its spec §6.1) does not apply in run 9b: the owner
   ruled that the refit, and any retirement, wait for the widening cycle. The author reports every xShot/xCross
   gate result to the owner and to the TF-46 session. The same holds for any other xShot/xCross instance that
   stops on an `AcceptanceGatesFailedError`: it is recorded and reported, and the run goes on (§17.6 item 19).

A miss of item 2, 3 or 4, a failure of an instance outside xShot and xCross, or a crash of any instance (any
error other than the gate error above) is a defect and stops at the owner.

### 17.4 Commits and order (this replaces §14.4)

1. This revision: reviews by A and B, then the owner's approval. The plan changes with it (revision 3), in the
   same review.
2. **Run 9b** at C1.
3. **C2: the record.** The record of runs 9a and 9b (a new section of this spec; the ADR-128 amendment). This
   revision and the plan revision are in the amended C1 (§17.1). The version: see §17.6. Gate: the implementation
   review by A and B, the answers, `/final-review`, the diff shown to the owner, his yes for commit and push,
   green CI.
4. The PR ready, the merge, the tag and each `prune`: each on the owner's yes.

Two commits, each for provenance: run 9b clones C1, and C2 holds its record. There is no adoption, no Hub publish
and no C3. ADR-130 does not change in this cycle.

| Commit | It holds | It does not hold |
|---|---|---|
| C1 (amended; it was `57a17a42`, then `d79311d2`) | As §14.4 states for C1, and also Part E (§19), revision 5a of this spec (with the facts of run 9a in §17.1 and §17.2) and revision 4a of the plan. | A version. The record of run 9b. |
| C2 | The record of runs 9a and 9b. Any review answers after C1. The version, if the owner sets one: `silly_kicks/_version.py`, the `CHANGELOG.md` heading, `uv.lock`, the `TODO.md` "Current" line. | Weights. Model cards. `observed.json`. Anything for ADR-130. |

Done: the acceptance of §17.3 holds, CI is green on `main`, the release (if any) is on PyPI, and the `TODO.md` row
"Find the cause of four launch-8 timings" is gone.

### 17.5 Docs (this replaces §14.5)

- `docs/howto/hf-jobs-training.md` keeps the C1 text, including the order "refit run, adoption, full parity run"
  as a general procedure. No refit result enters it.
- ADR-128 amendment of 2026-11-11: the record of runs 9a and 9b (C2).
- `CHANGELOG.md`: the `[Unreleased]` entries stay scripts-only. No refit entry.

### 17.6 Decisions for the owner to check (this replaces §14.6 items 10, 11 and 13)

16. **The version.** The change is scripts only; the wheel holds only `silly_kicks/` (`pyproject.toml`
    `[tool.hatch.build.targets.wheel]`), so the wheel does not change. On 2026-11-11, before the refit joined
    this cycle, the owner said "we will do a patch version to signal the change in general". With the refit out,
    that statement covers the cycle again: 4.142.1. The owner confirms or changes it at C2.
17. Run 9b has one baseline (`cycle1-a1a3b52`) and accepts 12 artifacts, not 20.
18. The order with the TF-46 work (§17.1) is agreed with that session. Its next code (the widening) starts after
    this PR merges.
19. **The owner's choice:** a gate failure of an xShot/xCross instance other than the expected one. This
    revision records and reports it, and run 9b goes on, because those instances are outside the acceptance and
    their gates are TF-46 work. The other path: stop run 9b at such a failure and ask the owner first.
20. Without a version (if the owner sets none at C2): `CHANGELOG.md` keeps the `[Unreleased]` section, and there
    is no tag and no PyPI release.

### 17.7 Not verified (this adds to §14.7)

- That the 12 trainer artifacts outside xShot and xCross still equal launch 7 after PR #301 and PR #303. PR #303
  changed `scripts/_study_shared.py`, which `train_pass_completion.py`, `train_win_probability.py` and
  `train_xsuccess.py` import. Run 9b is that test.
- The run time and the bucket size of the nested studies on HF. The TF-46 session measured them on the DGX only
  (single-threaded: 34 to 44 min for each xShot `sc_extended` study, 9 to 21 min for each xCross `sc_extended`
  study).
- That the 8 xShot/xCross instances of run 9b end as §17.3 item 6 expects. That is a prediction on the run 9a
  data.

## 18. Review round 5 answers (revision 4a)

Reports: `D:\Development\_reviews\2026-11-12-hfu-launch-topup-spec-r5-{A,B}.md` and
`…-plan-r4-{A,B}.md` (spec sha256 `17a3029c…`, plan sha256 `5202b6b5…`). Spec: A and B APPROVE. Plan: B
APPROVE, A REQUEST CHANGES (one should-fix).

| Finding | Answer |
|---|---|
| A TOPUP-PLAN-08 (should fix): no stop for green CI on the rebased C1 before run 9b | §17.3 item 1 and plan Task 11 step 0: the precondition and the stop. Met: CI run 43288396560 on `57a17a42`, 59 of 59. |
| B HFU2-SPEC-11 (should fix), A TOPUP-SPEC-17 (consider): "four xShot/xCross instances"; there are 8 | §17.3 item 6, §17.7 and plan Task 11 step 2 say 8 (an instance is job/tier/variant). |
| B HFU2-PLAN-30 (should fix): the plan's Global Constraints still name the adoption, C3 and the DGX | Plan Global Constraints and Task 9 step 4: two commits, the version in C2 if any, no DGX step. |
| B HFU2-SPEC-12 (consider), A TOPUP-SPEC-17: "2.55 s" is attempt 3 only; "no failed job" does not say what it counts | §17.2 gives the three driver steps and attempt 3 as the evidence and says what is not available; §17.1 says what was counted. |
| A TOPUP-PLAN-09 (consider): no branch for "no version" | Plan Task 12 steps 1, 2 and 4; §17.6 item 20. |
| B HFU2-PLAN-31 (consider): plan and spec differ on a gate failure of another xShot/xCross arm | Aligned: recorded and reported, the run goes on (§17.3 item 6, plan Task 11 step 1). The other path is the owner's choice (§17.6 item 19). |

Round 6 (spec r6, plan r5): A and B APPROVE both; spec 0 findings. A TOPUP-PLAN-10 (consider): plan Task 11 step 1
said "a unit in two jobs", which a retry is by design; it now uses the words of §17.3 item 4. The status of the
three retry jobs of run 9a (COMPLETED), which neither reviewer could read, is saved in
`D:\Temp\hfu-topup\run9a\dup_check.log` (read with `HfApi.inspect_job` without a namespace).

After the owner's approval of revision 4a: the first run 9b and the amend (§17.1). The edits that follow from it
are mechanical: §17.3 item 1 (the CI of the amended commit, a clean tree), the C1 and C2 rows of §17.4, and plan
Task 11 steps 0 and 1. The scope does not change.

## 19. Part E: no launch from a tree that the jobs do not run (revision 5a, 2026-11-12)

**Why.** The first run 9b (§17.1) ran with tracked edits in the tree. The driver plans every stage with the
controller's code identity (`scripts/_provenance.code_identity(git_provenance())`, `scripts/_stages.py:1308-1311`).
With tracked edits that value is `dirty-<tree hash>`, but every job clones the commit. So the driver and the jobs
computed different stage tokens, and the driver did not see any record. Preflight passed, because no check looks
at the tree, and the how-to rule against tracked edits (`docs/howto/hf-jobs-training.md`) did not stop it.

**Owner rulings, 2026-11-12.** "do this" (add the check in this cycle, in C1, with an A and B implementation
review); then "1+2:yes": (1) the rule goes inside check 1; (2) the driver gets the same guard for every step.

### 19.1 E1: preflight check 1 (`ref`)

Check 1 fails when the controller's code identity is not the campaign commit. It reads the same value that the
driver's `--plan` digests:

- clean tree, or only untracked files: the identity is the commit; the check goes on to its remote and CI tests;
- tracked changes: the identity is `dirty-<tree hash>`; the check fails;
- unknown provenance: the identity is `unknown-<nonce>`; the check fails.

The tree test runs after the HEAD test (a clean tree's identity is its HEAD, so a wrong HEAD keeps its own
message) and before `git fetch` and the `gh` call. One compare covers all three cases. The FAIL detail names the cause and at most 5
changed tracked paths (`git status --porcelain --untracked-files=no`), for example: `the tree has tracked changes
(2): docs/a.md, docs/b.md; the driver would plan under dirty-1a2b3c4d5e6f, the jobs run 57a17a4; commit or stash
them`. For unknown provenance and for any other identity the detail names the cause and both identities,
without the "commit or stash" advice. Signature: `check_ref(commit, *, run=subprocess.run, identity=_controller_identity)`, where
`_controller_identity()` returns `code_identity(git_provenance())`. The count of checks stays seven.

**The working directory.** `--plan` runs with `cwd=REPO_ROOT` (`scripts/hf_jobs/campaign.py:258`); preflight runs
git in its own working directory, as check 1 does today. Any directory inside the repo gives the same whole-repo
answer for both `git status` and the tree hash. Preflight is run from the repo root (the how-to command), so the
two values are equal.

### 19.2 E2: the driver refuses a plan with another identity

Every `--plan` result already holds `code_identity` (`scripts/_stages.py:1185`). In `driver.step`, after
`_gather` and before `decide`: if any readable plan has a `code_identity` that is not the commit, the step stops
like a FATAL, before any launch. It prints `FATAL code_identity: <instance> planned under <identity>, the jobs run
<commit7>; the tree changed after preflight; restore it, then resume` and returns the FATAL result, so `run`
exits 2. A plan without `code_identity` counts as another identity (fail closed). A plan from the cache (`_cached_reader`) is a reduced instance: the step launches nothing for it, so the
cache needs no change. No new git call: the value comes from the plan. `launch --dry-run` shows the same: its
output has the FATAL `code_identity` and no launch (final review, 2026-11-12).

### 19.3 The window without tracked edits

From the preflight of run 9b to the end of its `verify` and `compare`, the tree has no tracked edit: the driver
plans in every step, and `retrieve` reads the same plan (`scripts/hf_jobs/adopt.py:224-226`). E2 stops the driver
on such an edit; `retrieve` has no guard of its own, so the window rule covers it. Meanwhile the record of runs 9a
and 9b is drafted outside the repo, in `D:\Temp\hfu-topup\record\`, and enters the tree only after `compare`, for
C2.

### 19.4 Tests (each seen to fail first)

In `tests/scripts/hf_jobs/test_preflight.py` (E1), with an injected `identity` and an injected `run`:

- `test_check_ref_passes_when_the_identity_is_the_commit`;
- `test_check_ref_fails_on_a_tracked_dirty_tree_and_names_the_paths` (the detail holds `dirty-`, the commit and
  the paths; the `gh` call is not made);
- `test_check_ref_fails_on_unknown_provenance`;
- `test_check_ref_names_at_most_five_paths`;
- `test_check_ref_on_a_clean_tree_at_another_head_names_the_head`;
- `test_check_ref_fails_on_any_other_identity_at_the_right_head`;
- `test_check_ref_without_identity_uses_the_driver_identity` (no `identity=`; `git_provenance` and
  `code_identity` monkeypatched to a `dirty-` value: the production call shape of `cmd_preflight`);
- `test_check_ref_default_identity_is_the_driver_one`: `_controller_identity` returns `code_identity(
  git_provenance())` (a monkeypatched `git_provenance` and `code_identity`; no second rule).

In `tests/scripts/hf_jobs/test_hf_driver.py` (E2):

- `test_step_stops_before_any_launch_when_a_plan_has_another_identity` (a FATAL `code_identity`, 0 launches, the
  line names the instance and both identities);
- `test_step_launches_when_every_plan_has_the_commit_identity` (a positive control: it passes before E2 too; the
  mutant "the guard refuses every plan" fails it);
- `test_step_stops_when_a_plan_has_no_identity` (fail closed);
- `test_dry_run_reports_a_foreign_identity_as_fatal_and_no_launch`;
- `test_step_stops_when_a_later_plan_has_another_identity` and `test_dry_run_reports_a_later_foreign_identity`
  (two plans: the first in sorted order at the commit, as a cached reduced instance is, and a later one dirty).

The shared fixture `_ctx` of `test_hf_driver.py` builds its plans with `"c" * 40` as `code_identity`, and the
step tests use `_COMMIT`. The fixture changes to `_COMMIT` for both values, so the existing step tests keep
their meaning.

The semantics of `code_identity` (untracked files do not count) are already tested in
`tests/scripts/test_env_provenance.py`.

### 19.5 Docs, commit and order

- Docs (in C1): `docs/howto/hf-jobs-training.md` (the check 1 line, the driver's FATAL `code_identity`, the window
  of §19.3); ADR-128 amendment of 2026-11-11, item 56; `CHANGELOG.md` `[Unreleased]`, one line under "Changed
  (scripts only)".
- **Commit.** E1, E2, their tests and docs, this revision 5a of the spec and revision 4a of the plan all go into
  C1 by an amend of `d79311d2` (no launch has cloned it). The amend and the force-push each need the owner's yes.
  §17.4 is read with this change: the C1 row also holds Part E, spec revision 5a and plan revision 4a; the C2 row
  is unchanged.
- **Order.** This revision: A and B review, the owner's approval. Then TDD (plan Task 13); the A and B
  implementation review of the C1 delta; `/final-review`; the amend and the push on the owner's yes; green CI;
  then run 9b (§17.3, plan Task 11), from a clean tree.

## 20. Review round 7 answers (revision 5a)

Reports: `D:\Development\_reviews\2026-11-12-hfu-launch-topup-spec-r7-{A,B}.md` and `…-plan-r6-{A,B}.md`
(revision 5, sha256 `9fce7979…`; plan `bd2dfde7…`). Spec: B APPROVE, A REQUEST CHANGES. Plan: B APPROVE, A
REQUEST CHANGES. Both reviewers recommended choice 1. The owner ruled "1+2:yes".

| Finding | Answer |
|---|---|
| A TOPUP-SPEC-18 (should fix): the driver-side gap is left on a default | Owner ruling 2: §19.2 adds E2, the driver guard in every step. |
| A TOPUP-PLAN-11 (should fix): no window without tracked edits during the run | §19.3 and plan Task 11: the window from preflight to the end of `compare`; the record is drafted in `D:\Temp\hfu-topup\record\`. |
| B HFU2-SPEC-13 (should fix): the commit of revision 5 and of plan revision 4 is not named; the §17.4 C1 row is false | §19.5: everything goes into C1 by an amend of `d79311d2`; the §17.4 C1 row names Part E, spec revision 5a and plan revision 4a. |
| B HFU2-SPEC-14 (consider): "cannot disagree" assumes the working directory | §19.1, "The working directory". |
| B HFU2-SPEC-15 (consider): 7 places say "seven" under choice 2 | Not needed: the owner chose 1, so no count changes. |
| A TOPUP-PLAN-12 (consider): the doc list does not match; no named test per mutant; the `tree_state` mutant depends on the real tree | Plan Task 13: the files are named; each mutant has a named test with injected identities only. |
| Round 8 (A and B APPROVE): B HFU2-SPEC-16 (consider), A note: one E2 test passes before E2 | §19.4 calls it a positive control; plan Task 13 step 5 adds its mutant. |

## 21. Implementation review round 3 answers (Part E)

Reports: `D:\Development\_reviews\2026-11-12-hfu-launch-topup-impl-r3-{A,B}.md` (delta sha256 `ab2870e4…`). A:
REQUEST CHANGES; B: APPROVE. `/final-review` ran before this round (owner, 2026-11-12).

| Finding | Answer |
|---|---|
| A TOPUP-IMPL-02 (A), B HFU2-IMPL-15 (should fix): no test pins the default identity of check 1 | `test_check_ref_without_identity_uses_the_driver_identity`; it catches both reviewers' mutants. |
| A TOPUP-IMPL-02 (C) (should fix): every E2 test has one plan | `test_step_stops_when_a_later_plan_has_another_identity` and `test_dry_run_reports_a_later_foreign_identity`; they catch the mutant "the guard stops at the first plan at the commit". |
| B HFU2-IMPL-16 (consider): plan Task 13 step 2 says "the tree test first" | The plan says "after the HEAD test". |

## 22. Part F: one decision per shared generation, and a retrieve that waits out a rate limit (revision 6, 2026-11-12)

**Why.** Run 9b (§17.3) missed acceptance item 4, and its `retrieve` crashed twice.

- **The miss.** One `validate` unit of `ghost_gk-owner` (generation `ghost_gk/validate/85b04edb4327ce3e`, unit
  `only`) ran in 5 jobs at the same time, one job per variant (`default`, `position_only`, `sweeper`,
  `sweeper_position_only`, `full`), from 20:59:15Z to 21:01:06Z. All 5 ended COMPLETED.
- **The cause.** The token of `validate` does not depend on the variant, so the 5 variant instances plan the same
  generation: the same directory under the same run key (`_stages.RunContext.generation` is
  `<pipeline>/<stage>/<token>`). `driver.decide` walks each instance on its own. `_stage_flight` sees only the jobs
  of its own `(run_key, variant, stage)`, and `coverage`, `timeouts` and `reduce_errors` are keyed by the instance
  id. So each of the 5 instances saw the unit pending and launched a job for it in the same step.
- **The scope of the fault.** `ghost_gk-owner` has two groups of shared generations. The set of peers changes from
  stage to stage:
  - `validate`: all 5 variants share one generation;
  - `extract`, `cache`, `cv`, `fit` and `importance`: `default` and `full` share each generation (evidence E11:
    `scripts/train_ghost_gk.py:250-252`, `scripts/hf_jobs/registry.py:183`; `variant` is a param of `assemble`
    only). `position_only`, `sweeper` and `sweeper_position_only` each have their own;
  - `assemble`: no stage is shared.

  In run 9b, only `validate` ran in jobs of more than one variant (scratch script
  `D:\Temp\hfu-topup\run9b2\accept\shared.py` over the 325 job manifests). The second group did not overlap
  because of the instance order and the mount cap, not because of the token. `full` is the last of the 5 instances,
  and in each pass the instances before it used up the headroom of the `tc3` mount cap (6). By the time `full` had
  headroom, the stage was reduced. In `driver.log`, `full` launched only `validate` and `assemble`. A campaign with
  another order or a higher cap would also run the units of the second group twice. The launches of `a1a3b52`,
  `abcfda8` and `c40ec7b` show the same 5 concurrent `validate` jobs: the fault is older than this cycle.
- **The effect.** No output changed: `compare` gives the 12 trainer artifacts `IDENTICAL`. The cost in run 9b is 4
  extra jobs of about 2 min each. Without the cap, `full` would repeat the `extract`, `cache`, `cv`, `fit` and
  `importance` jobs of `default`: more than half of the ghost_gk-owner work of that variant pair. The risk is jobs
  that write the same files at the same time. That is safe today only because the units are deterministic.
- **The retrieve crash.** The first two `retrieve` attempts of run 9b crashed on HTTP 429 at
  `api/buckets/silly-kicks/hf-refit-scratch/xet-read-token`. The account limit is 2500 requests in 5 min.
  `huggingface_hub` retries a 429 on `paths-info`, but not the one raised by `download_bucket_files` through the
  xet token (`RuntimeError: Previous task error: Rate limited: ... (429 Too Many Requests)`). One counters read in
  `adopt._counters` raised it, and the whole `retrieve` stopped. The third attempt used the unchanged `retrieve`
  with `workers=1, read_workers=2` (a scratch wrapper) and finished in 12.5 min without a 429.

**Owner rulings, 2026-11-12.** "1+2: approved": fix both in this cycle, in C2, with tests first and an A and B
implementation review.

### 22.1 F1: a shared generation is decided once in a pass

The **generation key** of a stage of an instance is `(inst.run_key, stage["generation"])`. Two instances whose
stages have the same generation key are **peers** for that stage. A plan stage without a `generation` key has no
peers. (Every real `plan()` record has the key; the test-built records that lack it keep their meaning.)

In `decide`, for each instance in instance order, at its first un-reduced ready stage:

1. If an earlier instance of this pass reached a stage with the same generation key, this instance launches
   nothing for the stage, makes no report for it, and stops its walk (`break`), as a stage that waits does. The
   first instance that reaches a generation in the pass decides it for all peers. When that instance has no
   readable plan, the next peer is the first one.
2. Else the instance decides the stage as today, with four changes. Each one reads the peers of the stage:
   - the **flight** is the in-flight jobs of every peer (the labels `(run_key, variant, stage)` of any peer);
   - the **coverage** of a unit is the sum of `coverage[(p, stage, unit)]` over the peers `p`;
   - the **timeouts** of a unit are the jobs of `timeouts[(p, stage, unit)]` over the peers, in instance order;
   - the **reduce errors** of the stage are the sum of `reduce_errors[(p, stage)]` over the peers.

3. **A peer without a plan (fail closed).** An instance of the same run key whose plan is not readable in this
   step has an unknown generation, so it can be a peer. Its in-flight jobs of a stage with the same name count in
   the flight. Their units are not known (`_flight_units` reads only indexed instances), so `_top_up` returns no
   launch and the stage waits until the plan is readable again or those jobs end. Its coverage, timeouts and
   reduce errors are not added, because they can belong to another generation. The cost of that is at most more
   attempts of a unit, never two jobs that run one unit at the same time.

A stage without peers, and with no in-flight job of an instance without a plan, gives the same values as today.
So every such decision is unchanged.
`_attribute`, `_fatal_block`, the mount cap and `step` do not change: a launch of the first peer is an ordinary
launch of that instance, with its labels and its manifest. `_dry_run` calls `_decide`, so it shows the same.

The attempts of a shared unit are counted once over all peers. Before F1, each peer had its own `attempts`, so a
shared unit could run up to `attempts` times the number of peers.

### 22.2 F2: `retrieve` waits out a rate limit

`retrieve` (`scripts/hf_jobs/adopt.py`) retries a rate-limited call after each wait of
`_READ_BACKOFF_S = (30.0, 60.0, 120.0, 240.0)` seconds (450 s in total: more than the 5-min window of the
account limit). When the waits are spent, it raises the last error. Each wait prints one line:
`[retrieve] rate-limited (<what>); retry <i>/4 in <w> s`, where `<what>` is `read <rel>`, `names <rel>` or
`plan <instance>`.

- **The calls:** `read_bytes` and `names` of `_BoundedReads`, and the `--plan` call of `bounded_plan`.
- **The rule for a rate limit:** the rule that `step` already uses for a `--plan` failure
  (`driver._plan_rate_limited`): the text holds "too many requests" or "rate limit", and the error is not a
  `PlanTimeoutError`. A bare "429" is not a rate limit, because a path or a match id can hold those digits (the
  `statsbomb__3869429` case of `test_a_failing_plan`). The xet error and the `HfHubHTTPError` of `paths-info` both
  hold "too many requests". The rule moves to `scripts/hf_jobs/campaign.py` as the public
  `rate_limited(exc) -> bool`, next to `read_plan` and `PlanTimeoutError`. `driver._plan_rate_limited` returns
  `campaign.rate_limited(exc)` (its test in `test_hub_client.py` stays), and `adopt` calls `campaign.rate_limited`,
  which it already imports from. So there is one rule and no import between `adopt` and `driver`.
- **The read bound:** a call that waits holds no slot of the `read_workers` semaphore, so the other reads go on.
- **Any other error** raises at once, as today.
- `retrieve` gets a keyword `sleep: Callable[[float], object] = time.sleep` (a test seam), passed to
  `_BoundedReads` and `bounded_plan`. The defaults `workers=4, read_workers=8` do not change.

The driver does not change: `step` already backs off on a 429 (`_is_contention`, `_plan_rate_limited`).

### 22.2a F3: `retrieve` reads the counters in batches (revision 6b; owner ruling "3: fold it in")

`adopt._counters` reads one `<key>.counters.json` file for each present unit of the `extract` stage, with one
`read_bytes` call each. Each call is one `download_bucket_files` call for one file. In run 9b (`retrieve.log`): 13
instances read 1,436 counters files in all (ghost_gk 5 × 179; xshot 17, 17, 115, 115; xcross 17, 115, 115;
receiver_model 30), out of 17,638 `extract` units; 6 instances have `extract` units but no counters files
(xsuccess, win_probability, pass_completion, match_outcome_dependence: 0 of 3,961 each; ghost_outfield 2 × 0 of
179). F3 reads the same files in batches: one `download_bucket_files` call for up to `_READ_BATCH = 200` files. For
run 9b that is 13 calls instead of 1,436. (Revision 6b gave 21,734, the units of every stage of the run; that
number was wrong. Reviewer A, TOPUP-SPEC-22.)

- `BucketStore.read_many(rels: Sequence[str]) -> dict[str, bytes | None]`: one `download_bucket_files` call for
  the rels that are not memoized, each to its own file in one temporary directory, with
  `raise_on_missing_files=False`. A rel whose file is absent after the call is `None`. Every result goes into the
  memo of `read_bytes`, so a later `read_bytes` of the same rel makes no call. An empty list makes no call.
- `adopt._read_many(store, rels) -> dict[str, bytes | None]`: the one place that chooses the read. It calls
  `store.read_many(rels)` when the store has `read_many`; otherwise it calls `store.read_bytes` once for each rel.
  The test fakes (`_CountingStore`, `_SlowStore`) and a `LocalStore` have no `read_many`, and `_counters` must work
  with any store that has `read_bytes` and `names`.
- `_BoundedReads.read_many(rels)`: one slot of the read bound for the whole batch, and the retry of §22.2 for the
  whole batch (`<what>` is `read_many <n> files`), around `_read_many(<wrapped store>, rels)`.
- `_counters`: the present keys, in sorted order, in batches of `_READ_BATCH`, over its `workers` threads, one
  `_read_many(store, batch)` call for each batch. The merge stays in sorted key order, so the totals do not change.
  The progress line counts the files read, as today. The existing `_counters` tests
  (`test_parallel_counters_equal_serial`, `test_counters_read_count`) do not change.
- 200 is the batch size of the read-only scratch script of this cycle (`accept.py`). It downloaded the 21,734
  ledgers and 325 manifests of run 9b (22,059 files) in 200-file calls, and its log holds no rate-limit line. Its
  first run was stopped by hand to free the quota for `retrieve`, so the files came in more than one run; this is
  evidence for the batch size, not a measured request rate.
- No other read changes: the `_inputs.json` reads, the output FileSet and the extract output of `retrieve` stay one
  call each (a few files for each instance).

### 22.3 Tests (each seen to fail first)

In `tests/scripts/hf_jobs/test_hf_driver.py` (F1). The plans are real `plan()` records (`_realplan`). Two
instances of one run key are peers at a stage only when that stage has the same `generation` in both plans. A
param that the stage's token does not read does not matter. Two instances that read one plan are peers at every
stage.

- `test_a_shared_generation_is_decided_once_in_a_pass`: two peers, one pending unit: 1 launch, for the first
  instance;
- `test_a_peer_does_not_launch_units_that_another_peer_runs`: the second peer has a job in flight that holds
  every unit; the first peer makes no launch;
- `test_a_shared_generation_tops_up_only_the_units_no_peer_holds`: two pending units, a job of the second peer
  in flight with one of them: 1 launch with the other unit;
- `test_the_coverage_of_a_shared_unit_counts_every_peer`: coverage 2 for one peer and 1 for the other, 3
  attempts: no launch of that unit;
- `test_the_timeouts_of_a_shared_unit_count_every_peer`: one timed-out job for each peer: the unit is STUCK,
  with one report;
- `test_the_reduce_errors_of_a_shared_stage_count_every_peer`: the reduce triggers of the two peers together
  reach the limit: a report and no reduce trigger;
- `test_the_next_peer_decides_when_the_first_has_no_plan`: the first peer's plan is `None`: 1 launch, for the
  second (it passes before F1 too; the mutant "a `None` plan owns the generation" fails it);
- `test_instances_with_their_own_generations_each_launch` (a positive control: it passes before F1 too; the
  mutant "every instance of a run key is a peer" fails it);
- `test_the_peers_of_a_stage_follow_that_stage_s_generation`: a two-stage pipeline whose first stage's token
  does not read a param and whose second stage's token reads it, the way `validate` and `extract` differ in
  ghost_gk. Two instances with different values of that param: at the first stage, 1 launch; after its reduce, at
  the second stage, 2 launches. The mutant "the peers are grouped once for all stages" fails it;
- `test_a_peer_without_a_plan_still_holds_its_units_in_flight`: the first peer's plan is `None` and it has a
  job in flight on the stage; the second peer makes no launch for that stage (§22.1 item 3);
- `test_dry_run_shows_one_launch_for_a_shared_generation`.

In `tests/scripts/hf_jobs/test_adopt.py` (F2), with an injected `sleep` that records its waits:

- `test_retrieve_retries_a_rate_limited_read` (the xet error text, twice, then the bytes: waits `[30.0, 60.0]`,
  the artifact and the record are complete);
- `test_retrieve_retries_a_rate_limited_listing`;
- `test_retrieve_retries_a_rate_limited_plan`;
- `test_retrieve_raises_when_the_rate_limit_outlasts_the_waits` (waits `[30.0, 60.0, 120.0, 240.0]`, then the
  error);
- `test_retrieve_does_not_retry_another_error` (an error that holds "3869429" and no rate-limit words: raised at
  once, no wait);
- `test_a_waiting_read_frees_its_slot` (`read_workers=1`: while one read waits, the fake `sleep` takes the
  semaphore without a block).

For F3, in `tests/scripts/hf_jobs/test_adopt.py` and `tests/scripts/hf_jobs/test_bucket_store.py` (a fake `api` that
records each `download_bucket_files` call and writes the files it is given):

- `test_counters_reads_in_batches_of_200` (450 present keys: 3 `read_many` calls of 200, 200 and 50 rels in sorted
  key order; the totals equal those of one `read_bytes` for each file);
- `test_bucket_store_read_many_makes_one_call_for_a_batch` (one call with every rel; an absent file is `None`);
- `test_bucket_store_read_many_memoizes` (a second `read_many` or `read_bytes` of the same rels makes no call;
  an empty list makes no call);
- `test_retrieve_retries_a_rate_limited_batch` (the xet error text twice: waits `[30.0, 60.0]`, then the totals);
- `test_read_many_without_a_batch_read_uses_read_bytes` (`_read_many` over a store without `read_many`: one
  `read_bytes` for each rel, the same dict).

Thread order is not fixed: `_counters` runs its batches on several threads, so the batch test asserts the set of
batches, not their order.

The existing tests that give one plan to several variants of one run key (`test_a_failing_plan`,
`test_step_prints_one_timed_progress_line`, `test_step_line_instances_done` and any others) now describe peers.
The implementer reads each one and records in plan Task 14 whether its meaning changed. A test whose assertion
changes is a finding for the owner, not a silent edit.

### 22.4 Evidence on HF plans

A dry run cannot run before the commit: E2 (§19.2) stops a plan of a dirty tree. So, after the C2 push and its
green CI, on the owner's go: `campaign cycle1.toml launch --dry-run` at C2, from a clean tree. It launches nothing
and only reads the bucket. Expected: `ghost_gk-owner` `validate` has 1 launch, not 5. The dry run shows the first
pass only; the second group of §22 (`default` and `full` from `extract` to `importance`) is not reached in it. The result is reported to the
owner and in the PR text. It is not in C2, because C2 is pushed before the dry run. The next real launch (the
widening cycle) is the first field run of F1 and F2.

### 22.4a A read-only field check of F3 before the commit

`retrieve` reads only the run keys of HEAD (the run key holds the commit), so no `retrieve` at C2 can read run 9b. A
scratch script can: `D:\Temp\hfu-topup\f3check\counters_check.py`, with the C2 code in the tree, before the commit.
For each of the 13 instances of run 9b that read counters files (§22.2a), it finds the `extract` generation under
the run key `cycle1-70915ce-<job>-<tier>` (the token of `token_inputs["extract"]` in the instance's record in
`D:\Development\_staging\cycle1-70915ce`, through `_stages._token`). The records hold no unit list, so the units of
the check are the keys of the `<key>.counters.json` files in that generation's listing. It then calls
`adopt._counters` over a `BucketStore` wrapped in `_BoundedReads` with the default bound: one new `BucketStore` for
each instance, as `retrieve`'s `open_store` does. (`default` and `full` of ghost_gk share one `extract`
generation; one store for each run key would serve `full` from the memo and make 12 calls, not 13.) It reads only.

- Expected: for each of the 13 instances, `totals` and `n_read` equal the `counters` of its run 9b record, and no
  rate-limit line. (`n_counters_unrecorded` is not compared: it needs the full unit list.) The 6 instances that
  read no counters files are left out: their check would be empty.
- The check makes 13 `download_bucket_files` calls for 1,436 files. No instance has more than 179 files, so no
  instance splits into two batches: the split is proved by `test_counters_reads_in_batches_of_200` only.
- The record states the number of calls, the time, and any `[retrieve] rate-limited` line.

This check is evidence for F3 on the real bucket. It does not run the CLI and launches nothing.

### 22.5 Docs, commit and order

- **Docs (in C2):** `docs/howto/hf-jobs-training.md` (`retrieve` waits out a rate limit; the driver decides a
  shared generation once; `retrieve` reads the counters in batches); ADR-128 amendment of 2026-11-11, item 57;
  `CHANGELOG.md` `[Unreleased]`, three lines under "Changed (scripts only)".
- **Commit.** F1, F2, F3, their tests and docs, this revision 6d of the spec, revision 5d of the plan, and the record of
  runs 9a and 9b go into C2. §17.4 is read with this change: the C2 row also holds Part F. The version stays a
  scripts-only patch (§17.6 item 16): the wheel does not change.
- **Order.** This revision: A and B review, the owner's approval. Then TDD (plan Task 14); the field check of
  §22.4a; the record (plan Task 12 step 1); `/final-review`; the A and B implementation review of C2; the owner's yes for commit and push; green CI;
  the dry run of §22.4 on the owner's go; then plan Task 12 step 4.

### 22.6 Decisions for the owner to check

Owner rulings, 2026-11-12: item 21 "dry run" (no HF launch for F1); item 22 "fold it in" (F3, §22.2a).

21. **The field proof of F1.** This revision proves F1 with tests on real `plan()` records and the dry run of
    §22.4. It adds no HF launch. The dry run shows the first pass only (the `validate` group). The second group
    (`default` and `full` at `extract` to `importance`) is shown by the tests only, until a real launch reaches it. The other path: a real launch at C2 before the merge, for example `cycle1.toml`
    whole again (about 3 h and 325 jobs, like run 9b).
22. **Fewer reads in `retrieve`.** F2 waits out the limit; it does not lower the number of requests. `_counters`
    reads one file for each counters file (1,436 in run 9b; revision 6 gave 21,734, which was wrong). One
    `download_bucket_files` call can take many files, so a batched read sends far fewer requests (13 calls for
    run 9b). The owner added it: F3 (§22.2a).

### 22.7 Not verified

- Which other campaigns or trainers have shared generations. Run 9b showed one group with an overlap and one
  group that the mount cap hid; a trainer without a mount cap can show more.
- The request count of the xet path for one `read_bytes` and for one batch of `read_many`. The xet token request
  is made outside the `httpx` client, so `hub_client.request_counts` does not see it. The 450 s of waits cover one
  5-min window; a retrieve that hits the limit again after the waits stops with the error, as today.
- The field run of the whole `retrieve` with F2 and F3. `retrieve` reads the run keys of HEAD only, so a retrieve
  at C2 cannot read run 9b again. §22.4a checks the batched counter reads on the real bucket; the next real
  `retrieve` (the widening cycle) is the first whole one, and its time and rate-limit lines go into that cycle's
  record.

**A limit of F1.** When the `--plan` of a peer fails in a step, its coverage, timeouts and reduce errors are not
added in that step (§22.1 item 3). A unit can then get more attempts than `attempts`. Its jobs in flight still
hold the stage, so no unit runs in two jobs at the same time. The failed plan is reported (a STUCK line), and the
next step, with a readable plan, counts the peer again.

## 23. Review round 9 answers (revision 6a)

Reports: `D:\Development\_reviews\2026-11-12-hfu-launch-topup-spec-r9-{A,B}.md` and `…-plan-r8-{A,B}.md`
(revision 6, sha256 `93790d81…`; plan `209be713…`). Spec: A REQUEST CHANGES, B APPROVE. Plan: A REQUEST CHANGES
(it carries the spec findings), B APPROVE.

| Finding | Answer |
|---|---|
| A TOPUP-SPEC-19, B HFU2-SPEC-18 (should fix): the scope of the fault is wrong; `default` and `full` share `extract` to `importance` | §22 "The scope of the fault" names both groups and how the instance order and the `tc3` cap hid the second; "The effect", §22.4, decision 21 and §22.7 follow it; §22.3 defines a peer by the stage's generation; the draft record is corrected for plan Task 12 Step 1. |
| A TOPUP-SPEC-20 (a) (consider): no test has a peer set that changes from stage to stage | §22.3 `test_the_peers_of_a_stage_follow_that_stage_s_generation`; plan Task 14 Step 5 has its mutant. |
| B HFU2-SPEC-17 (should fix), A TOPUP-SPEC-20 (b) (consider): a peer whose `--plan` fails drops out, and the next peer can run its held units at the same time | §22.1 item 3: fail closed. The in-flight jobs of an instance without a plan hold every same-named stage of its run key. §22.3 `test_a_peer_without_a_plan_still_holds_its_units_in_flight`; §22.7 states what stays open (more attempts, never a concurrent run). |
| B HFU2-SPEC-19 (consider): `adopt` would call the private `driver._plan_rate_limited`, and the two modules would import each other | §22.2: the rule moves to the public `campaign.rate_limited`; `driver._plan_rate_limited` and `adopt` call it. |
| Round 10 (spec r10, plan r9: A and B APPROVE; spec 0 findings). A TOPUP-PLAN-13, B HFU2-PLAN-32 (consider): Task 14 Step 6 runs no pyright on `campaign.py`; Files names `test_campaign.py`, which no step changes | Plan Task 14: pyright on `campaign.py`; `test_campaign.py` leaves Files. Editorial notes: §22.7 "A limit of F1" follows the bullets; the E11 cite is `:250-252`. |
| Round 11 (spec r11, plan r10 of revision 6b / 5b; B APPROVE, A REQUEST CHANGES). A TOPUP-SPEC-21 (should fix): `_counters` would call `read_many` on stores without it (`_CountingStore` in the existing `_counters` tests) | §22.2a: `adopt._read_many` is the one helper that chooses `read_many` or `read_bytes`; `_counters` and `_BoundedReads.read_many` call it; the two existing tests do not change. |
| B HFU2-SPEC-20 (consider): a read-only field check of F3 against run 9b is possible before the merge | §22.4a: `counters_check.py` with `adopt._counters` over the real bucket; its totals must equal the run 9b records. |
| B note: the batches run on several threads | §22.3: the batch test asserts the set of batches, not their order. |
| Round 12 (spec r12, plan r11 of revision 6c / 5c; B APPROVE, A REQUEST CHANGES). A TOPUP-SPEC-22 (should fix): the counter-read count is wrong (21,734 is all units of every stage); §22.4a has vacuous rows and no unit source | §22.2a, §22.4a and decision 22: 1,436 files, 13 instances, 17,638 `extract` units, 13 calls; §22.4a names the 13 instances, leaves out the 6 empty ones, says no instance passes one batch, and takes the units from the listing. B note: the run key holds the commit (§22.4a wording). |
| Round 13 (spec r13, plan r12 of revision 6d / 5d: A and B APPROVE; A 0 findings). B HFU2-SPEC-21 (consider), A note: "13 calls" needs one store for each instance | §22.4a and plan Step 4c: one new `BucketStore` for each instance (12 calls with one store for each run key). |

The owner approved revision 6d and plan revision 5d on 2026-11-12 ("approved").

## 24. The record of runs 9a and 9b, and of Part F (C2)

Evidence (outside the repo): `D:\Temp\hfu-topup\run9a\` and `D:\Temp\hfu-topup\run9b2\` (`driver.log`,
`accept\*.log`, `accept\shared_9b.log`, `accept\verify_xsx.txt`), `D:\Temp\hfu-topup\f3check\counters_check.log`,
`D:\Temp\hfu-topup\mut\mutants.log`, `D:\Temp\hfu-topup\record\task14_ledger.md`.

### 24.1 Run 9a

As §17.1 and §17.2 state: `tf46_refit.toml` at `aadd4cf7`, 85 launches in 80 steps, 64.4 min from the first job
to the last; 7 of 8 arms passed. Item 4 (§17.3): 619 units in 85 job manifests, one unit in more than one job (the
failed `xcross_attempt-public` `fit` unit: 3 retries in 3 steps), no overlap.

### 24.2 Run 9b (`cycle1.toml` whole, C1 `70915ce6`, 2026-11-12)

- **The run.** Preflight: 7 checks ok. The driver ran from 20:58:43Z to 23:57:51Z (2 h 59 min) in 199 steps, with
  325 launches and no rate-limited step (the largest step sent 304 Hub requests). It ended with exit 1: one
  STUCK instance, `xcross_attempt-public` / `position_only`, as §17.3 item 6 expects. Jobs: 324 COMPLETED and 1
  ERROR, a `mount_auth_flake` job that never started (its retry ran). From the first job created (20:59:09Z) to
  the last job finished (23:57:00Z): 177.8 min.
- **`retrieve` and `verify` (§17.3 item 1).** 20 of 21 instances were retrieved (`xcross_attempt-public` /
  `position_only` is not reduced). The first two `retrieve` attempts crashed on HTTP 429 at the xet token (§22).
  The third attempt used the unchanged `adopt.retrieve` with `workers=1, read_workers=2` (a scratch wrapper) and
  took 12.5 min. `verify`: the 13 instances outside xShot and xCross pass. The 7 retrieved xShot/xCross instances
  fail the integrity gate (`reproduced` false), as §17.3 item 6 expects in `rematerialize` mode; `xshot_occurrence
  public` `default` and `position_only` also fail CV parity (pr_auc 0.0819 to 0.0889, 0.0754 to 0.0849).
- **`compare` (item 2): passes.** Against `cycle1-a1a3b52`: the 12 trainer artifacts outside xShot and xCross are
  `IDENTICAL`, and the census verdict is ok. The 8 xShot/xCross rows differ or are missing, as expected.
- **Item 3: passes.** The xsuccess `extract` stage took 10.8 min from its first job (created 21:01:54Z) to its
  inline reduce (21:12:39Z), against 36.7 min at launch 8. Its 18 jobs ran 71 to 367 s each and waited at most 5 s
  to start. In that window 47 other jobs ran (xshot `extract` 14, xcross `extract` 14, xshot `prep` 4, xshot
  `study` 4, xcross `probe` 2, `probe_comparison` 2, ghost_gk `extract` 6, xsuccess `hpo` 1), with at most 25 jobs
  running at once.
- **Item 4: a miss.** 325 job manifests, 21,734 units, a ledger for each unit, and every ledger unit in a manifest.
  17 units ran in more than one job: 15 `receiver_model` `extract` units (the ERROR job that never started, then
  its retry), the failed `xcross_attempt-public` `fit` unit (3 retries in turn), and one overlap: the ghost_gk-owner
  `validate` unit `only` in 5 jobs at once (§22). The owner ruled to fix it in this cycle: Part F.
- **Item 5.** Wall time: above, and 64.4 min for run 9a. The `fit` unit times of run 9b (wall, with CPU time about
  equal to it where not stated): ghost_gk 92.6 to 109.2 s (5.3 GB peak RSS); ghost_outfield 548 and 623 s;
  win_probability 1,252.4 s (CPU 1,247.9 s; 6.5 GB); xsuccess 36 units, 296.6 to 630.9 s (6.2 to 7.4 GB), against
  655 to 995 s at launch 8 (§1.3); pass_completion 23.4 s; match_outcome_dependence 8.9 s; receiver_model 15.9 s
  (CPU 4.1 s); xShot/xCross 1.5 to 31.7 s.
- **Item 6: the xShot/xCross gates** (the metrics of the candidate each arm shipped; reported to the owner; the
  report to the TF-46 session waits for the merge, on the owner's word):

| arm | gates | pr_auc | brier | base-rate brier | log loss | folds |
|---|---|---|---|---|---|---|
| xshot public default | pass | 0.08896 | 0.01715 | 0.01763 | 0.07647 | 5 |
| xshot public position_only | pass | 0.08498 | 0.01719 | 0.01763 | 0.07690 | 5 |
| xshot sc_extended default | pass | 0.14227 | 0.01505 | 0.01625 | 0.06529 | 5 |
| xshot sc_extended position_only | pass | 0.13554 | 0.01514 | 0.01625 | 0.06651 | 5 |
| xcross public default | pass | 0.08638 | 0.03681 | 0.03690 | 0.15508 | 5 |
| xcross public position_only | fails `brier_gain_lb95_gt_0` | 0.06172 | 0.03729 | 0.03690 | 0.16015 | 5 |
| xcross sc_extended default | pass | 0.20693 | 0.05982 | 0.06477 | 0.22025 | 5 |
| xcross sc_extended position_only | pass | 0.14778 | 0.06234 | 0.06477 | 0.23468 | 5 |

### 24.3 Part F (plan Task 14)

- **F1.** 11 tests; 9 failed first, the 2 positive controls passed by design; then `test_hf_driver.py` 149 passed.
  Round 5 (§25) added 2 tests against the finished code; each was proved by a mutant that makes it fail.
  The existing tests that now describe peers keep their meaning (plan Task 14 Step 2).
- **F2.** 6 tests, all failed first; then `test_adopt.py`, `test_hub_client.py` and `test_hf_driver.py` passed.
- **F3.** 5 tests, all failed first; then `test_bucket_store.py` 11 and `test_adopt.py` 73 passed; the existing
  `_counters` tests did not change.
- **Mutations.** 19 mutants, each caught by its named test, and 3 more from implementation review round 5 (§25),
  also caught: 22 in all. One ruling: the plan's mutant "the owner of a
  generation is the first instance with any plan entry, a `None` plan included" cannot be built (a `None` plan
  names no generation), so the mutant "a `None` plan stops the walk" stands in for it; the same test catches it.
- **The field check of §22.4a.** 13 instances, 0 mismatches: `totals` and `n_read` equal the run 9b records. 13
  `download_bucket_files` calls for 1,436 files in 51.0 s, with one new `BucketStore` for each instance, and no
  rate-limit line.
- **The dry run of §22.4** comes after the push, on the owner's go; its result goes into the PR text.

### 24.4 The version

Owner, 2026-11-12: "patch release is good" (§17.6 item 16): 4.142.1 in `silly_kicks/_version.py`, the
`CHANGELOG.md` heading and the `TODO.md` "Current" line. `uv.lock` does not change: the version is dynamic
(ADR-079), so the lock holds no version of the package. The `TODO.md` row of this cycle left with C1.

## 25. Implementation review round 5 answers (C2)

Reports: `D:\Development\_reviews\2026-11-12-hfu-launch-topup-impl-r5-{A,B}.md` (delta sha256 `c923bb0b…`).
`/final-review` ran before this round. A: REQUEST CHANGES; B: APPROVE. Both re-ran the GREEN and RED runs and
matched the ledger; both accept the mutation ruling of §24.3.

| Finding | Answer |
|---|---|
| A TOPUP-IMPL-03, B HFU2-IMPL-18 (should fix): the ADR-128 edit deleted the "**Consequences.**" label of the 2026-11-11 amendment | Restored: item 57 ends at its own text; the paragraph is byte-equal to C1. |
| A TOPUP-IMPL-04 (should fix): after F3 the counters reads go through `read_many`, so no test fails when `_BoundedReads.read_bytes` loses its retry; the read test matched the prefix "(read" | `test_retrieve_retries_a_rate_limited_read` rate-limits `extract/g0/_inputs.json` (a `read_bytes` of `retrieve`) and asserts the whole line `(read extract/g0/_inputs.json); retry 1/4 in 30 s`. A's mutant A is caught. |
| B HFU2-IMPL-17 (should fix): no test pins "of the same run key" in §22.1 item 3 | `test_a_planless_instance_of_another_run_key_does_not_hold_the_stage`; the mutant "the planless instances of every run key hold the stage" is caught. |
| B HFU2-IMPL-19 (consider): a later peer that `continue`s could start its next ready stage in the same pass | `test_a_waiting_peer_starts_no_later_stage_in_the_pass`; the mutant `break` to `continue` is caught. |
| Round 6 (`…-impl-r6-{A,B}.md`, delta sha256 `3787397c…`): A and B APPROVE, 0 findings | Each reviewer ran its own mutants of round 5 again; all are caught. No product file changed in round 6. |
