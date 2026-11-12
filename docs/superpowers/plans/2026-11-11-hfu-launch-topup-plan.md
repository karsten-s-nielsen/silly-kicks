# HFU launch top-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the owner forbids subagents) to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a stage gets more HF jobs while its jobs run, every keyed job has a unique tag, `compare` reads what
shipped from several launches, and launch 9 (the xShot/xCross refit, then a full parity run) proves it.

**Architecture:** `driver.decide` stays a pure function; it gets the unit lists of the running jobs as a new
argument. A new module, `scripts/hf_jobs/launched.py`, holds the jobs that one driver run launched and decides
which of them the listing does not show yet. `launch.py` owns the tag (the keys file bytes, the job name, the
labels). `adopt.compare_staging` looks a baseline up by job and tier under any campaign name.

**Tech Stack:** Python 3.12, pytest, `huggingface_hub` 2.2.0 (HF Jobs, buckets), ruff 0.15.7, pyright.

**Spec:** `docs/superpowers/specs/2026-11-11-hfu-launch-topup-design.md`, revision 3b (approved by the owner
2026-11-11, "Spec approved, proceed"). Section numbers below (§) are the spec's.

**Revision 2** (2026-11-11): the answers to plan review round 1 (A, B) and the owner ruling "2: use the DGX"
for the adoption of run 9a (Task 10); revision 2a: the two DGX environments of Task 10 (plan review round 2).

**Revision 3** (2026-11-12): spec revision 4 (§17). The refit leaves this cycle (owner rulings: the failed arm to
the TF-46 session, ADR-131, "Widen before refit", the decoupling). Tasks 1 to 9 are done (C1: rebased to
`57a17a42`, then amended on the owner's word to hold this plan and spec revision 4a; spec §17.1). Task 10 is a record of run 9a; Task 11 is run 9b as a parity run; Task 12 is C2, the record and the
release. The goal line above still names the refit; spec §17 is the current scope. Revision 3a: the answers to
plan review round 4 (spec §18). Revision 4a: Task 13, Part E (spec §19: E1 in preflight check 1, E2 in the driver), before Task 11.

## Global Constraints

- No subagents. Every test is seen to fail first, then to pass. Byte-preserving edits; the repo files are CRLF.
- One commit per provenance step (revision 3, spec §17.4): C1 (the build) and C2 (the record of runs 9a and 9b,
  and the version if the owner sets one). There is no adoption commit and no C3. Each commit: A and B
  implementation review, the answers, `/final-review`, the diff to the owner, his yes for commit and push, green
  CI. Never an amend after a launch has cloned the commit. (The C1 amend of 2026-11-12, PR-S206 to PR-S207, came
  with the rebase on the owner's yes; the run that cloned the old C1, run 9a, is superseded.)
- Every HF launch, every preflight, every adoption, every publish run, every `prune --delete`, the PR, the merge
  and the tag: the owner's go at that time.
- Nothing heavy on this PC (owner 2026-11-11): lint, format, targeted pyright and single test files, one file per
  pytest process. The full suite is CI's. Temp files and `--basetemp` under `D:\Temp\hfu-topup\`; `TMP`/`TEMP` set
  to `D:\Temp\hfu-topup\tmp`.
- C1 holds no version. The version is the owner's; if he sets one, it goes into C2 (spec §17.6 item 16).
- Constants: `UNLISTED_STEPS = 5`; tag = the first 12 hex characters of the sha256 of the keys file bytes;
  `_BOX_RAM = 32_000_000_000`; `_HEADROOM = 2 * 1024**3` (unchanged).
- Tools: tests with `.venv-hfjobs/Scripts/python.exe -m pytest <file> -m "not e2e and not slow" -p no:randomly
  -q --benchmark-skip --basetemp D:\Temp\hfu-topup\pt\<name>`; ruff `D:\Temp\hfu-next\lintvenv\Scripts\ruff.exe`;
  pyright `.venv/Scripts/python.exe -m pyright --pythonpath .venv-hfjobs/Scripts/python.exe <files>`. The doc and
  bundle gates are in `tests/` (not `tests/scripts/`). `int_key` is `s.int_key` in the driver tests.
- Revision 2a ran the adoption of run 9a on the DGX (owner 2026-11-11, "2: use the DGX"). With the refit out of
  this cycle (spec §17), no `adopt` runs, so no step uses the DGX.
- CI runs only on a pull request and on `main` (`.github/workflows/ci.yml`), so the CI of C1 and C2 runs on
  the PR of this branch. Opening that PR is its own owner gate.

## Review Focus

1. A resume over jobs that a driver BEFORE this cycle launched: their labels have no `tag`. `label_of`, the
   ended-job line and the top-up must work on them (Task 3 and Task 6 tests).
2. A job name over the 100-character limit: the tag must still make two names differ (Task 3 test).
3. A top-up of an instance under a mount cap (`tc3` = 6): the launches of one step never pass the cap
   (Task 5 test).
4. A launch whose manifest write got a 429, then a driver restart: the job is unknown, so its stage waits
   (Task 6 test).
5. `compare` with several baselines: the census row still reads only `<NEW>/verify.json` (Task 7 test).

---

### Task 1: Part B, the memory limit of the box (§4)

**Files:**
- Modify: `scripts/hf_jobs/driver.py` (`_BOX_RAM`, its comment)
- Test: `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:** Produces nothing new.

- [ ] **Step 1: Write the failing test**

```python
def test_an_escalation_job_is_capped_under_the_cgroup_limit(tmp_path) -> None:
    """HFU launch top-up spec §4: `memory.max` of a cpu-upgrade job is 32,000,000,000 bytes."""
    from scripts.hf_jobs.driver import _HEADROOM

    inst, plan = _inst(), _single_stage_plan(tmp_path)
    launch = Launch(inst.id, ("extract",), (0, 1), "run", 1, ("a",), escalate=True)
    spec = _job_spec(_campaign(), _COMMIT, inst, launch, plan, store=_FakeStore())
    assert spec.cap_bytes == 32_000_000_000 - _HEADROOM
```

- [ ] **Step 2: Run it.** Expected: FAIL, `34359738368 - 2147483648` against `29852516352`.
- [ ] **Step 3: Set `_BOX_RAM = 32_000_000_000`** with the comment "cgroup `memory.max` of a cpu-upgrade job,
  measured 2026-11-11 (job 6af4d4e79737d9a9e093649c)".
- [ ] **Step 4: Run `test_hf_driver.py`.** Expected: all pass.

### Task 2: Part C, CPU time in the unit ledger (§5)

**Files:**
- Modify: `scripts/_stages.py` (`_run_unit`, the ledger record)
- Test: `tests/scripts/test_stage_contract.py`

**Interfaces:** Produces the ledger keys `cpu_user_s: float`, `cpu_sys_s: float`, `cpu_scope: "process"`.

- [ ] **Step 1: Write the failing test** in `test_stage_contract.py`, next to the ledger test of line 283. Use a
  one-stage pipeline whose `work` burns CPU (a loop of about 0.2 s), run one unit with `s.run_unit`, read
  `<gen>/<key>.ledger.json`:

```python
assert rec["cpu_scope"] == "process"
assert isinstance(rec["cpu_user_s"], float) and isinstance(rec["cpu_sys_s"], float)
assert rec["cpu_user_s"] > 0 and rec["cpu_sys_s"] >= 0
```

- [ ] **Step 2: Run it.** Expected: FAIL, `KeyError: 'cpu_scope'`.
- [ ] **Step 3: In `_run_unit`,** take `os.times()` next to `started` and again next to `finished`; add the three
  keys to `record` (the difference of `.user` and of `.system`).
- [ ] **Step 4: Run `test_stage_contract.py`, `test_stage_reads.py`, `test_unit_log_lines.py`, one per process.**
  Expected: all pass.

### Task 3: The tag (§3.4)

**Files:**
- Modify: `scripts/hf_jobs/launch.py` (`JobSpec`, `job_name`, `labels_for`, `label_of`; new helpers)
- Modify: `scripts/hf_jobs/driver.py` (`Launch`, `_job_spec`, the `[driver] launch` line, `_say_ended_jobs`)
- Test: `tests/scripts/hf_jobs/test_launch_typed.py`, `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:**
- Produces, in `launch.py`:
  - `keys_file_bytes(keys: Sequence[str]) -> bytes`: `json.dumps(sorted(keys)).encode("utf-8")` (the bytes that
    `_job_spec` writes today).
  - `keys_tag(keys: Sequence[str]) -> str`: `hashlib.sha256(keys_file_bytes(keys)).hexdigest()[:12]`.
  - `shard_text(labels: Mapping[str, str]) -> str`: `labels["shard"]`, plus `-<tag>` when `labels.get("tag")`.
  - `JobSpec.tag: str = ""` (the last field).
- Produces, in `driver.py`: `Launch.tag: str = ""`, set in `Launch.__post_init__` to `keys_tag(self.keys)` when
  `self.keyed` and no tag was given (`object.__setattr__`, the dataclass is frozen). Every keyed launch, a reduce
  trigger included (`keys_tag(())`), has the tag, whoever builds it.

- [ ] **Step 1: Write the failing tests.**

In `test_hf_driver.py`, the collision of today's code (spec test 1), with the plan helpers of the escalation
test at line 1118:

```python
def test_two_launches_of_one_step_never_share_a_keys_file_or_a_name(tmp_path) -> None:
    """Spec §3.4: an escalation launch and a run launch of one stage, one step, both `(0, 1)`."""
    from scripts.hf_jobs.launch import job_name, label_of, labels_for

    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work()),))
    _run(ctx, "extract", "a")
    plan = _realplan(ctx)
    plan["stages"][0]["memory_units"] = ["b"]
    inst = _inst()
    launches = _dec(inst, plan).launches
    assert [(lx.escalate, lx.shard, lx.keys) for lx in launches] == [(True, (0, 1), ("b",)), (False, (0, 1), ("c",))]
    store = _FakeStore()
    specs = [_job_spec(_campaign(), _COMMIT, inst, lx, plan, store=store) for lx in launches]
    assert len(store.uploads) == 2
    assert {json.loads(v)[0] for v in store.uploads.values()} == {"b", "c"}
    assert len({job_name(sp) for sp in specs}) == 2
    assert len({label_of(labels_for(sp)) for sp in specs}) == 2
```

In `test_launch_typed.py` (spec test 13):

```python
def test_the_tag_names_the_keys():
    assert L.keys_tag(["b", "a"]) == L.keys_tag(["a", "b"]) == hashlib.sha256(b'["a", "b"]').hexdigest()[:12]
    assert L.keys_tag(["a"]) != L.keys_tag(["b"])
    s = _spec(tag=L.keys_tag(["a"]))
    assert L.job_name(s).endswith(f"--1of4-{s.tag}")
    assert L.labels_for(s)["tag"] == s.tag and L.labels_for(s)["shard"] == "1of4"
    assert L.label_of(L.labels_for(s)) == f"ghost_gk.default.extract.1of4-{s.tag}" == L.job_env(s)["SILLY_KICKS_JOB_LABEL"]


def test_a_spec_without_a_tag_keeps_the_names_of_today():
    s = _spec()
    assert "tag" not in L.labels_for(s) and L.job_name(s).endswith("--1of4")
    assert L.label_of({"job": "j", "variant": "v", "stage": "s", "shard": "0of1"}) == "j.v.s.0of1"  # an old job


def test_two_long_names_that_differ_only_by_the_tag_differ():
    long = dict(run_key="r" * 90)
    a, b = L.job_name(_spec(**long, tag=L.keys_tag(["a"]))), L.job_name(_spec(**long, tag=L.keys_tag(["b"])))
    assert a != b and len(a) <= L.LABEL_MAX and len(b) <= L.LABEL_MAX
```

In `test_hf_driver.py` (spec tests 8 and 18): a launch of `_download_inst()` has `tag == ""`, its spec has no
`tag` label and its keys file is not written; `_job_spec` of a keyed launch writes
`.../<variant>/<index>of<count>-<tag>.json`; a step's `[driver] launch` line ends with `<index>of<count>-<tag>
units=<n>`; the ended-job line of a record with a `tag` label names `<shard>-<tag>`, and of a record without one
names `<shard>` (Review Focus 1).

- [ ] **Step 2: Run the two files.** Expected: FAIL (the collision test: one upload, one name; the others:
  `AttributeError` or the old forms). `test_a_spec_without_a_tag_keeps_the_names_of_today` PASSES today: it is a
  guard of today's form, and the mutation "a tag on every spec" (Task 8) turns it red.
- [ ] **Step 3: Implement the helpers and the fields of the Interfaces block.** `job_name`: the last part is
  `f"{i}of{n}"` plus `f"-{spec.tag}"` when set, before the length rule. `labels_for`: add `"tag": spec.tag` only
  when set. `label_of`: use `shard_text`. `_job_spec`: the keys file name `f"{index}of{count}-{launch.tag}.json"`,
  its bytes `keys_file_bytes(launch.keys)`, and `tag=launch.tag` in the `JobSpec`. The `[driver] launch` line and
  `_say_ended_jobs`: `shard_text`.
- [ ] **Step 4: Run `test_launch_typed.py` and `test_hf_driver.py`.** Expected: all pass. Fix the tests of today
  that pin the old keys file name (`test_launch_typed.py:236` builds its own path and stays).

### Task 4: The record of launched jobs (§3.3)

**Files:**
- Create: `scripts/hf_jobs/launched.py`
- Test: `tests/scripts/hf_jobs/test_launched.py`

**Interfaces:**
- Produces:
  - `UNLISTED_STEPS: Final = 5`
  - `class LaunchedJobs` with:
    - `add(self, job_id: str, labels: Mapping[str, str], units: Sequence[str], created_at: str) -> None`
    - `reconcile(self, listing: Sequence[Mapping]) -> tuple[list[dict], list[tuple[str, str]]]`: called once for
      each listing that a step read. Returns `(held, dropped)`. `held`: one job record (the `ledger.job_record`
      shape: `id`, `labels`, `stage="SCHEDULING"`, `message=None`, `created_at`, `started_at=None`,
      `finished_at=None`, `wall_s=None`) for each job of the record that the listing does not show and that no
      listing has shown ENDED. `dropped`: `(job_id, label)` of each job that the listing has not shown at
      `UNLISTED_STEPS` listings in a row; it leaves the record.
    - `units_of(self, job_id: str) -> tuple[str, ...] | None`: the units of a launched job, or the remembered
      answer of the store for another job; `None` when not known.
    - `knows(self, job_id: str) -> bool`
    - `remember(self, job_id: str, units: Sequence[str] | None) -> None`: the store's answer for a job that this
      run did not launch (`None` = no manifest); it is kept for the run.
- A job is ENDED when `failures.classify(record).outcome` is not `JobOutcome.IN_FLIGHT`.

- [ ] **Step 1: Write the failing tests** (`test_launched.py`), each with records built as dicts:
  - a job added and absent from the listing: `held` has its record with `stage == "SCHEDULING"` and its labels;
  - the same job in the listing as `RUNNING`: `held` is empty; the next listing leaves it out: it is held again
    (a job listed in flight and then missing is held, §3.3);
  - the same job listed as `COMPLETED` once, then missing: never held again;
  - absent from 5 listings: `dropped == [(id, label)]` at the fifth `reconcile`, not before; `units_of` is then
    `None` and `held` no longer has it;
  - absent, then listed, then absent: the count restarts (5 more listings to drop);
  - `remember("x", None)`: `knows("x")` is true and `units_of("x")` is `None`; `remember("y", ["a"])`:
    `units_of("y") == ("a",)`.
- [ ] **Step 2: Run it.** Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 3: Implement `launched.py`** (a module docstring that names spec §3.3 and the two limits of §11).
- [ ] **Step 4: Run it.** Expected: all pass.

### Task 5: The top-up rule in `decide` (§3.2, §3.6)

**Files:**
- Modify: `scripts/hf_jobs/driver.py` (`decide`; replace `_stage_in_flight` by `_stage_flight`)
- Test: `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:**
- Consumes: `Launch.tag` (Task 3).
- Produces: `decide(..., flight_units: Mapping[str, Sequence[str] | None] | None = None)`;
  `_stage_flight(live_jobs, inst, stage) -> list[Mapping]` (the in-flight records of the stage, as
  `_stage_in_flight` matched them).

The rule, in `decide`'s stage walk, after `units_per_job` is set:

1. Compute `pending`, `candidates`, `memory_units`, `to_run` and `stuck` as today (lines 740 to 754), but append
   the `timed out` reports only in step 3.
2. `flight = _stage_flight(live_jobs, inst, name)`. If `flight`:
   - when `not _keyed(inst)`, or a record has no `id`, or `(flight_units or {}).get(id)` is `None`, or a unit
     list is empty: no launch, no report; `break` (today's barrier);
   - else `held` = the union of the unit lists; `rest = [u for u in to_run if u not in held]`;
     `rest_memory = [u for u in memory_units if u not in held]`; `room = max_shards - len(flight)`; when
     `room <= 0` or both lists are empty: no launch; else
     `esc = _keyed_launches(inst, name, rest_memory, units_per_job, room, workers, escalate=True)` (none when
     `rest_memory` is empty) and, when `room - len(esc) >= 1` and `rest`,
     `_keyed_launches(inst, name, rest, units_per_job, room - len(esc), workers)`. No report. Then the mount
     headroom cut of today, then `break`.
3. No flight: today's code from line 744 on, unchanged (the reports, the launches, the reduce).

- [ ] **Step 1: Write the failing tests** (spec tests 2 to 12, Review Focus 3). Build the flight with
  `_running(inst, "extract", shard=…)` plus an `"id"`, and pass `flight_units={id: [...]}`:
  - 2: three units `a`, `b`, `c`, one job `j1` holds `a`: the launches hold exactly `b` and `c`.
  - 3: `units=[int_key(i) for i in range(3961)]`, 6 jobs hold the first 1,980 keys: 6 launches, 1,981 keys in all,
    none of a held key, each of 330 or 331.
  - 4: 11 jobs in flight: one launch holding every unit left; 12: no launch.
  - 5: `room == 2`, `memory_units == ["m"]`, two units to run: one escalation launch (`("m",)`) and one run launch.
  - 6: a flight record without `id`, and one with `flight_units[id] is None`: no launch, no report;
    `test_no_launch_while_stage_in_flight` stays as it is.
  - 7: a job of the flight with `[]`: no launch; every unit held and the stage not reduced: no launch, no reduce.
  - 8: `_download_inst()` with a known flight: no launch.
  - 9: a memory unit that no job holds: one escalation launch.
  - 10: a unit with two timeouts (`timeouts=`) and a flight: in no launch, and `decision.reports == ()`; with no
    flight the `timed out 2x` report comes (today's test, unchanged).
  - 11: a recorded failed unit held by a running job: not launched; with no flight and coverage 1: launched.
  - 12: three jobs, one ERROR (its units no longer held: not in the flight), two RUNNING: the step launches the
    crashed job's units.
  - Review Focus 3: a `_mounting(…, "tc3")` instance with `mount_cap_for=lambda d: 6`, 4 jobs in flight that
    mount `tc3`: at most 2 launches.
- [ ] **Step 2: Run them.** Expected: FAIL. Tests 2 to 5, 9, 11 and 12 and Review Focus 3 fail on the behaviour
  (no launch while a stage is in flight). Tests 6, 7, 8 and 10 (its flight half) fail only by the `TypeError` of
  the new `flight_units` argument: they guard today's barrier. Task 8 names the mutation that turns each red:
  6 "an unknown unit list is topped up", 7 "a reduce trigger in flight is topped up", 8 "a `download` instance is
  topped up", 10 "a STUCK report during a flight".
- [ ] **Step 3: Implement the rule above.** `_stage_flight` keeps the match of `_stage_in_flight`; remove
  `_stage_in_flight`.
- [ ] **Step 4: Run `test_hf_driver.py`.** Expected: all pass.

### Task 6: Wire the record into a run (§3.3)

**Files:**
- Modify: `scripts/hf_jobs/driver.py` (`_State`, `_gather`, `_decide`, `step`, `run`, `_dry_run`)
- Test: `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:**
- Consumes: `LaunchedJobs`, `UNLISTED_STEPS` (Task 4); `decide(flight_units=)` (Task 5); `labels_for` (Task 3).
- Produces: `step(..., launched: LaunchedJobs | None = None)`; `_gather(..., launched: LaunchedJobs)`;
  `_State.flight_units: dict[str, tuple[str, ...] | None]` (default `{}`); `run` owns one `LaunchedJobs`.

- [ ] **Step 1: Write the failing tests** (spec tests 14 to 17, Review Focus 1 and 4), through `_step` with a
  `_LaunchRecorder`, a `_ManifestStore`-backed API or `live_records=`:
  - 14: a launch whose manifest write fails: the store raises a 429 only for a path with `/by_job/` (the keys file
    write of `_job_spec` must succeed), and `driver._PUT_BACKOFF_S` is monkeypatched to `(0.0, 0.0, 0.0)`: the job
    is in the record with its units. An in-flight job that the record does not know: `_gather` reads its manifest
    from the store once over two steps, also when the store has none (count the reads).
  - 15: a launched job that the next listing leaves out: `StepResult.in_flight` counts it, its units are in no
    launch, `run` reports no stall; after 5 listings it is dropped, the line `[driver] job <id> <label> was
    launched but is not in the listing after 5 steps; its units are free again` is printed and its units are
    launched; a step that ends in a 429 at the listing does not count.
  - 16: one stage of 24 units, `units_per_job=2`, `max_shards=12`, `concurrency_cap=12`, `launch_batch=6`, two
    steps, the second with a listing that shows the first step's 6 jobs RUNNING. Step 1: budget
    `min(12 - 0, 6) = 6`, the split is 12 jobs of 2, 6 start. Step 2: budget `min(12 - 6, 6) = 6`, `room = 6`, the
    12 units left in 6 jobs of 2. So 12 jobs, every unit in exactly one job.
  - 17: `_dry_run` with a known flight (manifests in the store) prints the top-up launches, each with its
    `tag`.
  - Review Focus 1: a resume (empty record) over in-flight jobs without a `tag` label whose manifests are in the
    store: the top-up launches the rest.
  - Review Focus 4: a resume over an in-flight job whose manifest is not in the store: no launch for its stage.
- [ ] **Step 2: Run them.** Expected: FAIL. Review Focus 4 fails only by the `TypeError` of the new `launched`
  argument: it guards today's barrier (mutation "the unknown case launches").
- [ ] **Step 3: Implement.** The new parameter of `_gather` is keyword-only with a default
  (`*, launched: LaunchedJobs | None = None`), after `fetch_logs=None`.
  - `_gather`: after the listing, `held, dropped = launched.reconcile(live_jobs)`; `_say` one line per
    `dropped`; `live_jobs += held`. Then, for each in-flight job of this run's instances (`_instance_index`):
    `launched.units_of(id)` when `launched.knows(id)`, else `_manifest_for(...)` and `launched.remember(...)`.
  - `step`: when `do_launch` returns `job_id`, `launched.add(job_id, labels_for(spec), launch.keys, <now, ISO
    8601 UTC>)` before the manifest write. A `step` without `launched` makes its own (a test seam).
  - `run`: one `LaunchedJobs` for the run, passed to each `step`, as `plan_cache` is.
  - `_dry_run`: an empty `LaunchedJobs`; its JSON gives each launch's `tag`.
- [ ] **Step 4: Run `test_hf_driver.py`, `test_launched.py`, `test_launch_typed.py`.** Expected: all pass.

### Task 7: Part D, `compare` with several baselines (§14.2)

**Files:**
- Modify: `scripts/hf_jobs/adopt.py` (`_staging_run_key`, `compare_staging`)
- Modify: `scripts/hf_jobs/driver.py` (`main`: `--baseline`; `_compare`; `_BASELINE_FORMS` line)
- Test: `tests/scripts/hf_jobs/test_adopt.py`, `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:**
- Produces: `_staging_run_key(folder, campaign: str | None, job, tier) -> Path | None` (`None` = any campaign
  name: the pattern `[A-Za-z0-9_]+-[0-9a-f]{7}-<job>-<tier>`); `compare_staging(c, *, staging: Path,
  baselines: Sequence[Path]) -> dict[str, dict]`; each trainer row with a baseline gets
  `"baseline_run_key": <run key folder name>`; `_compare(campaign, staging, baselines) -> int`; `--baseline`
  with `action="append"`.

- [ ] **Step 1: Write the failing tests** (D1 to D6, Review Focus 5); `_compare_staging_dir` gets a
  `campaign: str = "cycle1"` argument for its run key names.
  - D1: one baseline of the same campaign: today's rows, each with `baseline_run_key ==
    f"cycle1-{_CA[:7]}-xsuccess-public"`. Update the exact-dict assertions of the tests of lines 953 to 1065 to
    the new key and to `baselines=[old]`.
  - D2: a baseline whose run keys are `tf46_refit-…`: found; the row names that run key.
  - D3: two baselines both holding xsuccess: the row reads the second (`baseline_run_key` of the second); an
    instance only the first holds: read from the first.
  - D4: no baseline holds the instance: `{"identical": False, "differences": ["present in only one staging
    folder"]}` (no form, no run key).
  - D5: `cycle1-<c7>-xsuccess-public` and `tf46_refit-<c7>-xsuccess-public` in one baseline folder:
    `AdoptionRefusedError`.
  - D6 (`test_hf_driver.py`): `--baseline A --baseline B` reaches `compare_staging` as `[A, B]`; the line reads
    `IDENTICAL (baseline: adopted copy of <run key>)`; exit 0 and 1 as today; no `--baseline`: `SystemExit`.
  - Review Focus 5: two baselines; the census row equals the row of one baseline (it reads `<new>/verify.json`).
- [ ] **Step 2: Run the two files.** Expected: FAIL.
- [ ] **Step 3: Implement.** The NEW side keeps `campaign=c.name`. For each instance, walk `baselines` in order
  and keep the last folder with a run key and a variant folder.
- [ ] **Step 4: Run the two files.** Expected: all pass.

### Task 8: The docs of C1, the local checks, the mutation checks

**Files:**
- Modify: `docs/howto/hf-jobs-training.md`, `docs/superpowers/adrs/ADR-128-compute-contract-hf-campaigns.md`,
  `CHANGELOG.md`, `TODO.md`, `docs/superpowers/specs/2026-11-10-hfu-fit-units-carrier-index-hygiene-design.md`
- Add: the spec and this plan (both untracked today)

- [ ] **Step 1: The docs (§14.4, the C1 row).** How-to: the launch rule, the tag (keys file, job name, labels),
  the unlisted-job rule, the ledger keys, `compare` with several `--baseline` folders, the order "refit run,
  adoption, full parity run". ADR-128: an amendment of 2026-11-11 (Parts A to D and the investigation of §1).
  `CHANGELOG.md`: a new `## [Unreleased]` section above `[4.141.0]` with the scripts-only entries. `TODO.md`:
  remove the row "Find the cause of four launch-8 timings". The fit-units spec §17.1: one pointer line to §1 of
  the new spec. No version, no word on the refit.
- [ ] **Step 2: Run the checks.** ruff check and format (CI scope), pyright on the changed `.py` files, then one
  pytest process for each changed test file and for `test_stage_reads.py`, `test_howto_links_wired.py`,
  `test_todo_md_format.py`, `test_agents_md_budget.py`, `test_version_single_source.py`. Expected: 0, 0, 0
  errors, all pass. Check the CRLF of every changed file.
- [ ] **Step 3: The mutation checks** (§6), one at a time in a scratch script that restores the file: the top-up
  runs a held unit; the tag is dropped from the keys file name; `room` is `max_shards`; the unknown case launches;
  a job in the record that the listing leaves out holds nothing; `compare` keeps the first baseline; the lookup
  uses the campaign's own name; and the guard mutations of Tasks 3, 5 and 6 (a tag on every spec; a reduce
  trigger in flight is topped up; a `download` instance is topped up; a STUCK report during a flight). Expected:
  each caught by a named test.

### Task 9: The C1 gate

- [ ] **Step 1:** A and B implementation review of the C1 diff (reports `D:\Development\_reviews\
  2026-11-11-hfu-launch-topup-impl-{A,B}.md`); fix or answer every finding; rounds until both approve.
- [ ] **Step 2:** `/final-review`.
- [ ] **Step 3:** Show the owner the file list and the diff; ask "commit and push C1?". On his yes: one commit
  (Conventional Commits, the attribution line) and a push.
- [ ] **Step 4:** Ask the owner "open the draft PR?". On his yes: the draft PR. Its CI is the CI of C1 (and later
  of C2). Watch it until green; a red CI stops before run 9a. (Done. After the rebase, Task 11 step 0 repeats
  this stop for run 9b.)

### Task 10: Run 9a (done; superseded as an adoption source)

Run 9a ran on 2026-11-12 at the pre-rebase C1 `aadd4cf7` (spec §17.1). 7 of 8 arms passed; `xcross_attempt-public`
/ `position_only` failed its Brier gate. Nothing was retrieved or adopted. The driver log, the gates of the 8 arms,
the 704 per-job files and the check for units in two jobs are saved under `D:\Temp\hfu-topup\run9a\`. Its run keys
are pruned. The steps of revision 2a (the DGX adoption, C2 as an adoption commit) are removed with the refit.

### Task 11: Run 9b, the parity run (spec §17.3)

Each step on the owner's go.

- [ ] **Step 0 (precondition):** the PR CI of the amended C1 is green before any preflight; a red CI stops before
  run 9b (the CI of `57a17a42` does not count). `git status --porcelain --untracked-files=no` prints nothing: a
  tracked edit moves the driver's code identity away from the commit (the first run 9b, spec §17.1).
- [ ] **Step 1:** `cycle1.toml` (whole) `preflight` and `launch` at the amended C1; the log under
  `D:\Temp\hfu-topup\run9b2\`; a watcher. A STUCK line of an xShot/xCross instance whose unit failed with
  `AcceptanceGatesFailedError` is recorded and reported, and is no stop signal (spec §17.3 item 6; the expected one
  is `xcross_attempt-public/position_only`). Every other failure signal is a stop: a STUCK or failed job outside
  xShot and xCross, any crash (`Traceback` other than that gate error, FATAL), or a unit started by two jobs
  whose run times overlap (spec §17.3 item 4; a retry is in two or more jobs by design).
- **The window (spec §19.3):** from Step 0 to the end of Step 3, no tracked edit. The record is drafted in
  `D:\Temp\hfu-topup\record\` and enters the tree only after Step 3, for C2. A `FATAL code_identity` line is a
  stop: restore the tree, then resume on the owner's go.
- [ ] **Step 2:** `retrieve` and `verify`. Expected: every instance outside xShot and xCross passes. The 8
  xShot/xCross instances (4 job/tier pairs, 2 variants each): record the result (spec §17.3 item 6).
- [ ] **Step 3:** `compare --staging <9b> --baseline D:\Development\_staging\cycle1-a1a3b52`. Expected: the 12
  trainer artifacts outside xShot and xCross `IDENTICAL`, the census verdict ok; the 8 xShot/xCross rows differ
  or are missing (expected), so the exit code is non-zero. Any other row that is not `IDENTICAL` stops here:
  report the rows to the owner.
- [ ] **Step 4:** The acceptance of spec §17.3 items 3 to 5 from the job records, the logs and the manifests (a
  read-only scratch script): the xsuccess `extract` time (under 15 min) with the load from the nested studies next
  to it; every started unit in its job's manifest and no unit started by two jobs at once, for runs 9a and 9b; the
  wall times, the step lines, the `fit` unit times with their CPU time. A miss stops at the owner.
- [ ] **Step 5:** Report every xShot/xCross gate result to the owner and to the TF-46 session.

### Task 12: C2, the record

- [ ] **Step 1:** The record of runs 9a and 9b (a new section of the spec, the ADR-128 amendment). Ask the owner
  to confirm the version (spec §17.6 item 16: 4.142.1).
  - With a version: set it in `silly_kicks/_version.py`, the `CHANGELOG.md` heading (the `[Unreleased]` section
    becomes that version), `uv.lock`, the `TODO.md` "Current" line.
  - Without a version: `CHANGELOG.md` keeps the `[Unreleased]` section; `_version.py`, `uv.lock` and the `TODO.md`
    "Current" line do not change; Step 4 has no tag.
- [ ] **Step 2: The local checks of C2,** one file per process: `tests/test_version_single_source.py`,
  `tests/test_todo_md_format.py`, `tests/test_howto_links_wired.py`, `tests/test_agents_md_budget.py`; `uv lock
  --check` (with a version); ruff and format. Expected: all pass.
- [ ] **Step 3:** The C2 gate: A and B review, `/final-review`, the owner's yes, commit, push, CI green on the PR.
- [ ] **Step 4:** Each on the owner's yes: the PR ready, the merge (a merge commit), the tag after green CI on
  `main` (it publishes to PyPI; only with a version), and each `prune --delete`.

### Task 13: Part E, no launch from a tree that the jobs do not run (spec §19; runs BEFORE Task 11)

**Files:** `scripts/hf_jobs/preflight.py` (check 1), `scripts/hf_jobs/driver.py` (`step`),
`tests/scripts/hf_jobs/test_preflight.py`, `tests/scripts/hf_jobs/test_hf_driver.py` (the new tests and the `_ctx`
fixture), `docs/howto/hf-jobs-training.md`, the ADR-128 amendment (item 56), `CHANGELOG.md`. No "seven checks"
text changes (owner ruling 1: the rule is inside check 1).

**Interfaces:** `preflight._controller_identity() -> str` returns `code_identity(git_provenance())`;
`check_ref(commit: str, *, run=subprocess.run, identity: Callable[[], str] = _controller_identity) -> Check`.
`driver.step` returns `StepResult(fatal=("code_identity", <detail>), launched=0, ...)` when a readable plan's
`code_identity` is not `commit` (a missing key counts as another identity).

- [ ] **Step 1 (E1 RED):** the four `test_check_ref_*` tests of spec §19.4. The `run` fake answers `git status
  --porcelain --untracked-files=no` and records every argv, so the dirty test also asserts that no `gh` argv was
  made. Run `test_preflight.py`. Expected: the four FAIL (`identity` is not a parameter), the others pass.
- [ ] **Step 2 (E1 GREEN):** implement; the tree test in check 1 after the HEAD test and before the fetch and `gh` (spec §19.1). Run the file. Expected: all pass.
- [ ] **Step 3 (E2 RED):** the three `test_step_*identity*` tests of spec §19.4, and `_ctx` with `_COMMIT` for
  `commit` and `code_identity`. Run `test_hf_driver.py`. Expected: the identity-mismatch and no-identity tests
  FAIL (a launch happens); every other test passes.
- [ ] **Step 4 (E2 GREEN):** implement in `step`, after `_gather` and before `_decide`. Run the file. Expected:
  all pass.
- [ ] **Step 5: mutations,** each caught by the named test, with injected identities only (the real tree of the
  test run plays no part):
  - check 1 accepts a `dirty-` identity: `test_check_ref_fails_on_a_tracked_dirty_tree_and_names_the_paths`;
  - check 1 accepts an `unknown-` identity: `test_check_ref_fails_on_unknown_provenance`;
  - check 1 skips the identity compare: `test_check_ref_fails_on_any_other_identity_at_the_right_head`;
  - check 1 shows every path: `test_check_ref_names_at_most_five_paths`;
  - `_controller_identity` returns `git_provenance()["commit"]` (a second rule): `test_check_ref_default_identity_is_the_driver_one`;
  - `step` skips the guard: `test_step_stops_before_any_launch_when_a_plan_has_another_identity`;
  - `step` reads a missing key as the commit: `test_step_stops_when_a_plan_has_no_identity`.
  - check 1 never compares with the default identity, or its default is HEAD only:
    `test_check_ref_without_identity_uses_the_driver_identity`;
  - the guard stops at the first plan at the commit: `test_step_stops_when_a_later_plan_has_another_identity`
    and `test_dry_run_reports_a_later_foreign_identity`;
  - `_dry_run` skips the guard: `test_dry_run_reports_a_foreign_identity_as_fatal_and_no_launch`;
  - `step` refuses every plan: `test_step_launches_when_every_plan_has_the_commit_identity` (the positive
    control; it passes at RED by design).
- [ ] **Step 6: docs and local checks.** The how-to (check 1, the FATAL `code_identity`, the window of §19.3),
  ADR-128 item 56, `CHANGELOG.md` `[Unreleased]`. Run `test_preflight.py`, `test_hf_driver.py`, `test_campaign.py`,
  `tests/test_howto_links_wired.py`, one process each; ruff check and format on the changed files; pyright on
  `preflight.py`, `driver.py` and the two test files. Expected: all pass, 0 errors.
- [ ] **Step 7: the gate.** A and B implementation review of the C1 delta (reports
  `D:\Development\_reviews\2026-11-12-hfu-launch-topup-impl-r3-{A,B}.md`); `/final-review`; the owner's yes for
  the amend of `d79311d2` (it takes Part E, spec revision 5a and plan revision 4a) and for the force-push; green CI
  on the amended C1. Then Task 11.
