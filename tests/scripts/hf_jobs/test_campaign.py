"""Stateless campaign controller (spec 5.5): pure next-action logic over --plan JSON + job labels."""

from __future__ import annotations

import types

import pytest

from scripts.hf_jobs import campaign as C
from scripts.hf_jobs.planner import Constraints, UnitCost

SHA = "abcdef1" + "0" * 33


def _plan(stages):
    return {"pipeline": "ghost_gk", "stages": stages}


def _st(
    name,
    *,
    n=4,
    done=0,
    pending=None,
    ready=True,
    reduced=False,
    inner=False,
    hk=None,
    units=None,
    needs=(),
    failed=None,
    max_failures: int | None = 0,
):
    return {
        "name": name,
        "n_units": n,
        "done": done,
        "pending": (n - done) if pending is None else pending,
        "failed": 0,
        "excluded": 0,
        "ready": ready,
        "reduced": reduced,
        "inner_parallel": inner,
        "inner_width": 5 if inner else 1,
        "history_key": hk or name,
        "units": units or [f"u{i}" for i in range(n)],
        "pending_units": (units or [f"u{i}" for i in range(n)])[done:],
        "failed_units": dict(failed or {}),
        "max_failures": max_failures,
        "token": f"t_{name}",
        "needs": list(needs),
    }


def test_run_key_is_derived():
    c = C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )
    assert C.run_key(c, SHA, "ghost_gk", "owner") == "cycle1-abcdef1-ghost_gk-owner"
    assert C.run_key(c, SHA, "ghost_gk", "owner", replica=2) == "cycle1-abcdef1-ghost_gk-owner-r2"


def _campaign():
    return C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )


def test_the_planning_constraints_use_the_committed_median_startup(tmp_path, monkeypatch):
    """Spec 5.5: the planner learns job startup from the committed summaries; 300 s only before any."""
    import json

    assert C.planning_constraints(_campaign(), summary_dir=tmp_path).startup_s == Constraints().startup_s == 300.0
    (tmp_path / "cycle0.json").write_text(json.dumps({"stages": [{"startup_s": 140.0}, {"startup_s": 160.0}]}))
    c = C.planning_constraints(_campaign(), summary_dir=tmp_path)
    assert c.startup_s == 140.0 and c.target_s == 3600.0 and c.concurrency_cap == 10


def test_a_summary_row_carries_workers_startup_and_the_floor():
    """Plan Task 21 step 8: per stage `workers`, `inner_workers`, `startup_s` and the irreducible floor."""
    import datetime

    def iso(t):
        return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()

    labels = {"job": "ghost_gk", "variant": "default", "stage": "cv", "shard": "1of1", "run_key": "rk"}
    job = {"labels": labels, "created_at": iso(990), "started_at": iso(1000), "finished_at": iso(9000), "wall_s": 8000}
    recs = [
        {"key": k, "outcome": "done", "wall_s": 3400.0, "peak_rss_bytes": 2**30, "attempt": 1,
         "job_label": "ghost_gk.default.cv.1of1", "started_ns": int((1000 + s) * 1e9), "worker": w, "inner_workers": 1}
        for k, s, w in (("a", 120, 0), ("b", 125, 1))
    ]  # fmt: skip
    st = {"name": "cv", "history_key": "hk", "n_units": 2}
    row = C.stage_summary("rk/default", st, recs, [job], None, Constraints(startup_s=300), job_recs=recs)
    assert (row["workers"], row["inner_workers"], row["startup_s"]) == (2, 1, 120.0)
    assert row["irreducible_floor_s"] == 3400.0  # p95 3400 s > the 3300 s budget
    assert (row["n_jobs"], row["max_job_wall_s"], row["unit_cost"]["n_samples"]) == (1, 8000, 2)
    empty = C.stage_summary("rk/default", st, [], [], None, Constraints(), job_recs=[])
    assert (empty["workers"], empty["startup_s"], empty["irreducible_floor_s"], empty["unit_cost"]) == (
        None,
        None,
        None,
        None,
    )


def test_a_grouped_job_reports_its_startup_once_and_never_an_upstream_stage_wall():
    """Review r3 A T21-CORR-01 / B CCHC-IMPL-01: a grouped single-unit job (`--stages select,fit`) runs its
    stages in order. Its startup is its first unit across ALL its stages minus `started_at`, recorded on the
    group's FIRST stage only; the later stage's first unit starts after the earlier stage's wall, which is
    not startup."""
    import datetime

    def iso(t):
        return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()

    labels = {"job": "xshot", "variant": "paired", "stage": "select-fit", "shard": "1of1", "run_key": "rk"}
    job = {"labels": labels, "created_at": iso(990), "started_at": iso(1000), "finished_at": iso(9000), "wall_s": 8000}

    def rec(key, start):
        return {"key": key, "outcome": "done", "wall_s": 600.0, "peak_rss_bytes": 2**30, "attempt": 1,
                "job_label": "xshot.paired.select-fit.1of1", "started_ns": int(start * 1e9), "worker": 0}  # fmt: skip

    select, fit = [rec("select", 1060)], [rec("fit", 1660)]  # true startup 60 s; select ran 600 s
    both = select + fit
    rows = {
        name: C.stage_summary(
            "rk/paired",
            {"name": name, "history_key": name, "n_units": 1},
            recs,
            [job],
            None,
            Constraints(),
            job_recs=both,
        )
        for name, recs in (("select", select), ("fit", fit))
    }
    assert rows["select"]["startup_s"] == 60.0
    assert rows["fit"]["startup_s"] is None  # never 660 s (= startup + the select stage's wall)
    assert rows["fit"]["max_job_wall_s"] == 8000  # the job still counts for the later stage's walls


def test_a_grouped_job_whose_first_stage_ran_no_unit_measures_startup_from_any_stage():
    """Review r4 B observation: a grouped job's first unit can belong to a LATER stage (here `select` was
    already finished by an earlier job, so this `select-fit` job ran only `fit`). The startup is measured
    over every stage's records (`job_recs`) and still recorded on the group's first stage."""
    import datetime

    def iso(t):
        return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()

    labels = {"job": "xshot", "variant": "paired", "stage": "select-fit", "shard": "1of1", "run_key": "rk"}
    job = {"labels": labels, "created_at": iso(990), "started_at": iso(1000), "finished_at": iso(2000), "wall_s": 1000}
    fit = [
        {"key": "fit", "outcome": "done", "wall_s": 600.0, "peak_rss_bytes": 2**30, "attempt": 1,
         "job_label": "xshot.paired.select-fit.1of1", "started_ns": int(1045 * 1e9), "worker": 0}
    ]  # fmt: skip
    row = C.stage_summary(
        "rk/paired",
        {"name": "select", "history_key": "select", "n_units": 1},
        [],
        [job],
        None,
        Constraints(),
        job_recs=fit,
    )
    assert row["startup_s"] == 45.0 and row["unit_cost"] is None


def test_a_stage_without_history_is_profiled_first():
    acts = C.next_actions(_plan([_st("extract", n=179)]), running=set(), history={}, c=Constraints())
    assert [a.kind for a in acts] == ["profile"] and acts[0].stages == ("extract",)


def test_a_stage_with_history_is_fanned_by_the_planner():
    hist = {"extract": {"cost": UnitCost(600, 700, 800, 6 * 2**30, 20), "efficiency": 0.9}}
    acts = C.next_actions(_plan([_st("extract", n=179)]), running=set(), history=hist, c=Constraints())
    assert acts[0].kind == "run" and acts[0].n_jobs > 1 and acts[0].workers == 3


def test_a_resumed_stage_hands_jobs_only_its_pending_units():
    """Real HF Cycle 1 (xshot public/default extract): 16 of 17 done, and the LPT partition `0of1.json`
    held one key that was ALREADY done (the history's per-unit walls are the done units'). Every relaunch
    skipped it in ~20 s and the pending unit never ran. A partition covers exactly the pending units."""
    units = [f"m{i:02d}" for i in range(17)]
    done = units[:16]
    per_unit = {k: 30.0 + i for i, k in enumerate(done)}
    hist = {"extract": {"cost": UnitCost(30, 40, 50, 2**30, 16), "efficiency": 1.0, "per_unit": per_unit}}
    acts = C.next_actions(
        _plan([_st("extract", n=17, done=16, units=units)]), running=set(), history=hist, c=Constraints()
    )
    assert acts[0].kind == "run"
    covered = sorted(k for part in (acts[0].partition or ()) for k in part)
    assert covered == ["m16"], acts[0].partition


def test_a_partition_never_names_a_unit_outside_the_pending_set():
    units = [f"m{i:03d}" for i in range(179)]
    per_unit = {k: 600.0 for k in units[:40]}
    hist = {"extract": {"cost": UnitCost(600, 700, 800, 2**30, 40), "efficiency": 1.0, "per_unit": per_unit}}
    acts = C.next_actions(
        _plan([_st("extract", n=179, done=40, units=units)]), running=set(), history=hist, c=Constraints()
    )
    parts = acts[0].partition
    assert parts is not None
    covered = sorted(k for part in parts for k in part)
    assert covered == units[40:] and acts[0].n_jobs == len(parts) > 1


def test_without_lpt_the_planner_shards_round_robin():
    """A `download` dataset fetches by `--shard i/N`, so it cannot take an LPT partition (`_job_specs`
    refuses one); the controller plans such an instance with `lpt=False`."""
    units = [f"m{i:03d}" for i in range(179)]
    hist = {"extract": {"cost": UnitCost(600, 700, 800, 2**30, 40), "efficiency": 1.0,
                        "per_unit": {k: 600.0 for k in units[:40]}}}  # fmt: skip
    acts = C.next_actions(
        _plan([_st("extract", n=179, done=40, units=units)]), running=set(), history=hist, c=Constraints(), lpt=False
    )
    assert acts[0].partition is None and acts[0].n_jobs > 1


def test_a_running_stage_is_not_relaunched():
    acts = C.next_actions(_plan([_st("extract", n=179)]), running={"extract"}, history={}, c=Constraints())
    assert acts == []


def test_the_barrier_waits_for_upstream():
    p = _plan([_st("extract", done=4, reduced=True), _st("cv", ready=True), _st("fit", n=1, ready=False)])
    hist = {"cv": {"cost": UnitCost(60, 60, 60, 2**30, 5), "efficiency": 1.0}}
    acts = C.next_actions(p, running=set(), history=hist, c=Constraints())
    assert [a.stages for a in acts] == [("cv",)]


def test_consecutive_single_unit_stages_run_in_one_job():
    p = _plan([_st("extract", done=4, reduced=True), _st("cache", n=1), _st("fit", n=1, ready=False)])
    acts = C.next_actions(p, running=set(), history={}, c=Constraints())
    assert acts[0].stages == ("cache",) and acts[0].kind == "run"  # fit is not ready: only cache now


def test_a_ready_chain_of_single_unit_stages_is_grouped():
    p = _plan(
        [
            _st("cv", n=4, done=4, reduced=True),
            _st("fit", n=1, needs=("cv",)),
            _st("importance", n=1, ready=False, needs=("fit",), inner=True),
            _st("assemble", n=1, ready=False, needs=("fit", "importance")),
        ]
    )
    acts = C.next_actions(p, running=set(), history={}, c=Constraints())
    assert acts[0].stages == ("fit",)  # an inner_parallel stage never joins a group (it owns the pool)
    p2 = _plan(
        [
            _st("cv", n=4, done=4, reduced=True),
            _st("fit", n=1, needs=("cv",)),
            _st("assemble", n=1, ready=False, needs=("fit",)),
        ]
    )
    assert C.next_actions(p2, running=set(), history={}, c=Constraints())[0].stages == ("fit", "assemble")


def test_a_finished_but_unreduced_output_stage_gets_a_reduce_job():
    p = _plan([_st("assemble", n=1, done=1, reduced=False)])
    assert C.next_actions(p, running=set(), history={}, c=Constraints())[0].kind == "reduce"


def test_everything_reduced_means_done():
    plan = _plan([_st("a", done=4, reduced=True)])
    assert C.next_actions(plan, running=set(), history={}, c=Constraints()) == []


# ---- one stateless step (spec 5.5): job labels -> running / latest-failed ------------------------

_RK = "cycle1-abcdef1-ghost_gk-owner"


def _job(stage, status, *, created_at=1, jid="j1", message=None):
    return types.SimpleNamespace(
        id=jid,
        labels={"run_key": _RK, "variant": "default", "stage": stage},
        status=types.SimpleNamespace(stage=status, message=message),
        created_at=created_at,
    )


@pytest.fixture
def step_world(monkeypatch):
    """``step`` over ONE instance whose next action is ``profile extract``, every network seam stubbed.
    Returns (run(jobs) -> report, launched specs)."""
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import preflight

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), _RK)
    c = C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )
    launched: list = []
    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([_st("extract", n=179)]))
    monkeypatch.setattr(C, "committed_history", dict)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "new")

    def run(jobs):
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
        return C.step(c, SHA, api=object(), token="hf_x")[inst.id]

    return run, launched


def test_a_stage_whose_latest_job_ended_in_error_is_reported_never_relaunched(step_world):
    """The legacy ``wait_all`` raised on a terminal ERROR. The stateless controller reports it per
    instance and launches nothing for that stage: a blind relaunch would loop on a deterministic failure."""
    run, launched = step_world
    r = run([_job("extract", "ERROR", message="boom")])
    assert r["failed"] == "job j1 ERROR: boom" and r["jobs"] == [] and launched == []


def test_a_later_job_supersedes_an_older_error(step_world):
    """Both sides of the ERROR band: only the LATEST job of a stage decides, so a COMPLETED re-run after
    an owner-fixed error lets the instance advance again."""
    run, launched = step_world
    r = run([_job("extract", "ERROR", created_at=1), _job("extract", "COMPLETED", created_at=2, jid="j2")])
    assert r["failed"] is None and r["jobs"] == ["new"] and launched == ["spec"]


def test_a_running_job_label_holds_its_stage(step_world):
    """A non-terminal job's ``stage`` label marks that stage running, so ``step`` waits (the stage barrier)."""
    run, launched = step_world
    r = run([_job("extract", "RUNNING")])
    assert r["action"] == "waiting" and launched == []


# ---- failed-unit retry (ADR-128: a transient MemoryError was never retried, Cycle 1) -------------------


def _hist(name, units):
    return {name: {"cost": UnitCost(60, 70, 80, 2**30, 40), "efficiency": 1.0, "per_unit": {k: 60.0 for k in units}}}


def test_a_failed_unit_below_the_attempt_limit_is_rescheduled_with_the_pending_ones():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=30, units=units, failed={"u001": 1})
    st["pending_units"] = units[30:]
    acts = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())
    parts = acts[0].partition
    assert acts[0].kind == "run" and parts is not None
    assert sorted(k for part in parts for k in part) == sorted(["u001", *units[30:]])


def test_a_stage_with_only_a_retryable_failure_runs_it_instead_of_reducing():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 1})
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "run"
    assert [k for part in acts[0].partition or () for k in part] == ["u007"]


def test_an_exhausted_failure_halts_a_strict_stage_and_names_the_units():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2})
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "halt" and acts[0].stages == ("extract",)
    assert "u007" in (acts[0].reason or "") and "2" in (acts[0].reason or "")


def test_an_exhausted_failure_waits_for_the_rest_of_the_stage_first():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=30, units=units, failed={"u001": 2})
    st["pending_units"] = units[31:]
    acts = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "run" and "u001" not in {k for part in acts[0].partition or () for k in part}


def test_step_reports_a_halt_and_launches_nothing(step_world, monkeypatch):
    """Through `step` itself: a halt launches no job and reports the spent units (the instance settles)."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2})
    st["pending_units"] = []
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([st]))
    monkeypatch.setattr(C, "committed_history", lambda: _hist("extract", units))
    r = run([])
    assert r["action"] == "halt" and r["jobs"] == [] and launched == []
    assert "u007" in (r["failed"] or "")


def test_a_tolerant_stage_reduces_without_its_spent_units_and_names_them():
    """A stage that declares `max_failures=None` (the legacy "record failures and fit the rest") reduces
    once its failures are retried and spent; the action names the dropped units (Reviewer B
    CCHC-IMPL-06, owner ruling 2026-10-20)."""
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2}, max_failures=None)
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "reduce" and "u007" in (acts[0].reason or "")


def test_a_stage_over_its_declared_tolerance_halts():
    units = [f"u{i:03d}" for i in range(40)]
    spent = {"u007": 2, "u008": 2}
    st = _st("extract", n=40, done=38, pending=0, units=units, failed=spent, max_failures=1)
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "halt"
    within = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2}, max_failures=1)
    within["pending_units"] = []
    acts = C.next_actions(_plan([within]), running=set(), history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "reduce"


def test_step_reports_the_units_a_tolerant_reduce_drops(step_world, monkeypatch):
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2}, max_failures=None)
    st["pending_units"] = []
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([st]))
    monkeypatch.setattr(C, "committed_history", lambda: _hist("extract", units))
    r = run([])
    assert r["action"] == "reduce" and launched == ["spec"] and "u007" in r["reason"]


# ---- memory escalation and resume after an ERROR (ADR-128: real HF, 2026-10-20) ------------------------


def _mem_st(units, *, done, failed, memory, pending=()):
    st = _st("extract", n=len(units), done=done, pending=len(pending), units=units, failed=failed)
    st["pending_units"], st["memory_units"] = list(pending), list(memory)
    return st


def test_only_memory_failures_left_escalate_each_unit_to_a_job_of_its_own_with_the_whole_box():
    units = [f"u{i:03d}" for i in range(40)]
    st = _mem_st(units, done=38, failed={"u003": 1, "u009": 1}, memory=["u003", "u009"])
    c = Constraints()
    a = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=c)[0]
    assert a.kind == "run" and a.workers == 1 and a.inner_workers == 1 and a.escalate
    assert a.cap_bytes == c.usable_bytes and a.partition == (("u003",), ("u009",)) and a.n_jobs == 2


def test_pending_and_ordinary_retries_run_before_the_memory_escalation():
    units = [f"u{i:03d}" for i in range(40)]
    st = _mem_st(units, done=30, failed={"u003": 1, "u004": 1}, memory=["u003"], pending=units[31:])
    a = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())[0]
    covered = {k for part in a.partition or () for k in part}
    assert a.workers > 1 and "u003" not in covered and covered == {"u004", *units[31:]}


def test_a_spent_memory_failure_is_not_escalated_again():
    units = [f"u{i:03d}" for i in range(40)]
    st = _mem_st(units, done=39, failed={"u003": 2}, memory=["u003"])
    assert (
        C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())[0].kind == "halt"
    )


def _world_plan(monkeypatch, st, units):
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([st]))
    monkeypatch.setattr(C, "committed_history", lambda: _hist("extract", units))


def test_an_error_with_retryable_failures_is_retried_not_halted(step_world, monkeypatch):
    """A strict stage's job ends in ERROR on any failed unit, so the retry of a failed unit must pass the
    ERROR guard; it stays bounded by `unit_attempts`."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=39, failed={"u007": 1}, memory=[]), units)
    r = run([_job("extract", "ERROR", message="Job failed with exit code: 3. Reason: Error")])
    assert r["failed"] is None and launched == ["spec"]


def test_a_timed_out_job_is_resumed_once_more_then_halted(step_world, monkeypatch):
    """HF timed out xshot sc_extended with 83 units done: they persist, so a fresh job resumes the pending
    units -- until the stage's jobs have ended in ERROR twice in a row (`Constraints.error_resumes`)."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=30, failed={}, memory=[], pending=units[30:]), units)
    one = [_job("extract", "ERROR", message="Job timeout")]
    assert run(one)["failed"] is None and launched == ["spec"]
    two = [
        _job("extract", "ERROR", created_at=1, message="Job timeout"),
        _job("extract", "ERROR", created_at=2, jid="j2", message="Job timeout"),
    ]
    r = run(two)
    assert "2 job(s) in a row ended in ERROR" in (r["failed"] or "") and launched == ["spec"]


def test_an_error_with_nothing_retryable_still_halts(step_world, monkeypatch):
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=30, failed={}, memory=[], pending=units[30:]), units)
    r = run([_job("extract", "ERROR", message="boom")])
    assert r["failed"] == "job j1 ERROR: boom" and launched == []


def test_repeated_errors_halt_even_while_a_failed_unit_looks_retryable(step_world, monkeypatch):
    """Reviewer B CCHC-IMPL-09: a timeout or a crash before any unit writes no failure record, so the
    failed unit's attempt never grows. The bound on ERRORs in a row needs no unit record."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=39, failed={"u007": 1}, memory=[]), units)
    for message in ("boom", "Job timeout"):
        launched.clear()
        assert run([_job("extract", "ERROR", message=message)])["failed"] is None and launched == ["spec"]
        many = [_job("extract", "ERROR", created_at=i, jid=f"j{i}", message=message) for i in range(50)]
        r = run(many)
        assert "in a row ended in ERROR" in (r["failed"] or "") and launched == ["spec"]  # no new launch


def test_a_completed_job_resets_the_error_count(step_world, monkeypatch):
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=39, failed={"u007": 1}, memory=[]), units)
    jobs = [
        _job("extract", "ERROR", created_at=1, jid="j1", message="boom"),
        _job("extract", "COMPLETED", created_at=2, jid="j2"),
        _job("extract", "ERROR", created_at=3, jid="j3", message="boom"),
    ]
    assert run(jobs)["failed"] is None and launched == ["spec"]


def test_a_timeout_then_a_job_that_records_a_memory_unit_still_escalates(step_world, monkeypatch):
    """Reviewer B CCHC-IMPL-12: the job that records a memory unit COMPLETES (the unit is deferred), so it
    resets the ERRORs-in-a-row count and the escalation is launched after a timeout."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=39, failed={"u007": 1}, memory=["u007"]), units)
    seen: list = []
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store: seen.append(a) or ["spec"])
    jobs = [
        _job("extract", "ERROR", created_at=1, jid="j1", message="Job timeout"),
        _job("extract", "COMPLETED", created_at=2, jid="j2"),
    ]
    r = run(jobs)
    assert r["failed"] is None and launched == ["spec"] and seen[0].escalate and seen[0].workers == 1


def test_the_escalate_flag_reaches_the_job_command():
    from typing import Any

    from scripts.hf_jobs import launch as L

    base: dict[str, Any] = dict(
        campaign="cycle1",
        run_key="rk",
        job="j",
        variant="default",
        stages=("extract",),
        shard=(0, 1),
        module="scripts.train_ghost_gk",
        args=(),
        commit="4" * 40,
        bucket="b/x",
        workers=1,
        inner_workers=1,
        cap_bytes=None,
        peak_rss_bytes=None,
        timeout_s=3600,
        attempts=1,
        datasets=(),
        extras=(),
    )
    assert "--escalate" in L._run_argv(L.JobSpec(**base, escalate=True))
    assert "--escalate" not in L._run_argv(L.JobSpec(**base))
