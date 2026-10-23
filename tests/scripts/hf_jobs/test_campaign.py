"""Stateless campaign controller (spec 5.5): pure next-action logic over --plan JSON + job labels."""

from __future__ import annotations

import types
from typing import Any

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


def test_main_stores_are_content_addressed_and_replica_and_canary_stay_fresh():
    c = C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )
    assert C.run_key(c, SHA, "ghost_gk", "owner") == "ghost_gk-owner"  # content-addressed: no campaign/commit
    assert C.run_key(c, SHA, "ghost_gk", "owner", replica=1) == "cycle1-abcdef1-ghost_gk-owner-r1"
    assert C.run_key(c, SHA, "ghost_gk", "owner", canary=True) == "cycle1-abcdef1-ghost_gk-owner-canary"


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
    acts = C.next_actions(_plan([_st("extract", n=179)]), {}, history={}, c=Constraints())
    assert [a.kind for a in acts] == ["profile"] and acts[0].stages == ("extract",)


def test_a_stage_with_history_is_fanned_by_the_planner():
    hist = {"extract": {"cost": UnitCost(600, 700, 800, 6 * 2**30, 20), "efficiency": 0.9}}
    acts = C.next_actions(_plan([_st("extract", n=179)]), {}, history=hist, c=Constraints())
    assert acts[0].kind == "run" and acts[0].n_jobs > 1 and acts[0].workers == 3


def test_a_resumed_stage_hands_jobs_only_its_pending_units():
    """Real HF Cycle 1 (xshot public/default extract): 16 of 17 done, and the LPT partition `0of1.json`
    held one key that was ALREADY done (the history's per-unit walls are the done units'). Every relaunch
    skipped it in ~20 s and the pending unit never ran. A partition covers exactly the pending units."""
    units = [f"m{i:02d}" for i in range(17)]
    done = units[:16]
    per_unit = {k: 30.0 + i for i, k in enumerate(done)}
    hist = {"extract": {"cost": UnitCost(30, 40, 50, 2**30, 16), "efficiency": 1.0, "per_unit": per_unit}}
    acts = C.next_actions(_plan([_st("extract", n=17, done=16, units=units)]), {}, history=hist, c=Constraints())
    assert acts[0].kind == "run"
    covered = sorted(k for part in (acts[0].partition or ()) for k in part)
    assert covered == ["m16"], acts[0].partition


def test_a_partition_never_names_a_unit_outside_the_pending_set():
    units = [f"m{i:03d}" for i in range(179)]
    per_unit = {k: 600.0 for k in units[:40]}
    hist = {"extract": {"cost": UnitCost(600, 700, 800, 2**30, 40), "efficiency": 1.0, "per_unit": per_unit}}
    acts = C.next_actions(_plan([_st("extract", n=179, done=40, units=units)]), {}, history=hist, c=Constraints())
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
        _plan([_st("extract", n=179, done=40, units=units)]), {}, history=hist, c=Constraints(), lpt=False
    )
    assert acts[0].partition is None and acts[0].n_jobs > 1


def test_a_running_stage_is_not_relaunched():
    owned = {("extract", "t_extract"): C.Ownership(True, frozenset())}  # a running action owns the whole stage
    acts = C.next_actions(_plan([_st("extract", n=179)]), owned, history={}, c=Constraints())
    assert acts == []


def test_the_barrier_waits_for_upstream():
    p = _plan([_st("extract", done=4, reduced=True), _st("cv", ready=True), _st("fit", n=1, ready=False)])
    hist = {"cv": {"cost": UnitCost(60, 60, 60, 2**30, 5), "efficiency": 1.0}}
    acts = C.next_actions(p, {}, history=hist, c=Constraints())
    assert [a.stages for a in acts] == [("cv",)]


def test_consecutive_single_unit_stages_run_in_one_job():
    p = _plan([_st("extract", done=4, reduced=True), _st("cache", n=1), _st("fit", n=1, ready=False)])
    acts = C.next_actions(p, {}, history={}, c=Constraints())
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
    acts = C.next_actions(p, {}, history={}, c=Constraints())
    assert acts[0].stages == ("fit",)  # an inner_parallel stage never joins a group (it owns the pool)
    p2 = _plan(
        [
            _st("cv", n=4, done=4, reduced=True),
            _st("fit", n=1, needs=("cv",)),
            _st("assemble", n=1, ready=False, needs=("fit",)),
        ]
    )
    assert C.next_actions(p2, {}, history={}, c=Constraints())[0].stages == ("fit", "assemble")


def test_a_finished_but_unreduced_output_stage_gets_a_reduce_job():
    p = _plan([_st("assemble", n=1, done=1, reduced=False)])
    assert C.next_actions(p, {}, history={}, c=Constraints())[0].kind == "reduce"


def test_everything_reduced_means_done():
    plan = _plan([_st("a", done=4, reduced=True)])
    assert C.next_actions(plan, {}, history={}, c=Constraints()) == []


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
    # ADR-128 Cycle 1b: ownership and the action/prediction records go through the store; stub those seams so
    # the fixture needs no live bucket. A test passes `owned` to exercise the action-record barrier (S5.3).
    monkeypatch.setattr(C, "read_ownership", lambda store, plan, jobs: {})
    monkeypatch.setattr(L, "jobs_for_run_keys", lambda api, run_keys: [])
    monkeypatch.setattr(C, "_write_action_record", lambda *a, **k: None)
    monkeypatch.setattr(C, "_append_prediction", lambda *a, **k: None)
    monkeypatch.setattr(C, "_append_campaign_prediction", lambda *a, **k: None)
    monkeypatch.setattr(C, "_make_infra_checker", lambda store, jobs: lambda job: False)  # not an infra failure
    monkeypatch.setattr(C, "_newest_operator", lambda store, p, s, t: None)

    def run(jobs, owned=None):
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
        monkeypatch.setattr(L, "jobs_for_run_keys", lambda api, run_keys: jobs)
        if owned is not None:
            monkeypatch.setattr(C, "read_ownership", lambda store, plan, jobs: owned)
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


def test_a_running_action_owns_its_stage_and_step_waits(step_world):
    """S5.3: an action record owns the whole stage while its job runs, so `step` waits (the stage barrier is
    now the ownership record read from the store, not a job label)."""
    run, launched = step_world
    owned = {("extract", "t_extract"): C.Ownership(True, frozenset())}
    r = run([_job("extract", "RUNNING")], owned=owned)
    assert r["action"] == "waiting" and launched == []
    assert r["running"] is True  # CCTP-IMPL-14: a still-running instance is not settled for --watch


# ---- failed-unit retry (ADR-128: a transient MemoryError was never retried, Cycle 1) -------------------


def _hist(name, units):
    return {name: {"cost": UnitCost(60, 70, 80, 2**30, 40), "efficiency": 1.0, "per_unit": {k: 60.0 for k in units}}}


def test_a_failed_unit_below_the_attempt_limit_is_rescheduled_with_the_pending_ones():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=30, units=units, failed={"u001": 1})
    st["pending_units"] = units[30:]
    acts = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())
    parts = acts[0].partition
    assert acts[0].kind == "run" and parts is not None
    assert sorted(k for part in parts for k in part) == sorted(["u001", *units[30:]])


def test_a_stage_with_only_a_retryable_failure_runs_it_instead_of_reducing():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 1})
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "run"
    assert [k for part in acts[0].partition or () for k in part] == ["u007"]


def test_an_exhausted_failure_halts_a_strict_stage_and_names_the_units():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2})
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "halt" and acts[0].stages == ("extract",)
    assert "u007" in (acts[0].reason or "") and "2" in (acts[0].reason or "")


def test_an_exhausted_failure_waits_for_the_rest_of_the_stage_first():
    units = [f"u{i:03d}" for i in range(40)]
    st = _st("extract", n=40, done=30, units=units, failed={"u001": 2})
    st["pending_units"] = units[31:]
    acts = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())
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
    acts = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "reduce" and "u007" in (acts[0].reason or "")


def test_a_stage_over_its_declared_tolerance_halts():
    units = [f"u{i:03d}" for i in range(40)]
    spent = {"u007": 2, "u008": 2}
    st = _st("extract", n=40, done=38, pending=0, units=units, failed=spent, max_failures=1)
    st["pending_units"] = []
    acts = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())
    assert acts[0].kind == "halt"
    within = _st("extract", n=40, done=39, pending=0, units=units, failed={"u007": 2}, max_failures=1)
    within["pending_units"] = []
    acts = C.next_actions(_plan([within]), {}, history=_hist("extract", units), c=Constraints())
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
    a = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=c)[0]
    assert a.kind == "run" and a.workers == 1 and a.inner_workers == 1 and a.escalate
    assert a.cap_bytes == c.usable_bytes and a.partition == (("u003",), ("u009",)) and a.n_jobs == 2


def test_pending_and_ordinary_retries_run_before_the_memory_escalation():
    units = [f"u{i:03d}" for i in range(40)]
    st = _mem_st(units, done=30, failed={"u003": 1, "u004": 1}, memory=["u003"], pending=units[31:])
    a = C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())[0]
    covered = {k for part in a.partition or () for k in part}
    assert a.workers > 1 and "u003" not in covered and covered == {"u004", *units[31:]}


def test_a_spent_memory_failure_is_not_escalated_again():
    units = [f"u{i:03d}" for i in range(40)]
    st = _mem_st(units, done=39, failed={"u003": 2}, memory=["u003"])
    assert C.next_actions(_plan([st]), {}, history=_hist("extract", units), c=Constraints())[0].kind == "halt"


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
    )
    assert "--escalate" in L._run_argv(L.JobSpec(**base, escalate=True))
    assert "--escalate" not in L._run_argv(L.JobSpec(**base))


def test_read_plan_strips_an_operator_shell_switch_from_the_plan_subprocess(monkeypatch):
    """CCTP-IMPL-26: read_plan runs `--plan` under controller_plan_env, so an output switch in the operator's
    shell never reaches the token-computing subprocess -- it cannot move the --plan token away from the job's."""
    monkeypatch.setenv("SILLY_KICKS_ASSERT_INVARIANTS", "1")
    captured: dict = {}

    def fake_run(argv, **kw):
        captured.update(kw)
        return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")

    inst = C.Instance(
        job="ghost_gk",
        tier="owner",
        variant="default",
        module="scripts.train_ghost_gk",
        args=(),
        datasets=(),
        run_key="ghost_gk-owner",
    )
    C.read_plan(inst, bucket="b/x", run=fake_run)
    assert "env" in captured and "SILLY_KICKS_ASSERT_INVARIANTS" not in captured["env"]


def test_job_env_refuses_an_output_switch_the_controller_cannot_mirror():
    """CCTP-IMPL-27: the controller strips every output switch from its --plan token, so the job's declared
    environment must not set one (OPEN_DATA_REVISION aside) -- else the two tokens diverge and a generation
    never matches. job_env sets none, and it refuses an output switch injected through extra_env."""

    from scripts.hf_jobs import env_contract as ec
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs.closures import ClosureViolationError

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
    )
    env = L.job_env(L.JobSpec(**base))
    assert not ({k for k in ec.OUTPUT if k != "SILLY_KICKS_OPEN_DATA_REVISION"} & set(env))
    with pytest.raises(ClosureViolationError, match="SILLY_KICKS_ASSERT_INVARIANTS"):
        L.job_env(L.JobSpec(**base, extra_env={"SILLY_KICKS_ASSERT_INVARIANTS": "1"}))


# ---- DAG-parallel actions, ownership and partial grants (S5.3) ----------------------------------
_NOONE: dict = {}


def test_independent_ready_stages_get_actions_in_the_same_step():
    """S5.3: a -> {b, c}; {a, c} -> d. With a reduced, b and c both act; d waits for both."""
    p = _plan(
        [
            _st("a", reduced=True),
            _st("b", needs=("a",)),
            _st("c", needs=("a",)),
            _st("d", needs=("a", "c"), ready=False),
        ]
    )
    hist = {n: {"cost": UnitCost(10, 12, 15, 2**30, 8), "efficiency": 1.0, "per_unit": None} for n in "bcd"}
    acts = C.next_actions(p, _NOONE, hist, Constraints())
    assert sorted(a.stages for a in acts) == [("b",), ("c",)]


def test_the_real_ghost_gk_graph_launches_cv_and_fit_together():
    p = _plan(
        [
            _st("cache", n=1, reduced=True),
            _st("cv", n=5, needs=("cache",)),
            _st("fit", n=1, needs=("cache",)),
            _st("importance", n=1, needs=("cache", "fit"), ready=False, inner=True),
        ]
    )
    hist = {"cv": {"cost": UnitCost(400, 500, 600, 2**30, 5), "efficiency": 1.0, "per_unit": None}}
    acts = C.next_actions(p, _NOONE, hist, Constraints())
    assert {a.stages[0] for a in acts} == {"cv", "fit"}


def test_no_reduce_and_no_duplicate_while_running_actions_own_every_pending_unit():
    st = _st("x", n=40)
    hist = {"x": {"cost": UnitCost(60, 70, 80, 2**30, 40), "efficiency": 1.0, "per_unit": None}}
    owned = {("x", "t_x"): C.Ownership(False, frozenset(st["pending_units"]))}
    assert C.next_actions(_plan([st]), owned, hist, Constraints()) == []
    done = _st("x", n=40, done=40)
    acts = C.next_actions(_plan([done]), {}, hist, Constraints())
    assert [a.kind for a in acts] == ["reduce"]


def test_a_second_grant_with_the_same_n_writes_under_a_new_action_id():
    a = C.partition_rel("g", "x", "t", "20261021T143000Z_aaaaaaaa", 0, 2)
    b = C.partition_rel("g", "x", "t", "20261021T143100Z_bbbbbbbb", 0, 2)
    assert a != b and a.startswith("_partitions/g/x/t/20261021T143000Z_aaaaaaaa/")


def test_a_round_robin_action_never_runs_partially():
    a = C.Action("run", ("x",), 4, 8, 1, 1800, owns_all=True)
    assert C.allocate([("i1", a)], budget=3) == []
    assert [k for _, _, k in C.allocate([("i1", a)], budget=4)] == [4]


def test_a_running_job_maps_to_its_action_through_its_label_even_from_another_campaign():
    jobs = [
        types.SimpleNamespace(
            id="j1",
            labels={"run_key": "g-o", "action": "A", "campaign": "other"},
            status=types.SimpleNamespace(stage="RUNNING"),
        ),
        types.SimpleNamespace(
            id="j2",
            labels={"run_key": "g-o", "action": "B", "campaign": "cycle1b"},
            status=types.SimpleNamespace(stage="COMPLETED"),
        ),
    ]
    records = [
        {"pipeline": "g", "stage": "x", "token": "t", "keys": ["u1"], "owns_all": False, "action_id": "A"},
        {"pipeline": "g", "stage": "x", "token": "t", "keys": ["u2"], "owns_all": False, "action_id": "B"},
    ]
    owned = C.read_ownership_from_records(C.mark_running(records, jobs))
    assert owned[("x", "t")].keys == frozenset({"u1"})


def test_a_running_profile_owns_its_stage():
    owned = {("x", "t_x"): C.Ownership(True, frozenset())}
    assert C.next_actions(_plan([_st("x", n=40)]), owned, {}, Constraints()) == []


def test_action_ids_match_the_label_pattern_and_differ_for_same_second_actions():
    import datetime

    from scripts.hf_jobs.launch import LABEL_MAX, LABEL_PATTERN

    now = datetime.datetime(2026, 10, 21, 14, 30, tzinfo=datetime.timezone.utc)
    a = C.make_action_id("rk/default", ("cv",), ("t1",), None, now)
    b = C.make_action_id("rk/default", ("fit",), ("t2",), None, now)
    assert a != b and a.startswith("20261021T143000Z_") and "-" not in a
    assert LABEL_PATTERN.fullmatch(a) and len(a) <= LABEL_MAX


def test_two_variants_in_one_store_do_not_block_each_other():
    a = _st("cv", n=5)
    b = {**_st("cv", n=5), "token": "t_other"}
    owned = C.read_ownership_from_records(
        [
            {
                "pipeline": "ghost_gk",
                "stage": "cv",
                "token": "t_cv",
                "keys": a["units"],
                "owns_all": False,
                "action_id": "X",
                "running": True,
            }
        ]
    )
    assert owned.get(("cv", "t_cv")) and not owned.get(("cv", b["token"]))


# ---- critical-path scheduler (S5.5) --------------------------------------------------------------
def test_the_grant_order_is_longest_remaining_chain_first_and_deterministic():
    a, b, c3 = (C.Action("run", ("x",), 4, 8, 1, 1800) for _ in range(3))
    order = C.grant_order([("i2", 100.0, a), ("i1", 300.0, b), ("i3", 300.0, c3)])
    assert [i for i, _ in order] == ["i1", "i3", "i2"]
    assert order == C.grant_order(list(reversed([("i2", 100.0, a), ("i1", 300.0, b), ("i3", 300.0, c3)])))


def test_remaining_critical_path_follows_the_dag():
    p = _plan([_st("a", reduced=True), _st("b", needs=("a",)), _st("c", needs=("b",)), _st("d", needs=("a",))])
    hist = {n: {"cost": UnitCost(600, 600, 600, 2**30, 4), "efficiency": 1.0, "per_unit": None} for n in "bcd"}
    # CCTP-IMPL-02: the PLANNED wall (makespan), not a fluid estimate. 4 units of 600 s on 8 workers at
    # cap=1 is one wave = 600 s per stage; chain b -> c = 1200 s dominates d (600 s).
    cp_bc = C.remaining_critical_path(p, hist, Constraints(startup_s=0, concurrency_cap=1))
    assert cp_bc == pytest.approx(2 * 600, rel=0.1)  # chain b -> c dominates d


def test_allocate_grants_in_order_and_a_zero_budget_launches_nothing():
    a = C.Action("run", ("x",), 4, 8, 1, 1800)
    b = C.Action("run", ("y",), 4, 8, 1, 1800)
    ordered = [("i1", a), ("i2", b)]
    assert C.allocate(ordered, budget=0) == []
    assert [(i, k) for i, _, k in C.allocate(ordered, budget=5)] == [("i1", 4), ("i2", 1)]
    assert [(i, k) for i, _, k in C.allocate(ordered, budget=20)] == [("i1", 4), ("i2", 4)]


# ---- committed history with per-unit costs (S5.7, E-5) -------------------------------------------
def test_committed_history_carries_per_unit_costs_and_suppresses_the_profile(tmp_path, monkeypatch):
    import json

    (tmp_path / "cycle1.json").write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "history_key": "hk",
                        "unit_cost": {"p50_s": 10, "p95_s": 12, "p99_s": 15, "peak_rss_bytes": 2**30, "n_samples": 40},
                    }
                ]
            }
        )
    )
    (tmp_path / "cycle1.per_unit.json").write_text(
        json.dumps({"run_commit": "x", "run_tree_dirty": False, "per_unit": {"hk": {"u0": 99.0, "u1": 1.0}}})
    )
    monkeypatch.setattr(C, "SUMMARY_DIR", tmp_path)
    hist = C.committed_history()
    assert hist["hk"]["per_unit"] == {"u0": 99.0, "u1": 1.0}
    acts = C.next_actions(_plan([_st("x", n=40, hk="hk")]), {}, hist, Constraints())
    assert acts and acts[0].kind == "run"  # never "profile" with committed history


# ---- infra-failure resume and the operator --clear-halt (S5.8) -----------------------------------
def _ejob(stage="ERROR", msg="", created=1, label_action="A"):
    return types.SimpleNamespace(
        id=f"j{created}",
        status=types.SimpleNamespace(stage=stage, message=msg),
        created_at=created,
        labels={"action": label_action},
    )


def test_an_infra_error_resumes_twice_then_halts_with_its_cause():
    plan = _plan([_st("x")])
    c = Constraints(error_resumes=2)
    no_unit = lambda job: True  # noqa: E731 -- the job's phase record has no first_unit
    assert C.resume_after_error(plan, "x", [_ejob(created=1)], c, infra=no_unit) is None
    why = C.resume_after_error(plan, "x", [_ejob(created=2), _ejob(created=1)], c, infra=no_unit)
    assert why and "infra" in why


def test_clear_halt_resets_the_error_count_and_refuses_when_not_halted():
    jobs = [_ejob(created=3), _ejob(created=2)]
    assert C._trailing_errors(jobs, operator_after=2.5) == 1
    with pytest.raises(SystemExit, match="not halted"):
        C.clear_halt_guard(halted=False)


def test_step_gates_each_action_independently(monkeypatch):
    """CCTP-IMPL-01: a halted/ERROR stage never blocks a sibling. b has 3 ERRORs (>= error_resumes); c is
    clean. c launches; b is not relaunched."""
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import preflight

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), _RK)
    camp = C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )
    plan = _plan([_st("a", reduced=True), _st("b", n=4, needs=("a",)), _st("c", n=4, needs=("a",))])
    hist = {n: {"cost": UnitCost(10, 10, 10, 2**30, 4), "efficiency": 1.0, "per_unit": None} for n in "bc"}
    launched: list = []
    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: plan)
    monkeypatch.setattr(C, "committed_history", lambda: hist)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "read_ownership", lambda store, plan, jobs: {})
    monkeypatch.setattr(C, "_make_infra_checker", lambda store, jobs: lambda job: False)
    monkeypatch.setattr(C, "_newest_operator", lambda store, p, s, t: None)
    monkeypatch.setattr(C, "_write_action_record", lambda *a, **k: None)
    monkeypatch.setattr(C, "_append_prediction", lambda *a, **k: None)
    monkeypatch.setattr(C, "_append_campaign_prediction", lambda *a, **k: None)
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store: [a.stages])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "new")
    b_jobs = [
        _job("b", "ERROR", created_at=1, jid="j1"),
        _job("b", "ERROR", created_at=2, jid="j2"),
        _job("b", "ERROR", created_at=3, jid="j3"),
    ]
    monkeypatch.setattr(L, "jobs_for", lambda api, **k: b_jobs)
    monkeypatch.setattr(L, "jobs_for_run_keys", lambda api, run_keys: [])
    C.step(camp, SHA, api=object(), token="t")
    assert ("c",) in launched and ("b",) not in launched  # c proceeds; b stays halted


def test_clear_halt_refuses_healthy_and_unit_halt_accepts_the_error_band(monkeypatch):
    """CCTP-IMPL-03: refuse when the stage is not halted; write the operator record for a unit-cause halt."""
    import scripts.hf_jobs.store as store_mod
    from scripts.hf_jobs import launch as L

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), _RK)
    camp = C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )
    uploads: dict = {}

    class _FakeStore:
        def __init__(self, *a, **k):
            pass

        def upload_bytes(self, rel, data):
            uploads[rel] = data

    monkeypatch.setattr(store_mod, "BucketStore", _FakeStore)
    monkeypatch.setattr(C, "instances", lambda c, commit: [inst])
    # healthy stage (no errors, no failures) -> refuse
    monkeypatch.setattr(L, "jobs_for", lambda api, **k: [])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([_st("x", n=4, max_failures=0)]))
    with pytest.raises(SystemExit, match="not halted"):
        C.cmd_clear_halt(camp, SHA, f"{inst.id}/x", "note", api=object())
    # unit-cause halt (failed over max_failures) -> REFUSE (clearing cannot resume it, CCTP-IMPL-13)
    monkeypatch.setattr(
        C, "read_plan", lambda inst, bucket: _plan([_st("x", n=4, max_failures=1, failed={"u0": 2, "u1": 2})])
    )
    with pytest.raises(SystemExit, match="unit failure"):
        C.cmd_clear_halt(camp, SHA, f"{inst.id}/x", "note", api=object())
    # error-band halt (2 ERRORs in a row >= error_resumes) -> accept, operator record written
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([_st("x", n=4, max_failures=0)]))
    monkeypatch.setattr(
        L, "jobs_for", lambda api, **k: [_job("x", "ERROR", created_at=1), _job("x", "ERROR", created_at=2, jid="j2")]
    )
    out = C.cmd_clear_halt(camp, SHA, f"{inst.id}/x", "note", api=object())
    assert out["record"].startswith("_operator/ghost_gk/x/") and out["record"] in uploads


def test_an_operator_record_resumes_the_error_band():
    """CCTP-IMPL-03: after --clear-halt writes a record newer than the latest ERROR, the next step resumes."""
    plan = _plan([_st("x")])
    c = Constraints(error_resumes=2)
    jobs = [_ejob(created=3), _ejob(created=2)]  # 2 ERRORs in a row -> halted
    assert C.resume_after_error(plan, "x", jobs, c, infra=lambda j: False) is not None
    assert C.resume_after_error(plan, "x", jobs, c, infra=lambda j: False, operator_after=3.5) is None


# ---- S7.2 acceptance bars (Task 20) --------------------------------------------------------------


def _bar_inputs(scale: float = 1.0) -> dict[str, Any]:
    import datetime

    iso = lambda t: datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()  # noqa: E731
    jobs = [  # one unit job (action A, 1000 s) and one separate reduce job (action R, 100 s), main role
        {
            "id": "j1",
            "labels": {"action": "A", "run_key": "g-o"},
            "role": "main",
            "created_at": iso(0),
            "started_at": iso(10),
            "finished_at": iso(1010 * scale),
            "wall_s": 1000.0 * scale,
        },
        {
            "id": "j2",
            "labels": {"action": "R", "run_key": "g-o"},
            "role": "main",
            "created_at": iso(1010 * scale),
            "started_at": iso(1020 * scale),
            "finished_at": iso(1120 * scale),
            "wall_s": 100.0,
        },
    ]
    preds = [
        {"kind": "campaign", "makespan_s": 1120.0},
        {"action_id": "A", "predicted_wall_s": 1000.0},
        {"action_id": "R", "predicted_wall_s": 100.0, "predicted_reduce_wall_s": 90.0},
    ]
    stages = [
        {
            "instance": "g-o/default",
            "stage": "x",
            "needs": [],
            "per_unit": {"u0": 1000.0},
            "unit_cost": {"p50_s": 1000.0, "p95_s": 1000.0, "p99_s": 1000.0, "peak_rss_bytes": 1, "n_samples": 1},
        }
    ]
    reduces = [
        {"stage": "x", "token": "t", "wall_s": 90.0, "n_results": 2000, "in_job": False, "predicted_wall_s": 90.0}
    ]
    phases = [{"phases": {"script_start": 0.0, "first_unit": 60.0}}]
    return dict(
        jobs=jobs,
        predictions=preds,
        stages=stages,
        reduce_records=reduces,
        job_phases=phases,
        cycle1_main_job_hours=1100.0 / 3600,
        cap=20,
        declared_startup_s=60.0,
        comparisons=[True, True],
    )


def test_the_summary_computes_each_bar_from_its_records():
    bars = C.compute_bars(**_bar_inputs())
    assert set(bars) == {"schedule_efficiency", "planner_calibration", "reduce", "startup", "cost", "determinism"}
    assert all(b["pass"] for b in bars.values()), bars
    assert bars["startup"]["value"] == pytest.approx(1.0)


def test_each_bar_fails_from_the_other_side():
    slow = C.compute_bars(**_bar_inputs(scale=2.0))  # the unit job ran twice as long as planned
    assert not slow["planner_calibration"]["pass"] and not slow["cost"]["pass"]
    bad = _bar_inputs()
    bad["comparisons"] = [True, False]
    assert not C.compute_bars(**bad)["determinism"]["pass"]
    bad = _bar_inputs()
    bad["job_phases"] = [{"phases": {"script_start": 0.0, "install": 5.0, "first_unit": 200.0}}]
    assert not C.compute_bars(**bad)["startup"]["pass"]
    # No replica/canary comparisons is NOT a determinism pass (B#35): the bar cannot claim what it never measured.
    assert not C.compute_bars(**{**_bar_inputs(), "comparisons": []})["determinism"]["pass"]


def test_cycle1_denominator_excludes_the_profile_jobs(tmp_path, monkeypatch):
    # B#35 regression: the Cycle-1 bar-5 baseline drops the Cycle-1 profile jobs (CYCLE1_EXCLUDED).
    (tmp_path / "cycle1.json").write_text(
        __import__("json").dumps(
            {
                "jobs": [
                    {"id": "6ad89c3b-aaa", "wall_s": 1000.0},
                    {"id": "6ad8ccde-bbb", "wall_s": 2000.0},
                    {"id": "keep", "wall_s": 3600.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(C, "SUMMARY_DIR", tmp_path)
    assert C._cycle1_main_job_hours() == 1.0  # only the 3600 s "keep" job counts; the two profile ids dropped


def test_the_cost_bar_excludes_the_named_profile_and_cycle1_jobs():
    base = C.compute_bars(**_bar_inputs())["cost"]["value"]
    excluded = C.compute_bars(**{**_bar_inputs(), "cost_exclude_ids": {"j1"}})["cost"]["value"]
    assert excluded < base  # dropping the 1000 s unit job lowers the main-run job-hours


def test_remaining_critical_path_tolerates_a_partial_per_unit_history():
    # cycle1b run 1 crash (KeyError 'skillcorner__skillcorner-1036527'): the history measured a subset of the
    # stage's units, and this path passed the raw map to plan_stage. It must cost the rest at the p50.
    p = _plan([_st("b", n=40)])
    hist = {"b": {"cost": UnitCost(60, 70, 80, 2**30, 16), "efficiency": 1.0, "per_unit": {"0": 90.0, "1": 90.0}}}
    assert C.remaining_critical_path(p, hist, Constraints(startup_s=0, concurrency_cap=4)) > 0
