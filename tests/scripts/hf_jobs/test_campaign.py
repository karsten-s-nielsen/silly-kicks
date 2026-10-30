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


def test_load_campaign_reads_dispatch_and_shard_caps(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text(
        '[campaign]\nname="c"\nbucket="b/x"\ntarget_minutes=60\ndispatch_cap=25\nshard_cap=20\nattempts=3\n'
    )
    c = C.load_campaign(toml)
    assert c.dispatch_cap == 25 and c.shard_cap == 20
    assert c.constraints().dispatch_cap == 25 and c.constraints().shard_cap == 20


def test_load_campaign_shard_cap_defaults_to_dispatch_cap(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[campaign]\nname="c"\nbucket="b/x"\ntarget_minutes=60\ndispatch_cap=25\nattempts=3\n')
    c = C.load_campaign(toml)
    assert c.dispatch_cap == 25 and c.shard_cap == 25


def test_run_key_is_derived():
    c = C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=60,
        dispatch_cap=10,
        shard_cap=10,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    assert C.run_key(c, SHA, "ghost_gk", "owner") == "cycle1-abcdef1-ghost_gk-owner"
    assert C.run_key(c, SHA, "ghost_gk", "owner", replica=2) == "cycle1-abcdef1-ghost_gk-owner-r2"


def _campaign():
    return C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=60,
        dispatch_cap=10,
        shard_cap=10,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )


def test_the_planning_constraints_use_the_committed_median_startup(tmp_path, monkeypatch):
    """Spec 5.5: the planner learns job startup from the committed summaries; 300 s only before any."""
    import json

    assert C.planning_constraints(_campaign(), summary_dir=tmp_path).startup_s == Constraints().startup_s == 300.0
    (tmp_path / "cycle0.json").write_text(json.dumps({"stages": [{"startup_s": 140.0}, {"startup_s": 160.0}]}))
    c = C.planning_constraints(_campaign(), summary_dir=tmp_path)
    assert c.startup_s == 140.0 and c.target_s == 3600.0 and c.dispatch_cap == 10 and c.shard_cap == 10


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
        name="cycle1",
        bucket="b/x",
        target_minutes=60,
        dispatch_cap=10,
        shard_cap=10,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    launched: list = []

    class _Api:  # the step now lists each store once (TF-85); this instance's store is empty
        def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
            return []

    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([_st("extract", n=179)]))
    monkeypatch.setattr(C, "committed_history", dict)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "new")

    def run(jobs):
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
        return C.step(c, SHA, api=_Api(), token="hf_x")[inst.id]

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


# ---- D1: read commands refuse a dirty tree before touching the Hub (reviewer B B04) -----------


def test_main_refuses_a_dirty_tree_for_retrieve(monkeypatch, tmp_path):
    """D1/B04: `retrieve` resolves `--plan` tokens; on a dirty tree every token is `dirty-<hash>` and
    misses the committed-token stores, so `retrieve` would silently skip instances. `main` must refuse
    before it imports a token. Deleting the guard lets `retrieve` run (reviewer B mutant M4)."""
    c = C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=60,
        dispatch_cap=10,
        shard_cap=10,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    monkeypatch.setattr(C, "load_campaign", lambda p: c)
    monkeypatch.setattr(C, "_head", lambda: "c0ffee1")
    monkeypatch.setattr(C, "_tree_identity", lambda: "dirty-deadbee")
    with pytest.raises(SystemExit, match="dirty tree"):
        C.main([str(tmp_path / "x.toml"), "retrieve"])


# ---- T6: dispatch-overflow handling wired into step() (spec §D) ----


def _raise(exc):
    def _f(*a, **k):
        raise exc

    return _f


def test_a_capacity_reject_degrades_without_crashing_the_step(step_world, monkeypatch):
    from scripts.hf_jobs import launch as L

    run, _launched = step_world
    monkeypatch.setattr(L, "launch", _raise(RuntimeError("You have too many concurrent jobs running")))
    r = run([])  # no prior jobs -> profile extract is launched, but the account limit rejects it
    assert r["failed"] is None and r["jobs"] == [] and r.get("dispatch_overflow") is True


def test_a_non_capacity_reject_fails_loud(step_world, monkeypatch):
    from scripts.hf_jobs import launch as L

    run, _launched = step_world
    monkeypatch.setattr(L, "launch", _raise(RuntimeError("image pull failed: manifest unknown")))
    with pytest.raises(RuntimeError, match="manifest unknown"):
        run([])


def test_consecutive_never_started_timeouts_halt_and_surface(step_world):
    run, launched = step_world
    jobs = [
        _job("extract", "ERROR", created_at=1, jid="j1", message="Job timeout"),
        _job("extract", "ERROR", created_at=2, jid="j2", message="Job timeout"),
    ]
    for j in jobs:
        j.started_at = None  # container never started -> dispatch overflow (not a run error)
    r = run(jobs)  # cap == floor == 10 here, so two never-started timeouts trip the halt (B24)
    assert r["action"] == "halt" and "dispatch overflow" in (r["failed"] or "") and launched == []
    assert "observed_max_concurrency" in r


def test_an_ambiguous_reject_fails_loud_not_degrades(step_world, monkeypatch):
    # B48 / plan T6 (h): a quota/ambiguous reject must fail loud (no silent degrade loop)
    from scripts.hf_jobs import launch as L

    run, _launched = step_world
    monkeypatch.setattr(L, "launch", _raise(RuntimeError("Monthly quota exceeded for your account")))
    with pytest.raises(RuntimeError, match="quota exceeded"):
        run([])


def test_load_campaign_sha256_is_line_ending_invariant(tmp_path):
    # A Windows preflight (CRLF working file) must gate a Linux launch (LF): the campaign sha256 is the
    # committed/canonical content, so it hashes LF-normalized bytes, not the raw working-file bytes.
    body = '[campaign]\nname="c"\nbucket="b/x"\ntarget_minutes=60\ndispatch_cap=25\nshard_cap=20\nattempts=3\n'
    lf = tmp_path / "lf.toml"
    lf.write_bytes(body.encode())
    crlf = tmp_path / "crlf.toml"
    crlf.write_bytes(body.replace("\n", "\r\n").encode())
    assert C.load_campaign(lf).sha256 == C.load_campaign(crlf).sha256


# --- HFU T2: every action timeout is decoupled from target_s (R3) -----------------------------------------


def _tmo(st, c, history=None):
    return C.next_actions(_plan([st]), running=set(), history=history or {}, c=c)[0].timeout_s


def test_tiny_stage_timeout_is_the_bundled_work_not_a_flat_ceiling_or_the_target():
    """n_units<=2 run (spec §D, owner-ratified gold standard 2026-10-29; reviewer A D-IMPL-07): the tiny/grouped
    timeout is sized to the job's ACTUAL sequential work -- timeout_safety*(sum n_units*p99 over the bundled
    stages + inline reduce_wall) + startup, floored -- NOT a flat ceiling (which kills a heavy grouped chain >
    7200 s) and never a function of target_s (R3). First run (no committed p99): the generous profile fallback."""
    st = _st("fit", n=1)  # n_units<=2 -> the tiny-run branch
    c = Constraints()
    # first run: no committed p99 -> the profile ceiling, independent of target
    assert _tmo(st, Constraints(target_s=3600)) == _tmo(st, Constraints(target_s=1800)) == int(c.profile_timeout_s)
    # measured: sized to the work (1 unit * p99 2000 + reduce 100), independent of target, below the flat 7200 s
    cost = UnitCost(1800, 1900, 2000, 10**9, 1)
    hist = {"fit": {"cost": cost, "efficiency": None, "per_unit": None, "reduce_wall": 100.0}}
    want = int(max(c.timeout_floor_s, c.timeout_safety * (1 * 2000 + 100.0) + c.startup_s))
    assert _tmo(st, Constraints(target_s=3600), hist) == _tmo(st, Constraints(target_s=1800), hist) == want
    assert want < int(c.profile_timeout_s)  # the computed timeout is tighter than the old flat ceiling


def test_a_heavy_grouped_chain_gets_a_timeout_above_the_flat_ceiling():
    """Spec §D gold standard (reviewer A D-IMPL-07): a grouped chain of heavy single-unit stages whose sequential
    wall exceeds the flat profile ceiling (7200 s) gets a timeout sized to THAT wall, so R3 is honored. The flat
    ceiling that reviewer A's ratify-as-shipped option would keep would kill this job at 7200 s."""
    c = Constraints()
    a, b, cc = _st("a", n=1, hk="a"), _st("b", n=1, hk="b", needs=("a",)), _st("c", n=1, hk="c", needs=("b",))

    def heavy():  # a single-unit stage at ~3000 s; three chained -> a 9000 s sequential wall > 7200 s
        return {"cost": UnitCost(2900, 3000, 3000, 10**9, 1), "efficiency": None, "per_unit": None, "reduce_wall": 0.0}

    acts = C.next_actions(_plan([a, b, cc]), set(), {"a": heavy(), "b": heavy(), "c": heavy()}, c)
    run = next(x for x in acts if x.kind == "run")
    assert set(run.stages) == {"a", "b", "c"}  # the ready single-unit chain is grouped into ONE job
    assert run.timeout_s == int(max(c.timeout_floor_s, c.timeout_safety * (3 * 3000) + c.startup_s))
    assert run.timeout_s > int(c.profile_timeout_s)  # ABOVE the flat ceiling that would have killed it


def test_profile_action_timeout_is_profile_timeout_s():
    st = _st("x", n=179)  # no history -> the profile branch
    want = int(Constraints().profile_timeout_s)
    assert _tmo(st, Constraints(target_s=3600)) == _tmo(st, Constraints(target_s=1800)) == want


def test_reduce_action_timeout_independent_of_target():
    st = _st("x", n=4, done=4)  # all units done, no failures -> the reduce branch
    want = int(Constraints().profile_timeout_s)
    assert _tmo(st, Constraints(target_s=3600)) == _tmo(st, Constraints(target_s=1800)) == want


def test_escalate_timeout_independent_of_target():
    st = _st("fit", n=1, done=1, failed={"u0": 1})  # u0 failed-retryable AND over its fair share -> escalate
    st["memory_units"] = ["u0"]
    want = int(Constraints().profile_timeout_s)
    assert _tmo(st, Constraints(target_s=3600)) == _tmo(st, Constraints(target_s=1800)) == want


def test_shardable_over_budget_reduce_emits_subreduces_then_final():
    """HFU T8: a shardable stage whose measured reduce wall exceeds the per-job budget emits sub-reduce
    actions (one per shard, each carrying the shard COUNT; the worker owns the range) until all partials
    exist, then the final concat; a stage without shardable_reduce stays a single reduce even when heavy."""
    c = Constraints()  # budget = target_s - startup_s = 3600 - 300 = 3300

    def mk(shardable, present):
        s = _st("x", n=20, done=20, hk="x")  # all units done -> the reduce branch
        if shardable:
            s["shardable_reduce"] = ["game_id"]
        s["subreduce_present"] = present
        return s

    big = {"cost": UnitCost(1, 1, 1, 1, 20), "efficiency": None, "per_unit": None, "reduce_wall": 10000.0}  # > budget
    a1 = C.next_actions(_plan([mk(True, [])]), set(), {"x": big}, c)  # phase 1: no partials
    assert a1 and all(a.kind == "subreduce" for a in a1)
    assert {a.shard_id for a in a1} == set(range(len(a1)))  # one sub-reduce per shard, indexed 0..N-1
    assert all(a.n_shards == len(a1) and a.partition is None for a in a1)  # each carries the COUNT, no key range
    n = len(a1)
    a2 = C.next_actions(_plan([mk(True, list(range(n)))]), set(), {"x": big}, c)  # phase 2: all partials present
    assert len(a2) == 1 and a2[0].kind == "reduce" and a2[0].n_shards == n  # the final concat over N partials
    a3 = C.next_actions(_plan([mk(False, [])]), set(), {"x": big}, c)  # non-shardable heavy reduce
    assert len(a3) == 1 and a3[0].kind == "reduce" and a3[0].n_shards is None  # stays a single reduce


def test_non_concat_reduce_not_shardable():
    """HFU T8 S4: a stage WITHOUT shardable_reduce is never sharded; even a reduce wall far over the per-job
    budget stays ONE dedicated reduce, surfaced as an irreducible floor under R3 (a generous
    timeout_safety*wall timeout), never a halt and never silently replaced (B-03)."""
    c = Constraints()  # budget = target_s - startup_s
    st = _st("x", n=20, done=20, hk="x")  # all done -> the reduce branch; no shardable_reduce declared
    heavy = {"x": {"cost": UnitCost(1, 1, 1, 1, 20), "efficiency": None, "per_unit": None, "reduce_wall": 10000.0}}
    acts = C.next_actions(_plan([st]), set(), heavy, c)
    assert len(acts) == 1 and acts[0].kind == "reduce"  # one dedicated reduce, not sharded, not halted
    assert acts[0].n_shards is None  # never range-partitioned without the declaration
    assert acts[0].timeout_s == int(max(c.timeout_floor_s, c.timeout_safety * 10000.0))  # generous R3 floor, surfaced


def test_sharded_reduce_disables_inline_try_reduce():
    """HFU T8 S5: a sharded-reduce stage's RUN jobs carry n_shards (the disable marker), so `_job_specs` sets
    no_inline_reduce=True -- the reduce is left to the dedicated sub-reduce jobs, never also run inline."""
    c = Constraints()
    st = _st("x", n=20, done=10, hk="x")  # pending units -> the run branch
    st["shardable_reduce"] = ["game_id"]
    over = {"x": {"cost": UnitCost(1, 1, 1, 10**9, 20), "efficiency": None, "per_unit": None, "reduce_wall": 10000.0}}
    run = next(a for a in C.next_actions(_plan([st]), set(), over, c) if a.kind == "run")
    assert run.n_shards is not None  # the run jobs are marked: skip the inline reduce

    class _Store:  # the LPT partition branch writes shard-key files; record, do not touch HF
        def upload_bytes(self, rel, data):
            return None

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk1")
    specs = C._job_specs(_campaign(), SHA, inst, _plan([st]), run, store=_Store())
    assert specs and all(s.no_inline_reduce for s in specs)  # every run job disables its inline reduce


def test_final_concat_over_budget_is_irreducible_floor():
    """HFU T8 S6: when the final ordered concat alone exceeds the per-job budget (one concat is O(rows), not
    splittable without reordering), the final reduce is SURFACED with a generous R3 timeout once every
    sub-reduce partial is present -- emitted, never a halt, never failed."""
    c = Constraints()
    st = _st("x", n=20, done=20, hk="x")
    st["shardable_reduce"] = ["game_id"]
    heavy = {"x": {"cost": UnitCost(1, 1, 1, 1, 20), "efficiency": None, "per_unit": None, "reduce_wall": 10000.0}}
    subs = C.next_actions(_plan([st]), set(), heavy, c)  # phase 1: discover the shard count
    n = len(subs)
    assert n >= 2 and all(a.kind == "subreduce" for a in subs)
    st["subreduce_present"] = list(range(n))  # phase 2: all partials present -> the final concat
    final = C.next_actions(_plan([st]), set(), heavy, c)
    assert len(final) == 1 and final[0].kind == "reduce" and final[0].n_shards == n  # surfaced, never halt
    assert final[0].timeout_s == int(max(c.timeout_floor_s, c.timeout_safety * 10000.0))  # generous R3 timeout


def test_committed_history_carries_reduce_wall(tmp_path, monkeypatch):
    """HFU T4/B-04: committed_history carries reduce_wall from the summary (not only cost/efficiency), so
    next_actions can budget the inline reduce end-to-end, not from a hand-built dict."""
    import json

    summ = {
        "stages": [
            {
                "history_key": "x",
                "unit_cost": {"p50_s": 1.0, "p95_s": 1.0, "p99_s": 1.0, "peak_rss_bytes": 1, "n_samples": 1},
                "efficiency": None,
                "reduce_wall": 2000.0,
            }
        ]
    }
    d = tmp_path / "docs"
    d.mkdir()
    (d / "c.json").write_text(json.dumps(summ))
    monkeypatch.setattr(C, "SUMMARY_DIR", d)
    assert C.committed_history()["x"]["reduce_wall"] == 2000.0


def test_run_action_timeout_budgets_the_inline_reduce():
    """HFU T4/B-19: a run job's timeout budgets the INLINE reduce -- timeout_safety*(work+reduce) when the
    reduce_wall is recorded; else max(plan_stage, profile_timeout_s), never the bare 1800 floor."""
    from scripts.hf_jobs.planner import plan_stage

    st = _st("x", n=10)
    cost = UnitCost(600, 650, 700, 10**9, 10)
    c = Constraints()
    p = plan_stage("x", st["units"], cost, inner_parallel=False, c=c)
    measured = {"x": {"cost": cost, "efficiency": None, "per_unit": None, "reduce_wall": 2000.0}}
    first = {"x": {"cost": cost, "efficiency": None, "per_unit": None}}  # no reduce_wall yet
    assert _tmo(st, c, measured) == p.timeout_s + int(c.timeout_safety * 2000.0)
    assert _tmo(st, c, first) == max(p.timeout_s, int(c.profile_timeout_s))


def test_reduce_action_timeout_uses_measured_reduce_wall():
    st = _st("x", n=4, done=4)  # all done -> the dedicated reduce action
    c = Constraints()
    rw = {"x": {"cost": UnitCost(1, 1, 1, 1, 4), "efficiency": None, "per_unit": None, "reduce_wall": 2000.0}}
    assert _tmo(st, c, rw) == int(max(c.timeout_floor_s, c.timeout_safety * 2000.0))


def test_reduce_timeout_is_floored_not_tiny():
    """Partial review B D1-IMPL-01: a tiny measured reduce_wall must NOT give a near-zero timeout
    (0.4s -> int(1.5*0.4)=0 would never-start -> dispatch back-off/halt). Floored at timeout_floor_s."""
    st = _st("x", n=4, done=4)
    c = Constraints()
    rw = {"x": {"cost": UnitCost(1, 1, 1, 1, 4), "efficiency": None, "per_unit": None, "reduce_wall": 0.4}}
    assert _tmo(st, c, rw) == int(c.timeout_floor_s)


# --- HFU T5: the DAG schedule (no linear stage barrier) ---------------------------------------------------


def test_dag_emits_an_action_for_every_ready_stage():
    """HFU T5: the DAG emits an action per READY stage (no linear barrier) -- two independent ready stages
    (cv and fit, both past a reduced extract) both launch in one step."""
    cost = {"cost": UnitCost(600, 650, 700, 10**9, 10), "efficiency": None, "per_unit": None}
    acts = C.next_actions(
        _plan([_st("cv", n=10, hk="cv"), _st("fit", n=10, hk="fit")]), set(), {"cv": cost, "fit": cost}, Constraints()
    )
    assert {s for a in acts for s in a.stages} == {"cv", "fit"}


def test_dag_skips_running_and_unready_but_continues_the_walk():
    """A running stage is not relaunched and an unready stage is not launched, yet the DAG walk CONTINUES to
    a later independent ready stage -- the pre-HFU barrier would have returned nothing."""
    cost = {"cost": UnitCost(600, 650, 700, 10**9, 10), "efficiency": None, "per_unit": None}
    plan = _plan([_st("a", n=10, hk="a"), _st("b", n=10, hk="b", ready=False), _st("c", n=10, hk="c")])
    acts = C.next_actions(plan, running={"a"}, history={"a": cost, "c": cost}, c=Constraints())
    names = {s for act in acts for s in act.stages}
    assert "a" not in names and "b" not in names and "c" in names


def test_history_merge_keeps_committed_reduce_wall_when_live_is_none():
    """Partial review B D1-IMPL-04: live_history caches reduce_wall=None for a done stage whose reduce has
    not run this wave; the merge must keep the committed reduce_wall so T8's shard decision still sees it."""
    committed = {"x": {"cost": None, "efficiency": None, "per_unit": None, "reduce_wall": 2000.0}}
    live = {"x": {"cost": None, "efficiency": None, "per_unit": None, "reduce_wall": None}}
    assert C._merge_history(committed, live)["x"]["reduce_wall"] == 2000.0


def test_live_history_reads_reduce_wall_from_the_store(tmp_path):
    """Partial review B D1-IMPL-04 / T4 S2: reduce_wall flows END-TO-END from the store's {gen}/_reduce.json
    through live_history to the controller -- not just a hand-built dict."""
    from scripts._stages import LocalStore, make_context
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor
    from tests.scripts import _toy_pipeline
    from tests.scripts._stage_harness import params_from_args

    pipeline = _toy_pipeline.PIPELINE
    ctx = make_context(
        pipeline,
        params_from_args(pipeline, []),
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=None,
    )
    ex = LocalExecutor(ExecutorConfig(workers=1, poll_interval_s=0.05))
    ex.run_stage(ctx, "gen")
    assert ex.try_reduce(ctx, "gen")  # writes {gen}/_reduce.json
    gen = ctx.generation("gen")
    plan = {"stages": [{"name": "gen", "generation": gen, "history_key": "gen", "done": 12}]}
    out = C.live_history(plan, ctx.store, {})
    assert out["gen"]["reduce_wall"] is not None and out["gen"]["reduce_wall"] >= 0


def test_reduce_wall_from_store_flows_to_the_run_job_timeout(tmp_path):
    """Partial review A D-IMPL-03 / T4 S2: a {gen}/_reduce.json in the store flows through live_history INTO
    the run-job timeout = int(timeout_safety*(work_p99 + reduce_p99)) -- the LIVE producer->consumer path,
    not a hand-built history dict. A regression in the filename/key would silently drop reduce_wall."""
    import json

    from scripts._stages import LocalStore
    from scripts.hf_jobs.planner import plan_stage

    store = LocalStore(tmp_path / "s")
    gen = "g1"
    for i in range(3):  # done unit ledgers so live_history can cost the stage
        store.write_bytes(
            f"{gen}/u{i}.ledger.json",
            (json.dumps({"key": f"u{i}", "wall_s": 600.0, "outcome": "done", "peak_rss_bytes": 10**9}) + "\n").encode(),
        )
    store.write_bytes(f"{gen}/_reduce.json", (json.dumps({"stage": "x", "wall_s": 2000.0}) + "\n").encode())
    hist = C.live_history({"stages": [{"name": "x", "generation": gen, "history_key": "x", "done": 3}]}, store, {})
    assert hist["x"]["reduce_wall"] == 2000.0
    c = Constraints()
    pend = _st("x", n=10, hk="x")
    p = plan_stage("x", pend["units"], hist["x"]["cost"], inner_parallel=False, c=c)
    act = C.next_actions(_plan([pend]), set(), {"x": hist["x"]}, c)[0]
    assert act.timeout_s == p.timeout_s + int(c.timeout_safety * 2000.0)


def test_next_actions_passes_thread_safe_to_plan_stage():
    """Partial review B D1-IMPL-06: next_actions must thread thread_safe into plan_stage -- else a thread_safe
    non-inner stage plans >1 worker in production (W=1 dead). A multi-unit stage hits the plan_stage path."""
    st = _st("fit", n=10, hk="fit")
    st["thread_safe"] = True
    hist = {"fit": {"cost": UnitCost(600, 650, 700, 10**9, 10), "efficiency": None, "per_unit": None}}
    act = C.next_actions(_plan([st]), set(), hist, Constraints())[0]
    assert act.workers == 1  # threaded through -> W=1; drop the keyword and this is _workers(peak) > 1


# --- HFU T6: per-instance halt cancel by (run_key, variant) ------------------------------------------------


def test_halt_cancels_only_this_instance_by_run_key_variant():
    """HFU T6/B-25/B-30: a halting instance cancels its OWN in-flight jobs by (run_key, variant); a sibling
    sharing job+variant on another TIER (a different run_key) survives; a terminal job is left alone."""

    class Job:
        def __init__(self, jid, rk, variant, stage):
            self.id, self.labels = jid, {"run_key": rk, "variant": variant}
            self.status = types.SimpleNamespace(stage=stage)

    jobs = [
        Job("j1", "rk-public", "default", "RUNNING"),
        Job("j2", "rk-sc", "default", "RUNNING"),  # same job+variant, OTHER tier (run_key) -> survives
        Job("j3", "rk-public", "default", "COMPLETED"),  # terminal -> not cancelled
    ]
    cancelled = []

    class Api:
        def list_jobs(self, *, labels=None):
            return jobs

        def cancel_job(self, *, job_id):
            cancelled.append(job_id)

    out = C._cancel_instance(Api(), campaign="cycle1", commit="c", run_key="rk-public", variant="default")
    assert out["cancelled"] == ["j1"] and cancelled == ["j1"]


def test_failed_cancel_retried_next_step():
    """HFU T6 S2 / reviewer B D1-IMPL-12: a cancel that raises the cancel-API error leaves the job targeted (in
    cancel_failed), so the next stateless step re-derives it from live state and retries. ONLY the cancel-API
    error (HfHubHTTPError) is caught -- a programming error would fail loud."""
    from huggingface_hub.errors import HfHubHTTPError

    class Job:
        def __init__(self, jid):
            self.id, self.labels = jid, {"run_key": "rk", "variant": "default"}
            self.status = types.SimpleNamespace(stage="RUNNING")

    class Api:
        def list_jobs(self, *, labels=None):
            return [Job("j1")]

        def cancel_job(self, *, job_id):
            raise HfHubHTTPError.__new__(HfHubHTTPError)  # the real cancel-API error type, bypassing __init__

    out = C._cancel_instance(Api(), campaign="cycle1", commit="c", run_key="rk", variant="default")
    assert out["cancelled"] == [] and [f["job"] for f in out["cancel_failed"]] == ["j1"]
    out2 = C._cancel_instance(Api(), campaign="cycle1", commit="c", run_key="rk", variant="default")
    assert [f["job"] for f in out2["cancel_failed"]] == ["j1"]  # still targeted on the next step (re-derived)


def test_watch_not_settled_while_a_cancel_is_pending():
    """HFU T6 / reviewer B D1-IMPL-12: a halted instance whose cancel-API call failed is NOT settled, so --watch
    keeps stepping until the cancel lands and never exits leaving the instance's jobs alive."""

    def halt_rep(**extra):
        return {"i1": {"action": "halt", "stages": ("cv",), "jobs": [], "failed": "halt", **extra}}

    assert C.all_settled(halt_rep(cancel_failed=[{"job": "j1"}])) is False
    assert C.all_settled(halt_rep(cancelled=["j1"], cancel_failed=[])) is True  # the cancel landed: settled


def test_step_dispatches_every_ready_stage_in_one_step(monkeypatch):
    """HFU T5 S1/S3 (partial review B-05): step() dispatches EVERY ready stage this step (the DAG), not just
    the first; the per-instance report lists concurrent siblings under `also`."""
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs import preflight

    inst = C.Instance("ghost_outfield", "owner", "default", "scripts.train_ghost_outfield", (), (), "rk1")
    plan = {"pipeline": "ghost_outfield", "stages": [_st("cv", n=10, hk="cv"), _st("fit", n=10, hk="fit")]}
    cost = {"cost": UnitCost(600, 650, 700, 10**9, 10), "efficiency": None, "per_unit": None}
    c = C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=30,
        dispatch_cap=25,
        shard_cap=20,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "_tree_identity", lambda: "tree")  # the WIP working tree is dirty; step() refuses it
    monkeypatch.setattr(C, "store_snapshots", lambda keys, **k: {inst.run_key: None})
    monkeypatch.setattr(C, "read_plans", lambda insts, snaps, **k: {inst.id: plan})
    monkeypatch.setattr(C, "instance_views", lambda insts, plans, snaps, cache: {inst.id: {}})
    monkeypatch.setattr(C, "committed_history", lambda: {"cv": cost, "fit": cost})
    monkeypatch.setattr(C, "planning_constraints", lambda c, **k: c.constraints())
    monkeypatch.setattr(launcher, "jobs_for", lambda api, *, campaign, commit: [])
    rep = C.step(c, SHA, api=object(), token="t", dry_run=True, cache={})
    entry = rep[inst.id]
    dispatched = set(entry["stages"]) | {s for e in entry.get("also", []) for s in e["stages"]}
    assert dispatched == {"cv", "fit"}  # BOTH dispatched in one step (the DAG, no linear barrier)


def test_concurrent_stage_resume_is_per_stage_independent(monkeypatch):
    """HFU T5 S3 / B-12 (partial review A D-IMPL-01): resume is keyed PER STAGE, so a prior ERROR on one
    concurrent stage reports THAT stage failed without blocking or colliding with its independent siblings --
    which still dispatch, including a grouped "a-b" tiny-stage job. A shared "-".join cursor would cross them."""
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import preflight

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), _RK)
    c = C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=30,
        dispatch_cap=10,
        shard_cap=10,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    cost = {"cost": UnitCost(600, 650, 700, 10**9, 10), "efficiency": None, "per_unit": None}
    plan = _plan([_st("cv", n=10, hk="cv"), _st("fit", n=10, hk="fit"), _st("a", n=1, hk="a"), _st("b", n=1, hk="b")])

    class _Api:
        def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
            return []

    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: plan)
    monkeypatch.setattr(C, "committed_history", lambda: {"cv": cost, "fit": cost, "a": cost, "b": cost})
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: "new")
    monkeypatch.setattr(L, "jobs_for", lambda api, **k: [_job("cv", "ERROR", message="boom")])
    rep = C.step(c, SHA, api=_Api(), token="hf_x")[inst.id]
    by_stage = {e["stages"]: e for e in [rep, *rep.get("also", [])]}
    assert by_stage[("cv",)]["failed"] == "job j1 ERROR: boom" and by_stage[("cv",)]["jobs"] == []  # cv resume-errored
    assert by_stage[("fit",)]["jobs"] == ["new"]  # independent sibling dispatched -- no cursor collision
    assert by_stage[("a", "b")]["jobs"] == ["new"]  # the grouped tiny stage keyed separately, also dispatched


def test_watch_not_settled_when_a_concurrent_sibling_has_live_jobs():
    """Partial review B D1-IMPL-05 / B-07: a DAG instance whose first reported entry is a resume-error but
    whose concurrent sibling (in `also`) launched jobs is NOT settled -- --watch must keep watching, never
    exit with jobs still running. Terminal states (done, halt, plan-error) still settle."""
    live_sibling = {
        "i": {
            "action": "run",
            "stages": ("cv",),
            "jobs": [],
            "failed": "cv resume error",
            "also": [{"action": "run", "stages": ("fit",), "jobs": ["j1"], "failed": None}],
        }
    }
    assert C.all_settled(live_sibling) is False
    assert C.all_settled({"i": {"action": "done", "jobs": [], "failed": None}}) is True
    assert C.all_settled({"i": {"action": "halt", "jobs": [], "failed": "over max_failures"}}) is True
    assert C.all_settled({"i": {"action": None, "jobs": [], "failed": "--plan failed"}}) is True


def test_planner_uses_committed_median_startup(tmp_path):
    """HFU T10 S1: with a committed summary present, planning_constraints' startup_s is the committed median,
    not the 300 s Constraints default -- so the per-job budget (target_s - startup_s) reflects the real box."""
    import json as _json

    (tmp_path / "s.json").write_text(
        _json.dumps({"stages": [{"stage": "extract", "startup_s": 42.0}]}), encoding="utf-8"
    )
    cons = C.planning_constraints(_campaign(), summary_dir=tmp_path)
    assert cons.startup_s == 42.0  # learned from the committed summary, not the default
    assert C.Constraints().startup_s != 42.0  # guard: 42.0 is not the default, so the assertion has teeth


# ---- HFU T12: per-stage utilization audit (each stage saturates the vCPU or records an irreducible floor) --


def test_audit_artifact_saturates_or_only_a_structural_floor_is_ok():
    """HFU T12 S3 (reviewer B D1-IMPL-15): a stage is ok when it SATURATES the vCPU OR its idle cores are a
    STRUCTURAL floor -- an inner pool AT its declared fold width, or a wall floor on a LONE unit. An inner pool
    UNDER its fold width (2-of-5) is NOT a floor, and a wall floor on a MULTI-unit stage is NOT a floor (those
    units must fan). A pinned stage under the vCPU with no floor is flagged (R1)."""
    v = 8
    pinned = C.audit_utilization({"workers": 8, "inner_workers": 1, "thread_budget": 1, "irreducible_floor_s": None}, v)
    assert pinned["cores_used"] == 8 and pinned["saturated"] and pinned["ok"]
    thread = C.audit_utilization({"workers": 1, "inner_workers": 1, "thread_budget": 8, "irreducible_floor_s": None}, v)
    assert thread["cores_used"] == 8 and thread["saturated"] and thread["ok"]
    at_width = C.audit_utilization(
        {"workers": 1, "inner_workers": 5, "inner_width": 5, "thread_budget": 1, "irreducible_floor_s": None}, v
    )
    assert not at_width["saturated"] and at_width["idle_cores"] == 3 and at_width["ok"]  # fold width -> the floor
    under_width = C.audit_utilization(
        {"workers": 1, "inner_workers": 2, "inner_width": 5, "thread_budget": 1, "irreducible_floor_s": None}, v
    )
    assert not under_width["ok"]  # 2-of-5: an UNDER-sized pool is NOT a floor (D1-IMPL-15)
    lone_wall = C.audit_utilization(
        {"workers": 1, "inner_workers": 1, "thread_budget": 1, "n_units": 1, "irreducible_floor_s": 99.0}, v
    )
    assert not lone_wall["saturated"] and lone_wall["ok"]  # a single over-budget unit cannot fan -> the floor
    multi_wall = C.audit_utilization(
        {"workers": 1, "inner_workers": 1, "thread_budget": 1, "n_units": 20, "irreducible_floor_s": 99.0}, v
    )
    assert not multi_wall["ok"]  # a wall floor on 20 units must fan to saturate -> NOT a floor (D1-IMPL-15)
    under = C.audit_utilization({"workers": 2, "inner_workers": 1, "thread_budget": 1, "irreducible_floor_s": None}, v)
    assert not under["ok"]  # 2 cores, no floor, not inner_parallel -> flagged as under-utilized (R1)


# ---- HFU T7 S3/S4: the per-instance fair share caps TOTAL launches (reviewer B D1-IMPL-13) ----------------


def _share_world(monkeypatch, insts, plan_by_rk, *, dispatch_cap):
    """A `step` harness where `_job_specs` offers MANY specs per action, so the only thing that caps launches
    is the per-instance share / inst_budget. Each launched spec is tagged by its instance run_key."""
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import preflight

    c = C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=60,
        dispatch_cap=dispatch_cap,
        shard_cap=dispatch_cap,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    launched: list[str] = []
    hist = {k: {"cost": UnitCost(600, 600, 600, 10**9, 20), "efficiency": None, "per_unit": None} for k in ("x", "y")}

    class _Api:
        def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
            return []

    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: insts)
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: plan_by_rk[inst.run_key])
    monkeypatch.setattr(C, "committed_history", lambda: hist)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(  # 100 specs per action: more than any share, so the share/inst_budget is the cap
        C, "_job_specs", lambda c, commit, inst, plan, a, *, store: [f"{inst.run_key}|{i}" for i in range(100)]
    )
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "new")
    monkeypatch.setattr(L, "jobs_for", lambda api, **k: [])
    C.step(c, SHA, api=_Api(), token="hf_x")
    return launched


def test_two_instances_split_free_slots_fairly(monkeypatch):
    """HFU T7 S3 (reviewer B D1-IMPL-13): with two active instances and a dispatch budget B, neither instance's
    step launches more than its fair share max(1, B // 2) -- the first instance cannot take every free slot."""
    i1 = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk-1")
    i2 = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk-2")
    plans = {"rk-1": _plan([_st("x", n=179, hk="x")]), "rk-2": _plan([_st("x", n=179, hk="x")])}
    launched = _share_world(monkeypatch, [i1, i2], plans, dispatch_cap=10)
    per_inst = {rk: sum(1 for s in launched if s.startswith(rk + "|")) for rk in ("rk-1", "rk-2")}
    assert per_inst["rk-1"] <= 5 and per_inst["rk-2"] <= 5  # neither beyond the B//2 = 5 share
    assert sum(per_inst.values()) <= 10  # the whole budget, never exceeded


def test_two_ready_stages_share_one_instance_allowance(monkeypatch):
    """HFU T7 S4 (reviewer B D1-IMPL-13): an instance with TWO ready stages does not let each stage fan into the
    full share -- the two stages together stay within the instance's share (inst_budget is shared across them)."""
    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk-1")
    plan = _plan([_st("x", n=179, hk="x"), _st("y", n=179, hk="y")])  # two independent ready multi-unit stages
    launched = _share_world(monkeypatch, [inst], {"rk-1": plan}, dispatch_cap=10)
    assert len(launched) <= 10  # both stages TOGETHER within the single instance's share, not 10 per stage
    assert len(launched) >= 1  # non-vacuity: it really launched


def test_campaign_constraints_reprofiled_startup_lowers_budget_floor():
    """Reviewer B D1-IMPL-10 round 2: the PRODUCTION path. Campaign.constraints() builds Constraints() (startup
    300) then dataclasses.replace(startup_s=15); the effective floor is read live through budget_floor_s, so the
    re-profiled startup gives 10 x 15 = 150, NOT the pre-profile 3000 a stored-derived field would have pinned."""
    c = C.Campaign(
        name="cycle1",
        bucket="b/x",
        target_minutes=30,
        dispatch_cap=25,
        shard_cap=20,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    assert c.constraints().budget_floor_s == 3000.0  # before any measurement: the 300 s default
    assert c.constraints(startup_s=15.0).budget_floor_s == 150.0  # re-profiled: tracks through replace()
