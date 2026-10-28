"""Stateless campaign controller (spec 5.5): pure next-action logic over --plan JSON + job labels."""

from __future__ import annotations

import json
import pathlib
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
    # spec r4 D18 (c): one store per (job, DATASET) -- tiers that read one dataset share it
    assert C.run_key(c, SHA, "ghost_gk", "tc3") == "cycle1-abcdef1-ghost_gk-tc3"
    assert C.run_key(c, SHA, "ghost_gk", "tc3", replica=2) == "cycle1-abcdef1-ghost_gk-tc3-r2"


def test_an_instance_id_names_its_tier():
    """Spec r4 D18 (c): two tiers share a run key, so the instance id carries the tier."""
    inst = C.Instance("xshot_occurrence", "public", "default", "scripts.train_xshot_occurrence", (), (), "rk")
    assert inst.id == "rk/public/default"


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

    labels = {"job": "ghost_gk", "tier": "owner", "variant": "default", "stage": "cv", "shard": "1of1", "run_key": "rk"}
    job = {"labels": labels, "created_at": iso(990), "started_at": iso(1000), "finished_at": iso(9000), "wall_s": 8000}
    recs = [
        {"key": k, "outcome": "done", "wall_s": 3400.0, "peak_rss_bytes": 2**30, "attempt": 1,
         "job_label": "ghost_gk.owner.default.cv.1of1", "started_ns": int((1000 + s) * 1e9), "worker": w,
         "inner_workers": 1}
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

    labels = {
        "job": "xshot",
        "tier": "owner",
        "variant": "paired",
        "stage": "select-fit",
        "shard": "1of1",
        "run_key": "rk",
    }
    job = {"labels": labels, "created_at": iso(990), "started_at": iso(1000), "finished_at": iso(9000), "wall_s": 8000}

    def rec(key, start):
        return {"key": key, "outcome": "done", "wall_s": 600.0, "peak_rss_bytes": 2**30, "attempt": 1,
                "job_label": "xshot.owner.paired.select-fit.1of1", "started_ns": int(start * 1e9),
                "worker": 0}  # fmt: skip

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

    labels = {
        "job": "xshot",
        "tier": "owner",
        "variant": "paired",
        "stage": "select-fit",
        "shard": "1of1",
        "run_key": "rk",
    }
    job = {"labels": labels, "created_at": iso(990), "started_at": iso(1000), "finished_at": iso(2000), "wall_s": 1000}
    fit = [
        {"key": "fit", "outcome": "done", "wall_s": 600.0, "peak_rss_bytes": 2**30, "attempt": 1,
         "job_label": "xshot.owner.paired.select-fit.1of1", "started_ns": int(1045 * 1e9), "worker": 0}
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


def _gens(stage_label: str) -> str:
    """The `gens` label of a job running ``stage_label`` over `_st` plans (token ``t_<name>``; spec r4 D17)."""
    return "-".join(f"t_{s}"[:8] for s in stage_label.split("-"))


def _job(stage, status, *, created_at=1, jid="j1", message=None):
    return types.SimpleNamespace(
        id=jid,
        labels={"run_key": _RK, "tier": "owner", "variant": "default", "stage": stage, "gens": _gens(stage)},
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
    cancelled: list = []

    class _Api:  # the step now lists each store once (TF-85); this instance's store is empty
        def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
            return []

        def cancel_job(self, *, job_id):  # a halting instance cancels its live jobs after the loop
            cancelled.append(job_id)

    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([_st("extract", n=179)]))
    monkeypatch.setattr(C, "committed_history", dict)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store, unit_peaks=None: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "new")

    def run(jobs, *, only=None):
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
        return C.step(c, SHA, api=_Api(), token="hf_x", only=only)[inst.id]

    run.cancelled = cancelled
    return run, launched


def test_a_stage_whose_latest_job_ended_in_error_with_nothing_to_resume_halts(step_world):
    """The legacy ``wait_all`` raised on a terminal ERROR. The stateless controller, finding nothing to resume
    (no retryable unit, not a timeout with pending work), halts the INSTANCE and launches nothing: a blind
    relaunch would loop on a deterministic failure."""
    run, launched = step_world
    r = run([_job("extract", "ERROR", message="boom")])
    assert r["action"] == "halt" and r["jobs"] == [] and launched == []
    assert r["failed"] == "extract: latest job j1 ended in ERROR, nothing to resume"


def test_a_profile_only_step_halts_on_a_nonresumable_profile_error(step_world):
    """HFU rung 5 / review R5-HALT-01: the instance halt on a non-resumable ERROR fires under ``only="profile"``
    too (the `profile` command) -- the halt arm's `strict` crash is a profile-phase ERROR. The stage has no
    committed history, so its action is `profile`; its latest job ERRORed with pending units and nothing to
    resume, so the instance halts and launches nothing."""
    run, launched = step_world
    r = run([_job("extract", "ERROR", message="boom")], only="profile")
    assert r["action"] == "halt" and launched == []
    assert r["failed"] == "extract: latest job j1 ended in ERROR, nothing to resume"


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
    assert r["action"] == "halt" and launched == []
    assert r["failed"] == "extract: latest job j1 ended in ERROR, nothing to resume"


def test_a_nonresumable_error_halts_the_instance_and_cancels_its_live_sibling(step_world, monkeypatch):
    """HFU rung 5 (the canary caught this): a stage whose latest job ended in ERROR with nothing to resume halts
    the INSTANCE and cancels its concurrent live sibling (the halt arm's ``hold``), so --watch settles. Before,
    ``step`` only reported the ERROR per stage and left the sibling orphaned -- on the live run its ``hold`` ran
    ~47 min, uncancelled."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=30, failed={}, memory=[], pending=units[30:]), units)
    err = _job("extract", "ERROR", message="boom")  # the extract stage's latest job
    hold = _job("hold", "RUNNING", jid="hold1")  # the concurrent sibling, same (run_key, tier, variant)
    r = run([err, hold])
    assert r["action"] == "halt"
    assert r["failed"] == "extract: latest job j1 ended in ERROR, nothing to resume"
    assert launched == [] and run.cancelled == ["hold1"]


def test_a_never_started_infra_error_resumes_not_halts(step_world, monkeypatch):
    """HFU: a job whose container NEVER started (``started_at`` is None) -- e.g. a HF
    ``Volume mount failed: MountVolume.SetUp failed`` -- is a transient infra fault, not a run error, so it
    RESUMES (relaunches) instead of halting the instance the way a deterministic crash (``started_at`` set) does."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=30, failed={}, memory=[], pending=units[30:]), units)
    j = _job("extract", "ERROR", message="Volume mount failed: MountVolume.SetUp failed for volume hf-vol-2")
    j.started_at = None  # the container never started: infra fault before the run
    r = run([j])
    assert r["failed"] is None and r["jobs"] == ["new"] and launched == ["spec"]


def test_repeated_never_started_infra_errors_halt_via_errors_exhausted(step_world, monkeypatch):
    """A PERSISTENT infra fault is still bounded: a mount failure is NOT a never-started TIMEOUT (dispatch
    overflow), so ``_trailing_errors`` counts it and two in a row halt the instance via ``error_resumes``."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=30, failed={}, memory=[], pending=units[30:]), units)
    jobs = [
        _job("extract", "ERROR", created_at=1, jid="j1", message="Volume mount failed: MountVolume.SetUp failed"),
        _job("extract", "ERROR", created_at=2, jid="j2", message="Volume mount failed: MountVolume.SetUp failed"),
    ]
    for j in jobs:
        j.started_at = None
    r = run(jobs)
    assert r["action"] == "halt" and "2 job(s) in a row ended in ERROR" in (r["failed"] or "") and launched == []


def test_a_crash_with_started_at_set_still_halts(step_world, monkeypatch):
    """INF-B-04: the never-started signal must REJECT a code crash. A deterministic crash RAN (``started_at`` set,
    a datetime) then exited non-zero, so `job_never_started` is False, `resume_after_error` returns its halt
    default, and the instance halts -- it does NOT get the infra retry."""
    import datetime as dt

    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=30, failed={}, memory=[], pending=units[30:]), units)
    j = _job("extract", "ERROR", message="Job failed with exit code: 3. Reason: Error")
    j.started_at = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)  # the container RAN, then crashed
    r = run([j])
    assert r["action"] == "halt" and launched == []
    assert r["failed"] == "extract: latest job j1 ended in ERROR, nothing to resume"


def test_shard_cap_equal_dispatch_cap_leaves_no_dispatch_backoff_range():
    """INF-B-01 (owner-accepted 2026-11-01): with `shard_cap` = `dispatch_cap` = 25, `dispatch_floor` defaults to
    25, so the dispatch back-off has NO range -- the live cap stays 25 and `dispatch_overflow_halt` fires at
    `dispatch_halt_after` (2) never-started timeouts with no slack. The documented cost of `shard_cap=25`."""
    c = Constraints(dispatch_cap=25, shard_cap=25)
    assert c.dispatch_floor == 25  # defaults to shard_cap
    assert C.effective_dispatch_cap(dispatch_cap=25, dispatch_floor=25, never_started=5) == 25  # no back-off
    m = c.dispatch_halt_after
    assert not C.dispatch_overflow_halt(never_started=m - 1, dispatch_cap=25, dispatch_floor=25, m=m)
    assert C.dispatch_overflow_halt(never_started=m, dispatch_cap=25, dispatch_floor=25, m=m)


_GEN = "ghost_gk/extract/g1"


@pytest.fixture
def guard_world(monkeypatch):
    """``step`` over ONE instance with a REAL `StoreSnapshot` over a fake bucket listing (paths relative to the
    run key), for the zero-progress guard (spec r3 D5). Returns run(stage, jobs, files) -> report, launched."""
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
    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "committed_history", dict)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store, unit_peaks=None: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "new")

    def run(stage, jobs, files):
        entries = [
            types.SimpleNamespace(path=f"{_RK}/{rel}", type="file", xet_hash=f"h{i}") for i, rel in enumerate(files)
        ]

        class _Api:
            def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
                return list(entries)

        monkeypatch.setattr(C, "read_plan", lambda inst, bucket: _plan([stage]))
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
        return C.step(c, SHA, api=_Api(), token="hf_x")[inst.id]

    return run, launched


def _done_job(jid: str, *, records: str, finished_ago_s: float, created_at: int, stage: str = "extract"):
    import datetime as dt

    return types.SimpleNamespace(
        id=jid,
        labels={
            "run_key": _RK,
            "tier": "owner",
            "variant": "default",
            "stage": stage,
            "gens": _gens(stage),
            "records": records,
        },
        status=types.SimpleNamespace(stage="COMPLETED", message=None),
        created_at=created_at,
        finished_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=finished_ago_s),
    )


def _results(n: int) -> list[str]:
    return [f"{_GEN}/u{i}{sfx}" for i in range(n) for sfx in (".parquet", ".sha256", ".ledger.json")]


def _gst(**kw):
    st = _st("extract", **kw)
    st["generation"] = _GEN
    return st


def _mark(st: dict, files: list[str]) -> str:
    """The progress mark `step` computes for stage ``st`` over the listing ``files`` (paths under the run key)."""
    names: dict[str, set] = {}
    for f in files:
        d, _, n = f.rpartition("/")
        names.setdefault(d, set()).add(n)
    return C.progress_mark_of(names, _plan([st]), [st["name"]])


def _loop(mark: str, *, n: int, newest_ago_s: float, spacing_s: float) -> list:
    """``n`` COMPLETED jobs launched at ``mark``, newest first, finished every ``spacing_s`` seconds."""
    return [
        _done_job(f"j{i}", records=mark, finished_ago_s=newest_ago_s + i * spacing_s, created_at=n - i)
        for i in range(n)
    ]


def test_zero_progress_guard_halts_after_two_jobs_with_no_new_committed_record(guard_world):
    """Spec r3 D5: two COMPLETED jobs of a stage, both launched at the stage's current progress mark, both settled,
    committed nothing -- the 2474b73 354-job loop. The instance halts, named, and launches nothing."""
    run, launched = guard_world
    st, files = _gst(n=10, done=2), _results(2)
    r = run(st, _loop(_mark(st, files), n=2, newest_ago_s=1200, spacing_s=60), files)
    assert r["action"] == "halt" and "zero progress" in (r["failed"] or "") and launched == []


def test_a_fast_no_op_loop_halts_within_the_settle_window_plus_two_jobs(guard_world):
    """B r4 R4-01: a loop relaunches every step, so its NEWEST job is never settled. The guard judges the 2 most
    recent SETTLED jobs and ignores the fresh ones, so the 2474b73 cadence (66 s, newest 60 s old, 354 jobs) halts;
    while fewer than 2 jobs are settled (the first settle window of the loop) it does not."""
    run, launched = guard_world
    st, files = _gst(n=10, done=2), _results(2)
    mark = _mark(st, files)
    r = run(st, _loop(mark, n=354, newest_ago_s=60, spacing_s=66), files)
    assert r["action"] == "halt" and "zero progress" in (r["failed"] or "") and launched == []
    young = _loop(mark, n=9, newest_ago_s=60, spacing_s=66)  # 60 .. 588 s old: none settled yet
    assert run(st, young, files)["failed"] is None


def test_zero_progress_guard_waits_for_late_records(guard_world):
    """B plan review PLAN-R3-08: a job inside the settle window may have records not yet visible in the listing,
    so it is never judged; only settled jobs are evidence."""
    run, launched = guard_world
    st, files = _gst(n=10, done=2), _results(2)
    r = run(st, _loop(_mark(st, files), n=2, newest_ago_s=60, spacing_s=30), files)
    assert r["failed"] is None and launched == ["spec"]


def test_a_settled_job_whose_records_landed_is_progress(guard_world):
    """A settled job launched at an EARLIER mark committed work (its records landed), so the stage progressed."""
    run, launched = guard_world
    st, files = _gst(n=10, done=3), _results(3)
    before = _mark(_gst(n=10, done=2), _results(2))
    r = run(st, _loop(before, n=2, newest_ago_s=1200, spacing_s=60), files)
    assert r["failed"] is None and launched == ["spec"]


def test_sub_reduce_jobs_never_trip_the_zero_progress_guard(guard_world):
    """B review SPEC-R3-01: sub-reduce jobs write only PARTIALS, which are committed records, so a sharded reduce is
    progress, never a stall."""
    run, launched = guard_world
    st = _gst(n=10, done=10)
    before = _mark(st, _results(10))
    files = [*_results(10), f"{_GEN}/_subreduce/0000.parquet", f"{_GEN}/_subreduce/0001.parquet"]
    r = run(st, _loop(before, n=2, newest_ago_s=1200, spacing_s=60), files)
    assert r["failed"] is None and launched == ["spec"]


def test_a_retry_changes_the_progress_mark_whether_it_fails_again_or_succeeds():
    """B r4 R4-02: a retry that fails again rewrites the SAME `.failed.json` (only its attempt moves), and a retry
    that succeeds swaps the failure record for a result; a record COUNT stands still in both cases. The mark
    covers names and attempts, so both are progress."""
    files = [f"{_GEN}/u0.failed.json", *_results(1)]
    first = _gst(n=2, done=1, failed={"u0": 1})
    again = _gst(n=2, done=1, failed={"u0": 2})
    assert _mark(first, files) != _mark(again, files)
    succeeded = _gst(n=2, done=2)
    assert _mark(succeeded, [f"{_GEN}/u0.parquet", *_results(1)]) != _mark(first, files)


def _two_retry_shards(state: dict, files: list[str]) -> list:
    """Two settled jobs launched in ONE step at ``state`` (one retry shard per failed unit), newest first."""
    mark = _mark(state, files)
    return [
        _done_job("j2", records=mark, finished_ago_s=1200, created_at=2),
        _done_job("j1", records=mark, finished_ago_s=1210, created_at=1),
    ]


_U2 = [f"{_GEN}/u2{sfx}" for sfx in (".parquet", ".sha256", ".ledger.json")]
_RETRY_FILES = [f"{_GEN}/u0.failed.json", f"{_GEN}/u1.failed.json", *_U2]


def _retry_state(failed: dict, *, done: int) -> dict:
    st = _gst(n=3, done=done, failed=failed, max_failures=None)
    st["pending_units"] = []
    return st


def test_a_tolerant_stage_whose_retries_fail_again_reduces_instead_of_halting(guard_world):
    """B r4 R4-02: u0 and u1 each failed once; the next step retries both, one shard each, and both fail again. The
    files are unchanged (each retry rewrote its own `.failed.json`), so a record COUNT reads both jobs as a stall
    and halts. Their attempts moved, which is progress: the tolerant stage, now spent, takes its reduce."""
    run, launched = guard_world
    jobs = _two_retry_shards(_retry_state({"u0": 1, "u1": 1}, done=1), _RETRY_FILES)
    r = run(_retry_state({"u0": 2, "u1": 2}, done=1), jobs, _RETRY_FILES)
    assert r["failed"] is None and r["action"] == "reduce" and launched == ["spec"]


def test_a_stage_whose_retries_succeed_is_progress_not_a_stall(guard_world):
    """B r4 R4-02, the twin case: both retries SUCCEED. Each swaps its failure record for a result, so a record COUNT
    stands still and reads two productive jobs as a stall. The mark covers the names: the stage reduces."""
    run, launched = guard_world
    jobs = _two_retry_shards(_retry_state({"u0": 1, "u1": 1}, done=1), _RETRY_FILES)
    now_files = [f"{_GEN}/u0.parquet", f"{_GEN}/u1.parquet", *_U2]
    r = run(_retry_state({}, done=3), jobs, now_files)
    assert r["failed"] is None and r["action"] == "reduce" and launched == ["spec"]


def test_zero_progress_guard_survives_a_controller_restart(guard_world):
    """B plan review PLAN-R3-01: the controller is stateless; the baseline rides the job labels, so a FRESH step
    (an empty cache, as after a restart) halts on the same evidence."""
    run, _ = guard_world
    st, files = _gst(n=10, done=2), _results(2)
    jobs = _loop(_mark(st, files), n=2, newest_ago_s=1200, spacing_s=60)
    for _attempt in range(2):  # each `step` call builds a fresh cache
        r = run(st, jobs, files)
        assert r["action"] == "halt" and "zero progress" in (r["failed"] or "")


_SUM_RK = "cycle1-abcdef1-ghost_outfield-owner"
_SUM_UNITS = 40


@pytest.fixture
def summary_world(monkeypatch, tmp_path):
    """`cmd_summary` over TWO variants sharing one run key, each with a 40-unit `cv` generation, over a fake bucket
    that counts download calls and downloaded files (spec r3 D7). Returns run(*, shared=False) -> (rows, counts);
    ``shared``: both variants hold ONE `cv` generation and only `default` launched jobs (spec r4 D17)."""
    import json

    import scripts._provenance as P
    from scripts.hf_jobs import launch as L

    monkeypatch.setattr(P, "require_clean_tree", lambda prov, allow_dirty: prov)
    monkeypatch.setattr(P, "git_provenance", lambda: {"commit": SHA, "dirty": False})
    monkeypatch.setattr(P, "env_provenance", lambda: {})
    monkeypatch.setattr(C, "SUMMARY_DIR", tmp_path / "summ")
    (tmp_path / "summ").mkdir()
    monkeypatch.setattr(C, "planning_constraints", lambda c: Constraints())
    insts = [
        C.Instance("ghost_outfield", "owner", v, "scripts.train_ghost_outfield", (), (), _SUM_RK)
        for v in ("default", "position_only")
    ]
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: insts)
    counts = {"calls": 0, "files": 0}

    def job(jid, variant, token, tier="owner"):
        lab = {
            "campaign": "cycle1",
            "run_key": _SUM_RK,
            "job": "ghost_outfield",
            "tier": tier,
            "variant": variant,
            "stage": "cv",
            "gens": token[:8],
            "shard": "0of1",
            "commit": SHA,
        }
        return types.SimpleNamespace(
            id=jid,
            labels=lab,
            status=types.SimpleNamespace(stage="COMPLETED", message=None),
            started_at=None,
            finished_at=None,
            created_at=None,
        )

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

    def run(*, shared: bool = False, tiers: bool = False):
        """``tiers``: two TIERS of one dataset and one variant in one store (spec r4 D18 (c)) instead of two
        variants; only the `public` tier launched jobs."""
        if tiers:
            members = [
                C.Instance("ghost_outfield", t, "default", "scripts.train_ghost_outfield", (), (), _SUM_RK)
                for t in ("public", "sc_extended")
            ]
            monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: members)
        tok = {v: ("g_shared" if shared else f"g_{v}") for v in ("default", "position_only")}

        def plan_of(inst, bucket):
            gen = f"ghost_outfield/cv/{tok[inst.variant]}"
            return _plan([{**_st("cv", n=_SUM_UNITS, done=_SUM_UNITS), "generation": gen, "token": tok[inst.variant]}])

        files = {
            f"{_SUM_RK}/ghost_outfield/cv/{tok[v]}/{name}": json.dumps(rec).encode()
            for v in ("default", "position_only")
            for name, rec in [
                *(
                    (f"u{i:03d}.ledger.json", {"key": f"u{i:03d}", "outcome": "done", "wall_s": 60.0})
                    for i in range(_SUM_UNITS)
                ),
                ("_profile.json", {"efficiency": 0.9}),
            ]
        }

        class _Api:
            def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
                p = prefix or ""
                return [
                    types.SimpleNamespace(path=k, type="file", xet_hash=f"h{abs(hash(k))}")
                    for k in files
                    if k.startswith(p)
                ]

            def download_bucket_files(self, bucket, pairs, raise_on_missing_files=False):
                counts["calls"] += 1
                for remote, dest in pairs:
                    if remote in files:
                        counts["files"] += 1
                        pathlib.Path(dest).write_bytes(files[remote])

        jobs = [job(f"d{i}", "default", tok["default"], tier="public" if tiers else "owner") for i in (1, 2, 3)]
        if not shared and not tiers:
            jobs.append(job("p1", "position_only", tok["position_only"]))
        monkeypatch.setattr(C, "read_plan", plan_of)
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
        path = C.cmd_summary(c, SHA, api=_Api())
        return {r["instance"]: r for r in json.loads(path.read_text())["stages"]}, counts

    return run


def test_summary_downloads_scale_with_stages_not_units(summary_world):
    """Spec r3 D7: the 2474b73 summary downloaded every ledger with its own call (about 70 min, 14.86 GB read).
    It reads records in bulk: download calls scale with stages, and only record files are fetched."""
    rows, counts = summary_world()
    assert len(rows) == 2  # non-vacuous: both instances summarized
    assert counts["calls"] <= 4, f"{counts['calls']} download calls for 2 stages of {_SUM_UNITS} units each"
    assert counts["files"] <= 2 * (_SUM_UNITS + 1)  # bytes bound: record files only


def test_summary_attributes_jobs_by_run_key_variant_and_stage(summary_world):
    """Spec r3 D7: two variants share a run key; each instance's row counts only its OWN variant's jobs (2474b73:
    both `ghost_outfield` variants showed 333 cv jobs)."""
    rows, _ = summary_world()
    by_variant = {k.split("/")[-1]: r for k, r in rows.items()}
    assert by_variant["default"]["n_jobs"] == 3 and by_variant["position_only"]["n_jobs"] == 1


def test_step_names_units_lost_after_a_completed_run_job(step_world, monkeypatch):
    """Spec r3 D6(c): the stage's newest job is a COMPLETED run job, yet units are pending -- what it wrote did not
    persist. The step re-runs them AND names the loss in its report."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=37, failed={}, memory=[], pending=units[37:]), units)
    r = run([_job("extract", "COMPLETED")])
    assert launched == ["spec"] and "3 unit(s) still pending after completed run job j1" in (r.get("lost") or "")


def test_a_completed_profile_job_is_not_reported_as_a_loss(step_world, monkeypatch):
    """A profile leaves units pending by design, so its completion is never reported as a storage fault."""
    run, _ = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=4, failed={}, memory=[], pending=units[4:]), units)
    job = _job("extract", "COMPLETED")
    job.labels["kind"] = "profile"
    assert "lost" not in run([job])


def test_a_timed_out_run_job_with_nothing_pending_resumes_into_its_reduce(step_world, monkeypatch):
    """Spec r3 D3: the run job that completed the stage timed out in its inline reduce, so every unit is done and
    nothing is pending. The next action is the stage's reduce, not a halt (the 2474b73 `win_probability` halt)."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=40, failed={}, memory=[]), units)
    r = run([_job("extract", "ERROR", message="Job timeout")])
    assert r["action"] == "reduce" and r["failed"] is None and launched == ["spec"]


def test_a_reduce_that_times_out_again_halts_with_a_named_reason(step_world, monkeypatch):
    """Reviewer B DIAG-05: the resume into the reduce is bounded -- a second ERROR in a row halts, named, and
    launches nothing."""
    run, launched = step_world
    units = [f"u{i:03d}" for i in range(40)]
    _world_plan(monkeypatch, _mem_st(units, done=40, failed={}, memory=[]), units)
    two = [
        _job("extract", "ERROR", created_at=1, message="Job timeout"),
        _job("extract", "ERROR", created_at=2, jid="j2", message="Job timeout"),
    ]
    r = run(two)
    assert "2 job(s) in a row ended in ERROR" in (r["failed"] or "") and launched == []


def test_watch_exit_code_is_nonzero_when_an_instance_halted():
    """Spec r3 D15: a settled `--watch` with a halted instance must not exit 0 like a finished campaign."""
    rep = {
        "a": {"action": "done", "jobs": [], "failed": None},
        "b": {"action": "reduce", "stages": ("extract",), "jobs": [], "failed": "job j1 ERROR: Job timeout"},
    }
    assert C.all_settled(rep) and C.watch_exit_code(rep) == 1


def test_watch_exit_code_is_zero_when_every_instance_is_done():
    rep = {"a": {"action": "done", "jobs": [], "failed": None}, "b": {"action": "done", "jobs": [], "failed": None}}
    assert C.all_settled(rep) and C.watch_exit_code(rep) == 0


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
    monkeypatch.setattr(
        C, "_job_specs", lambda c, commit, inst, plan, a, *, store, unit_peaks=None: seen.append(a) or ["spec"]
    )
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
    # S3 item 5: a WIO stage's job tells the executor to bound W by the planned W, not by nproc
    assert "--wio" in L._run_argv(L.JobSpec(**base, wio=True))
    assert "--wio" not in L._run_argv(L.JobSpec(**base))


def test_a_stage_with_unit_peaks_is_gated_and_its_job_reads_them():
    """S3 item 6 (MEM): with per-unit history the run action is gated at W = vcpu, and each job of it reads the
    staged per-unit peaks (`--unit-peaks`)."""
    hist = {"extract": {"cost": UnitCost(60, 90, 120, 5_510_000_000, 179), "efficiency": 1.0}}
    peaks = {"extract": {f"u{i}": 2_000_000_000 + i for i in range(20)}}
    st = _st("extract", n=20)
    run = C.next_actions(_plan([st]), set(), hist, Constraints(), unit_peaks=peaks)[0]
    assert run.kind == "run" and (run.workers, run.gated) == (8, True)
    ungated = C.next_actions(_plan([st]), set(), hist, Constraints())[0]
    assert (ungated.workers, ungated.gated) == (4, False)

    uploads: dict[str, bytes] = {}

    class _Store:
        def upload_bytes(self, rel, data):
            uploads[rel] = data

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk1")
    specs = C._job_specs(_campaign(), SHA, inst, _plan([st]), run, store=_Store(), unit_peaks=peaks)
    (rel,) = [r for r in uploads if r.startswith("_unit_peaks/")]
    assert json.loads(uploads[rel]) == peaks["extract"]
    assert specs and all(s.unit_peaks_path == f"/scratch/rk1/{rel}" for s in specs)
    from scripts.hf_jobs import launch as L

    argv = L._run_argv(specs[0])
    assert argv[argv.index("--unit-peaks") + 1] == f"/scratch/rk1/{rel}"


def test_the_gated_run_stages_its_peaks_through_the_real_bucket_store():
    """HFU rung 5 (canary a4aa128): the controller stages the per-unit peaks through `BucketStore.upload_bytes`,
    whose prefix guard refused `_unit_peaks/` on HF (PermissionError at the first gated run). The fake store of
    the test above has no guard; this one is the store the controller uses, behind its step snapshot."""
    from scripts.hf_jobs.store import BucketStore, StoreSnapshot

    hist = {"extract": {"cost": UnitCost(60, 90, 120, 5_510_000_000, 179), "efficiency": 1.0}}
    peaks = {"extract": {f"u{i}": 2_000_000_000 + i for i in range(20)}}
    st = _st("extract", n=20)
    run = C.next_actions(_plan([st]), set(), hist, Constraints(), unit_peaks=peaks)[0]
    assert run.gated

    added: list[str] = []

    class _Api:
        def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
            added.extend(path for _, path in add or ())

        def list_bucket_tree(self, bucket_id, *, prefix, recursive):
            return []

    store = StoreSnapshot(BucketStore("ns/b", "rk1", api=_Api()))
    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk1")
    specs = C._job_specs(_campaign(), SHA, inst, _plan([st]), run, store=store, unit_peaks=peaks)
    (path,) = added
    assert path.startswith("rk1/_unit_peaks/")
    assert specs and all(s.unit_peaks_path == f"/scratch/{path}" for s in specs)
    # the guard still refuses a unit result: the controller never writes one
    with pytest.raises(PermissionError, match="controller uploads are limited to"):
        BucketStore("ns/b", "rk1", api=_Api()).upload_bytes("ghost_gk/extract/tok/u0.parquet", b"x")


def test_every_other_controller_upload_passes_the_real_bucket_store_guard():
    """Review R5-FIX-01: the other controller uploads (the LPT partition of `_job_specs`, the preflight report of
    `cmd_preflight`, the preflight cache) go through the real `BucketStore` guard here too, not only a guard-less
    fake (the escape that hid the `_unit_peaks/` refusal)."""
    from scripts.hf_jobs import preflight as P
    from scripts.hf_jobs.store import BucketStore, StoreSnapshot

    added: list[str] = []

    class _Api:
        def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
            added.extend(path for _, path in add or ())

        def list_bucket_tree(self, bucket_id, *, prefix, recursive):
            return []

    store = StoreSnapshot(BucketStore("ns/b", "rk1", api=_Api()))
    units = [f"u{i:03d}" for i in range(40)]
    st = _mem_st(units, done=38, failed={"u003": 1, "u009": 1}, memory=["u003", "u009"])
    a = C.next_actions(_plan([st]), running=set(), history=_hist("extract", units), c=Constraints())[0]
    assert a.partition
    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "rk1")
    specs = C._job_specs(_campaign(), SHA, inst, _plan([st]), a, store=store)
    parts = [p for p in added if p.startswith("rk1/_partitions/")]
    assert len(parts) == len(a.partition) == len(specs)

    store.upload_bytes(P.REPORT_REL.format(campaign="canary", commit7=SHA[:7]), b"{}")  # as `cmd_preflight` does
    P._write_cache(store, {})
    assert {f"rk1/_campaign/canary/{SHA[:7]}/preflight.json", "rk1/_campaign/preflight_cache.json"} <= set(added)


def test_a_wio_plan_reaches_the_run_action():
    """S3 item 5: the planner's WIO mark and W ride on the run action (rows 33, 38: 0.71 share, W=12)."""
    hist = {"extract": {"cost": UnitCost(0.45, 0.6, 0.8, 360 * 10**6, 50, cpu_share=0.71), "efficiency": 1.0}}
    acts = C.next_actions(_plan([_st("extract", n=3961)]), running=set(), history=hist, c=Constraints())
    assert acts[0].kind == "run" and (acts[0].workers, acts[0].wio) == (12, True)


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


def _first_run(read_s: float, combine_s: float | None) -> dict:
    """A history entry of a profiled first run: a unit cost, NO measured reduce wall, the profile's reduce cost."""
    return {
        "cost": UnitCost(1, 1, 1, 10**8, 16),
        "efficiency": None,
        "per_unit": None,
        "reduce_wall": None,
        "reduce_cost": {"n": 16, "read_s_per_unit": read_s, "combine_s_per_unit": combine_s},
    }


def test_a_fresh_store_with_a_heavy_shardable_reduce_plans_the_sharded_path():
    """Spec r3 D2: on a first run no reduce has run, so no reduce wall is measured, but the profile measured the
    reduce's per-unit cost. The controller estimates the wall (units x (read / read workers + combine)) and, over
    the per-job budget, plans the sharded path from the FIRST run: the run jobs skip the inline reduce (the 2474b73
    `win_probability` `extract` reduce ran inline until the 7200 s timeout), and the sub-reduces follow."""
    from scripts._stages import REDUCE_READ_WORKERS

    c = Constraints()  # budget = 3600 - 300 = 3300 s
    hist = {"x": _first_run(8.0, 0.05)}
    est = 8000 * (8.0 / REDUCE_READ_WORKERS + 0.05)  # 4400 s, over the budget
    st = _st("x", n=8000, done=16, hk="x")  # profiled: pending units remain -> the run branch
    st["shardable_reduce"] = []
    run = next(a for a in C.next_actions(_plan([st]), set(), hist, c) if a.kind == "run")
    assert run.n_shards == -(-int(est) // 3300)  # sharded from the first run: no inline reduce
    done = _st("x", n=8000, done=8000, hk="x")
    done["shardable_reduce"] = []
    done["subreduce_present"] = []
    subs = C.next_actions(_plan([done]), set(), hist, c)
    assert subs and all(a.kind == "subreduce" and a.n_shards == run.n_shards for a in subs)
    assert all(a.timeout_s == int(max(c.timeout_floor_s, c.timeout_safety * est)) for a in subs)


def test_a_fresh_store_with_a_light_reduce_keeps_it_inline_with_an_estimated_timeout():
    """Spec r3 D2: an estimate under the budget keeps the inline reduce, and the run job's timeout carries the
    ESTIMATED reduce wall, exactly as it would carry a measured one, instead of the flat profile ceiling."""
    from scripts._stages import REDUCE_READ_WORKERS

    c = Constraints()
    st = _st("x", n=400, done=16, hk="x")
    st["shardable_reduce"] = []
    est = 400 * (1.6 / REDUCE_READ_WORKERS + 0.01)
    measured = {**_first_run(1.6, 0.01), "reduce_wall": est, "reduce_cost": None}
    run = next(a for a in C.next_actions(_plan([st]), set(), {"x": _first_run(1.6, 0.01)}, c) if a.kind == "run")
    ref = next(a for a in C.next_actions(_plan([st]), set(), {"x": measured}, c) if a.kind == "run")
    assert run.n_shards is None
    assert run.timeout_s == ref.timeout_s < int(c.profile_timeout_s)


def test_a_first_run_reduce_with_an_unmeasured_combine_keeps_the_ceiling_or_the_read_estimate():
    """Spec r3 D2: a non-shardable reduce's combine is never run on a sample (it may not accept a subset), so only
    its read is estimated. Its timeout stays the generous profile ceiling, raised to the read estimate when the
    read alone would exceed it; it is never sized below the unmeasured combine's protection."""
    from scripts._stages import REDUCE_READ_WORKERS

    c = Constraints()
    st = _st("x", n=20, done=20, hk="x")  # all done -> the reduce branch; not shardable
    small = C.next_actions(_plan([st]), set(), {"x": _first_run(1.0, None)}, c)
    assert small[0].kind == "reduce" and small[0].timeout_s == int(c.profile_timeout_s)
    big = C.next_actions(_plan([st]), set(), {"x": _first_run(4000.0, None)}, c)
    read_est = 20 * 4000.0 / REDUCE_READ_WORKERS  # 5000 s: 1.5 x 5000 > the 7200 s ceiling
    assert big[0].kind == "reduce" and big[0].timeout_s == int(c.timeout_safety * read_est)


def test_the_reduce_estimate_never_trusts_a_read_faster_than_a_cold_hf_read():
    """B r4 R4-03: the profile reads back units its OWN container just wrote, which a local write cache may serve,
    while the real reduce reads them cold in a fresh container. The per-unit read the estimate uses is never below
    the HF-measured cold read for the unit's size (per-file latency plus size over cold throughput)."""
    from scripts._stages import REDUCE_READ_WORKERS

    cached = {"reduce_cost": {"n": 16, "read_s_per_unit": 0.001, "combine_s_per_unit": 0.01, "bytes_per_unit": 2e6}}
    got = C._reduce_estimate(cached, 1000)
    assert got is not None
    est, combine_measured = got
    cold = C.COLD_READ_S_PER_UNIT + 2e6 / C.COLD_READ_BYTES_PER_S
    assert combine_measured and est == pytest.approx(1000 * (cold / REDUCE_READ_WORKERS + 0.01))
    slow = {"reduce_cost": {"n": 16, "read_s_per_unit": 5.0, "combine_s_per_unit": 0.01, "bytes_per_unit": 2e6}}
    slow_est = C._reduce_estimate(slow, 1000)
    assert slow_est is not None and slow_est[0] == pytest.approx(1000 * (5.0 / REDUCE_READ_WORKERS + 0.01))


def test_the_reduce_shard_budget_uses_the_stage_measured_startup():
    """B r4 R4-09 (spec D11): the sharded-reduce decision budgets a reduce job with the stage's own measured startup,
    like its run jobs, not the campaign-wide one. A reduce job loads no upstream, so no load term."""
    c = Constraints()  # campaign startup 300 s: budget 3300 s
    st = _st("x", n=20, done=20, hk="x")
    st["shardable_reduce"] = []
    st["subreduce_present"] = []
    entry = {"cost": UnitCost(1, 1, 1, 1, 20), "efficiency": None, "per_unit": None, "reduce_wall": 3000.0}
    whole = C.next_actions(_plan([st]), set(), {"x": entry}, c)
    assert [a.kind for a in whole] == ["reduce"]  # 3000 s fits the 3300 s budget
    slow = {**entry, "startup_s": 1000.0, "upstream_load_s": 500.0}  # budget 2600 s; the load term is not charged
    subs = C.next_actions(_plan([st]), set(), {"x": slow}, c)
    assert subs and all(a.kind == "subreduce" for a in subs)
    assert subs[0].n_shards == 2  # ceil(3000 / 2600), never ceil(3000 / 2100)


def test_live_history_refreshes_when_the_profile_record_lands_after_the_ledgers(tmp_path):
    """B r4 R4-10: a step can read a generation after its ledgers landed and before `_profile.json` did; the cached
    entry refreshes when the profile record lands, not only when a new ledger does."""
    import json

    from scripts._stages import LocalStore

    store = LocalStore(tmp_path / "s")
    gen = "g1"
    plan = {"stages": [{"name": "x", "generation": gen, "history_key": "x", "done": 1}]}
    _ledger(store, gen, "u0", 600.0)
    cache: dict = {}
    assert C.live_history(plan, store, cache)["x"]["reduce_cost"] is None
    cost = {"n": 1, "read_s_per_unit": 0.4, "combine_s_per_unit": 0.01, "bytes_per_unit": 10.0}
    store.write_bytes(f"{gen}/_profile.json", (json.dumps({"stage": "x", "reduce_cost": cost}) + "\n").encode())
    assert C.live_history(plan, store, cache)["x"]["reduce_cost"] == cost


def test_live_history_carries_the_profiled_reduce_cost(tmp_path):
    """Spec r3 D2: the profile's reduce cost reaches next_actions through live history."""
    import json

    from scripts._stages import LocalStore

    store = LocalStore(tmp_path / "s")
    gen = "g1"
    _ledger(store, gen, "u0", 600.0)
    cost = {"n": 1, "read_s_per_unit": 0.4, "combine_s_per_unit": 0.01}
    store.write_bytes(f"{gen}/_profile.json", (json.dumps({"stage": "x", "reduce_cost": cost}) + "\n").encode())
    plan = {"stages": [{"name": "x", "generation": gen, "history_key": "x", "done": 1}]}
    assert C.live_history(plan, store, {})["x"]["reduce_cost"] == cost


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
    raw = ctx.store.read_bytes(f"{gen}/_reduce.json")
    assert raw is not None
    rec = json.loads(raw)
    assert rec["cpu_s"] >= 0  # S3 item 3 (S2B-r2-04): the reduce's CPU, for the "too small to pay" class


def _ledger(store, gen: str, key: str, wall_s: float) -> None:
    import json

    rec = {"key": key, "wall_s": wall_s, "outcome": "done", "peak_rss_bytes": 10**9}
    store.write_bytes(f"{gen}/{key}.ledger.json", (json.dumps(rec) + "\n").encode())


def test_live_history_rereads_a_stage_whose_ledger_lands_after_its_result(tmp_path):
    """Spec r3 D4 (B DIAG-03): the plan counts a unit done when its result and `.sha256` exist, but `run_unit`
    writes the ledger LAST. A step inside that window must not freeze the stage's cost as None for the life of
    the controller: the next step, with the ledger landed, gets a cost from the SAME long-lived cache."""
    from scripts._stages import LocalStore

    store = LocalStore(tmp_path / "s")
    gen = "g1"
    plan = {"stages": [{"name": "x", "generation": gen, "history_key": "x", "done": 1}]}
    store.write_bytes(f"{gen}/u0.parquet", b"result")
    store.write_bytes(f"{gen}/u0.sha256", b"digest")
    cache: dict = {}
    assert "x" not in C.live_history(plan, store, cache)  # no ledger yet: no cost
    _ledger(store, gen, "u0", 600.0)
    assert C.live_history(plan, store, cache)["x"]["cost"] is not None


def test_live_history_refreshes_a_cached_cost_when_a_new_ledger_lands(tmp_path):
    """Spec r3 D4 (B review R3-08): a cached NON-None cost is replaced when the stage's ledger set grows."""
    from scripts._stages import LocalStore

    store = LocalStore(tmp_path / "s")
    gen = "g1"
    plan = {"stages": [{"name": "x", "generation": gen, "history_key": "x", "done": 1}]}
    _ledger(store, gen, "u0", 600.0)
    cache: dict = {}
    assert C.live_history(plan, store, cache)["x"]["cost"].n_samples == 1
    _ledger(store, gen, "u1", 1200.0)
    plan["stages"][0]["done"] = 2
    assert C.live_history(plan, store, cache)["x"]["cost"].n_samples == 2


def _load_ledger(key: str, wall_s: float, load_s: float) -> dict:
    """A done unit ledger with a measured upstream load (spec r3 D11 (b)); ``load_s`` 0 = the worker's copy reused."""
    return {
        "key": key,
        "wall_s": wall_s,
        "outcome": "done",
        "peak_rss_bytes": 10**9,
        "upstream_loads": 1 if load_s else 0,
        "upstream_fetch_s": load_s / 3,
        "upstream_verify_s": load_s / 6,
        "upstream_decode_s": load_s / 2,
    }


def test_live_history_costs_units_without_their_upstream_load(tmp_path):
    """Spec r3 D11 (b): the live per-unit walls (the LPT costs) and the stage cost exclude the upstream load (a
    worker's first unit carries it), and the stage's measured load travels with its history entry."""
    import json

    from scripts._stages import LocalStore

    store = LocalStore(tmp_path / "s")
    gen = "g1"
    for rec in (_load_ledger("u0", 630.0, 30.0), _load_ledger("u1", 600.0, 0.0)):
        store.write_bytes(f"{gen}/{rec['key']}.ledger.json", (json.dumps(rec) + "\n").encode())
    plan = {"stages": [{"name": "x", "generation": gen, "history_key": "x", "done": 2}]}
    h = C.live_history(plan, store, {})["x"]
    assert h["per_unit"] == pytest.approx({"u0": 600.0, "u1": 600.0})
    assert h["cost"].p50_s == pytest.approx(600.0)
    assert h["upstream_load_s"] == pytest.approx(30.0)


def test_committed_history_carries_the_stage_startup_and_upstream_load(tmp_path, monkeypatch):
    """Spec r3 D11: the stage's own measured startup and upstream load reach next_actions from the summary."""
    import json

    cost = {"p50_s": 1.0, "p95_s": 1.0, "p99_s": 1.0, "peak_rss_bytes": 1, "n_samples": 1}
    summ = {"stages": [{"history_key": "x", "unit_cost": cost, "startup_s": 40.0, "upstream_load_s": 25.0}]}
    d = tmp_path / "docs"
    d.mkdir()
    (d / "c.json").write_text(json.dumps(summ))
    monkeypatch.setattr(C, "SUMMARY_DIR", d)
    h = C.committed_history()["x"]
    assert (h["startup_s"], h["upstream_load_s"]) == (40.0, 25.0)


def test_history_merge_keeps_the_committed_overhead_when_live_has_none():
    """Spec r3 D11: live history has no HF job start to measure a startup from, and no load until a unit loads;
    the merge keeps the committed measurements instead of dropping them."""
    committed = {"x": {"cost": None, "reduce_wall": None, "startup_s": 40.0, "upstream_load_s": 25.0}}
    live = {"x": {"cost": None, "reduce_wall": None, "upstream_load_s": None}}
    merged = C._merge_history(committed, live)["x"]
    assert (merged["startup_s"], merged["upstream_load_s"]) == (40.0, 25.0)


def test_next_actions_charges_the_stage_measured_overhead(monkeypatch):
    """Spec r3 D11: the planner gets the stage's measured per-job overhead: its own measured startup (HF start to
    the first unit) plus its upstream load; the campaign-wide startup stands in only when the stage has none."""
    seen: list = []
    real = C.plan_stage

    def spy(*args, **kwargs):
        seen.append(kwargs.get("overhead_s"))
        return real(*args, **kwargs)

    monkeypatch.setattr(C, "plan_stage", spy)
    st = _st("fit", n=10, hk="fit")
    entry = {"cost": UnitCost(600, 650, 700, 10**9, 10), "efficiency": None, "per_unit": None}
    C.next_actions(_plan([st]), set(), {"fit": {**entry, "startup_s": 40.0, "upstream_load_s": 25.0}}, Constraints())
    C.next_actions(_plan([st]), set(), {"fit": {**entry, "upstream_load_s": 25.0}}, Constraints(startup_s=15.0))
    C.next_actions(_plan([st]), set(), {"fit": entry}, Constraints(startup_s=15.0))
    assert seen == [pytest.approx(65.0), pytest.approx(40.0), pytest.approx(15.0)]


def test_a_summary_row_records_the_stage_upstream_load():
    """Spec r3 D11 (b): the committed summary row carries the stage's measured upstream load (the planner's input)."""
    st = {"name": "cv", "history_key": "hk", "n_units": 2}
    recs = [_load_ledger("a", 630.0, 30.0), _load_ledger("b", 600.0, 0.0)]
    row = C.stage_summary("rk/default", st, recs, [], None, Constraints(), job_recs=recs)
    assert row["upstream_load_s"] == pytest.approx(30.0)
    assert row["unit_cost"]["p50_s"] == pytest.approx(600.0)


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


# --- HFU T6: per-instance halt cancel by (run_key, tier, variant) ------------------------------------------


def test_halt_cancels_only_this_instance_by_run_key_tier_variant():
    """HFU T6/B-25/B-30 + spec r4 D18 (c): a halting instance cancels its OWN in-flight jobs by (run_key, tier,
    variant). Two tiers that read one dataset share the run key, so a same-variant sibling on the OTHER tier
    survives, as does one in another store; a terminal job is left alone."""

    class Job:
        def __init__(self, jid, rk, tier, variant, stage):
            self.id, self.labels = jid, {"run_key": rk, "tier": tier, "variant": variant, "gens": "0a1b2c3d"}
            self.status = types.SimpleNamespace(stage=stage)

    jobs = [
        Job("j1", "rk", "public", "default", "RUNNING"),
        Job("j2", "rk", "sc_extended", "default", "RUNNING"),  # same store and variant, OTHER tier -> survives
        Job("j3", "rk", "public", "default", "COMPLETED"),  # terminal -> not cancelled
        Job("j4", "rk-other", "public", "default", "RUNNING"),  # another store -> survives
    ]
    cancelled = []

    class Api:
        def list_jobs(self, *, labels=None):
            return jobs

        def cancel_job(self, *, job_id):
            cancelled.append(job_id)

    out = C._cancel_instance(Api(), campaign="cycle1", commit="c", run_key="rk", tier="public", variant="default")
    assert out["cancelled"] == ["j1"] and cancelled == ["j1"]


def test_failed_cancel_retried_next_step():
    """HFU T6 S2 / reviewer B D1-IMPL-12: a cancel that raises the cancel-API error leaves the job targeted (in
    cancel_failed), so the next stateless step re-derives it from live state and retries. ONLY the cancel-API
    error (HfHubHTTPError) is caught -- a programming error would fail loud."""
    from huggingface_hub.errors import HfHubHTTPError

    class Job:
        def __init__(self, jid):
            labels = {"run_key": "rk", "tier": "owner", "variant": "default", "gens": "0a1b2c3d"}
            self.id, self.labels = jid, labels
            self.status = types.SimpleNamespace(stage="RUNNING")

    class Api:
        def list_jobs(self, *, labels=None):
            return [Job("j1")]

        def cancel_job(self, *, job_id):
            raise HfHubHTTPError.__new__(HfHubHTTPError)  # the real cancel-API error type, bypassing __init__

    out = C._cancel_instance(Api(), campaign="cycle1", commit="c", run_key="rk", tier="owner", variant="default")
    assert out["cancelled"] == [] and [f["job"] for f in out["cancel_failed"]] == ["j1"]
    out2 = C._cancel_instance(Api(), campaign="cycle1", commit="c", run_key="rk", tier="owner", variant="default")
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
    """HFU T5 S3 / B-12 (partial review A D-IMPL-01): resume is keyed PER STAGE, so a RESUMABLE ERROR on one
    concurrent stage resumes THAT stage without blocking or colliding with its independent siblings -- which
    still dispatch, including a grouped "a-b" tiny-stage job. A shared "-".join cursor would cross them. (A
    non-resumable ERROR halts the whole instance instead, HFU rung 5.)"""
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
    plan = _plan(
        [
            _st("cv", n=10, hk="cv", failed={"u001": 1}),
            _st("fit", n=10, hk="fit"),
            _st("a", n=1, hk="a"),
            _st("b", n=1, hk="b"),
        ]
    )

    class _Api:
        def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
            return []

    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: plan)
    monkeypatch.setattr(C, "committed_history", lambda: {"cv": cost, "fit": cost, "a": cost, "b": cost})
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store, unit_peaks=None: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: "new")
    monkeypatch.setattr(L, "jobs_for", lambda api, **k: [_job("cv", "ERROR", message="boom")])
    rep = C.step(c, SHA, api=_Api(), token="hf_x")[inst.id]
    by_stage = {e["stages"]: e for e in [rep, *rep.get("also", [])]}
    # cv's ERROR is RESUMABLE (a retryable failed unit), so cv resumes its OWN stage; a non-resumable ERROR would
    # halt the whole instance instead (test_a_nonresumable_error_halts_the_instance_and_cancels_its_live_sibling).
    assert by_stage[("cv",)]["jobs"] == ["new"] and by_stage[("cv",)]["failed"] is None  # cv resumes per stage
    assert by_stage[("fit",)]["jobs"] == ["new"]  # independent sibling dispatched -- no cursor collision
    assert by_stage[("a", "b")]["jobs"] == ["new"]  # the grouped tiny stage keyed separately, also dispatched


def test_watch_not_settled_when_a_concurrent_sibling_has_live_jobs():
    """Partial review B D1-IMPL-05 / B-07: a DAG instance whose first reported entry is a resume-error but
    whose concurrent sibling (in `also`) launched jobs is NOT settled -- --watch must keep watching, never
    exit with jobs still running. Terminal states (done, halt, plan-error) still settle.

    Since HFU rung 5 a non-resumable ERROR halts the whole instance, so `step` no longer emits a resume-error
    entry beside a live `also` sibling; this drives `all_settled` on a hand-built report and stays a defensive
    `all_settled` unit guard (review R5-HALT-02 / HALT-B-03)."""
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
    """HFU T12 S3 (reviewer B D1-IMPL-15), measured per spec r3 D8: a stage is ok when its MEASURED cores saturate the
    vCPU OR its idle cores are a STRUCTURAL floor -- an inner pool AT its declared fold width, a lone unit at the
    FULL thread budget (a final fit), or a declared serial reason. An inner pool UNDER its fold width is NOT a
    floor, and neither is a multi-unit stage over budget (those units must fan)."""
    v = 8
    procs = C.audit_utilization({"workers": 8, "inner_workers": 1, "thread_budget": 1, "measured_cores": 7.5}, v)
    assert procs["saturated"] and procs["ok"]
    fit = C.audit_utilization({"workers": 1, "thread_budget": 8, "n_units": 1, "measured_cores": 5.0}, v)
    assert not fit["saturated"] and fit["floor"] and fit["ok"]  # a lone 8-thread final fit: the structural floor
    at_width = C.audit_utilization(
        {"workers": 1, "inner_workers": 5, "inner_width": 5, "thread_budget": 1, "measured_cores": 4.8}, v
    )
    assert not at_width["saturated"] and at_width["ok"]  # fold width -> the floor
    under_width = C.audit_utilization(
        {"workers": 1, "inner_workers": 2, "inner_width": 5, "thread_budget": 1, "measured_cores": 1.9}, v
    )
    assert not under_width["ok"]  # 2-of-5: an UNDER-sized pool is NOT a floor (D1-IMPL-15)
    multi_wall = C.audit_utilization(
        {"workers": 1, "thread_budget": 1, "n_units": 20, "irreducible_floor_s": 99.0, "measured_cores": 1.0}, v
    )
    assert not multi_wall["ok"]  # a wall floor on 20 units must fan to saturate -> NOT a floor (D1-IMPL-15)
    declared = C.audit_utilization({"workers": 1, "thread_budget": 1, "measured_cores": 1.0, "serial_reason": "x"}, v)
    assert declared["floor"] and declared["ok"]


def test_a_w8_job_of_single_thread_units_audits_saturated():
    """Spec r3 D8 (A review SPEC-R3-01): 8 concurrent single-thread units fill the box, measured at the job."""
    a = C.audit_utilization(
        {"workers": 8, "inner_workers": 1, "thread_budget": 1, "n_units": 40, "measured_cores": 7.4}, 8
    )
    assert a["saturated"] and a["ok"]


def test_a_lone_single_thread_unit_audits_idle():
    """Spec r3 D8: a lone SINGLE-THREAD unit is never a floor by itself (2474b73: `match_outcome_dependence`, 1 core
    and 7 idle, passed as a floor)."""
    row = {"workers": 1, "inner_workers": 1, "thread_budget": 1, "n_units": 1, "irreducible_floor_s": 99.0}
    a = C.audit_utilization({**row, "measured_cores": 1.0}, 8)
    assert not a["saturated"] and not a["floor"] and not a["ok"]


def test_a_missing_job_record_audits_unmeasured():
    """Spec r3 D8: no job record -> UNMEASURED, never passing on configuration alone."""
    a = C.audit_utilization({"workers": 8, "inner_workers": 1, "thread_budget": 1, "n_units": 40}, 8)
    assert a["unmeasured"] and not a["ok"]


def test_a_bundled_job_is_judged_once_and_shown_on_each_stage():
    """Spec r3 D8 (B review R3-11): a bundled job ran two stages; both stages show that ONE job's measurement."""
    job = {"labels": {"job": "toy", "tier": "owner", "variant": "default", "stage": "a-b", "shard": "0of1"}}
    recs = {"toy.owner.default.a-b.0of1": [{"cpu_s": 600.0, "wall_s": 100.0}]}
    assert C.stage_measured_cores([job], recs) == C.stage_measured_cores([job], recs) == 6.0
    assert C.stage_measured_cores([], recs) is None


# ---- HFU plan R3-8 S3 items 3 and 4: the floors need measured use; the "too small to pay" class; 0.85 ----------


def test_the_saturation_bar_is_the_owner_ruled_0_85_and_it_holds_at_the_boundary():
    """Owner ruling 2026-10-30 (S0): calibrated on the HF W=8 stages (0.883-0.937)."""
    assert C.SATURATION_FRACTION == 0.85
    row = {"workers": 8, "inner_workers": 1, "thread_budget": 1, "n_units": 40}
    assert not C.audit_utilization({**row, "measured_cores": 0.849 * 8}, 8)["saturated"]
    assert C.audit_utilization({**row, "measured_cores": 0.851 * 8}, 8)["saturated"]


@pytest.mark.parametrize(
    ("row", "floor"),
    [
        # the final-fit floor, W = 1 at budget 8: job-record cores per process must be MORE than 4.0
        ({"workers": 1, "thread_budget": 8, "n_units": 1, "measured_cores": 0.97}, False),  # dead lever (S1 job record)
        ({"workers": 1, "thread_budget": 8, "n_units": 1, "measured_cores": 1.75}, False),  # row 11 as measured in S1
        ({"workers": 1, "thread_budget": 8, "n_units": 1, "measured_cores": 3.77}, False),  # xCross fit, S1d job record
        ({"workers": 1, "thread_budget": 8, "n_units": 1, "measured_cores": 4.0}, False),  # synthetic: the strict edge
        ({"workers": 1, "thread_budget": 8, "n_units": 1, "measured_cores": 6.0}, True),  # synthetic
        # the generalized floor, every unit at once with the whole budget
        ({"workers": 5, "thread_budget": 2, "n_units": 5, "measured_cores": 5.0}, False),  # synthetic dead lever
        ({"workers": 5, "thread_budget": 2, "n_units": 5, "measured_cores": 6.12}, True),  # S1d ghost_gk cv job record
        ({"workers": 5, "thread_budget": 2, "n_units": 5, "measured_cores": 5.39}, True),  # S1d xCross cv job record
        ({"workers": 5, "thread_budget": 1, "n_units": 5, "measured_cores": 4.6}, False),  # 5 x 1 < 8: no floor
        # the study at its fold width: the divisor counts the inner children (S2B-r3-03)
        ({"workers": 1, "inner_workers": 5, "inner_width": 5, "thread_budget": 2, "measured_cores": 4.7}, False),
        # synthetic
        ({"workers": 1, "inner_workers": 5, "inner_width": 5, "thread_budget": 2, "measured_cores": 6.0}, True),
        ({"workers": 1, "inner_workers": 5, "inner_width": 5, "thread_budget": 1, "measured_cores": 4.74}, True),
    ],
)
def test_every_structural_floor_needs_measured_use_above_half_its_budget(row, floor):
    a = C.audit_utilization(row, 8)
    assert not a["saturated"]
    assert a["floor"] is floor and a["ok"] is floor


def test_the_too_small_class_is_checked_against_the_measured_unit_and_reduce_cpu():
    small = "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU."
    row = {"workers": 1, "thread_budget": 1, "n_units": 1, "measured_cores": 0.4, "serial_reason": small}
    assert C.audit_utilization({**row, "unit_cpu_s": 45.0, "reduce_cpu_s": 2.0}, 8)["ok"]
    grown = C.audit_utilization({**row, "unit_cpu_s": 70.0, "reduce_cpu_s": None}, 8)
    assert not grown["ok"]  # the class cannot hide a stage that grew past 60 s
    reduce_only = {**row, "n_units": 0, "unit_cpu_s": 0.0, "reduce_cpu_s": 157.0}
    assert not C.audit_utilization(reduce_only, 8)["ok"]  # a heavy reduce counts (S2B-r2-04)
    unknown = C.audit_utilization({**row, "unit_cpu_s": 10.0, "reduce_wall": 30.0, "reduce_cpu_s": None}, 8)
    assert not unknown["ok"]  # a reduce ran but its CPU is unknown: fail closed


def test_a_stage_instance_under_sixty_seconds_passes_as_too_small_by_measurement():
    """C06: a smaller tier's instance of a shaped stage (here a cv at W=5 x 2 that is start-up bound) passes by the
    same measured rule; the audit records its measured CPU."""
    row = {"workers": 5, "thread_budget": 2, "n_units": 5, "measured_cores": 2.0, "unit_cpu_s": 21.0}
    a = C.audit_utilization(row, 8)
    assert a["ok"] and a["too_small"] and a["stage_cpu_s"] == 21.0
    big = C.audit_utilization({**row, "unit_cpu_s": 61.0}, 8)
    assert not big["ok"] and not big["too_small"]


def test_a_reviewed_serial_reason_outside_the_class_stays_a_floor():
    row = {"workers": 1, "thread_budget": 1, "n_units": 1, "measured_cores": 1.0, "unit_cpu_s": 900.0}
    reason = "Optuna TPE: each trial reads every earlier trial"
    assert C.audit_utilization({**row, "serial_reason": reason}, 8)["ok"]


def test_the_stage_summary_records_unit_and_reduce_cpu():
    st = {"name": "gen", "history_key": "gen", "n_units": 2}
    recs = [
        {"key": "0", "outcome": "done", "wall_s": 3.0, "cpu_s": 2.5, "peak_rss_bytes": 10**8},
        {"key": "1", "outcome": "done", "wall_s": 3.0, "cpu_s": 1.5, "peak_rss_bytes": 10**8},
    ]
    red = (json.dumps({"stage": "gen", "wall_s": 5.0, "cpu_s": 4.25}) + "\n").encode()
    row = C.stage_summary("i", st, recs, [], None, Constraints(), job_recs=recs, red=red, measured_cores=1.0)
    assert row["unit_cpu_s"] == 4.0 and row["reduce_cpu_s"] == 4.25


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
        C,
        "_job_specs",
        lambda c, commit, inst, plan, a, *, store, unit_peaks=None: [f"{inst.run_key}|{i}" for i in range(100)],
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


# ---- spec r4 D17: work is keyed by (run_key, generation), not by instance ------------------------------------

_GRK = "cycle1-abcdef1-ghost_gk-owner"


def _tok(stage: str, variant: str | None = None) -> str:
    """A 16-hex token: ONE per stage when ``variant`` is None (a generation the variants share), else per variant."""
    import hashlib

    return hashlib.sha256(f"{stage}/{variant or '*'}".encode()).hexdigest()[:16]


def _gstage(name: str, token: str, **kw) -> dict:
    st = _st(name, **kw)
    st.update(token=token, generation=f"ghost_gk/{name}/{token}")
    return st


def _gjob(
    jid, variant, stages, status, *, tokens, created_at=1, message=None, finished_at=None, records=None, tier="owner"
):
    """A campaign job as HF lists it: launched by ``variant`` (its instance), carrying each stage's generation."""
    labels = {
        "run_key": _GRK,
        "tier": tier,
        "variant": variant,
        "stage": "-".join(stages),
        "gens": "-".join(tokens[s][:8] for s in stages),
        "kind": "run",
    }
    if records is not None:
        labels["records"] = records
    return types.SimpleNamespace(
        id=jid,
        labels=labels,
        status=types.SimpleNamespace(stage=status, message=message),
        created_at=created_at,
        finished_at=finished_at,
    )


def _gid(variant: str, tier: str = "owner") -> str:
    return f"{_GRK}/{tier}/{variant}"


@pytest.fixture
def gen_world(monkeypatch):
    """`step` over instances of ONE run key (spec r4 D17), every network seam stubbed.

    ``run(plans, jobs, *, cache=None, history=None, n_specs=1)`` -> ``(report, launched, cancelled)``: ``plans`` maps a
    variant to its plan (or the RuntimeError its ``--plan`` raises); ``launched`` lists ``(variant, stages)`` per
    launched job; ``cancelled`` lists the cancelled job ids. Every call is a fresh step over a fresh tree identity,
    so a persistent ``cache`` re-reads every plan, as a controller does once the store or the tree moved."""
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import preflight

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
    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    steps = iter(range(10**6))

    def run(plans, jobs, *, cache=None, history=None, n_specs=1):
        # a key is a variant (the `owner` tier) or a (tier, variant) pair (spec r4 D18 (c): tiers share a store)
        pairs = [k if isinstance(k, tuple) else ("owner", k) for k in plans]
        insts = [
            C.Instance("ghost_gk", tier, variant, "scripts.train_ghost_gk", (), (), _GRK) for tier, variant in pairs
        ]
        launched: list = []
        cancelled: list = []

        def key(inst):
            return (inst.tier, inst.variant) if (inst.tier, inst.variant) in plans else inst.variant

        def read_plan(inst, bucket):
            p = plans[key(inst)]
            if isinstance(p, Exception):
                raise p
            return p

        class _Api:
            def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
                return []

            def cancel_job(self, *, job_id):
                cancelled.append(job_id)

        monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: insts)
        monkeypatch.setattr(C, "read_plan", read_plan)
        monkeypatch.setattr(C, "committed_history", lambda: dict(history or {}))
        monkeypatch.setattr(C, "_tree_identity", lambda: f"t{next(steps)}")
        monkeypatch.setattr(
            C,
            "_job_specs",
            lambda c, commit, inst, plan, a, *, store, unit_peaks=None: [(key(inst), a.stages)] * n_specs,
        )
        monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or f"new{len(launched)}")
        monkeypatch.setattr(L, "jobs_for", lambda api, **k: list(jobs))
        rep = C.step(c, SHA, api=_Api(), token="hf_x", cache=cache)
        return rep, launched, cancelled

    return run


def _shared_extract(**kw) -> dict:
    """Both variants' plans: ONE `extract` generation (the 2474b73 E11 shape)."""
    return {v: _plan([_gstage("extract", _tok("extract"), n=179, **kw)]) for v in ("default", "full")}


def test_a_shared_generation_launches_once_per_step(gen_world):
    """Spec r4 D17 (B R4S-01): two instances share `extract` (one generation in one store). The first in input order
    launches it and the second waits, in the SAME step: the 2474b73 controller launched each shared stage twice, in
    the same second (279 job-minutes)."""
    rep, launched, _ = gen_world(_shared_extract(), [])
    assert launched == [("default", ("extract",))]
    assert rep[_gid("full")]["action"] == "waiting" and rep[_gid("full")]["jobs"] == []
    # The other side of the band: two DIFFERENT generations of one stage name both launch.
    own = {v: _plan([_gstage("extract", _tok("extract", v), n=179)]) for v in ("default", "full")}
    _, launched, _ = gen_world(own, [])
    assert launched == [("default", ("extract",)), ("full", ("extract",))]


def test_two_tiers_of_one_dataset_launch_a_shared_extract_once_and_never_cancel_each_other(gen_world):
    """Spec r4 D18 (c) + D17: the public and sc_extended tiers read one dataset into one store, so their `extract` is
    ONE generation: the first tier launches it, the other waits. When the other tier halts on its own stage, it
    cancels none of the first tier's jobs."""
    toks = {"extract": _tok("extract"), "x": _tok("x", "sc_extended")}
    shared = {(t, "default"): _plan([_gstage("extract", toks["extract"], n=179)]) for t in ("public", "sc_extended")}
    rep, launched, _ = gen_world(shared, [])
    assert launched == [(("public", "default"), ("extract",))]
    assert rep[_gid("default", "sc_extended")]["action"] == "waiting"
    x = _gstage("x", toks["x"], n=40, done=39, pending=0, failed={"u007": 2})
    x["pending_units"] = []
    halting = {**shared, ("sc_extended", "default"): _plan([_gstage("extract", toks["extract"], n=179), x])}
    pub_job = _gjob("j_pub", "default", ("extract",), "RUNNING", tokens=toks, tier="public")
    rep, launched, cancelled = gen_world(halting, [pub_job])
    assert rep[_gid("default", "sc_extended")]["action"] == "halt" and cancelled == [] and launched == []


def test_a_shared_stage_in_flight_is_not_relaunched(gen_world):
    """Spec r4 D17: a stage is in flight when ANY live job carries its generation, whoever launched it. `full` never
    launched `extract`; the live `default` job holds it for both."""
    toks = {"extract": _tok("extract")}
    live = [_gjob("j1", "default", ("extract",), "RUNNING", tokens=toks)]
    rep, launched, _ = gen_world(_shared_extract(), live)
    assert launched == [] and {r["action"] for r in rep.values()} == {"waiting"}
    # A live job of ANOTHER generation holds nothing for `full`.
    other = {**_shared_extract(), "full": _plan([_gstage("extract", _tok("extract", "full"), n=179)])}
    _, launched, _ = gen_world(other, live)
    assert launched == [("full", ("extract",))]


def _halting_launcher() -> tuple[dict, dict]:
    """`default` launched the shared `extract` and its own `y` (both in flight); its own strict `x` has a unit out of
    attempts, so `default` halts. `full` needs only `extract`. Returns (plans, tokens)."""
    toks = {"extract": _tok("extract"), "x": _tok("x", "default"), "y": _tok("y", "default")}
    x = _gstage("x", toks["x"], n=40, done=39, pending=0, failed={"u007": 2})
    x["pending_units"] = []
    plans = {
        "default": _plan([_gstage("extract", toks["extract"], n=179), x, _gstage("y", toks["y"], n=179)]),
        "full": _plan([_gstage("extract", toks["extract"], n=179)]),
    }
    return plans, toks


def test_a_shared_stage_is_relaunched_when_its_launcher_halts_on_its_own_stage(gen_world):
    """Spec r4 D17: the launcher of a shared stage halts on its OWN stage while the shared stage is in flight. It
    cancels its own jobs but NOT the shared one `full` still needs (`full` waits on it). When the shared job ends
    without reducing the stage, the next step relaunches it for `full`."""
    plans, toks = _halting_launcher()
    j_ext = _gjob("j_ext", "default", ("extract",), "RUNNING", tokens=toks)
    j_y = _gjob("j_y", "default", ("y",), "RUNNING", tokens=toks)
    rep, launched, cancelled = gen_world(plans, [j_ext, j_y])
    d = rep[_gid("default")]
    assert d["action"] == "halt" and "u007" in d["failed"]
    assert cancelled == ["j_y"] and d["cancelled"] == ["j_y"]
    assert d["kept"] == [{"job": "j_ext", "for": [_gid("full")]}]
    assert rep[_gid("full")]["action"] == "waiting" and launched == []
    # The shared job ends (here CANCELED by hand) without reducing `extract`: `full` relaunches it.
    j_ext.status.stage = "CANCELED"
    j_y.status.stage = "CANCELED"
    rep, launched, cancelled = gen_world(plans, [j_ext, j_y])
    assert launched == [("full", ("extract",))] and cancelled == []
    assert rep[_gid("default")]["action"] == "halt"


def test_a_plan_error_in_one_instance_does_not_duplicate_a_shared_launch(gen_world):
    """Spec r4 D17: the in-flight and history sets come from job labels, not from plans, so a plan error of one
    instance changes nothing for the others."""
    toks = {"extract": _tok("extract")}
    plans = {**_shared_extract(), "default": RuntimeError("--plan failed (rc=1): boom")}
    live = [_gjob("j1", "default", ("extract",), "RUNNING", tokens=toks)]
    rep, launched, _ = gen_world(plans, live)
    assert launched == [] and rep[_gid("full")]["action"] == "waiting"
    assert "boom" in rep[_gid("default")]["failed"]
    # Nothing in flight: `full` launches the shared stage once; the errored `default` adds no second launch.
    _, launched, _ = gen_world(plans, [])
    assert launched == [("full", ("extract",))]


def test_a_refused_resume_on_a_shared_stage_halts_every_instance_that_needs_it(gen_world):
    """Spec r4 D17: `error_resumes` ERRORs in a row of jobs that carry the shared generation -- whoever launched them --
    refuse the resume for EVERY instance that needs it, with the same reason, in the same step. Below the bound the
    shared stage resumes ONCE (one launch, not one per instance)."""
    toks = {"extract": _tok("extract")}
    n = Constraints().error_resumes
    errs = [
        _gjob(f"j{i}", "default", ("extract",), "ERROR", tokens=toks, created_at=i, message="boom")
        for i in range(1, n + 1)
    ]
    rep, launched, _ = gen_world(_shared_extract(), errs)
    assert launched == []
    assert {rep[_gid(v)]["failed"] for v in ("default", "full")} == {f"extract: {n} job(s) in a row ended in ERROR"}
    # Below the bound, with a retryable failed unit, the stage resumes once for both instances.
    _, launched, _ = gen_world(_shared_extract(failed={"u007": 1}), errs[:1])
    assert launched == [("default", ("extract",))]


def test_a_shared_stage_out_of_attempts_halts_every_instance_that_needs_it(gen_world):
    """Spec r4 D17: a shared strict stage with a unit out of attempts halts EVERY instance that needs it, with the same
    reason, in the same step; no instance is left that needs a job, so every job of the halting ones is cancelled."""
    toks = {"extract": _tok("extract"), "y": _tok("y", "default")}

    def spent() -> dict:
        st = _gstage("extract", toks["extract"], n=40, done=39, pending=0, failed={"u007": 2})
        st["pending_units"] = []
        return st

    plans = {"default": _plan([spent(), _gstage("y", toks["y"], n=179)]), "full": _plan([spent()])}
    rep, launched, cancelled = gen_world(plans, [_gjob("j_y", "default", ("y",), "RUNNING", tokens=toks)])
    assert launched == [] and cancelled == ["j_y"]
    assert {rep[_gid(v)]["action"] for v in ("default", "full")} == {"halt"}
    assert rep[_gid("default")]["failed"] == rep[_gid("full")]["failed"] and "u007" in rep[_gid("full")]["failed"]


def test_a_zero_progress_stall_on_a_shared_head_halts_every_instance_that_needs_it(gen_world):
    """Spec r4 D17 + r3 D5: the zero-progress guard judges the jobs of the action's FIRST generation against that
    generation's progress mark. `default` runs the shared single-unit `fit` alone; `full` groups it with its own ready
    `z`, which holds a failure record. Two settled jobs that `default` launched at `fit`'s mark committed nothing:
    both instances halt in the same step (a mark over the whole group differs for `full` and would let it launch)."""
    import datetime as dt

    toks = {"fit": _tok("fit"), "z": _tok("z", "full")}
    plans = {
        "default": _plan([_gstage("fit", toks["fit"], n=1)]),
        "full": _plan([_gstage("fit", toks["fit"], n=1), _gstage("z", toks["z"], n=1, failed={"u0": 1})]),
    }
    mark = C.progress_mark_of({}, plans["default"], ["fit"])
    ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=2 * C.PROGRESS_SETTLE_S)
    jobs = [
        _gjob(f"j{i}", "default", ("fit",), "COMPLETED", tokens=toks, created_at=i, finished_at=ago, records=mark)
        for i in (1, 2)
    ]
    rep, launched, _ = gen_world(plans, jobs)
    assert launched == []
    assert {rep[_gid(v)]["action"] for v in ("default", "full")} == {"halt"}
    assert all("zero progress" in rep[_gid(v)]["failed"] for v in ("default", "full"))


def test_a_halting_instance_defers_a_possibly_shared_cancel_while_another_plan_errors(gen_world):
    """Spec r4 D17 (B r4-spec-r2 CONSIDER): `full`'s plan errors in the step `default` halts. `default` cancels none of
    its jobs that carry a generation `full` held in its LAST plan, and retries next step (not settled); a job only it
    needs is cancelled. With no last plan known (a fresh controller), every cancel waits."""
    plans, toks = _halting_launcher()
    j_ext = _gjob("j_ext", "default", ("extract",), "RUNNING", tokens=toks)
    j_y = _gjob("j_y", "default", ("y",), "RUNNING", tokens=toks)
    healthy = {"default": _plan([_gstage("extract", toks["extract"], n=179), _gstage("y", toks["y"], n=179)])}
    cache: dict = {}
    gen_world({**plans, **healthy}, [j_ext, j_y], cache=cache)  # step 1: both plans read; `full` holds `extract`
    errored = {**plans, "full": RuntimeError("--plan failed (rc=1): boom")}
    rep, _, cancelled = gen_world(errored, [j_ext, j_y], cache=cache)
    d = rep[_gid("default")]
    assert d["action"] == "halt" and cancelled == ["j_y"] and d["cancel_deferred"] == ["j_ext"]
    assert not C.all_settled(rep)
    j_y.status.stage = "CANCELED"
    rep, _, cancelled = gen_world(plans, [j_ext, j_y], cache=cache)  # `full` plans again and needs `extract`
    assert cancelled == [] and "cancel_deferred" not in rep[_gid("default")]
    assert rep[_gid("default")]["kept"] == [{"job": "j_ext", "for": [_gid("full")]}]
    # A fresh controller has no last plan for `full`: nothing is cancelled until it plans again.
    j_y.status.stage = "RUNNING"
    rep, _, cancelled = gen_world(errored, [j_ext, j_y], cache={})
    assert cancelled == [] and rep[_gid("default")]["cancel_deferred"] == ["j_ext", "j_y"]


def test_a_waiting_instance_takes_no_budget_share(gen_world):
    """Spec r4 D17: an instance whose ready stages are all in flight takes no share of the step budget, so the one
    instance with work gets the whole free budget (dispatch_cap 10, one stage in flight: 9), not half of it."""
    toks = {"extract": _tok("extract"), "x": _tok("x", "default")}
    hist = {"x": {"cost": UnitCost(600, 600, 600, 10**9, 20), "efficiency": None, "per_unit": None}}
    plans = {
        "default": _plan([_gstage("x", toks["x"], n=179, hk="x")]),
        "full": _plan([_gstage("extract", toks["extract"], n=179)]),
    }
    live = [_gjob("j1", "full", ("extract",), "RUNNING", tokens=toks)]
    rep, launched, _ = gen_world(plans, live, history=hist, n_specs=100)  # more specs than any share
    assert rep[_gid("full")]["action"] == "waiting"
    assert len(launched) == 9  # the whole free budget; a share split over 2 "active" instances gave 4


def test_the_step_budget_subtracts_every_live_job_not_every_live_stage(gen_world):
    """Owner ruling 2026-10-30 (spec §C: `dispatch_cap` caps total JOBS in flight). One stage running 6 jobs holds 6 of
    the 10 slots, so the step launches at most 4 more; the 2474b73 controller subtracted 1 (one stage) and could
    launch 9, over the cap."""
    toks = {"extract": _tok("extract", "full"), "x": _tok("x", "default")}
    hist = {"x": {"cost": UnitCost(600, 600, 600, 10**9, 20), "efficiency": None, "per_unit": None}}
    plans = {
        "default": _plan([_gstage("x", toks["x"], n=179, hk="x")]),
        "full": _plan([_gstage("extract", toks["extract"], n=179)]),
    }
    live = [_gjob(f"j{i}", "full", ("extract",), "RUNNING", tokens=toks) for i in range(6)]
    _, launched, _ = gen_world(plans, live, history=hist, n_specs=100)
    assert len(launched) == 4
    # The other side: the same 6 jobs, now ended, hold no slot.
    for j in live:
        j.status.stage = "COMPLETED"
    _, launched, _ = gen_world(plans, live, history=hist, n_specs=100)
    assert len(launched) > 4


def test_a_generation_prefix_collision_is_refused(gen_world):
    """Spec r4 D17: jobs carry 8-hex generation prefixes, so two generations of ONE store with the same prefix would
    alias each other's jobs. The controller refuses such a step and launches nothing; a generation the plans SHARE is
    no collision."""
    a, b = "abcdef01" + "0" * 8, "abcdef01" + "1" * 8
    plans = {"default": _plan([_gstage("extract", a, n=179)]), "full": _plan([_gstage("extract", b, n=179)])}
    with pytest.raises(RuntimeError, match="prefix"):
        gen_world(plans, [])
    shared = {v: _plan([_gstage("extract", a, n=179)]) for v in ("default", "full")}
    _, launched, _ = gen_world(shared, [])
    assert launched == [("default", ("extract",))]


def test_the_job_labels_carry_every_stage_generation():
    """Spec r4 D17: every job carries the generation prefix of each stage it runs, in stage order (the `gens` label),
    so the stateless controller can key work by generation."""
    from scripts.hf_jobs import launch as L

    inst = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), _GRK)
    toks = {n: _tok(n) for n in ("fit", "assemble")}
    plan = _plan([_gstage(n, toks[n], n=1) for n in ("fit", "assemble")])

    class _Store:
        def names(self, prefix):
            return []

    a = C.Action("run", ("fit", "assemble"), 1, 1, 1, 600)
    specs = C._job_specs(_campaign(), SHA, inst, plan, a, store=_Store())
    want = (toks["fit"][:8], toks["assemble"][:8])
    assert [s.gens for s in specs] == [want]
    assert L.labels_for(specs[0])["gens"] == "-".join(want)
    assert L.gens_of(L.labels_for(specs[0])) == want


def test_every_case_settles_the_watch(gen_world):
    """Spec r4 D17: `--watch` settles in every shared-generation case. A small simulated HF: a job lives two steps,
    then ends -- ERROR (a timeout) when it carries a failing generation, else COMPLETED, reducing each stage it ran in
    the ONE store, for every instance that holds that generation. The world must settle within a bound, never launch
    a generation a live job carries, and launch a failing shared stage no more often than one instance would."""
    toks = {"extract": _tok("extract"), "x": _tok("x", "default")}
    for v in ("default", "full"):
        toks[f"fit_{v}"] = _tok("fit", v)

    def plans_for(reduced: set[str], *, halt_x: bool, errored: bool) -> dict:
        out: dict = {}
        for v in ("default", "full"):
            stages = []
            for name, tok, needs in (("extract", toks["extract"], ()), ("fit", toks[f"fit_{v}"], ("extract",))):
                st = _gstage(name, tok, n=1, ready=all(toks[n] in reduced for n in needs), needs=needs)
                if tok in reduced:
                    st.update(reduced=True, done=1, pending=0, pending_units=[])
                stages.append(st)
            if v == "default" and halt_x:
                x = _gstage("x", toks["x"], n=2, done=1, pending=0, failed={"u1": 2})
                x["pending_units"] = []
                stages.append(x)
            out[v] = RuntimeError("--plan failed (rc=1): boom") if (v == "full" and errored) else _plan(stages)
        return out

    def simulate(*, fail=(), halt_from=None, errored_steps=()):
        reduced: set[str] = set()
        jobs: list = []
        born: dict[str, int] = {}
        cache: dict = {}
        n_launch: dict[str, int] = {}
        failing = {toks[s][:8] for s in fail}
        for n in range(16):
            for j in jobs:  # a job ends two steps after its launch
                if j.status.stage == "RUNNING" and n - born[j.id] >= 2:
                    carried = set(j.labels["gens"].split("-"))
                    if carried & failing:
                        j.status.stage, j.status.message = "ERROR", "Job timeout"
                    else:
                        j.status.stage = "COMPLETED"
                        reduced |= {t for t in toks.values() if t[:8] in carried}
            halt_x = halt_from is not None and n >= halt_from
            plans = plans_for(reduced, halt_x=halt_x, errored=n in errored_steps)
            live = {g for j in jobs if j.status.stage == "RUNNING" for g in j.labels["gens"].split("-")}
            rep, launched, cancelled = gen_world(plans, jobs, cache=cache)
            for j in jobs:
                if j.id in cancelled:
                    j.status.stage = "CANCELED"
            for i, (v, stages) in enumerate(launched):
                tokens = {"extract": toks["extract"], "fit": toks[f"fit_{v}"], "x": toks["x"]}
                gens = {tokens[s][:8] for s in stages}
                assert not gens & live, f"step {n}: launched a generation a live job carries: {stages}"
                live |= gens
                for g in gens:
                    n_launch[g] = n_launch.get(g, 0) + 1
                jobs.append(_gjob(f"s{n}j{i}", v, stages, "RUNNING", tokens=tokens, created_at=100 * n + i))
                born[jobs[-1].id] = n
            if C.all_settled(rep):
                return rep, n_launch
        raise AssertionError("the watch never settled")

    rep, n_launch = simulate()  # idle sharing: everything done, the shared `extract` launched once
    assert {r["action"] for r in rep.values()} == {"done"} and n_launch[toks["extract"][:8]] == 1
    rep, n_launch = simulate(halt_from=1)  # the launcher halts on its own stage with the shared stage in flight
    assert rep[_gid("default")]["action"] == "halt" and rep[_gid("full")]["action"] == "done"
    assert n_launch[toks["extract"][:8]] == 1  # its shared job was kept for `full`, not cancelled and relaunched
    rep, n_launch = simulate(fail=("extract",))  # the shared stage times out every time
    n = Constraints().error_resumes
    assert {r["failed"] for r in rep.values()} == {f"extract: {n} job(s) in a row ended in ERROR"}
    assert n_launch[toks["extract"][:8]] == n  # once per resume, not once per instance
    rep, _ = simulate(halt_from=1, errored_steps=(1,))  # `full`'s plan errors in the step the launcher halts
    assert rep[_gid("default")]["action"] == "halt" and rep[_gid("full")]["action"] == "done"


def test_a_shared_stage_is_audited_once_under_its_launcher(summary_world):
    """Spec r4 D17: both variants share the `cv` generation and `default` launched all its jobs. `default`'s row is the
    stage's ONE audit; `position_only`'s row names the launcher (`shared_with`) and carries no verdict, so it is never
    reported as unmeasured."""
    rows, _ = summary_world(shared=True)
    by_variant = {k.split("/")[-1]: r for k, r in rows.items()}
    assert by_variant["default"]["n_jobs"] == 3 and by_variant["default"]["shared_with"] == []
    assert by_variant["default"]["audit"] is not None
    po = by_variant["position_only"]
    assert po["shared_with"] == [f"{_SUM_RK}/owner/default"] and po["audit"] is None and po["n_jobs"] == 0


def test_a_tier_shared_stage_is_audited_once_under_its_launching_tier(summary_world):
    """Spec r4 D18 (c): two tiers of one dataset share a store and the variant name, so a job belongs to the
    (run_key, tier, variant) that launched it: the launching tier's row audits the stage, the other tier's row names
    it (`shared_with`), never counting its jobs twice."""
    rows, _ = summary_world(shared=True, tiers=True)
    pub, sc = rows[f"{_SUM_RK}/public/default"], rows[f"{_SUM_RK}/sc_extended/default"]
    assert pub["n_jobs"] == 3 and pub["shared_with"] == [] and pub["audit"] is not None
    assert sc["n_jobs"] == 0 and sc["shared_with"] == [f"{_SUM_RK}/public/default"] and sc["audit"] is None


def test_a_shared_row_never_masks_its_launchers_history(tmp_path):
    """Spec r4 D17: `load_history` keys the summary rows by history key, a later row winning. A shared row has no job
    measurement (no startup), so it must never replace its launcher's row, or the planner would lose that history."""
    import json

    from scripts.hf_jobs.ledger import load_history

    cost = {"p50_s": 60.0, "p95_s": 70.0, "p99_s": 80.0, "peak_rss_bytes": 2**30, "n_samples": 40}
    launcher = {"history_key": "hk", "instance": "rk/default", "startup_s": 140.0, "unit_cost": cost, "shared_with": []}
    shared = {**launcher, "instance": "rk/full", "startup_s": None, "shared_with": ["rk/default"]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"stages": [launcher, shared]}), encoding="utf-8")
    assert load_history([p])["hk"]["instance"] == "rk/default"
