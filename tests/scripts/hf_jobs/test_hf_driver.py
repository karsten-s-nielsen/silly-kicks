"""Driver scheduling decision.

Both halves use REAL data (owner rule, memory ``tdd-tests-from-real-data``):

* Failure records come from ``fixtures/real_jobinfo.json`` -- real captured
  ``JobInfo`` (``status.stage`` + ``status.message`` + real label schema).
* Plan records come from REAL ``scripts._stages.plan()`` output, produced by
  driving a small real pipeline over a ``LocalStore`` (``_realplan``). The earlier
  hand-built plan dicts omitted ``excluded`` / ``max_failures`` -- exactly the keys
  that decide whether a stage reduces -- and that is how two reduce bugs shipped
  (review HFUD-IMPL-06). A real ``plan()`` carries every key.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from scripts import _stages as s
from scripts.hf_jobs.campaign import Campaign, Instance
from scripts.hf_jobs.driver import (
    Launch,
    StepResult,
    _attribute,
    _job_spec,
    _seed_baseline,
    by_job_rel,
    decide,
    main,
    run,
    step,
)

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "real_jobinfo.json"


def _cover(insts, plans, live, baseline, store_for, *, attempts=3):
    """Per-unit coverage from ``_attribute`` (fix 3): the first of its (cover, reduce_errors, mount_faults)."""
    return _attribute(insts, plans, live, baseline, store_for, attempts=attempts)[0]


def _rerr(insts, plans, live, baseline, store_for, *, attempts=3):
    """Per-stage reduce-trigger failures from ``_attribute`` (fix 3): the second of its three maps."""
    return _attribute(insts, plans, live, baseline, store_for, attempts=attempts)[1]


def _mfaults(insts, plans, live, baseline, store_for, *, attempts=3):
    """Per-dataset mount faults from ``_attribute`` (fix 3 by-locus): the third of its three maps."""
    return _attribute(insts, plans, live, baseline, store_for, attempts=attempts)[2]


def _records() -> dict[str, dict]:
    return {r["_class"]: r for r in json.loads(_FIXTURE.read_text())}


# --- real plan() construction -------------------------------------------------


def _work(fail: tuple[str, ...] = (), exclude: tuple[str, ...] = ()):
    def w(k, c):
        if k in exclude:
            raise s.ItemExcluded("deterministic exclusion")
        if k in fail:
            raise ValueError(f"injected failure {k}")
        return {"k": k}

    return w


_PARAMS = (s.Param("x", int, 1),)  # module-level singleton (B008: no call in a default argument)


def _pipeline(stages, params=_PARAMS, **kw) -> s.Pipeline:
    return s.Pipeline(name="t", module="tests.scripts._toy_pipeline", stages=stages, params=params, **kw)


def _ctx(tmp_path, stages, params=None):
    prov = {"commit": "c" * 40, "tree_state": "clean", "code_identity": "c" * 40}
    return s.make_context(_pipeline(stages), params or {"x": 1}, prov, store=s.LocalStore(tmp_path), data_root=None)


def _realplan(ctx) -> dict:
    """The genuine ``plan()`` dict for ``ctx``'s current store state (every schema key present)."""
    return s.plan(ctx)


def _run(ctx, stage_name: str, *keys: str) -> None:
    st = ctx.pipeline.stage(stage_name)
    for k in keys:
        s.run_unit(ctx, st, k, worker="w0")


# --- instances + live-job records (real label schema) -------------------------


def _inst(run_key: str = "rk", tier: str = "public", variant: str = "default") -> Instance:
    return Instance("job", tier, variant, "mod", (), (), run_key)


def _labels(inst: Instance, stage: str, *, shard: str = "0of1") -> dict:
    """The REAL label schema: main's ``launch.labels_for`` over a ``JobSpec`` for this instance and stage
    (campaign / run_key / job / variant / stage / shard / commit). No tier / gens / kind label exists on main,
    so the driver matches jobs on ``(run_key, variant, stage)`` only (reviews HFD1-TEST-06 / D1-IMPL-08)."""
    from scripts.hf_jobs.launch import JobSpec, labels_for

    i, _, n = shard.partition("of")
    spec = JobSpec(
        campaign="cycle1",
        run_key=inst.run_key,
        job=inst.job,
        variant=inst.variant,
        stages=(stage,),
        shard=(int(i), int(n)),
        module=inst.module,
        args=(),
        commit=_COMMIT,
        bucket="silly-kicks/hf-refit-scratch",
        workers=8,
        inner_workers=1,
        cap_bytes=None,
        peak_rss_bytes=None,
        timeout_s=7200,
        attempts=1,
        datasets=(),
        extras=(),
    )
    return labels_for(spec)


def _running(inst: Instance, stage: str, *, shard: str = "0of1", created="2026-11-02T03:00:00+00:00") -> dict:
    """A RUNNING job record. No real RUNNING sample exists (nothing ran at capture); the labels are main's
    real ``labels_for`` schema."""
    return {"stage": "RUNNING", "message": None, "created_at": created, "labels": _labels(inst, stage, shard=shard)}


def _errored(
    inst: Instance,
    stage: str,
    message: str,
    *,
    shard: str = "0of1",
    job_id: str = "e1",
    created="2026-11-02T03:00:00+00:00",
) -> dict:
    """A terminal ERROR record with a REAL captured ``message`` (classify's only input) and main's real labels."""
    return {
        "id": job_id,
        "stage": "ERROR",
        "message": message,
        "created_at": created,
        "labels": _labels(inst, stage, shard=shard),
    }


def _terminal(inst: Instance, stage: str, raw_stage: str, *, created: str) -> dict:
    return {"stage": raw_stage, "message": None, "created_at": created, "labels": _labels(inst, stage)}


def _campaign(
    concurrency_cap: int = 20,
    *,
    units_per_job: int = 20,
    max_shards: int = 12,
    launch_batch: int = 100,
    mount_cap: int = 10**9,
    mount_cap_by_dataset: dict | None = None,
    mount_fault_budget: int = 10**9,
    peak_rss_by_stage: dict | None = None,
    backoff_max_s: float = 300.0,
) -> Campaign:
    # concurrency_cap is the in-flight ceiling; sizing is units_per_job/max_shards (fix 1). The caps default
    # effectively off here so the pre-fix behavioural tests are unaffected; fix-specific tests set them.
    return Campaign(
        name="cycle1",
        bucket="silly-kicks/hf-refit-scratch",
        concurrency_cap=concurrency_cap,
        attempts=3,
        jobs=(),
        path=None,
        sha256="0" * 64,
        units_per_job=units_per_job,
        max_shards=max_shards,
        launch_batch=launch_batch,
        mount_cap=mount_cap,
        mount_cap_by_dataset=mount_cap_by_dataset or {},
        mount_fault_budget=mount_fault_budget,
        peak_rss_by_stage=peak_rss_by_stage or {},
        backoff_max_s=backoff_max_s,
    )


# --- stage-walk decisions (real plans) ----------------------------------------


def test_launch_pending_units_when_nothing_in_flight(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work()),))
    inst = _inst()
    decision = decide(
        instances=[inst], plans={inst.id: _realplan(ctx)}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12
    )
    assert decision.fatal is None and decision.done is False
    assert all(lx.kind == "run" and lx.stages == ("extract",) for lx in decision.launches)
    assert sorted(k for lx in decision.launches for k in lx.keys) == ["a", "b", "c"]


def test_reduce_launched_when_all_units_done(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a", "b")
    inst = _inst()
    decision = decide(
        instances=[inst], plans={inst.id: _realplan(ctx)}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12
    )
    assert [lx.kind for lx in decision.launches] == ["reduce"]
    assert decision.launches[0].stages == ("extract",)


def test_reduce_launched_when_units_done_or_excluded(tmp_path) -> None:
    """HFUD-IMPL-01: an excluded unit counts in ``n_units`` but not ``done``. The stage still reduces."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work(exclude=("c",))),))
    _run(ctx, "extract", "a", "b", "c")  # c raises ItemExcluded -> real .excluded.json
    plan = _realplan(ctx)
    extract = plan["stages"][0]
    assert (extract["n_units"], extract["done"], extract["excluded"]) == (3, 2, 1)  # real plan() shape
    inst = _inst()
    decision = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12
    )
    assert [lx.kind for lx in decision.launches] == ["reduce"]
    assert decision.reports == ()


def test_tolerant_stage_reduces_over_a_spent_failure(tmp_path) -> None:
    """HFUD-IMPL-02: a ``max_failures=None`` stage reduces over a unit that is spent (failed ``attempts`` times)."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",)), max_failures=None),))
    _run(ctx, "fit", "a")  # a done
    for _ in range(3):  # b fails 3 times == attempts -> spent
        _run(ctx, "fit", "b")
    plan = _realplan(ctx)
    fit = plan["stages"][0]
    assert fit["max_failures"] is None and fit["failed_units"] == {"b": 3}
    inst = _inst()
    cover = {(inst.id, "fit", "b"): 3}  # b covered 3 times without completing -> spent (coverage == attempts)
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        coverage=cover,
    )
    assert [lx.kind for lx in decision.launches] == ["reduce"]  # tolerant: reduces over the rest
    assert decision.reports == ()


def test_strict_stage_reports_a_spent_failure_over_tolerance(tmp_path) -> None:
    """A ``max_failures=0`` stage with a spent failure cannot reduce: it is reported, not retried, not reduced."""
    ctx = _ctx(
        tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",))),)
    )  # default max_failures=0
    _run(ctx, "fit", "a")
    for _ in range(3):
        _run(ctx, "fit", "b")
    plan = _realplan(ctx)
    assert plan["stages"][0]["max_failures"] == 0
    inst = _inst()
    cover = {(inst.id, "fit", "b"): 3}  # b spent (coverage == attempts); max_failures=0 does not tolerate it
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        coverage=cover,
    )
    assert decision.launches == ()
    assert len(decision.reports) == 1 and "fit" in decision.reports[0][1]


def test_crash_under_attempts_reruns_the_failed_unit(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",))),))
    _run(ctx, "fit", "a")
    _run(ctx, "fit", "b")  # 1 failure < attempts(3)
    plan = _realplan(ctx)
    assert plan["stages"][0]["failed_units"] == {"b": 1}
    inst = _inst()
    crash = _records()["exit_1"]["message"]
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[_errored(inst, "fit", crash)],
        attempts=3,
        units_per_job=20,
        max_shards=12,
    )
    assert [k for lx in decision.launches for k in lx.keys] == ["b"]  # re-run the one failed unit
    assert decision.reports == ()


def test_timeouts_rerun_the_pending_units_not_stall_the_wave(tmp_path) -> None:
    """HFUD-IMPL-07 / P5: a wide wave with several timed-out shards re-runs the pending units; it does NOT
    report the stage stuck on a per-stage error count. A timed-out unit leaves no failure record (the job was
    killed), so it comes back pending and is simply re-run."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    _run(ctx, "extract", *[s.int_key(i) for i in range(22)])  # 22 done, 3 still pending (timed-out shards)
    plan = _realplan(ctx)
    inst = _inst()
    timeout = _records()["job_timeout"]["message"]
    live = [_errored(inst, "extract", timeout, shard=f"{i}of25", job_id=f"t{i}") for i in range(3)]
    decision = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=live, attempts=3, units_per_job=20, max_shards=12
    )
    assert decision.reports == ()  # not stalled
    assert sorted(k for lx in decision.launches for k in lx.keys) == sorted(s.int_key(i) for i in range(22, 25))


def test_no_launch_while_stage_in_flight(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[_running(inst, "extract")],
        attempts=3,
        units_per_job=20,
        max_shards=12,
    )
    assert decision.launches == () and decision.done is False


def test_stage_barrier_one_stage_at_a_time(tmp_path) -> None:
    ctx = _ctx(
        tmp_path,
        (
            s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),
            s.Stage(name="study", units=lambda c: ["x"], work=_work(), needs=("extract",)),
        ),
    )
    inst = _inst()
    decision = decide(
        instances=[inst], plans={inst.id: _realplan(ctx)}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12
    )
    assert {lx.stages for lx in decision.launches} == {("extract",)}  # study waits behind the barrier


def test_reduce_failure_is_bounded_then_reported(tmp_path) -> None:
    """HFUD-IMPL-03: a reduce that keeps crashing is bounded by ``attempts`` reduce-trigger ERROR jobs, then
    reported -- not relaunched on every step. ``decide`` takes the per-stage reduce-error count from ``step``
    (``_reduce_errors``), so here it is passed directly."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a", "b")  # all done -> would reduce
    plan = _realplan(ctx)
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        reduce_errors={(inst.id, "extract"): 3},  # == attempts
    )
    assert decision.launches == ()
    assert len(decision.reports) == 1 and "reduce" in decision.reports[0][1]


def test_done_when_every_stage_reduced(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    _run(ctx, "extract", "a")
    s.reduce_stage(ctx, "extract")  # real reduce -> output present -> plan reports reduced=True
    plan = _realplan(ctx)
    assert plan["stages"][0]["reduced"] is True
    inst = _inst()
    decision = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12
    )
    assert decision.launches == () and decision.done is True


# --- FATAL scoping (HFUD-IMPL-05, owner ruling 2026-11-02: recoverable) --------


def _fatal_rec(inst: Instance, stage: str, *, created: str) -> dict:
    base = _records()["storage_403_started_set"]  # real storage-limit 403
    return {**base, "created_at": created, "labels": _labels(inst, stage)}


def test_fatal_blocks_current_unreduced_work(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[_fatal_rec(inst, "extract", created="2026-11-02T03:00:00+00:00")],
        attempts=3,
        units_per_job=20,
        max_shards=12,
    )
    assert decision.fatal is not None and decision.fatal[0] == "storage_limit_403"


def test_a_baseline_fatal_does_not_block_so_resume_recovers(tmp_path) -> None:
    """HFUD-IMPL-05/r2: the documented recovery is a fresh resume after freeing storage. ``run`` seeds the
    FATALs already terminal at the start into ``baseline_error_ids``; ``decide`` looks past those, so the one
    pending unit launches. The round-1 fix was circular (it waited for a superseding job the blocked driver
    could never create) -- this proves recovery from a state the driver reaches: the 403 is the only job."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    fatal = _fatal_rec(inst, "extract", created="2026-11-02T03:00:00+00:00")
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[fatal],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        baseline_error_ids=frozenset({fatal["id"]}),  # what run() seeds at the start of a resume
    )
    assert decision.fatal is None and decision.launches  # looks past the resolved fault and launches


def test_a_new_fatal_blocks_on_any_shard_without_sibling_masking(tmp_path) -> None:
    """HFUD-IMPL-17: a FATAL on one shard must block even when a sibling shard is still RUNNING. The round-1
    'latest by created_at in the group' logic let a 1-2 ms-later sibling mask the FATAL; the baseline check
    blocks on ANY new FATAL for current work."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    live = [
        _fatal_rec(inst, "extract", created="2026-11-02T03:00:00.000+00:00"),  # shard 0 FATAL
        _running(inst, "extract", shard="1of2", created="2026-11-02T03:00:00.002+00:00"),  # sibling, later
    ]
    decision = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=live, attempts=3, units_per_job=20, max_shards=12
    )
    assert decision.fatal is not None and decision.fatal[0] == "storage_limit_403"


def test_historical_fatal_of_another_instance_does_not_block(tmp_path) -> None:
    """HFUD-IMPL-05 / P6: a 403 from a different instance no longer stops THIS instance's current work."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    plan = _realplan(ctx)
    here = _inst()
    other = _inst(run_key="rk2")
    decision = decide(
        instances=[here],
        plans={here.id: plan},
        live_jobs=[_fatal_rec(other, "extract", created="2026-11-02T03:00:00+00:00")],
        attempts=3,
        units_per_job=20,
        max_shards=12,
    )
    assert decision.fatal is None and decision.launches  # here's work proceeds


# --- per-unit coverage bound (HFUD-IMPL-16/19/23/26/28, owner 2026-11-02) ------
#
# One per-unit bound: a unit covered `attempts` times by terminal RUN jobs of THIS run, without completing, is
# SPENT. `step` builds the coverage map from the per-job partition manifests it writes at launch (a job record
# never says which units it held); `decide` consumes it. This unifies crashes (recorded or not), worker deaths
# and timeouts, which exit codes and plan state could not tell apart (16/19/23/26/28).


class _ManifestStore:
    """A store double that round-trips the per-job partition manifests ``step`` writes at launch."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload_bytes(self, rel: str, data: bytes) -> None:
        self.files[rel] = data

    def read_bytes(self, rel: str) -> bytes | None:
        return self.files.get(rel)


def _ran_job(inst, plan, token, job_id, keys, *, outcome="COMPLETED", message=None, store=None) -> dict:
    """A terminal job record for stage ``extract`` + (for a run job with a store) its per-job manifest."""
    if store is not None:
        store.upload_bytes(by_job_rel(plan, "extract", token, job_id), json.dumps(sorted(keys)).encode("utf-8"))
    return {
        "id": job_id,
        "stage": outcome,
        "message": message,
        "created_at": "2026-11-02T03:00:00+00:00",
        "labels": _labels(inst, "extract", shard="0of1"),
    }


def _one_stage(tmp_path, units):
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: list(units), work=_work()),))
    plan = _realplan(ctx)
    return plan, plan["stages"][0]["token"]


def test_a_unit_spent_by_coverage_is_reported(tmp_path) -> None:
    """A unit covered ``attempts`` times without completing is SPENT; a strict stage reports it (B's V5/28: a
    worker death or pod-OOM that writes NO record still bounds, because coverage counts the covering jobs)."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")  # a done, b pending (its 3 jobs all crashed, no record)
    plan = _realplan(ctx)
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        coverage={(inst.id, "extract", "b"): 3},
    )
    assert decision.launches == () and len(decision.reports) == 1 and "extract" in decision.reports[0][1]


def test_a_unit_under_coverage_is_relaunched(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")
    plan = _realplan(ctx)
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        coverage={(inst.id, "extract", "b"): 2},  # < attempts
    )
    assert [k for lx in decision.launches for k in lx.keys] == ["b"] and decision.reports == ()


def test_a_done_unit_with_high_coverage_is_not_reported(tmp_path) -> None:
    """HFUD-IMPL-26 (B's T3): a unit that failed then RECOVERED is done, so it is not a candidate -- its (now
    high) coverage never causes a phantom report. Only the still-pending (e.g. timed-out) unit matters."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")  # a done (it had crashed earlier, recovered); b pending (timed out once)
    plan = _realplan(ctx)
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        coverage={(inst.id, "extract", "a"): 5, (inst.id, "extract", "b"): 1},  # a recovered (done); b 1 timeout
    )
    assert decision.reports == () and [k for lx in decision.launches for k in lx.keys] == ["b"]


def test_coverage_counts_every_terminal_run_job_covering_a_unit(tmp_path) -> None:
    """Two crashes (RETRY) and a COMPLETED job each attempted their units once: coverage counts them by the
    per-job manifest, with NO dependence on a `.failed.json` record (B's V5/28)."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    crash = _records()["exit_127"]["message"]
    live = [
        _ran_job(inst, plan, token, "j1", ["a"], outcome="ERROR", message=crash, store=store),  # worker-death
        _ran_job(inst, plan, token, "j2", ["a"], outcome="ERROR", message=crash, store=store),  # crash again
        _ran_job(inst, plan, token, "j3", ["b"], outcome="COMPLETED", store=store),  # completed (ran b)
    ]
    cov = _cover([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, attempts=3)
    assert cov == {(inst.id, "extract", "a"): 2, (inst.id, "extract", "b"): 1}


def test_coverage_excludes_timeouts(tmp_path) -> None:
    """A `Job timeout` (RERUN) does NOT count toward coverage: only crashes do. The 2-timeout STUCK rule is separate
    (HFU spec 13.5, replacing HFU-TIMEOUT-05's "rides"; see test_two_timeouts_make_a_unit_stuck)."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    timeout = _records()["job_timeout"]["message"]
    live = [
        _ran_job(inst, plan, token, f"t{i}", ["a"], outcome="ERROR", message=timeout, store=store) for i in range(5)
    ]
    cov = _cover([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, attempts=3)
    assert cov == {}  # five timeouts, zero coverage -> a (always coverage 0) rides, never spent


def test_coverage_attributes_a_job_to_the_exact_instance_sharing_a_run_key(tmp_path) -> None:
    """HFUD-IMPL-30: several instances share a run_key (it omits tier/variant), so a job must be credited to
    the EXACT (run_key, tier, variant). Variant A's crashes must not land on variant B even on the same token."""
    a = _inst(variant="A")
    b = _inst(variant="B")  # same run_key + tier, different variant
    plan, token = _one_stage(tmp_path, ["u"])  # same plan/token for both instances (equal-token case)
    plans = {a.id: plan, b.id: plan}
    store = _ManifestStore()
    crash = _records()["exit_127"]["message"]
    live = [_ran_job(a, plan, token, f"ja{i}", ["u"], outcome="ERROR", message=crash, store=store) for i in range(3)]
    cov = _cover([a, b], plans, live, frozenset(), lambda _i: store, attempts=3)
    assert cov == {(a.id, "extract", "u"): 3}  # all 3 on A; B gets nothing despite the shared run_key + token


def test_coverage_skips_baseline_and_jobs_that_never_ran(tmp_path) -> None:
    """A job in the baseline (a prior run) does not count, so a resume re-attempts (HFUD-IMPL-20). A FATAL
    (infra fault, never ran its units) and a CANCELED job never count either."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    crash, fatal = _records()["exit_127"]["message"], _records()["storage_403_started_set"]["message"]
    live = [
        _ran_job(inst, plan, token, "old", ["a"], outcome="ERROR", message=crash, store=store),  # baseline
        _ran_job(inst, plan, token, "fatal", ["a"], outcome="ERROR", message=fatal, store=store),  # FATAL: never ran
        _ran_job(inst, plan, token, "cancel", ["a"], outcome="CANCELED", store=store),  # canceled
        _ran_job(inst, plan, token, "new", ["a"], outcome="ERROR", message=crash, store=store),  # counts
    ]
    cov = _cover([inst], {inst.id: plan}, live, frozenset({"old"}), lambda _i: store, attempts=3)
    assert cov == {(inst.id, "extract", "a"): 1}  # only the new crash


def test_coverage_ignores_a_job_with_no_manifest(tmp_path) -> None:
    """A job whose per-job manifest is absent (launched outside the controller) is not attributed to any unit:
    ``step`` writes the manifest for every job it launches, so a missing one is not this run's work."""
    inst = _inst()
    plan, _token = _one_stage(tmp_path, ["a"])
    no_manifest = {
        "id": "j1",
        "stage": "ERROR",
        "message": _records()["exit_127"]["message"],
        "created_at": "2026-11-02T03:00:00+00:00",
        "labels": _labels(inst, "extract", shard="0of1"),
    }
    cov = _cover([inst], {inst.id: plan}, [no_manifest], frozenset(), lambda _i: _ManifestStore(), attempts=3)
    assert cov == {}  # no manifest -> no unit attribution


def test_a_tolerant_stage_reports_a_spent_pending_unit_rather_than_a_doomed_reduce(tmp_path) -> None:
    """HFUD-IMPL-32: a spent PENDING unit (coverage == attempts, but NO record -- a worker death that wrote
    nothing) cannot be reduced over; `reduce_stage` would re-run it and refuse it (rc 3). The driver reports it
    instead of launching a doomed reduce, even on a tolerant stage that would otherwise reduce over failures."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(), max_failures=None),))
    _run(ctx, "fit", "a")  # a done; b pending (its jobs crashed with no record)
    plan = _realplan(ctx)
    assert plan["stages"][0]["failed_units"] == {} and "b" in plan["stages"][0]["pending_units"]
    inst = _inst()
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        coverage={(inst.id, "fit", "b"): 3},  # b spent but has NO record
    )
    assert decision.launches == ()  # NOT a reduce
    assert len(decision.reports) == 1 and "no record" in decision.reports[0][1]


def test_manifest_id_matches_the_live_record_id(tmp_path) -> None:
    """A-Deux pin: the whole coverage model depends on the id `step` writes the manifest under being the same
    id the live record carries. `launch.launch` returns ``info.id`` and `ledger.job_record` reads the same
    ``info.id``, so they match by construction."""
    from scripts.hf_jobs import launch as launch_mod
    from scripts.hf_jobs.ledger import job_record

    _jobs_api = pytest.importorskip("huggingface_hub._jobs_api")
    info = _jobs_api.JobInfo(
        id="job-xyz",
        owner={"id": "u", "name": "silly-kicks", "type": "user"},
        status={"stage": "ERROR", "message": "Job failed with exit code: 1. Reason: Error"},
        labels={"stage": "extract"},
    )

    class _Api:
        def run_job(self, **_kw):
            return info

    returned = launch_mod.launch(_minimal_spec(), api=_Api(), token="t")
    assert returned == info.id == job_record(info)["id"] == "job-xyz"


def _minimal_spec():
    from scripts.hf_jobs.launch import JobSpec

    return JobSpec(
        campaign="cycle1",
        run_key="rk",
        job="j",
        variant="default",
        stages=("extract",),
        shard=(0, 1),
        module="m",
        args=(),
        commit=_COMMIT,
        bucket="silly-kicks/scratch",
        workers=8,
        inner_workers=1,
        cap_bytes=None,
        peak_rss_bytes=None,
        timeout_s=7200,
        attempts=3,
        datasets=(),
        extras=(),
    )


def _reduce_trigger_job(inst, plan, token, job_id, message, store) -> dict:
    """A terminal ERROR reduce trigger: an EMPTY per-job manifest (it ran no unit) + its ERROR record."""
    store.upload_bytes(by_job_rel(plan, "extract", token, job_id), json.dumps([]).encode("utf-8"))
    return {
        "id": job_id,
        "stage": "ERROR",
        "message": message,
        "created_at": "2026-11-02T03:00:00+00:00",
        "labels": _labels(inst, "extract", shard="0of1"),
    }


def test_reduce_crashes_in_the_baseline_reattempt(tmp_path) -> None:
    """HFUD-IMPL-24 (owner 2026-11-02): the reduce bound is baseline-recoverable too. A reduce trigger that
    crashed before this run started is in ``baseline_error_ids`` and is NOT counted, so a resume re-attempts
    it. ``_reduce_errors`` skips the baseline exactly as ``_coverage`` does."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    crash = _records()["exit_1"]["message"]
    live = [_reduce_trigger_job(inst, plan, token, f"r{i}", crash, store) for i in range(3)]
    seeded = frozenset(j["id"] for j in live)  # all terminal at resume start
    assert _rerr([inst], {inst.id: plan}, live, seeded, lambda _i: store) == {}  # all in baseline -> 0
    assert _rerr([inst], {inst.id: plan}, live, frozenset(), lambda _i: store) == {(inst.id, "extract"): 3}


def test_a_completed_reduce_trigger_that_left_the_stage_unreduced_is_bounded(tmp_path) -> None:
    """A reduce trigger can exit 0 without reducing (``try_reduce`` returns False while a memory-deferred unit
    waits). The stage stays un-reduced, so ``decide`` would relaunch the trigger on every step and ``run`` would
    never stall. ``_reduce_errors`` counts a COMPLETED trigger too: any terminal trigger of an un-reduced stage
    is a reduce that did not happen."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    live = []
    for i in range(3):
        jid = f"c{i}"
        store.upload_bytes(by_job_rel(plan, "extract", token, jid), json.dumps([]).encode("utf-8"))
        live.append(
            {
                "id": jid,
                "stage": "COMPLETED",
                "message": None,
                "created_at": "2026-11-02T03:00:00+00:00",
                "labels": _labels(inst, "extract", shard="0of1"),
            }
        )
    assert _rerr([inst], {inst.id: plan}, live, frozenset(), lambda _i: store) == {(inst.id, "extract"): 3}


def _download_inst(variant: str = "default") -> Instance:
    from scripts.hf_jobs.launch import DatasetMount

    return Instance("job", "public", variant, "mod", (), (DatasetMount("d", "ns/d", "rev", "download"),), "rk")


def test_a_download_instance_runs_round_robin_shards_not_keys(tmp_path) -> None:
    """A ``download`` dataset is fetched in-job by ``--shard I/N``, which a keyed launch omits. So, as main's
    controller did, its run shards are unkeyed round-robin slices; each records the slice's candidate units
    (what the executor will run), so coverage stays exact (reviews HFD1-DOC-05 / D1-IMPL-05, owner 2026-11-02)."""
    from scripts._driver import shard_select

    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(5)], work=_work()),))
    _run(ctx, "extract", s.int_key(0))  # unit 0 done: not a candidate
    plan = _realplan(ctx)
    units = plan["stages"][0]["units"]
    inst = _download_inst()
    decision = decide(instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=3, max_shards=12)
    assert [(lx.shard, lx.keyed) for lx in decision.launches] == [((0, 2), False), ((1, 2), False)]
    for lx in decision.launches:
        assert list(lx.keys) == [u for u in shard_select(units, *lx.shard) if u != s.int_key(0)]
    spec = _job_spec(_campaign(), _COMMIT, inst, decision.launches[1], plan, store=_FakeStore())
    assert spec.shard == (1, 2) and spec.shard_keys_path is None  # `--shard 1/2`, no keys file


def test_shard_slice_mirrors_shard_select() -> None:
    """``decide`` keeps a local copy of ``_driver.shard_select`` (its purity); the copy must not drift."""
    from scripts._driver import shard_select
    from scripts.hf_jobs.driver import _shard_slice

    seq = [s.int_key(i) for i in range(11)]
    assert all(_shard_slice(seq, i, n) == shard_select(seq, i, n) for n in (1, 3, 4) for i in range(n))


def test_a_download_instance_reduce_trigger_stays_keyed_on_an_empty_set(tmp_path) -> None:
    """The reduce trigger runs no unit, so it stays keyed (``[]``) even for a download instance."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    _run(ctx, "extract", "a")
    plan = _realplan(ctx)
    inst = _download_inst()
    decision = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12
    )
    assert [(lx.kind, lx.keyed, lx.keys) for lx in decision.launches] == [("reduce", True, ())]


def test_job_spec_gives_hf_one_attempt_so_the_driver_owns_retries(tmp_path) -> None:
    """D1-IMPL-04 (owner 2026-11-02): HF ``run_job(attempts=)`` retries inside a job would multiply the driver's
    per-unit bound (up to attempts**2 runs). Every job gets ONE HF attempt; ``campaign.attempts`` is the bound."""
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    spec = _job_spec(
        _campaign(), _COMMIT, inst, Launch(inst.id, ("extract",), (0, 1), "run", 8, ("a",)), plan, store=_FakeStore()
    )
    assert spec.attempts == 1 and _campaign().attempts == 3


def test_shard_keys_files_are_per_variant(tmp_path) -> None:
    """HFD1-CONC-07: sibling variants share a run_key and can share a stage token; their keys files must not
    collide, or one variant's write would replace the other's keys between their launches."""
    plan = _single_stage_plan(tmp_path)
    a, b = _inst(variant="A"), _inst(variant="B")
    store = _FakeStore()
    sa = _job_spec(_campaign(), _COMMIT, a, Launch(a.id, ("extract",), (0, 1), "run", 8, ("a",)), plan, store=store)
    sb = _job_spec(_campaign(), _COMMIT, b, Launch(b.id, ("extract",), (0, 1), "run", 8, ("b",)), plan, store=store)
    assert sa.shard_keys_path != sb.shard_keys_path and len(store.uploads) == 2
    assert "/A/0of1.json" in str(sa.shard_keys_path) and "/B/0of1.json" in str(sb.shard_keys_path)


def _waves_until_report(inst, plan, *, max_shards: int, max_waves: int) -> tuple[int, tuple]:
    """Drive ``decide`` -> launch -> a no-record crash (real ``exit_3`` RETRY) of every launched job ->
    ``_coverage``, wave after wave, as ``step`` would. Returns (wave, reports) at the first report, or
    (max_waves + 1, ()) if the driver never reports."""
    store = _ManifestStore()
    crash = _records()["exit_3"]["message"]
    jobs: list[dict] = []
    cov: dict = {}
    for wave in range(1, max_waves + 1):
        decision = decide(
            instances=[inst],
            plans={inst.id: plan},
            live_jobs=[],
            attempts=3,
            units_per_job=20,
            max_shards=max_shards,
            coverage=cov,
        )
        if decision.reports:
            return wave, decision.reports
        assert decision.launches, f"wave {wave}: no launch and no report"
        for i, lx in enumerate(decision.launches):
            jid = f"w{wave}-{i}"
            store.upload_bytes(
                by_job_rel(plan, "fit", plan["stages"][0]["token"], jid), json.dumps(list(lx.keys)).encode()
            )
            created = f"2026-11-02T03:{wave:02d}:{i:02d}+00:00"
            jobs.append(
                {
                    "id": jid,
                    "stage": "ERROR",
                    "message": crash,
                    "created_at": created,
                    "labels": _labels(inst, "fit", shard=f"{lx.shard[0]}of{lx.shard[1]}"),
                }
            )
        cov = _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store, attempts=3)
    return max_waves + 1, ()


@pytest.mark.parametrize("download", [True, False])
def test_a_no_record_crasher_beside_a_recorded_failure_is_reported_in_bounded_waves(tmp_path, download) -> None:
    """D1-IMPL-10: a round-robin slice re-runs a spent recorded failure F every wave. If every crash were charged
    to F alone, the no-record crasher X would never be charged and the shard would relaunch for ever. Every RETRY
    must raise the coverage of a not-yet-spent unit it held, so X is reported within 2 x attempts waves -- on the
    download (round-robin) path and on the keyed path alike."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["F", "X"], work=_work(fail=("F",)), max_failures=None),))
    _run(ctx, "fit", "F")  # F: a recorded failure; X: pending, its jobs crash with no record
    plan = _realplan(ctx)
    assert plan["stages"][0]["failed_units"] == {"F": 1} and plan["stages"][0]["pending_units"] == ["X"]
    inst = _download_inst() if download else _inst()
    wave, reports = _waves_until_report(inst, plan, max_shards=1, max_waves=12)
    assert wave <= 2 * 3 + 1, f"never reported within 12 waves (download={download})"
    assert len(reports) == 1 and "no record" in reports[0][1]


def test_a_strict_abort_charges_only_the_recorded_failure(tmp_path) -> None:
    """D1-IMPL-07: a strict stage aborts its job (rc 3) on a recorded failure, so the job's co-sharded units never
    ran. Coverage charges the recorded unit only; the never-run unit is not spent toward a false "crashed with no
    record" report. A crash with NO recorded unit still charges every unit it held."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("a",))),))
    _run(ctx, "fit", "a")  # a recorded failed; b never ran (the abort)
    plan = _realplan(ctx)
    token = plan["stages"][0]["token"]
    assert plan["stages"][0]["failed_units"] == {"a": 1} and "b" in plan["stages"][0]["pending_units"]
    inst = _inst()
    store = _ManifestStore()
    abort = _records()["exit_3"]["message"]
    jobs = []
    for jid in ("j1", "j2"):
        store.upload_bytes(by_job_rel(plan, "fit", token, jid), json.dumps(["a", "b"]).encode("utf-8"))
        jobs.append(
            {
                "id": jid,
                "stage": "ERROR",
                "message": abort,
                "created_at": "2026-11-02T03:00:00+00:00",
                "labels": _labels(inst, "fit"),
            }
        )
    cov = _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store, attempts=3)
    assert cov == {(inst.id, "fit", "a"): 2}  # b uncharged: it never ran


def test_a_completed_run_job_charges_its_recorded_failure(tmp_path) -> None:
    """HFU spec §4.2 (T6): a run job that records a unit failure now exits 0 (COMPLETED, no email). Coverage still
    charges every unit the job held, so a unit that keeps failing exhausts its attempts and is reported; until
    then the failed unit stays a launch candidate."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("a",))),))
    _run(ctx, "fit", "a")  # a recorded failed; b still pending
    plan = _realplan(ctx)
    token = plan["stages"][0]["token"]
    inst = _inst()
    store = _ManifestStore()
    store.upload_bytes(by_job_rel(plan, "fit", token, "j1"), json.dumps(["a", "b"]).encode("utf-8"))
    job = {"id": "j1", "stage": "COMPLETED", "created_at": "2026-11-02T03:00:00+00:00", "labels": _labels(inst, "fit")}
    cov = _cover([inst], {inst.id: plan}, [job], frozenset(), lambda _i: store, attempts=3)
    assert cov == {(inst.id, "fit", "a"): 1, (inst.id, "fit", "b"): 1}
    decision = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12, coverage=cov
    )
    assert "a" in {k for lx in decision.launches for k in lx.keys}


def test_an_empty_shard_keys_list_runs_no_unit(tmp_path) -> None:
    """HFD1-TEST-08: the reduce trigger depends on main's executor running EXACTLY the listed keys, so ``[]``
    runs nothing (``LocalExecutor.run_stage(keys=[])``) -- every unit stays as it was."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")
    before = _realplan(ctx)["stages"][0]
    report = LocalExecutor(ExecutorConfig(workers=1)).run_stage(ctx, "extract", keys=[])
    after = _realplan(ctx)["stages"][0]
    assert report.n_mine == 0 and report.workers == 0
    assert (after["done"], after["pending_units"]) == (before["done"], before["pending_units"]) == (1, ["b"])


def test_reduce_timeouts_are_bounded_then_reported(tmp_path) -> None:
    """HFU-TIMEOUT (owner 2026-11-02): a UNIT timeout rides (``_coverage`` excludes RERUN), but a REDUCE is
    deterministic whole-stage work, so a reduce timeout IS bounded -- ``_reduce_errors`` counts a reduce trigger
    (empty manifest) that TIMED OUT. ``attempts`` of them make ``decide`` report the stage."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    _run(ctx, "extract", "a")  # all done -> the reduce path
    plan = _realplan(ctx)
    token = plan["stages"][0]["token"]
    inst = _inst()
    store = _ManifestStore()
    timeout = _records()["job_timeout"]["message"]
    live = [_reduce_trigger_job(inst, plan, token, f"r{i}", timeout, store) for i in range(3)]
    red = _rerr([inst], {inst.id: plan}, live, frozenset(), lambda _i: store)
    assert red == {(inst.id, "extract"): 3}  # a reduce timeout IS counted (a unit timeout would not be)
    decision = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        reduce_errors=red,
    )
    assert decision.launches == () and len(decision.reports) == 1 and "reduce" in decision.reports[0][1]


# --- job spec (real plan for token/sizing) ------------------------------------


_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class _FakeStore:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_bytes(self, rel: str, data: bytes) -> None:
        self.uploads[rel] = data


def _single_stage_plan(tmp_path, *, inner_parallel=False, inner_width=1) -> dict:
    stage = s.Stage(
        name="extract",
        units=lambda c: ["a", "b", "c"],
        work=_work(),
        inner_parallel=inner_parallel,
        inner_width=inner_width,  # ignored unless inner_parallel (Stage default 1)
    )
    return _realplan(_ctx(tmp_path, (stage,)))


def test_job_spec_run_shard_is_valid_and_sharded(tmp_path) -> None:
    from scripts.hf_jobs.launch import FLAVOR, labels_for

    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    launch = Launch(inst.id, ("extract",), (1, 3), "run", 8, ("b", "c"))
    store = _FakeStore()
    spec = _job_spec(_campaign(), _COMMIT, inst, launch, plan, store=store)
    labels = labels_for(spec)  # must not raise (length / pattern)
    assert labels["stage"] == "extract" and labels["shard"] == "1of3"
    assert FLAVOR == "cpu-upgrade"
    assert (spec.workers, spec.inner_workers, spec.timeout_s) == (8, 1, 7200)
    assert spec.shard_keys_path is not None
    assert store.uploads


def test_job_spec_reduce_trigger_stages_an_empty_key_set(tmp_path) -> None:
    """A reduce trigger is a run over an EMPTY ``--shard-keys``: the executor runs no unit and reduces the
    stage inline (main's engine, unchanged). So the spec carries a ``shard_keys_path`` to a staged ``[]``."""
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    launch = Launch(inst.id, ("extract",), (0, 1), "reduce", 8, ())
    store = _FakeStore()
    spec = _job_spec(_campaign(), _COMMIT, inst, launch, plan, store=store)
    assert spec.shard == (0, 1) and spec.shard_keys_path is not None
    assert len(store.uploads) == 1  # the empty shard-keys file was staged
    assert json.loads(next(iter(store.uploads.values()))) == []  # [] -> executor runs nothing, reduces


def test_job_spec_inner_parallel_sizing(tmp_path) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path, inner_parallel=True, inner_width=5)
    launch = Launch(inst.id, ("extract",), (0, 1), "run", 8, ("a",))
    spec = _job_spec(_campaign(), _COMMIT, inst, launch, plan, store=_FakeStore())
    assert spec.workers == 1 and spec.inner_workers == 5  # the inner unit runs alone, box threads inside


# --- step / run / main --------------------------------------------------------


class _LaunchRecorder:
    def __init__(self) -> None:
        self.specs: list = []

    def __call__(self, spec, *, api, token) -> str:
        self.specs.append(spec)
        return f"id{len(self.specs)}"


class _FakeApi:
    def __init__(self) -> None:
        self.uploads: list = []

    def batch_bucket_files(self, bucket, add=None, **kwargs) -> None:
        self.uploads.append((bucket, add))

    def download_bucket_files(self, bucket, pairs, raise_on_missing_files=False) -> None:
        pass  # empty bucket: nothing downloaded, so BucketStore.read_bytes returns None (fix-3 cache absent)


def _open_gate() -> None:
    """A passing launch gate for the step tests; the real gate has its own tests below."""


def test_step_refuses_at_the_launch_gate_before_anything(tmp_path) -> None:
    """HFD1-GATE-01 / D1-IMPL-01: without a passing preflight report for THIS commit and campaign hash, a step
    launches nothing -- it refuses before reading a plan or listing jobs."""
    recorder = _LaunchRecorder()

    def refuse():
        raise SystemExit("no preflight report for cycle1@0123456: run `campaign <file> preflight` first")

    with pytest.raises(SystemExit, match="preflight"):
        step(
            _campaign(),
            _COMMIT,
            api=_FakeApi(),
            token="t",
            instances=[_inst()],
            read_plan=lambda _i: pytest.fail("the gate must refuse before any plan is read"),
            live_records=lambda: pytest.fail("the gate must refuse before any job is listed"),
            launch_fn=recorder,
            gate=refuse,
        )
    assert recorder.specs == []


def test_the_default_gate_is_require_preflight_for_this_commit_and_campaign(monkeypatch) -> None:
    """The real gate is ``preflight.require_preflight(name, commit, campaign sha256)`` on the bucket root."""
    from scripts.hf_jobs import preflight
    from scripts.hf_jobs.driver import _default_gate

    seen: dict = {}
    monkeypatch.setattr(
        preflight, "require_preflight", lambda name, commit, sha, *, store: seen.update(a=(name, commit, sha, store))
    )
    camp = _campaign()
    _default_gate(camp, _COMMIT, _FakeApi())()
    name, commit, sha, store = seen["a"]
    assert (name, commit, sha) == ("cycle1", _COMMIT, camp.sha256)
    assert (store.bucket, store.prefix) == (camp.bucket, "")


def test_a_failing_plan_is_reported_and_the_other_instances_still_launch(tmp_path) -> None:
    """HFD1-ROBUST-03: one instance's ``--plan`` failing (``campaign.read_plan`` raises RuntimeError) must not stop
    the step for every instance. It becomes a report; the healthy instance launches."""
    ok, bad = _inst(variant="ok"), _inst(variant="bad")
    plan = _single_stage_plan(tmp_path)

    def read(inst):
        if inst is bad:
            raise RuntimeError(f"{inst.id}: --plan failed (rc=1): boom")
        return plan

    recorder = _LaunchRecorder()
    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[ok, bad],
        read_plan=read,
        live_records=lambda: [],
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.launched == 1 and result.done is False  # the 3-unit stage is ONE sized job (fix 1)
    assert [r[0] for r in result.reports] == [bad.id] and "--plan failed" in result.reports[0][1]


def test_step_reads_the_plans_in_parallel(tmp_path, monkeypatch) -> None:
    """Owner decision 2026-11-05: a step read the 21 ``--plan`` one after another (minutes, silent) before it
    launched anything. The plans are independent, so they are read in parallel."""
    import time as _time

    from scripts.hf_jobs import driver as driver_mod

    monkeypatch.setattr(driver_mod, "PLAN_WORKERS", 6)  # B HFURR-IMPL-08: never the host's CPU count

    insts = [_inst(variant=f"v{i}") for i in range(6)]
    plan = _single_stage_plan(tmp_path)

    def slow(_inst):
        _time.sleep(0.5)
        return plan

    t0 = _time.monotonic()
    step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=insts,
        read_plan=slow,
        live_records=lambda: [],
        launch_fn=_LaunchRecorder(),
        gate=_open_gate,
    )
    assert _time.monotonic() - t0 < 2.0  # serial would be >= 3 s


def test_a_rate_limited_plan_read_backs_off_instead_of_reporting_stuck(tmp_path) -> None:
    """Impl review B HFURR-IMPL-07: a 429 inside a ``--plan`` subprocess surfaces as a RuntimeError; it must reach
    the contention back-off, not become a STUCK report (with nothing in flight, ``run`` would then return 1)."""
    insts = [_inst(variant="a"), _inst(variant="b")]
    plan = _single_stage_plan(tmp_path)

    def read(inst):
        if inst is insts[1]:
            raise RuntimeError(f"{inst.id}: --plan failed (rc=1): HfHubHTTPError: 429 Too Many Requests")
        return plan

    recorder = _LaunchRecorder()
    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=insts,
        read_plan=read,
        live_records=lambda: [],
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.contention is True and result.launched == 0 and result.reports == ()
    assert recorder.specs == []


def test_a_plan_failure_that_only_contains_429_is_still_stuck(tmp_path) -> None:
    """Impl review B HFURR-IMPL-09: the --plan error carries up to 2000 chars of stderr, so a bare "429" (a game id,
    a line number) must not read as a rate limit -- that would back off forever and never report the instance."""
    inst = _inst(variant="a")

    def read(_inst):
        raise RuntimeError(
            f"{inst.id}: --plan failed (rc=1): File \"x.py\", line 429, in f\nKeyError: 'statsbomb__3869429'"
        )

    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=read,
        live_records=lambda: [],
        launch_fn=_LaunchRecorder(),
        gate=_open_gate,
    )
    assert result.contention is False
    assert [r[0] for r in result.reports] == [inst.id] and "--plan failed" in result.reports[0][1]


def test_step_prints_one_timed_progress_line(tmp_path, capsys) -> None:
    ok, bad = _inst(variant="ok"), _inst(variant="bad")
    plan = _single_stage_plan(tmp_path)

    def read(inst):
        if inst is bad:
            raise RuntimeError(f"{inst.id}: --plan failed (rc=1): boom")
        return plan

    step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[ok, bad],
        read_plan=read,
        live_records=lambda: [],
        launch_fn=_LaunchRecorder(),
        gate=_open_gate,
    )
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] step")]
    assert len(lines) == 1, lines
    assert re.fullmatch(
        r"\[driver\] step: plans 2 in \d+\.\ds \(1 failed\); in flight 0; launched 1; stuck 1", lines[0]
    ), lines


def test_plan_reports_keep_the_instance_order_when_read_in_parallel(tmp_path) -> None:
    import time as _time

    insts = [_inst(variant=f"v{i}") for i in range(4)]

    def read(inst):
        _time.sleep(0.3 - 0.1 * insts.index(inst) if insts.index(inst) < 3 else 0)  # finish in reverse order
        raise RuntimeError(f"{inst.id}: --plan failed (rc=1): boom")

    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=insts,
        read_plan=read,
        live_records=lambda: [],
        launch_fn=_LaunchRecorder(),
        gate=_open_gate,
    )
    assert [r[0] for r in result.reports] == [i.id for i in insts]


def test_dry_run_decides_on_the_same_coverage_as_launch(tmp_path, monkeypatch, capsys) -> None:
    """HFD1-DRYRUN-04: a unit that the run's jobs already covered ``attempts`` times is reported by ``launch``;
    the dry run must show the same report, not a relaunch."""
    from scripts.hf_jobs import store as store_mod
    from scripts.hf_jobs.driver import _dry_run

    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")  # b pending; three jobs covered it and crashed with no record
    plan = _realplan(ctx)
    inst = _inst()
    manifests = _ManifestStore()
    crash = _records()["exit_127"]["message"]
    live = [
        _ran_job(
            inst, plan, plan["stages"][0]["token"], f"j{i}", ["b"], outcome="ERROR", message=crash, store=manifests
        )
        for i in range(3)
    ]
    monkeypatch.setattr(store_mod, "BucketStore", lambda *a, **k: manifests)
    _dry_run(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: live,
        baseline_error_ids=frozenset(),
    )
    out = json.loads(capsys.readouterr().out)
    assert out["launches"] == [] and len(out["reports"]) == 1 and "no record" in out["reports"][0][1]


def test_step_launches_what_decide_decides(tmp_path) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    recorder, api = _LaunchRecorder(), _FakeApi()
    result = step(
        _campaign(),
        _COMMIT,
        api=api,
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [],
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.fatal is None and result.done is False and result.launched == 1  # 3 units -> 1 sized job
    assert all(spec.stages == ("extract",) for spec in recorder.specs)
    assert api.uploads
    # HFUD-IMPL-28: each run launch also stages a per-job partition manifest keyed by the returned job id,
    # so a later terminated job can be attributed to its units for the coverage bound.
    paths = [path for _bucket, add in api.uploads for _data, path in (add or [])]
    assert any("/by_job/" in p for p in paths)


def test_step_enforces_dispatch_cap(tmp_path) -> None:
    """HFUD-IMPL-08: one step launches at most ``dispatch_cap`` jobs (minus those already in flight)."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    recorder = _LaunchRecorder()
    result = step(
        _campaign(concurrency_cap=20, units_per_job=1, max_shards=100),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [],
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.launched == 20 and len(recorder.specs) == 20  # 25 one-unit shards capped to the dispatch budget


def test_step_stops_on_real_fatal_before_launching(tmp_path) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    fatal = _fatal_rec(inst, "extract", created="2026-11-02T03:00:00+00:00")
    recorder = _LaunchRecorder()
    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [fatal],
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.fatal is not None and result.fatal[0] == "storage_limit_403"
    assert result.launched == 0 and recorder.specs == []


def test_step_recovers_when_the_fatal_is_in_the_baseline(tmp_path) -> None:
    """HFUD-IMPL-05/r2: a resume passes ``run``'s seeded baseline through ``step``. With the pre-existing 403
    in the baseline, the step launches the pending work instead of stopping -- the reachable recovery."""
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    fatal = _fatal_rec(inst, "extract", created="2026-11-02T03:00:00+00:00")
    recorder = _LaunchRecorder()
    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [fatal],
        launch_fn=recorder,
        gate=_open_gate,
        baseline_error_ids=frozenset({fatal["id"]}),
    )
    assert result.fatal is None and result.launched == 1 and recorder.specs  # 3 units -> 1 sized job


def test_step_dispatch_cap_subtracts_in_flight(tmp_path) -> None:
    """HFUD-IMPL-18: the budget is dispatch_cap MINUS jobs already in flight. Five unrelated RUNNING jobs
    leave room for 15 of the 25 shards."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    running = [_running(inst, "warmup", shard=f"{i}of5", created="2026-11-02T03:00:00+00:00") for i in range(5)]
    recorder = _LaunchRecorder()
    result = step(
        _campaign(concurrency_cap=20, units_per_job=1, max_shards=100),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: running,  # 5 in flight (a different stage, so they do not block extract)
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.in_flight == 5 and result.launched == 15  # 20 - 5 (25 one-unit shards, budget 15)


def test_seed_baseline_collects_every_terminal_id() -> None:
    """The baseline is EVERY pre-existing terminal job (HFUD-IMPL-20/24/28): a resume looks past the whole prior
    history -- FATALs do not block, and prior crashes / timeouts / completions do not count toward this run's
    per-unit coverage. An in-flight job is NOT seeded (it finishes during the run and counts)."""
    inst = _inst()
    fatal = _fatal_rec(inst, "extract", created="2026-11-02T03:00:00+00:00")
    crash = _errored(inst, "extract", _records()["exit_127"]["message"], job_id="crash1")
    done = _terminal(inst, "extract", "COMPLETED", created="2026-11-02T03:00:00+00:00")
    done["id"] = "ok1"
    timeout = _errored(inst, "extract", _records()["job_timeout"]["message"], job_id="to1")
    running = _running(inst, "extract")
    running["id"] = "run1"
    live = [fatal, crash, done, timeout, running]
    baseline = _seed_baseline(_campaign(), _COMMIT, api=_FakeApi(), live_records=lambda: live)
    assert baseline == frozenset({fatal["id"], "crash1", "ok1", "to1"})  # every terminal; NOT in-flight run1


def test_run_seeds_baseline_and_passes_it_to_step() -> None:
    """``run`` seeds the baseline once and threads it into every step (so a resume looks past a resolved
    fault for its whole loop)."""
    seen: dict = {}

    def step_fn(campaign, commit, **kwargs) -> StepResult:
        seen["baseline"] = kwargs.get("baseline_error_ids")
        return _res(done=True)

    rc = run(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        step_fn=step_fn,
        sleep_fn=_Sleeps(),
        baseline_fn=lambda: frozenset({"old-403"}),
    )
    assert rc == 0 and seen["baseline"] == frozenset({"old-403"})


def _res(
    *,
    launched: int = 0,
    done: bool = False,
    in_flight: int = 0,
    fatal=None,
    reports=(),
    contention: bool = False,
    mount_faults: int = 0,
) -> StepResult:
    return StepResult(
        fatal=fatal,
        launched=launched,
        done=done,
        reports=reports,
        in_flight=in_flight,
        contention=contention,
        mount_faults=mount_faults,
    )


def _scripted(results: list[StepResult]):
    it = iter(results)

    def step_fn(campaign, commit, **kwargs) -> StepResult:
        return next(it)

    return step_fn


class _Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_run_loops_until_done() -> None:
    sleeps = _Sleeps()
    results = [_res(launched=3, in_flight=0), _res(launched=0, in_flight=3), _res(done=True)]
    rc = run(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        poll_s=5,
        step_fn=_scripted(results),
        sleep_fn=sleeps,
        baseline_fn=lambda: frozenset(),
    )
    assert rc == 0 and sleeps.calls == [5, 5]


def test_run_stops_on_fatal_without_sleeping() -> None:
    sleeps = _Sleeps()
    rc = run(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        step_fn=_scripted([_res(fatal=("storage_limit_403", "detail"))]),
        sleep_fn=sleeps,
        baseline_fn=lambda: frozenset(),
    )
    assert rc == 2 and sleeps.calls == []


def test_run_returns_1_when_stalled() -> None:
    sleeps = _Sleeps()
    rc = run(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        step_fn=_scripted([_res(launched=0, in_flight=0, done=False, reports=(("id", "stuck"),))]),
        sleep_fn=sleeps,
        baseline_fn=lambda: frozenset(),
    )
    assert rc == 1 and sleeps.calls == []


def test_main_launch_routes_to_run_loop() -> None:
    seen: dict = {}

    def fake_runner(campaign, commit, *, api, token, replica, poll_s) -> int:
        seen.update(commit=commit, poll_s=poll_s, replica=replica)
        return 7

    rc = main(
        ["camp.toml", "launch", "--poll-s", "5"],
        load=lambda _f: _campaign(),
        make_api=lambda: (_FakeApi(), "tok"),
        head=lambda: _COMMIT,
        runner=fake_runner,
    )
    assert rc == 7 and seen == {"commit": _COMMIT, "poll_s": 5.0, "replica": None}


def _main_args(*argv):
    return dict(load=lambda _f: _campaign(), make_api=lambda: (_FakeApi(), "tok"), head=lambda: _COMMIT)


def test_main_preflight_routes_to_cmd_preflight(monkeypatch, capsys) -> None:
    from scripts.hf_jobs import campaign as camp

    monkeypatch.setattr(camp, "cmd_preflight", lambda c, commit, *, api: {"ok": True})
    rc = main(["camp.toml", "preflight"], **_main_args())
    assert rc == 0 and "ok" in capsys.readouterr().out


def test_main_preflight_returns_1_when_not_ok(monkeypatch) -> None:
    from scripts.hf_jobs import campaign as camp

    monkeypatch.setattr(camp, "cmd_preflight", lambda c, commit, *, api: {"ok": False})
    assert main(["camp.toml", "preflight"], **_main_args()) == 1


def test_main_status_routes_to_cmd_status(monkeypatch) -> None:
    from scripts.hf_jobs import campaign as camp

    seen: dict = {}
    monkeypatch.setattr(camp, "cmd_status", lambda c, commit, *, api, replica: seen.setdefault("hit", True) or [])
    assert main(["camp.toml", "status"], **_main_args()) == 0 and seen["hit"]


def test_main_cancel_routes_to_cmd_cancel(monkeypatch) -> None:
    from scripts.hf_jobs import campaign as camp

    seen: dict = {}
    monkeypatch.setattr(camp, "cmd_cancel", lambda c, commit, *, api: seen.setdefault("hit", True) or [])
    assert main(["camp.toml", "cancel"], **_main_args()) == 0 and seen["hit"]


def test_main_adopt_family_requires_staging() -> None:
    with pytest.raises(SystemExit, match="staging"):
        main(["camp.toml", "retrieve"], **_main_args())


def test_main_retrieve_routes_to_adopt(monkeypatch, tmp_path) -> None:
    from scripts.hf_jobs import adopt

    seen: dict = {}

    def fake_retrieve(c, commit, *, staging, api, replica):
        seen["hit"] = True
        return {}

    monkeypatch.setattr(adopt, "retrieve", fake_retrieve)
    assert main(["camp.toml", "retrieve", "--staging", str(tmp_path)], **_main_args()) == 0 and seen["hit"]


def test_package_main_routes_campaign_to_the_driver(monkeypatch) -> None:
    """``python -m scripts.hf_jobs campaign ...`` routes to the driver's ``main`` -- the minimal driver is the
    controller now (ADR-128), not the old campaign CLI."""
    import scripts.hf_jobs.__main__ as pkg_main
    from scripts.hf_jobs import driver

    seen: dict = {}
    monkeypatch.setattr(driver, "main", lambda rest: (seen.setdefault("rest", rest), 0)[1])
    rc = pkg_main.main(["campaign", "camp.toml", "status"])
    assert rc == 0 and seen["rest"] == ["camp.toml", "status"]


# --- fix 1: bounded job-sizing (few big jobs, not cap-many) -----------------------------------------


def test_job_sizing_few_big_jobs(tmp_path) -> None:
    """Fix 1: a 50-unit stage launches ceil(50/units_per_job)=3 jobs, not 50 (or the concurrency cap)."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(50)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    d = decide(instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12)
    assert len(d.launches) == 3
    assert sorted(k for lx in d.launches for k in lx.keys) == sorted(s.int_key(i) for i in range(50))  # disjoint cover


def test_job_sizing_respects_max_shards(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(500)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    d = decide(instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12)
    assert len(d.launches) == 12  # ceil(500/20)=25 capped at max_shards=12


# --- fix 2: per-dataset in-flight mount cap ---------------------------------------------------------


def _mount_inst(dataset: str = "tc", variant: str = "default", run_key: str = "rk") -> Instance:
    from scripts.hf_jobs.launch import DatasetMount

    mount = DatasetMount(dataset, f"ns/{dataset}", "r", "mount", 4)
    return Instance("job", "owner", variant, "mod", (), (mount,), run_key)


def test_mount_cap_limits_concurrent_mounts_in_one_pass(tmp_path) -> None:
    """Fix 2 / B-SPEC-02: five instances mount one dataset; mount_cap=2 => only 2 launch in ONE decide pass
    (same-pass counting, not just pre-existing in-flight)."""
    plan = _single_stage_plan(tmp_path)
    insts = [_mount_inst(variant=f"v{i}", run_key=f"rk{i}") for i in range(5)]
    d = decide(
        instances=insts,
        plans={i.id: plan for i in insts},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        mount_cap_for=lambda ds: 2 if ds == "tc" else 10**9,
    )
    assert len(d.launches) == 2


def test_mount_cap_counts_in_flight(tmp_path) -> None:
    plan = _single_stage_plan(tmp_path)
    insts = [_mount_inst(variant=f"v{i}", run_key=f"rk{i}") for i in range(3)]
    inflight = _running(insts[0], "extract")  # one tc mount already in flight
    d = decide(
        instances=insts,
        plans={i.id: plan for i in insts},
        live_jobs=[inflight],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        mount_cap_for=lambda ds: 2 if ds == "tc" else 10**9,
    )
    assert len(d.launches) == 1  # in-flight(1) + v1 chosen(1) == cap 2; v2 blocked


def test_mount_cap_ignores_open_and_download(tmp_path) -> None:
    from scripts.hf_jobs.launch import DatasetMount

    plan = _single_stage_plan(tmp_path)
    openi = Instance("job", "public", "default", "mod", (), (DatasetMount("sb", "ns/sb", "r", "open"),), "rko")
    d = decide(
        instances=[openi],
        plans={openi.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        mount_cap_for=lambda _ds: 0,  # cap 0 for everything -- an open dataset is not CSI-mounted, so not capped
    )
    assert len(d.launches) == 1


def test_mount_cap_counts_every_mount_dataset(tmp_path) -> None:
    """A-PLAN-02: an xcross instance mounts TWO (dataset + probe); a cap on the SHARED probe limits all of them."""
    from scripts.hf_jobs.launch import DatasetMount

    plan = _single_stage_plan(tmp_path)

    def xc(v: str, rk: str) -> Instance:
        mounts = (DatasetMount("xshot", "ns/x", "r", "mount", 4), DatasetMount("xprobe", "ns/p", "r", "mount", 4))
        return Instance("xcross", "public", v, "mod", (), mounts, rk)

    insts = [xc(f"v{i}", f"rk{i}") for i in range(4)]
    d = decide(
        instances=insts,
        plans={i.id: plan for i in insts},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        mount_cap_for=lambda ds: 2 if ds == "xprobe" else 10**9,
    )
    assert len(d.launches) == 2  # capped by the shared probe dataset across the 4 instances


# --- fix 3: by-locus attribution + fold-order-exact recompute ---------------------------------------


def test_coverage_mount_flake_is_per_dataset_not_per_unit(tmp_path) -> None:
    """Owner by-locus ruling: a mount_auth_flake (RETRY, like unit_crash, separated only by reason) bumps a
    per-DATASET fault tally, NEVER the unit's attempts."""
    inst = _mount_inst(dataset="tc")
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    flake = _records()["mount_401_auth"]["message"]
    jobs = [_ran_job(inst, plan, token, f"f{i}", ["a"], outcome="ERROR", message=flake, store=store) for i in range(3)]
    assert _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {}
    assert _mfaults([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {"tc": 3}


def test_coverage_bare_mount_failure_is_per_dataset_not_per_unit(tmp_path) -> None:
    """A generic `mount_failed` (RETRY, the catch-all transient mount fault) routes like `mount_auth_flake`:
    a per-DATASET fault tally, NEVER the unit's attempts -- a transient mount fault is not the unit's fault."""
    inst = _mount_inst(dataset="tc")
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    msg = 'Volume mount failed: MountVolume.SetUp failed for volume "hf-vol-0": sidecar not ready'
    jobs = [_ran_job(inst, plan, token, f"m{i}", ["a"], outcome="ERROR", message=msg, store=store) for i in range(3)]
    assert _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {}
    assert _mfaults([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {"tc": 3}


def test_mount_outage_reports_over_budget(tmp_path) -> None:
    inst = _mount_inst(dataset="tc")
    plan = _single_stage_plan(tmp_path)
    d = decide(
        instances=[inst],
        plans={inst.id: plan},
        live_jobs=[],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        mount_faults={"tc": 5},
        mount_fault_budget=4,
        mount_cap_for=lambda _d: 10**9,
    )
    assert any(iid == "dataset:tc" and "mount outage" in reason for iid, reason in d.reports)


def test_coverage_is_fold_order_exact(tmp_path) -> None:
    """B-SPEC-01: the recompute MUST sort by created_at. Here `a` is a recorded failure; one job covers `[a]`
    (early) and one covers `[a,b]` (late), at attempts=1. Under the recorded-first + not-yet-spent rule, whether
    `b` is charged depends on PROCESSING order -- so a fold that keeps insertion/termination order (not
    created_at) diverges. The created_at sort makes both fold orders give the same, correct tally."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work(fail=("a",))),))
    _run(ctx, "extract", "a")  # a is a RECORDED failure
    plan = _realplan(ctx)
    token = plan["stages"][0]["token"]
    inst = _inst()
    crash = _records()["exit_3"]["message"]

    def cov(order):
        store = _ManifestStore()
        jobs = []
        for jid, keys, ts in order:
            j = _ran_job(inst, plan, token, jid, keys, outcome="ERROR", message=crash, store=store)
            j["created_at"] = ts
            jobs.append(j)
        return _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store, attempts=1)

    jx = ("jx", ["a"], "2026-11-02T03:00:01+00:00")  # early: covers a
    jj = ("jj", ["a", "b"], "2026-11-02T03:00:09+00:00")  # late: covers a,b
    # created_at order jx->jj: jx spends a; jj finds a spent -> charges b. b IS charged exactly once.
    assert cov([jx, jj]) == cov([jj, jx]) == {(inst.id, "extract", "a"): 1, (inst.id, "extract", "b"): 1}


def test_coverage_cache_rebuild_identity(tmp_path) -> None:
    import scripts.hf_jobs.driver as drv

    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    crash = _records()["exit_127"]["message"]
    store = _ManifestStore()
    jobs = [_ran_job(inst, plan, token, f"j{i}", ["a"], outcome="ERROR", message=crash, store=store) for i in range(2)]
    first = _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store)
    store.files.pop(drv._COVERAGE_STATE, None)  # wipe the cache (keep the manifests): a rebuild must be identical
    assert _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == first


# --- fix 4: memory-aware sizing + escalation --------------------------------------------------------


def test_escalate_launches_memory_units_solo(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",)), max_failures=None),))
    _run(ctx, "fit", "a")  # a done
    _run(ctx, "fit", "b")  # b failed
    plan = _realplan(ctx)
    plan["stages"][0]["memory_units"] = ["b"]  # b deferred for memory (the engine's plan field; fix 4 reads it)
    inst = _inst()
    d = decide(instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12)
    assert [(lx.escalate, lx.workers, lx.keys) for lx in d.launches] == [(True, 1, ("b",))]


def test_job_spec_escalate_is_solo_whole_box(tmp_path) -> None:
    from scripts.hf_jobs.driver import _BOX_RAM, _HEADROOM
    from scripts.hf_jobs.launch import _run_argv

    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    launch = Launch(inst.id, ("extract",), (0, 1), "run", 1, ("a",), escalate=True)
    spec = _job_spec(_campaign(), _COMMIT, inst, launch, plan, store=_FakeStore())
    assert spec.escalate is True and spec.workers == 1 and spec.cap_bytes == _BOX_RAM - _HEADROOM
    assert "--escalate" in _run_argv(spec) and "--cap-bytes" in _run_argv(spec)


def test_job_spec_peak_hint_caps_and_passes_argv(tmp_path) -> None:
    from scripts.hf_jobs.launch import _run_argv

    inst = _inst()  # job "job"
    plan = _single_stage_plan(tmp_path)  # stage "extract"
    camp = _campaign(peak_rss_by_stage={"job/extract": 8 * 1024**3})
    launch = Launch(inst.id, ("extract",), (0, 3), "run", 8, ("a",))
    spec = _job_spec(camp, _COMMIT, inst, launch, plan, store=_FakeStore())
    assert spec.peak_rss_bytes == 8 * 1024**3 and spec.cap_bytes == 8 * 1024**3
    argv = _run_argv(spec)
    assert "--peak-rss-bytes" in argv and "--cap-bytes" in argv


# --- fix 5: launch pace + contention backoff --------------------------------------------------------


def test_step_paces_launches_to_launch_batch(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    recorder = _LaunchRecorder()
    result = step(
        _campaign(concurrency_cap=100, units_per_job=1, max_shards=100, launch_batch=8),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [],
        launch_fn=recorder,
        gate=_open_gate,
    )
    assert result.launched == 8  # 25 one-unit shards, paced to launch_batch=8


def test_run_backs_off_on_contention_then_resets() -> None:
    sleeps = _Sleeps()
    results = [_res(launched=1, contention=True), _res(launched=1, contention=True), _res(launched=1), _res(done=True)]
    rc = run(
        _campaign(backoff_max_s=300.0),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        poll_s=5,
        step_fn=_scripted(results),
        sleep_fn=sleeps,
        baseline_fn=lambda: frozenset(),
    )
    assert rc == 0 and sleeps.calls == [10, 20, 5]  # 5*2, 10*2, then a clean step resets to 5


def test_step_launch_429_signals_contention(tmp_path) -> None:
    plan = _single_stage_plan(tmp_path)
    inst = _inst()

    def boom(spec, *, api, token):
        raise RuntimeError("HTTP Error 429: rate limited")

    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [],
        launch_fn=boom,
        gate=_open_gate,
    )
    assert result.contention is True and result.launched == 0


def test_step_live_429_signals_contention(tmp_path) -> None:
    """A-IMPL-01: the live-jobs read (``list_jobs``) is the per-poll, most-frequent API call and the 429
    surface fix 5 was approved to absorb (spec fix 5 / plan task 6 name a ``live()`` 429). A transient 429
    there must back off, NOT crash the unattended loop."""
    plan = _single_stage_plan(tmp_path)
    inst = _inst()

    def boom() -> list:
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=boom,
        launch_fn=_LaunchRecorder(),
        gate=_open_gate,
    )
    assert result.contention is True and result.launched == 0


def test_run_backs_off_on_rising_mount_faults() -> None:
    """A-IMPL-02 / spec fix 5: a rising ``mount_auth_flake`` rate is a named contention signal, so the poll
    backs off while faults climb and resets once they stop -- with no launch-429 at all."""
    sleeps = _Sleeps()
    results = [
        _res(launched=1, in_flight=1, mount_faults=1),  # 0 -> 1 rising: back off
        _res(launched=1, in_flight=1, mount_faults=3),  # 1 -> 3 rising: back off again
        _res(launched=1, in_flight=1, mount_faults=3),  # 3 -> 3 flat: reset
        _res(done=True),
    ]
    rc = run(
        _campaign(backoff_max_s=300.0),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        poll_s=5,
        step_fn=_scripted(results),
        sleep_fn=sleeps,
        baseline_fn=lambda: frozenset(),
    )
    assert rc == 0 and sleeps.calls == [10, 20, 5]  # 5*2, 10*2, then flat faults reset to 5


def test_run_holds_mount_faults_across_contention_step() -> None:
    """A r2 observation: a 429 step's `mount_faults` reading is meaningless (``_gather`` aborted before the
    tally), so a contention step must NOT clobber `prev_mount_faults` and make the next FLAT-faults step read
    as 'rising'. prev is held across the contention step, so the flat step after it RESETS the poll."""
    sleeps = _Sleeps()
    results = [
        _res(launched=1, in_flight=1, mount_faults=5),  # first obs: 0->5 rising -> 10
        _res(launched=1, in_flight=1, mount_faults=5),  # flat 5->5 -> reset 5
        _res(launched=1, contention=True),  # 429 step: back off (10); prev must STAY 5, not reset to 0
        _res(launched=1, in_flight=1, mount_faults=5),  # still flat 5 -> RESET 5 (not a spurious rise from 0)
        _res(done=True),
    ]
    rc = run(
        _campaign(backoff_max_s=300.0),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        poll_s=5,
        step_fn=_scripted(results),
        sleep_fn=sleeps,
        baseline_fn=lambda: frozenset(),
    )
    assert rc == 0 and sleeps.calls == [10, 5, 10, 5]  # the 4th is 5 (fix); a prev-clobber would make it back off


def test_coverage_excludes_baseline_on_resume(tmp_path) -> None:
    """HFRS-IMPL-01 (reviewer B): the fix-3 cache persists per run_key ACROSS runs. On a RESUME the prior
    run's jobs are THIS run's baseline, so the recompute must NOT count them -- else a crashed unit reads as
    already-spent and the campaign stalls. The persisted-cache resume must equal a from-empty rebuild."""
    import scripts.hf_jobs.driver as drv

    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    crash = _records()["exit_127"]["message"]
    store = _ManifestStore()
    jobs = [_ran_job(inst, plan, token, f"j{i}", ["a"], outcome="ERROR", message=crash, store=store) for i in range(2)]
    run1 = _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store)
    assert run1 == {(inst.id, "extract", "a"): 2}  # run 1: both no-record crash jobs charge a
    baseline = frozenset(j["id"] for j in jobs)  # resume: those jobs are now terminal baseline
    resume = _cover([inst], {inst.id: plan}, jobs, baseline, lambda _i: store)  # SAME store (cache persisted)
    assert resume == {}  # a baseline job contributes nothing on resume
    store.files.pop(drv._COVERAGE_STATE, None)  # a from-empty rebuild on the resume is identical
    assert _cover([inst], {inst.id: plan}, jobs, baseline, lambda _i: store) == {}


def test_coverage_prunes_reduced_stage(tmp_path) -> None:
    """A-IMPL-03 (plan task 4 'Stage-prune reduced stages'): a cached job for a stage already reduced is moot
    to ``decide`` (it inspects only the first ready, un-reduced stage) and is pruned from the in-memory tally."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a"])
    plan["stages"][0]["reduced"] = True  # the stage is already reduced
    crash = _records()["exit_127"]["message"]
    store = _ManifestStore()
    jobs = [_ran_job(inst, plan, token, "j0", ["a"], outcome="ERROR", message=crash, store=store)]
    assert _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {}


# --- HFU spec 13.5: a unit that times out twice is STUCK (only units that STARTED in the job are charged) -------


def _timeouts(insts, plans, live, baseline, store_for, fetch):
    return _attribute(insts, plans, live, baseline, store_for, attempts=3, fetch_logs=fetch)[3]


def _timeout_jobs(inst, plan, token, store, n, keys=("a", "b")):
    msg = _records()["job_timeout"]["message"]
    return [
        _ran_job(inst, plan, token, f"t{i}", list(keys), outcome="ERROR", message=msg, store=store) for i in range(n)
    ]


def test_started_unfinished_parses_unit_lines() -> None:
    from scripts.hf_jobs.driver import started_unfinished

    lines = [
        "[unit] extract/a start",
        "[unit] extract/b start",
        "[unit] extract/b done in 1.0 s",
        "[unit] cv/c start",
        "[alive] t w0 pid=1 up=300s unit=extract/a",
    ]
    assert started_unfinished(lines, "extract") == frozenset({"a"})


def test_two_timeouts_make_a_unit_stuck(tmp_path) -> None:
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    live = _timeout_jobs(inst, plan, token, store, 2)
    logs = {"t0": ["[unit] extract/a start"], "t1": ["[unit] extract/a start"]}
    to = _timeouts([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, logs.__getitem__)
    assert to == {(inst.id, "extract", "a"): ("t0", "t1")}
    d = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=live, attempts=3, units_per_job=20, max_shards=8, timeouts=to
    )
    launched = {k for launch in d.launches for k in launch.keys}
    assert "a" not in launched and "b" in launched
    assert (inst.id, "extract/a timed out 2x (jobs t0, t1)") in d.reports


def test_one_timeout_still_relaunches(tmp_path) -> None:
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    live = _timeout_jobs(inst, plan, token, store, 1)
    to = _timeouts([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, lambda jid: ["[unit] extract/a start"])
    d = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=live, attempts=3, units_per_job=20, max_shards=8, timeouts=to
    )
    assert "a" in {k for launch in d.launches for k in launch.keys} and not d.reports


def test_queued_unit_is_never_charged(tmp_path) -> None:
    """A unit held by 2 timed-out jobs that never started in either (8 workers, 20 held units) is not charged."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    live = _timeout_jobs(inst, plan, token, store, 2)
    to = _timeouts([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, lambda jid: ["[unit] extract/a start"])
    assert (inst.id, "extract", "b") not in to


def test_unit_done_after_a_timeout_is_not_reported(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")  # a finished in a later job
    plan = _realplan(ctx)
    inst = _inst()
    to = {(inst.id, "extract", "a"): ("t0", "t1")}
    d = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=8, timeouts=to
    )
    assert not [r for r in d.reports if "timed out" in r[1]]


def test_log_fetch_failure_charges_nothing_and_retries(tmp_path, capsys) -> None:
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    live = _timeout_jobs(inst, plan, token, store, 2)

    def broken(jid):
        raise OSError("logs unavailable")

    assert _timeouts([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, broken) == {}
    err = capsys.readouterr().err
    assert "[driver] job t0: log fetch failed (OSError: logs unavailable); timeout not charged yet" in err
    good = _timeouts([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, lambda j: ["[unit] extract/a start"])
    assert good == {(inst.id, "extract", "a"): ("t0", "t1")}  # fetched again on the next step


def test_baseline_timeouts_do_not_count(tmp_path) -> None:
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    live = _timeout_jobs(inst, plan, token, store, 2)
    fetch = lambda jid: ["[unit] extract/a start"]  # noqa: E731
    assert _timeouts([inst], {inst.id: plan}, live, frozenset({"t0", "t1"}), lambda _i: store, fetch) == {}


# --- HFU spec 13.6: the driver's log trail ----------------------------------------------------------------------


def test_error_job_line_once(tmp_path, capsys) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    crash = _records()["exit_127"]["message"]
    job = {
        "id": "e1",
        "stage": "ERROR",
        "message": crash,
        "created_at": "2026-11-06T03:00:00+00:00",
        "labels": _labels(inst, "extract", shard="0of1"),
        "wall_s": 12.0,
    }
    reported: set[str] = set()
    for _ in range(2):
        step(
            _campaign(),
            _COMMIT,
            api=_FakeApi(),
            token="t",
            instances=[inst],
            read_plan=lambda i: plan,
            live_records=lambda: [job],
            launch_fn=_LaunchRecorder(),
            gate=_open_gate,
            reported=reported,
        )
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] job e1 ")]
    assert lines == [f"[driver] job e1 {inst.id}/extract 0of1 unit_crash after 12 s"], lines


def test_launch_line_per_launch(tmp_path, capsys) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    recorder = _LaunchRecorder()
    step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[inst],
        read_plan=lambda i: plan,
        live_records=lambda: [],
        launch_fn=recorder,
        gate=_open_gate,
    )
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] launch ")]
    assert len(lines) == len(recorder.specs) >= 1
    assert all(re.fullmatch(rf"\[driver\] launch {re.escape(inst.id)}/extract \d+of\d+ units=\d+", ln) for ln in lines)


def test_a_stuck_memory_unit_is_not_escalated(tmp_path) -> None:
    """HFURR-IMPL-10: a memory-deferred unit charged by 2 timeouts is STUCK on the escalate path too."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",)), max_failures=None),))
    _run(ctx, "fit", "a")
    _run(ctx, "fit", "b")
    plan = _realplan(ctx)
    plan["stages"][0]["memory_units"] = ["b"]
    inst = _inst()
    to = {(inst.id, "fit", "b"): ("t0", "t1")}
    d = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=20, max_shards=12, timeouts=to
    )
    assert "b" not in {k for lx in d.launches for k in lx.keys}
    assert (inst.id, "fit/b timed out 2x (jobs t0, t1)") in d.reports


def test_a_round_robin_shard_holding_a_stuck_unit_is_held_back(tmp_path) -> None:
    """HFURR-IMPL-10: an unkeyed shard runs its whole slice, so a shard that holds a STUCK unit is not launched
    (it would run the unit again); it is reported, and the other shards still launch."""
    from scripts._driver import shard_select

    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(6)], work=_work()),))
    plan = _realplan(ctx)
    units = plan["stages"][0]["units"]
    inst = _download_inst()
    stuck_unit = shard_select(units, 0, 2)[0]
    to = {(inst.id, "extract", stuck_unit): ("t0", "t1")}
    d = decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=[], attempts=3, units_per_job=3, max_shards=12, timeouts=to
    )
    assert [lx.shard for lx in d.launches] == [(1, 2)]
    assert stuck_unit not in {k for lx in d.launches for k in lx.keys}
    assert (inst.id, f"extract/{stuck_unit} timed out 2x (jobs t0, t1)") in d.reports
    assert any("shard 0of2 held back" in r for _i, r in d.reports)


def test_a_long_plan_timeout_dump_keeps_the_innermost_frames(tmp_path) -> None:
    """HFU spec 13.4: a `PlanTimeoutError` is a `--plan failed` report, never a rate-limit back-off or a step crash.
    A faulthandler dump lists the most recent call FIRST, so the report keeps the HEAD: the timeout text and the
    innermost frames (impl review B HFURR-IMPL-11)."""
    from scripts.hf_jobs.campaign import PlanTimeoutError

    frames = ['  File "deep.py", line 10 in _hang'] + [f'  File "deep.py", line {i} in level{i}' for i in range(30)]
    dump = "Timeout (0:09:30)!\nThread 0x0001 (most recent call first):\n" + "\n".join(frames) + "\n"
    assert len(dump) > 300
    ok, slow = _inst(variant="ok"), _inst(variant="slow")
    plan = _single_stage_plan(tmp_path)

    def read(inst):
        if inst is slow:
            raise PlanTimeoutError(f"{inst.id}: --plan failed (timeout after 600 s): {dump}")
        return plan

    result = step(
        _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        instances=[ok, slow],
        read_plan=read,
        live_records=lambda: [],
        launch_fn=_LaunchRecorder(),
        gate=_open_gate,
    )
    assert result.contention is False and result.launched >= 1  # no back-off; the readable instance launched
    report = dict(result.reports)[slow.id]
    assert report.startswith("--plan failed") and "timeout after 600 s" in report and "_hang" in report, report
