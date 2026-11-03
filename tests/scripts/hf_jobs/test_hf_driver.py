"""Driver scheduling decision.

Both halves use REAL data (owner rule, memory ``tdd-tests-from-real-data``):

* Failure records come from ``fixtures/real_jobinfo.json`` -- real captured
  ``JobInfo`` (``status.stage`` + ``status.message`` + real label schema).
* Plan records come from REAL ``scripts._stages.plan()`` output, produced by
  driving a small real pipeline over a ``LocalStore`` (``_realplan``). The earlier
  hand-built plan dicts omitted ``excluded`` / ``max_failures`` -- exactly the keys
  that decide whether a stage reduces -- and that is how two reduce bugs shipped. A real ``plan()`` carries every key.
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
    """Per-unit coverage from ``_attribute``: the first of its (cover, reduce_errors, mount_faults)."""
    return _attribute(insts, plans, live, baseline, store_for, attempts=attempts)[0]


def _mfaults(insts, plans, live, baseline, store_for, *, attempts=3):
    """Per-dataset mount faults from ``_attribute``: the third of its three maps."""
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
    so the driver matches jobs on ``(run_key, variant, stage)`` only."""
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
    # concurrency_cap is the in-flight ceiling; sizing is units_per_job/max_shards. The caps default
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


def _dec(inst, plan, live=(), **kw):
    """``decide`` for one instance at the test campaign's limits (3 attempts, 20 units per job, 12 shards)."""
    return decide(
        instances=[inst], plans={inst.id: plan}, live_jobs=list(live), attempts=3, units_per_job=20, max_shards=12, **kw
    )


# --- stage-walk decisions (real plans) ----------------------------------------


def test_launch_pending_units_when_nothing_in_flight(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work()),))
    inst = _inst()
    decision = _dec(inst, _realplan(ctx))
    assert decision.fatal is None and decision.done is False
    assert all(lx.kind == "run" and lx.stages == ("extract",) for lx in decision.launches)
    assert sorted(k for lx in decision.launches for k in lx.keys) == ["a", "b", "c"]


def test_reduce_launched_when_all_units_done(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a", "b")
    inst = _inst()
    decision = _dec(inst, _realplan(ctx))
    assert [lx.kind for lx in decision.launches] == ["reduce"]
    assert decision.launches[0].stages == ("extract",)


def test_reduce_launched_when_units_done_or_excluded(tmp_path) -> None:
    """An excluded unit counts in ``n_units`` but not ``done``. The stage still reduces."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work(exclude=("c",))),))
    _run(ctx, "extract", "a", "b", "c")  # c raises ItemExcluded -> real .excluded.json
    plan = _realplan(ctx)
    extract = plan["stages"][0]
    assert (extract["n_units"], extract["done"], extract["excluded"]) == (3, 2, 1)  # real plan() shape
    inst = _inst()
    decision = _dec(inst, plan)
    assert [lx.kind for lx in decision.launches] == ["reduce"]
    assert decision.reports == ()


def test_tolerant_stage_reduces_over_a_spent_failure(tmp_path) -> None:
    """A ``max_failures=None`` stage reduces over a unit that is spent (failed ``attempts`` times)."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",)), max_failures=None),))
    _run(ctx, "fit", "a")  # a done
    for _ in range(3):  # b fails 3 times == attempts -> spent
        _run(ctx, "fit", "b")
    plan = _realplan(ctx)
    fit = plan["stages"][0]
    assert fit["max_failures"] is None and fit["failed_units"] == {"b": 3}
    inst = _inst()
    cover = {(inst.id, "fit", "b"): 3}  # b covered 3 times without completing -> spent (coverage == attempts)
    decision = _dec(inst, plan, coverage=cover)
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
    decision = _dec(inst, plan, coverage=cover)
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
    decision = _dec(inst, plan, [_errored(inst, "fit", crash)])
    assert [k for lx in decision.launches for k in lx.keys] == ["b"]  # re-run the one failed unit
    assert decision.reports == ()


def test_timeouts_rerun_the_pending_units_not_stall_the_wave(tmp_path) -> None:
    """A wide wave with several timed-out shards re-runs the pending units; it does NOT
    report the stage stuck on a per-stage error count. A timed-out unit leaves no failure record (the job was
    killed), so it comes back pending and is simply re-run."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    _run(ctx, "extract", *[s.int_key(i) for i in range(22)])  # 22 done, 3 still pending (timed-out shards)
    plan = _realplan(ctx)
    inst = _inst()
    timeout = _records()["job_timeout"]["message"]
    live = [_errored(inst, "extract", timeout, shard=f"{i}of25", job_id=f"t{i}") for i in range(3)]
    decision = _dec(inst, plan, live)
    assert decision.reports == ()  # not stalled
    assert sorted(k for lx in decision.launches for k in lx.keys) == sorted(s.int_key(i) for i in range(22, 25))


def test_no_launch_while_stage_in_flight(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    decision = _dec(inst, plan, [_running(inst, "extract")])
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
    decision = _dec(inst, _realplan(ctx))
    assert {lx.stages for lx in decision.launches} == {("extract",)}  # study waits behind the barrier


def test_reduce_failure_is_bounded_then_reported(tmp_path) -> None:
    """A reduce that keeps crashing is bounded by ``attempts`` reduce-trigger ERROR jobs, then
    reported -- not relaunched on every step. ``decide`` takes the per-stage reduce-error count from ``step``
    (``_reduce_errors``), so here it is passed directly."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a", "b")  # all done -> would reduce
    plan = _realplan(ctx)
    inst = _inst()
    decision = _dec(inst, plan, reduce_errors={(inst.id, "extract"): 3})
    assert decision.launches == ()
    assert len(decision.reports) == 1 and "reduce" in decision.reports[0][1]


def test_done_when_every_stage_reduced(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a"], work=_work()),))
    _run(ctx, "extract", "a")
    s.reduce_stage(ctx, "extract")  # real reduce -> output present -> plan reports reduced=True
    plan = _realplan(ctx)
    assert plan["stages"][0]["reduced"] is True
    inst = _inst()
    decision = _dec(inst, plan)
    assert decision.launches == () and decision.done is True


# --- FATAL scoping: a FATAL new to this run on current work stops it; a resolved one does not -----------


def _fatal_rec(inst: Instance, stage: str, *, created: str = "2026-11-02T03:00:00+00:00") -> dict:
    base = _records()["storage_403_started_set"]  # real storage-limit 403
    return {**base, "created_at": created, "labels": _labels(inst, stage)}


@pytest.mark.parametrize(
    ("case", "blocks"),
    [
        ("new", True),  # a new 403 on current work blocks
        ("baseline", False),  # the same 403 seeded as baseline (a resume after freeing storage) does not
        ("sibling_running", True),  # a later RUNNING sibling shard never masks the FATAL
        ("other_instance", False),  # another instance's 403 does not stop this one
    ],
)
def test_fatal_scoping(tmp_path, case, blocks) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    inst = _inst()
    fatal = _fatal_rec(_inst(run_key="rk2") if case == "other_instance" else inst, "extract")
    live = [fatal]
    if case == "sibling_running":
        live.append(_running(inst, "extract", shard="1of2", created="2026-11-02T03:00:00.002+00:00"))
    baseline = frozenset({fatal["id"]}) if case == "baseline" else frozenset()
    decision = _dec(inst, _realplan(ctx), live, baseline_error_ids=baseline)
    if blocks:
        assert decision.fatal is not None and decision.fatal[0] == "storage_limit_403"
    else:
        assert decision.fatal is None and decision.launches


# --- per-unit coverage bound ------
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


@pytest.mark.parametrize(
    ("coverage", "launched", "reported"),
    [
        ({"b": 3}, [], True),  # b covered `attempts` times with no record: spent, reported
        ({"b": 2}, ["b"], False),  # under the bound: relaunched
        ({"a": 5, "b": 1}, ["b"], False),  # a recovered (done): its high coverage never reports
    ],
)
def test_the_per_unit_coverage_bound(tmp_path, coverage, launched, reported) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    _run(ctx, "extract", "a")
    inst = _inst()
    decision = _dec(inst, _realplan(ctx), coverage={(inst.id, "extract", u): n for u, n in coverage.items()})
    assert [k for lx in decision.launches for k in lx.keys] == launched
    assert bool(decision.reports) is reported


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
    (HFU spec 13.5; see test_two_timeouts_make_a_unit_stuck)."""
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    timeout = _records()["job_timeout"]["message"]
    live = [
        _ran_job(inst, plan, token, f"t{i}", ["a"], outcome="ERROR", message=timeout, store=store) for i in range(5)
    ]
    cov = _cover([inst], {inst.id: plan}, live, frozenset(), lambda _i: store, attempts=3)
    assert cov == {}  # five timeouts, zero coverage -> a (always coverage 0) rides, never spent


def test_coverage_skips_baseline_and_jobs_that_never_ran(tmp_path) -> None:
    """A job in the baseline (a prior run) does not count, so a resume re-attempts. A FATAL
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


def _download_inst(variant: str = "default") -> Instance:
    from scripts.hf_jobs.launch import DatasetMount

    return Instance("job", "public", variant, "mod", (), (DatasetMount("d", "ns/d", "rev", "download"),), "rk")


def test_a_download_instance_runs_round_robin_shards_not_keys(tmp_path) -> None:
    """A ``download`` dataset is fetched in-job by ``--shard I/N``, which a keyed launch omits. So, as main's
    controller did, its run shards are unkeyed round-robin slices; each records the slice's candidate units
    (what the executor will run), so coverage stays exact."""
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
    assert spec is not None
    assert spec.shard == (1, 2) and spec.shard_keys_path is None  # `--shard 1/2`, no keys file


def test_job_spec_gives_hf_one_attempt_so_the_driver_owns_retries(tmp_path) -> None:
    """HF ``run_job(attempts=)`` retries inside a job would multiply the driver's
    per-unit bound (up to attempts**2 runs). Every job gets ONE HF attempt; ``campaign.attempts`` is the bound."""
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    spec = _job_spec(
        _campaign(), _COMMIT, inst, Launch(inst.id, ("extract",), (0, 1), "run", 8, ("a",)), plan, store=_FakeStore()
    )
    assert spec is not None
    assert spec.attempts == 1 and _campaign().attempts == 3


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
    decision = _dec(inst, plan, coverage=cov)
    assert "a" in {k for lx in decision.launches for k in lx.keys}


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
    assert spec is not None
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
    assert spec is not None
    assert spec.shard == (0, 1) and spec.shard_keys_path is not None
    assert len(store.uploads) == 1  # the empty shard-keys file was staged
    assert json.loads(next(iter(store.uploads.values()))) == []  # [] -> executor runs nothing, reduces


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


def _step(campaign, **kw):
    """``step`` at ``_COMMIT`` with a fake API, a token and an open launch gate (``api=`` may override)."""
    kw.setdefault("api", _FakeApi())
    return step(campaign, _COMMIT, token="t", gate=_open_gate, **kw)


def test_step_refuses_at_the_launch_gate_before_anything(tmp_path) -> None:
    """Without a passing preflight report the step refuses before it reads a plan or lists a job."""
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


@pytest.mark.parametrize(
    ("error", "contention", "launched"),
    [
        ("boom", False, 1),  # reported; the healthy instance still launches
        ("HfHubHTTPError: 429 Too Many Requests", True, 0),  # a rate limit backs off: no report, no launch
        ("File \"x.py\", line 429, in f\nKeyError: 'statsbomb__3869429'", False, 1),  # a bare 429 is no limit
    ],
)
def test_a_failing_plan(tmp_path, error, contention, launched) -> None:
    ok, bad = _inst(variant="ok"), _inst(variant="bad")
    plan = _single_stage_plan(tmp_path)

    def read(inst):
        if inst is bad:
            raise RuntimeError(f"{inst.id}: --plan failed (rc=1): {error}")
        return plan

    result = _step(
        _campaign(), instances=[ok, bad], read_plan=read, live_records=lambda: [], launch_fn=_LaunchRecorder()
    )
    assert (result.contention, result.launched) == (contention, launched)
    expected = [] if contention else [bad.id]
    assert [r[0] for r in result.reports] == expected and all("--plan failed" in r[1] for r in result.reports)


def test_step_reads_the_plans_in_parallel(tmp_path, monkeypatch) -> None:
    """The ``--plan`` reads run in parallel."""
    import time as _time

    from scripts.hf_jobs import driver as driver_mod

    monkeypatch.setattr(driver_mod, "PLAN_WORKERS", 6)  # never the host's CPU count

    insts = [_inst(variant=f"v{i}") for i in range(6)]
    plan = _single_stage_plan(tmp_path)

    def slow(_inst):
        _time.sleep(0.5)
        return plan

    t0 = _time.monotonic()
    _step(_campaign(), instances=insts, read_plan=slow, live_records=lambda: [], launch_fn=_LaunchRecorder())
    assert _time.monotonic() - t0 < 2.0  # serial would be >= 3 s


def test_step_prints_one_timed_progress_line(tmp_path, capsys) -> None:
    ok, bad = _inst(variant="ok"), _inst(variant="bad")
    plan = _single_stage_plan(tmp_path)

    def read(inst):
        if inst is bad:
            raise RuntimeError(f"{inst.id}: --plan failed (rc=1): boom")
        return plan

    _step(_campaign(), instances=[ok, bad], read_plan=read, live_records=lambda: [], launch_fn=_LaunchRecorder())
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] step")]
    assert len(lines) == 1, lines
    assert re.fullmatch(
        r"\[driver\] step: plans 2 in \d+\.\ds \(1 failed\); in flight 0; launched 1; stuck 1; instances done 0/2",
        lines[0],
    ), lines


def test_step_line_instances_done(tmp_path, capsys) -> None:
    """HFU spec 14.8: the step line counts the instances whose output stage is reduced."""
    done, open_ = _inst(variant="done"), _inst(variant="open")
    plan = _single_stage_plan(tmp_path)
    plan = {**plan, "output": "extract"}  # the helper pipeline names no output stage
    reduced = {**plan, "stages": [{**st, "reduced": True} for st in plan["stages"]]}
    _step(
        _campaign(),
        instances=[done, open_],
        read_plan=lambda inst: reduced if inst.variant == "done" else plan,
        live_records=lambda: [],
        launch_fn=_LaunchRecorder(),
    )
    line = next(ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] step"))
    assert line.endswith("; instances done 1/2"), line


def test_dry_run_decides_on_the_same_coverage_as_launch(tmp_path, monkeypatch, capsys) -> None:
    """A unit that the run's jobs already covered ``attempts`` times is reported by ``launch``;
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
    result = _step(
        _campaign(), api=api, instances=[inst], read_plan=lambda _i: plan, live_records=lambda: [], launch_fn=recorder
    )
    assert result.fatal is None and result.done is False and result.launched == 1  # 3 units -> 1 sized job
    assert all(spec.stages == ("extract",) for spec in recorder.specs)
    assert api.uploads
    # each run launch also stages a per-job partition manifest keyed by the returned job id,
    # so a later terminated job can be attributed to its units for the coverage bound.
    paths = [path for _bucket, add in api.uploads for _data, path in (add or [])]
    assert any("/by_job/" in p for p in paths)


def test_step_enforces_dispatch_cap(tmp_path) -> None:
    """One step launches at most ``dispatch_cap`` jobs (minus those already in flight)."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    recorder = _LaunchRecorder()
    result = _step(
        _campaign(concurrency_cap=20, units_per_job=1, max_shards=100),
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [],
        launch_fn=recorder,
    )
    assert result.launched == 20 and len(recorder.specs) == 20  # 25 one-unit shards capped to the dispatch budget


def test_step_stops_on_real_fatal_before_launching(tmp_path) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    fatal = _fatal_rec(inst, "extract", created="2026-11-02T03:00:00+00:00")
    recorder = _LaunchRecorder()
    result = _step(
        _campaign(), instances=[inst], read_plan=lambda _i: plan, live_records=lambda: [fatal], launch_fn=recorder
    )
    assert result.fatal is not None and result.fatal[0] == "storage_limit_403"
    assert result.launched == 0 and recorder.specs == []


def test_seed_baseline_collects_every_terminal_id() -> None:
    """The baseline is EVERY pre-existing terminal job: a resume looks past the whole prior
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


def _res(**kw) -> StepResult:
    return StepResult(**{"fatal": None, "launched": 0, "done": False, "reports": (), **kw})


def _scripted(results: list[StepResult]):
    it = iter(results)
    return lambda campaign, commit, **kwargs: next(it)


class _Sleeps(list):
    """Records each sleep. Always truthy: ``run`` falls back to ``time.sleep`` on a falsy ``sleep_fn``."""

    def __call__(self, seconds: float) -> None:
        self.append(seconds)

    def __bool__(self) -> bool:
        return True


def _loop(step_fn, sleeps=None, *, baseline=frozenset(), campaign=None, **kw) -> int:
    """``run`` over a scripted ``step_fn`` with recorded sleeps and a fixed baseline."""
    return run(
        campaign or _campaign(),
        _COMMIT,
        api=_FakeApi(),
        token="t",
        step_fn=step_fn,
        sleep_fn=_Sleeps() if sleeps is None else sleeps,
        baseline_fn=lambda: baseline,
        **kw,
    )


def test_run_seeds_baseline_and_passes_it_to_step() -> None:
    """``run`` seeds the baseline once and passes it to every step."""
    seen: dict = {}

    def step_fn(campaign, commit, **kwargs) -> StepResult:
        seen["baseline"] = kwargs.get("baseline_error_ids")
        return _res(done=True)

    assert _loop(step_fn, baseline=frozenset({"old-403"})) == 0 and seen["baseline"] == frozenset({"old-403"})


@pytest.mark.parametrize(
    ("results", "rc", "slept"),
    [
        ([_res(launched=3), _res(in_flight=3), _res(done=True)], 0, [5, 5]),  # loops until done
        ([_res(fatal=("storage_limit_403", "detail"))], 2, []),  # FATAL: stops without sleeping
        ([_res(reports=(("id", "stuck"),))], 1, []),  # nothing launched or in flight: stalled
    ],
)
def test_run_exit_codes(results, rc, slept) -> None:
    sleeps = _Sleeps()
    assert _loop(_scripted(results), sleeps, poll_s=5) == rc and sleeps == slept


def _main_args():
    return dict(load=lambda _f: _campaign(), make_api=lambda: (_FakeApi(), "tok"), head=lambda: _COMMIT)


def test_main_launch_routes_to_run_loop() -> None:
    seen: dict = {}

    def fake_runner(campaign, commit, *, api, token, replica, poll_s) -> int:
        seen.update(commit=commit, poll_s=poll_s, replica=replica)
        return 7

    rc = main(["camp.toml", "launch", "--poll-s", "5"], runner=fake_runner, **_main_args())
    assert rc == 7 and seen == {"commit": _COMMIT, "poll_s": 5.0, "replica": None}


@pytest.mark.parametrize(
    ("module", "attr", "argv", "ret", "rc"),
    [
        ("campaign", "cmd_preflight", ["preflight"], {"ok": True}, 0),
        ("campaign", "cmd_preflight", ["preflight"], {"ok": False}, 1),
        ("campaign", "cmd_status", ["status"], [], 0),
        ("campaign", "cmd_cancel", ["cancel"], [], 0),
        ("adopt", "retrieve", ["retrieve", "--staging", "STAGING"], {}, 0),
    ],
)
def test_main_routes_each_command(monkeypatch, tmp_path, module, attr, argv, ret, rc) -> None:
    import importlib

    seen: list = []
    target = importlib.import_module(f"scripts.hf_jobs.{module}")
    monkeypatch.setattr(target, attr, lambda *a, **k: (seen.append(attr), ret)[1])
    argv = [str(tmp_path) if a == "STAGING" else a for a in argv]
    assert main(["camp.toml", *argv], **_main_args()) == rc and seen == [attr]


def test_main_adopt_family_requires_staging() -> None:
    with pytest.raises(SystemExit, match="staging"):
        main(["camp.toml", "retrieve"], **_main_args())


def test_package_main_routes_campaign_to_the_driver(monkeypatch) -> None:
    """``python -m scripts.hf_jobs campaign ...`` routes to the driver's ``main``."""
    import scripts.hf_jobs.__main__ as pkg_main
    from scripts.hf_jobs import driver

    seen: dict = {}
    monkeypatch.setattr(driver, "main", lambda rest: (seen.setdefault("rest", rest), 0)[1])
    assert pkg_main.main(["campaign", "camp.toml", "status"]) == 0 and seen["rest"] == ["camp.toml", "status"]


# --- bounded job-sizing (few big jobs, not cap-many) -----------------------------------------


def test_job_sizing_few_big_jobs(tmp_path) -> None:
    """A 50-unit stage launches ceil(50/units_per_job)=3 jobs, not 50 (or the concurrency cap)."""
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(50)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    d = _dec(inst, plan)
    assert len(d.launches) == 3
    assert sorted(k for lx in d.launches for k in lx.keys) == sorted(s.int_key(i) for i in range(50))  # disjoint cover


def test_job_sizing_respects_max_shards(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(500)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    d = _dec(inst, plan)
    assert len(d.launches) == 12  # ceil(500/20)=25 capped at max_shards=12


# --- per-dataset in-flight mount cap -----------------------------------------------------------------


def _mounting(variant: str, run_key: str, *datasets: str, access: str = "mount") -> Instance:
    from scripts.hf_jobs.launch import DatasetMount

    cap = 4 if access == "mount" else None
    mounts = tuple(DatasetMount(d, f"ns/{d}", "r", access, cap) for d in datasets)
    return Instance("job", "owner", variant, "mod", (), mounts, run_key)


@pytest.mark.parametrize(
    ("datasets", "n", "in_flight", "capped", "access", "launched"),
    [
        (("tc",), 5, 0, "tc", "mount", 2),  # 5 instances, cap 2: counted within one decide pass
        (("tc",), 3, 1, "tc", "mount", 1),  # one already in flight + one chosen == cap 2
        (("xshot", "xprobe"), 4, 0, "xprobe", "mount", 2),  # every mount dataset counts (the shared probe)
        (("sb",), 1, 0, "sb", "open", 1),  # an open dataset is not a mount: cap 0 does not apply
    ],
)
def test_mount_cap(tmp_path, datasets, n, in_flight, capped, access, launched) -> None:
    plan = _single_stage_plan(tmp_path)
    insts = [_mounting(f"v{i}", f"rk{i}", *datasets, access=access) for i in range(n)]
    cap = 0 if access == "open" else 2
    d = decide(
        instances=insts,
        plans={i.id: plan for i in insts},
        live_jobs=[_running(insts[0], "extract")] if in_flight else [],
        attempts=3,
        units_per_job=20,
        max_shards=12,
        mount_cap_for=lambda ds: cap if ds == capped else 10**9,
    )
    assert len(d.launches) == launched


# --- by-locus attribution + fold-order-exact recompute ---------------------------------------


def test_coverage_mount_flake_is_per_dataset_not_per_unit(tmp_path) -> None:
    """Owner by-locus ruling: a mount_auth_flake (RETRY, like unit_crash, separated only by reason) bumps a
    per-DATASET fault tally, NEVER the unit's attempts."""
    inst = _mounting("default", "rk", "tc")
    plan, token = _one_stage(tmp_path, ["a"])
    store = _ManifestStore()
    flake = _records()["mount_401_auth"]["message"]
    jobs = [_ran_job(inst, plan, token, f"f{i}", ["a"], outcome="ERROR", message=flake, store=store) for i in range(3)]
    assert _cover([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {}
    assert _mfaults([inst], {inst.id: plan}, jobs, frozenset(), lambda _i: store) == {"tc": 3}


def test_mount_outage_reports_over_budget(tmp_path) -> None:
    inst = _mounting("default", "rk", "tc")
    plan = _single_stage_plan(tmp_path)
    d = _dec(inst, plan, mount_faults={"tc": 5}, mount_fault_budget=4, mount_cap_for=lambda _d: 10**9)
    assert any(iid == "dataset:tc" and "mount outage" in reason for iid, reason in d.reports)


def test_coverage_is_fold_order_exact(tmp_path) -> None:
    """The recompute MUST sort by created_at. Here `a` is a recorded failure; one job covers `[a]`
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


# --- memory-aware sizing + escalation --------------------------------------------------------


def test_escalate_launches_memory_units_solo(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",)), max_failures=None),))
    _run(ctx, "fit", "a")  # a done
    _run(ctx, "fit", "b")  # b failed
    plan = _realplan(ctx)
    plan["stages"][0]["memory_units"] = ["b"]  # b deferred for memory (the engine's plan field)
    inst = _inst()
    d = _dec(inst, plan)
    assert [(lx.escalate, lx.workers, lx.keys) for lx in d.launches] == [(True, 1, ("b",))]


def test_job_spec_escalate_is_solo_whole_box(tmp_path) -> None:
    from scripts.hf_jobs.driver import _BOX_RAM, _HEADROOM
    from scripts.hf_jobs.launch import _run_argv

    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    launch = Launch(inst.id, ("extract",), (0, 1), "run", 1, ("a",), escalate=True)
    spec = _job_spec(_campaign(), _COMMIT, inst, launch, plan, store=_FakeStore())
    assert spec is not None
    assert spec.escalate is True and spec.workers == 1 and spec.cap_bytes == _BOX_RAM - _HEADROOM
    assert "--escalate" in _run_argv(spec) and "--cap-bytes" in _run_argv(spec)


# --- launch pace + contention backoff --------------------------------------------------------


def test_step_paces_launches_to_launch_batch(tmp_path) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: [s.int_key(i) for i in range(25)], work=_work()),))
    plan = _realplan(ctx)
    inst = _inst()
    recorder = _LaunchRecorder()
    result = _step(
        _campaign(concurrency_cap=100, units_per_job=1, max_shards=100, launch_batch=8),
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [],
        launch_fn=recorder,
    )
    assert result.launched == 8  # 25 one-unit shards, paced to launch_batch=8


@pytest.mark.parametrize(
    ("results", "slept"),
    [
        # contention doubles the poll (5*2, 10*2); a clean step resets it
        ([_res(launched=1, contention=True), _res(launched=1, contention=True), _res(launched=1)], [10, 20, 5]),
        # rising mount faults back off; flat faults reset
        ([_res(launched=1, in_flight=1, mount_faults=m) for m in (1, 3, 3)], [10, 20, 5]),
        # a contention step keeps the previous mount-fault count, so the flat step after it resets
        (
            [
                _res(launched=1, in_flight=1, mount_faults=5),
                _res(launched=1, in_flight=1, mount_faults=5),
                _res(launched=1, contention=True),
                _res(launched=1, in_flight=1, mount_faults=5),
            ],
            [10, 5, 10, 5],
        ),
    ],
)
def test_run_backoff(results, slept) -> None:
    sleeps = _Sleeps()
    rc = _loop(_scripted([*results, _res(done=True)]), sleeps, campaign=_campaign(backoff_max_s=300.0), poll_s=5)
    assert rc == 0 and sleeps == slept


def _boom(*_a, **_k):
    raise RuntimeError("HTTP Error 429: Too Many Requests")


@pytest.mark.parametrize("where", ["launch_fn", "live_records"])
def test_step_429_signals_contention(tmp_path, where) -> None:
    """A 429 on launch or on the live-jobs read backs off; it never crashes the loop."""
    plan = _single_stage_plan(tmp_path)
    kw = {"launch_fn": _LaunchRecorder(), "live_records": lambda: [], where: _boom}
    result = _step(_campaign(), instances=[_inst()], read_plan=lambda _i: plan, **kw)
    assert result.contention is True and result.launched == 0


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


@pytest.mark.parametrize(
    ("n_jobs", "baseline", "charged"),
    [
        (2, frozenset(), ("t0", "t1")),  # only `a` started in both jobs: `a` charged twice, `b` (queued) never
        (1, frozenset(), ("t0",)),
        (2, frozenset({"t0", "t1"}), None),  # a prior run's timeouts never count
    ],
)
def test_timeout_charges_only_units_that_started(tmp_path, n_jobs, baseline, charged) -> None:
    inst = _inst()
    plan, token = _one_stage(tmp_path, ["a", "b"])
    store = _ManifestStore()
    live = _timeout_jobs(inst, plan, token, store, n_jobs)
    to = _timeouts([inst], {inst.id: plan}, live, baseline, lambda _i: store, lambda _j: ["[unit] extract/a start"])
    assert to == ({(inst.id, "extract", "a"): charged} if charged else {})


@pytest.mark.parametrize(
    ("charged", "a_done", "a_launched", "reported"),
    [
        (("t0", "t1"), False, False, True),  # 2 timeouts: STUCK, reported, not relaunched
        (("t0",), False, True, False),  # 1 timeout: relaunched
        (("t0", "t1"), True, False, False),  # done in a later job: never reported
    ],
)
def test_two_timeouts_make_a_unit_stuck(tmp_path, charged, a_done, a_launched, reported) -> None:
    ctx = _ctx(tmp_path, (s.Stage(name="extract", units=lambda c: ["a", "b"], work=_work()),))
    if a_done:
        _run(ctx, "extract", "a")
    inst = _inst()
    d = _dec(inst, _realplan(ctx), timeouts={(inst.id, "extract", "a"): charged})
    assert ("a" in {k for lx in d.launches for k in lx.keys}) is a_launched
    assert any("extract/a timed out 2x" in r[1] for r in d.reports) is reported


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
        _step(
            _campaign(),
            instances=[inst],
            read_plan=lambda i: plan,
            live_records=lambda: [job],
            launch_fn=_LaunchRecorder(),
            reported=reported,
        )
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] job e1 ")]
    assert lines == [f"[driver] job e1 {inst.id}/extract 0of1 unit_crash after 12 s"], lines


def test_launch_line_per_launch(tmp_path, capsys) -> None:
    inst = _inst()
    plan = _single_stage_plan(tmp_path)
    recorder = _LaunchRecorder()
    _step(_campaign(), instances=[inst], read_plan=lambda i: plan, live_records=lambda: [], launch_fn=recorder)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[driver] launch ")]
    assert len(lines) == len(recorder.specs) >= 1
    assert all(re.fullmatch(rf"\[driver\] launch {re.escape(inst.id)}/extract \d+of\d+ units=\d+", ln) for ln in lines)


def test_a_stuck_memory_unit_is_not_escalated(tmp_path) -> None:
    """A memory-deferred unit charged by 2 timeouts is STUCK on the escalate path too."""
    ctx = _ctx(tmp_path, (s.Stage(name="fit", units=lambda c: ["a", "b"], work=_work(fail=("b",)), max_failures=None),))
    _run(ctx, "fit", "a")
    _run(ctx, "fit", "b")
    plan = _realplan(ctx)
    plan["stages"][0]["memory_units"] = ["b"]
    inst = _inst()
    to = {(inst.id, "fit", "b"): ("t0", "t1")}
    d = _dec(inst, plan, timeouts=to)
    assert "b" not in {k for lx in d.launches for k in lx.keys}
    assert (inst.id, "fit/b timed out 2x (jobs t0, t1)") in d.reports


def test_a_round_robin_shard_holding_a_stuck_unit_is_held_back(tmp_path) -> None:
    """An unkeyed shard runs its whole slice, so a shard that holds a STUCK unit is not launched
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
    innermost frames."""
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

    result = _step(
        _campaign(), instances=[ok, slow], read_plan=read, live_records=lambda: [], launch_fn=_LaunchRecorder()
    )
    assert result.contention is False and result.launched >= 1  # no back-off; the readable instance launched
    report = dict(result.reports)[slow.id]
    assert report.startswith("--plan failed") and "timeout after 600 s" in report and "_hang" in report, report


# --- a 429 on a driver store write never stops the driver (spec §15.4) -----------------------------------------


class _BucketApi(_FakeApi):
    """A bucket double that stores uploads and serves them back; an upload whose path holds ``poison`` is a 429."""

    def __init__(self, poison: str = "", files: dict | None = None) -> None:
        super().__init__()
        self.poison, self.files = poison, dict(files or {})

    def batch_bucket_files(self, bucket, add=None, **kwargs) -> None:
        for data, path in add or ():
            if self.poison and self.poison in path:
                raise RuntimeError("Rate limited: HTTP status client error (429 Too Many Requests)")
            self.files[path] = data

    def download_bucket_files(self, bucket, pairs, raise_on_missing_files=False) -> None:
        for remote, local in pairs:
            if remote in self.files:
                pathlib.Path(local).write_bytes(self.files[remote])


@pytest.mark.parametrize("write", ["shard_keys", "manifest", "cache"])
def test_step_survives_a_429_on_each_write(tmp_path, monkeypatch, capsys, write) -> None:
    from scripts.hf_jobs import driver as D

    monkeypatch.setattr(D, "_PUT_BACKOFF_S", (0.0, 0.0, 0.0))
    inst, plan = _inst(), _single_stage_plan(tmp_path)
    token = plan["stages"][0]["token"]
    old = {"id": "old1", "stage": "COMPLETED", "message": None, "created_at": "2026-11-02T03:00:00+00:00",
           "labels": _labels(inst, "extract")}  # fmt: skip
    files = {f"{inst.run_key}/{by_job_rel(plan, 'extract', token, 'old1')}": b'["a"]'}  # a manifest to fold
    poison = {"shard_keys": f"/{inst.variant}/0of", "manifest": "/by_job/id", "cache": "_coverage/state.json"}[write]
    recorder = _LaunchRecorder()
    result = _step(
        _campaign(),
        api=_BucketApi(poison, files),
        instances=[inst],
        read_plan=lambda _i: plan,
        live_records=lambda: [old] if write == "cache" else [],
        launch_fn=recorder,
    )
    assert result.contention is True and result.fatal is None
    if write == "shard_keys":
        assert recorder.specs == [] and result.launched == 0  # nothing runs without its keys
    elif write == "manifest":
        assert (
            result.launched == 1 and "[driver] job id1 manifest not recorded (rate-limited)" in capsys.readouterr().err
        )
    else:
        assert result.launched == 1  # the step still launches; the cache waits for the next step


def test_guarded_put_retries_a_429_then_succeeds(monkeypatch) -> None:
    from scripts.hf_jobs import driver as D

    slept: list[float] = []
    calls = {"n": 0}

    class _Flaky:
        def upload_bytes(self, rel, data):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("429 Too Many Requests")

    assert D._guarded_put(_Flaky(), "_partitions/x", b"1", sleep=slept.append) is True
    assert calls["n"] == 3 and slept == list(D._PUT_BACKOFF_S[:2])


def test_guarded_put_raises_a_non_contention_error() -> None:
    from scripts.hf_jobs import driver as D

    class _Broken:
        def upload_bytes(self, rel, data):
            raise PermissionError("not allowed")

    with pytest.raises(PermissionError):
        D._guarded_put(_Broken(), "_partitions/x", b"1", sleep=lambda _s: None)


@pytest.mark.parametrize("timeout_s", [None, 14400])
def test_job_spec_takes_a_stage_timeout_from_the_real_plan(tmp_path, timeout_s) -> None:
    """HFU spec 15.7: a stage's declared ``timeout_s`` reaches the job through the real ``plan()`` record;
    without one the job gets 7,200 s."""
    kw = {} if timeout_s is None else {"timeout_s": timeout_s}
    stage = s.Stage(name="extract", units=lambda c: ["a", "b", "c"], work=_work(), **kw)
    plan = _realplan(_ctx(tmp_path, (stage,)))
    inst = _inst()
    spec = _job_spec(_campaign(), _COMMIT, inst, _dec(inst, plan).launches[0], plan, store=_FakeStore())
    assert spec is not None and spec.timeout_s == (timeout_s or 7200)


def test_a_reduced_instance_is_planned_once_per_driver_run(tmp_path) -> None:
    """HFU spec 16.4 (A): a reduced output never changes under its run key, so a step reuses that instance's
    cached plan; an open instance is planned every step."""
    done_i, open_i = _inst(variant="done"), _inst(variant="open")
    open_plan = _single_stage_plan(tmp_path)
    done_plan = {**open_plan, "output": "extract", "stages": [{**st, "reduced": True} for st in open_plan["stages"]]}
    reads: list[str] = []

    def read(inst):
        reads.append(inst.id)
        return done_plan if inst is done_i else open_plan

    cache: dict = {}
    for _ in range(2):
        _step(
            _campaign(),
            instances=[done_i, open_i],
            read_plan=read,
            live_records=lambda: [],
            launch_fn=_LaunchRecorder(),
            plan_cache=cache,
        )
    assert reads.count(done_i.id) == 1 and reads.count(open_i.id) == 2


def test_run_keeps_one_plan_cache_across_steps() -> None:
    seen: list = []

    def step_fn(campaign, commit, **kwargs):
        seen.append(kwargs.get("plan_cache"))
        return _res(in_flight=1) if len(seen) == 1 else _res(done=True)

    assert _loop(step_fn) == 0
    assert isinstance(seen[0], dict) and seen[0] is seen[1]
