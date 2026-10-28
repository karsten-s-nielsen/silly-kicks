from __future__ import annotations

import json

import pytest

from scripts._stages import LocalStore
from scripts.hf_jobs.ledger import (
    history_key,
    job_startups,
    median_startup_s,
    percentile,
    stage_workers,
    summarize,
    unit_cost,
    unit_records,
)

_LABELS = {"job": "ghost_gk", "tier": "owner", "variant": "default", "stage": "extract", "shard": "1of2"}
_T0 = 1_800_000_000  # epoch seconds of the first job's start


def _rec(store, gen, key, wall, peak, outcome="done"):
    rec = {"key": key, "outcome": outcome, "wall_s": wall, "peak_rss_bytes": peak, "attempt": 1}
    store.write_bytes(f"{gen}/{key}.ledger.json", json.dumps(rec).encode())


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.5) == 2 and percentile([1, 2, 3, 4], 0.95) == 4


def test_unit_cost_from_done_records_only(tmp_path):
    store = LocalStore(tmp_path)
    for i, w in enumerate([10, 20, 30, 40]):
        _rec(store, "p/s/t", f"k{i}", w, 100 + i)
    _rec(store, "p/s/t", "bad", 999, 10**12, outcome="failed")
    c = unit_cost(unit_records(store, "p/s/t"))
    assert c is not None
    assert (c.p50_s, c.p95_s, c.peak_rss_bytes, c.n_samples) == (20, 40, 103, 4)


def test_cost_statistics_exclude_null_measurements():
    """Spec r3 D6(a) (B review R3-04): a ledger lost on the mount is repaired with NULL measurements (D7). Every cost
    and worker statistic skips the nulls instead of failing or counting them."""
    measured = [
        {
            "key": f"k{i}",
            "outcome": "done",
            "wall_s": float(w),
            "peak_rss_bytes": 100 + i,
            "worker": "w0",
            "job_label": "j.a",
            "inner_workers": 1,
            "started_ns": (_T0 + i) * 10**9,
        }
        for i, w in enumerate([10, 20, 30])
    ]
    repaired = {
        "key": "k9",
        "outcome": "done",
        "wall_s": None,
        "peak_rss_bytes": None,
        "worker": None,
        "job_label": None,
        "inner_workers": None,
        "started_ns": None,
        "repaired": True,
    }
    c = unit_cost([*measured, repaired])
    assert c is not None and (c.n_samples, c.p50_s, c.peak_rss_bytes) == (3, 20.0, 102)
    assert stage_workers([*measured, repaired]) == (1, 1)
    assert job_startups([_job(_LABELS, _T0 - 5, _T0 + 60)], [repaired]) == []  # no measured start: no startup


def test_history_key_ignores_code_and_upstream():
    base = {"pipeline": "g", "stage": "cv", "params": {"x": 1}, "dataset": {"name": "tc3", "revision": "r"}}
    a = history_key({**base, "code": "c1", "upstream": {"cache": "t1"}})
    b = history_key({**base, "code": "c2", "upstream": {"cache": "t2"}})
    assert a == b != history_key({**base, "params": {"x": 2}, "code": "c1", "upstream": {}})


def test_summary_carries_top_level_provenance():
    s = summarize(
        campaign="cycle1",
        campaign_sha256="ab",
        prov={"commit": "c" * 40, "dirty": False},
        stages=[],
        jobs=[],
        artifacts=[],
        observed_max_concurrency=7,
    )
    assert s["run_commit"] == "c" * 40 and s["run_tree_dirty"] is False
    assert s["observed_max_concurrency"] == 7  # §D probe-first: the real HF ceiling this run


def _job(labels, start_s, end_s):
    import datetime

    def iso(t):
        return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()

    return {"labels": labels, "created_at": iso(start_s - 5), "started_at": iso(start_s), "finished_at": iso(end_s)}


def _urec(label, start_s, worker, inner=None):
    return {"job_label": label, "started_ns": int(start_s * 1e9), "worker": worker, "inner_workers": inner}


def test_job_startup_is_the_first_unit_start_minus_the_job_start():
    """Plan Task 21 step 8: startup = first-unit start minus `JobInfo.started_at`, per job. A retry of the
    same shard (same label, later window) is its own job; a unit of another label never counts."""
    label = "ghost_gk.owner.default.extract.1of2"
    jobs = [_job(_LABELS, _T0, _T0 + 1000), _job(_LABELS, _T0 + 2000, _T0 + 3000)]
    recs = [
        _urec(label, _T0 + 130, 0),
        _urec(label, _T0 + 90, 1),  # the FIRST unit of job 1: startup 90 s
        _urec(label, _T0 + 2200, 0),  # job 2 (the retry): startup 200 s
        _urec("ghost_gk.owner.default.extract.2of2", _T0 + 10, 0),  # another shard: never matched
    ]
    assert job_startups(jobs, recs) == [90.0, 200.0]
    assert job_startups([_job(_LABELS, _T0, _T0 + 50)], recs[3:]) == []  # a job with no unit has no startup


def test_a_unit_of_an_earlier_in_job_attempt_never_counts():
    """Review r3 B CCHC-IMPL-02: one HF job id can hold several `attempts` (the store persists, so a retry
    resumes). A surviving record from before `started_at` belongs to an earlier attempt; counting it (and
    clamping the negative to 0) would record a plausible but false 0 s startup."""
    label = "ghost_gk.owner.default.extract.1of2"
    job = _job(_LABELS, _T0, _T0 + 1000) | {"created_at": _job(_LABELS, _T0 - 600, _T0)["started_at"]}
    recs = [_urec(label, _T0 - 300, 0), _urec(label, _T0 + 75, 1)]
    assert job_startups([job], recs) == [75.0]


def test_a_committed_startup_that_is_negative_or_not_finite_fails_loud(tmp_path):
    """Review r3 B CCHC-IMPL-02: a hand-edited summary must not poison the planner silently (a negative
    startup under-predicts every wall; NaN makes every predicted wall NaN, so preflight flags nothing)."""
    import pytest

    for bad in (-5.0, float("nan"), float("inf")):
        (tmp_path / "x.json").write_text(json.dumps({"stages": [{"startup_s": bad}]}))
        with pytest.raises(ValueError, match="startup_s"):
            median_startup_s(tmp_path.glob("*.json"))


def test_stage_workers_are_the_distinct_outer_workers_of_the_widest_job():
    recs = [_urec("a.v.s.1of2", 0, w, inner=1) for w in (0, 1, 2)] + [_urec("a.v.s.2of2", 0, 0, inner=1)]
    assert stage_workers(recs) == (3, 1)
    assert stage_workers([_urec("a.v.study.1of1", 0, 0, inner=5)]) == (1, 5)
    assert stage_workers([]) == (None, None)


def test_median_startup_is_read_from_the_committed_summaries(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"stages": [{"startup_s": 100.0}, {"startup_s": None}]}))
    (tmp_path / "b.json").write_text(json.dumps({"stages": [{"startup_s": 40.0}, {"startup_s": 220.0}]}))
    assert median_startup_s(tmp_path.glob("*.json")) == 100.0
    assert median_startup_s([]) is None


# ---- HFU T10: bootstrap the first run's startup with the HF-measured value ------------------------------


def test_bootstrap_summary_supplies_first_run_startup(tmp_path):
    """HFU T10 S2 (B-15): before any campaign measured a startup, the committed bootstrap summary supplies the
    HF-measured median (~15 s: git clone + venv + uv sync on cpu-upgrade), so the first planning run does NOT
    fall back to the 300 s Constraints default. The bootstrap stage carries startup_s alone -- no history_key,
    no unit_cost -- so it feeds median_startup_s only, never the cost history."""
    from scripts.hf_jobs.ledger import BOOTSTRAP_STARTUP_S, load_history, write_bootstrap_summary

    p = tmp_path / "_bootstrap_startup.json"
    write_bootstrap_summary(p)
    assert median_startup_s(tmp_path.glob("*.json")) == BOOTSTRAP_STARTUP_S  # supplies the first-run startup
    assert load_history(tmp_path.glob("*.json")) == {}  # never pollutes the cost history


def test_unit_cost_surfaces_inner_child_peak_rss():
    """HFU T11 S5b/B-19/B-20: unit_cost surfaces the MAX per-child pool RSS from the unit ledgers as
    UnitCost.inner_child_peak_rss_bytes; on an older record that lacks the field it is None (defaulted)."""
    from scripts.hf_jobs.ledger import unit_cost as _uc

    recs = [
        {"outcome": "done", "wall_s": 10.0, "peak_rss_bytes": 100, "inner_child_peak_rss_bytes": 30},
        {"outcome": "done", "wall_s": 12.0, "peak_rss_bytes": 120, "inner_child_peak_rss_bytes": 50},
    ]
    cost = _uc(recs)
    assert cost is not None and cost.inner_child_peak_rss_bytes == 50  # the max child peak across the ledgers
    old = _uc([{"outcome": "done", "wall_s": 10.0, "peak_rss_bytes": 100}])  # older ledger: no child field
    assert old is not None and old.inner_child_peak_rss_bytes is None


def _load_rec(wall: float, load: float | None, loads: int | None = 1) -> dict:
    """A done unit record; ``load`` None = a pre-D11 ledger without the load fields."""
    rec = {"outcome": "done", "wall_s": wall, "peak_rss_bytes": 1}
    if load is not None:
        rec.update(
            upstream_loads=loads,
            upstream_fetch_s=load * 0.4,
            upstream_verify_s=load * 0.1,
            upstream_decode_s=load * 0.5,
        )
    return rec


def test_the_unit_cost_excludes_the_upstream_load():
    """Spec r3 D11 (b): a worker's FIRST unit carries the upstream load (`ctx.upstream` inside the unit). The
    planner's per-unit cost is the unit's compute alone; the load is a per-job overhead, never multiplied by the
    unit count."""
    from scripts.hf_jobs.ledger import unit_cost

    cost = unit_cost([_load_rec(110.0, 10.0), _load_rec(100.0, 0.0, loads=0), _load_rec(100.0, None)])
    assert cost is not None
    assert (cost.p50_s, cost.p95_s, cost.p99_s) == (100.0, 100.0, 100.0)


def test_the_stage_upstream_load_is_the_median_over_the_loading_units():
    """Spec r3 D11 (b): the stage's measured load is the median load of the units that loaded; units that reused
    the worker's copy, and pre-D11 ledgers, carry no load measurement."""
    from scripts.hf_jobs.ledger import upstream_load_s

    recs = [_load_rec(110.0, 10.0), _load_rec(112.0, 12.0), _load_rec(100.0, 0.0, loads=0), _load_rec(100.0, None)]
    assert upstream_load_s(recs) == pytest.approx(10.0)
    assert upstream_load_s([_load_rec(100.0, 0.0, loads=0), _load_rec(100.0, None)]) is None


def test_unit_cost_records_the_p50_unit_cpu_share():
    """S3 item 5 (WIO): the p50 of the done units' ``cpu_s / wall_s``; None when no unit recorded its CPU."""
    recs = [
        {"outcome": "done", "wall_s": w, "cpu_s": c, "peak_rss_bytes": 10**8}
        for w, c in [(1.0, 0.7), (2.0, 1.42), (1.0, 0.9)]
    ]
    cost = unit_cost(recs)
    assert cost is not None and cost.cpu_share == pytest.approx(0.71)
    no_cpu = unit_cost([{"outcome": "done", "wall_s": 1.0, "peak_rss_bytes": 10**8}])
    assert no_cpu is not None and no_cpu.cpu_share is None
