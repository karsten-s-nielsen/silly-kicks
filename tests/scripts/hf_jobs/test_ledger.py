from __future__ import annotations

import json

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

_LABELS = {"job": "ghost_gk", "variant": "default", "stage": "extract", "shard": "1of2"}
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
    )
    assert s["run_commit"] == "c" * 40 and s["run_tree_dirty"] is False


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
    label = "ghost_gk.default.extract.1of2"
    jobs = [_job(_LABELS, _T0, _T0 + 1000), _job(_LABELS, _T0 + 2000, _T0 + 3000)]
    recs = [
        _urec(label, _T0 + 130, 0),
        _urec(label, _T0 + 90, 1),  # the FIRST unit of job 1: startup 90 s
        _urec(label, _T0 + 2200, 0),  # job 2 (the retry): startup 200 s
        _urec("ghost_gk.default.extract.2of2", _T0 + 10, 0),  # another shard: never matched
    ]
    assert job_startups(jobs, recs) == [90.0, 200.0]
    assert job_startups([_job(_LABELS, _T0, _T0 + 50)], recs[3:]) == []  # a job with no unit has no startup


def test_a_unit_of_an_earlier_in_job_attempt_never_counts():
    """Review r3 B CCHC-IMPL-02: one HF job id can hold several `attempts` (the store persists, so a retry
    resumes). A surviving record from before `started_at` belongs to an earlier attempt; counting it (and
    clamping the negative to 0) would record a plausible but false 0 s startup."""
    label = "ghost_gk.default.extract.1of2"
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


def test_two_actions_with_the_same_n_and_a_retry_never_share_a_file(tmp_path):
    from scripts._stages import LocalStore
    from scripts.hf_jobs import ledger, phases

    st = LocalStore(tmp_path)
    labels = ["g.default.extract.0of3.20261021T143000Z_aaaaaaaa", "g.default.extract.0of3.20261021T150000Z_bbbbbbbb"]
    for label, attempt, key in (
        (labels[0], "20261021T143001Z_0001", "u1"),
        (labels[0], "20261021T143500Z_0002", "u2"),
        (labels[1], "20261021T150001Z_0003", "u3"),
    ):
        rel = phases.cost_rel("g", "extract", "tok", label, attempt)
        row = {key: {"outcome": "done", "wall_s": 1.0, "started_ns": 1, "finished_ns": 2}}
        st.write_bytes(rel, json.dumps(row).encode())
    got = ledger.cost_aggregates(st, "g", "extract", "tok")
    assert sorted(got) == ["u1", "u2", "u3"]
    assert len(list((tmp_path / "_ledger/g/extract/tok/_costs").iterdir())) == 3


def test_job_startups_reads_an_old_job_record_without_an_action_label():
    import datetime

    from scripts.hf_jobs import ledger

    iso = lambda t: datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()  # noqa: E731
    labels = {"job": "g", "variant": "v", "stage": "cv", "shard": "0of1"}
    job = {"labels": labels, "started_at": iso(100), "finished_at": iso(900)}
    rec = {"job_label": "g.v.cv.0of1", "started_ns": int(160e9)}
    assert ledger.job_startups([job], [rec]) == [60.0]
