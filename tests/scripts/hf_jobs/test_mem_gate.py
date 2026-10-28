"""HFU plan R3-8 S3 item 6 (MEM, row 2): the memory admission gate's decision logic (`scripts.hf_jobs.mem_gate`).

The executor-level behaviour (W, the claim protocol, the box guard, completion) is in `test_executor.py`; the history
source (the newest earlier generation) is in `test_campaign.py`."""

from __future__ import annotations

import pytest

from scripts.hf_jobs.mem_gate import GRANT, WAIT, MemGate, WorkerMem

GB = 10**9


def _gate(budget_gb: float, peaks_gb: dict[str, float]) -> MemGate:
    return MemGate.from_peaks({k: int(v * GB) for k, v in peaks_gb.items()}, budget=int(budget_gb * GB))


def test_each_unit_reserves_1_3_times_its_predicted_peak():
    g = _gate(30, {"a": 2.0, "b": 5.51})
    assert g.reservation("a") == int(1.3 * 2.0 * GB) and g.reservation("b") == int(1.3 * 5.51 * GB)


def test_a_unit_with_no_record_reserves_the_stage_maximum():
    """Plan item 6: a unit with no earlier record takes the earlier generation's stage MAXIMUM, never the p95."""
    g = _gate(30, {"a": 2.0, "b": 5.51, "c": 1.0})
    assert g.reservation("new") == g.reservation("b")


def test_admission_counts_the_larger_of_measured_rss_and_the_reservation_in_flight():
    g = _gate(16, {"u": 5.0})  # each unit reserves 6.5 GB
    busy_low = {"w0": WorkerMem(rss=1 * GB, inflight="u")}  # measured below its reservation: the reservation counts
    assert g.decide(busy_low, "w1", "u") is GRANT  # 6.5 + 6.5 = 13 <= 16
    busy_high = {"w0": WorkerMem(rss=10 * GB, inflight="u")}  # measured above it: the measurement counts
    assert g.decide(busy_high, "w1", "u") is WAIT  # 10 + 6.5 > 16, and no idle worker holds retained heap


def test_eight_units_of_5_gb_under_a_16_gb_budget_run_at_most_two_at_once():
    g = _gate(16, {f"u{i}": 5.0 for i in range(8)})
    live: dict[str, WorkerMem] = {}
    admitted = []
    for i in range(8):
        wid = f"w{i}"
        if g.decide(live, wid, f"u{i}") is GRANT:
            live[wid] = WorkerMem(rss=0, inflight=f"u{i}")
            admitted.append(wid)
        else:
            live[wid] = WorkerMem(rss=0, inflight=None)
    assert admitted == ["w0", "w1"]


def test_a_worker_that_retains_4_gb_after_a_5_gb_unit_blocks_a_claim_that_would_overfill():
    g = _gate(16, {"u": 5.0})
    live = {"w0": WorkerMem(rss=4 * GB, inflight=None), "w1": WorkerMem(rss=5 * GB, inflight="u")}
    d = g.decide(live, "w2", "u")  # 4 + 6.5 + 6.5 = 17 > 16
    assert d is not GRANT
    assert d.stop == ("w0",)  # progress rule (i): the idle worker whose retained heap is the reason exits
    released = {**live, "w0": WorkerMem(rss=0, inflight=None)}
    assert g.decide(released, "w2", "u") is GRANT  # non-vacuity: without the retained heap the claim fits


def test_the_requester_retaining_heap_above_its_next_reservation_is_part_of_the_reason():
    g = _gate(16, {"small": 1.0, "u": 5.0})
    live = {"w0": WorkerMem(rss=12 * GB, inflight=None), "w1": WorkerMem(rss=5 * GB, inflight="u")}
    assert g.decide(live, "w0", "small").stop == ("w0",)  # 5 + max(12, 1.3) > 16: its own retained heap blocks


def test_retained_heap_never_stalls_the_stage():
    """Progress rule (S2B-r2-01): seven idle workers retain 4 GB each and the next unit reserves 7.2 GB. Rule (i) stops
    the idle workers; rule (ii) then admits the claim with no unit in flight, whatever its reservation."""
    g = _gate(30, {"big": 5.51, "small": 1.0})
    live = {f"w{i}": WorkerMem(rss=4 * GB, inflight=None) for i in range(1, 8)}
    live["w0"] = WorkerMem(rss=0, inflight=None)
    d = g.decide(live, "w0", "big")  # 28 + 7.2 > 30
    assert d.stop == tuple(f"w{i}" for i in range(1, 8))
    alone = {"w0": WorkerMem(rss=0, inflight=None)}
    assert g.decide(alone, "w0", "big") is GRANT


def test_rule_ii_admits_a_unit_alone_even_over_the_budget_but_only_with_nothing_in_flight():
    g = _gate(6, {"big": 5.51})  # one unit reserves 7.2 GB, over the 6 GB budget
    assert g.decide({"w0": WorkerMem(rss=0, inflight=None)}, "w0", "big") is GRANT  # rule (ii): alone always runs
    busy = {"w0": WorkerMem(rss=0, inflight=None), "w1": WorkerMem(rss=0, inflight="big")}
    assert g.decide(busy, "w0", "big") is WAIT  # a unit in flight: wait for it, never overfill


def test_a_fresh_worker_is_launched_only_when_the_gate_would_admit_it():
    g = _gate(16, {"u": 5.0})
    one_busy = {"w0": WorkerMem(rss=0, inflight="u")}
    assert g.fresh_fits(one_busy, "u")  # 6.5 + 6.5 <= 16
    two_busy = {**one_busy, "w1": WorkerMem(rss=0, inflight="u")}
    assert not g.fresh_fits(two_busy, "u")  # 13 + 6.5 > 16


@pytest.mark.parametrize("peaks", [{}, {"a": 0.0}])
def test_a_gate_needs_a_measured_peak(peaks):
    with pytest.raises(ValueError, match="peak"):
        MemGate.from_peaks({k: int(v * GB) for k, v in peaks.items()}, budget=30 * GB)


def test_the_box_guard_picks_the_busy_worker_with_the_largest_excess_over_its_reservation():
    g = _gate(16, {"a": 2.0, "b": 5.0})  # reservations 2.6 and 6.5 GB
    fits = {"w0": WorkerMem(rss=3 * GB, inflight="a"), "w1": WorkerMem(rss=7 * GB, inflight="b")}
    assert g.box_victim(fits) is None  # 10 GB <= 16 GB
    over = {"w0": WorkerMem(rss=6 * GB, inflight="a"), "w1": WorkerMem(rss=11 * GB, inflight="b")}
    assert g.box_victim(over) == "w1"  # excess 4.5 GB over 3.4 GB
    idle_only = {"w0": WorkerMem(rss=17 * GB, inflight=None)}
    assert g.box_victim(idle_only) is None  # an idle worker has no unit to escalate; rule (i) handles its heap


def test_rule_i_does_not_stop_idle_workers_when_the_claim_still_waits_for_units_in_flight():
    """Review S3B-08: four units of 7.2 GB in flight on a 30 GB budget, three idle workers at an interpreter's
    baseline. Stopping the idle workers frees only their baseline; the fifth unit still waits for a unit in flight.
    So the gate WAITS (no churn) instead of stopping them."""
    g = _gate(30, {"big": 5.51})
    live = {f"w{i}": WorkerMem(rss=int(0.3 * GB), inflight="big") for i in range(4)}
    live.update({f"w{i}": WorkerMem(rss=int(0.3 * GB), inflight=None) for i in range(4, 7)})
    assert g.decide(live, "w4", "big") is WAIT  # 4 x 7.16 + 7.16 > 30 even with every idle worker gone
