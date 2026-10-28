from __future__ import annotations

import pytest

from scripts._parallel_launch import RefusalError
from scripts.hf_jobs.planner import Constraints, UnitCost, irreducible_floor_s, lpt_partition, plan_stage

GiB = 1024**3
KEYS = [f"m{i:03d}" for i in range(179)]


def test_workers_are_ram_bounded_not_vcpu_bounded():
    p = plan_stage("extract", KEYS, UnitCost(30.0, 40.0, 50.0, 6 * GiB, 20), inner_parallel=False)
    assert p.workers == 3  # (30 - 2) // (6 * 1.3) == 3, below 8 vCPU
    assert p.cap_bytes == 30 * GiB // 3  # the fair share, above 1.3 x the measured peak


def test_n_jobs_meets_the_target_with_startup():
    c = Constraints(target_s=3600, startup_s=300)
    p = plan_stage("extract", KEYS, UnitCost(60.0, 70.0, 80.0, 1 * GiB, 20), inner_parallel=False, c=c)
    # 179 units * 60 s / (8 workers * 3300 s) -> 1 job
    assert p.n_jobs == 1 and p.workers == 8
    slow = plan_stage("extract", KEYS, UnitCost(600.0, 700.0, 800.0, 1 * GiB, 20), inner_parallel=False, c=c)
    assert slow.n_jobs == 5  # ceil(179*600 / (8*3300)) = ceil(4.07)
    assert slow.predicted_job_wall_s <= 3600 + 800


def test_measured_efficiency_reduces_effective_parallelism():
    full = plan_stage("fit", KEYS, UnitCost(600, 600, 600, 1 * GiB, 5), inner_parallel=False)
    poor = plan_stage("fit", KEYS, UnitCost(600, 600, 600, 1 * GiB, 5), inner_parallel=False, efficiency=0.5)
    assert poor.n_jobs > full.n_jobs


def test_shard_cap_clips_n_jobs_and_dispatch_cap_does_not():
    # shard_cap is the planner's per-stage shard ceiling; dispatch_cap is the campaign dispatch budget
    # and must NOT raise the shard count (the Cycle-1 decouple: dispatch 25, shard 20 -> no re-shard).
    c = Constraints(shard_cap=20, dispatch_cap=25)
    p = plan_stage("x", KEYS, UnitCost(3000, 3000, 3000, 1 * GiB, 5), inner_parallel=False, c=c)
    assert p.n_jobs == 20 and p.predicted_job_wall_s > 3600
    hi = Constraints(shard_cap=20, dispatch_cap=99)
    p_hi = plan_stage("x", KEYS, UnitCost(3000, 3000, 3000, 1 * GiB, 5), inner_parallel=False, c=hi)
    assert p_hi.n_jobs == 20  # a higher dispatch_cap never re-shards


def test_shard_cap_defaults_to_dispatch_cap():
    # omitting shard_cap keeps a single effective cap (back-compat for callers that set one number)
    assert Constraints(dispatch_cap=25).shard_cap == 25
    assert Constraints(dispatch_cap=25, shard_cap=20).shard_cap == 20


def test_a_unit_longer_than_the_target_is_the_irreducible_floor():
    p = plan_stage("fit", ["only"], UnitCost(5000, 5400, 5600, 9 * GiB, 1), inner_parallel=False)
    assert p.irreducible_floor_s == 5400 and p.n_jobs == 1


def test_the_irreducible_floor_is_single_sourced_and_reads_the_startup():
    cost = UnitCost(3200, 3400, 3500, 1 * GiB, 1)
    for c in (Constraints(startup_s=300), Constraints(startup_s=100)):
        assert (
            irreducible_floor_s(cost, c) == plan_stage("f", ["k"], cost, inner_parallel=False, c=c).irreducible_floor_s
        )
    # the budget is target - startup: 3400 s is past a 3300 s budget, inside a 3500 s one
    assert irreducible_floor_s(cost, Constraints(startup_s=300)) == 3400
    assert irreducible_floor_s(cost, Constraints(startup_s=100)) is None


def test_inner_parallel_uses_one_top_level_worker_and_bounded_inner_workers():
    p = plan_stage("study", ["single"], UnitCost(2400, 2600, 2700, 3 * GiB, 1), inner_parallel=True, n_inner=5)
    assert (p.workers, p.inner_workers, p.n_jobs) == (1, 5, 1)


def test_inner_parallel_fans_units_across_jobs():
    tags = [f"a_f{i}" for i in range(15)]
    p = plan_stage("study", tags, UnitCost(1200, 1300, 1400, 2 * GiB, 3), inner_parallel=True, n_inner=5)
    assert p.n_jobs == 6  # ceil(15*1200 / 3300)


def test_timeout_is_derived_with_a_floor():
    p = plan_stage("x", KEYS[:4], UnitCost(10, 10, 10, 1 * GiB, 4), inner_parallel=False)
    assert p.timeout_s == 1800
    big = plan_stage("x", KEYS, UnitCost(1000, 1200, 1500, 1 * GiB, 9), inner_parallel=False)
    assert big.timeout_s >= int(1.5 * big.predicted_job_wall_s)


def test_a_unit_that_cannot_fit_refuses():
    with pytest.raises(RefusalError):
        plan_stage("x", KEYS, UnitCost(1, 1, 1, 40 * GiB, 1), inner_parallel=False)


def test_unmeasured_stage_plans_nothing():
    p = plan_stage("x", KEYS, None, inner_parallel=False)
    assert p.basis == "unmeasured" and p.n_jobs == 0


def test_lpt_partition_is_balanced_disjoint_and_deterministic():
    costs = {f"k{i}": float(c) for i, c in enumerate([9, 7, 6, 5, 5, 4, 3, 2])}
    a = lpt_partition(costs, 3)
    assert a == lpt_partition(dict(reversed(costs.items())), 3)
    assert sorted(k for part in a for k in part) == sorted(costs)
    loads = sorted(sum(costs[k] for k in part) for part in a)
    assert loads[-1] - loads[0] <= 4


# ---- the fair-share cap (ADR-128: real HF, 2026-10-20) -------------------------------------------------
# A cap of 1.3 x the profile sample's peak killed every heavy UNSAMPLED unit: xshot's sportec J03WR9 needed
# 2.21 GB against 2.20 GB, ghost_gk's gradientsports 10506/10517 more than 8.11 GB against 8.11 GB. The cap
# protects the box, so each of the W workers gets its fair share of the usable RAM.


@pytest.mark.parametrize(("peak_gib", "workers"), [(1.6, 8), (6.0, 3), (12.0, 1)])
def test_the_cap_is_the_fair_share_of_usable_ram_and_never_below_the_measured_margin(peak_gib, workers):
    c = Constraints()
    peak = int(peak_gib * GiB)
    p = plan_stage("extract", KEYS, UnitCost(30.0, 40.0, 50.0, peak, 20), inner_parallel=False, c=c)
    assert p.workers == workers and p.cap_bytes is not None
    assert p.cap_bytes == max(int(peak * c.cap_factor), c.usable_bytes // workers)
    assert p.workers * p.cap_bytes <= c.usable_bytes  # the box stays protected
    assert p.cap_bytes >= int(peak * c.cap_factor)


def test_an_inner_parallel_cap_shares_the_ram_with_the_unit_and_its_pool():
    c = Constraints()
    p = plan_stage("cv", ["0000", "0001"], UnitCost(300, 300, 300, 2 * GiB, 5), inner_parallel=True, n_inner=5, c=c)
    assert p.cap_bytes == max(int(2 * GiB * c.cap_factor), c.usable_bytes // (p.inner_workers + 1))


def test_thread_safe_non_inner_stage_plans_one_worker(tmp_path):
    """A thread_safe non-inner stage plans W=1 so its single worker gets the full vCPU thread budget
    (ADR-128 full-utilization; HFU plan T1). The flags come from REAL plan() output, not a hand-built dict."""
    from scripts._stages import LocalStore, make_context, plan
    from tests.scripts import _toy_pipeline
    from tests.scripts._stage_harness import PROV, params_from_args

    pipeline = _toy_pipeline.PIPELINE
    ctx = make_context(
        pipeline,
        params_from_args(pipeline, []),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "d",
    )
    nfit = next(s for s in plan(ctx)["stages"] if s["name"] == "nfit")
    assert nfit["thread_safe"] and not nfit["inner_parallel"]  # the real stage flags, via plan()
    p = plan_stage(
        "nfit",
        ["0000", "0001", "0002", "0003"],
        UnitCost(2000, 2000, 2000, 1 * GiB, 4),
        inner_parallel=nfit["inner_parallel"],
        thread_safe=nfit["thread_safe"],
    )
    assert p.workers == 1  # one worker, full 8-thread budget — NOT _workers(peak)=vcpu at budget 1
    assert p.n_jobs > 1  # the 4 heavy units fan across jobs, not bundled into one W=1 job (plan S2, A D-IMPL-04)


def test_budget_floor_s_tracks_the_current_startup_through_replace():
    """HFU T7 / spec §G-2 (reviewer B D1-IMPL-10 rounds 1+2): the effective minimum work per slot-filled job is
    the ``budget_floor_s`` PROPERTY -- 10 x startup_s unless an explicit budget_floor is set. It is read from the
    CURRENT startup_s, so it tracks a dataclasses.replace(startup_s=...): a stored-derived field would be copied
    from the pre-profile 300 s and pinned at 3000 (the r2 bug). An explicit budget_floor is respected."""
    import dataclasses

    assert Constraints(startup_s=300.0).budget_floor_s == 3000.0  # the placeholder before any measurement
    # the production path: re-profiling via replace(startup_s=15) must give 150, NOT the pinned 3000
    assert dataclasses.replace(Constraints(), startup_s=15.0).budget_floor_s == 150.0
    assert Constraints(startup_s=15.0, budget_floor=1200.0).budget_floor_s == 1200.0  # explicit knob respected


def test_slot_filling_fans_toward_free_slots_bounded_by_budget_floor():
    """HFU T7: with free dispatch slots the planner fans n_jobs toward the share to shorten each job's wall,
    never below budget_floor work per job; free_slots=0 (default) leaves the minimum untouched. Both branches."""
    cost = UnitCost(600, 600, 600, 1 * GiB, 20)
    keys = KEYS[:20]
    base = Constraints(shard_cap=50)  # room to fan; free_slots=0 (default) -> slot-filling off whatever budget_floor
    floored = Constraints(shard_cap=50, budget_floor=1200.0)  # >= 1200s (2 units) of work per job
    for inner in (False, True):
        p0 = plan_stage("x", keys, cost, inner_parallel=inner, n_inner=5, c=base)
        p = plan_stage("x", keys, cost, inner_parallel=inner, n_inner=5, free_slots=40, c=floored)
        assert p.n_jobs > p0.n_jobs  # free slots increased the fan
        assert p.n_jobs <= 40  # never beyond the share
        assert p.n_jobs <= (20 * 600) // 1200  # >= budget_floor work per job => <= 10 jobs


# ---- HFU T11: size the study inner pool to the fold width by the per-child RSS (spec B / B-19 / B-21) -----


def test_study_pool_sizes_to_fold_width_from_store_record():
    """A study unit with a LARGE whole-unit peak but a SMALL per-child peak sizes the inner pool to the fold
    width (5) when the child RSS is recorded -- the whole-unit clamp alone under-sizes it. Fallback (no child
    field, first run): the whole-unit clamp + an unsized-pending note; a child RSS too big to fit 5 records a
    RAM floor (surfaced), never a silent 1."""
    big_unit = 10 * GiB  # _workers(10 GiB) == 2, so the whole-unit clamp alone under-sizes the pool
    sized = plan_stage("study", ["s"], UnitCost(2400, 2600, 2700, big_unit, 1, 2 * GiB), inner_parallel=True, n_inner=5)
    assert sized.inner_workers == 5 and sized.note is None  # fold width: the 2 GiB child RSS fits 5

    first = plan_stage("study", ["s"], UnitCost(2400, 2600, 2700, big_unit, 1), inner_parallel=True, n_inner=5)
    assert first.inner_workers == 2  # the whole-unit clamp, under-sized, before any child RSS is measured
    assert first.note == "study-unsized-pending-measurement"

    tight = plan_stage("study", ["s"], UnitCost(2400, 2600, 2700, big_unit, 1, 5 * GiB), inner_parallel=True, n_inner=5)
    # (usable - floor_cap) // child_cap = (30 - 13) // 6.5 = 2: the pool is sized against the WORKER CAP, not the
    # actual peak, so the tree budget stays within the box (reviewer B D1-IMPL-24). A surfaced floor, never a silent 1.
    assert tight.inner_workers == 2
    assert tight.note is not None and tight.note.startswith("inner-ram-floor")


def test_child_sized_pool_cap_protects_the_box_AND_the_unit_worker():
    """B-21 + reviewer B D1-IMPL-07: the inner branch gives the UNIT WORKER a cap >= its measured margin
    (unit_peak * cap_factor), so a child-sized cap never starves the worker that runs the unit itself, and
    the POOL CHILDREN a child-sized cap, so unit_peak + inner * child_cap stays within the usable RAM at the
    fold width. cap_bytes is the worker's cap; inner_cap_bytes is the pool child's -- two distinct limits."""
    c = Constraints()
    unit_peak, child_peak = 10 * GiB, 2 * GiB
    p = plan_stage(
        "study", ["s"], UnitCost(2400, 2600, 2700, unit_peak, 1, child_peak), inner_parallel=True, n_inner=5, c=c
    )
    assert p.inner_workers == 5 and p.cap_bytes is not None and p.inner_cap_bytes is not None
    assert p.cap_bytes >= int(unit_peak * c.cap_factor)  # the UNIT WORKER is not starved by the child-sized cap
    assert p.inner_cap_bytes == int(child_peak * c.cap_factor)  # the pool children get the child-sized cap
    assert unit_peak + p.inner_workers * p.inner_cap_bytes <= c.usable_bytes  # the box holds the unit plus its pool
    # the TREE BUDGET the executor enforces (worker cap + pool caps) stays within the box (reviewer B D1-IMPL-24)
    assert p.cap_bytes + p.inner_workers * p.inner_cap_bytes <= c.usable_bytes


def test_the_inner_tree_budget_never_exceeds_the_box_for_a_heavy_unit():
    """Reviewer B D1-IMPL-24: a heavy unit whose worker cap (1.3 x peak) dominates must still leave the pool
    within the box. inner is sized against the WORKER CAP, not the actual peak, so the tree budget the executor
    enforces -- cap_bytes + inner_workers * inner_cap_bytes -- never exceeds usable_bytes (it would overcommit a
    32 GiB-class box otherwise: a 20 GiB unit with 2 GiB children once summed to 33.8 GiB)."""
    c = Constraints()
    cost = UnitCost(5000, 5200, 5400, 20 * GiB, 1, 2 * GiB)
    p = plan_stage("study", ["s"], cost, inner_parallel=True, n_inner=5, c=c)
    assert p.cap_bytes is not None and p.inner_cap_bytes is not None
    assert p.cap_bytes >= int(20 * GiB * c.cap_factor)  # the worker keeps its 1.3 x peak margin
    assert p.cap_bytes + p.inner_workers * p.inner_cap_bytes <= c.usable_bytes  # the tree budget is within the box
