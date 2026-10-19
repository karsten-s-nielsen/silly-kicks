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


def test_concurrency_cap_clips_n_and_the_plan_says_so():
    p = plan_stage(
        "x", KEYS, UnitCost(3000, 3000, 3000, 1 * GiB, 5), inner_parallel=False, c=Constraints(concurrency_cap=10)
    )
    assert p.n_jobs == 10 and p.predicted_job_wall_s > 3600


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
