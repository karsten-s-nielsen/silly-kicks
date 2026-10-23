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


def test_the_makespan_minimizer_fans_to_the_shortest_finish():
    # ADR-128 Cycle 1b (owner ruling 5): the planner minimizes the stage finish time, not "fill to the
    # target". 179 units x 60 s on 8 workers: 8 jobs run in 3 waves (~480 s) and the fewest-jobs-within-5 %
    # rule keeps it at 8 (10 jobs give the same wall). All stay under the 3600 s job limit, so no remainder.
    c = Constraints(target_s=3600, startup_s=300)
    p = plan_stage("extract", KEYS, UnitCost(60.0, 70.0, 80.0, 1 * GiB, 20), inner_parallel=False, c=c)
    assert p.n_jobs == 8 and p.workers == 8 and p.remainder == ()
    assert p.predicted_job_wall_s <= c.target_s
    slow = plan_stage("extract", KEYS, UnitCost(600.0, 700.0, 800.0, 1 * GiB, 20), inner_parallel=False, c=c)
    assert slow.n_jobs == 8 and slow.predicted_job_wall_s <= c.target_s


def test_measured_efficiency_reduces_effective_parallelism():
    full = plan_stage("fit", KEYS, UnitCost(600, 600, 600, 1 * GiB, 5), inner_parallel=False)
    poor = plan_stage("fit", KEYS, UnitCost(600, 600, 600, 1 * GiB, 5), inner_parallel=False, efficiency=0.5)
    assert poor.n_jobs > full.n_jobs


def test_concurrency_cap_clips_n_and_the_overflow_waits():
    # ADR-128 Cycle 1b: with heavy units the cap clips n_jobs; the target is the job LIMIT, so each job stays
    # under it (one wave of 3000 s + startup = 3300 s) and the units that do not fit the 10 jobs wait (S5.4).
    p = plan_stage(
        "x", KEYS, UnitCost(3000, 3000, 3000, 1 * GiB, 5), inner_parallel=False, c=Constraints(concurrency_cap=10)
    )
    assert p.n_jobs == 10 and p.predicted_job_wall_s <= 3600 and p.remainder


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
    assert p.n_jobs == 8  # ADR-128 Cycle 1b: 15 units in 2 waves on 8 jobs is the shortest finish (fewest jobs)


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


# ---- ADR-128 Cycle 1b: the makespan planner and plan_reduce (S5.2, S5.4) -------------------------
from scripts.hf_jobs import planner as P  # noqa: E402


def test_job_makespan_runs_units_in_whole_waves():
    assert P.job_makespan([2000.0] * 8, 8) == 2000
    assert P.job_makespan([2000.0] * 9, 8) == 4000
    assert P.job_makespan([], 8) == 0.0


def test_the_planner_picks_the_shortest_finish_within_the_grant_and_fewest_jobs_within_5_percent():
    c = P.Constraints(startup_s=120, concurrency_cap=20, target_s=3600)
    cost = P.UnitCost(60, 70, 80, 2**30, 100)
    one = P.plan_stage("x", [f"k{i}" for i in range(5)], cost, inner_parallel=False, c=c)
    assert one.n_jobs == 1  # 5 x 60 s fits one job; a second job saves < 5 % after its startup
    many = P.plan_stage("x", [f"k{i}" for i in range(3961)], cost, inner_parallel=False, c=c)
    assert many.n_jobs == 20 and many.predicted_job_wall_s < c.target_s
    capped = P.plan_stage("x", [f"k{i}" for i in range(3961)], cost, inner_parallel=False, c=c, grant=3)
    assert capped.n_jobs == 3 and capped.predicted_job_wall_s <= c.target_s
    assert capped.partition is not None
    taken = {k for part in capped.partition for k in part}
    assert capped.remainder and not taken & set(capped.remainder) and len(taken) + len(capped.remainder) == 3961


def test_cv_folds_fan_out_when_a_shorter_makespan_wins():
    c = P.Constraints(startup_s=120, concurrency_cap=20)
    # 20 GiB peak -> W = 1 per job: one job runs the 5 folds in 5 waves (2200 s), 5 jobs run them in one (440 s).
    cost = P.UnitCost(440, 450, 460, 20 * 2**30, 5)
    p = P.plan_stage("cv", [f"{i:04d}" for i in range(5)], cost, inner_parallel=False, c=c)
    assert p.n_jobs >= 2


def test_thirteen_long_units_predict_two_waves_at_w8():
    c = P.Constraints(startup_s=120, concurrency_cap=20, target_s=100000)  # no job-length limit
    cost = P.UnitCost(2000, 2000, 2000, 2**30, 13)
    p = P.plan_stage("x", [f"k{i}" for i in range(13)], cost, inner_parallel=False, c=c, grant=1)
    assert p.n_jobs == 1 and p.predicted_job_wall_s == 120 + 4000  # 13 on 8 workers -> 2 waves


def test_the_makespan_search_is_fast_at_scale():
    import time

    c = P.Constraints(startup_s=120, concurrency_cap=20)
    cost = P.UnitCost(60, 70, 80, 2**30, 100)
    t0 = time.monotonic()
    P.plan_stage("x", [f"k{i}" for i in range(3961)], cost, inner_parallel=False, c=c, grant=20)
    assert time.monotonic() - t0 < 1.0


def test_the_incremental_fill_keeps_every_job_under_the_limit_and_covers_the_units():
    import random

    c = P.Constraints(startup_s=120, concurrency_cap=20, target_s=3600)
    for seed in range(50):
        rng = random.Random(seed)  # noqa: S311 -- test fixture, not cryptographic
        per = {f"k{i}": rng.uniform(100.0, 3000.0) for i in range(200)}
        cost = P.UnitCost(max(per.values()), max(per.values()), max(per.values()), 2**30, 200)
        p = P.plan_stage("x", list(per), cost, inner_parallel=False, per_unit=per, c=c, grant=5)
        assert p.partition is not None
        taken = {k for part in p.partition for k in part}
        assert taken.isdisjoint(p.remainder) and taken | set(p.remainder) == set(per)
        for part in p.partition:  # each bin's LPT makespan stays under the limit (the span == job_makespan)
            span = P.job_makespan([per[k] + c.unit_overhead_s for k in part], p.workers)
            if len(part) > 1:  # a lone over-limit unit is allowed to exceed it
                assert c.startup_s + span <= c.target_s + 1e-6


def test_plan_reduce_is_linear_with_and_without_history_and_respects_the_floor():
    c = P.Constraints(store_read_s=0.05, startup_s=60, timeout_floor_s=1800, timeout_safety=1.5)
    assert P.plan_reduce(0, c).timeout_s == 1800 and P.plan_reduce(1, c).timeout_s == 1800
    big = P.plan_reduce(3961, c)
    assert abs(big.wall_s - (60 + 3961 * 0.05)) < 1e-6
    assert P.plan_reduce(3961, c, per_result_s=2.0).timeout_s == int(1.5 * (60 + 3961 * 2.0))


def test_a_partial_grant_leaves_an_unowned_remainder_within_the_job_length_limit():
    """S5.3 / S5.4 through the controller: k granted jobs take only what fits k jobs under `target_s`; the rest
    stays for the next step. 40 units x 2000 s, W = 8 -> a job of 8 units runs one wave (2120 s); a 9th makes
    2 waves (4120 s > 3600), so 2 jobs take 16 and leave 24."""
    from scripts.hf_jobs import campaign as C
    from tests.scripts.hf_jobs.test_campaign import _plan, _st

    st = _st("x", n=40)
    hist = {"x": {"cost": P.UnitCost(2000, 2100, 2200, 2**30, 40), "efficiency": 1.0, "per_unit": None}}
    c = P.Constraints(target_s=3600, startup_s=120)
    first = C.next_actions(_plan([st]), {}, hist, c, grant=lambda a: 2)[0]
    assert first.partition is not None
    assert first.n_jobs == 2 and len(first.partition) == 2 and first.workers == 8
    assert first.predicted_job_wall_s is not None and first.predicted_job_wall_s <= c.target_s
    taken = frozenset(k for part in first.partition for k in part)
    assert len(taken) == 16 and all(len(part) == 8 for part in first.partition)
    owned = {("x", "t_x"): C.Ownership(False, taken)}
    second = C.next_actions(_plan([st]), owned, hist, c)
    assert second[0].partition is not None
    rest = {k for part in second[0].partition for k in part}
    assert rest == set(st["pending_units"]) - taken


def test_ideal_stage_wall_uses_an_unlimited_budget():
    c = P.Constraints(startup_s=120, concurrency_cap=2, target_s=100000)
    per = {f"k{i}": 600.0 for i in range(20)}
    cost = P.UnitCost(600, 600, 600, 2**30, 20)
    ideal = P.ideal_stage_wall(per, cost, c, inner_parallel=False, n_inner=1, efficiency=1.0)
    capped = P.plan_stage("x", sorted(per), cost, inner_parallel=False, per_unit=per, c=c)
    assert ideal < capped.predicted_job_wall_s  # the unlimited budget beats the cap-2 plan
    assert ideal == pytest.approx(120 + 600, rel=0.1)  # ~one 600 s wave


def test_the_incremental_fill_is_fast_at_scale():
    import time

    c = P.Constraints(startup_s=120, concurrency_cap=20, target_s=3600)
    cost = P.UnitCost(2000, 2100, 2200, 2**30, 4000)  # units too long -> the over-limit incremental fill runs
    per = {f"k{i}": 2000.0 for i in range(4000)}
    t0 = time.monotonic()
    p = P.plan_stage("x", list(per), cost, inner_parallel=False, per_unit=per, c=c)
    assert time.monotonic() - t0 < 1.0 and p.remainder  # the fill path ran (remainder non-empty)


def test_a_unit_missing_from_the_per_unit_history_is_costed_at_the_p50():
    """cycle1b run 1 (2026-10-23): the profile measured 16 of a stage's units, and `plan_stage` indexed
    `per_unit[k]` for every key -> KeyError on the first unmeasured unit. A missing unit costs the stage p50."""
    cost = UnitCost(100, 120, 140, 1 * GiB, 16)
    c = Constraints(startup_s=0, concurrency_cap=4)
    keys = [f"u{i:02d}" for i in range(40)]
    partial = {k: 250.0 for k in keys[:16]}
    full = {**partial, **{k: cost.p50_s for k in keys[16:]}}
    got = plan_stage("x", keys, cost, inner_parallel=False, per_unit=partial, c=c)
    want = plan_stage("x", keys, cost, inner_parallel=False, per_unit=full, c=c)
    assert (got.n_jobs, got.predicted_job_wall_s, got.partition) == (
        want.n_jobs,
        want.predicted_job_wall_s,
        want.partition,
    )
