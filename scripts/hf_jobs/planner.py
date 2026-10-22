"""Pure planning (spec 5.5): N jobs x W workers per stage from measured unit costs, a 60-minute target,
the cpu-upgrade box (8 vCPU, ~30 GB usable) and the account concurrency cap. Outputs never depend on
the plan (INV-1), so the planner may partition freely (LPT when per-unit costs are known)."""

from __future__ import annotations

import dataclasses
import heapq
import math
from collections.abc import Mapping, Sequence

from scripts._stages import UNIT_ATTEMPTS

GiB = 1024**3


@dataclasses.dataclass(frozen=True)
class Constraints:
    vcpu: int = 8
    usable_bytes: int = 30 * GiB
    headroom_bytes: int = 2 * GiB
    target_s: float = 3600.0
    startup_s: float = 300.0  # before any history; campaign.planning_constraints passes the committed median
    concurrency_cap: int = 10
    usd_per_hour: float = 0.03  # the cpu-upgrade list price (spec 3); preflight prints the live catalog price
    timeout_safety: float = 1.5
    timeout_floor_s: float = 1800.0
    cap_factor: float = 1.3
    #: A failed unit is retried until it has failed this many times; then the instance halts and names it.
    unit_attempts: int = UNIT_ATTEMPTS
    #: A stage's jobs may end in ERROR this many times IN A ROW before its instance halts, whatever
    #: the cause (a timeout or a crash before any unit writes no failure record to count on).
    error_resumes: int = 2


@dataclasses.dataclass(frozen=True)
class UnitCost:
    p50_s: float
    p95_s: float
    p99_s: float
    peak_rss_bytes: int
    n_samples: int


@dataclasses.dataclass(frozen=True)
class StagePlan:
    stage: str
    n_units: int
    n_jobs: int
    workers: int
    inner_workers: int
    partition: tuple[tuple[str, ...], ...] | None  # None: round-robin `--shard I/N`
    predicted_job_wall_s: float
    predicted_cost_usd: float
    timeout_s: int
    cap_bytes: int | None
    peak_rss_bytes: int | None
    irreducible_floor_s: float | None
    basis: str  # "measured" | "unmeasured"


def lpt_partition(costs: Mapping[str, float], n_jobs: int) -> list[list[str]]:
    """Longest-processing-time greedy: biggest unit first onto the least-loaded job. Ties break by key, so
    the partition is a pure function of the costs (input order is irrelevant)."""
    n_jobs = max(1, n_jobs)
    heap = [(0.0, j) for j in range(n_jobs)]
    parts: list[list[str]] = [[] for _ in range(n_jobs)]
    for key in sorted(costs, key=lambda k: (-costs[k], k)):
        load, j = heapq.heappop(heap)
        parts[j].append(key)
        heapq.heappush(heap, (load + costs[key], j))
    return [sorted(p) for p in parts if p]


def _workers(peak: int, c: Constraints) -> int:
    from scripts._parallel_launch import size_workers  # RefusalError when one unit cannot fit

    return size_workers(int(peak * c.cap_factor), c.usable_bytes + c.headroom_bytes, c.vcpu, c.headroom_bytes)


def irreducible_floor_s(cost: UnitCost, c: Constraints) -> float | None:
    """A unit whose p95 wall exceeds the per-job budget (target minus startup) is the irreducible wall
    floor: no N or W brings that stage under the target (spec 5.5). Single-sourced for the plan and the
    committed summary."""
    budget = max(1.0, c.target_s - c.startup_s)
    return cost.p95_s if cost.p95_s > budget else None


def plan_stage(
    stage: str,
    keys: Sequence[str],
    cost: UnitCost | None,
    *,
    inner_parallel: bool,
    n_inner: int = 1,
    efficiency: float | None = None,
    per_unit: Mapping[str, float] | None = None,
    c: Constraints = Constraints(),  # noqa: B008 -- frozen dataclass: an immutable default
) -> StagePlan:
    n = len(keys)
    if cost is None or n == 0:
        return StagePlan(stage, n, 0, 0, 0, None, 0.0, 0.0, 0, None, None, None, "unmeasured")
    eff = efficiency if efficiency is not None else 1.0
    budget = max(1.0, c.target_s - c.startup_s)
    # The cap protects the BOX, so each process gets its fair share of the usable RAM, never less than
    # 1.3 x the measured peak: a cap of 1.3 x the profile SAMPLE's peak killed every heavier unsampled unit
    # on real HF (2026-10-20). W x cap stays within the usable RAM either way.
    floor_cap = int(cost.peak_rss_bytes * c.cap_factor)
    if inner_parallel:
        inner = max(1, min(c.vcpu, n_inner, _workers(cost.peak_rss_bytes, c)))
        cap = max(floor_cap, c.usable_bytes // (inner + 1))  # the unit's process and its pool share the box
        n_jobs = max(1, min(n, c.concurrency_cap, math.ceil(n * cost.p50_s / budget)))
        per_job = math.ceil(n / n_jobs)
        wall = c.startup_s + per_job * cost.p50_s
        wall_p99 = c.startup_s + per_job * cost.p99_s
        workers, partition = 1, None
    else:
        workers = max(1, _workers(cost.peak_rss_bytes, c))
        cap = max(floor_cap, c.usable_bytes // workers)
        inner = 1
        par = workers * eff
        total = sum(per_unit.values()) if per_unit else cost.p50_s * n
        n_jobs = max(1, min(n, c.concurrency_cap, math.ceil(total / (par * budget))))
        if per_unit:
            parts = lpt_partition(per_unit, n_jobs)
            n_jobs = len(parts)
            partition = tuple(tuple(p) for p in parts)
            busiest = max(sum(per_unit[k] for k in p) for p in parts)
        else:
            partition = None
            busiest = math.ceil(n / n_jobs) * cost.p50_s
        wall = c.startup_s + max(busiest / par, cost.p95_s)
        wall_p99 = c.startup_s + max(math.ceil(n / n_jobs) * cost.p99_s / par, cost.p99_s)
    floor = irreducible_floor_s(cost, c)
    timeout = int(max(c.timeout_floor_s, c.timeout_safety * wall_p99))
    usd = n_jobs * wall / 3600.0 * c.usd_per_hour
    return StagePlan(
        stage, n, n_jobs, workers, inner, partition, wall, usd, timeout, cap, cost.peak_rss_bytes, floor, "measured"
    )
