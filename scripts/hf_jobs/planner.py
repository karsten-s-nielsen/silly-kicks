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
    #: Per-unit store overhead (claim, outcome, result write, checksum), added to each unit's wall in the
    #: makespan model (S5.4). Measured; 0 before a measurement.
    unit_overhead_s: float = 0.0
    #: The measured wall per RESULT read in a reduce (S5.2), from the §7.1 store probe. `plan_reduce` uses it
    #: before a stage has its own measured reduce cost. `campaign.planning_constraints` sets it from
    #: `datasets.store_settings().per_result_read_s`.
    store_read_s: float = 0.05
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
    #: Units left over when the best plan within the grant/cap still has a job over `target_s` (S5.4): they
    #: stay pending and unowned, so the next step plans them. A round-robin action never has a remainder.
    remainder: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ReducePlan:
    n_results: int
    wall_s: float
    timeout_s: int


def plan_reduce(n_results: int, c: Constraints, *, per_result_s: float | None = None) -> ReducePlan:
    """S5.2: a reduce reads each result once; its wall is linear in the results. Before history, the cost per
    result is the store probe's ``store_read_s``; with history, the measured seconds per result."""
    per = c.store_read_s if per_result_s is None else per_result_s
    wall = c.startup_s + max(0, n_results) * per
    return ReducePlan(n_results, wall, int(max(c.timeout_floor_s, c.timeout_safety * wall)))


def ideal_stage_wall(
    per_unit: Mapping[str, float],
    cost: UnitCost,
    c: Constraints,
    *,
    inner_parallel: bool,
    n_inner: int,
    efficiency: float | None,
) -> float:
    """S7.2 bar 1: the stage's wall with an UNLIMITED budget, from measured unit walls (plus the startup).
    The jobs the scheduler would launch with no grant ceiling -- the floor the schedule-efficiency bar
    measures the real makespan against."""
    unlimited = dataclasses.replace(c, concurrency_cap=max(1, len(per_unit)))
    p = plan_stage(
        "ideal",
        sorted(per_unit),
        cost,
        inner_parallel=inner_parallel,
        n_inner=n_inner,
        efficiency=efficiency,
        per_unit=per_unit,
        c=unlimited,
    )
    return p.predicted_job_wall_s


def job_makespan(unit_costs: Sequence[float], workers: int) -> float:
    """The wall of ONE job: LPT list scheduling of its units on its W workers (owner ruling 12). Units run
    whole, so 8 units of 2000 s on 8 workers take 2000 s and a 9th makes it 4000 s; a fluid ``sum / W`` would
    say 2250 s."""
    loads = [0.0] * max(1, int(workers))
    for cost in sorted(unit_costs, reverse=True):
        i = min(range(len(loads)), key=lambda j: (loads[j], j))
        loads[i] += cost
    return max(loads) if len(unit_costs) else 0.0


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
    grant: int | None = None,
    lpt: bool = True,
) -> StagePlan:
    """Minimize the stage finish time within the grant/cap (S5.4, owner ruling 12): a job's wall is its LPT
    makespan on its W workers (units run whole, in waves), plus the startup; among plans within 5 % of the
    best finish, the one with the fewest jobs wins (each job costs a startup). A job over ``target_s`` is the
    limit: the plan fills jobs incrementally and leaves the overflow as ``remainder`` for the next step."""
    n = len(keys)
    if cost is None or n == 0:
        return StagePlan(stage, n, 0, 0, 0, None, 0.0, 0.0, 0, None, None, None, "unmeasured")
    eff = efficiency if efficiency is not None else 1.0
    # The cap protects the BOX, so each process gets its fair share of the usable RAM, never less than
    # 1.3 x the measured peak: a cap of 1.3 x the profile SAMPLE's peak killed every heavier unsampled unit
    # on real HF (2026-10-20). W x cap stays within the usable RAM either way.
    floor_cap = int(cost.peak_rss_bytes * c.cap_factor)
    ceiling = max(1, min(n, int(grant) if grant is not None else c.concurrency_cap))
    # Exhaustive for a small ceiling (exact fewest-jobs, CCTP-IMPL-10: 15 not 16); coarse at scale (speed).
    probe = (
        list(range(1, ceiling + 1))
        if ceiling <= 24
        else sorted({1, 2, 3, 4, 5, 8, 10, 12, 16, 20, ceiling} & set(range(1, ceiling + 1)))
    )
    remainder: tuple[str, ...] = ()
    if inner_parallel:
        inner = max(1, min(c.vcpu, n_inner, _workers(cost.peak_rss_bytes, c)))
        cap = max(floor_cap, c.usable_bytes // (inner + 1))  # the unit's process and its pool share the box
        workers, partition = 1, None

        def inner_wall(nj: int) -> float:
            return c.startup_s + math.ceil(n / nj) * cost.p50_s

        walls = {j: inner_wall(j) for j in probe}
        best = min(walls.values())
        n_jobs = min(j for j, w in walls.items() if w <= 1.05 * best)
        per_job = math.ceil(n / n_jobs)
        wall = c.startup_s + per_job * cost.p50_s
        wall_p99 = c.startup_s + per_job * cost.p99_s
    else:
        workers = max(1, _workers(cost.peak_rss_bytes, c))
        cap = max(floor_cap, c.usable_bytes // workers)
        inner = 1
        costs = {k: (per_unit[k] if per_unit else cost.p50_s) + c.unit_overhead_s for k in keys}

        def job_wall(part: Sequence[str]) -> float:  # owner ruling 12: LPT on the job's W workers, in waves
            return c.startup_s + max(job_makespan([costs[k] for k in part], workers) / eff, cost.p95_s)

        def finish(nj: int) -> tuple[float, tuple[tuple[str, ...], ...]]:
            parts = lpt_partition(costs, nj)
            ordered = tuple(tuple(sorted(p, key=lambda k: (-costs[k], k))) for p in parts)  # LPT order per job
            return max(job_wall(p) for p in ordered), ordered

        trials = {j: finish(j) for j in probe}
        best = min(w for w, _ in trials.values())
        n_jobs = min(j for j, (w, _) in trials.items() if w <= 1.05 * best)
        wall, parts = trials[n_jobs]
        partition = parts if (per_unit or lpt) else None
        if wall > c.target_s and lpt:
            # Incremental fill (r4 B R4-01): units arrive in LPT order (longest first); a bin's makespan is
            # kept by its W worker loads (the next unit goes to the least-loaded worker). A unit that would
            # push a bin over the limit goes to the remainder (but a unit over the limit still runs alone).
            bins: list[list[str]] = [[] for _ in range(ceiling)]
            loads: list[list[float]] = [[0.0] * max(1, workers) for _ in range(ceiling)]
            spans = [0.0] * ceiling
            overflow: list[str] = []

            def wall_of(span: float) -> float:
                return c.startup_s + max(span / eff, cost.p95_s)

            for k in sorted(costs, key=lambda k: (-costs[k], k)):
                j = min(range(ceiling), key=lambda i: (loads[i][0], spans[i], i))  # the globally least-loaded worker
                new_span = max(spans[j], loads[j][0] + costs[k])
                if not bins[j] or wall_of(new_span) <= c.target_s:  # a unit over the limit still runs alone
                    heapq.heapreplace(loads[j], loads[j][0] + costs[k])
                    spans[j] = new_span
                    bins[j].append(k)
                else:
                    overflow.append(k)
            parts = tuple(tuple(b) for b in bins if b)
            n_jobs, partition = len(parts), parts
            wall = max(wall_of(spans[i]) for i, b in enumerate(bins) if b)
            remainder = tuple(sorted(overflow))
        taken = [k for part in parts for k in part] if partition is not None else keys
        busiest = max((len(p) for p in parts), default=math.ceil(len(taken) / max(1, n_jobs)))
        wall_p99 = c.startup_s + max(
            job_makespan([cost.p99_s + c.unit_overhead_s] * busiest, workers) / eff, cost.p99_s
        )
    floor = irreducible_floor_s(cost, c)
    timeout = int(max(c.timeout_floor_s, c.timeout_safety * wall_p99))
    usd = n_jobs * wall / 3600.0 * c.usd_per_hour
    return StagePlan(
        stage,
        n,
        n_jobs,
        workers,
        inner,
        partition,
        wall,
        usd,
        timeout,
        cap,
        cost.peak_rss_bytes,
        floor,
        "measured",
        remainder,
    )
