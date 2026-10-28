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
    #: The campaign dispatch budget: how many jobs may be in flight at once (probed against the HF account
    #: limit). Decoupled from `shard_cap` so raising it (Cycle 1: 25) never re-shards a stage (ADR-128; r3 B17).
    dispatch_cap: int = 10
    #: The planner's per-stage shard ceiling. 0 (the default) means "same as `dispatch_cap`" so a single-cap
    #: caller is unchanged; Cycle 1 pins it at 20 while `dispatch_cap` rises to 25 (no re-shard).
    shard_cap: int = 0
    usd_per_hour: float = 0.03  # the cpu-upgrade list price (spec 3); preflight prints the live catalog price
    timeout_safety: float = 1.5
    timeout_floor_s: float = 1800.0
    #: A generous fixed ceiling for actions that cannot derive a timeout from their own measurement (the
    #: profile action; and the first-run fallback for the tiny/reduce/escalate actions). Independent of
    #: target_s (HFU R3): no job timeout is a function of the 30-minute target. The HF 70-min cv observation
    #: sets the per-part scale; chosen from existing HF observations, never a DGX/PC run.
    profile_timeout_s: float = 7200.0
    #: Slot-filling (HFU T7 / spec §G-2): the minimum work per job, so a slot-filled job never shrinks to
    #: startup-sized. 0 (the default) means "derive": read the effective value through the ``budget_floor_s``
    #: PROPERTY (= `10 x startup_s`, so the startup is ~10% of a slot-filled job). An explicit value > 0 wins.
    #: It is NOT derived into this field (that would not survive ``dataclasses.replace``; reviewer B D1-IMPL-10 r2).
    budget_floor: float = 0.0
    cap_factor: float = 1.3
    #: A failed unit is retried until it has failed this many times; then the instance halts and names it.
    unit_attempts: int = UNIT_ATTEMPTS
    #: A stage's jobs may end in ERROR this many times IN A ROW before its instance halts, whatever
    #: the cause (a timeout or a crash before any unit writes no failure record to count on).
    error_resumes: int = 2
    #: The live dispatch back-off (§D) never falls below this many jobs: >= shard_cap so a stage's wave keeps
    #: its parallelism (r6 consider 21). 0 (the default) resolves to shard_cap.
    dispatch_floor: int = 0
    #: After the back-off reaches dispatch_floor, this many MORE consecutive never-started timeouts halt the
    #: instance and surface the real HF ceiling (owner ruling B24: surface, not degrade to slow waves).
    dispatch_halt_after: int = 2

    def __post_init__(self) -> None:
        # shard_cap defaults to the dispatch_cap: a caller that sets one number gets one effective cap.
        if self.shard_cap <= 0:
            object.__setattr__(self, "shard_cap", self.dispatch_cap)
        # dispatch_floor defaults to shard_cap (a stage's wave is never throttled below its parallelism).
        if self.dispatch_floor <= 0:
            object.__setattr__(self, "dispatch_floor", self.shard_cap)

    @property
    def budget_floor_s(self) -> float:
        """The minimum work per slot-filled job (HFU T7 / spec §G-2): an explicit ``budget_floor``, else
        ``10 x startup_s`` so the re-profiled startup is ~10% of it. Computed from the CURRENT ``startup_s`` on
        every read, so it tracks a ``dataclasses.replace(startup_s=...)`` -- the derivation must NOT be a stored
        field, which ``replace`` would copy from the pre-profile value and pin (reviewer B D1-IMPL-10 round 2)."""
        return self.budget_floor if self.budget_floor > 0 else 10.0 * self.startup_s


@dataclasses.dataclass(frozen=True)
class UnitCost:
    p50_s: float
    p95_s: float
    p99_s: float
    peak_rss_bytes: int
    n_samples: int
    #: HFU T11/B-19: the max per-CHILD peak RSS of an inner_parallel stage's pool (sizes the study pool to the
    #: fold width). DEFAULTED (B-20), so `UnitCost(**cost)` on older summaries and positional constructors keep
    #: working; None before any pool child recorded one.
    inner_child_peak_rss_bytes: int | None = None


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
    #: HFU T11 + reviewer B D1-IMPL-07: the POOL CHILD cap of an inner_parallel stage (child_peak * cap_factor),
    #: DISTINCT from ``cap_bytes`` (the unit worker's cap). None for a non-inner stage (no pool).
    inner_cap_bytes: int | None
    peak_rss_bytes: int | None
    irreducible_floor_s: float | None
    basis: str  # "measured" | "unmeasured"
    #: HFU T11: a surfaced planning note for the study inner pool -- "study-unsized-pending-measurement" (no
    #: child RSS yet, the whole-unit clamp applied) or "inner-ram-floor:<k>of<n>" (the child RSS cannot fit the
    #: fold width, surfaced under R3 rather than silently sized to 1). None otherwise.
    note: str | None = None


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


def _slot_fill(min_jobs: int, *, total_work: float, free_slots: int, n: int, c: Constraints) -> int:
    """HFU T7: with free dispatch slots, raise n_jobs toward the share to shorten each job's wall, never
    below ``budget_floor`` work per job (0 = no floor). Bounded by free_slots, shard_cap and the unit count.
    free_slots <= 0 (the default) leaves the minimum untouched, so a non-slot-filling caller is unchanged."""
    if free_slots <= 0:
        return min_jobs
    bf = c.budget_floor_s  # the effective floor from the CURRENT startup (survives replace; reviewer B D1-IMPL-10 r2)
    floor_cap = int(total_work / bf) if bf > 0 else n
    return max(min_jobs, min(free_slots, c.shard_cap, n, max(1, floor_cap)))


def plan_stage(
    stage: str,
    keys: Sequence[str],
    cost: UnitCost | None,
    *,
    inner_parallel: bool,
    thread_safe: bool = False,
    free_slots: int = 0,
    n_inner: int = 1,
    efficiency: float | None = None,
    per_unit: Mapping[str, float] | None = None,
    c: Constraints = Constraints(),  # noqa: B008 -- frozen dataclass: an immutable default
) -> StagePlan:
    n = len(keys)
    if cost is None or n == 0:
        return StagePlan(stage, n, 0, 0, 0, None, 0.0, 0.0, 0, None, None, None, None, "unmeasured")
    eff = efficiency if efficiency is not None else 1.0
    budget = max(1.0, c.target_s - c.startup_s)
    # The cap protects the BOX, so each process gets its fair share of the usable RAM, never less than
    # 1.3 x the measured peak: a cap of 1.3 x the profile SAMPLE's peak killed every heavier unsampled unit
    # on real HF (2026-10-20). W x cap stays within the usable RAM either way.
    floor_cap = int(cost.peak_rss_bytes * c.cap_factor)
    note: str | None = None
    if inner_parallel:
        # HFU T11/B-19/B-21: size the pool by the per-CHILD peak when it is measured (the whole-unit peak
        # over-counts -- each child needs a fraction, not a whole unit). The unit process and `inner` children
        # share the box. Reserve the WORKER's CAP (floor_cap = 1.3 x peak), not its actual peak, so the sum of
        # ceilings -- worker cap + inner x child cap -- stays within the usable RAM and the tree budget the
        # executor enforces never exceeds the box (reviewer B D1-IMPL-24). Before any child RSS is recorded,
        # fall back to the whole-unit clamp and surface that the study is unsized.
        child = cost.inner_child_peak_rss_bytes
        if child:
            child_cap = int(child * c.cap_factor)
            fit = max(1, (c.usable_bytes - floor_cap) // max(1, child_cap))
            inner = max(1, min(c.vcpu, n_inner, fit))
            # Two distinct caps (reviewer B D1-IMPL-07): the POOL CHILDREN get the child-sized cap; the UNIT
            # WORKER (which runs the unit itself, peak = cost.peak_rss_bytes) gets its own margin -- the RAM left
            # after the pool, never below floor_cap -- so the child-sized cap never starves the worker.
            inner_cap = child_cap
            cap = max(floor_cap, c.usable_bytes - inner * child_cap)
            if inner < n_inner:  # the child RSS cannot fit the fold width: a RAM floor, surfaced (not a silent 1)
                note = f"inner-ram-floor:{inner}of{n_inner}"
        else:
            inner = max(1, min(c.vcpu, n_inner, _workers(cost.peak_rss_bytes, c)))
            cap = max(floor_cap, c.usable_bytes // (inner + 1))  # the unit's process and its pool share the box
            inner_cap = cap  # child peak unmeasured: the pool children share the whole-unit clamp (same as before)
            note = "study-unsized-pending-measurement"
        n_jobs = _slot_fill(
            max(1, min(n, c.shard_cap, math.ceil(n * cost.p50_s / budget))),
            total_work=n * cost.p50_s,
            free_slots=free_slots,
            n=n,
            c=c,
        )
        per_job = math.ceil(n / n_jobs)
        wall = c.startup_s + per_job * cost.p50_s
        wall_p99 = c.startup_s + per_job * cost.p99_s
        workers, partition = 1, None
    else:
        # A thread_safe non-inner stage runs at W=1 so its single worker gets the full vCPU thread budget
        # (thread_budget(vcpu, 1, 1) = vcpu); units fan across n_jobs by wall (ADR-128 full-utilization).
        workers = 1 if thread_safe else max(1, _workers(cost.peak_rss_bytes, c))
        cap = max(floor_cap, c.usable_bytes // workers)
        inner = 1
        inner_cap = None  # a non-inner stage has no pool
        par = workers * eff
        total = sum(per_unit.values()) if per_unit else cost.p50_s * n
        n_jobs = _slot_fill(
            max(1, min(n, c.shard_cap, math.ceil(total / (par * budget)))),
            total_work=total,
            free_slots=free_slots,
            n=n,
            c=c,
        )
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
        inner_cap,
        cost.peak_rss_bytes,
        floor,
        "measured",
        note,
    )
