"""The executor's memory admission gate (HFU plan R3-8 S3 item 6, MEM; ADR-128 Amendment 2026-10-31).

For a stage with per-unit peak history (the newest EARLIER generation of the same ``(run_key, stage)``), each unit
reserves ``cap_factor`` (1.3) times its predicted peak RSS. A worker gets its next unit only while the other live
workers' ``max(measured RSS, reservation of the unit in flight)`` plus its own ``max(measured RSS, reservation of the
next unit)`` fit the job's usable RAM. An idle live worker counts its measured RSS, so heap that a heavy unit leaves
behind is counted.

Progress rule (S2B-r2-01, S2B-r3-04): retained heap never stalls the stage. (i) The idle workers whose measured RSS is
part of the reason a claim is refused exit, which returns their heap to the box. (ii) Only when no idle worker holds
retained heap and no unit is in flight, the claim is admitted whatever its reservation: one unit alone always runs.

This module is the decision logic only (pure); `executor.LocalExecutor._dispatch` measures and acts on it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from scripts.hf_jobs.planner import Constraints

#: The reservation factor: the planner's per-worker cap factor (`Constraints.cap_factor`), single-sourced.
CAP_FACTOR = Constraints().cap_factor


@dataclasses.dataclass(frozen=True)
class WorkerMem:
    """One live worker as the gate sees it: its measured resident memory (0 when unmeasured) and the unit it runs
    (None when idle)."""

    rss: int
    inflight: str | None


@dataclasses.dataclass(frozen=True)
class Decision:
    """``grant`` the claim, ``wait`` for a unit in flight to finish, or ``stop`` the listed idle workers first."""

    kind: str
    stop: tuple[str, ...] = ()


GRANT = Decision("grant")
WAIT = Decision("wait")


@dataclasses.dataclass(frozen=True)
class MemGate:
    reservations: Mapping[str, int]
    budget: int
    #: A unit with no earlier record reserves the stage MAXIMUM, never a percentile (plan item 6).
    default: int

    @classmethod
    def from_peaks(cls, peaks: Mapping[str, int], *, budget: int, cap_factor: float = CAP_FACTOR) -> MemGate:
        if not peaks or max(peaks.values()) <= 0:
            raise ValueError("a memory gate needs a measured per-unit peak RSS (an earlier generation's records)")
        res = {k: int(v * cap_factor) for k, v in peaks.items()}
        return cls(res, int(budget), max(res.values()))

    def reservation(self, key: str) -> int:
        return self.reservations.get(key, self.default)

    def _held(self, m: WorkerMem) -> int:
        return max(m.rss, self.reservation(m.inflight) if m.inflight is not None else 0)

    def fresh_fits(self, live: Mapping[str, WorkerMem], key: str) -> bool:
        """Whether a NEW worker (no heap yet) would be admitted for ``key``: the condition to launch one in a free
        slot, so a stopped worker's slot is refilled only once the gate admits it."""
        return sum(self._held(m) for m in live.values()) + self.reservation(key) <= self.budget

    def box_victim(self, live: Mapping[str, WorkerMem]) -> str | None:
        """The box guard: when the live workers' summed measured RSS passes the budget, the BUSY worker with the
        largest excess of RSS over its reservation (the executor kills it and its unit goes to the memory
        escalation). None while the sum fits, or when no worker is busy (an idle worker's heap is rule (i)'s)."""
        if sum(m.rss for m in live.values()) <= self.budget:
            return None
        busy = [(m.rss - self.reservation(m.inflight), w) for w, m in live.items() if m.inflight is not None]
        return max(busy)[1] if busy else None

    def decide(self, live: Mapping[str, WorkerMem], wid: str, key: str) -> Decision:
        """The decision on worker ``wid``'s claim of ``key``."""
        res = self.reservation(key)
        others = sum(self._held(m) for w, m in live.items() if w != wid)
        own = live[wid].rss if wid in live else 0
        if others + max(own, res) <= self.budget:
            return GRANT
        # Rule (i): an idle worker's measured heap counts in full; the requester's own heap is a reason only where it
        # is above the reservation it asks for (below it, the reservation is what counts). Stop them only when that
        # admits the claim: with every one of them gone (the requester down to its reservation), the claim fits
        # (review S3B-08: never churn workers while the claim still waits for a unit in flight).
        retained = tuple(
            sorted(w for w, m in live.items() if m.inflight is None and m.rss > 0 and (w != wid or m.rss > res))
        )
        busy = sum(self._held(m) for w, m in live.items() if w != wid and w not in retained)
        if retained and busy + res <= self.budget:
            return Decision("stop", retained)
        if not any(m.inflight is not None for m in live.values()):
            if retained:  # nothing in flight: the retained heap is all that stands in the way
                return Decision("stop", retained)
            return GRANT  # rule (ii): nothing in flight and no retained heap, so this unit runs alone
        return WAIT
