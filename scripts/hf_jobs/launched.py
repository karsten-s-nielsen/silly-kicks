"""The record of the jobs that one driver run launched (HFU launch top-up spec §3.3).

The driver tops up a stage with the units that no running job holds (`driver.decide`). It learns the running jobs
from the HF listing, and a listing can lag behind a launch. So the run keeps its own record: each job it launched,
with its labels and its units. A job of the record that the listing of a step does not show is added to the job
records of that step as a scheduling job, so it holds its units, counts toward the concurrency budget and keeps the
run from a stall.

- A job that a listing has shown ENDED is never added again.
- Any other job that the listing leaves out is added, for at most `UNLISTED_STEPS` listings in a row. Then it is
  dropped and its units are free again.

The two limits of the rule (spec §11): a dropped job can still exist, and then a second job can run its units; and
the record lives in one `driver.run`, so a `resume` inside a lag starts without it.

The record also keeps the store's answer for a job that this run did not launch (its per-job manifest, or "none"),
so each such job costs one store read in a run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from scripts.hf_jobs.failures import JobOutcome, classify
from scripts.hf_jobs.launch import label_of

#: The listings in a row that may leave out a launched job before it is dropped (spec §3.3).
UNLISTED_STEPS: Final = 5


@dataclass
class _Launched:
    labels: dict[str, str]
    units: tuple[str, ...]
    created_at: str
    ended: bool = False
    unlisted: int = 0


class LaunchedJobs:
    """The jobs that one driver run launched, and the store's answers for the other jobs it met."""

    def __init__(self) -> None:
        self._launched: dict[str, _Launched] = {}
        self._remembered: dict[str, tuple[str, ...] | None] = {}

    def add(self, job_id: str, labels: Mapping[str, str], units: Sequence[str], created_at: str) -> None:
        """A job that this run launched: its labels, its units and its launch time (ISO 8601)."""
        self._launched[job_id] = _Launched(dict(labels), tuple(units), created_at)

    def reconcile(self, listing: Sequence[Mapping]) -> tuple[list[dict], list[tuple[str, str]]]:
        """One listing that a step read. Returns ``(held, dropped)``: a scheduling job record for each launched job
        that the listing leaves out and that no listing has shown ended, and ``(job id, in-job label)`` of each job
        that ``UNLISTED_STEPS`` listings in a row have left out (it leaves the record)."""
        listed = {str(job.get("id") or ""): job for job in listing}
        held: list[dict] = []
        dropped: list[tuple[str, str]] = []
        for job_id, rec in list(self._launched.items()):
            if rec.ended:
                continue
            job = listed.get(job_id)
            if job is not None:
                rec.unlisted = 0
                rec.ended = classify(job).outcome is not JobOutcome.IN_FLIGHT
                continue
            rec.unlisted += 1
            if rec.unlisted >= UNLISTED_STEPS:
                dropped.append((job_id, label_of(rec.labels)))
                del self._launched[job_id]
                continue
            held.append(
                {
                    "id": job_id,
                    "labels": dict(rec.labels),
                    "stage": "SCHEDULING",
                    "message": None,
                    "created_at": rec.created_at,
                    "started_at": None,
                    "finished_at": None,
                    "wall_s": None,
                }
            )
        return held, dropped

    def units_of(self, job_id: str) -> tuple[str, ...] | None:
        """The units of a launched job, or the remembered answer of the store; ``None`` when not known."""
        rec = self._launched.get(job_id)
        if rec is not None:
            return rec.units
        return self._remembered.get(job_id)

    def knows(self, job_id: str) -> bool:
        """True for a launched job of the record and for a job whose store answer is remembered."""
        return job_id in self._launched or job_id in self._remembered

    def remember(self, job_id: str, units: Sequence[str] | None) -> None:
        """The store's answer for a job that this run did not launch (``None``: it has no manifest)."""
        self._remembered[job_id] = None if units is None else tuple(units)
