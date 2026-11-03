"""Message-driven failure classification for the HF Jobs driver.

The controller rebuilds run state from ``api.list_jobs(labels=...)``. Each job
carries a terminal ``status.stage`` and, on failure, a ``status.message``. This
module maps one job record to a :class:`Verdict` that tells the driver what to do.

The reliable signal is the MESSAGE, never ``started_at``: the real storage-limit
403 appears with ``started_at`` both set and None (memory ``hf-storage-limit``).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass


class JobOutcome(enum.Enum):
    """What the driver does about one job record."""

    COMPLETED = "completed"  # terminal success
    CANCELED = "canceled"  # terminal, operator-canceled, not an error
    IN_FLIGHT = "in_flight"  # RUNNING / SCHEDULING / not yet terminal
    FATAL = "fatal"  # class 1: stop the run, never retry, surface to operator
    RERUN = "rerun"  # class 2: timeout/partial, re-run the missing units, bounded
    RETRY = "retry"  # class 3: deterministic unit crash, bounded retry then report


@dataclass(frozen=True)
class Verdict:
    outcome: JobOutcome
    reason: str = ""
    sizing: bool = False  # True for resource faults (OOM / disk evict): report as sizing
    detail: str = ""  # the raw status.message, for operator surfacing on FATAL


def classify(record: Mapping) -> Verdict:
    """Map one job record (``ledger.job_record`` shape: ``stage`` + ``message``)
    to a driver action. The driver serializes live ``JobInfo`` through
    ``job_record`` first, so this operates on the same shape the tests hold.

    Routing is by MESSAGE substring and is ORDER-SENSITIVE: a resource fault
    (OOM / disk evict) carries ``exit code`` text too, so it is matched before
    the generic unit-crash branch. An ERROR that matches no known signature is
    FATAL/``unclassified`` (fail loud, never silently retry an unseen fault).
    """
    stage = (record.get("stage") or "").upper()
    if stage == "COMPLETED":
        return Verdict(JobOutcome.COMPLETED)
    if stage in ("CANCELED", "DELETED"):  # both terminal in launch.TERMINAL + hf_hub JobStage
        return Verdict(JobOutcome.CANCELED, stage.lower())
    if stage != "ERROR":  # RUNNING / SCHEDULING / absent: not yet terminal
        return Verdict(JobOutcome.IN_FLIGHT, stage.lower())

    message = record.get("message") or ""
    low = message.lower()

    # Class 1 FATAL STOP: infra / auth / config faults that recur identically.
    if "storage limit reached" in low or ("403" in message and "storage" in low):
        return Verdict(JobOutcome.FATAL, "storage_limit_403", detail=message)
    if "invalid username or password" in low:
        # A 401 resolving a bucket/dataset at MOUNT time is a TRANSIENT CSI mount-sidecar flake, not a real
        # auth denial: `preflight.check_api` already proved the token, and the same private repo mounts fine
        # for most jobs (the 2026-11-02 retrain: 63 jobs mounted `tc3`, a few flaked 401). So retry it, bounded
        # by `attempts`; a genuinely bad token would be caught at preflight, and a persistent flake exhausts
        # attempts and is reported -- it never FATAL-stops the whole campaign on one transient mount.
        return Verdict(JobOutcome.RETRY, "mount_auth_flake", detail=message)
    if "init container exhausted retries" in low:
        return Verdict(JobOutcome.FATAL, "mount_init_retries", detail=message)
    if "invalid image name" in low or "image specification error" in low:
        return Verdict(JobOutcome.FATAL, "invalid_image", detail=message)
    if "volume mount failed" in low:
        # A generic CSI mount fault with no recognized sub-signature (the specific ones -- a 401, a storage
        # 403, "init container exhausted retries" -- are matched ABOVE) is a TRANSIENT mount-sidecar flake,
        # same class as the 401: RETRY bounded by `attempts`, never a FATAL stop of the whole campaign on one
        # transient mount. A persistent mount problem exhausts attempts and is reported. `_attribute` routes
        # `mount_failed` like `mount_auth_flake` -- a per-DATASET fault, so a mount flake never charges a unit.
        return Verdict(JobOutcome.RETRY, "mount_failed", detail=message)

    # Class 2 RERUN: partial progress, re-run the missing units (bounded).
    if low.startswith("job timeout"):
        return Verdict(JobOutcome.RERUN, "timeout", detail=message)

    # Class 3 RETRY: unit-level failure, bounded retry then report. OOM / disk
    # evict are resource faults: still class 3, but flagged ``sizing``.
    if "oomkilled" in low:
        return Verdict(JobOutcome.RETRY, "oom", sizing=True, detail=message)
    if "ephemeral local storage" in low or ("evicted" in low and "storage" in low):
        return Verdict(JobOutcome.RETRY, "disk_evict", sizing=True, detail=message)
    if "exit code" in low:
        return Verdict(JobOutcome.RETRY, "unit_crash", detail=message)

    # Residual safety net: an ERROR with no known signature stops the run.
    return Verdict(JobOutcome.FATAL, "unclassified", detail=message)
