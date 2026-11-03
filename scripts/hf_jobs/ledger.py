"""The job record the driver reads (spec 5.5). The full per-unit record lives in the store
(`<key>.ledger.json`, written by `_stages.run_unit`); job records come from the HF API (`JobInfo` via
labels). `job_record` turns one live `JobInfo` into the compact dict `failures.classify` and the driver
read (ADR-128 amendment 2026-11-02)."""

from __future__ import annotations


def _iso(dt) -> str | None:
    return None if dt is None else dt.isoformat()


def job_record(info) -> dict:
    started, finished = getattr(info, "started_at", None), getattr(info, "finished_at", None)
    wall = (finished - started).total_seconds() if (started and finished) else None
    status = getattr(info, "status", None)
    return {
        "id": info.id,
        "labels": dict(info.labels or {}),
        "stage": getattr(status, "stage", None),
        "message": getattr(status, "message", None),
        "created_at": _iso(getattr(info, "created_at", None)),
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "wall_s": wall,
    }
