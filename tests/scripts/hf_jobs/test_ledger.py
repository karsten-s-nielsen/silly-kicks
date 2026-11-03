"""``ledger.job_record``: the compact record the driver reads from one HF ``JobInfo``.

Driven by the REAL ``huggingface_hub`` ``JobInfo`` / ``JobStatus`` dataclasses (not a hand-built double), so
the serializer is pinned against the shape the live ``list_jobs`` returns (review HFUD-IMPL-06 / A-TEST-03).
"""

from __future__ import annotations

import pytest

from scripts.hf_jobs.ledger import job_record

_jobs_api = pytest.importorskip("huggingface_hub._jobs_api")


def _jobinfo(**resp):
    """A real ``JobInfo`` built from an API-response-shaped dict (the shape ``list_jobs`` parses). ``owner`` and
    ``status`` are the only structurally-required sub-objects."""
    resp.setdefault("owner", {"id": "u1", "name": "silly-kicks", "type": "user"})
    return _jobs_api.JobInfo(**resp)


def test_job_record_extracts_stage_message_labels_and_wall() -> None:
    info = _jobinfo(
        id="abc123",
        status={"stage": "ERROR", "message": "Job failed with exit code: 1. Reason: Error"},
        labels={"campaign": "cycle1", "stage": "extract", "gens": "deadbeef", "kind": "run"},
        createdAt="2026-11-02T02:55:00.000000Z",
        startedAt="2026-11-02T02:56:00.000000Z",
        finishedAt="2026-11-02T03:06:00.000000Z",
    )
    rec = job_record(info)
    assert rec["id"] == "abc123"
    assert rec["stage"] == "ERROR"
    assert rec["message"] == "Job failed with exit code: 1. Reason: Error"
    assert rec["labels"]["gens"] == "deadbeef" and rec["labels"]["kind"] == "run"
    assert rec["created_at"] == info.created_at.isoformat()
    assert rec["wall_s"] == 600.0  # finished - started


def test_job_record_handles_a_job_that_never_started() -> None:
    """The real storage-limit 403 appears with ``started_at`` None (memory ``hf-storage-limit``); the record
    must still serialize, with ``wall_s`` None rather than raising."""
    info = _jobinfo(
        id="job-1",
        status={"stage": "ERROR", "message": "Volume mount failed: ... storage limit reached"},
        labels={},
    )
    rec = job_record(info)
    assert rec["started_at"] is None and rec["wall_s"] is None and rec["stage"] == "ERROR"
