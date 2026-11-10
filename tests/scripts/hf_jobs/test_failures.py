"""Failure classification, verified against REAL captured HF ``JobInfo`` records.

The fixture ``fixtures/real_jobinfo.json`` holds real ``JobInfo`` serializations
(``status.stage``, ``status.message``, ``started_at`` ...) captured from the
silly-kicks HF account on 2026-11-02. Every failure signature here occurred on a
real job. No hand-built records. See the memory ``tdd-tests-from-real-data``: the
two earlier HFU cycles failed because the failure handling was tested against
invented fixtures that encoded wrong assumptions.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from scripts.hf_jobs.failures import JobOutcome, classify

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "real_jobinfo.json"


def _records() -> dict[str, dict]:
    data = json.loads(_FIXTURE.read_text())
    return {r["_class"]: r for r in data}


# Fixture class key -> (outcome, reason, sizing). The owner confirmed the mapping
# (2026-11-02): infra/auth/config faults all FATAL; timeout re-runs; unit crashes
# plus OOM/disk-evict retry (the resource faults flagged ``sizing``).
_EXPECTED = {
    "storage_403_started_none": (JobOutcome.FATAL, "storage_limit_403", False),
    "storage_403_started_set": (JobOutcome.FATAL, "storage_limit_403", False),
    "mount_401_auth": (JobOutcome.RETRY, "mount_auth_flake", False),
    "mount_init_exhausted": (JobOutcome.FATAL, "mount_init_retries", False),
    "invalid_image": (JobOutcome.FATAL, "invalid_image", False),
    "job_timeout": (JobOutcome.RERUN, "timeout", False),
    "exit_1": (JobOutcome.RETRY, "unit_crash", False),
    "exit_2": (JobOutcome.RETRY, "unit_crash", False),
    "exit_3": (JobOutcome.RETRY, "unit_crash", False),
    "exit_127": (JobOutcome.RETRY, "unit_crash", False),
    "oom_137": (JobOutcome.RETRY, "oom", True),
    "disk_evict": (JobOutcome.RETRY, "disk_evict", True),
    "completed_extract": (JobOutcome.COMPLETED, "", False),
    "completed_study": (JobOutcome.COMPLETED, "", False),
    "completed_assemble": (JobOutcome.COMPLETED, "", False),
    "canceled": (JobOutcome.CANCELED, "", False),
}


@pytest.mark.parametrize("key", list(_EXPECTED))
def test_classify_real_record(key: str) -> None:
    rec = _records()[key]
    want_outcome, want_reason, want_sizing = _EXPECTED[key]
    verdict = classify(rec)
    assert verdict.outcome is want_outcome
    if want_reason:
        assert verdict.reason == want_reason
    assert verdict.sizing is want_sizing


def test_storage_403_fatal_regardless_of_started_at() -> None:
    """The reverted bug keyed transience on ``started_at``. The two real 403
    records differ ONLY in ``started_at`` (one None, one set) yet are the same
    permanent storage-limit fault. Both MUST classify FATAL on the message."""
    recs = _records()
    none_rec = recs["storage_403_started_none"]
    set_rec = recs["storage_403_started_set"]
    assert none_rec["started_at"] is None
    assert set_rec["started_at"] is not None
    assert classify(none_rec).outcome is JobOutcome.FATAL
    assert classify(set_rec).outcome is JobOutcome.FATAL
    assert classify(none_rec).reason == classify(set_rec).reason == "storage_limit_403"


def test_mount_401_is_a_transient_sidecar_flake_not_fatal() -> None:
    """The real `mount_401_auth` record is a CSI mount-sidecar 401. `preflight.check_api` already proved the
    token is valid, and on the 2026-11-02 retrain 63 jobs mounted the SAME private dataset fine, so an in-job
    mount 401 is a transient flake, never a permanent auth denial. It must RETRY (bounded by `attempts`), not
    FATAL-stop the whole campaign. A storage-limit fault still FATALs -- it carries the same sidecar text but
    `storage limit`/`403`, matched first."""
    rec = _records()["mount_401_auth"]
    assert "401" in rec["message"] and "sidecar" in rec["message"].lower()
    assert classify(rec).outcome is JobOutcome.RETRY
    assert classify(_records()["storage_403_started_set"]).outcome is JobOutcome.FATAL  # storage stays permanent


def test_bare_volume_mount_failure_retries_not_fatal() -> None:
    """A 'volume mount failed' with NO recognized sub-signature -- not a 401, not a storage 403, not
    'init container exhausted retries' -- is a generic TRANSIENT CSI mount fault: RETRY (bounded by
    `attempts`), never a FATAL stop of the whole campaign on one transient mount. Every mount fault captured
    on 2026-11-02 carried a 401/403/init signature caught by an EARLIER branch, so no real bare record exists;
    this one message is synthetic by necessity to pin the catch-all (like the DELETED/unknown records below)."""
    rec = {
        "stage": "ERROR",
        "message": 'Volume mount failed: MountVolume.SetUp failed for volume "hf-vol-0": sidecar not ready',
    }
    verdict = classify(rec)
    assert verdict.outcome is JobOutcome.RETRY and verdict.reason == "mount_failed"


def test_fatal_mount_signatures_precede_the_generic_mount_retry() -> None:
    """GF-03/branch-order: the real storage-403 AND init-retries records BOTH contain 'Volume mount failed'
    (the mount failed BECAUSE of the permanent fault), so the generic `volume mount failed` -> RETRY branch
    must sit BELOW their specific FATAL branches. A reorder would flip a real permanent storage limit /
    exhausted init to RETRY -- retried forever instead of stopping. This pins the order with real data."""
    recs = _records()
    for key, reason in (
        ("storage_403_started_set", "storage_limit_403"),
        ("mount_init_exhausted", "mount_init_retries"),
    ):
        rec = recs[key]
        assert "volume mount failed" in rec["message"].lower()  # the co-occurrence that makes order matter
        verdict = classify(rec)
        assert verdict.outcome is JobOutcome.FATAL and verdict.reason == reason, key


def test_deleted_is_terminal_not_in_flight() -> None:
    """DELETED is terminal (`launch.TERMINAL` + hf_hub both include it). It must NOT read as IN_FLIGHT,
    or the run loop never stalls. No real DELETED record was captured, so this one record is synthetic by
    necessity; the status value DELETED is from the real `JobStage` enum."""
    verdict = classify({"stage": "DELETED", "message": None})
    assert verdict.outcome is JobOutcome.CANCELED  # terminal, not an error, not in flight


def test_unknown_error_message_is_fatal_unclassified() -> None:
    """Residual safety net. An ERROR whose message matches no known signature
    STOPS the run (FATAL, reason ``unclassified``) rather than being silently
    retried or dropped. No real sample can exist for an unseen message, so this
    one record is synthetic by necessity; every other test uses real data."""
    rec = {"stage": "ERROR", "message": "some brand new failure never seen before"}
    verdict = classify(rec)
    assert verdict.outcome is JobOutcome.FATAL
    assert verdict.reason == "unclassified"
