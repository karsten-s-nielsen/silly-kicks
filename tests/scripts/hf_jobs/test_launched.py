"""The record of the jobs that one driver run launched (HFU launch top-up spec §3.3).

A job that this run launched holds its units until a listing has shown it ENDED. A job that the listing of a step
does not show is held as in flight (a lag of the listing must not free its units), for at most `UNLISTED_STEPS`
listings in a row; then it is dropped and its units are free again.
"""

from __future__ import annotations

from scripts.hf_jobs.launched import UNLISTED_STEPS, LaunchedJobs

_LABELS = {"run_key": "rk", "variant": "default", "stage": "extract", "shard": "0of2", "tag": "0123456789ab",
           "job": "job"}  # fmt: skip


def _listed(job_id: str, stage: str) -> dict:
    return {"id": job_id, "labels": dict(_LABELS), "stage": stage, "message": None}


def _record() -> LaunchedJobs:
    rec = LaunchedJobs()
    rec.add("j1", _LABELS, ["a", "b"], "2026-11-12T10:00:00+00:00")
    return rec


def test_a_launched_job_that_the_listing_does_not_show_is_held_in_flight():
    rec = _record()
    held, dropped = rec.reconcile([])
    assert dropped == []
    assert held == [
        {
            "id": "j1",
            "labels": _LABELS,
            "stage": "SCHEDULING",
            "message": None,
            "created_at": "2026-11-12T10:00:00+00:00",
            "started_at": None,
            "finished_at": None,
            "wall_s": None,
        }
    ]
    assert rec.units_of("j1") == ("a", "b") and rec.knows("j1")


def test_a_job_listed_in_flight_and_then_missing_is_held_again():
    rec = _record()
    assert rec.reconcile([_listed("j1", "RUNNING")]) == ([], [])
    held, _ = rec.reconcile([])
    assert [h["id"] for h in held] == ["j1"]


def test_a_job_that_a_listing_showed_ended_is_never_held_again():
    rec = _record()
    assert rec.reconcile([_listed("j1", "COMPLETED")]) == ([], [])
    for _ in range(UNLISTED_STEPS + 1):
        assert rec.reconcile([]) == ([], [])


def test_a_job_missing_from_five_listings_is_dropped_and_its_units_are_free():
    rec = _record()
    assert UNLISTED_STEPS == 5
    for _ in range(UNLISTED_STEPS - 1):
        held, dropped = rec.reconcile([])
        assert [h["id"] for h in held] == ["j1"] and dropped == []
    held, dropped = rec.reconcile([])
    assert held == [] and dropped == [("j1", "job.default.extract.0of2-0123456789ab")]
    assert rec.units_of("j1") is None and not rec.knows("j1")
    assert rec.reconcile([]) == ([], [])


def test_the_count_restarts_when_a_listing_shows_the_job():
    rec = _record()
    for _ in range(UNLISTED_STEPS - 1):
        rec.reconcile([])
    rec.reconcile([_listed("j1", "RUNNING")])
    for _ in range(UNLISTED_STEPS - 1):
        held, dropped = rec.reconcile([])
        assert [h["id"] for h in held] == ["j1"] and dropped == []
    _, dropped = rec.reconcile([])
    assert [d[0] for d in dropped] == ["j1"]


def test_the_answer_of_the_store_is_remembered():
    rec = LaunchedJobs()
    assert not rec.knows("x") and rec.units_of("x") is None
    rec.remember("x", None)
    assert rec.knows("x") and rec.units_of("x") is None
    rec.remember("y", ["b", "a"])
    assert rec.units_of("y") == ("b", "a")
    assert rec.reconcile([]) == ([], [])  # a remembered job is not a launched job: never held
