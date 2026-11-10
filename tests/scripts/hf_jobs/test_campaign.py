"""The campaign model and the operator commands ``preflight`` / ``status`` / ``cancel`` (ADR-128).

The driver loop is tested in ``test_hf_driver.py``. These tests run the command BODIES (review D1-IMPL-06):
``cmd_cancel`` and ``cmd_status`` on real ``huggingface_hub`` ``JobInfo`` objects, ``cmd_preflight`` over the
real cycle1 campaign with the six checks stubbed.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from scripts.hf_jobs import campaign as C
from scripts.hf_jobs import preflight
from scripts.hf_jobs import store as store_mod

_jobs_api = pytest.importorskip("huggingface_hub._jobs_api")

SHA = "abcdef1" + "0" * 33
CYCLE1 = pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml")


def _campaign() -> C.Campaign:
    return C.Campaign(name="cycle1", bucket="b/x", concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h")


def _jobinfo(**resp):
    resp.setdefault("owner", {"id": "u1", "name": "silly-kicks", "type": "user"})
    return _jobs_api.JobInfo(**resp)


class _ListApi:
    """A minimal HfApi double: ``list_jobs`` returns the given real ``JobInfo``s; ``cancel_job`` records ids."""

    def __init__(self, jobs=()) -> None:
        self._jobs = list(jobs)
        self.canceled: list[str] = []

    def list_jobs(self, labels=None):
        return self._jobs

    def cancel_job(self, *, job_id):
        self.canceled.append(job_id)


def test_run_key_is_derived():
    c = _campaign()
    assert C.run_key(c, SHA, "ghost_gk", "owner") == "cycle1-abcdef1-ghost_gk-owner"
    assert C.run_key(c, SHA, "ghost_gk", "owner", replica=2) == "cycle1-abcdef1-ghost_gk-owner-r2"


def test_load_campaign_reads_the_driver_settings():
    c = C.load_campaign(CYCLE1)
    assert (c.name, c.concurrency_cap, c.attempts) == ("cycle1", 25, 3)
    assert c.launch_batch == 16  # HFU spec 17.5
    assert len(c.sha256) == 64 and c.jobs


def test_cmd_cancel_cancels_only_non_terminal_jobs() -> None:
    running = _jobinfo(id="r1", status={"stage": "RUNNING"}, labels={"stage": "extract"})
    done = _jobinfo(id="c1", status={"stage": "COMPLETED"}, labels={"stage": "extract"})
    api = _ListApi([running, done])
    ids = C.cmd_cancel(_campaign(), SHA, api=api)
    assert ids == ["r1"] and api.canceled == ["r1"]  # the COMPLETED job is terminal, left alone


def test_cmd_status_reports_progress_running_and_errors_per_instance(monkeypatch) -> None:
    plan = {
        "stages": [
            {"name": "extract", "done": 1, "pending": 2, "failed": 0, "reduced": False, "token": "deadbeefcafe0001"}
        ]
    }
    monkeypatch.setattr(C, "read_plan", lambda inst, *, bucket: plan)
    camp = C.load_campaign(CYCLE1)
    first = C.instances(camp, SHA)[0]
    labels = {"run_key": first.run_key, "variant": first.variant, "stage": "extract"}
    live = [
        _jobinfo(id="run1", status={"stage": "RUNNING"}, labels=labels, createdAt="2026-11-02T03:00:00.000000Z"),
        _jobinfo(
            id="err1",
            status={"stage": "ERROR", "message": "Job failed with exit code: 1. Reason: Error"},
            labels={**labels, "stage": "study"},
            createdAt="2026-11-02T03:00:00.000000Z",
        ),
    ]
    rows = C.cmd_status(camp, SHA, api=_ListApi(live))
    assert len(rows) == len(C.instances(camp, SHA))
    row = next(r for r in rows if r["instance"] == first.id)
    assert row["stages"] == [{"name": "extract", "done": 1, "pending": 2, "failed": 0, "reduced": False}]
    assert row["running"] == ["extract"] and row["errors"] == ["study:err1"]


def test_cmd_status_reports_a_plan_failure_as_a_row(monkeypatch) -> None:
    def boom(inst, *, bucket):
        raise RuntimeError(f"{inst.id}: --plan failed (rc=1): boom")

    monkeypatch.setattr(C, "read_plan", boom)
    rows = C.cmd_status(C.load_campaign(CYCLE1), SHA, api=_ListApi([]))
    assert rows and all("boom" in r["error"] for r in rows)


def test_cmd_preflight_runs_the_six_checks_and_uploads_the_gate_report(monkeypatch, capsys) -> None:
    calls: list[str] = []
    for name in ("check_ref", "check_lock", "check_numba", "check_api", "check_datasets", "check_plans"):
        monkeypatch.setattr(
            preflight, name, lambda *a, _n=name, **k: (calls.append(_n), preflight.Check(_n, True, "ok"))[1]
        )
    uploaded: dict[str, bytes] = {}

    class _Store:
        def __init__(self, *a, **k) -> None: ...

        def upload_bytes(self, rel, data) -> None:
            uploaded[rel] = data

    monkeypatch.setattr(store_mod, "BucketStore", _Store)  # cmd_preflight imports it from the store module
    camp = C.load_campaign(CYCLE1)
    report = C.cmd_preflight(camp, SHA, api=_ListApi([]))
    assert calls == ["check_ref", "check_lock", "check_numba", "check_api", "check_datasets", "check_plans"]
    assert report["ok"] is True and report["commit"] == SHA and report["campaign_sha256"] == camp.sha256
    lines = capsys.readouterr().out.splitlines()  # one timed progress line per check, as it finishes
    assert [ln.split()[1] for ln in lines] == [
        "check_ref",
        "check_lock",
        "check_numba",
        "check_api",
        "check_datasets",
        "check_plans",
    ]
    assert all(re.fullmatch(r"\[preflight\] \S+ ok \d+\.\ds: ok", ln) for ln in lines), lines
    assert list(uploaded) == [preflight.REPORT_REL.format(campaign="cycle1", commit7=SHA[:7])]
