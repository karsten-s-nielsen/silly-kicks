"""The campaign model and the operator commands ``preflight`` / ``status`` / ``cancel`` (ADR-128).

The driver loop is tested in ``test_hf_driver.py``. These tests run the command BODIES (review D1-IMPL-06):
``cmd_cancel`` and ``cmd_status`` on real ``huggingface_hub`` ``JobInfo`` objects, ``cmd_preflight`` over the
real cycle1 campaign with the seven checks stubbed.
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
    assert (c.storage_limit_bytes, c.storage_headroom_bytes) == (100_000_000_000, 20_000_000_000)
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


def test_cmd_preflight_runs_the_seven_checks_and_uploads_the_gate_report(monkeypatch, capsys) -> None:
    calls: list[str] = []
    names = ("check_ref", "check_lock", "check_numba", "check_api", "check_datasets", "check_plans", "check_storage")
    for name in names:
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
    assert calls == list(names)
    assert report["ok"] is True and report["commit"] == SHA and report["campaign_sha256"] == camp.sha256
    lines = capsys.readouterr().out.splitlines()  # one timed progress line per check, as it finishes
    assert [ln.split()[1] for ln in lines] == list(names)
    assert all(re.fullmatch(r"\[preflight\] \S+ ok \d+\.\ds: ok", ln) for ln in lines), lines
    assert list(uploaded) == [preflight.REPORT_REL.format(campaign="cycle1", commit7=SHA[:7])]


def test_storage_values_default_when_the_campaign_file_has_none(tmp_path):
    text = CYCLE1.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith(("storage_limit_bytes", "storage_headroom_bytes"))]
    assert len(lines) == len(text.splitlines()) - 2  # cycle1.toml states both values
    p = tmp_path / "c.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    c = C.load_campaign(p)
    assert (c.storage_limit_bytes, c.storage_headroom_bytes) == (100_000_000_000, 20_000_000_000)


# ---- ADR-130: the campaign mode and the refit campaign file --------------------------------------------------

REFIT = pathlib.Path("scripts/hf_jobs/campaigns/tf46_refit.toml")


def _with_mode(tmp_path, mode: str | None) -> pathlib.Path:
    text = CYCLE1.read_text(encoding="utf-8")
    if mode is not None:
        text = text.replace("[campaign]\n", f'[campaign]\nmode = "{mode}"\n', 1)
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_the_campaign_mode_defaults_to_rematerialize_and_accepts_refit(tmp_path):
    assert C.load_campaign(_with_mode(tmp_path, None)).mode == "rematerialize"
    assert C.load_campaign(_with_mode(tmp_path, "refit")).mode == "refit"


def test_campaign_mode_accepts_retrain(tmp_path):
    assert C.load_campaign(_with_mode(tmp_path, "retrain")).mode == "retrain"


def _with_job_mode(tmp_path, mode: str) -> pathlib.Path:
    p = tmp_path / "j.toml"
    text = CYCLE1.read_text(encoding="utf-8").replace('job = "xsuccess"\n', f'job = "xsuccess"\nmode = "{mode}"\n', 1)
    p.write_text(text, encoding="utf-8")
    return p


def test_job_mode_defaults_to_the_campaign_mode(tmp_path):
    assert {i.mode for i in C.instances(C.load_campaign(_with_mode(tmp_path, "refit")), "0" * 40)} == {"refit"}


@pytest.mark.parametrize("mode", ["rematerialize", "refit", "retrain"])
def test_job_mode_accepts_the_three_values(tmp_path, mode):
    insts = C.instances(C.load_campaign(_with_job_mode(tmp_path, mode)), "0" * 40)
    assert {(i.job == "xsuccess", i.mode) for i in insts} == {(True, mode), (False, "rematerialize")}


def test_job_mode_refuses_another_value(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        C.load_campaign(_with_job_mode(tmp_path, "nonsense"))


def test_the_campaign_mode_refuses_any_other_value(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        C.load_campaign(_with_mode(tmp_path, "nonsense"))


def test_tf46_refit_holds_the_21_release_instances():
    # The 4.144.0 release refit (delta spec 4.1): every instance with its verify mode. A ghost-GK job without
    # `retrain` would pass on its absolute acceptance gates alone, so the mode is pinned per instance.
    c = C.load_campaign(REFIT)
    got = {(i.job, i.tier, i.variant): i.mode for i in C.instances(c, "0" * 40)}
    want = {
        (job, tier, variant): "refit"
        for job in ("xshot_occurrence", "xcross_attempt")
        for tier in ("public", "sc_extended")
        for variant in ("default", "position_only")
    }
    want[("xsuccess", "public", "default")] = "retrain"
    want[("pass_completion", "public", "default")] = "retrain"
    for v in ("default", "position_only", "sweeper", "sweeper_position_only", "full"):
        want[("ghost_gk", "owner", v)] = "retrain"
    for v in ("default", "position_only"):
        want[("ghost_outfield", "owner", v)] = "retrain"
    want[("receiver_model", "owner", "default")] = "retrain"
    for job in ("win_probability", "match_outcome_dependence", "build_territory_ranking_census"):
        want[(job, "public", "default")] = "rematerialize"
    assert len(want) == 21 and got == want
    assert c.mode == "refit" and c.bucket == "karstenskyt/hf-refit-scratch"
    assert c.storage_limit_bytes == 10**12 and c.jobs[0].job == "xsuccess"
    cycle1 = C.load_campaign(CYCLE1)  # the sizing tables of cycle1.toml, the campaign that runs these jobs today
    assert c.mount_cap_by_dataset == cycle1.mount_cap_by_dataset
    assert c.peak_rss_by_stage == cycle1.peak_rss_by_stage
    # action-time repair plan Task 14: the xShot study runs 2 units per job (the 2 h job limit); the rest as cycle1
    assert c.units_per_job_by_stage == {**cycle1.units_per_job_by_stage, "xshot_occurrence/study": 2}
    # the two wheel jobs force the public arm; the Hub jobs keep the rule's choice
    forced = {(j.job, j.tier): j.ship_variant for j in c.jobs if j.job in ("xshot_occurrence", "xcross_attempt")}
    assert forced == {(job, tier): ("public" if tier == "public" else None) for job, tier in forced}
    assert sorted(forced.values(), key=str) == [None, None, "public", "public"]  # 2 wheel + 2 Hub jobs
    assert {j.dataset for j in c.jobs if j.job in ("xshot_occurrence", "xcross_attempt")} == {"xshot187"}
