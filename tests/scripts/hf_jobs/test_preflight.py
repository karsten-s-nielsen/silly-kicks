"""Preflight (spec 5.4): six no-spend checks; launch refuses without a passing report for this commit."""

from __future__ import annotations

import json

import pytest

from scripts.hf_jobs import preflight as P

SHA = "a" * 40


def _ok(name):
    return P.Check(name, True, "ok")


def test_report_ok_only_when_every_check_passes():
    r = P.assemble_report("cycle1", SHA, "sha", [_ok("ref"), _ok("lock"), P.Check("api", False, "no labels")], flags=[])
    assert r["ok"] is False and r["commit"] == SHA


def test_estimate_flags_do_not_fail_the_report():
    r = P.assemble_report("cycle1", SHA, "sha", [_ok("ref")], flags=["ghost_gk/fit predicted 74 min > 60"])
    assert r["ok"] is True and r["flags"]


def test_require_preflight_refuses_another_commit_or_campaign_hash():
    class _Store:
        def __init__(self, data):
            self.data = data

        def read_bytes(self, rel):
            return self.data

    rep = json.dumps({"ok": True, "commit": SHA, "campaign_sha256": "h1"}).encode()
    assert P.require_preflight("cycle1", SHA, "h1", store=_Store(rep))["ok"]
    with pytest.raises(SystemExit, match="commit"):
        P.require_preflight("cycle1", "b" * 40, "h1", store=_Store(rep))
    with pytest.raises(SystemExit, match="campaign file changed"):
        P.require_preflight("cycle1", SHA, "h2", store=_Store(rep))
    with pytest.raises(SystemExit, match="no preflight"):
        P.require_preflight("cycle1", SHA, "h1", store=_Store(None))


def test_api_contract_check_reads_the_signature_and_the_flavor_catalog():
    class _Hw:  # the HF catalog prices per `unit_label` (huggingface_hub JobHardwareInfo), not per hour
        name, cpu, ram, unit_cost_usd, unit_label = "cpu-upgrade", "8 vCPU", "32 GB", 0.0005, "minute"

    class _Api:
        def run_job(self, *, image, command, env, secrets, flavor, timeout, attempts, labels, volumes, name):
            raise AssertionError

        def list_jobs(self, *, labels=None):
            return []

        def list_jobs_hardware(self):
            return [_Hw()]

    c = P.check_api(_Api())
    assert c.ok, c.detail
    assert "$0.0005/minute" in c.detail and "/h" not in c.detail


def test_api_contract_check_requires_the_job_name_parameter():
    """The launcher names every job (HF derives a >100-char default name from a long command otherwise)."""

    class _Api:
        def run_job(self, *, image, command, env, secrets, flavor, timeout, attempts, labels, volumes):
            raise AssertionError

        def list_jobs(self, *, labels=None):
            return []

        def list_jobs_hardware(self):
            return []

    c = P.check_api(_Api())
    assert not c.ok and "name" in c.detail


def test_api_contract_check_fails_when_volumes_disappear():
    class _Api:
        def run_job(self, *, image, command, env, secrets, flavor, timeout, attempts, labels):
            raise AssertionError

        def list_jobs(self, *, labels=None):
            return []

        def list_jobs_hardware(self):
            return []

    assert not P.check_api(_Api()).ok


def test_ref_check_requires_head_pushed_and_green():
    def run(argv, **kw):
        out = {
            ("git", "rev-parse", "HEAD"): SHA + "\n",
            ("git", "branch", "-r", "--contains", SHA): "  origin/feat/x\n",
            ("gh",): json.dumps([{"status": "completed", "conclusion": "success", "workflowName": "CI"}]),
        }
        key = tuple(argv) if argv[0] == "git" else ("gh",)
        return type("R", (), {"returncode": 0, "stdout": out.get(key, ""), "stderr": ""})()

    assert P.check_ref(SHA, run=run).ok


def test_preflight_live_vs_lock(monkeypatch):
    # Regression guard (plan Task 2, spec section 7; mostly unchanged code, B-10): check_datasets reads the LIVE
    # files AT the lock's pinned revision and compares them with the lock's units. A matching revision passes;
    # a unit shard missing at that revision fails and is named.
    from scripts.hf_jobs import datasets as D

    gen = "tc3-frames-4-bbb"
    rec = {
        "revision": "r" * 40,
        "generation": gen,
        "home_teams": True,
        "units": [{"stem": "idsse__DFL-MAT-J03WMX"}, {"stem": "skillcorner__1886347"}],
    }
    monkeypatch.setattr(D, "locked", lambda name: rec)
    live = {
        f"shards/{gen}/idsse__DFL-MAT-J03WMX.parquet",
        f"shards/{gen}/skillcorner__1886347.parquet",
        "home_teams.json",
    }

    class _Api:
        def __init__(self, files):
            self.files, self.revisions = files, []

        def list_repo_files(self, repo, repo_type=None, revision=None):
            self.revisions.append(revision)
            return sorted(self.files)

    ok = _Api(live)
    assert P.check_datasets(["xshot17"], api=ok).ok
    assert ok.revisions == [rec["revision"]]  # read at the PINNED revision, never the moving head
    stale = P.check_datasets(["xshot17"], api=_Api(live - {f"shards/{gen}/skillcorner__1886347.parquet"}))
    assert not stale.ok and "skillcorner__1886347.parquet missing at the pinned revision" in stale.detail
