"""Preflight (spec 5.4, HFU spec 13.2): six no-spend checks; launch refuses without a passing report for this commit."""

from __future__ import annotations

import json
import re
import threading
import time
import types

import pytest

from scripts.hf_jobs import preflight as P

SHA = "a" * 40


def _ok(name):
    return P.Check(name, True, "ok")


def test_report_ok_only_when_every_check_passes():
    r = P.assemble_report("cycle1", SHA, "sha", [_ok("ref"), _ok("lock"), P.Check("api", False, "no labels")])
    assert r["ok"] is False and r["commit"] == SHA
    assert "flags" not in r  # the planner estimate that filled it is gone


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


# ---- check 5: every instance's --plan, in parallel, one progress line each --------------------------------


def _inst(job: str, variant: str = "default", module: str = "scripts.train_x"):
    return types.SimpleNamespace(job=job, variant=variant, module=module, args=("--dataset", "d"))


def _plan_run(*, delay: float = 0.0, fail: frozenset[str] = frozenset(), seen: list | None = None):
    lock = threading.Lock()

    def run(argv, **kw):
        with lock:
            if seen is not None:
                seen.append(list(argv))
        time.sleep(delay)
        job = argv[argv.index("--store") + 1].rsplit("/", 2)[-2]  # the per-instance temp dir names the job
        if job in fail:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Traceback ...\nValueError: no refs\n")
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"stages": [{"name": "s"}]}), stderr="")

    return run


def test_check_plans_runs_only_plan_never_a_toy_training_run():
    # Owner decision 2026-11-05: the toy --run-all cost ~15 min per preflight and duplicates CI's parity tests.
    seen: list = []
    c = P.check_plans([_inst("a"), _inst("b"), _inst("c")], run=_plan_run(seen=seen), say=lambda line: None)
    assert c.ok, c.detail
    assert len(seen) == 3 and all("--plan" in a and "--run-all" not in a for a in seen)


def test_check_plans_runs_the_instances_in_parallel():
    insts = [_inst(f"j{i}") for i in range(6)]
    t0 = time.monotonic()
    assert P.check_plans(insts, run=_plan_run(delay=0.5), workers=6, say=lambda line: None).ok
    assert time.monotonic() - t0 < 2.0  # serial would be >= 3 s


def test_check_plans_prints_one_timed_line_per_instance():
    lines: list[str] = []
    P.check_plans([_inst("a"), _inst("b", "v2")], run=_plan_run(), say=lines.append)
    assert sorted(re.sub(r"\d+\.\ds", "Ns", ln) for ln in lines) == [
        "[preflight] plan a/default ok Ns",
        "[preflight] plan b/v2 ok Ns",
    ]


def test_a_failed_plan_fails_the_check_and_names_its_error():
    lines: list[str] = []
    c = P.check_plans([_inst("a"), _inst("bad")], run=_plan_run(fail=frozenset({"bad"})), say=lines.append)
    assert not c.ok and "bad/default" in c.detail and "ValueError: no refs" in c.detail
    assert any(ln.startswith("[preflight] plan bad/default FAILED") and "ValueError: no refs" in ln for ln in lines)


# ---- check_numba (HFU spec 13.2): numba is in the job's LOCKED install set for the job platform --------------------


def _export_run(text, rc=0, seen=None):
    def run(argv, **kwargs):
        if seen is not None:
            seen.append(list(argv))
        return types.SimpleNamespace(returncode=rc, stdout=text, stderr="")

    return run


def test_check_numba_exports_the_job_install_set():
    seen = []
    P.check_numba(run=_export_run("numba==0.65.1\n", seen=seen))
    assert seen == [["uv", "export", "--frozen", "--no-hashes", "--extra", "train", "--group", "hfjobs"]]


def test_check_numba_passes():
    c = P.check_numba(run=_export_run("numpy==2.4.4 ; python_full_version >= '3.11'\nnumba==0.65.1\n"))
    assert c.ok and "0.65.1" in c.detail


def test_check_numba_fails_without_numba():
    c = P.check_numba(run=_export_run("numpy==2.4.4\nscipy==1.16.0\n"))
    assert not c.ok and "numba" in c.detail


def test_check_numba_ignores_other_platform_line():
    c = P.check_numba(run=_export_run("numba==0.65.1 ; sys_platform == 'win32'\n"))
    assert not c.ok


def test_check_numba_fails_when_the_export_fails():
    assert not P.check_numba(run=_export_run("", rc=2)).ok
