"""Preflight (spec 5.4, HFU spec 13.2, HFU fit-units spec C3): seven no-spend checks.

Launch refuses without a passing report for this commit.
"""

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

    assert P.check_ref(SHA, run=run, identity=lambda: SHA).ok


# ---- check 1, the tree (spec §19.1 E1): the controller plans with the commit the jobs clone --------------


def _ref_run(calls: list, *, status: str = ""):
    """A `run` fake for check 1: HEAD is SHA, pushed, CI green; `git status` lists ``status``. Records argv."""

    def run(argv, **kw):
        calls.append(tuple(argv))
        out = {
            ("git", "rev-parse", "HEAD"): SHA + "\n",
            ("git", "branch", "-r", "--contains", SHA): "  origin/feat/x\n",
            ("git", "status", "--porcelain", "--untracked-files=no"): status,
            ("gh",): json.dumps([{"status": "completed", "conclusion": "success", "workflowName": "CI"}]),
        }
        key = tuple(argv) if argv[0] == "git" else ("gh",)
        return type("R", (), {"returncode": 0, "stdout": out.get(key, ""), "stderr": ""})()

    return run


def test_check_ref_passes_when_the_identity_is_the_commit():
    calls: list = []
    c = P.check_ref(SHA, run=_ref_run(calls), identity=lambda: SHA)
    assert c.ok and c.name == "ref"


def test_check_ref_fails_on_a_tracked_dirty_tree_and_names_the_paths():
    calls: list = []
    status = " M docs/a.md\n M docs/b.md\n"
    c = P.check_ref(SHA, run=_ref_run(calls, status=status), identity=lambda: "dirty-1a2b3c4d5e6f")
    assert not c.ok and c.name == "ref"
    assert "dirty-1a2b3c4d5e6f" in c.detail and f"the jobs run {SHA[:7]};" in c.detail
    assert c.detail.endswith("commit or stash them")
    assert "docs/a.md" in c.detail and "docs/b.md" in c.detail and "(2)" in c.detail
    assert not any(a[0] == "gh" for a in calls)  # it fails before the network calls
    assert ("git", "fetch", "-q", "origin") not in calls


def test_check_ref_names_at_most_five_paths():
    calls: list = []
    status = "".join(f" M f{i}.py\n" for i in range(7))
    c = P.check_ref(SHA, run=_ref_run(calls, status=status), identity=lambda: "dirty-0")
    assert not c.ok and "(7)" in c.detail
    assert "f4.py" in c.detail and "f5.py" not in c.detail


def test_check_ref_fails_on_unknown_provenance():
    calls: list = []
    c = P.check_ref(SHA, run=_ref_run(calls), identity=lambda: "unknown-" + "9" * 32)
    assert not c.ok and "unknown-" in c.detail and "stash" not in c.detail
    assert not any(a[0] == "gh" for a in calls)


def test_check_ref_on_a_clean_tree_at_another_head_names_the_head():
    calls: list = []
    other = "b" * 40

    def run(argv, **kw):
        calls.append(tuple(argv))
        stdout = other + "\n" if tuple(argv) == ("git", "rev-parse", "HEAD") else ""
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    c = P.check_ref(SHA, run=run, identity=lambda: other)
    assert not c.ok and f"HEAD {other[:12]} != campaign commit {SHA[:12]}" == c.detail


def test_check_ref_fails_on_any_other_identity_at_the_right_head():
    calls: list = []
    c = P.check_ref(SHA, run=_ref_run(calls), identity=lambda: "c" * 40)
    assert not c.ok and "c" * 18 in c.detail
    assert not any(a[0] == "gh" for a in calls)


def test_check_ref_without_identity_uses_the_driver_identity(monkeypatch):
    """Production calls ``check_ref(commit)`` (``cmd_preflight``): the DEFAULT must be the driver's identity."""
    from scripts import _provenance

    monkeypatch.setattr(_provenance, "git_provenance", lambda: {"commit": SHA, "tree_state": "dirty"})
    monkeypatch.setattr(_provenance, "code_identity", lambda prov: "dirty-0badc0de0bad")
    calls: list = []
    c = P.check_ref(SHA, run=_ref_run(calls, status=" M scripts/x.py\n"))
    assert not c.ok and "dirty-0badc0de0bad" in c.detail and "scripts/x.py" in c.detail
    assert not any(a[0] == "gh" for a in calls)


def test_check_ref_default_identity_is_the_driver_one(monkeypatch):
    from scripts import _provenance

    prov = {"commit": SHA, "tree_state": "dirty"}
    monkeypatch.setattr(_provenance, "git_provenance", lambda: prov)
    seen = []

    def fake_identity(p):
        seen.append(p)
        return "dirty-feed"

    monkeypatch.setattr(_provenance, "code_identity", fake_identity)
    assert P._controller_identity() == "dirty-feed"
    assert seen == [prov]


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


# ---- check 7: storage (HFU fit-units spec C3) --------------------------------------------------------

GB = 1_000_000_000


class _StorageApi:
    """Two buckets, two private datasets, one public dataset, one private model -- sizes in bytes."""

    def __init__(self, *, fail: str | None = None, missing: str | None = None) -> None:
        self.fail, self.missing = fail, missing
        self.buckets = {"org/scratch": 20 * GB, "org/canary": 1 * GB}
        self.datasets = {
            "org/private-a": (True, 30 * GB),
            "org/private-b": (True, 9 * GB),
            "org/public": (False, 500 * GB),
        }
        self.models = {"org/sandbox-m": (True, 0 * GB + 5_000_000)}

    def list_buckets(self, org):
        return [types.SimpleNamespace(id=k, size=v) for k, v in self.buckets.items()]

    def list_datasets(self, *, author):
        return [types.SimpleNamespace(id=k, private=p) for k, (p, _s) in self.datasets.items()]

    def list_models(self, *, author):
        return [types.SimpleNamespace(id=k, private=p) for k, (p, _s) in self.models.items()]

    def _info(self, table, repo_id):
        if self.fail == repo_id:
            raise RuntimeError("429 Too Many Requests")
        used = None if self.missing == repo_id else table[repo_id][1]
        return types.SimpleNamespace(id=repo_id, used_storage=used)

    def dataset_info(self, repo_id, *, expand):
        assert expand == ["usedStorage"]
        return self._info(self.datasets, repo_id)

    def model_info(self, repo_id, *, expand):
        assert expand == ["usedStorage"]
        return self._info(self.models, repo_id)


_USED = 20 * GB + 1 * GB + 30 * GB + 9 * GB + 5_000_000  # the public dataset is not counted


def test_storage_under_the_bound_passes():
    c = P.check_storage(_StorageApi(), org="org", limit_bytes=100 * GB, headroom_bytes=20 * GB)
    assert c.ok and c.name == "storage"
    assert f"{_USED / GB:.2f} GB used" in c.detail and "org/private-a (dataset) 30.00 GB" in c.detail


def test_storage_at_the_bound_passes():
    assert P.check_storage(_StorageApi(), org="org", limit_bytes=_USED + 20 * GB, headroom_bytes=20 * GB).ok


def test_storage_over_the_bound_fails_and_names_the_largest_repo():
    c = P.check_storage(_StorageApi(), org="org", limit_bytes=_USED + 20 * GB - 1, headroom_bytes=20 * GB)
    assert not c.ok
    assert c.detail.index("org/private-a") < c.detail.index("org/scratch") < c.detail.index("org/private-b")


def test_a_public_repo_is_not_counted():
    c = P.check_storage(_StorageApi(), org="org", limit_bytes=100 * GB, headroom_bytes=20 * GB)
    assert "org/public" not in c.detail


def test_an_api_error_fails_the_storage_check_with_its_reason():
    c = P.check_storage(_StorageApi(fail="org/private-b"), org="org", limit_bytes=100 * GB, headroom_bytes=20 * GB)
    assert not c.ok and "429 Too Many Requests" in c.detail


def test_a_missing_size_fails_the_storage_check():
    c = P.check_storage(_StorageApi(missing="org/private-a"), org="org", limit_bytes=100 * GB, headroom_bytes=20 * GB)
    assert not c.ok and "org/private-a" in c.detail and "no size" in c.detail


def test_storage_fails_when_a_repo_has_no_private_field():
    """A listing entry without ``private`` cannot be classed: the count is not a pass (fail closed)."""
    api = _StorageApi()
    api.list_datasets = lambda *, author: [types.SimpleNamespace(id="org/private-a")]
    c = P.check_storage(api, org="org", limit_bytes=100 * GB, headroom_bytes=20 * GB)
    assert not c.ok and "org/private-a" in c.detail and "private" in c.detail
