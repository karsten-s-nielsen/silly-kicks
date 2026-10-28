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


def _ref_run(argv, **kw):
    out = {
        ("git", "rev-parse", "HEAD"): SHA + "\n",
        ("git", "branch", "-r", "--contains", SHA): "  origin/feat/x\n",
        ("git", "remote", "get-url", "origin"): "https://github.com/karsten-s-nielsen/silly-kicks.git\n",
    }
    return type("R", (), {"returncode": 0, "stdout": out.get(tuple(argv), ""), "stderr": ""})()


def test_ref_check_requires_head_pushed_and_green():
    def fetch_ci(repo, commit):
        assert repo == "karsten-s-nielsen/silly-kicks" and commit == SHA
        return [{"status": "completed", "conclusion": "success", "name": "CI"}]

    assert P.check_ref(SHA, run=_ref_run, fetch_ci=fetch_ci).ok


def test_ref_check_needs_no_gh_cli_or_token():
    """The controller verifies CI-green on the PUBLIC repo without the `gh` CLI or any token (owner: a
    controller cannot be brittle to missing GH access). check_ref reads CI status through an injectable
    anonymous GitHub read, never shells `gh`, and a read failure degrades to a clear not-ok rather than
    crashing preflight."""
    calls = []

    def run(argv, **kw):
        calls.append(tuple(argv))
        return _ref_run(argv, **kw)

    green = [{"status": "completed", "conclusion": "success", "name": "CI"}]
    ok = P.check_ref(SHA, run=run, fetch_ci=lambda repo, commit: green)
    assert ok.ok and "CI green" in ok.detail
    assert not any(a and a[0] == "gh" for a in calls)  # never invoked the gh CLI

    def boom(repo, commit):
        raise OSError("no network")

    degraded = P.check_ref(SHA, run=run, fetch_ci=boom)
    assert not degraded.ok and "CI status" in degraded.detail  # graceful: a read failure is not a crash


# ---- HFU T9: incremental preflight -- only check_plans' ~20-30 min toy train is cached (spec F) -----------


class _DictStore:
    """A dict-backed preflight store: read_bytes/upload_bytes over one in-memory namespace."""

    def __init__(self):
        self.data: dict[str, bytes] = {}

    def read_bytes(self, rel):
        return self.data.get(rel)

    def upload_bytes(self, rel, data):
        self.data[rel] = data


def test_unchanged_tree_reuses_check_plans(monkeypatch):
    """HFU T9 S1 (B-10): a second preflight with no changed input skips the check_plans toy train and reuses
    its stored pass -- the ~20-30 min train runs ONCE across two runs."""
    store = _DictStore()
    calls = []
    monkeypatch.setattr(P, "check_plans", lambda insts, *, run=None: calls.append(1) or P.Check("plan", True, "toy ok"))
    r1 = P.cached_check_plans([], store=store, campaign_sha256="abc")
    r2 = P.cached_check_plans([], store=store, campaign_sha256="abc")
    assert r1.ok and r2.ok and r2.detail == "toy ok"  # the reused Check is byte-identical to the stored pass
    assert len(calls) == 1  # the toy train ran once; the second run reused the cache


def test_library_change_invalidates_check_plans(monkeypatch):
    """HFU T9 S2: a silly_kicks/**-only change changes the check_plans key, so the toy train re-runs."""
    monkeypatch.setattr(P, "_tree_digest", lambda globs, *, exclude=(): "treeA")
    k1 = P.check_plans_key("abc", image="img")
    monkeypatch.setattr(P, "_tree_digest", lambda globs, *, exclude=(): "treeB")  # a library edit moved the tree digest
    k2 = P.check_plans_key("abc", image="img")
    assert k1 != k2  # a library change invalidates the key

    store = _DictStore()
    calls = []
    monkeypatch.setattr(P, "check_plans", lambda insts, *, run=None: calls.append(1) or P.Check("plan", True, "ok"))
    monkeypatch.setattr(P, "check_plans_key", lambda sha, instances=(), *, image=None: "keyA")
    P.cached_check_plans([], store=store, campaign_sha256="abc")
    monkeypatch.setattr(P, "check_plans_key", lambda sha, instances=(), *, image=None: "keyB")  # the input changed
    P.cached_check_plans([], store=store, campaign_sha256="abc")
    assert len(calls) == 2  # a changed key re-runs the toy train


def test_changed_lock_or_image_reruns_check_plans(monkeypatch):
    """HFU T9 S3 (B-10, B-18): a changed uv.lock or image digest changes the check_plans key; only check_plans
    is cached, so the cheap checks always re-observe (the cache holds the plan key alone)."""
    monkeypatch.setattr(P, "_tree_digest", lambda globs, *, exclude=(): "tree")
    monkeypatch.setattr(P, "_file_digest", lambda p: f"d:{p}")  # a stable per-path digest
    base = P.check_plans_key("abc", image="img1")
    assert P.check_plans_key("abc", image="img2") != base  # the pinned image digest is in the key
    monkeypatch.setattr(P, "_file_digest", lambda p: f"CHANGED:{p}")  # uv.lock (and the other lock files) moved
    assert P.check_plans_key("abc", image="img1") != base  # a changed lock re-runs the train

    store = _DictStore()
    monkeypatch.setattr(P, "check_plans", lambda insts, *, run=None: P.Check("plan", True, "ok"))
    P.cached_check_plans([], store=store, campaign_sha256="abc")
    keys = list(json.loads(store.data[P.PREFLIGHT_CACHE_REL]))
    assert keys and all(k.startswith("plan:") for k in keys)  # nothing but check_plans is cached


def test_controller_only_files_are_unreachable_by_the_toy_train():
    """The gate for `_CONTROLLER_ONLY_FILES`: the static import closure (over-approximating what a `--plan`/
    `--run-all` loads, with ancestor packages and bare sibling imports followed) must NOT reach any excluded file,
    so excluding them from the toy-train key can never drop code the train runs (reviews R5-PF-01 / CK-B-01/02).
    The closure reaches the shared machinery, the library families and the job-side `hf_jobs` modules."""
    files, unresolved = P._repo_import_closure(P._train_entry_modules())
    assert not unresolved, f"unresolved repo imports (resolver gap): {unresolved}"
    for excluded in P._CONTROLLER_ONLY_FILES:
        assert excluded not in files, f"{excluded} is excluded from the key but the toy train can load it"
    # the ancestor-seeding works: the library families reached only via deep submodule imports ARE in the closure
    for reached in (
        "scripts/_stages.py",
        "scripts/hf_jobs/executor.py",
        "scripts/hf_jobs/store.py",
        "silly_kicks/vaep/__init__.py",
        "silly_kicks/xtgk/__init__.py",
    ):
        assert reached in files, reached
    # review CK-B-07: bare `from _x import` (a sibling on sys.path) is resolved, so the gate does not under-report
    for sibling in ("scripts/_crossed_icc.py", "scripts/_loader_databricks.py"):
        assert sibling in files, sibling


class _Inst:
    def __init__(self, module: str, args: tuple[str, ...] = ()) -> None:
        self.module = module
        self.args = args


def test_check_plans_key_covers_the_checked_instance_argv():
    """Review CK-B-06: `check_plans` runs `--plan` with each instance's (module, args) built by
    `campaign.instances()` (in the excluded `campaign.py`). The key must cover those, so an argv change (a new
    `--arms-json`, a dropped flag) re-runs the train instead of reusing a stale PASS and sending an unplanned argv
    to HF."""
    base = P.check_plans_key("sha", [_Inst("scripts.train_x", ("--arms-json", "a"))], image="img")
    same = P.check_plans_key("sha", [_Inst("scripts.train_x", ("--arms-json", "a"))], image="img")
    assert base == same
    assert P.check_plans_key("sha", [_Inst("scripts.train_x", ("--arms-json", "b"))], image="img") != base  # args
    assert P.check_plans_key("sha", [_Inst("scripts.train_y", ("--arms-json", "a"))], image="img") != base  # module
    assert P.check_plans_key("sha", [], image="img") != base  # the instance set changed


def test_cached_check_plans_reruns_when_an_instance_arg_changes(monkeypatch):
    """Review CK-B-09: the (module, args) must reach the key THROUGH `cached_check_plans`, not only the direct
    `check_plans_key` call -- a dropped `instances` wiring (so an argv change reuses a stale PASS) must fail a test."""
    store = _DictStore()
    runs: list = []
    monkeypatch.setattr(P, "check_plans", lambda insts, *, run=None: runs.append(1) or P.Check("plan", True, "ok"))
    a = [_Inst("scripts.train_x", ("--arms-json", "a"))]
    P.cached_check_plans(a, store=store, campaign_sha256="sha")
    P.cached_check_plans(list(a), store=store, campaign_sha256="sha")  # same argv -> cached, no re-run
    assert len(runs) == 1
    P.cached_check_plans([_Inst("scripts.train_x", ("--arms-json", "b"))], store=store, campaign_sha256="sha")
    assert len(runs) == 2  # a changed instance argv re-runs the toy train (not a stale cached PASS)


def test_every_registry_job_module_is_a_train_entry():
    """Review R5-PF-02: the entry set is derived from the authoritative registry, not only filename globs, so an
    off-convention trainer is never silently dropped from the closure gate."""
    from scripts.hf_jobs.registry import DRIVERS, REGISTRY

    entries = set(P._train_entry_modules())
    for spec in (*REGISTRY.values(), *DRIVERS.values()):
        assert spec.module in entries, spec.module


def test_check_plans_key_excludes_the_controller_but_hashes_the_library(tmp_path, monkeypatch):
    """`check_plans_key` hashes all of `scripts/`+`silly_kicks/` except `_CONTROLLER_ONLY_FILES`: a controller-only
    edit leaves the key unchanged (no retrain), a library or trainer edit moves it (retrain). A bare-import helper
    and a deep library submodule both stay hashed -- the under-hash holes reviews R5-PF-01 / CK-B-01/02 found."""
    (tmp_path / "scripts" / "hf_jobs").mkdir(parents=True)
    (tmp_path / "silly_kicks" / "vaep").mkdir(parents=True)
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("proj", encoding="utf-8")
    ctrl = tmp_path / "scripts" / "hf_jobs" / "campaign.py"
    ctrl.write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "_crossed_icc.py").write_text("B = 1\n", encoding="utf-8")  # a bare-import helper
    lib = tmp_path / "silly_kicks" / "vaep" / "base.py"
    lib.write_text("C = 1\n", encoding="utf-8")  # a deep library submodule
    monkeypatch.setattr(P, "_ROOT", tmp_path)
    monkeypatch.setattr(P, "_CONTROLLER_ONLY_FILES", ("scripts/hf_jobs/campaign.py",))
    monkeypatch.setattr(P, "check_plans_key", P.check_plans_key)  # use the real one
    from scripts.hf_jobs import registry

    monkeypatch.setattr(registry, "CANARY", {})
    base = P.check_plans_key("sha", image="img")
    ctrl.write_text("A = 2\n", encoding="utf-8")  # controller-only edit
    assert P.check_plans_key("sha", image="img") == base  # excluded -> key unchanged
    lib.write_text("C = 2\n", encoding="utf-8")  # deep library edit (the 32-file hole)
    moved = P.check_plans_key("sha", image="img")
    assert moved != base  # hashed -> key moves
    (tmp_path / "scripts" / "_crossed_icc.py").write_text("B = 2\n", encoding="utf-8")  # bare-import helper edit
    assert P.check_plans_key("sha", image="img") != moved  # hashed -> key moves


def test_check_plans_key_is_cwd_independent(monkeypatch, tmp_path):
    """Reviewer B D1-IMPL-14: the key anchors its paths to the repo root, so running the controller from a
    different working directory does NOT collapse the tree/lock digests to a constant and reuse a stale PASS."""
    from_root = P.check_plans_key("abc", image="img")
    monkeypatch.chdir(tmp_path)  # run from an unrelated cwd
    assert P.check_plans_key("abc", image="img") == from_root  # same code -> same key, wherever the process ran
    # and the tree digest really hashed the repo (not an empty cwd glob): a real glob differs from an empty one
    assert P._tree_digest(["scripts/**/*.py"]) != P._tree_digest(["nonexistent/**/*.py"])


def test_force_pushed_ref_rechecks(monkeypatch):
    """HFU T9 S4: check_ref is NEVER cached -- a force-push or a not-yet-green CI run always re-checks. The
    cache wraps check_plans only, so a ref result never lands in it and check_ref re-observes every run."""
    store = _DictStore()
    monkeypatch.setattr(P, "check_plans", lambda insts, *, run=None: P.Check("plan", True, "ok"))
    P.cached_check_plans([], store=store, campaign_sha256="abc")
    cached = json.loads(store.data[P.PREFLIGHT_CACHE_REL])
    assert not any(k.startswith("ref") or k == "ref" for k in cached)  # check_ref is excluded from the cache

    runs = {"n": 0}

    def fake_run(argv, **kw):
        import types

        if argv[:2] == ["git", "rev-parse"]:
            runs["n"] += 1
            return types.SimpleNamespace(returncode=0, stdout="deadbeef\n", stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")  # not on a remote branch -> not-ok

    c1 = P.check_ref("deadbeef", run=fake_run)
    c2 = P.check_ref("deadbeef", run=fake_run)
    assert not c1.ok and not c2.ok and runs["n"] == 2  # re-observed both times, never served from a cache
