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
    monkeypatch.setattr(P, "_tree_digest", lambda globs: "treeA")
    k1 = P.check_plans_key("abc", image="img")
    monkeypatch.setattr(P, "_tree_digest", lambda globs: "treeB")  # a library edit moved the tree digest
    k2 = P.check_plans_key("abc", image="img")
    assert k1 != k2  # a library change invalidates the key

    store = _DictStore()
    calls = []
    monkeypatch.setattr(P, "check_plans", lambda insts, *, run=None: calls.append(1) or P.Check("plan", True, "ok"))
    monkeypatch.setattr(P, "check_plans_key", lambda sha, *, image=None: "keyA")
    P.cached_check_plans([], store=store, campaign_sha256="abc")
    monkeypatch.setattr(P, "check_plans_key", lambda sha, *, image=None: "keyB")  # the input changed
    P.cached_check_plans([], store=store, campaign_sha256="abc")
    assert len(calls) == 2  # a changed key re-runs the toy train


def test_changed_lock_or_image_reruns_check_plans(monkeypatch):
    """HFU T9 S3 (B-10, B-18): a changed uv.lock or image digest changes the check_plans key; only check_plans
    is cached, so the cheap checks always re-observe (the cache holds the plan key alone)."""
    monkeypatch.setattr(P, "_tree_digest", lambda globs: "tree")
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
