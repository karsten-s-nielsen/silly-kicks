"""ADR-128 environment provenance (INV-6) and the stage-token code identity (decision P9)."""

from __future__ import annotations

import hashlib
import pathlib

import scripts._provenance as prov_mod

_KEYS = {"image", "python", "uv_lock_sha256", "packages", "platform", "machine", "commit", "tree_state", "run_key"}


def test_env_provenance_has_exactly_the_declared_keys(monkeypatch):
    monkeypatch.setenv(prov_mod.IMAGE_ENV, "python:3.12.14-bookworm@sha256:abc")
    monkeypatch.setenv(prov_mod.RUN_KEY_ENV, "cycle1-abc1234-ghost_gk-owner")
    out = prov_mod.env_provenance()
    assert set(out) == _KEYS
    assert out["image"] == "python:3.12.14-bookworm@sha256:abc"
    assert out["run_key"] == "cycle1-abc1234-ghost_gk-owner"
    assert out["packages"]["numpy"]  # installed in every test environment
    assert set(out["packages"]) == set(prov_mod._ENV_PACKAGES)


def test_env_provenance_hashes_the_committed_lock():
    root = pathlib.Path(prov_mod.__file__).resolve().parents[1]
    want = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    assert prov_mod.env_provenance(root=root)["uv_lock_sha256"] == want


def test_env_provenance_unset_image_is_none(monkeypatch):
    monkeypatch.delenv(prov_mod.IMAGE_ENV, raising=False)
    assert prov_mod.env_provenance()["image"] is None


def test_code_identity_clean_is_the_commit():
    assert prov_mod.code_identity({"commit": "a" * 40, "tree_state": "clean"}) == "a" * 40


def test_code_identity_tracked_dirt_is_the_deterministic_tree_hash(monkeypatch):
    monkeypatch.setattr(prov_mod, "_has_tracked_changes", lambda: True)
    monkeypatch.setattr(prov_mod, "git_tree_hash", lambda: "deadbeef")
    p = {"commit": "a" * 40, "tree_state": "dirty"}
    assert prov_mod.code_identity(p) == "dirty-deadbeef"
    assert prov_mod.code_identity(p) == prov_mod.code_identity(p)  # two processes agree


def test_code_identity_untracked_only_dirt_is_the_commit(monkeypatch):
    # A tree dirty ONLY by untracked files (e.g. a controller's own `_retrain.log`) runs HEAD's committed
    # code -- an HF job clones HEAD and never sees those files -- so the GEN identity is the commit, stable
    # across untracked litter. The 2026-11-04 campaign wedge: the controller's logs changed the gen between
    # launch and status, so results written under one gen read EMPTY under another -> done=0 forever.
    monkeypatch.setattr(prov_mod, "_has_tracked_changes", lambda: False)
    assert prov_mod.code_identity({"commit": "a" * 40, "tree_state": "dirty"}) == "a" * 40


def test_code_identity_over_a_real_git_tree_untracked_does_not_move_the_gen(tmp_path, monkeypatch):
    """Drive code_identity(git_provenance()) over an ACTUAL git tree with NO monkeypatch of _has_tracked_changes,
    so the real git predicate that IS the 2026-11-04 fix is guarded (GF-01 / GSF-IMPL-01). A controller's own
    `_retrain.log` is exactly the untracked case that must NOT move the gen; a TRACKED edit must. Mutating the
    predicate body (always-True / always-False) fails this test -- the monkeypatched tests above cannot."""
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603, S607

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = rev.stdout.strip()
    assert prov_mod.code_identity(prov_mod.git_provenance()) == commit  # clean -> commit
    (tmp_path / "_retrain.log").write_text("noise\n")  # an untracked controller log
    assert prov_mod.code_identity(prov_mod.git_provenance()) == commit  # untracked-only -> STILL the commit
    (tmp_path / "code.py").write_text("x = 2\n")  # a TRACKED edit
    ident = prov_mod.code_identity(prov_mod.git_provenance())
    assert ident != commit and ident.startswith("dirty-")  # tracked edit -> content hash, invalidates the store


def test_code_identity_unknown_never_repeats():
    p = {"commit": "unknown", "tree_state": "unknown"}
    a, b = prov_mod.code_identity(p), prov_mod.code_identity(p)
    assert a.startswith("unknown-") and b.startswith("unknown-") and a != b
