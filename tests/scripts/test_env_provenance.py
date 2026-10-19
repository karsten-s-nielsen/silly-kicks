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


def test_code_identity_dirty_is_the_deterministic_tree_hash(monkeypatch):
    monkeypatch.setattr(prov_mod, "git_tree_hash", lambda: "deadbeef")
    p = {"commit": "a" * 40, "tree_state": "dirty"}
    assert prov_mod.code_identity(p) == "dirty-deadbeef"
    assert prov_mod.code_identity(p) == prov_mod.code_identity(p)  # two processes agree


def test_code_identity_unknown_never_repeats():
    p = {"commit": "unknown", "tree_state": "unknown"}
    a, b = prov_mod.code_identity(p), prov_mod.code_identity(p)
    assert a.startswith("unknown-") and b.startswith("unknown-") and a != b
