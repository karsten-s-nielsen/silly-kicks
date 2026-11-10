"""Verify checks the recipe, not only the score (HFU spec 14.3, Change 7b; launch-3 findings F9, F13).

One corpus mechanism, one source: the lock's ``sb_open.identity.open`` digest, compared by the manifest
``digest`` rule; every certified artifact has a corpus-scope check (ADR-056: derive, assert exactly)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from scripts._corpus_identity import open_data_identity
from scripts._stages import _dataset_identity
from scripts._toml import load_toml
from scripts.hf_jobs import adopt as A
from scripts.hf_jobs import manifest as M
from scripts.hf_jobs.datasets import declared, expected_identity, locked

C = "c" * 40
MAN = M.load()
#: The four event-only artifacts trained on the pinned open-data corpus (spec 14.3).
EVENT_ONLY = ("win_probability", "match_outcome", "expected_passing", "xsuccess")


def _record(dataset: str) -> dict:
    """A job record whose dataset identity is the form a job writes (`_stages._dataset_identity`)."""
    return {"token_inputs": {"run": {"dataset": _dataset_identity({"dataset": dataset}, None)}}}


def _fresh_xsuccess(d: Path, identity: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"training_commit": C}), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps({"hpo_trials": 30, **identity}), encoding="utf-8")
    return d


def test_pinned_competitions_equal_committed():
    pinned = [tuple(p) for p in locked("sb_open")["competitions"]]
    assert pinned == sorted(set(pinned)) and len(pinned) == 80
    wp = json.loads(Path("silly_kicks/win_probability/weights/metrics.json").read_text(encoding="utf-8"))
    assert pinned == sorted(tuple(p) for p in wp["competitions"])
    card = Path("silly_kicks/xsuccess/weights/MODEL_CARD.md").read_text(encoding="utf-8")
    pairs = {(int(c), int(s)) for c, s in re.findall(r"\(comp (\d+), season (\d+)\)", card)}
    assert pinned == sorted(pairs)


def test_lock_open_identity_is_the_digest_of_the_pinned_corpus():
    rec = locked("sb_open")
    ident = rec["identity"]["open"]
    assert set(ident) == {"corpus_match_ids_sha256", "corpus_n_matches"}
    assert ident["corpus_n_matches"] == rec["n_units"] == 3961


def test_open_data_identity_is_order_free_and_provider_keyed():
    a = open_data_identity(["3", "1", "2"])
    assert a == open_data_identity(["1", "2", "3"]) and a["corpus_n_matches"] == 3
    assert a != open_data_identity(["1", "2"])


def test_fresh_sb_open_64_matches_fails(tmp_path):
    fresh = _fresh_xsuccess(tmp_path / "f", open_data_identity([str(i) for i in range(64)]))
    bad = A.expectation_violations(MAN.artifacts["xsuccess"], fresh, _record("sb_open"), C)
    assert any("corpus identity" in v for v in bad), bad


def test_fresh_sb_open_lock_identity_passes(tmp_path):
    fresh = _fresh_xsuccess(tmp_path / "f", expected_identity("sb_open", "open"))
    assert A.expectation_violations(MAN.artifacts["xsuccess"], fresh, _record("sb_open"), C) == []


def test_fresh_sb_open_without_identity_fails(tmp_path):
    fresh = _fresh_xsuccess(tmp_path / "f", {})
    bad = A.expectation_violations(MAN.artifacts["xsuccess"], fresh, _record("sb_open"), C)
    assert any("corpus identity" in v for v in bad), bad


#: ADR-056: the derived population of certified artifacts, asserted exactly.
CERTIFIED = {
    "xshot_default", "xshot_position_only", "xcross_default", "xcross_position_only",
    "ghost_default", "ghost_position_only", "ghost_sweeper", "ghost_sweeper_position_only",
    "outfield_default", "outfield_position_only", "receiver_default",
    "win_probability", "match_outcome", "expected_passing", "xsuccess",
    "hub_xshot_sc_extended", "hub_xshot_sc_extended_position_only",
    "hub_xcross_sc_extended", "hub_xcross_sc_extended_position_only",
    "ghost_full", "hub_ghost_sweeper", "hub_ghost_sweeper_position_only",
    "hub_outfield_default", "hub_outfield_position_only", "hub_xsuccess",
}  # fmt: skip


def test_every_certified_artifact_has_corpus_scope():
    population = {a.id for a in MAN.artifacts.values() if a.certification in ("public", "owner")}
    assert population == CERTIFIED
    missing = {aid for aid in population if A.corpus_scope(MAN.artifacts[aid]) is None}
    assert missing == set()


def test_corpus_scope_gate_has_teeth():
    """Anti-rot: an artifact with no corpus rule, no count expectation and a staged dataset has no scope."""
    bare = dataclasses.replace(MAN.artifacts["receiver_default"], corpus=None, expect={})
    assert A.corpus_scope(bare) is None


@pytest.mark.parametrize("aid", EVENT_ONLY)
def test_bundle_metrics_json_is_listed(aid):
    from scripts._hub_publish import MODEL_ONLY_ALLOWLIST

    a = MAN.artifacts[aid]
    cfg = load_toml(Path("pyproject.toml"))["tool"]["hatch"]["build"]["targets"]
    wheel = cfg["wheel"]
    assert "silly_kicks" in wheel["packages"]
    assert not any(a.path.startswith(x) or x.startswith(a.path) for x in wheel.get("exclude", [])), a.path
    assert "metrics.json" in MODEL_ONLY_ALLOWLIST  # the hub_xsuccess publish seam (ADR-088)
    assert a.dataset is not None and declared()[a.dataset].kind == "statsbomb_open"


def test_corpus_check_adds_no_read(tmp_path, monkeypatch):
    """Owner limit: the corpus check is a local compare. Verify opens no store; the check reads only the
    retrieved ``metrics.json`` and the lock."""
    from scripts.hf_jobs import store as S

    def _no_store(*_a, **_k):
        raise AssertionError("verify must not read the bucket")

    monkeypatch.setattr(S.BucketStore, "from_uri", _no_store)
    reads: list[str] = []
    real = A._read_json
    monkeypatch.setattr(A, "_read_json", lambda p: reads.append(Path(p).name) or real(p))
    fresh = _fresh_xsuccess(tmp_path / "f", expected_identity("sb_open", "open"))
    assert A.expectation_violations(MAN.artifacts["xsuccess"], fresh, _record("sb_open"), C) == []
    assert sorted(set(reads)) == ["metadata.json", "metrics.json"]
