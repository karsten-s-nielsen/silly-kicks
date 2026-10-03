"""G1: gk_completion `skillcorner` is the wheel-bundled PUBLIC arm -- a restricted request is refused
before extraction; the requested ids are recorded (spec section 5)."""

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
train = importlib.import_module("train_gk_completion")
import _loader_pining  # noqa: E402


def _args(tmp_path, cache=None):
    return Namespace(
        max_per_provider=10,
        tracking_limit=None,
        cache_features=str(cache) if cache else None,
        mode="rebundle",
        reason="guard test",
        feature_space=None,
        probe_old=None,
        shard_dir=str(tmp_path / "shards"),
        cache_dir=None,
    )


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_SKILLCORNER_WEIGHTS_DIR", tmp_path / "skillcorner")
    monkeypatch.setattr(train, "_WEIGHTS_ROOT", tmp_path)


def test_skillcorner_refuses_a_restricted_request_before_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "_corpus_taxonomy", lambda providers, mpp: ("sc_extended", False))

    def _no_extraction(*a, **k):
        raise AssertionError("extraction started -- the G1 refusal must come first")

    monkeypatch.setattr(train, "_extract", _no_extraction)
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="not all-public"):
        train._train_skillcorner(_args(tmp_path))


def test_skillcorner_passes_a_public_request_to_extraction(tmp_path, monkeypatch):
    """The other side of the band: an all-public request reaches extraction."""
    monkeypatch.setattr(train, "_corpus_taxonomy", lambda providers, mpp: ("public", True))
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: [("skillcorner", "1886347")])

    def _reached(*a, **k):
        raise AssertionError("extraction reached")

    monkeypatch.setattr(train, "_extract", _reached)
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="extraction reached"):
        train._train_skillcorner(_args(tmp_path))


@pytest.mark.slow
def test_skillcorner_records_requested_match_ids(tmp_path, monkeypatch):
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES as FEATS

    rng = np.random.RandomState(0)
    n = 160
    df = pd.DataFrame({f: rng.randn(n) for f in FEATS})
    df["is_goalkick"] = (np.arange(n) % 4 == 0).astype(float)
    df["is_throw_in"] = 0.0
    df["_y"] = (rng.rand(n) < 0.6).astype(int)
    df["_group"] = np.arange(n) % 5
    cache = tmp_path / "feat.parquet"
    df.to_parquet(cache)
    monkeypatch.setattr(train, "_corpus_taxonomy", lambda providers, mpp: ("public", True))
    monkeypatch.setattr(
        _loader_pining, "select_match_ids", lambda **kw: [("skillcorner", "1886347"), ("skillcorner", "1899585")]
    )
    _isolate(monkeypatch, tmp_path)
    assert train._train_skillcorner(_args(tmp_path, cache)) == 0
    written = tmp_path / "skillcorner" / "metrics.json"
    if not written.exists():
        written = tmp_path / "skillcorner_remeasurement.json"
    m = json.loads(written.read_text(encoding="utf-8"))
    assert m["requested_match_ids"] == [["skillcorner", "1886347"], ["skillcorner", "1899585"]]
    assert m["artifact_label"] == "public" and m["all_public"] is True
