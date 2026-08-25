"""Task 17 (ADR-068): _load_fold refuses LOUDLY before an OOM instead of silently capping the
calibration corpus (a silent subset would corrupt the sweep -- the fold IS the objective's input)."""

import argparse

import calibrate_tracking_defaults as cal
import pandas as pd
import pytest


def _fake_match(mid: str, game_id: int = 1):
    actions = pd.DataFrame({"game_id": [game_id], "action_id": [0]})
    frames = pd.DataFrame({"game_id": [game_id], "frame_id": [0], "x": [1.0]})
    return "gradientsports", mid, actions, frames, 1


def _args(**kw):
    d = dict(
        source="pining",
        providers=["gradientsports"],
        match_ids=None,
        tracking_limit=None,
        max_matches_per_provider=None,
        cache_dir=None,
        allow_large=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def _patch_loader(monkeypatch, matches):
    import scripts._loader_pining as loader

    monkeypatch.setattr(loader, "load_matches", lambda **kw: iter(matches))
    monkeypatch.setattr(cal, "_assert_match_game_id_consistent", lambda *a, **k: None)


def test_load_fold_fails_fast_over_budget(monkeypatch):
    _patch_loader(monkeypatch, [_fake_match("1"), _fake_match("2")])
    monkeypatch.setattr(cal, "_load_budget_bytes", lambda: 1)  # 1-byte budget -> first match trips it
    with pytest.raises(RuntimeError, match=r"fail-fast budget|--allow-large"):
        cal._load_fold(_args())


def test_allow_large_disables_the_guard(monkeypatch):
    _patch_loader(monkeypatch, [_fake_match("1")])
    monkeypatch.setattr(cal, "_load_budget_bytes", lambda: 1)  # would trip, but --allow-large skips it
    fold, used = cal._load_fold(_args(allow_large=True))
    assert used == {"gradientsports": ["1"]}
    assert len(fold["gradientsports"]) == 1


def test_within_budget_loads_all(monkeypatch):
    _patch_loader(monkeypatch, [_fake_match("1"), _fake_match("2")])
    monkeypatch.setattr(cal, "_load_budget_bytes", lambda: 10**12)  # roomy -> no refusal
    _fold, used = cal._load_fold(_args())
    assert used == {"gradientsports": ["1", "2"]}
