"""TF-24 calibrate CLI arg parsing + fold-loader threading (plan Task 9b).

The local TF-24 sweep OOMs when the loader pulls all matches at full tracking
depth; these flags bound memory. Verify ``--match-ids`` parsing and that
``_load_fold`` threads ``match_ids`` / ``tracking_limit`` / ``max_per_provider``
into ``load_matches`` instead of the old hardcoded ``match_ids=None``.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

import scripts.calibrate_tracking_defaults as C


def _actions(game_id):
    return pd.DataFrame({"action_id": [1, 2], "game_id": [game_id, game_id]})


def _frames(game_id):
    return pd.DataFrame({"frame_id": [1, 2], "game_id": [game_id, game_id], "is_ball": [True, False]})


def test_game_id_guard_passes_when_consistent():
    # Consistent game_id (string or int) on both sides -> no raise.
    C._assert_match_game_id_consistent("idsse", "M1", _actions("DFL-MAT-1"), _frames("DFL-MAT-1"))
    C._assert_match_game_id_consistent("gradientsports", "10502", _actions(10502), _frames(10502))


@pytest.mark.parametrize(
    "a_id, f_id",
    [
        (None, "DFL-MAT-1"),  # defensive: a None-vs-value game_id mismatch must still fail loud.
        # (The IDSSE spadl_kloppy path historically left actions game_id None; the PR-S95 native DFL
        # re-route now derives game_id from match_id on BOTH sides -> IDSSE is consistent. The guard
        # stays as a generic safety net against any future drop-causing mismatch.)
        ("DFL-MAT-1", "DFL-MAT-2"),  # genuine value mismatch
    ],
)
def test_game_id_guard_raises_on_mismatch(a_id, f_id):
    # A game_id mismatch silently drops the whole match from every tracking-feature join; the guard
    # must fail loud rather than let the provider be quietly excluded by signal_sanity.
    with pytest.raises(ValueError, match="game_id"):
        C._assert_match_game_id_consistent("idsse", "M1", _actions(a_id), _frames(f_id))


def test_load_fold_raises_on_game_id_mismatch(monkeypatch):
    # _load_fold must invoke the guard so a mismatched match aborts the sweep up front.
    import scripts._loader_pining as L

    monkeypatch.setattr(
        L, "load_matches", lambda **k: iter([("idsse", "M1", _actions(None), _frames("DFL-MAT-1"), "H")])
    )
    args = types.SimpleNamespace(
        source="pining", providers=["idsse"], match_ids=None, tracking_limit=None, max_matches_per_provider=None
    )
    with pytest.raises(ValueError, match="game_id"):
        C._load_fold(args)


def test_parse_match_ids_repeatable():
    out = C._parse_match_ids(["gradientsports:10517,10519", "idsse:M1"])
    assert out == {"gradientsports": ["10517", "10519"], "idsse": ["M1"]}


def test_parse_match_ids_none():
    assert C._parse_match_ids(None) is None
    assert C._parse_match_ids([]) is None


def test_parse_match_ids_rejects_malformed():
    with pytest.raises(ValueError, match="PROVIDER:id1"):
        C._parse_match_ids(["gradientsports"])  # no ':ids'


def test_load_fold_threads_memory_bounds(monkeypatch):
    captured = {}

    acts, frms = _actions("10517"), _frames("10517")  # consistent game_id passes the fold guard

    def _fake_load_matches(**kwargs):
        captured.update(kwargs)
        return iter([("gradientsports", "10517", acts, frms, "H")])

    import scripts._loader_pining as L

    monkeypatch.setattr(L, "load_matches", _fake_load_matches)
    args = types.SimpleNamespace(
        source="pining",
        providers=["gradientsports"],
        match_ids=["gradientsports:10517"],
        tracking_limit=200,
        max_matches_per_provider=3,
    )
    fold, used_ids = C._load_fold(args)

    assert captured["match_ids"] == {"gradientsports": ["10517"]}
    assert captured["tracking_limit"] == 200
    assert captured["max_per_provider"] == 3
    assert captured["providers"] == ["gradientsports"]
    assert used_ids == {"gradientsports": ["10517"]}
    assert len(fold["gradientsports"]) == 1
    a, f, h = fold["gradientsports"][0]
    assert h == "H" and a is acts and f is frms
