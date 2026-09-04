"""compute_duel_ratings -- per-(player, match) Glicko-2 orchestration, schema, conservation, ids."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.duels import DUEL_COLUMNS, compute_duel_ratings

_TACKLE, _TAKE_ON = 9, 7
_SUCCESS, _FAIL = 1, 0


def _actions(rows: list[dict]) -> pd.DataFrame:
    base = {
        "game_id": 1,
        "period_id": 1,
        "action_id": 0,
        "time_seconds": 0.0,
        "team_id": 10,
        "player_id": 100,
        "type_id": _TAKE_ON,
        "result_id": _SUCCESS,
    }
    return pd.DataFrame([{**base, **r, "action_id": i} for i, r in enumerate(rows)])


def _native(game_id, w, wt, loser, lt, t=5.0):
    return {
        "game_id": game_id,
        "type_id": _TACKLE,
        "time_seconds": t,
        "player_id": w,
        "team_id": wt,
        "tackle_winner_player_id": w,
        "tackle_winner_team_id": wt,
        "tackle_loser_player_id": loser,
        "tackle_loser_team_id": lt,
    }


def test_schema_and_dtypes():
    a = _actions([_native(1, 100, 10, 200, 20)])
    samples, report = compute_duel_ratings(a)
    assert list(samples.columns) == list(DUEL_COLUMNS)
    for col, dt in DUEL_COLUMNS.items():
        assert str(samples[col].dtype) == dt
    assert report.labeling_strategy == "native"


def test_one_duel_two_rows_winner_above_loser():
    a = _actions([_native(1, 100, 10, 200, 20)])
    samples, report = compute_duel_ratings(a)
    assert report.n_player_match_rows == 2 and len(samples) == 2
    r100 = samples[samples["player_id"] == 100]
    r200 = samples[samples["player_id"] == 200]
    assert r100["duel_rating"].iloc[0] > 1500.0 > r200["duel_rating"].iloc[0]
    assert r100["duels_won"].iloc[0] == 1 and r100["duels_lost"].iloc[0] == 0
    assert r200["duels_lost"].iloc[0] == 1 and r200["duels_contested"].iloc[0] == 1
    assert set(samples["duel_winner_source"]) == {"native"}


def test_report_conserves():
    a = _actions(
        [
            _native(1, 100, 10, 200, 20, t=5.0),
            _native(1, 100, 10, 300, 20, t=6.0),  # player 100 wins two duels this match
        ]
    )
    _, report = compute_duel_ratings(a)
    assert report.n_matches == 1 and report.n_duels == 2 and report.n_duels_excluded == 0
    # player 100 contests twice -> one deduped row with contested=2
    samples, _ = compute_duel_ratings(a)
    assert samples[samples["player_id"] == 100]["duels_contested"].iloc[0] == 2


def test_carries_rating_across_matches():
    # player 100 wins in match 1 (rating up), then again in match 2 -> even higher post-match rating.
    a = _actions(
        [
            _native(1, 100, 10, 200, 20, t=5.0),
            _native(2, 100, 10, 300, 30, t=5.0),
        ]
    )
    samples, report = compute_duel_ratings(a)
    assert report.n_matches == 2
    r1 = samples[(samples["game_id"] == 1) & (samples["player_id"] == 100)]["duel_rating"].iloc[0]
    r2 = samples[(samples["game_id"] == 2) & (samples["player_id"] == 100)]["duel_rating"].iloc[0]
    assert r2 > r1 > 1500.0  # winning again lifts the carried-forward rating further


def test_empty_actions_returns_empty_schema():
    a = _actions([{"type_id": _TAKE_ON, "result_id": _SUCCESS}])  # a lone take_on -> no duel
    samples, report = compute_duel_ratings(a)
    assert len(samples) == 0 and list(samples.columns) == list(DUEL_COLUMNS)
    assert report.n_duels == 0 and report.n_player_match_rows == 0


def test_mixed_dtype_player_id_does_not_fragment():
    # same player as int 100 (match 1) and str "100" (match 2) -> canonical key unifies the trajectory.
    a = _actions(
        [
            _native(1, 100, 10, 200, 20, t=5.0),
            _native(2, "100", "10", "300", "30", t=5.0),
        ]
    )
    samples, _ = compute_duel_ratings(a)
    r1 = samples[samples["game_id"] == 1]
    r2 = samples[samples["game_id"] == 2]
    win1 = r1[r1["player_id"] == 100]["duel_rating"].iloc[0]
    win2 = r2[r2["player_id"] == "100"]["duel_rating"].iloc[0]
    assert win2 > win1  # carried forward as ONE player, not two seeds


def test_resume_equivalence():
    # B4.1b: one batch of N matches == two batches threaded via initial_ratings (byte-equal final state).
    a = _actions([_native(1, 100, 10, 200, 20, t=5.0), _native(2, 100, 10, 300, 30, t=5.0)])
    _, full = compute_duel_ratings(a)
    _, r1 = compute_duel_ratings(a[a["game_id"] == 1])
    _, r2 = compute_duel_ratings(a[a["game_id"] == 2], initial_ratings=r1.final_ratings)
    assert r2.final_ratings == full.final_ratings  # GlickoState dataclass equality, field-for-field
    # non-vacuity: the resumed batch actually carried the seed (player 100 above its 1500 seed).
    assert full.final_ratings["100"].rating > 1500.0


def test_resume_equivalence_holds_for_inactive_player():
    # player 200 loses in match 1, is INACTIVE in match 2 -> RD grows; resume must reproduce that.
    a = _actions([_native(1, 100, 10, 200, 20, t=5.0), _native(2, 300, 30, 400, 40, t=5.0)])
    _, full = compute_duel_ratings(a)
    _, r1 = compute_duel_ratings(a[a["game_id"] == 1])
    _, r2 = compute_duel_ratings(a[a["game_id"] == 2], initial_ratings=r1.final_ratings)
    assert r2.final_ratings == full.final_ratings
    assert r2.final_ratings["200"].rd > r1.final_ratings["200"].rd  # inactivity grew 200's RD in match 2


def test_order_insensitive_permuted_rows():
    # B4.1c / ADR-065: permuting input row order yields an identical result (modulo row order).
    a = _actions(
        [
            _native(1, 100, 10, 200, 20, t=5.0),
            _native(1, 300, 30, 400, 40, t=6.0),
            _native(2, 100, 10, 300, 30, t=5.0),
        ]
    )
    base, _ = compute_duel_ratings(a)
    shuffled, _ = compute_duel_ratings(a.iloc[[2, 0, 1]].reset_index(drop=True))
    key = ["game_id", "player_id"]
    b = base.sort_values(key).reset_index(drop=True)
    s = shuffled.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(b, s)


def _climb() -> pd.DataFrame:
    # player 100 wins in matches 1, 2, 3 -> a monotone rating climb.
    return _actions(
        [
            _native(1, 100, 10, 200, 20, t=5.0),
            _native(2, 100, 10, 300, 30, t=5.0),
            _native(3, 100, 10, 400, 40, t=5.0),
        ]
    )


def test_window_slice_as_of_end():
    # B4.1d: window=[2,3] -> one row per player, game_id NA, rating AS OF the last game in the window.
    a = _climb()
    full, _ = compute_duel_ratings(a)
    r3 = full[(full["game_id"] == 3) & (full["player_id"] == 100)]["duel_rating"].iloc[0]
    win, _ = compute_duel_ratings(a, window=[2, 3])
    row = win[win["player_id"] == 100]
    assert len(win) == len(win["player_id"].unique())  # one row per player
    assert row["game_id"].isna().all()  # windowed rows carry NA game_id
    assert row["duel_rating"].iloc[0] == r3  # as-of-end == the match-3 rating
    assert row["duels_contested"].iloc[0] == 2  # matches 2 + 3 summed


def test_window_slice_change_from_before_window():
    # window_stat="change": rating delta from just BEFORE the window (after match 1) to window end (match 3).
    a = _climb()
    full, _ = compute_duel_ratings(a)
    r1 = full[(full["game_id"] == 1) & (full["player_id"] == 100)]["duel_rating"].iloc[0]
    r3 = full[(full["game_id"] == 3) & (full["player_id"] == 100)]["duel_rating"].iloc[0]
    win, _ = compute_duel_ratings(a, window=[2, 3], window_stat="change")
    change = win[win["player_id"] == 100]["duel_rating"].iloc[0]
    assert change == pytest.approx(r3 - r1)


def test_unknown_window_stat_raises():
    with pytest.raises(ValueError, match="window_stat"):
        compute_duel_ratings(_climb(), window=[1], window_stat="bogus")
