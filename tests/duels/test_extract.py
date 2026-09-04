"""extract_duels -- native (sportec winner/loser) + derived (tackle/take_on adjacency) + conservation."""

from __future__ import annotations

import pandas as pd

from silly_kicks.duels._extract import extract_duels

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


def test_native_extraction_preferred():
    a = _actions(
        [
            {
                "type_id": _TACKLE,
                "player_id": 100,
                "team_id": 10,
                "tackle_winner_player_id": 100,
                "tackle_winner_team_id": 10,
                "tackle_loser_player_id": 200,
                "tackle_loser_team_id": 20,
            },
        ]
    )
    games, report = extract_duels(a)
    assert report.labeling_strategy == "native"
    assert len(games) == 1 and report.n_native == 1
    g = games[0]
    assert g.winner_player == 100 and g.loser_player == 200 and g.source == "native"
    assert g.winner_team == 10 and g.loser_team == 20


def test_native_row_without_winner_is_not_a_duel():
    a = _actions(
        [
            {
                "type_id": _TACKLE,
                "tackle_winner_player_id": 100,
                "tackle_loser_player_id": 200,
                "tackle_winner_team_id": 10,
                "tackle_loser_team_id": 20,
            },
            {
                "type_id": _TACKLE,
                "tackle_winner_player_id": pd.NA,
                "tackle_loser_player_id": pd.NA,
                "tackle_winner_team_id": pd.NA,
                "tackle_loser_team_id": pd.NA,
            },
        ]
    )
    games, report = extract_duels(a)
    assert len(games) == 1 and report.n_candidate == 1  # the NA-winner tackle is skipped


def test_derived_tackle_wins():
    a = _actions(
        [
            {"type_id": _TACKLE, "player_id": 100, "team_id": 10, "result_id": _SUCCESS, "time_seconds": 5.0},
            {"type_id": _TAKE_ON, "player_id": 200, "team_id": 20, "result_id": _FAIL, "time_seconds": 5.1},
        ]
    )
    games, report = extract_duels(a)
    assert report.labeling_strategy == "derived"
    assert len(games) == 1
    g = games[0]
    assert g.winner_player == 100 and g.loser_player == 200 and g.source == "derived"


def test_derived_takeon_wins():
    a = _actions(
        [
            {"type_id": _TAKE_ON, "player_id": 200, "team_id": 20, "result_id": _SUCCESS, "time_seconds": 5.0},
            {"type_id": _TACKLE, "player_id": 100, "team_id": 10, "result_id": _FAIL, "time_seconds": 5.1},
        ]
    )
    games, _ = extract_duels(a)
    assert len(games) == 1 and games[0].winner_player == 200 and games[0].loser_player == 100


def test_derived_indeterminate_excluded_and_counted():
    a = _actions(
        [
            {"type_id": _TACKLE, "player_id": 100, "team_id": 10, "result_id": _SUCCESS, "time_seconds": 5.0},
            {"type_id": _TAKE_ON, "player_id": 200, "team_id": 20, "result_id": _SUCCESS, "time_seconds": 5.1},
        ]
    )
    games, report = extract_duels(a)
    assert len(games) == 0
    assert report.n_excluded == 1 and report.n_candidate == 1
    assert report.n_native + report.n_derived + report.n_excluded == report.n_candidate


def test_derived_same_team_is_not_a_duel():
    a = _actions(
        [
            {"type_id": _TACKLE, "player_id": 100, "team_id": 10, "result_id": _SUCCESS, "time_seconds": 5.0},
            {"type_id": _TAKE_ON, "player_id": 101, "team_id": 10, "result_id": _FAIL, "time_seconds": 5.1},
        ]
    )
    games, report = extract_duels(a)
    assert len(games) == 0 and report.n_candidate == 0


def test_derived_outside_window_is_not_a_duel():
    a = _actions(
        [
            {"type_id": _TACKLE, "player_id": 100, "team_id": 10, "result_id": _SUCCESS, "time_seconds": 5.0},
            {"type_id": _TAKE_ON, "player_id": 200, "team_id": 20, "result_id": _FAIL, "time_seconds": 30.0},
        ]
    )
    games, report = extract_duels(a)
    assert len(games) == 0 and report.n_candidate == 0
