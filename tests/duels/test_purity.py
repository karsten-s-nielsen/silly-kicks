"""compute_duel_ratings is PURE -- never mutates the caller's actions.

Two variants (native sportec winner/loser + derived tackle/take_on adjacency) exercise both extract paths.
"""

from __future__ import annotations

import pandas as pd
import pytest

_TACKLE, _TAKE_ON = 9, 7
_SUCCESS, _FAIL = 1, 0


def _native_actions() -> pd.DataFrame:
    rows = [
        {
            "game_id": 1,
            "period_id": 1,
            "action_id": 0,
            "time_seconds": 5.0,
            "team_id": 10,
            "player_id": 100,
            "type_id": _TACKLE,
            "result_id": _SUCCESS,
            "tackle_winner_player_id": 100,
            "tackle_winner_team_id": 10,
            "tackle_loser_player_id": 200,
            "tackle_loser_team_id": 20,
        },
    ]
    return pd.DataFrame(rows)


def _derived_actions() -> pd.DataFrame:
    rows = [
        {
            "game_id": 1,
            "period_id": 1,
            "action_id": 0,
            "time_seconds": 5.0,
            "team_id": 10,
            "player_id": 100,
            "type_id": _TACKLE,
            "result_id": _SUCCESS,
        },
        {
            "game_id": 1,
            "period_id": 1,
            "action_id": 1,
            "time_seconds": 5.1,
            "team_id": 20,
            "player_id": 200,
            "type_id": _TAKE_ON,
            "result_id": _FAIL,
        },
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("factory", [_native_actions, _derived_actions])
def test_compute_does_not_mutate_actions(factory):
    from silly_kicks.duels import compute_duel_ratings

    actions = factory()
    snapshot = actions.copy(deep=True)
    out, _ = compute_duel_ratings(actions)
    pd.testing.assert_frame_equal(actions, snapshot)  # input untouched
    assert out is not actions
