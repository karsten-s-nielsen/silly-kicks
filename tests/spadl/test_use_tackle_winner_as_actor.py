"""Tests for ``silly_kicks.spadl.use_tackle_winner_as_actor`` (added in 2.0.0).

Mirrors the PR-S8 ``boundary_metrics`` test discipline. The helper is a
post-conversion enrichment that restores pre-2.0.0 sportec SPADL
"actor = winner" semantic for consumers whose upstream identifier
conventions match DFL's tackle_winner_* qualifier format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import use_tackle_winner_as_actor

_TACKLE_ID = spadlconfig.actiontype_id["tackle"]


def _actions_one_tackle(
    *,
    team_id: object = "home",
    player_id: object = "P-LOSER",
    winner_player_id: object = "P-WINNER",
    winner_team_id: object = "DFL-CLU-X",
    type_id: int = _TACKLE_ID,
) -> pd.DataFrame:
    """Build a one-row sportec-shape SPADL action DataFrame."""
    return pd.DataFrame(
        {
            "team_id": [team_id],
            "player_id": [player_id],
            "tackle_winner_player_id": [winner_player_id],
            "tackle_winner_team_id": [winner_team_id],
            "tackle_loser_player_id": ["P-LOSER"],
            "tackle_loser_team_id": ["DFL-CLU-Y"],
            "type_id": [type_id],
            "extra_col": ["preserved"],
        }
    )


class TestUseTackleWinnerAsActorContract:
    def test_returns_dataframe(self):
        result = use_tackle_winner_as_actor(_actions_one_tackle())
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self):
        actions = _actions_one_tackle()
        original_team = actions["team_id"].iloc[0]
        use_tackle_winner_as_actor(actions)
        assert actions["team_id"].iloc[0] == original_team

    def test_missing_team_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["team_id"])
        with pytest.raises(ValueError, match="team_id"):
            use_tackle_winner_as_actor(actions)

    def test_missing_player_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["player_id"])
        with pytest.raises(ValueError, match="player_id"):
            use_tackle_winner_as_actor(actions)

    def test_missing_tackle_winner_player_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["tackle_winner_player_id"])
        with pytest.raises(ValueError, match="tackle_winner_player_id"):
            use_tackle_winner_as_actor(actions)

    def test_missing_tackle_winner_team_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["tackle_winner_team_id"])
        with pytest.raises(ValueError, match="tackle_winner_team_id"):
            use_tackle_winner_as_actor(actions)


class TestUseTackleWinnerAsActorCorrectness:
    def test_overwrites_team_id_from_winner_team_when_non_null(self):
        result = use_tackle_winner_as_actor(_actions_one_tackle())
        assert result["team_id"].iloc[0] == "DFL-CLU-X"

    def test_overwrites_player_id_from_winner_player_when_non_null(self):
        result = use_tackle_winner_as_actor(_actions_one_tackle())
        assert result["player_id"].iloc[0] == "P-WINNER"

    def test_atomic_overwrite_both_or_neither(self):
        actions = pd.DataFrame(
            {
                "team_id": ["home", "home"],
                "player_id": ["P-LOSER1", "P-LOSER2"],
                "tackle_winner_player_id": ["P-WIN1", np.nan],
                "tackle_winner_team_id": ["DFL-CLU-A", np.nan],
                "tackle_loser_player_id": ["P-LOSER1", "P-LOSER2"],
                "tackle_loser_team_id": ["DFL-CLU-B", "DFL-CLU-B"],
                "type_id": [_TACKLE_ID, _TACKLE_ID],
            }
        )
        result = use_tackle_winner_as_actor(actions)
        assert result["team_id"].iloc[0] == "DFL-CLU-A"
        assert result["player_id"].iloc[0] == "P-WIN1"
        assert result["team_id"].iloc[1] == "home"
        assert result["player_id"].iloc[1] == "P-LOSER2"

    def test_rows_with_nan_winner_left_unchanged(self):
        actions = _actions_one_tackle(winner_player_id=np.nan, winner_team_id=np.nan)
        result = use_tackle_winner_as_actor(actions)
        assert result["team_id"].iloc[0] == "home"
        assert result["player_id"].iloc[0] == "P-LOSER"

    def test_preserves_all_other_columns(self):
        actions = _actions_one_tackle()
        result = use_tackle_winner_as_actor(actions)
        assert result["extra_col"].iloc[0] == "preserved"
        assert result["type_id"].iloc[0] == _TACKLE_ID

    def test_preserves_tackle_winner_columns_themselves(self):
        actions = _actions_one_tackle()
        result = use_tackle_winner_as_actor(actions)
        assert result["tackle_winner_player_id"].iloc[0] == "P-WINNER"
        assert result["tackle_winner_team_id"].iloc[0] == "DFL-CLU-X"

    def test_non_tackle_rows_with_non_null_winner_still_swap(self):
        # Helper acts purely on tackle_winner_*_id non-null status, not
        # type_id. This is by design — the converter guarantees winner
        # cols are NaN on non-tackle rows, so this scenario only arises
        # with hand-crafted fixtures.
        actions = _actions_one_tackle(type_id=spadlconfig.actiontype_id["pass"])
        result = use_tackle_winner_as_actor(actions)
        assert result["team_id"].iloc[0] == "DFL-CLU-X"


class TestUseTackleWinnerAsActorDegenerate:
    def test_empty_dataframe_returns_empty(self):
        actions = pd.DataFrame(
            {
                "team_id": pd.Series([], dtype="object"),
                "player_id": pd.Series([], dtype="object"),
                "tackle_winner_player_id": pd.Series([], dtype="object"),
                "tackle_winner_team_id": pd.Series([], dtype="object"),
            }
        )
        result = use_tackle_winner_as_actor(actions)
        assert len(result) == 0
        assert list(result.columns) == ["team_id", "player_id", "tackle_winner_player_id", "tackle_winner_team_id"]

    def test_all_nan_winner_returns_identity(self):
        actions = pd.DataFrame(
            {
                "team_id": ["home", "away"],
                "player_id": ["P1", "P2"],
                "tackle_winner_player_id": [np.nan, np.nan],
                "tackle_winner_team_id": [np.nan, np.nan],
            }
        )
        result = use_tackle_winner_as_actor(actions)
        assert result["team_id"].tolist() == ["home", "away"]
        assert result["player_id"].tolist() == ["P1", "P2"]

    def test_preserves_preserve_native_columns(self):
        actions = _actions_one_tackle()
        actions["my_preserved"] = ["xyz"]
        result = use_tackle_winner_as_actor(actions)
        assert "my_preserved" in result.columns
        assert result["my_preserved"].iloc[0] == "xyz"
