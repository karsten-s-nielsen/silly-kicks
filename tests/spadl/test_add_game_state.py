"""Tests for add_game_state enrichment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config
from silly_kicks.spadl.utils import add_game_state


def _make_actions(events: list[tuple[str, str, str, int, float]]) -> pd.DataFrame:
    """Helper: create minimal SPADL actions from (team, type, result, period, time)."""
    rows = []
    for i, (team, type_name, result_name, period, time_s) in enumerate(events):
        rows.append(
            {
                "game_id": 1,
                "action_id": i,
                "team_id": team,
                "type_name": type_name,
                "result_name": result_name,
                "type_id": config.actiontype_id.get(type_name, 0),
                "result_id": config.result_id.get(result_name, 0),
                "period_id": period,
                "time_seconds": time_s,
                "start_x": 50.0,
                "start_y": 34.0,
                "end_x": 60.0,
                "end_y": 34.0,
                "bodypart_id": 0,
                "player_id": 100,
            }
        )
    return pd.DataFrame(rows)


class TestAddGameState:
    """Unit tests for add_game_state."""

    def test_all_drawing_no_goals(self):
        """Match with no goals -> all actions are 'drawing'."""
        actions = _make_actions(
            [
                ("A", "pass", "success", 1, 10.0),
                ("B", "pass", "fail", 1, 15.0),
                ("A", "shot", "fail", 1, 20.0),
            ]
        )
        result = add_game_state(actions)
        assert (result["game_state"] == "drawing").all()

    def test_winning_losing_after_goal(self):
        """After team A scores, A is winning and B is losing."""
        actions = _make_actions(
            [
                ("A", "pass", "success", 1, 10.0),
                ("A", "shot", "success", 1, 20.0),  # GOAL for A
                ("B", "pass", "success", 1, 25.0),  # B is losing
                ("A", "pass", "success", 1, 30.0),  # A is winning
            ]
        )
        result = add_game_state(actions)
        assert result.iloc[0]["game_state"] == "drawing"
        assert result.iloc[1]["game_state"] == "winning"  # inclusive
        assert result.iloc[2]["game_state"] == "losing"
        assert result.iloc[3]["game_state"] == "winning"

    def test_equalizer_returns_to_drawing(self):
        """After A scores then B equalizes, both are drawing."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 10.0),  # 1-0
                ("B", "pass", "success", 1, 15.0),  # B losing
                ("B", "shot", "success", 1, 20.0),  # 1-1 EQUALIZER
                ("A", "pass", "success", 1, 25.0),  # drawing
                ("B", "pass", "success", 1, 30.0),  # drawing
            ]
        )
        result = add_game_state(actions)
        assert result.iloc[0]["game_state"] == "winning"  # A scored
        assert result.iloc[1]["game_state"] == "losing"  # B before equalizer
        # B scored the equalizer: score is 1-1. From B's perspective:
        # acting_goals = goals_b = 1, opponent_goals = goals_a = 1 → diff = 0 → drawing
        assert result.iloc[2]["game_state"] == "drawing"  # B scored equalizer: 1-1
        assert result.iloc[3]["game_state"] == "drawing"
        assert result.iloc[4]["game_state"] == "drawing"

    def test_multiple_goals_cumulative(self):
        """Score tracks correctly through multiple goals."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 10.0),  # 1-0
                ("A", "shot", "success", 1, 20.0),  # 2-0
                ("B", "shot", "success", 1, 30.0),  # 2-1
                ("B", "shot", "success", 1, 40.0),  # 2-2
                ("B", "shot", "success", 1, 50.0),  # 2-3
                ("A", "pass", "success", 1, 55.0),  # A is losing
            ]
        )
        result = add_game_state(actions)
        assert result.iloc[0]["game_state"] == "winning"  # A 1-0
        assert result.iloc[1]["game_state"] == "winning"  # A 2-0
        assert result.iloc[2]["game_state"] == "losing"  # B scored 2-1, still losing
        assert result.iloc[3]["game_state"] == "drawing"  # B scored 2-2
        assert result.iloc[4]["game_state"] == "winning"  # B scored 2-3
        assert result.iloc[5]["game_state"] == "losing"  # A is now 2-3

    def test_works_with_type_id_only(self):
        """Works when only type_id/result_id are present (no name columns)."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 10.0),
                ("B", "pass", "success", 1, 15.0),
            ]
        )
        actions = actions.drop(columns=["type_name", "result_name"])
        result = add_game_state(actions)
        assert result.iloc[0]["game_state"] == "winning"
        assert result.iloc[1]["game_state"] == "losing"

    def test_single_team_all_drawing(self):
        """Edge case: only one team in actions (incomplete data)."""
        actions = _make_actions(
            [
                ("A", "pass", "success", 1, 10.0),
                ("A", "shot", "success", 1, 20.0),
            ]
        )
        result = add_game_state(actions)
        assert (result["game_state"] == "drawing").all()

    def test_cross_period_goals_accumulate(self):
        """Goals in period 1 carry into period 2."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 2700.0),  # Goal end of first half
                ("B", "pass", "success", 2, 5.0),  # Start of second half
            ]
        )
        result = add_game_state(actions)
        assert result.iloc[0]["game_state"] == "winning"
        assert result.iloc[1]["game_state"] == "losing"

    def test_failed_shot_not_a_goal(self):
        """Shots with result != success don't count as goals."""
        actions = _make_actions(
            [
                ("A", "shot", "fail", 1, 10.0),
                ("A", "shot", "offside", 1, 20.0),
                ("B", "pass", "success", 1, 25.0),
            ]
        )
        result = add_game_state(actions)
        assert (result["game_state"] == "drawing").all()

    def test_preserves_original_columns(self):
        """All original columns are preserved, game_state is appended."""
        actions = _make_actions(
            [
                ("A", "pass", "success", 1, 10.0),
            ]
        )
        original_cols = set(actions.columns)
        result = add_game_state(actions)
        assert original_cols.issubset(set(result.columns))
        assert "game_state" in result.columns

    def test_does_not_mutate_input(self):
        """Input DataFrame is not modified in place."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 10.0),
            ]
        )
        original_cols = list(actions.columns)
        _ = add_game_state(actions)
        assert list(actions.columns) == original_cols
        assert "game_state" not in actions.columns

    def test_nan_team_id_tolerant(self):
        """NaN team_id rows get 'drawing' (ADR-003 NaN safety)."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 10.0),
                ("B", "pass", "success", 1, 15.0),
                ("B", "pass", "success", 1, 20.0),
                ("A", "pass", "success", 1, 25.0),
            ]
        )
        # Set one B row to NaN — both teams still discoverable via remaining rows
        actions.loc[1, "team_id"] = np.nan
        result = add_game_state(actions)
        assert len(result) == 4
        # Row 0: A scored, winning
        assert result.iloc[0]["game_state"] == "winning"
        # Row 1: NaN team → drawing (NaN != A, NaN != B → both where() branches false)
        assert result.iloc[1]["game_state"] == "drawing"
        # Row 2: B, losing (A leads 1-0)
        assert result.iloc[2]["game_state"] == "losing"
        # Row 3: A, winning
        assert result.iloc[3]["game_state"] == "winning"

    def test_importable_from_spadl(self):
        """add_game_state is importable from silly_kicks.spadl."""
        from silly_kicks.spadl import add_game_state as gs

        assert callable(gs)

    def test_output_column_values(self):
        """game_state column only contains the three valid values."""
        actions = _make_actions(
            [
                ("A", "shot", "success", 1, 10.0),
                ("B", "shot", "success", 1, 20.0),
                ("A", "pass", "success", 1, 25.0),
                ("B", "shot", "success", 1, 30.0),
            ]
        )
        result = add_game_state(actions)
        valid = {"winning", "losing", "drawing"}
        assert set(result["game_state"].unique()).issubset(valid)

    def test_empty_actions(self):
        """Empty DataFrame returns empty with game_state column."""
        actions = pd.DataFrame(
            columns=[
                "game_id",
                "action_id",
                "team_id",
                "type_name",
                "result_name",
                "type_id",
                "result_id",
                "period_id",
                "time_seconds",
            ]
        )
        result = add_game_state(actions)
        assert "game_state" in result.columns
        assert len(result) == 0
