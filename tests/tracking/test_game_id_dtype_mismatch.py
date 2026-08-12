"""Regression tests: game_id dtype mismatch between actions (int64) and frames (str).

Lakehouse SPADL pipelines produce actions.game_id as int64 (hash_native_id_to_bigint)
while frames.game_id retains native string values. Merges/lookups on game_id must
tolerate mixed dtypes.

PR-S53: _defensive_line_at_actions, ball_carrier_at_action, _team_shape_at_actions.
PR-S44 (prior art): _off_ball_runs, _line_breaking.
"""

from __future__ import annotations

import pandas as pd

from tests.tracking.test_defensive_line import _make_frame_rows


def _actions_with_str_game_id(game_id: str = "3817") -> pd.DataFrame:
    """Actions with string game_id (native provider ID)."""
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "action_id": [1, 2],
            "period_id": [1, 1],
            "time_seconds": [1.0, 1.0],
            "team_id": [1, 1],
            "player_id": [50, 51],
            "start_x": [50.0, 55.0],
            "start_y": [34.0, 34.0],
            "end_x": [55.0, 60.0],
            "end_y": [34.0, 34.0],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "bodypart_id": [0, 0],
        }
    )


def _actions_with_int_game_id(game_id: int = 3817) -> pd.DataFrame:
    """Actions with int64 game_id (lakehouse hash_native_id_to_bigint)."""
    df = _actions_with_str_game_id(str(game_id))
    df["game_id"] = game_id
    return df


def _frames_with_str_game_id(game_id: str = "3817") -> pd.DataFrame:
    """Frames with string game_id (native provider value)."""
    frames = _make_frame_rows(
        home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
        away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
    )
    frames["game_id"] = game_id
    # ball_carrier_at_action requires ball_state for infer_ball_carrier
    if "ball_state" not in frames.columns:
        frames["ball_state"] = "alive"
    return frames


class TestDefensiveLineDtypeMismatch:
    """_defensive_line_at_actions merge on game_id."""

    def test_int_actions_str_frames(self):
        from silly_kicks.tracking.features import add_defensive_line

        actions = _actions_with_int_game_id(3817)
        frames = _frames_with_str_game_id("3817")
        result = add_defensive_line(actions, frames)
        # Must not raise ValueError on int64 vs object merge
        assert "defensive_line_x" in result.columns
        # At least one non-NaN value (actions are linkable to frames)
        assert result["defensive_line_x"].notna().any()

    def test_str_actions_str_frames(self):
        """Baseline: same dtype should work."""
        from silly_kicks.tracking.features import add_defensive_line

        actions = _actions_with_str_game_id("3817")
        frames = _frames_with_str_game_id("3817")
        result = add_defensive_line(actions, frames)
        assert result["defensive_line_x"].notna().any()


class TestBallCarrierAtActionDtypeMismatch:
    """ball_carrier_at_action merge on game_id."""

    def test_int_actions_str_frames(self):
        from silly_kicks.tracking.features import ball_carrier_at_action

        actions = _actions_with_int_game_id(3817)
        frames = _frames_with_str_game_id("3817")
        result = ball_carrier_at_action(actions, frames)
        # Must not raise ValueError on int64 vs object merge
        assert len(result) == len(actions)


class TestTeamShapeAtActionsDtypeMismatch:
    """_team_shape_at_actions dict-key lookup on game_id."""

    def test_int_actions_str_frames(self):
        from silly_kicks.tracking.features import add_team_shape

        actions = _actions_with_int_game_id(3817)
        frames = _frames_with_str_game_id("3817")
        result = add_team_shape(actions, frames)
        # Must not raise ValueError or silently produce all-NaN
        assert "team_shape_convex_hull_area_attacking" in result.columns
        assert result["team_shape_convex_hull_area_attacking"].notna().any()
