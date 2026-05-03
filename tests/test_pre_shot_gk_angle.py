"""TF-12 -- pre_shot_gk_angle_to_shot_trajectory + pre_shot_gk_angle_off_goal_line."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import (
    add_pre_shot_gk_angle,
    pre_shot_gk_angle_off_goal_line,
    pre_shot_gk_angle_to_shot_trajectory,
)

_SHOT_TYPE_ID = spadlconfig.actiontype_id["shot"]


def _toy_actions_and_frames(gk_x: float, gk_y: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "game_id": ["G1"],
            "period_id": [1],
            "type_id": [_SHOT_TYPE_ID],
            "team_id": [1],
            "player_id": [10],
            "start_x": [95.0],
            "start_y": [34.0],
            "time_seconds": [10.0],
            "defending_gk_player_id": [99.0],
        }
    )
    frames = pd.DataFrame(
        {
            "game_id": ["G1"] * 2,
            "period_id": [1, 1],
            "frame_id": [100, 100],
            "time_seconds": [10.0, 10.0],
            "frame_rate": [25.0, 25.0],
            "player_id": [10, 99],
            "team_id": [1, 2],
            "is_ball": [False, False],
            "is_goalkeeper": [False, True],
            "x": [95.0, gk_x],
            "y": [34.0, gk_y],
            "z": [np.nan, np.nan],
            "speed": [0.0, 0.0],
            "speed_source": ["native", "native"],
            "ball_state": ["alive", "alive"],
            "team_attacking_direction": ["ltr", "ltr"],
            "source_provider": ["sportec", "sportec"],
        }
    )
    return actions, frames


def test_gk_on_shot_trajectory_returns_zero():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert math.isclose(float(s.iloc[0]), 0.0, abs_tol=1e-6)


def test_gk_displaced_positive_y_returns_positive_angle():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=36.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert float(s.iloc[0]) > 0


def test_gk_displaced_negative_y_returns_negative_angle():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=32.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert float(s.iloc[0]) < 0


def test_off_goal_line_zero_when_gk_on_goal_line_normal():
    """GK at (102, 34) is on the goal-line normal at goal-mouth centre -> angle approx 0."""
    actions, frames = _toy_actions_and_frames(gk_x=102.0, gk_y=34.0)
    s = pre_shot_gk_angle_off_goal_line(actions, frames)
    assert math.isclose(float(s.iloc[0]), 0.0, abs_tol=1e-6)


def test_off_goal_line_sign_flips_with_y():
    a_up, f_up = _toy_actions_and_frames(gk_x=102.0, gk_y=36.0)
    a_dn, f_dn = _toy_actions_and_frames(gk_x=102.0, gk_y=32.0)
    s_up = float(pre_shot_gk_angle_off_goal_line(a_up, f_up).iloc[0])
    s_dn = float(pre_shot_gk_angle_off_goal_line(a_dn, f_dn).iloc[0])
    assert s_up * s_dn < 0


def test_nan_when_defending_gk_id_missing():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    actions.loc[0, "defending_gk_player_id"] = np.nan
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert pd.isna(s.iloc[0])


def test_add_pre_shot_gk_angle_emits_two_columns():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    out = add_pre_shot_gk_angle(actions, frames=frames)
    new_cols = set(out.columns) - set(actions.columns)
    assert {"pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"} <= new_cols


def test_add_pre_shot_gk_angle_requires_defending_gk_player_id():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    actions = actions.drop(columns=["defending_gk_player_id"])
    with pytest.raises(ValueError, match="defending_gk_player_id"):
        add_pre_shot_gk_angle(actions, frames=frames)


def test_bounds_within_pi():
    actions, frames = _toy_actions_and_frames(gk_x=80.0, gk_y=10.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert -math.pi <= float(s.iloc[0]) <= math.pi


def test_atomic_mirror_emits_same_columns():
    """Atomic mirror reads (x, y) instead of (start_x, start_y)."""
    from silly_kicks.atomic.tracking.features import (
        add_pre_shot_gk_angle as atomic_add_pre_shot_gk_angle,
    )

    atomic_actions = pd.DataFrame(
        {
            "action_id": [1],
            "game_id": ["G1"],
            "period_id": [1],
            "type_id": [_SHOT_TYPE_ID],
            "team_id": [1],
            "player_id": [10],
            "x": [95.0],
            "y": [34.0],
            "time_seconds": [10.0],
            "defending_gk_player_id": [99.0],
        }
    )
    _, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    out = atomic_add_pre_shot_gk_angle(atomic_actions, frames=frames)
    assert {"pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"} <= set(out.columns)
