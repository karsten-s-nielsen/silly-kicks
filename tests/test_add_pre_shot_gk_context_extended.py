"""Extended add_pre_shot_gk_context -- 4 columns when frames=None (PR-S21 backcompat),
6 columns when frames=... (PR-S24 umbrella facade extension).

Backcompat: frames=None path is bit-identical to silly-kicks 2.9.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_pre_shot_gk_context

_SHOT_TYPE_ID = spadlconfig.actiontype_id["shot"]
_KEEPER_SAVE_TYPE_ID = spadlconfig.actiontype_id["keeper_save"]


def _toy_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [1, 2, 3],
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "type_id": [_KEEPER_SAVE_TYPE_ID, 0, _SHOT_TYPE_ID],
            "team_id": [2, 1, 1],
            "player_id": [99, 10, 11],
            "start_x": [5.0, 50.0, 95.0],
            "start_y": [34.0, 30.0, 34.0],
            "end_x": [10.0, 60.0, 105.0],
            "end_y": [34.0, 32.0, 34.0],
            "time_seconds": [1.0, 5.0, 10.0],
        }
    )


def _toy_frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1] * 2,
            "period_id": [1, 1],
            "frame_id": [100, 100],
            "time_seconds": [10.0, 10.0],
            "frame_rate": [25.0, 25.0],
            "player_id": [11, 99],
            "team_id": [1, 2],
            "is_ball": [False, False],
            "is_goalkeeper": [False, True],
            "x": [95.0, 102.0],
            "y": [34.0, 34.0],
            "z": [np.nan, np.nan],
            "speed": [0.0, 0.0],
            "speed_source": ["native", "native"],
            "ball_state": ["alive", "alive"],
            "team_attacking_direction": ["ltr", "ltr"],
            "source_provider": ["sportec", "sportec"],
        }
    )


def test_frames_none_emits_four_pr_s21_cols():
    out = add_pre_shot_gk_context(_toy_actions())
    assert {
        "gk_was_distributing",
        "gk_was_engaged",
        "gk_actions_in_possession",
        "defending_gk_player_id",
    } <= set(out.columns)
    # No frames-related columns
    assert "pre_shot_gk_x" not in out.columns
    assert "pre_shot_gk_angle_to_shot_trajectory" not in out.columns
    assert "pre_shot_gk_angle_off_goal_line" not in out.columns


def test_frames_supplied_emits_six_cols():
    out = add_pre_shot_gk_context(_toy_actions(), frames=_toy_frames())
    expected_new = {
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
        "pre_shot_gk_angle_to_shot_trajectory",
        "pre_shot_gk_angle_off_goal_line",
    }
    assert expected_new <= set(out.columns)
