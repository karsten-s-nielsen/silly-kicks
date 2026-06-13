"""ADR-028: pre-shot GK + pressure are emitted in the action-LTR frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_pre_shot_gk_context

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]


def _frame_rows():
    # Home-attacks-right frame: home GK defends x=0 (~3), away GK defends x=105 (~102).
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=250,
        time_seconds=10.0,
        frame_rate=25.0,
        z=0.0,
        speed=0.0,
        speed_source="native",
        ball_state="alive",
        confidence=None,
        visibility=None,
        source_provider="synthetic",
        is_goalkeeper_source="native",
    )
    rows = [
        dict(
            player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=3.0, y=34.0, team_attacking_direction="ltr"
        ),
        dict(
            player_id=50,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=True,
            x=102.0,
            y=34.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=11,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=40.0,
            y=20.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=61,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=65.0,
            y=40.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=13.0,
            y=34.0,
            team_attacking_direction=None,
        ),
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_away_shot_gk_reprojected_to_attacked_goal():
    frames = _frame_rows()
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=HOME,
                player_id=1.0,
                type_id=GOALKICK,
                result_id=1,
                start_x=5.0,
                start_y=34.0,
                end_x=40.0,
                end_y=34.0,
                time_seconds=9.6,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=AWAY,
                player_id=99.0,
                type_id=SHOT,
                result_id=1,
                start_x=92.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=10.0,
            ),
        ]
    )
    enriched = add_pre_shot_gk_context(actions, frames=frames)
    shot = enriched[enriched["type_id"] == SHOT].iloc[0]
    # Defending GK (home, frame x=3) re-projected to action-LTR: 105-3 = 102, near attacked goal.
    assert shot["pre_shot_gk_x"] == 102.0
    assert shot["pre_shot_gk_y"] == 34.0
    assert shot["pre_shot_gk_distance_to_goal"] == 3.0  # |105-102|
    assert abs(shot["pre_shot_gk_distance_to_shot"] - 10.0) < 1e-9  # |102-92|


def test_home_shot_unchanged_no_flip():
    frames = _frame_rows()
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=AWAY,
                player_id=50.0,
                type_id=GOALKICK,
                result_id=1,
                start_x=5.0,
                start_y=34.0,
                end_x=40.0,
                end_y=34.0,
                time_seconds=9.6,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=HOME,
                player_id=11.0,
                type_id=SHOT,
                result_id=1,
                start_x=92.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=10.0,
            ),
        ]
    )
    enriched = add_pre_shot_gk_context(actions, frames=frames)
    shot = enriched[enriched["type_id"] == SHOT].iloc[0]
    # Home shot: defending GK is away GK (frame x=102), no flip → stays 102, near attacked goal.
    assert shot["pre_shot_gk_x"] == 102.0
    assert shot["pre_shot_gk_distance_to_goal"] == 3.0


def test_frame_fallback_preserves_numeric_gk_id_dtype():
    """Regression (pandas >=3.0): the frame-fallback GK resolver must keep
    defending_gk_player_id as float64 for numeric-id input. pandas 3.0 stopped
    silently downcasting an object fill on .fillna, which left the column object and
    made the downstream float-vs-object GK id match find zero rows -> NaN GK position
    (CI 3.11/3.12 only; 3.10 uses pandas 2.x which downcast and masked it)."""
    frames = _frame_rows()
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=HOME,
                player_id=1.0,
                type_id=GOALKICK,
                result_id=1,
                start_x=5.0,
                start_y=34.0,
                end_x=40.0,
                end_y=34.0,
                time_seconds=9.6,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=AWAY,
                player_id=99.0,
                type_id=SHOT,
                result_id=1,
                start_x=92.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=10.0,
            ),
        ]
    )
    enriched = add_pre_shot_gk_context(actions, frames=frames)
    # goalkick is NOT a keeper type, so the GK is resolved via the frame fallback.
    assert enriched["defending_gk_player_id"].dtype == np.float64
    shot = enriched[enriched["type_id"] == SHOT].iloc[0]
    assert shot["defending_gk_player_id"] == 1.0
