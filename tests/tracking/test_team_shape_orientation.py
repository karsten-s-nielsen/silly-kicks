"""ADR-028: team-shape centroids / line-height emitted in action-LTR frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import add_team_shape

HOME, AWAY = 1, 2
PASS = spadlconfig.actiontype_id["pass"]


def _frames():
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=100,
        time_seconds=4.0,
        frame_rate=25.0,
        z=0.0,
        speed=0.0,
        speed_source="native",
        ball_state="alive",
        confidence=None,
        visibility=None,
        source_provider="synthetic",
        is_goalkeeper_source="native",
        is_goalkeeper=False,
    )
    rows = []
    for i in range(10):
        rows.append(
            dict(
                player_id=10 + i,
                team_id=HOME,
                is_ball=False,
                x=20.0 + i,
                y=10.0 + i * 5,
                team_attacking_direction="ltr",
            )
        )
    for i in range(10):
        rows.append(
            dict(
                player_id=60 + i,
                team_id=AWAY,
                is_ball=False,
                x=70.0 + i,
                y=10.0 + i * 5,
                team_attacking_direction="rtl",
            )
        )
    rows.append(dict(player_id=np.nan, team_id=np.nan, is_ball=True, x=50.0, y=34.0, team_attacking_direction=None))
    return pd.DataFrame([{**base, **r} for r in rows])


def test_away_action_centroids_reprojected_both_axes():
    frames = _frames()
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=AWAY,
                player_id=99.0,
                type_id=PASS,
                result_id=1,
                start_x=70.0,
                start_y=34.0,
                end_x=80.0,
                end_y=40.0,
                time_seconds=4.0,
            ),
        ]
    )
    out = add_team_shape(actions, frames, home_team_id=HOME).iloc[0]
    # Attacking team = AWAY. Frame centroid_x = mean(70..79)=74.5 -> action-LTR 105-74.5 = 30.5.
    assert abs(out["team_shape_centroid_x_attacking"] - 30.5) < 1e-9
    # Frame centroid_y = mean(10,15,...,55)=32.5 -> action-LTR 68-32.5 = 35.5.
    assert abs(out["team_shape_centroid_y_attacking"] - 35.5) < 1e-9
    # team_length is a span (max-min x) -> invariant: 9.0.
    assert abs(out["team_shape_team_length_attacking"] - 9.0) < 1e-9


def test_home_action_centroids_unchanged():
    frames = _frames()
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=HOME,
                player_id=11.0,
                type_id=PASS,
                result_id=1,
                start_x=70.0,
                start_y=34.0,
                end_x=80.0,
                end_y=40.0,
                time_seconds=4.0,
            ),
        ]
    )
    out = add_team_shape(actions, frames, home_team_id=HOME).iloc[0]
    # Attacking team = HOME, attacks 105, no flip. Frame centroid_x = mean(20..29)=24.5.
    assert abs(out["team_shape_centroid_x_attacking"] - 24.5) < 1e-9
    assert abs(out["team_shape_centroid_y_attacking"] - 32.5) < 1e-9
