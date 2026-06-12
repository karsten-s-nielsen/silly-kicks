"""ADR-028: defensive_line_x / back_line_high_x emitted in action-LTR frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import add_defensive_line

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
    # Home back line near x=20 (defends x=0). Away back line near x=85 (defends x=105).
    for i, x in enumerate((18.0, 20.0, 22.0, 24.0)):
        rows.append(
            dict(player_id=10 + i, team_id=HOME, is_ball=False, x=x, y=20.0 + i * 8, team_attacking_direction="ltr")
        )
    for i, x in enumerate((81.0, 83.0, 85.0, 87.0)):
        rows.append(
            dict(player_id=60 + i, team_id=AWAY, is_ball=False, x=x, y=20.0 + i * 8, team_attacking_direction="rtl")
        )
    rows.append(dict(player_id=np.nan, team_id=np.nan, is_ball=True, x=50.0, y=34.0, team_attacking_direction=None))
    return pd.DataFrame([{**base, **r} for r in rows])


def test_away_action_defending_line_reprojected():
    frames = _frames()
    # Away team passes (LTR-normalized: away attacks x=105). Defending team = HOME (near x=20).
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
    out = add_defensive_line(actions, frames, home_team_id=HOME)
    # Home defenders at mean x=21 in frame; re-projected to action-LTR (away attacks 105): 105-21 = 84.
    assert abs(out["defensive_line_x"].iloc[0] - 84.0) < 1e-9
    # compactness_x (a span) is invariant: max(24)-min(18)=6 in frame, unchanged.
    assert abs(out["compactness_x"].iloc[0] - 6.0) < 1e-9


def test_home_action_defending_line_unchanged():
    frames = _frames()
    # Home team passes (attacks x=105). Defending team = AWAY (near x=85). No flip.
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
    out = add_defensive_line(actions, frames, home_team_id=HOME)
    # Away defenders mean x=84 in frame; home attacks 105, no flip -> stays 84.
    assert abs(out["defensive_line_x"].iloc[0] - 84.0) < 1e-9
