"""ADR-028: ghost_gk_x/y emitted in action-LTR frame (defended goal at x=105)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import add_ghost_gk

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]


def _frames_two_shots():
    base = dict(
        game_id=1,
        period_id=1,
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
    rows = []
    for fid, t in ((100, 4.0), (200, 8.0)):
        rows += [
            dict(
                frame_id=fid,
                time_seconds=t,
                player_id=1,
                team_id=HOME,
                is_ball=False,
                is_goalkeeper=True,
                x=4.0,
                y=34.0,
                team_attacking_direction="ltr",
            ),
            dict(
                frame_id=fid,
                time_seconds=t,
                player_id=50,
                team_id=AWAY,
                is_ball=False,
                is_goalkeeper=True,
                x=101.0,
                y=34.0,
                team_attacking_direction="rtl",
            ),
            dict(
                frame_id=fid,
                time_seconds=t,
                player_id=11,
                team_id=HOME,
                is_ball=False,
                is_goalkeeper=False,
                x=40.0,
                y=30.0,
                team_attacking_direction="ltr",
            ),
            dict(
                frame_id=fid,
                time_seconds=t,
                player_id=61,
                team_id=AWAY,
                is_ball=False,
                is_goalkeeper=False,
                x=65.0,
                y=38.0,
                team_attacking_direction="rtl",
            ),
            dict(
                frame_id=fid,
                time_seconds=t,
                player_id=np.nan,
                team_id=np.nan,
                is_ball=True,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                team_attacking_direction=None,
            ),
        ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_ghost_gk_x_is_action_ltr_near_attacked_goal():
    frames = _frames_two_shots()
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=HOME,
                player_id=11.0,
                type_id=SHOT,
                result_id=1,
                start_x=90.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=4.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=AWAY,
                player_id=61.0,
                type_id=SHOT,
                result_id=1,
                start_x=90.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=8.0,
            ),
        ]
    )
    out = add_ghost_gk(actions, frames, home_team_id=HOME)
    gx = out["ghost_gk_x"].to_numpy()
    # In action-LTR the defended goal is x=105; a ghost defending GK sits near it (x >> 50)
    # for BOTH home and away shots (no own-goal-end bimodality).
    assert np.all(gx[np.isfinite(gx)] > 70.0), f"ghost_gk_x not near attacked goal: {gx}"
    assert np.isfinite(gx).any(), "expected at least one finite ghost_gk_x"
