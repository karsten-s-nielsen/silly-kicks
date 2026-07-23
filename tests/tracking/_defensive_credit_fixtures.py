"""Synthetic (actions, frames) builders for TF-51 rule tests. Import from here; never duplicate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_action(
    *,
    action_id=1,
    type_name="shot",
    result_name="fail",
    team_id=10,
    player_id=100,
    start_x=95.0,
    start_y=34.0,
    end_x=105.0,
    end_y=34.0,
    period_id=1,
    time_seconds=50.0,
    game_id="g1",
    **extra,
) -> pd.DataFrame:
    from silly_kicks.spadl import config as spadlconfig

    row = {
        "game_id": game_id,
        "period_id": period_id,
        "action_id": action_id,
        "time_seconds": time_seconds,
        "team_id": team_id,
        "player_id": player_id,
        "type_id": spadlconfig.actiontype_id[type_name],
        "result_id": spadlconfig.result_id[result_name],
        "bodypart_id": spadlconfig.bodypart_id["foot"],
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
    }
    row.update(extra)
    return pd.DataFrame([row])


def _direction(team_id, home_team_id) -> str:
    """Home team attacks x=105 in convert_to_frames coords -> 'ltr'; away attacks x=0 -> 'rtl'."""
    return "ltr" if team_id == home_team_id else "rtl"


def frame_with_defender(
    *,
    action_time=50.0,
    period_id=1,
    game_id="g1",
    acting_team_id=10,
    defender_team_id=20,
    defender_x=96.0,
    defender_y=34.0,
    frame_id=500,
    home_team_id=10,
) -> pd.DataFrame:
    """One frame at the action time: a single defender (opponent) + one acting-team player + the ball.

    Home team attacks x=105 (convert_to_frames convention); ``team_attacking_direction`` is the
    ``"ltr"``/``"rtl"`` string ``acting_team_attacks_rtl`` reads. The acting-team player row is
    REQUIRED so the away-action flip resolves (the direction lookup keys on the ACTING team's rows;
    a defender-only frame would default an away action to no-flip). For a home action (acting == home)
    action-LTR == frame coords (no flip); for an away action place the defender in FRAME coords.
    """
    common = dict(
        game_id=game_id,
        period_id=period_id,
        frame_id=frame_id,
        time_seconds=action_time,
        vx=0.0,
        vy=0.0,
        is_goalkeeper=False,
        home_team_id=home_team_id,
        source_provider="test",
    )
    rows = [
        # defender (opponent) -- row 0, so `frames.iloc[[0]]` in tests is the defender
        {
            **common,
            "team_id": defender_team_id,
            "player_id": 900,
            "x": defender_x,
            "y": defender_y,
            "is_ball": False,
            "team_attacking_direction": _direction(defender_team_id, home_team_id),
        },
        # acting-team player (far from any anchor; opponent-filter excludes it, but its direction
        # is what acting_team_attacks_rtl reads to decide the per-action flip)
        {
            **common,
            "team_id": acting_team_id,
            "player_id": 800,
            "x": 52.0,
            "y": 34.0,
            "is_ball": False,
            "team_attacking_direction": _direction(acting_team_id, home_team_id),
        },
        # ball
        {
            **common,
            "team_id": np.nan,
            "player_id": np.nan,
            "x": 100.0,
            "y": 34.0,
            "is_ball": True,
            "team_attacking_direction": _direction(acting_team_id, home_team_id),
        },
    ]
    return pd.DataFrame(rows)
