"""Atomic mirror -- minimal pin for Link surface."""

from __future__ import annotations

import pandas as pd

from silly_kicks.atomic.tracking.features import pressure_on_actor


def test_atomic_link_runs() -> None:
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "x": [50.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
            "type_id": [0],
        }
    )
    # Frames are home-attacks-right (the convention convert_to_frames(output_convention="ltr")
    # emits), so the home team carries "ltr" and the away team "rtl". Without this column
    # acting_team_attacks_rtl cannot resolve a direction at all and returns an all-False flip
    # for the wrong reason (ADR-028).
    #
    # The acting team's OWN row is present deliberately: with only the away defender in the
    # frame the acting team ("home") is absent from the direction lookup, so the flip defaults
    # to False no matter what the column says -- measured, "ltr" and "rtl" both yield 0.39347,
    # i.e. the column would be inert and the orientation still untested. With this row the
    # column is load-bearing (an "rtl" home team yields 0.22120 instead). "home" attacks
    # left-to-right, so the correct resolution is "no flip" and the asserted value is unchanged.
    frames = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "home",
                "player_id": 10,
                "is_ball": False,
                "team_attacking_direction": "ltr",
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "away",
                "player_id": 100,
                "is_ball": False,
                "team_attacking_direction": "rtl",
                "x": 52.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
        ]
    )
    out = pressure_on_actor(actions, frames, method="link_zones")
    assert out.name == "pressure_on_actor__link_zones"
    assert 0.0 <= out.iloc[0] <= 1.0
