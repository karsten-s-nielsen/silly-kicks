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
    frames = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "away",
                "player_id": 100,
                "is_ball": False,
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
