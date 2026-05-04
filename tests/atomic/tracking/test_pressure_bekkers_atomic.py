"""Atomic mirror -- minimal pin for Bekkers surface."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.tracking.features import pressure_on_actor


def test_atomic_bekkers_runs() -> None:
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
                "team_id": "home",
                "player_id": 10,
                "is_ball": False,
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
                "team_id": None,
                "player_id": None,
                "is_ball": True,
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
                "x": 52.0,
                "y": 34.0,
                "vx": -3.0,
                "vy": 0.0,
                "speed": 3.0,
                "source_provider": "synthetic",
            },
        ]
    )
    out = pressure_on_actor(actions, frames, method="bekkers_pi")
    assert out.name == "pressure_on_actor__bekkers_pi"
    assert 0.0 <= out.iloc[0] <= 1.0


def test_atomic_bekkers_missing_velocities_raises() -> None:
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
    frames = pd.DataFrame()
    with pytest.raises(ValueError, match="missing velocity columns"):
        pressure_on_actor(actions, frames, method="bekkers_pi")
