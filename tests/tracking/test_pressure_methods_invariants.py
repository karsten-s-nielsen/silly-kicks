"""Cross-method physical invariants per spec section 8.3."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor


def _make_one_action_frame(actor_xy, defender_xy_v):
    """Build minimal actions+frames for a single (actor, single defender) test."""
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [actor_xy[0]],
            "start_y": [actor_xy[1]],
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
                "x": actor_xy[0],
                "y": actor_xy[1],
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
                "x": actor_xy[0],
                "y": actor_xy[1],
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
                "x": defender_xy_v[0],
                "y": defender_xy_v[1],
                "vx": defender_xy_v[2],
                "vy": defender_xy_v[3],
                "speed": math.hypot(defender_xy_v[2], defender_xy_v[3]),
                "source_provider": "synthetic",
            },
        ]
    )
    return actions, frames


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones"])
def test_monotone_decreasing_in_distance(method: str) -> None:
    """Position-only methods: pressure decreases as defender->actor distance increases."""
    pressures = []
    for d in [1.0, 2.0, 3.0, 5.0]:
        actions, frames = _make_one_action_frame((50.0, 34.0), (50.0 + d, 34.0, 0.0, 0.0))
        out = pressure_on_actor(actions, frames, method=method)
        pressures.append(float(out.iloc[0]))
    for i in range(1, len(pressures)):
        assert pressures[i] <= pressures[i - 1]


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones"])
def test_axially_symmetric(method: str) -> None:
    actions_pos, frames_pos = _make_one_action_frame((50.0, 34.0), (51.0, 34.0 + 1.0, 0.0, 0.0))
    actions_neg, frames_neg = _make_one_action_frame((50.0, 34.0), (51.0, 34.0 - 1.0, 0.0, 0.0))
    p_pos = pressure_on_actor(actions_pos, frames_pos, method=method).iloc[0]
    p_neg = pressure_on_actor(actions_neg, frames_neg, method=method).iloc[0]
    assert p_pos == pytest.approx(p_neg, rel=1e-9)


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_non_negative(method: str) -> None:
    actions, frames = _make_one_action_frame((50.0, 34.0), (52.0, 34.0, 3.0, 0.0))
    out = pressure_on_actor(actions, frames, method=method)
    assert (out.dropna() >= 0.0).all()


@pytest.mark.parametrize("method", ["link_zones", "bekkers_pi"])
def test_bounded_in_zero_one(method: str) -> None:
    actions, frames = _make_one_action_frame((50.0, 34.0), (50.5, 34.0, 3.0, 0.0))
    out = pressure_on_actor(actions, frames, method=method)
    assert ((out.dropna() >= 0.0) & (out.dropna() <= 1.0)).all()
