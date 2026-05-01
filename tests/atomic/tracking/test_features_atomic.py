"""Tests for silly_kicks.atomic.tracking.features (atomic SPADL public surface).

Atomic-SPADL has columns (x, y, dx, dy) instead of (start_x, start_y, end_x, end_y).
The wrappers consume the same _kernels with atomic-shaped column reads.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


@pytest.fixture
def atomic_actions_and_frames():
    """1 atomic action at (50, 34), dx=10 dy=0 -> end at (60, 34).
    Defender at (52, 34) -> 2.0 m from start; 8.0 m from end."""
    actions = pd.DataFrame(
        {
            "action_id": [101],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "x": [50.0],
            "y": [34.0],
            "dx": [10.0],
            "dy": [0.0],
            "type_id": [0],
            "bodypart_id": [0],
        }
    )
    rows = [
        dict(
            game_id=1,
            period_id=1,
            frame_id=1000,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=11,
            team_id=1,
            is_ball=False,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            z=float("nan"),
            speed=2.0,
            speed_source="native",
            ball_state="alive",
            team_attacking_direction="ltr",
            confidence=None,
            visibility=None,
            source_provider="test",
        ),
        dict(
            game_id=1,
            period_id=1,
            frame_id=1000,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=22,
            team_id=2,
            is_ball=False,
            is_goalkeeper=False,
            x=52.0,
            y=34.0,
            z=float("nan"),
            speed=1.5,
            speed_source="native",
            ball_state="alive",
            team_attacking_direction="ltr",
            confidence=None,
            visibility=None,
            source_provider="test",
        ),
    ]
    frames = pd.DataFrame(rows)
    return actions, frames


def test_atomic_nearest_defender_distance(atomic_actions_and_frames):
    from silly_kicks.atomic.tracking.features import nearest_defender_distance

    actions, frames = atomic_actions_and_frames
    out = nearest_defender_distance(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_atomic_receiver_zone_density(atomic_actions_and_frames):
    """End at (60, 34); defender at (52, 34) -> 8m from end -> outside radius=5 -> 0."""
    from silly_kicks.atomic.tracking.features import receiver_zone_density

    actions, frames = atomic_actions_and_frames
    out = receiver_zone_density(actions, frames, radius=5.0)
    assert int(out.iloc[0]) == 0


def test_atomic_actor_speed(atomic_actions_and_frames):
    from silly_kicks.atomic.tracking.features import actor_speed

    actions, frames = atomic_actions_and_frames
    out = actor_speed(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_atomic_defenders_in_triangle_to_goal(atomic_actions_and_frames):
    """Defender at (52, 34) is inside triangle (50,34) -> goal posts."""
    from silly_kicks.atomic.tracking.features import defenders_in_triangle_to_goal

    actions, frames = atomic_actions_and_frames
    out = defenders_in_triangle_to_goal(actions, frames)
    assert int(out.iloc[0]) == 1


def test_atomic_zero_dx_dy_is_degenerate_density(atomic_actions_and_frames):
    """Atomic action with dx=dy=0 -> end == start -> density at start equals density-at-anchor.
    Documented degenerate case per spec."""
    from silly_kicks.atomic.tracking.features import receiver_zone_density

    actions, frames = atomic_actions_and_frames
    actions = actions.copy()
    actions["dx"] = 0.0
    actions["dy"] = 0.0
    out = receiver_zone_density(actions, frames, radius=5.0)
    # Defender at (52, 34) is 2m from (50, 34) -> 1 defender within radius=5
    assert int(out.iloc[0]) == 1
