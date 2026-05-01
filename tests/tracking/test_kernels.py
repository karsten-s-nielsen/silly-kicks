"""Tests for silly_kicks.tracking._kernels --- pure-compute analytical-truth fixtures.

Loop 6 covers: nearest_defender_distance, actor_speed, receiver_zone_density,
defenders_in_triangle_to_goal kernels. Inputs are tiny in-memory ActionFrameContext
objects with known geometric ground truth.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from silly_kicks.tracking.feature_framework import ActionFrameContext


@pytest.fixture
def ctx_three_defenders():
    """Action at (50, 34); 3 opposite-team defenders at distances 2.0, 5.0, 10.0 m east.

    nearest_defender_distance kernel must return 2.0.
    receiver_zone_density at radius=5.0 must return 2 (the defenders 5m and 0m from end=(60,34)).
    """
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [60.0],
            "end_y": [34.0],
            "team_id": [1],
            "player_id": [11],
        }
    )
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [1000]})
    actor_rows = pd.DataFrame(
        {
            "action_id": [1],
            "x": [50.0],
            "y": [34.0],
            "speed": [1.5],
        }
    )
    opposite = pd.DataFrame(
        {
            "action_id": [1, 1, 1],
            "x": [52.0, 55.0, 60.0],
            "y": [34.0, 34.0, 34.0],
            "team_id_frame": [2, 2, 2],
        }
    )
    return ActionFrameContext(
        actions=actions, pointers=pointers, actor_rows=actor_rows, opposite_rows_per_action=opposite
    )


def test_nearest_defender_distance_kernel(ctx_three_defenders):
    from silly_kicks.tracking._kernels import _nearest_defender_distance

    result = _nearest_defender_distance(
        ctx_three_defenders.actions["start_x"],
        ctx_three_defenders.actions["start_y"],
        ctx_three_defenders,
    )
    assert math.isclose(result.iloc[0], 2.0, abs_tol=1e-9)


def test_receiver_zone_density_kernel(ctx_three_defenders):
    from silly_kicks.tracking._kernels import _receiver_zone_density

    # End at (60, 34); defenders at x=52,55,60 -> dists 8, 5, 0. radius=5 -> count=2.
    result = _receiver_zone_density(
        ctx_three_defenders.actions["end_x"],
        ctx_three_defenders.actions["end_y"],
        ctx_three_defenders,
        radius=5.0,
    )
    assert int(result.iloc[0]) == 2


def test_actor_speed_kernel(ctx_three_defenders):
    from silly_kicks.tracking._kernels import _actor_speed_from_ctx

    result = _actor_speed_from_ctx(ctx_three_defenders)
    assert math.isclose(result.iloc[0], 1.5, abs_tol=1e-9)


def test_defenders_in_triangle_to_goal_kernel():
    """Defender at (80, 34) is between (50, 34) and the goal mouth -> IN.
    Defender at (60, 60) is outside (above goal-mouth y range) -> OUT.
    """
    from silly_kicks.tracking._kernels import _defenders_in_triangle_to_goal

    actions = pd.DataFrame(
        {
            "action_id": [1],
            "start_x": [50.0],
            "start_y": [34.0],
            "team_id": [1],
            "player_id": [11],
        }
    )
    actor_rows = pd.DataFrame({"action_id": [1], "x": [50.0], "y": [34.0], "speed": [1.5]})
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [1000]})
    opposite = pd.DataFrame(
        {
            "action_id": [1, 1],
            "x": [80.0, 60.0],
            "y": [34.0, 60.0],
            "team_id_frame": [2, 2],
        }
    )
    ctx = ActionFrameContext(
        actions=actions, pointers=pointers, actor_rows=actor_rows, opposite_rows_per_action=opposite
    )
    result = _defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)
    assert int(result.iloc[0]) == 1


def test_unlinked_action_returns_nan():
    """Action with no actor_row and no opposite_rows -> NaN feature output."""
    from silly_kicks.tracking._kernels import _nearest_defender_distance

    actions = pd.DataFrame(
        {
            "action_id": [1],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [60.0],
            "end_y": [34.0],
            "team_id": [1],
            "player_id": [11],
        }
    )
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [pd.NA]}, dtype="object")
    actor_rows = pd.DataFrame({"action_id": [1], "x": [float("nan")], "y": [float("nan")], "speed": [float("nan")]})
    opposite = pd.DataFrame({"action_id": [], "x": [], "y": [], "team_id_frame": []}, dtype="float64")
    ctx = ActionFrameContext(
        actions=actions, pointers=pointers, actor_rows=actor_rows, opposite_rows_per_action=opposite
    )
    result = _nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)
    assert pd.isna(result.iloc[0])


def test_receiver_zone_density_zero_when_no_defenders_in_radius():
    """Linked action with all defenders far from end -> count = 0 (NOT NaN)."""
    from silly_kicks.tracking._kernels import _receiver_zone_density

    actions = pd.DataFrame(
        {
            "action_id": [1],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [60.0],
            "end_y": [34.0],
            "team_id": [1],
            "player_id": [11],
        }
    )
    actor_rows = pd.DataFrame({"action_id": [1], "x": [50.0], "y": [34.0], "speed": [1.5]})
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [1000]})
    # Defender far away: dist = 100 m from end (60, 34)
    opposite = pd.DataFrame({"action_id": [1], "x": [160.0], "y": [34.0], "team_id_frame": [2]})
    ctx = ActionFrameContext(
        actions=actions, pointers=pointers, actor_rows=actor_rows, opposite_rows_per_action=opposite
    )
    result = _receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=5.0)
    assert int(result.iloc[0]) == 0
