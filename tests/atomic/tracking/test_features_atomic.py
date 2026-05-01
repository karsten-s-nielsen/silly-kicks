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


# ---------------------------------------------------------------------------
# PR-S21 — atomic-side pre_shot_gk_* mirrors
# ---------------------------------------------------------------------------


@pytest.fixture
def atomic_shot_actions_and_frames_with_gk():
    """1 atomic SHOT at (90, 34), dx=dy=0; defending GK at (104, 34) in linked frame."""
    from silly_kicks.spadl import config as spadlconfig

    actions = pd.DataFrame(
        {
            "action_id": [301],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "x": [90.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
            "type_id": [spadlconfig.actiontype_id["shot"]],
            "bodypart_id": [0],
            "defending_gk_player_id": [99.0],
        }
    )
    rows = [
        dict(
            game_id=1,
            period_id=1,
            frame_id=2000,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=11,
            team_id=1,
            is_ball=False,
            is_goalkeeper=False,
            x=90.0,
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
            frame_id=2000,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=99,
            team_id=2,
            is_ball=False,
            is_goalkeeper=True,
            x=104.0,
            y=34.0,
            z=float("nan"),
            speed=0.5,
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


@pytest.mark.parametrize(
    "helper_name,expected_value",
    [
        ("pre_shot_gk_x", 104.0),
        ("pre_shot_gk_y", 34.0),
        ("pre_shot_gk_distance_to_goal", 1.0),
        ("pre_shot_gk_distance_to_shot", 14.0),
    ],
)
def test_atomic_pre_shot_gk_helpers_exact_values(
    atomic_shot_actions_and_frames_with_gk,
    helper_name,
    expected_value,
):
    from silly_kicks.atomic.tracking import features as atomic_features

    actions, frames = atomic_shot_actions_and_frames_with_gk
    helper = getattr(atomic_features, helper_name)
    s = helper(actions, frames)
    assert s.name == helper_name
    assert math.isclose(float(s.iloc[0]), expected_value, abs_tol=1e-6)


def test_atomic_pre_shot_gk_distance_to_shot_uses_atomic_x_y_anchor(
    atomic_shot_actions_and_frames_with_gk,
):
    """Atomic anchors on action.x/y (not start_x/start_y). Standard would use start_x — verify
    atomic explicitly uses x/y by changing only the atomic-side anchor and observing distance.
    """
    from silly_kicks.atomic.tracking.features import pre_shot_gk_distance_to_shot

    actions, frames = atomic_shot_actions_and_frames_with_gk
    actions = actions.copy()
    actions["x"] = 100.0  # move shot anchor to (100, 34); GK still at (104, 34) -> distance=4
    out = pre_shot_gk_distance_to_shot(actions, frames)
    assert math.isclose(float(out.iloc[0]), 4.0, abs_tol=1e-6)


def test_atomic_add_pre_shot_gk_position_emits_4_features_plus_4_provenance(
    atomic_shot_actions_and_frames_with_gk,
):
    from silly_kicks.atomic.tracking.features import add_pre_shot_gk_position

    actions, frames = atomic_shot_actions_and_frames_with_gk
    out = add_pre_shot_gk_position(actions, frames)
    expected = {
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    }
    assert expected.issubset(set(out.columns))


def test_atomic_add_pre_shot_gk_position_raises_on_missing_defending_gk_player_id_column(
    atomic_shot_actions_and_frames_with_gk,
):
    from silly_kicks.atomic.tracking.features import add_pre_shot_gk_position

    actions, frames = atomic_shot_actions_and_frames_with_gk
    actions = actions.drop(columns=["defending_gk_player_id"])
    with pytest.raises(ValueError, match="defending_gk_player_id"):
        add_pre_shot_gk_position(actions, frames)


def test_atomic_pre_shot_gk_default_xfns_count():
    from silly_kicks.atomic.tracking.features import atomic_pre_shot_gk_default_xfns

    assert len(atomic_pre_shot_gk_default_xfns) == 4


def test_atomic_pre_shot_gk_recognizes_shot_penalty():
    """Atomic uses {shot, shot_penalty} (no shot_freekick)."""
    from silly_kicks.atomic.tracking.features import pre_shot_gk_x
    from silly_kicks.spadl import config as spadlconfig

    rows_actions = pd.DataFrame(
        {
            "action_id": [401],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "x": [90.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
            "type_id": [spadlconfig.actiontype_id["shot_penalty"]],
            "bodypart_id": [0],
            "defending_gk_player_id": [99.0],
        }
    )
    rows = [
        dict(
            game_id=1,
            period_id=1,
            frame_id=3000,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=99,
            team_id=2,
            is_ball=False,
            is_goalkeeper=True,
            x=104.0,
            y=34.0,
            z=float("nan"),
            speed=0.5,
            speed_source="native",
            ball_state="alive",
            team_attacking_direction="ltr",
            confidence=None,
            visibility=None,
            source_provider="test",
        ),
    ]
    frames = pd.DataFrame(rows)
    out = pre_shot_gk_x(rows_actions, frames)
    assert math.isclose(float(out.iloc[0]), 104.0, abs_tol=1e-6)
