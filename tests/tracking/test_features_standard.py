"""Tests for silly_kicks.tracking.features (standard SPADL public surface)."""

from __future__ import annotations

import math

import pandas as pd
import pytest


@pytest.fixture
def actions_and_frames_for_features():
    """1 action at (50,34) ending at (60,34); actor and 3 defenders + ball in 1 frame."""
    actions = pd.DataFrame(
        {
            "action_id": [101],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [60.0],
            "end_y": [34.0],
        }
    )
    rows = []
    fid, t = 1000, 10.0
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=fid,
            time_seconds=t,
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
        )
    )
    for x in (52.0, 55.0, 60.0):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=20 + int(x),
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=x,
                y=34.0,
                z=float("nan"),
                speed=1.5,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="test",
            )
        )
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=fid,
            time_seconds=t,
            frame_rate=25.0,
            player_id=float("nan"),
            team_id=float("nan"),
            is_ball=True,
            is_goalkeeper=False,
            x=52.5,
            y=34.0,
            z=0.0,
            speed=5.0,
            speed_source="native",
            ball_state="alive",
            team_attacking_direction=None,
            confidence=None,
            visibility=None,
            source_provider="test",
        )
    )
    frames = pd.DataFrame(rows)
    return actions, frames


def test_nearest_defender_distance_standard(actions_and_frames_for_features):
    from silly_kicks.tracking.features import nearest_defender_distance

    actions, frames = actions_and_frames_for_features
    out = nearest_defender_distance(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_actor_speed_standard(actions_and_frames_for_features):
    from silly_kicks.tracking.features import actor_speed

    actions, frames = actions_and_frames_for_features
    out = actor_speed(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_receiver_zone_density_standard(actions_and_frames_for_features):
    """End at (60, 34); defenders at 52,55,60 -> 8,5,0 from end. radius=5 -> count=2."""
    from silly_kicks.tracking.features import receiver_zone_density

    actions, frames = actions_and_frames_for_features
    out = receiver_zone_density(actions, frames, radius=5.0)
    assert int(out.iloc[0]) == 2


def test_defenders_in_triangle_to_goal_standard(actions_and_frames_for_features):
    """All 3 defenders are on y=34, x in [52,60] -> all in triangle to goal."""
    from silly_kicks.tracking.features import defenders_in_triangle_to_goal

    actions, frames = actions_and_frames_for_features
    out = defenders_in_triangle_to_goal(actions, frames)
    assert int(out.iloc[0]) == 3


def test_tracking_default_xfns_count():
    from silly_kicks.tracking.features import tracking_default_xfns

    assert len(tracking_default_xfns) == 4
