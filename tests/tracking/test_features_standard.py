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


# ---------------------------------------------------------------------------
# PR-S21 — pre_shot_gk_* Series helpers + add_pre_shot_gk_position aggregator
# ---------------------------------------------------------------------------


@pytest.fixture
def shot_actions_and_frames_with_gk():
    """1 SHOT at (90, 34); defending GK (team 2, player 99) at (104, 34) in linked frame.
    Pre-engagement events not modeled — defending_gk_player_id is set directly on the action.
    """
    from silly_kicks.spadl import config as spadlconfig

    actions = pd.DataFrame(
        {
            "action_id": [201],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "type_id": [spadlconfig.actiontype_id["shot"]],
            "start_x": [90.0],
            "start_y": [34.0],
            "end_x": [105.0],
            "end_y": [34.0],
            "defending_gk_player_id": [99.0],
        }
    )
    rows = []
    fid, t = 2000, 10.0
    # Actor (team 1)
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
        )
    )
    # Defending GK (team 2, player_id=99) at known coords
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=fid,
            time_seconds=t,
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
        )
    )
    # Ball
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
            x=90.0,
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


@pytest.mark.parametrize(
    "helper_name,expected_value",
    [
        ("pre_shot_gk_x", 104.0),
        ("pre_shot_gk_y", 34.0),
        ("pre_shot_gk_distance_to_goal", 1.0),  # |105-104| = 1
        ("pre_shot_gk_distance_to_shot", 14.0),  # |104-90| = 14
    ],
)
def test_pre_shot_gk_helpers_emit_named_series_with_correct_value(
    shot_actions_and_frames_with_gk,
    helper_name,
    expected_value,
):
    from silly_kicks.tracking import features as track_features

    actions, frames = shot_actions_and_frames_with_gk
    helper = getattr(track_features, helper_name)
    s = helper(actions, frames)
    assert isinstance(s, pd.Series)
    assert s.name == helper_name
    assert len(s) == len(actions)
    assert math.isclose(float(s.iloc[0]), expected_value, abs_tol=1e-6)


def test_add_pre_shot_gk_position_emits_4_features_plus_4_provenance(shot_actions_and_frames_with_gk):
    from silly_kicks.tracking.features import add_pre_shot_gk_position

    actions, frames = shot_actions_and_frames_with_gk
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


def test_add_pre_shot_gk_position_raises_on_missing_defending_gk_player_id_column(
    shot_actions_and_frames_with_gk,
):
    from silly_kicks.tracking.features import add_pre_shot_gk_position

    actions, frames = shot_actions_and_frames_with_gk
    actions = actions.drop(columns=["defending_gk_player_id"])
    with pytest.raises(ValueError, match="defending_gk_player_id"):
        add_pre_shot_gk_position(actions, frames)


def test_add_pre_shot_gk_position_provenance_columns_match_link_actions_to_frames(
    shot_actions_and_frames_with_gk,
):
    from silly_kicks.tracking.features import add_pre_shot_gk_position
    from silly_kicks.tracking.utils import link_actions_to_frames

    actions, frames = shot_actions_and_frames_with_gk
    out = add_pre_shot_gk_position(actions, frames)
    pointers, _ = link_actions_to_frames(actions, frames)
    out_indexed = out.set_index("action_id")
    pointers_indexed = pointers.set_index("action_id")
    assert (out_indexed.loc[pointers_indexed.index, "frame_id"].astype("Int64") == pointers_indexed["frame_id"]).all()


def test_pre_shot_gk_default_xfns_count():
    from silly_kicks.tracking.features import pre_shot_gk_default_xfns

    assert len(pre_shot_gk_default_xfns) == 4
