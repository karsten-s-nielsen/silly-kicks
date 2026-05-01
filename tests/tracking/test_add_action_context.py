"""Tests for add_action_context aggregator: provenance columns, NaN safety, dtypes."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def actions_and_frames_aggregator():
    """2 linked actions + 1 unlinked action (period 1, t=1000s --- no frame at 1000s)."""
    actions = pd.DataFrame(
        {
            "action_id": [101, 102, 999],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 20.0, 1000.0],
            "team_id": [1, 1, 1],
            "player_id": [11, 11, 11],
            "start_x": [50.0, 60.0, 50.0],
            "start_y": [34.0, 30.0, 34.0],
            "end_x": [55.0, 65.0, 55.0],
            "end_y": [34.0, 30.0, 34.0],
        }
    )
    rows = []
    for fid, t in [(1000, 10.0), (2000, 20.0)]:
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
                x=50.0 if fid == 1000 else 60.0,
                y=34.0 if fid == 1000 else 30.0,
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
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=22,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=(50.0 if fid == 1000 else 60.0) + 5.0,
                y=34.0 if fid == 1000 else 30.0,
                z=float("nan"),
                speed=1.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="test",
            )
        )
    frames = pd.DataFrame(rows)
    return actions, frames


def test_add_action_context_returns_input_plus_8_columns(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    new_cols = set(enriched.columns) - set(actions.columns)
    expected = {
        "nearest_defender_distance",
        "actor_speed",
        "receiver_zone_density",
        "defenders_in_triangle_to_goal",
        "frame_id",
        "time_offset_seconds",
        "link_quality_score",
        "n_candidate_frames",
    }
    assert expected.issubset(new_cols)


def test_add_action_context_unlinked_action_has_nan_features(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    unlinked = enriched[enriched["action_id"] == 999].iloc[0]
    assert pd.isna(unlinked["nearest_defender_distance"])
    assert pd.isna(unlinked["actor_speed"])
    assert pd.isna(unlinked["frame_id"])


def test_add_action_context_linked_action_has_features(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    linked = enriched[enriched["action_id"] == 101].iloc[0]
    assert linked["nearest_defender_distance"] == 5.0
    assert linked["actor_speed"] == 2.0


def test_add_action_context_dtypes(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    assert enriched["nearest_defender_distance"].dtype == "float64"
    assert enriched["actor_speed"].dtype == "float64"
    assert enriched["receiver_zone_density"].dtype.name == "Int64"
    assert enriched["defenders_in_triangle_to_goal"].dtype.name == "Int64"


def test_add_action_context_is_nan_safe_decorated():
    """ADR-003 contract: add_action_context is in the auto-discovered NaN-safety registry."""
    from silly_kicks._nan_safety import is_nan_safe_enrichment
    from silly_kicks.tracking.features import add_action_context

    assert is_nan_safe_enrichment(add_action_context) is True
