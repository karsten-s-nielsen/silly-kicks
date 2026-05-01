"""Aggregator + provenance for atomic SPADL."""

from __future__ import annotations

import pandas as pd


def test_atomic_add_action_context_smoke():
    from silly_kicks.atomic.tracking.features import add_action_context

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
    frames = pd.DataFrame(
        [
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
    )
    enriched = add_action_context(actions, frames)
    assert "nearest_defender_distance" in enriched.columns
    assert enriched["nearest_defender_distance"].iloc[0] == 2.0
