"""Tests for TF-30 cover shadow atomic mirror."""

from __future__ import annotations

import pandas as pd

from tests.tracking._gk_test_helpers import _make_two_team_frame


def _make_atomic_actions_and_frames():
    """Build atomic actions + frames for cover shadow testing."""
    frame = _make_two_team_frame(
        home_positions=[
            (40.0, 34.0),
            (20.0, 15.0),
            (25.0, 55.0),
            (30.0, 10.0),
            (35.0, 60.0),
            (40.0, 20.0),
            (45.0, 50.0),
            (15.0, 34.0),
            (10.0, 15.0),
            (10.0, 55.0),
        ],
        away_positions=[
            (50.0, 34.0),
            (70.0, 34.0),
            (75.0, 20.0),
            (80.0, 50.0),
            (55.0, 10.0),
            (55.0, 58.0),
            (60.0, 20.0),
            (65.0, 50.0),
            (85.0, 30.0),
            (90.0, 40.0),
        ],
    )
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 999.0],
            "team_id": [2, 2],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "x": [50.0, 50.0],
            "y": [34.0, 34.0],
            "dx": [20.0, 20.0],
            "dy": [0.0, 0.0],
            "bodypart_id": [0, 0],
            "player_id": [60, 60],
        }
    )
    return actions, frame


class TestAtomicCoverShadows:
    """Atomic mirror for cover shadow features."""

    def test_add_cover_shadows_runs(self, fitted_xt):
        """Atomic add_cover_shadows produces 5 columns."""
        from silly_kicks.atomic.tracking.features import add_cover_shadows

        actions, frames = _make_atomic_actions_and_frames()
        result = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
        )
        for col in [
            "n_blocked_receivers",
            "n_potential_receivers",
            "blocking_score",
            "blocked_threat_fraction",
            "max_single_defender_blocking_score",
        ]:
            assert col in result.columns

    def test_cover_shadow_xfns_column_count(self, fitted_xt):
        """Atomic xfns factory produces 15 columns."""
        from silly_kicks.atomic.tracking.features import cover_shadow_xfns

        xfns = cover_shadow_xfns(fitted_xt, home_team_id=1)
        assert len(xfns) == 1
        transformer = xfns[0]
        assert getattr(transformer, "_frame_aware", False) is True

        dummy = pd.DataFrame(
            {
                "game_id": [1] * 3,
                "action_id": [0, 1, 2],
                "period_id": [1] * 3,
                "time_seconds": [1.0, 2.0, 3.0],
                "team_id": [1] * 3,
                "player_id": [10, 11, 12],
                "x": [50.0] * 3,
                "y": [34.0] * 3,
                "dx": [10.0] * 3,
                "dy": [0.0] * 3,
                "type_id": [0] * 3,
                "result_id": [1] * 3,
                "bodypart_id": [0] * 3,
            }
        )
        states = [dummy, dummy, dummy]
        result = transformer(states, None)
        assert result.shape[1] == 15
