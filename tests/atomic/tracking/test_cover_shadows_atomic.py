"""Tests for TF-30 cover shadow atomic mirror."""

from __future__ import annotations

import pandas as pd

from tests.tracking._gk_test_helpers import _make_two_team_frame
from tests.tracking._goal_map_helpers import goal_map_for

#: ADR-055: `_make_two_team_frame` emits game 1 / period 1, teams {1, 2}, keeper at each end.
HOME_GOAL_MAP = goal_map_for({1: 0.0, 2: 105.0})


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
            goal_map=HOME_GOAL_MAP,
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

        xfns = cover_shadow_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
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

    def test_atomic_max_single_equals_standard(self, fitted_xt):
        """Atomic max_single == standard on a shared frame (pure delegation, PR-S65).

        Pins the 'atomic inherits by pure delegation' claim so a future atomic fork
        cannot silently drift from the standard computation.
        """
        import numpy as np

        from silly_kicks.atomic.tracking.features import add_cover_shadows as atomic_cs
        from silly_kicks.tracking.features import add_cover_shadows as std_cs

        atomic_actions, frames = _make_atomic_actions_and_frames()
        # Standard SPADL actions referencing the SAME frame/passer (x->start_x, dx->end_x).
        std_actions = pd.DataFrame(
            {
                "action_id": [0, 1],
                "game_id": [1, 1],
                "period_id": [1, 1],
                "time_seconds": [1.0, 999.0],
                "team_id": [2, 2],
                "type_id": [0, 0],
                "result_id": [1, 1],
                "start_x": [50.0, 50.0],
                "start_y": [34.0, 34.0],
                "end_x": [70.0, 70.0],
                "end_y": [34.0, 34.0],
                "bodypart_id": [0, 0],
                "player_id": [60, 60],
            }
        )
        std = std_cs(std_actions, frames, fitted_xt, goal_map=HOME_GOAL_MAP, detailed=False)
        atom = atomic_cs(atomic_actions, frames, fitted_xt, goal_map=HOME_GOAL_MAP, detailed=False)
        np.testing.assert_allclose(
            atom["max_single_defender_blocking_score"].to_numpy(),
            std["max_single_defender_blocking_score"].to_numpy(),
            rtol=1e-10,
            equal_nan=True,
        )
