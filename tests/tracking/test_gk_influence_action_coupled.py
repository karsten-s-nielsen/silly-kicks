"""Tests for TF-15 GK influence action-coupled features."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.tracking._gk_test_helpers import _make_two_team_frame
from tests.tracking._goal_map_helpers import goal_map_for

#: ADR-055 replaced ``home_team_id=1`` at this file's re-keyed call sites. Its frames carry
#: game 1 / period 1 with teams {1, 2} and each keeper at its own end, so this states exactly
#: what ``home_team_id=1`` meant and matches what ``resolve_defended_goals`` derives there.
HOME_GOAL_MAP = goal_map_for({1: 0.0, 2: 105.0})

# fitted_xt fixture inherited from tests/conftest.py


def _make_actions_and_frames():
    """Build an action+frame pair for action-coupled testing.

    Uses 10 outfield players per team (realistic 11v11) with non-zero
    velocities. The action team_id=2 corresponds to the away team so
    the defending GK (player_id=1, team_id=1) is at home goal (x=3).
    """
    frame = _make_two_team_frame(
        home_positions=[
            (20, 15),
            (25, 25),
            (30, 40),
            (35, 55),
            (45, 10),
            (50, 30),
            (55, 45),
            (60, 55),
            (70, 30),
            (75, 40),
        ],
        away_positions=[
            (80, 20),
            (85, 30),
            (75, 40),
            (70, 55),
            (60, 10),
            (55, 30),
            (50, 45),
            (45, 55),
            (35, 30),
            (30, 40),
        ],
        home_velocities=[
            (1.5, 0.0),
            (0.0, 2.0),
            (-1.0, 0.5),
            (2.0, -1.0),
            (0.0, 0.0),
            (3.0, 1.0),
            (-0.5, -2.0),
            (0.0, 0.0),
            (4.0, 0.0),
            (-1.0, 1.5),
        ],
        away_velocities=[
            (-2.0, 0.5),
            (0.0, -1.0),
            (1.0, 0.0),
            (-3.0, 1.0),
            (0.0, 0.0),
            (2.0, -2.0),
            (-1.0, 0.0),
            (0.0, 3.0),
            (5.0, 0.0),
            (-0.5, -1.0),
        ],
    )
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 999.0],  # second action unlinked
            "team_id": [2, 2],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "start_x": [80.0, 50.0],
            "start_y": [30.0, 34.0],
            "end_x": [85.0, 55.0],
            "end_y": [35.0, 40.0],
            "bodypart_id": [0, 0],
            "player_id": [60, 61],
        }
    )
    return actions, frame


class TestPerSeriesHelpers:
    """T-7: Per-Series helper functions."""

    def test_known_gk_produces_scalar(self, fitted_xt):
        from silly_kicks.tracking.features import gk_pitch_control_share_weighted

        actions, frames = _make_actions_and_frames()
        result = gk_pitch_control_share_weighted(
            actions,
            frames,
            fitted_xt,
            goal_map=HOME_GOAL_MAP,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        # First action linked, should have valid value
        assert not np.isnan(result.iloc[0])

    def test_unlinked_action_nan(self, fitted_xt):
        from silly_kicks.tracking.features import gk_reachable_area_m2

        actions, frames = _make_actions_and_frames()
        result = gk_reachable_area_m2(
            actions,
            frames,
            fitted_xt,
            goal_map=HOME_GOAL_MAP,
        )
        # Second action time_seconds=999 -> no matching frame -> NaN
        assert np.isnan(result.iloc[1])

    def test_nan_team_id_nan(self, fitted_xt):
        from silly_kicks.tracking.features import gk_closing_time_min_s

        actions, frames = _make_actions_and_frames()
        actions.loc[0, "team_id"] = np.nan
        result = gk_closing_time_min_s(
            actions,
            frames,
            goal_map=HOME_GOAL_MAP,
        )
        assert np.isnan(result.iloc[0])

    def test_introspection_all_nan(self, fitted_xt):
        from silly_kicks.tracking.features import gk_pitch_control_share_weighted

        actions, _ = _make_actions_and_frames()
        result = gk_pitch_control_share_weighted(
            actions,
            None,
            fitted_xt,
            goal_map=HOME_GOAL_MAP,
        )
        assert result.isna().all()
        assert result.name == "gk_pitch_control_share_weighted"


class TestAggregator:
    """T-8: add_gk_influence aggregator."""

    def test_correct_column_set(self, fitted_xt):
        from silly_kicks.tracking.features import add_gk_influence

        actions, frames = _make_actions_and_frames()
        result = add_gk_influence(
            actions,
            frames,
            fitted_xt,
            goal_map=HOME_GOAL_MAP,
        )
        expected_cols = {
            "gk_pitch_control_share_weighted",
            "gk_reachable_area_m2",
            "gk_closing_time_min_s__six_yard_box",
            "gk_closing_time_mean_s__six_yard_box",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_idempotent_provenance(self, fitted_xt):
        """Provenance columns skipped if already present."""
        from silly_kicks.tracking.features import add_action_context, add_gk_influence

        actions, frames = _make_actions_and_frames()
        enriched = add_action_context(actions, frames)
        result = add_gk_influence(enriched, frames, fitted_xt, goal_map=HOME_GOAL_MAP)
        # Should not duplicate provenance columns
        assert result.columns.duplicated().sum() == 0

    def test_nan_safe_decorator(self):
        from silly_kicks._nan_safety import is_nan_safe_enrichment
        from silly_kicks.tracking.features import add_gk_influence

        assert is_nan_safe_enrichment(add_gk_influence)

    def test_additional_zones(self, fitted_xt):
        from silly_kicks.tracking.features import add_gk_influence

        actions, frames = _make_actions_and_frames()
        result = add_gk_influence(
            actions,
            frames,
            fitted_xt,
            goal_map=HOME_GOAL_MAP,
            zone_names=["six_yard_box", "near_post", "far_post"],
        )
        assert "gk_closing_time_min_s__near_post" in result.columns
        assert "gk_closing_time_mean_s__far_post" in result.columns


# === T-9: xfns factory ===


class TestXfnsFactory:
    """T-9: gk_influence_xfns factory."""

    def test_returns_frame_aware_transformer(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
        assert len(xfns) == 1
        assert hasattr(xfns[0], "_frame_aware")
        assert xfns[0]._frame_aware is True

    def test_introspection_column_names(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
        transformer = xfns[0]

        # Create dummy states (3 slots of 2 actions)
        dummy_actions = pd.DataFrame(
            {
                "action_id": [0, 1],
                "game_id": [1, 1],
                "period_id": [1, 1],
                "time_seconds": [1.0, 2.0],
                "team_id": [1, 1],
                "type_id": [0, 0],
                "result_id": [1, 1],
                "start_x": [50.0, 60.0],
                "start_y": [34.0, 34.0],
                "end_x": [55.0, 65.0],
                "end_y": [34.0, 34.0],
                "bodypart_id": [0, 0],
                "player_id": [10, 11],
            }
        )
        states = [dummy_actions.copy() for _ in range(3)]

        result = transformer(states, None)  # frames=None -> introspection
        assert result.isna().all().all()
        # 4 cols x 3 states = 12 columns
        assert len(result.columns) == 12

    def test_full_mode_column_count(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        actions, frames = _make_actions_and_frames()
        xfns = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
        transformer = xfns[0]
        states = [actions.copy() for _ in range(3)]
        result = transformer(states, frames)
        assert len(result.columns) == 12

    def test_column_naming_convention(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
        states = [
            pd.DataFrame(
                {
                    "action_id": [0],
                    "team_id": [1],
                    "period_id": [1],
                    "time_seconds": [1.0],
                    "game_id": [1],
                    "type_id": [0],
                    "result_id": [1],
                    "start_x": [50.0],
                    "start_y": [34.0],
                    "end_x": [55.0],
                    "end_y": [34.0],
                    "bodypart_id": [0],
                    "player_id": [10],
                }
            )
            for _ in range(3)
        ]
        result = xfns[0](states, None)
        for col in result.columns:
            assert col.endswith(("_a0", "_a1", "_a2"))

    def test_cache_avoids_redundant_calls(self, fitted_xt):
        """Frame precomputation: same frame_id shared across states -> single call."""
        from silly_kicks.tracking.features import gk_influence_xfns

        # Two actions sharing same frame
        actions = pd.DataFrame(
            {
                "action_id": [0, 1],
                "game_id": [1, 1],
                "period_id": [1, 1],
                "time_seconds": [1.0, 1.0],
                "team_id": [2, 2],
                "type_id": [0, 0],
                "result_id": [1, 1],
                "start_x": [80.0, 82.0],
                "start_y": [30.0, 35.0],
                "end_x": [85.0, 87.0],
                "end_y": [35.0, 40.0],
                "bodypart_id": [0, 0],
                "player_id": [60, 61],
            }
        )
        _, frames = _make_actions_and_frames()
        xfns = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
        states = [actions.copy() for _ in range(3)]

        call_count = [0]

        from silly_kicks.tracking import _gk_influence

        original_fn = _gk_influence.compute_gk_influence

        def counting_wrapper(*args, **kwargs):
            call_count[0] += 1
            return original_fn(*args, **kwargs)

        with patch.object(
            _gk_influence,
            "compute_gk_influence",
            side_effect=counting_wrapper,
        ):
            xfns[0](states, frames)

        # Both actions share same frame_id=1, team_id=2 -> exactly 1 call
        assert call_count[0] == 1

    def test_different_params_no_stale_cache(self, fitted_xt):
        """Different method produces different results (no stale cache)."""
        from silly_kicks.tracking.features import gk_influence_xfns

        actions, frames = _make_actions_and_frames()
        xfns_s = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP, method="spearman")
        xfns_v = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP, method="voronoi")

        states = [actions.copy() for _ in range(3)]
        result_s = xfns_s[0](states, frames)
        result_v = xfns_v[0](states, frames)

        # Different methods -> different share values (at least for non-NaN)
        share_col = "gk_pitch_control_share_weighted_a0"
        if share_col in result_s.columns and share_col in result_v.columns:
            s_vals = result_s[share_col].dropna()
            v_vals = result_v[share_col].dropna()
            if len(s_vals) > 0 and len(v_vals) > 0:
                assert not np.allclose(s_vals.values, v_vals.values, equal_nan=True)


# === T-10: Atomic mirror ===


class TestAtomicMirror:
    """T-10: Atomic SPADL produces same values via x/y anchor."""

    def test_atomic_share_matches_standard(self, fitted_xt):
        """Standard and atomic paths produce identical share values."""
        from silly_kicks.atomic.tracking.features import (
            gk_pitch_control_share_weighted as atomic_share,
        )
        from silly_kicks.tracking.features import (
            gk_pitch_control_share_weighted as std_share,
        )

        actions, frames = _make_actions_and_frames()
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]

        std_result = std_share(actions, frames, fitted_xt, goal_map=HOME_GOAL_MAP)
        atomic_result = atomic_share(atomic_actions, frames, fitted_xt, goal_map=HOME_GOAL_MAP)
        pd.testing.assert_series_equal(std_result, atomic_result, check_names=False)

    def test_atomic_closing_time_matches(self, fitted_xt):
        from silly_kicks.atomic.tracking.features import (
            gk_closing_time_min_s as atomic_ct,
        )
        from silly_kicks.tracking.features import gk_closing_time_min_s as std_ct

        actions, frames = _make_actions_and_frames()
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]

        std_result = std_ct(actions, frames, goal_map=HOME_GOAL_MAP)
        atomic_result = atomic_ct(atomic_actions, frames, goal_map=HOME_GOAL_MAP)
        pd.testing.assert_series_equal(std_result, atomic_result, check_names=False)

    def test_atomic_aggregator_column_set(self, fitted_xt):
        from silly_kicks.atomic.tracking.features import add_gk_influence

        actions, frames = _make_actions_and_frames()
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]

        result = add_gk_influence(atomic_actions, frames, fitted_xt, goal_map=HOME_GOAL_MAP)
        assert "gk_pitch_control_share_weighted" in result.columns
        assert "gk_reachable_area_m2" in result.columns

    def test_atomic_xfns_column_count(self, fitted_xt):
        from silly_kicks.atomic.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, goal_map=HOME_GOAL_MAP)
        assert len(xfns) == 1
        assert xfns[0]._frame_aware is True
