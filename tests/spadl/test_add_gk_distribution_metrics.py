"""Tests for ``silly_kicks.spadl.utils.add_gk_distribution_metrics`` (1.4.0).

Adds four columns to GK distribution actions:

  - ``gk_pass_length_m: float`` — Euclidean ``(start, end)`` distance
  - ``gk_pass_length_class: Categorical[short, medium, long]``
  - ``is_launch: bool`` — long-distance pass-type distribution
  - ``gk_xt_delta: float | NaN`` -- xT(end_zone) - xT(start_zone), only when
    ``xt_grid`` is provided AND the pass succeeded

Locked design (Q2.1, Q2.2, Q2.3):

  - Auto-calls ``add_gk_role`` if ``gk_role`` column absent (``require_gk_role=True``)
  - xT is skipped for failed passes (end coordinates are interception points)
  - ``is_launch`` requires both length > ``long_threshold`` AND pass-type action
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_gk_distribution_metrics, add_gk_role
from tests.spadl._gk_test_fixtures import _df, _make_action, _make_gk_action, _make_pass_action

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_returns_dataframe(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_four_columns(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        assert "gk_pass_length_m" in result.columns
        assert "gk_pass_length_class" in result.columns
        assert "is_launch" in result.columns
        assert "gk_xt_delta" in result.columns

    def test_pass_length_class_is_categorical(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=999, time_seconds=1.0),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert isinstance(result["gk_pass_length_class"].dtype, pd.CategoricalDtype)

    def test_categorical_categories_are_locked(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        expected = {"short", "medium", "long"}
        assert set(result["gk_pass_length_class"].cat.categories) == expected

    def test_preserves_all_input_columns(self):
        actions = _df([_make_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_gk_distribution_metrics(actions)
        assert "custom_col" in result.columns

    def test_does_not_mutate_input(self):
        actions = _df([_make_action(action_id=0)])
        cols_before = list(actions.columns)
        add_gk_distribution_metrics(actions)
        assert list(actions.columns) == cols_before

    def test_empty_input(self):
        actions = _df([_make_action(action_id=0)]).iloc[0:0]
        result = add_gk_distribution_metrics(actions)
        assert "gk_pass_length_m" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Auto-call add_gk_role (Q2.1)
# ---------------------------------------------------------------------------


class TestAutoCallAddGkRole:
    def test_auto_calls_add_gk_role_when_absent(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=999, time_seconds=1.0),
            ]
        )
        # gk_role column not pre-computed
        result = add_gk_distribution_metrics(actions)
        # gk_role column should now be present
        assert "gk_role" in result.columns

    def test_uses_existing_gk_role_when_present(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=999, time_seconds=1.0),
            ]
        )
        actions = add_gk_role(actions)
        # Pre-computed gk_role; helper should use it directly.
        result = add_gk_distribution_metrics(actions)
        assert "gk_role" in result.columns

    def test_require_gk_role_false_skips_auto_call(self):
        # When require_gk_role=False and gk_role absent, helper SHOULD NOT auto-call
        # → gk_pass_length_m / etc. are all NaN/False/None for every row (no
        # distribution detection possible).
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=999, time_seconds=1.0),
            ]
        )
        result = add_gk_distribution_metrics(actions, require_gk_role=False)
        # No distribution detection without gk_role context.
        assert result["gk_pass_length_m"].isna().all()


# ---------------------------------------------------------------------------
# Length classification
# ---------------------------------------------------------------------------


class TestLengthClassification:
    def _build_distribution(self, *, distance_m: float) -> pd.DataFrame:
        """Build a (save → pass) sequence with a controlled pass distance."""
        end_x = 5.0 + distance_m  # pass starts at x=5, lands at x=5+distance
        return _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,
                    start_y=34.0,
                    end_x=end_x,
                    end_y=34.0,
                ),
            ]
        )

    def test_short_pass_under_32m(self):
        actions = self._build_distribution(distance_m=20.0)
        result = add_gk_distribution_metrics(actions)
        # Action 1 is the pass row.
        assert result["gk_pass_length_m"].iloc[1] == pytest.approx(20.0)
        assert result["gk_pass_length_class"].iloc[1] == "short"

    def test_medium_pass_at_45m(self):
        actions = self._build_distribution(distance_m=45.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_long_pass_over_60m(self):
        actions = self._build_distribution(distance_m=70.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "long"

    def test_short_threshold_boundary(self):
        # Exactly 32m → "medium" (short is strictly < short_threshold)
        actions = self._build_distribution(distance_m=32.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_long_threshold_boundary(self):
        # Exactly 60m → "medium" (long is strictly > long_threshold)
        actions = self._build_distribution(distance_m=60.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_just_over_long_threshold(self):
        actions = self._build_distribution(distance_m=60.5)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "long"

    def test_custom_thresholds(self):
        actions = self._build_distribution(distance_m=30.0)
        # With short_threshold=20, this 30m pass is medium (no longer short).
        result = add_gk_distribution_metrics(actions, short_threshold=20.0, long_threshold=80.0)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_diagonal_distance(self):
        # Pass from (5, 34) to (35, 64): dx=30, dy=30, distance = sqrt(1800) ≈ 42.4m
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,
                    start_y=34.0,
                    end_x=35.0,
                    end_y=64.0,
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_m"].iloc[1] == pytest.approx(np.sqrt(1800), rel=1e-3)


# ---------------------------------------------------------------------------
# is_launch (Q2.3) — requires pass-type AND length > long_threshold
# ---------------------------------------------------------------------------


class TestIsLaunch:
    def test_long_pass_is_launch(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    pass_type="pass",
                    time_seconds=1.0,
                    start_x=5.0,
                    end_x=80.0,
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is True

    def test_long_goalkick_is_launch(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_pick_up", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    pass_type="goalkick",
                    time_seconds=1.0,
                    start_x=5.0,
                    end_x=75.0,
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is True

    def test_short_pass_is_not_launch(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,
                    end_x=20.0,
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is False

    def test_long_distance_action_outside_pass_types_is_not_launch(self):
        # A long clearance — not a launch (literature convention; clearance != deliberate distribution).
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_action(
                    action_id=1,
                    type_name="clearance",
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,
                    end_x=80.0,
                ),
            ]
        )
        # A clearance after a save is NOT distribution per gk_role logic anyway,
        # so is_launch must be False either way.
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is False


# ---------------------------------------------------------------------------
# xT delta (Q2.2)
# ---------------------------------------------------------------------------


class TestXtDelta:
    def _identity_grid(self) -> np.ndarray:
        """12x8 xT grid where xT[zx, zy] = zx (so end_zone_x - start_zone_x captures progression)."""
        grid = np.zeros((12, 8), dtype=np.float64)
        for zx in range(12):
            grid[zx, :] = float(zx)
        return grid

    def test_xt_delta_positive_for_forward_pass(self):
        grid = self._identity_grid()
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,  # zone_x = 0
                    end_x=70.0,  # zone_x = 8
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        # Distribution at row 1: zone_x diff = 8 - 0 = 8.
        assert result["gk_xt_delta"].iloc[1] == pytest.approx(8.0)

    def test_xt_delta_skipped_for_failed_pass(self):
        # Q2.2 lock: xT is skipped for failed passes (interception coords).
        grid = self._identity_grid()
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(
                    action_id=1,
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,
                    end_x=70.0,
                    result_name="fail",
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_nan_without_grid(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=999, time_seconds=1.0),
            ]
        )
        result = add_gk_distribution_metrics(actions)  # no xt_grid
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_nan_for_non_distribution(self):
        grid = self._identity_grid()
        actions = _df([_make_pass_action(action_id=0)])  # outfield pass, not distribution
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[0])


# ---------------------------------------------------------------------------
# Non-distribution rows: NaN/False
# ---------------------------------------------------------------------------


class TestNonDistributionRows:
    def test_keeper_save_action_has_nan_metrics(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save")])
        result = add_gk_distribution_metrics(actions)
        assert pd.isna(result["gk_pass_length_m"].iloc[0])
        assert pd.isna(result["gk_pass_length_class"].iloc[0])
        assert bool(result["is_launch"].iloc[0]) is False
        assert pd.isna(result["gk_xt_delta"].iloc[0])

    def test_outfield_pass_has_nan_metrics(self):
        actions = _df([_make_pass_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        assert pd.isna(result["gk_pass_length_m"].iloc[0])
        assert bool(result["is_launch"].iloc[0]) is False


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_action(action_id=0)]).drop(columns=["start_x"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_gk_distribution_metrics(actions)

    def test_negative_short_threshold_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"threshold"):
            add_gk_distribution_metrics(actions, short_threshold=-5.0)

    def test_invalid_threshold_ordering_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"threshold"):
            add_gk_distribution_metrics(actions, short_threshold=50.0, long_threshold=40.0)

    def test_invalid_xt_grid_shape_raises(self):
        actions = _df([_make_action(action_id=0)])
        bad_grid = np.zeros((5, 5), dtype=np.float64)
        with pytest.raises(ValueError, match=r"xt_grid"):
            add_gk_distribution_metrics(actions, xt_grid=bad_grid)
