"""Tests for ``silly_kicks.atomic.spadl.utils.add_gk_distribution_metrics`` (1.5.0).

Atomic-SPADL counterpart to ``silly_kicks.spadl.utils.add_gk_distribution_metrics``.

Three atomic-specific adaptations:

  - Length is ``sqrt(dx² + dy²)`` from atomic's ``(dx, dy)`` columns.
  - xT delta is from ``(x, y)`` (start) to ``(x+dx, y+dy)`` (end).
  - Pass success is detected from the FOLLOWING atomic action by row index:
    ``receival`` (same-team pickup) → success; ``interception`` /
    ``out`` / ``offside`` → failure; no following action (last row of
    game/period) → conservative failure (xT delta = NaN).

Same four output columns:

  - ``gk_pass_length_m``
  - ``gk_pass_length_class`` (Categorical short/medium/long)
  - ``is_launch``
  - ``gk_xt_delta``

Atomic launch types collapse standard SPADL's
``{pass, goalkick, freekick_short, freekick_crossed}`` into
``{pass, goalkick, freekick}`` (where ``freekick`` is the post-collapse name
in atomic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl.utils import add_gk_distribution_metrics, add_gk_role
from tests.atomic._atomic_test_fixtures import (
    _df,
    _make_atomic_action,
    _make_atomic_gk_action,
    _make_atomic_interception,
    _make_atomic_offside,
    _make_atomic_out,
    _make_atomic_pass_action,
    _make_atomic_receival,
)


def _save_then_pass_then_outcome(
    *,
    pass_dx: float = 10.0,
    pass_dy: float = 0.0,
    pass_type: str = "pass",
    outcome: str = "receival",
    pass_x: float = 5.0,
    pass_y: float = 34.0,
) -> pd.DataFrame:
    """Build a 3-row fixture: keeper_save → GK distribution pass → outcome.

    ``outcome`` must be ``receival`` (success), ``interception`` /
    ``out`` / ``offside`` (failure), or ``none`` (no follow-up — pass is the
    last row of game/period).
    """
    rows = [
        _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
        _make_atomic_pass_action(
            action_id=1,
            player_id=999,
            team_id=100,
            pass_type=pass_type,
            time_seconds=1.0,
            x=pass_x,
            y=pass_y,
            dx=pass_dx,
            dy=pass_dy,
        ),
    ]
    if outcome == "receival":
        rows.append(_make_atomic_receival(action_id=2, team_id=100, time_seconds=1.5))
    elif outcome == "interception":
        rows.append(_make_atomic_interception(action_id=2, team_id=200, time_seconds=1.5))
    elif outcome == "out":
        rows.append(_make_atomic_out(action_id=2, team_id=100, time_seconds=1.5))
    elif outcome == "offside":
        rows.append(_make_atomic_offside(action_id=2, team_id=100, time_seconds=1.5))
    elif outcome == "none":
        pass  # no follow-up
    else:
        raise AssertionError(f"unknown outcome: {outcome}")
    return _df(rows)


class TestContract:
    def test_returns_dataframe(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_four_columns(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        assert "gk_pass_length_m" in result.columns
        assert "gk_pass_length_class" in result.columns
        assert "is_launch" in result.columns
        assert "gk_xt_delta" in result.columns

    def test_pass_length_class_is_categorical(self):
        actions = _save_then_pass_then_outcome()
        result = add_gk_distribution_metrics(actions)
        assert isinstance(result["gk_pass_length_class"].dtype, pd.CategoricalDtype)

    def test_categorical_categories_are_locked(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        expected = {"short", "medium", "long"}
        assert set(result["gk_pass_length_class"].cat.categories) == expected

    def test_preserves_all_input_columns(self):
        actions = _df([_make_atomic_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_gk_distribution_metrics(actions)
        assert "custom_col" in result.columns

    def test_does_not_mutate_input(self):
        actions = _df([_make_atomic_action(action_id=0)])
        cols_before = list(actions.columns)
        add_gk_distribution_metrics(actions)
        assert list(actions.columns) == cols_before

    def test_empty_input(self):
        actions = _df([_make_atomic_action(action_id=0)]).iloc[0:0]
        result = add_gk_distribution_metrics(actions)
        assert "gk_pass_length_m" in result.columns
        assert len(result) == 0


class TestAutoCallAddGkRole:
    def test_auto_calls_add_gk_role_when_absent(self):
        actions = _save_then_pass_then_outcome()
        result = add_gk_distribution_metrics(actions)
        assert "gk_role" in result.columns

    def test_uses_existing_gk_role_when_present(self):
        actions = _save_then_pass_then_outcome()
        actions = add_gk_role(actions)
        result = add_gk_distribution_metrics(actions)
        assert "gk_role" in result.columns

    def test_require_gk_role_false_skips_auto_call(self):
        actions = _save_then_pass_then_outcome()
        result = add_gk_distribution_metrics(actions, require_gk_role=False)
        assert result["gk_pass_length_m"].isna().all()


class TestLengthClassification:
    def test_short_pass_under_32m(self):
        actions = _save_then_pass_then_outcome(pass_dx=20.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_m"].iloc[1] == pytest.approx(20.0)
        assert result["gk_pass_length_class"].iloc[1] == "short"

    def test_medium_pass_at_45m(self):
        actions = _save_then_pass_then_outcome(pass_dx=45.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_long_pass_over_60m(self):
        actions = _save_then_pass_then_outcome(pass_dx=70.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "long"

    def test_short_threshold_boundary(self):
        actions = _save_then_pass_then_outcome(pass_dx=32.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_long_threshold_boundary(self):
        actions = _save_then_pass_then_outcome(pass_dx=60.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_just_over_long_threshold(self):
        actions = _save_then_pass_then_outcome(pass_dx=60.5, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_class"].iloc[1] == "long"

    def test_custom_thresholds(self):
        actions = _save_then_pass_then_outcome(pass_dx=30.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions, short_threshold=20.0, long_threshold=80.0)
        assert result["gk_pass_length_class"].iloc[1] == "medium"

    def test_diagonal_distance(self):
        # dx=30, dy=30 → length = sqrt(1800)
        actions = _save_then_pass_then_outcome(pass_dx=30.0, pass_dy=30.0)
        result = add_gk_distribution_metrics(actions)
        assert result["gk_pass_length_m"].iloc[1] == pytest.approx(np.sqrt(1800), rel=1e-3)


class TestIsLaunch:
    def test_long_pass_is_launch(self):
        actions = _save_then_pass_then_outcome(pass_dx=75.0, pass_dy=0.0, pass_type="pass")
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is True

    def test_long_goalkick_is_launch(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_pick_up", player_id=999),
                _make_atomic_pass_action(
                    action_id=1,
                    player_id=999,
                    team_id=100,
                    pass_type="goalkick",
                    time_seconds=1.0,
                    x=5.0,
                    dx=70.0,
                    dy=0.0,
                ),
                _make_atomic_receival(action_id=2, team_id=100, time_seconds=1.5),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is True

    def test_long_freekick_is_launch(self):
        """Atomic ``freekick`` is the post-collapse name for both pass-class and
        shot-class freekicks; treated as a launch type for distribution purposes."""
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_atomic_pass_action(
                    action_id=1,
                    player_id=999,
                    team_id=100,
                    pass_type="freekick",
                    time_seconds=1.0,
                    x=5.0,
                    dx=70.0,
                    dy=0.0,
                ),
                _make_atomic_receival(action_id=2, team_id=100, time_seconds=1.5),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is True

    def test_short_pass_is_not_launch(self):
        actions = _save_then_pass_then_outcome(pass_dx=15.0, pass_dy=0.0)
        result = add_gk_distribution_metrics(actions)
        assert bool(result["is_launch"].iloc[1]) is False

    def test_long_distance_action_outside_pass_types_is_not_launch(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_atomic_action(
                    action_id=1,
                    type_name="clearance",
                    player_id=999,
                    team_id=100,
                    time_seconds=1.0,
                    x=5.0,
                    dx=70.0,
                    dy=0.0,
                ),
            ]
        )
        result = add_gk_distribution_metrics(actions)
        # Clearance is not in launch types AND clearance is not a distribution type per
        # add_gk_role (clearance is itself a defensive action, not a follow-up).
        assert bool(result["is_launch"].iloc[1]) is False


class TestXtDelta:
    def _identity_grid(self) -> np.ndarray:
        grid = np.zeros((12, 8), dtype=np.float64)
        for zx in range(12):
            grid[zx, :] = float(zx)
        return grid

    def test_xt_delta_positive_for_forward_successful_pass(self):
        grid = self._identity_grid()
        # x=5 (zone_x=0), dx=65 → end_x=70 (zone_x=8). Receival → success.
        actions = _save_then_pass_then_outcome(pass_dx=65.0, pass_x=5.0, outcome="receival")
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert result["gk_xt_delta"].iloc[1] == pytest.approx(8.0)

    def test_xt_delta_skipped_for_intercepted_pass(self):
        grid = self._identity_grid()
        actions = _save_then_pass_then_outcome(pass_dx=65.0, outcome="interception")
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_skipped_for_out_pass(self):
        grid = self._identity_grid()
        actions = _save_then_pass_then_outcome(pass_dx=65.0, outcome="out")
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_skipped_for_offside_pass(self):
        grid = self._identity_grid()
        actions = _save_then_pass_then_outcome(pass_dx=65.0, outcome="offside")
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_skipped_when_no_following_action(self):
        """Per Q2 lock: no following action (last row of game/period) → conservative
        failure (xT delta = NaN)."""
        grid = self._identity_grid()
        actions = _save_then_pass_then_outcome(pass_dx=65.0, outcome="none")
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_nan_without_grid(self):
        actions = _save_then_pass_then_outcome()
        result = add_gk_distribution_metrics(actions)
        assert pd.isna(result["gk_xt_delta"].iloc[1])

    def test_xt_delta_nan_for_non_distribution(self):
        grid = self._identity_grid()
        actions = _df([_make_atomic_pass_action(action_id=0)])
        result = add_gk_distribution_metrics(actions, xt_grid=grid)
        assert pd.isna(result["gk_xt_delta"].iloc[0])


class TestNonDistributionRows:
    def test_keeper_save_action_has_nan_metrics(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save")])
        result = add_gk_distribution_metrics(actions)
        assert pd.isna(result["gk_pass_length_m"].iloc[0])
        assert pd.isna(result["gk_pass_length_class"].iloc[0])
        assert bool(result["is_launch"].iloc[0]) is False
        assert pd.isna(result["gk_xt_delta"].iloc[0])

    def test_outfield_pass_has_nan_metrics(self):
        actions = _df([_make_atomic_pass_action(action_id=0)])
        result = add_gk_distribution_metrics(actions)
        assert pd.isna(result["gk_pass_length_m"].iloc[0])
        assert bool(result["is_launch"].iloc[0]) is False


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_atomic_action(action_id=0)]).drop(columns=["x"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_gk_distribution_metrics(actions)

    def test_negative_short_threshold_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match=r"threshold"):
            add_gk_distribution_metrics(actions, short_threshold=-5.0)

    def test_invalid_threshold_ordering_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match=r"threshold"):
            add_gk_distribution_metrics(actions, short_threshold=50.0, long_threshold=40.0)

    def test_invalid_xt_grid_shape_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        bad_grid = np.zeros((5, 5), dtype=np.float64)
        with pytest.raises(ValueError, match=r"xt_grid"):
            add_gk_distribution_metrics(actions, xt_grid=bad_grid)
