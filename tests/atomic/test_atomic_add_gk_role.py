"""Tests for ``silly_kicks.atomic.spadl.utils.add_gk_role`` (added in 1.5.0).

Atomic-SPADL counterpart to ``silly_kicks.spadl.utils.add_gk_role``.

Algorithm parity with the standard helper, with one atomic-specific adaptation:

  - Reads ``x`` (NOT ``start_x``) for the penalty-area threshold check.

Same five role categories:

  - ``shot_stopping`` (keeper_save)
  - ``cross_collection`` (keeper_claim, keeper_punch)
  - ``pick_up`` (keeper_pick_up)
  - ``sweeping`` (any keeper_* with x > penalty_area_x_threshold)
  - ``distribution`` (non-keeper action by the same player whose preceding
    action was a keeper action)
  - ``None`` for everything else
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.spadl.utils import add_gk_role
from tests.atomic._atomic_test_fixtures import (
    _df,
    _make_atomic_action,
    _make_atomic_gk_action,
    _make_atomic_pass_action,
)


class TestAddGkRoleContract:
    def test_returns_dataframe(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_role(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_gk_role_column(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_role(actions)
        assert "gk_role" in result.columns

    def test_gk_role_is_categorical(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_role(actions)
        assert isinstance(result["gk_role"].dtype, pd.CategoricalDtype)

    def test_gk_role_categories_are_locked(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_gk_role(actions)
        expected = {"shot_stopping", "cross_collection", "sweeping", "pick_up", "distribution"}
        assert set(result["gk_role"].cat.categories) == expected

    def test_preserves_all_input_columns(self):
        actions = _df([_make_atomic_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_gk_role(actions)
        assert "custom_col" in result.columns
        assert list(result["custom_col"]) == ["preserved"]

    def test_empty_input_returns_empty_with_gk_role(self):
        actions = _df([_make_atomic_action(action_id=0)]).iloc[0:0]
        result = add_gk_role(actions)
        assert "gk_role" in result.columns
        assert len(result) == 0

    def test_does_not_mutate_input(self):
        actions = _df([_make_atomic_action(action_id=0)])
        cols_before = list(actions.columns)
        add_gk_role(actions)
        assert list(actions.columns) == cols_before


class TestGkActionTypeMapping:
    def test_keeper_save_maps_to_shot_stopping(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "shot_stopping"

    def test_keeper_claim_maps_to_cross_collection(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_claim")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "cross_collection"

    def test_keeper_punch_maps_to_cross_collection(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_punch")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "cross_collection"

    def test_keeper_pick_up_maps_to_pick_up(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_pick_up")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "pick_up"


class TestSweepingOverride:
    def test_keeper_save_outside_box_becomes_sweeping(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save", x=20.0)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_keeper_punch_outside_box_becomes_sweeping(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_punch", x=18.0)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_keeper_pickup_outside_box_becomes_sweeping(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_pick_up", x=20.0)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_keeper_save_at_threshold_stays_shot_stopping(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save", x=16.5)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "shot_stopping"

    def test_keeper_save_just_past_threshold_becomes_sweeping(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save", x=16.6)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_custom_threshold(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save", x=20.0)])
        result = add_gk_role(actions, penalty_area_x_threshold=25.0)
        assert result["gk_role"].iloc[0] == "shot_stopping"

    def test_non_keeper_action_outside_box_is_not_sweeping(self):
        actions = _df([_make_atomic_action(action_id=0, type_name="pass", x=20.0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])


class TestDistributionDetection:
    def test_pass_immediately_after_save_by_same_player_is_distribution(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_atomic_pass_action(action_id=1, player_id=999, time_seconds=2.0),
            ]
        )
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "shot_stopping"
        assert result["gk_role"].iloc[1] == "distribution"

    def test_goalkick_after_pickup_by_same_player_is_distribution(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_pick_up", player_id=999),
                _make_atomic_pass_action(action_id=1, player_id=999, pass_type="goalkick", time_seconds=3.0),
            ]
        )
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[1] == "distribution"

    def test_pass_after_save_but_different_player_is_not_distribution(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_atomic_pass_action(action_id=1, player_id=200, time_seconds=2.0),
            ]
        )
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[1])

    def test_pass_two_actions_after_save_is_not_distribution_default(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_atomic_pass_action(action_id=1, player_id=200, time_seconds=2.0),
                _make_atomic_pass_action(action_id=2, player_id=999, time_seconds=4.0),
            ]
        )
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[1])
        assert pd.isna(result["gk_role"].iloc[2])

    def test_lookback_2_catches_two_step_distribution(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_atomic_pass_action(action_id=1, player_id=200, time_seconds=2.0),
                _make_atomic_pass_action(action_id=2, player_id=999, time_seconds=4.0),
            ]
        )
        result = add_gk_role(actions, distribution_lookback_actions=2)
        assert result["gk_role"].iloc[2] == "distribution"

    def test_distribution_does_not_span_games(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, game_id=1, keeper_action="keeper_save", player_id=999),
                _make_atomic_pass_action(action_id=0, game_id=2, player_id=999, time_seconds=1.0),
            ]
        )
        result = add_gk_role(actions).sort_values(["game_id", "action_id"]).reset_index(drop=True)
        game1_save = result[result["game_id"] == 1].iloc[0]
        game2_pass = result[result["game_id"] == 2].iloc[0]
        assert game1_save["gk_role"] == "shot_stopping"
        assert pd.isna(game2_pass["gk_role"]), "distribution must not span game boundaries"

    def test_distribution_first_action_of_game_is_never_distribution(self):
        actions = _df([_make_atomic_pass_action(action_id=0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])


class TestNonGkActions:
    def test_outfield_pass_gets_none(self):
        actions = _df([_make_atomic_pass_action(action_id=0, player_id=200)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])

    def test_outfield_shot_gets_none(self):
        actions = _df([_make_atomic_action(action_id=0, type_name="shot", x=95.0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])

    def test_clearance_outside_box_is_not_sweeping(self):
        actions = _df([_make_atomic_action(action_id=0, type_name="clearance", x=20.0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])


class TestSortOrder:
    def test_unsorted_input_is_sorted_internally(self):
        actions = _df(
            [
                _make_atomic_pass_action(action_id=1, player_id=999, time_seconds=2.0),
                _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
            ]
        )
        result = add_gk_role(actions).sort_values("action_id").reset_index(drop=True)
        assert result["gk_role"].iloc[0] == "shot_stopping"
        assert result["gk_role"].iloc[1] == "distribution"


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_atomic_action(action_id=0)]).drop(columns=["player_id"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_gk_role(actions)

    def test_negative_threshold_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match=r"penalty_area_x_threshold"):
            add_gk_role(actions, penalty_area_x_threshold=-1.0)

    def test_zero_lookback_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match=r"distribution_lookback_actions"):
            add_gk_role(actions, distribution_lookback_actions=0)


class TestMultiGameMultiPeriod:
    def test_per_game_independent_role_assignment(self):
        actions = _df(
            [
                _make_atomic_gk_action(action_id=0, game_id=1, keeper_action="keeper_save"),
                _make_atomic_gk_action(action_id=0, game_id=2, keeper_action="keeper_claim"),
            ]
        )
        result = add_gk_role(actions).sort_values(["game_id", "action_id"]).reset_index(drop=True)
        assert result.iloc[0]["gk_role"] == "shot_stopping"
        assert result.iloc[1]["gk_role"] == "cross_collection"
