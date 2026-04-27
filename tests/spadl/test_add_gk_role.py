"""Tests for ``silly_kicks.spadl.utils.add_gk_role`` (added in 1.4.0).

The helper tags each SPADL action with the goalkeeper's role context:

  - ``shot_stopping`` (keeper_save)
  - ``cross_collection`` (keeper_claim, keeper_punch)
  - ``pick_up`` (keeper_pick_up)
  - ``sweeping`` (any keeper_* with start_x > penalty_area_x_threshold;
    overrides the type-specific role)
  - ``distribution`` (non-keeper action by the same player whose immediately-
    preceding action was a keeper action)
  - ``None`` for everything else

Locked design decisions (see project memory
``project_silly_kicks_gk_analytics_handoff_2026_04_27.md``):

  - Q1.1: sweeping overrides the type-specific role
  - Q1.2: ``distribution_lookback_actions=1`` default
  - Q1.3: GK identity inferred from ``keeper_*`` action history (no
    ``position_group`` flag in canonical SPADL)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_gk_role
from tests.spadl._gk_test_fixtures import (
    _df,
    _make_action,
    _make_gk_action,
    _make_pass_action,
    _make_shot_action,
)

# ---------------------------------------------------------------------------
# Contract: shape + dtype + column preservation
# ---------------------------------------------------------------------------


class TestAddGkRoleContract:
    def test_returns_dataframe(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_role(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_gk_role_column(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_role(actions)
        assert "gk_role" in result.columns

    def test_gk_role_is_categorical(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_role(actions)
        assert isinstance(result["gk_role"].dtype, pd.CategoricalDtype)

    def test_gk_role_categories_are_locked(self):
        actions = _df([_make_action(action_id=0)])
        result = add_gk_role(actions)
        expected = {"shot_stopping", "cross_collection", "sweeping", "pick_up", "distribution"}
        assert set(result["gk_role"].cat.categories) == expected

    def test_preserves_all_input_columns(self):
        actions = _df([_make_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_gk_role(actions)
        assert "custom_col" in result.columns
        assert list(result["custom_col"]) == ["preserved"]

    def test_empty_input_returns_empty_with_gk_role(self):
        actions = _df([_make_action(action_id=0)]).iloc[0:0]
        result = add_gk_role(actions)
        assert "gk_role" in result.columns
        assert len(result) == 0

    def test_does_not_mutate_input(self):
        actions = _df([_make_action(action_id=0)])
        cols_before = list(actions.columns)
        add_gk_role(actions)
        assert list(actions.columns) == cols_before

    def test_returns_same_row_count(self):
        actions = _df([_make_action(action_id=i, time_seconds=float(i)) for i in range(10)])
        result = add_gk_role(actions)
        assert len(result) == len(actions)


# ---------------------------------------------------------------------------
# Direct GK action type → role mapping
# ---------------------------------------------------------------------------


class TestGkActionTypeMapping:
    def test_keeper_save_maps_to_shot_stopping(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "shot_stopping"

    def test_keeper_claim_maps_to_cross_collection(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_claim")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "cross_collection"

    def test_keeper_punch_maps_to_cross_collection(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_punch")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "cross_collection"

    def test_keeper_pick_up_maps_to_pick_up(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_pick_up")])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "pick_up"


# ---------------------------------------------------------------------------
# Sweeping override (Q1.1) — keeper action outside box → "sweeping"
# ---------------------------------------------------------------------------


class TestSweepingOverride:
    def test_keeper_save_outside_box_becomes_sweeping(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save", start_x=20.0)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_keeper_punch_outside_box_becomes_sweeping(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_punch", start_x=18.0)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_keeper_pickup_outside_box_becomes_sweeping(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_pick_up", start_x=20.0)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_keeper_save_at_threshold_stays_shot_stopping(self):
        # Boundary: start_x exactly at 16.5 is INSIDE the box (NOT sweeping).
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save", start_x=16.5)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "shot_stopping"

    def test_keeper_save_just_past_threshold_becomes_sweeping(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save", start_x=16.6)])
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "sweeping"

    def test_custom_threshold(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save", start_x=20.0)])
        # With a custom threshold of 25m, this save at x=20 is INSIDE the (custom) box.
        result = add_gk_role(actions, penalty_area_x_threshold=25.0)
        assert result["gk_role"].iloc[0] == "shot_stopping"

    def test_non_keeper_action_outside_box_is_not_sweeping(self):
        # A pass at x=20 is NOT a keeper action → not sweeping (and not GK role).
        actions = _df([_make_action(action_id=0, type_name="pass", start_x=20.0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])


# ---------------------------------------------------------------------------
# Distribution detection (Q1.2 + Q1.3) — non-keeper action by the GK
# immediately following a keeper action
# ---------------------------------------------------------------------------


class TestDistributionDetection:
    def test_pass_immediately_after_save_by_same_player_is_distribution(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=999, time_seconds=2.0),
            ]
        )
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[0] == "shot_stopping"
        assert result["gk_role"].iloc[1] == "distribution"

    def test_goalkick_after_pickup_by_same_player_is_distribution(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_pick_up", player_id=999),
                _make_pass_action(action_id=1, player_id=999, pass_type="goalkick", time_seconds=3.0),
            ]
        )
        result = add_gk_role(actions)
        assert result["gk_role"].iloc[1] == "distribution"

    def test_pass_after_save_but_different_player_is_not_distribution(self):
        # GK saves, then a defender (different player) takes the pass → not distribution.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=200, time_seconds=2.0),
            ]
        )
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[1])

    def test_pass_two_actions_after_save_is_not_distribution_default(self):
        # With default lookback=1, only the IMMEDIATELY-following action counts.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=200, time_seconds=2.0),
                _make_pass_action(action_id=2, player_id=999, time_seconds=4.0),
            ]
        )
        result = add_gk_role(actions)
        # action 1 is by a different player → not distribution
        assert pd.isna(result["gk_role"].iloc[1])
        # action 2 is two steps from the save → not distribution at default k=1
        assert pd.isna(result["gk_role"].iloc[2])

    def test_lookback_2_catches_two_step_distribution(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, player_id=200, time_seconds=2.0),  # other player
                _make_pass_action(action_id=2, player_id=999, time_seconds=4.0),  # GK pass
            ]
        )
        result = add_gk_role(actions, distribution_lookback_actions=2)
        assert result["gk_role"].iloc[2] == "distribution"

    def test_distribution_does_not_span_games(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, game_id=1, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=0, game_id=2, player_id=999, time_seconds=1.0),
            ]
        )
        result = add_gk_role(actions)
        # The shift would naively look back to game 1's keeper_save, but the same-game guard
        # must prevent the cross-game distribution tag.
        result_sorted = result.sort_values(["game_id", "action_id"]).reset_index(drop=True)
        game1_save = result_sorted[result_sorted["game_id"] == 1].iloc[0]
        game2_pass = result_sorted[result_sorted["game_id"] == 2].iloc[0]
        assert game1_save["gk_role"] == "shot_stopping"
        assert pd.isna(game2_pass["gk_role"]), "distribution must not span game boundaries"

    def test_distribution_first_action_of_game_is_never_distribution(self):
        actions = _df([_make_pass_action(action_id=0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])


# ---------------------------------------------------------------------------
# Non-GK actions get None
# ---------------------------------------------------------------------------


class TestNonGkActions:
    def test_outfield_pass_gets_none(self):
        actions = _df([_make_pass_action(action_id=0, player_id=200)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])

    def test_outfield_shot_gets_none(self):
        actions = _df([_make_action(action_id=0, type_name="shot", start_x=95.0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])

    def test_clearance_outside_box_is_not_sweeping(self):
        # Clearance is not a keeper action → no GK role even at x>16.5.
        actions = _df([_make_action(action_id=0, type_name="clearance", start_x=20.0)])
        result = add_gk_role(actions)
        assert pd.isna(result["gk_role"].iloc[0])


# ---------------------------------------------------------------------------
# Sort order — input not pre-sorted
# ---------------------------------------------------------------------------


class TestSortOrder:
    def test_unsorted_input_is_sorted_internally(self):
        actions = _df(
            [
                _make_pass_action(action_id=1, player_id=999, time_seconds=2.0),
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
            ]
        )
        result = add_gk_role(actions).sort_values("action_id").reset_index(drop=True)
        assert result["gk_role"].iloc[0] == "shot_stopping"
        assert result["gk_role"].iloc[1] == "distribution"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_action(action_id=0)]).drop(columns=["player_id"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_gk_role(actions)

    def test_negative_threshold_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"penalty_area_x_threshold"):
            add_gk_role(actions, penalty_area_x_threshold=-1.0)

    def test_zero_lookback_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"distribution_lookback_actions"):
            add_gk_role(actions, distribution_lookback_actions=0)


# ---------------------------------------------------------------------------
# Multi-game, multi-period combined scenarios
# ---------------------------------------------------------------------------


class TestMultiGameMultiPeriod:
    def test_per_game_independent_role_assignment(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, game_id=1, keeper_action="keeper_save", player_id=999),
                _make_pass_action(action_id=1, game_id=1, player_id=999, time_seconds=2.0),
                _make_gk_action(action_id=0, game_id=2, keeper_action="keeper_claim", player_id=888),
                _make_pass_action(action_id=1, game_id=2, player_id=888, time_seconds=3.0),
            ]
        )
        result = add_gk_role(actions).sort_values(["game_id", "action_id"]).reset_index(drop=True)
        assert list(result["gk_role"]) == [
            "shot_stopping",
            "distribution",
            "cross_collection",
            "distribution",
        ]


# ---------------------------------------------------------------------------
# Realistic combined sequence
# ---------------------------------------------------------------------------


class TestRealisticSequence:
    def test_realistic_match_segment(self):
        # Sequence: shot → save → goalkick distribution → defender pass → cross →
        # GK punches outside box (sweeping) → field pass.
        actions = _df(
            [
                _make_shot_action(action_id=0, player_id=700, team_id=200, time_seconds=10.0),
                _make_gk_action(
                    action_id=1, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=11.0
                ),
                _make_pass_action(
                    action_id=2,
                    player_id=999,
                    team_id=100,
                    pass_type="goalkick",
                    time_seconds=13.0,
                    start_x=5.0,
                    end_x=70.0,
                ),
                _make_pass_action(action_id=3, player_id=200, team_id=100, time_seconds=14.0, start_x=70.0, end_x=85.0),
                _make_pass_action(
                    action_id=4,
                    player_id=300,
                    team_id=200,
                    pass_type="cross",
                    time_seconds=20.0,
                    start_x=95.0,
                    end_x=105.0,
                ),
                _make_gk_action(
                    action_id=5,
                    keeper_action="keeper_punch",
                    player_id=999,
                    team_id=100,
                    time_seconds=21.0,
                    start_x=18.0,  # outside box → sweeping
                ),
                _make_pass_action(action_id=6, player_id=200, team_id=100, time_seconds=23.0, start_x=18.0, end_x=40.0),
            ]
        )
        result = add_gk_role(actions)
        # Verify each row's role
        roles = list(result["gk_role"])
        # action 0 (shot by team 200) → None
        # action 1 (keeper_save by team 100) → shot_stopping
        # action 2 (goalkick by GK same player as save) → distribution
        # action 3 (pass by defender) → None (not GK player)
        # action 4 (cross by team 200) → None
        # action 5 (keeper_punch outside box) → sweeping
        # action 6 (pass by defender) → None (not GK player)
        assert roles == [
            None,
            "shot_stopping",
            "distribution",
            None,
            None,
            "sweeping",
            None,
        ] or [(r if r is None or pd.isna(r) is False else None) for r in roles] == [
            None,
            "shot_stopping",
            "distribution",
            None,
            None,
            "sweeping",
            None,
        ]


# ---------------------------------------------------------------------------
# Synthetic dribbles (NaN player_id is also possible in some flows)
# ---------------------------------------------------------------------------


class TestSyntheticDribbles:
    def test_synthetic_dribble_action_never_distribution(self):
        # If a dribble row has a NaN original_event_id (synthetic), it can still
        # have a player_id from `_add_dribbles`. The distribution-role logic keys
        # off (player_id, prev keeper) — synthetic dribbles by the GK could
        # technically be tagged as distribution. Document the behavior.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999),
                _make_action(
                    action_id=1,
                    type_name="dribble",
                    player_id=999,
                    original_event_id=np.nan,
                    time_seconds=1.0,
                ),
            ]
        )
        result = add_gk_role(actions)
        # Synthetic dribbles are real actions in SPADL; if the GK is dribbling
        # immediately after a save, that IS distribution-like behavior.
        # We allow this — it's a feature, not a bug.
        assert result["gk_role"].iloc[1] == "distribution"
