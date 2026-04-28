"""Tests for ``silly_kicks.atomic.spadl.utils.add_possessions`` (added in 1.5.0).

Atomic-SPADL counterpart to ``silly_kicks.spadl.utils.add_possessions``.

Algorithm parity with the standard helper, with two atomic-specific adaptations:

  - Set-piece restart names match the post-collapse atomic action types:
    ``corner`` / ``freekick`` (NOT ``corner_short`` / ``corner_crossed`` /
    ``freekick_short`` / ``freekick_crossed``) plus ``throw_in`` / ``goalkick``
    which are not collapsed.
  - ``yellow_card`` and ``red_card`` synthetic atomic rows do NOT trigger
    possession boundaries — they inherit the surrounding possession state.
    The carve-out for set-piece-after-foul still fires when the chain is
    ``... → foul → yellow_card → freekick`` (cards appear between foul and
    restart in atomic streams).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl.utils import add_possessions
from tests.atomic._atomic_test_fixtures import (
    _df,
    _make_atomic_action,
    _make_atomic_card,
    _make_atomic_pass_action,
    _make_atomic_receival,
)


class TestAddPossessionsContract:
    def test_returns_dataframe(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_possessions(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_possession_id_column(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_possessions(actions)
        assert "possession_id" in result.columns

    def test_possession_id_dtype_is_int64(self):
        actions = _df([_make_atomic_action(action_id=0), _make_atomic_action(action_id=1, time_seconds=1.0)])
        result = add_possessions(actions)
        assert result["possession_id"].dtype == np.int64

    def test_preserves_all_input_columns(self):
        actions = _df([_make_atomic_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_possessions(actions)
        assert "custom_col" in result.columns
        assert list(result["custom_col"]) == ["preserved"]

    def test_empty_input_returns_empty_with_possession_id(self):
        actions = _df([_make_atomic_action(action_id=0)]).iloc[0:0]
        result = add_possessions(actions)
        assert "possession_id" in result.columns
        assert len(result) == 0

    def test_does_not_mutate_input(self):
        actions = _df([_make_atomic_action(action_id=0)])
        cols_before = list(actions.columns)
        add_possessions(actions)
        assert list(actions.columns) == cols_before, "input df should not be mutated"

    def test_returns_same_row_count(self):
        actions = _df([_make_atomic_action(action_id=i, time_seconds=float(i)) for i in range(10)])
        result = add_possessions(actions)
        assert len(result) == len(actions)


class TestSameTeamSamePossession:
    def test_two_same_team_passes_share_possession_id(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]

    def test_pass_then_receival_same_possession(self):
        actions = _df(
            [
                _make_atomic_pass_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_atomic_receival(action_id=1, team_id=100, time_seconds=0.5),
            ]
        )
        result = add_possessions(actions)
        # Pass + receival are both team 100 → same possession.
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]


class TestTeamChangeNewPossession:
    def test_team_change_increments_possession_id(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=200, time_seconds=1.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1

    def test_alternating_teams_unique_possession_ids(self):
        actions = _df(
            [_make_atomic_action(action_id=i, team_id=100 + (i % 2) * 100, time_seconds=float(i)) for i in range(6)]
        )
        result = add_possessions(actions)
        assert list(result["possession_id"]) == [0, 1, 2, 3, 4, 5]


class TestPeriodChange:
    def test_period_change_within_game_increments_counter(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, period_id=1, team_id=100, time_seconds=2700.0),
                _make_atomic_action(action_id=1, period_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1


class TestGameChange:
    def test_game_change_resets_counter_to_zero(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, game_id=1, team_id=200, time_seconds=1.0),
                _make_atomic_action(action_id=0, game_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 0

    def test_multi_game_counter_independent(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, game_id=1, team_id=200, time_seconds=1.0),
                _make_atomic_action(action_id=0, game_id=2, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, game_id=2, team_id=100, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions)
        per_game = result.groupby("game_id")["possession_id"].agg(["min", "max"])
        assert per_game.loc[1, "min"] == 0
        assert per_game.loc[1, "max"] == 1
        assert per_game.loc[2, "min"] == 0
        assert per_game.loc[2, "max"] == 0


class TestMaxGapTimeout:
    def test_long_gap_starts_new_possession_same_team(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=10.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[1] == 1

    def test_default_5s_threshold_just_under_keeps_possession(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=4.9),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]

    def test_custom_max_gap_seconds(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.5),
            ]
        )
        result = add_possessions(actions, max_gap_seconds=1.0)
        assert result["possession_id"].iloc[0] != result["possession_id"].iloc[1]


class TestSetPieceCarveOut:
    """Set-piece carve-out adapted for atomic's collapsed type names."""

    def test_freekick_after_opposing_foul_retains_possession(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="freekick"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 1, (
            "free kick after opposing-team foul should retain the previous possession"
        )

    def test_corner_after_opposing_foul_retains_possession(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="corner"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 1

    def test_throw_in_after_foul_retains(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="throw_in"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 1

    def test_goalkick_after_foul_retains(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="goalkick"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 1

    def test_carve_out_disabled(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="freekick"),
            ]
        )
        result = add_possessions(actions, retain_on_set_pieces=False)
        assert result["possession_id"].iloc[2] == 2

    def test_carve_out_does_not_apply_to_normal_pass(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 2

    def test_corner_after_opposing_clearance_does_not_retain(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="clearance"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0, type_name="corner"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 2


class TestYellowRedCardsBetweenFoulAndRestart:
    """In atomic streams, yellow_card / red_card synthetic rows appear between
    a foul and the resulting set-piece restart. They must NOT break the carve-out."""

    def test_yellow_card_between_foul_and_freekick_carve_out_still_fires(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_card(action_id=2, card="yellow_card", team_id=100, time_seconds=1.0),
                _make_atomic_action(action_id=3, team_id=200, time_seconds=2.0, type_name="freekick"),
            ]
        )
        result = add_possessions(actions)
        # foul gets a new possession (team change A → fouler), card inherits the foul's id,
        # freekick retains via carve-out (looks past the card to the foul).
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1  # foul
        assert result["possession_id"].iloc[2] == 1  # yellow_card inherits
        assert result["possession_id"].iloc[3] == 1, (
            "freekick after foul→yellow_card→freekick should still retain via carve-out"
        )

    def test_red_card_between_foul_and_freekick_carve_out_still_fires(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_card(action_id=2, card="red_card", team_id=100, time_seconds=1.0),
                _make_atomic_action(action_id=3, team_id=200, time_seconds=2.0, type_name="freekick"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[3] == 1

    def test_yellow_card_alone_does_not_create_boundary(self):
        # Pass(A) → foul(B) → yellow_card(B): yellow_card inherits foul's possession id,
        # does not increment.
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=200, time_seconds=1.0, type_name="foul"),
                _make_atomic_card(action_id=2, card="yellow_card", team_id=200, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions)
        # foul at team-change → possession 1; card inherits → also 1.
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 1, (
            "yellow_card has the fouler's team id but should NOT create its own possession boundary"
        )


class TestMonotonic:
    def test_possession_id_monotonic_within_game(self):
        rng = np.random.default_rng(42)
        rows = []
        teams = [100, 200]
        for i in range(30):
            team = int(teams[rng.integers(0, 2)])
            rows.append(_make_atomic_action(action_id=i, team_id=team, time_seconds=float(i), type_name="pass"))
        actions = _df(rows)
        result = add_possessions(actions)
        ids = result.sort_values(["game_id", "period_id", "action_id"])["possession_id"].to_numpy()
        diffs = np.diff(ids)
        assert (diffs >= 0).all(), f"possession_id must be non-decreasing within game; got diffs {diffs}"

    def test_possession_id_starts_at_zero_per_game(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=0, game_id=2, team_id=200, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        per_game_first = (
            result.sort_values(["game_id", "period_id", "action_id"]).groupby("game_id")["possession_id"].first()
        )
        assert per_game_first.loc[1] == 0
        assert per_game_first.loc[2] == 0


class TestCombinedBoundaries:
    def test_period_change_overrides_set_piece_carve_out(self):
        actions = _df(
            [
                _make_atomic_action(
                    action_id=0,
                    period_id=1,
                    team_id=100,
                    time_seconds=2700.0,
                    type_name="foul",
                ),
                _make_atomic_action(action_id=1, period_id=2, team_id=200, time_seconds=0.0, type_name="freekick"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1

    def test_long_gap_overrides_set_piece_carve_out(self):
        actions = _df(
            [
                _make_atomic_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul"),
                _make_atomic_action(action_id=2, team_id=200, time_seconds=10.0, type_name="freekick"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 2


class TestSortOrder:
    def test_unsorted_input_is_sorted_internally(self):
        actions = _df(
            [
                _make_atomic_action(action_id=2, team_id=200, time_seconds=2.0),
                _make_atomic_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_atomic_action(action_id=1, team_id=100, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions).sort_values("action_id").reset_index(drop=True)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]
        assert result["possession_id"].iloc[2] != result["possession_id"].iloc[0]


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_atomic_action(action_id=0)]).drop(columns=["game_id"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_possessions(actions)

    def test_negative_max_gap_seconds_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match="max_gap_seconds"):
            add_possessions(actions, max_gap_seconds=-1.0)
