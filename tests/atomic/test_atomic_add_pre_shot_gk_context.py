"""Tests for ``silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`` (1.5.0).

Atomic-SPADL counterpart to ``silly_kicks.spadl.utils.add_pre_shot_gk_context``.

Same algorithm; atomic-specific shot-type set:

  - Standard SPADL recognises ``shot``, ``shot_freekick``, ``shot_penalty``.
  - In atomic, ``shot_freekick`` is collapsed into ``freekick`` (which also
    contains pass-class freekicks). Atomic loses the shot/pass distinction
    on free kicks, so the atomic helper recognises only ``shot`` and
    ``shot_penalty`` as shot rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
from tests.atomic._atomic_test_fixtures import (
    _df,
    _make_atomic_action,
    _make_atomic_gk_action,
    _make_atomic_pass_action,
    _make_atomic_shot_action,
)


class TestContract:
    def test_returns_dataframe(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_four_columns(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert "gk_was_distributing" in result.columns
        assert "gk_was_engaged" in result.columns
        assert "gk_actions_in_possession" in result.columns
        assert "defending_gk_player_id" in result.columns

    def test_preserves_input_columns(self):
        actions = _df([_make_atomic_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_pre_shot_gk_context(actions)
        assert "custom_col" in result.columns

    def test_does_not_mutate_input(self):
        actions = _df([_make_atomic_action(action_id=0)])
        cols_before = list(actions.columns)
        add_pre_shot_gk_context(actions)
        assert list(actions.columns) == cols_before

    def test_empty_input(self):
        actions = _df([_make_atomic_action(action_id=0)]).iloc[0:0]
        result = add_pre_shot_gk_context(actions)
        assert "gk_was_distributing" in result.columns
        assert len(result) == 0

    def test_returns_same_row_count(self):
        actions = _df([_make_atomic_action(action_id=i, time_seconds=float(i)) for i in range(5)])
        result = add_pre_shot_gk_context(actions)
        assert len(result) == len(actions)


class TestNonShotRows:
    def test_pass_row_has_default_values(self):
        actions = _df([_make_atomic_pass_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert bool(result["gk_was_distributing"].iloc[0]) is False
        assert bool(result["gk_was_engaged"].iloc[0]) is False
        assert int(result["gk_actions_in_possession"].iloc[0]) == 0
        assert pd.isna(result["defending_gk_player_id"].iloc[0])

    def test_keeper_action_row_has_default_values(self):
        actions = _df([_make_atomic_gk_action(action_id=0, keeper_action="keeper_save")])
        result = add_pre_shot_gk_context(actions)
        assert bool(result["gk_was_engaged"].iloc[0]) is False
        assert pd.isna(result["defending_gk_player_id"].iloc[0])


class TestGkAbsent:
    def test_shot_with_no_gk_history_gets_defaults(self):
        actions = _df([_make_atomic_shot_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert bool(result["gk_was_distributing"].iloc[0]) is False
        assert bool(result["gk_was_engaged"].iloc[0]) is False
        assert int(result["gk_actions_in_possession"].iloc[0]) == 0
        assert pd.isna(result["defending_gk_player_id"].iloc[0])

    def test_only_outfield_actions_before_shot(self):
        actions = _df(
            [
                _make_atomic_pass_action(action_id=0, player_id=200, team_id=100, time_seconds=0.0),
                _make_atomic_pass_action(action_id=1, player_id=201, team_id=100, time_seconds=1.0),
                _make_atomic_pass_action(action_id=2, player_id=202, team_id=100, time_seconds=2.0),
                _make_atomic_shot_action(action_id=3, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot_row = result.iloc[3]
        assert bool(shot_row["gk_was_engaged"]) is False
        assert pd.isna(shot_row["defending_gk_player_id"])


class TestDefendingGkIdentification:
    def test_defending_gk_was_engaged_recently(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_pass_action(action_id=1, player_id=200, team_id=100, time_seconds=1.0),
                _make_atomic_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[2]
        assert bool(shot["gk_was_engaged"]) is True
        assert int(shot["defending_gk_player_id"]) == 999
        assert int(shot["gk_actions_in_possession"]) == 1

    def test_defending_gk_was_distributing(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_pass_action(
                    action_id=1, player_id=999, team_id=100, pass_type="goalkick", time_seconds=2.0
                ),
                _make_atomic_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=4.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[2]
        assert bool(shot["gk_was_distributing"]) is True
        assert bool(shot["gk_was_engaged"]) is True
        assert int(shot["defending_gk_player_id"]) == 999

    def test_shooter_team_gk_actions_excluded(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=888, team_id=200, time_seconds=0.0
                ),
                _make_atomic_shot_action(action_id=1, player_id=700, team_id=200, time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[1]
        assert bool(shot["gk_was_engaged"]) is False
        assert pd.isna(shot["defending_gk_player_id"])


class TestLookbackBounds:
    def test_action_just_outside_lookback_actions_excluded(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                *[
                    _make_atomic_pass_action(action_id=i, player_id=200, team_id=100, time_seconds=float(i))
                    for i in range(1, 7)
                ],
                _make_atomic_shot_action(action_id=7, player_id=700, team_id=200, time_seconds=7.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[7]
        assert bool(shot["gk_was_engaged"]) is False

    def test_action_just_inside_lookback_actions_included(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                *[
                    _make_atomic_pass_action(action_id=i, player_id=200, team_id=100, time_seconds=float(i))
                    for i in range(1, 5)
                ],
                _make_atomic_shot_action(action_id=5, player_id=700, team_id=200, time_seconds=5.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[5]
        assert bool(shot["gk_was_engaged"]) is True

    def test_action_outside_lookback_seconds_excluded(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_pass_action(action_id=1, player_id=200, team_id=100, time_seconds=5.0),
                _make_atomic_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=15.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        assert bool(result.iloc[2]["gk_was_engaged"]) is False

    def test_custom_lookback_actions(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                *[
                    _make_atomic_pass_action(action_id=i, player_id=200, team_id=100, time_seconds=float(i))
                    for i in range(1, 4)
                ],
                _make_atomic_shot_action(action_id=4, player_id=700, team_id=200, time_seconds=4.0),
            ]
        )
        result = add_pre_shot_gk_context(actions, lookback_actions=2, lookback_seconds=100.0)
        assert bool(result.iloc[4]["gk_was_engaged"]) is False

    def test_custom_lookback_seconds(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_shot_action(action_id=1, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions, lookback_seconds=2.0, lookback_actions=10)
        assert bool(result.iloc[1]["gk_was_engaged"]) is False


class TestActionCount:
    def test_multiple_gk_actions_counted(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_gk_action(
                    action_id=1, keeper_action="keeper_pick_up", player_id=999, team_id=100, time_seconds=2.0
                ),
                _make_atomic_gk_action(
                    action_id=2, keeper_action="keeper_punch", player_id=999, team_id=100, time_seconds=4.0
                ),
                _make_atomic_shot_action(action_id=3, player_id=700, team_id=200, time_seconds=5.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[3]
        assert int(shot["gk_actions_in_possession"]) == 3
        assert int(shot["defending_gk_player_id"]) == 999

    def test_only_defending_gk_actions_counted(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=888, team_id=200, time_seconds=0.0
                ),
                _make_atomic_gk_action(
                    action_id=1, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=2.0
                ),
                _make_atomic_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[2]
        assert int(shot["gk_actions_in_possession"]) == 1
        assert int(shot["defending_gk_player_id"]) == 999


class TestMultiGameScoping:
    def test_lookback_does_not_span_games(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0,
                    game_id=1,
                    keeper_action="keeper_save",
                    player_id=999,
                    team_id=100,
                    time_seconds=0.0,
                ),
                _make_atomic_shot_action(action_id=0, game_id=2, player_id=700, team_id=200, time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions).sort_values(["game_id", "action_id"]).reset_index(drop=True)
        shot_row = result[result["game_id"] == 2].iloc[0]
        assert bool(shot_row["gk_was_engaged"]) is False

    def test_lookback_does_not_span_periods(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0,
                    period_id=1,
                    keeper_action="keeper_save",
                    player_id=999,
                    team_id=100,
                    time_seconds=2700.0,
                ),
                _make_atomic_shot_action(action_id=1, period_id=2, player_id=700, team_id=200, time_seconds=10.0),
            ]
        )
        result = add_pre_shot_gk_context(actions, lookback_actions=10, lookback_seconds=10000.0)
        assert bool(result.iloc[1]["gk_was_engaged"]) is False


class TestShotTypes:
    def test_shot_penalty_treated_as_shot(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_shot_action(
                    action_id=1, player_id=700, team_id=200, shot_type="shot_penalty", time_seconds=1.0
                ),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        assert bool(result.iloc[1]["gk_was_engaged"]) is True

    def test_freekick_not_treated_as_shot(self):
        """In atomic, ``freekick`` is the post-collapse name for both pass-class
        and shot-class free kicks. The helper does NOT treat ``freekick`` as a
        shot — atomic users explicitly opt in to that lossy collapse."""
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_action(action_id=1, type_name="freekick", player_id=700, team_id=200, time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        # Freekick is not in atomic shot types → defaults applied (gk_was_engaged=False).
        assert bool(result.iloc[1]["gk_was_engaged"]) is False


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_atomic_action(action_id=0)]).drop(columns=["team_id"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_pre_shot_gk_context(actions)

    def test_negative_lookback_seconds_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match=r"lookback_seconds"):
            add_pre_shot_gk_context(actions, lookback_seconds=-1.0)

    def test_zero_lookback_actions_raises(self):
        actions = _df([_make_atomic_action(action_id=0)])
        with pytest.raises(ValueError, match=r"lookback_actions"):
            add_pre_shot_gk_context(actions, lookback_actions=0)


class TestRealisticScenario:
    def test_save_distribute_attack_shot_sequence(self):
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_atomic_pass_action(
                    action_id=1, player_id=999, team_id=100, pass_type="goalkick", time_seconds=2.0
                ),
                _make_atomic_pass_action(action_id=2, player_id=200, team_id=100, time_seconds=4.0),
                _make_atomic_pass_action(action_id=3, player_id=300, team_id=200, time_seconds=6.0),
                _make_atomic_shot_action(action_id=4, player_id=700, team_id=200, time_seconds=8.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[4]
        # Defending GK 999 had a save + a goalkick distribution within 8s / 5 actions.
        assert bool(shot["gk_was_engaged"]) is True
        assert bool(shot["gk_was_distributing"]) is True
        assert int(shot["gk_actions_in_possession"]) == 1
        assert int(shot["defending_gk_player_id"]) == 999
