"""Tests for ``silly_kicks.spadl.utils.add_pre_shot_gk_context`` (1.4.0).

For each shot action, look back up to (``lookback_actions`` OR
``lookback_seconds``, smaller wins) within the same ``(game_id, period_id)``
and tag the defending GK's recent activity:

  - ``gk_was_distributing`` — bool, defending GK had a non-keeper action
  - ``gk_was_engaged`` — bool, defending GK had a ``keeper_*`` action
  - ``gk_actions_in_possession`` — int, count of ``keeper_*`` by defending GK
  - ``defending_gk_player_id`` — int (NaN when defending GK absent)

Locked design (Q3.1, Q3.2, Q3.3, Q3.4):

  - Q3.1: defaults to False/0/NaN when defending GK absent in lookback window
  - Q3.2: smaller of (lookback_actions, lookback_seconds) wins
  - Q3.3: independent — no dependency on add_possessions
  - Q3.4: features only populated on shot rows (non-shot rows = defaults)

Note: this is a novel approach — no published OSS / academic equivalent.
The defending GK is identified as the most recent ``keeper_*`` action by a
team OTHER than the shooter's team within the lookback window.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from tests.spadl._gk_test_fixtures import (
    _df,
    _make_action,
    _make_gk_action,
    _make_pass_action,
    _make_shot_action,
)

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_returns_dataframe(self):
        actions = _df([_make_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_four_columns(self):
        actions = _df([_make_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert "gk_was_distributing" in result.columns
        assert "gk_was_engaged" in result.columns
        assert "gk_actions_in_possession" in result.columns
        assert "defending_gk_player_id" in result.columns

    def test_preserves_input_columns(self):
        actions = _df([_make_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_pre_shot_gk_context(actions)
        assert "custom_col" in result.columns

    def test_does_not_mutate_input(self):
        actions = _df([_make_action(action_id=0)])
        cols_before = list(actions.columns)
        add_pre_shot_gk_context(actions)
        assert list(actions.columns) == cols_before

    def test_empty_input(self):
        actions = _df([_make_action(action_id=0)]).iloc[0:0]
        result = add_pre_shot_gk_context(actions)
        assert "gk_was_distributing" in result.columns
        assert len(result) == 0

    def test_returns_same_row_count(self):
        actions = _df([_make_action(action_id=i, time_seconds=float(i)) for i in range(5)])
        result = add_pre_shot_gk_context(actions)
        assert len(result) == len(actions)


# ---------------------------------------------------------------------------
# Non-shot rows: defaults
# ---------------------------------------------------------------------------


class TestNonShotRows:
    def test_pass_row_has_default_values(self):
        actions = _df([_make_pass_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert bool(result["gk_was_distributing"].iloc[0]) is False
        assert bool(result["gk_was_engaged"].iloc[0]) is False
        assert int(result["gk_actions_in_possession"].iloc[0]) == 0
        assert pd.isna(result["defending_gk_player_id"].iloc[0])

    def test_keeper_action_row_has_default_values(self):
        actions = _df([_make_gk_action(action_id=0, keeper_action="keeper_save")])
        result = add_pre_shot_gk_context(actions)
        assert bool(result["gk_was_engaged"].iloc[0]) is False
        assert pd.isna(result["defending_gk_player_id"].iloc[0])


# ---------------------------------------------------------------------------
# GK absence (Q3.1) — defaults when defending GK has no actions in lookback
# ---------------------------------------------------------------------------


class TestGkAbsent:
    def test_shot_with_no_gk_history_gets_defaults(self):
        # Just a shot; no preceding actions.
        actions = _df([_make_shot_action(action_id=0)])
        result = add_pre_shot_gk_context(actions)
        assert bool(result["gk_was_distributing"].iloc[0]) is False
        assert bool(result["gk_was_engaged"].iloc[0]) is False
        assert int(result["gk_actions_in_possession"].iloc[0]) == 0
        assert pd.isna(result["defending_gk_player_id"].iloc[0])

    def test_only_outfield_actions_before_shot(self):
        # 3 outfield passes by team 100 → shot by team 200.
        actions = _df(
            [
                _make_pass_action(action_id=0, player_id=200, team_id=100, time_seconds=0.0),
                _make_pass_action(action_id=1, player_id=201, team_id=100, time_seconds=1.0),
                _make_pass_action(action_id=2, player_id=202, team_id=100, time_seconds=2.0),
                _make_shot_action(action_id=3, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        # No keeper_* actions anywhere → all defaults on the shot row.
        shot_row = result.iloc[3]
        assert bool(shot_row["gk_was_engaged"]) is False
        assert pd.isna(shot_row["defending_gk_player_id"])


# ---------------------------------------------------------------------------
# Defending GK identification
# ---------------------------------------------------------------------------


class TestDefendingGkIdentification:
    def test_defending_gk_was_engaged_recently(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_pass_action(action_id=1, player_id=200, team_id=100, time_seconds=1.0),
                _make_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[2]
        assert bool(shot["gk_was_engaged"]) is True
        assert int(shot["defending_gk_player_id"]) == 999
        assert int(shot["gk_actions_in_possession"]) == 1

    def test_defending_gk_was_distributing(self):
        # GK saves, then distributes via a goalkick → opposing shot.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_pass_action(action_id=1, player_id=999, team_id=100, pass_type="goalkick", time_seconds=2.0),
                _make_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=4.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[2]
        assert bool(shot["gk_was_distributing"]) is True
        assert bool(shot["gk_was_engaged"]) is True
        assert int(shot["defending_gk_player_id"]) == 999

    def test_shooter_team_gk_actions_excluded(self):
        # GK of team 200 (the shooter's own team) has actions; should NOT count.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=888, team_id=200, time_seconds=0.0),
                _make_shot_action(action_id=1, player_id=700, team_id=200, time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[1]
        # Defending team is 100, no GK actions by team 100 → defaults.
        assert bool(shot["gk_was_engaged"]) is False
        assert pd.isna(shot["defending_gk_player_id"])


# ---------------------------------------------------------------------------
# Lookback bounds (Q3.2) — smaller of lookback_actions / lookback_seconds wins
# ---------------------------------------------------------------------------


class TestLookbackBounds:
    def test_action_just_outside_lookback_actions_excluded(self):
        # Default lookback_actions=5; 6 actions back should be excluded.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                *[
                    _make_pass_action(action_id=i, player_id=200, team_id=100, time_seconds=float(i))
                    for i in range(1, 7)
                ],
                _make_shot_action(action_id=7, player_id=700, team_id=200, time_seconds=7.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[7]
        # GK save was 7 actions back → outside default lookback_actions=5 → defaults.
        assert bool(shot["gk_was_engaged"]) is False

    def test_action_just_inside_lookback_actions_included(self):
        # GK save 5 actions back is at lookback_actions=5 (boundary, included).
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                *[
                    _make_pass_action(action_id=i, player_id=200, team_id=100, time_seconds=float(i))
                    for i in range(1, 5)
                ],
                _make_shot_action(action_id=5, player_id=700, team_id=200, time_seconds=5.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[5]
        # GK save was 5 actions back → at the boundary, included.
        assert bool(shot["gk_was_engaged"]) is True

    def test_action_outside_lookback_seconds_excluded(self):
        # GK save was only 2 actions back BUT >10s ago → excluded.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_pass_action(action_id=1, player_id=200, team_id=100, time_seconds=5.0),
                _make_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=15.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        # 15s after the GK action → beyond default lookback_seconds=10.0
        assert bool(result.iloc[2]["gk_was_engaged"]) is False

    def test_custom_lookback_actions(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                *[
                    _make_pass_action(action_id=i, player_id=200, team_id=100, time_seconds=float(i))
                    for i in range(1, 4)
                ],
                _make_shot_action(action_id=4, player_id=700, team_id=200, time_seconds=4.0),
            ]
        )
        # With lookback_actions=2, the GK save (4 back) is excluded.
        result = add_pre_shot_gk_context(actions, lookback_actions=2, lookback_seconds=100.0)
        assert bool(result.iloc[4]["gk_was_engaged"]) is False

    def test_custom_lookback_seconds(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_shot_action(action_id=1, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        # With lookback_seconds=2.0, the 3s-old GK save is excluded.
        result = add_pre_shot_gk_context(actions, lookback_seconds=2.0, lookback_actions=10)
        assert bool(result.iloc[1]["gk_was_engaged"]) is False


# ---------------------------------------------------------------------------
# Counting GK actions in window
# ---------------------------------------------------------------------------


class TestActionCount:
    def test_multiple_gk_actions_counted(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_gk_action(
                    action_id=1, keeper_action="keeper_pick_up", player_id=999, team_id=100, time_seconds=2.0
                ),
                _make_gk_action(
                    action_id=2, keeper_action="keeper_punch", player_id=999, team_id=100, time_seconds=4.0
                ),
                _make_shot_action(action_id=3, player_id=700, team_id=200, time_seconds=5.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[3]
        assert int(shot["gk_actions_in_possession"]) == 3
        assert int(shot["defending_gk_player_id"]) == 999

    def test_only_defending_gk_actions_counted(self):
        # Both teams' GKs have actions; only the defending team's count.
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=888, team_id=200, time_seconds=0.0),
                _make_gk_action(action_id=1, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=2.0),
                _make_shot_action(action_id=2, player_id=700, team_id=200, time_seconds=3.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[2]
        # Defending team is 100, so only player 999's actions count → 1.
        assert int(shot["gk_actions_in_possession"]) == 1
        assert int(shot["defending_gk_player_id"]) == 999


# ---------------------------------------------------------------------------
# Multi-game / multi-period scoping
# ---------------------------------------------------------------------------


class TestMultiGameScoping:
    def test_lookback_does_not_span_games(self):
        actions = _df(
            [
                _make_gk_action(
                    action_id=0, game_id=1, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0
                ),
                _make_shot_action(action_id=0, game_id=2, player_id=700, team_id=200, time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions).sort_values(["game_id", "action_id"]).reset_index(drop=True)
        # The shot in game 2 must NOT see the GK save from game 1.
        shot_row = result[result["game_id"] == 2].iloc[0]
        assert bool(shot_row["gk_was_engaged"]) is False

    def test_lookback_does_not_span_periods(self):
        actions = _df(
            [
                _make_gk_action(
                    action_id=0,
                    period_id=1,
                    keeper_action="keeper_save",
                    player_id=999,
                    team_id=100,
                    time_seconds=2700.0,
                ),
                _make_shot_action(action_id=1, period_id=2, player_id=700, team_id=200, time_seconds=10.0),
            ]
        )
        result = add_pre_shot_gk_context(actions, lookback_actions=10, lookback_seconds=10000.0)
        # Period change → no carryover.
        assert bool(result.iloc[1]["gk_was_engaged"]) is False


# ---------------------------------------------------------------------------
# Different shot types
# ---------------------------------------------------------------------------


class TestShotTypes:
    def test_shot_freekick_treated_as_shot(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_shot_action(action_id=1, player_id=700, team_id=200, shot_type="shot_freekick", time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        assert bool(result.iloc[1]["gk_was_engaged"]) is True

    def test_shot_penalty_treated_as_shot(self):
        actions = _df(
            [
                _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=0.0),
                _make_shot_action(action_id=1, player_id=700, team_id=200, shot_type="shot_penalty", time_seconds=1.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        assert bool(result.iloc[1]["gk_was_engaged"]) is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_required_column_raises(self):
        actions = _df([_make_action(action_id=0)]).drop(columns=["team_id"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_pre_shot_gk_context(actions)

    def test_negative_lookback_seconds_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"lookback_seconds"):
            add_pre_shot_gk_context(actions, lookback_seconds=-1.0)

    def test_zero_lookback_actions_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"lookback_actions"):
            add_pre_shot_gk_context(actions, lookback_actions=0)


# ---------------------------------------------------------------------------
# Realistic combined scenario
# ---------------------------------------------------------------------------


class TestRealisticScenario:
    def test_save_distribute_attack_shot_sequence(self):
        # Save → distribute → opposing build-up → shot.
        actions = _df(
            [
                _make_gk_action(
                    action_id=0, keeper_action="keeper_save", player_id=999, team_id=100, time_seconds=10.0
                ),
                _make_pass_action(
                    action_id=1, player_id=999, team_id=100, pass_type="goalkick", time_seconds=12.0
                ),  # GK distribution
                _make_pass_action(action_id=2, player_id=200, team_id=100, time_seconds=14.0),  # team 100 transitions
                _make_pass_action(action_id=3, player_id=300, team_id=200, time_seconds=16.0),  # team 200 wins ball
                _make_shot_action(action_id=4, player_id=700, team_id=200, time_seconds=17.0),
            ]
        )
        result = add_pre_shot_gk_context(actions)
        shot = result.iloc[4]
        # Defending team's GK (player 999) was engaged (action 0) AND distributed (action 1).
        # Both within 7-second window from shot at t=17.
        assert bool(shot["gk_was_engaged"]) is True
        assert bool(shot["gk_was_distributing"]) is True
        assert int(shot["defending_gk_player_id"]) == 999
        # 1 keeper_* action by player 999 within window (the save at action 0).
        assert int(shot["gk_actions_in_possession"]) == 1
