"""Tests for ``silly_kicks.spadl.utils.add_possessions`` (added in 1.2.0).

The helper assigns a per-match possession-sequence integer to each SPADL
action via a team-change-with-carve-outs heuristic (locked design 1.2.0):

  - Sort actions by ``(game_id, period_id, action_id)``.
  - Possession boundary on team change, EXCEPT a set-piece restart
    (``freekick_*``, ``corner_*``, ``throw_in``, ``goalkick``) following
    the opposing team's foul retains the previous possession (the team
    that won the foul resumes its sequence).
  - Possession boundary forced at period change (within a game) and at
    time gap >= ``max_gap_seconds`` regardless of team.
  - Counter resets to 0 at each new ``game_id``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_possessions, boundary_metrics

# ---------------------------------------------------------------------------
# Test fixture builder
# ---------------------------------------------------------------------------

_ACT = spadlconfig.actiontype_id
_RES = spadlconfig.result_id
_BP = spadlconfig.bodypart_id


def _make_action(
    *,
    action_id: int,
    game_id: int = 1,
    period_id: int = 1,
    time_seconds: float = 0.0,
    team_id: int = 100,
    player_id: int = 200,
    type_name: str = "pass",
    result_name: str = "success",
    bodypart_name: str = "foot",
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    original_event_id: object = None,
) -> dict[str, object]:
    """Build a minimal valid SPADL action row."""
    return {
        "game_id": game_id,
        "original_event_id": str(action_id) if original_event_id is None else original_event_id,
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": time_seconds,
        "team_id": team_id,
        "player_id": player_id,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "type_id": _ACT[type_name],
        "result_id": _RES[result_name],
        "bodypart_id": _BP[bodypart_name],
    }


def _df(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a SPADL DataFrame from a list of action dicts."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Contract: shape + dtype + column preservation
# ---------------------------------------------------------------------------


class TestAddPossessionsContract:
    def test_returns_dataframe(self):
        actions = _df([_make_action(action_id=0)])
        result = add_possessions(actions)
        assert isinstance(result, pd.DataFrame)

    def test_adds_possession_id_column(self):
        actions = _df([_make_action(action_id=0)])
        result = add_possessions(actions)
        assert "possession_id" in result.columns

    def test_possession_id_dtype_is_int64(self):
        actions = _df([_make_action(action_id=0), _make_action(action_id=1, time_seconds=1.0)])
        result = add_possessions(actions)
        assert result["possession_id"].dtype == np.int64

    def test_preserves_all_input_columns(self):
        actions = _df([_make_action(action_id=0)])
        actions["custom_col"] = "preserved"
        result = add_possessions(actions)
        assert "custom_col" in result.columns
        assert list(result["custom_col"]) == ["preserved"]

    def test_empty_input_returns_empty_with_possession_id(self):
        actions = _df([_make_action(action_id=0)]).iloc[0:0]
        result = add_possessions(actions)
        assert "possession_id" in result.columns
        assert len(result) == 0

    def test_does_not_mutate_input(self):
        actions = _df([_make_action(action_id=0)])
        cols_before = list(actions.columns)
        add_possessions(actions)
        assert list(actions.columns) == cols_before, "input df should not be mutated"

    def test_returns_same_row_count(self):
        actions = _df([_make_action(action_id=i, time_seconds=float(i)) for i in range(10)])
        result = add_possessions(actions)
        assert len(result) == len(actions)


# ---------------------------------------------------------------------------
# Same team → same possession
# ---------------------------------------------------------------------------


class TestSameTeamSamePossession:
    def test_two_same_team_passes_share_possession_id(self):
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]

    def test_long_chain_same_team_one_possession(self):
        actions = _df([_make_action(action_id=i, team_id=100, time_seconds=float(i)) for i in range(20)])
        result = add_possessions(actions)
        assert result["possession_id"].nunique() == 1
        assert (result["possession_id"] == 0).all()


# ---------------------------------------------------------------------------
# Team change → new possession
# ---------------------------------------------------------------------------


class TestTeamChangeNewPossession:
    def test_team_change_increments_possession_id(self):
        # Pass by team A then pass by team B (no set-piece carve-out, no foul) → new possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1

    def test_alternating_teams_unique_possession_ids(self):
        actions = _df([_make_action(action_id=i, team_id=100 + (i % 2) * 100, time_seconds=float(i)) for i in range(6)])
        result = add_possessions(actions)
        # 6 alternating actions → 6 unique possession_ids 0..5
        assert list(result["possession_id"]) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Period change within game → new possession (counter increments, no reset)
# ---------------------------------------------------------------------------


class TestPeriodChange:
    def test_period_change_within_game_increments_counter(self):
        # Same team, new period → still new possession (period restart is a possession boundary).
        actions = _df(
            [
                _make_action(action_id=0, period_id=1, team_id=100, time_seconds=2700.0),
                _make_action(action_id=1, period_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1, "period change should start a new possession"

    def test_period_does_not_reset_counter(self):
        # Three possessions in period 1, then period 2 → period 2 starts at possession_id=3, not 0.
        actions = _df(
            [
                _make_action(action_id=0, period_id=1, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, period_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, period_id=1, team_id=100, time_seconds=2.0),
                _make_action(action_id=3, period_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 2  # third possession
        assert result["possession_id"].iloc[3] == 3  # period 2 → new possession, counter continues


# ---------------------------------------------------------------------------
# Game change → counter resets to 0
# ---------------------------------------------------------------------------


class TestGameChange:
    def test_game_change_resets_counter_to_zero(self):
        actions = _df(
            [
                _make_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, game_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=0, game_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 0, "new game must reset counter"

    def test_multi_game_counter_independent(self):
        # Game 1: 2 alternating possessions. Game 2: 1 possession. Counter independent per game.
        actions = _df(
            [
                _make_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, game_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=0, game_id=2, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, game_id=2, team_id=100, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions)
        per_game = result.groupby("game_id")["possession_id"].agg(["min", "max"])
        assert per_game.loc[1, "min"] == 0
        assert per_game.loc[1, "max"] == 1
        assert per_game.loc[2, "min"] == 0
        assert per_game.loc[2, "max"] == 0


# ---------------------------------------------------------------------------
# Time gap → new possession even if same team
# ---------------------------------------------------------------------------


class TestMaxGapTimeout:
    def test_long_gap_starts_new_possession_same_team(self):
        # Same team but >5s gap → still new possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=10.0),  # 10s gap
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1

    def test_default_5s_threshold_just_under_keeps_possession(self):
        # 4.9s gap < 5.0s default → same possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=4.9),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]

    def test_custom_max_gap_seconds(self):
        # Custom 1.0s threshold → 1.5s gap triggers new possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=1.5),
            ]
        )
        result = add_possessions(actions, max_gap_seconds=1.0)
        assert result["possession_id"].iloc[0] != result["possession_id"].iloc[1]


# ---------------------------------------------------------------------------
# Set-piece carve-out — foul → opposing-team set-piece retains possession
# ---------------------------------------------------------------------------


class TestSetPieceCarveOut:
    def test_freekick_after_opposing_foul_retains_possession(self):
        # Pass by team B (possession 0), foul by team A (still possession 0 — same team change rule
        # would say new possession but it's still the same SPADL-level possession we're tracking),
        # free kick by team B (carve-out: same possession as before the foul).
        # Actually with team change A->B and current team B taking free kick → carve-out triggers.
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _make_action(
                    action_id=2,
                    team_id=200,
                    time_seconds=2.0,
                    type_name="freekick_short",
                ),
            ]
        )
        result = add_possessions(actions)
        # action 0: pass by team B, possession 0
        # action 1: foul by team A — team change → new possession 1
        # action 2: free kick by team B — team change BUT carve-out applies (prev was foul by other team)
        #   → SAME possession as action 1 (possession 1, NOT new possession 2)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 1, (
            "free kick after opposing-team foul should retain the previous possession"
        )

    def test_carve_out_disabled(self):
        # Same scenario but retain_on_set_pieces=False → free kick is a new possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="freekick_short"),
            ]
        )
        result = add_possessions(actions, retain_on_set_pieces=False)
        # All three possessions distinct.
        assert result["possession_id"].iloc[2] == 2, "carve-out disabled → free kick is new possession"

    def test_corner_after_opposing_clearance_does_not_retain(self):
        # Clearance != foul → carve-out does NOT apply → new possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="clearance"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="corner_short"),
            ]
        )
        result = add_possessions(actions)
        # Clearance is not a foul → no carve-out → corner is a new possession.
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 2

    def test_throw_in_after_foul_retains(self):
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="throw_in"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 1

    def test_goalkick_after_foul_retains(self):
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="goalkick"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 1

    def test_carve_out_does_not_apply_to_normal_pass(self):
        # If current action is a regular pass (not a set piece), team change → new possession even after foul.
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 2, "regular pass is not a set-piece restart"


# ---------------------------------------------------------------------------
# Monotonicity invariants
# ---------------------------------------------------------------------------


class TestMonotonic:
    def test_possession_id_monotonic_within_game(self):
        # Build a 30-action mixed sequence; possession_id must be monotonic non-decreasing within game.
        rng = np.random.default_rng(42)
        rows = []
        teams = [100, 200]
        for i in range(30):
            team = int(teams[rng.integers(0, 2)])
            rows.append(_make_action(action_id=i, team_id=team, time_seconds=float(i), type_name="pass"))
        actions = _df(rows)
        result = add_possessions(actions)
        # Possession IDs must be monotonic non-decreasing within each game when sorted.
        ids = result.sort_values(["game_id", "period_id", "action_id"])["possession_id"].to_numpy()
        diffs = np.diff(ids)
        assert (diffs >= 0).all(), f"possession_id must be non-decreasing within game; got diffs {diffs}"

    def test_possession_id_starts_at_zero_per_game(self):
        actions = _df(
            [
                _make_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_action(action_id=0, game_id=2, team_id=200, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions)
        per_game_first = (
            result.sort_values(["game_id", "period_id", "action_id"]).groupby("game_id")["possession_id"].first()
        )
        assert per_game_first.loc[1] == 0
        assert per_game_first.loc[2] == 0


# ---------------------------------------------------------------------------
# Combined scenarios — boundary interactions
# ---------------------------------------------------------------------------


class TestCombinedBoundaries:
    def test_period_change_overrides_set_piece_carve_out(self):
        # Foul at end of period 1, free kick at start of period 2 → period change forces new possession
        # regardless of carve-out.
        actions = _df(
            [
                _make_action(
                    action_id=0,
                    period_id=1,
                    team_id=100,
                    time_seconds=2700.0,
                    type_name="foul",
                    result_name="fail",
                ),
                _make_action(action_id=1, period_id=2, team_id=200, time_seconds=0.0, type_name="freekick_short"),
            ]
        )
        result = add_possessions(actions)
        # Period change forces increment regardless of carve-out — they yield the same answer here
        # but the design intent is that period change wins.
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1

    def test_long_gap_overrides_set_piece_carve_out(self):
        # >5s gap between foul and free kick → long-gap rule forces new possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _make_action(action_id=2, team_id=200, time_seconds=10.0, type_name="freekick_short"),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[2] == 2, "long gap should force new possession"


# ---------------------------------------------------------------------------
# Sort order — input not pre-sorted
# ---------------------------------------------------------------------------


class TestSortOrder:
    def test_unsorted_input_is_sorted_internally(self):
        # Provide actions in reverse order; result should still be canonical.
        actions = _df(
            [
                _make_action(action_id=2, team_id=200, time_seconds=2.0),
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=1.0),
            ]
        )
        result = add_possessions(actions).sort_values("action_id").reset_index(drop=True)
        # Action 0 and 1 are same team → same possession
        # Action 2 is different team → new possession
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]
        assert result["possession_id"].iloc[2] != result["possession_id"].iloc[0]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_required_column_raises(self):
        # Drop game_id — required for the algorithm.
        actions = _df([_make_action(action_id=0)]).drop(columns=["game_id"])
        with pytest.raises(ValueError, match=r"missing|required"):
            add_possessions(actions)

    def test_negative_max_gap_seconds_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match="max_gap_seconds"):
            add_possessions(actions, max_gap_seconds=-1.0)


# ---------------------------------------------------------------------------
# Rule 1: brief-opposing-action merge — added in 2.1.0
# ---------------------------------------------------------------------------


class TestBriefOpposingMerge:
    """``merge_brief_opposing_actions`` + ``brief_window_seconds`` pair.

    Suppresses team-change boundaries when team B has 1..N consecutive
    actions sandwiched between team A actions within ``brief_window_seconds``.
    Both kwargs must be > 0 to enable; both 0 to disable.
    """

    def test_aba_within_window_suppresses_boundary(self):
        # A (team 100) at t=0, B (team 200) at t=1.0, A (team 100) at t=2.0.
        # Window measured from first B-action (t=1.0) to next A-action (t=2.0):
        # 1.0s <= 2.0s threshold. With N=1, T=2.0: detect 1 sandwiched B,
        # suppress both team-change boundaries.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, team_id=100, time_seconds=2.0),
            ]
        )
        result = add_possessions(actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0, "B at t=1 should be merged"
        assert result["possession_id"].iloc[2] == 0, "A at t=2 should be merged"

    def test_aba_outside_window_keeps_boundary(self):
        # Same A B A pattern but B's window exceeded — boundaries stand.
        # Window measured from first B (t=1.0) to A (t=3.0) = 2.0s, but threshold
        # is 1.5 — so 2.0 > 1.5 → no merge.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, team_id=100, time_seconds=3.0),
            ]
        )
        result = add_possessions(actions, merge_brief_opposing_actions=1, brief_window_seconds=1.5)
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 2

    def test_abba_within_window_with_n_eq_2_suppresses(self):
        # A, B, B, A within window — 2 consecutive B's, N=2 covers it.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=0.5),
                _make_action(action_id=2, team_id=200, time_seconds=1.0),
                _make_action(action_id=3, team_id=100, time_seconds=1.5),
            ]
        )
        result = add_possessions(actions, merge_brief_opposing_actions=2, brief_window_seconds=2.0)
        assert result["possession_id"].nunique() == 1, "all merged into one possession"

    def test_abbba_with_n_eq_2_keeps_boundary(self):
        # A, B, B, B, A — 3 consecutive B's. With N=2 the rule looks ahead at
        # most 2 rows; row 1's lookahead k=1 sees B (no), k=2 sees B (no);
        # no match → boundary at row 1 stands. Row 4 (B→A) has no further
        # lookahead → boundary stands.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=0.5),
                _make_action(action_id=2, team_id=200, time_seconds=1.0),
                _make_action(action_id=3, team_id=200, time_seconds=1.5),
                _make_action(action_id=4, team_id=100, time_seconds=2.0),
            ]
        )
        result = add_possessions(actions, merge_brief_opposing_actions=2, brief_window_seconds=3.0)
        # Three distinct possessions: A, B, A
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[4] == 2

    def test_disabled_when_both_zero(self):
        # Default values (both 0) → identical to no-rule baseline.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, team_id=100, time_seconds=2.0),
            ]
        )
        with_rule_off = add_possessions(actions, merge_brief_opposing_actions=0, brief_window_seconds=0.0)
        baseline = add_possessions(actions)
        assert (with_rule_off["possession_id"] == baseline["possession_id"]).all()

    def test_partial_config_actions_only_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"both"):
            add_possessions(actions, merge_brief_opposing_actions=2, brief_window_seconds=0.0)

    def test_partial_config_seconds_only_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"both"):
            add_possessions(actions, merge_brief_opposing_actions=0, brief_window_seconds=2.0)

    def test_negative_actions_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"merge_brief_opposing_actions"):
            add_possessions(actions, merge_brief_opposing_actions=-1, brief_window_seconds=2.0)

    def test_negative_seconds_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"brief_window_seconds"):
            add_possessions(actions, merge_brief_opposing_actions=1, brief_window_seconds=-0.5)

    def test_game_boundary_blocks_lookahead(self):
        # Brief-merge must not cross game_id boundaries.
        actions = _df(
            [
                _make_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, game_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=0, game_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0)
        # Game 1 has 2 distinct possessions (A then B). Game 2 has 1 possession (resets to 0).
        assert result["possession_id"].iloc[0] == 0  # game 1 first possession
        assert result["possession_id"].iloc[1] == 1  # game 1 second possession (no merge across game)
        assert result["possession_id"].iloc[2] == 0  # game 2 starts fresh

    def test_period_boundary_blocks_lookahead(self):
        # Same intent across period boundary.
        actions = _df(
            [
                _make_action(action_id=0, period_id=1, team_id=100, time_seconds=2700.0),
                _make_action(action_id=1, period_id=1, team_id=200, time_seconds=2700.5),
                _make_action(action_id=2, period_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0)
        # Period change forces boundary regardless of brief-merge.
        assert result["possession_id"].iloc[2] == 2


# ---------------------------------------------------------------------------
# Rule 2: defensive transitions — added in 2.1.0
# ---------------------------------------------------------------------------


class TestDefensiveTransitions:
    """``defensive_transition_types`` rule.

    Action types listed do NOT trigger team-change boundaries on their own.
    Recommended types per measurement: ``interception``, ``clearance``.
    """

    def test_interception_does_not_trigger_boundary(self):
        # A passes, B intercepts, B passes → expected to merge into one possession.
        # With defensive=("interception",): the team-change boundary at the intercept
        # is suppressed; the next row (B-pass) has no team_change vs the intercept,
        # so no boundary fires there either. Result: all rows in possession 0.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="interception"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions, defensive_transition_types=("interception",))
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0
        assert result["possession_id"].iloc[2] == 0

    def test_pass_after_interception_keeps_separate_possession_when_recovered(self):
        # A pass, B intercept, A pass. Intercept's team-change boundary is
        # suppressed by the rule, but the A-pass at row 2 IS a team change vs
        # the intercept (B→A) and is NOT itself defensive — boundary fires.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="interception"),
                _make_action(action_id=2, team_id=100, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions, defensive_transition_types=("interception",))
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0  # intercept boundary suppressed
        assert result["possession_id"].iloc[2] == 1  # A pass after B intercept = new possession

    def test_unknown_type_raises_value_error(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"unknown action types"):
            add_possessions(actions, defensive_transition_types=("not_a_type",))

    def test_empty_tuple_no_op(self):
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
            ]
        )
        with_rule = add_possessions(actions, defensive_transition_types=())
        baseline = add_possessions(actions)
        assert (with_rule["possession_id"] == baseline["possession_id"]).all()

    def test_multi_type_tuple(self):
        # ("interception", "clearance") → both types suppress team-change boundary.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="clearance"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(actions, defensive_transition_types=("interception", "clearance"))
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0
        assert result["possession_id"].iloc[2] == 0


# ---------------------------------------------------------------------------
# max_gap_seconds default change 5.0 → 7.0 — added in 2.1.0
# ---------------------------------------------------------------------------


class TestMaxGapDefaultIs7Seconds:
    """The ``max_gap_seconds`` default changed from 5.0 to 7.0 in 2.1.0.

    Behavior break: same-team actions with time gap in [5, 7) seconds are
    now in the same possession (previously a new possession at gap >= 5).
    """

    def test_default_value_is_7(self):
        import inspect

        sig = inspect.signature(add_possessions)
        assert sig.parameters["max_gap_seconds"].default == 7.0

    def test_5_to_6s_gap_no_boundary_at_default(self):
        # Same team, gap = 6.0s. Under 5.0 default this would be new possession;
        # under 7.0 default this is same possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=6.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]

    def test_7_to_8s_gap_boundary_at_default(self):
        # Same team, gap = 7.5s — new possession under 7.0 default (>= 7.0).
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=7.5),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] != result["possession_id"].iloc[1]

    def test_explicit_5_0_still_works(self):
        # Opt-out path: explicitly set max_gap_seconds=5.0.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=6.0),
            ]
        )
        result = add_possessions(actions, max_gap_seconds=5.0)
        assert result["possession_id"].iloc[0] != result["possession_id"].iloc[1]


# ---------------------------------------------------------------------------
# End-to-end: add_possessions vs StatsBomb native possession_id
# ---------------------------------------------------------------------------


class TestBoundaryAgainstStatsBombNative:
    """Validate add_possessions against StatsBomb's native possession_id.

    PR-S12 (silly-kicks 2.1.0) updated the empirical baselines to the
    new ``max_gap_seconds=7.0`` default: per-match boundary recall ~0.94
    and boundary F1 ~0.60. The precision gap remains intrinsic to the
    team-change-with-carve-outs algorithm class. The CI gate below tests
    recall AND precision because both are observable behaviors that
    downstream consumers can develop dependencies on — F1 conflates two
    signals with very different magnitudes and is recorded for diagnostics
    only.

    Fixtures (committed under ``tests/datasets/statsbomb/raw/events/``)
    are 3 diverse matches measured during the luxury-lakehouse PR-LL2
    boundary-metrics campaign:

    - 7298  — Women's World Cup
    - 7584  — Champions League
    - 3754058 — Premier League

    Per-match independent gates: any single match falling below
    ``recall >= 0.83 AND precision >= 0.30`` fires the regression.

    Companion test: :class:`TestBoundaryAgainstStatsBomb64Match` runs the
    same gate across 64 FIFA WorldCup-2018 matches via the committed
    HDF5 fixture. Cross-competition coverage here; within-competition
    variance there.
    """

    @pytest.mark.parametrize("match_id", [7298, 7584, 3754058])
    def test_boundary_metrics_against_native_possession_id(self, match_id: int):
        import json
        import os

        fixture_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "statsbomb", "raw", "events", f"{match_id}.json"
        )
        if not os.path.exists(fixture_path):
            pytest.fail(
                f"StatsBomb fixture not found at {fixture_path}. "
                f"This test requires committed fixtures under tests/datasets/statsbomb/raw/events/ — "
                f"see tests/datasets/statsbomb/README.md for vendoring details."
            )

        from silly_kicks.spadl import statsbomb

        with open(fixture_path, encoding="utf-8") as f:
            events_raw = json.load(f)

        # Adapter: StatsBomb open data top-level keys → silly-kicks EXPECTED_INPUT_COLUMNS.
        _top_level_keys = {"id", "period", "timestamp", "team", "player", "type", "location"}
        adapted = pd.DataFrame(
            [
                {
                    "game_id": match_id,
                    "event_id": e.get("id"),
                    "period_id": e.get("period"),
                    "timestamp": e.get("timestamp"),
                    "team_id": (e.get("team") or {}).get("id"),
                    "player_id": (e.get("player") or {}).get("id"),
                    "type_name": (e.get("type") or {}).get("name"),
                    "location": e.get("location"),
                    "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
                    # preserve_native target: top-level possession sequence number.
                    "possession": e.get("possession"),
                }
                for e in events_raw
            ]
        )

        actions, _report = statsbomb.convert_to_actions(
            adapted,
            home_team_id=int(adapted["team_id"].dropna().iloc[0]),
            preserve_native=["possession"],
        )

        # Keep only non-synthetic rows. Synthetic dribbles inserted by
        # _add_dribbles have possession=NaN (no source event to inherit
        # from); we want to compare heuristic vs native only where both
        # are defined. .copy() avoids SettingWithCopyWarning on the
        # subsequent add_possessions call.
        non_synthetic = actions[actions["possession"].notna()].copy()
        non_synthetic = add_possessions(non_synthetic)

        m = boundary_metrics(
            heuristic=non_synthetic["possession_id"],
            native=non_synthetic["possession"].astype(np.int64),
        )

        # Per-match independent gates (PR-S12, silly-kicks 2.1.0).
        # Recall floor: 0.83 — 4pp below worst observed (R_min=0.854 at gap=7.0
        # across 64 WC-2018 matches; 3 committed JSON fixtures have higher
        # R_min). Loosened from 0.85 in 2.1.0 alongside the max_gap_seconds
        # default change 5.0→7.0.
        # Precision floor: 0.30 — 5pp below worst observed (P_min=0.350 at gap=7.0).
        # F1 in message only — gating on it conflates two independent signals.
        assert m["recall"] >= 0.83 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.83 precision>=0.30"
        )


# ---------------------------------------------------------------------------
# Within-competition variance: 64 WC-2018 matches via committed HDF5 fixture
# ---------------------------------------------------------------------------


class TestBoundaryAgainstStatsBomb64Match:
    """Validate add_possessions against StatsBomb native across 64 WC-2018 matches.

    Reads the committed HDF5 fixture
    ``tests/datasets/statsbomb/spadl-WorldCup-2018.h5`` (regenerated in
    silly-kicks 2.1.0 to preserve the StatsBomb ``possession`` column).
    Each match is gated independently at ``recall >= 0.83 AND
    precision >= 0.30`` per the same per-match contract as
    :class:`TestBoundaryAgainstStatsBombNative`.

    Within-competition variance complement to the cross-competition
    3-fixture test. ~1-2s additional CI runtime after HDFStore cold-load.
    """

    def test_fixture_has_possession_column(self, sb_worldcup_data: pd.HDFStore):
        # Sentinel: if the HDF5 was built without preserve_native=["possession"],
        # the entire test class will fail at this guard. Clear failure mode
        # vs 64 cryptic per-match KeyErrors.
        keys = [k for k in sb_worldcup_data.keys() if k.startswith("/actions/game_")]
        assert keys, "no actions/game_<id> keys in HDF5 fixture"
        first = sb_worldcup_data.get(keys[0])
        assert "possession" in first.columns, (
            "HDF5 fixture missing `possession` column. Regenerate with "
            "`uv run python scripts/build_worldcup_fixture.py --verbose`."
        )

    @pytest.mark.parametrize(
        "match_id",
        # The 64 FIFA WorldCup-2018 match IDs — hardcoded from the manifest
        # because pytest.parametrize cannot defer to a fixture for IDs. If the
        # manifest changes upstream, regenerate this list from
        # tests/datasets/statsbomb/raw/.cache/events/*.json filenames.
        [
            7525,
            7529,
            7530,
            7531,
            7532,
            7533,
            7534,
            7535,
            7536,
            7537,
            7538,
            7539,
            7540,
            7541,
            7542,
            7543,
            7544,
            7545,
            7546,
            7547,
            7548,
            7549,
            7550,
            7551,
            7552,
            7553,
            7554,
            7555,
            7556,
            7557,
            7558,
            7559,
            7560,
            7561,
            7562,
            7563,
            7564,
            7565,
            7566,
            7567,
            7568,
            7569,
            7570,
            7571,
            7572,
            7576,
            7577,
            7578,
            7579,
            7580,
            7581,
            7582,
            7583,
            7584,
            7585,
            7586,
            8649,
            8650,
            8651,
            8652,
            8655,
            8656,
            8657,
            8658,
        ],
    )
    def test_boundary_metrics_against_native_per_match(self, sb_worldcup_data: pd.HDFStore, match_id: int):
        actions = sb_worldcup_data.get(f"actions/game_{match_id}")
        non_synth = actions[actions["possession"].notna()].copy()
        non_synth = add_possessions(non_synth)

        m = boundary_metrics(
            heuristic=non_synth["possession_id"],
            native=non_synth["possession"].astype(np.int64),
        )

        assert m["recall"] >= 0.83 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.83 precision>=0.30"
        )


# ---------------------------------------------------------------------------
# Unit tests for the public boundary_metrics utility (added in 1.8.0)
# ---------------------------------------------------------------------------


class TestBoundaryMetricsContract:
    def test_returns_dict_with_required_keys(self):
        m = boundary_metrics(
            heuristic=pd.Series([0, 0, 1, 1]),
            native=pd.Series([0, 0, 1, 1]),
        )
        assert set(m.keys()) == {"precision", "recall", "f1"}

    def test_all_metric_values_are_floats(self):
        m = boundary_metrics(
            heuristic=pd.Series([0, 0, 1, 1]),
            native=pd.Series([0, 0, 1, 1]),
        )
        assert isinstance(m["precision"], float)
        assert isinstance(m["recall"], float)
        assert isinstance(m["f1"], float)

    def test_keyword_only_args_required(self):
        # Positional invocation must raise TypeError. The args are asymmetric
        # (swapping inputs swaps precision and recall), so positional usage is
        # a silent footgun we eliminate at the API surface.
        with pytest.raises(TypeError):
            boundary_metrics(pd.Series([0, 0, 1]), pd.Series([0, 0, 1]))  # type: ignore[misc]


class TestBoundaryMetricsCorrectness:
    def test_identical_sequences_all_metrics_one(self):
        s = pd.Series([0, 0, 1, 1, 2, 2])
        m = boundary_metrics(heuristic=s, native=s)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_relabeled_identical_sequences_all_metrics_one(self):
        # Boundaries are invariant under counter-relabeling: [0,0,1,1] and
        # [5,5,7,7] emit boundaries at the same row. Both should report
        # perfect agreement.
        h = pd.Series([0, 0, 1, 1, 2, 2])
        n = pd.Series([5, 5, 7, 7, 9, 9])
        m = boundary_metrics(heuristic=h, native=n)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_completely_disjoint_boundaries_all_zero(self):
        # Heuristic emits boundary at idx 2; native emits at idx 1.
        # No overlap → TP=0, FP=1, FN=1 → precision=recall=f1=0.
        h = pd.Series([0, 0, 1, 1])
        n = pd.Series([0, 1, 1, 1])
        m = boundary_metrics(heuristic=h, native=n)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_partial_overlap_hand_computed(self):
        # Heuristic boundaries at idx 1, 2, 4 (3 total)
        # Native boundaries at idx 1, 4 (2 total)
        # Shared boundaries at idx 1, 4 → TP=2, FP=1 (idx 2), FN=0
        # precision = 2/(2+1) = 2/3 ≈ 0.6667
        # recall = 2/(2+0) = 1.0
        # f1 = 2 * 0.6667 * 1.0 / (0.6667 + 1.0) = 0.8
        h = pd.Series([0, 1, 2, 2, 3])
        n = pd.Series([0, 1, 1, 1, 2])
        m = boundary_metrics(heuristic=h, native=n)
        assert abs(m["precision"] - 2 / 3) < 1e-9
        assert m["recall"] == 1.0
        assert abs(m["f1"] - 0.8) < 1e-9


class TestBoundaryMetricsDegenerate:
    def test_empty_sequences_returns_zeros(self):
        m = boundary_metrics(
            heuristic=pd.Series([], dtype=np.int64),
            native=pd.Series([], dtype=np.int64),
        )
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_single_row_returns_zeros(self):
        # Single-row sequences have no consecutive pairs → no boundaries
        # detectable in either series → degenerate → all zeros.
        m = boundary_metrics(
            heuristic=pd.Series([0]),
            native=pd.Series([0]),
        )
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_constant_sequences_returns_zeros(self):
        # All-constant sequences have zero boundaries → degenerate → all zeros.
        m = boundary_metrics(
            heuristic=pd.Series([0, 0, 0, 0, 0]),
            native=pd.Series([7, 7, 7, 7, 7]),
        )
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0


class TestBoundaryMetricsErrors:
    def test_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match=r"length"):
            boundary_metrics(
                heuristic=pd.Series([0, 0, 1, 1]),
                native=pd.Series([0, 0, 1]),
            )
