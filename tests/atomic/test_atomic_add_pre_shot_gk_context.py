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


# ---------------------------------------------------------------------------
# PR-S21 — frames= kwarg backcompat + tracking integration (atomic)
# ---------------------------------------------------------------------------


class TestAtomicFramesKwargBackcompat:
    def test_frames_none_bit_identical_to_v280(self):
        """Backward-compat: silly-kicks 2.8.0 atomic behavior pinned by golden fixture."""
        from pathlib import Path

        # Reuse standard fixture builder — the atomic helper iterates type_id and
        # ignores standard-only columns gracefully (game_id, period_id, action_id,
        # team_id, player_id, type_id, time_seconds are present in both).
        from tests.spadl._gk_test_fixtures import (
            _df as _std_df,
        )
        from tests.spadl._gk_test_fixtures import (
            _make_action as _std_make_action,
        )
        from tests.spadl._gk_test_fixtures import (
            _make_gk_action as _std_make_gk_action,
        )
        from tests.spadl._gk_test_fixtures import (
            _make_pass_action as _std_make_pass_action,
        )
        from tests.spadl._gk_test_fixtures import (
            _make_shot_action as _std_make_shot_action,
        )

        actions = _std_df(
            [
                _std_make_pass_action(action_id=1, time_seconds=0.0, team_id=200, player_id=701),
                _std_make_gk_action(
                    action_id=2,
                    keeper_action="keeper_save",
                    time_seconds=2.0,
                    team_id=100,
                    player_id=999,
                    start_x=5.0,
                    start_y=34.0,
                ),
                _std_make_pass_action(action_id=3, time_seconds=4.0, team_id=200, player_id=702),
                _std_make_pass_action(action_id=4, time_seconds=6.0, team_id=200, player_id=703),
                _std_make_shot_action(
                    action_id=5,
                    time_seconds=8.0,
                    team_id=200,
                    player_id=704,
                    start_x=95.0,
                    start_y=34.0,
                    shot_type="shot",
                ),
                _std_make_pass_action(action_id=6, time_seconds=12.0, team_id=100, player_id=205),
                _std_make_shot_action(
                    action_id=7,
                    time_seconds=14.0,
                    team_id=100,
                    player_id=206,
                    start_x=10.0,
                    start_y=34.0,
                    shot_type="shot_freekick",
                ),
                _std_make_action(action_id=8, time_seconds=16.0, team_id=200, player_id=708),
            ]
        )
        actual = add_pre_shot_gk_context(actions)
        expected = pd.read_parquet(Path(__file__).parent / "_golden_atomic_pre_shot_gk_context_v280.parquet")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


class TestAtomicFramesKwargWithTracking:
    def test_frames_supplied_emits_8_extra_columns(self):
        # Atomic shot at (90, 34); GK at (104, 34) -> distances 1, 14.
        actions = _df(
            [
                _make_atomic_gk_action(
                    action_id=0,
                    keeper_action="keeper_save",
                    player_id=999,
                    team_id=100,
                    time_seconds=0.0,
                ),
                _make_atomic_shot_action(
                    action_id=1,
                    player_id=704,
                    team_id=200,
                    time_seconds=2.0,
                    x=90.0,
                    y=34.0,
                ),
            ]
        )
        frames = pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2000,
                    time_seconds=2.0,
                    frame_rate=25.0,
                    player_id=999,
                    team_id=100,
                    is_ball=False,
                    is_goalkeeper=True,
                    x=104.0,
                    y=34.0,
                    z=float("nan"),
                    speed=0.5,
                    speed_source="native",
                    ball_state="alive",
                    team_attacking_direction="ltr",
                    confidence=None,
                    visibility=None,
                    source_provider="test",
                ),
            ]
        )
        out = add_pre_shot_gk_context(actions, frames=frames)
        expected_extra = {
            "pre_shot_gk_x",
            "pre_shot_gk_y",
            "pre_shot_gk_distance_to_goal",
            "pre_shot_gk_distance_to_shot",
            "frame_id",
            "time_offset_seconds",
            "n_candidate_frames",
            "link_quality_score",
        }
        assert expected_extra.issubset(set(out.columns))
        # Atomic shot has type_id matching atomic shot ids; should populate.
        shot_row = out[out["action_id"] == 1].iloc[0]
        assert float(shot_row["pre_shot_gk_x"]) == pytest.approx(104.0)
        assert float(shot_row["pre_shot_gk_distance_to_shot"]) == pytest.approx(14.0)


class TestAtomicFramesKwargNoModuleImportCycle:
    def test_no_top_level_atomic_tracking_import_in_atomic_spadl_utils(self):
        """Lazy-import contract per ADR-005 § 5 (atomic mirror): static AST inspection."""
        import ast
        from pathlib import Path

        import silly_kicks.atomic.spadl.utils as _utils

        source = Path(_utils.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("silly_kicks.atomic.tracking"):
                        offenders.append(f"top-level `import {alias.name}` at line {node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("silly_kicks.atomic.tracking"):
                    offenders.append(f"top-level `from {node.module} import ...` at line {node.lineno}")
            elif isinstance(node, ast.If):
                test = node.test
                is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                )
                if is_type_checking:
                    continue
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.ImportFrom)
                        and stmt.module
                        and stmt.module.startswith("silly_kicks.atomic.tracking")
                    ):
                        offenders.append(
                            f"non-TYPE_CHECKING top-level if-block imports {stmt.module} at line {stmt.lineno}"
                        )
        assert not offenders, f"silly_kicks.atomic.spadl.utils violates ADR-005 s 5 lazy-import: {offenders}"
