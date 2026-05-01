"""Tests for silly_kicks.tracking.feature_framework + frame_aware marker.

Loop 1 covers: frame_aware decorator, is_frame_aware predicate, ActionFrameContext
frozen dataclass shape, type aliases.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest


def test_frame_aware_marker_sets_attribute():
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def my_xfn(states, frames):
        return pd.DataFrame()

    assert my_xfn._frame_aware is True


def test_is_frame_aware_returns_true_for_marked():
    from silly_kicks.vaep.feature_framework import frame_aware, is_frame_aware

    @frame_aware
    def marked(states, frames):
        return pd.DataFrame()

    assert is_frame_aware(marked) is True


def test_is_frame_aware_returns_false_for_unmarked():
    from silly_kicks.vaep.feature_framework import is_frame_aware

    def regular(states):
        return pd.DataFrame()

    assert is_frame_aware(regular) is False


def test_is_frame_aware_returns_false_for_lambda_without_attr():
    from silly_kicks.vaep.feature_framework import is_frame_aware

    assert is_frame_aware(lambda x: x) is False


def test_frames_type_alias_exists():
    from silly_kicks.vaep import feature_framework as ff

    assert ff.Frames is pd.DataFrame


def test_action_frame_context_is_frozen_dataclass():
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    assert dataclasses.is_dataclass(ActionFrameContext)
    params = ActionFrameContext.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen is True


def test_action_frame_context_has_required_fields():
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    field_names = {f.name for f in dataclasses.fields(ActionFrameContext)}
    expected = {"actions", "pointers", "actor_rows", "opposite_rows_per_action"}
    assert expected.issubset(field_names)


def test_action_frame_context_has_defending_gk_rows_field():
    """ActionFrameContext exposes a defending_gk_rows: pd.DataFrame field per ADR-005 (PR-S21)."""
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    field_names = {f.name for f in dataclasses.fields(ActionFrameContext)}
    assert "defending_gk_rows" in field_names


@pytest.fixture
def tiny_actions_and_frames():
    """3 actions + 3 frames at known times; 4 players + ball per frame."""
    actions = pd.DataFrame(
        {
            "action_id": [101, 102, 103],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 20.0, 30.0],
            "team_id": [1, 1, 2],
            "player_id": [11, 12, 21],
            "start_x": [50.0, 60.0, 40.0],
            "start_y": [34.0, 30.0, 38.0],
            "end_x": [55.0, 65.0, 45.0],
            "end_y": [34.0, 30.0, 38.0],
        }
    )
    # 3 frames at t=10.0/20.0/30.0; each frame has 4 players (2 per team) + 1 ball row
    rows = []
    for fid, t in [(1000, 10.0), (2000, 20.0), (3000, 30.0)]:
        for pid, tid, x, y in [(11, 1, 50.0, 34.0), (12, 1, 60.0, 30.0), (21, 2, 40.0, 38.0), (22, 2, 70.0, 35.0)]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "time_seconds": t,
                    "frame_rate": 25.0,
                    "player_id": pid,
                    "team_id": tid,
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": x,
                    "y": y,
                    "z": float("nan"),
                    "speed": 1.5,
                    "speed_source": "native",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "test",
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "frame_rate": 25.0,
                "player_id": float("nan"),
                "team_id": float("nan"),
                "is_ball": True,
                "is_goalkeeper": False,
                "x": 52.5,
                "y": 34.0,
                "z": 0.0,
                "speed": 5.0,
                "speed_source": "native",
                "ball_state": "alive",
                "team_attacking_direction": None,
                "confidence": None,
                "visibility": None,
                "source_provider": "test",
            }
        )
    frames = pd.DataFrame(rows)
    return actions, frames


def test_resolve_action_frame_context_links_all_actions(tiny_actions_and_frames):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)

    assert len(ctx.pointers) == 3
    assert ctx.pointers["frame_id"].notna().all()


def test_resolve_action_frame_context_actor_rows_one_per_action(tiny_actions_and_frames):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)

    # actor_rows: one row per action_id with the actor's frame data
    assert len(ctx.actor_rows) == 3
    assert set(ctx.actor_rows["action_id"]) == {101, 102, 103}


def test_resolve_action_frame_context_opposite_excludes_actor_team(tiny_actions_and_frames):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)

    # action_id 101 is team_id=1; opposite rows should all be team_id=2
    opp_101 = ctx.opposite_rows_per_action[ctx.opposite_rows_per_action["action_id"] == 101]
    assert (opp_101["team_id_frame"] == 2).all()
    # 2 team-2 players + 1 ball, but ball is excluded — 2 rows
    assert len(opp_101) == 2


def test_resolve_action_frame_context_unlinked_action():
    """Action with no frame within tolerance -> NaN pointer + empty opposite_rows."""
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions = pd.DataFrame(
        {
            "action_id": [999],
            "period_id": [1],
            "time_seconds": [1000.0],  # no frame at 1000s
            "team_id": [1],
            "player_id": [11],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [55.0],
            "end_y": [34.0],
        }
    )
    frames = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "frame_id": [1000],
            "time_seconds": [10.0],
            "frame_rate": [25.0],
            "player_id": [11],
            "team_id": [1],
            "is_ball": [False],
            "is_goalkeeper": [False],
            "x": [50.0],
            "y": [34.0],
            "z": [float("nan")],
            "speed": [1.5],
            "speed_source": ["native"],
            "ball_state": ["alive"],
            "team_attacking_direction": ["ltr"],
            "confidence": [None],
            "visibility": [None],
            "source_provider": ["test"],
        }
    )
    ctx = _resolve_action_frame_context(actions, frames)
    assert pd.isna(ctx.pointers["frame_id"].iloc[0])


@pytest.fixture
def tiny_actions_and_frames_with_gk(tiny_actions_and_frames):
    """tiny_actions_and_frames with a defending-GK player_id flagged on action 101.

    Action 101 is taken by team 1; the defending GK is team-2 player 22 (already
    in the frame at (70, 35)). The other actions get NaN defending_gk_player_id.
    """
    actions, frames = tiny_actions_and_frames
    actions = actions.copy()
    actions["defending_gk_player_id"] = [22.0, float("nan"), float("nan")]
    return actions, frames


def test_resolve_action_frame_context_populates_defending_gk_rows_when_player_id_matches(
    tiny_actions_and_frames_with_gk,
):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames_with_gk
    ctx = _resolve_action_frame_context(actions, frames)
    gk_for_action_101 = ctx.defending_gk_rows[ctx.defending_gk_rows["action_id"] == 101]
    assert len(gk_for_action_101) == 1
    assert float(gk_for_action_101["x"].iloc[0]) == pytest.approx(70.0)
    assert float(gk_for_action_101["y"].iloc[0]) == pytest.approx(35.0)


def test_resolve_action_frame_context_excludes_action_with_nan_defending_gk_player_id(
    tiny_actions_and_frames_with_gk,
):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames_with_gk
    ctx = _resolve_action_frame_context(actions, frames)
    gk_for_action_102 = ctx.defending_gk_rows[ctx.defending_gk_rows["action_id"] == 102]
    assert len(gk_for_action_102) == 0


def test_resolve_action_frame_context_defending_gk_rows_empty_when_column_absent(
    tiny_actions_and_frames,
):
    """Backward-compat: PR-S20 callers without defending_gk_player_id get empty rows."""
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)
    assert len(ctx.defending_gk_rows) == 0


def test_resolve_action_frame_context_excludes_ball_from_defending_gk_rows(tiny_actions_and_frames):
    """Even if a ball row had matching player_id (it shouldn't), is_ball=True excludes it."""
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    actions = actions.copy()
    # Force defending_gk_player_id to NaN (ball player_id) for all rows; result should be empty.
    actions["defending_gk_player_id"] = float("nan")
    ctx = _resolve_action_frame_context(actions, frames)
    assert len(ctx.defending_gk_rows) == 0


def test_lift_to_states_marks_output_frame_aware(tiny_actions_and_frames):
    from silly_kicks.tracking.feature_framework import lift_to_states
    from silly_kicks.vaep.feature_framework import is_frame_aware

    def stub(actions, frames):
        return pd.Series([1.0] * len(actions), index=actions.index)

    stub.__name__ = "stub"

    lifted = lift_to_states(stub, nb_states=3)
    assert is_frame_aware(lifted) is True


def test_lift_to_states_produces_a0_a1_a2_columns(tiny_actions_and_frames):
    from silly_kicks.tracking.feature_framework import lift_to_states
    from silly_kicks.vaep.feature_framework import gamestates

    actions, frames = tiny_actions_and_frames
    actions = actions.assign(game_id=1)

    def stub(actions, frames):
        return pd.Series([1.0] * len(actions), index=actions.index)

    stub.__name__ = "stub"

    states = gamestates(actions, nb_prev_actions=3)
    lifted = lift_to_states(stub, nb_states=3)
    out = lifted(states, frames)

    assert list(out.columns) == ["stub_a0", "stub_a1", "stub_a2"]
    assert len(out) == len(actions)


def test_lift_to_states_a0_matches_direct_call(tiny_actions_and_frames):
    """The _a0 column should equal the helper called on states[0] directly."""
    from silly_kicks.tracking.feature_framework import lift_to_states
    from silly_kicks.vaep.feature_framework import gamestates

    actions, frames = tiny_actions_and_frames
    actions = actions.assign(game_id=1)

    def stub_increasing(actions, frames):
        return pd.Series(range(len(actions)), index=actions.index, dtype="float64")

    stub_increasing.__name__ = "stub_inc"

    states = gamestates(actions, nb_prev_actions=3)
    lifted = lift_to_states(stub_increasing, nb_states=3)
    out = lifted(states, frames)

    expected = stub_increasing(states[0], frames).to_numpy()
    actual = out["stub_inc_a0"].to_numpy()
    import numpy as np

    np.testing.assert_array_equal(actual, expected)
