"""select_rest_defense_samples (TF-60, ADR-080) -- action-grid sampling + gates + loss instant."""

import pandas as pd

from silly_kicks.restdefense._config import RestDefenseParams
from silly_kicks.restdefense._windows import GATE_DROP_REASONS, select_rest_defense_samples
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_rest_defense_fixture

_KEYS = ["game_id", "period_id", "team_id", "action_id"]
_EXPECTED_COLS = [
    *_KEYS,
    "possession_id",
    "frame_id",
    "ball_x",
    "own_goal_x",
    "attacked_goal_x",
    "is_possession_loss",
    "gate_drop_reason",
]


def test_selects_advanced_in_possession_actions_and_flags_loss():
    actions, frames = make_rest_defense_fixture()
    gm = resolve_defended_goals(frames)
    s = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    scored = s[s["gate_drop_reason"].isna()]
    assert len(scored) >= 2  # home + away advanced actions
    assert scored["is_possession_loss"].any()  # the terminal loss is flagged
    # the non-advanced action is dropped, not scored:
    assert (s["gate_drop_reason"] == "not_committed_forward").any()
    # conservation: every input in-possession on-ball action is either scored or dropped exactly once
    assert len(s) == len(s.drop_duplicates(subset=_KEYS))


def test_output_schema_and_dtypes():
    actions, frames = make_rest_defense_fixture()
    gm = resolve_defended_goals(frames)
    s = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    assert list(s.columns) == _EXPECTED_COLS
    assert s["is_possession_loss"].dtype == bool
    # one output row per input action (full-population conservation)
    assert len(s) == len(actions)


def test_scored_rows_have_both_orientations():
    actions, frames = make_rest_defense_fixture()
    gm = resolve_defended_goals(frames)
    s = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    scored = s[s["gate_drop_reason"].isna()]
    # a home-possession (team 1, own goal x=0) and an away-possession (team 2, own goal x=105) sample
    assert (scored["own_goal_x"] == 0.0).any()
    assert (scored["own_goal_x"] == 105.0).any()
    # attacked_goal is a real opponent lookup, never 105 - own
    home = scored[scored["own_goal_x"] == 0.0].iloc[0]
    assert home["attacked_goal_x"] == 105.0


def test_committed_forward_gate_boundary():
    actions, frames = make_rest_defense_fixture()
    gm = resolve_defended_goals(frames)
    # a3: home ball at x=30, own goal x=0 -> advance 30 m. Gate at 25 keeps it; at 40 drops it.
    keep = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams(min_ball_advance_m=25.0))
    drop = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams(min_ball_advance_m=40.0))
    a3_keep = keep[keep["action_id"] == 3].iloc[0]
    a3_drop = drop[drop["action_id"] == 3].iloc[0]
    assert pd.isna(a3_keep["gate_drop_reason"])
    assert a3_drop["gate_drop_reason"] == "not_committed_forward"


def test_unlinked_action_is_dropped_and_counted():
    actions, frames = make_rest_defense_fixture()
    # move a0 far in time from any frame -> unlinked
    actions.loc[actions["action_id"] == 0, "time_seconds"] = 999.0
    gm = resolve_defended_goals(frames)
    s = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    a0 = s[s["action_id"] == 0].iloc[0]
    assert a0["gate_drop_reason"] == "unlinked"
    assert pd.isna(a0["frame_id"])


def test_all_drop_reasons_are_in_the_declared_vocabulary():
    actions, frames = make_rest_defense_fixture()
    gm = resolve_defended_goals(frames)
    s = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    seen = set(s["gate_drop_reason"].dropna().unique())
    assert seen <= set(GATE_DROP_REASONS)


def test_purity_does_not_mutate_inputs():
    actions, frames = make_rest_defense_fixture()
    a_before, f_before = actions.copy(), frames.copy()
    gm = resolve_defended_goals(frames)
    select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)
