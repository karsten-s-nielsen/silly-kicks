"""Orientation / D3 for restdefense (ADR-051 / ADR-055): direction from the GoalMap, never team id.

Two complementary gates:
  * Gate C  -- hold the frames fixed, SWAP the GoalMap ends; the geometry columns must MOVE (proving
    the map is consulted, not that direction is baked into team identity).
  * direction-invariance -- point-reflect the frames AND rebuild the GoalMap from them; the action-LTR
    metrics must be UNCHANGED (the goal-relative geometry is orientation-invariant).
"""

import pandas as pd

from silly_kicks.restdefense import RD_METRIC_COLUMNS, RestDefenseParams
from silly_kicks.restdefense._compute import compute_rest_defense
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture

# Direction-dependent columns that MUST move when the GoalMap is swapped. The two Layer-2 zone-Z
# metrics flip because Z flips ends with the defended goal (space control + gk coverage over Z).
_GATE_C_MUST_MOVE = [
    "rd_line_height",
    "rd_gk_to_line_distance",
    "rd_num_superiority",
    "rd_attacker_space_control",
    "rd_gk_coverage_behind_line",
]
_NUMERIC = [c for c in RD_METRIC_COLUMNS if c != "rd_shape_2_3_vs_3_2"]


def _resolved_by_action(samples):
    r = samples[samples["rd_geometry_source"] == "resolved"].copy()
    r["action_id"] = r["action_id"].astype("int64")
    return r.set_index("action_id").sort_index()


def _swapped_goal_map(frames):
    """A GoalMap with the two teams' defended ends EXCHANGED (built by relabelling team ids)."""
    relabelled = frames.copy()
    tid = relabelled["team_id"]
    relabelled["team_id"] = tid.where(tid.isna(), tid.map({1: 2, 2: 1}).astype("Int64"))
    return resolve_defended_goals(relabelled)


def test_gate_c_swapping_the_goal_map_moves_geometry():
    actions, frames = make_rest_defense_fixture()
    # min_ball_advance_m=0 keeps every sample scored under BOTH maps, so the committed-forward gate
    # (which also reads own_goal_x) cannot confound the geometry-moves assertion -- only the map moves.
    params = RestDefenseParams(min_ball_advance_m=0.0)
    xt = make_fitted_xt()
    correct, _ = compute_rest_defense(actions, frames, xt=xt, params=params)
    swapped, _ = compute_rest_defense(actions, frames, xt=xt, goal_map=_swapped_goal_map(frames), params=params)
    a, b = _resolved_by_action(correct), _resolved_by_action(swapped)
    common = a.index.intersection(b.index)
    assert len(common) >= 1
    for c in _GATE_C_MUST_MOVE:
        assert (a.loc[common, c] != b.loc[common, c]).any(), f"{c} did not move when the GoalMap was swapped"


def test_direction_invariance_under_point_reflection():
    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    base, _ = compute_rest_defense(actions, frames, xt=xt)
    reflected = frames.copy()
    reflected["x"] = 105.0 - reflected["x"]
    reflected["y"] = 68.0 - reflected["y"]
    mirrored, _ = compute_rest_defense(actions, reflected, xt=xt, goal_map=resolve_defended_goals(reflected))
    a, b = _resolved_by_action(base), _resolved_by_action(mirrored)
    assert list(a.index) == list(b.index)
    for c in _NUMERIC:
        pd.testing.assert_series_equal(a[c].reset_index(drop=True), b[c].reset_index(drop=True), check_names=False)
    assert list(a["rd_shape_2_3_vs_3_2"]) == list(b["rd_shape_2_3_vs_3_2"])
