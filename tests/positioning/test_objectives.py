"""TF-56 positioning objectives (spec section 5)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.positioning import Objective


def test_threat_objective_is_frame_to_float_and_moving_a_defender_changes_it(
    two_team_velocity_frame, fitted_xt, goal_map, attacking_team_id
):
    """Non-vacuity: moving a defender must change the threat surface -- proof that the
    ThreatObjective computes the surface DIRECTLY (PitchControlCache is NOT served; ADR-043)."""
    from silly_kicks.positioning import ThreatObjective

    frames = two_team_velocity_frame
    obj = ThreatObjective(xt=fitted_xt, goal_map=goal_map, attacking_team_id=attacking_team_id)
    assert isinstance(obj, Objective)

    s0 = obj.score(frames)
    assert isinstance(s0, float)

    moved = frames.copy()
    di = moved.index[moved["player_id"] == 10][0]  # deep flank defender: pivotal (see conftest)
    moved.loc[di, "x"] = float(moved.loc[di, "x"]) - 4.0  # deeper toward the x=0 goal
    s1 = obj.score(moved)
    assert s1 != s0, "moving a defender must change the threat surface (PitchControlCache NOT served)"


def test_threat_objective_does_not_mutate_the_frame(two_team_velocity_frame, fitted_xt, goal_map, attacking_team_id):
    from silly_kicks.positioning import ThreatObjective

    frames = two_team_velocity_frame
    before = frames.copy(deep=True)
    ThreatObjective(xt=fitted_xt, goal_map=goal_map, attacking_team_id=attacking_team_id).score(frames)
    pd.testing.assert_frame_equal(frames, before)


class _Const:
    """A constant objective (test double)."""

    def __init__(self, v: float) -> None:
        self.v = v

    def score(self, frame) -> float:
        return self.v


def test_weighted_sum_is_objective_and_combines():
    from silly_kicks.positioning import WeightedSum

    ws = WeightedSum([(_Const(2.0), 1.0), (_Const(4.0), 0.5)])
    assert isinstance(ws, Objective)
    # None is the documented degenerate input (constant terms ignore the frame); the protocol type
    # is DataFrame for real objectives, so the arg-type is suppressed narrowly here.
    assert ws.score(None) == 2.0 * 1.0 + 4.0 * 0.5  # type: ignore[arg-type]


def test_capped_contribution_is_objective_and_caps_whole_value_with_no_agents():
    from silly_kicks.positioning import CappedContribution

    capped = CappedContribution(_Const(100.0), cap=1.0)
    assert isinstance(capped, Objective)
    # No decomposable agents (None frame) -> the whole value is one contribution, clipped at +/-cap.
    assert capped.score(None) == 1.0  # type: ignore[arg-type]  (None is the documented degenerate input)


def test_capped_contribution_recovers_objective_as_cap_grows(one_frame, threat_objective):
    from silly_kicks.positioning import CappedContribution

    # cap -> inf makes the clip a no-op, recovering the wrapped objective EXACTLY.
    huge = CappedContribution(threat_objective, cap=1e9)
    assert huge.score(one_frame) == pytest.approx(threat_objective.score(one_frame))


def test_pressure_objective_is_lower_is_better_frame_to_float(one_frame):
    from silly_kicks.positioning import PressureObjective

    obj = PressureObjective(method="bekkers_pi")
    assert isinstance(obj, Objective)
    s = obj.score(one_frame)
    assert isinstance(s, float)  # == -pressure_on_target(carrier); higher pressure -> lower score


def test_pressure_objective_is_negative_of_pressure_on_carrier(one_frame):
    from silly_kicks.positioning import PressureObjective
    from silly_kicks.positioning._objectives import _nearest_to_ball_player_id
    from silly_kicks.tracking import pressure_on_target

    carrier = _nearest_to_ball_player_id(one_frame)
    expected = -float(pressure_on_target(one_frame, carrier, method="bekkers_pi"))
    assert PressureObjective(method="bekkers_pi").score(one_frame) == expected


def test_das_objective_is_frame_to_float(one_frame):
    pytest.importorskip("accessible_space")
    from silly_kicks.positioning import DasObjective
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    # derive_team_in_possession adds ball_carrier_player_id, so the DAS offside mask excludes the
    # passer (the CORRECT path). Scoring a bare frame otherwise emits a legitimate _das.py notice
    # ("no ball-carrier column ... proceeding without passer exclusion") -- a real correctness signal
    # to fix at the caller, never to suppress.
    frame = derive_team_in_possession(one_frame, infer_ball_carrier(one_frame))
    obj = DasObjective()
    assert isinstance(obj, Objective)
    s = obj.score(frame)
    assert isinstance(s, float)


class _SumX:
    """Objective = sum of non-ball players' x. Leave-one-out marginal of player p is p's x, so a
    tight cap clips each large per-agent contribution -- an explicit, non-vacuous cap test."""

    def score(self, frame) -> float:
        players = frame[~frame["is_ball"].astype(bool)]
        return float(players["x"].dropna().sum())


def test_capped_contribution_limits_a_single_agent_marginal(one_frame, movable_ids):
    from silly_kicks.positioning import CappedContribution

    obj = _SumX()
    total = obj.score(one_frame)  # sum over all non-ball players
    cap = 1.0
    tight = CappedContribution(obj, cap=cap, agents=movable_ids)

    # A tight cap clips each movable defender's (large) marginal -> a measurably lower score.
    capped = tight.score(one_frame)
    assert capped != total

    # Exact: residual (non-movable x) is kept intact; each movable marginal (its x) is clipped to cap.
    players = one_frame[~one_frame["is_ball"].astype(bool)]
    movable_x = players.loc[players["player_id"].isin(movable_ids), "x"].sum()
    residual = total - movable_x
    assert capped == pytest.approx(residual + len(movable_ids) * cap)
