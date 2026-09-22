"""TF-56 positioning reachability constraint (spec section 5)."""

from __future__ import annotations

from silly_kicks.positioning import Constraint, ReachabilityConstraint, ReachabilityParams


def _pos(frame, pid):
    row = frame.loc[frame["player_id"] == pid].iloc[0]
    return float(row["x"]), float(row["y"])


def test_reachability_is_a_constraint(one_frame):
    assert isinstance(ReachabilityConstraint(ReachabilityParams.default()), Constraint)


def test_reachability_rejects_beyond_horizon_accepts_within(one_frame):
    c = ReachabilityConstraint(ReachabilityParams(max_reach_seconds=0.7))
    pid = 10  # deep flank defender at (12, 30), velocity toward goal
    px, py = _pos(one_frame, pid)
    near = (px - 1.0, py)  # ~1 m toward goal -> tti ~0.36 s, reachable
    far = (px - 40.0, py)  # unreachable in 0.7 s
    assert c.is_feasible(pid, near, one_frame) is True
    assert c.is_feasible(pid, far, one_frame) is False


def test_tighter_horizon_shrinks_feasible_set(one_frame):
    pid = 10
    px, py = _pos(one_frame, pid)
    cand = (px - 3.0, py)  # ~3 m toward goal -> tti ~0.69 s
    loose = ReachabilityConstraint(ReachabilityParams(max_reach_seconds=1.5))
    tight = ReachabilityConstraint(ReachabilityParams(max_reach_seconds=0.3))
    assert loose.is_feasible(pid, cand, one_frame) is True
    assert tight.is_feasible(pid, cand, one_frame) is False


def test_unknown_player_is_infeasible(one_frame):
    c = ReachabilityConstraint(ReachabilityParams.default())
    assert c.is_feasible(9999, (10.0, 34.0), one_frame) is False
