"""Rest-defense geometry (TF-60, ADR-080) -- the danger-behind-line zone.

The rearguard LINE itself is NOT computed here: it is ``compute_defensive_line(frames, goal_map=…,
n=params.n_rearguard).defensive_line_x`` (TF-14, GK-excluded + adaptive-n), computed once per match
in the orchestrator (Task 7) and threaded in. Spec §6 mandates that single source so the danger-zone
boundary and ``rd_line_height`` cannot disagree. This module is pure-stdlib: it only turns a
(line_x, own_goal_x) pair into the oriented danger strip's x-bounds.
"""

from __future__ import annotations


def danger_zone_bounds(line_x: float, own_goal_x: float, *, zone_depth_m: float | None = None) -> tuple[float, float]:
    """``(x_min, x_max)`` of the danger strip between the rearguard line and the OWN goal.

    The strip runs the full pitch width (y in ``[0, 68]``); only the x-band varies. Orientation
    comes from which end the own goal is at (``own_goal_x``), never from team identity. When
    ``zone_depth_m`` is given, the strip is capped to that depth measured FROM the own goal (a cap
    wider than the strip is a no-op, never a widening past the rearguard line).
    """
    lo, hi = (own_goal_x, line_x) if own_goal_x <= line_x else (line_x, own_goal_x)
    if zone_depth_m is not None:
        if own_goal_x <= line_x:  # own goal at low x: keep the strip nearest the goal
            hi = min(hi, own_goal_x + zone_depth_m)
        else:  # own goal at high x
            lo = max(lo, own_goal_x - zone_depth_m)
    return (float(lo), float(hi))
