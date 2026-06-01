"""Shared goal-relative coordinate transforms for tracking features.

A frame is "goal-relative" when the *defended* goal sits at x=0, so that
LTR and RTL frames map to identical feature values (doubling effective data
and removing direction asymmetry). ``goal_x`` is the absolute x of the
defended goal: 0.0 for the goal at the low-x end, 105.0 for the high-x end.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math

FIELD_LENGTH = 105.0
GOAL_Y = 34.0  # pitch half-width (68 / 2) --- goal centre y


def _flip(goal_x: float) -> bool:
    return goal_x > 50.0


def to_goal_relative_x(x: float, *, goal_x: float) -> float:
    """Map absolute pitch x to goal-relative x (defended goal at 0).

    Examples
    --------
    >>> to_goal_relative_x(30.0, goal_x=0.0)
    30.0
    >>> to_goal_relative_x(30.0, goal_x=105.0)
    75.0
    """
    if math.isnan(x):
        return x
    return (FIELD_LENGTH - x) if _flip(goal_x) else x


def to_goal_relative_vx(vx: float, *, goal_x: float) -> float:
    """Map absolute x-velocity to goal-relative x-velocity (negated when flipped).

    Examples
    --------
    >>> to_goal_relative_vx(2.0, goal_x=0.0)
    2.0
    >>> to_goal_relative_vx(2.0, goal_x=105.0)
    -2.0
    """
    if math.isnan(vx):
        return vx
    return -vx if _flip(goal_x) else vx
