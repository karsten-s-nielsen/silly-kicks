"""Shared goal-relative coordinate transforms for tracking features.

A frame is "goal-relative" when the *defended* goal sits at x=0, so that
LTR and RTL frames map to identical feature values (doubling effective data
and removing direction asymmetry). ``goal_x`` is the absolute x of the
defended goal: 0.0 for the goal at the low-x end, 105.0 for the high-x end.

That identity holds because the two ends differ by a 180-degree POINT REFLECTION
``(x, y) -> (105 - x, 68 - y)`` -- a ROTATION, not an x mirror. It is enforced by
``tests/tracking/test_pr5_chirality_gates.py``. Before PR 5 the y counterpart was MISSING, so
``goal_x=105`` was an x-only mirror (determinant -1) against an identity (+1) at the other end:
the claim above was FALSE for every signed-y feature, with every radial byte-identical and every
bearing negated (measured: xS 12 of 27 features, xCross 3 of 16). Adding a signed-y quantity
without routing it through :func:`to_goal_relative_y` reintroduces exactly that defect.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math

import silly_kicks.spadl.config as _spadlconfig

FIELD_LENGTH = 105.0
GOAL_Y = 34.0  # pitch half-width (68 / 2) --- goal centre y

PITCH_LENGTH = FIELD_LENGTH  # 105.0 m --- physical pitch length the goal-relative features assume
PITCH_WIDTH = GOAL_Y * 2.0  # 68.0 m
# Bump when the goal-relative transform's NUMERIC output changes (NOT for a pure origin
# translation like TF-38, which is invariant). Consumed by trained-model metadata as the
# coordinate-change fail-closed guard. See the TF-16 weights spec S6.
GEOMETRY_VERSION = "goal-relative-2"  # PR 5: x-only mirror -> 180-degree point reflection


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


def to_goal_relative_y(y: float, *, goal_x: float) -> float:
    """Map absolute pitch y to goal-relative y (mirrored when the defended goal is at high x).

    Paired with :func:`to_goal_relative_x` this is the 180-degree POINT REFLECTION
    ``(x, y) -> (105 - x, 68 - y)``, so the two goal ends differ by a ROTATION rather than a
    reflection. Before PR 5 there was no y counterpart: ``goal_x=105`` was an x-only mirror
    (determinant -1) while ``goal_x=0`` was the identity (+1), so the ends used frames of OPPOSITE
    handedness -- every RADIAL feature stayed byte-identical and every BEARING negated (measured:
    xS 12 of 27 features, xCross 3 of 16).

    Examples
    --------
    >>> to_goal_relative_y(20.0, goal_x=0.0)
    20.0
    >>> to_goal_relative_y(20.0, goal_x=105.0)
    48.0
    """
    if math.isnan(y):
        return y
    return (PITCH_WIDTH - y) if _flip(goal_x) else y


def to_goal_relative_vy(vy: float, *, goal_x: float) -> float:
    """Map absolute y-velocity to goal-relative y-velocity (negated when flipped).

    Added for symmetry with :func:`to_goal_relative_vx`. NOTE that BOTH are currently unused in
    production: no shipped feature consumes a directional velocity (xS's ``bvx``/``bvy`` and
    xCross's feed only ``hypot``), so neither is exercised by the PR 5 feature-identity gate.

    Examples
    --------
    >>> to_goal_relative_vy(2.0, goal_x=0.0)
    2.0
    >>> to_goal_relative_vy(2.0, goal_x=105.0)
    -2.0
    """
    if math.isnan(vy):
        return vy
    return -vy if _flip(goal_x) else vy


def in_penalty_area_goal_relative(gr_x: float, y: float) -> bool:
    """Penalty-area membership in GOAL-RELATIVE coords (the reference goal sits at ``gr_x = 0``).

    Takes NO goal argument on purpose: the caller has already resolved attacked-vs-defended by
    producing ``gr_x``, so that ambiguity cannot re-enter here. Boundary is non-strict on both
    axes -- the Law's area includes its own lines.

    Examples
    --------
    >>> in_penalty_area_goal_relative(16.5, 34.0)
    True
    >>> in_penalty_area_goal_relative(16.51, 34.0)
    False
    """
    # NOTE: no lower bound on gr_x, DELIBERATELY. The shipped xCross predicate is
    # `gr_x <= _BOX_DEPTH_M` with no `0 <= gr_x` guard, and real tracking carries x beyond the
    # goal line (gr_x < 0), so adding one would CHANGE xCross behaviour for behind-the-line
    # players. Whether a behind-the-line point should count as in-box is a separate, measurable
    # question -- not this cycle's.
    return bool((gr_x <= _spadlconfig.penalty_area_depth) and (abs(y - GOAL_Y) <= _spadlconfig.penalty_area_half_width))


def in_penalty_area_absolute(x: float, y: float, *, attacked_goal_x: float) -> bool:
    """Penalty-area membership in ABSOLUTE (action-LTR) coords.

    ``attacked_goal_x`` is the absolute x of the goal whose area is being tested (0.0 or 105.0).
    Named to avoid colliding with this module's ``goal_x``, which means the *defended* goal in the
    to-goal-relative transforms above.

    Examples
    --------
    >>> in_penalty_area_absolute(88.5, 34.0, attacked_goal_x=105.0)
    True
    >>> in_penalty_area_absolute(88.49, 34.0, attacked_goal_x=105.0)
    False
    """
    gr_x = abs(float(attacked_goal_x) - float(x))
    return in_penalty_area_goal_relative(gr_x, y)
