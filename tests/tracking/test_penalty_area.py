"""Canonical penalty-area constants + the two frame-explicit membership predicates (ADR-050).

The three hand-rolled box tests in this repo differ on FRAME (absolute vs goal-relative) and on
which goal is the reference, so a single ``goal_x``-taking helper would mean different things at
different call sites -- an 88.5 m error, not an epsilon. Hence two named entry points.
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._geometry import in_penalty_area_absolute, in_penalty_area_goal_relative


def test_canonical_constants_are_the_law_values():
    assert spadlconfig.penalty_area_half_width == 20.16  # FIFA: 40.32 m wide
    assert spadlconfig.penalty_area_depth == 16.5


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (88.5, 34.0, True),  # EXACTLY on the depth line -> inside (Law: the area includes its lines)
        (88.49, 34.0, False),  # one cm outside the depth line
        (105.0, 13.84, True),  # EXACTLY on the y edge -> inside
        (105.0, 13.83, False),
        (105.0, 54.16, True),  # the mirrored y edge
        (105.0, 54.17, False),
    ],
)
def test_absolute_frame_edges(x, y, expected):
    """A mid-box fixture passes under EVERY wrong convention. Only the edges discriminate."""
    assert in_penalty_area_absolute(x, y, attacked_goal_x=105.0) is expected


def test_absolute_frame_mirrored_goal():
    """The other goal: depth measured from x=0, not x=105."""
    assert in_penalty_area_absolute(16.5, 34.0, attacked_goal_x=0.0) is True
    assert in_penalty_area_absolute(16.51, 34.0, attacked_goal_x=0.0) is False
    assert in_penalty_area_absolute(88.5, 34.0, attacked_goal_x=0.0) is False


@pytest.mark.parametrize(
    ("gr_x", "y", "expected"),
    [(16.5, 34.0, True), (16.51, 34.0, False), (0.0, 13.84, True), (0.0, 13.83, False)],
)
def test_goal_relative_frame_edges(gr_x, y, expected):
    """Takes NO goal argument: the caller resolved attacked-vs-defended by producing gr_x, so the
    ambiguity cannot re-enter the helper."""
    assert in_penalty_area_goal_relative(gr_x, y) is expected


def test_migration_is_byte_identical_for_both_20_16_sites():
    """Both sites were already non-strict on x with the same abs(y-34) form, so the canonical
    constants must reproduce them exactly. Grid-sweep, not spot-check -- and BOTH sites: the
    scalar one and the vectorized one can diverge independently."""
    import silly_kicks.tracking._geometry as _geo
    from silly_kicks.tracking._xcross_attempt import _BOX_DEPTH_M, _BOX_HALF_WIDTH_M
    from silly_kicks.tracking.defensive_credit._params import _is_inside_attacked_box

    xs = np.arange(80.0, 120.01, 0.25)
    ys = np.arange(10.0, 58.01, 0.25)

    # -- site 1: defensive_credit, ABSOLUTE frame
    for x in xs:
        for y in ys:
            old = (x >= 105.0 - 16.5) and (abs(y - 34.0) <= 20.16)
            assert bool(_is_inside_attacked_box(float(x), float(y))) is bool(old), (x, y)

    # -- site 2: xCross, GOAL-RELATIVE frame, vectorized. Reproduce the shipped expression against
    #    the post-migration constants over the same grid.
    gr_x = np.abs(105.0 - xs)[:, None]
    yy = ys[None, :]
    new = (gr_x <= _BOX_DEPTH_M) & (np.abs(yy - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M)
    old_vec = (gr_x <= 16.5) & (np.abs(yy - 34.0) <= 20.16)
    assert np.array_equal(new, old_vec)


def test_absolute_helper_diverges_only_beyond_the_reachable_pitch():
    """``in_penalty_area_absolute`` uses gr_x = abs(105 - x), so it has an UPPER bound the old
    ``x >= 105 - 16.5`` form does not -- they disagree for x > 121.5 (i.e. >16.5 m PAST the goal
    line).

    DOCUMENTED, not proven unreachable. The nearest cap is ``_SPADL_X_MAX = 120.0``
    (``_gk_identification.py``, raised inside ``derive_goalkeepers``) -- but that validates
    TRACKING FRAME coords, whereas this helper's only production caller is ``defensive_credit``,
    which works on SPADL ACTION coords. Different path; no equivalent validation is known to guard
    it. So: the tracking path caps x at 120.0 and 120.0 < 121.5; the action path is not shown to
    be validated, and this test records the divergence rather than dismissing it. If the cap
    moves, the assertion below fails and the behind-the-goal semantics get decided explicitly
    instead of inherited.
    """
    from silly_kicks.tracking._gk_identification import _SPADL_X_MAX

    assert in_penalty_area_absolute(121.4, 34.0, attacked_goal_x=105.0) is True
    assert in_penalty_area_absolute(121.6, 34.0, attacked_goal_x=105.0) is False  # old form: True
    assert _SPADL_X_MAX < 121.5, (
        f"the x cap moved to {_SPADL_X_MAX}; the abs() divergence is now reachable and the "
        f"behind-the-goal semantics must be decided, not inherited"
    )
