"""The array box predicate must agree with the scalar one, everywhere.

This is the DURABLE artifact of the ADR-050 §6 migration. The one-off grid sweep in
``test_xcross_box_migration_identity.py`` is a characterization test against an expression being
deleted -- once xCross migrates, the thing it compared against is gone. This test is what
permanently pins the two forms together.
"""

from __future__ import annotations

import numpy as np
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.tracking._geometry import (
    GOAL_Y,
    in_penalty_area_goal_relative,
    in_penalty_area_goal_relative_array,
)


def _ulp_neighbourhood(x: float, n: int = 50) -> list[float]:
    """The ``2n+1`` doubles centred on ``x``.

    MUST WALK. The obvious comprehension --
    ``[np.nextafter(x, inf if d > 0 else -inf) for d in range(-n, n + 1)]`` -- returns the SAME two
    neighbours over and over: measured 101 entries collapsing to **3 distinct doubles** at n=50, and
    an outcome-based non-vacuity check (any/not-all) cannot see it. That would gut the one dimension
    this grid exists to cover, since bound equality does NOT imply predicate equality.
    """
    out = [x]
    lo = hi = x
    for _ in range(n):
        lo = np.nextafter(lo, -np.inf)
        out.append(float(lo))
        hi = np.nextafter(hi, np.inf)
        out.append(float(hi))
    return out


def _grid() -> tuple[np.ndarray, np.ndarray]:
    half = spadlconfig.penalty_area_half_width
    depth = spadlconfig.penalty_area_depth
    ys: list[float] = [0.0, GOAL_Y, 68.0, 13.85, 13.84, 54.16, 54.15]
    for c in (GOAL_Y - half, GOAL_Y + half):
        ys.extend(_ulp_neighbourhood(c))
    # The depth boundary needs the SAME treatment: the `<` -> `<=` change lives exactly at
    # `gr_x == depth`, so ULP-walking only the y bounds would leave that contributor covered by
    # three hand-picked points.
    xs: list[float] = [-5.0, -0.001, 0.0, 5.0, 16.49, 16.51, 120.0]
    xs.extend(_ulp_neighbourhood(depth))
    gx, gy = np.meshgrid(np.array(xs, dtype=float), np.array(ys, dtype=float))
    return gx.ravel(), gy.ravel()


def test_the_ulp_neighbourhood_actually_walks():
    """Non-vacuity of the GRID ITSELF, not merely of its outcomes.

    A neighbourhood that collapses to 3 values still yields a grid that is neither all-True nor
    all-False, so the outcome-based check below passes while covering nothing.
    """
    assert len(set(_ulp_neighbourhood(13.84, n=50))) == 101


def test_array_form_agrees_with_scalar_everywhere():
    gr_x, y = _grid()
    got = in_penalty_area_goal_relative_array(gr_x, y)
    want = np.array([in_penalty_area_goal_relative(float(a), float(b)) for a, b in zip(gr_x, y, strict=True)])
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (
        f"{bad.size} disagreements, first at gr_x={gr_x[bad[0]]!r} y={y[bad[0]]!r}: "
        f"array={got[bad[0]]} scalar={want[bad[0]]}"
    )


def test_the_grid_is_not_vacuous():
    """A parity test over a grid that is all-True or all-False proves nothing."""
    gr_x, y = _grid()
    got = in_penalty_area_goal_relative_array(gr_x, y)
    assert got.any() and not got.all(), f"grid is degenerate: {got.sum()}/{got.size} True"


@pytest.mark.parametrize(
    "gr_x,y",
    [(float("nan"), 34.0), (5.0, float("nan")), (float("nan"), float("nan"))],
)
def test_nan_is_False_on_both_forms(gr_x, y):
    """SPECIFIED contract, not incidental behaviour: NaN on either argument -> False.

    The scalar form yields this because ``NaN <= depth`` is False. Pinning it here stops a future
    array implementation (e.g. one masking with ``np.abs``) from silently returning True.
    """
    assert in_penalty_area_goal_relative(gr_x, y) is False
    out = in_penalty_area_goal_relative_array(np.array([gr_x]), np.array([y]))
    assert out.dtype == np.bool_ and not bool(out[0])
