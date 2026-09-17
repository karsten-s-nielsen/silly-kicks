"""Ground-truth geometry tests for the synthetic-interception substrate (SPEC-02 / SPEC-09).

``perturb_interception`` places a synthetic interception death at a flight-fraction ``f`` along the
``origin -> end`` ray and rotates the offset vector by ``delta`` OFF that ray. The two consequences
documented in the module docstring are the ground truth these tests pin:

* ``delta == 0`` -> ``death == origin + f * (end - origin)`` exactly (zero perpendicular distance to
  the ``origin -> end`` line).
* ``delta != 0`` -> the perpendicular distance from ``death`` to the ``origin -> end`` line equals
  ``f * |end - origin| * |sin(delta)|`` exactly.

Plus: the return preserves the broadcast shape of the inputs (scalar and array).
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic_interception import perturb_interception


def _perp_distance_to_line(px, py, ox, oy, ex, ey):
    """Exact perpendicular distance from point ``(px, py)`` to the line through ``o`` and ``e``.

    Uses the 2-D cross product of ``(e - o)`` with ``(p - o)`` divided by ``|e - o|``.
    """
    vx = ex - ox
    vy = ey - oy
    cross = vx * (py - oy) - vy * (px - ox)
    return np.abs(cross) / np.hypot(vx, vy)


def test_zero_offset_lands_exactly_on_segment_at_fraction():
    # delta == 0 -> death == origin + f * (end - origin), exactly.
    origin = (10.0, 20.0)
    end = (40.0, 60.0)
    f = 0.375

    dx, dy = perturb_interception(origin, end, fraction=f, angle_offset_rad=0.0)

    expected_x = origin[0] + f * (end[0] - origin[0])
    expected_y = origin[1] + f * (end[1] - origin[1])
    assert dx == pytest.approx(expected_x, abs=0.0, rel=0.0)
    assert dy == pytest.approx(expected_y, abs=0.0, rel=0.0)

    # Zero perpendicular distance to the origin -> end line.
    perp = _perp_distance_to_line(dx, dy, origin[0], origin[1], end[0], end[1])
    assert perp == pytest.approx(0.0, abs=1e-12)


def test_nonzero_offset_perpendicular_distance_matches_ground_truth():
    # delta != 0 -> perp distance == f * |v| * |sin(delta)|, exactly.
    origin = (5.0, 5.0)
    end = (35.0, 45.0)  # |v| = 50.0 (a clean 30-40-50 triangle)
    f = 0.42

    for delta in (0.1, 0.5, 1.0, -0.7, 2.3):
        dx, dy = perturb_interception(origin, end, fraction=f, angle_offset_rad=delta)

        vx = end[0] - origin[0]
        vy = end[1] - origin[1]
        vmag = np.hypot(vx, vy)
        expected_perp = f * vmag * abs(np.sin(delta))

        perp = _perp_distance_to_line(dx, dy, origin[0], origin[1], end[0], end[1])
        assert perp == pytest.approx(expected_perp, abs=1e-9), f"delta={delta}"

        # Strictly positive for f > 0, |v| > 0, delta not a multiple of pi.
        assert perp > 0.0


def test_rotation_preserves_distance_from_origin():
    # A rotation about the origin preserves |death - origin| = f * |v| for any delta.
    origin = (0.0, 0.0)
    end = (30.0, 40.0)  # |v| = 50.0
    f = 0.6
    vmag = np.hypot(end[0] - origin[0], end[1] - origin[1])

    for delta in (0.0, 0.3, 1.2, -2.0):
        dx, dy = perturb_interception(origin, end, fraction=f, angle_offset_rad=delta)
        radius = np.hypot(dx - origin[0], dy - origin[1])
        assert radius == pytest.approx(f * vmag, abs=1e-9), f"delta={delta}"


def test_scalar_inputs_return_scalars():
    dx, dy = perturb_interception((1.0, 2.0), (3.0, 4.0), fraction=0.5, angle_offset_rad=0.0)
    assert np.ndim(dx) == 0
    assert np.ndim(dy) == 0
    assert dx == pytest.approx(2.0)
    assert dy == pytest.approx(3.0)


def test_array_inputs_preserve_broadcast_shape():
    ox = np.array([0.0, 10.0, 20.0])
    oy = np.array([0.0, 5.0, 10.0])
    ex = np.array([30.0, 40.0, 50.0])
    ey = np.array([40.0, 5.0, 10.0])
    f = np.array([0.5, 0.25, 0.75])
    delta = np.array([0.0, 0.0, 0.0])

    dx, dy = perturb_interception((ox, oy), (ex, ey), fraction=f, angle_offset_rad=delta)

    assert dx.shape == (3,)
    assert dy.shape == (3,)
    np.testing.assert_allclose(dx, ox + f * (ex - ox))
    np.testing.assert_allclose(dy, oy + f * (ey - oy))


def test_broadcasting_scalar_endpoints_against_array_fraction():
    origin = (0.0, 0.0)
    end = (10.0, 0.0)
    f = np.linspace(0.1, 0.9, 5)
    delta = np.zeros(5)

    dx, dy = perturb_interception(origin, end, fraction=f, angle_offset_rad=delta)

    assert dx.shape == (5,)
    assert dy.shape == (5,)
    np.testing.assert_allclose(dx, f * 10.0)
    np.testing.assert_allclose(dy, np.zeros(5))
