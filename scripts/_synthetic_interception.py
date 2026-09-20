"""Synthetic-interception substrate for the TF-54b target-recovery battery (SPEC-02 / SPEC-09).

A completed pass carries ground truth: its true end. To validate the counterfactual metric's
failed-pass target model (the direction-conditioned distribution ``q`` and the cone half-angle
``direction_cone_degrees``) we take a completed pass, SYNTHESIZE an interception somewhere along
its flight, hide the true end, and ask the estimator to recover it.

``perturb_interception`` places that synthetic death at a flight-fraction ``f`` in ``(0, 1)`` along
the ``origin -> end`` ray AND rotates it by an angular offset ``delta`` OFF the ray. Corrupting BOTH
the intended distance AND the intended direction is the whole point: an offset of zero would leave
the death exactly on the ray and would validate only distance recovery, so a metric whose cone
half-angle was wildly wrong could still pass. The angular perturbation is what exercises and
validates ``direction_cone_degrees`` against known ground truth.

Geometry: let ``o`` be the origin, ``v = end - o``. The unrotated point at fraction ``f`` is
``o + f*v``. Rotating the offset vector ``f*v`` about the origin by ``delta`` gives
``death = o + R(delta) @ (f*v)``. Consequences, both used as ground truth by the tests:

* ``delta == 0`` -> ``death == o + f*v`` (exactly on the segment at fraction ``f``; zero
  perpendicular distance to the ``o -> end`` line).
* ``delta != 0`` -> the perpendicular distance from ``death`` to the ``o -> end`` line is exactly
  ``f * |v| * |sin(delta)|`` (strictly positive for ``f > 0``, ``|v| > 0``, ``delta`` not a
  multiple of pi) -- the death is off the ray.

Pure numpy and fully vectorizable: every input may be a scalar or a broadcastable array. The two
coordinate axes are passed as ``origin = (x, y)`` and ``end = (x, y)`` sequences.
"""

from __future__ import annotations

import numpy as np


def perturb_interception(origin, end, *, fraction, angle_offset_rad):
    """Return ``(death_x, death_y)`` for a synthetic interception (SPEC-02 / SPEC-09).

    Parameters
    ----------
    origin : sequence ``(x, y)``
        The pass origin. ``x`` and ``y`` may be scalars or broadcastable arrays.
    end : sequence ``(x, y)``
        The completed pass's true end (the hidden ground-truth target).
    fraction : float or array
        Flight-fraction ``f`` in ``(0, 1)`` at which the interception is synthesized.
    angle_offset_rad : float or array
        Angular offset ``delta`` (radians) by which the offset vector is rotated OFF the ray, so
        the synthesized death corrupts the intended direction as well as the intended distance.

    Returns
    -------
    (death_x, death_y)
        The synthetic death coordinates, same broadcast shape as the inputs.
    """
    ox = np.asarray(origin[0], dtype=float)
    oy = np.asarray(origin[1], dtype=float)
    ex = np.asarray(end[0], dtype=float)
    ey = np.asarray(end[1], dtype=float)
    f = np.asarray(fraction, dtype=float)
    delta = np.asarray(angle_offset_rad, dtype=float)

    # The offset vector from the origin, scaled to the flight-fraction.
    sx = f * (ex - ox)
    sy = f * (ey - oy)

    # Rotate that offset about the origin by delta (a 2-D rotation; corrupts the direction).
    cos = np.cos(delta)
    sin = np.sin(delta)
    rx = cos * sx - sin * sy
    ry = sin * sx + cos * sy

    return ox + rx, oy + ry
