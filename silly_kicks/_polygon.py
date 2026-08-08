"""Dependency-free polygon geometry: area, convexity, membership, convex clipping.

A NEUTRAL root module, in the same position as ``id_compat`` and ``reflection``, and for the
same reason: two packages need it and neither may depend on the other. ``tracking/_visibility``
answers "what did the camera observe" in SPADL metres; ``providers/statsbomb/parse`` answers the
same question in StatsBomb's native 120x80 before any transform exists. Putting the primitives
in either one would either invert the port layering (``providers`` -> ``tracking``) or duplicate
a Sutherland-Hodgman implementation -- and a second copy of a clipper is exactly the
fork-by-duplication this cycle exists to delete.

Coordinate-system agnostic by construction: nothing here knows a pitch length. Callers supply
both rings in whatever frame they are already working in.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np

#: Below this a ring bounds no region, so every quantity here is undefined rather than zero.
MIN_VERTICES = 3


def as_polygon(poly) -> np.ndarray:
    """``(N, 2)`` float ring, or ``(0, 2)`` when the input cannot bound a region.

    Collapsing "absent", "too few vertices" and "carries a NaN" to one empty result is
    deliberate: all three mean the same thing to a caller -- no region is described -- and each
    must be distinguishable from a genuine zero-area answer, which callers signal with NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks._polygon import as_polygon
    >>> as_polygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]).shape
    (3, 2)
    >>> as_polygon([[0.0, 0.0], [1.0, 1.0]]).shape
    (0, 2)
    >>> as_polygon(None).shape
    (0, 2)
    """
    if poly is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < MIN_VERTICES:
        return np.empty((0, 2), dtype=float)
    if not np.isfinite(arr).all():
        return np.empty((0, 2), dtype=float)
    return arr


def shoelace_area(poly: np.ndarray) -> float:
    """UNSIGNED ring area, so a clockwise ring is not reported negative.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks._polygon import shoelace_area
    >>> square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    >>> shoelace_area(square)
    4.0
    >>> shoelace_area(square[::-1])   # orientation must not change the magnitude
    4.0
    """
    if len(poly) < MIN_VERTICES:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def is_convex(poly: np.ndarray) -> bool:
    """Do all cross products along the ring share one sign? Collinear vertices are tolerated.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks._polygon import is_convex
    >>> is_convex(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]))
    True
    >>> is_convex(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 0.5], [0.0, 2.0]]))
    False
    """
    if len(poly) < MIN_VERTICES:
        return False
    d = np.roll(poly, -1, axis=0) - poly
    cross = d[:, 0] * np.roll(d[:, 1], -1) - d[:, 1] * np.roll(d[:, 0], -1)
    nz = cross[np.abs(cross) > 1e-12]
    return bool(nz.size == 0 or np.all(nz > 0) or np.all(nz < 0))


def counter_clockwise(poly: np.ndarray) -> np.ndarray:
    """The ring, reversed if needed, so ``clip_to_convex``'s inside test has a fixed sign.

    Examples
    --------
    Asserts the PROPERTY (the returned ring has positive signed area) rather than a vertex
    index, which says nothing about orientation and is easy to write down wrongly:

    >>> import numpy as np
    >>> from silly_kicks._polygon import counter_clockwise
    >>> def signed(p):
    ...     x, y = p[:, 0], p[:, 1]
    ...     return round(float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))), 6)
    >>> cw = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
    >>> signed(cw), signed(counter_clockwise(cw))
    (-4.0, 4.0)
    >>> ccw = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    >>> signed(counter_clockwise(ccw))   # already CCW: returned untouched
    4.0
    """
    x, y = poly[:, 0], poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return poly if signed >= 0 else poly[::-1]


def clip_to_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: ``subject`` clipped to the CONVEX ``clip`` ring.

    ``clip`` must already be counter-clockwise (see :func:`counter_clockwise`) and convex; the
    caller validates convexity, because this returns a wrong ring rather than an error for a
    concave clip and a silent wrong area is the failure mode worth refusing.

    ``subject`` carries NO convexity requirement, which is the property that decides the
    argument order at every call site: provider polygons are arbitrary and go here, regions are
    caller-constructed and go in ``clip``.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks._polygon import clip_to_convex, counter_clockwise, shoelace_area
    >>> big = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    >>> window = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    >>> shoelace_area(clip_to_convex(big, counter_clockwise(window)))
    4.0
    """
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            return np.empty((0, 2), dtype=float)
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a

        def side(p, _a=a, _e=edge):
            return _e[0] * (p[1] - _a[1]) - _e[1] * (p[0] - _a[0])

        prev = out[-1]
        clipped: list[np.ndarray] = []
        for cur in out:
            s_cur, s_prev = side(cur), side(prev)
            if s_cur >= 0:
                if s_prev < 0:
                    t = s_prev / (s_prev - s_cur)
                    clipped.append(prev + t * (cur - prev))
                clipped.append(cur)
            elif s_prev >= 0:
                t = s_prev / (s_prev - s_cur)
                clipped.append(prev + t * (cur - prev))
            prev = cur
        out = clipped
    return np.asarray(out, dtype=float) if out else np.empty((0, 2), dtype=float)


def point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    """Ray-casting membership. Correct for NON-CONVEX rings, which broadcast polygons can be.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks._polygon import point_in_polygon
    >>> notched = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [5.0, 1.0], [0.0, 10.0]])
    >>> point_in_polygon(notched, 1.0, 1.0)
    True
    >>> point_in_polygon(notched, 5.0, 8.0)   # inside the hull, OUTSIDE the polygon
    False
    """
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    straddles = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.count_nonzero(straddles & (x < x_at_y)) % 2 == 1)


def covered_fraction(polygon, region) -> float:
    """``area(region ∩ polygon) / area(region)`` in ``[0, 1]``; ``nan`` when undefined.

    ``nan`` -- never ``0.0`` -- when the polygon describes no region or the region has no area:
    zero would claim "measured, and none of it", which is a different statement from "no
    measurement exists".

    Raises
    ------
    ValueError
        If ``region`` has fewer than 3 finite vertices, or is not convex.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks._polygon import covered_fraction
    >>> left = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
    >>> triangle = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0]])
    >>> round(covered_fraction(left, triangle), 6)
    0.75
    """
    poly = as_polygon(polygon)
    reg = as_polygon(region)
    if len(reg) < MIN_VERTICES:
        raise ValueError("covered_fraction: region needs >= 3 finite (x, y) vertices")
    if not is_convex(reg):
        raise ValueError(
            "covered_fraction: region is not convex. Sutherland-Hodgman clips against a convex "
            "region only; a concave one would return a plausible but wrong area. Split it into "
            "convex parts and sum."
        )
    denom = shoelace_area(reg)
    if len(poly) < MIN_VERTICES or denom <= 0.0:
        return float("nan")
    return float(min(1.0, shoelace_area(clip_to_convex(poly, counter_clockwise(reg))) / denom))
