"""Trimmed defensive hull geometry for TF-54 (spec §5.2).

``build_trimmed_hull`` keeps the ``trim_fraction`` of points nearest their centroid (robust to the odd
deep recovery), then takes the convex hull of the survivors. A hull with fewer than 3 non-collinear
survivors is DEGENERATE -> ``None`` (the caller drops-and-counts that player; never a fabricated 0/NaN).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError  # QhullError: scipy >= 1.8


class Hull:
    """A convex hull over trimmed defensive-action locations: membership + area + centroid.

    ``contains`` is vectorized (an ``(M, 2)`` array of query points -> an ``(M,)`` bool array); a single
    ``(2,)`` point returns a numpy bool.
    """

    __slots__ = ("_delaunay", "area", "centroid")

    def __init__(self, survivors: np.ndarray, hull: ConvexHull, delaunay: Delaunay) -> None:
        self._delaunay = delaunay
        # ConvexHull.volume is the enclosed AREA in 2D (.area would be the perimeter).
        self.area = float(hull.volume)
        c = survivors.mean(axis=0)
        self.centroid = (float(c[0]), float(c[1]))

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return self._delaunay.find_simplex(np.asarray(xy, dtype=float)) >= 0


def build_trimmed_hull(defensive_actions_xy: np.ndarray, *, trim_fraction: float) -> Hull | None:
    """Trimmed convex hull of a player's defensive-action locations, or ``None`` if degenerate.

    ``defensive_actions_xy`` is an ``(N, 2)`` array; NaN rows are dropped. Keeps
    ``max(3, ceil(N * trim_fraction))`` points nearest the centroid (never fewer than a triangle).
    """
    xy = np.asarray(defensive_actions_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return None
    xy = xy[np.isfinite(xy).all(axis=1)]
    n = len(xy)
    if n < 3:
        return None
    centroid = xy.mean(axis=0)
    d2 = ((xy - centroid) ** 2).sum(axis=1)
    keep = min(n, max(3, int(np.ceil(n * trim_fraction))))
    idx = np.argsort(d2, kind="stable")[:keep]
    survivors = xy[idx]
    try:
        hull = ConvexHull(survivors)
        delaunay = Delaunay(survivors)
    except (QhullError, ValueError):  # collinear / coincident / too few distinct points
        return None
    return Hull(survivors, hull, delaunay)
