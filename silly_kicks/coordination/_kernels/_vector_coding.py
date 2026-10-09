"""Vector coding of a coordination dyad (Moura 2016 Table 1 coupling-angle patterns).

Pure numpy. The coupling angle is the direction of the (Delta a, Delta b) step; the four patterns are the
45-degree octants centred on the axes and diagonals. Every bin edge (22.5 + 45k) is exactly representable
in binary, so the classification is exact at the boundaries.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

PATTERNS: tuple[str, ...] = ("in_phase", "anti_phase", "a_phase", "b_phase")

# octant (0..7) -> PATTERNS index. Octant j spans [22.5 + 45(j-1), 22.5 + 45j); octant 0 straddles 0/360.
_OCTANT_TO_PATTERN = np.array([2, 0, 3, 1, 2, 0, 3, 1], dtype=np.int8)  # a, in, b, anti, a, in, b, anti


def coupling_angle_deg(da: npt.ArrayLike, db: npt.ArrayLike) -> np.ndarray:
    """Coupling angle in ``[0, 360)`` degrees: the direction of the ``(da, db)`` step, ``atan2(db, da)``."""
    ang = np.degrees(np.arctan2(np.asarray(db, dtype=np.float64), np.asarray(da, dtype=np.float64)))
    return np.asarray(ang % 360.0)


def classify(angle_deg: npt.ArrayLike) -> np.ndarray:
    """Classify coupling angles (degrees) into the four :data:`PATTERNS` (int8 index, Moura 2016 Table 1)."""
    angle = np.asarray(angle_deg, dtype=np.float64)
    octant = np.floor((angle + 22.5) / 45.0).astype(np.int64) % 8  # 337.5 and 360 -> octant 0 (A-phase)
    return np.asarray(_OCTANT_TO_PATTERN[octant])


def stationary_mask(da: npt.ArrayLike, db: npt.ArrayLike, eps_a: float, eps_b: float) -> np.ndarray:
    """True where BOTH components are below their epsilon: ``(|da| < eps_a) & (|db| < eps_b)`` (spec 7.8.3)."""
    below_a = np.abs(np.asarray(da, dtype=np.float64)) < eps_a
    below_b = np.abs(np.asarray(db, dtype=np.float64)) < eps_b
    return np.asarray(below_a & below_b)
