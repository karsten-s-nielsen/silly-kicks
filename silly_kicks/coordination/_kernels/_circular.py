"""Circular statistics for coordination phase analysis (Bourbousson 2010 relative-phase bins).

Pure numpy. All angles are degrees at the surface; the phasor ``z`` is a unit complex number carrying the
relative phase, so summaries reduce a vector sum ``z_sum`` and a count ``n`` rather than raw angle lists.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Bourbousson relative-phase histogram: twelve 30-deg bins, centred on these degrees (the -180 bin wraps).
HIST_BIN_CENTRES_DEG: tuple[int, ...] = (-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150)
HIST_BIN_LABELS: tuple[str, ...] = (
    "m180",
    "m150",
    "m120",
    "m090",
    "m060",
    "m030",
    "p000",
    "p030",
    "p060",
    "p090",
    "p120",
    "p150",
)


def wrap_deg(a: npt.ArrayLike) -> np.ndarray:
    """Wrap degrees into ``(-180, 180]`` (180 kept, -180 folded up to +180)."""
    a = np.asarray(a, dtype=np.float64)
    w = (a + 180.0) % 360.0 - 180.0
    return np.where(w == -180.0, 180.0, w)


def circular_summary(z_sum: npt.ArrayLike, n: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a phasor vector sum to ``(mean_deg, R, circ_sd_deg)``.

    ``mean_deg = deg(arg(z_sum))``; ``R = |z_sum| / n`` (mean resultant length); the circular SD is
    ``deg(sqrt(-2 ln R))``. ``n == 0`` yields NaN for all three; ``R == 0`` yields ``+inf`` for the SD.
    """
    z_sum = np.asarray(z_sum, dtype=np.complex128)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(z_sum) / n
        mean_deg = np.degrees(np.angle(z_sum))
        circ_sd_deg = np.degrees(np.sqrt(-2.0 * np.log(r)))
    bad = n == 0
    mean_deg = np.where(bad, np.nan, mean_deg)
    r = np.where(bad, np.nan, r)
    circ_sd_deg = np.where(bad, np.nan, circ_sd_deg)
    return mean_deg, r, circ_sd_deg


def hist_bin_index(z: npt.ArrayLike) -> np.ndarray:
    """Index (int8, 0..11) into :data:`HIST_BIN_CENTRES_DEG` for each unit phasor ``z``.

    Each bin spans ``[centre - 15, centre + 15)``; the +/-180 bin wraps (an angle in ``[165, 180]`` and one
    in ``(-180, -165)`` both fall in bin 0).
    """
    a = np.degrees(np.angle(np.asarray(z)))  # (-180, 180]
    return np.asarray((np.floor((a + 195.0) / 30.0).astype(np.int64) % 12).astype(np.int8))


def near_in_phase(z: npt.ArrayLike, near_deg: float) -> np.ndarray:
    """True where the unit phasor ``z`` is within ``near_deg`` of 0 phase: ``Re(z) >= cos(near_deg)``."""
    return np.asarray(np.real(z) >= np.cos(np.radians(near_deg)))


def circular_reliability(values_deg: npt.ArrayLike, groups: npt.ArrayLike) -> tuple[float, float]:
    """Rotation-invariant circular reliability (TF-58 §8.5) and the overall mean resultant length.

    The circular analogue of one-way ICC(1) (``scripts/_reliability.icc1``) for circular-mean
    constructs: ``1 - V_within / V_total``, where circular variance ``V = 1 - R`` is taken from mean
    resultant lengths ``R``. ``V_total`` pools every observation; ``V_within = 1 - (sum_g |S_g|) / n``
    is the n-weighted pooled within-group circular variance (``S_g`` the per-group phasor sum). Because
    every resultant magnitude is invariant to a global phase rotation, the statistic is unchanged when a
    constant is added to every angle -- a cos/sin-component ICC is NOT (it is origin-dependent), which is
    why the plain ICC is wrong for a circular mean.

    Returns ``(reliability, rbar_total)``. ``rbar_total`` is the overall concentration ``R`` so the
    caller can apply the ``CIRCULAR_RELIABILITY_MIN_RBAR`` floor (circular reliability is undefined as
    concentration -> 0). ``reliability`` is NaN when there are fewer than two groups, no more
    observations than groups, or no total variance (every observation in one direction).
    """
    v = np.asarray(values_deg, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    mask = np.isfinite(v)
    v, g = v[mask], g[mask]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan")
    c = np.cos(np.radians(v))
    s = np.sin(np.radians(v))
    rbar_total = float(np.hypot(c.sum(), s.sum()) / n)
    # group the phasor sums by unique group label (numpy only -- kernels forbid pandas)
    _uniq, inverse = np.unique(g, return_inverse=True)
    k = int(_uniq.size)
    if k < 2 or n <= k:
        return float("nan"), rbar_total
    sc = np.zeros(k, dtype=np.float64)
    ss = np.zeros(k, dtype=np.float64)
    np.add.at(sc, inverse, c)
    np.add.at(ss, inverse, s)
    v_within = 1.0 - float(np.hypot(sc, ss).sum()) / n
    v_total = 1.0 - rbar_total
    if v_total <= 0.0:
        return float("nan"), rbar_total
    return 1.0 - v_within / v_total, rbar_total
