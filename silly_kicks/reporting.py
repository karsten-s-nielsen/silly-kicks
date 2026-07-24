"""Reporting / wordalisation helpers.

Seed module: ``describe_level`` — a generic z-score -> verbal-band transform. Deliberately separate
from ``feature_glossary`` (this is a generic transform, not feature metadata; different responsibility
and change cadence). See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Owner-specified coach-facing DESCRIPTIVE bands (relative-to-cohort, NOT absolute quality;
# z>=0.5 -> "good" is ~69th percentile == "above average"). Provisional / adjustable.
_BANDS: list[tuple[float, str]] = [
    (1.5, "outstanding"),
    (1.0, "excellent"),
    (0.5, "good"),
    (-0.5, "average"),
    (-1.0, "below average"),
]
_FLOOR = "poor"
_UNKNOWN = "unknown"


def describe_level(z, *, higher_is_better: bool = True):
    """Map a z-score (or array/Series of them) to a verbal band, direction-aware and NaN-safe.

    Parameters
    ----------
    z : float | numpy.ndarray | pandas.Series
        Standardised score(s). A z-score of 0 is exactly average.
    higher_is_better : bool, default True
        When ``False`` the sign is flipped internally, so a lower-is-better metric (turnovers,
        times beaten) at a high z is correctly labelled ``"poor"``, not ``"outstanding"``.

    Returns
    -------
    str | numpy.ndarray | pandas.Series
        A scalar ``str`` for scalar input; an object ``ndarray`` for array input; a ``Series``
        (index preserved) for ``Series`` input. ``NaN`` maps to ``"unknown"``.

    Notes
    -----
    Bands (upper-open, ``>=`` boundaries): ``>=1.5`` outstanding, ``>=1.0`` excellent, ``>=0.5``
    good, ``>=-0.5`` average, ``>=-1.0`` below average, else poor. These are relative-to-cohort
    descriptive labels, not absolute-quality claims.

    Examples
    --------
    >>> describe_level(1.6)
    'outstanding'
    >>> describe_level(1.6, higher_is_better=False)
    'poor'
    >>> describe_level(float("nan"))
    'unknown'
    """
    is_series = isinstance(z, pd.Series)
    index = z.index if is_series else None
    arr = np.asarray(z, dtype=float)
    scalar = arr.ndim == 0
    flat = arr.reshape(-1)
    score = flat if higher_is_better else -flat
    out = np.full(flat.shape, _FLOOR, dtype=object)
    for thr, label in reversed(_BANDS):  # ascending thresholds; the highest satisfied wins
        out[score >= thr] = label
    out[np.isnan(flat)] = _UNKNOWN  # applied last: NaN never mislabels
    if scalar:
        return str(out[0])
    result = out.reshape(arr.shape)
    return pd.Series(result, index=index) if is_series else result
