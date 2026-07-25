"""Shared velocity-availability contract (extracted from _das.py, TF-51 v2).

The one place the "structurally unavailable vs caller-forgot-derive_velocities()" distinction lives,
so every velocity consumer (DAS, press-commitment) reads it identically instead of each raising
unconditionally or silently filling zeros.
"""

from __future__ import annotations

import pandas as pd

from .schema import SPEED_SOURCE_UNAVAILABLE


def velocity_unavailable_by_design(frames: pd.DataFrame) -> bool:
    """True iff EVERY row declares kinematics structurally unavailable.

    Reads the ``speed_source == SPEED_SOURCE_UNAVAILABLE`` marker a frame builder stamps when its
    source has no per-player temporal history to differentiate (the freeze-frame shape -- see
    :data:`~silly_kicks.tracking.SPEED_SOURCE_UNAVAILABLE`). Absent the marker, missing ``vx``/``vy``
    is a caller bug ("forgot ``derive_velocities()``") and must fail loud; the whole point of the
    marker is that the two shapes are otherwise byte-identical at this seam.

    ALL rows must be marked, not any: a PARTIALLY marked frame set means some genuine velocity-bearing
    source is also missing its velocity, which is the caller bug -- so the fail-loud branch wins. An
    empty frame set is not marked (nothing declared it).
    """
    if "speed_source" not in frames.columns or len(frames) == 0:
        return False
    return bool((frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all())
