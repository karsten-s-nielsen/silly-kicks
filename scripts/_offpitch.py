"""Shared gross-off-pitch predicate for the keeper-box detection-quality cycle.

A per-position boolean mask over the SPADL pitch rectangle [0, 105] x [0, 68] plus a
tolerance margin for legitimately-off-pitch positions (a keeper stepping behind the goal
line). Owned here because there is NO reusable off-pitch constant in the pining loader
(measured: only the per-player ``is_visible`` detection bit exists). Imported by both
``scripts/validate_skillcorner_keeper_origin.py`` (the S1 gross-off-pitch rate-gate) and
``scripts/measure_box_constant_delta.py`` so the two use one implementation.

N-2 (open confirmation): align ``OFF_PITCH_MARGIN_M`` with ADR-024 S1's own gross-off-pitch
fail-loud bound -- defined in the SkillCorner converter / resolver, not the loader -- or
document the deliberate difference here, so the CI rate-gate and S1 measure the same thing.
"""

from __future__ import annotations

import numpy as np

#: Tolerance (metres) beyond the pitch rectangle before a position counts as gross-off-pitch.
#: The "few-metre tolerance for legitimately off-pitch keepers" that ADR-024 S1 allows.
OFF_PITCH_MARGIN_M = 2.0

_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0


def off_pitch_mask(x: np.ndarray, y: np.ndarray, *, margin_m: float = OFF_PITCH_MARGIN_M) -> np.ndarray:
    """Boolean mask: each ``(x, y)`` lies beyond the pitch rectangle by more than ``margin_m``.

    NaN coordinates yield False (an unknown position makes no gross-off-pitch claim), matching
    the "False is a claim" discipline the tracking visibility seam uses.

    Examples
    --------
    >>> import numpy as np
    >>> off_pitch_mask(np.array([52.5, -3.0]), np.array([34.0, 34.0])).tolist()
    [False, True]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beyond = (x < -margin_m) | (x > _PITCH_LENGTH + margin_m) | (y < -margin_m) | (y > _PITCH_WIDTH + margin_m)
    return beyond & ~(np.isnan(x) | np.isnan(y))
