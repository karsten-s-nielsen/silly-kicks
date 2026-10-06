"""The detection share the ``insufficient_detection`` gate tests (spec 7.11; owner rulings 2026-10-04, review A-08).

ONE construct, single-sourced here, read by every family compute AND by the D1 occlusion derivation that sets the
thresholds -- a threshold only means something on the quantity it was derived from:

- a SIDE's detected share over a window ``[s, e)`` is the fraction of its on-pitch player-samples there that were
  RAW-detected (samples bridged across a detection gap count as NOT detected), so a fully observed provider's share
  is always 1.0;
- a row's share is the MINIMUM over its sides (coordination is two-sided: the worse-seen side limits it), and the row
  is ``insufficient_detection`` when that minimum is below its family's ``min_observed_fraction``;
- a window in which no side was on the pitch has no share (NaN): no detection verdict, the family's own minimum-length
  rule decides (``too_short``); a side on the pitch but never detected has share 0 and fails.

Pure numpy; prefix sums make every window's share O(1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionCounts:
    """One side's cumulative on-pitch and raw-detected player-sample counts on the analysis grid.

    ``cum_on[i]`` / ``cum_det[i]`` count samples ``[0, i)`` (summed over the side's players), so the share over
    ``[s, e)`` is a difference of two prefix sums.

    Examples
    --------
    >>> import numpy as np
    >>> c = DetectionCounts.from_masks(np.array([1, 1, 1, 1], bool), np.array([1, 0, 1, 1], bool))
    >>> c.share(0, 4), c.share(1, 2)
    (0.75, 0.0)
    """

    cum_on: np.ndarray
    cum_det: np.ndarray

    @classmethod
    def from_masks(cls, on_pitch: np.ndarray, detected: np.ndarray) -> DetectionCounts:
        """From ``(N,)`` or ``(N, P)`` boolean masks; a detected sample must be on the pitch (``detected <= on``)."""
        on = np.asarray(on_pitch, dtype=bool)
        det = np.asarray(detected, dtype=bool)
        if on.shape != det.shape:
            raise ValueError(f"on_pitch {on.shape} and detected {det.shape} masks differ in shape")
        if (det & ~on).any():
            raise ValueError("a detected sample must be on the pitch")
        if on.ndim == 2:
            on, det = on.sum(axis=1), det.sum(axis=1)
        zero = np.zeros(1, dtype=np.int64)
        return cls(
            cum_on=np.concatenate((zero, np.cumsum(on, dtype=np.int64))),
            cum_det=np.concatenate((zero, np.cumsum(det, dtype=np.int64))),
        )

    def share(self, s: int, e: int) -> float:
        """The detected share of the side's on-pitch player-samples in ``[s, e)``; NaN when it was never on."""
        on = int(self.cum_on[e] - self.cum_on[s])
        return float(self.cum_det[e] - self.cum_det[s]) / on if on > 0 else float("nan")


def row_detected_share(sides: Sequence[DetectionCounts], s: int, e: int) -> float:
    """The row's share: the MINIMUM over its sides' shares in ``[s, e)``, ignoring a side never on the pitch there;
    NaN when no side was.

    Examples
    --------
    >>> import numpy as np
    >>> on = np.ones(4, bool)
    >>> a = DetectionCounts.from_masks(on, np.array([1, 1, 1, 1], bool))
    >>> b = DetectionCounts.from_masks(on, np.array([1, 0, 0, 0], bool))
    >>> row_detected_share([a, b], 0, 4)
    0.25
    """
    shares = [x for x in (side.share(s, e) for side in sides) if not math.isnan(x)]
    return min(shares) if shares else float("nan")


def insufficient_detection(share: float, threshold: float) -> bool:
    """The gate: a share below the family threshold. NaN (no side on the pitch) carries no detection verdict.

    Examples
    --------
    >>> insufficient_detection(0.49, 0.5), insufficient_detection(0.5, 0.5), insufficient_detection(float("nan"), 0.5)
    (True, False, False)
    """
    return not math.isnan(share) and share < threshold
