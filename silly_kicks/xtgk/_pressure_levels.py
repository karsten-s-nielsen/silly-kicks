"""Continuous pressure -> {1,2,3} tercile quantizer (ADR-036 §5).

fit() learns cutpoints on the fit cohort; apply() maps new actions; cutpoints persist with the
surface. Occupancy is reported so a degenerate deep-zone stratification is visible.
NOTE: heavy ties skew tercile fill; continuous tracking pressure is fine, but a 2-value input
collapses to two levels — always feed >=3 well-separated bands in fixtures.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Mode = Literal["global", "zone_conditional"]


def coalesce_frame_present_null_pressure(
    pressure: pd.Series[float], frame_present: pd.Series[bool]
) -> pd.Series[float]:
    """Frame-aware null-pressure rule (ADR-036 §5, G8): distinguish a genuine tracking gap from a
    genuinely unpressured restart.

    - **frame present & pressure null** → 0.0 (no opponent in the pressure region — e.g. an
      unpressured goal-kick; a real zero → LOW tercile, keep it).
    - **frame absent & pressure null** → left null (a genuine tracking gap; ``PressureLevels.apply``
      then fail-loud-drops it — the backstop).
    - non-null → unchanged.

    Pure: returns a new Series, never mutates ``pressure``. Owner-run data-prep applies this BEFORE
    ``fit`` so the ~60% of WC goal-kicks that are frame-present-null are not silently dropped.
    """
    out = pressure.copy()
    fill_mask = frame_present.to_numpy(dtype=bool) & out.isna().to_numpy()
    out[fill_mask] = 0.0
    return out


class PressureLevels:
    def __init__(self, *, mode: Mode = "global") -> None:
        self.mode: Mode = mode
        self.cutpoints: tuple[float, float] | None = None

    def fit(self, pressure: pd.Series[float]) -> PressureLevels:
        p = pressure.dropna().to_numpy(dtype=float)
        if p.size == 0:
            raise ValueError("cannot fit pressure terciles on empty/all-NaN pressure")
        lo, hi = np.quantile(p, [1 / 3, 2 / 3])
        self.cutpoints = (float(lo), float(hi))
        return self

    @classmethod
    def from_cutpoints(cls, cutpoints: tuple[float, float], *, mode: Mode = "global") -> PressureLevels:
        obj = cls(mode=mode)
        obj.cutpoints = (float(cutpoints[0]), float(cutpoints[1]))
        return obj

    def apply(self, pressure: pd.Series[float]) -> np.ndarray:
        if self.cutpoints is None:
            raise ValueError("PressureLevels not fitted")
        if pressure.isna().any():
            raise ValueError("missing pressure value(s); never default a level (ADR-036 §5)")
        lo, hi = self.cutpoints
        p = pressure.to_numpy(dtype=float)
        return np.where(p <= lo, 1, np.where(p <= hi, 2, 3)).astype(int)

    def occupancy(self, pressure: pd.Series[float]) -> dict[int, int]:
        lv = self.apply(pressure)
        return {k: int((lv == k).sum()) for k in (1, 2, 3)}
