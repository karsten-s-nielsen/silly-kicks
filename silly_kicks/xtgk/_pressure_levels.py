"""Continuous pressure -> {1,2,3} tercile quantizer (ADR-036 §5, §1c).

fit() learns cutpoints on the fit cohort; apply() maps new actions; cutpoints persist with the
surface. mode="global" is the default and byte-identical to SP1. mode="zone_conditional" learns
per-BAND terciles (deep band = grid columns xi in {0,1} vs the rest) so a systematically
low-pressure deep zone still populates all three deep terciles (the M3 fix / gate fallback rung).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.xthreat._grid import N

Mode = Literal["global", "zone_conditional"]

_DEEP_MAX_XI = 1  # deep band = grid columns xi in {0,1} (matches _diagnostics.DEEP_ZONE_CELLS)


def band_of_zone(zone: int, l: int = N) -> int:
    """0 for the deep band (xi in {0,1}), 1 otherwise. Flat index layout: xi = zone % l."""
    return 0 if (int(zone) % l) <= _DEEP_MAX_XI else 1


def _bands(zones: np.ndarray, l: int) -> np.ndarray:
    return np.where((np.asarray(zones).astype(int) % l) <= _DEEP_MAX_XI, 0, 1)


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

    Pure: returns a new Series, never mutates ``pressure``.
    """
    out = pressure.copy()
    fill_mask = frame_present.to_numpy(dtype=bool) & out.isna().to_numpy()
    out[fill_mask] = 0.0
    return out


class PressureLevels:
    def __init__(self, *, mode: Mode = "global", l: int = N) -> None:
        self.mode: Mode = mode
        self.l = l
        self.cutpoints: tuple[float, float] | None = None  # global
        self.band_cutpoints: dict[int, tuple[float, float]] | None = None  # zone_conditional

    def fit(self, pressure: pd.Series[float], *, zones: np.ndarray | None = None) -> PressureLevels:
        p_all = pressure.to_numpy(dtype=float)
        valid = ~np.isnan(p_all)
        if not valid.any():
            raise ValueError("cannot fit pressure terciles on empty/all-NaN pressure")
        if self.mode == "global":
            lo, hi = np.quantile(p_all[valid], [1 / 3, 2 / 3])
            self.cutpoints = (float(lo), float(hi))
            return self
        if zones is None:
            raise ValueError("zone_conditional fit requires zones= (each action's flat grid cell)")
        bands = _bands(zones, self.l)
        bc: dict[int, tuple[float, float]] = {}
        for b in (0, 1):
            sel = valid & (bands == b)
            if not sel.any():
                raise ValueError(f"zone band {b} has no non-NaN pressure at fit (check deep-zone coverage)")
            lo, hi = np.quantile(p_all[sel], [1 / 3, 2 / 3])
            bc[b] = (float(lo), float(hi))
        self.band_cutpoints = bc
        return self

    @classmethod
    def from_cutpoints(cls, cutpoints: tuple[float, float], *, mode: Mode = "global") -> PressureLevels:
        obj = cls(mode=mode)
        obj.cutpoints = (float(cutpoints[0]), float(cutpoints[1]))
        return obj

    @classmethod
    def from_band_cutpoints(cls, band_cutpoints: dict, *, l: int = N) -> PressureLevels:
        obj = cls(mode="zone_conditional", l=l)
        obj.band_cutpoints = {int(k): (float(v[0]), float(v[1])) for k, v in band_cutpoints.items()}
        return obj

    def apply(self, pressure: pd.Series[float], *, zones: np.ndarray | None = None) -> np.ndarray:
        # fitted-check BEFORE the isna-check (preserves SP1 apply ordering)
        if self.mode == "global":
            if self.cutpoints is None:
                raise ValueError("PressureLevels not fitted")
        elif self.band_cutpoints is None:
            raise ValueError("PressureLevels not fitted")
        if pressure.isna().any():
            raise ValueError("missing pressure value(s); never default a level (ADR-036 §5)")
        p = pressure.to_numpy(dtype=float)
        if self.mode == "global":
            lo, hi = self.cutpoints  # type: ignore[misc]
            return np.where(p <= lo, 1, np.where(p <= hi, 2, 3)).astype(int)
        if zones is None:
            raise ValueError("zone_conditional apply requires zones= (each action's flat grid cell)")
        bands = _bands(zones, self.l)
        bc = self.band_cutpoints
        if bc is None:
            raise ValueError("PressureLevels not fitted")
        los = np.array([bc[int(b)][0] for b in bands])
        his = np.array([bc[int(b)][1] for b in bands])
        return np.where(p <= los, 1, np.where(p <= his, 2, 3)).astype(int)

    def occupancy(self, pressure: pd.Series[float], *, zones: np.ndarray | None = None) -> dict[int, int]:
        lv = self.apply(pressure, zones=zones)
        return {k: int((lv == k).sum()) for k in (1, 2, 3)}

    def to_meta(self) -> dict:
        """Serialize state. Global form is byte-identical to SP1 (``{"cutpoints": [lo, hi]}``)."""
        if self.mode == "global":
            if self.cutpoints is None:
                raise ValueError("cannot serialize an unfitted PressureLevels")
            return {"cutpoints": list(self.cutpoints)}
        if self.band_cutpoints is None:
            raise ValueError("cannot serialize an unfitted PressureLevels")
        return {
            "pressure_mode": "zone_conditional",
            "band_cutpoints": {str(b): list(c) for b, c in self.band_cutpoints.items()},
        }

    @classmethod
    def from_meta(cls, meta: dict, *, l: int = N) -> PressureLevels:
        """Reconstruct. Absent ``pressure_mode`` => global (back-compat with SP1 artifacts)."""
        if meta.get("pressure_mode") == "zone_conditional":
            return cls.from_band_cutpoints({int(b): tuple(c) for b, c in meta["band_cutpoints"].items()}, l=l)
        cut = meta["cutpoints"]
        return cls.from_cutpoints((float(cut[0]), float(cut[1])))
