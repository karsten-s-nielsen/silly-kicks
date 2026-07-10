"""EmpiricalPossessionValue — model-free cross-check (ADR-036 §M1). NOT shipped.

Per action, outcome = xG of the FIRST shot after it in the same possession (0 if none),
averaged per (cell, tercile). This 'first_shot' aggregation is the like-for-like estimator of
the Markov target (per-action conditioning, ball-at-z). Independent of the Markov estimator (no
shared transitions), which is what makes disagreement diagnostic (§8.5). Partial port:
surface/value only.

Coincidence band: first_shot EXCLUDES the action's own shot, so at shot-origin (final-third)
cells it undercounts vs the Markov gs immediate term — harmless BECAUSE the gate compares only
on BUILD-UP cells, exactly the region where the two coincide. Never compare on final-third cells.

O(n) per possession via a right-to-left scan.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk._possession_value import PressureLevel
from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

Aggregation = Literal["first_shot", "noisy_or", "sum"]
_LEVELS: tuple[PressureLevel, ...] = (1, 2, 3)
_SHOT = spadlconfig.actiontype_id["shot"]


def _possession_outcomes(a: pd.DataFrame, xg_column: str, aggregation: Aggregation) -> np.ndarray:
    """For each action, the aggregated xG of shots strictly AFTER it within its possession."""
    out = np.zeros(len(a), dtype=float)
    pos = {ix: i for i, ix in enumerate(a.index)}
    group_cols = ["game_id", "possession_id"] if "game_id" in a.columns else ["possession_id"]
    for _key, grp in a.groupby(group_cols, sort=False):
        idx = list(grp.index)
        is_shot = (grp["type_id"] == _SHOT).to_numpy()
        xg = grp[xg_column].fillna(0.0).to_numpy(dtype=float)
        acc_first = 0.0  # first-shot value for the current position
        acc_sum = 0.0  # sum of shot xg after
        acc_prod = 1.0  # product of (1-xg) after -> noisy_or = 1 - prod
        for i in range(len(idx) - 1, -1, -1):
            if aggregation == "first_shot":
                out[pos[idx[i]]] = acc_first
            elif aggregation == "sum":
                out[pos[idx[i]]] = acc_sum
            else:
                out[pos[idx[i]]] = 1.0 - acc_prod
            if is_shot[i]:
                acc_first = xg[i]
                acc_sum += xg[i]
                acc_prod *= 1.0 - xg[i]
    return out


class EmpiricalPossessionValue:
    def __init__(self, *, l: int = N, w: int = M) -> None:
        self.l, self.w = l, w
        self._surfaces: dict[PressureLevel, npt.NDArray[np.float64]] = {}
        self._fitted = False

    def fit(
        self,
        actions: pd.DataFrame,
        *,
        xg_column: str,
        pressure_column: str,
        aggregation: Aggregation = "first_shot",
        pressure_levels=None,
    ) -> EmpiricalPossessionValue:
        from silly_kicks.xtgk._pressure_levels import PressureLevels

        pl = pressure_levels or PressureLevels().fit(actions[pressure_column])
        a = actions.reset_index(drop=True).copy()
        zones = None
        if pl.mode == "zone_conditional":
            from silly_kicks.xtgk._possession_value import flat_zones

            zones = flat_zones(a.start_x, a.start_y, self.l, self.w)
        a["_p_level"] = pl.apply(a[pressure_column], zones=zones)
        a["_outcome"] = _possession_outcomes(a, xg_column, aggregation)
        for p in _LEVELS:
            sub = a[a["_p_level"] == p].dropna(subset=["start_x", "start_y"])
            flat = _get_flat_indexes(sub.start_x, sub.start_y, self.l, self.w).to_numpy()
            num = np.zeros(self.w * self.l)
            den = np.zeros(self.w * self.l)
            np.add.at(num, flat, sub["_outcome"].to_numpy(dtype=float))
            np.add.at(den, flat, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                surf = np.where(den > 0, num / den, 0.0)
            self._surfaces[p] = surf.reshape((self.w, self.l))
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted:
            raise NotFittedError("EmpiricalPossessionValue.fit not called")

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        self._check()
        return self._surfaces[p]

    def value(self, zone: int, p: PressureLevel) -> float:
        self._check()
        return float(self._surfaces[p].ravel()[zone])
