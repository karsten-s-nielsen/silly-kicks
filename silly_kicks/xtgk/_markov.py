"""MarkovPossessionValue — production possession-value surface (ADR-036 §4).

Reuses xthreat.value_iteration verbatim with (i) an xG-calibrated immediate reward, (ii) a
goal-kick-inclusive move-set, (iii) pressure stratification. V(z,p) = E[xG of the possession's
FIRST shot | ball at z under pressure p] — the shoot branch is terminal, so the recursion
values the first shot; deep-zone value is pure forward propagation. See NOTICE / ADR-036 §4.2.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

from silly_kicks.xtgk._moves import extended_move_actions, xtgk_action_prob, xtgk_transition_matrix
from silly_kicks.xtgk._possession_value import DeltaV, PressureLevel, State
from silly_kicks.xtgk._validate import validate_possession_value_input
from silly_kicks.xtgk._xg_reward import xg_scoring_prob
from silly_kicks.xthreat import GridSpec, value_iteration
from silly_kicks.xthreat._grid import M, N, _count
from silly_kicks.xthreat._params import KDEParams

if TYPE_CHECKING:
    from silly_kicks.xtgk._pressure_levels import PressureLevels

Method = Literal["singh_counts", "kde_smoothed"]
_LEVELS: tuple[PressureLevel, ...] = (1, 2, 3)


class MarkovPossessionValue:
    def __init__(self, *, l: int = N, w: int = M, eps: float = 1e-5, method: Method = "singh_counts") -> None:
        self.l, self.w, self.eps = l, w, eps
        self.method: Method = method
        self.grid = GridSpec(n_zones_x=l, n_zones_y=w)
        self._surfaces: dict[PressureLevel, npt.NDArray[np.float64]] = {}
        self._support: dict[PressureLevel, npt.NDArray[np.int_]] = {}
        self._fitted = False
        self.xg_column: str | None = None
        self.pressure_levels: PressureLevels | None = None
        self.provenance: dict = {}

    def fit(
        self,
        actions: pd.DataFrame,
        *,
        xg_column: str,
        pressure_column: str,
        pressure_levels=None,
        reward_provenance: dict | None = None,
    ) -> MarkovPossessionValue:
        diag = validate_possession_value_input(actions, xg_column=xg_column, pressure_column=pressure_column)
        if not diag.ok:
            raise ValueError("invalid fit input: " + "; ".join(diag.problems))
        from silly_kicks.xtgk._pressure_levels import PressureLevels

        pl = pressure_levels or PressureLevels().fit(actions[pressure_column])
        zones = None
        if pl.mode == "zone_conditional":
            from silly_kicks.xtgk._possession_value import flat_zones

            zones = flat_zones(actions["start_x"], actions["start_y"], self.l, self.w)
        levels = pl.apply(actions[pressure_column], zones=zones)
        actions = actions.assign(_p_level=levels)
        for p in _LEVELS:
            sub = actions[actions["_p_level"] == p]
            if len(sub) == 0:
                warnings.warn(
                    f"pressure tercile {p} has zero actions at fit; its surface is all-zero "
                    f"(check pressure distribution / cutpoints — ADR-036 §5)",
                    stacklevel=2,
                )
            self._surfaces[p] = self._solve_level(sub, xg_column)
            self._support[p] = self._support_counts(sub)
        self.xg_column, self.pressure_levels, self._fitted = xg_column, pl, True
        self.provenance = {
            "xg_column": xg_column,
            "method": self.method,
            "grid": (self.l, self.w),
            "cutpoints": pl.cutpoints,
            "n_actions": len(actions),
        }
        # Q3 (ADR-036 §6): the caller (owner-run) summarizes the injected reward's quality
        # (OOD-rate, xg-CI width) from fct_shot_xg and passes it here — silly-kicks records but
        # never interprets ood_flag/CI semantics (no xG model shipped).
        if reward_provenance is not None:
            self.provenance["reward_provenance"] = reward_provenance
        return self

    def _solve_level(self, sub: pd.DataFrame, xg_column: str) -> npt.NDArray[np.float64]:
        xg_scoring = xg_scoring_prob(sub, xg_column=xg_column, l=self.l, w=self.w)
        p_shot, p_move = xtgk_action_prob(sub, self.l, self.w)
        transition = xtgk_transition_matrix(
            sub,
            self.grid,
            method=self.method,
            params=KDEParams() if self.method == "kde_smoothed" else None,
        )
        xt, _ = value_iteration(xg_scoring, p_shot, p_move, transition, eps=self.eps)
        return xt

    def _support_counts(self, sub: pd.DataFrame) -> npt.NDArray[np.int_]:
        moves = extended_move_actions(sub).dropna(subset=["start_x", "start_y"])
        return _count(moves.start_x, moves.start_y, self.l, self.w)

    def _check(self) -> None:
        if not self._fitted:
            raise NotFittedError("MarkovPossessionValue.fit not called")

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        self._check()
        return self._surfaces[p]

    def value(self, zone: int, p: PressureLevel) -> float:
        self._check()
        return float(self._surfaces[p].ravel()[zone])

    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]:
        self._check()
        return self._support[p]

    def delta_v(self, s: State, s_next: State) -> DeltaV:
        self._check()
        z, p, zp, pp = s.zone, s.pressure_level, s_next.zone, s_next.pressure_level
        v_zp, v_zpp, v_zpp_, v_zp_pp = (
            self.value(z, p),
            self.value(z, pp),
            self.value(zp, p),
            self.value(zp, pp),
        )
        delta = v_zp_pp - v_zp
        pressure = 0.5 * ((v_zpp - v_zp) + (v_zp_pp - v_zpp_))
        position = 0.5 * ((v_zpp_ - v_zp) + (v_zp_pp - v_zpp))
        return DeltaV(delta=delta, pressure_component=pressure, position_component=position)

    # -- serialization (ADR-036 §4/G4, pickle-free) -----------------------
    def save(self, directory) -> None:
        self._check()
        from silly_kicks.xtgk._serialize import save_surface

        pl = cast("PressureLevels", self.pressure_levels)  # _check() guarantees fitted
        meta = dict(self.provenance)
        meta.update(pl.to_meta())  # global: {"cutpoints":[lo,hi]} (byte-identical); zone_cond adds band_cutpoints
        save_surface(directory, surfaces=self._surfaces, support=self._support, metadata=meta)

    @classmethod
    def load(cls, directory) -> MarkovPossessionValue:
        from silly_kicks.xtgk._pressure_levels import PressureLevels
        from silly_kicks.xtgk._serialize import load_surface

        surfaces, support, meta = load_surface(directory)
        l, w = meta["grid"]
        obj = cls(l=int(l), w=int(w), method=cast(Method, meta.get("method", "singh_counts")))
        obj._surfaces = {cast(PressureLevel, p): surfaces[p] for p in (1, 2, 3)}
        obj._support = {cast(PressureLevel, p): support[p] for p in (1, 2, 3)}
        obj.provenance = meta
        obj.xg_column = meta.get("xg_column")
        obj.pressure_levels = PressureLevels.from_meta(meta, l=int(l))
        obj._fitted = True
        return obj
