"""TurnoverCost port + MirroredTurnoverCost adapter + EmpiricalTurnoverValue cross-check
(ADR-036 §Part 2).

V(z,p) is team-agnostic (pooled attack-LTR), so the opponent's threat after winning the ball at
zone z is V at the 180-degree mirror zone. MirroredTurnoverCost wraps an already-fit
PossessionValue -- zero new fitting. EmpiricalTurnoverValue is a model-free cross-check (not shipped)
that validates the p_opp = p mirror assumption on real post-turnover chains.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions
from silly_kicks.xtgk._moves import _is_turnover
from silly_kicks.xtgk._possession_value import M, N, PossessionValue, PressureLevel, mirror_zone
from silly_kicks.xthreat._grid import _get_flat_indexes

_SHOT = spadlconfig.actiontype_id["shot"]


@runtime_checkable
class TurnoverCost(Protocol):
    def value(self, zone: int, p: PressureLevel) -> float: ...
    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]: ...
    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]: ...


class MirroredTurnoverCost:
    """E[opp threat | turnover at (zone, p)] = V(mirror_zone(zone), policy(p)). Zero new fitting."""

    def __init__(
        self,
        possession_value: PossessionValue,
        *,
        pressure_policy: Callable[[PressureLevel], PressureLevel] | None = None,
        l: int = N,
        w: int = M,
    ) -> None:
        self._v = possession_value
        self._policy = pressure_policy or (lambda p: p)  # default p_opp = p
        self.l, self.w = l, w

    def value(self, zone: int, p: PressureLevel) -> float:
        return float(self._v.value(mirror_zone(zone, self.l, self.w), self._policy(p)))

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        # point-reflect the whole V(policy(p)) surface (row + column reversal)
        base = np.asarray(self._v.surface(self._policy(p)))
        return base[::-1, ::-1].copy()

    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]:
        # support of the mirrored cell = mirrored V-support; sparsity is load-bearing (expose it).
        # The wrapped value must expose support (the TurnoverCost contract requires it); every
        # production PossessionValue (MarkovPossessionValue) does, though the base port omits it.
        base = np.asarray(self._v.support(self._policy(p)))  # type: ignore[attr-defined]
        return base[::-1, ::-1].copy()


class EmpiricalTurnoverValue:
    """Model-free cross-check for the mirror assumption (ADR-036 §Part 2.5). NOT shipped.

    For each turnover action, credit the xG of the OPPONENT's first shot in the BOUNDED post-turnover
    window (same game, within window_seconds, before the ball returns to the loser), binned to the
    loss zone/tercile. More sparse than V -- apply the support gate before trusting a cell."""

    def __init__(self, *, l: int = N, w: int = M, window_seconds: float = 10.0) -> None:
        self.l, self.w = l, w
        self.window_seconds = window_seconds
        self._surfaces: dict[int, np.ndarray] = {}
        self._support: dict[int, np.ndarray] = {}
        self._fitted = False

    def fit(self, actions: pd.DataFrame, *, xg_column: str, pressure_column: str, pressure_levels=None):
        from silly_kicks.xtgk._pressure_levels import PressureLevels

        a = actions.reset_index(drop=True).copy()
        if "possession_id" not in a.columns:
            a = add_possessions(a)
        pl = pressure_levels or PressureLevels().fit(a[pressure_column])
        zones = (
            _get_flat_indexes(a.start_x, a.start_y, self.l, self.w).to_numpy()
            if pl.mode == "zone_conditional"
            else None
        )
        a["_p_level"] = pl.apply(a[pressure_column], zones=zones)
        a["_turnover"] = _is_turnover(a)
        a["_opp_shot_xg"] = self._opp_first_shot_after_turnover(a, xg_column, window_seconds=self.window_seconds)
        turnovers = a[a["_turnover"]].dropna(subset=["start_x", "start_y"])
        for p in (1, 2, 3):
            sub = turnovers[turnovers["_p_level"] == p]
            flat = _get_flat_indexes(sub.start_x, sub.start_y, self.l, self.w).to_numpy()
            num = np.zeros(self.w * self.l)
            den = np.zeros(self.w * self.l)
            np.add.at(num, flat, sub["_opp_shot_xg"].to_numpy(dtype=float))
            np.add.at(den, flat, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                surf = np.where(den > 0, num / den, 0.0)
            self._surfaces[p] = surf.reshape((self.w, self.l))
            self._support[p] = den.reshape((self.w, self.l)).astype(int)
        self._fitted = True
        return self

    def _opp_first_shot_after_turnover(self, a: pd.DataFrame, xg_column: str, *, window_seconds: float) -> np.ndarray:
        """Per turnover action, the xG of the OPPONENT's first shot in the BOUNDED post-turnover
        window: same game, within window_seconds, and before the ball returns to the loser's team.
        A minute-10 turnover must NOT be charged an unrelated minute-40 opponent shot (the scan that
        validates the mirror V_opp cannot itself be noisy)."""
        out = np.zeros(len(a), dtype=float)
        team = a["team_id"].to_numpy()
        typ = a["type_id"].to_numpy()
        xg = a[xg_column].fillna(0.0).to_numpy(dtype=float)
        game = a["game_id"].to_numpy() if "game_id" in a.columns else np.zeros(len(a))
        poss = a["possession_id"].to_numpy()
        t = a["time_seconds"].to_numpy(dtype=float)
        turn = a["_turnover"].to_numpy()
        n = len(a)
        for i in range(n):
            if not turn[i]:
                continue
            for j in range(i + 1, n):
                if game[j] != game[i] or (t[j] - t[i]) > window_seconds:
                    break  # out of the bounded window
                if poss[j] == poss[i]:
                    continue  # still the loser's own (briefly interrupted) possession
                if team[j] == team[i]:
                    break  # ball back with the loser -> no opponent-threat credit
                if typ[j] == _SHOT:
                    out[i] = xg[j]
                    break
        return out

    def _check(self):
        if not self._fitted:
            raise NotFittedError("EmpiricalTurnoverValue.fit not called")

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        self._check()
        return self._surfaces[p]

    def value(self, zone: int, p: PressureLevel) -> float:
        self._check()
        return float(self._surfaces[p].ravel()[zone])

    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]:
        self._check()
        return self._support[p]
