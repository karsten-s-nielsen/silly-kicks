"""TurnoverCost port + two adapters (ADR-036 §Part 2).

The turnover term needs `V_opp` = the expected OPPONENT threat after a loss of possession at (zone, p).

- **EmpiricalTurnoverValue** (the FAITHFUL production adapter, Eyestone §2.3, promoted 4.45.0/PR-S112):
  estimates V_opp from OBSERVED post-turnover possessions, indexed by loss-zone x pressure,
  possession-bound with a support-gated hierarchical fallback. This is what the metric injects.
- **MirroredTurnoverCost** (geometric proxy, retained as a cross-check / disentanglement comparator):
  since V(z,p) is team-agnostic (pooled attack-LTR), it approximates V_opp as V at the 180-degree
  mirror zone (`p_opp = p`) — zero new fitting. On real data this OVER-STATES deep opponent threat
  ~10-50x vs the faithful estimate, which is why it was demoted from production to comparator.

`surface_divergence` reports the per-zone gap between the two (the gap is itself a finding).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import _sort_actions_chronological_or_action_id, add_possessions
from silly_kicks.xtgk._moves import _is_turnover
from silly_kicks.xtgk._possession_value import M, N, PossessionValue, PressureLevel, mirror_zone
from silly_kicks.xthreat._grid import _get_flat_indexes

_SHOT = spadlconfig.actiontype_id["shot"]


def _equality_codes(series: pd.Series) -> np.ndarray:
    """Map ``series`` to int64 codes that preserve ``==`` EXACTLY, so the turnover scan can run on a
    numeric (numba-compatible, dtype-agnostic) array (ADR-068). ``pd.factorize`` gives equal values
    equal codes; the one place it would diverge from a raw ``==`` is NA -- factorize collapses every
    NA to a single ``-1`` (so ``-1 == -1`` would read as "same"), but raw ``NaN == NaN`` is False.
    So each NA gets a DISTINCT negative code, reproducing NaN-never-equals-NaN (GS null-actor teams,
    ADR-027). game_id is NA-free (fit's input guard) and possession_id is an int; only team_id hits
    this in practice."""
    codes, _ = pd.factorize(series, use_na_sentinel=True)
    codes = np.asarray(codes, dtype=np.int64).copy()
    na = codes == -1
    n_na = int(na.sum())
    if n_na:
        codes[na] = -np.arange(1, n_na + 1, dtype=np.int64)  # distinct negatives; real codes are >= 0
    return codes


def _opp_first_shot_scan(
    turn: np.ndarray,
    game_c: np.ndarray,
    poss_c: np.ndarray,
    team_c: np.ndarray,
    typ: np.ndarray,
    xg: np.ndarray,
    t: np.ndarray,
    shot_id: int,
    window: float,
) -> np.ndarray:
    """Per turnover, the xG of the opponent's first post-turnover shot. VERBATIM arithmetic of the
    prior nested loop, on equality-preserving int codes: match boundary (``game_c``) always bounds
    the scan; ``poss_c`` skips the loser's own possession; ``team_c`` equal = ball back (break); a
    finite window is passed as-is and ``window=inf`` reproduces ``window_seconds is None`` (the
    ``t[j]-t[i] > inf`` test is always False). @njit-compiled when numba is present."""
    n = turn.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not turn[i]:
            continue
        gi = game_c[i]
        pi = poss_c[i]
        team_i = team_c[i]
        ti = t[i]
        for j in range(i + 1, n):
            if game_c[j] != gi or (t[j] - ti) > window:
                break  # out of the match boundary / bounded window
            if poss_c[j] == pi:
                continue  # still the loser's own (briefly interrupted) possession
            if team_c[j] == team_i:
                break  # ball back with the loser -> no opponent-threat credit
            if typ[j] == shot_id:
                out[i] = xg[j]
                break
    return out


# The numba dispatcher and the pure-Python fn share this call signature; annotate so callers
# (and pyright) see one callable regardless of whether numba resolved.
_opp_first_shot_scan_fast: Callable[..., np.ndarray]
try:  # numba accelerates the O(n*k) scan ~100x; pure-Python fallback stays byte-identical (ADR-068)
    from numba import njit as _njit

    _opp_first_shot_scan_fast = _njit(cache=True)(_opp_first_shot_scan)
    _NUMBA_TURNOVER = True
except Exception:  # numba absent (it is an optional extra) -> the pure-Python kernel above
    _opp_first_shot_scan_fast = _opp_first_shot_scan
    _NUMBA_TURNOVER = False


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
    """Faithful production `V_opp` (ADR-036 §Part 2; Eyestone §2.3): expected OPPONENT threat following a
    loss of possession, estimated from OBSERVED post-turnover possessions, indexed by loss-zone x pressure.

    **Scope: POSSESSION-BOUND by default** (``window_seconds=None``) — the opponent's first-shot xG over
    their WON possession, before the ball returns to the loser. This is scope-symmetric with V (the
    possessing team's first shot over its possession), so a `dzv`-magnitude change vs the old mirror proxy
    reflects a real over-statement, not a window artifact. A finite ``window_seconds`` gives the
    immediate-danger SENSITIVITY variant only.

    **Sparsity (§2.3): a support-gated hierarchical fallback.** Deep keeper turnovers are rare → the native
    cell is 0/noise there. Every cell resolves to the finest estimate with ``>= min_support`` support:
    native (zone,p) cell → coarse block (``coarsen``x``coarsen`` native cells) → global-per-pressure.
    ``support(p)`` exposes the honest native n; ``resolution_level(p)`` the level used (0/1/2; -1 = no data).
    ``min_support`` defaults to the pre-registered deep-zone gate ``n_min``.

    Caveat (``zone_conditional`` terciles): native cells pooled into one coarse block may have had their
    p-level from different per-band cutpoints, so a block's ``p=2`` can mix different absolute pressures
    (second-order — a coarse block is a spatial neighborhood, usually one band).

    Requires a non-null ``game_id`` (or ``match_key``): possession-bound scope uses the match boundary as
    the only scan bound, so a missing/null id would let the scan cross matches (ADR-017/019 input guard).
    """

    def __init__(
        self, *, l: int = N, w: int = M, window_seconds: float | None = None, min_support: int = 30, coarsen: int = 4
    ) -> None:
        self.l, self.w = l, w
        self.window_seconds = window_seconds
        self.min_support = min_support
        self.coarsen = coarsen
        self._surfaces: dict[int, np.ndarray] = {}
        self._support: dict[int, np.ndarray] = {}
        self._levels: dict[int, np.ndarray] = {}
        self._fitted = False

    def fit(self, actions: pd.DataFrame, *, xg_column: str, pressure_column: str, pressure_levels=None):
        from silly_kicks.xtgk._pressure_levels import PressureLevels

        a = actions.reset_index(drop=True).copy()
        if "game_id" not in a.columns or a["game_id"].isna().any():
            raise ValueError(
                "EmpiricalTurnoverValue.fit requires a non-null game_id (or match_key) on every row: the "
                "possession-bound scope (window_seconds=None) uses the match boundary as the only scan bound, "
                "so a missing/null game_id lets the scan charge a turnover with an opponent shot from a "
                "different match. (ADR-017/019 input guard.)"
            )
        # ``_opp_first_shot_after_turnover`` is a POSITIONAL forward scan: its correctness requires rows
        # contiguous-by-game AND chronological within a game. A mart read (e.g. Databricks EXTERNAL_LINKS +
        # Arrow chunks) carries NO order guarantee, so sort here rather than trust the caller -- V_opp is a
        # pure function of event CONTENT, not input row order (ADR-065 robust key; period_id is load-bearing
        # because time_seconds is period-relative). A no-op on already-chronological input.
        a = _sort_actions_chronological_or_action_id(a)
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
        ncell, cf = self.w * self.l, self.coarsen
        lb, wb = (self.l + cf - 1) // cf, (self.w + cf - 1) // cf  # coarse cols, rows
        xi = np.arange(ncell) % self.l
        yj = np.arange(ncell) // self.l
        block = (yj // cf) * lb + (xi // cf)  # native cell -> coarse block id
        for p in (1, 2, 3):
            sub = turnovers[turnovers["_p_level"] == p]
            flat = _get_flat_indexes(sub.start_x, sub.start_y, self.l, self.w).to_numpy().astype(int)
            num = np.zeros(ncell)
            den = np.zeros(ncell)
            np.add.at(num, flat, sub["_opp_shot_xg"].to_numpy(dtype=float))
            np.add.at(den, flat, 1.0)
            bnum = np.zeros(lb * wb)
            bden = np.zeros(lb * wb)
            np.add.at(bnum, block, num)
            np.add.at(bden, block, den)
            gnum, gden = num.sum(), den.sum()
            native_ok = den >= self.min_support
            block_ok = bden[block] >= self.min_support
            with np.errstate(invalid="ignore", divide="ignore"):
                nat = np.where(den > 0, num / den, 0.0)
                blk = np.where(bden[block] > 0, bnum[block] / bden[block], 0.0)
            glob = gnum / gden if gden >= self.min_support else 0.0
            resolved = np.where(native_ok, nat, np.where(block_ok, blk, glob if gden >= self.min_support else 0.0))
            level = np.where(native_ok, 0, np.where(block_ok, 1, np.where(gden >= self.min_support, 2, -1)))
            self._surfaces[p] = resolved.reshape((self.w, self.l))
            self._support[p] = den.reshape((self.w, self.l)).astype(int)
            self._levels[p] = level.reshape((self.w, self.l))
        self._fitted = True
        return self

    def _opp_first_shot_after_turnover(
        self, a: pd.DataFrame, xg_column: str, *, window_seconds: float | None
    ) -> np.ndarray:
        """Per turnover, the xG of the OPPONENT's first shot over their won possession (possession-bound,
        ``window_seconds=None``) or within a bounded window (a finite ``window_seconds``). Bounds: the match
        boundary (``game_id``) always; the ball returning to the loser always; the time cap only if finite.

        ADR-068: the O(n*k) forward scan runs in :func:`_opp_first_shot_scan` on equality-preserving int
        codes (numba-compiled when available; byte-identical pure-Python fallback otherwise). ``a`` is
        pre-sorted chronologically by ``fit`` (item A), which the scan relies on."""
        xg = a[xg_column].fillna(0.0).to_numpy(dtype=float)
        t = a["time_seconds"].to_numpy(dtype=float)
        turn = a["_turnover"].to_numpy(dtype=bool)
        typ = a["type_id"].to_numpy(dtype=np.int64)
        # window_seconds=None => inf, so `(t[j]-t[i]) > inf` is always False (== the None branch).
        window = np.inf if window_seconds is None else float(window_seconds)
        return _opp_first_shot_scan_fast(
            turn,
            _equality_codes(a["game_id"]),
            _equality_codes(a["possession_id"]),
            _equality_codes(a["team_id"]),
            typ,
            xg,
            t,
            _SHOT,
            window,
        )

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

    def resolution_level(self, p: PressureLevel) -> npt.NDArray[np.int_]:
        """Per-cell fallback level used: 0 native / 1 coarse block / 2 global / -1 no data (report guard)."""
        self._check()
        return self._levels[p]


def surface_divergence(a, b, p: PressureLevel) -> npt.NDArray[np.float64]:
    """Per-zone ``|a.surface(p) - b.surface(p)|`` between two TurnoverCost surfaces (empirical-vs-mirror
    report helper; deep-cell divergence is itself a finding — it is what the mirror silently assumed)."""
    return np.abs(np.asarray(a.surface(p)) - np.asarray(b.surface(p)))
