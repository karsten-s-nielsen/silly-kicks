"""SK-xT-3 xT bandwidth/resolution HPO objective — held-out transition NLL (MINIMIZE).

Plain duck-typed object (NOT a ruthless.CachedObjective — the resolution axis means the invariant
is keyed by grid, which a single prepare() does not model). Caches per-(grid, fold) the small
grouped destinations (NOT D²) and re-runs only the shared vectorized gaussian seam per trial — the
same seam the library bottoms out in (definitional equivalence).

See docs/superpowers/specs/2026-06-08-xt-bandwidth-calibration-design.md.

Examples
--------
>>> from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
>>> from ruthless import Candidate
>>> # obj = XtBandwidthObjective(actions, seed=42)  # actions: SPADL + game_id
>>> # obj.evaluate(Candidate(id="t0", params={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"}))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from ruthless.result import Candidate, Metrics

from silly_kicks.calibration._cv import cv_standard_error, match_cv_splits
from silly_kicks.calibration._spaces import grid_from_str
from silly_kicks.xthreat._eval import compute_holdout_nll
from silly_kicks.xthreat._grid import _get_flat_indexes, _get_successful_move_actions
from silly_kicks.xthreat._params import KDEParams
from silly_kicks.xthreat._transitions import (
    _bin_destinations_by_source,
    _gaussian_transition_from_grouped,
    singh_transition_matrix,
)

_EPS = 1e-10  # matches compute_holdout_nll


@dataclass
class _PreparedFold:
    """Param-invariant per-(grid, fold) state: cached grouped train destinations + centres, the
    held-out (src, dst) flat-zone indexes, and the param-free Singh held-out NLL."""

    grouped: dict[int, npt.NDArray[np.float64]]
    centres: npt.NDArray[np.float64]
    holdout_src: npt.NDArray[np.int_]
    holdout_dst: npt.NDArray[np.int_]
    singh_nll: float
    n_holdout_moves: int


class XtBandwidthObjective:
    """ruthless-compatible objective (MINIMIZE ``xt_holdout_nll``).

    Examples
    --------
    >>> from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
    >>> from ruthless import Candidate
    >>> # obj = XtBandwidthObjective(actions, seed=42)
    >>> # obj.evaluate(Candidate(id="t0", params={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"}))
    """

    def __init__(self, actions: pd.DataFrame, *, seed: int = 42, max_points_per_zone: int | None = None) -> None:
        self._actions = actions.reset_index(drop=True)
        self._seed = seed
        self._cap = max_points_per_zone
        self.diagnostics: dict = {}
        self._prepared: dict[tuple[str, int], _PreparedFold] = {}
        # CV folds over game_id — computed once (invariant across all trials). astype(str) makes the
        # grouping robust to provider-asymmetric game_id dtypes: a multi-provider corpus concatenates
        # int and str game_ids into one object column, which match_cv_splits -> np.unique -> sort()
        # cannot order ('<' not supported between int and str). Callers must supply globally-unique
        # game_ids (provider-qualified) — this only guards the dtype crash, not id collisions.
        self._game_ids = self._actions["game_id"].astype(str).to_numpy()
        self._folds = match_cv_splits(self._game_ids)

    def _prepare(self, grid_str: str, train_idx, test_idx) -> _PreparedFold:
        grid = grid_from_str(grid_str)
        train = self._actions.iloc[train_idx]
        test = self._actions.iloc[test_idx]
        grouped, centres = _bin_destinations_by_source(train, grid, max_points_per_zone=self._cap, rng_seed=self._seed)
        move = _get_successful_move_actions(test).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        nx, ny = grid.n_zones_x, grid.n_zones_y
        src = _get_flat_indexes(move.start_x, move.start_y, nx, ny).to_numpy()
        dst = _get_flat_indexes(move.end_x, move.end_y, nx, ny).to_numpy()
        # Singh baseline (param-free) on the SAME split — its filter (_get_move_actions) differs from
        # the KDE grouped cache, so it is computed via the library function, not the cache.
        singh_nll = compute_holdout_nll(singh_transition_matrix(train, grid), test, grid=grid)
        return _PreparedFold(grouped, centres, src, dst, float(singh_nll), len(move))

    def evaluate(self, candidate: Candidate) -> Metrics:
        """K-fold held-out transition NLL for a (bandwidth, adaptive, grid) candidate.

        Examples
        --------
        >>> # obj.evaluate(Candidate(id="t0",  # doctest: +SKIP
        >>> #     params={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"}))["xt_holdout_nll"]
        """
        p = candidate.params
        bandwidth, adaptive, grid_str = float(p["bandwidth"]), bool(p["adaptive"]), str(p["grid"])
        grid = grid_from_str(grid_str)
        params = KDEParams(bandwidth=bandwidth, adaptive=adaptive)  # kernel defaults to gaussian
        kde_nlls: list[float] = []
        singh_nlls: list[float] = []
        n_moves = 0
        for fi, (tr, te) in enumerate(self._folds):
            key = (grid_str, fi)
            prep = self._prepared.get(key)
            if prep is None:
                prep = self._prepare(grid_str, tr, te)
                self._prepared[key] = prep
            if prep.n_holdout_moves == 0:
                continue  # empty holdout fold excluded from the mean
            transition = _gaussian_transition_from_grouped(prep.grouped, prep.centres, grid, params)
            probs = transition[prep.holdout_src, prep.holdout_dst]
            kde_nlls.append(float(-np.mean(np.log(np.maximum(probs, _EPS)))))
            singh_nlls.append(prep.singh_nll)
            n_moves += prep.n_holdout_moves
        if not kde_nlls:
            return {"xt_holdout_nll": float("inf")}  # no-signal: worst score, competes honestly
        return {
            "xt_holdout_nll": float(np.mean(kde_nlls)),
            "xt_holdout_nll_se": cv_standard_error(kde_nlls),
            "singh_holdout_nll": float(np.mean(singh_nlls)),
            "n_folds": float(len(kde_nlls)),
            "n_holdout_moves": float(n_moves),
        }
