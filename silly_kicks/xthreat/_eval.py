"""Held-out transition-model NLL — negative log-likelihood of pass destination zone given
source zone under the transition matrix. NOT an xT-quality metric. See NOTICE / ADR-021.
"""

import hashlib

import numpy as np
import numpy.typing as npt
import pandas as pd

from silly_kicks.xthreat._grid import _get_flat_indexes, _get_successful_move_actions
from silly_kicks.xthreat._params import GridSpec


def holdout_split(
    actions: pd.DataFrame,
    *,
    holdout_fraction: float = 0.15,
    key_cols: tuple[str, ...] = ("game_id",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic match-level holdout split (silly-kicks-native ``game_id`` key).

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions.
    holdout_fraction : float
        Approximate fraction of keys routed to holdout (sha256-bucketed, deterministic).
    key_cols : tuple[str, ...]
        Columns whose joined string identifies a match. ``game_id`` by default; callers with
        richer schemas (e.g. lakehouse ``competition_id``+``match_key``) can override.

    Returns
    -------
    (train, holdout)
        Disjoint at the key level.

    Examples
    --------
    Split a SPADL action stream by match::

        from silly_kicks.xthreat import holdout_split

        train, holdout = holdout_split(actions, holdout_fraction=0.15)
    """
    threshold = round(holdout_fraction * 100)
    keys = actions[list(key_cols)].astype(str).agg("|".join, axis=1)

    def _bucket(k: str) -> int:
        return int(hashlib.sha256(k.encode()).hexdigest(), 16) % 100

    is_holdout = keys.map(lambda k: _bucket(k) < threshold)
    return actions[~is_holdout].copy(), actions[is_holdout].copy()


def compute_holdout_nll(
    transition_matrix: npt.NDArray[np.float64],
    holdout: pd.DataFrame,
    *,
    grid: GridSpec,
    eps: float = 1e-10,
) -> float:
    """Mean held-out NLL of pass destination zone given source zone under ``transition_matrix``.

    ``-mean_i log( T[src_zone_i, dst_zone_i] )`` over successful move rows with valid coords.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Row-stochastic ``(grid.n_zones, grid.n_zones)`` matrix.
    holdout : pd.DataFrame
        Held-out SPADL actions (the successful-move + valid-coord filter is applied internally).
    grid : GridSpec
        The grid the matrix was built on (its resolution defines the zone binning).
    eps : float
        Floor for ``log(0)`` on unobserved (source, destination) pairs.

    Returns
    -------
    float
        Mean negative log-likelihood; ``nan`` if the holdout has no eligible move rows.

    Examples
    --------
    Score a fitted transition matrix on held-out actions::

        from silly_kicks.xthreat import GridSpec, compute_holdout_nll, singh_transition_matrix

        grid = GridSpec(16, 12)
        nll = compute_holdout_nll(singh_transition_matrix(train, grid), holdout, grid=grid)
    """
    # Guard the purity trade-off: going pure (no bundled model) means the matrix and grid
    # could silently disagree. Fail loud instead.
    if transition_matrix.shape != (grid.n_zones, grid.n_zones):
        raise ValueError(f"transition_matrix {transition_matrix.shape} does not match grid {grid.n_zones} zones")
    move = _get_successful_move_actions(holdout).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if len(move) == 0:
        return float("nan")
    l, w = grid.n_zones_x, grid.n_zones_y
    src = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    dst = _get_flat_indexes(move.end_x, move.end_y, l, w).to_numpy()
    probs = transition_matrix[src, dst]
    return float(-np.mean(np.log(np.maximum(probs, eps))))


def compute_holdout_nll_per_group(
    transition_matrix: npt.NDArray[np.float64],
    holdout: pd.DataFrame,
    *,
    grid: GridSpec,
    group_col: str = "game_id",
    eps: float = 1e-10,
) -> dict[str, float]:
    """Per-group held-out NLL (e.g. per game or, with ``group_col`` override, per competition).

    Examples
    --------
    Break held-out NLL down by game::

        from silly_kicks.xthreat import GridSpec, compute_holdout_nll_per_group, singh_transition_matrix

        grid = GridSpec(16, 12)
        per_game = compute_holdout_nll_per_group(
            singh_transition_matrix(train, grid), holdout, grid=grid, group_col="game_id"
        )
    """
    return {
        str(g): compute_holdout_nll(transition_matrix, sub, grid=grid, eps=eps) for g, sub in holdout.groupby(group_col)
    }
