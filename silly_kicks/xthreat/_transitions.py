"""Transition-matrix builders for the xT model. See NOTICE for full bibliographic citations."""

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _get_flat_indexes, _get_move_actions, _get_successful_move_actions
from silly_kicks.xthreat._params import GridSpec, KDEParams


def singh_transition_matrix(actions: pd.DataFrame, grid: GridSpec) -> npt.NDArray[np.float64]:
    """Row-normalized empirical move-transition counts (classic Singh 2018).

    Byte-identical to the legacy ``_move_transition_matrix(actions, grid.n_zones_x, grid.n_zones_y)``.

    Examples
    --------
    Build the Singh transition matrix for a grid::

        from silly_kicks.xthreat import GridSpec, singh_transition_matrix

        T = singh_transition_matrix(actions, GridSpec(16, 12))  # (192, 192) row-stochastic
    """
    l, w = grid.n_zones_x, grid.n_zones_y
    n = w * l
    move_actions = _get_move_actions(actions)
    move_actions = move_actions.dropna(subset=["start_x", "start_y", "end_x", "end_y"])

    start_cell = _get_flat_indexes(move_actions.start_x, move_actions.start_y, l, w).to_numpy()
    end_cell = _get_flat_indexes(move_actions.end_x, move_actions.end_y, l, w).to_numpy()
    is_success = (move_actions.result_id == spadlconfig.result_id["success"]).to_numpy()

    # Vectorized, byte-identical to the legacy per-zone boolean-mask loop (same integer
    # operands -> same float64 division). O(n_actions + n_zones^2) instead of
    # O(n_zones * n_actions). Denominator = ALL moves per start cell; numerator =
    # successful moves per (start, end) cell.
    start_counts = np.zeros(n)
    np.add.at(start_counts, start_cell, 1.0)
    counts = np.zeros((n, n))
    np.add.at(counts, (start_cell[is_success], end_cell[is_success]), 1.0)

    transition_matrix = np.zeros((n, n))
    nz = start_counts > 0
    transition_matrix[nz] = counts[nz] / start_counts[nz, None]
    return transition_matrix


def silverman_2d(n: int, sigma: float) -> float:
    """Silverman's rule-of-thumb bandwidth in 2D: h = n^(-1/6) * sigma.

    (4/(d+2))^(1/(d+4)) with d=2 simplifies to 1. Silverman (1986). See NOTICE.

    Examples
    --------
    Compute a 2D rule-of-thumb bandwidth::

        from silly_kicks.xthreat import silverman_2d

        h = silverman_2d(n=400, sigma=8.0)
    """
    return float(n ** (-1 / 6) * sigma)


def _zone_centres(grid: GridSpec) -> npt.NDArray[np.float64]:
    """(n_zones, 2) SPADL coords of each flat-index zone centre, matching ``_get_flat_indexes``.

    Legacy flat index = (w-1 - yj)*l + xi  =>  xi = flat % l ;  yj = (w-1) - flat // l.
    """
    l, w = grid.n_zones_x, grid.n_zones_y
    flat = np.arange(l * w)
    xi = flat % l
    yj = (w - 1) - (flat // l)
    cx = (xi + 0.5) * grid.cell_length
    cy = (yj + 0.5) * grid.cell_width
    return np.column_stack([cx, cy]).astype(np.float64)


def kde_smoothed_transition_matrix(actions: pd.DataFrame, grid: GridSpec, params: KDEParams) -> npt.NDArray[np.float64]:
    """Per-source-zone 2D KDE-smoothed move-transition matrix.

    Salimi et al. 2026 (poster) reproduction; Silverman 1986 bandwidth. See NOTICE. Indexed by
    silly-kicks flat zone indices (consistent with ``singh_transition_matrix`` + value iteration).

    Examples
    --------
    Build a KDE-smoothed transition matrix::

        from silly_kicks.xthreat import GridSpec, KDEParams, kde_smoothed_transition_matrix

        T = kde_smoothed_transition_matrix(actions, GridSpec(16, 12), KDEParams(bandwidth=2.0))
    """
    from sklearn.neighbors import KernelDensity

    l, w = grid.n_zones_x, grid.n_zones_y
    n = l * w
    move = _get_successful_move_actions(actions).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
    centres = _zone_centres(grid)

    T = np.zeros((n, n), dtype=np.float64)
    populated: list[int] = []
    for s in range(n):
        rows = end_xy[start_cell == s]
        if rows.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((rows[:, 0].var() + rows[:, 1].var()) / 2.0))
            if sigma == 0.0:
                sigma = 1e-6
            h = params.bandwidth * silverman_2d(rows.shape[0], sigma)
        else:
            h = params.bandwidth
        kde = KernelDensity(kernel=params.kernel, bandwidth=h).fit(rows)
        dens = np.exp(kde.score_samples(centres))
        total = dens.sum()
        if total > 0:
            T[s] = dens / total
            populated.append(s)

    if populated:
        mean_row = T[populated].mean(axis=0)
        s_mean = mean_row.sum()
        mean_row = mean_row / s_mean if s_mean > 0 else np.full(n, 1.0 / n)
        for s in range(n):
            if s not in populated:
                T[s] = mean_row
    else:
        T[:] = 1.0 / n
    return T
