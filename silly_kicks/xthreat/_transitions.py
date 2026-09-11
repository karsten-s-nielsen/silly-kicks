"""Transition-matrix builders for the xT model. See NOTICE for full bibliographic citations."""

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _get_flat_indexes, _get_move_actions, _get_successful_move_actions
from silly_kicks.xthreat._params import GridSpec, KDEParams


def singh_transition_matrix(actions: pd.DataFrame, grid: GridSpec) -> npt.NDArray[np.float64]:
    """Empirical move-transition matrix (classic Singh 2018): successful moves per (start,end) over
    ALL moves per start cell. Byte-identical to the legacy
    ``_move_transition_matrix(actions, grid.n_zones_x, grid.n_zones_y)``.

    NOTE: rows are **sub-stochastic** — ``Σ_j T[i,j] = P(success | move from i) ≤ 1`` (the missing
    mass is the failure probability). Contrast ``kde_smoothed_transition_matrix``, whose rows ARE
    row-stochastic (density normalized to 1).

    Examples
    --------
    Build the Singh transition matrix for a grid::

        from silly_kicks.xthreat import GridSpec, singh_transition_matrix

        T = singh_transition_matrix(actions, GridSpec(16, 12))  # (192, 192), rows sub-stochastic
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


def _bin_destinations_by_source(
    actions: pd.DataFrame,
    grid: GridSpec,
    *,
    max_points_per_zone: int | None = None,
    rng_seed: int | None = None,
) -> tuple[dict[int, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Group successful-move destinations by source zone in a SINGLE vectorized pass.

    Returns ``(grouped, centres)`` where ``grouped[s]`` is the ``(n_s, 2)`` destination coords of
    moves starting in flat zone ``s`` and ``centres`` is ``(n_zones, 2)``. ``grouped`` is the small
    param-invariant artifact the calibration objective caches (NOT pairwise D², which is
    ``n_s x n_zones`` and OOMs at scale). ``argsort + split`` replaces the legacy
    ``for s: end_xy[start_cell == s]`` mask-in-loop. Optional deterministic per-zone subsample bounds
    per-trial cdist FLOPs / pathological-zone memory; default ``(None, None)`` keeps every row
    (byte-identical grouping to the legacy binning).

    Examples
    --------
    Group a small SPADL corpus by source zone::

        from silly_kicks.xthreat import GridSpec
        from silly_kicks.xthreat._transitions import _bin_destinations_by_source

        grouped, centres = _bin_destinations_by_source(actions, GridSpec(16, 12))
    """
    l, w = grid.n_zones_x, grid.n_zones_y
    centres = _zone_centres(grid)
    move = _get_successful_move_actions(actions).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if len(move) == 0:
        return {}, centres
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
    order = np.argsort(start_cell, kind="stable")
    sc_sorted = start_cell[order]
    end_sorted = end_xy[order]
    boundaries = np.flatnonzero(np.diff(sc_sorted)) + 1
    zone_per_group = sc_sorted[np.concatenate(([0], boundaries))]
    groups = np.split(end_sorted, boundaries)
    rng = np.random.default_rng(rng_seed)
    grouped: dict[int, npt.NDArray[np.float64]] = {}
    for s, pts in zip(zone_per_group, groups, strict=True):
        if max_points_per_zone is not None and len(pts) > max_points_per_zone:
            pts = pts[rng.choice(len(pts), size=max_points_per_zone, replace=False)]
        grouped[int(s)] = pts
    return grouped, centres


def _gaussian_transition_from_grouped(
    grouped: dict[int, npt.NDArray[np.float64]],
    centres: npt.NDArray[np.float64],
    grid: GridSpec,
    params: KDEParams,
) -> npt.NDArray[np.float64]:
    """SHARED vectorized gaussian KDE seam — called by both the library core and the calibration
    objective (equivalence is definitional, one function).

    Per source zone with destinations ``pts``: ``logits = -D2 / (2h^2)`` where ``D2`` is the
    ``(n_zones, n_s)`` pairwise squared distance from centres to pts; subtract the SCALAR global max
    of ``logits`` (softmax stabilization — cancels in the row-normalization, prevents small-h
    underflow; a per-centre max would corrupt the distribution); ``dens = exp(stabilized).sum``
    over pts; row-normalize. Unpopulated zones get the populated mean row (matches the legacy
    sklearn path's ``if total > 0 else mean-row`` branch).

    Examples
    --------
    Build a gaussian transition matrix from pre-grouped destinations::

        from silly_kicks.xthreat import GridSpec, KDEParams
        from silly_kicks.xthreat._transitions import _bin_destinations_by_source, _gaussian_transition_from_grouped

        grouped, centres = _bin_destinations_by_source(actions, GridSpec(16, 12))
        T = _gaussian_transition_from_grouped(grouped, centres, GridSpec(16, 12), KDEParams())
    """
    n = grid.n_zones_x * grid.n_zones_y
    T = np.zeros((n, n), dtype=np.float64)
    populated: list[int] = []
    for s, pts in grouped.items():
        if pts.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((pts[:, 0].var() + pts[:, 1].var()) / 2.0))
            if sigma == 0.0:
                sigma = 1e-6
            h = params.bandwidth * silverman_2d(pts.shape[0], sigma)
        else:
            h = params.bandwidth
        d2 = ((centres[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)  # (n_zones, n_s)
        logits = -d2 / (2.0 * h * h)
        logits = logits - logits.max()  # SCALAR global max — stabilize, cancels in normalization
        dens = np.exp(logits).sum(axis=1)  # (n_zones,)
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


def _kde_transition_from_grouped(
    grouped: dict[int, npt.NDArray[np.float64]],
    centres: npt.NDArray[np.float64],
    grid: GridSpec,
    params: KDEParams,
) -> npt.NDArray[np.float64]:
    """Dispatch the KDE core on ``params.kernel``: ``"gaussian"`` -> the vectorized shared seam;
    any other kernel -> the sklearn ``KernelDensity`` fallback (unchanged generality).

    Examples
    --------
    ::

        from silly_kicks.xthreat import GridSpec, KDEParams
        from silly_kicks.xthreat._transitions import _bin_destinations_by_source, _kde_transition_from_grouped

        grouped, centres = _bin_destinations_by_source(actions, GridSpec(16, 12))
        T = _kde_transition_from_grouped(grouped, centres, GridSpec(16, 12), KDEParams())
    """
    if params.kernel == "gaussian":
        return _gaussian_transition_from_grouped(grouped, centres, grid, params)
    from sklearn.neighbors import KernelDensity

    n = grid.n_zones_x * grid.n_zones_y
    T = np.zeros((n, n), dtype=np.float64)
    populated: list[int] = []
    for s, pts in grouped.items():
        if pts.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((pts[:, 0].var() + pts[:, 1].var()) / 2.0))
            if sigma == 0.0:
                sigma = 1e-6
            h = params.bandwidth * silverman_2d(pts.shape[0], sigma)
        else:
            h = params.bandwidth
        dens = np.exp(KernelDensity(kernel=params.kernel, bandwidth=h).fit(pts).score_samples(centres))
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


def kde_smoothed_transition_matrix(actions: pd.DataFrame, grid: GridSpec, params: KDEParams) -> npt.NDArray[np.float64]:
    """Per-source-zone 2D KDE-smoothed move-transition matrix.

    Salimi et al. 2026 (poster) reproduction; Silverman 1986 bandwidth. See NOTICE. Indexed by
    silly-kicks flat zone indices (consistent with ``singh_transition_matrix`` + value iteration).
    The gaussian kernel (default) runs the vectorized shared seam; other kernels use sklearn.

    Examples
    --------
    Build a KDE-smoothed transition matrix::

        from silly_kicks.xthreat import GridSpec, KDEParams, kde_smoothed_transition_matrix

        T = kde_smoothed_transition_matrix(actions, GridSpec(16, 12), KDEParams(bandwidth=2.0))
    """
    grouped, centres = _bin_destinations_by_source(actions, grid)
    return _kde_transition_from_grouped(grouped, centres, grid, params)
