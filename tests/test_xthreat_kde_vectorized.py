"""SK-xT-3 Task 1: the vectorized gaussian KDE seam == sklearn gaussian (row-normalized) in the
non-underflow regime, and is finite/stable in the small-h underflow regime where the old
sklearn-wrapper degenerated to the mean-row fallback. See the spec §0 (review M5)."""

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

from silly_kicks.xthreat import KDEParams
from silly_kicks.xthreat._eval import compute_holdout_nll, holdout_split
from silly_kicks.xthreat._params import GridSpec
from silly_kicks.xthreat._transitions import (
    _bin_destinations_by_source,
    _gaussian_transition_from_grouped,
    kde_smoothed_transition_matrix,
)
from tests._xthreat_helpers import _moves, _sparse_overfit_corpus


def _sklearn_reference(grouped, centres, grid, params):
    """The old per-zone sklearn gaussian path, row-normalized — the parity reference."""
    n = grid.n_zones_x * grid.n_zones_y
    from silly_kicks.xthreat._transitions import silverman_2d

    T = np.zeros((n, n))
    populated = []
    for s, pts in grouped.items():
        if pts.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((pts[:, 0].var() + pts[:, 1].var()) / 2.0)) or 1e-6
            h = params.bandwidth * silverman_2d(pts.shape[0], sigma)
        else:
            h = params.bandwidth
        dens = np.exp(KernelDensity(kernel="gaussian", bandwidth=h).fit(pts).score_samples(centres))
        if dens.sum() > 0:
            T[s] = dens / dens.sum()
            populated.append(s)
    if populated:
        mean_row = T[populated].mean(axis=0)
        mean_row = mean_row / mean_row.sum() if mean_row.sum() > 0 else np.full(n, 1.0 / n)
        for s in range(n):
            if s not in populated:
                T[s] = mean_row
    return T


@pytest.mark.parametrize("adaptive,bandwidth", [(True, 0.5), (True, 1.0), (True, 2.0), (False, 2.0), (False, 5.0)])
def test_vectorized_gaussian_matches_sklearn_non_underflow(adaptive, bandwidth):
    grid = GridSpec(6, 4)
    grouped, centres = _bin_destinations_by_source(_moves(n_per_zone=120), grid)
    params = KDEParams(bandwidth=bandwidth, adaptive=adaptive)
    vec = _gaussian_transition_from_grouped(grouped, centres, grid, params)
    ref = _sklearn_reference(grouped, centres, grid, params)
    np.testing.assert_allclose(vec, ref, rtol=0, atol=1e-9)


def test_vectorized_gaussian_finite_in_underflow_regime():
    # adaptive=False, bandwidth=0.1 (raw metres) -> the old sklearn wrapper underflowed to 0 -> mean
    # row. The vectorized path must stay FINITE and row-stochastic (strictly more correct).
    grid = GridSpec(6, 4)
    grouped, centres = _bin_destinations_by_source(_moves(n_per_zone=120), grid)
    T = _gaussian_transition_from_grouped(grouped, centres, grid, KDEParams(bandwidth=0.1, adaptive=False))
    assert np.all(np.isfinite(T))
    np.testing.assert_allclose(T.sum(axis=1), np.ones(grid.n_zones), atol=1e-9)


def test_library_composes_the_shared_seam():
    # M6: kde_smoothed_transition_matrix bottoms out in the same seam — definitional, not a gate.
    grid = GridSpec(6, 4)
    df = _moves(n_per_zone=120)
    params = KDEParams(bandwidth=1.5, adaptive=True)
    grouped, centres = _bin_destinations_by_source(df, grid)
    np.testing.assert_array_equal(
        kde_smoothed_transition_matrix(df, grid, params),
        _gaussian_transition_from_grouped(grouped, centres, grid, params),
    )


def test_kde_holdout_nll_characterization_pin():
    # Scalar golden: pins the re-pinned gaussian numerics against accidental future drift, tolerant
    # to numpy micro-version noise. Value generated from the committed implementation (Step 6).
    df = _sparse_overfit_corpus(seed=3, n_games=20)
    train, holdout = holdout_split(df, holdout_fraction=0.25)
    grid = GridSpec(16, 12)
    T = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=1.0, adaptive=True))
    nll = compute_holdout_nll(T, holdout, grid=grid)
    assert nll == pytest.approx(_EXPECTED_NLL, abs=1e-4)


_EXPECTED_NLL = 3.069325  # characterization pin (vectorized gaussian core, GridSpec(16,12))
