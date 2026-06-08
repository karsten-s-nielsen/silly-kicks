import numpy as np
import pytest

from silly_kicks.xthreat import (
    GridSpec,
    KDEParams,
    compute_holdout_nll,
    holdout_split,
    kde_smoothed_transition_matrix,
    singh_transition_matrix,
)
from tests._xthreat_helpers import _sparse_overfit_corpus, _worldcup_ltr


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kde_strictly_beats_singh_on_synthetic_sparse_corpus(seed):
    """Singh overfits the sparse, wide-jitter rows; KDE smoothing strictly lowers held-out NLL.
    Asserted across 5 seeds — the mechanism, not one favorable draw."""
    grid = GridSpec(n_zones_x=12, n_zones_y=8)
    df = _sparse_overfit_corpus(seed=seed)
    train, holdout = holdout_split(df, holdout_fraction=0.25, key_cols=("game_id",))
    assert len(train) > 0 and len(holdout) > 0, f"seed={seed}: degenerate split"
    singh = singh_transition_matrix(train, grid)
    kde = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=3.0, adaptive=True))
    nll_singh = compute_holdout_nll(singh, holdout, grid=grid)
    nll_kde = compute_holdout_nll(kde, holdout, grid=grid)
    assert nll_kde < nll_singh, f"seed={seed}: KDE {nll_kde} should beat Singh {nll_singh}"


def test_kde_bandwidth_sweep_worldcup_diagnostic(sb_worldcup_data, capsys):
    """Widen past the lakehouse's saturated 2.0 edge. Logs the Singh baseline + the KDE NLL
    curve over bandwidths so the chosen KDEParams.bandwidth default is justified (NOT asserting)."""
    actions = _worldcup_ltr(sb_worldcup_data)
    grid = GridSpec(n_zones_x=16, n_zones_y=12)  # silly-kicks default resolution
    train, holdout = holdout_split(actions, holdout_fraction=0.15, key_cols=("game_id",))
    nll_singh = compute_holdout_nll(singh_transition_matrix(train, grid), holdout, grid=grid)
    curve = {}
    for bw in (1.0, 2.0, 4.0, 8.0):
        T = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=bw, adaptive=True))
        curve[bw] = compute_holdout_nll(T, holdout, grid=grid)
    with capsys.disabled():
        print(f"\n[xT bandwidth sweep 16x12 WC2018] Singh NLL={nll_singh:.5f}")
        for bw, nll in curve.items():
            print(f"  bw={bw:>4}: KDE NLL={nll:.5f}  delta_vs_singh={nll_singh - nll:+.5f}")
    assert all(np.isfinite(v) for v in curve.values())  # sanity only
