"""Golden-master tests: numba kernels produce identical output to NumPy.

numba is a test dependency (pyproject.toml [test] extra) — these always run in CI.
"""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking.pitch_control._numba_kernels import (
    gaussian_influence_numba,
    influence_numba,
    tti_numba,
)
from silly_kicks.tracking.pitch_control._spearman import _compute_influence, compute_tti


class TestTTIParity:
    def test_fixed_seed_parity(self):
        rng = np.random.default_rng(42)
        pos = rng.uniform(0, 105, (22, 2))
        vel = rng.uniform(-5, 5, (22, 2))
        targets = np.column_stack(
            [
                np.linspace(0, 105, 50).repeat(32),
                np.tile(np.linspace(0, 68, 32), 50),
            ]
        )
        numpy_out = compute_tti(pos, vel, targets, 0.7, 7.0)
        numba_out = tti_numba(pos, vel, targets, 0.7, 7.0)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-12)


class TestInfluenceParity:
    def test_fixed_seed_parity(self):
        rng = np.random.default_rng(123)
        team_tti = rng.uniform(0.5, 5.0, (11, 1600))
        opp_min = rng.uniform(0.5, 5.0, (1600,))
        numpy_out = _compute_influence(team_tti, opp_min, 0.45)
        numba_out = influence_numba(team_tti, opp_min, 0.45)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-12)


class TestGaussianInfluenceParity:
    def test_fixed_seed_parity(self):
        from silly_kicks.tracking.pitch_control._fernandez_bornn import _compute_gaussian_influence

        rng = np.random.default_rng(456)
        targets = rng.uniform(0, 105, (1600, 2))
        mu = rng.uniform(20, 80, (11, 2))
        # Build per-player inverse covariance and determinants from random PD matrices
        inv_cov = np.zeros((11, 2, 2))
        det_cov = np.zeros(11)
        for i in range(11):
            A = rng.standard_normal((2, 2))
            cov = A @ A.T + 0.1 * np.eye(2)
            inv_cov[i] = np.linalg.inv(cov)
            det_cov[i] = np.linalg.det(cov)
        numpy_out = _compute_gaussian_influence(targets, mu, inv_cov, det_cov)
        numba_out = gaussian_influence_numba(targets, mu, inv_cov, det_cov)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-10)
