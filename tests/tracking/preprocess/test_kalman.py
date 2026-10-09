"""TF-65 Task 6: constant-acceleration Kalman/RTS smoother."""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking.preprocess import PreprocessConfig
from silly_kicks.tracking.preprocess._kalman import kalman_ca


def test_kalman_recovers_constant_velocity():
    hz = 25.0
    dt = 1 / hz
    n = 100
    z = 2.0 * np.arange(n) * dt  # 2 m/s
    out = kalman_ca(z, dt, PreprocessConfig.default())
    assert np.allclose(out.vel[20:80], 2.0, atol=0.1)


def test_kalman_variance_grows_through_gap():
    hz = 25.0
    dt = 1 / hz
    n = 120
    z = np.linspace(0, 10, n)
    z[50:80] = np.nan  # a 30-frame occlusion
    out = kalman_ca(z, dt, PreprocessConfig.default())
    assert out.pos_var[65] > out.pos_var[20]  # mid-gap uncertainty exceeds a dense-interior frame
    assert np.isfinite(out.vel[65])  # still predicts through (not NaN) -- honest-uncertainty
