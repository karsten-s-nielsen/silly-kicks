import numpy as np

from silly_kicks._calibration_metrics import ece, reliability_slope


def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert ece(y, p) < 0.05


def test_reliability_slope_near_one_for_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert 0.75 <= reliability_slope(y, p) <= 1.25


def test_reliability_slope_nan_on_degenerate_predictions():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)  # single occupied bin -> slope undefined
    assert np.isnan(reliability_slope(y, p))
