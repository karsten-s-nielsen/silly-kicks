import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_prepare_retention_training_data_builds_labels_and_features():
    from scripts.train_gk_retention import prepare_retention_training_data
    from tests.xtgk.conftest import three_band_cohort

    actions = three_band_cohort()  # no is_gk_distribution column -> goalkick-only domain fallback
    X, y, groups = prepare_retention_training_data(actions)
    assert len(X) == len(y) == len(groups)
    assert set(np.unique(y)) <= {0, 1}


def test_calibration_gate_passes_on_calibrated_predictions():
    from scripts.train_gk_retention import calibration_gate

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)
    ok, metrics = calibration_gate(y, p)
    assert ok is True
    assert metrics["ece"] <= 0.10
