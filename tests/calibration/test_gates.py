import numpy as np
import pandas as pd

from silly_kicks.calibration._features import _TRIAL_DEPENDENT_COLS
from silly_kicks.calibration._gates import (
    default_feature_variances,
    h1_penalty_fires,
    signal_sanity,
)


def test_default_variances_computed_over_trial_cols():
    X = pd.DataFrame({c: np.linspace(0, 1, 50) for c in _TRIAL_DEPENDENT_COLS})
    variances = default_feature_variances(X)
    for c in _TRIAL_DEPENDENT_COLS:
        assert variances[c] > 0


def test_h1_fires_when_a_trial_col_collapses():
    # Default variances measured from healthy data; then a degenerate candidate collapses one col.
    healthy = pd.DataFrame({c: np.linspace(0, 1, 50) for c in _TRIAL_DEPENDENT_COLS})
    defaults = default_feature_variances(healthy)
    degenerate = healthy.copy()
    degenerate["pressure_on_actor__link_zones"] = 0.5  # constant => variance 0
    assert h1_penalty_fires(degenerate, defaults) is True


def test_h1_does_not_fire_for_healthy_features():
    # A healthy trial has ~the same variance as the default params => ratio ~1.0, no penalty.
    X = pd.DataFrame({c: np.linspace(0, 1, 50) for c in _TRIAL_DEPENDENT_COLS})
    defaults = default_feature_variances(X)
    assert h1_penalty_fires(X, defaults) is False


def test_signal_sanity_excludes_zero_signal_provider():
    per_provider = {"gs": 0.85, "idsse": 0.0, "skillcorner": 0.80}
    kept, excluded = signal_sanity(per_provider, min_value=0.01)
    assert "idsse" in excluded
    assert set(kept) == {"gs", "skillcorner"}
