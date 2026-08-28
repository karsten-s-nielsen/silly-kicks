"""Performance benchmark for Ghost-GK (TF-18)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES,
    GhostGkModel,
    _vectorized_leaf_indices,
)


@pytest.fixture(scope="module")
def trained_model():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((500, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 500).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 500), "gk_y": rng.uniform(25, 45, 500)})
    model = GhostGkModel(n_estimators=50)
    model.fit(X, labels)
    return model, X


def test_leaf_traversal_performance(trained_model, benchmark):
    """Vectorized leaf traversal should handle 1000 samples."""
    model, X = trained_model
    batch = pd.concat([X] * 2, ignore_index=True).iloc[:1000]
    X_arr = batch.values.astype(np.float64)

    # Measure the PRODUCTION path: pass the cached flat trees so the traversal dispatches to the
    # numba kernel when installed (matching predict_mean/predict_density), not the numpy fallback.
    result = benchmark(lambda: _vectorized_leaf_indices(model._tree_nodes, X_arr, flat=model._flat_trees))
    assert result.shape == (1000, 50)


def test_predict_density_performance(trained_model, benchmark):
    """predict_density for 10 samples."""
    model, X = trained_model
    batch = X.iloc[:10]

    result = benchmark(lambda: model.predict_density(batch))
    assert len(result) == 10
