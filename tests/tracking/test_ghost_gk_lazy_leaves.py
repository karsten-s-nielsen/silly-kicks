"""`GhostGkModel._training_leaves` is computed on first read, never inside `fit` (HFU plan R3-8 S3 item 14).

No trainer reads the training leaves: the ghost_gk `cv` scores only `predict_mean`, the `fit` stage runs the parity
gate, `save` and `load`, and `save` never persists the per-sample arrays. Only `predict_density` on a locally fit
model reads them. The eager walk in `fit` was 44.6 s of a 101 s `cv` fold on HF (S1g). The ORACLE below is the
eager body that `fit` ran before this change, verbatim: the same function on the same fit-time matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _training_set(n_samples: int = 120, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n_samples, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n_samples).astype(float)
    X["team_in_possession"] = rng.integers(0, 2, n_samples).astype(float)
    X["ball_in_own_half"] = rng.integers(0, 2, n_samples).astype(float)
    X.iloc[::11, 3] = np.nan  # NaN routing through missing_go_to_left
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, n_samples), "gk_y": rng.uniform(25, 45, n_samples)})
    return X, labels


def _fit(features: pd.DataFrame, labels: pd.DataFrame):
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    return GhostGkModel(n_estimators=12).fit(features, labels)


def _oracle_training_leaves(model, features: pd.DataFrame) -> np.ndarray:
    """The pre-change eager body of `fit`, verbatim (with its own fit-time column order)."""
    from silly_kicks.tracking._ghost_gk import _vectorized_leaf_indices

    X_fit = features[model._feature_names()].values.astype(np.float64)
    return _vectorized_leaf_indices(model._tree_nodes, X_fit, flat=model._flat_trees)


def test_fit_runs_no_leaf_walk_over_the_training_rows(monkeypatch):
    import silly_kicks.tracking._ghost_gk as gg

    X, labels = _training_set()
    calls: list[int] = []
    real = gg._vectorized_leaf_indices

    def spy(nodes_list, rows, *, flat=None):
        calls.append(len(rows))
        return real(nodes_list, rows, flat=flat)

    monkeypatch.setattr(gg, "_vectorized_leaf_indices", spy)
    model = _fit(X, labels)
    assert calls == [], f"fit walked {calls} rows; the training leaves must wait for their first read"
    model.predict_mean(X.iloc[:5])  # the served mean never needs them either
    assert calls == []


def test_the_lazy_leaves_equal_the_eager_oracle():
    X, labels = _training_set()
    model = _fit(X, labels)
    expected = _oracle_training_leaves(model, X)
    got = model._training_leaves
    assert got is not None
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape == (len(X), 12)
    assert np.array_equal(got, expected)


def test_the_first_read_releases_the_training_matrix_and_later_reads_reuse_the_leaves():
    X, labels = _training_set()
    model = _fit(X, labels)
    first = model._training_leaves
    assert model._training_X is None
    assert model._training_leaves is first


def test_predict_density_is_byte_identical_to_the_eager_leaves():
    X, labels = _training_set()
    lazy = _fit(X, labels)
    eager = _fit(X, labels)
    eager._training_leaves = _oracle_training_leaves(eager, X)  # the pre-change state, set explicitly
    for backend in ("vectorized", "scipy"):
        got = lazy.predict_density(X.iloc[:4], kde_backend=backend)
        want = eager.predict_density(X.iloc[:4], kde_backend=backend)
        assert len(got) == len(want) == 4
        for a, b in zip(got, want, strict=True):
            assert a.probabilities.tobytes() == b.probabilities.tobytes()
            got_stats = (a.mode_x, a.mode_y, a.mean_x, a.mean_y, a.spread)
            assert got_stats == (b.mode_x, b.mode_y, b.mean_x, b.mean_y, b.spread)


def test_a_loaded_model_still_refuses_predict_density(tmp_path):
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    X, labels = _training_set()
    model = _fit(X, labels)
    model.save(tmp_path / "m")
    loaded = GhostGkModel.load(tmp_path / "m")
    assert loaded._training_leaves is None
    with pytest.raises(RuntimeError, match="parameters-only artifact"):
        loaded.predict_density(X.iloc[:2])


def test_an_unfitted_model_has_no_training_leaves():
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    model = GhostGkModel()
    assert model._training_leaves is None
    assert model._training_X is None
