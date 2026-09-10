import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.expected_passing import PassCompletionIntegrityError, PassCompletionModel


def _passes(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ox = rng.uniform(0, 105, n)
    oy = rng.uniform(0, 68, n)
    tx = np.clip(ox + rng.uniform(-5, 30, n), 0, 105)
    ty = np.clip(oy + rng.uniform(-20, 20, n), 0, 68)
    dist = np.hypot(tx - ox, ty - oy)
    p = 1.0 / (1.0 + np.exp((dist - 20) / 8))  # longer pass -> lower completion (true DGP)
    completed = rng.uniform(size=n) < p
    return pd.DataFrame(
        {
            "type_id": spadlconfig.actiontype_id["pass"],
            "result_id": np.where(completed, spadlconfig.result_id["success"], spadlconfig.result_id["fail"]),
            "start_x": ox,
            "start_y": oy,
            "end_x": tx,
            "end_y": ty,
        }
    )


def test_predict_is_monotone_decreasing_in_distance():
    m = PassCompletionModel().fit(_passes())
    short = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([28.0]), np.array([34.0]))
    long = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([34.0]))
    assert 0 <= long[0] < short[0] <= 1


def test_predict_is_pure_numpy_no_sklearn_import(monkeypatch):
    m = PassCompletionModel().fit(_passes())
    import sys

    monkeypatch.setitem(sys.modules, "sklearn", None)  # break sklearn
    out = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([40.0]), np.array([40.0]))
    assert np.isfinite(out).all()  # serve path must not import sklearn


def test_nan_target_predicts_nan():
    m = PassCompletionModel().fit(_passes())
    out = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([np.nan]), np.array([40.0]))
    assert np.isnan(out[0])


def test_unfitted_predict_raises():
    with pytest.raises(PassCompletionIntegrityError):
        PassCompletionModel().predict_completion(np.array([1.0]), np.array([1.0]), np.array([2.0]), np.array([2.0]))
