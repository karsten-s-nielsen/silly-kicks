"""TF-61 XSuccessModel — fit / predict_success / calibration (unit-scale synthetic corpus)."""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as cfg

xgb = pytest.importorskip("xgboost")  # serve/fit path requires the [xgboost] extra
pytest.importorskip("sklearn")

from silly_kicks.xsuccess import XSuccessIntegrityError, XSuccessModel  # noqa: E402


def _corpus(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 105, n)
    # completion depends on START distance-to-goal only (end-blind is safe to test):
    p = 1.0 / (1.0 + np.exp(-(3.0 - 0.04 * (105.0 - x))))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(
        dict(
            type_id=cfg.actiontype_id["pass"],
            bodypart_id=cfg.bodypart_id["foot"],
            start_x=x,
            start_y=rng.uniform(0, 68, n),
            end_x=x,
            end_y=rng.uniform(0, 68, n),
            result_id=np.where(y == 1, cfg.result_id["success"], cfg.result_id["fail"]),
            time_seconds=rng.uniform(0, 3000, n),
            period_id=1,
        )
    )


def test_unfitted_refuses():
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel().predict_success(_corpus(1))


def test_fit_predict_range():
    m = XSuccessModel().fit(_corpus())
    assert m.is_fitted
    p = m.predict_success(_corpus(50, seed=1))
    assert p.shape == (50,)
    assert np.nanmin(p) >= 0.0 and np.nanmax(p) <= 1.0


def test_nan_feature_predicts_nan():
    m = XSuccessModel().fit(_corpus())
    a = _corpus(1)
    a.loc[a.index[0], "start_x"] = np.nan
    assert np.isnan(m.predict_success(a)[0])


def test_feature_set_default_is_xgboost():
    m = XSuccessModel().fit(_corpus())
    assert m.feature_set == "xgboost"


def test_per_type_logistic_family_fit_predict():
    m = XSuccessModel().fit(_corpus(), family="per_type_logistic")
    assert m.is_fitted and m.feature_set == "per_type_logistic"
    p = m.predict_success(_corpus(50, seed=1))
    assert p.shape == (50,)
    assert np.nanmin(p) >= 0.0 and np.nanmax(p) <= 1.0


def test_per_type_logistic_nan_feature_predicts_nan():
    m = XSuccessModel().fit(_corpus(), family="per_type_logistic")
    a = _corpus(1)
    a.loc[a.index[0], "start_x"] = np.nan
    assert np.isnan(m.predict_success(a)[0])


def test_unknown_family_raises():
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel().fit(_corpus(20), family="bogus")
