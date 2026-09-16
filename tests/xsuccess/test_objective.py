"""TF-61 XSuccessObjective — ruthless CachedObjective cache-equivalence (evaluate == evaluate_patch)."""

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
Candidate = pytest.importorskip("ruthless.result").Candidate

import silly_kicks.spadl.config as cfg  # noqa: E402
from silly_kicks.xsuccess._features import xsuccess_features  # noqa: E402
from silly_kicks.xsuccess._objective import XSuccessObjective  # noqa: E402
from tests.xsuccess.test_model_fit_predict import _corpus  # noqa: E402


def _fold():
    a = _corpus(300)
    X = xsuccess_features(a)
    y = (a["result_id"].to_numpy() == cfg.result_id["success"]).astype(int)
    g = np.arange(len(a)) % 6  # 6 pseudo-matches
    return {"synthetic": [(X, y, g)]}


def test_cache_equivalence():
    obj = XSuccessObjective(fold=_fold())
    c = Candidate(
        id="t1",
        params=dict(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            subsample=1.0,
            colsample_bytree=1.0,
        ),
    )
    inv = obj.prepare()
    assert abs(obj.evaluate_patch(inv, c)["logloss"] - obj.evaluate(c)["logloss"]) < 1e-9
