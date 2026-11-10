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


_PARAMS = dict(
    n_estimators=40, max_depth=3, learning_rate=0.1, min_child_weight=1,
    reg_lambda=1.0, reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0,
)  # fmt: skip


def test_fold_map_result_equals_the_serial_result():
    """HFU spec 15.7: the per-fold path (``fold_map``) gives the serial result bit for bit."""
    from silly_kicks.xsuccess import _objective as O

    obj = XSuccessObjective(fold=_fold())
    inv = obj.prepare()
    c = Candidate(id="t1", params=_PARAMS)
    splits = O._cv_splits(inv.y, inv.groups)

    def fold_map(params):  # what the pipeline's InnerPool does, here in-process and out of order
        out = {i: O._fold_score(inv.X, inv.y, tr, te, params) for i, (tr, te) in reversed(list(enumerate(splits)))}
        return [out[i] for i in range(len(splits))]

    assert obj.evaluate_patch(inv, c, fold_map=fold_map) == obj.evaluate_patch(inv, c)


def test_the_trainer_fold_map_returns_folds_in_fold_order(tmp_path):
    """HFU spec 15.7: ``train_xsuccess._fold_map`` (job-local files, ``_fold_task``) gives the serial fold scores
    in fold order when the pool finishes them out of order."""
    from scripts.train_xsuccess import _fold_map
    from silly_kicks.xsuccess import _objective as O

    class _ReversedPool:  # an InnerPool whose units complete in reverse key order
        def map_keyed(self, fn, items):
            return {k: fn(*a) for k, a in reversed(list(items.items()))}

    inv = XSuccessObjective(fold=_fold()).prepare()
    fold_map = _fold_map(_ReversedPool(), inv, tmp_path)
    serial = [O._fold_score(inv.X, inv.y, tr, te, _PARAMS) for tr, te in O._cv_splits(inv.y, inv.groups)]
    assert fold_map(_PARAMS) == serial


def test_fold_map_keeps_a_skipped_fold(monkeypatch):
    """A skipped fold (one training class) is skipped the same way on both paths, and it moves the result."""
    from silly_kicks.xsuccess import _objective as O

    obj = XSuccessObjective(fold=_fold())
    inv = obj.prepare()
    c = Candidate(id="t1", params=_PARAMS)
    splits = O._cv_splits(inv.y, inv.groups)
    unskipped = obj.evaluate_patch(inv, c)
    score = O._fold_score
    monkeypatch.setattr(
        O, "_fold_score", lambda x, y, tr, te, p: None if np.array_equal(tr, splits[0][0]) else score(x, y, tr, te, p)
    )

    def fold_map(params):
        return [O._fold_score(inv.X, inv.y, tr, te, params) for tr, te in splits]

    skipped = obj.evaluate_patch(inv, c)
    assert obj.evaluate_patch(inv, c, fold_map=fold_map) == skipped != unskipped


def test_combine_keeps_the_skip_rules():
    """A skipped fold (one training class) adds nothing; a fold with one test class adds no PR-AUC; no fold
    scoring gives ``inf`` (the ``_cv_logloss`` rules)."""
    import math

    from silly_kicks.xsuccess import _objective as O

    assert O._combine([None, (1.0, None, 0.2), (3.0, 0.5, 0.4)]) == (2.0, 0.5, pytest.approx(0.3))
    ll, pr, br = O._combine([None, None])
    assert ll == math.inf and math.isnan(pr) and math.isnan(br)
    assert O._fold_score(np.zeros((4, 2)), np.array([1, 1, 1, 1]), np.arange(4), np.arange(2), _PARAMS) is None
