"""The split fit steps of `XSuccessModel` give the bytes of the 4.140.0 fit (HFU fit-units spec 3.3, A-R1).

`OracleXSuccessModel` (`tests/xsuccess/_fit_oracle.py`) holds the old `fit`, `_fit_xgboost`, `predict_success`
and `to_dict` verbatim. For every input the new model must save the same `model.json`, `metadata.json` and
`SHA256SUMS` and predict the same values. Every input holds a `non_action` row and a row with a non-finite
feature, so the kept-row selection is compared too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as cfg

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from silly_kicks.xsuccess import XSuccessModel
from silly_kicks.xsuccess import _model as M
from silly_kicks.xsuccess._features import xsuccess_features
from tests.scripts.test_progress_lines import _xsuccess_pooled
from tests.xsuccess._fit_oracle import OracleXSuccessModel

_SUCCESS, _FAIL = cfg.result_id["success"], cfg.result_id["fail"]
_NON_ACTION = cfg.actiontype_id["non_action"]


def _actions(*, n: int = 400, n_fail: int | None = None) -> pd.DataFrame:
    """``n`` on-ball rows, plus one `non_action` row and one row with a NaN ``start_x`` (never kept).
    ``n_fail``: exactly that many failures among the ``n`` rows (``None``: the seeded labels)."""
    df = _xsuccess_pooled(n=n, seed=3).reset_index(drop=True)
    if n_fail is not None:
        df["result_id"] = _SUCCESS
        df.loc[df.index[:n_fail], "result_id"] = _FAIL
    extra = df.iloc[[0, 1]].copy()
    extra.loc[extra.index[0], "type_id"] = _NON_ACTION
    extra.loc[extra.index[1], "start_x"] = np.nan
    return pd.concat([df, extra], ignore_index=True)


#: name -> (actions, family, calibration splits expected; 0 = no calibration)
CASES = {
    "five_splits": (lambda: _actions(), "xgboost", 5),
    "four_splits": (lambda: _actions(n_fail=4), "xgboost", 4),
    "three_splits": (lambda: _actions(n_fail=3), "xgboost", 3),
    "two_splits": (lambda: _actions(n_fail=2), "xgboost", 2),
    "under_fifty_rows": (lambda: _actions(n=40), "xgboost", 0),
    "one_class_all_fail": (lambda: _actions(n_fail=400), "xgboost", 0),
    "per_type_logistic": (lambda: _actions(), "per_type_logistic", None),
}


def _saved(model: XSuccessModel, d) -> dict[str, bytes]:
    model.save(d)
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_fit_steps_equal_the_oracle(case, tmp_path):
    build, family, n_splits = CASES[case]
    actions = build()
    # non-vacuity: the two rows that the fit must drop are there
    assert (actions["type_id"] == _NON_ACTION).sum() == 1
    assert (~np.isfinite(xsuccess_features(actions)).all(axis=1)).sum() == 1

    new = XSuccessModel().fit(actions, family=family)
    old = OracleXSuccessModel().fit(actions, family=family)
    assert _saved(new, tmp_path / "new") == _saved(old, tmp_path / "old")  # the RAW bytes of every file
    assert np.array_equal(new.predict_success(actions), old.predict_success(actions), equal_nan=True)
    assert np.isnan(new.predict_success(actions)).sum() == 1  # the non-finite row serves NaN

    X, y, _tid = M._training_arrays(actions)
    assert len(y) == len(actions) - 2
    if n_splits is None:
        assert new._iso_x is None and new._booster is None  # the logistic family has neither
    else:
        assert len(M._calibration_splits(X, y, 42)) == n_splits
        assert (new._iso_x is not None) is (n_splits > 0)


def test_a_corpus_of_successes_only_raises_as_before():
    """xgboost refuses a label set of {1} (`Invalid classes inferred`): the new fit raises what the old one did."""
    actions = _actions(n_fail=0)
    with pytest.raises(Exception) as want:  # the type is asserted on the next lines
        OracleXSuccessModel().fit(actions)
    with pytest.raises(type(want.value)):
        XSuccessModel().fit(actions)
    assert isinstance(want.value, ValueError)


def test_the_out_of_fold_vector_keeps_its_dtype(monkeypatch):
    """The vector that reaches the isotonic fit has the dtype `cross_val_predict` gave the old code."""
    from sklearn.isotonic import IsotonicRegression

    seen: list = []
    real = IsotonicRegression.fit

    def spy(self, X, y, sample_weight=None):  # noqa: N803 - the sklearn parameter name
        seen.append(np.asarray(X).dtype)
        return real(self, X, y, sample_weight)

    monkeypatch.setattr(IsotonicRegression, "fit", spy)
    actions = _actions()
    OracleXSuccessModel().fit(actions)
    XSuccessModel().fit(actions)
    assert len(seen) == 2 and seen[0] == seen[1]


def test_the_oracle_test_detects_a_fault(monkeypatch, tmp_path):
    """A deliberate fault in one fit step (1e-7 on a calibration fold) must change the saved bytes."""
    actions = _actions()
    old = _saved(OracleXSuccessModel().fit(actions), tmp_path / "old")
    real = M._calibration_fold_proba
    monkeypatch.setattr(M, "_calibration_fold_proba", lambda *a, **k: real(*a, **k) + np.float32(1e-7))
    assert _saved(XSuccessModel().fit(actions), tmp_path / "new") != old
