"""TF-61 XSuccessModel — pickle-free save/load + fail-closed integrity (ADR-011/040/050)."""

import json

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from silly_kicks.xsuccess import XSuccessIntegrityError, XSuccessModel  # noqa: E402
from tests.xsuccess.test_model_fit_predict import _corpus  # noqa: E402


def test_roundtrip(tmp_path):
    m = XSuccessModel().fit(_corpus())
    m.save(tmp_path)
    m2 = XSuccessModel.load(tmp_path)
    a = _corpus(30, seed=2)
    assert np.allclose(m.predict_success(a), m2.predict_success(a), atol=1e-6, equal_nan=True)


def test_sha_tamper_raises(tmp_path):
    XSuccessModel().fit(_corpus()).save(tmp_path)
    (tmp_path / "model.json").write_bytes((tmp_path / "model.json").read_bytes() + b" ")
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel.load(tmp_path)


def test_feature_contract_drift_raises(tmp_path):
    XSuccessModel().fit(_corpus()).save(tmp_path)
    d = json.loads((tmp_path / "metadata.json").read_text())
    d["feature_contract"]["geometry"]["field_length"] = 100.0
    (tmp_path / "metadata.json").write_text(json.dumps(d))
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel.load(tmp_path)


def test_chirality_mismatch_raises(tmp_path):
    # Precondition (a design constraint this pins): SHA256SUMS covers model.json (the booster), NOT
    # metadata.json, so a tampered chirality fingerprint trips the CHIRALITY check, not the SHA check.
    XSuccessModel().fit(_corpus()).save(tmp_path)
    d = json.loads((tmp_path / "metadata.json").read_text())
    d["chirality"]["probe_prediction"] = [min(1.0, v + 0.5) for v in d["chirality"]["probe_prediction"]]
    (tmp_path / "metadata.json").write_text(json.dumps(d))
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel.load(tmp_path)


def test_per_type_logistic_roundtrip(tmp_path):
    m = XSuccessModel().fit(_corpus(), family="per_type_logistic")
    m.save(tmp_path)
    m2 = XSuccessModel.load(tmp_path)
    assert m2.feature_set == "per_type_logistic"
    a = _corpus(30, seed=2)
    assert np.allclose(m.predict_success(a), m2.predict_success(a), atol=1e-6, equal_nan=True)


def test_bundled_loads_and_predicts():
    # Weights bundled in Commit 2 -> bundled() must load (fail-closed SHA + feature-contract +
    # chirality) and serve finite probabilities in [0, 1]. No skip: the artifact is committed.
    from silly_kicks.xsuccess._features import _probe_actions

    m = XSuccessModel.bundled()
    p = np.asarray(m.predict_success(_probe_actions()), dtype=float)
    assert np.all(np.isfinite(p)), "bundled xSuccess produced a non-finite probability"
    assert ((p >= 0.0) & (p <= 1.0)).all(), "bundled xSuccess probability out of [0, 1]"
