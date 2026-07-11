import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._retention import GkRetentionModel, RetentionModel
from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES


def test_a_stub_satisfies_the_port():
    class _Stub:
        def predict_proba(self, features):
            return np.full(len(features), 0.7)

    stub = _Stub()
    assert isinstance(stub, RetentionModel)
    assert list(stub.predict_proba(pd.DataFrame({"x": [1, 2]}))) == [0.7, 0.7]


def _fake_training():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.normal(size=n) for c in RETENTION_FEATURE_NAMES})
    y = (X["forwardness"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_fit_serve_roundtrip_pure_numpy(tmp_path):
    X, y = _fake_training()
    m = GkRetentionModel().fit(X, pd.Series(y))
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()
    m.save(tmp_path / "ret")
    reloaded = GkRetentionModel.load(tmp_path / "ret")
    assert np.allclose(reloaded.predict_proba(X), p)


def test_load_detects_tamper(tmp_path):
    X, y = _fake_training()
    GkRetentionModel().fit(X, pd.Series(y)).save(tmp_path / "ret")
    (tmp_path / "ret" / "model.json").write_text('{"version":"9"}')
    with pytest.raises(ValueError, match="integrity"):
        GkRetentionModel.load(tmp_path / "ret")


def test_skillcorner_variant_bundled_and_routed():
    # PR-S111: SkillCorner clears the gate on the broadened domain -> its own variant ships.
    from silly_kicks.xtgk._retention import _PROVIDER_VARIANT, variant_key_for_provider

    assert _PROVIDER_VARIANT.get("skillcorner") == "skillcorner"
    assert variant_key_for_provider("skillcorner") == "skillcorner"
    assert variant_key_for_provider("gradientsports") == "gs"  # others still fall back to default
    m = GkRetentionModel.from_variant("skillcorner")
    X = pd.DataFrame({c: [0.0] for c in RETENTION_FEATURE_NAMES})
    p = m.predict_proba(X)
    assert 0.0 <= float(p[0]) <= 1.0


def test_bundled_default_variant_loads_if_present():
    try:
        m = GkRetentionModel.from_variant("default")
    except FileNotFoundError:
        pytest.skip("retention weights not yet bundled (owner-run training)")
    X = pd.DataFrame({c: [0.0] for c in RETENTION_FEATURE_NAMES})
    p = m.predict_proba(X)
    assert 0.0 <= float(p[0]) <= 1.0
