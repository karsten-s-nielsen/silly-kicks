import hashlib
import json

import numpy as np
import pytest

from silly_kicks.win_probability import (
    WinProbabilityIntegrityError,
    WinProbabilityModel,
    WinProbabilityParams,
)


def _fitted(tmp_path):
    m = WinProbabilityModel(params=WinProbabilityParams.default())
    m._beta = np.array([0.3, -0.005, 0.2, 0.1, 0.05])
    m._intercept = -3.0
    m._isotonic = None
    m._fitted = True
    return m


def test_save_load_roundtrip_identical_serve(tmp_path):
    m = _fitted(tmp_path)
    m.save(tmp_path)
    loaded = WinProbabilityModel.load(tmp_path)
    a = m.predict_outcome(score_diff=1, minutes_remaining=30.0, base_strength=0.2, home=True, man_advantage=0)
    b = loaded.predict_outcome(score_diff=1, minutes_remaining=30.0, base_strength=0.2, home=True, man_advantage=0)
    assert np.allclose(a, b, atol=1e-12)


def test_load_raises_on_sha_tamper(tmp_path):
    m = _fitted(tmp_path)
    m.save(tmp_path)
    # tamper model.json without re-signing SHA256SUMS
    p = tmp_path / "model.json"
    data = json.loads(p.read_text())
    data["intercept"] = -2.0
    p.write_text(json.dumps(data, sort_keys=True, indent=2))
    with pytest.raises(WinProbabilityIntegrityError):
        WinProbabilityModel.load(tmp_path)


def test_load_raises_on_feature_contract_mismatch(tmp_path):
    m = _fitted(tmp_path)
    m.save(tmp_path)
    # corrupt beta but keep the stored probe_hazards; RE-SIGN the SHA so only the contract prong fires.
    p = tmp_path / "model.json"
    data = json.loads(p.read_text())
    data["beta"] = [9.9, 9.9, 9.9, 9.9, 9.9]  # changes recomputed probe -> mismatch stored probe
    new_bytes = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
    p.write_bytes(new_bytes)
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(new_bytes).hexdigest()}  model.json\n", encoding="utf-8")
    with pytest.raises(WinProbabilityIntegrityError):
        WinProbabilityModel.load(tmp_path)
