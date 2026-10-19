"""TF-46 RunZoneModel ADR-011 serialization: round-trip, integrity, chirality, no-sklearn load."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from silly_kicks.corners._zones import IntegrityError, RunZoneModel


def _synthetic():
    rng = np.random.default_rng(0)
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),
            rng.normal([86.0, 34.0], 1.0, size=(40, 2)),  # top-of-box / second-ball edge (in-region)
        ]
    )
    return init, target


def _toy_model() -> RunZoneModel:
    init, target = _synthetic()
    return RunZoneModel().fit(init, target, init_k=2, target_k=3)


_PROBE_INIT = np.array([[90.0, 25.0], [90.0, 43.0], [95.0, 34.0]])
_PROBE_TARGET = np.array([[100.0, 28.0], [100.0, 40.0], [100.0, 34.0]])


def test_round_trip_predicts_identically(tmp_path):
    m = _toy_model()
    v0 = m.encode_runs(_PROBE_INIT, _PROBE_TARGET)
    m.save(tmp_path)
    v1 = RunZoneModel.load(tmp_path).encode_runs(_PROBE_INIT, _PROBE_TARGET)
    np.testing.assert_allclose(v0, v1)


def test_missing_sha_raises(tmp_path):
    with pytest.raises(IntegrityError):
        RunZoneModel.load(tmp_path)  # empty dir, no SHA256SUMS


def test_sha_tamper_raises(tmp_path):
    m = _toy_model()
    m.save(tmp_path)
    npz = tmp_path / "model.npz"
    npz.write_bytes(npz.read_bytes() + b"\x00")  # corrupt the payload
    with pytest.raises(IntegrityError):
        RunZoneModel.load(tmp_path)


def _rewrite_without_chirality(tmp_path):
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta.pop("chirality", None)
    (tmp_path / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    # recompute SHA256SUMS so the integrity check passes and we reach the chirality check
    import hashlib

    with open(tmp_path / "SHA256SUMS", "w", newline="\n") as f:
        for fname in ["model.npz", "metadata.json"]:
            raw = (tmp_path / fname).read_bytes()
            if fname.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")


def test_missing_chirality_raises_without_override(tmp_path):
    _toy_model().save(tmp_path)
    _rewrite_without_chirality(tmp_path)
    with pytest.raises(IntegrityError):
        RunZoneModel.load(tmp_path)


def test_missing_chirality_allowed_with_override(tmp_path):
    _toy_model().save(tmp_path)
    _rewrite_without_chirality(tmp_path)
    with pytest.warns(UserWarning):
        m = RunZoneModel.load(tmp_path, legacy_override=True)
    assert m.n_initial_zones == 2


def test_load_imports_no_sklearn(tmp_path, monkeypatch):
    m0 = _toy_model()  # fit (needs sklearn) happens here, before we block it
    expected = m0.encode_runs(_PROBE_INIT, _PROBE_TARGET)
    m0.save(tmp_path)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.mixture", None)
    m = RunZoneModel.load(tmp_path)  # must not import sklearn
    np.testing.assert_allclose(m.encode_runs(_PROBE_INIT, _PROBE_TARGET), expected)
