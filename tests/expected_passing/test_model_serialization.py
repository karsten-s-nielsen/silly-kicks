import json

import numpy as np
import pytest

from silly_kicks.expected_passing import PassCompletionIntegrityError, PassCompletionModel
from tests.expected_passing.test_model_fit_predict import _passes  # reuse the fixture


def test_round_trip_predicts_identically(tmp_path):
    m = PassCompletionModel().fit(_passes())
    q = np.array([30.0])
    m.save(tmp_path)
    m2 = PassCompletionModel.load(tmp_path)
    np.testing.assert_allclose(
        m.predict_completion(q, q, q + 10, q),
        m2.predict_completion(q, q, q + 10, q),
        rtol=0,
        atol=1e-12,
    )


def _repoison(tmp_path, mutate):
    """Save a fitted model, mutate ``model.json`` via ``mutate(d)``, then RECOMPUTE ``SHA256SUMS``.

    Re-writing the digest makes the SHA guard PASS, so ``load`` reaches the downstream chirality /
    feature-contract guards. Without this, a model.json edit trips the SHA check first and the guard
    named in the test never fires (the "band tested from one side" defect CLAUDE.md forbids).
    """
    m = PassCompletionModel().fit(_passes())
    m.save(tmp_path)
    d = json.loads((tmp_path / "model.json").read_text())
    mutate(d)
    (tmp_path / "model.json").write_text(json.dumps(d))
    (tmp_path / "SHA256SUMS").write_text(f"{PassCompletionModel._sha(tmp_path)}  model.json\n", encoding="utf-8")


def test_sha256sums_written_and_checked(tmp_path):
    PassCompletionModel().fit(_passes()).save(tmp_path)
    assert (tmp_path / "SHA256SUMS").exists()
    (tmp_path / "model.json").write_text(json.dumps({"tampered": True}))
    with pytest.raises(PassCompletionIntegrityError):
        PassCompletionModel.load(tmp_path)


def test_chirality_field_corruption_without_sha_rewrite_trips_sha_guard(tmp_path):
    # The other side of the SHA guard: a model.json edit that does NOT re-write SHA256SUMS is caught
    # at the digest step (defense in depth), before any downstream guard runs.
    m = PassCompletionModel().fit(_passes())
    m.save(tmp_path)
    d = json.loads((tmp_path / "model.json").read_text())
    d["chirality"]["probe_prediction"][0] += 0.5
    (tmp_path / "model.json").write_text(json.dumps(d))  # SHA256SUMS left stale on purpose
    with pytest.raises(PassCompletionIntegrityError, match="SHA"):
        PassCompletionModel.load(tmp_path)


def _corrupt_chirality(d):
    d["chirality"]["probe_prediction"][0] += 0.5


def test_chirality_mismatch_raises_with_valid_sha(tmp_path):
    # VALID SHA -> the chirality guard is what fires (not the SHA guard).
    _repoison(tmp_path, _corrupt_chirality)
    with pytest.raises(PassCompletionIntegrityError, match="chirality"):
        PassCompletionModel.load(tmp_path)


def test_chirality_mismatch_legacy_override_warns_and_loads(tmp_path):
    # legacy_override downgrades the chirality raise to a warning AND still returns a served model.
    _repoison(tmp_path, _corrupt_chirality)
    with pytest.warns(UserWarning, match="chirality"):
        m2 = PassCompletionModel.load(tmp_path, legacy_override=True)
    assert m2.is_fitted


def test_feature_name_contract_mismatch_raises_with_valid_sha(tmp_path):
    # VALID SHA, contract intact except a drifted feature name -> the feature-contract guard fires.
    def _drift_name(d):
        d["feature_contract"]["feature_names"][0] = "not_a_feature"

    _repoison(tmp_path, _drift_name)
    with pytest.raises(PassCompletionIntegrityError, match="feature-contract"):
        PassCompletionModel.load(tmp_path)


def test_geometry_constant_contract_mismatch_raises_with_valid_sha(tmp_path):
    # VALID SHA, a declared geometry constant drifted -> the feature-contract guard fires.
    def _drift_constant(d):
        d["feature_contract"]["geometry"]["field_length"] = 999.0

    _repoison(tmp_path, _drift_constant)
    with pytest.raises(PassCompletionIntegrityError, match="feature-contract"):
        PassCompletionModel.load(tmp_path)
