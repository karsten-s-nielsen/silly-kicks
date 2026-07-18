import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xshot_occurrence as xs
from silly_kicks.tracking._chirality import chirality_fingerprint, verify_chirality
from silly_kicks.tracking._xshot_occurrence import IntegrityError


def test_chirality_fingerprint_raises_on_nonfinite():
    with pytest.raises(ValueError, match="non-finite"):
        chirality_fingerprint(lambda frame: np.array([np.nan, 0.5]))


def test_verify_chirality_passes_identical():
    fp = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5, 0.25]}
    verify_chirality(fp, dict(fp), legacy_override=False, model_name="xS")


def test_verify_chirality_tolerates_float_noise():
    stored = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5000000, 0.2500000]}
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5000004, 0.2499997]}
    verify_chirality(recomputed, stored, legacy_override=False, model_name="xS")


def test_verify_chirality_raises_on_mirror_scale_mismatch():
    stored = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.80, 0.20]}
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.20, 0.80]}
    with pytest.raises(IntegrityError, match="chirality"):
        verify_chirality(recomputed, stored, legacy_override=False, model_name="xS")


def test_verify_chirality_raises_on_missing_stored():
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5]}
    with pytest.raises(IntegrityError, match="missing"):
        verify_chirality(recomputed, None, legacy_override=False, model_name="xS")


def test_verify_chirality_legacy_override_allows_missing():
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5]}
    with pytest.warns(UserWarning, match="legacy"):
        verify_chirality(recomputed, None, legacy_override=True, model_name="xS")


def test_verify_chirality_raises_on_frame_sha_change():
    stored = {"version": "chirality-probe-1", "frame_sha256": "OLDSHA", "outputs": [0.5]}
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "NEWSHA", "outputs": [0.5]}
    with pytest.raises(IntegrityError, match="probe frame"):
        verify_chirality(recomputed, stored, legacy_override=False, model_name="xS")


# --- Task 2: xS load() chirality enforcement (TF-19 PR-2) ---


def _toy_xs_model():
    """Fit a tiny xS model from the canonical toy fixture (mirrors test_xshot_occurrence._toy_xy)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = (X["r"] + rng.normal(scale=0.5, size=400) < 0).astype(int)
    return xs.XShotOccurrenceModel().fit(X, pd.Series(y))


def _resum(d):
    """Rewrite SHA256SUMS (LF-normalized) so the SHA guard passes and the chirality guard fires."""
    with open(d / "SHA256SUMS", "w", newline="\n") as f:
        for fn in ["model.json", "metadata.json"]:
            raw = (d / fn).read_bytes()
            if fn.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            f.write(f"{hashlib.sha256(raw).hexdigest()}  {fn}\n")


def test_xs_load_raises_on_chirality_output_mismatch(tmp_path):
    d = tmp_path / "xs_v1"
    _toy_xs_model().save(d)
    meta = json.loads((d / "metadata.json").read_text())
    # Gross shift beyond the cross-platform tolerance -> the y-mirror-mis-serving signature.
    meta["chirality"]["outputs"] = [v + 0.5 for v in meta["chirality"]["outputs"]]
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _resum(d)
    with pytest.raises(IntegrityError, match="chirality"):
        xs.XShotOccurrenceModel.load(d)


def test_xs_load_raises_on_missing_chirality_and_legacy_override_allows(tmp_path):
    d = tmp_path / "xs_v1"
    _toy_xs_model().save(d)
    meta = json.loads((d / "metadata.json").read_text())
    del meta["chirality"]
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _resum(d)
    with pytest.raises(IntegrityError, match=r"missing|chirality"):
        xs.XShotOccurrenceModel.load(d)
    # legacy_override lets a fingerprint-less (pre-PR-2) artifact through, with a loud warning.
    with pytest.warns(UserWarning, match="legacy"):
        model = xs.XShotOccurrenceModel.load(d, legacy_override=True)
    assert model._booster is not None


# --- Task 3: xCross load() chirality enforcement (TF-19 PR-2) ---


def _resum_from_sums(d):
    """Re-checksum whatever files the existing SHA256SUMS already lists (LF-normalize .json).

    Generalizes ``_resum`` for artifacts whose SHA256SUMS covers a fileset other than
    ``{model,metadata}.json`` --- ghost-GK lists ``rfcde_weights.npz`` + ``metadata.json``.
    """
    sums = d / "SHA256SUMS"
    fnames = [line.split("  ", 1)[1] for line in sums.read_text().splitlines() if line.strip()]
    with open(sums, "w", newline="\n") as f:
        for fn in fnames:
            raw = (d / fn).read_bytes()
            if fn.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            f.write(f"{hashlib.sha256(raw).hexdigest()}  {fn}\n")


def _toy_xcross_model():
    """Fit a tiny xCross model from the canonical toy fixture (mirrors test_xcross_attempt._fit_tiny_model)."""
    from silly_kicks.tracking import _xcross_attempt as xc

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (X["gk_r"] + rng.normal(scale=0.5, size=200) > 0).astype(int).to_numpy()
    return xc.XCrossAttemptModel().fit(X, pd.Series(y))


def test_xcross_load_raises_on_chirality_output_mismatch(tmp_path):
    from silly_kicks.tracking import _xcross_attempt as xc

    d = tmp_path / "xcross_v1"
    _toy_xcross_model().save(d)
    meta = json.loads((d / "metadata.json").read_text())
    # Gross shift beyond the cross-platform tolerance -> the y-mirror-mis-serving signature.
    meta["chirality"]["outputs"] = [v + 0.5 for v in meta["chirality"]["outputs"]]
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _resum_from_sums(d)
    with pytest.raises(IntegrityError, match="chirality"):
        xc.XCrossAttemptModel.load(d)


def test_xcross_load_raises_on_missing_chirality_and_legacy_override_allows(tmp_path):
    from silly_kicks.tracking import _xcross_attempt as xc

    d = tmp_path / "xcross_v1"
    _toy_xcross_model().save(d)
    meta = json.loads((d / "metadata.json").read_text())
    del meta["chirality"]
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _resum_from_sums(d)
    with pytest.raises(IntegrityError, match=r"missing|chirality"):
        xc.XCrossAttemptModel.load(d)
    # legacy_override lets a fingerprint-less (pre-PR-2) artifact through, with a loud warning.
    with pytest.warns(UserWarning, match="legacy"):
        model = xc.XCrossAttemptModel.load(d, legacy_override=True)
    assert model._booster is not None


# --- Task 4: ghost-GK load() chirality enforcement (TF-19 PR-2) ---


def _toy_ghost_model():
    """Fit a tiny GhostGkModel from the canonical toy fixture (mirrors test_ghost_gk._fitted_model)."""
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 100).astype(float)
    X["team_in_possession"] = rng.integers(0, 2, 100).astype(float)
    X["ball_in_own_half"] = rng.integers(0, 2, 100).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 100), "gk_y": rng.uniform(25, 45, 100)})
    model = GhostGkModel(n_estimators=10)
    model.fit(X, labels)
    return model


def test_ghost_load_raises_on_chirality_output_mismatch(tmp_path):
    # ghost-GK's load() raises its OWN module-local IntegrityError for every artifact-integrity
    # failure (SHA-256, pitch dims); the chirality failure must share that taxonomy so a consumer
    # catching _ghost_gk.IntegrityError catches this too (NOT the foreign _xshot_occurrence one).
    from silly_kicks.tracking._ghost_gk import GhostGkModel
    from silly_kicks.tracking._ghost_gk import IntegrityError as GhostIntegrityError

    assert GhostIntegrityError is not IntegrityError  # genuinely distinct types -> the raise below is meaningful

    d = tmp_path / "ghost_v1"
    _toy_ghost_model().save(d)
    meta = json.loads((d / "metadata.json").read_text())
    # Gross shift beyond the cross-platform tolerance -> the y-mirror-mis-serving signature.
    meta["chirality"]["outputs"] = [v + 0.5 for v in meta["chirality"]["outputs"]]
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _resum_from_sums(d)
    with pytest.raises(GhostIntegrityError, match="chirality"):
        GhostGkModel.load(d)


def test_ghost_load_raises_on_missing_chirality_and_legacy_override_allows(tmp_path):
    from silly_kicks.tracking._ghost_gk import GhostGkModel
    from silly_kicks.tracking._ghost_gk import IntegrityError as GhostIntegrityError

    d = tmp_path / "ghost_v1"
    _toy_ghost_model().save(d)
    meta = json.loads((d / "metadata.json").read_text())
    del meta["chirality"]
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _resum_from_sums(d)
    with pytest.raises(GhostIntegrityError, match=r"missing|chirality"):
        GhostGkModel.load(d)
    # legacy_override lets a fingerprint-less (pre-PR-2) artifact through, with a loud warning.
    with pytest.warns(UserWarning, match="legacy"):
        model = GhostGkModel.load(d, legacy_override=True)
    assert model._tree_nodes is not None


def test_load_xgb_booster_base_score_safe_normalizes_bracketed(tmp_path):
    """The xgboost-3.x bracketed base_score '[X]' is normalized so xgboost-2.x reads it correctly
    (TF-19 PR-2 defensive guard); a scalar base_score passes through unchanged."""
    import json

    import xgboost as xgb

    from silly_kicks.tracking._xshot_occurrence import load_xgb_booster_base_score_safe

    Xt = np.array([[0.0], [1.0], [2.0], [3.0]])
    yt = np.array([0, 0, 1, 1])
    bst = xgb.train(
        {"objective": "binary:logistic", "base_score": 0.3, "max_depth": 1},
        xgb.DMatrix(Xt, label=yt),
        num_boost_round=3,
    )
    mjp = tmp_path / "model.json"
    bst.save_model(str(mjp))
    d = xgb.DMatrix(Xt)
    correct = load_xgb_booster_base_score_safe(mjp).predict(d)

    # simulate the xgboost-3.x serialization: base_score as a bracketed STRING
    mj = json.loads(mjp.read_text())
    bs = mj["learner"]["learner_model_param"]["base_score"]
    mj["learner"]["learner_model_param"]["base_score"] = f"[{bs}]"
    mjp.write_text(json.dumps(mj))

    guarded = load_xgb_booster_base_score_safe(mjp).predict(d)
    np.testing.assert_allclose(guarded, correct, atol=1e-9)  # guard normalized -> identical

    raw = xgb.Booster()
    raw.load_model(str(mjp))  # raw 2.x load drops the bracketed base_score to the 0.5 default
    assert not np.allclose(raw.predict(d), correct, atol=1e-6)


# --- Task 5: sc_extended variant routes to the Hub (TF-19 PR-2) ---


def test_xs_sc_extended_routes_to_hub(monkeypatch):
    from silly_kicks.tracking import _xshot_occurrence as X

    seen = {}

    def fake(cls, repo_id=X._HF_REPO_ID):
        seen["repo"] = repo_id
        raise RuntimeError("HUB_MARKER")

    monkeypatch.setattr(X.XShotOccurrenceModel, "from_hub", classmethod(fake))
    X._VARIANT_CACHE.pop("sc_extended", None)
    with pytest.raises(RuntimeError, match="HUB_MARKER"):
        X.XShotOccurrenceModel.from_variant("sc_extended")
    assert seen["repo"] == "silly-kicks/xshot-occurrence-v1"


def test_xcross_sc_extended_routes_to_hub(monkeypatch):
    from silly_kicks.tracking import _xcross_attempt as XC

    seen = {}

    def fake(cls, repo_id=XC._HF_REPO_ID):
        seen["repo"] = repo_id
        raise RuntimeError("HUB_MARKER")

    monkeypatch.setattr(XC.XCrossAttemptModel, "from_hub", classmethod(fake))
    XC._VARIANT_CACHE.pop("sc_extended", None)
    with pytest.raises(RuntimeError, match="HUB_MARKER"):
        XC.XCrossAttemptModel.from_variant("sc_extended")
    assert seen["repo"] == "silly-kicks/xcross-attempt-v1"
