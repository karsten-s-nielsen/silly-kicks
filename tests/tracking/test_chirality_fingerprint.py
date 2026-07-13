"""Spec §3.4/M2: chirality fingerprints are BEHAVIORAL (model outputs on a canonical
asymmetric probe frame), never self-declared strings; emitted at save() time (PR-1).
load()-enforcement is PR-2 -- here we pin emission + round-trip determinism."""

import json
import sys

import numpy as np
import pandas as pd

from silly_kicks.tracking._chirality import canonical_probe_frame, chirality_fingerprint
from tests.tracking._probe_fixtures import planted_model


def test_canonical_probe_frame_is_y_asymmetric_and_deterministic():
    f1, f2 = canonical_probe_frame(), canonical_probe_frame()
    import pandas.testing as pdt

    pdt.assert_frame_equal(f1, f2)
    mirrored = f1.copy()
    mirrored["y"] = 68.0 - mirrored["y"]
    assert not np.allclose(np.sort(f1["y"]), np.sort(mirrored["y"]))  # genuinely asymmetric


def _predict_on(model):
    def predict(frame):
        from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

        feats = extract_xshot_features(frame, gk_team_id="B", goal_x=105.0)
        return model.predict_proba(feats)

    return predict


def test_fingerprint_changes_under_a_y_mirror_for_a_chiral_model():
    """MUST use the 'chiral' planted kind: 'mixed'/'gk_blind' consume only MAGNITUDES
    (GK_r, Def/OffDist), which are invariant under y->68-y (GOAL_Y=34 is ON the mirror
    axis) -- a fingerprint test built on them passes while proving nothing (review B3)."""
    predict = _predict_on(planted_model("chiral"))
    fp = chirality_fingerprint(predict)
    mirrored_fp = chirality_fingerprint(lambda f: predict(f.assign(y=68.0 - f["y"])))
    assert fp["frame_sha256"] == mirrored_fp["frame_sha256"]  # same canonical frame
    assert not np.allclose(fp["outputs"], mirrored_fp["outputs"])  # chirality DETECTABLE


def test_fingerprint_is_blind_to_a_mirror_for_a_magnitude_only_model():
    """Guard-the-guard: if this ever FAILS, the fixture drifted (a signed term leaked
    into 'mixed') and the chirality test above has become vacuous."""
    predict = _predict_on(planted_model("mixed"))
    fp = chirality_fingerprint(predict)
    mirrored_fp = chirality_fingerprint(lambda f: predict(f.assign(y=68.0 - f["y"])))
    assert np.allclose(fp["outputs"], mirrored_fp["outputs"])


# --- Round-trip: emission survives save()/load() and reproduces exactly (PR-1). ---


def test_chirality_block_roundtrips_for_xshot(tmp_path):
    from silly_kicks.tracking import _xshot_occurrence as xs
    from silly_kicks.tracking._xshot_occurrence import _chirality_block

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(200, len(xs.XSHOT_FEATURE_NAMES_FAITHFUL))),
        columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL,
    )
    y = (X["r"] + rng.normal(scale=0.5, size=200) < 0).astype(int)
    model = xs.XShotOccurrenceModel().fit(X, pd.Series(y))
    model.save(tmp_path / "xs")

    meta = json.loads((tmp_path / "xs" / "metadata.json").read_text())
    assert "chirality" in meta
    assert meta["chirality"]["version"] == "chirality-probe-1"

    loaded = xs.XShotOccurrenceModel.load(tmp_path / "xs")
    recomputed = _chirality_block(loaded)
    assert recomputed["frame_sha256"] == meta["chirality"]["frame_sha256"]
    assert recomputed["outputs"] == meta["chirality"]["outputs"]  # exact (round-to-10dp)


def test_chirality_block_roundtrips_for_xcross(tmp_path):
    from silly_kicks.tracking import _xcross_attempt as xc
    from silly_kicks.tracking._xcross_attempt import _chirality_block

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (X["gk_r"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    model.save(tmp_path / "xc")

    meta = json.loads((tmp_path / "xc" / "metadata.json").read_text())
    assert "chirality" in meta
    assert meta["chirality"]["version"] == "chirality-probe-1"

    loaded = xc.XCrossAttemptModel.load(tmp_path / "xc")
    recomputed = _chirality_block(loaded)
    assert recomputed["frame_sha256"] == meta["chirality"]["frame_sha256"]
    assert recomputed["outputs"] == meta["chirality"]["outputs"]  # exact (round-to-10dp)


def test_chirality_block_roundtrips_for_ghost_gk(tmp_path):
    from silly_kicks.tracking._ghost_gk import (
        GHOST_GK_FEATURE_NAMES,
        GhostGkModel,
        _chirality_block,
    )

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 100).astype(float)
    X["team_in_possession"] = rng.integers(0, 2, 100).astype(float)
    X["ball_in_own_half"] = rng.integers(0, 2, 100).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 100), "gk_y": rng.uniform(25, 45, 100)})
    model = GhostGkModel(n_estimators=10)
    model.fit(X, labels)
    model.save(tmp_path / "gg")

    meta = json.loads((tmp_path / "gg" / "metadata.json").read_text())
    assert "chirality" in meta
    assert meta["chirality"]["version"] == "chirality-probe-1"
    assert len(meta["chirality"]["outputs"]) == 2  # served (x, y)

    loaded = GhostGkModel.load(tmp_path / "gg")
    recomputed = _chirality_block(loaded)
    assert recomputed["frame_sha256"] == meta["chirality"]["frame_sha256"]
    assert recomputed["outputs"] == meta["chirality"]["outputs"]  # exact (round-to-10dp)


# --- Step 4: probe-sample provenance schema (trainer code already landed in Task 6b). ---


def test_write_probe_sample_persists_match_provenance(tmp_path):
    """The TF-19 probe cohort writer records ``probe_matches`` provenance (M5) into
    meta.json. Unit-testable without Databricks (the write is extracted for exactly this)."""
    sys.path.insert(0, "scripts")
    from train_xcross_attempt import _write_probe_sample

    cohort = {
        "frames": [pd.DataFrame({"game_id": ["g"], "x": [1.0]})],
        "actions": [pd.DataFrame({"game_id": ["g"], "action_id": [0]})],
        "home": "A",
        "matches": [["skillcorner", "m1"], ["idsse", "m2"]],
        "match_groups": {"m1": ["g"], "m2": ["g2"]},
    }
    ps = tmp_path / "probe"
    _write_probe_sample(ps, cohort, ["skillcorner", "idsse"])

    meta = json.loads((ps / "meta.json").read_text())
    assert "probe_matches" in meta
    assert meta["probe_matches"] == cohort["matches"]
    assert meta["home_team_id"] == "A"
    assert meta["probe_providers"] == ["skillcorner", "idsse"]
