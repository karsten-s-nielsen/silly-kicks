"""Tests for the TF-60 PR5 ghost-outfield model (rest-defense rearguard positioning).

Mirrors the ghost-GK test surface, adapted for the multi-agent (per-slot) outfield model.
Spec: docs/superpowers/specs/2026-09-03-tf60-pr5-ghost-outfield-model-design.md
Plan: docs/superpowers/plans/2026-09-04-tf60-pr5-ghost-outfield-model.md
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._ghost_outfield import (
    _GHOST_OUTFIELD_VELOCITY_FEATURES,
    _WEIGHTS_ROOT,
    GHOST_OUTFIELD_FEATURE_NAMES,
    GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY,
    GHOST_OUTFIELD_SOURCE_VALUES,
    GhostOutfieldFeatureSet,
    GhostOutfieldModel,
    IntegrityError,
    _canonical_frame_sha,
    _extract_all_ghost_outfield_features,
    _outfield_probe_actions,
    _outfield_probe_frame,
    _resolve_outfield_model_for_frames,
    ghost_rearguard_coherence,
    serve_ghost_outfield_positions,
)

# --------------------------------------------------------------------------- #
# Toy fixtures
# --------------------------------------------------------------------------- #
# One frame: team 1 (in possession, attacks right, defends x=0) + team 2 (defends x=105) + ball.
# Team 1's deepest-4 are WELL SEPARATED in both x (>=2 m apart) and y (>=13 m apart) so a small
# perturbation cannot change the depth or lateral rank. Velocities are present (faithful).


def _player(pid, team, x, y, *, gk=False, ball=False, vx=0.0, vy=0.0):
    return {
        "game_id": "G1",
        "period_id": 1,
        "frame_id": 1000,
        "team_id": (pd.NA if ball else team),
        "player_id": (pd.NA if ball else pid),
        "is_ball": ball,
        "is_goalkeeper": gk,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "time_seconds": 100.0,
        "team_in_possession": 1,  # team 1 has the ball
    }


def _toy_two_team_frame() -> pd.DataFrame:
    rows = [
        # Team 1 (A, in possession): GK near x=0, deepest-4 well separated, 6 more upfield.
        _player(101, 1, 3.0, 34.0, gk=True),
        _player(102, 1, 20.0, 10.0),  # deepest-4: slot 1 (lowest y) after lateral sort
        _player(103, 1, 22.0, 27.0),  # slot 2
        _player(104, 1, 24.0, 44.0),  # slot 3
        _player(105, 1, 26.0, 60.0),  # slot 4
        _player(106, 1, 40.0, 15.0),
        _player(107, 1, 45.0, 50.0),
        _player(108, 1, 52.0, 30.0),
        _player(109, 1, 58.0, 40.0),
        _player(110, 1, 63.0, 20.0),
        _player(111, 1, 68.0, 48.0),
        # Team 2 (B): GK near x=105, some players deep in A's half as counter-threats.
        _player(201, 2, 102.0, 34.0, gk=True),
        _player(202, 2, 15.0, 30.0, vx=1.5),
        _player(203, 2, 25.0, 40.0, vx=1.0),
        _player(204, 2, 50.0, 20.0),
        _player(205, 2, 55.0, 50.0),
        _player(206, 2, 70.0, 34.0),
        _player(207, 2, 75.0, 15.0),
        _player(208, 2, 80.0, 55.0),
        _player(209, 2, 85.0, 25.0),
        _player(210, 2, 90.0, 45.0),
        _player(211, 2, 95.0, 34.0),
        # Ball (team 1 attacking, ~x=60)
        _player(pd.NA, pd.NA, 60.0, 34.0, ball=True, vx=2.0, vy=0.0),
    ]
    return pd.DataFrame(rows)


def _toy_actions() -> pd.DataFrame:
    """Minimal actions: one pass, no shots/set-pieces -> score_diff 0, phase 0 (open play)."""
    return pd.DataFrame(
        [
            {
                "game_id": "G1",
                "period_id": 1,
                "time_seconds": 50.0,
                "team_id": 1,
                "player_id": 108,
                "type_name": "pass",
                "result_name": "success",
                "start_x": 40.0,
                "start_y": 30.0,
            }
        ]
    )


def _extract(frame, *, feature_set: GhostOutfieldFeatureSet = "faithful", n_rearguard=4):
    return _extract_all_ghost_outfield_features(
        frame,
        _toy_actions(),
        home_team_id=1,
        feature_set=feature_set,
        n_rearguard=n_rearguard,
    )


def _toy_training_frames(n_frames: int = 60) -> pd.DataFrame:
    """Many 2-team frames with deterministic jitter -> enough per-slot rows to fit a small HGBR."""
    base = _toy_two_team_frame()
    non_ball = ~base["is_ball"].astype(bool)
    out = []
    for k in range(n_frames):
        f = base.copy()
        f["frame_id"] = 1000 + k
        f["time_seconds"] = 100.0 + 0.5 * k
        dx = 4.0 * np.sin(0.3 * k)
        dy = 3.0 * np.cos(0.4 * k)
        f.loc[non_ball, "x"] = (f.loc[non_ball, "x"] + dx).clip(1.0, 104.0)
        f.loc[non_ball, "y"] = (f.loc[non_ball, "y"] + dy).clip(1.0, 67.0)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def _fit_toy(*, feature_set: GhostOutfieldFeatureSet = "faithful", n_estimators=30, max_depth=4, n_frames=60):
    frames = _toy_training_frames(n_frames)
    model = GhostOutfieldModel(n_estimators=n_estimators, max_depth=max_depth, feature_set=feature_set).fit(
        frames, _toy_actions(), home_team_id=1
    )
    return model, frames


def _restamp_sha256sums(d):
    lines = []
    for fname in ["model.npz", "metadata.json"]:
        raw = (d / fname).read_bytes()
        if fname.endswith(".json"):
            raw = raw.replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {fname}")
    with open(d / "SHA256SUMS", "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Task 2 --- fit / predict_mean / save / load
# --------------------------------------------------------------------------- #


def test_predict_mean_matches_sklearn_within_1e_6():
    model, frames = _fit_toy()
    feats = _extract(frames)
    pred = model.predict_mean(feats)
    X = feats[model._feature_names()].to_numpy(dtype=np.float64)
    assert model._sk_reg_x is not None and model._sk_reg_y is not None
    ref = np.column_stack([model._sk_reg_x.predict(X), model._sk_reg_y.predict(X)])
    assert np.max(np.abs(pred - ref)) <= 1e-6


def test_save_load_roundtrip_field_level(tmp_path):
    model, frames = _fit_toy()
    model.save(tmp_path)
    loaded = GhostOutfieldModel.load(tmp_path)
    feats = _extract(frames)
    np.testing.assert_array_equal(model.predict_mean(feats), loaded.predict_mean(feats))
    assert loaded.feature_set == "faithful"
    assert loaded._feature_names() == model._feature_names()


def test_position_only_fit_save_load(tmp_path):
    model, frames = _fit_toy(feature_set="position_only", n_estimators=20, max_depth=3)
    assert model._feature_names() == GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY
    feats = _extract(frames, feature_set="position_only")
    assert model.predict_mean(feats).shape == (len(feats), 2)
    model.save(tmp_path)
    GhostOutfieldModel.load(tmp_path)  # guards pass on the position_only artifact


# --------------------------------------------------------------------------- #
# Task 3 --- load-guards fire on tamper (non-vacuity)
# --------------------------------------------------------------------------- #


def test_load_raises_on_tampered_sha256sums(tmp_path):
    model, _ = _fit_toy(n_estimators=20, max_depth=3)
    model.save(tmp_path)
    sums = (tmp_path / "SHA256SUMS").read_text().splitlines()
    with open(tmp_path / "SHA256SUMS", "w", newline="\n") as fh:
        fh.write("\n".join(["0" * 64 + "  model.npz", *sums[1:]]) + "\n")
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(tmp_path)


def test_load_raises_on_perturbed_weight_via_chirality(tmp_path):
    """Perturb a weight AND re-stamp the SHA so the chirality guard (not the SHA check) fires."""
    model, _ = _fit_toy(n_estimators=20, max_depth=3)
    model.save(tmp_path)
    npz = dict(np.load(tmp_path / "model.npz"))
    npz["baseline_x"] = npz["baseline_x"] + 7.5  # shifts every served x -> chirality outputs move
    np.savez_compressed(tmp_path / "model.npz", **npz)
    _restamp_sha256sums(tmp_path)
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(tmp_path)


def test_load_raises_on_changed_pitch_dimension(tmp_path):
    """Change a declared geometry constant (pitch_length) AND re-stamp -> the pitch guard fires."""
    import json

    model, _ = _fit_toy(n_estimators=20, max_depth=3)
    model.save(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["pitch_length"] = meta["pitch_length"] + 1.0
    with open(tmp_path / "metadata.json", "w", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
    _restamp_sha256sums(tmp_path)
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(tmp_path)


def test_load_raises_on_tampered_feature_contract(tmp_path):
    """Perturb the recorded feature-contract fingerprint + re-stamp -> the feature-contract guard fires
    (the probe is unchanged, so the fingerprint comparison applies and raises IntegrityError)."""
    import json

    model, _ = _fit_toy(n_estimators=20, max_depth=3)
    model.save(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["feature_contract"]["fingerprint"][0] = float(meta["feature_contract"]["fingerprint"][0]) + 5.0
    with open(tmp_path / "metadata.json", "w", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
    _restamp_sha256sums(tmp_path)
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(tmp_path)


# --------------------------------------------------------------------------- #
# Task 4 --- serve seam (orientation, per-slot, velocity-keyed)
# --------------------------------------------------------------------------- #


def _point_reflect_frames(frames):
    """180-degree point reflection of positions AND velocities (ball included)."""
    f = frames.copy()
    f["x"] = 105.0 - f["x"]
    f["y"] = 68.0 - f["y"]
    f["vx"] = -f["vx"]
    f["vy"] = -f["vy"]
    return f


def _velocityless_frame(base=None):
    f = (base if base is not None else _toy_two_team_frame()).copy()
    f = f.drop(columns=[c for c in ("vx", "vy") if c in f.columns])
    f["speed_source"] = "unavailable"
    return f


def test_serve_source_vocab_is_closed():
    assert GHOST_OUTFIELD_SOURCE_VALUES == frozenset({"computed", "variant_unavailable", "fov_cropped"})


def test_serve_yields_distinct_positions_per_slot():
    model, frames = _fit_toy()
    one = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    out = serve_ghost_outfield_positions(one, model=model, home_team_id=1, actions=_toy_actions())
    assert set(out["slot_index"]) == {1.0, 2.0, 3.0, 4.0}
    assert (out["ghost_outfield_source"] == "computed").all()
    assert out[["ghost_gr_x", "ghost_gr_y"]].drop_duplicates().shape[0] >= 2


def test_serve_is_orientation_invariant_under_frame_mirror():
    model, frames = _fit_toy()
    one = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].copy()
    a = serve_ghost_outfield_positions(one, model=model, home_team_id=1, actions=_toy_actions())
    b = serve_ghost_outfield_positions(_point_reflect_frames(one), model=model, home_team_id=1, actions=_toy_actions())
    m = a.merge(b, on=["game_id", "period_id", "frame_id", "team_id", "slot_index"], suffixes=("_a", "_b"))
    assert len(m) == 4
    assert np.allclose(m["ghost_gr_x_a"], m["ghost_gr_x_b"], atol=1e-6)
    assert np.allclose(m["ghost_gr_y_a"], m["ghost_gr_y_b"], atol=1e-6)


def test_variant_key_default_on_velocity_bearing_frames():
    _, frames = _fit_toy(n_frames=1)
    _m, key = _resolve_outfield_model_for_frames(frames, None)
    assert key == "default"


def test_variant_key_position_only_on_declared_velocityless():
    _m, key = _resolve_outfield_model_for_frames(_velocityless_frame(), None)
    assert key == "position_only"


def test_missing_position_only_serves_nan_not_default(tmp_path, monkeypatch):
    """Declared-velocity-less frames must serve NaN when position_only is unbundled -- never default."""
    default_model, _ = _fit_toy(n_frames=40)
    default_model.save(tmp_path / "default")  # bundle ONLY default
    monkeypatch.setenv("SILLY_KICKS_GHOST_OUTFIELD_PATH", str(tmp_path))
    out = serve_ghost_outfield_positions(_velocityless_frame(), home_team_id=1, actions=_toy_actions())
    assert len(out) > 0
    assert (out["ghost_outfield_source"] == "variant_unavailable").all()
    assert out["ghost_gr_x"].isna().all()


def test_mixed_velocity_availability_raises():
    good = _toy_two_team_frame()
    good["speed_source"] = "derived"
    bad = _velocityless_frame()
    bad["frame_id"] = 1001
    mixed = pd.concat([good, bad], ignore_index=True)
    with pytest.raises(ValueError, match="mixed"):
        serve_ghost_outfield_positions(mixed, model=None, home_team_id=1, actions=_toy_actions())


# --------------------------------------------------------------------------- #
# Bundled-artifact CI gate (loads the REAL committed weights -- runs on every leg / both pandas
# majors; the toy-model tests above never touch the bundled artifacts). IMPL-06.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ["default", "position_only"])
def test_bundled_weights_load_and_predict_finite(variant):
    """The COMMITTED bundled artifact loads (SHA-256 + chirality + feature-contract) and predicts
    finite. Unlike the toy-model tests, this loads the REAL weights via ``from_variant``, so it runs
    on EVERY CI leg -- both pandas majors (ADR-057) -- and is the gate that catches a pandas-major
    chirality frame-hash skew (the class of bug that otherwise ships all-NaN)."""
    assert (_WEIGHTS_ROOT / variant / "SHA256SUMS").exists(), (
        f"bundled {variant} weights are MISSING -- a dropped artifact, asserted (never skipped)"
    )
    model = GhostOutfieldModel.from_variant(variant)  # the fail-closed load-guards run here
    assert model.feature_set == ("position_only" if variant == "position_only" else "faithful")
    feats = _extract_all_ghost_outfield_features(
        _outfield_probe_frame(), _outfield_probe_actions(), home_team_id="A", feature_set=model.feature_set
    )
    preds = model.predict_mean(feats)
    assert preds.shape[1] == 2 and np.isfinite(preds).all()


def test_present_but_corrupt_bundle_raises_not_variant_unavailable(tmp_path, monkeypatch):
    """IMPL-06b: a PRESENT-but-unloadable artifact must RAISE at the serve resolver, never silently
    serve an all-NaN ``variant_unavailable`` column that hides the corruption (a bad SHA, a
    pandas-major chirality skew, ...). An ABSENT variant stays honest-NaN --- see
    ``test_missing_position_only_serves_nan_not_default``."""
    model, _ = _fit_toy(n_frames=40)
    model.save(tmp_path / "default")
    # Corrupt model.npz WITHOUT re-stamping SHA256SUMS -> the SHA-256 check fails on load.
    npz = tmp_path / "default" / "model.npz"
    npz.write_bytes(npz.read_bytes() + b"tamper")
    monkeypatch.setenv("SILLY_KICKS_GHOST_OUTFIELD_PATH", str(tmp_path))
    with pytest.raises(IntegrityError):
        _resolve_outfield_model_for_frames(_toy_two_team_frame(), None)
    with pytest.raises(IntegrityError):
        serve_ghost_outfield_positions(_toy_two_team_frame(), model=None, home_team_id=1, actions=_toy_actions())


def test_canonical_frame_sha_is_cross_major_invariant():
    """IMPL-06c: the chirality/contract frame hash must be IDENTICAL across pandas majors, or a bundle
    hashed under one major fails its load-time frame-hash check under the other (ADR-057; CI's test
    legs float to pandas 3). The real gap is the probe's ball-row ``team_id``/``player_id`` = ``None``,
    which stays ``None`` on pandas 2 (JSON ``null``) but coerces to a float ``NaN`` on pandas 3 (JSON
    ``NaN``) --- ``.item()`` alone does NOT bridge None<->NaN. These assertions are major-AGNOSTIC (they
    hold on whatever host runs them; CI runs both majors)."""
    # (1) PRIMARY cross-major gate: the REAL probe hashes to the bundled artifact's stored digest on
    #     WHATEVER major runs this. A major-fragile hash fails here on the pandas-3 legs (the r4 red).
    stored = json.loads((_WEIGHTS_ROOT / "default" / "metadata.json").read_text())["chirality"]["frame_sha256"]
    assert _canonical_frame_sha(_outfield_probe_frame()) == stored

    # (2) A None cell and a NaN cell canonicalize to the SAME digest (the None<->NaN gap .item() does
    #     not bridge). Minimal frames, so it is major-agnostic.
    f_none = pd.DataFrame({"k": ["A", None], "v": [1.0, 2.0]})
    f_nan = pd.DataFrame({"k": ["A", np.nan], "v": [1.0, 2.0]})
    assert _canonical_frame_sha(f_none) == _canonical_frame_sha(f_nan)

    # (3) Red-green: the NA-normalization is load-bearing -- a hash WITHOUT it serializes a NaN cell as
    #     the JSON token ``NaN`` (vs the canonical ``null``), the very skew that differs across majors.
    naive = hashlib.sha256(json.dumps(f_nan.to_dict("records"), sort_keys=True, default=str).encode()).hexdigest()
    assert naive != _canonical_frame_sha(f_nan)


# --------------------------------------------------------------------------- #
# Task 6 --- SB360 (velocity-less freeze-frame) serving
# --------------------------------------------------------------------------- #


def test_serve_on_velocityless_sb360_frames_with_position_only_model():
    """A declared-velocity-less freeze frame + a position_only model -> real ghosts on the visible
    rearguard (position_only drops the 4 velocity features; no fabrication, no NaN)."""
    model, frames = _fit_toy(feature_set="position_only", n_estimators=20, max_depth=3)
    one = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].copy()
    vless = _velocityless_frame(one)
    out = serve_ghost_outfield_positions(vless, model=model, home_team_id=1, actions=_toy_actions())
    assert len(out) > 0
    assert (out["ghost_outfield_source"] == "computed").all()
    assert out["ghost_gr_x"].notna().all()
    assert out["ghost_gr_y"].notna().all()


def _visible_area_row(frame_id, polygon):
    return pd.DataFrame(
        [{"game_id": "G1", "period_id": 1, "frame_id": frame_id, "visible_area": np.asarray(polygon, dtype=float)}]
    )


def test_serve_fov_cropped_rearguard_is_honest_nan():
    """SB360 spec §8: a frame whose rearguard region is FOV-cropped serves honest-NaN (`fov_cropped`),
    NOT a fabricated ghost for a promoted midfielder (team 1 defends x=0 -> defensive third x∈[0,35])."""
    model, frames = _fit_toy(feature_set="position_only", n_estimators=20, max_depth=3)
    one = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].copy()
    vless = _velocityless_frame(one)
    fid = int(vless["frame_id"].iloc[0])
    # visible polygon = attacking half only (x>=52.5) -> A's defensive third is UNOBSERVED -> cropped.
    va = _visible_area_row(fid, [[52.5, 0.0], [105.0, 0.0], [105.0, 68.0], [52.5, 68.0]])
    out = serve_ghost_outfield_positions(vless, model=model, home_team_id=1, actions=_toy_actions(), visible_area=va)
    assert len(out) > 0
    cropped = out[out["ghost_outfield_source"] != "computed"]
    assert (out["ghost_outfield_source"] == "fov_cropped").all()
    assert cropped["ghost_gr_x"].isna().all()
    assert cropped["ghost_gr_y"].isna().all()


def test_serve_fov_observed_rearguard_is_computed():
    """The complementary side: a frame whose rearguard region IS observed serves `computed`."""
    model, frames = _fit_toy(feature_set="position_only", n_estimators=20, max_depth=3)
    one = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].copy()
    vless = _velocityless_frame(one)
    fid = int(vless["frame_id"].iloc[0])
    va = _visible_area_row(fid, [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])  # full pitch
    out = serve_ghost_outfield_positions(vless, model=model, home_team_id=1, actions=_toy_actions(), visible_area=va)
    assert (out["ghost_outfield_source"] == "computed").all()
    assert out["ghost_gr_x"].notna().all()


# --------------------------------------------------------------------------- #
# Task 5 --- ghost-rearguard coherence (measure; non-vacuity)
# --------------------------------------------------------------------------- #


def test_coherence_metric_reports_ordering_and_min_distance():
    model, frames = _fit_toy()
    served = serve_ghost_outfield_positions(frames, model=model, home_team_id=1, actions=_toy_actions())
    c = ghost_rearguard_coherence(served)
    assert c["n_groups"] > 0
    assert 0.0 <= c["ordering_fraction"] <= 1.0
    assert c["min_pairwise_distance_m"] >= 0.0


def test_coherence_metric_is_not_vacuous():
    """A perfectly-ordered ghost line scores 1.0; reversing its lateral order scores 0.0."""
    ordered = pd.DataFrame(
        {
            "game_id": ["g"] * 4,
            "period_id": [1] * 4,
            "frame_id": [1] * 4,
            "team_id": ["A"] * 4,
            "slot_index": [1.0, 2.0, 3.0, 4.0],
            "player_id": [1, 2, 3, 4],
            "ghost_gr_x": [20.0, 20.0, 20.0, 20.0],
            "ghost_gr_y": [10.0, 25.0, 40.0, 55.0],
            "ghost_outfield_source": ["computed"] * 4,
        }
    )
    assert ghost_rearguard_coherence(ordered)["ordering_fraction"] == 1.0
    shuffled = ordered.copy()
    shuffled["ghost_gr_y"] = [55.0, 40.0, 25.0, 10.0]  # reversed -> lateral order broken
    assert ghost_rearguard_coherence(shuffled)["ordering_fraction"] == 0.0


def test_feature_name_counts_and_partition():
    assert len(GHOST_OUTFIELD_FEATURE_NAMES) == 20
    assert len(_GHOST_OUTFIELD_VELOCITY_FEATURES) == 4
    assert len(GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY) == 16
    # position_only is exactly faithful minus the 4 velocity features, order-preserved.
    assert GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY == [
        f for f in GHOST_OUTFIELD_FEATURE_NAMES if f not in _GHOST_OUTFIELD_VELOCITY_FEATURES
    ]
    # No feature name references A's rearguard geometry (the leakage rule, at the name level).
    banned = ("rearguard", "back_line", "defensive_line", "deepest_defender", "def_compact")
    assert not [f for f in GHOST_OUTFIELD_FEATURE_NAMES if any(b in f for b in banned)]
    # slot_index is present (the one multi-agent feature).
    assert "slot_index" in GHOST_OUTFIELD_FEATURE_NAMES
    # Every velocity feature is actually in the faithful set.
    assert set(_GHOST_OUTFIELD_VELOCITY_FEATURES) <= set(GHOST_OUTFIELD_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Task 1 --- the leakage-safe feature extractor
# --------------------------------------------------------------------------- #


def test_features_do_not_leak_the_target_players_own_position():
    """The crux gate: no feature encodes the target rearguard player's own coordinates."""
    frame = _toy_two_team_frame()
    base = _extract(frame)
    slot1 = base.sort_values("slot_index").iloc[0]
    target_pid = slot1["player_id"]  # player_id is a KEY column (NOT a feature)
    # Small perturbation: large enough to move the target, small enough to preserve the slot ranks.
    moved = frame.copy()
    pmask = ids_match(moved["player_id"], target_pid) & (~moved["is_ball"].astype(bool))
    moved.loc[pmask, "x"] = moved.loc[pmask, "x"] + 0.5
    moved.loc[pmask, "y"] = moved.loc[pmask, "y"] - 0.5
    after = _extract(moved)
    a_slot1 = after[ids_match(after["player_id"], target_pid)].sort_values("slot_index").iloc[0]
    # The same player still occupies slot 1 (like-with-like, not a re-ranked slot).
    assert int(a_slot1["slot_index"]) == int(slot1["slot_index"])
    # Its FEATURE columns are byte-identical: no feature encodes the target's own coordinates.
    pd.testing.assert_series_equal(
        a_slot1[GHOST_OUTFIELD_FEATURE_NAMES],
        slot1[GHOST_OUTFIELD_FEATURE_NAMES],
        check_names=False,
    )
    # ...and the TARGET moved (non-vacuity: the perturbation was real).
    assert not np.isclose(a_slot1["target_x"], slot1["target_x"])


def test_extractor_shape_and_slots():
    frame = _toy_two_team_frame()
    out = _extract(frame, n_rearguard=4)
    assert len(out) == 4  # 4 slots for the one in-possession team
    assert set(out["slot_index"]) == {1, 2, 3, 4}
    assert out[GHOST_OUTFIELD_FEATURE_NAMES].notna().all().all()
    # player_id is present as a bookkeeping key (NOT a feature).
    assert "player_id" in out.columns
    assert "player_id" not in GHOST_OUTFIELD_FEATURE_NAMES
    # Targets are goal-relative (deepest-4 near A's defended goal at x=0 -> small target_x).
    assert (out["target_x"] < 52.5).all()
    # Slots ranked left-to-right by goal-relative y: slot 1 = smallest target_y.
    ordered = out.sort_values("slot_index")
    assert list(ordered["target_y"]) == sorted(ordered["target_y"])


def test_position_only_drops_velocity_columns():
    frame = _toy_two_team_frame()
    out = _extract(frame, feature_set="position_only")
    assert not set(_GHOST_OUTFIELD_VELOCITY_FEATURES) & set(out.columns)
    assert out[GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY].notna().all().all()


def test_extractor_single_team_frame_yields_no_rows():
    """Honest-NaN: a non-two-team frame produces no fabricated rows (never a raise-through)."""
    frame = _toy_two_team_frame()
    one_team = frame[ids_match(frame["team_id"], 1) | frame["is_ball"].astype(bool)]
    out = _extract(one_team)
    assert len(out) == 0


def test_both_teams_makes_team_in_possession_a_live_feature():
    """Possession-conditioned (Option 2): ``both_teams=True`` (training) models BOTH teams' deepest-n, so
    ``team_in_possession`` VARIES (1 for the ball-carrier, 0 for the other); the default (serving) models
    only the carrier, where it is constant 1.0."""
    frame = _toy_two_team_frame()  # team 1 in possession
    both = _extract_all_ghost_outfield_features(frame, _toy_actions(), home_team_id=1, both_teams=True)
    assert set(both["team_id"]) == {1, 2}
    assert set(both["team_in_possession"]) == {0.0, 1.0}
    assert (both.loc[ids_match(both["team_id"], 1), "team_in_possession"] == 1.0).all()  # carrier
    assert (both.loc[ids_match(both["team_id"], 2), "team_in_possession"] == 0.0).all()  # out of possession
    # Default (serve): carrier only -> constant 1.0.
    only = _extract_all_ghost_outfield_features(frame, _toy_actions(), home_team_id=1)
    assert set(only["team_id"]) == {1}
    assert (only["team_in_possession"] == 1.0).all()
