"""Velocity-keyed variant auto-select (Tasks 3-5, D4).

Layer A: `variant_key_for_velocity` (pure 2-way key) + `velocity_availability_is_mixed` (the
partially-marked predicate that closes the M3 fabrication hole -- a mixed set must RAISE, not resolve
to the default velocity model). Layer B (the per-model `(model, variant_key)` resolvers) and the
serve-seam behavioral tests are added by Tasks 4-5.
"""

from __future__ import annotations

import contextlib
import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking._ghost_gk as _gg
import silly_kicks.tracking._xcross_attempt as _xc
import silly_kicks.tracking._xshot_occurrence as _xs
from silly_kicks.tracking._velocity_availability import (
    variant_key_for_velocity,
    velocity_availability_is_mixed,
)
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames
from tests.tracking.test_xshot_occurrence import _synthetic_match_frames


def _frames(marks: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"speed_source": marks, "vx": [0.0] * len(marks), "vy": [0.0] * len(marks)})


# -- Layer A: variant_key_for_velocity (pure 2-way) ------------------------------------------------


def test_key_declared_unavailable_is_position_only():
    assert variant_key_for_velocity(_frames([SPEED_SOURCE_UNAVAILABLE] * 3)) == "position_only"


def test_key_velocity_bearing_is_default():
    assert variant_key_for_velocity(_frames(["native"] * 3)) == "default"


def test_key_empty_is_default():
    # The VAEP column-discovery pass sends an EMPTY frame set -> default, byte-identical, no raise.
    assert variant_key_for_velocity(pd.DataFrame({"speed_source": []})) == "default"


def test_key_partial_marked_is_default_not_position_only():
    # A partially-marked set is NOT all-unavailable, so the key is "default"; the seam RAISES on it via
    # velocity_availability_is_mixed (below) rather than silently resolving here.
    assert variant_key_for_velocity(_frames([SPEED_SOURCE_UNAVAILABLE, "native"])) == "default"


# -- Layer A: velocity_availability_is_mixed (the M3 guard) ----------------------------------------


def test_mixed_true_on_partially_marked():
    assert velocity_availability_is_mixed(_frames([SPEED_SOURCE_UNAVAILABLE, "native"])) is True


def test_mixed_false_on_all_marked():
    assert velocity_availability_is_mixed(_frames([SPEED_SOURCE_UNAVAILABLE] * 2)) is False


def test_mixed_false_on_none_marked():
    assert velocity_availability_is_mixed(_frames(["native"] * 2)) is False


def test_mixed_false_on_empty():
    assert velocity_availability_is_mixed(pd.DataFrame({"speed_source": []})) is False


def test_mixed_false_when_no_speed_source_column():
    assert velocity_availability_is_mixed(pd.DataFrame({"x": [1.0, 2.0]})) is False


# -- Layer B: per-model (model, variant_key) resolvers (Task 4) -------------------------------------


def _declared(n: int = 2) -> pd.DataFrame:
    return _frames([SPEED_SOURCE_UNAVAILABLE] * n)


def _velocity(n: int = 2) -> pd.DataFrame:
    return _frames(["native"] * n)


# All FOUR resolver branches, PARAMETRIZED over the three models (was xShot 4/4 + xCross/ghost 2/4
# hand-clones -- the "representative xShot only" gap the reviewer flagged as G3/G4). The (model, key)
# contract is identical across models; the ONE genuine asymmetry is `warns`: xShot/xCross warn on an
# unbundled position_only, ghost does NOT (its degrade is column-signalled, not a log line).
# (name, resolve_fn, model_cls, warns_on_unbundled_po)
_RESOLVER_CASES = [
    ("xshot", _xs._resolve_xshot_model_for_frames, _xs.XShotOccurrenceModel, True),
    ("xcross", _xc._resolve_xcross_model_for_frames, _xc.XCrossAttemptModel, True),
    ("ghost", _gg._resolve_ghost_model_for_frames, _gg.GhostGkModel, False),
]
_RESOLVER_IDS = [c[0] for c in _RESOLVER_CASES]


@pytest.mark.parametrize("name,resolve,model_cls,warns", _RESOLVER_CASES, ids=_RESOLVER_IDS)
def test_resolver_override_is_custom(name, resolve, model_cls, warns, monkeypatch):
    monkeypatch.delenv("SILLY_KICKS_GHOST_GK_PATH", raising=False)  # ghost treats env as an override too
    # A REAL (unfitted) instance -- object() would (correctly) fail _resolve_model's type check (P3).
    m = model_cls(feature_set="position_only")
    resolved, key = resolve(_declared(), model=m)
    assert resolved is m
    assert key == "custom"  # V1: override ALWAYS "custom" (closed set; never shipped_variant)


@pytest.mark.parametrize("name,resolve,model_cls,warns", _RESOLVER_CASES, ids=_RESOLVER_IDS)
def test_resolver_declared_resolves_position_only(name, resolve, model_cls, warns, monkeypatch):
    monkeypatch.delenv("SILLY_KICKS_GHOST_GK_PATH", raising=False)
    stub = object()
    monkeypatch.setattr(model_cls, "from_variant", classmethod(lambda cls, v: stub))
    resolved, key = resolve(_declared(), model=None)
    assert resolved is stub
    assert key == "position_only"


@pytest.mark.parametrize("name,resolve,model_cls,warns", _RESOLVER_CASES, ids=_RESOLVER_IDS)
def test_resolver_declared_unbundled_is_NaN_not_default(name, resolve, model_cls, warns, monkeypatch):
    monkeypatch.delenv("SILLY_KICKS_GHOST_GK_PATH", raising=False)

    def _boom(cls, v):
        if v == "position_only":
            raise FileNotFoundError
        raise AssertionError("must NOT fall back to the default velocity model on velocity-less frames")

    monkeypatch.setattr(model_cls, "from_variant", classmethod(_boom))
    ctx = pytest.warns(UserWarning) if warns else contextlib.nullcontext()
    with ctx:
        resolved, key = resolve(_declared(), model=None)
    assert resolved is None  # NaN sentinel -- NEVER the default model (the load-bearing asymmetry)
    assert key == "position_only"


@pytest.mark.parametrize("name,resolve,model_cls,warns", _RESOLVER_CASES, ids=_RESOLVER_IDS)
def test_resolver_velocity_bearing_is_default(name, resolve, model_cls, warns, monkeypatch):
    monkeypatch.delenv("SILLY_KICKS_GHOST_GK_PATH", raising=False)
    stub = object()
    monkeypatch.setattr(model_cls, "from_variant", classmethod(lambda cls, v: stub))
    resolved, key = resolve(_velocity(), model=None)
    assert resolved is stub
    assert key == "default"


# -- Task 5: serve-seam restructure (compute_xshot_occurrence, representative) ----------------------


def _declare_unavailable(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns]).copy()
    out["speed"] = np.nan
    out["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    return out


class _MockPositionOnlyXShot:
    """Minimal stand-in for a fitted position_only model (Task 10 exercises the REAL weights)."""

    feature_set: ClassVar[str] = "position_only"
    carrier_params: ClassVar[dict] = dict(_xs._DEFAULT_CARRIER_PARAMS)

    def predict_proba(self, feats):
        return np.full(len(feats), 0.5)


class _MockPositionOnlyXCross:
    """Minimal stand-in for a fitted position_only xCross model (mirrors _MockPositionOnlyXShot)."""

    feature_set: ClassVar[str] = "position_only"
    carrier_params: ClassVar[dict] = dict(_xc._DEFAULT_CARRIER_PARAMS)

    def predict_proba(self, feats):
        return np.full(len(feats), 0.5)


def test_xshot_declared_serves_via_position_only(monkeypatch):
    # The auto-select unlock: a declared-velocity-less (SB360) frame now SERVES a value via the
    # position_only variant (was honest-NaN in 4.90.0). Uses a mock so no bundled weights are needed.
    monkeypatch.setattr(_xs.XShotOccurrenceModel, "from_variant", classmethod(lambda cls, v: _MockPositionOnlyXShot()))
    frames = _declare_unavailable(_synthetic_match_frames(n_frames=5))
    out = _xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    assert out["xshot_occurrence"].notna().any()  # position_only served a value on velocity-less frames


def test_xcross_declared_serves_via_position_only(monkeypatch):
    # G5: the xCross seam mirror of the unlock above -- a declared-velocity-less frame SERVES a value
    # via the position_only variant. `_freeze_frame` is xcross-valid (the velocity-bearing sibling in
    # test_xcross_attempt_velocity_contract asserts notna on it), so the mock scores a real row.
    from tests.tracking.test_xcross_attempt_velocity_contract import _freeze_frame

    monkeypatch.setattr(_xc.XCrossAttemptModel, "from_variant", classmethod(lambda cls, v: _MockPositionOnlyXCross()))
    frames, actions = _freeze_frame(declare=SPEED_SOURCE_UNAVAILABLE, with_velocity=False)
    out = _xc.compute_xcross_attempt(frames, model=None, home_team_id="A", actions=actions)
    assert out["xcross_attempt"].notna().any()  # position_only served a value on velocity-less frames


def test_xshot_declared_no_bundled_position_only_returns_nan_not_default():
    # Phase-A state (no bundled position_only): auto-select falls back to NaN, NEVER the default
    # velocity model -- the load-bearing asymmetry -- and warns.
    frames = _declare_unavailable(_synthetic_match_frames(n_frames=5))
    with pytest.warns(UserWarning, match="position_only"):
        out = _xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    assert out["xshot_occurrence"].isna().all()


# The mixed-raise (M3 fabrication-hole guard) and undeclared-raise (forgot derive_velocities()) guards
# are UNIFORM across the two compute seams -- parametrized so xCross's seam is covered (G5), not just
# xShot's. Ghost's mixed-raise is asserted separately below (its seam is _serve_positions_core, which
# raises _GhostVelocityUnavailableError rather than returning a NaN column -- the reviewer's asymmetry).
_COMPUTE_SEAMS = [("xshot", _xs.compute_xshot_occurrence), ("xcross", _xc.compute_xcross_attempt)]


@pytest.mark.parametrize("name,compute", _COMPUTE_SEAMS, ids=[c[0] for c in _COMPUTE_SEAMS])
def test_compute_mixed_availability_raises(name, compute):
    frames = _synthetic_match_frames(n_frames=5)
    frames.loc[frames.index[:3], "speed_source"] = SPEED_SOURCE_UNAVAILABLE  # partial-mark -> mixed
    with pytest.raises(ValueError, match="mixed"):
        compute(frames, model=None, home_team_id=1)


@pytest.mark.parametrize("name,compute", _COMPUTE_SEAMS, ids=[c[0] for c in _COMPUTE_SEAMS])
def test_compute_undeclared_missing_velocity_raises(name, compute):
    frames = _synthetic_match_frames(n_frames=5).drop(columns=["vx", "vy"])  # no marker
    with pytest.raises(ValueError, match="derive_velocities"):
        compute(frames, model=None, home_team_id=1)


def test_ghost_mixed_availability_raises():
    # G6: the ghost seam's M3 guard (compute_ghost_gk / _serve_positions_core / add_ghost_gk all raise
    # on a partially-marked set -- otherwise the marked rows would fabricate speed=NaN).
    frames = _make_ghost_gk_frames()
    frames.loc[frames.index[:2], "speed_source"] = SPEED_SOURCE_UNAVAILABLE  # partial-mark -> mixed
    with pytest.raises(ValueError, match="mixed"):
        _gg.serve_ghost_gk_positions(frames, model=None, home_team_id=1)


# -- Task 6: load-guard feature_set threading (r2) -------------------------------------------------


def test_feature_contract_block_length_is_feature_set_aware():
    # r2: the model-INDEPENDENT contract block threads feature_set through to the extractor, so a
    # position_only artifact fingerprints its SHORTER vector. This is what makes a position_only
    # model's save/load guards correct; the full fitted round-trip with real weights is Task 10, and
    # the faithful round-trip regression (existing model save/load suites) covers the threading
    # MECHANISM (feature_set='faithful' must stay byte-identical).
    assert len(_xs._feature_contract_block("faithful")["fingerprint"]) == 27
    assert len(_xs._feature_contract_block("position_only")["fingerprint"]) == 26
    assert len(_xc._feature_contract_block("faithful")["fingerprint"]) == 16
    assert len(_xc._feature_contract_block("position_only")["fingerprint"]) == 15
    assert len(_gg._feature_contract_block("faithful")["fingerprint"]) == 26
    assert len(_gg._feature_contract_block("position_only")["fingerprint"]) == 21


# -- Task 7: provenance columns on the add_* path -------------------------------------------------


def test_add_xshot_emits_variant_provenance_column():
    from tests.tracking.test_xshot_occurrence import _actions_and_frames_for_add

    actions, frames = _actions_and_frames_for_add()

    # Velocity-bearing -> auto-select serves the bundled default variant.
    out = _xs.add_xshot_occurrence(actions, frames, home_team_id=1)
    assert (out["xshot_occurrence_variant"] == "default").all()
    # D2: the provenance survives @nan_safe_enrichment AS STRINGS (uncoerced). Assert the VALUE type,
    # never the dtype literal -- a Python-str column is `object` on pandas 2 but `StringDtype("str")` on
    # pandas 3, so `dtype == object` is a spurious cross-major failure (ADR-057).
    assert all(isinstance(v, str) for v in out["xshot_occurrence_variant"])

    # Declared velocity-less -> auto-select chooses position_only (unbundled -> NaN value + warn), but
    # the PROVENANCE still records the chosen variant.
    declared = _declare_unavailable(frames)
    with pytest.warns(UserWarning, match="position_only"):
        out2 = _xs.add_xshot_occurrence(actions, declared, home_team_id=1)
    assert (out2["xshot_occurrence_variant"] == "position_only").all()
    assert out2["xshot_occurrence"].isna().all()  # NaN value (unbundled), NOT the default model


def test_add_xshot_explicit_override_is_custom():
    from tests.tracking.test_xshot_occurrence import _actions_and_frames_for_add

    actions, frames = _actions_and_frames_for_add()
    m = _xs.XShotOccurrenceModel.from_variant("default")  # a real instance override
    out = _xs.add_xshot_occurrence(actions, frames, model=m, home_team_id=1)
    assert (out["xshot_occurrence_variant"] == "custom").all()


# -- Position-only MODEL round-trips: fit -> save -> load -> predict -------------------------------
# TDD backfill for the model-class feature_set sites (fit/predict/save/load/width-check). Every one
# hardcoded the FAITHFUL feature set; a position_only model has a SHORTER vector, so a bare
# `features[FAITHFUL]` select (or a `!= len(FAITHFUL)` width check) raises `KeyError`/`ValueError` on
# it. These exercise the whole cycle on a synthetic position_only matrix and assert (a) predict is
# finite, (b) save/load is exact-parity, (c) the recorded metadata feature_names match the SHORTER
# set -- the shape the xShot train smoke caught in `prepare_*` and these fixes closed everywhere else.


def test_ghost_position_only_fit_save_load_predict_roundtrip(tmp_path):
    names = list(_gg.GHOST_GK_FEATURE_NAMES_POSITION_ONLY)
    assert len(names) == 21
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    labels = pd.DataFrame({"gk_x": rng.uniform(2.0, 18.0, n), "gk_y": rng.uniform(24.0, 44.0, n)})

    m = _gg.GhostGkModel(feature_set="position_only", n_estimators=8, max_depth=3)
    assert m._feature_names() == names  # helper picks the 21-col set, not the faithful 26
    m.fit(X, labels)
    pred1 = m.predict(X)
    assert pred1.shape == (n, 2)
    assert np.isfinite(pred1).all()

    out_dir = tmp_path / "ghost_po"
    m.save(out_dir)
    m2 = _gg.GhostGkModel.load(out_dir)
    assert m2.feature_set == "position_only"
    np.testing.assert_allclose(m2.predict(X), pred1, atol=1e-9, rtol=0)

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == names  # 21 position_only names, not the faithful 26
    assert meta["feature_set"] == "position_only"


def test_xshot_position_only_fit_save_load_predict_roundtrip(tmp_path):
    pytest.importorskip("xgboost")
    names = list(_xs.XSHOT_FEATURE_NAMES_POSITION_ONLY)
    assert len(names) == 26  # faithful 27 minus `speed`
    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    y = pd.Series(((X["r"] + rng.normal(scale=0.5, size=n)) < -0.3).astype(int))
    if int(y.sum()) == 0:
        y.iloc[:5] = 1

    m = _xs.XShotOccurrenceModel(feature_set="position_only")
    m.fit(X, y)
    p1 = m.predict_proba(X)
    assert p1.shape == (n,)
    assert np.isfinite(p1).all()

    out_dir = tmp_path / "xshot_po"
    m.save(out_dir)
    m2 = _xs.XShotOccurrenceModel.load(out_dir)
    assert m2.feature_set == "position_only"
    np.testing.assert_allclose(m2.predict_proba(X), p1, atol=1e-9, rtol=0)

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == names
    assert len(meta["feature_names"]) == 26


def test_xcross_position_only_fit_save_load_predict_roundtrip(tmp_path):
    pytest.importorskip("xgboost")
    names = list(_xc.XCROSS_FEATURE_NAMES_POSITION_ONLY)
    assert len(names) == 15  # faithful 16 minus `ball_speed`
    rng = np.random.default_rng(2)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    y = pd.Series(((X.iloc[:, 0] + rng.normal(scale=0.5, size=n)) < -0.3).astype(int))
    if int(y.sum()) == 0:
        y.iloc[:5] = 1

    m = _xc.XCrossAttemptModel(feature_set="position_only")
    m.fit(X, y)
    p1 = m.predict_proba(X)
    assert p1.shape == (n,)
    assert np.isfinite(p1).all()

    out_dir = tmp_path / "xcross_po"
    m.save(out_dir)
    m2 = _xc.XCrossAttemptModel.load(out_dir)
    assert m2.feature_set == "position_only"
    np.testing.assert_allclose(m2.predict_proba(X), p1, atol=1e-9, rtol=0)

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == names
    assert len(meta["feature_names"]) == 15


# -- Task 5 Step 5: the gkdv non-vacuity rung (serve_ghost_gk_positions on a declared frame) --------


def _fitted_position_only_ghost():
    """A REAL fitted position_only ghost (tiny random fit -- values are irrelevant; the rung asserts
    that serve yields ROWS, not that they are correct)."""
    rng = np.random.default_rng(7)
    n = 60
    names = list(_gg.GHOST_GK_FEATURE_NAMES_POSITION_ONLY)
    X = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    labels = pd.DataFrame({"gk_x": rng.uniform(2.0, 18.0, n), "gk_y": rng.uniform(24.0, 44.0, n)})
    return _gg.GhostGkModel(feature_set="position_only", n_estimators=8, max_depth=3).fit(X, labels)


def test_gkdv_serve_ghost_positions_serves_rows_on_declared_frame_with_po_model():
    # The gkdv unlock (Task 5 Step 5): serve_ghost_gk_positions on a DECLARED (velocity-less, SB360)
    # freeze frame yields ROWS once a position_only ghost model is available -- it was ZERO under the
    # ADR-054 refusal, so gkdv's ghost arm could not work on SB360. BOTH sides asserted: the refusal
    # (model=None, no bundled PO -> 0 rows) AND the unlock (an explicit PO model -> rows).
    from silly_kicks.tracking import serve_ghost_gk_positions

    declared = _declare_unavailable(_make_ghost_gk_frames())

    # WAS zero: no PO model available (auto-select, PO unbundled) -> serve REFUSES, returns no rows.
    refused = serve_ghost_gk_positions(declared, model=None, home_team_id=1)
    assert len(refused) == 0, "auto-select with no bundled PO variant must refuse (no rows), not fabricate"

    # NOW rows: an explicit position_only model -> serve extracts the 21-col PO vector and predicts.
    out = serve_ghost_gk_positions(declared, model=_fitted_position_only_ghost(), home_team_id=1)
    assert len(out) >= 1, "serve must yield rows on a declared freeze frame once a PO model exists"
    assert np.isfinite(out["ghost_gr_x"]).all()
    assert np.isfinite(out["ghost_gr_y"]).all()
