"""Per-type serve-gate: pure decision fn + model fields + save/load + real-artifact lock."""

import math

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._gk_completion import (
    _GATE_LCB_FLOOR,
    _GATE_N_MIN,
    GkCompletionModel,
    serve_mode_from_lcb,
)

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
_PASS = spadlconfig.actiontype_id["pass"]


def test_serve_mode_lcb_above_floor_is_model():
    assert serve_mode_from_lcb(0.55, n=200) == "model"


def test_serve_mode_lcb_at_or_below_floor_is_base_rate():
    assert serve_mode_from_lcb(_GATE_LCB_FLOOR, n=200) == "base_rate"  # 0.5 is NOT > 0.5
    assert serve_mode_from_lcb(0.42, n=200) == "base_rate"


def test_serve_mode_none_or_nan_lcb_is_base_rate():
    assert serve_mode_from_lcb(None, n=200) == "base_rate"
    assert serve_mode_from_lcb(float("nan"), n=200) == "base_rate"


def test_serve_mode_too_few_samples_is_base_rate():
    assert serve_mode_from_lcb(0.99, n=_GATE_N_MIN - 1) == "base_rate"
    assert serve_mode_from_lcb(0.55, n=_GATE_N_MIN) == "model"  # exactly n_min is enough


def _fitted_model_with_gate(serve_mode):
    # A minimally-fitted model with explicit per-type gate + base rates (no real corpus needed).
    rng = np.random.default_rng(0)
    n = 200
    feats = pd.DataFrame(
        {
            "length": rng.normal(20, 5, n),
            "dx": rng.normal(10, 3, n),
            "dy_abs": np.abs(rng.normal(0, 5, n)),
            "forwardness": rng.normal(0, 1, n),
            "dest_x": rng.uniform(0, 105, n),
            "dest_y": rng.uniform(0, 68, n),
            "dest_y_off": np.abs(rng.uniform(0, 34, n)),
            "dest_defender_density": rng.uniform(0, 1, n),
            "is_goalkick": (np.arange(n) % 3 == 0).astype(float),
            "is_throw_in": (np.arange(n) % 3 == 1).astype(float),
        }
    )
    y = pd.Series((rng.random(n) < 0.6).astype(int))
    m = GkCompletionModel().fit(feats, y)
    m._type_serve_mode = dict(serve_mode)
    m._type_gate_metrics = {k: {"auc": 0.5, "lcb": 0.49, "n": 80} for k in serve_mode}
    return m


def test_serve_mode_for_types_maps_per_gate():
    m = _fitted_model_with_gate({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    tids = np.array([_GOALKICK, _THROW_IN, _PASS])
    assert list(m.serve_mode_for_types(tids)) == ["base_rate", "base_rate", "model"]


def test_serve_mode_for_types_absent_type_defaults_model():
    m = _fitted_model_with_gate({})  # no gate -> fail-open
    assert list(m.serve_mode_for_types(np.array([_GOALKICK, _PASS]))) == ["model", "model"]


def test_base_rate_for_types_returns_per_type_rate():
    m = _fitted_model_with_gate({"goalkick": "base_rate"})
    br = m.base_rate_for_types(np.array([_GOALKICK, _THROW_IN, _PASS]))
    assert math.isclose(br[0], m._base_rates["goalkick"])
    assert math.isclose(br[1], m._base_rates["throw_in"])
    assert math.isclose(br[2], m._base_rates["other"])


def test_save_load_roundtrips_gate(tmp_path):
    m = _fitted_model_with_gate({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    m.save(tmp_path)
    back = GkCompletionModel.load(tmp_path)
    assert back._type_serve_mode == m._type_serve_mode
    assert back._type_gate_metrics["goalkick"]["n"] == 80


def test_load_fail_open_when_gate_absent(tmp_path):
    import json

    m = _fitted_model_with_gate({"goalkick": "base_rate"})
    m.save(tmp_path)
    d = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    del d["type_serve_mode"]  # simulate a pre-gate (4.21.0) artifact
    (tmp_path / "model.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(f"{GkCompletionModel._sha(tmp_path)}  model.json\n", encoding="utf-8")
    back = GkCompletionModel.load(tmp_path)
    assert back._type_serve_mode == {}  # absent -> empty
    assert list(back.serve_mode_for_types(np.array([_GOALKICK]))) == ["model"]  # fail-open


def test_per_type_gate_from_oof_smoke():
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    tg = importlib.import_module("train_gk_completion")
    n = 300
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {"is_goalkick": (np.arange(n) % 3 == 0).astype(float), "is_throw_in": (np.arange(n) % 3 == 1).astype(float)}
    )
    y = (rng.random(n) < 0.5).astype(int)
    oof = rng.random(n)  # random scores -> AUC ~ 0.5 -> LCB <= 0.5 -> base_rate
    sm, gm = tg._per_type_gate_from_oof(oof, y, X)
    assert set(sm) == {"goalkick", "throw_in", "other"}
    assert all(v in ("model", "base_rate") for v in sm.values())
    assert all(set(gm[k]) == {"auc", "lcb", "n"} for k in gm)


def test_bundled_skillcorner_goalkick_is_base_rate():
    # Real-artifact lock (review M3): the committed skillcorner gate routes goal-kicks to base_rate
    # (goal-kick AUC 0.461 < chance, retrained 4.73.0; 0.433 pre-RC4). Stronger than an owner e2e --
    # the variable is committed. Only the MODE is asserted, so a retrain that keeps the mode is fine.
    m = GkCompletionModel.from_variant("skillcorner")
    assert m._type_serve_mode.get("goalkick") == "base_rate"
    assert m._type_serve_mode.get("other") == "model"  # GK-passes stay model-scored (AUC 0.740)


def test_bundled_gs_default_goalkick_mode_is_locked():
    # Measured-value golden (review-2 L-A): the GS goal-kick mode is "model" -- GS goal-kick completion
    # IS predictable from geometry (OOF AUC 0.835, LCB 0.809 > 0.5 floor after the 4.73.0 retrain;
    # 0.836/0.798 before it), unlike SkillCorner.
    # Permanent regression lock set from the SK-91 owner-run re-bundle.
    m = GkCompletionModel.from_variant("default")
    assert m._type_serve_mode.get("goalkick") == "model"
