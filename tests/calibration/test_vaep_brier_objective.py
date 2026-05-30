import math

import numpy as np
import pandas as pd
from ruthless import Candidate, assert_cache_equivalence

import silly_kicks.calibration._vaep_brier_objective as vbo
from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective

_CP = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}


def _candidates():
    # MUST vary all 3 patch params across >=2 values each (assert_cache_equivalence contract, L4).
    return [
        Candidate(id="c0", params={"k3": 1.0, "pre_seconds": 1.5, "min_displacement_m": 3.0}),
        Candidate(id="c1", params={"k3": 2.5, "pre_seconds": 3.0, "min_displacement_m": 5.0}),
    ]


def _obj(fold, xt):
    return AugmentedVaepBrierObjective(fold=fold, xt=xt, carrier_params=_CP, seed=42)


def test_patch_params_declared():
    assert AugmentedVaepBrierObjective.patch_params == frozenset({"k3", "pre_seconds", "min_displacement_m"})


def test_returns_finite_brier_and_per_provider_attrs(stage2_fold, frozen_xt):
    obj = _obj(stage2_fold, frozen_xt)
    m = obj.evaluate(_candidates()[0])
    assert np.isfinite(m["brier"])
    assert any(k.startswith("brier__") for k in m)  # per-provider Brier
    assert any(k.startswith("brier_se__") for k in m)  # per-provider CV SE (M1)


def test_cache_equivalence_fast_equals_full(stage2_fold, frozen_xt):
    obj = _obj(stage2_fold, frozen_xt)
    # Deterministic XGBoost + independent enrich_full => fast path ≡ full recompute to 1e-9.
    assert_cache_equivalence(obj, _candidates())


def test_feature_matrix_parity_full_vs_invariant_patch(synth, frozen_xt):
    # N2: assert parity at the FEATURE level (not just downstream Brier).
    from silly_kicks.calibration._features import (
        ALL_FEATURES,
        enrich_full,
        enrich_invariant,
        patch_trial_columns,
    )

    actions, frames, home = synth
    params = {"k3": 2.0, "pre_seconds": 2.5, "min_displacement_m": 4.0}
    full = enrich_full(actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home, carrier_params=_CP, **params)
    base, links, _das = enrich_invariant(
        actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home, carrier_params=_CP
    )
    patched = patch_trial_columns(base_actions=base, frames=frames, links=links, home_team_id=home, **params)
    pd.testing.assert_frame_equal(
        full[ALL_FEATURES].reset_index(drop=True),
        patched[ALL_FEATURES].reset_index(drop=True),
        check_dtype=False,
    )


def test_h1_penalty_is_path_stable(stage2_fold, frozen_xt, monkeypatch):
    # When H1 fires, evaluate and evaluate_patch must return the SAME default-Brier-anchored
    # penalty (R1 stateless penalty). Force the gate to fire on every call.
    monkeypatch.setattr(vbo, "h1_penalty_fires", lambda *a, **k: True)
    obj = _obj(stage2_fold, frozen_xt)
    cand = _candidates()[0]
    inv = obj.prepare()
    full = obj.evaluate(cand)
    patch = obj.evaluate_patch(inv, cand)
    assert full == patch
    assert math.isfinite(full["brier"]) and full["brier"] > 0  # a deliberately-bad finite penalty


def test_provider_cv_keeps_labels_aligned_on_single_class_fold():
    # M4: if a fold is single-class for ONE label, it is skipped for BOTH (no zip misalignment).
    from silly_kicks.calibration._vaep_brier_objective import _provider_cv

    n = 60
    x = pd.DataFrame({"f": np.linspace(0, 1, n)})
    mids = np.array([f"m{i // 6}" for i in range(n)])  # 10 matches => GroupKFold(5)
    y_scores = (np.arange(n) % 2).astype(int)
    y_concedes = np.ones(n, dtype=int)
    y_concedes[mids == "m0"] = 0  # makes some train folds single-class for concedes
    mean_b, _se = _provider_cv(x, y_scores, y_concedes, mids, seed=42)
    assert mean_b is None or np.isfinite(mean_b)  # no crash, finite-or-None
