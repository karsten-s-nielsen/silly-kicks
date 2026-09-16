"""TF-61 VAEP_adjusted — eqs 8-12 combiner (adjusted_value) + rate_adjusted (conftest fixtures)."""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as cfg
from silly_kicks.spadl.utils import add_names
from silly_kicks.vaep import formula
from silly_kicks.vaep.adjusted import adjusted_value


def _acts(team):
    n = len(team)
    df = pd.DataFrame(
        dict(
            team_id=team,
            time_seconds=np.arange(n, dtype=float),
            type_id=cfg.actiontype_id["pass"],
            result_id=cfg.result_id["success"],
            bodypart_id=cfg.bodypart_id["foot"],
            start_x=50.0,
            start_y=34.0,
            end_x=60.0,
            end_y=34.0,
        )
    )
    return add_names(df)


def test_weighting_matches_formula():
    a = _acts([1, 1, 1])
    pss = pd.Series([0.2, 0.5, 0.7])
    pcf = pd.Series([0.1, 0.1, 0.2])
    xs = pd.Series([0.9, 0.4, 0.8])
    got = adjusted_value(a, pss, pcf, xs)
    want = formula.value(a, xs * pss, (1 - xs) * pcf)
    assert np.allclose(got["vaep_value"], want["vaep_value"])
    assert list(got.columns) == ["offensive_value", "defensive_value", "vaep_value"]


def test_prev_state_uses_xs_of_previous_action():
    # team switch at index 2 -> the delta for row 2 draws on the PREVIOUS action's ADJUSTED prob
    # (xSuccess[1]), never xSuccess[2]; pins per-action element-wise weighting before formula.value.
    a = _acts([1, 1, 2, 2])
    pss = pd.Series([0.3, 0.6, 0.2, 0.5])
    pcf = pd.Series([0.1, 0.2, 0.1, 0.3])
    xs = pd.Series([0.5, 0.9, 0.5, 0.7])
    got = adjusted_value(a, pss, pcf, xs)
    ref = formula.value(a, xs * pss, (1 - xs) * pcf)
    assert np.allclose(got["vaep_value"], ref["vaep_value"])


# --- VAEP.rate_adjusted (surgical result-feature counterfactual) ---
import pytest  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402

from silly_kicks.vaep import VAEP  # noqa: E402


def test_rate_adjusted_requires_fitted(fitted_xs, game, actions):
    with pytest.raises(NotFittedError):
        VAEP().rate_adjusted(game, actions, fitted_xs)


def test_hybrid_raises_non_vacuity(fitted_hybrid, fitted_xs, game, actions):
    with pytest.raises(ValueError, match=r"result-bearing|no-op|STANDARD"):
        fitted_hybrid.rate_adjusted(game, actions, fitted_xs)


def test_rate_adjusted_shapes(fitted_vaep, fitted_xs, game, actions):
    out = fitted_vaep.rate_adjusted(game, actions, fitted_xs)
    assert list(out.columns) == ["offensive_value", "defensive_value", "vaep_value"]
    assert len(out) == len(actions)


def test_nan_xsuccess_propagates(fitted_vaep, fitted_xs, game, actions):
    a = actions.copy()
    a.loc[a.index[0], "start_x"] = np.nan  # -> xSuccess NaN on row 0
    out = fitted_vaep.rate_adjusted(game, a, fitted_xs)
    assert np.isnan(out["vaep_value"].iloc[0])


def test_purity_no_mutation(fitted_vaep, fitted_xs, game, actions):
    snap = actions.copy(deep=True)
    fitted_vaep.rate_adjusted(game, actions, fitted_xs)
    pd.testing.assert_frame_equal(actions, snap)


def test_surgical_flip_scope(fitted_vaep, game, actions):
    # TF61-SPEC-01/07: only the a0 result columns change; predecessors (a1/a2) + goalscore +
    # locations are held from the real feature matrix (ceteris-paribus counterfactual).
    X, X_succ, _X_fail, a0_cols = fitted_vaep._surgical_result_flip(game, actions)
    assert a0_cols and all(c.endswith("_a0") for c in a0_cols)
    for c in X.columns:
        if c in a0_cols:
            continue
        assert X_succ[c].equals(X[c]), f"non-a0 column changed by the surgical flip: {c}"
    a1_result = [c for c in X.columns if c.startswith("result_") and c.endswith("_a1")]
    assert a1_result, "standard VAEP should carry a1 result columns"


def test_frames_threaded_to_all_builds(fitted_vaep, fitted_xs, game, actions, monkeypatch):
    # PLAN-02: frames must be forwarded to EVERY compute_features build (real + both counterfactuals).
    seen = []
    real = fitted_vaep.compute_features

    def spy(g, a, *, frames=None, **kw):
        seen.append(frames)
        return real(g, a, frames=frames, **kw)

    monkeypatch.setattr(fitted_vaep, "compute_features", spy)
    fitted_vaep.rate_adjusted(game, actions, fitted_xs, frames=None)
    assert len(seen) == 3 and all(f is None for f in seen)  # X + X_succ_full + X_fail_full


def test_return_components(fitted_vaep, fitted_xs, game, actions):
    out = fitted_vaep.rate_adjusted(game, actions, fitted_xs, return_components=True)
    assert {"p_scores_success_adj", "p_concedes_fail_adj"} <= set(out.columns)
