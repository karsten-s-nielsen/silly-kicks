"""Unit tests for the TF-17 xCross maintainer-eval module (PR-B)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xcross_eval as ev
from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCROSS_GK_BLOCK
from tests.tracking._probe_fixtures import planted_model, probe_frames


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    # make gk_r + dist_endline genuinely informative so CV scoring is non-degenerate
    y = ((X["gk_r"] + X["dist_endline"] * 0.5 + rng.normal(scale=0.5, size=n)) > 0).astype(int).to_numpy()
    groups = np.array((["g1"] * (n // 2)) + (["g2"] * (n - n // 2)))
    return X, y, groups


def test_tf19_constants_present_and_typed():
    assert ev.TF19_PROBE_RATIO == 2.0
    assert ev.TF19_PROBE_ABS_FLOOR == 0.01


def test_gk_block_ablation_emits_with_without_and_deltas():
    X, y, groups = _synth()
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}
    out = ev.gk_block_ablation(X, y, groups, params, seed=42)
    for k in (
        "with_gk_pr_auc",
        "without_gk_pr_auc",
        "with_gk_log_loss",
        "without_gk_log_loss",
        "delta_pr_auc",
        "delta_log_loss",
    ):
        assert k in out, k
    assert out["delta_pr_auc"] == pytest.approx(out["with_gk_pr_auc"] - out["without_gk_pr_auc"], abs=1e-9)
    assert len([c for c in X.columns if c not in XCROSS_GK_BLOCK]) == 10


def _fit_probe_model():
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    X, y, _ = _synth()
    return XCrossAttemptModel().fit(X, pd.Series(y))


def test_gk_substitution_probe_emits_gk_and_two_controls():
    m = _fit_probe_model()
    out = ev.gk_substitution_probe(m, probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=42)
    for k in (
        "gk_median_abs_delta",
        "nearest_def_median_abs_delta",
        "random_band_median_abs_delta",
        "tf19_ready",
        "n_frames_used",
    ):
        assert k in out, k
    assert isinstance(out["tf19_ready"], bool)
    assert out["n_frames_used"] >= 1


def test_gk_substitution_probe_is_deterministic():
    m = _fit_probe_model()
    a = ev.gk_substitution_probe(m, probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=7)
    b = ev.gk_substitution_probe(m, probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=7)
    assert a["gk_median_abs_delta"] == b["gk_median_abs_delta"]
    assert a["random_band_median_abs_delta"] == b["random_band_median_abs_delta"]


def test_probe_report_matches_pre_refactor_golden():
    import json
    import pathlib

    golden = json.loads((pathlib.Path(__file__).parent / "goldens" / "xcross_probe_golden.json").read_text())
    report = ev.gk_substitution_probe(planted_model("mixed"), probe_frames(), home_team_id="A")
    for k, v in golden.items():
        if isinstance(v, float):
            assert report[k] == pytest.approx(v, rel=1e-12), k
        else:
            assert report[k] == v, k


def test_tf19_ready_reads_pinned_constants(monkeypatch):
    """C1: the gate uses TF19_PROBE_RATIO/TF19_PROBE_ABS_FLOOR from the module, not an inline literal."""
    assert ev._tf19_ready(gk=0.05, nearest_def=0.02, rand=0.01) is True  # 0.05 >= 2*0.02 and >= 0.01
    assert ev._tf19_ready(gk=0.03, nearest_def=0.02, rand=0.01) is False  # 0.03 < 2*0.02 -> ratio fails
    assert ev._tf19_ready(gk=0.008, nearest_def=0.02, rand=0.01) is False  # below abs floor
    assert ev._tf19_ready(gk=0.05, nearest_def=0.0, rand=0.0) is False  # M2: no control band (nd==0)
    assert ev._tf19_ready(gk=0.05, nearest_def=float("nan"), rand=float("nan")) is False  # M2: no control band
    monkeypatch.setattr(ev, "TF19_PROBE_RATIO", 10.0)
    assert ev._tf19_ready(gk=0.05, nearest_def=0.02, rand=0.01) is False  # respects the constant


def test_probe_report_carries_report_only_diagnostics():
    report = ev.gk_substitution_probe(planted_model("mixed"), probe_frames(), home_team_id="A")
    assert "gk_zero_fraction" in report  # report-only; NOT part of _tf19_ready
    assert "random_band_zero_fraction" in report  # S5: post-B1 THE diagnostic separating
    assert "gk_median_abs_delta_at_2m" in report  # 'unmeasurable' from 'clean fail'
    assert "gk_median_abs_delta_at_4m" in report  # P9: REAL dose diagnostics, not prose
    # the FROZEN verdict fields are untouched -- golden still green


def test_permutation_importance_cv_held_out_and_reports_coverage():
    X, y, groups = _synth()
    X = X.copy()
    X["score_differential"] = 0.0  # fully covered -> coverage 1.0
    X.loc[:9, "score_differential"] = np.nan  # a few missing -> coverage < 1.0
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}
    out = ev.permutation_importance_report(X, y, groups, params, n_repeats=5, seed=42)
    assert "importances" in out and "score_differential" in out["importances"]
    assert "score_differential_importance" in out
    assert out["score_differential_coverage"] == pytest.approx(1 - 10 / len(X), abs=1e-9)
    assert out["held_out"] is True
    assert set(out["importances"]) == set(XCROSS_FEATURE_NAMES_FAITHFUL)
