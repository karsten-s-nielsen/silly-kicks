"""Unit tests for the TF-17 xCross maintainer-eval module (PR-B)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xcross_eval as ev
from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCROSS_GK_BLOCK


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


def _probe_frames():
    """Two wide-area frames, ball near the left byline, carrier A1, one defender, a GK, ball row.
    Attacked goal at x=105 (GK near x~104)."""
    rows = []
    for fr, t in [(1, 40.0), (2, 40.4)]:
        rows += [
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="A",
                player_id="A1",
                x=96.0,
                y=8.0,
                vx=1.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="A",
                player_id="A2",
                x=99.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="B",
                player_id="B1",
                x=100.0,
                y=20.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="B",
                player_id="Bgk",
                x=104.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=True,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="ball",
                player_id=None,
                x=96.0,
                y=8.0,
                vx=1.0,
                vy=0.0,
                is_ball=True,
                is_goalkeeper=False,
                ball_state="alive",
            ),
        ]
    return pd.DataFrame(rows)


def _fit_probe_model():
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    X, y, _ = _synth()
    return XCrossAttemptModel().fit(X, pd.Series(y))


def test_gk_substitution_probe_emits_gk_and_two_controls():
    m = _fit_probe_model()
    out = ev.gk_substitution_probe(m, _probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=42)
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
    a = ev.gk_substitution_probe(m, _probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=7)
    b = ev.gk_substitution_probe(m, _probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=7)
    assert a["gk_median_abs_delta"] == b["gk_median_abs_delta"]
    assert a["random_band_median_abs_delta"] == b["random_band_median_abs_delta"]


def test_tf19_ready_reads_pinned_constants(monkeypatch):
    """C1: the gate uses TF19_PROBE_RATIO/TF19_PROBE_ABS_FLOOR from the module, not an inline literal."""
    assert ev._tf19_ready(gk=0.05, nearest_def=0.02, rand=0.01) is True  # 0.05 >= 2*0.02 and >= 0.01
    assert ev._tf19_ready(gk=0.03, nearest_def=0.02, rand=0.01) is False  # 0.03 < 2*0.02 -> ratio fails
    assert ev._tf19_ready(gk=0.008, nearest_def=0.02, rand=0.01) is False  # below abs floor
    assert ev._tf19_ready(gk=0.05, nearest_def=0.0, rand=0.0) is False  # M2: no control band (nd==0)
    assert ev._tf19_ready(gk=0.05, nearest_def=float("nan"), rand=float("nan")) is False  # M2: no control band
    monkeypatch.setattr(ev, "TF19_PROBE_RATIO", 10.0)
    assert ev._tf19_ready(gk=0.05, nearest_def=0.02, rand=0.01) is False  # respects the constant


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
