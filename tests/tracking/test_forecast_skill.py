"""Forecast-skill acceptance gates (ADR-131): `forecast_skill` and `skill_gates`.

Spec: docs/superpowers/specs/2026-11-12-forecast-skill-gates-design.md, sections 4.2-4.5 and section 7 Tests 1-7.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking._artifact_integrity import (
    SKILL_BOOTSTRAP_DRAWS,
    SKILL_BOOTSTRAP_SEED,
    SKILL_LEVEL,
    SKILL_LOG_CLIP,
    SKILL_LOWER_PERCENTILE,
    forecast_skill,
    skill_gates,
    weighted_average_precision,
)

GATE_KEYS = [
    "enough_usable_folds",
    "brier_skill_gt_0",
    "log_loss_skill_gt_0",
    "pr_auc_lift_gt_0",
    "brier_gain_lb95_gt_0",
    "log_loss_gain_lb95_gt_0",
    "pr_auc_lift_lb95_gt_0",
]


def _gate_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """The trainers' gate fold plan: per row, its fold id and its training folds' TRUE positive rate."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = np.full(len(y), -1)
    r = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(skf.split(np.zeros((len(y), 1)), y, groups)):
        fold[te] = k
        r[te] = y[tr].mean()
    return fold, r


def _metrics(skill: dict, n_usable_folds: int = 5) -> dict:
    return {"n_usable_folds": n_usable_folds, "skill": skill}


def test_constant_train_rate_forecast_fails_old_bar_and_is_neutral():
    rng = np.random.default_rng(1)
    match_rates = np.linspace(0.01, 0.25, 10)  # unequal match rates -> unequal fold rates
    groups = np.repeat([f"m{i}" for i in range(10)], 300)
    y = (rng.random(len(groups)) < np.repeat(match_rates, 300)).astype(int)
    fold, r = _gate_folds(y, groups)
    base = y.mean()
    fold_briers = [np.mean((y[fold == k] - r[fold == k]) ** 2) for k in range(5)]
    assert np.mean(fold_briers) >= base * (1 - base)  # the old bar fails a zero-skill forecast

    s = forecast_skill(y, r.copy(), r, fold, groups, nested=True)
    assert s["brier_skill"] == 0.0
    assert s["log_loss_skill"] == 0.0
    assert s["pr_auc_lift"] == 0.0
    g = skill_gates(_metrics(s))
    assert not g["brier_skill_gt_0"]
    assert not g["log_loss_skill_gt_0"]
    assert not g["pr_auc_lift_gt_0"]


def test_informative_forecast_passes_all_seven():
    rng = np.random.default_rng(2)
    groups = np.repeat([f"m{i}" for i in range(60)], 200)
    x = rng.normal(size=len(groups))
    p_true = 1.0 / (1.0 + np.exp(-(-3.0 + 1.5 * x)))
    y = (rng.random(len(groups)) < p_true).astype(int)
    fold, r = _gate_folds(y, groups)
    g = skill_gates(_metrics(forecast_skill(y, p_true, r, fold, groups, nested=True)))
    assert list(g) == GATE_KEYS
    assert all(g.values()), g


def _mixed_effect(n_matches: int) -> dict:
    """Half the matches gain from p (p moves halfway to y), half lose (p moves 15% away from y)."""
    rows = 20
    groups = np.repeat([f"m{i:04d}" for i in range(n_matches)], rows)
    y = np.tile(np.r_[np.ones(4), np.zeros(rows - 4)], n_matches).astype(int)
    r = np.full(len(y), 0.2)
    good = np.repeat(np.arange(n_matches) % 2 == 0, rows)
    p = np.where(good, r + 0.5 * (y - r), r - 0.15 * (y - r))
    fold = np.repeat(np.arange(n_matches) % min(5, n_matches), rows)
    return forecast_skill(y, p, r, fold, groups, nested=True)


def test_bounds_react_to_sample_size():
    assert _mixed_effect(4)["bootstrap"]["brier_gain_lb"] <= 0
    assert _mixed_effect(200)["bootstrap"]["brier_gain_lb"] > 0


def test_weighted_ap_equals_sklearn():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(200):
        n = int(rng.integers(5, 60))
        y = (rng.random(n) < 0.3).astype(int)
        p = np.round(rng.random(n), 2)  # ties
        w = rng.integers(0, 4, n).astype(float)  # zero weights
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            continue
        assert abs(weighted_average_precision(y, p, w) - average_precision_score(y, p, sample_weight=w)) <= 1e-12
        checked += 1
    assert checked > 100


def _reference_lower_bounds(y, p, r, fold, groups):
    """The spec 4.3 reference code, verbatim in its draw sequence (as gold3.py implements it)."""
    bm, bref = (y - p) ** 2, (y - r) ** 2
    pc, rc = np.clip(p, 1e-15, 1 - 1e-15), np.clip(r, 1e-15, 1 - 1e-15)
    lm = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    lref = -(y * np.log(rc) + (1 - y) * np.log(1 - rc))
    labels, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    m_count = len(labels)
    rng = np.random.default_rng(0)
    bg, lg, pl = np.empty(2000), np.empty(2000), np.empty(2000)
    n_empty = n_one = 0
    for b in range(2000):
        idx = rng.integers(0, m_count, m_count)
        w = np.bincount(idx, minlength=m_count)[inv].astype(np.float64)
        bg[b] = (w * (bref - bm)).sum() / w.sum()
        lg[b] = (w * (lref - lm)).sum() / w.sum()
        num = den = 0.0
        for k in np.unique(fold):
            m = fold == k
            wk = w[m]
            sk = wk.sum()
            if sk == 0:
                n_empty += 1
                continue
            rate = (wk * y[m]).sum() / sk
            if not 0 < rate < 1:
                n_one += 1
                continue
            num += sk * (average_precision_score(y[m], p[m], sample_weight=wk) - rate)
            den += sk
        pl[b] = num / den if den else np.nan
    return np.percentile(bg, 5), np.percentile(lg, 5), np.percentile(pl, 5), n_empty, n_one


def test_pinned_sequence_matches_reference():
    rng = np.random.default_rng(4)
    groups = np.repeat([f"g{i:02d}" for i in range(12)], 40)
    x = rng.normal(size=len(groups))
    p = 1.0 / (1.0 + np.exp(-(-1.5 + x)))
    y = (rng.random(len(groups)) < p).astype(int)
    y[groups == "g03"] = 0  # one match without positives -> one-class fold draws
    fold = np.repeat(np.arange(12) % 4, 40)
    r = np.array([y[fold != k].mean() for k in fold])
    s = forecast_skill(y, p, r, fold, groups, nested=False)
    lb_b, lb_l, lb_p, n_empty, n_one = _reference_lower_bounds(y, p, r, fold, groups)
    b = s["bootstrap"]
    # Exact: the Brier and log-loss bounds are the reference's `np.percentile(gain, 5)` bit for bit (FSG-IMPL-03: an
    # earlier q = 100 * (1 - 0.95) = 5.000000000000004 was off by ~2e-18; the cause was the percentile, not memory
    # alignment). The PR-AUC bound differs only by the single-sort AP, which Test 4 pins to 1e-12.
    assert b["brier_gain_lb"] == lb_b
    assert b["log_loss_gain_lb"] == lb_l
    assert abs(b["pr_auc_lift_lb"] - lb_p) <= 1e-12
    assert b["n_empty_fold_draws"] == n_empty
    assert b["n_one_class_fold_draws"] == n_one
    assert n_empty > 0 and n_one > 0  # both kinds occur, so both counts are exercised


def test_byte_equal_json():
    rng = np.random.default_rng(5)
    groups = np.repeat([f"m{i}" for i in range(8)], 50)
    y = (rng.random(len(groups)) < 0.2).astype(int)
    fold, r = _gate_folds(y, groups, n_splits=4)
    p = np.clip(r + 0.1 * (y - r), 0, 1)
    a = json.dumps(forecast_skill(y, p, r, fold, groups, nested=True), sort_keys=True)
    b = json.dumps(forecast_skill(y, p, r, fold, groups, nested=True), sort_keys=True)
    assert a == b


def test_constants_and_shape():
    assert (SKILL_BOOTSTRAP_DRAWS, SKILL_BOOTSTRAP_SEED, SKILL_LEVEL, SKILL_LOG_CLIP) == (2000, 0, 0.95, 1e-15)
    assert SKILL_LOWER_PERCENTILE == 5.0 and abs(100 * (1 - SKILL_LEVEL) - SKILL_LOWER_PERCENTILE) < 1e-9
    rng = np.random.default_rng(6)
    groups = np.repeat([f"m{i}" for i in range(6)], 30)
    y = (rng.random(len(groups)) < 0.3).astype(int)
    fold, r = _gate_folds(y, groups, n_splits=3)
    s = forecast_skill(y, r.copy(), r, fold, groups, nested=True)
    assert s["reference"] == "train_fold_true_rate"
    assert s["nested"] is True
    assert (s["n_oof_rows"], s["n_matches"]) == (180, 6)
    assert {"unit": "match", "draws": 2000, "seed": 0, "level": 0.95}.items() <= s["bootstrap"].items()


def test_nan_and_dropped_folds():
    # A NaN skill fails its gate.
    rng = np.random.default_rng(7)
    groups = np.repeat([f"m{i}" for i in range(10)], 40)
    y = (rng.random(len(groups)) < 0.3).astype(int)
    fold, r = _gate_folds(y, groups)
    s = forecast_skill(y, np.clip(r + 0.3 * (y - r), 0, 1), r, fold, groups, nested=True)
    s["brier_skill"] = float("nan")
    assert not skill_gates(_metrics(s))["brier_skill_gt_0"]

    # One match without positives inside a 3-match fold: dropped one-class fold draws, finite bound.
    groups = np.repeat([f"m{i}" for i in range(9)], 40)
    y = np.tile(np.r_[np.ones(8), np.zeros(32)], 9).astype(int)
    y[groups == "m0"] = 0
    fold = np.repeat(np.arange(9) % 3, 40)  # m0, m3, m6 share fold 0
    r = np.array([y[fold != k].mean() for k in fold])
    p = np.clip(r + 0.4 * (y - r), 0, 1)
    b = forecast_skill(y, p, r, fold, groups, nested=True)["bootstrap"]
    assert b["n_one_class_fold_draws"] > 0
    assert np.isfinite(b["pr_auc_lift_lb"])

    # Two matches, one of them without positives, each its own fold: some draws have no usable fold.
    groups = np.repeat(["a", "b"], 30)
    y = np.r_[np.tile(np.r_[1, 0, 0], 10), np.zeros(30)].astype(int)
    fold = np.repeat([0, 1], 30)
    r = np.full(60, y.mean())
    p = np.clip(r + 0.4 * (y - r), 0, 1)
    s = forecast_skill(y, p, r, fold, groups, nested=True)
    assert np.isnan(s["bootstrap"]["pr_auc_lift_lb"])
    assert not skill_gates(_metrics(s))["pr_auc_lift_lb95_gt_0"]

    # enough_usable_folds follows n_usable_folds >= 2.
    assert not skill_gates(_metrics(s, n_usable_folds=1))["enough_usable_folds"]
    assert skill_gates(_metrics(s, n_usable_folds=2))["enough_usable_folds"]


def test_missing_skill_fails_the_six_skill_gates():
    g = skill_gates({"n_usable_folds": 5})
    assert g["enough_usable_folds"]
    assert not any(v for k, v in g.items() if k != "enough_usable_folds")


@pytest.mark.parametrize("key", GATE_KEYS)
def test_gate_keys_and_order(key):
    assert key in skill_gates({"n_usable_folds": 5})
    assert list(skill_gates({"n_usable_folds": 5})) == GATE_KEYS
