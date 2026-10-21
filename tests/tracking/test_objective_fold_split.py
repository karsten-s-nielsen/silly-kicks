"""ADR-128 ruling 4: the fold-parallel path computes the SAME numbers as the serial objective.

Exact equality, never approx: the fold split moves WHERE a fold is evaluated (an InnerPool child), not
WHAT it computes, and the combination iterates sorted fold keys. Covered: both HPO objectives (xShot and
its xCross twin), the xShot trainer's fixed-params CV (`_cv_metrics`) and the xCross gate scorer
(`_xcross_eval._cv_score`).

The oracle is a VERBATIM copy of each pre-split monolith (below), not the refactored function: after the
split the library's `_cv_logloss` IS the composition, so comparing against it alone would be a tautology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

from ruthless.result import Candidate
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking import _xcross_attempt_objective as XC
from silly_kicks.tracking import _xshot_occurrence_objective as XS
from silly_kicks.tracking._xshot_occurrence import subsample_negatives

OBJECTIVES = [(XS, XS.XShotOccurrenceObjective), (XC, XC.XCrossAttemptObjective)]
P = {"n_estimators": 20.0, "max_depth": 3.0, "learning_rate": 0.1, "min_child_weight": 1.0, "reg_lambda": 1.0}


def _legacy_cv_logloss(pinned, X, y, groups, params, *, negative_subsample=None, subsample_seed=42):  # noqa: N803
    """The pre-ADR-128 `_cv_logloss` loop, verbatim (both objectives shared it; only `_pinned_params` differs)."""
    import xgboost as xgb

    groups = np.asarray(groups).astype(str)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        n_splits = 2
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lls, prs, brs = [], [], []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X.iloc[tr], y[tr]
        if negative_subsample:
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=subsample_seed + fold_i)
            if len(np.unique(ytr)) < 2:
                continue
        clf = xgb.XGBClassifier(**pinned(params))
        clf.fit(Xtr.to_numpy(dtype=float), ytr)
        p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
        brs.append(brier_score_loss(y[te], p))
    if not lls:
        return float("inf"), float("nan"), float("nan")
    return (float(np.mean(lls)), float(np.mean(prs)) if prs else float("nan"), float(np.mean(brs)))


def _legacy_cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42) -> dict:  # noqa: N803
    """The pre-ADR-128 `scripts/train_xshot_occurrence._cv_metrics`, verbatim."""
    import xgboost as xgb

    from silly_kicks.tracking._xshot_occurrence import _pinned_params

    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prs, brs, lls = [], [], []
    for fold_i, (tr, te) in enumerate(skf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X.iloc[tr], y[tr]
        if negative_subsample:
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed + fold_i)
            if len(np.unique(ytr)) < 2:
                continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(ytr.mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(Xtr.to_numpy(float), ytr)
        p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        brs.append(brier_score_loss(y[te], p))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
    base = float(y.mean())
    return {
        "pr_auc": float(np.mean(prs)) if prs else float("nan"),
        "brier": float(np.mean(brs)) if brs else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("inf"),
        "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
        "positive_rate": base,
        "base_rate_brier": base * (1 - base),
        "n_usable_folds": len(lls),
    }


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    y = (X["a"] + rng.normal(scale=0.5, size=400) > 1.2).astype(int).to_numpy()
    groups = np.repeat(np.arange(20), 20).astype(str)
    return X, y, groups


@pytest.mark.parametrize(("mod", "_cls"), OBJECTIVES, ids=["xshot", "xcross"])
@pytest.mark.parametrize("ns", [None, 0.5], ids=["full-balance", "train-subsample"])
def test_split_equals_the_monolith(mod, _cls, ns):
    X, y, g = _data()
    want = _legacy_cv_logloss(mod._pinned_params, X, y, g, P, negative_subsample=ns, subsample_seed=7)
    plan = mod.cv_fold_plan(y, g)
    got = mod.combine_folds(
        [mod.eval_fold(X, y, i, tr, te, P, negative_subsample=ns, subsample_seed=7) for i, tr, te in plan]
    )
    assert got == want  # exact, not approx
    assert mod._cv_logloss(X, y, g, P, negative_subsample=ns, subsample_seed=7) == want
    assert np.isfinite(got[0]), "non-vacuity: the folds really fitted"


@pytest.mark.parametrize(("mod", "cls"), OBJECTIVES, ids=["xshot", "xcross"])
def test_fold_map_objective_equals_the_serial_objective(mod, cls):
    X, y, g = _data(1)
    plan = mod.cv_fold_plan(y, g)
    seen: list[str] = []

    def fold_map(params):
        # deliberately REVERSED completion order; the objective must combine by sorted key
        out = {}
        for i, tr, te in reversed(plan):
            out[f"{i:04d}"] = mod.eval_fold(X, y, i, tr, te, params)
            seen.append(f"{i:04d}")
        return out

    serial = cls(fold={"t": [(X, pd.Series(y), g)]})
    mapped = cls(fold={"t": [(X, pd.Series(y), g)]}, fold_map=fold_map)
    c = Candidate(id="t0", params=P)
    assert mapped.evaluate_patch(mapped.prepare(), c) == serial.evaluate_patch(serial.prepare(), c)
    assert seen and seen != sorted(seen), "non-vacuity: the map really ran, out of key order"


def test_a_skipped_fold_is_dropped_from_the_combination():
    """A single-class train fold returns None (the monolith's `continue`); combine ignores it, and an
    all-skipped plan reports the monolith's (inf, nan, nan)."""
    assert XS.combine_folds([None, (1.0, None, 0.5), (3.0, 0.2, 0.25)]) == (2.0, 0.2, 0.375)
    ll, pr, br = XS.combine_folds([None, None])
    assert ll == float("inf") and np.isnan(pr) and np.isnan(br)


@pytest.mark.parametrize("ns", [None, 0.5], ids=["full-balance", "train-subsample"])
def test_cv_metrics_split_equals_the_monolith(ns):
    """The trainer's fixed-params CV, split into plan / fold / combine for the `cv` stage's fold units."""
    from scripts import train_xshot_occurrence as tr

    X, y, g = _data(2)
    want = _legacy_cv_metrics(X, y, g, P, negative_subsample=ns, seed=3)
    folds = [tr._cv_metrics_fold(X, y, i, a, b, P, ns, 3) for i, a, b in tr._cv_metrics_plan(y, g)]
    got = tr._cv_metrics_combine(folds, float(y.mean()))
    assert list(got) == list(want)  # same keys, same order (metrics.json bytes)
    assert got == want
    assert tr._cv_metrics(X, y, g, P, negative_subsample=ns, seed=3) == want
    assert got["n_usable_folds"] >= 2, "non-vacuity: the folds really fitted"


def _legacy_xcross_cv_score(X, y, groups, params, *, seed=42, negative_subsample=None) -> dict:  # noqa: N803
    """The pre-ADR-128 `silly_kicks.tracking._xcross_eval._cv_score`, verbatim."""
    import xgboost as xgb

    from silly_kicks.tracking._xcross_attempt import _pinned_params

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prs, brs, lls = [], [], []
    for fold_i, (tr, te) in enumerate(skf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X.iloc[tr], y[tr]
        if negative_subsample:
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed + fold_i)
            if len(np.unique(ytr)) < 2:
                continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(ytr.mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(Xtr.to_numpy(float), ytr)
        p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        brs.append(brier_score_loss(y[te], p))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
    return {
        "pr_auc": float(np.mean(prs)) if prs else float("nan"),
        "brier": float(np.mean(brs)) if brs else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("inf"),
        "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
        "n_usable_folds": len(lls),
    }


@pytest.mark.parametrize("ns", [None, 0.5], ids=["full-balance", "train-subsample"])
def test_xcross_cv_score_split_equals_the_monolith(ns):
    """The xCross gate scorer, split into plan / fold / combine for the xcross `cv` stage's fold units."""
    from silly_kicks.tracking import _xcross_eval as ev

    X, y, g = _data(3)
    want = _legacy_xcross_cv_score(X, y, g, P, seed=5, negative_subsample=ns)
    folds = [ev._cv_score_fold(X, y, i, a, b, P, seed=5, negative_subsample=ns) for i, a, b in ev._cv_score_plan(y, g)]
    got = ev._cv_score_combine(folds)
    assert list(got) == list(want)
    assert got == want
    assert ev._cv_score(X, y, g, P, seed=5, negative_subsample=ns) == want
    assert got["n_usable_folds"] >= 2, "non-vacuity: the folds really fitted"
