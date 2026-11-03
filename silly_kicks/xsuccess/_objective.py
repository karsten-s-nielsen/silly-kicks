"""ruthless CachedObjective for XSuccess XGBoost HPO (TF-61, ADR-009 pattern).

Mirrors ``tracking/_xshot_occurrence_objective.py``: ``prepare()`` builds the trial-invariant
``(X, y, groups)`` once; ``evaluate_patch()`` runs StratifiedGroupKFold-by-match CV with the candidate
hyperparameters and returns held-out **log-loss** (a proper score — no ``scale_pos_weight``, since the
output is a calibrated probability) plus Brier / PR-AUC diagnostics; ``evaluate()`` is the independent
recompute so ``assert_cache_equivalence`` is non-tautological to 1e-9.

NOT imported by ``silly_kicks.xsuccess.__init__`` or the inference path. Requires the ``[train]`` extra
(ruthless-efficiency[optuna] + xgboost). xgboost/sklearn/ruthless are imported function-locally / via
string annotations so this module still passes the event-only import-allowlist gate.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# scale_pos_weight deliberately EXCLUDED (xShot precedent): xSuccess is consumed as a calibrated
# P(success), so we keep log-loss (a proper score) and do NOT reweight. The natural base rate stands.
_SEARCH_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "reg_lambda",
    "reg_alpha",
    "subsample",
    "colsample_bytree",
)


@dataclasses.dataclass
class _Invariant:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def _pinned(params: dict) -> dict:
    base = {"eval_metric": "logloss", "tree_method": "hist", "random_state": 42}
    base.update({k: params[k] for k in _SEARCH_KEYS if k in params})
    return base


def _cv_splits(y, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """Label-stratified, match-grouped folds (groups stringified for cross-provider dtype safety)."""
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(gkf.split(np.zeros(len(y)), y, groups))


def _fold_score(X, y, tr, te, params) -> tuple[float, float | None, float] | None:
    """One fold's (log-loss, PR-AUC or None for a one-class test fold, Brier); None for a one-class train fold."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    if len(np.unique(y[tr])) < 2:
        return None
    clf = xgb.XGBClassifier(**_pinned(params))
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    pr = float(average_precision_score(y[te], p)) if len(np.unique(y[te])) == 2 else None
    return float(log_loss(y[te], p, labels=[0, 1])), pr, float(brier_score_loss(y[te], p))


def _combine(scores) -> tuple[float, float, float]:
    """The fold scores, in fold order -> (mean log-loss, mean PR-AUC, mean Brier); ``inf`` when none scored."""
    kept = [s for s in scores if s is not None]
    if not kept:
        return float("inf"), float("nan"), float("nan")
    prs = [s[1] for s in kept if s[1] is not None]
    return (
        float(np.mean([s[0] for s in kept])),
        float(np.mean(prs)) if prs else float("nan"),
        float(np.mean([s[2] for s in kept])),
    )


def _cv_logloss(X, y, groups, params) -> tuple[float, float, float]:
    """Label-stratified, match-grouped CV -> (mean log-loss, mean PR-AUC, mean Brier)."""
    return _combine([_fold_score(X, y, tr, te, params) for tr, te in _cv_splits(y, groups)])


class XSuccessObjective:
    """CachedObjective: minimize held-out log-loss over XGBoost hyperparameters."""

    patch_params = frozenset(_SEARCH_KEYS)

    def __init__(self, *, fold: dict[str, list[tuple]]) -> None:
        self._fold = fold

    def prepare(self) -> _Invariant:
        """Build the trial-independent ``(X, y, groups)`` invariant once."""
        Xs, ys, gs = [], [], []
        for matches in self._fold.values():
            for X, y, groups in matches:
                Xs.append(np.asarray(X, dtype=float))
                ys.append(np.asarray(y, dtype=int))
                gs.append(np.asarray(groups))
        return _Invariant(np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs))

    def _params(self, candidate) -> dict:
        return {k: candidate.params[k] for k in _SEARCH_KEYS if k in candidate.params}

    def evaluate_patch(self, invariant: _Invariant, candidate, *, fold_map=None) -> dict:
        """Cheap per-trial CV log-loss on the cached invariant (+ diagnostics). ``fold_map(params)``, when given,
        returns the ``_fold_score`` of each ``_cv_splits`` fold in fold order (e.g. computed in parallel by the
        caller); the result equals the serial one."""
        params = self._params(candidate)
        if fold_map is None:
            ll, pr, br = _cv_logloss(invariant.X, invariant.y, invariant.groups, params)
        else:
            ll, pr, br = _combine(fold_map(params))
        return {"logloss": ll, "pr_auc": pr, "brier": br}

    def evaluate(self, candidate) -> dict:
        """Full from-scratch CV (independent recompute; cache-equivalence to 1e-9)."""
        inv = self.prepare()
        ll, pr, br = _cv_logloss(inv.X, inv.y, inv.groups, self._params(candidate))
        return {"logloss": ll, "pr_auc": pr, "brier": br}
