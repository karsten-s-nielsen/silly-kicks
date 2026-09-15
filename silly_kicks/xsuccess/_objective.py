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


def _cv_logloss(X, y, groups, params) -> tuple[float, float, float]:
    """Label-stratified, match-grouped CV -> (mean log-loss, mean PR-AUC, mean Brier).

    Groups are stringified for cross-provider dtype safety (kloppy str vs GS int game ids).
    """
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(groups).astype(str)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        n_splits = 2
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lls, prs, brs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = xgb.XGBClassifier(**_pinned(params))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
        brs.append(brier_score_loss(y[te], p))
    if not lls:
        return float("inf"), float("nan"), float("nan")
    return (
        float(np.mean(lls)),
        float(np.mean(prs)) if prs else float("nan"),
        float(np.mean(brs)),
    )


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

    def evaluate_patch(self, invariant: _Invariant, candidate) -> dict:
        """Cheap per-trial CV log-loss on the cached invariant (+ diagnostics)."""
        ll, pr, br = _cv_logloss(invariant.X, invariant.y, invariant.groups, self._params(candidate))
        return {"logloss": ll, "pr_auc": pr, "brier": br}

    def evaluate(self, candidate) -> dict:
        """Full from-scratch CV (independent recompute; cache-equivalence to 1e-9)."""
        inv = self.prepare()
        ll, pr, br = _cv_logloss(inv.X, inv.y, inv.groups, self._params(candidate))
        return {"logloss": ll, "pr_auc": pr, "brier": br}
