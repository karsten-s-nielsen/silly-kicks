"""ruthless CachedObjective for xCrossAttempt HPO (ADR-009 pattern; mirror of the xS objective).

prepare(): build the trial-invariant (X, y, groups) once.
evaluate_patch(): fit XGBoost with the candidate hyperparams + StratifiedGroupKFold CV, return
held-out log-loss (+ PR-AUC/Brier diagnostics).
evaluate(): independent monolith (recompute), so assert_cache_equivalence is non-tautological to 1e-9.

NOT imported by silly_kicks/__init__ or by the inference path. Requires the [train] extra
(ruthless-efficiency[optuna] + xgboost).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from ruthless.result import Candidate, Metrics
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking._xcross_attempt import _pinned_params
from silly_kicks.tracking._xshot_occurrence import subsample_negatives

# scale_pos_weight deliberately EXCLUDED (PR-S80 lesson): xCross is consumed as a calibrated
# P(cross), so we keep log-loss (a proper score) and do NOT reweight.
_SEARCH_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "reg_lambda",
)


@dataclasses.dataclass
class _Invariant:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def _cv_logloss(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    *,
    negative_subsample: float | None = None,
    subsample_seed: int = 42,
) -> tuple[float, float, float]:
    """Label-stratified, match-grouped CV -> (mean log-loss, mean PR-AUC, mean Brier).

    StratifiedGroupKFold keeps per-fold positive counts stable under a low base rate.
    ``negative_subsample`` thins negatives in the TRAIN fold only (held-out fold never touched),
    deterministic per fold so the cache-equivalence gate still holds.
    """
    import xgboost as xgb

    # game_id dtype is provider-asymmetric (kloppy str vs Gradient Sports int) -> normalize to str.
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
        if negative_subsample:  # TRAIN fold only; eval fold (te) never subsampled
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=subsample_seed + fold_i)
            if len(np.unique(ytr)) < 2:
                continue
        clf = xgb.XGBClassifier(**_pinned_params(params))
        clf.fit(Xtr.to_numpy(dtype=float), ytr)
        p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
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


class XCrossAttemptObjective:
    """CachedObjective: minimize held-out log-loss over XGBoost hyperparameters."""

    patch_params = frozenset(_SEARCH_KEYS)

    def __init__(
        self,
        *,
        fold: dict[str, list[tuple]],
        negative_subsample: float | None = None,
        subsample_seed: int = 42,
    ) -> None:
        self._fold = fold
        self._negative_subsample = negative_subsample
        self._subsample_seed = subsample_seed

    def prepare(self) -> _Invariant:
        """Build the trial-independent (X, y, groups) invariant once."""
        Xs, ys, gs = [], [], []
        for matches in self._fold.values():
            for X, y, groups in matches:
                Xs.append(X)
                ys.append(np.asarray(y, dtype=int))
                gs.append(np.asarray(groups))
        return _Invariant(
            pd.concat(Xs, ignore_index=True),
            np.concatenate(ys),
            np.concatenate(gs),
        )

    def _params(self, candidate: Candidate) -> dict:
        return {k: candidate.params[k] for k in _SEARCH_KEYS if k in candidate.params}

    def evaluate_patch(self, invariant: _Invariant, candidate: Candidate) -> Metrics:
        """Cheap per-trial CV log-loss on the cached invariant (+ diagnostics)."""
        ll, pr, br = _cv_logloss(
            invariant.X,
            invariant.y,
            invariant.groups,
            self._params(candidate),
            negative_subsample=self._negative_subsample,
            subsample_seed=self._subsample_seed,
        )
        return {"logloss": ll, "pr_auc": pr, "brier": br}

    def evaluate(self, candidate: Candidate) -> Metrics:
        """Full from-scratch CV (independent recompute; H1) -- == evaluate_patch to 1e-9."""
        inv = self.prepare()
        ll, pr, br = _cv_logloss(
            inv.X,
            inv.y,
            inv.groups,
            self._params(candidate),
            negative_subsample=self._negative_subsample,
            subsample_seed=self._subsample_seed,
        )
        return {"logloss": ll, "pr_auc": pr, "brier": br}
