"""ruthless CachedObjective for xShotOccurrence HPO (ADR-009 pattern).

prepare(): build the trial-invariant (X, y, groups) per provider once.
evaluate_patch(): fit XGBoost with the candidate hyperparams + GroupKFold CV,
return held-out log-loss (+ PR-AUC/Brier diagnostics).
evaluate(): independent monolith (recompute), so assert_cache_equivalence is
non-tautological to 1e-9.

NOT imported by silly_kicks/__init__ or by the inference path. Requires the
[train] extra (ruthless-efficiency[optuna] + xgboost).

Note (spec S2): unlike TF-24 (where feature extraction is the per-trial cost the
cache eliminates), here feature extraction happens upstream in the trainer, so
prepare() only concats pre-built X and the expensive XGB fit+CV runs per-trial in
BOTH paths. The CachedObjective shape is kept for consistency with the house
pattern + the assert_cache_equivalence correctness gate, not for a speedup.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from ruthless.result import Candidate, Metrics
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking._xshot_occurrence import _pinned_params, subsample_negatives

# scale_pos_weight deliberately EXCLUDED (PR-S80 M2): xS is consumed as a calibrated
# P(shot), so we keep log-loss (a proper score) and do NOT reweight. The natural base
# rate stands (XGBoost base_score). Imbalance handling is calibration, not recall-trading.
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

    StratifiedGroupKFold (PR-S80 M1) keeps per-fold positive counts stable under the ~0.02
    base rate; plain GroupKFold left fold-assignment noise in the cross-fold spread.

    ``negative_subsample`` thins negatives in the **TRAIN fold only** (the held-out fold is never
    touched, so log-loss/PR-AUC/Brier stay on the true balance — PR-S80 M3). Deterministic per fold
    (``subsample_seed + fold_index``) so the cache-equivalence gate (evaluate vs evaluate_patch)
    still holds.
    """
    import xgboost as xgb

    # game_id dtype is provider-asymmetric (kloppy str hashes vs Gradient Sports int); a mixed
    # cross-provider groups array breaks np.unique/StratifiedGroupKFold's sort. Normalize to str
    # for grouping (PR-S80: real-multi-provider fix; the model itself never stores game_id).
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
        if negative_subsample:  # TRAIN fold only; eval fold (te) is never subsampled
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


class XShotOccurrenceObjective:
    """CachedObjective: minimize held-out log-loss over XGBoost hyperparameters.

    Examples
    --------
    >>> # obj = XShotOccurrenceObjective(fold={"synthetic": [(X, y, groups)]})
    >>> # inv = obj.prepare(); obj.evaluate_patch(inv, candidate)["logloss"]
    """

    patch_params = frozenset(_SEARCH_KEYS)

    def __init__(
        self, *, fold: dict[str, list[tuple]], negative_subsample: float | None = None, subsample_seed: int = 42
    ) -> None:
        self._fold = fold
        # TRAIN-fold-only negative subsampling (PR-S80 M3); off by default. Never touches eval folds.
        self._negative_subsample = negative_subsample
        self._subsample_seed = subsample_seed

    def prepare(self) -> _Invariant:
        """Build the trial-independent (X, y, groups) invariant once.

        Examples
        --------
        >>> # inv = obj.prepare()
        """
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
        """Cheap per-trial CV log-loss on the cached invariant (+ diagnostics).

        Examples
        --------
        >>> # obj.evaluate_patch(inv, candidate)["logloss"]
        """
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
        """Full from-scratch CV (independent recompute; H1).

        Examples
        --------
        >>> # obj.evaluate(candidate)["logloss"]  # == evaluate_patch to 1e-9
        """
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
