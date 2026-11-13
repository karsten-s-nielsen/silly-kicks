"""Trained-artifact integrity predicates (ADR-126) and forecast-skill acceptance gates (ADR-131). Pure; no I/O.

The base-rate band is DERIVED physically per artifact (``n_events * (fps * horizon) / n_rows``
is the maximum possible per-row positive rate for an occurrence label), never a fixed constant:
the committed xShot defect sat at a 22% positive rate that cleared ``pr_auc > base_rate`` but is
13x the physical maximum. ``tol`` is the only slack.

The forecast-skill gates (ADR-131) replace the old ``brier < base*(1-base)``, ``pr_auc > base`` and
``log_loss < ln 2`` bars of the xShot and xCross trainers. They compare the model's pooled out-of-fold forecast with
an honest reference, the training folds' true positive rate, on the same held-out rows: Brier skill, log-loss skill
and the within-fold PR-AUC lift. Each gain must also hold under a match-cluster bootstrap (the one-sided 95% lower
bound above 0). The draw count, seed, level and clip are pre-registered constants: no flag or config changes them.
The bound is approximate: it holds the fold models fixed and covers only the sampling of held-out matches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: Per-family slack over the physical maximum positive rate (the only tunable). Justified in ADR-126:
#: 1.5x leaves headroom for the share of pre-event frames that fall inside the domain filter while
#: still rejecting a >=1.5x inflation.
MODEL_INTEGRITY_TOL: dict[str, float] = {"xshot_occurrence": 1.5, "xcross_attempt": 1.5}


def physical_upper_bound(n_events: float, horizon_s: float, fps: float, n_rows: float, *, tol: float = 1.5) -> float:
    """Maximum plausible per-row positive rate for an occurrence label, with ``tol`` slack.

    Physical maximum = ``n_events * (fps * horizon_s) / n_rows`` (the most positive frames an
    event set can flag, over the extracted rows). For a mixed-fps corpus pass
    ``fps = n_rows / total_duration_s`` so the bound reads ``tol * n_events * horizon_s /
    total_duration_s`` (fps-independent). Returns ``inf`` for a degenerate empty corpus.
    """
    if n_rows <= 0:
        return float("inf")
    return tol * n_events * horizon_s * fps / n_rows


def base_rate_plausible(
    positive_rate: float,
    *,
    n_events: float,
    horizon_s: float,
    fps: float,
    n_rows: float,
    lo: float = 0.0005,
    tol: float = 1.5,
) -> bool:
    """True iff ``lo <= positive_rate <= physical_upper_bound(...)`` (NaN-safe: NaN -> False)."""
    hi = physical_upper_bound(n_events, horizon_s, fps, n_rows, tol=tol)
    return bool(positive_rate == positive_rate and lo <= positive_rate <= hi)


# --- Forecast-skill acceptance gates (ADR-131) ------------------------------------------------------------------

#: Match-bootstrap draws, generator seed, one-sided level and the log-loss clip. Pre-registered (ADR-131 spec 4.3/4.4).
SKILL_BOOTSTRAP_DRAWS = 2000
SKILL_BOOTSTRAP_SEED = 0
SKILL_LEVEL = 0.95
SKILL_LOG_CLIP = 1e-15
#: The lower bound's percentile, ``100 * (1 - SKILL_LEVEL)`` written exactly: that product is 5.000000000000004 in
#: floating point, and the pre-registered rule is ``np.percentile(gain, 5)`` (ADR-131; FSG-IMPL-03).
SKILL_LOWER_PERCENTILE = 5.0


class _PresortedFold:
    """One fold's rows, held in row order (for the weighted rate) and in descending-score order with the
    distinct-threshold ends (for the weighted average precision). The bootstrap sorts each fold ONCE."""

    def __init__(self, rows: np.ndarray, y: np.ndarray, p: np.ndarray) -> None:
        self.rows = rows
        self.y = y[rows]
        order = np.argsort(p[rows], kind="mergesort")[::-1]  # scikit-learn's descending order
        self.desc = rows[order]
        self.y_desc = y[self.desc].astype(np.float64)
        p_desc = p[self.desc]
        self.thr = np.r_[np.flatnonzero(np.diff(p_desc)), len(p_desc) - 1]

    def average_precision(self, w: np.ndarray) -> float:
        """scikit-learn's ``average_precision_score(y, p, sample_weight=w)`` for this fold's rows."""
        w_desc = w[self.desc]
        tps = np.cumsum(self.y_desc * w_desc)[self.thr]
        fps = np.cumsum((1.0 - self.y_desc) * w_desc)[self.thr]
        ps = tps + fps
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(ps != 0, tps / ps, 0.0)
        recall = tps / tps[-1]
        precision = np.r_[precision[::-1], 1.0]
        recall = np.r_[recall[::-1], 0.0]
        return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def weighted_average_precision(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """``average_precision_score(y, p, sample_weight=w)`` (scikit-learn's step integral), from one sort."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return _PresortedFold(np.arange(len(y)), y, p).average_precision(w)


def _within_fold_lift(folds: list[_PresortedFold], w: np.ndarray) -> tuple[float, int, int]:
    """The weighted within-fold PR-AUC lift of one draw (spec 4.3): ``sum_k s_k (AP_k - rate_k) / sum_k s_k`` over the
    usable folds; a fold with no weight or a one-class weighted rate is dropped. Returns ``(lift or NaN,
    n_empty, n_one_class)``."""
    num = den = 0.0
    n_empty = n_one_class = 0
    for f in folds:
        wk = w[f.rows]
        sk = wk.sum()
        if sk == 0:
            n_empty += 1
            continue
        rate = (wk * f.y).sum() / sk
        if not 0 < rate < 1:
            n_one_class += 1
            continue
        num += sk * (f.average_precision(w) - rate)
        den += sk
    return (num / den if den else float("nan")), n_empty, n_one_class


def forecast_skill(
    y: np.ndarray, p: np.ndarray, r: np.ndarray, fold: np.ndarray, groups: np.ndarray, *, nested: bool
) -> dict[str, Any]:
    """The ``skill`` dict of one candidate (ADR-131; spec 4.2, 4.3, 4.5).

    ``y`` are the pooled out-of-fold labels, ``p`` the model's predictions, ``r`` the reference forecast (each row's
    training folds' TRUE positive rate), ``fold`` each row's gate fold and ``groups`` each row's match. The Brier and
    log-loss quantities are pooled over the rows; the PR-AUC lift is within each fold (``AP_k - rate_k``, weighted by
    the fold's rows), so a forecast that is constant within a fold has a lift of exactly 0. The lower bounds come
    from the pinned match-cluster bootstrap.
    """
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    fold = np.asarray(fold)
    bm, bref = (y - p) ** 2, (y - r) ** 2
    pc = np.clip(p, SKILL_LOG_CLIP, 1 - SKILL_LOG_CLIP)
    rc = np.clip(r, SKILL_LOG_CLIP, 1 - SKILL_LOG_CLIP)
    lm = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    lref = -(y * np.log(rc) + (1 - y) * np.log(1 - rc))
    folds = [_PresortedFold(np.flatnonzero(fold == k), y, p) for k in np.unique(fold)]
    point_lift, _, _ = _within_fold_lift(folds, np.ones(len(y)))

    labels, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    n_matches = len(labels)
    rng = np.random.default_rng(SKILL_BOOTSTRAP_SEED)
    brier_gain = np.empty(SKILL_BOOTSTRAP_DRAWS)
    log_loss_gain = np.empty(SKILL_BOOTSTRAP_DRAWS)
    lift = np.empty(SKILL_BOOTSTRAP_DRAWS)
    n_empty = n_one_class = 0
    for b in range(SKILL_BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n_matches, n_matches)
        w = np.bincount(idx, minlength=n_matches)[inv].astype(np.float64)
        brier_gain[b] = (w * (bref - bm)).sum() / w.sum()
        log_loss_gain[b] = (w * (lref - lm)).sum() / w.sum()
        lift[b], e, o = _within_fold_lift(folds, w)
        n_empty += e
        n_one_class += o
    q = SKILL_LOWER_PERCENTILE
    rate = float(y.mean())
    pooled_ap = weighted_average_precision(y, p, np.ones(len(y))) if 0 < rate < 1 else float("nan")
    return {
        "reference": "train_fold_true_rate",
        "n_oof_rows": len(y),
        "n_matches": n_matches,
        "brier_model": float(bm.mean()),
        "brier_ref": float(bref.mean()),
        "brier_skill": float(1 - bm.mean() / bref.mean()),
        "log_loss_model": float(lm.mean()),
        "log_loss_ref": float(lref.mean()),
        "log_loss_skill": float(1 - lm.mean() / lref.mean()),
        "pr_auc_lift": float(point_lift),
        "pr_auc_pooled": pooled_ap,
        "positive_rate_pooled": rate,
        "bootstrap": {
            "unit": "match",
            "draws": SKILL_BOOTSTRAP_DRAWS,
            "seed": SKILL_BOOTSTRAP_SEED,
            "level": SKILL_LEVEL,
            "brier_gain_lb": float(np.percentile(brier_gain, q)),
            "log_loss_gain_lb": float(np.percentile(log_loss_gain, q)),
            "pr_auc_lift_lb": float(np.percentile(lift, q)),
            "frac_brier_gain_le_0": float(np.mean(brier_gain <= 0)),
            "frac_log_loss_gain_le_0": float(np.mean(log_loss_gain <= 0)),
            "frac_pr_auc_lift_le_0": float(np.mean(lift <= 0)),
            "n_empty_fold_draws": n_empty,
            "n_one_class_fold_draws": n_one_class,
        },
        "nested": nested,
    }


def _point_skill(model_loss: np.ndarray, ref_loss: np.ndarray) -> float:
    """``1 - mean(model) / mean(reference)``; NaN when the reference loss is zero or not finite (fail-closed)."""
    ref = float(ref_loss.mean())
    if not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return float(1 - model_loss.mean() / ref)


def provider_skill(
    y: np.ndarray, p: np.ndarray, r: np.ndarray, providers: np.ndarray, groups: np.ndarray, fold: np.ndarray
) -> dict[str, dict[str, Any]]:
    """The per-provider skill of one candidate (ADR-132; corpus widening spec 5.7), keyed by provider identity.

    The same pooled out-of-fold rows as :func:`forecast_skill`. ``r`` is each row's OWN provider's training-fold true
    positive rate (stricter than the pooled reference: recognising the provider alone earns no skill); ``providers``
    are the row-aligned licensing identities; ``groups`` the bare match ids. Per provider: ``n_matches`` (distinct
    ``(provider, group)``), ``n_oof_rows``, ``n_positive``, ``row_share``, the point ``brier_skill`` and
    ``log_loss_skill`` (``SKILL_LOG_CLIP``; NaN on a NaN reference or a zero reference loss) and the within-fold
    ``pr_auc_lift`` over the provider's rows (NaN allowed). No bootstrap: a 7-match provider's cluster bound is too
    wide to be useful (spec 5.7).
    """
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    providers = np.asarray(providers).astype(str)
    groups = np.asarray(groups).astype(str)
    fold = np.asarray(fold)
    out: dict[str, dict[str, Any]] = {}
    for prov in sorted(set(providers.tolist())):
        sel = providers == prov
        ys, ps, rs = y[sel], p[sel], r[sel]
        pc = np.clip(ps, SKILL_LOG_CLIP, 1 - SKILL_LOG_CLIP)
        rc = np.clip(rs, SKILL_LOG_CLIP, 1 - SKILL_LOG_CLIP)
        lm = -(ys * np.log(pc) + (1 - ys) * np.log(1 - pc))
        lref = -(ys * np.log(rc) + (1 - ys) * np.log(1 - rc))
        folds = [_PresortedFold(np.flatnonzero(fold[sel] == k), ys, ps) for k in np.unique(fold[sel])]
        lift, _, _ = _within_fold_lift(folds, np.ones(len(ys)))
        out[prov] = {
            "n_matches": len(set(groups[sel].tolist())),
            "n_oof_rows": int(sel.sum()),
            "n_positive": int(ys.sum()),
            "row_share": float(sel.sum() / len(y)),
            "brier_skill": _point_skill((ys - ps) ** 2, (ys - rs) ** 2),
            "log_loss_skill": _point_skill(lm, lref),
            "pr_auc_lift": float(lift),
        }
    return out


def _gt0(v: Any) -> bool:
    """Strict ``> 0`` that fails a missing or NaN value."""
    return bool(v is not None and v == v and v > 0)


def skill_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    """The ADR-131 acceptance block (spec 4.4) from one candidate's metrics: ``n_usable_folds`` and ``skill``, plus the
    ADR-132 per-provider pair from ``skill["by_provider"]`` (True iff at least one provider has held-out rows and EVERY
    provider's skill is strictly above 0). A missing ``skill`` or ``by_provider`` fails its gates (fail-closed)."""
    s = metrics.get("skill") or {}
    b = s.get("bootstrap") or {}
    by = s.get("by_provider") or {}
    return {
        "enough_usable_folds": metrics.get("n_usable_folds", 0) >= 2,
        "brier_skill_gt_0": _gt0(s.get("brier_skill")),
        "log_loss_skill_gt_0": _gt0(s.get("log_loss_skill")),
        "pr_auc_lift_gt_0": _gt0(s.get("pr_auc_lift")),
        "brier_gain_lb95_gt_0": _gt0(b.get("brier_gain_lb")),
        "log_loss_gain_lb95_gt_0": _gt0(b.get("log_loss_gain_lb")),
        "pr_auc_lift_lb95_gt_0": _gt0(b.get("pr_auc_lift_lb")),
        "provider_brier_skill_gt_0": bool(by) and all(_gt0(v.get("brier_skill")) for v in by.values()),
        "provider_log_loss_skill_gt_0": bool(by) and all(_gt0(v.get("log_loss_skill")) for v in by.values()),
    }
