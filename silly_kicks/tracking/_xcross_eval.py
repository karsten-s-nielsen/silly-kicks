"""TF-17 xCrossAttempt maintainer-eval helpers (PR-B). PRIVATE, single-repo.

Three shipped-surface GK validations, all REPORTED (never assert "GK wins"):
- gk_block_ablation             -> marginal predictive value (reported context)
- gk_substitution_probe         -> does the surface move when the GK moves? (THE TF-19 gate)
- permutation_importance_report -> CV-held-out feature weights incl. score_differential (context)

Not promoted to ruthless-efficiency (an optimisation/search substrate, not model-evaluation);
promote to a public model-eval home only if/when a 2nd consumer (TF-19 / retro-xS) lands.
See docs/superpowers/specs/2026-06-06-tf17-xcross-pr-b-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Pre-registered TF-19 viability threshold (C1; frozen before the run) -----------------
# GK median |Δ| must be >= 2x the stronger positional control (TF19_PROBE_RATIO) AND
# >= 0.01 (1 pp of P(cross)) in absolute terms (TF19_PROBE_ABS_FLOOR). The constants are
# OWNED by _model_eval (ADR-037) and RE-EXPORTED here: the frozen wrapper surface (and its
# monkeypatch tests) read them as _xcross_eval module attributes.
from silly_kicks.tracking._model_eval import TF19_PROBE_ABS_FLOOR, TF19_PROBE_RATIO
from silly_kicks.tracking._xcross_attempt import (
    XCROSS_FEATURE_NAMES_FAITHFUL,
    XCROSS_GK_BLOCK,
    _pinned_params,
)


def _cv_score(
    X: pd.DataFrame,
    y,
    groups,
    params: dict,
    *,
    seed: int = 42,
    negative_subsample: float | None = None,
) -> dict:
    """SINGLE shared CV scorer for BOTH the acceptance gate (trainer ``_cv_metrics``) AND the
    ablation (M4 -- one implementation, so they cannot drift on seed / negative_subsample). The
    splitter is ALWAYS ``random_state=42`` (the gate's fold construction); ``seed`` drives ONLY
    ``negative_subsample`` (``seed + fold_i``), exactly as the trainer's original ``_cv_metrics``
    did. Returns the per-fold means both callers need (the trainer adds positive_rate /
    base_rate_brier on top).
    """
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)  # gate's fold construction
    prs, brs, lls = [], [], []
    for fold_i, (tr, te) in enumerate(skf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X.iloc[tr], y[tr]
        if negative_subsample:  # TRAIN fold only; the eval fold keeps the true balance
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


def gk_block_ablation(
    X: pd.DataFrame, y, groups, params: dict, *, seed: int = 42, negative_subsample: float | None = None
) -> dict:
    """Reported context: held-out PR-AUC + log-loss WITH vs WITHOUT XCROSS_GK_BLOCK, scored via
    the SAME ``_cv_score`` the acceptance gate uses (so deltas are gate-comparable for any seed/ns)."""
    with_ = _cv_score(
        X[XCROSS_FEATURE_NAMES_FAITHFUL], y, groups, params, seed=seed, negative_subsample=negative_subsample
    )
    base_cols = [c for c in XCROSS_FEATURE_NAMES_FAITHFUL if c not in XCROSS_GK_BLOCK]
    wo_ = _cv_score(X[base_cols], y, groups, params, seed=seed, negative_subsample=negative_subsample)
    return {
        "with_gk_pr_auc": with_["pr_auc"],
        "without_gk_pr_auc": wo_["pr_auc"],
        "with_gk_log_loss": with_["log_loss"],
        "without_gk_log_loss": wo_["log_loss"],
        "delta_pr_auc": with_["pr_auc"] - wo_["pr_auc"],
        "delta_log_loss": with_["log_loss"] - wo_["log_loss"],
        "note": "reported context (marginal predictive value); NOT the tf19_ready gate",
    }


def _tf19_ready(gk: float, nearest_def: float, rand: float) -> bool:
    """Pre-registered numeric gate (C1 + M2): GK median |Δ| must (a) be >= RATIO x the stronger
    positional control AND (b) clear the absolute floor (a big ratio over a negligible band is
    still negligible). M2: a real control band is REQUIRED -- the nearest-defender control must be
    finite and > 0, else there was no placebo comparison at all and we must NOT pass on the
    abs-floor alone (that would re-open the A1 placebo hole at the gate)."""
    if not np.isfinite(gk):
        return False
    nd_ok = np.isfinite(nearest_def) and nearest_def > 0.0
    if not nd_ok:  # M2: no control band -> cannot declare tf19_ready
        return False
    control = max(float(nearest_def), float(rand) if np.isfinite(rand) else 0.0)
    return bool(gk >= TF19_PROBE_RATIO * control and gk >= TF19_PROBE_ABS_FLOOR)


def gk_substitution_probe(
    model,
    frames: pd.DataFrame,
    actions=None,
    *,
    home_team_id,
    n_frames: int = 200,
    n_random: int = 3,
    seed: int = 42,
    advance_m: float = 35.0,
) -> dict:
    """THE TF-19 viability gate (deterministic). For a fixed sample of wide-area frames, measure
    |P(cross|actual) - P(cross|shifted)| for the GK vs a nearest-defender control vs an averaged
    random-outfielder band, over a geometrically-matched displacement panel. Establishes the
    surface is GK-RESPONSIVE (necessary for TF-19); NOT causal GK importance (that is PR-C).

    Since ADR-037 the sampling core lives in ``_model_eval.substitution_deltas`` (arm='xcross',
    mode='panel' -- byte-equivalent, golden-pinned); this wrapper keeps the frozen report."""
    from silly_kicks.tracking._model_eval import substitution_deltas

    deltas = substitution_deltas(
        model,
        frames,
        arm="xcross",
        mode="panel",
        n_frames=n_frames,
        n_random=n_random,
        seed=seed,
        advance_m=advance_m,
    )
    if deltas.empty:
        return {
            "gk_median_abs_delta": float("nan"),
            "nearest_def_median_abs_delta": float("nan"),
            "random_band_median_abs_delta": float("nan"),
            "tf19_ready": False,
            "tf19_reason": "no eligible wide-area frames with a resolvable carrier + GK",
            "n_frames_used": 0,
        }

    # Role-filtered slices preserve the legacy per-frame append order (gk, nearest_def, picks).
    gk_d = deltas.loc[deltas["actor_role"] == "gk", "delta_p"].to_numpy(float)
    nd_d = deltas.loc[deltas["actor_role"] == "nearest_def", "delta_p"].to_numpy(float)
    rb_d = deltas.loc[deltas["actor_role"] == "placebo_out", "delta_p"].to_numpy(float)
    # Every sampled frame emits GK rows (eligibility requires a GK), so this is the legacy len(idx).
    n_frames_used = int(
        deltas.loc[deltas["actor_role"] == "gk", ["game_id", "period_id", "frame_id"]].drop_duplicates().shape[0]
    )

    gk_med = float(np.median(gk_d)) if len(gk_d) else float("nan")
    nd_med = float(np.median(nd_d)) if len(nd_d) else float("nan")
    rb_med = float(np.median(rb_d)) if len(rb_d) else float("nan")
    ready = _tf19_ready(gk_med, nd_med, rb_med)
    # P9 report-only dose diagnostics: median |Δp| at each frozen-panel dose level (the two
    # displacement_m magnitudes math.hypot() emits, exactly 2.0 m and 4.0 m). These are context,
    # NEVER inputs to _tf19_ready (which stays on the pooled gk/control medians above).
    gk_by_dose = deltas.loc[deltas["actor_role"] == "gk"].groupby("displacement_m")["delta_p"].median()
    med_2m = float(gk_by_dose.get(2.0, float("nan")))
    med_4m = float(gk_by_dose.get(4.0, float("nan")))
    # NaN-safe: a zero/NaN 2 m median yields NaN (never a ZeroDivisionError or an inf).
    dose_ratio_4m_over_2m = med_4m / med_2m if (np.isfinite(med_2m) and med_2m != 0.0) else float("nan")
    if not (np.isfinite(nd_med) and nd_med > 0.0):
        reason = "no control band (nearest-defender |Δ| absent/zero) -- cannot compare; False (M2)"
    elif not ready:
        reason = "GK |Δ| did not clear ratio>=2.0 x control AND abs-floor>=0.01"
    else:
        reason = "GK |Δ| cleared both controls and the absolute floor"
    return {
        "gk_median_abs_delta": gk_med,
        "gk_mean_abs_delta": float(np.mean(gk_d)) if len(gk_d) else float("nan"),
        "gk_p90_abs_delta": float(np.percentile(gk_d, 90)) if len(gk_d) else float("nan"),
        "nearest_def_median_abs_delta": nd_med,
        "random_band_median_abs_delta": rb_med,
        # S5 report-only: zero-fraction of each arm (post-B1, the diagnostic that separates an
        # 'unmeasurable' all-zero band from a live-controls 'clean fail'); NOT a _tf19_ready input.
        "gk_zero_fraction": float((gk_d == 0).mean()) if len(gk_d) else float("nan"),
        "random_band_zero_fraction": float((rb_d == 0).mean()) if len(rb_d) else float("nan"),
        # P9 report-only dose diagnostics (see the median computation above).
        "gk_median_abs_delta_at_2m": med_2m,
        "gk_median_abs_delta_at_4m": med_4m,
        "gk_dose_ratio_4m_over_2m": dose_ratio_4m_over_2m,
        "tf19_ready": ready,
        "tf19_reason": reason,
        "tf19_probe_ratio": TF19_PROBE_RATIO,
        "tf19_probe_abs_floor": TF19_PROBE_ABS_FLOOR,
        "n_frames_used": n_frames_used,
        "note": "responsiveness (necessary for TF-19), NOT causal GK primacy (PR-C); "
        "nearest-def control is partially self-limiting (floating identity)",
    }


def permutation_importance_report(
    X: pd.DataFrame, y, groups, params: dict, *, n_repeats: int = 10, seed: int = 42
) -> dict:
    """Reported context: CV-HELD-OUT permutation importance (C2 -- never permuted on the all-data
    shipped model's own training data). For each fold: fit on K-1, permute+score on fold K with
    scoring='average_precision' (B3); average importances across folds. Also report
    score_differential coverage (non-NaN fraction over the full matrix, B2)."""
    import xgboost as xgb
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.utils import Bunch

    X = X[XCROSS_FEATURE_NAMES_FAITHFUL]
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    per_fold: list[np.ndarray] = []
    for tr, te in skf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(y[tr].mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(X.iloc[tr].to_numpy(float), y[tr])
        r = permutation_importance(
            clf, X.iloc[te].to_numpy(float), y[te], scoring="average_precision", n_repeats=n_repeats, random_state=seed
        )
        # A single scoring string returns one Bunch; sklearn returns a dict[str, Bunch] only for
        # multi-metric scoring. Narrow without an `assert` (S101) -- handle both for safety.
        bunch = r if isinstance(r, Bunch) else next(iter(r.values()))
        per_fold.append(bunch.importances_mean)

    if per_fold:
        mean_imp = np.mean(np.vstack(per_fold), axis=0)
        importances = {f: float(v) for f, v in zip(XCROSS_FEATURE_NAMES_FAITHFUL, mean_imp, strict=True)}
    else:
        importances = {f: float("nan") for f in XCROSS_FEATURE_NAMES_FAITHFUL}

    coverage = float(X["score_differential"].notna().mean())
    return {
        "importances": importances,
        "score_differential_importance": importances["score_differential"],
        "score_differential_coverage": coverage,
        "scoring": "average_precision",
        "n_repeats": n_repeats,
        "n_folds_used": len(per_fold),
        "held_out": True,
        "note": "CV-held-out, architecture-representative; NOT measured on the production "
        "weights' own training data. score_differential importance is qualified by coverage.",
    }
