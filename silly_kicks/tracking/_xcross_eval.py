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

from silly_kicks.tracking._xcross_attempt import (
    XCROSS_FEATURE_NAMES_FAITHFUL,
    XCROSS_GK_BLOCK,
    _pinned_params,
)

# --- Pre-registered TF-19 viability threshold (C1; frozen before the run) -----------------
TF19_PROBE_RATIO = 2.0  # GK median |Δ| must be >= 2x the stronger positional control
TF19_PROBE_ABS_FLOOR = 0.01  # AND GK median |Δ| >= 0.01 (1 pp of P(cross)) in absolute terms


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


def _displacement_panel(goal_x: float) -> list[tuple[str, float, float]]:
    """Geometrically-matched (dx, dy) panel applied identically to GK / nearest-def / random
    outfielders so 'same-magnitude' is comparable (A1). 'depth' is signed toward the attacked goal."""
    toward = 1.0 if goal_x >= 105.0 / 2 else -1.0
    return [
        ("lat+2", 0.0, 2.0),
        ("lat-2", 0.0, -2.0),
        ("lat+4", 0.0, 4.0),
        ("lat-4", 0.0, -4.0),
        ("depth+2", toward * 2.0, 0.0),
        ("depth-2", -toward * 2.0, 0.0),
    ]


def _abs_delta_for_player(model, grp, *, row_mask, panel, gk_team_id, goal_x, carrier_pid, sd) -> list[float]:
    """Baseline predict vs each panel-perturbed predict for the single player row(s) in row_mask."""
    from silly_kicks.tracking._xcross_attempt import extract_xcross_features

    mask = np.asarray(row_mask, dtype=bool)
    base_feats = extract_xcross_features(
        grp, gk_team_id=gk_team_id, goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
    )
    base_p = float(model.predict_proba(base_feats)[0])
    deltas = []
    for _name, dx, dy in panel:
        pert = grp.copy()
        pert.loc[mask, "x"] = pert.loc[mask, "x"].to_numpy(float) + dx
        pert.loc[mask, "y"] = pert.loc[mask, "y"].to_numpy(float) + dy
        feats = extract_xcross_features(
            pert, gk_team_id=gk_team_id, goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
        )
        deltas.append(abs(float(model.predict_proba(feats)[0]) - base_p))
    return deltas


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
    surface is GK-RESPONSIVE (necessary for TF-19); NOT causal GK importance (that is PR-C)."""
    from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._xcross_attempt import _build_goal_map, _in_wide_area

    rng = np.random.default_rng(seed)
    cp = dict(getattr(model, "carrier_params", None) or {})
    carrier = infer_ball_carrier(frames, **cp) if cp else infer_ball_carrier(frames)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)

    # Collect eligible (resolvable carrier + GK row + wide-area) frame groups deterministically.
    groups_list = []
    for (gid, pid, _fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        # Defending team(s) = non-ball player rows of the OTHER team. Filter by is_ball (not the
        # string "ball") so a provider/fixture that encodes the ball's team_id differently can't be
        # mistaken for a defending team (it would then have no GK row -> the frame would be dropped).
        non_ball = grp[~grp["is_ball"].astype(bool)]
        # .dropna() guards a non-ball player row with NA team_id (unresolved GS jersey): `pd.NA !=
        # poss_team` is NA -> `if` raises "boolean value of NA is ambiguous" (mirrors prepare/compute).
        defending = [t for t in non_ball["team_id"].dropna().unique() if t != poss_team]
        if not defending:
            continue
        goal_x = goal_map.get((gid, pid, defending[0]))
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
        by = float(ball["y"].iloc[0]) if len(ball) else np.nan
        if not _in_wide_area(bx, by, goal_x, advance_m):
            continue
        cpid = grp["ball_carrier_player_id"].dropna()
        cpid = cpid.iloc[0] if not cpid.empty else None
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == defending[0])
        if cpid is None or not gk_mask.any():
            continue
        groups_list.append((grp.reset_index(drop=True), defending[0], goal_x, cpid))

    if not groups_list:
        return {
            "gk_median_abs_delta": float("nan"),
            "nearest_def_median_abs_delta": float("nan"),
            "random_band_median_abs_delta": float("nan"),
            "tf19_ready": False,
            "tf19_reason": "no eligible wide-area frames with a resolvable carrier + GK",
            "n_frames_used": 0,
        }

    # Deterministic sample of up to n_frames.
    idx = np.arange(len(groups_list))
    if len(idx) > n_frames:
        idx = np.sort(rng.choice(idx, size=n_frames, replace=False))

    gk_d, nd_d, rb_d = [], [], []
    for i in idx:
        grp, gk_team, goal_x, cpid = groups_list[i]
        panel = _displacement_panel(goal_x)
        sd = float("nan")  # probe measures positional sensitivity; score held at NaN
        # GK
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == gk_team)
        gk_d += _abs_delta_for_player(
            model, grp, row_mask=gk_mask, panel=panel, gk_team_id=gk_team, goal_x=goal_x, carrier_pid=cpid, sd=sd
        )
        # Nearest defender to the carrier (control a)
        carr = grp[grp["player_id"].astype(str) == str(cpid)]
        defenders = grp[(grp["team_id"] == gk_team) & ~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]
        if len(carr) and len(defenders):
            cx, cy = float(carr["x"].iloc[0]), float(carr["y"].iloc[0])
            d2 = (defenders["x"].to_numpy(float) - cx) ** 2 + (defenders["y"].to_numpy(float) - cy) ** 2
            nd_id = defenders["player_id"].to_numpy()[int(np.argmin(d2))]
            nd_mask = grp["player_id"].to_numpy() == nd_id
            nd_d += _abs_delta_for_player(
                model, grp, row_mask=nd_mask, panel=panel, gk_team_id=gk_team, goal_x=goal_x, carrier_pid=cpid, sd=sd
            )
        # Averaged random-outfielder band (control b)
        outs = grp[~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]
        out_ids = outs["player_id"].to_numpy()
        if len(out_ids):
            pick = rng.choice(out_ids, size=min(n_random, len(out_ids)), replace=False)
            for rid in pick:
                rb_d += _abs_delta_for_player(
                    model,
                    grp,
                    row_mask=grp["player_id"].to_numpy() == rid,
                    panel=panel,
                    gk_team_id=gk_team,
                    goal_x=goal_x,
                    carrier_pid=cpid,
                    sd=sd,
                )

    gk_med = float(np.median(gk_d)) if gk_d else float("nan")
    nd_med = float(np.median(nd_d)) if nd_d else float("nan")
    rb_med = float(np.median(rb_d)) if rb_d else float("nan")
    ready = _tf19_ready(gk_med, nd_med, rb_med)
    if not (np.isfinite(nd_med) and nd_med > 0.0):
        reason = "no control band (nearest-defender |Δ| absent/zero) -- cannot compare; False (M2)"
    elif not ready:
        reason = "GK |Δ| did not clear ratio>=2.0 x control AND abs-floor>=0.01"
    else:
        reason = "GK |Δ| cleared both controls and the absolute floor"
    return {
        "gk_median_abs_delta": gk_med,
        "gk_mean_abs_delta": float(np.mean(gk_d)) if gk_d else float("nan"),
        "gk_p90_abs_delta": float(np.percentile(gk_d, 90)) if gk_d else float("nan"),
        "nearest_def_median_abs_delta": nd_med,
        "random_band_median_abs_delta": rb_med,
        "tf19_ready": ready,
        "tf19_reason": reason,
        "tf19_probe_ratio": TF19_PROBE_RATIO,
        "tf19_probe_abs_floor": TF19_PROBE_ABS_FLOOR,
        "n_frames_used": len(idx),
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
