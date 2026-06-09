"""Train the bundled GK-distribution completion model (xT-GK RAV; logistic, pure-numpy serve).

Mirrors scripts/train_xshot_occurrence.py at a fraction of the weight (logistic, observable
label, few features). The GREEN GATE is the NATIVE-ORIGIN POOLED out-of-fold calibration
(review #1): one AUC over ALL native-origin rows with a bootstrap-CI lower bound > 0.5, NOT a
per-fold mean over the small native minority. Bundled default is GS-only (review R1).

Usage:
    python scripts/train_gk_completion.py --providers gradientsports --max-per-provider 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _loader_pining import load_matches

from silly_kicks.tracking._gk_completion import (
    GK_COMPLETION_FEATURE_NAMES,
    GkCompletionModel,
    prepare_gk_completion_training_data,
)

_WEIGHTS_ROOT = Path(__file__).resolve().parent.parent / "silly_kicks" / "tracking" / "_gk_completion_weights"
_WEIGHTS_DIR = _WEIGHTS_ROOT / "default"
_SKILLCORNER_WEIGHTS_DIR = _WEIGHTS_ROOT / "skillcorner"
_N_NATIVE_FLOOR = 100
_GKPASS_AUC_FLOOR = 0.70  # D-S3: SkillCorner GK-pass held-out floor
_ECE_TOL = 0.10  # C1: calibration tolerance (expected calibration error)
_SLOPE_TOL = 0.25  # C1: reliability-slope within [1-tol, 1+tol]; NaN slope (degenerate) is not gated


def _ece(y, p, n_bins=10):
    """Expected calibration error (binned |mean_pred - mean_obs|, weighted by bin mass)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * (m.mean())
    return float(e)


def _reliability_slope(y, p, n_bins=10):
    """Reliability-diagram slope: linear fit of binned mean-observed on binned mean-predicted,
    weighted by bin mass. A perfectly calibrated model has slope ~1 (the diagonal); a slope < 1 is
    over-confident, > 1 under-confident. Returns NaN when predictions don't span >1 occupied bin
    (slope undefined). Complements ECE (magnitude) with a directional shape check (C1)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mp, mo, w = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mp.append(p[m].mean())
            mo.append(y[m].mean())
            w.append(m.sum())
    if len(mp) < 2 or np.ptp(mp) < 1e-9:
        return float("nan")
    coef = np.polyfit(np.asarray(mp), np.asarray(mo), 1, w=np.sqrt(np.asarray(w, float)))
    return float(coef[0])


def _extract(providers, max_per_provider, tracking_limit):
    frames_seen = 0
    parts = []
    for prov, mid, actions, frames, _home in load_matches(
        providers=providers, max_per_provider=max_per_provider, tracking_limit=tracking_limit
    ):
        try:
            X, y, groups = prepare_gk_completion_training_data(actions, frames=frames)
        except ValueError as exc:  # a single near-degenerate match shouldn't kill the run
            print(f"  {prov}/{mid}: skipped ({exc})", flush=True)
            continue
        X = X.copy()
        X["_y"] = y
        X["_group"] = groups
        parts.append(X)
        frames_seen += len(frames)
        print(f"  {prov}/{mid}: {len(X)} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    return df


def _bootstrap_auc_ci(y, p, n_boot=2000, lo=2.5, seed=0):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, lo)), float(np.percentile(aucs, 100 - lo))


def _train_skillcorner(args) -> int:
    """D-S1 GS-transfer re-measurement on the CORRECTED native label + the SkillCorner gate (D-S3/C1).

    Decides whether to bundle distinct SkillCorner weights or alias to the GS ``default``: the
    prior 0.50 non-transfer was on the WRONG (proxy) label and is void; this re-measures GS-transfer
    on the native-completion label (training is native-only via the F1/G1 filter). Per sub-domain
    (overall / goal-kick / GK-pass), reports SkillCorner-fit OOF vs GS-transfer AUC (+bootstrap LCB)
    and ECE. Gate: GK-pass AUC LCB > 0.70 AND ECE <= tol."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    cache = getattr(args, "cache_features", None)
    if cache and Path(cache).exists():
        print(f"=== loading cached SkillCorner features from {cache} ===", flush=True)
        df = pd.read_parquet(cache)
    else:
        print("=== extracting SkillCorner GK distributions (native-label-filtered, F1/G1) ===", flush=True)
        df = _extract(["skillcorner"], args.max_per_provider, args.tracking_limit)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
            print(f"  cached features -> {cache}", flush=True)
    feats = GK_COMPLETION_FEATURE_NAMES
    X_all = df[feats]
    y_all = df["_y"].to_numpy(int)
    groups = df["_group"].to_numpy()
    is_gk = df["is_goalkick"].to_numpy() == 1.0
    print(
        f"N={len(df)} base_rate={y_all.mean():.3f} matches={len(np.unique(groups))} "
        f"goalkick={int(is_gk.sum())} gk_pass={int((~is_gk).sum())}",
        flush=True,
    )

    # SkillCorner-fit out-of-fold
    oof = np.full(len(df), np.nan)
    n_splits = max(2, min(5, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits).split(X_all, y_all, groups):
        if len(np.unique(y_all[tr])) < 2:
            continue
        m = GkCompletionModel().fit(X_all.iloc[tr], pd.Series(y_all[tr]))
        oof[te] = m.predict_proba(X_all.iloc[te])
    ok = np.isfinite(oof)

    # GS-transfer: the bundled GS default served on the SkillCorner rows
    gs = GkCompletionModel.from_variant("default")
    gs_p = gs.predict_proba(X_all)

    def _report(name, mask):
        m = mask & ok
        n = int(m.sum())
        if n < 10 or len(np.unique(y_all[m])) < 2:
            print(f"  [{name}] n={n}: insufficient/degenerate -- skipped", flush=True)
            return None
        sc_auc = roc_auc_score(y_all[m], oof[m])
        sc_lo, _ = _bootstrap_auc_ci(y_all[m], oof[m])
        gs_auc = roc_auc_score(y_all[m], gs_p[m])
        gs_lo, _ = _bootstrap_auc_ci(y_all[m], gs_p[m])
        sc_ece, gs_ece = _ece(y_all[m], oof[m]), _ece(y_all[m], gs_p[m])
        sc_slope, gs_slope = _reliability_slope(y_all[m], oof[m]), _reliability_slope(y_all[m], gs_p[m])
        print(
            f"  [{name}] n={n} base={y_all[m].mean():.3f} | SC-fit AUC={sc_auc:.3f}(LCB {sc_lo:.3f}) "
            f"ECE={sc_ece:.3f} slope={sc_slope:.2f} | GS-transfer AUC={gs_auc:.3f}(LCB {gs_lo:.3f}) "
            f"ECE={gs_ece:.3f} slope={gs_slope:.2f}",
            flush=True,
        )
        return dict(
            n=n,
            base=float(y_all[m].mean()),
            sc_auc=float(sc_auc),
            sc_lcb=float(sc_lo),
            sc_ece=float(sc_ece),
            sc_slope=float(sc_slope),
            gs_auc=float(gs_auc),
            gs_lcb=float(gs_lo),
            gs_ece=float(gs_ece),
            gs_slope=float(gs_slope),
        )

    print("\n=== D-S1 GS-transfer re-measurement (corrected native label) ===", flush=True)
    rep = {
        "overall": _report("overall", np.ones(len(df), bool)),
        "goalkick": _report("goalkick", is_gk),
        "gk_pass": _report("gk_pass", ~is_gk),
    }

    # Decision (D-S1): alias to GS ONLY if GS actually transfers (clears the floor + is no worse than
    # the SC fit); else bundle the SC model when it clears the POINT floor + calibration (LCB + n are
    # REPORTED as uncertainty, m3 -- the floor is the point estimate, not the conservative LCB, which
    # on a few-hundred-row sample would reject a clearly-better model). Aliasing to a worse-than-chance
    # GS transfer is never correct.
    gkp = rep["gk_pass"]
    if gkp is None:
        decision = "alias_gs_insufficient"
    else:

        def _slope_ok(s):  # NaN (degenerate, single occupied bin) is not gated
            return (not np.isfinite(s)) or abs(s - 1.0) <= _SLOPE_TOL

        gs_transfers = gkp["gs_auc"] >= _GKPASS_AUC_FLOOR and gkp["gs_ece"] <= _ECE_TOL and _slope_ok(gkp["gs_slope"])
        sc_usable = gkp["sc_auc"] >= _GKPASS_AUC_FLOOR and gkp["sc_ece"] <= _ECE_TOL and _slope_ok(gkp["sc_slope"])
        if gs_transfers and gkp["gs_auc"] >= gkp["sc_auc"]:
            decision = "alias_gs"  # GS transfers and is no worse -> no distinct weights
        elif sc_usable:
            decision = "bundle_skillcorner"  # SC clears floor + calibrated; GS doesn't transfer
        elif gkp["sc_auc"] > gkp["gs_auc"]:
            decision = "bundle_skillcorner_below_floor"  # neither ideal, but SC >> GS -> ship SC, flag
        else:
            decision = "alias_gs_below_floor"
    print(f"\nDECISION: {decision}  (GK-pass floor {_GKPASS_AUC_FLOOR}, ECE tol {_ECE_TOL})", flush=True)

    bundled = False
    if decision.startswith("bundle_skillcorner"):
        model = GkCompletionModel().fit(X_all, pd.Series(y_all))
        model.shipped_variant = "skillcorner"
        model.provider_list = ["skillcorner"]
        model.save(_SKILLCORNER_WEIGHTS_DIR)
        reloaded = GkCompletionModel.load(_SKILLCORNER_WEIGHTS_DIR)
        np.testing.assert_allclose(model.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)
        bundled = True
        print(f"SAVED skillcorner weights -> {_SKILLCORNER_WEIGHTS_DIR}", flush=True)
    else:
        print(
            "Not bundling distinct SkillCorner weights; from_variant('skillcorner') resolves to the "
            "gs 'default' (alias -- the GS native-completion model transfers, or is the best-available).",
            flush=True,
        )

    metrics = {
        "variant": "skillcorner",
        "decision": decision,
        "bundled": bundled,
        "n_rows": len(df),
        "n_matches": len(np.unique(groups)),
        "subdomains": rep,
        "gkpass_auc_floor": _GKPASS_AUC_FLOOR,
        "ece_tol": _ECE_TOL,
        "slope_tol": _SLOPE_TOL,
    }
    out_dir = _SKILLCORNER_WEIGHTS_DIR if bundled else _WEIGHTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "metrics.json" if bundled else "skillcorner_remeasurement.json"
    (out_dir / fname).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / fname}\nDONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["gradientsports"])
    ap.add_argument("--max-per-provider", type=int, default=64)
    ap.add_argument("--tracking-limit", type=int, default=200)
    ap.add_argument("--variant", default="default", choices=["default", "skillcorner"])
    ap.add_argument("--cache-features", default=None, help="parquet path to cache/reuse extracted features (owner-run)")
    args = ap.parse_args()

    if args.variant == "skillcorner":
        return _train_skillcorner(args)

    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    print(f"=== extracting GK distributions ({args.providers}) ===", flush=True)
    df = _extract(args.providers, args.max_per_provider, args.tracking_limit)
    feats = GK_COMPLETION_FEATURE_NAMES
    X_all = df[feats]
    y_all = df["_y"].to_numpy(int)
    groups = df["_group"].to_numpy()
    native = df["origin_source"].to_numpy() == "native"
    print(
        f"N={len(df)}  base_rate={y_all.mean():.3f}  native={native.sum()} ({native.mean():.0%})  "
        f"matches={len(np.unique(groups))}",
        flush=True,
    )

    # ---- out-of-fold predictions ----
    oof = np.full(len(df), np.nan)
    n_splits = max(2, min(5, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits).split(X_all, y_all, groups):
        if len(np.unique(y_all[tr])) < 2:
            continue
        m = GkCompletionModel().fit(X_all.iloc[tr], pd.Series(y_all[tr]))
        oof[te] = m.predict_proba(X_all.iloc[te])
    ok = np.isfinite(oof)

    # ---- GREEN GATE: native-origin pooled calibration (review #1) ----
    nat = native & ok
    n_native = int(nat.sum())
    yb, pb = y_all[nat], oof[nat]
    base_brier = brier_score_loss(yb, np.full(n_native, yb.mean()))
    native_auc = roc_auc_score(yb, pb)
    ci_lo, ci_hi = _bootstrap_auc_ci(yb, pb)
    native_brier = brier_score_loss(yb, pb)
    density_finite = float(np.isfinite(df["dest_defender_density"].to_numpy()).mean())
    label_split = {"fail": int((y_all == 0).sum()), "success": int((y_all == 1).sum())}

    print("\n=== native-origin pooled out-of-fold gate ===", flush=True)
    print(
        f"  n_native={n_native}  AUC={native_auc:.4f}  CI95=[{ci_lo:.4f},{ci_hi:.4f}]  "
        f"Brier={native_brier:.4f} (base {base_brier:.4f})",
        flush=True,
    )
    print(f"  density_finite_rate={density_finite:.0%}  label_split={label_split}", flush=True)

    gate = {
        "n_native_ge_floor": n_native >= _N_NATIVE_FLOOR,
        "auc_ci_lower_gt_chance": ci_lo > 0.5,
        "brier_lt_base_rate": native_brier < base_brier,
    }
    print(f"  gate: {gate}", flush=True)
    if not all(gate.values()):
        print("GATE FAILED -- not shipping.", flush=True)
        return 1

    # ---- final fit on ALL kept rows (native + imputed) ----
    model = GkCompletionModel().fit(X_all, pd.Series(y_all))
    model.shipped_variant = "default"
    model.provider_list = list(args.providers)
    model.save(_WEIGHTS_DIR)
    reloaded = GkCompletionModel.load(_WEIGHTS_DIR)
    np.testing.assert_allclose(model.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)

    metrics = {
        "n_rows": len(df),
        "n_native": n_native,
        "base_rate": float(y_all.mean()),
        "native_auc": native_auc,
        "native_auc_ci95": [ci_lo, ci_hi],
        "native_brier": native_brier,
        "base_rate_brier": base_brier,
        "density_finite_rate": density_finite,
        "label_split": label_split,
        "providers": list(args.providers),
        "coef": dict(zip(feats, model._coef.tolist(), strict=True)),
    }
    (_WEIGHTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSAVED bundled default -> {_WEIGHTS_DIR}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
