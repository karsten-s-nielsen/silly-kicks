#!/usr/bin/env python
"""Apples-to-apples non-regression gate for the Ghost-GK re-fit (PR-S81 / P2/N4).

Evaluates BOTH the re-fit model and the incumbent model on the SAME held-out
folds (the re-fit's CV split), so the MAE delta isolates "different model" from
"different data". Prints per-fold + aggregate MAE for both, and the gate verdict.

N4: keeping the incumbent is AVAILABILITY-safe, not correctness-safe. If the gate
rejects the re-fit, this script prints a KEEP INCUMBENT verdict; the operator must
record why and file a bronze-scale-refresh follow-up rather than declaring staleness
resolved.

Usage:
    python scripts/validate_ghost_gk_refit.py \
        --features /path/_feature_cache/features.parquet \
        --labels   /path/_feature_cache/labels.parquet \
        --groups   /path/_feature_cache/groups.npy \
        --providers /path/_feature_cache/providers.npy \
        --incumbent-variant default \
        --epsilon 0.02 --cv-folds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking._ghost_gk import GhostGkModel


def _euclid_mae(model: GhostGkModel, X: pd.DataFrame, y: pd.DataFrame) -> float:
    # KDE mode via the fft-cic backend (~2000x faster than the default vectorized per-sample
    # 60x64 grid; mode is raw-grid-tight, negligible aggregate-MAE diff). load()ed incumbents
    # lack predict_mean, so both models are scored the SAME (mode) way -> apples-to-apples.
    densities = model.predict_density(X, kde_backend="fft-cic")
    px = np.array([d.mode_x for d in densities])
    py = np.array([d.mode_y for d in densities])
    return float(np.mean(np.sqrt((px - y["gk_x"].values) ** 2 + (py - y["gk_y"].values) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--providers", type=Path, required=True)
    ap.add_argument(
        "--incumbent-variants",
        default="default,full",
        help="Comma-separated incumbent variants to gate against in ONE pass (amortizes the per-fold re-fit).",
    )
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument(
        "--eval-cap",
        type=int,
        default=4000,
        help="Subsample each test fold to this many rows for KDE-mode eval (predict is a per-sample "
        "60x64 grid; full folds are ~178k rows). Both models gated on the SAME capped set. 0 = no cap.",
    )
    ap.add_argument(
        "--train-cap",
        type=int,
        default=0,
        help="Subsample each train fold to this many rows before re-fit (0 = full). Use ~29000 to gate the "
        "wheel-bundled 'default' variant fairly (it ships a 36k-sample model); leave 0 for the 'full' variant.",
    )
    args = ap.parse_args()

    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    groups = np.load(args.groups, allow_pickle=True)
    provs = np.load(args.providers, allow_pickle=True)
    variants = [v.strip() for v in args.incumbent_variants.split(",") if v.strip()]
    incumbents = {v: GhostGkModel.from_variant(v) for v in variants}

    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    refit_maes: list[float] = []
    inc_maes: dict[str, list[float]] = {v: [] for v in variants}
    for fold, (tr, te) in enumerate(cv.split(features, provs, groups)):
        # Cap the held-out for KDE-mode eval (deterministic per-fold seed); all models
        # scored on the SAME capped set -> apples-to-apples, just on a manageable subset.
        if args.eval_cap and len(te) > args.eval_cap:
            te = np.sort(np.random.default_rng(1000 + fold).choice(te, size=args.eval_cap, replace=False))
        Xte, yte = features.iloc[te], labels.iloc[te]
        # optionally cap the train-fold to gate a specific variant's sample budget fairly
        tr_use = tr
        if args.train_cap and len(tr) > args.train_cap:
            tr_use = np.sort(np.random.default_rng(2000 + fold).choice(tr, size=args.train_cap, replace=False))
        # re-fit model for this fold (fit on train-fold so the test-fold is a true held-out
        # for the re-fit; each incumbent gets the same test-fold -- L3 overlap caveat applies).
        rf = GhostGkModel(n_estimators=args.n_estimators, max_depth=args.max_depth)
        rf.fit(features.iloc[tr_use], labels.iloc[tr_use])
        rf_mae = _euclid_mae(rf, Xte, yte)
        refit_maes.append(rf_mae)
        line = f"fold {fold + 1}: refit={rf_mae:.3f}m"
        for v in variants:
            im = _euclid_mae(incumbents[v], Xte, yte)
            inc_maes[v].append(im)
            line += f"  {v}={im:.3f}m (d={rf_mae - im:+.3f})"
        print(line)

    rf_mean = float(np.mean(refit_maes))
    print(f"\nAGGREGATE refit={rf_mean:.3f}m (KDE-mode, eval_cap={args.eval_cap})  eps={args.epsilon}")
    any_keep = False
    for v in variants:
        inc_mean = float(np.mean(inc_maes[v]))
        verdict = "SHIP REFIT" if rf_mean <= inc_mean + args.epsilon else "KEEP INCUMBENT"
        any_keep = any_keep or verdict == "KEEP INCUMBENT"
        print(f"  vs incumbent {v}: incumbent={inc_mean:.3f}m  ->  VERDICT: {verdict}")
    print("L3 caveat: if held-out overlaps an incumbent's TRAINING corpus, that incumbent MAE is optimistic.")
    if any_keep:
        print(
            "N4: KEEP is availability-safe only. Record why + file a bronze-scale refresh follow-up;"
            " do NOT declare staleness resolved."
        )


if __name__ == "__main__":
    main()
