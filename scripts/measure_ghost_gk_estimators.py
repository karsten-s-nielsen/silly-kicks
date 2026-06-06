"""Measure Ghost-GK point-estimate candidates on the pining-81 CV folds (maintainer).

Option A (4.13.0) ship gate (spec 2026-06-04 §3.6). Reports, per fold + aggregate,
euclidean MAE and RMSE for the candidates **stratified by a density-intrinsic
multimodal flag** (mode<->grid-mean gap > 4 m):
  boosted  (predict_mean — the SERVED Option-A estimate; exact HGBR boosted mean) <- primary
  mode     (predict_density argmax — the status-quo served value, the bar to beat)
  grid_mean(density.mean_*)
  b_central(leaf-weighted conditional mean — the rejected Option B, recomputed inline
            on the SAME split as a reference; _central_estimate was removed from the lib)
  geom_median (measurement-only Weiszfeld)

Stratification is the gate's whole point: a good *pooled* MAE can hide a large miss on
the high-leverage multimodal subset (wide crosses, set-pieces) where GK position matters
most — and a single pooled scalar is exactly the aggregate that hid Option B's failure.
boosted's predict_mean is CHEAP (leaf-value traversal, no leaf-match, no KDE) so it needs
no chunking; the density-based candidates (mode/grid_mean/b_central/geom_median) still use
the fft-cic backend + memory chunking.

Loads the trainer's existing feature cache directly (features/labels/groups/providers),
so the folds are IDENTICAL to scripts/train_ghost_gk.py and to the prior Option-B
measurement (same StratifiedGroupKFold(shuffle=True, random_state=42) split + same
eval-subsample seeds), making mode / b_central / boosted directly comparable on one split.

Usage:
  python scripts/measure_ghost_gk_estimators.py \
      --feature-cache <output_dir>/ghost_gk_v1/_feature_cache \
      --cv-folds 5 --eval-subsample 8000 --out metrics_estimators.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES,
    GhostGkModel,
    _leaf_match_weights,
    _vectorized_leaf_indices,
)

_MULTIMODAL_GAP_M = 4.0  # mode<->grid-mean gap proxy for a multimodal conditional density


def _weiszfeld(weights: np.ndarray, pts: np.ndarray, iters: int = 64, eps: float = 1e-6) -> np.ndarray:
    """Measurement-only guarded Weiszfeld geometric median (fixed iters).

    pts: (k, 2); weights: (k,). The eps-floor guards the coincident-point singularity.
    Not production code — only needs ~mm accuracy for an MAE estimate.
    """
    est = np.average(pts, axis=0, weights=weights)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(pts - est, axis=1), eps)
        w = weights / d
        est = (w[:, None] * pts).sum(axis=0) / w.sum()
    return est


def _b_central(weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Leaf-weighted conditional mean (the rejected Option B) — recomputed inline.

    weights: (k,) leaf-match weights for one query; labels: (k, 2). All-zero weights
    fall back to the global label mean (the Option-B convention).
    """
    wsum = weights.sum()
    if wsum <= 0:
        return labels.mean(axis=0)
    return (weights[:, None] * labels).sum(axis=0) / wsum


def _euclid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def _strat_stats(pred: np.ndarray, truth: np.ndarray, mm: np.ndarray) -> dict:
    """MAE + RMSE pooled and split by the multimodal mask `mm` (bool, per-frame)."""
    e = _euclid(pred, truth)

    def _mr(err: np.ndarray) -> dict:
        if len(err) == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
        return {"mae": float(err.mean()), "rmse": float(np.sqrt((err**2).mean())), "n": len(err)}

    return {"pooled": _mr(e), "multimodal": _mr(e[mm]), "unimodal": _mr(e[~mm])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-cache", required=True, type=Path, help="dir with features/labels/groups/providers")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--eval-subsample", type=int, default=8000)
    ap.add_argument("--eval-chunk", type=int, default=1000, help="memory chunk: peak ~(chunk, n_train) leaf-match")
    ap.add_argument("--out", default="metrics_estimators.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # live progress under nohup

    c = args.feature_cache
    features = pd.read_parquet(c / "features.parquet")
    labels = pd.read_parquet(c / "labels.parquet")
    groups = np.load(c / "groups.npy", allow_pickle=True)
    provider_labels = np.load(c / "providers.npy", allow_pickle=True)
    print(f"Loaded {len(features)} samples, {len(set(groups))} games, providers={set(provider_labels)}")

    # Replicate the trainer's split EXACTLY (no refactor): stratify on provider_labels.
    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    rows = []
    for fold, (tr, te) in enumerate(cv.split(features, provider_labels, groups)):
        Xtr, ytr = features.iloc[tr], labels.iloc[tr]
        Xte, yte = features.iloc[te], labels.iloc[te]
        model = GhostGkModel(verbose=0).fit(Xtr, ytr)

        # Held-out subsample for tractability of the density-based candidates.
        rng = np.random.default_rng(fold)
        n = min(args.eval_subsample, len(Xte))
        idx = rng.choice(len(Xte), n, replace=False)
        Xe = Xte.iloc[idx]
        truth = yte.iloc[idx][["gk_x", "gk_y"]].values
        print(f"[fold {fold}] fit done; scoring {n} held-out (fft-cic backend)...", flush=True)

        # SERVED Option-A estimate — cheap leaf-value traversal, scored all at once (no chunking,
        # no leaf-match, no KDE). This is the number the ship gate evaluates.
        boosted = model.predict_mean(Xe)

        # Density-based references: a single (n, n_train) leaf-match matrix is ~45 GB at
        # n=8000 / n_train~710k, so chunk to keep peak ~(CHUNK, n_train). mode/grid-mean use
        # the fast fft-cic backend (~2000x; aggregate MAE ~= vectorized per PR-S81).
        pts = np.column_stack([model._training_gk_x, model._training_gk_y])
        chunk = args.eval_chunk
        mode_l, grid_l, bcen_l, gmed_l, spread_l = [], [], [], [], []
        for s in range(0, n, chunk):
            Xc = Xe.iloc[s : s + chunk]
            dens = model.predict_density(Xc, kde_backend="fft-cic")
            mode_l.extend([[d.mode_x, d.mode_y] for d in dens])
            grid_l.extend([[d.mean_x, d.mean_y] for d in dens])
            spread_l.extend([d.spread for d in dens])
            ql = _vectorized_leaf_indices(model._tree_nodes, Xc[GHOST_GK_FEATURE_NAMES].values.astype(np.float64))
            w = _leaf_match_weights(model._training_leaves, ql)
            for i in range(len(Xc)):
                nz = w[i] > 0
                bcen_l.append(_b_central(w[i][nz], pts[nz]) if nz.any() else pts.mean(0))
                gmed_l.append(_weiszfeld(w[i][nz], pts[nz]) if nz.any() else pts.mean(0))
            del dens, ql, w
            print(f"[fold {fold}] scored {min(s + chunk, n)}/{n}", flush=True)

        mode = np.array(mode_l)
        gridm = np.array(grid_l)
        bcen = np.array(bcen_l)
        gmed = np.array(gmed_l)

        # Density-intrinsic multimodal flag (independent of which estimator we serve).
        mm = _euclid(mode, gridm) > _MULTIMODAL_GAP_M
        rows.append(
            {
                "fold": fold,
                "n_eval": n,
                "multimodal_frac": float(mm.mean()),
                "spread_mean": float(np.mean(spread_l)),
                "boosted": _strat_stats(boosted, truth, mm),
                "mode": _strat_stats(mode, truth, mm),
                "grid_mean": _strat_stats(gridm, truth, mm),
                "b_central": _strat_stats(bcen, truth, mm),
                "geom_median": _strat_stats(gmed, truth, mm),
            }
        )
        print(rows[-1], flush=True)

    # Aggregate MAE per estimator per stratum (mean over all folds).
    ests = ("boosted", "mode", "grid_mean", "b_central", "geom_median")
    strata = ("pooled", "multimodal", "unimodal")
    agg = {est: {s: float(np.mean([r[est][s]["mae"] for r in rows])) for s in strata} for est in ests}
    out = {"folds": rows, "aggregate_mae": agg, "multimodal_gap_m": _MULTIMODAL_GAP_M}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("AGGREGATE MAE (pooled / multimodal / unimodal):")
    for est in ests:
        a = agg[est]
        print(f"  {est:12s} {a['pooled']:.3f} / {a['multimodal']:.3f} / {a['unimodal']:.3f}")
    print(
        "\nSHIP GATE (spec §3.6): HARD-FAIL if boosted MAE >= mode on pooled OR multimodal; "
        "CLEAR-PASS if boosted beats mode by a clear margin on BOTH + multimodal not pathological; "
        "else owner checkpoint with this table."
    )


if __name__ == "__main__":
    main()
