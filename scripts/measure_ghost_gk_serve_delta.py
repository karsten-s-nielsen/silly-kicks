#!/usr/bin/env python
"""Measure the buggy-vs-fixed Ghost-GK serve delta (PR-S81 / P3).

The "buggy" serve passed no carrier -> team_in_possession == 0 everywhere.
The "fixed" serve computes the carrier on full frames. This script runs the
internal extractor both ways on a real match, predicts both, and reports the
max/median |ghost_gk_x/y| delta in metres -- the real number for the CHANGELOG
and the lakehouse heads-up (NOT the word "small").

Usage:
    python scripts/measure_ghost_gk_serve_delta.py --frames match.parquet --home-team-id 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking._ball_carrier import infer_ball_carrier
from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features, _resolve_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--home-team-id", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Cap the GK-sample rows scored (KDE-mode predict is per-sample); both serve paths use the "
        "SAME row indices so the per-frame delta is exact. 0 = all.",
    )
    args = ap.parse_args()

    frames = pd.read_parquet(args.frames)
    model = _resolve_model(args.variant)
    carrier = infer_ball_carrier(frames, **model.carrier_params)[
        ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
    ]
    # Extract on FULL frames (cross-frame velocity/goal-mean deps); subsample the feature ROWS after.
    feats_fixed, _meta = _extract_all_ghost_gk_features(frames, home_team_id=args.home_team_id, carrier=carrier)
    feats_bug, _ = _extract_all_ghost_gk_features(frames, home_team_id=args.home_team_id, carrier=None)
    if args.max_samples and len(feats_fixed) > args.max_samples:
        idx = np.sort(np.random.default_rng(0).choice(len(feats_fixed), args.max_samples, replace=False))
        feats_fixed = feats_fixed.iloc[idx].reset_index(drop=True)
        feats_bug = feats_bug.iloc[idx].reset_index(drop=True)

    # fft-cic KDE mode (production serves the mode; ~2000x faster than the default vectorized predict;
    # both paths scored identically so the delta is the true serve-fix shift).
    df = model.predict_density(feats_fixed, kde_backend="fft-cic")
    db = model.predict_density(feats_bug, kde_backend="fft-cic")
    pred_fixed = np.array([[d.mode_x, d.mode_y] for d in df])
    pred_bug = np.array([[d.mode_x, d.mode_y] for d in db])
    dx = np.abs(pred_fixed[:, 0] - pred_bug[:, 0])
    dy = np.abs(pred_fixed[:, 1] - pred_bug[:, 1])
    d = np.sqrt(dx**2 + dy**2)
    n_changed = int((d > 1e-9).sum())
    print(f"samples={len(d)}  changed={n_changed} ({100 * n_changed / max(len(d), 1):.1f}%)")
    print(f"euclidean delta (m): max={d.max():.3f}  median={np.median(d):.3f}  mean={d.mean():.3f}")
    print(f"x delta (m): max={dx.max():.3f}  median={np.median(dx):.3f}")
    print(f"y delta (m): max={dy.max():.3f}  median={np.median(dy):.3f}")


if __name__ == "__main__":
    main()
