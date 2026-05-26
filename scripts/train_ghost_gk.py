#!/usr/bin/env python
"""Train Ghost-GK positioning model (TF-18).

Usage:
    uv run python scripts/train_ghost_gk.py \
        --data-dir /path/to/tracking/parquet/ \
        --output-dir models/ \
        --home-teams home_teams.json \
        --subsample-fps 1.0 \
        --n-estimators 500 \
        --max-depth 8 \
        --cv-folds 5

Requires: silly-kicks installed (uv run handles this).

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ghost-GK model")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of tracking parquets")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Where to save model artifact")
    parser.add_argument("--actions-dir", type=Path, default=None, help="Optional: directory of SPADL actions parquets")
    parser.add_argument("--home-teams", type=Path, required=True, help="JSON file: {game_id: home_team_id, ...}")
    parser.add_argument("--subsample-fps", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"Data: {args.data_dir}, subsample_fps={args.subsample_fps}")
    print(f"CV: {args.cv_folds}-fold StratifiedGroupKFold (match+provider)")
    print(f"Output: {args.output_dir}")

    # --- 1. Load tracking data ---
    parquets = sorted(args.data_dir.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: No .parquet files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    frames = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    # Validate schema
    required = {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x",
        "y",
    }
    missing = required - set(frames.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if "vx" not in frames.columns or "vy" not in frames.columns:
        print("ERROR: vx/vy columns missing. Run smooth_frames + derive_velocities first.", file=sys.stderr)
        sys.exit(1)

    n_games = frames["game_id"].nunique()
    n_frames_total = frames[["game_id", "period_id", "frame_id"]].drop_duplicates().shape[0]
    providers = frames["source_provider"].unique().tolist() if "source_provider" in frames.columns else ["unknown"]
    print(f"\nLoaded: {n_games} games, {n_frames_total} frames, providers: {providers}")

    # --- 2. Load actions (optional) ---
    actions: pd.DataFrame | None = None
    if args.actions_dir is not None:
        action_parquets = sorted(args.actions_dir.glob("*.parquet"))
        if action_parquets:
            actions = pd.concat(
                [pd.read_parquet(p) for p in action_parquets],
                ignore_index=True,
            )
            print(f"Loaded: {len(actions)} actions from {len(action_parquets)} files")

    # --- 3. Load home team mapping ---
    with open(args.home_teams) as f:
        home_team_map: dict[str, str] = json.load(f)
    print(f"Home team mapping: {len(home_team_map)} games")

    # --- 4. Per-game feature extraction ---
    from silly_kicks.tracking import prepare_ghost_gk_training_data

    frames_by_game = dict(list(frames.groupby("game_id")))
    actions_by_game = dict(list(actions.groupby("game_id"))) if actions is not None else {}

    all_features: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    all_game_ids: list = []
    all_providers: list[str] = []
    t0 = time.time()

    for game_id in sorted(frames_by_game):
        game_frames = frames_by_game[game_id]
        game_actions = actions_by_game.get(game_id) if actions is not None else None
        home = home_team_map.get(str(game_id))
        if home is None:
            print(f"  SKIP game {game_id}: no home_team_id in mapping")
            continue

        feats, labs = prepare_ghost_gk_training_data(
            game_frames,
            home_team_id=home,
            actions=game_actions,
            subsample_fps=args.subsample_fps,
        )
        if len(feats) > 0:
            all_features.append(feats)
            all_labels.append(labs)
            all_game_ids.extend([game_id] * len(feats))
            prov = (
                str(game_frames["source_provider"].iloc[0]) if "source_provider" in game_frames.columns else "unknown"
            )
            all_providers.extend([prov] * len(feats))

    if not all_features:
        print("ERROR: No training samples extracted.", file=sys.stderr)
        sys.exit(1)

    features = pd.concat(all_features, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    groups = np.array(all_game_ids)
    provider_labels = np.array(all_providers)
    elapsed = time.time() - t0
    print(f"\nExtracted {len(features)} samples from {len(set(all_game_ids))} games in {elapsed:.1f}s")

    # --- 5. StratifiedGroupKFold CV ---
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._ghost_gk import GhostGkModel

    cv = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=42,
    )
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(features, provider_labels, groups)):
        print(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model = GhostGkModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)  # shape (n, 2)

        mae_x = float(np.mean(np.abs(preds[:, 0] - y_test["gk_x"].values)))
        mae_y = float(np.mean(np.abs(preds[:, 1] - y_test["gk_y"].values)))
        mae_euclid = float(
            np.mean(np.sqrt((preds[:, 0] - y_test["gk_x"].values) ** 2 + (preds[:, 1] - y_test["gk_y"].values) ** 2))
        )

        # Per-provider MAE
        test_provs = provider_labels[test_idx]
        per_prov: dict[str, float] = {}
        for prov in np.unique(test_provs):
            mask = test_provs == prov
            per_prov[prov] = float(
                np.mean(
                    np.sqrt(
                        (preds[mask, 0] - y_test["gk_x"].values[mask]) ** 2
                        + (preds[mask, 1] - y_test["gk_y"].values[mask]) ** 2
                    )
                )
            )

        print(f"  MAE x={mae_x:.3f}m  y={mae_y:.3f}m  euclid={mae_euclid:.3f}m")
        print(f"  Per-provider: {per_prov}")
        fold_metrics.append(
            {
                "mae_x": mae_x,
                "mae_y": mae_y,
                "mae_euclidean": mae_euclid,
                "per_provider": per_prov,
            }
        )

    # Aggregate CV
    mae_x_vals = [m["mae_x"] for m in fold_metrics]
    mae_y_vals = [m["mae_y"] for m in fold_metrics]
    mae_e_vals = [m["mae_euclidean"] for m in fold_metrics]
    print("\n=== CV Summary ===")
    print(f"MAE x: {np.mean(mae_x_vals):.3f} +/- {np.std(mae_x_vals):.3f}")
    print(f"MAE y: {np.mean(mae_y_vals):.3f} +/- {np.std(mae_y_vals):.3f}")
    print(f"MAE euclid: {np.mean(mae_e_vals):.3f} +/- {np.std(mae_e_vals):.3f}")

    # --- 6. Feature importance ---
    from sklearn.inspection import permutation_importance

    print("\n--- Feature importance (full model, x-coordinate only) ---")
    print("NOTE: Importance measured for gk_x predictions only.")
    print("Features primarily influencing gk_y may show artificially low importance.")
    final_model = GhostGkModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    final_model.fit(features, labels)

    # Use a simple sklearn wrapper for permutation importance
    from sklearn.base import BaseEstimator, RegressorMixin

    class _SklearnWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
        def __init__(self, m: GhostGkModel | None = None) -> None:
            self.m = m

        def fit(self, X: np.ndarray, y: np.ndarray) -> _SklearnWrapper:
            return self  # already fitted

        def predict(self, X: np.ndarray) -> np.ndarray:
            assert self.m is not None  # noqa: S101
            return self.m.predict(pd.DataFrame(X, columns=features.columns))[:, 0]

    pi = permutation_importance(
        _SklearnWrapper(m=final_model),
        features.values,
        labels["gk_x"].values,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
    )
    importances = sorted(
        zip(features.columns, pi.importances_mean, strict=True),
        key=lambda x: -x[1],
    )
    print("Top 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")

    # --- 7. Save final model ---
    artifact_dir = args.output_dir / "ghost_gk_v1"
    final_model.save(artifact_dir)
    print(f"\nModel saved to {artifact_dir}")

    # Round-trip verify
    loaded = GhostGkModel.load(artifact_dir)
    sample_pred = loaded.predict(features.head(10))
    expected = final_model.predict(features.head(10))
    np.testing.assert_allclose(sample_pred, expected, atol=1e-10)
    print("Round-trip verification: PASS")

    # --- 8. Metrics JSON ---
    # Aggregate per-provider MAE across folds
    all_provs_set: set[str] = set()
    for m in fold_metrics:
        all_provs_set.update(m["per_provider"].keys())
    per_prov_agg: dict[str, float] = {}
    for prov in sorted(all_provs_set):
        vals = [m["per_provider"].get(prov, np.nan) for m in fold_metrics]
        per_prov_agg[prov] = float(np.nanmean(vals))

    artifact_bytes = sum(f.stat().st_size for f in artifact_dir.rglob("*") if f.is_file())

    metrics = {
        "n_games": len(set(all_game_ids)),
        "n_samples": len(features),
        "n_providers": len(set(all_providers)),
        "providers": sorted(set(all_providers)),
        "cv_folds": args.cv_folds,
        "subsample_fps": args.subsample_fps,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
        "cv_mae_x_mean": float(np.mean(mae_x_vals)),
        "cv_mae_x_std": float(np.std(mae_x_vals)),
        "cv_mae_y_mean": float(np.mean(mae_y_vals)),
        "cv_mae_y_std": float(np.std(mae_y_vals)),
        "cv_mae_euclidean_mean": float(np.mean(mae_e_vals)),
        "cv_mae_euclidean_std": float(np.std(mae_e_vals)),
        "per_provider_mae_euclidean": per_prov_agg,
        "acceptance": {
            "overall_mae_lt_2m": float(np.mean(mae_e_vals)) < 2.0,
            "per_provider_mae_lt_3m": all(v < 3.0 for v in per_prov_agg.values()),
            "cross_fold_std_lt_05m": float(np.std(mae_e_vals)) < 0.5,
            "artifact_size_lt_15mb": artifact_bytes < 15_000_000,
        },
        "artifact_size_bytes": artifact_bytes,
    }
    metrics_path = args.output_dir / "ghost_gk_v1" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Acceptance criteria
    print("\n=== Acceptance Criteria ===")
    for key, passed in metrics["acceptance"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
