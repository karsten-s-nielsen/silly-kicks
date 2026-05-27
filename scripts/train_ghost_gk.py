#!/usr/bin/env python
"""Train Ghost-GK positioning model (TF-18).

Usage:
    uv run python scripts/train_ghost_gk.py \
        --data-dir /path/to/tc3_cache/ \
        --output-dir models/ \
        --subsample-fps 1.0 \
        --n-estimators 500 \
        --max-depth 8 \
        --cv-folds 5

Supports two directory layouts (prefers TC3 when both exist):
  - TC3 cache: data-dir/{provider}/{game_id}/frames.parquet
    (auto-reads meta.json siblings for home_team_id)
  - Flat: data-dir/*.parquet

Override home team mapping with --home-teams JSON file.

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
    parser.add_argument(
        "--home-teams",
        type=Path,
        default=None,
        help="JSON file: {game_id: home_team_id, ...} (auto-read from meta.json if omitted)",
    )
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

    # --- 1. Discover tracking parquets ---
    # Support both tc3 cache ({provider}/{game_id}/frames.parquet) and flat (*.parquet) layouts.
    # Prefer tc3 layout (more specific) to avoid picking up stale non-tracking parquets in the root.
    parquets = sorted(args.data_dir.glob("**/frames.parquet"))
    if not parquets:
        parquets = sorted(args.data_dir.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: No .parquet files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Schema validation on first file only (all files share the same pipeline)
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
    import pyarrow.parquet as pq

    probe_cols = set(pq.read_schema(parquets[0]).names)
    missing = required - probe_cols
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if "vx" not in probe_cols or "vy" not in probe_cols:
        print("ERROR: vx/vy columns missing. Run smooth_frames + derive_velocities first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquets)} parquet files in {args.data_dir}")

    # --- 2. Load actions (optional, small — OK to hold in memory) ---
    actions_by_game: dict[str, pd.DataFrame] = {}
    if args.actions_dir is not None:
        action_parquets = sorted(args.actions_dir.glob("*.parquet"))
        if action_parquets:
            actions = pd.concat(
                [pd.read_parquet(p) for p in action_parquets],
                ignore_index=True,
            )
            actions_by_game = dict(list(actions.groupby("game_id")))
            del actions  # Release concatenated copy
            print(f"Loaded actions for {len(actions_by_game)} games")

    # --- 3. Load home team mapping ---
    # Auto-discover from meta.json siblings when --home-teams is not provided.
    home_team_map: dict[str, str] = {}
    if args.home_teams is not None:
        with open(args.home_teams) as f:
            home_team_map = json.load(f)
    else:
        for pq_path in parquets:
            meta_path = pq_path.parent / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                home_val = str(meta["home_team_id"])
                # Key by directory name (works for providers where dir name == game_id)
                home_team_map[pq_path.parent.name] = home_val
                # Also key by actual game_id from the parquet (handles SkillCorner
                # where dir name is match_id but game_id column is a kloppy hash)
                actual_ids = pq.read_table(pq_path, columns=["game_id"]).column("game_id").unique().to_pylist()
                for gid in actual_ids:
                    home_team_map[str(gid)] = home_val
    if not home_team_map:
        print(
            "ERROR: No home team mapping. Provide --home-teams or use tc3 cache layout with meta.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Home team mapping: {len(home_team_map)} games")

    # --- 4. Per-game feature extraction (on-demand frame loading) ---
    # Following lakehouse TC-3 pattern: load frames per-file, extract features,
    # then delete frames immediately.  Only the extracted feature matrix (small)
    # stays in memory — raw frames (large) are never held simultaneously.
    from silly_kicks.tracking import prepare_ghost_gk_training_data

    all_features: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    all_game_ids: list = []
    all_providers: list[str] = []
    n_skipped = 0
    t0 = time.time()

    for pq_idx, pq_path in enumerate(parquets):
        file_frames = pd.read_parquet(pq_path)
        game_ids_in_file = sorted(file_frames["game_id"].unique())
        print(
            f"  [{pq_idx + 1}/{len(parquets)}] {pq_path.name}: {len(game_ids_in_file)} game(s), {len(file_frames)} rows"
        )

        for game_id in game_ids_in_file:
            home = home_team_map.get(str(game_id))
            if home is None:
                print(f"    SKIP game {game_id}: no home_team_id in mapping")
                n_skipped += 1
                continue

            game_frames = file_frames[file_frames["game_id"] == game_id]
            game_actions = actions_by_game.get(game_id) if actions_by_game else None

            feats, labs = prepare_ghost_gk_training_data(
                game_frames,
                home_team_id=home,
                actions=game_actions,
                subsample_fps=args.subsample_fps,
            )
            del game_frames  # Release per-game slice

            if len(feats) > 0:
                all_features.append(feats)
                all_labels.append(labs)
                all_game_ids.extend([game_id] * len(feats))
                prov = (
                    str(file_frames["source_provider"].iloc[0])
                    if "source_provider" in file_frames.columns
                    else "unknown"
                )
                all_providers.extend([prov] * len(feats))

        del file_frames  # Release entire file's frames before loading next

    if not all_features:
        print("ERROR: No training samples extracted.", file=sys.stderr)
        sys.exit(1)

    features = pd.concat(all_features, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    del all_features, all_labels  # Release intermediate lists
    groups = np.array(all_game_ids)
    provider_labels = np.array(all_providers)
    elapsed = time.time() - t0
    print(
        f"\nExtracted {len(features)} samples from {len(set(all_game_ids))} games"
        f" ({n_skipped} skipped) in {elapsed:.1f}s"
    )

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
