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
    # PR-S81: carrier params (single source -> prepare AND fit; recorded in metadata, R3).
    parser.add_argument(
        "--carrier-beta", type=float, default=None, help="Carrier velocity weight (default: library default)"
    )
    parser.add_argument("--carrier-gamma", type=float, default=None, help="Carrier hysteresis (default: library)")
    parser.add_argument("--carrier-tolerance", type=float, default=None, help="Carrier radius m (default: library)")
    parser.add_argument(
        "--variant",
        choices=["default", "full"],
        default="full",
        help="Which variant this run produces (recorded in metrics/metadata)",
    )
    parser.add_argument(
        "--subsample-cap",
        type=int,
        default=None,
        help="Cap total training samples (default None=all; ~36000 for the bundled 'default')",
    )
    parser.add_argument(
        "--training-platform", type=str, default=None, help="Recorded in metadata (e.g. 'dgx-spark-aarch64')"
    )
    parser.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="Skip the (slow, metrics-only) permutation importance pass. The artifact + CV "
        "metrics + acceptance criteria are unaffected; only the printed feature-importance "
        "ranking is omitted. At full scale (887k) the pass dominates wall-clock.",
    )
    parser.add_argument(
        "--perm-importance-sample",
        type=int,
        default=150000,
        help="Cap the permutation-importance EVAL rows to a seeded subsample (importance is a "
        "statistical estimate; the ranking is stable on a representative sample). The full "
        "corpus (887k) is memory-bandwidth-bound and intractable even with n_jobs. Default "
        "150000 (~4x the default variant's whole training set). Set 0 for all rows.",
    )
    return parser.parse_args()


def main() -> None:
    # Force unbuffered stdout so background tasks show progress immediately
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # PR-S81: resolve ONE carrier cp from CLI (default = library) and pass the SAME dict
    # to both prepare (compute) and fit (record) so metadata records exactly what was used.
    import subprocess

    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    cp = dict(DEFAULT_CARRIER_PARAMS)
    if args.carrier_tolerance is not None:
        cp["tolerance_m"] = args.carrier_tolerance
    if args.carrier_beta is not None:
        cp["beta"] = args.carrier_beta
    if args.carrier_gamma is not None:
        cp["gamma"] = args.carrier_gamma
    print(f"Carrier params (single source, recorded + used): {cp}")

    try:
        training_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        training_commit = None
    print(f"training_commit={training_commit}, training_platform={args.training_platform}")

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

    # --- 4. Per-game feature extraction (with disk cache) ---
    # Cache extracted features to avoid re-reading 78 x 4M-row parquets on re-runs.
    cache_dir = args.output_dir / "ghost_gk_v1" / "_feature_cache"
    cache_feats = cache_dir / "features.parquet"
    cache_labels = cache_dir / "labels.parquet"
    cache_groups = cache_dir / "groups.npy"
    cache_provs = cache_dir / "providers.npy"

    if cache_feats.exists() and cache_labels.exists() and cache_groups.exists() and cache_provs.exists():
        print(f"\nLoading cached features from {cache_dir}")
        t0 = time.time()
        features = pd.read_parquet(cache_feats)
        labels = pd.read_parquet(cache_labels)
        groups = np.load(cache_groups, allow_pickle=True)
        provider_labels = np.load(cache_provs, allow_pickle=True)
        elapsed = time.time() - t0
        print(f"Loaded {len(features)} samples in {elapsed:.1f}s (cached)")
    else:
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
                f"  [{pq_idx + 1}/{len(parquets)}] {pq_path.name}:"
                f" {len(game_ids_in_file)} game(s), {len(file_frames)} rows"
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
                    carrier_params=cp,
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

        # Save cache for subsequent runs
        cache_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_feats)
        labels.to_parquet(cache_labels)
        np.save(cache_groups, groups)
        np.save(cache_provs, provider_labels)
        print(f"Cached features to {cache_dir}")

    # PR-S81: variant axis = sample count -> wheel size. Cap AFTER extraction so the
    # bundled "default" stays small while "full" keeps all in-domain samples.
    if args.subsample_cap is not None and len(features) > args.subsample_cap:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(features), size=args.subsample_cap, replace=False)
        keep.sort()
        features = features.iloc[keep].reset_index(drop=True)
        labels = labels.iloc[keep].reset_index(drop=True)
        groups = groups[keep]
        provider_labels = provider_labels[keep]
        print(f"Subsampled to {len(features)} samples (variant={args.variant}, cap={args.subsample_cap})")

    # --- 5. StratifiedGroupKFold CV ---
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._ghost_gk import GhostGkModel

    cv = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=42,
    )
    fold_metrics: list[dict] = []

    cv_t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(cv.split(features, provider_labels, groups)):
        print(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model = GhostGkModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            verbose=1,
        )
        fit_t0 = time.time()
        model.fit(X_train, y_train, carrier_params=cp)
        fit_elapsed = time.time() - fit_t0
        print(f"  Fit: {fit_elapsed:.1f}s")

        pred_t0 = time.time()
        preds = model.predict_mean(X_test)  # shape (n, 2) — exact boosted HGBR mean (Option A)
        pred_elapsed = time.time() - pred_t0
        print(f"  Predict (boosted mean): {pred_elapsed:.1f}s")

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

        fold_wall = time.time() - cv_t0
        avg_per_fold = fold_wall / (fold + 1)
        remaining = avg_per_fold * (args.cv_folds - fold - 1)
        print(f"  MAE x={mae_x:.3f}m  y={mae_y:.3f}m  euclid={mae_euclid:.3f}m")
        print(f"  Per-provider: {per_prov}")
        print(f"  CV elapsed: {fold_wall:.0f}s, ETA remaining: {remaining:.0f}s")
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
    print("Training final model on all data...")
    final_model = GhostGkModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        verbose=1,
    )
    final_t0 = time.time()
    final_model.fit(features, labels, carrier_params=cp)
    print(f"Final model fit: {time.time() - final_t0:.1f}s")

    # --- BLOCKING parity-on-fresh-fit gate (Option A, ADR-016) ---
    # predict_mean (pickle-free numpy reconstruction) must equal the live sklearn
    # regressors' .predict() to <=1e-6 on the ACTUAL fitted model. The regressors are
    # transient (not serialized), so this can only run on the fresh fit, here. Abort
    # before the expensive permutation importance + publish if parity fails.
    _par_n = min(20000, len(features))
    _par_idx = np.random.default_rng(0).choice(len(features), _par_n, replace=False)
    _par_Xv = features.iloc[_par_idx][features.columns].values
    _par_sk = np.column_stack([final_model._sk_reg_x.predict(_par_Xv), final_model._sk_reg_y.predict(_par_Xv)])
    _par_err = float(np.abs(final_model.predict_mean(features.iloc[_par_idx]) - _par_sk).max())
    _par_ncat = sum(int(t["is_categorical"].sum()) for t in final_model._tree_nodes) + sum(
        int(t["is_categorical"].sum()) for t in final_model._tree_nodes_y
    )
    print(f"PARITY GATE: max|predict_mean - sklearn| = {_par_err:.2e} over {_par_n} rows; n_cat = {_par_ncat}")
    if _par_err > 1e-6 or _par_ncat != 0:
        msg = f"BLOCKING parity gate FAILED (err={_par_err:.2e}, ncat={_par_ncat}); refusing to publish."
        raise RuntimeError(msg)
    print("PARITY GATE: PASS (boosted reconstruction is exact; safe to publish)")

    # Permutation importance is metrics-only (printed, not saved to metrics.json) and
    # dominates wall-clock at full scale (887k x 5 repeats x 26 features). Skippable.
    if args.skip_permutation_importance:
        print("Skipping permutation importance (--skip-permutation-importance).")
    else:
        # Use a simple sklearn wrapper for permutation importance
        from sklearn.base import BaseEstimator, RegressorMixin

        class _SklearnWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
            def __init__(self, m: GhostGkModel | None = None) -> None:
                self.m = m

            def fit(self, X: np.ndarray, y: np.ndarray) -> _SklearnWrapper:
                return self  # already fitted

            def predict(self, X: np.ndarray) -> np.ndarray:
                assert self.m is not None  # noqa: S101
                return self.m.predict_mean(pd.DataFrame(X, columns=features.columns))[:, 0]

        # Subsample the EVAL rows (importance is a statistical estimate; the ranking is stable
        # on a representative sample). The full 887k corpus is memory-bandwidth-bound — each
        # boosted predict_mean scans the full leaf arrays, so 20 workers contend for bandwidth
        # and n_jobs gives no speedup. A 150k subsample is ~8x less traffic so n_jobs parallelizes.
        _pi_cap = args.perm_importance_sample
        if _pi_cap and _pi_cap < len(features):
            _pi_idx = np.random.default_rng(42).choice(len(features), _pi_cap, replace=False)
            _pi_X = features.values[_pi_idx]
            _pi_y = labels["gk_x"].values[_pi_idx]
        else:
            _pi_X, _pi_y = features.values, labels["gk_x"].values
        print(f"Running permutation importance (5 repeats, n_jobs=-1) on {len(_pi_X)} eval rows...")
        pi_t0 = time.time()
        # n_jobs=-1: parallelize the per-(feature, repeat) scorer calls; pair with
        # OMP_NUM_THREADS=1 in the launch env so each loky worker stays single-threaded.
        pi = permutation_importance(
            _SklearnWrapper(m=final_model),
            _pi_X,
            _pi_y,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        print(f"Permutation importance: {time.time() - pi_t0:.1f}s")
        importances = sorted(
            zip(features.columns, pi.importances_mean, strict=True),
            key=lambda x: -x[1],
        )
        print("Top 10 features:")
        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.4f}")

    # --- 7. Save final model ---
    final_model.training_commit = training_commit
    final_model.training_platform = args.training_platform
    artifact_dir = args.output_dir / "ghost_gk_v1"
    final_model.save(artifact_dir)
    print(f"\nModel saved to {artifact_dir}")

    # Round-trip verify (compare serialized weights, not KDE predictions —
    # predict() through KDE is intractable at training scale)
    loaded = GhostGkModel.load(artifact_dir)
    for attr in ("_tree_nodes", "_training_gk_x", "_training_gk_y", "_training_leaves"):
        orig = getattr(final_model, attr)
        back = getattr(loaded, attr)
        if isinstance(orig, list):
            for i, (a, b) in enumerate(zip(orig, back, strict=True)):
                np.testing.assert_array_equal(a, b, err_msg=f"{attr}[{i}]")
        else:
            np.testing.assert_array_equal(orig, back, err_msg=attr)
    assert loaded.carrier_params == cp, f"carrier_params drift: {loaded.carrier_params} != {cp}"  # noqa: S101
    print(f"Round-trip verification: PASS (R3 carrier_params={loaded.carrier_params})")

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

    # Derive game/provider counts from groups/provider_labels (always defined in BOTH
    # the fresh-extract and cache-load branches, and subsample-cap-aware) -- all_game_ids/
    # all_providers exist only on the fresh-extract path (PR-S81).
    metrics = {
        "n_games": len(set(groups.tolist())),
        "n_samples": len(features),
        "n_providers": len(set(provider_labels.tolist())),
        "providers": sorted({str(p) for p in provider_labels.tolist()}),
        "cv_folds": args.cv_folds,
        "subsample_fps": args.subsample_fps,
        "variant": args.variant,
        "carrier_params": cp,
        "training_commit": training_commit,
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
