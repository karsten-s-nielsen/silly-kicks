#!/usr/bin/env python
"""Train the xShotOccurrence (xS) model (TF-16).

Reads {data-dir}/*/frames.parquet + shots.parquet, builds time-windowed labels
+ faithful features, runs a ruthless Optuna study over XGBoost hyperparameters,
fits the final model on the best params, and writes a pickle-free artifact
(model.json + metadata.json + SHA256SUMS).

Requires: silly-kicks[train].  See the TF-16 spec.

Usage:
    python scripts/train_xshot_occurrence.py \
        --data-dir /path/to/games/ \
        --output-dir models/ \
        --n-trials 50 \
        --horizon-seconds 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--horizon-seconds", type=float, default=1.0)
    ap.add_argument("--home-team-id", default=1)
    ap.add_argument("--tolerance-m", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-subsample", type=float, default=None)
    ap.add_argument(
        "--no-attacking-third-only",
        dest="attacking_third_only",
        action="store_false",
        help="Score all in-possession frames, not just the attacking third (default: third only).",
    )
    ap.set_defaults(attacking_third_only=True)
    args = ap.parse_args()

    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, prepare_xshot_training_data
    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    carrier_params = {"tolerance_m": args.tolerance_m, "beta": args.beta, "gamma": args.gamma}
    data_dir = Path(args.data_dir)
    fold: dict[str, list[tuple]] = {"synthetic": []}
    n_games = 0
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        shots = pd.read_parquet(game_dir / "shots.parquet")
        # Shared train/serve-parity entry point (the public API) does the domain
        # filter + labels + features in one place -- no duplicated loop here.
        X, y, groups = prepare_xshot_training_data(
            frames,
            shots,
            home_team_id=args.home_team_id,
            horizon_seconds=args.horizon_seconds,
            attacking_third_only=args.attacking_third_only,
            carrier_params=carrier_params,
            negative_subsample=args.negative_subsample,
            seed=args.seed,
        )
        if len(X) == 0:
            continue
        fold["synthetic"].append((X, y, groups))
        n_games += 1
        print(f"Prepared game {game_dir.name}: {len(X)} frames, {int(y.sum())} positive labels")

    if not fold["synthetic"]:
        raise SystemExit("No usable training data found in --data-dir.")
    print(f"Prepared {n_games} game(s).")

    obj = XShotOccurrenceObjective(fold=fold)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OptunaConfig(
        kind="optuna",
        metric="logloss",
        direction=Direction.MINIMIZE,
        n_trials=args.n_trials,
        sampler="tpe",
        param_space={
            "n_estimators": FloatRange(kind="float", lo=50.0, hi=400.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=6.0),
            "learning_rate": FloatRange(kind="float", lo=0.02, hi=0.4, log=True),
            "min_child_weight": FloatRange(kind="float", lo=1.0, hi=20.0),
            "scale_pos_weight": FloatRange(kind="float", lo=1.0, hi=200.0, log=True),
            "reg_lambda": FloatRange(kind="float", lo=0.0, hi=5.0),
        },
        store=StoreConfig(kind="sqlite", path=str(out_dir / "study.db")),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    # ruthless result API: result.best is an Evaluation(candidate, metrics, ok).
    best = dict(result.best.candidate.params)
    print(f"Best params: {best}")
    print(f"Best metrics: {dict(result.best.metrics)}")

    inv = obj.prepare()
    model = XShotOccurrenceModel(params=best).fit(
        inv.X,
        pd.Series(inv.y),
        carrier_params=carrier_params,
        horizon_seconds=args.horizon_seconds,
    )
    model.save(out_dir / "xshot_occurrence_v1")
    print(f"Wrote artifact to {out_dir / 'xshot_occurrence_v1'}.")


if __name__ == "__main__":
    main()
