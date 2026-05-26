#!/usr/bin/env python
"""Train Ghost-GK positioning model (TF-18).

Usage:
    uv run python scripts/train_ghost_gk.py \
        --data-dir /path/to/tracking/parquet/ \
        --output-dir models/ \
        --subsample-fps 1.0 \
        --n-estimators 500 \
        --max-depth 8 \
        --cv-folds 5

Requires: silly-kicks installed (uv run handles this).
Training-only dep: skl2onnx (for validation ONNX export).

See docs/superpowers/specs/2026-05-25-tf18-ghost-gk-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ghost-GK model")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
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
    print(f"CV: {args.cv_folds}-fold GroupKFold (match-level)")
    print(f"Output: {args.output_dir}")

    # Training data loading + feature extraction + CV evaluation
    # implemented when real multi-provider tracking data is available.
    # The model class + feature extraction are fully implemented in _ghost_gk.py.
    print("\nTraining script ready for data integration.")


if __name__ == "__main__":
    main()
