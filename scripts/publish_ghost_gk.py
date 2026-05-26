#!/usr/bin/env python
"""Publish trained Ghost-GK model to HuggingFace Hub.

Usage:
    uv run python scripts/publish_ghost_gk.py \
        --artifact-dir models/ghost_gk_v1 \
        --repo-id karsten-s-nielsen/ghost-gk-v1

    # Verify only (dry run):
    uv run python scripts/publish_ghost_gk.py \
        --artifact-dir models/ghost_gk_v1 \
        --verify-only

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish Ghost-GK model to HF Hub",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Model artifact directory (from train script)")
    parser.add_argument("--repo-id", type=str, default="karsten-s-nielsen/ghost-gk-v1", help="HF Hub repo ID")
    parser.add_argument("--verify-only", action="store_true", help="Dry run: verify integrity without uploading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    # --- 1. Load and verify artifact ---
    print(f"Loading artifact from {args.artifact_dir}")
    model = GhostGkModel.load(args.artifact_dir)
    print("  SHA-256 verification: PASS (automatic in load)")

    # Sanity check: predict on synthetic sample
    rng = np.random.default_rng(99)
    n = 5
    X = pd.DataFrame(
        rng.standard_normal((n, len(GHOST_GK_FEATURE_NAMES))),
        columns=GHOST_GK_FEATURE_NAMES,
    )
    X["phase"] = 0.0
    X["team_in_possession"] = 1.0
    X["ball_in_own_half"] = 0.0
    local_preds = model.predict(X)
    print(f"  Sanity check: predicted {n} samples, shape {local_preds.shape}")

    if args.verify_only:
        print("\n--verify-only: artifact integrity confirmed. Skipping upload.")
        return

    # --- 2. Upload to HF Hub ---
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "ERROR: huggingface_hub not installed. pip install silly-kicks[ghost-gk]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nUploading to {args.repo_id}...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(args.artifact_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("  Upload complete.")

    # --- 3. Verify download ---
    print(f"\nVerifying download from {args.repo_id}...")
    downloaded = GhostGkModel.from_hub(args.repo_id)
    remote_preds = downloaded.predict(X)
    np.testing.assert_allclose(remote_preds, local_preds, atol=1e-10)
    print("  Download + predict verification: PASS")
    print("\nDone.")


if __name__ == "__main__":
    main()
