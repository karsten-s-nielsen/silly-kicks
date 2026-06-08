#!/usr/bin/env python
"""Publish a trained xCrossAttempt artifact to HuggingFace Hub (TF-17 weights, PR-B).

Verifies SHA-256 + a sanity prediction, uploads the folder, then re-downloads via from_hub and
asserts identical predictions. ``--verify-only`` stops before upload (no network/token needed).

NOTE: PR-B shipped the ``public`` variant bundled-in-wheel with NO Hub repo (the paired
public-vs-full test degraded public generalization in all 5 folds -- mirrors xS). This script
exists for the contingency where a future ``full`` variant DOES ship to Hub.

Requires: silly-kicks[xgboost,xcross].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/xcross-attempt-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCrossAttemptModel

    art = Path(args.artifact_dir)
    model = XCrossAttemptModel.load(art)  # SHA-256 verified
    sample = pd.DataFrame(np.zeros((3, len(XCROSS_FEATURE_NAMES_FAITHFUL))), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    local_pred = model.predict_proba(sample)
    print(f"Loaded + verified {art}; sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(art), repo_id=args.repo_id, repo_type="model")
    back = XCrossAttemptModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_proba(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified.")


if __name__ == "__main__":
    main()
