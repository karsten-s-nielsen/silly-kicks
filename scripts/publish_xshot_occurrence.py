#!/usr/bin/env python
"""Publish a trained xShotOccurrence artifact to HuggingFace Hub (TF-16 weights, PR-S80).

Verifies SHA-256 + a sanity prediction, uploads the folder, then re-downloads via from_hub and
asserts identical predictions. ``--verify-only`` stops before upload (no network/token needed).

Requires: silly-kicks[xgboost] (+ huggingface_hub for the actual upload).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/xshot-occurrence-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

    art = Path(args.artifact_dir)
    model = XShotOccurrenceModel.load(art)  # SHA-256 verified
    sample = pd.DataFrame(np.zeros((3, len(XSHOT_FEATURE_NAMES_FAITHFUL))), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    local_pred = model.predict_proba(sample)
    print(f"Loaded + verified {art}; sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(art), repo_id=args.repo_id, repo_type="model")
    back = XShotOccurrenceModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_proba(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified.")


if __name__ == "__main__":
    main()
