#!/usr/bin/env python
"""Publish a trained xCrossAttempt artifact to HuggingFace Hub (TF-17 weights, PR-B).

Verifies SHA-256 + a sanity prediction, uploads the folder, then re-downloads via from_hub and
asserts identical predictions. ``--verify-only`` stops before upload (no network/token needed).

Publishes to ``silly-kicks/xcross-attempt-v1`` (faithful ``sc_extended``) or, via ``--repo-id``, to
``silly-kicks/xcross-attempt-position-only-v1`` (the position-only owner-tier variant, ADR-070). The
verify sample is chosen from the artifact's ``feature_set`` so a position-only fit is not fed a
faithful-width sample.

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

    from silly_kicks.tracking._xcross_attempt import (
        XCROSS_FEATURE_NAMES_FAITHFUL,
        XCROSS_FEATURE_NAMES_POSITION_ONLY,
        XCrossAttemptModel,
    )

    art = Path(args.artifact_dir)
    model = XCrossAttemptModel.load(art)  # SHA-256 verified
    # Feature-set-aware verify sample (ADR-070): a position-only fit has a shorter vector; a hard-coded
    # faithful sample would raise an xgboost feature-count mismatch.
    names = (
        XCROSS_FEATURE_NAMES_POSITION_ONLY if model.feature_set == "position_only" else XCROSS_FEATURE_NAMES_FAITHFUL
    )
    sample = pd.DataFrame(np.zeros((3, len(names))), columns=names)
    local_pred = model.predict_proba(sample)
    print(f"Loaded + verified {art} (feature_set={model.feature_set}); sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from _hub_publish import upload_model_only
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)  # no-op if it exists
    upload_model_only(api, str(art), args.repo_id)  # model-only allowlist + leak guard
    back = XCrossAttemptModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_proba(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified.")


if __name__ == "__main__":
    main()
