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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/xshot-occurrence-v1")
    ap.add_argument(
        "--model-card",
        default=None,
        help="Path to the model card (.md). REQUIRED for a real publish; uploaded as README.md. "
        "Making it a required input is deliberate -- a hand-staged card is how it gets dropped.",
    )
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    from silly_kicks.tracking._xshot_occurrence import (
        XSHOT_FEATURE_NAMES_FAITHFUL,
        XSHOT_FEATURE_NAMES_POSITION_ONLY,
        XShotOccurrenceModel,
    )

    art = Path(args.artifact_dir)
    model = XShotOccurrenceModel.load(art)  # SHA-256 verified
    # Feature-set-aware verify sample: a position-only artifact was fit on the SHORTER vector, so a
    # hard-coded faithful (27-col) sample raises an xgboost feature-count mismatch (ADR-070). Pick the
    # column set the artifact actually declares.
    names = XSHOT_FEATURE_NAMES_POSITION_ONLY if model.feature_set == "position_only" else XSHOT_FEATURE_NAMES_FAITHFUL
    sample = pd.DataFrame(np.zeros((3, len(names))), columns=names)
    local_pred = model.predict_proba(sample)
    print(f"Loaded + verified {art} (feature_set={model.feature_set}); sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    if not args.model_card:
        raise SystemExit("--model-card is REQUIRED for a real publish (uploaded as README.md).")

    from _hub_publish import publish_model_with_card
    from huggingface_hub import HfApi

    publish_model_with_card(HfApi(), str(art), args.repo_id, model_card=args.model_card)  # create + card + model-only
    back = XShotOccurrenceModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_proba(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} (weights + model card as README) + round-trip verified.")


if __name__ == "__main__":
    main()
