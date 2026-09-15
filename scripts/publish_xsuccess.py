#!/usr/bin/env python
"""Publish a trained XSuccessModel artifact to HuggingFace Hub (TF-61).

Verifies SHA-256 + a sanity prediction, then uploads via the ONE card-required seam
``_hub_publish.publish_model_with_card`` (ADR-088: create_repo + card-as-README + model-only
allowlist -- a card-less/weightless publish is unrepresentable). ``--verify-only`` stops before
upload (no network/token needed).

Requires: silly-kicks[xgboost] (+ huggingface_hub for the actual upload).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import silly_kicks.spadl.config as cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/xsuccess-v1")
    ap.add_argument(
        "--model-card",
        default=None,
        help="Path to the model card (.md). REQUIRED for a real publish; uploaded as README.md. "
        "Making it a required input is deliberate -- a hand-staged card is how it gets dropped (ADR-088).",
    )
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    from silly_kicks.xsuccess import XSuccessModel

    art = Path(args.artifact_dir)
    model = XSuccessModel.load(art)  # SHA-256 + chirality + feature-contract verified
    sample = pd.DataFrame(
        dict(
            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]],
            bodypart_id=[cfg.bodypart_id["foot"], cfg.bodypart_id["head"]],
            start_x=[30.0, 90.0],
            start_y=[34.0, 30.0],
            time_seconds=[10.0, 2600.0],
            period_id=[1, 2],
        )
    )
    local_pred = model.predict_success(sample)
    print(f"Loaded + verified {art} (feature_set={model.feature_set}); sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    if not args.model_card:
        raise SystemExit("--model-card is REQUIRED for a real publish (uploaded as README.md).")

    from _hub_publish import publish_model_with_card
    from huggingface_hub import HfApi

    publish_model_with_card(HfApi(), str(art), args.repo_id, model_card=args.model_card)
    print(f"Published to {args.repo_id} (weights + model card as README).")


if __name__ == "__main__":
    main()
