#!/usr/bin/env python
"""Publish a trained ghost-outfield artifact to HuggingFace Hub (TF-60 PR5).

Mirrors ``publish_ghost_gk.py``: verify SHA-256 + chirality + feature contract on load, upload the
MODEL FILES ONLY (``_hub_publish.upload_model_only``'s allowlist leak-guard -- never the whole folder,
the 4.94.0 raw-shard-leak lesson), then re-download via ``from_hub`` and assert identical predictions.
``--verify-only`` stops before upload (no network, no token).

Takes ``--artifact-dir`` rather than a bundled-weights path, deliberately: a ``"_weights"`` string
literal in CODE would make this a derived-artifact driver
(``tests/scripts/test_provenance_wiring.py``), which it is not -- it publishes an artifact someone
else produced. The ghost-GK / xShot / xCross publishers take the same argument for the same reason.

Requires: silly-kicks[train] (huggingface_hub).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Publish a ghost-outfield artifact to HuggingFace Hub.")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/ghost-outfield-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    from silly_kicks.tracking._ghost_outfield import GHOST_OUTFIELD_FEATURE_NAMES, GhostOutfieldModel

    art = Path(args.artifact_dir)

    # REFUSE a contract-less artifact rather than publishing one (mirrors the ghost-GK publisher: a
    # pre-contract artifact re-breaks `from_hub` for consumers that escalate MissingFeatureContractWarning).
    meta = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
    contract = meta.get("feature_contract")
    if not contract:
        raise SystemExit(
            f"{art}/metadata.json carries no `feature_contract`. Publishing it would re-break "
            f"`from_hub` for every consumer that escalates MissingFeatureContractWarning."
        )

    model = GhostOutfieldModel.load(art)  # SHA-256 + chirality + feature contract all verified here
    # Sanity sample from the ARTIFACT's own feature_names (a position_only variant has 16, not 20).
    feature_names = meta.get("feature_names", GHOST_OUTFIELD_FEATURE_NAMES)
    sample = pd.DataFrame(np.zeros((3, len(feature_names))), columns=feature_names)
    local_pred = model.predict_mean(sample)
    print(f"Loaded + verified {art}")
    print(f"  feature_set:        {meta.get('feature_set')}")
    print(f"  sklearn at fit:     {meta.get('sklearn_version')}")
    print(f"  training_commit:    {meta.get('training_commit')}")
    print(f"  sample predictions: {np.asarray(local_pred).tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from _hub_publish import upload_model_only
    from huggingface_hub import HfApi

    upload_model_only(HfApi(), str(art), args.repo_id)  # model-only allowlist + leak guard

    from silly_kicks.tracking import MissingFeatureContractWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFeatureContractWarning)
        back = GhostOutfieldModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_mean(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified (no MissingFeatureContractWarning).")


if __name__ == "__main__":
    main()
