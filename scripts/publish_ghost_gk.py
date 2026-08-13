#!/usr/bin/env python
"""Publish a trained Ghost-GK artifact to HuggingFace Hub.

Mirrors `publish_xcross_attempt.py`: verify SHA-256 + a sanity prediction, upload the folder, then
re-download via `from_hub` and assert identical predictions. `--verify-only` stops before upload
(no network, no token).

WHAT THIS DISCHARGES. The Hub artifact predated ADR-050, so `from_hub` served a model whose
extractor could not be verified and emitted `MissingFeatureContractWarning` -- and a consumer
escalating that category (as this repo's CI does) got an exception on the `"full"` path. Uploading a
contract-bearing artifact is what fixes it, so this script REFUSES to publish one without a contract
and asserts the round-tripped model loads without that warning. A publish that leaves `from_hub`
broken is the failure mode worth spending a gate on.

Takes `--artifact-dir` rather than naming a bundled weights path, deliberately: a `"_weights"`
string literal in CODE makes a script a derived artifact driver
(`tests/scripts/test_provenance_wiring.py::_writes_bundled_weights`), which this is not -- it
publishes an artifact someone else produced. The two existing publishers take the same argument for
the same reason.

Requires: silly-kicks[ghost-gk].
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish a Ghost-GK artifact to HuggingFace Hub.")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/ghost-gk-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    art = Path(args.artifact_dir)

    # REFUSE a contract-less artifact rather than publishing one. Uploading a pre-ADR-050 artifact
    # is precisely how `from_hub` came to be broken; re-creating that state would make this script
    # the cause of the thing it exists to fix.
    meta = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
    contract = meta.get("feature_contract")
    if not contract:
        raise SystemExit(
            f"{art}/metadata.json carries no `feature_contract`. Publishing it would re-break "
            f"`from_hub` for every consumer that escalates MissingFeatureContractWarning. Re-stamp "
            f"with scripts/stamp_feature_contracts.py, or train on a tree that writes one."
        )

    model = GhostGkModel.load(art)  # SHA-256 + chirality + feature contract all verified here
    sample = pd.DataFrame(np.zeros((3, len(GHOST_GK_FEATURE_NAMES))), columns=GHOST_GK_FEATURE_NAMES)
    local_pred = model.predict_mean(sample)
    print(f"Loaded + verified {art}")
    print(f"  declared constants: {contract.get('constants')}")
    print(f"  sklearn at fit:     {meta.get('sklearn_version')}")
    print(f"  training_commit:    {meta.get('training_commit')}")
    print(f"  sample predictions: {np.asarray(local_pred).tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(art), repo_id=args.repo_id, repo_type="model")

    # Round-trip. `error` on the contract category, not a filter -- the point of this publish is
    # that the served artifact no longer warns, so a warning here is a FAILED publish, not noise.
    from silly_kicks.tracking import MissingFeatureContractWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFeatureContractWarning)
        back = GhostGkModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_mean(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified (no MissingFeatureContractWarning).")


if __name__ == "__main__":
    main()
