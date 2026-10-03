#!/usr/bin/env python
"""Push ONE registered model card to its EXISTING Hub repo as README.md (combined-cycle spec 9, D9).

The card-only seam (ADR-088 amendment). The card comes from CARD_SOURCE, never a free path. The repo
must already exist. An unchanged card is not re-uploaded. After an upload the README is read back and
must be byte-identical. --verify-only reports changed / unchanged without uploading.
Run from the repo root of a clean checkout at the release tag:
    PYTHONPATH=. python scripts/publish_model_card.py --repo-id silly-kicks/xshot-occurrence-v1 --verify-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._hub_publish import CARD_SOURCE, publish_card_only

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, choices=sorted(CARD_SOURCE))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    out = publish_card_only(HfApi(), args.repo_id, root=_REPO_ROOT, verify_only=args.verify_only)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
