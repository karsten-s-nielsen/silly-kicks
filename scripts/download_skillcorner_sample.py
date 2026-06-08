#!/usr/bin/env python
"""Idempotently download the 10 public SkillCorner matches into SKILLCORNER_SAMPLE_DIR.

Populates SAMPLE_DIR/<match_id>/<original_filename> for the match.json, tracking
(extrapolated jsonl), and dynamic-events CSV artifacts — the layout both
tests/spadl/test_skillcorner_e2e.py and tests/tracking/test_gk_skillcorner_roster_e2e.py
read. Uses the pining PUBLIC token (no owner tier needed). Skips files already on disk.

Run: python scripts/download_skillcorner_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run as a bare script (`python scripts/download_skillcorner_sample.py`): put the REPO
# ROOT on sys.path so the namespace-package imports below resolve to the SAME module
# objects pytest uses (no dual-module footgun; scripts/ has no __init__.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._loader_pining import (
    _artifact_key,
    _base_url,
    _download_to_temp,
    _list_matches,
    _resolve_token,
)
from tests._skillcorner_sample import MATCH_IDS, SAMPLE_DIR

_SUFFIXES = ("_match.json", "_tracking_extrapolated.jsonl", "_dynamic_events.csv")


def main() -> int:
    tok, base = _resolve_token(None), _base_url()
    manifest = {m["id"]: m for m in _list_matches("skillcorner", tok, base)}
    wanted = [mid for mid in MATCH_IDS if mid in manifest] or list(manifest)
    for mid in wanted:
        artifacts = manifest[mid]["artifacts"]
        dest = SAMPLE_DIR / mid
        dest.mkdir(parents=True, exist_ok=True)
        for stale in dest.glob("skillcorner_*"):  # sweep interrupted-download temps
            stale.unlink()
        for suffix in _SUFFIXES:
            key = _artifact_key(artifacts, suffix=suffix)
            target = dest / str(artifacts[key])  # original filename, ends with suffix
            if target.exists():
                print(f"  skip {mid}/{target.name} (present)")
                continue
            tmp = _download_to_temp("skillcorner", mid, key, tok, base, dest)
            tmp.replace(target)  # same-dir atomic move
            print(f"  saved {mid}/{target.name}")
    print(f"done -> {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
