"""TF-23b G1 dump: convert every cached GS + IDSSE match and write its frames (ADR-035 ship gate).

Run once per library version (the PYTHONPATH selects 4.33.0 baseline vs 4.34.0 tree):

    PYTHONPATH=~/sk-base  .venv/bin/python scripts/_tf23b_g1_dump.py --cache-dir <cache> --out /tmp/g1_base
    PYTHONPATH=~/sk-434   .venv/bin/python scripts/_tf23b_g1_dump.py --cache-dir <cache> --out /tmp/g1_434

Then `_tf23b_g1_compare.py` asserts byte-identity per match (strict dtypes) and reports the changed
set (the enumerated retrain scope). Bypasses the pining manifest/network — builds artifact paths from
the cache directory directly (GS needs no token this way), reusing the loader's own `_build_match`
provider dispatch so the conversion path is exactly production's.

NOT committed long-term tooling — a one-shot ship-gate harness; kept in scripts/ for reproducibility.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from scripts._loader_pining import _build_match  # provider dispatch -> (actions, frames, home_team_id)

_ROLES = {
    "idsse": ("events", "metadata", "tracking"),
    "gradientsports": ("events", "metadata", "roster", "tracking"),
}


def _cache_paths(cache_dir: Path, provider: str, match_id: str) -> dict[str, Path]:
    d = cache_dir / provider / match_id
    return {role: d / f"{provider}_{match_id}_{role}" for role in _ROLES[provider]}


def _matches(cache_dir: Path, provider: str) -> list[str]:
    pdir = cache_dir / provider
    if not pdir.is_dir():
        return []
    return sorted(m.name for m in pdir.iterdir() if m.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--providers", nargs="+", default=["gradientsports", "idsse"])
    ap.add_argument("--tracking-limit", type=int, default=None)
    args = ap.parse_args()

    cache_dir = args.cache_dir.expanduser()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    for provider in args.providers:
        for match_id in _matches(cache_dir, provider):
            paths = _cache_paths(cache_dir, provider, match_id)
            if not all(p.exists() for p in paths.values()):
                print(f"SKIP {provider}/{match_id}: missing cached artifact(s)")
                continue
            try:
                _actions, frames, _home = _build_match(provider, match_id, paths, args.tracking_limit)
            except Exception as exc:
                print(f"ERROR {provider}/{match_id}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            dest = out / f"{provider}_{match_id}.parquet"
            frames.to_parquet(dest, index=False)
            periods = sorted(int(p) for p in frames["period_id"].dropna().unique())
            print(f"OK {provider}/{match_id}: {len(frames)} rows, periods={periods} -> {dest.name}")


if __name__ == "__main__":
    main()
