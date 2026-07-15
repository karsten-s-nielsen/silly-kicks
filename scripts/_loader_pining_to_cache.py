#!/usr/bin/env python
"""Stream pining matches -> per-match tc3-layout cache for train_ghost_gk.py (PR-S81).

Writes {out}/{provider}/{match_id}/frames.parquet + meta.json (home_team_id), and
optional actions to {out}/_actions/{match_id}.parquet. Frames carry vx/vy because
_loader_pining yields smooth_frames+derive_velocities output.

Usage:
    set -a; source ~/.pining_env; set +a
    python scripts/_loader_pining_to_cache.py --providers skillcorner idsse gradientsports \
        --out ~/Development/ghost_gk_refit/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def write_match_cache(
    out: Path,
    *,
    provider: str,
    match_id: str,
    frames: pd.DataFrame,
    actions: pd.DataFrame | None,
    home_team_id: object,
) -> None:
    """Write one match into the {provider}/{match_id}/ layout the trainer consumes."""
    gdir = out / provider / str(match_id)
    gdir.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(gdir / "frames.parquet")
    (gdir / "meta.json").write_text(json.dumps({"home_team_id": home_team_id}))
    if actions is not None and len(actions) > 0:
        adir = out / "_actions"
        adir.mkdir(parents=True, exist_ok=True)
        actions.to_parquet(adir / f"{match_id}.parquet")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from _loader_pining import load_matches

    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        type=Path,
        default=None,
        help="JSON file mapping {provider: [match_id, ...]} -- a per-provider allowlist threaded to "
        "load_matches(match_ids=). Default None (load every listed match).",
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    match_ids = json.loads(args.match_ids_json.read_text()) if args.match_ids_json else None
    n = 0
    for provider, match_id, actions, frames, home in load_matches(
        providers=args.providers,
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        if "vx" not in frames.columns or "vy" not in frames.columns:
            print(f"  SKIP {provider}/{match_id}: no vx/vy", file=sys.stderr)
            continue
        write_match_cache(
            args.out, provider=provider, match_id=match_id, frames=frames, actions=actions, home_team_id=home
        )
        n += 1
        print(f"  [{n}] cached {provider}/{match_id}: {len(frames)} rows")
    print(f"Done: cached {n} matches to {args.out}")


if __name__ == "__main__":
    main()
