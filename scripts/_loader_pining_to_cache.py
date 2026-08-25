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


def _cached(out: Path, provider: str, match_id: str) -> bool:
    """A match is already cached iff BOTH its unconditional artifacts exist. ``write_match_cache``
    writes frames.parquet THEN meta.json, so meta.json is the LAST write; a crash/OOM between the two
    (frames.parquet is a multi-hundred-MB download) leaves frames.parquet without meta.json. Requiring
    both means such a partial cache is re-done on resume -- never silently skipped, which would lose
    home_team_id forever and surface only much later when the trainer reads a missing meta.json."""
    gdir = out / provider / str(match_id)
    return (gdir / "frames.parquet").exists() and (gdir / "meta.json").exists()


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from _loader_pining import load_matches, select_match_ids

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

    # ADR-052/ADR-068 resume: list the wanted (provider, match_id) pairs UP FRONT (a cheap manifest
    # listing) and skip any already cached, so a crashed run re-fetches ONLY the missing matches --
    # not the whole corpus (a GS match alone is a multi-hundred-MB download + ~74s parse). Then load
    # ONLY the remaining ids, dropping providers with nothing left: an empty list in match_ids would
    # otherwise EXPAND back to the full manifest (the `(match_ids.get(p) ...) or list(manifest_ids)`
    # falsy trap in _wanted_for_provider).
    wanted = select_match_ids(providers=args.providers, match_ids=match_ids, max_per_provider=args.max_per_provider)
    todo: dict[str, list[str]] = {}
    n_cached = 0
    for provider, mid in wanted:
        if _cached(args.out, provider, mid):
            n_cached += 1
            continue
        todo.setdefault(provider, []).append(mid)
    if n_cached:
        print(f"Resume: {n_cached}/{len(wanted)} matches already cached -- skipping their fetch")
    if not todo:
        print(f"Done: all {n_cached} wanted matches already cached at {args.out}")
        return

    n = 0
    for provider, match_id, actions, frames, home in load_matches(
        providers=[p for p in args.providers if todo.get(p)],  # drop providers with no remaining ids
        match_ids=todo,
        max_per_provider=None,  # already applied by select_match_ids above
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
    print(f"Done: cached {n} new matches ({n_cached} already present) to {args.out}")


if __name__ == "__main__":
    main()
