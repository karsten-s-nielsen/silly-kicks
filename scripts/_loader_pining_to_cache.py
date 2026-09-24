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
    from _driver import for_each
    from _item_outcome import ItemExcluded
    from _loader_pining import pining_source, resolve_cache_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="raw-artifact cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR). Distinct from --out, which "
        "is this driver's MATERIALIZED frames/actions cache; --cache-dir persists the raw downloads so "
        "a re-fetch is a disk read.",
    )
    ap.add_argument(
        "--match-ids-json",
        type=Path,
        default=None,
        help="JSON file mapping {provider: [match_id, ...]} -- a per-provider allowlist. Default None "
        "(cache every listed match).",
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    match_ids = json.loads(args.match_ids_json.read_text()) if args.match_ids_json else None
    cache_dir = resolve_cache_dir(args.cache_dir)

    # ADR-052/ADR-068 resume: list the wanted refs UP FRONT (a cheap manifest listing) and skip any
    # already MATERIALIZED, so a crashed run re-fetches ONLY the missing matches -- not the whole
    # corpus (a GS match alone is a multi-hundred-MB download + ~74s parse). The `_cached` pre-filter
    # iterates refs and performs NO load, so Rule C allows it; the load happens per surviving ref
    # inside `for_each`, which itself skips a match whose marker shard already exists.
    refs, load = pining_source(
        providers=args.providers,
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
        cache_dir=cache_dir,
    )
    todo = [r for r in refs if not _cached(args.out, r.provider, r.match_id)]
    n_cached = len(refs) - len(todo)
    if n_cached:
        print(f"Resume: {n_cached}/{len(refs)} matches already cached -- skipping their fetch")
    if not todo:
        print(f"Done: all {n_cached} wanted matches already cached at {args.out}")
        return

    _last: dict = {}

    def _work(item):
        # A frame set with no velocity cannot serve train_ghost_gk, so it is a DETERMINISTIC exclusion
        # (counted, replayed on resume; ADR-052 D13), NOT a fabricated cache entry nor a silent skip.
        if item.frames is None or "vx" not in item.frames.columns or "vy" not in item.frames.columns:
            raise ItemExcluded("no vx/vy", details={"provider": item.provider, "match_id": str(item.match_id)})
        write_match_cache(
            args.out,
            provider=item.provider,
            match_id=item.match_id,
            frames=item.frames,
            actions=item.actions,
            home_team_id=item.home_team_id,
        )
        _last.clear()
        _last.update({"n_rows": len(item.frames), "n_matches": 1})
        # The materialized cache under --out IS the real output; this marker shard records that the
        # write happened (so for_each's conservation + resume are honest about what ran).
        return pd.DataFrame(
            {"provider": [item.provider], "match_id": [str(item.match_id)], "n_rows": [len(item.frames)]}
        )

    res = for_each(
        todo,
        key=lambda r: r.key,
        load=load,
        work=_work,
        counters=lambda _i, _f: dict(_last),
        shard_root=args.out / "_prefetch_shards",
        token_inputs={
            "providers": sorted(args.providers),
            "tracking_limit": args.tracking_limit,
            "schema": "prefetch-cache-1",
        },
        tag="prefetch",
        label="match",
    )
    print(
        f"Done: cached {len(res.shard_keys)} new matches ({n_cached} already present, "
        f"{res.excluded} skipped no-vx/vy) to {args.out}"
    )


if __name__ == "__main__":
    main()
