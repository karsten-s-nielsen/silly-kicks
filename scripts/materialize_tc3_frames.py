"""Materialize pining-loaded tracking frames into the TC3 layout the ghost trainer reads.

`_loader_pining.load_matches(cache_dir=...)` caches RAW provider artifacts under
`cache_dir/{provider}/{match_id}/` and parses frames in memory; it never writes them (verified: no
`to_parquet` anywhere in that module). The ghost trainer globs `**/frames.parquet` and exits
otherwise. The directory SHAPE coincides and the CONTENTS do not, so pointing `--data-dir` at the
pining cache finds nothing.

This bridges the two, so ONE download serves both the ghost re-fit and the TF-24 refresh, and leaves
a reusable corpus cache for the next cycle.

Writing the frames is only half of it: a divergent parse that gets positions right and velocities
wrong would land UNDERNEATH the delta measurement built to detect trouble. `assert_frames_parity`
is the guard, and it is checked against a known-good TC3 parquet BEFORE the corpus run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._driver import for_each
from scripts._provenance import git_provenance, require_clean_tree

#: Identity columns -- used to ORDER rows deterministically, never to restrict what is hashed.
_KEY_COLUMNS = ("game_id", "period_id", "frame_id", "player_id", "x", "y", "team_id")

#: Bump whenever the SHAPE of a shard changes (columns added/renamed/dropped). Pinned into
#: `token_inputs` so a schema change lands in a NEW generation instead of silently combining with
#: stale shards -- the 4.77.1 defect, where 22 stale shards were skipped as already-done.
_SHARD_SCHEMA_VERSION = "tc3-frames-1"


def _checksum(df: pd.DataFrame) -> str:
    """Hash EVERY column, ordered deterministically.

    Restricting the hash to `_KEY_COLUMNS` loses the content this gate exists to protect. Measured:
    with only those hashed, a parse that gets positions right and drifts `vx` from 0.5 to 99.0
    passes -- while ghost's extractor consumes velocity (`to_gr_vx`, `_VELOCITY_WINDOW_S`) and
    `infer_ball_carrier` consumes `vx`/`vy` for its `beta` term.

    `+ 0.0` normalises `-0.0` to `0.0`: they hash differently otherwise, and negative zero is
    reachable through the velocity NEGATION (`-vx` where `vx == 0.0`); subtraction of equal operands
    gives `+0.0`, so positions are not the source.

    Sorting on ALL columns, not just the identity ones: two rows tying on every identity column but
    differing on `vx` hash differently when reordered, because `sort_values` leaves ties in input
    order. GS duplicate frames make that reachable. With every column in the key, remaining ties are
    fully identical rows, whose order cannot affect the hash.
    """
    ordered = df.reindex(sorted(df.columns), axis=1)
    float_cols = [c for c in ordered.columns if pd.api.types.is_float_dtype(ordered[c])]
    for c in float_cols:
        ordered[c] = ordered[c] + 0.0
    ordered = ordered.sort_values(list(ordered.columns), kind="mergesort").reset_index(drop=True)
    hashed = np.asarray(pd.util.hash_pandas_object(ordered, index=False))
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def assert_frames_parity(produced: pd.DataFrame, reference: pd.DataFrame, *, match_id: str) -> None:
    """Raise unless `produced` matches `reference` in schema, row count, dtypes and content.

    Raises `AssertionError` EXPLICITLY rather than using bare `assert`: this is a guard whose whole
    job is to stop a corpus run, and `python -O` strips `assert` statements. (ruff's S101 forbids
    them under `scripts/` for exactly that reason.)
    """
    missing = set(reference.columns) - set(produced.columns)
    if missing:
        raise AssertionError(f"{match_id}: produced frames missing column(s) {sorted(missing)}")
    if len(produced) != len(reference):
        raise AssertionError(f"{match_id}: row count {len(produced)} != reference {len(reference)}")
    for col in reference.columns:
        if produced[col].dtype != reference[col].dtype:
            raise AssertionError(f"{match_id}: dtype drift on {col}: {produced[col].dtype} != {reference[col].dtype}")
    if _checksum(produced) != _checksum(reference):
        raise AssertionError(
            f"{match_id}: content checksum differs despite matching schema -- the trainer would fit "
            f"on different data than the established pipeline produces"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize pining frames into a resumable corpus cache.")
    ap.add_argument("--cache-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument(
        "--reference-parquet",
        type=pathlib.Path,
        default=None,
        help="An existing TC3 frames.parquet to assert parity against on the FIRST match.",
    )
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    from scripts._loader_pining import load_matches

    reference = pd.read_parquet(args.reference_parquet) if args.reference_parquet else None
    checked: list[str] = []

    def _work(item):
        _provider, match_id, _actions, frames, _home = item
        # Parity is asserted on the FIRST match only: it is a pipeline-shape check, not a per-row
        # one, and re-reading the reference per match would cost the corpus pass for no new signal.
        if reference is not None and not checked:
            assert_frames_parity(frames, reference, match_id=str(match_id))
            checked.append(str(match_id))
            print(f"parity OK against {args.reference_parquet}", flush=True)
        return frames

    # ADR-052: this walks a corpus and each item costs minutes, so it MUST persist per item.
    # Materializing 179 matches with a naive loop loses everything on a crash at match 150 -- the
    # exact failure that seam exists to prevent. The shards ARE the deliverable here: a `for_each`
    # generation directory holds only per-item parquets, and `train_ghost_gk.py:291` falls back to
    # a flat `*.parquet` glob, so the trainer reads the generation directory directly.
    res = for_each(
        load_matches(
            providers=args.providers,
            cache_dir=args.cache_dir,
            max_per_provider=args.max_per_provider,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=args.out / "shards",
        # What determines the CONTENT of a materialized frame set: which providers were requested
        # and the parse that produced them. `--reference-parquet` is deliberately absent -- it is a
        # verification switch, not an input, so toggling it must not orphan a generation.
        token_inputs={
            "providers": sorted(args.providers),
            "schema": _SHARD_SCHEMA_VERSION,
        },
        label="match",
    )
    # Write the manifest beside the shards. Every other `for_each` adopter does, and here it is what
    # gives the corpus CACHE its provenance: a consumer that trains on these frames can check which
    # commit produced them and whether the pass conserved, rather than trusting a directory.
    # `res.manifest()`, never a hand-written `manifest_fields(...)` -- only the method threads
    # `counters_unrecorded`, so a hand-rolled call silently defaults it to 0 and a resumed pass
    # reports a complete corpus it never walked.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest_all.json").write_text(
        json.dumps(
            {
                **res.manifest(),
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
                "providers": sorted(args.providers),
                "parity_checked_against": str(args.reference_parquet) if reference is not None else None,
                "parity_checked_match": checked[0] if checked else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"materialized {len(res.keys)} matches; shards at {res.shard_dir}")
    print("point train_ghost_gk.py --data-dir at that directory (flat *.parquet fallback)")


if __name__ == "__main__":
    main()
