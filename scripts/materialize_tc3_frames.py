"""Materialize pining-loaded tracking frames into the TC3 layout the ghost trainer reads.

`_loader_pining.load_matches(cache_dir=...)` caches RAW provider artifacts under
`cache_dir/{provider}/{match_id}/` and parses frames in memory; it never writes them (verified: no
`to_parquet` anywhere in that module). The ghost trainer globs `**/frames.parquet` and exits
otherwise. The directory SHAPE coincides and the CONTENTS do not, so pointing `--data-dir` at the
pining cache finds nothing.

**Relationship to `_loader_pining_to_cache.py`, which does the same job and came first (PR-S81).**
That script is the ESTABLISHED pipeline -- it produced the input the currently-bundled ghost weights
were fit on -- and it writes the TC3 tree `{provider}/{match_id}/frames.parquet` + `meta.json`, plus
actions under `_actions/`. This driver exists for ONE reason: that one has no resume, so a crash at
match 150 of 179 loses the pass, which is the failure ADR-052 was written for. It is NOT a
replacement parse: both consume the SAME `load_matches` generator and write the yielded frames
unchanged.

**That shared parse is also the honest limit of `assert_frames_parity`.** Since both pipelines share
their parse, the assertion cannot detect a divergent PARSE -- there is only one. What it does check
is the WRITE path: schema, row count, dtypes and a full-content checksum surviving the parquet
round-trip. Worth running, and worth not overclaiming: the design note that motivated it said "the
trainer's established input comes from a different pipeline", and that is not the case.

The trainer needs more than frames, and a `for_each` generation cannot carry it: `--home-teams` (or
a `meta.json` beside each parquet, which flat shards do not have) and `--actions-dir`. Both are
emitted here as item-keyed side artifacts; see `_SHARD_SCHEMA_VERSION` for why that forces a token
bump even though no shard column changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._driver import for_each, join_key
from scripts._provenance import git_provenance, require_clean_tree

#: Identity columns -- used to ORDER rows deterministically, never to restrict what is hashed.
_KEY_COLUMNS = ("game_id", "period_id", "frame_id", "player_id", "x", "y", "team_id")

#: Bump whenever the SHAPE of a shard changes (columns added/renamed/dropped). Pinned into
#: `token_inputs` so a schema change lands in a NEW generation instead of silently combining with
#: stale shards -- the 4.77.1 defect, where 22 stale shards were skipped as already-done.
#:
#: `-2` bumps for a reason the "shape of a shard" wording does NOT cover, and the wording is the
#: narrower of the two: the shard COLUMNS are unchanged, but this driver now emits per-item SIDE
#: artifacts (`_home/`, `_actions/`) that the trainer requires. `for_each` resumes by skipping items
#: whose SHARD exists, so a `-1` generation -- shards present, side artifacts absent -- would skip
#: every item, emit an EMPTY home map, and report a conserved pass. Exactly the 4.77.1 shape with a
#: different trigger. **The real invariant is "bump when a resumed pass would produce something
#: different", not "when a column changed".**
_SHARD_SCHEMA_VERSION = "tc3-frames-2"


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


def collect_home_team_map(home_dir: pathlib.Path, keys) -> dict[str, str]:
    """Assemble the trainer's `--home-teams` map from per-item sidecars, or RAISE.

    `train_ghost_gk.py` resolves home teams from a `meta.json` BESIDE each parquet (`:379-387`),
    which exists in the TC3 tree layout and does NOT exist in a `for_each` generation -- flat
    shards, no per-match directory. Without a map the trainer's is empty and it `sys.exit(1)`s at
    `:388`, AFTER the corpus pass has been paid for.

    **Completeness is the point, not the file.** A shard whose sidecar is missing -- a generation
    predating the side artifacts, or a kill between the two writes -- would otherwise drop that game
    from the map, and the trainer prints `SKIP game <id>: no home_team_id` per game and fits on a
    SHORTER corpus while reporting success. That is a quiet corpus truncation, so it raises here.

    Keyed by the `game_id` the FRAMES carry, never the match id: SkillCorner's `game_id` is a kloppy
    hash unrelated to its match id, and the trainer looks the map up as
    `home_team_map.get(str(game_id))`.
    """
    home_map: dict[str, str] = {}
    missing = []
    for key in keys:
        path = home_dir / f"{join_key(key)}.json"
        if not path.exists():
            missing.append(join_key(key))
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        for gid in rec["game_ids"]:
            home_map[str(gid)] = str(rec["home_team_id"])
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(list(keys))} shards have no home-team sidecar "
            f"(first: {missing[:3]}). The trainer would skip those games and fit on a shorter "
            f"corpus while reporting success. Re-run those items, or delete the generation."
        )
    return home_map


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

    home_dir = args.out / "_home"
    actions_dir = args.out / "_actions"

    def _work(item):
        provider, match_id, actions, frames, home = item
        # Parity is asserted on the FIRST match only: it is a pipeline-shape check, not a per-row
        # one, and re-reading the reference per match would cost the corpus pass for no new signal.
        if reference is not None and not checked:
            assert_frames_parity(frames, reference, match_id=str(match_id))
            checked.append(str(match_id))
            print(f"parity OK against {args.reference_parquet}", flush=True)

        # `home` and `actions` are TRAINER INPUTS, not decoration, and neither can ride the
        # `for_each` contract: `work` returns one tidy frame, and a `{game_id: team_id}` counter
        # dict is DROPPED by `aggregate_manifests` (`_partition.py:88` merges a dict counter only
        # when every value is numeric). So they are persisted as item-keyed FILES, which gives them
        # the same durability model as the shard: a resumed pass finds them already on disk.
        #
        # `join_key` rather than a local f-string, so a sidecar sits next to the shard it belongs to
        # and inherits that function's injectivity guarantee -- two providers sharing a `match_id`
        # must not overwrite each other here either.
        stem = join_key((str(provider), str(match_id)))
        home_dir.mkdir(parents=True, exist_ok=True)
        (home_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "home_team_id": str(home),
                    # Keyed by the game_id the FRAMES carry, never the directory or match id:
                    # SkillCorner's `game_id` is a kloppy hash unrelated to its match id, and the
                    # trainer looks the map up as `home_team_map.get(str(game_id))`.
                    "game_ids": sorted({str(g) for g in frames["game_id"].dropna().unique()}),
                },
                default=str,
            ),
            encoding="utf-8",
        )
        if actions is not None and len(actions) > 0:
            actions_dir.mkdir(parents=True, exist_ok=True)
            actions.to_parquet(actions_dir / f"{stem}.parquet")
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

    # --- assemble the trainer's home-team map, and REFUSE a partial one -------------------------
    #
    # `train_ghost_gk.py` resolves home teams from a `meta.json` BESIDE each parquet
    # (`:379-387`), which exists in the TC3 tree layout and does NOT exist in a `for_each`
    # generation -- flat shards, no per-match directory. Without this file the trainer's map is
    # empty and it `sys.exit(1)`s at `:388`, AFTER the corpus pass has been paid for.
    #
    # The completeness check is the point, not the file: a shard whose sidecar is missing (a
    # pre-`-2` generation, or a kill between the two writes) would otherwise silently drop that
    # game from the map, and the trainer would print `SKIP game <id>: no home_team_id` per game and
    # fit on a SHORTER corpus while reporting success. Fail here instead.
    home_map = collect_home_team_map(home_dir, res.keys)
    (args.out / "home_teams.json").write_text(json.dumps(home_map, indent=2, sort_keys=True), encoding="utf-8")

    (args.out / "manifest_all.json").write_text(
        json.dumps(
            {
                **res.manifest(),
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
                "providers": sorted(args.providers),
                "parity_checked_against": str(args.reference_parquet) if reference is not None else None,
                "parity_checked_match": checked[0] if checked else None,
                "n_home_team_entries": len(home_map),
                "n_action_shards": len(sorted(actions_dir.glob("*.parquet"))) if actions_dir.exists() else 0,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"materialized {len(res.keys)} matches; shards at {res.shard_dir}")
    print(f"home map: {len(home_map)} games -> {args.out / 'home_teams.json'}")
    print("train_ghost_gk.py needs BOTH: --data-dir <generation> --home-teams <out>/home_teams.json")
    print(f"and --actions-dir {actions_dir} to reproduce the established pipeline inputs")


if __name__ == "__main__":
    main()
