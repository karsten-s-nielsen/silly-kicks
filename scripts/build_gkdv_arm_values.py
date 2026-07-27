"""Maintainer driver: persist per-frame GKDV arm values for the S6.1 ICC power leg.

TF-19 sign-off package. `scripts/run_signoff_power.py --arm-values <parquet>` consumes the table
this writes: one row per SCORED counterfactual frame, carrying the arm value, the keeper it belongs
to, and the match it occurred in -- exactly the (values, groups, blocks) `icc_power_curve` needs.

Split out as its OWN pass, and run ONCE, because this is the expensive leg: accessible-space plus
Spearman pitch control on every domain frame, twice (factual and ghost). The power simulator then
resamples the persisted table rather than recomputing surfaces per replicate.

TWO CORRECTNESS CONSTRAINTS, both load-bearing:

* **No `PitchControlCache`, ever.** It keys on frame IDENTITY (game/period/frame/team/method/
  params/ball/decompose) and excludes player positions, so a ghost frame -- which carries its twin's
  identity -- would be served the FACTUAL leg's surface and every delta would collapse to exactly
  zero, silently. The arms deliberately accept no cache; this note exists so nobody adds one.
* **Dropped frames are dropped, never zero.** `build_ghost_frames` drops-and-counts a frame whose
  ghost is missing/NaN; scoring those as delta =0 would read as "no deterrence" and bias every keeper
  aggregate toward the null. Only rows with `drop_reason` NaN are scored, and the conservation
  identity (scored + drops == in) is asserted per match.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/build_gkdv_arm_values.py --out <DIR> [--providers gradientsports] \
      [--arm das|threat|both] [--max-per-provider N] [--tracking-limit N]

The `threat` arm is REFUSED, not defaulted: it needs a fitted ExpectedThreat, and none can be
loaded (`ExpectedThreat` exposes only fit/interpolator/rate -- there is no serialization in the
package, and `FrozenXt` wraps an already-fitted in-memory model). Fitting one in-process is a
leakage decision that belongs in its own registered cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _frame_slice(frames, gid, per, fid):
    """One (game, period, frame) slice -- the unit both arms consume."""
    return frames[(frames["game_id"] == gid) & (frames["period_id"] == per) & (frames["frame_id"] == fid)]


def _attacking_team_id(frame_slice, defending_team_id):
    """The in-possession team: the non-ball team that is NOT defending (ADR-019 dtype-safe)."""
    from silly_kicks.id_compat import same_id

    teams = frame_slice.loc[~frame_slice["is_ball"].astype(bool), "team_id"].dropna().unique()
    other = [t for t in teams if not same_id(t, defending_team_id)]
    return other[0] if other else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--arm", choices=["das", "threat", "both"], default="das")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help=(
            'JSON {"gradientsports": ["10502", ...]} pinning WHICH matches this process handles. '
            "This is how the corpus pass is PARALLELISED: split the id list N ways and launch N "
            "processes, each with its own slice and a SHARED --out. Ids are STRINGS in the "
            "manifest. Without it a second process would re-walk the corpus from the start and "
            "redo work, because shards are written on COMPLETION, not claimed up front."
        ),
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; manifest is marked)")
    ap.add_argument(
        "--list-matches",
        action="store_true",
        help="print the available match ids as JSON and exit (build the parallel split from this)",
    )
    args = ap.parse_args()

    # The threat arm is REFUSED here, and the reason is a library fact rather than a preference.
    #
    # `compute_threat_pc` types `xt` as a required ExpectedThreat; before this cycle's guard it
    # returned 0.0 for None, so a threat pass would have persisted structural zeros and the ICC
    # computed on them would be degenerate while looking like a measurement.
    #
    # And it cannot simply be handed one by path: `ExpectedThreat` exposes only fit/interpolator/
    # rate -- there is NO serialization anywhere in the package, and `FrozenXt` is a provenance
    # wrapper around an already-fitted in-memory model, not a loader. So the threat arm requires
    # an IN-PROCESS fit on a chosen corpus, which is a leakage decision (which matches? disjoint
    # from what?) that belongs in its own registered cycle -- not an inline default here.
    if args.arm in ("threat", "both"):
        raise SystemExit(
            "--arm threat|both is not runnable yet: it needs a fitted ExpectedThreat, and none "
            "can be loaded (ExpectedThreat has no save/load; FrozenXt wraps an in-memory model). "
            "Fitting one in-process is a registered leakage decision -- give the threat arm its "
            "own cycle. Use --arm das."
        )

    import numpy as np
    import pandas as pd

    from scripts._loader_pining import load_matches
    from silly_kicks.gkdv import build_ghost_frames, delta_das, delta_threat_suppression
    from silly_kicks.id_compat import ids_equal
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    want_das = args.arm in ("das", "both")
    want_threat = args.arm in ("threat", "both")
    totals = {"n_frames_in": 0, "n_frames_scored": 0, "drop_reasons": {}, "n_matches": 0}

    from scripts._provenance import git_provenance, require_clean_tree

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    # --list-matches writes no artifact, so it is exempt from the clean-tree requirement.
    prov = (
        {"commit": "n/a", "dirty": False, "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    if args.list_matches:
        # Consumes the loader's own `_list_matches` (private, but the exact call `load_matches`
        # makes internally, so the id set cannot drift from what a run would actually fetch).
        # `scripts/_loader_*` is read-only here -- this reads it, never edits it.
        from scripts._loader_pining import _base_url, _list_matches, _resolve_token

        tok, base = _resolve_token(None), _base_url()
        ids = {p: [m["id"] for m in _list_matches(p, tok, base)] for p in args.providers.split(",")}
        print(json.dumps(ids, indent=2))
        return

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    # PER-MATCH SHARDS, written as each match completes. A single accumulate-then-write-once pass
    # over a 64-match corpus is both an OOM risk and unresumable: a crash at match 60 discards
    # everything. Shards also make the run restartable -- an existing shard is skipped, so a
    # re-invocation resumes rather than recomputing surfaces that already cost hours.
    dest = Path(args.out)
    shard_dir = dest / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    xt_model = None  # unreachable-with-a-value today: the threat arm is refused above

    # load_matches yields (provider, match_id, ACTIONS, FRAMES, home_team_id) -- actions FIRST.
    for _provider, match_id, _actions, frames, home_team_id in load_matches(
        providers=args.providers.split(","),
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        shard = shard_dir / f"{_provider}__{match_id}.parquet"
        if shard.is_file():
            print(f"  skip {match_id}: shard exists")
            continue
        # The arms route through DAS, which REQUIRES `team_in_possession`; raw loader frames do
        # not carry it. Since ADR-043 removed the broad except that used to swallow this into an
        # all-NaN column, it now raises -- which is how this surfaced at all.
        #
        # The carrier is computed ONCE and shared by the possession derivation, the engine's
        # domain filter and its serving seam (spec S4.1 pins the carrier once for exactly this
        # reason: two carrier definitions would make the domain and the substitution disagree).
        carrier = infer_ball_carrier(frames)
        frames = derive_team_in_possession(frames, carrier)
        cf, provenance, report = build_ghost_frames(
            frames,
            home_team_id=home_team_id,  # type: ignore[arg-type]
            carrier=carrier,
        )
        totals["n_frames_in"] += report.n_frames_in
        totals["n_frames_scored"] += report.n_frames_scored
        for reason, n in report.drop_reasons.items():
            totals["drop_reasons"][reason] = totals["drop_reasons"].get(reason, 0) + n
        totals["n_matches"] += 1

        # Conservation (the engine guarantees it): a silent shortfall means frames vanished.
        # A raise, not an assert: asserts vanish under -O, and a frame that is neither scored nor
        # counted as a drop is exactly the silent-null shape this package exists to refuse.
        if report.n_frames_scored + sum(report.drop_reasons.values()) != report.n_frames_in:
            raise RuntimeError(
                f"{match_id}: scored ({report.n_frames_scored}) + drops "
                f"({sum(report.drop_reasons.values())}) != in ({report.n_frames_in}) -- frames vanished"
            )

        match_rows: list[dict] = []
        # SELECT THE DEFENDING KEEPER. The serving seam writes a row for BOTH teams' keepers, and
        # `build_ghost_frames` substitutes only the DEFENDING one -- so a naive pass-through
        # attributes each frame's delta to two keepers, one of whom never moved. MEASURED on real
        # data before this filter: 4448 rows from 2224 scored frames, both rows per frame carrying
        # an IDENTICAL arm_value under different keeper_keys. That is keeper-INDEPENDENT noise, and
        # it compresses between-keeper variance toward zero -- the same mechanism that made the
        # xT-GK v2 metric read "keeper-flat" on fabricated origins (ADR-036/PR-S113).
        # `provenance_to_targets` applies this same rule but drops `player_id`, so the selection is
        # reproduced here rather than the adapter reused.
        # `.reset_index` before masking: `ids_equal` returns a Series indexed 0..n-1, while the
        # drop_reason filter leaves the ORIGINAL non-contiguous index -- pandas then refuses the
        # mask as unalignable. The local test missed this because its fixture filtered nothing, so
        # the index stayed contiguous; the real provenance always has gaps.
        scored = provenance[provenance["drop_reason"].isna()].reset_index(drop=True)
        keep = np.asarray(ids_equal(scored["gk_team_id"], scored["defending_team_id"]), dtype=bool)
        scored = scored[keep]
        for rec in scored.to_dict("records"):
            gid, per, fid = rec["game_id"], rec["period_id"], rec["frame_id"]
            actual = _frame_slice(frames, gid, per, fid)
            ghost = _frame_slice(cf, gid, per, fid)
            atk = _attacking_team_id(actual, rec["defending_team_id"])
            if atk is None:
                continue
            base = {"keeper_key": rec["player_id"], "game_id": gid, "period_id": per, "frame_id": fid}
            if want_das:
                match_rows.append(
                    {**base, "arm": "delta_das", "arm_value": float(delta_das(actual, ghost, attacking_team_id=atk))}
                )
            if want_threat:
                match_rows.append(
                    {
                        **base,
                        "arm": "delta_threat",
                        "arm_value": float(
                            delta_threat_suppression(
                                actual,
                                ghost,
                                attacking_team_id=atk,
                                xt=xt_model,
                                home_team_id=home_team_id,  # type: ignore[arg-type]
                            )
                        ),
                    }
                )

        # Written even when EMPTY: an absent shard means "not yet run", a present empty one means
        # "run, scored nothing". Conflating them would make a resume silently recompute.
        pd.DataFrame(
            match_rows, columns=["keeper_key", "game_id", "period_id", "frame_id", "arm", "arm_value"]
        ).to_parquet(shard, index=False)
        print(f"  {match_id}: {len(match_rows)} rows -> {shard.name}")

    shards = sorted(shard_dir.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True) if shards else pd.DataFrame()
    written = {}
    for arm in ("delta_das", "delta_threat"):
        df = combined[combined["arm"] == arm] if len(combined) else combined
        if not len(df):
            continue
        path = dest / f"arm_values_{arm}.parquet"
        df.to_parquet(path, index=False)
        # Reported so a structurally-degenerate ICC input is visible BEFORE the power run:
        # a keeper appearing in one match makes the block permutation a pure relabelling.
        spanning = df.groupby("keeper_key")["game_id"].nunique()
        written[arm] = {
            "path": str(path),
            "n_rows": len(df),
            "n_keepers": len(spanning),
            "n_single_match_keepers": int((spanning <= 1).sum()),
            "n_nonzero": int((df["arm_value"] != 0).sum()),
        }

    # The arm-values table is what the S6.1 ICC number derives from, so it carries its own
    # provenance -- a clean SHA on the power metrics would otherwise launder a dirty input.
    manifest = {
        **totals,
        "arms_written": written,
        "arm_requested": args.arm,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    (dest / "arm_values_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
