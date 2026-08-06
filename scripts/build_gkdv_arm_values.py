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

from scripts._input_contract import declare_inputs


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-054).

    `GkdvParams` carries the arm configuration (`lambda_gk`, the domain radius, the pinned
    `method="spearman"`), so its field VALUES are what a rerun would have to match. The ghost model
    is declared by name rather than by weights: its own chirality and feature-contract stamps are
    what pin the artifact, per ADR-050.
    """
    from dataclasses import asdict

    from silly_kicks.gkdv import GkdvParams

    return declare_inputs(
        driver="build_gkdv_arm_values",
        params={"gkdv": asdict(GkdvParams())},
        extractors=("silly_kicks.gkdv._engine", "silly_kicks.gkdv._arms"),
        models=("silly_kicks.tracking._ghost_gk.GhostGkModel",),
    )


def _aggregate_manifests(dest) -> dict:
    """Corpus-wide totals, plus the conservation identity this pass is accountable for.

    The summing itself lives in :mod:`scripts._partition` and is shared with the Layer 2 spells
    producer -- the last-writer-wins defect it fixes must not be repairable in one producer while
    still live in the other. What stays HERE is the domain claim: every input frame is either
    scored or dropped for a named reason, and `drop_reasons` describes frames that produced no row
    at all, so it can never be recovered from the shard table.
    """
    from scripts._partition import aggregate_manifests

    corpus = aggregate_manifests(dest, defaults=("n_frames_in", "n_frames_scored", "n_matches"))
    corpus.setdefault("drop_reasons", {})
    scored, dropped = corpus["n_frames_scored"], sum(corpus["drop_reasons"].values())
    # Conservation across the WHOLE corpus, not merely within one worker.
    corpus["conservation_holds"] = scored + dropped == corpus["n_frames_in"]
    return corpus


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
        from scripts._partition import list_match_ids

        print(json.dumps(list_match_ids(args.providers.split(",")), indent=2))
        return

    from scripts._partition import providers_for_slice

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    # PER-MATCH SHARDS, written as each match completes. A single accumulate-then-write-once pass
    # over a 64-match corpus is both an OOM risk and unresumable: a crash at match 60 discards
    # everything. Shards also make the run restartable -- an existing shard is skipped, so a
    # re-invocation resumes rather than recomputing surfaces that already cost hours.
    dest = Path(args.out)
    xt_model = None  # unreachable-with-a-value today: the threat arm is refused above

    # Per-item counters that are NOT in the returned frame. `for_each`'s contract hands `counters`
    # the item and the tidy frame, but these come from `build_ghost_frames`'s REPORT -- frames seen,
    # frames scored, per-reason drops -- which the frame cannot carry. `for_each` calls `work(item)`
    # and then `counters(item, frame)` for the SAME item, in that order, so stashing the report here
    # is well-defined rather than a race.
    _last_report: dict = {}

    # load_matches yields (provider, match_id, ACTIONS, FRAMES, home_team_id) -- actions FIRST.
    def _work(item):
        _provider, match_id, _actions, frames, home_team_id = item
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
        _last_report.clear()
        _last_report.update(
            {
                "n_frames_in": report.n_frames_in,
                "n_frames_scored": report.n_frames_scored,
                "drop_reasons": dict(report.drop_reasons),
                "n_matches": 1,
            }
        )

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

        # An EMPTY frame still writes a shard (see `_driver.write_shard`): absent means "not yet
        # run", present-and-empty means "run, scored nothing". Conflating them would make a resume
        # silently recompute.
        return pd.DataFrame(match_rows, columns=["keeper_key", "game_id", "period_id", "frame_id", "arm", "arm_value"])

    # `reconcile` is deliberately NOT used here: this driver writes TWO per-arm tables
    # (`arm_values_delta_das` / `_delta_threat`) from one shard set, and `reconcile` writes a
    # single combined path. Its own per-arm loop below stays, on `write_table_atomically`.
    from scripts._driver import for_each
    from scripts._partition import worker_tag as _worker_tag
    from scripts._partition import write_table_atomically

    worker_tag = _worker_tag(args.match_ids_json)
    res = for_each(
        load_matches(
            providers=providers_for_slice(args.providers.split(","), match_ids),
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        counters=lambda _item, _frame: dict(_last_report),
        shard_root=dest / "shards",
        # What determines an arm VALUE: the ghost model that positions the counterfactual keeper,
        # the pitch-control method the arms integrate over, and the carrier parameters that pin the
        # domain. The downstream ICC/power analysis is NOT declared -- it re-reads these shards on
        # every invocation, so it consumes the content rather than determining it.
        token_inputs={
            # `ghost_model` is the BUNDLED default: this driver exposes no --ghost-model flag, so
            # `build_ghost_frames` resolves it internally. Declared as a literal rather than read
            # off `args` (an earlier draft wrote `args.ghost_model`, which does not exist and would
            # have raised AttributeError on the first real run). If a variant flag is ever added,
            # thread it here -- that is exactly the kind of change the token has to see.
            "ghost_model": "default",
            "pitch_control_method": "spearman",
            "arms": sorted(a for a, want in (("delta_das", want_das), ("delta_threat", want_threat)) if want),
            # `--tracking-limit` TRUNCATES the frames every downstream computation sees, so a
            # capped smoke run and a full run are DIFFERENT corpora and must never share a
            # generation. Omitting it let a smoke pass poison the real one: every match reports
            # "skip (shard exists)", the combined table is rebuilt from truncated shards, and the
            # replayed counters make `conservation_holds` corroborate a corpus never walked.
            "tracking_limit": args.tracking_limit,
        },
        tag=worker_tag,
        label="match",
    )
    totals.update(res.counters)

    shard_dir = res.shard_dir
    shards = sorted(shard_dir.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True) if shards else pd.DataFrame()
    written = {}
    for arm in ("delta_das", "delta_threat"):
        df = combined[combined["arm"] == arm] if len(combined) else combined
        if not len(df):
            continue
        path = dest / f"arm_values_{arm}.parquet"
        # Atomic: N parallel workers all rebuild this from the SHARED shard dir and write the same
        # path, so a plain to_parquet has N concurrent writers on one file and can be read -- or
        # left -- half-written. The 64-match corpus pass got away with it; that is luck, not safety.
        write_table_atomically(df, path, tag=worker_tag)
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
    # PER-WORKER manifest under a DISTINCT name, then aggregate.
    #
    # MEASURED defect this replaces: with N parallel workers all writing one shared
    # `arm_values_manifest.json`, the last writer won -- so `totals` described a SINGLE partition
    # (n_matches: 8) while `arms_written`, computed from the shared shard dir, covered all 64. The
    # data was never wrong; the artifact misdescribed its own scope, which for a provenance-bearing
    # file is the same class of defect as a false commit SHA.
    #
    # `drop_reasons` cannot be recovered from the shards (they hold only SCORED rows), so the
    # corpus totals must come from summing per-worker manifests -- not from re-reading the table.
    worker_manifest = {
        **totals,
        # Carries the generation token, so a reader can tell whether the arm tables beside
        # this manifest came from the generation directory beside them. `res.manifest()`, not a
        # bare `manifest_fields(...)`: only the method threads `counters_unrecorded`, and a
        # hand-written call silently defaults it to 0 -- so a resumed worker whose sidecars were
        # missing would report a complete corpus, which is the very defect the sidecar closed.
        **res.manifest(),
        "arm_requested": args.arm,
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov["tree_state"],
        "partition": worker_tag,
    }
    (dest / f"manifest_{worker_tag}.json").write_text(
        json.dumps(worker_manifest, indent=2, default=str), encoding="utf-8"
    )

    corpus = _aggregate_manifests(dest)
    # `run_commit` / `run_tree_dirty` are DELIBERATELY not re-stamped here. `aggregate_manifests`
    # derives them ACROSS WORKERS -- `run_tree_dirty` is an OR and `run_commit` is contributor-gated
    # against `commits_seen` -- and overwriting them with THIS process's values destroys exactly
    # that: a corpus whose last-finishing worker happened to be clean would record
    # `run_tree_dirty: false` while another worker's slice was built from a dirty tree. This
    # process's own values are already recorded, correctly, in its own `manifest_<tag>.json` above.
    # `build_layer2_spells` never did this; the two producers `_partition.py` exists to keep
    # identical had diverged on precisely the field its OR is for.
    corpus.update(arms_written=written, arm_requested=args.arm)
    (dest / "arm_values_manifest.json").write_text(json.dumps(corpus, indent=2, default=str), encoding="utf-8")
    print(json.dumps(corpus, indent=2, default=str))


if __name__ == "__main__":
    main()
