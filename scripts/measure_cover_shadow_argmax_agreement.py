"""Measure cheap-path vs exact-path agreement for ``max_single_defender_player_id`` (TF-30).

The aggregator's default (``detailed=False``) names the top blocking defender from
``score_per_blocker`` -- accumulated lane reception deltas. The ``detailed=True`` path names it from
the full pitch-control Voronoi counterfactual. These are DIFFERENT quantities, so the cheap path's
identity is an approximation of the exact one, and the column ships only if that approximation is
usually right.

Three numbers, not one:

1. **Agreement rate with a Wilson interval.** A bare rate hides its own precision.
2. **The HARM at disagreements, in exact-path units.** Agreement rate alone is the wrong decision
   input: a disagreement between two near-tied defenders is harmless, while the same rate with large
   gaps means the cheap path is naming materially wrong players. Note the harm is NOT
   ``max_def_exact - max_def_cheap`` -- the two paths compute different quantities, so their maxima
   differ in magnitude whether or not the argmax agrees. The harm is the cheap path's nominee scored
   THROUGH THE EXACT PATH and differenced against the exact winner, so both terms share one scale.
3. **The full ``max_def`` distribution, INCLUDING rows the qualification filter excludes.** This is
   what sets ``TOL_ATTRIB``. Look for a gap between the zero cluster and the smallest genuinely
   non-zero values. If no such gap exists, that is itself the finding -- "no attribution" and "small
   attribution" are not separable, and the NA rule needs rethinking rather than a tighter constant.

Pre-registered decision rule (named BEFORE any number was seen; do not move it afterwards):

| Outcome                          | Action                                                    |
|----------------------------------|-----------------------------------------------------------|
| agreement >= 0.9 at n >= 100     | ship as specified; record the number + gap distribution   |
| below either                     | gate the column to ``detailed=True``, or drop it          |

0.9 is a STATED ENGINEERING THRESHOLD, not derived -- "a consumer reading ``..._player_id`` assumes
it is usually right".

Requires ``PINING_FOR_THE_DATA_TOKEN`` in the environment (never hardcode it).

Usage::

    python scripts/measure_cover_shadow_argmax_agreement.py --max-matches 4 --out docs/research/...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._driver import for_each, shard_path
from scripts._loader_pining import load_matches
from scripts._provenance import git_provenance, require_clean_tree
from silly_kicks.tracking import link_actions_to_frames, resolve_defended_goals
from silly_kicks.tracking._action_orientation import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    acting_team_attacks_rtl,
)
from silly_kicks.tracking._cover_shadows import (
    TOL_ATTRIB,
    _compute_cover_shadow_dict,
    compute_blocking_score,
)
from silly_kicks.xthreat import ExpectedThreat


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it stays inside [0, 1] and behaves at the extremes
    -- an agreement rate near 1.0 is exactly where the normal interval is worst, and near 1.0 is
    where this measurement is expected to land.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _lane_blocker_count(frame_data, attacking_team_id, goal_map) -> int:
    """Number of lane blockers, mirroring ``_compute_cover_shadow_dict``'s own construction."""
    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.id_compat import ids_match

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    defenders_outfield = players[
        (~ids_match(players["team_id"], attacking_team_id)) & (~players["is_goalkeeper"].astype(bool))
    ]
    attackers = players[ids_match(players["team_id"], attacking_team_id)]
    # ADR-055: the DEFENDERS' own goal is the end the attacking team ATTACKS -- a real lookup
    # of the opponent's map entry, not `same_id(attacking_team_id, home_team_id)`.
    goal_x_own = goal_map.attacked_goal(
        frame_data["game_id"].iloc[0], frame_data["period_id"].iloc[0], attacking_team_id, allow_guess=True
    )
    if goal_x_own is None:
        return 0
    man_markers = cs._classify_man_markers(
        defenders_outfield, attackers, goal_x_own=goal_x_own, params=cs.CoverShadowParams()
    )
    return sum(1 for pid in defenders_outfield["player_id"] if pid not in man_markers)


def measure_match(actions, frames, home_team_id, xt, *, match_id: str) -> list[dict]:
    """One record per action that both paths could score.

    ``home_team_id`` is retained in the signature because callers pass it and it still
    identifies the match's home side; the GEOMETRY, however, now comes from the goal map
    derived from ``frames`` (ADR-055).
    """
    goal_map = resolve_defended_goals(frames)
    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    # ADR-028 (RC1). `start_x`/`start_y` are ACTION-LTR while every position they are compared
    # against inside `_compute_cover_shadow_dict` -- defenders, receivers, the ball -- is FRAME-LTR,
    # so an AWAY passer used to enter the geometry a 180-degree point reflection away, at the wrong
    # END of the pitch. 4.70.0/PR-S138 fixed the two `features.py` callers; this driver imports
    # `_compute_cover_shadow_dict` DIRECTLY (see the import block above), so it was never a
    # registered site and the defect stayed live on main. `_cover_shadows.py` itself contains no
    # `acting_team_attacks_rtl` at all -- the module never reprojects, its callers must.
    #
    # It does NOT cancel between the two arms this script compares: the CHEAP path consumes the
    # passer and the EXACT (pitch-control counterfactual) path does not, so the defect degraded
    # exactly the agreement being measured. `docs/research/cover_shadow_identity/`'s 0.1992 is
    # therefore a PRE-RC1 number and needs an owner re-run; the gating verdict itself survives by
    # arithmetic (0.157 x 970 = 152 agreements against a 0.90 floor needing 873 -- even if every
    # away row flipped to agreeing, the ceiling is 637/970 = 0.657 < 0.90).
    #
    # Reproject the PASSER into frame coords, not the frame into action-LTR: everything downstream
    # of this tuple is frame-convention, and the one place that steps out (the xT lookup at
    # `_cover_shadows.py:1164`) already reprojects itself. Computed ONCE per match.
    flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)

    records: list[dict] = []
    for j, (_idx, row) in enumerate(actions.iterrows()):
        aid, tid = row["action_id"], row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        try:
            frame_data = frame_groups.get_group((row["period_id"], int(float(fid_raw))))  # type: ignore[arg-type]
        except KeyError:
            continue

        passer_xy = (float(row["start_x"]), float(row["start_y"]))
        if flip[j]:
            passer_xy = (FIELD_LENGTH - passer_xy[0], FIELD_WIDTH - passer_xy[1])
        # `_ungated_cheap_identity=True` is REQUIRED: production gates the cheap identity to
        # None, so without it every row would compare None against a real id and this script
        # would report agreement 0.0 -- measuring the gate, not the cheap path.
        # Spelled out, not splatted: a `**common` dict widens to its value union and the checker
        # then tries to bind a GoalMap to `method` / `decision_rule` / `pitch_control_cache`.
        cheap = _compute_cover_shadow_dict(
            frame_data, passer_xy, tid, xt, goal_map=goal_map, detailed=False, _ungated_cheap_identity=True
        )
        exact = _compute_cover_shadow_dict(frame_data, passer_xy, tid, xt, goal_map=goal_map, detailed=True)
        if cheap is None or exact is None:
            continue

        # THE HARM, in EXACT-path units. `max_def_exact - max_def_cheap` is the WRONG quantity: the
        # two paths compute different things (PC Voronoi counterfactual vs accumulated lane
        # reception deltas), so their maxima differ in magnitude whether or not the argmax agrees.
        # The decision question is "when the cheap path names A and the exact path names B, how much
        # worse is A?" -- which is only answerable on ONE scale. So score the cheap path's nominee
        # through the EXACT path and difference it against the exact maximum. >= 0 by construction.
        pid_cheap = cheap["max_single_defender_player_id"]
        exact_of_cheap_pid = None
        if pid_cheap is not None:
            exact_of_cheap_pid = float(
                compute_blocking_score(
                    frame_data,
                    tid,
                    xt,
                    goal_map=goal_map,
                    defenders_to_remove=[pid_cheap],
                ).blocking_score
            )

        records.append(
            {
                "match_id": match_id,
                "action_id": str(aid),
                "n_lane_blockers": _lane_blocker_count(frame_data, tid, goal_map),
                "max_def_cheap": float(cheap["max_single_defender_blocking_score"]),
                "max_def_exact": float(exact["max_single_defender_blocking_score"]),
                "exact_score_of_cheap_pid": exact_of_cheap_pid,
                "pid_cheap": pid_cheap,
                "pid_exact": exact["max_single_defender_player_id"],
            }
        )
    return records


def summarize(df: pd.DataFrame) -> dict:
    """Compute the three reported quantities. Pure -- takes the records, returns the report."""
    # QUALIFICATION: >= 2 lane blockers (with one, both paths trivially agree and the measurement
    # would be inflated by cases that cannot disagree) and a real attribution on the exact path.
    qual = df[(df["n_lane_blockers"] >= 2) & (df["max_def_exact"] > TOL_ATTRIB)].copy()

    # String-compare identities: provider player ids are str on kloppy-family providers and numeric
    # on others, and `None` must compare equal to `None` without becoming NaN-unequal.
    def _key(v):
        return "<NA>" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)

    qual["agree"] = [_key(a) == _key(b) for a, b in zip(qual["pid_cheap"], qual["pid_exact"], strict=True)]
    n, k = len(qual), int(qual["agree"].sum())
    lo, hi = wilson_interval(k, n)

    disagree = qual[~qual["agree"]].copy()
    # Harm in EXACT-path units: how much blocking score the cheap path's nominee gives up against
    # the exact path's winner. Both terms are exact-path scores, so this is a like-for-like gap.
    harm = (disagree["max_def_exact"] - disagree["exact_score_of_cheap_pid"]).dropna()
    # As a FRACTION of the exact maximum -- an absolute gap of 0.9 means something very different
    # against a max of 1.0 than against a max of 90.
    denom = disagree.loc[harm.index, "max_def_exact"]
    harm_frac = (harm / denom.where(denom > 0)).dropna()

    all_max_def = df["max_def_exact"].to_numpy(dtype=float)
    nonzero = np.sort(all_max_def[all_max_def > 0.0])
    # The separability question: is there a gap between the zero cluster and the smallest real
    # values? Report the smallest non-zero magnitudes so TOL_ATTRIB is set from data, not asserted.
    smallest_nonzero = [float(v) for v in nonzero[:20]]

    return {
        "n_actions_scored": len(df),
        "n_qualifying": n,
        "agreement_rate": (k / n) if n else None,
        "agreement_wilson95": [lo, hi],
        "n_agree": k,
        "n_disagree": int(n - k),
        "disagreement_harm_exact_units": {
            "n": len(harm),
            "median": float(harm.median()) if len(harm) else None,
            "p90": float(harm.quantile(0.90)) if len(harm) else None,
            "max": float(harm.max()) if len(harm) else None,
        },
        "disagreement_harm_as_fraction_of_max": {
            "median": float(harm_frac.median()) if len(harm_frac) else None,
            "p90": float(harm_frac.quantile(0.90)) if len(harm_frac) else None,
        },
        "max_def_distribution": {
            "n_exactly_zero": int((all_max_def == 0.0).sum()),
            "n_nonzero": len(nonzero),
            "smallest_20_nonzero": smallest_nonzero,
            "percentiles": {str(p): float(np.percentile(all_max_def, p)) for p in (50, 75, 90, 99)}
            if len(all_max_def)
            else {},
        },
        "tol_attrib_in_effect": TOL_ATTRIB,
        "decision": _decide(n, (k / n) if n else 0.0),
    }


def _decide(n: int, rate: float) -> str:
    """Apply the PRE-REGISTERED rule. Thresholds are constants here so they cannot drift."""
    if n < 100:
        return f"INSUFFICIENT_N (n={n} < 100) -- do not ship on this evidence"
    if rate >= 0.9:
        return f"SHIP (agreement {rate:.4f} >= 0.90 at n={n})"
    return f"DO_NOT_SHIP_SILENTLY (agreement {rate:.4f} < 0.90) -- gate to detailed=True or drop"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="gradientsports")
    ap.add_argument("--max-matches", type=int, default=4)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None, help="Write the JSON report here.")
    ap.add_argument("--records-out", default=None, help="Write the per-action CSV here.")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a dev run from a modified tree. The artifact is still marked dirty.",
    )
    args = ap.parse_args()

    # FIRST, before paying for any corpus work: `git rev-parse HEAD` returns the same SHA whether
    # or not the tree is modified, so stamping the bare SHA would record a commit that does not
    # describe the code that ran. Enforcement lives in main(), not in the work functions -- a
    # `run()` that refuses on a dirty checkout cannot be tested without mocking git.
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    all_actions, loaded = [], []
    for provider, match_id, actions, frames, home in load_matches(
        providers=[args.provider],
        max_per_provider=args.max_matches,
        tracking_limit=args.tracking_limit,
        cache_dir=args.cache_dir,
    ):
        print(f"loaded {provider}/{match_id}: {len(actions)} actions", file=sys.stderr)
        all_actions.append(actions)
        loaded.append((match_id, actions, frames, home))

    if not loaded:
        print("no matches loaded", file=sys.stderr)
        return 1

    # Fit xT on the whole loaded corpus -- one surface for every match, so the identity comparison
    # is not confounded by a per-match threat surface.
    xt = ExpectedThreat()
    xt.fit(pd.concat(all_actions, ignore_index=True))

    # WHY THIS DRIVER CANNOT STREAM ITS CORPUS, unlike its neighbours. The xT surface is fit ONCE
    # on every loaded match's actions -- deliberately, so the identity comparison is not confounded
    # by a per-match threat surface -- which is a genuine cross-item barrier: no match can be
    # measured until all of them have been read. So `loaded` stays materialised (today's memory
    # profile, unchanged) and `for_each` walks it. The load is therefore re-paid on a resume; the
    # per-match MEASUREMENT is what it skips, and that is where the time goes -- the exact path was
    # measured at 98-125 ms per action over ~1000 actions a match.
    res = for_each(
        loaded,
        key=lambda item: (str(args.provider), str(item[0])),
        work=lambda item: pd.DataFrame.from_records(measure_match(item[1], item[2], item[3], xt, match_id=item[0])),
        shard_root=Path(args.out).parent / "shards" if args.out else Path("cover_shadow_argmax_shards"),
        # Unlike every other driver in this cycle, the corpus SELECTORS belong in the token here,
        # and the reason is specific: the xT surface above is fit on exactly this corpus and is an
        # input to both scored paths. A `--max-matches 8` run reusing shards computed against a
        # 4-match surface would silently mix two threat models in one agreement rate. The match ids
        # are declared rather than the selector so the digest describes what was actually loaded
        # (`--max-matches` picks the first N, and an excluded match changes the set).
        #
        # `passer_reprojected` is declared because ADR-028 RC1 changes the CHEAP path's nominee on
        # away rows: shards written before this fix must not be reused after it.
        token_inputs={
            "provider": args.provider,
            "match_ids": sorted(str(m) for m, _a, _f, _h in loaded),
            "tracking_limit": args.tracking_limit,
            "xt_surface": "corpus-fit",
            "tol_attrib": TOL_ATTRIB,
            "passer_reprojected": "adr028-rc1",
        },
        tag="cover_shadow_argmax",
        label="match",
    )
    if res.failures:
        print(f"{len(res.failures)} match(es) failed: {res.failures}", file=sys.stderr)
        return 1

    # Combined from THIS PASS'S keys rather than `_driver.reconcile`: there is no partition surface
    # here, so a whole-generation read would fold in matches from a wider earlier run.
    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys) if len(f)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        print("no scoreable actions", file=sys.stderr)
        return 1

    report = summarize(df)
    report["run_commit"] = prov["commit"]
    report["run_tree_dirty"] = prov["dirty"]
    report["run_tree_state"] = prov["tree_state"]
    report["corpus"] = {
        "provider": args.provider,
        "n_matches": len(loaded),
        "match_ids": [m for m, _a, _f, _h in loaded],
        "tracking_limit": args.tracking_limit,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.records_out:
        Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.records_out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
