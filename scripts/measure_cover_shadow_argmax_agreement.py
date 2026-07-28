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

from scripts._loader_pining import load_matches
from scripts._provenance import git_provenance, require_clean_tree
from silly_kicks.tracking import link_actions_to_frames
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


def _lane_blocker_count(frame_data, attacking_team_id, home_team_id) -> int:
    """Number of lane blockers, mirroring ``_compute_cover_shadow_dict``'s own construction."""
    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.id_compat import ids_match, same_id

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    defenders_outfield = players[
        (~ids_match(players["team_id"], attacking_team_id)) & (~players["is_goalkeeper"].astype(bool))
    ]
    attackers = players[ids_match(players["team_id"], attacking_team_id)]
    goal_x_own = 105.0 if same_id(attacking_team_id, home_team_id) else 0.0
    man_markers = cs._classify_man_markers(
        defenders_outfield, attackers, goal_x_own=goal_x_own, params=cs.CoverShadowParams()
    )
    return sum(1 for pid in defenders_outfield["player_id"] if pid not in man_markers)


def measure_match(actions, frames, home_team_id, xt, *, match_id: str) -> list[dict]:
    """One record per action that both paths could score."""
    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    records: list[dict] = []
    for _idx, row in actions.iterrows():
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
        common = dict(home_team_id=home_team_id)
        # `_ungated_cheap_identity=True` is REQUIRED: production gates the cheap identity to
        # None, so without it every row would compare None against a real id and this script
        # would report agreement 0.0 -- measuring the gate, not the cheap path.
        cheap = _compute_cover_shadow_dict(
            frame_data, passer_xy, tid, xt, detailed=False, _ungated_cheap_identity=True, **common
        )
        exact = _compute_cover_shadow_dict(frame_data, passer_xy, tid, xt, detailed=True, **common)
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
                    home_team_id=home_team_id,
                    defenders_to_remove=[pid_cheap],
                ).blocking_score
            )

        records.append(
            {
                "match_id": match_id,
                "action_id": str(aid),
                "n_lane_blockers": _lane_blocker_count(frame_data, tid, home_team_id),
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

    records: list[dict] = []
    for match_id, actions, frames, home in loaded:
        rec = measure_match(actions, frames, home, xt, match_id=match_id)
        print(f"  {match_id}: {len(rec)} scored actions", file=sys.stderr)
        records.extend(rec)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("no scoreable actions", file=sys.stderr)
        return 1

    report = summarize(df)
    report["run_commit"] = prov["commit"]
    report["run_tree_dirty"] = prov["dirty"]
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
