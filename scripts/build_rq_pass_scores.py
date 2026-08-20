"""Corpus-pass driver: shard the per-pass cover-shadow + pitch-control scores over GS WC2022.

The EXPENSIVE half of the cover-shadow RQ1 + pass-risk validation cycle: one shard per match, run
once; the two ``validate_*`` consumers read the persisted table and iterate the metrics cheaply.

Spec: docs/superpowers/specs/2026-08-19-cover-shadow-rq1-and-pass-risk-calibration-design.md
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from scripts import _rq_corpus as rqc
from scripts._driver import for_each, reconcile
from scripts._loader_pining import load_matches
from scripts._provenance import git_provenance, require_clean_tree
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking import (
    link_actions_to_frames,
    pitch_control_at_target,
    resolve_defended_goals,
)
from silly_kicks.tracking._cover_shadows import CoverShadowParams, lane_control
from silly_kicks.tracking._geometry import GEOMETRY_VERSION

_SHARD_SCHEMA_VERSION = "rq-scores-2"  # +p_received_{center,left,right}, +n_blocked (margin-based score)
_EMITTED_SHARD_COLUMNS = [
    "game_id",
    "period_id",
    "action_id",
    "frame_id",
    "attacking_team_id",
    "passer_x",
    "passer_y",
    "target_x",
    "target_y",
    "target_source",
    "is_cross",
    "is_completed",
    "is_fail",
    "p_blocked_center",
    "p_blocked_mean",
    "p_blocked_max",
    "p_received_center",
    "p_received_left",
    "p_received_right",
    "n_blocked",
    "is_blocked_majority",
    "control",
]

# Floors relative to GS WC2022's known volume (64 matches x ~900 passes) so a half-empty run trips
# them; the CLI args (Task 4) let the main() unit test lower them to 1.
_MIN_PASSES = 20_000
_MIN_COMPLETED = 12_000


def score_match(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """One tidy shard per match: per played pass, the cover-shadow lane probs + pitch control."""
    links, _ = link_actions_to_frames(actions, frames)  # (pointers, LinkReport) -- ADR-004
    passes = rqc.extract_played_passes(actions, frames, links=links)
    if passes.empty:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    gm = resolve_defended_goals(frames)
    control = pitch_control_at_target(actions, frames, links=links, method="spearman")  # SHARED link
    ctrl_by_aid = pd.Series(control.to_numpy(), index=actions["action_id"].to_numpy())
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}  # index ONCE per match
    recs = []
    for _, p in passes.iterrows():
        fr = by_frame.get(canonical_id(p["frame_id"]))
        if fr is None:  # symmetry with extract_played_passes
            continue
        r = lane_control(
            fr,
            (p["passer_x"], p["passer_y"]),
            (p["target_x"], p["target_y"]),
            goal_map=gm,
            attacking_team_id=p["attacking_team_id"],
        )
        lanes = (r.p_blocked_center, r.p_blocked_left, r.p_blocked_right)
        # n_blocked = the per-lane p_blocked > p_received count (the `blocked_flags` sum the
        # majority rule thresholds); the MARGIN it counts is the discriminating quantity, not
        # the absolute p_blocked intensity -- so the shard carries p_received per lane too.
        n_blocked = (
            int(r.p_blocked_center > r.p_received_center)
            + int(r.p_blocked_left > r.p_received_left)
            + int(r.p_blocked_right > r.p_received_right)
        )
        recs.append(
            {
                **p.to_dict(),
                "p_blocked_center": r.p_blocked_center,
                "p_blocked_mean": float(np.mean(lanes)),
                "p_blocked_max": float(np.max(lanes)),
                "p_received_center": r.p_received_center,
                "p_received_left": r.p_received_left,
                "p_received_right": r.p_received_right,
                "n_blocked": n_blocked,
                "is_blocked_majority": bool(r.is_blocked_majority),
                "control": float(ctrl_by_aid.get(p["action_id"], np.nan)),
            }
        )
    return pd.DataFrame(recs)[_EMITTED_SHARD_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard per-pass cover-shadow + pitch-control scores over GS WC2022.")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--shard-root", type=pathlib.Path, required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--min-passes", type=int, default=_MIN_PASSES)  # injectable floors
    ap.add_argument("--min-completed", type=int, default=_MIN_COMPLETED)
    args = ap.parse_args()

    prov = git_provenance()  # a DICT: prov["commit"], prov["dirty"]
    require_clean_tree(prov, allow_dirty=args.allow_dirty)  # BEFORE any corpus work

    cs = CoverShadowParams()
    token_inputs = {
        "schema": _SHARD_SCHEMA_VERSION,
        "sigma": cs.sigma,
        "lambda_ctrl": cs.lambda_ctrl,
        "pc_method": "spearman",
        "geometry_version": GEOMETRY_VERSION,
    }
    items = load_matches(providers=["gradientsports"], cache_dir=args.cache_dir)
    res = for_each(
        items,
        key=lambda t: t[1],
        work=lambda t: score_match(t[2], t[3]),
        shard_root=args.shard_root,
        token_inputs=token_inputs,
        label="match",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    combined = reconcile(res.shard_dir, args.out / "pass_scores.parquet", tag="all")

    n_passes = len(combined)
    n_completed = int(combined["is_completed"].sum()) if n_passes else 0
    if not (n_passes >= args.min_passes and bool(combined["is_fail"].any()) and bool(combined["is_completed"].any())):
        raise SystemExit(f"vacuous pass set: n_passes={n_passes} (need >= {args.min_passes}, both classes present)")
    if n_completed < args.min_completed:
        raise SystemExit(
            f"too few completed passes for the leakage-free headline: {n_completed} < {args.min_completed}"
        )

    manifest = {
        "schema": _SHARD_SCHEMA_VERSION,
        "n_matches": int(combined["game_id"].nunique()),
        "n_passes": n_passes,
        "n_completed": n_completed,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    manifest.update(res.manifest())
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
