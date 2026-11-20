"""Spec section 8 "the blank frame features" on a repaired stage, part 2.

Runs `run_tracking_features` with its default families on each match of a stage (--stage, read only) that has an
unmeasured row (a match with none is listed with the count 0; the producer blanks nothing there) and counts, per
added column, the "placeholder" and "order_only" rows whose value is not missing (must be 0 for every family
column). The four pointer columns are checked for their values (frame_id <NA>, time_offset_seconds NaN,
link_quality_score NaN, n_candidate_frames 0); the two keeper-identity columns keep their value. Lists each dtype.
"""

import argparse
import glob
import json
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import is_time_unmeasured
from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier, run_tracking_features
from silly_kicks.xthreat import ExpectedThreat

warnings.simplefilter("ignore")
ROOT = ""  # set from the arguments (the workers fork after that)
POINTER = ("frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score")
IDENTITY = ("defending_gk_player_id", "defending_gk_team_id")


XT = None  # fitted once in main on the stage matches; inherited by the fork workers
SHOTS = [spadlconfig.actiontype_id[t] for t in ("shot", "shot_freekick", "shot_penalty")]


def one(name):
    a = pd.read_parquet(f"{ROOT}/_actions/{name}")
    if not np.asarray(is_time_unmeasured(a), dtype=bool).any():  # the producer blanks nothing here (no row to count)
        return {"match": name, "n_unmeasured": 0, "families_run": [], "skipped": "producer not run: no unmeasured row",
                "cols": {}}
    # xg_column input of add_defensive_credit: a stand-in value on shots (the check is the blanking, not the value)
    a["xg_trial"] = np.where(a["type_id"].isin(SHOTS), 0.1, np.nan)
    fr = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
    fr = derive_team_in_possession(fr, infer_ball_carrier(fr))  # add_das needs the frames' team_in_possession
    homes = json.load(open(f"{ROOT}/home_teams.json"))
    gid = str(a["game_id"].iloc[0])
    home = homes.get(gid, homes.get(name.split("__")[1].removesuffix(".parquet").removeprefix("DFL-MAT-")))
    out, rep = run_tracking_features(a, fr, home_team_id=home, xt=XT, xg_column="xg_trial")
    um = pd.Series(is_time_unmeasured(out), index=out.index).to_numpy(bool)
    added = [c for c in out.columns if c not in a.columns]
    res = {"match": name, "n_unmeasured": int(um.sum()), "families_run": sorted(f for f, s in rep.family_status.items() if s == "ran"),
           "skipped": {f: s for f, s in rep.family_status.items() if s != "ran"},
           "cols": {}}
    for c in added:
        col = out.loc[um, c]
        if c == "frame_id":
            bad = int(col.notna().sum())
        elif c == "n_candidate_frames":
            bad = int((col.fillna(-1) != 0).sum())
        elif c in IDENTITY:
            bad = 0  # keeps its value (section 5.3); reported, not counted
        else:
            bad = int(col.notna().sum())
        res["cols"][c] = (str(out[c].dtype), bad)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The blank frame features on the unmeasured rows of a stage.")
    ap.add_argument("--stage", required=True, help="stage root (holds _actions/, shards/ and home_teams.json)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ROOT = args.stage
    names = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/_actions/*.parquet"))
    XT = ExpectedThreat().fit(pd.concat([pd.read_parquet(f"{ROOT}/_actions/{n}") for n in names], ignore_index=True))
    with Pool(args.workers) as pool:
        rs = pool.map(one, names)
    total = 0
    dtypes = {}
    for r in rs:
        bad = {c: v[1] for c, v in r["cols"].items() if v[1]}
        total += sum(bad.values())
        for c, (dt, _b) in r["cols"].items():
            dtypes.setdefault(c, set()).add(dt)
        print(f"BLANK {r['match']} unmeasured rows {r['n_unmeasured']} added cols {len(r['cols'])} "
              f"families ran {len(r['families_run'])} skipped {r['skipped']} non-missing on unmeasured rows: {bad or 0}")
    print("BLANK total non-missing family/pointer values on unmeasured rows (must 0):", total)
    for c in sorted(dtypes):
        print("DTYPE", c, sorted(dtypes[c]))
