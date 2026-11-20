"""The idsse action times on a repaired stage, the 7 idsse matches.

Inputs (read only): the repaired stage (--stage), the stage before the repair (--old-stage, the provider time t of
each event) and the raw provider cache (--cache-dir, idsse events). Reuses the measures of elastic_check.py (the
script in this folder): the release yardstick and the DFL CalculatedFrame map.
- Shots: of the 155 shots with a CalculatedFrame, the share whose repaired time_seconds is within 0.5 s of it (>= 0.93).
- Families: per family, the share of rows with time_source "elastic" (>= investigation found - 0.02) and, of the elastic
  rows with a yardstick release, the share within 0.5 s of it (>= investigation yard(el) - 0.02).
"""

import argparse
import glob
import os
from multiprocessing import Pool

import elastic_check as E  # the script in this folder (the folder of this script is on sys.path)
import numpy as np
import pandas as pd

ROOT = OLD = ""  # set from the arguments (the workers fork after that)
BASE = {  # the investigation on the f5fc2e2b stage (elastic_check.py rows): (found, yard(el))
    "shot": (0.988, 0.938), "pass": (0.995, 0.961), "cross": (0.991, 0.991), "corner": (0.860, 1.000),
    "freekick": (0.924, 0.987), "throw_in": (0.951, 0.941), "goalkick": (0.896, 0.975), "clearance": (0.994, 0.972),
}


def one(pa):
    name = os.path.basename(pa)
    acts = pd.read_parquet(pa)
    frames = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
    ftime = (frames[["period_id", "frame_id", "time_seconds"]].drop_duplicates(["period_id", "frame_id"])
             .set_index(["period_id", "frame_id"])["time_seconds"])
    a = acts[acts["type_id"].isin(list(E.FAM))].copy()
    a["fam"] = a["type_id"].map(E.FAM)
    a["pid_key"] = E.id_key(a["player_id"])
    # The investigation's yardstick is centred on the PROVIDER time t (elastic_summary.py: |el_off - yard_off|);
    # the provider time comes from the f5fc2e2b stage, keyed on original_event_id (idsse ids are unique).
    old = pd.read_parquet(f"{OLD}/_actions/{name}", columns=["original_event_id", "time_seconds"])
    t_old = a["original_event_id"].astype(str).map(dict(zip(old["original_event_id"].astype(str), old["time_seconds"])))
    a_old = a.assign(time_seconds=t_old.to_numpy(dtype=float))
    yard_t = t_old + a["action_id"].map(E.yardstick(a_old, frames))
    a["yard_off"] = yard_t - a["time_seconds"]  # release time - repaired time
    cf = E.calculated_frames(name.split("__")[1].removesuffix(".parquet"))
    cft = [ftime.get((int(p), cf[str(e)]), np.nan) if str(e) in cf else np.nan
           for p, e in zip(a["period_id"], a["original_event_id"])]
    a["cf_off"] = np.asarray(cft, dtype=float) - a["time_seconds"].to_numpy(dtype=float)
    a["has_cf"] = [str(e) in cf for e in a["original_event_id"]]
    a["match"] = name
    return a[["match", "action_id", "fam", "time_source", "yard_off", "cf_off", "has_cf"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The idsse action times against the CalculatedFrame and the release.")
    ap.add_argument("--stage", required=True, help="the repaired stage root (holds _actions/ and shards/)")
    ap.add_argument("--old-stage", required=True, help="the stage root before the repair (provider times)")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    args = ap.parse_args()
    ROOT, OLD, E.CACHE = args.stage, args.old_stage, os.path.join(args.cache_dir, "idsse")
    files = sorted(glob.glob(f"{ROOT}/_actions/idsse__*.parquet"))
    with Pool(len(files)) as pool:
        d = pd.concat(pool.map(one, files), ignore_index=True)
    print(f"idsse matches {len(files)}, on-ball rows {len(d)}, time_source {d['time_source'].value_counts().to_dict()}")
    shots = d[(d["fam"] == "shot") & d["has_cf"]]
    within = float((shots["cf_off"].abs() <= 0.5).mean())
    print(f"shots with CalculatedFrame: n {len(shots)}; within 0.5 s {within:.3f} (limit 0.93) "
          f"{'PASS' if len(shots) == 155 and within >= 0.93 else 'STOP'}")
    ok_all = True
    for fam, (found0, yard0) in BASE.items():
        g = d[d["fam"] == fam]
        el = g[g["time_source"] == "elastic"]
        share = len(el) / len(g) if len(g) else float("nan")
        y = el[el["yard_off"].notna()]
        agree = float((y["yard_off"].abs() <= 0.5).mean()) if len(y) else float("nan")
        ok = share >= found0 - 0.02 and agree >= yard0 - 0.02
        ok_all &= ok
        print(f"elastic share {fam:10s} n {len(g):5d} elastic {share:.3f} (limit {found0 - 0.02:.3f}) | "
              f"yard n {len(y):5d} agree {agree:.3f} (limit {yard0 - 0.02:.3f}) {'PASS' if ok else 'STOP'}")
    print("elastic share and release agreement, all families", "PASS" if ok_all else "STOP")
