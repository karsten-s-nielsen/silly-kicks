"""Read-only: for each "after"-only restart row B (order-only time), does the window of an earlier opener reach past B's time
if B is skipped by the phase reader (option i)? Openers: derived corner / free-kick / goal-kick rows with a first possession
(paired or "before"-only). Times on the frame clock (frame_start / 10), same period only. 10 s window, ended by the next opener.
Input (read only): the raw provider cache (--cache-dir, SkillCorner). Output: stdout.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

CACHE = ""  # set from the arguments (the workers fork after that)
SP = ("corner", "free_kick", "goal_kick")
ALL = SP + ("throw_in",)
W = 10.0


def kind(s):
    for k in ALL:
        if s.startswith(k + "_") and s.rsplit("_", 1)[1] in ("for", "against"):
            return k
    return ""


def one(m):
    f = glob.glob(f"{CACHE}/{m}/*_events.parquet")
    if not f:
        return None
    ev = pd.read_parquet(f[0], columns=["event_type", "frame_start", "period", "game_interruption_before", "game_interruption_after"])
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    gib = pp["game_interruption_before"].astype("string").fillna("").map(kind)
    gia = pp["game_interruption_after"].astype("string").fillna("").map(kind)
    per = pp["period"].to_numpy()
    t = pp["frame_start"].to_numpy() / 10.0
    n = len(pp)
    rows = []  # (period, time, kind, order_only)
    for i in range(n):
        if gib[i]:  # restart with a first possession i (paired or before-only)
            rows.append((per[i], t[i] - 0.02, gib[i], False))
        if gia[i] and i + 1 < n and per[i + 1] == per[i] and gib[i + 1] != gia[i]:
            rows.append((per[i], t[i + 1] - 0.03, gia[i], True))  # after-only
    d = pd.DataFrame(rows, columns=["period", "t", "kind", "oo"])
    out = {"oo_sp": 0, "oo_all": 0, "reach": 0, "reach_s": 0.0, "own_s": 0.0}
    for p, g in d.groupby("period"):
        g = g.sort_values("t", kind="mergesort")
        op_all = g[g["kind"].isin(SP)]  # today's rule on the repaired stage: every SP row opens
        op_i = op_all[~op_all["oo"]]  # option i: order-only rows skipped
        ti = op_i["t"].to_numpy()
        for _, b in g[g["oo"]].iterrows():
            out["oo_all"] += 1
            out["oo_sp"] += int(b["kind"] in SP)
            j = np.searchsorted(ti, b["t"], side="right") - 1
            if j < 0:
                continue
            out.setdefault("gaps", []).append(b["t"] - ti[j])
            end = ti[j] + W
            if j + 1 < len(ti):
                end = min(end, ti[j + 1])
            if end > b["t"]:
                out["reach"] += 1
                out["reach_s"] += end - b["t"]
        # seconds labelled by order-only windows today (repaired stage, no skip)
        ta = op_all["t"].to_numpy()
        oo = op_all["oo"].to_numpy()
        for k in range(len(ta)):
            if oo[k]:
                e = ta[k] + W
                if k + 1 < len(ta):
                    e = min(e, ta[k + 1])
                out["own_s"] += e - ta[k]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The order-only SkillCorner rows and the phase windows.")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    CACHE = os.path.join(args.cache_dir, "skillcorner")
    ms = sorted(os.listdir(CACHE))
    with Pool(args.workers) as pl:
        rs = [r for r in pl.map(one, ms) if r]
    gaps = np.concatenate([np.asarray(r.pop("gaps", []), dtype=float) for r in rs])
    print("gap from the previous opener to the order-only time: n", len(gaps), "min", gaps.min(), "q01", np.quantile(gaps, 0.01), "median", np.median(gaps), "share < 10 s", (gaps < 10).mean())
    s = pd.DataFrame(rs).sum()
    print(f"matches {len(rs)}")
    print(f"order-only rows: all kinds {int(s.oo_all)}, of which corner/free kick/goal kick {int(s.oo_sp)}")
    print(f"seconds labelled by order-only windows if they open (no skip): {s.own_s:.1f}")
    print(f"option i (skip): order-only rows inside an earlier opener's open window {int(s.reach)}; window seconds past the order-only time {s.reach_s:.1f}")
    print("DONE")
