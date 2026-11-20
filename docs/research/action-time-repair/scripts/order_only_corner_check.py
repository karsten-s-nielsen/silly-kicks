"""Read-only (spec r8, change 2): the 690 "after"-only SkillCorner corners and the rest runs in their bracket.

Bracket: [time of frame_end of a (the possession with the "after" corner marker), time of frame_start of b (the next
possession)]. In it: every rest-and-release run (>= 0.5 s of detected ball frames within 0.5 m of the run's first point;
a frame > 2 m away within 1 s after), with its distance to the nearest corner flag. Per corner: the follower kind (b's
"before" marker kind, or "goal" / "none"), the number of flag runs (first point <= 2 m of a flag) and whether a
rest-and-release run of any place starts after the first flag run's release.
Input (read only): the raw provider cache (--cache-dir, SkillCorner). Output: stdout and <--out> (a parquet file).
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

CACHE = ""  # set from the arguments (the workers fork after that)
FLAGS = np.array([[0.0, 0.0], [0.0, 68.0], [105.0, 0.0], [105.0, 68.0]])
KINDS = ("corner", "free_kick", "goal_kick", "throw_in")


def kind(s):
    s = str(s)
    for k in KINDS:
        if s.startswith(k + "_") and s.rsplit("_", 1)[1] in ("for", "against"):
            return k
    if s.startswith("goal_"):
        return "goal"
    if s.startswith("penalty"):
        return "penalty"
    return "none" if s in ("", "<NA>", "None", "nan") else "other:" + s


def runs(t, x, y, fps=10):
    """All maximal rest runs (start index, end index) followed by a release; non-overlapping, in time order."""
    n = max(2, int(round(0.5 * fps)))
    out = []
    s = 0
    while s <= len(t) - n:
        if t[s + n - 1] - t[s] > 0.5 + 1.5 / fps or not np.all(np.hypot(x[s:s + n] - x[s], y[s:s + n] - y[s]) < 0.5):
            s += 1
            continue
        e = s + n - 1
        while e + 1 < len(t) and np.hypot(x[e + 1] - x[s], y[e + 1] - y[s]) < 0.5 and t[e + 1] - t[e] < 1.5 / fps + 0.1:
            e += 1
        after = (t > t[e]) & (t <= t[e] + 1.0)
        if np.any(np.hypot(x[after] - x[s], y[after] - y[s]) > 2.0):
            d = float(np.min(np.hypot(FLAGS[:, 0] - x[s], FLAGS[:, 1] - y[s])))
            out.append((t[s], t[e], d))
        s = e + 1
    return out


def one(m):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    fe = glob.glob(f"{CACHE}/{m}/*_events.parquet")
    ft = glob.glob(f"{CACHE}/{m}/*_tracking.parquet")
    if not fe or not ft:
        return []
    ev = pd.read_parquet(fe[0], columns=["event_type", "frame_start", "frame_end", "period",
                                         "game_interruption_before", "game_interruption_after"])
    pp = ev[(ev["event_type"] == "player_possession") & ev["frame_start"].notna() & ev["frame_end"].notna()]
    pp = pp.sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    gia = pp["game_interruption_after"].astype("string").fillna("").map(kind)
    gib = pp["game_interruption_before"].astype("string").fillna("").map(kind)
    idx = [i for i in range(len(pp) - 1)
           if gia[i] == "corner" and pp["period"].iloc[i + 1] == pp["period"].iloc[i] and gib[i + 1] != "corner"]
    if not idx:
        return []
    tb = pq.read_table(ft[0], columns=["frame", "ball_data"])
    bd = tb["ball_data"].combine_chunks()
    fr = tb["frame"].to_numpy()
    bx = np.asarray(pc.fill_null(bd.field("x"), np.nan).to_numpy(zero_copy_only=False), float) + 52.5
    by = np.asarray(pc.fill_null(bd.field("y"), np.nan).to_numpy(zero_copy_only=False), float) + 34.0
    det = np.asarray(pc.fill_null(bd.field("is_detected"), False).to_numpy(zero_copy_only=False), bool)
    rows = []
    for i in idx:
        lo, hi = int(pp["frame_end"].iloc[i]), int(pp["frame_start"].iloc[i + 1])
        msk = (fr >= lo) & (fr <= hi) & det & np.isfinite(bx)
        t = fr[msk] / 10.0
        rr = runs(t, bx[msk], by[msk])
        flag = [r for r in rr if r[2] <= 2.0]
        later = bool(flag) and any(r[0] > flag[0][1] for r in rr)
        rows.append({"match": m, "follower": gib[i + 1], "gap_s": (hi - lo) / 10.0, "n_runs": len(rr),
                     "n_flag": len(flag), "later_run": later,
                     "first_flag_is_first_run": bool(flag) and rr[0] == flag[0]})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The rest runs in the gaps of the order-only SkillCorner corners.")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--out", required=True, help="output parquet file")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    CACHE = os.path.join(args.cache_dir, "skillcorner")
    ms = sorted(os.listdir(CACHE))
    with Pool(args.workers) as p:
        d = pd.DataFrame([r for rs in p.map(one, ms) for r in rs])
    d.to_parquet(args.out)
    print("order-only corners with tracking:", len(d))
    d["flag_class"] = np.select([d.n_flag == 0, d.n_flag == 1, d.n_flag >= 2], ["0", "1", "2+"], default="?")
    print(pd.crosstab(d.follower, d.flag_class, margins=True))
    f = d[d.n_flag >= 1]
    print("with a flag run: later run after the first flag run:")
    print(pd.crosstab(f.follower, f.later_run, margins=True))
    print("with a flag run: the first flag run is the first run of the bracket:", int(f.first_flag_is_first_run.sum()), "of", len(f))
    print("gap (s) quantiles:", d.gap_s.quantile([0.1, 0.5, 0.9]).round(1).tolist())
    print("DONE")
