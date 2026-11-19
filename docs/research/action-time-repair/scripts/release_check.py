"""Read-only measure: the ball's release after an on-ball action, 2026-11-18.

Input (read only): a stage root (--stage). Output: <--out>/<match>.parquet + summary.

Per on-ball action (the families of FAM below):
- yard_off: the release yardstick. Release = a frame with the ball within 2 m of the actor,
  then more than 3 m away inside the next second; the release nearest to t in [t-4, t+4]. Offset = release - t.
- tf79_off: TF-79 detect_ball_contacts (default params) on the match frames; the RESOLVED contact attributed to the
  actor nearest to t in [t-4, t+4]. Offset = contact - t.
- tf79_any_off: the same, any resolved player (attribution check).
Workers: one match per task, K processes, one output generation.
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from silly_kicks.tracking._ball_contact import detect_ball_contacts

ROOT = OUT = ""  # set from the arguments (the workers fork after that)
FAM = {0: "pass", 1: "cross", 2: "throw_in", 3: "freekick", 4: "freekick", 5: "corner", 6: "corner", 11: "shot",
       12: "shot", 13: "shot", 18: "clearance", 22: "goalkick"}
FRAME_COLS = ["period_id", "frame_id", "time_seconds", "frame_rate", "is_ball", "player_id", "x", "y", "z", "z_source"]


def id_key(series):
    if series.dtype.kind in "iuf" or str(series.dtype) in ("Int64", "Float64"):
        return series.astype("Float64").round().astype("Int64").astype(str)
    return series.astype(str)


def nearest(times, t, span=4.0):
    if len(times) == 0:
        return np.nan
    w = times[(times >= t - span) & (times <= t + span)]
    if len(w) == 0:
        return np.nan
    return float(w[np.argmin(np.abs(w - t))] - t)


def one_match(pa):
    name = os.path.basename(pa)
    prov = name.split("__")[0]
    a = pd.read_parquet(pa)
    a = a[a["type_id"].isin(list(FAM))].copy()
    a["fam"] = a["type_id"].map(FAM)
    a["pid_key"] = id_key(a["player_id"])
    fall = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0], columns=FRAME_COLS)
    # TF-79 on the whole match.
    con = detect_ball_contacts(fall)
    con = con[(con["contact_source"].astype(object) != "unresolved") & con["player_id"].notna()].copy()
    con["pid_key"] = id_key(con["player_id"])
    f = fall[fall["x"].notna()]
    isb = f["is_ball"].astype(bool)
    ball = f[isb].sort_values(["period_id", "time_seconds"])
    pl = f[~isb].copy()
    pl["pid_key"] = id_key(pl["player_id"])
    rows = []
    for per, act in a.groupby("period_id"):
        bp = ball[ball["period_id"] == per]
        tb = bp["time_seconds"].to_numpy(dtype=float)
        cper = con[con["period_id"] == per]
        c_all = np.sort(cper["time_seconds"].to_numpy(dtype=float))
        c_by = {k: np.sort(g["time_seconds"].to_numpy(dtype=float)) for k, g in cper.groupby("pid_key")}
        if len(tb) < 50:
            for ts, fam in zip(act["time_seconds"], act["fam"]):
                rows.append((prov, name, int(per), fam, float(ts), np.nan, "no_ball_period", np.nan, np.nan))
            continue
        bx, by = bp["x"].to_numpy(dtype=float), bp["y"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(tb)))
        m = max(1, int(round(1.0 / dt)))
        tracks = {}
        for pid, g in pl[pl["period_id"] == per].sort_values("time_seconds").groupby("pid_key", sort=False):
            tracks[pid] = (g["time_seconds"].to_numpy(dtype=float), g["x"].to_numpy(dtype=float),
                           g["y"].to_numpy(dtype=float))
        for ts, pid, fam in zip(act["time_seconds"].to_numpy(dtype=float), act["pid_key"].to_numpy(),
                                act["fam"].to_numpy()):
            tf = nearest(c_by.get(pid, np.array([])), ts) if not np.isnan(ts) else np.nan
            tfa = nearest(c_all, ts) if not np.isnan(ts) else np.nan
            if np.isnan(ts) or pid not in tracks:
                rows.append((prov, name, int(per), fam, ts, np.nan, "no_player", tf, tfa))
                continue
            tm, px, py = tracks[pid]
            lo, hi = np.searchsorted(tb, ts - 4.0), np.searchsorted(tb, ts + 4.0)
            if hi - lo < 5:
                rows.append((prov, name, int(per), fam, ts, np.nan, "no_ball", tf, tfa))
                continue
            tt = tb[lo:hi]
            j = np.clip(np.searchsorted(tm, tt), 0, len(tm) - 1)
            ok = np.abs(tm[j] - tt) < 0.11
            dist = np.hypot(bx[lo:hi] - px[j], by[lo:hi] - py[j])
            dist[~ok] = np.nan
            near = dist <= 2.0
            rel = []
            for i in np.flatnonzero(near):
                nxt = dist[i + 1:i + 1 + m]
                if len(nxt) and not near[i + 1] and np.nanmax(np.where(np.isnan(nxt), -1.0, nxt)) > 3.0:
                    rel.append(tt[i])
            if not rel:
                rows.append((prov, name, int(per), fam, ts, np.nan, "no_release", tf, tfa))
                continue
            rel = np.asarray(rel)
            rows.append((prov, name, int(per), fam, ts, float(rel[np.argmin(np.abs(rel - ts))] - ts), "ok", tf, tfa))
    d = pd.DataFrame(rows, columns=["prov", "match", "period", "fam", "t", "yard_off", "yard_state", "tf79_off",
                                    "tf79_any_off"])
    d.to_parquet(f"{OUT}/{name}")
    return name, len(d), len(con)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The ball's release after an on-ball action.")
    ap.add_argument("--stage", required=True, help="stage root (holds _actions/ and shards/)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--providers", required=True, help="comma-separated, e.g. skillcorner,idsse")
    ap.add_argument("--workers", type=int, required=True)
    args = ap.parse_args()
    ROOT, OUT = args.stage, args.out
    os.makedirs(OUT, exist_ok=True)
    provs = args.providers.split(",")
    k = args.workers
    files = []
    for p in provs:
        files += sorted(glob.glob(f"{ROOT}/_actions/{p}__*.parquet"))
    todo = [f for f in files if not os.path.exists(f"{OUT}/{os.path.basename(f)}")]
    print("matches", len(files), "todo", len(todo), flush=True)
    with Pool(k) as pool:
        for name, n, nc in pool.imap_unordered(one_match, todo):
            print(name, n, nc, flush=True)
    print("DONE", flush=True)
