"""Read-only measure: ELASTIC alignment of the event times, 2026-11-18.

Inputs (read only): a stage root (--stage) and the raw provider cache (--cache-dir, idsse events).
Matches: all idsse, the first 6 gradientsports (sorted). Output: <--out>/<match>.parquet.
Per on-ball action (the families of FAM below), offsets are (time - action time t):
- el_off: ELASTIC align_events_to_frames (default params) elastic_frame_id -> its frame time; el_conf, el_err.
- tf79_off: TF-79 detect_ball_contacts (default params), the actor's resolved contact nearest to t in [t-4, t+4].
- yard_off: the release yardstick of release_check.py (the same rule).
- cf_off (idsse only): the frame time of the DFL CalculatedFrame of the action's event.
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from silly_kicks.tracking._ball_contact import detect_ball_contacts
from silly_kicks.tracking._elastic_sync import align_events_to_frames

ROOT = CACHE = OUT = ""  # set from the arguments (the workers fork after that)
FAM = {0: "pass", 1: "cross", 2: "throw_in", 3: "freekick", 4: "freekick", 5: "corner", 6: "corner", 11: "shot",
       12: "shot", 13: "shot", 18: "clearance", 22: "goalkick"}


def id_key(series):
    if series.dtype.kind in "iuf" or str(series.dtype) in ("Int64", "Float64"):
        return series.astype("Float64").round().astype("Int64").astype(str)
    return series.astype(str)


def nearest(times, t, span=4.0):
    w = times[(times >= t - span) & (times <= t + span)] if len(times) else times
    if len(w) == 0:
        return np.nan
    return float(w[np.argmin(np.abs(w - t))] - t)


def yardstick(a, f):
    """The release measure of release_check.py, unchanged. Returns {action_id: offset}."""
    res = {}
    f = f[f["x"].notna()]
    isb = f["is_ball"].astype(bool)
    ball = f[isb].sort_values(["period_id", "time_seconds"])
    pl = f[~isb].copy()
    pl["pid_key"] = id_key(pl["player_id"])
    for per, act in a.groupby("period_id"):
        bp = ball[ball["period_id"] == per]
        tb = bp["time_seconds"].to_numpy(dtype=float)
        if len(tb) < 50:
            continue
        bx, by = bp["x"].to_numpy(dtype=float), bp["y"].to_numpy(dtype=float)
        m = max(1, int(round(1.0 / float(np.median(np.diff(tb))))))
        tracks = {pid: (g["time_seconds"].to_numpy(dtype=float), g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float))
                  for pid, g in pl[pl["period_id"] == per].sort_values("time_seconds").groupby("pid_key", sort=False)}
        for aid, ts, pid in zip(act["action_id"], act["time_seconds"].to_numpy(dtype=float), act["pid_key"]):
            if np.isnan(ts) or pid not in tracks:
                continue
            tm, px, py = tracks[pid]
            lo, hi = np.searchsorted(tb, ts - 4.0), np.searchsorted(tb, ts + 4.0)
            if hi - lo < 5:
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
            if rel:
                rel = np.asarray(rel)
                res[aid] = float(rel[np.argmin(np.abs(rel - ts))] - ts)
    return res


def calculated_frames(mid):
    from silly_kicks.providers.sportec import parse_dfl_events, parse_dfl_match_info, shape_events_to_native

    d = f"{CACHE}/{mid}"
    mi = parse_dfl_match_info(f"{d}/idsse_{mid}_metadata.xml")
    ev = shape_events_to_native(parse_dfl_events(f"{d}/idsse_{mid}_events.xml", match_info=mi,
                                                 match_id=mid.removeprefix("DFL-MAT-")))
    ev = ev[ev["calculated_frame"].notna()]
    return dict(zip(ev["event_id"].astype(str), ev["calculated_frame"].astype("int64")))


def one_match(pa):
    name = os.path.basename(pa)
    prov = name.split("__")[0]
    acts = pd.read_parquet(pa)
    frames = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
    ftime = (frames[["period_id", "frame_id", "time_seconds"]].drop_duplicates(["period_id", "frame_id"])
             .set_index(["period_id", "frame_id"])["time_seconds"])
    t0 = time.perf_counter()
    al = align_events_to_frames(acts, frames)
    t_el = time.perf_counter() - t0
    t0 = time.perf_counter()
    con = detect_ball_contacts(frames)
    t_tf = time.perf_counter() - t0
    con = con[(con["contact_source"].astype(object) != "unresolved") & con["player_id"].notna()].copy()
    con["pid_key"] = id_key(con["player_id"])
    a = acts[acts["type_id"].isin(list(FAM))].copy()
    a["fam"] = a["type_id"].map(FAM)
    a["pid_key"] = id_key(a["player_id"])
    yd = yardstick(a, frames)
    a = a.merge(al[["action_id", "elastic_frame_id", "elastic_confidence", "elastic_error_seconds"]], on="action_id", how="left")
    el_t = [ftime.get((int(p), int(fr)), np.nan) if pd.notna(fr) else np.nan
            for p, fr in zip(a["period_id"], a["elastic_frame_id"])]
    a["el_off"] = np.asarray(el_t, dtype=float) - a["time_seconds"].to_numpy(dtype=float)
    cby = {(int(p), k): np.sort(g["time_seconds"].to_numpy(dtype=float)) for (p, k), g in con.groupby(["period_id", "pid_key"])}
    a["tf79_off"] = [nearest(cby.get((int(p), k), np.array([])), float(t)) if pd.notna(t) else np.nan
                     for p, k, t in zip(a["period_id"], a["pid_key"], a["time_seconds"])]
    a["yard_off"] = a["action_id"].map(yd)
    a["cf_off"] = np.nan
    if prov == "idsse":
        cf = calculated_frames(name.split("__")[1].removesuffix(".parquet"))
        cft = [ftime.get((int(p), cf[str(e)]), np.nan) if str(e) in cf else np.nan
               for p, e in zip(a["period_id"], a["original_event_id"])]
        a["cf_off"] = np.asarray(cft, dtype=float) - a["time_seconds"].to_numpy(dtype=float)
    a["prov"], a["match"] = prov, name
    keep = ["prov", "match", "action_id", "period_id", "fam", "time_seconds", "elastic_confidence", "elastic_error_seconds",
            "el_off", "tf79_off", "yard_off", "cf_off"]
    a[keep].to_parquet(f"{OUT}/{name}")
    return name, len(acts), len(al), round(t_el, 1), round(t_tf, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ELASTIC alignment of the event times (idsse, 6 gradientsports).")
    ap.add_argument("--stage", required=True, help="stage root (holds _actions/ and shards/)")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--workers", type=int, required=True)
    args = ap.parse_args()
    ROOT, CACHE, OUT = args.stage, os.path.join(args.cache_dir, "idsse"), args.out
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{ROOT}/_actions/idsse__*.parquet")) + sorted(glob.glob(f"{ROOT}/_actions/gradientsports__*.parquet"))[:6]
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(one_match, files):
            print("match %s actions %d aligned %d elastic_s %.1f tf79_s %.1f" % r, flush=True)
    print("DONE", flush=True)
