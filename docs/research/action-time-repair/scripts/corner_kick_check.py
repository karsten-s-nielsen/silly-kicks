"""Read-only measure for the action-time spec (the corner kick time, sections 4.4 to 4.6), 2026-11-18.

Corner kick time from library engines, checked against the ball-at-rest rule. Inputs (read only): a stage root
(--stage) and the raw provider cache (--cache-dir, SkillCorner events and tracking).
Output: <--out>/<match>.parquet (one row per corner).

Bracket: SkillCorner = [frame_end of the possession before the restart, frame_start of the first possession]
(the GI frame ids are the tracking frame ids); GS / idsse = [t - 2 s, t + 4 s] around the corner row time t.
Engines (each: the LAST candidate in the bracket whose ball position is within 3 m of a corner flag):
- tf79: detect_ball_contacts (default params), any contact_source (resolved or unresolved).
- tf79r: the same, resolved contacts only (gives the attributed player).
- elastic: ELASTIC _detect_candidate_frames (default ElasticSyncParams), candidates with a feasible player.
Check (the ball at rest at the corner, then it leaves; the ball-at-rest rule of spec section 4.4):
- rest: the last run of >= 0.5 s of ball frames that stay within 0.5 m of the run's first point, with the run's first
  point within 2 m of a flag; release seen: within 1 s after the run a ball frame is more than 2 m from the rest point.
  t_rest = time of the last frame of the run. SkillCorner: only raw-detected ball frames count (is_detected).
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from silly_kicks.tracking._ball_contact import detect_ball_contacts
from silly_kicks.tracking._elastic_sync import ElasticSyncParams, _detect_candidate_frames

ROOT = CACHE = OUT = ""  # set from the arguments (the workers fork after that)
FLAGS = np.array([[0.0, 0.0], [0.0, 68.0], [105.0, 0.0], [105.0, 68.0]])
CORNER_IDS = (5, 6)


def norm_id(v):
    """An id as a comparable string: 363, 363.0 and "363" are equal."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def to_flag(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.min(np.hypot(x[:, None] - FLAGS[None, :, 0], y[:, None] - FLAGS[None, :, 1]), axis=1)


def rest_release(bt, bx, by, bdet, lo, hi, fps):
    """Ball-at-rest-at-the-corner check. Returns (t_rest or nan, rest_seen_at_corner)."""
    m = (bt >= lo) & (bt <= hi) & bdet
    t, x, y = bt[m], bx[m], by[m]
    run = max(2, int(round(0.5 * fps)))
    if len(t) < run + 1:
        return np.nan, False
    best = None
    for s in range(len(t) - run + 1):
        if t[s + run - 1] - t[s] > 0.5 + 1.5 / fps:  # a gap inside the run
            continue
        if np.all(np.hypot(x[s:s + run] - x[s], y[s:s + run] - y[s]) < 0.5) and to_flag([x[s]], [y[s]])[0] <= 2.0:
            best = s
    if best is None:
        return np.nan, False
    e = best + run - 1
    rx, ry = x[best], y[best]
    after = (t > t[e]) & (t <= t[e] + 1.0)
    seen = bool(np.any(np.hypot(x[after] - rx, y[after] - ry) > 2.0))
    return (float(t[e]) if seen else np.nan), True


def last_near_flag(times, xs, ys, lo, hi):
    m = (times >= lo) & (times <= hi)
    if not m.any():
        return np.nan, None
    idx = np.flatnonzero(m)
    near = idx[to_flag(xs[idx], ys[idx]) <= 3.0]
    if not len(near):
        return np.nan, None
    j = near[np.argmax(times[near])]
    return float(times[j]), j


def sc_brackets(mid, ftime):
    ev = pd.read_parquet(glob.glob(f"{CACHE}/{mid}/*_events.parquet")[0],
                         columns=["event_type", "frame_start", "frame_end", "team_id", "game_interruption_before"])
    pp = ev[(ev["event_type"] == "player_possession") & ev["frame_start"].notna() & ev["frame_end"].notna()]
    pp = pp.sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    gib = pp["game_interruption_before"].astype("string")
    rows = []
    teams = pd.unique(pp["team_id"].dropna())
    for i in np.flatnonzero(gib.isin(["corner_for", "corner_against"]).fillna(False).to_numpy()):
        side = str(gib.iloc[i]).rsplit("_", 1)[1]
        recv_team = pp["team_id"].iloc[i]
        restart_team = recv_team if side == "for" else next((t for t in teams if t != recv_team), pd.NA)
        fe = int(pp["frame_end"].iloc[i - 1]) if i > 0 else None
        fs = int(pp["frame_start"].iloc[i])
        lo = ftime.get(fe) if fe is not None else None
        hi = ftime.get(fs)
        if lo is None or hi is None or lo[0] != hi[0]:
            rows.append((side, restart_team, np.nan, np.nan, np.nan, "unmapped"))
            continue
        rows.append((side, restart_team, hi[0], lo[1], hi[1], "ok"))
    return pd.DataFrame(rows, columns=["side", "restart_team", "period_id", "lo", "hi", "bracket_state"])


def sc_detected(mid, frame_ids):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(glob.glob(f"{CACHE}/{mid}/*_tracking.parquet")[0], columns=["frame", "ball_data"])
    bd = t["ball_data"].combine_chunks()
    det = np.asarray(pc.fill_null(bd.field("is_detected"), False).to_numpy(zero_copy_only=False), bool)
    m = dict(zip(t["frame"].to_numpy().tolist(), det.tolist()))
    return np.array([m.get(int(f), False) for f in frame_ids], bool)


def one_match(pa):
    name = os.path.basename(pa)
    prov = name.split("__")[0]
    mid = name.split("__")[1].removesuffix(".parquet")
    frames = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
    ball = frames[frames["is_ball"].astype(bool)].sort_values(["period_id", "frame_id"])
    fps = float(frames["frame_rate"].dropna().iloc[0])
    con = detect_ball_contacts(frames)
    cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
    bxy = ball.set_index(["period_id", "frame_id"])[["x", "y", "time_seconds"]]
    ptm = frames.loc[~frames["is_ball"].astype(bool), ["player_id", "team_id"]].dropna().drop_duplicates("player_id")
    p2t = {norm_id(p): norm_id(t) for p, t in zip(ptm["player_id"], ptm["team_id"])}
    if prov == "skillcorner" and not (glob.glob(f"{CACHE}/{mid}/*_events.parquet") and glob.glob(f"{CACHE}/{mid}/*_tracking.parquet")):
        pd.DataFrame({"prov": [prov], "match": [name], "bracket_state": ["no_cached_events"]}).to_parquet(f"{OUT}/{name}")
        return name, 0
    if prov == "skillcorner":
        det = sc_detected(mid, ball["frame_id"].to_numpy())
        ftime = {int(f): (int(p), float(t)) for f, p, t in
                 frames[["frame_id", "period_id", "time_seconds"]].drop_duplicates("frame_id").itertuples(index=False)}
        br = sc_brackets(mid, ftime)
        br["t"] = np.nan
        share_det = float(det.mean())
    else:
        det = np.ones(len(ball), bool)
        share_det = 1.0
        a = pd.read_parquet(pa)
        c = a[a["type_id"].isin(CORNER_IDS)]
        br = pd.DataFrame({"side": "row", "restart_team": c["team_id"].astype(str).to_numpy(),
                           "period_id": c["period_id"].to_numpy(), "lo": (c["time_seconds"] - 2.0).to_numpy(),
                           "hi": (c["time_seconds"] + 4.0).to_numpy(), "bracket_state": "ok",
                           "t": c["time_seconds"].to_numpy(), "action_id": c["action_id"].to_numpy()})
    out = []
    for r in br.itertuples(index=False):
        rec = dict(r._asdict())
        if r.bracket_state != "ok":
            out.append(rec)
            continue
        per = int(r.period_id)
        bm = (ball["period_id"] == per).to_numpy()
        bt, bx, by = (ball.loc[bm, "time_seconds"].to_numpy(float), ball.loc[bm, "x"].to_numpy(float),
                      ball.loc[bm, "y"].to_numpy(float))
        rec["t_rest"], rec["rest_at_corner"] = rest_release(bt, bx, by, det[bm], r.lo, r.hi, fps)
        cp = con[con["period_id"] == per]
        rec["t_tf79"], _ = last_near_flag(cp["time_seconds"].to_numpy(float), cp["contact_x"].to_numpy(float),
                                          cp["contact_y"].to_numpy(float), r.lo, r.hi)
        cr = cp[cp["player_id"].notna() & (cp["contact_source"].astype(object) != "unresolved")]
        rec["t_tf79r"], j = last_near_flag(cr["time_seconds"].to_numpy(float), cr["contact_x"].to_numpy(float),
                                           cr["contact_y"].to_numpy(float), r.lo, r.hi)
        rec["tf79r_team_ok"] = (None if j is None else
                                p2t.get(norm_id(cr["player_id"].iloc[j])) == norm_id(r.restart_team))
        cl = [cd for (g, p), lst in cands.items() if p == per for cd in lst if cd.players]
        if cl:
            keys = [(per, cd.frame_id) for cd in cl]
            sub = bxy.reindex(keys)
            rec["t_elastic"], _ = last_near_flag(sub["time_seconds"].to_numpy(float), sub["x"].to_numpy(float),
                                                 sub["y"].to_numpy(float), r.lo, r.hi)
        else:
            rec["t_elastic"] = np.nan
        out.append(rec)
    d = pd.DataFrame(out)
    d["prov"], d["match"], d["share_ball_detected"] = prov, name, share_det
    d.to_parquet(f"{OUT}/{name}")
    return name, len(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Corner kick time from the library engines against the rest rule.")
    ap.add_argument("--stage", required=True, help="stage root (holds _actions/ and shards/)")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--workers", type=int, required=True)
    args = ap.parse_args()
    ROOT, CACHE, OUT = args.stage, os.path.join(args.cache_dir, "skillcorner"), args.out
    os.makedirs(OUT, exist_ok=True)
    files = (sorted(glob.glob(f"{ROOT}/_actions/skillcorner__*.parquet")) + sorted(glob.glob(f"{ROOT}/_actions/gradientsports__*.parquet"))
             + sorted(glob.glob(f"{ROOT}/_actions/idsse__*.parquet")))
    todo = [f for f in files if not os.path.exists(f"{OUT}/{os.path.basename(f)}")]
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(one_match, todo):
            print(*r, flush=True)
    print("DONE", flush=True)
