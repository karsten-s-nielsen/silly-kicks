"""SkillCorner penalties: what the data gives for a derived penalty row (read-only; stdout and one parquet file,
--out). Input: the raw provider cache (--cache-dir, SkillCorner). The derived penalty row of spec section 4.17.

For each penalty marker (penalty_for / penalty_against on a possession's game_interruption_after, or on the next
possession's game_interruption_before):
  - the outcome from the provider's running score, and what follows;
  - the team: marker side against the team whose score rises, and against the end where the ball rests;
  - the kick from the tracking: the last run of at least 0.5 s of detected ball frames within 0.5 m of its first point,
    that point within 1.5 m of a penalty mark, then a detected frame more than 2 m away within 1 s;
  - at the kick: the nearest player (the taker), the defending goalkeeper, and where the ball goes.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = ""  # set from the arguments (the workers fork after that)
EV_COLS = ["event_type", "team_id", "player_id", "period", "frame_start", "frame_end", "end_type", "start_type",
           "game_interruption_before", "game_interruption_after", "team_score", "opponent_team_score"]
R_MARK, R_REST, N_REST, R_REL, N_REL = 1.5, 0.5, 5, 2.0, 10


def kind_of(v):
    if pd.isna(v):
        return (None, "")
    s = str(v)
    k, _, side = s.rpartition("_")
    return (k, side) if side in ("for", "against") else (s, "")


def find_kick(bx, by, det, marks):
    """Last rest-then-release run at a penalty mark. Returns (index of the last rest frame, rest length, distance to
    the mark, sign of the mark) or None."""
    n = len(bx)
    best = None
    i = 0
    while i < n:
        if not det[i]:
            i += 1
            continue
        dm = np.hypot(marks[:, 0] - bx[i], marks[:, 1] - by[i])
        if dm.min() > R_MARK:
            i += 1
            continue
        e = i
        while e + 1 < n and det[e + 1] and np.hypot(bx[e + 1] - bx[i], by[e + 1] - by[i]) <= R_REST:
            e += 1
        if e - i + 1 >= N_REST:
            nxt = slice(e + 1, min(n, e + 1 + N_REL))
            d = np.hypot(bx[nxt] - bx[i], by[nxt] - by[i])
            if (det[nxt] & (d > R_REL)).any():
                best = (e, e - i + 1, float(dm.min()), float(np.sign(marks[int(dm.argmin()), 0])))
        i = e + 1
    return best


def one(m):
    evf = glob.glob(f"{ROOT}/{m}/*_events.parquet")
    mef = glob.glob(f"{ROOT}/{m}/*_metadata.json")
    trf = glob.glob(f"{ROOT}/{m}/*_tracking.parquet")
    if not evf or not mef:
        return []
    try:
        ev = pd.read_parquet(evf[0], columns=EV_COLS)
    except Exception:
        return []
    pp = ev[(ev["event_type"] == "player_possession") & ev["frame_start"].notna() & ev["frame_end"].notna()]
    pp = pp.sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    if pp.empty:
        return []
    ka = [kind_of(v) for v in pp["game_interruption_after"].astype("string")]
    kb = [kind_of(v) for v in pp["game_interruption_before"].astype("string")]
    per = pp["period"].to_numpy()
    N = len(pp)
    inst = []
    for i in range(N):
        if ka[i][0] == "penalty":
            b = i + 1 if i + 1 < N and per[i + 1] == per[i] else None
            inst.append((i, b, "after", ka[i][1]))
        if kb[i][0] == "penalty" and not (i > 0 and per[i - 1] == per[i] and ka[i - 1][0] == "penalty"):
            a = i - 1 if i > 0 and per[i - 1] == per[i] else None
            inst.append((a, i, "before_only", kb[i][1]))
    if not inst:
        return []
    meta = json.load(open(mef[0]))
    L = float(meta.get("pitch_length") or 105)
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    sides = meta.get("home_team_side") or []
    role = {p["id"]: ((p.get("player_role") or {}).get("acronym"), p.get("team_id")) for p in meta.get("players") or []}
    tid = pp["team_id"].to_numpy()
    tot = (pp["team_score"].fillna(0) + pp["opponent_team_score"].fillna(0)).to_numpy()
    hsc = np.where(tid == home, pp["team_score"], pp["opponent_team_score"]).astype(float)
    fs = pp["frame_start"].to_numpy(dtype="int64")
    fe = pp["frame_end"].to_numpy(dtype="int64")
    et = pp["end_type"].astype("string").fillna("").to_numpy()
    pid = pp["player_id"].to_numpy()
    T = None
    if trf:
        try:
            t = pq.read_table(trf[0], columns=["frame", "ball_data", "player_data"])
            fr = t["frame"].to_numpy()
            bd = t["ball_data"].combine_chunks()
            bx = np.asarray(bd.field("x").to_numpy(zero_copy_only=False), float)
            by = np.asarray(bd.field("y").to_numpy(zero_copy_only=False), float)
            det = np.asarray(bd.field("is_detected").fill_null(False).to_numpy(zero_copy_only=False), bool) & ~np.isnan(bx)
            T = (fr, bx, by, det, t["player_data"])
        except Exception:
            T = None
    marks = np.array([[L / 2 - 11.0, 0.0], [-(L / 2 - 11.0), 0.0]])
    out = []
    for a, b, cls, side in inst:
        ref = a if cls == "after" else b
        t_ref = tid[ref]
        other = away if t_ref == home else home
        team_h1 = t_ref if side == "for" else other  # hypothesis: `_for` = the marked possession's team takes it
        p = int(per[ref])
        att = None
        if 1 <= p <= len(sides):
            home_sign = 1.0 if sides[p - 1] == "left_to_right" else -1.0
            att = home_sign if team_h1 == home else -home_sign
        scored, score_team = None, None
        if a is not None and b is not None:
            scored = bool(tot[b] > tot[a])
            if scored:
                score_team = home if hsc[b] > hsc[a] else away
        row = dict(match=m, cls=cls, side=side, period=p, has_a=a is not None, has_b=b is not None, scored=scored,
                   score_team_is_h1=(score_team == team_h1) if score_team is not None else None,
                   follower=(kb[b][0] or "none") if b is not None else "period end",
                   b_end_type=et[b] if b is not None else "", b_team_is_h1=(tid[b] == team_h1) if b is not None else None,
                   has_tracking=T is not None, found=False)
        if T is not None and a is not None:
            fr, bx, by, det, pdata = T
            f0 = int(fe[a])
            f1 = int(fs[b]) if b is not None else f0 + 1500
            lo, hi = int(np.searchsorted(fr, f0)), int(np.searchsorted(fr, f1))
            row["bracket_s"] = (f1 - f0) / 10.0
            row["ball_det_share"] = float(det[lo:hi].mean()) if hi > lo else np.nan
            k = find_kick(bx[lo:hi], by[lo:hi], det[lo:hi], marks) if hi - lo > N_REST else None
            if k is not None:
                e, n_rest, d_mark, sgn = k
                g = lo + e
                row.update(found=True, rest_s=n_rest / 10.0, d_mark=d_mark, mark_is_attacked_end=(sgn == att) if att is not None else None,
                           wait_s=(int(fr[g]) - f0) / 10.0, to_b_s=(f1 - int(fr[g])) / 10.0 if b is not None else np.nan)
                # the ball after the release
                x0 = bx[g]
                aft = slice(g + 1, min(len(bx), g + 16))
                dd = det[aft]
                row["toward_goal"] = bool((dd & ((bx[aft] - x0) * sgn > 2.0)).any())
                row["ball_in_goal"] = bool((dd & (bx[aft] * sgn >= L / 2 - 0.5) & (np.abs(by[aft]) <= 4.5)).any())
                # players at the kick
                pl = pdata[g].as_py() or []
                if pl:
                    px = np.array([q["x"] for q in pl], float)
                    py = np.array([q["y"] for q in pl], float)
                    ids = [q["player_id"] for q in pl]
                    dets = np.array([bool(q["is_detected"]) for q in pl])
                    d = np.hypot(px - bx[g], py - by[g])
                    j = int(np.nanargmin(d))
                    row.update(taker_dist=float(d[j]), taker_detected=bool(dets[j]),
                               taker_team_is_h1=(role.get(ids[j], (None, None))[1] == team_h1),
                               taker_is_gk=(role.get(ids[j], (None, None))[0] == "GK"),
                               second_dist=float(np.sort(d)[1]) if len(d) > 1 else np.nan,
                               b_player_is_taker=(pid[b] == ids[j]) if b is not None else None)
                    def_team = away if team_h1 == home else home
                    gk = [i2 for i2, q in enumerate(ids) if role.get(q, (None, None)) == ("GK", def_team)]
                    row["gk_in_data"] = bool(gk)
                    if gk:
                        i2 = gk[0]
                        row.update(gk_detected=bool(dets[i2]), gk_to_line=float(abs(px[i2] - sgn * L / 2)), gk_abs_y=float(abs(py[i2])))
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SkillCorner penalties: what the data gives for a derived penalty row.")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--out", required=True, help="output parquet file (read by penalty_wait_check.py)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    ROOT = os.path.join(args.cache_dir, "skillcorner")
    ms = sorted(os.listdir(ROOT))
    with Pool(args.workers) as pool:
        res = pool.map(one, ms, chunksize=8)
    d = pd.DataFrame([r for rr in res for r in rr])
    d.to_parquet(args.out)
    pd.set_option("display.width", 220)
    print("matches", len(ms), "| penalties", len(d), "| matches with one", d["match"].nunique())
    print("by class:", d["cls"].value_counts().to_dict(), "| marker side:", d["side"].value_counts().to_dict(), "| period:", d["period"].value_counts().sort_index().to_dict())
    s = d[d["scored"].notna()]
    print("== outcome (the running score): scored %d of %d (%.3f)" % (int(s["scored"].sum()), len(s), s["scored"].mean()))
    print("   what follows, by scored:")
    print(pd.crosstab(d["follower"], d["scored"].astype("object").fillna("n/a")).to_string())
    sc = d[d["scored"] == True]  # noqa: E712
    # `score_team_is_h1` is an object column (True / False / None), so cast before the sum: an object sum of booleans
    # printed "1 of 173" in the first run of this script (2026-11-18); the count was 173 of 173.
    team_ok = sc["score_team_is_h1"].astype(bool)
    print("== team: the scoring team is the team that the marker rule names: %d of %d" % (int(team_ok.sum()), len(sc)))
    print("   by marker side:", team_ok.groupby(sc["side"]).agg(["sum", "count"]).to_dict("index"))
    t = d[d["has_tracking"] & d["has_a"]]
    f = t[t["found"]]
    print("== kick from the tracking: with tracking %d | kick found %d (%.3f) | by scored: %s" % (
        len(t), len(f), len(f) / len(t), t.groupby(t["scored"].astype("object").fillna("n/a"))["found"].agg(["sum", "count"]).to_dict("index")))
    nf = t[~t["found"]]
    print("   not found: ball detected share in the bracket q25 %.2f q50 %.2f q75 %.2f | found: %.2f %.2f %.2f" % (
        nf["ball_det_share"].quantile(.25), nf["ball_det_share"].median(), nf["ball_det_share"].quantile(.75),
        f["ball_det_share"].quantile(.25), f["ball_det_share"].median(), f["ball_det_share"].quantile(.75)))
    q = lambda c: "%.2f / %.2f / %.2f" % (f[c].quantile(.1), f[c].median(), f[c].quantile(.9))  # noqa: E731
    print("   found: rest length s (q10/q50/q90) %s | distance to the mark m %s | wait after the previous possession s %s | to the next possession s %s" % (
        q("rest_s"), q("d_mark"), q("wait_s"), q("to_b_s")))
    me = f[f["mark_is_attacked_end"].notna()]
    print("   the mark is at the end that the marker's `_for` team attacks: %d of %d (%.3f)" % (int(me["mark_is_attacked_end"].sum()), len(me), me["mark_is_attacked_end"].mean()))
    print("   after the release: ball moves toward the goal %.3f | scored and ball seen in the goal %d of %d" % (
        f["toward_goal"].mean(), int(f.loc[f["scored"] == True, "ball_in_goal"].sum()), int((f["scored"] == True).sum())))  # noqa: E712
    k = f[f["taker_dist"].notna()]
    print("== at the kick (%d): nearest player to the ball m (q10/q50/q90) %.2f / %.2f / %.2f | within 2 m %.3f | he is detected %.3f | he is of the `_for` team %.3f | he is a goalkeeper %.3f | second nearest m q10 %.1f q50 %.1f" % (
        len(k), k["taker_dist"].quantile(.1), k["taker_dist"].median(), k["taker_dist"].quantile(.9), (k["taker_dist"] <= 2).mean(),
        k["taker_detected"].mean(), k["taker_team_is_h1"].mean(), k["taker_is_gk"].mean(), k["second_dist"].quantile(.1), k["second_dist"].median()))
    good = k[(k["taker_dist"] <= 2) & k["taker_detected"] & k["taker_team_is_h1"]]
    print("   taker named with all three (within 2 m, detected, `_for` team): %d of %d kicks (%.3f) = %.3f of all penalties" % (len(good), len(k), len(good) / len(k), len(good) / len(d)))
    g = k[k["gk_in_data"] == True]  # noqa: E712
    print("   defending goalkeeper: in the frame data %d of %d | detected %.3f | distance to the goal line m (q10/q50/q90) %.2f / %.2f / %.2f | within 1.5 m of the line and |y| < 4 %.3f" % (
        len(g), len(k), g["gk_detected"].mean(), g["gk_to_line"].quantile(.1), g["gk_to_line"].median(), g["gk_to_line"].quantile(.9),
        ((g["gk_to_line"] <= 1.5) & (g["gk_abs_y"] <= 4)).mean()))
    pr = d[(d["follower"] == "penalty")]
    print("== a possession follows the penalty with a penalty marker (%d): end_type %s | it is of the `_for` team %s | scored %s" % (
        len(pr), pr["b_end_type"].value_counts().to_dict(), pr["b_team_is_h1"].value_counts().to_dict(), pr["scored"].value_counts().to_dict()))
    prf = pr[pr["found"]]
    print("   with a kick found %d: its player is the nearest player at the kick %s | kick to its start s q10 %.1f q50 %.1f q90 %.1f" % (
        len(prf), prf["b_player_is_taker"].value_counts().to_dict(), prf["to_b_s"].quantile(.1), prf["to_b_s"].median(), prf["to_b_s"].quantile(.9)))
    print("DONE")
    sys.stdout.flush()
