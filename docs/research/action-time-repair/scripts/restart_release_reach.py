"""How often does the SkillCorner ball track show the kick of a restart? All four restart types, both
sides. Read-only: reads the raw provider cache (--cache-dir, SkillCorner); writes nothing but stdout.

For each first possession after a restart (game_interruption_before = <type>_for or <type>_against), in the 15 s before
its frame_start (not earlier than the end of the possession before it):
- rest: the last run of 5 detected ball frames (0.5 s) that stay within 0.5 m of the run's first point;
- release seen: the frame after that run is detected, and inside 1 s a detected frame is more than 2 m from the rest point.
The release frame is the last frame of the rest run. A crude reach measure: it has no check against a true kick time.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ap = argparse.ArgumentParser(description="How often the SkillCorner ball track shows the kick of a restart.")
ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
ROOT = os.path.join(ap.parse_args().cache_dir, "skillcorner")
KINDS = ("corner", "free_kick", "goal_kick", "throw_in")
COLS = ["event_type", "frame_start", "frame_end", "game_interruption_before"]
LOOK, RUN = 150, 5
rows = []
n_match = 0
for m in sorted(os.listdir(ROOT)):
    evf = glob.glob(f"{ROOT}/{m}/*_events.parquet")
    trf = glob.glob(f"{ROOT}/{m}/*_tracking.parquet")
    if not evf or not trf:
        continue
    try:
        ev = pd.read_parquet(evf[0], columns=COLS)
        t = pq.read_table(trf[0], columns=["frame", "ball_data"])
    except Exception as e:
        print("skip", m, type(e).__name__, str(e)[:60])
        continue
    pp = ev[(ev["event_type"] == "player_possession") & ev["frame_start"].notna() & ev["frame_end"].notna()]
    pp = pp.sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    gib = pp["game_interruption_before"].astype("string")
    fr = t["frame"].to_numpy()
    bd = t["ball_data"].combine_chunks()
    bx = np.asarray(pc.cast(bd.field("x"), "float64").to_numpy(zero_copy_only=False), float)
    by = np.asarray(pc.cast(bd.field("y"), "float64").to_numpy(zero_copy_only=False), float)
    det = np.asarray(pc.fill_null(bd.field("is_detected"), False).to_numpy(zero_copy_only=False), bool)
    n = int(fr.max()) + 2
    X, Y, D = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    X[fr], Y[fr] = bx, by
    D[fr] = det & ~np.isnan(bx)
    n_match += 1
    fs_all = pp["frame_start"].to_numpy(dtype="int64")
    fe_all = pp["frame_end"].to_numpy(dtype="int64")
    for i in np.flatnonzero(gib.notna().to_numpy()):
        g = str(gib.iloc[i])
        kind, side = g.rsplit("_", 1)
        if kind not in KINDS or side not in ("for", "against"):
            continue
        fs = int(fs_all[i])
        if fs >= n:
            continue
        lo = max(0, fs - LOOK, int(fe_all[i - 1]) if i > 0 else 0)
        if fs - lo < RUN + 1:
            rows.append((kind, side, False, False, np.nan, 0.0))
            continue
        x, y, d = X[lo:fs], Y[lo:fs], D[lo:fs]
        L = len(x)
        share_det = float(d.mean())
        okrun = np.convolve(d.astype(int), np.ones(RUN, int), "valid") == RUN  # index s: frames s..s+RUN-1 detected
        still = np.ones(L - RUN + 1, bool)
        for k in range(1, RUN):
            still &= np.hypot(x[k:L - RUN + 1 + k] - x[:L - RUN + 1], y[k:L - RUN + 1 + k] - y[:L - RUN + 1]) < 0.5
        cand = np.flatnonzero(okrun & still)
        if not len(cand):
            rows.append((kind, side, False, False, np.nan, share_det))
            continue
        s = int(cand[-1])
        e = s + RUN - 1  # last frame of the last rest run, window index
        rx, ry = x[s], y[s]
        seen = False
        if e + 1 < L and d[e + 1]:
            nxt = slice(e + 1, min(L, e + 11))
            far = d[nxt] & (np.hypot(x[nxt] - rx, y[nxt] - ry) > 2.0)
            seen = bool(far.any())
        rows.append((kind, side, True, seen, (fs - (lo + e)) / 10.0 if seen else np.nan, share_det))

d = pd.DataFrame(rows, columns=["kind", "side", "rest", "seen", "gap", "share_det"])
print("matches read", n_match, "| restarts", len(d))
for kind in KINDS:
    s = d[d["kind"] == kind]
    g = s["gap"].dropna()
    print("%-9s n %6d (for %d, against %d) | ball detected in the window, mean share %.2f | rest seen %.3f | release seen %.3f | release to first possession: q25 %.1f, q50 %.1f, q75 %.1f s" % (
        kind, len(s), int((s["side"] == "for").sum()), int((s["side"] == "against").sum()), s["share_det"].mean(),
        s["rest"].mean(), s["seen"].mean(), g.quantile(0.25), g.median(), g.quantile(0.75)))
    for side in ("for", "against"):
        q = s[s["side"] == side]
        print("      %-7s n %6d | release seen %.3f" % (side, len(q), q["seen"].mean()))
print("DONE")
