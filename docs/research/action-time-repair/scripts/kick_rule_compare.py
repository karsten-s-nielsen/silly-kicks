"""Reviewer B RTR-IMPL-03 (owner: measure it), 2026-11-19.

The library kick rule (`silly_kicks.tracking._action_time._find_kick`, corner, paired: first_run_rule=False) against
the measured rule of `corner_kick_check.rest_release`, on the Gradient Sports corners of the f5fc2e2b stage (read
only), each in the script's GS bracket [t - 2 s, t + 4 s] around the provider corner time t.
Reports: corners; the share each rule finds a kick; agreement where both find one (|dt| <= 0.1 s, <= 0.5 s); each
rule's accuracy against the provider time (|kick - t| <= 0.5 s, of the corners it finds a kick for) and median.
Output: stdout.
"""

import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/refit144"))
sys.path.insert(0, os.path.expanduser("~/refit144/sk"))
import corner_kick_check as K  # noqa: E402  (the research script, copied next to this one)

from silly_kicks.tracking._action_time import _find_kick  # noqa: E402

ROOT = os.path.expanduser("~/cross-def/xshot187-f5fc2e2b/out")


def one(pa):
    name = os.path.basename(pa)
    a = pd.read_parquet(pa)
    c = a[a["type_id"].isin(K.CORNER_IDS)]
    fr = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
    ball = fr[fr["is_ball"].astype(bool)].sort_values(["period_id", "time_seconds"])
    out = []
    for r in c.itertuples(index=False):
        per, t = int(r.period_id), float(r.time_seconds)
        bp = ball[ball["period_id"] == per]
        bt, bx, by = (bp["time_seconds"].to_numpy(float), bp["x"].to_numpy(float), bp["y"].to_numpy(float))
        fps = 1.0 / float(np.median(np.diff(bt))) if len(bt) > 1 else 25.0
        lo, hi = t - 2.0, t + 4.0
        t_script, _ = K.rest_release(bt, bx, by, np.ones(len(bt), bool), lo, hi, fps)
        k = _find_kick(bp, lo, hi, "corner", first_run_rule=False, follower_cannot_rest=True)
        out.append({"match": name, "t": t, "t_script": t_script, "t_lib": np.nan if k is None else k[0]})
    return pd.DataFrame(out)


if __name__ == "__main__":
    files = sorted(glob.glob(f"{ROOT}/_actions/gradientsports__*.parquet"))
    with Pool(8) as pool:
        d = pd.concat(pool.map(one, files), ignore_index=True)
    n = len(d)
    fs, fl = d["t_script"].notna(), d["t_lib"].notna()
    both = d[fs & fl]
    dt = (both["t_lib"] - both["t_script"]).abs()
    print(f"GS matches {len(files)}, corners {n}")
    print(f"found: script {fs.mean():.3f} ({fs.sum()}), library {fl.mean():.3f} ({fl.sum()}), both {len(both)}, "
          f"script only {(fs & ~fl).sum()}, library only {(~fs & fl).sum()}")
    print(f"agreement where both: |dt|<=0.1 s {float((dt <= 0.1).mean()):.3f}, <=0.5 s {float((dt <= 0.5).mean()):.3f}, "
          f"median |dt| {float(dt.median()):.3f} s")
    for col, nm in (("t_script", "script"), ("t_lib", "library")):
        off = (d[col] - d["t"]).dropna()
        print(f"{nm:8s} vs provider time: within 0.5 s {float((off.abs() <= 0.5).mean()):.3f} of {len(off)}, "
              f"median {float(off.median()):+.2f} s, more than 0.5 s off {float((off.abs() > 0.5).mean()):.3f}")
