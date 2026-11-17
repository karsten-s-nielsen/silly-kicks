"""Read-only (D12 option A3): the lag of the running-score rule on rows only.

Rule: the new score takes effect at the row time (frame_end) of the possession AFTER the first possession whose
running score (team_score + opponent_team_score, side-free) is higher. Per goal with a goal marker on the scoring
possession s ("goal_" in game_interruption_after): K = s + 1 (the kick-off possession). Reported: the effect time E
minus frame_end(s) (the goal proxy; < 0 would break the bar), and the live lag E - frame_start(K). Same period only.
Output: stdout.
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

CACHE = os.path.expanduser("~/refit144/corners/cache/skillcorner")


def one(m):
    f = glob.glob(f"{CACHE}/{m}/*_events.parquet")
    if not f:
        return []
    ev = pd.read_parquet(f[0], columns=["event_type", "frame_start", "frame_end", "period", "team_score",
                                        "opponent_team_score", "game_interruption_after"])
    pp = ev[(ev.event_type == "player_possession") & ev.frame_start.notna() & ev.frame_end.notna()]
    pp = pp.sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    tot = (pp.team_score.astype(float) + pp.opponent_team_score.astype(float)).to_numpy()
    if np.any(np.diff(tot[~np.isnan(tot)]) < 0):
        return [{"match": m, "decreasing": True}]
    gia = pp.game_interruption_after.astype("string").fillna("")
    per = pp.period.to_numpy()
    fs, fe = pp.frame_start.to_numpy() / 10.0, pp.frame_end.to_numpy() / 10.0
    out = []
    for s in np.flatnonzero(gia.isin(["goal_for", "goal_against"]).to_numpy()):
        if s + 1 >= len(pp) or per[s + 1] != per[s]:
            continue
        rise = [i for i in range(max(s - 2, 0), min(s + 6, len(pp))) if i > 0 and tot[i] > tot[i - 1]]
        if not rise:
            out.append({"match": m, "found": False})
            continue
        i = rise[0]
        if i + 1 >= len(pp) or per[i + 1] != per[s]:
            continue
        e = fe[i + 1]
        out.append({"match": m, "found": True, "rise_at": i - s, "after_goal_s": e - fe[s], "live_lag_s": e - fs[s + 1]})
    return out


if __name__ == "__main__":
    with Pool(16) as p:
        d = pd.DataFrame([r for rs in p.map(one, sorted(os.listdir(CACHE))) for r in rs])
    print("decreasing matches:", int(d.get("decreasing", pd.Series(dtype=bool)).fillna(False).sum()))
    g = d[d.get("found") == True]  # noqa: E712
    print("goals with a marker and a score rise near it:", len(g), "; no rise found:", int((d.get("found") == False).sum()))  # noqa: E712
    print("rise at s+k:", g.rise_at.value_counts().sort_index().to_dict())
    print("effect before the goal proxy (E < frame_end(s)):", int((g.after_goal_s < 0).sum()))
    print("E - frame_end(s) quantiles 0.1/0.5/0.9:", g.after_goal_s.quantile([0.1, 0.5, 0.9]).round(1).tolist())
    print("live lag E - frame_start(K) quantiles 0.1/0.5/0.9:", g.live_lag_s.quantile([0.1, 0.5, 0.9]).round(1).tolist())
    print("DONE")
