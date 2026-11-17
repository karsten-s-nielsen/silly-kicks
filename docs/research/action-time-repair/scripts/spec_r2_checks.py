"""Read-only measures for the spec revision (supervisor points 1, 3, 6, 8), 2026-11-18.

1. The restart count rule with the "after" markers: raw GI events of the cache (889 matches).
3. The kick rule and the engines against the gradientsports provider corner time (~/refit144/corner_kick_check rows).
6. The 18 order inversions after the idsse retime: which order does the tracking support? (re-runs the retime on 7 idsse)
8. The size of the SkillCorner stoppage change: frames of the restart flight after a measured kick and of the first
   possession, against the frames each stage match has (98 SkillCorner stage matches with cached raw events).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

CACHE = os.path.expanduser("~/refit144/corners/cache/skillcorner")
ROOT = os.path.expanduser("~/cross-def/xshot187-f5fc2e2b/out")
KINDS = ("corner", "free_kick", "goal_kick", "throw_in")


def split(marker):
    if not isinstance(marker, str) or "_" not in marker:
        return None, None
    k, s = marker.rsplit("_", 1)
    return (k, s) if k in KINDS and s in ("for", "against") else (None, None)


def point1():
    tot = {k: {"paired": 0, "after_only": 0, "before_only": 0, "side_agree": 0, "side_n": 0} for k in KINDS}
    n = 0
    for m in sorted(os.listdir(CACHE)):
        f = glob.glob(f"{CACHE}/{m}/*_events.parquet")
        if not f:
            continue
        ev = pd.read_parquet(f[0], columns=["event_type", "frame_start", "period", "team_id", "game_interruption_before",
                                            "game_interruption_after"])
        pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start", kind="mergesort").reset_index(drop=True)
        n += 1
        teams = list(pd.unique(pp["team_id"].dropna()))
        other = {t: next((u for u in teams if u != t), None) for t in teams}
        per, team = pp["period"].to_numpy(), pp["team_id"].to_numpy()
        gia, gib = pp["game_interruption_after"].to_numpy(object), pp["game_interruption_before"].to_numpy(object)
        for i in range(len(pp)):
            ka, sa = split(gia[i])
            same_next = i + 1 < len(pp) and per[i + 1] == per[i]
            kb_next, sb_next = split(gib[i + 1]) if same_next else (None, None)
            if ka is not None:
                if kb_next == ka:
                    tot[ka]["paired"] += 1
                    ta = team[i] if sa == "for" else other.get(team[i])
                    tb = team[i + 1] if sb_next == "for" else other.get(team[i + 1])
                    tot[ka]["side_n"] += 1
                    tot[ka]["side_agree"] += int(ta == tb)
                else:
                    tot[ka]["after_only"] += 1
            kb, _ = split(gib[i])
            if kb is not None:
                prev_ka = split(gia[i - 1])[0] if i > 0 and per[i - 1] == per[i] else None
                if prev_ka != kb:
                    tot[kb]["before_only"] += 1
    print(f"== 1. restart markers, {n} matches")
    for k, t in tot.items():
        rule_old = t["paired"] + t["before_only"]
        rule_new = rule_old + t["after_only"]
        print(f"   {k:9s} paired {t['paired']:6d} | after only {t['after_only']:5d} | before only {t['before_only']:5d} | "
              f"side agreement on paired {t['side_agree'] / max(1, t['side_n']):.4f} | count: before-rule {rule_old}, with after markers {rule_new}")


def point3():
    D = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.expanduser("~/refit144/corner_kick_check/gradientsports__*.parquet"))])
    D = D[D["bracket_state"] == "ok"]
    print(f"== 3. gradientsports corners {len(D)}: each source against the provider corner time t (reference)")
    for col in ("t_rest", "t_tf79", "t_tf79r", "t_elastic"):
        d = (D[col] - D["t"]).dropna()
        print(f"   {col:9s} found {D[col].notna().mean():.3f} | within 0.5 s of t {(d.abs() <= 0.5).mean():.3f} | more than 0.5 s off {(d.abs() > 0.5).mean():.3f} | median {d.median():+.2f} s")
    both = D.dropna(subset=["t_rest"])
    for col in ("t_tf79", "t_tf79r", "t_elastic"):
        b = both.dropna(subset=[col])
        print(f"   on the {len(b)} corners where the rule and {col} both find: rule within 0.5 s of t {((b['t_rest'] - b['t']).abs() <= 0.5).mean():.3f}, {col} {((b[col] - b['t']).abs() <= 0.5).mean():.3f}")


def point6():
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking._elastic_sync import _map_category, align_events_to_frames

    names = list(spadlconfig.actiontypes)
    yard = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.expanduser("~/refit144/elastic_check/idsse__*.parquet"))])
    yard = yard.set_index(["match", "action_id"])["yard_off"]
    rows = []
    for pa in sorted(glob.glob(f"{ROOT}/_actions/idsse__*.parquet")):
        name = os.path.basename(pa)
        a = pd.read_parquet(pa).sort_values("action_id").reset_index(drop=True)
        f = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
        ft = f[["period_id", "frame_id", "time_seconds"]].drop_duplicates(["period_id", "frame_id"]).set_index(["period_id", "frame_id"])["time_seconds"]
        al = align_events_to_frames(a, f).set_index("action_id")["elastic_frame_id"]
        new, src = [], []
        for r in a.itertuples():
            fr = al.get(r.action_id)
            if fr is not None and pd.notna(fr):
                new.append(ft.get((int(r.period_id), int(fr)), np.nan)); src.append("aligned")
            else:
                new.append(r.time_seconds); src.append("kept")
        a["new_t"], a["cls"] = new, src
        a.loc[a["type_id"] == spadlconfig.actiontype_id["dribble"], "cls"] = "dribble"
        for per, g in a.groupby("period_id"):
            g = g.reset_index(drop=True)
            for i in np.flatnonzero(np.diff(g["new_t"].to_numpy(float)) < 0):
                p, q = g.iloc[i], g.iloc[i + 1]
                if "dribble" in (p.cls, q.cls):
                    continue
                def track_t(r):
                    if r.cls == "aligned":
                        return r.new_t, "elastic"
                    y = yard.get((name, int(r.action_id)), np.nan)
                    return (r.time_seconds + y, "yardstick") if pd.notna(y) else (np.nan, "none")
                tp, sp = track_t(p)
                tq, sq = track_t(q)
                verdict = "provider order" if tp < tq else ("time order" if tp > tq else "unknown")
                rows.append((name, int(per), names[int(p.type_id)], p.cls, names[int(q.type_id)], q.cls,
                             round(p.new_t - q.new_t, 2), sp, sq, verdict))
    d = pd.DataFrame(rows, columns=["match", "period", "type_a", "cls_a", "type_b", "cls_b", "inversion_s", "evidence_a", "evidence_b", "tracking_supports"])
    print(f"== 6. inversions (dribble pairs excluded): {len(d)}")
    print(d.to_string(index=False))
    print("   tracking supports:", d["tracking_supports"].value_counts().to_dict())


def point8():
    rows = []
    for pa in sorted(glob.glob(f"{ROOT}/_actions/skillcorner__*.parquet")):
        mid = os.path.basename(pa).split("__")[1].removesuffix(".parquet")
        f = glob.glob(f"{CACHE}/{mid}/*_events.parquet")
        if not f:
            continue
        ev = pd.read_parquet(f[0], columns=["event_type", "frame_start", "frame_end", "period", "game_interruption_before",
                                            "game_interruption_after"])
        pp = ev[(ev["event_type"] == "player_possession") & ev["frame_start"].notna()].sort_values("frame_start", kind="mergesort").reset_index(drop=True)
        fr = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/skillcorner__{mid}.parquet")[0], columns=["frame_id", "is_ball", "stoppage_state"])
        fr = fr[fr["is_ball"]]
        n_frames = fr["frame_id"].nunique()
        n_live = int((fr["stoppage_state"] == "live").sum())
        ck = pd.read_parquet(os.path.expanduser(f"~/refit144/corner_kick_check/skillcorner__{mid}.parquet"))
        gib = pp["game_interruption_before"].map(lambda m: split(m)[0])
        recv = 0
        for i in np.flatnonzero(gib.notna().to_numpy()):
            recv += int(pp["frame_end"].iloc[i] - pp["frame_start"].iloc[i] + 1)
        ok = ck[(ck.get("bracket_state") == "ok")] if "bracket_state" in ck else ck.iloc[0:0]
        flight = float(((ok["hi"] - ok["t_rest"]).dropna() * 10).sum()) if len(ok) else 0.0
        rows.append((mid, n_frames, n_live, recv, flight))
    d = pd.DataFrame(rows, columns=["match", "frames", "live", "first_possession_frames", "corner_flight_frames"])
    print(f"== 8. SkillCorner stage matches {len(d)}: frames {d.frames.sum()}, live {d.live.sum()} ({d.live.sum() / d.frames.sum():.3f})")
    print(f"   first-possession frames of all restarts (unknown today): {d.first_possession_frames.sum()} ({d.first_possession_frames.sum() / d.frames.sum():.4f} of all frames)")
    print(f"   flight frames after a measured corner kick (dead today): {int(d.corner_flight_frames.sum())} ({d.corner_flight_frames.sum() / d.frames.sum():.4f})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        globals()[f"point{p}"]()
    print("DONE")
