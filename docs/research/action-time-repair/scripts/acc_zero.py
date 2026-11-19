"""Spec section 8 zero-count checks on a repaired stage, part 1: phase, key, spans.

Inputs (read only): a stage root (--stage) and the raw provider cache (--cache-dir, SkillCorner raw files).
1. Phase: `_build_phase_lookup(actions)` as the trainers call it, against the same lookup with the
   order-only rows KEPT. Counted per match: frames inside a window an order-only row opens (kept lookup governed by
   an order-only row), and frames OUTSIDE those windows whose phase differs (must be 0). The repaired lookup
   drops order-only rows before it indexes, so no frame takes a phase from one.
2. Key (section 4.18): duplicates of (game_id, original_event_id, action_provenance, restart_kind(type_id)) among
   SkillCorner rows with an id (must be 0).
3. Spans (section 4.9): the loader's kick mapping rebuilt (`_build_skillcorner` -> `sync_action_times` ->
   `_skillcorner_kicks`); `live` frames in [k, end) of each "after" kick, end = the next kick or `b.frame_start`
   (must be 0); `live` frames from a "period_end" kick to the period's last frame (must be 0); kicks outside their
   gap (must be 0); gaps with two kicks; possessions that are b of one kicked gap and a of another.
"""

import argparse
import glob
import os
import warnings
from multiprocessing import Pool
from unittest import mock

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.spadl import schema
from silly_kicks.tracking._ghost_gk import _SET_PIECE_DECAY_SECONDS, _build_phase_lookup

warnings.simplefilter("ignore")
ROOT = CACHE_DIR = ""  # set from the arguments (the workers fork after that)


def phase_check(name):
    a = pd.read_parquet(f"{ROOT}/_actions/{name}")
    fr = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0], columns=["period_id", "frame_id", "time_seconds"])
    fr = fr.drop_duplicates(["period_id", "frame_id"])
    new = _build_phase_lookup(a)
    with mock.patch.object(schema, "is_order_only", lambda df: pd.Series(False, index=df.index)):
        kept = _build_phase_lookup(a)
    # windows the order-only set-piece rows open in the KEPT table: [t, min(t + decay, next set-piece row))
    oo = pd.Series(np.asarray(schema.is_order_only(a), dtype=bool), index=a.index)
    sp_ids = {i for i, n in enumerate(__import__("silly_kicks.spadl.config", fromlist=["x"]).actiontypes)
              if n in ("freekick_crossed", "freekick_short", "corner_crossed", "corner_short", "goalkick")}
    sp = a[a["type_id"].isin(sp_ids)].sort_values(["period_id", "time_seconds"], kind="mergesort")
    win = []
    for per, g in sp.groupby("period_id"):
        t, o = g["time_seconds"].to_numpy(float), oo.loc[g.index].to_numpy(bool)
        for i in np.flatnonzero(o):
            end = min(t[i] + _SET_PIECE_DECAY_SECONDS, t[i + 1] if i + 1 < len(t) else np.inf)
            win.append((int(per), t[i], end))
    gid = a["game_id"].iloc[0]
    n_in = n_out_diff = n_in_diff = 0
    for per, ts in zip(fr["period_id"].to_numpy(int), fr["time_seconds"].to_numpy(float)):
        inside = any(p == per and lo <= ts <= hi for p, lo, hi in win)
        d = new(gid, per, ts) != kept(gid, per, ts)
        if inside:
            n_in += 1
            n_in_diff += d
        else:
            n_out_diff += d
    return name, len(fr), int(oo.sum()), len(win), n_in, n_in_diff, n_out_diff


def key_check(name):
    a = pd.read_parquet(f"{ROOT}/_actions/{name}")
    a = a[a["original_event_id"].notna()]
    k = pd.DataFrame({"g": a["game_id"].astype(str), "e": a["original_event_id"].map(canonical_id),
                      "p": a["action_provenance"].astype(str), "k": [schema.restart_kind(t) for t in a["type_id"]]})
    return name, len(k), int(k.duplicated().sum())


def span_check(name):
    import scripts._loader_pining as lp

    mid = name.split("__")[1].removesuffix(".parquet")
    got = {}
    orig_kicks, orig_events = lp._skillcorner_kicks, lp._read_skillcorner_events

    def _kicks(*args, **kw):
        got["kicks"] = orig_kicks(*args, **kw)
        return got["kicks"]

    def _events(*args, **kw):
        got["events"] = orig_events(*args, **kw)
        return got["events"]

    with mock.patch.object(lp, "_skillcorner_kicks", _kicks), mock.patch.object(lp, "_read_skillcorner_events", _events):
        ref = lp.list_match_refs(providers=["skillcorner"], match_ids={"skillcorner": [mid]})[0]
        loaded = lp.load_match(ref, events_only=False, cache_dir=CACHE_DIR)
    kicks, outside = got["kicks"]
    ev = got["events"]
    pp = ev[ev["event_type"] == "player_possession"] if "event_type" in ev.columns else ev
    fstart = dict(zip(pp["event_id"].map(canonical_id), pp["frame_start"].astype("int64")))
    # the staged artifact's spans (the loader run above only rebuilds the kick mapping); same frame ids
    staged = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0], columns=["period_id", "frame_id", "stoppage_state"])
    assert set(staged["frame_id"].unique()) == set(loaded.frames["frame_id"].unique())
    staged = staged.drop_duplicates(["period_id", "frame_id"]).sort_values("frame_id")
    fid, live = staged["frame_id"].to_numpy(np.int64), (staged["stoppage_state"].astype(str) == "live").to_numpy()
    per_of = staged.set_index("frame_id")["period_id"]
    last_of = staged.groupby("period_id")["frame_id"].max()
    kframes = sorted(int(v[0]) for v in kicks.values())
    after_live = pe_live = n_after = n_pe = 0
    by_b = {}
    for (eid, kind), (k, side) in kicks.items():
        by_b.setdefault(eid, []).append(side)
        if side == "after":
            n_after += 1
            nxt = [f for f in kframes if f > k]
            end = min([fstart.get(eid, np.iinfo(np.int64).max)] + nxt[:1])
            after_live += int(live[(fid >= k) & (fid < end)].sum())
        elif side == "period_end":
            n_pe += 1
            end = int(last_of.get(per_of.get(k), k))
            pe_live += int(live[(fid >= k) & (fid <= end)].sum())
    two = sum(1 for s in by_b.values() if len(s) > 1)
    return name, len(kicks), n_after, after_live, n_pe, pe_live, len(outside), two


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zero-count checks: phase, key, spans.")
    ap.add_argument("--stage", required=True, help="stage root (holds _actions/ and shards/)")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ROOT, CACHE_DIR = args.stage, args.cache_dir
    allm = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/_actions/*.parquet"))
    sc = [n for n in allm if n.startswith("skillcorner__")]
    with Pool(args.workers) as pool:
        ph = pool.map(phase_check, allm)
        ks = pool.map(key_check, sc)
        sp = pool.map(span_check, sc)
    print("PHASE match | frames | order_only rows | oo set-piece windows | frames in windows | of them differ | frames OUTSIDE windows that differ (must 0)")
    for r in ph:
        print("PHASE", *r)
    print("PHASE outside-window differences total:", sum(r[6] for r in ph), "| windows total:", sum(r[3] for r in ph))
    for r in ks:
        print("KEY", *r)
    print("KEY duplicates total (must 0):", sum(r[2] for r in ks))
    print("SPANS match | kicks | after kicks | live in after spans (must 0) | period_end kicks | live after period_end kick (must 0) | outside gap (must 0) | b-ids with two kicks")
    for r in sp:
        print("SPANS", *r)
    print("SPANS totals: after_live", sum(r[3] for r in sp), "pe_live", sum(r[5] for r in sp), "outside", sum(r[6] for r in sp))
