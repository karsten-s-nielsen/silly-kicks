"""Acceptance counts on the stage of record (read only): the compares with the f5fc2e2b stage, the census, the corner
table and the kick shares, the spec section 8 counts of the
SkillCorner restart rows, and the section 4.9 live share.

Inputs (read only): the repaired stage (--stage), the f5fc2e2b stage (--old-stage) and the raw provider cache
(--cache-dir, SkillCorner events). Output: --out (acc_counts.json and acc_counts_census.parquet).

- Equal to the old stage: gradientsports, each match's actions table equal to the f5fc2e2b stage in action_id order, every column
  except time_source; SkillCorner, the native rows joined on (game_id, original_event_id): time_seconds equal.
- Census: an independent census of the raw GI markers (pining cache, read only) by the spec's count rule (section 4,
  "The count rule"; penalties by section 4.20's rule): one restart per marker; a same-kind "after" marker on a and
  "before" marker on b are one; an "after" marker with no possession after it in the period is not a restart
  (it is a penalty). Each census restart is keyed (id of b, or of a for a period-end penalty; kind). Per match: the
  census keys against the staged derived rows of that kind (both directions). Plus: derived restart rows with a
  player (must be 0) and native rows of a restart type.
- Corner table: per match, the corner rows of the actions against `identify_corners(actions)` (each corner row is in the
  corner table), and the SkillCorner corner rows against the census corners.
- Kick share: the share of derived restart rows with time_source "kick", by kind, with no limit; and by census class.
- Section 8: the "after"-only corners that get a kick, by follower marker; the derived restart rows with a NaN
  start, by kind; the penalty rows by class, result and time source.
- Section 4.9: the SkillCorner frames inside b of a marked pair whose stoppage_state is "live", by the marker kind
  of the pair, as a share of all staged SkillCorner frames (unique frame_id per match).
"""

import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from silly_kicks.corners import identify_corners
from silly_kicks.spadl import config as spadlconfig

STAGE = OLD = RAW = OUT = ""  # set from the arguments (the workers fork after that)
WORKERS = 12
KIND_TYPES = {
    "corner": ("corner_crossed", "corner_short"),
    "free_kick": ("freekick_crossed", "freekick_short", "shot_freekick"),
    "goal_kick": ("goalkick",),
    "throw_in": ("throw_in",),
    "penalty": ("shot_penalty",),
}
TYPE_KIND = {spadlconfig.actiontype_id[t]: k for k, ts in KIND_TYPES.items() for t in ts}
RESTARTS = ("corner", "free_kick", "goal_kick", "throw_in")
CORNER_IDS = {spadlconfig.actiontype_id[t] for t in KIND_TYPES["corner"]}
SUCCESS = spadlconfig.result_id["success"]


def _kind(marker, kinds):
    for k in kinds:
        if marker in (f"{k}_for", f"{k}_against"):
            return k
    return None


def census(m):
    """Every restart and penalty of one match by the count rule: (key, kind) -> (class, follower marker)."""
    cols = ["event_id", "event_type", "frame_start", "frame_end", "period", "game_interruption_before",
            "game_interruption_after"]
    pq_files = glob.glob(f"{RAW}/{m}/*_events.parquet")  # the newer artifact schema; else the open-data CSV
    ev = (pd.read_parquet(pq_files[0], columns=cols) if pq_files
          else pd.read_csv(glob.glob(f"{RAW}/{m}/*_dynamic_events.csv")[0], low_memory=False)[cols])
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    gib = pp["game_interruption_before"].astype("string").fillna("").tolist()
    gia = pp["game_interruption_after"].astype("string").fillna("").tolist()
    per = pp["period"].tolist()
    ids = pp["event_id"].astype(str).tolist()
    kinds = (*RESTARTS, "penalty")
    out, excluded, pairs = {}, Counter(), []
    n = len(pp)
    for i in range(n):
        a = i - 1 if i > 0 and per[i - 1] == per[i] else None
        kb = _kind(gib[i], kinds)
        ka = _kind(gia[a], kinds) if a is not None else None
        if kb is not None:
            out[(ids[i], kb)] = ("paired" if ka == kb else "before_only", "")
        if ka is not None and ka != kb:
            out[(ids[i], ka)] = ("after_only", gib[i])
        if a is not None and (gia[a] or gib[i]):  # a marked pair (any marker kind), for section 4.9
            pairs.append((int(pp.at[i, "frame_start"]), int(pp.at[i, "frame_end"]),
                          kb or ka or (gib[i] or gia[a]).rsplit("_", 1)[0]))
        last = i == n - 1 or per[i + 1] != per[i]
        ke = _kind(gia[i], kinds)
        if last and ke is not None:
            if ke == "penalty":
                out[(ids[i], "penalty")] = ("period_end", "")
            else:
                excluded[ke] += 1
    return out, excluded, pairs


def one(path):
    stem = os.path.basename(path).removesuffix(".parquet")
    prov, _, game = stem.partition("__")
    a = pd.read_parquet(path)
    rec = {"stem": stem, "provider": prov}
    old_p = os.path.join(OLD, "_actions", f"{stem}.parquet")
    old = pd.read_parquet(old_p) if os.path.exists(old_p) else None
    # The corner table: every corner row is in the corner table.
    corner_rows = int(a["type_id"].isin(CORNER_IDS).sum())
    try:
        rec["corner_table"] = int(len(identify_corners(a, provider=prov)))
    except Exception as exc:  # noqa: BLE001 -- reported, not hidden
        rec["corner_table_error"] = repr(exc)[:200]
        rec["corner_table"] = -1
    rec["corner_rows"] = corner_rows
    if prov == "gradientsports":
        if old is None:
            rec["vs_old_missing_old"] = 1
        else:
            cols = [c for c in a.columns if c != "time_source" and c in old.columns]
            new_s = a.sort_values("action_id", kind="mergesort").reset_index(drop=True)[cols]
            old_s = old.sort_values("action_id", kind="mergesort").reset_index(drop=True)[cols]
            rec["vs_old_rows"] = len(new_s)
            rec["vs_old_len_equal"] = int(len(new_s) == len(old_s))
            diff = []
            if len(new_s) == len(old_s):
                for c in cols:
                    x, y = new_s[c], old_s[c]
                    if not (x.isna().to_numpy() == y.isna().to_numpy()).all() or not (
                        x[x.notna()].astype(str).to_numpy() == y[y.notna()].astype(str).to_numpy()
                    ).all():
                        diff.append(c)
            rec["vs_old_diff_cols"] = diff
            rec["vs_old_cols_only_new"] = sorted(set(a.columns) - set(old.columns))
            rec["vs_old_cols_only_old"] = sorted(set(old.columns) - set(a.columns))
    if prov != "skillcorner":
        return rec
    # Equal to the old stage, SkillCorner: native rows joined on (game_id, original_event_id).
    nat = a[a["action_provenance"] == "native"]
    rec["sc_native_rows"] = len(nat)
    rec["sc_native_key_dups"] = int(nat.duplicated(["game_id", "original_event_id"]).sum())
    if old is not None:
        old_nat = old[old["action_provenance"] == "native"] if "action_provenance" in old else old
        j = nat.merge(old_nat[["game_id", "original_event_id", "time_seconds", "type_id"]],
                      on=["game_id", "original_event_id"], how="outer", suffixes=("", "_old"), indicator=True)
        both = j[j["_merge"] == "both"]
        rec["vs_old_sc_joined"] = len(both)
        rec["vs_old_sc_only_new"] = int((j["_merge"] == "left_only").sum())
        rec["vs_old_sc_only_old"] = int((j["_merge"] == "right_only").sum())
        rec["vs_old_sc_time_unequal"] = int((both["time_seconds"] != both["time_seconds_old"]).sum())
        rec["vs_old_sc_type_changed"] = int((both["type_id"] != both["type_id_old"]).sum())
    # Census, kick share, section 8: derived restart and penalty rows against the census.
    cen, excluded, pairs = census(game)
    rec["census_period_end_excluded"] = dict(excluded)
    der = a[(a["action_provenance"] == "derived") & a["type_id"].isin(TYPE_KIND)].copy()
    der["kind"] = der["type_id"].map(TYPE_KIND)
    keys = {(str(r.original_event_id), r.kind): r for r in der.itertuples(index=False)}
    rec["derived_key_dups"] = int(len(der) - len(keys))
    rows = []
    for (key, kind), (cls, follower) in cen.items():
        r = keys.get((key, kind))
        rows.append({"kind": kind, "cls": cls, "follower": follower, "found": r is not None,
                     "time_source": None if r is None else r.time_source,
                     "nan_start": None if r is None else bool(pd.isna(r.start_x)),
                     "has_player": None if r is None else bool(pd.notna(r.player_id)),
                     "success": None if r is None else bool(r.result_id == SUCCESS)})
    rec["census"] = rows
    rec["staged_not_in_census"] = [k for k in keys if k not in cen]
    rec["native_restart_type_rows"] = int(a[(a["action_provenance"] == "native") & a["type_id"].isin(TYPE_KIND)].shape[0])
    rec["derived_restart_with_player"] = int(der["player_id"].notna().sum())
    # Section 4.9: frames inside b of a marked pair that are live, by marker kind.
    shard = glob.glob(os.path.join(STAGE, "shards", "*", f"{stem}.parquet"))
    old_shard = glob.glob(os.path.join(OLD, "shards", "*", f"{stem}.parquet"))
    if shard:
        cols = ["frame_id", "stoppage_state", "stoppage_source"]
        fr = pd.read_parquet(shard[0], columns=cols).drop_duplicates("frame_id").set_index("frame_id").astype(str)
        rec["frames_total"] = int(len(fr))
        old_state = (pd.read_parquet(old_shard[0], columns=cols[:2]).drop_duplicates("frame_id").set_index("frame_id")
                     ["stoppage_state"].astype(str).reindex(fr.index)) if old_shard else None
        fid = fr.index.to_numpy()
        live = fr["stoppage_state"].to_numpy() == "live"
        ball_out_unknown = (fr["stoppage_state"].to_numpy() == "unknown") & (fr["stoppage_source"].to_numpy() == "ball_out")
        became = (old_state.to_numpy() == "unknown") & live if old_state is not None else None
        by_live, by_became, by_bo = Counter(), Counter(), Counter()
        for s, e, k in pairs:
            sel = (fid >= s) & (fid <= e)
            by_live[k] += int((sel & live).sum())
            by_bo[k] += int((sel & ball_out_unknown).sum())
            if became is not None:
                by_became[k] += int((sel & became).sum())
        rec["b_live_frames"] = dict(by_live)
        rec["b_ball_out_unknown_frames"] = dict(by_bo)
        if became is not None:
            rec["b_became_live_frames"] = dict(by_became)
            rec["became_live_all_frames"] = int(became.sum())
            rec["transitions"] = {f"{o}->{n}": int(v) for (o, n), v in
                                  pd.crosstab(old_state, fr["stoppage_state"]).stack().items() if v}
    return rec


def main():
    files = sorted(glob.glob(os.path.join(STAGE, "_actions", "*.parquet")))
    with Pool(WORKERS) as p:
        recs = p.map(one, files)
    summ = {"stage": STAGE, "old": OLD, "matches": Counter(r["provider"] for r in recs)}
    gs = [r for r in recs if r["provider"] == "gradientsports"]
    summ["gradientsports_equal_to_old_stage"] = {
        "matches": len(gs), "rows": sum(r.get("vs_old_rows", 0) for r in gs),
        "len_unequal": sum(1 - r.get("vs_old_len_equal", 1) for r in gs),
        "missing_old": sum(r.get("vs_old_missing_old", 0) for r in gs),
        "diff_cols": Counter(c for r in gs for c in r.get("vs_old_diff_cols", [])),
        "cols_only_new": sorted({c for r in gs for c in r.get("vs_old_cols_only_new", [])}),
        "cols_only_old": sorted({c for r in gs for c in r.get("vs_old_cols_only_old", [])}),
    }
    sc = [r for r in recs if r["provider"] == "skillcorner"]
    summ["skillcorner_native_times_equal"] = {k: sum(r.get(k, 0) for r in sc) for k in (
        "sc_native_rows", "sc_native_key_dups", "vs_old_sc_joined", "vs_old_sc_only_new", "vs_old_sc_only_old",
        "vs_old_sc_time_unequal", "vs_old_sc_type_changed")}
    cen = pd.DataFrame([dict(row, stem=r["stem"]) for r in sc for row in r["census"]])
    per_match = cen.groupby("stem")["found"].agg(["size", "sum"])
    summ["restart_census"] = {
        "census_restarts_and_penalties": int(len(cen)),
        "found": int(cen["found"].sum()),
        "census_not_staged": int((~cen["found"]).sum()),
        "census_not_staged_by_kind": cen[~cen["found"]]["kind"].value_counts().to_dict(),
        "staged_not_in_census": sum(len(r["staged_not_in_census"]) for r in sc),
        "matches_with_a_count_difference": int(((per_match["size"] != per_match["sum"])).sum()
                                               + sum(1 for r in sc if r["staged_not_in_census"])),
        "derived_key_dups": sum(r["derived_key_dups"] for r in sc),
        "derived_restart_with_player": sum(r["derived_restart_with_player"] for r in sc),
        "native_restart_type_rows": sum(r["native_restart_type_rows"] for r in sc),
        "period_end_excluded": dict(sum((Counter(r["census_period_end_excluded"]) for r in sc), Counter())),
        "by_kind_class": {f"{k}/{c}": int(v) for (k, c), v in cen.groupby(["kind", "cls"]).size().items()},
    }
    f = cen[cen["found"]]
    summ["kick_share_by_kind"] = {
        k: {"rows": int(len(g)), "kick": int((g["time_source"] == "kick").sum()),
            "share": round(float((g["time_source"] == "kick").mean()), 4),
            "time_source": g["time_source"].value_counts().to_dict()}
        for k, g in f.groupby("kind")}
    summ["kick_share_by_kind_and_census_class"] = {
        f"{k}/{c}": round(float((g["time_source"] == "kick").mean()), 4) for (k, c), g in f.groupby(["kind", "cls"])}
    oc = f[(f["kind"] == "corner") & (f["cls"] == "after_only")]
    summ["order_only_corners_kick_by_follower"] = {
        fl or "(none)": {"n": int(len(g)), "kick": int((g["time_source"] == "kick").sum())}
        for fl, g in oc.groupby(oc["follower"].str.rsplit("_", n=1).str[0])}
    summ["nan_start_by_kind"] = {k: int(g["nan_start"].sum()) for k, g in f.groupby("kind")}
    pen = f[f["kind"] == "penalty"]
    summ["penalty_rows"] = {
        "n": int(len(pen)), "by_class": pen["cls"].value_counts().to_dict(),
        "success": int(pen["success"].sum()), "by_time_source": pen["time_source"].value_counts().to_dict(),
        "kick_share": round(float((pen["time_source"] == "kick").mean()), 4) if len(pen) else None}
    summ["corner_table"] = {
        "matches": len(recs),
        "corner_rows": sum(r["corner_rows"] for r in recs),
        "corner_table": sum(r["corner_table"] for r in recs if r["corner_table"] >= 0),
        "matches_rows_ne_table": sorted(r["stem"] for r in recs if r["corner_rows"] != r["corner_table"]),
        "table_errors": {r["stem"]: r["corner_table_error"] for r in recs if "corner_table_error" in r},
        "skillcorner_corner_rows_vs_census_corners": [
            sum(r["corner_rows"] for r in sc), int(((cen["kind"] == "corner")).sum())],
    }
    tot = sum(r.get("frames_total", 0) for r in sc)

    def _sum(field):
        return sum((Counter(r.get(field, {})) for r in sc), Counter())

    became, live_b, bo_b = _sum("b_became_live_frames"), _sum("b_live_frames"), _sum("b_ball_out_unknown_frames")
    four = sum(v for k, v in became.items() if k in RESTARTS)
    summ["section_4_9_became_live"] = {
        "meaning": "SkillCorner frames inside b of a marked pair that are unknown on the f5fc2e2b stage and live on "
                   "this stage, as a share of all staged SkillCorner frames (unique frame_id per match)",
        "sc_frames": tot,
        "total_share": round(sum(became.values()) / tot, 4) if tot else None,
        "four_restart_types_share": round(four / tot, 4) if tot else None,
        "by_kind": {k: round(v / tot, 5) for k, v in sorted(became.items(), key=lambda kv: -kv[1])} if tot else {},
        "all_frames_unknown_to_live": sum(r.get("became_live_all_frames", 0) for r in sc),
        "b_frames_live": sum(live_b.values()),
        "b_frames_kept_unknown_by_ball_out_rule": sum(bo_b.values()),
        "transitions_all_frames": dict(_sum("transitions")),
    }
    out = OUT
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(cen).to_parquet(os.path.join(out, "acc_counts_census.parquet"))
    with open(os.path.join(out, "acc_counts.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda o: o if not isinstance(o, (np.integer, tuple)) else (
            int(o) if isinstance(o, np.integer) else "/".join(map(str, o))))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Acceptance counts of the repaired stage against the f5fc2e2b stage.")
    ap.add_argument("--stage", required=True, help="the repaired stage root (holds _actions/ and shards/)")
    ap.add_argument("--old-stage", required=True, help="the f5fc2e2b stage root")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    STAGE, OLD, OUT, WORKERS = args.stage, args.old_stage, args.out, args.workers
    RAW = os.path.join(args.cache_dir, "skillcorner")
    main()
