"""
GK observability by game context -- PUBLIC tier (A-League 2024/25 SkillCorner Open Data, MIT licence).

The ~20 PUBLIC SkillCorner Open Data matches are the only numbers citable in a public document.
(The dual-tier source of truth, including a restricted-pilot regression path and its match-id list,
lives in the research workstream outside this repo; this public copy carries the public tier only.)

Keeper's perspective, per frame: opponent possession by phase / ball zone; own block type; defending set
pieces (restart taker resolved from the `_for/_against` suffix, which is relative to the POSSESSING team);
moment-of-truth windows (corner delivery, goal-kick strike) recovered from ball position because the
restart-tagged possession starts at the RECEPTION; own build-up; baseline.
'unit' = keeper + the 4 deepest outfielders of her team, all is_detected.
Read-only research; prints aggregates only.
"""
import argparse, os, sys, io, json, subprocess, collections
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.json as pj
import pyarrow.parquet as pq
import pyarrow.compute as pc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = "https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1"
TOK = os.environ["PINING_FOR_THE_DATA_TOKEN"]
OUT = Path(__file__).resolve().parent
CACHE_PUB = OUT / "pining_sc_public_cache"
SET_PIECE_WINDOW = 150  # frames (15 s at 10 fps)
TRK_COLS = ["frame", "period", "ball_data", "possession", "player_data"]
TRK_SCHEMA = pa.schema([
    ("frame", pa.int64()),
    ("period", pa.int64()),
    ("ball_data", pa.struct([("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64()),
                             ("is_detected", pa.bool_())])),
    ("possession", pa.struct([("player_id", pa.int64()), ("group", pa.string())])),
    ("player_data", pa.list_(pa.struct([("x", pa.float64()), ("y", pa.float64()),
                                        ("player_id", pa.int64()), ("is_detected", pa.bool_())]))),
])


def curl_to(path, dest):
    subprocess.run(["curl", "-sL", "-o", str(dest), "-H", f"Authorization: Bearer {TOK}",
                    f"{BASE}/{path}"], check=True)
    return dest


def cached(cache, name, path):
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / name
    if not dest.exists() or dest.stat().st_size < 100:
        curl_to(path, dest)
    return dest


# ---- loader: returns (meta: dict, ev: DataFrame, t: pyarrow.Table[TRK_COLS]) --------------
def load_public(mid):
    meta = json.loads(cached(CACHE_PUB, f"{mid}_match.json",
                             f"skillcorner/matches/{mid}/{mid}_match").read_text("utf-8"))
    ev = pd.read_csv(cached(CACHE_PUB, f"{mid}_dynamic_events.csv",
                            f"skillcorner/matches/{mid}/{mid}_dynamic_events"), low_memory=False)
    pq_path = CACHE_PUB / f"{mid}_tracking.parquet"
    if not pq_path.exists():
        raw = curl_to(f"skillcorner/matches/{mid}/{mid}_tracking_extrapolated",
                      CACHE_PUB / f"{mid}_tracking_extrapolated.jsonl")
        with open(raw, "rb") as fh:
            if fh.read(2) == b"\x1f\x8b":
                raise RuntimeError(f"{mid}: gzipped tracking not handled")
        t_full = pj.read_json(str(raw), read_options=pj.ReadOptions(block_size=1 << 24),
                              parse_options=pj.ParseOptions(explicit_schema=TRK_SCHEMA,
                                                            unexpected_field_behavior="ignore"))
        pq.write_table(t_full, pq_path)
        os.remove(raw)  # keep only the compact parquet cache
    return meta, ev, pq.read_table(pq_path, columns=TRK_COLS)


def norm_restart(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).lower()
    for k in ("corner", "free_kick", "throw_in", "goal_kick", "kick_off", "penalty"):
        if k in s:
            return k
    return "other"


acc = collections.defaultdict(lambda: np.zeros(3))   # context -> [n, gk_det, unit_det]
per_gk = []
curve = {"opp": collections.defaultdict(lambda: np.zeros(2)),
         "own": collections.defaultdict(lambda: np.zeros(2))}


def process_match(mid, meta, ev, t, print_taxonomy=False):
    half = float(meta.get("pitch_length") or 105.0) / 2.0
    halfw = float(meta.get("pitch_width") or 68.0) / 2.0
    home_id, away_id = int(meta["home_team"]["id"]), int(meta["away_team"]["id"])
    pid2team, gk_ids = {}, set()
    for p in meta.get("players", []):
        pid = p.get("id") if p.get("id") is not None else p.get("player_id")
        if pid is None:
            continue
        if p.get("team_id") is not None:
            pid2team[int(pid)] = int(p["team_id"])
        role = p.get("player_role") or {}
        if role.get("position_group") == "Goalkeeper" or role.get("acronym") == "GK":
            gk_ids.add(int(pid))
    for pid, tid in ev[["player_id", "team_id"]].dropna().drop_duplicates().itertuples(index=False):
        pid2team.setdefault(int(pid), int(tid))

    frame = t.column("frame").to_numpy()
    NF = len(frame)
    period = pc.fill_null(t.column("period").combine_chunks(), -1).to_numpy(zero_copy_only=False).astype(int)
    ball = t.column("ball_data").combine_chunks()
    bx = pc.struct_field(ball, "x").to_numpy(zero_copy_only=False).astype(float)
    by = pc.struct_field(ball, "y").to_numpy(zero_copy_only=False).astype(float)
    groups = pc.struct_field(t.column("possession").combine_chunks(), "group").to_pylist()
    gmap = {"home team": home_id, "away team": away_id, "home": home_id, "away": away_id}
    poss_team = np.array([gmap.get(str(g).lower(), np.nan) if g is not None else np.nan for g in groups],
                         dtype=float)

    pdl = t.column("player_data").combine_chunks()
    flat = pc.list_flatten(pdl)
    parent = pc.list_parent_indices(pdl).to_numpy()
    L = pd.DataFrame({
        "fi": parent,
        "pid": pc.struct_field(flat, "player_id").to_numpy(zero_copy_only=False),
        "x": pc.struct_field(flat, "x").to_numpy(zero_copy_only=False).astype(float),
        "det": pc.fill_null(pc.struct_field(flat, "is_detected"), False).to_numpy(zero_copy_only=False).astype(bool),
    }).dropna(subset=["pid", "x"])
    L["pid"] = L["pid"].astype(np.int64)
    L["team"] = L["pid"].map(pid2team)
    L["is_gk"] = L["pid"].isin(gk_ids)
    L["period"] = period[L["fi"].to_numpy()]

    pp = ev[ev["event_type"] == "player_possession"].copy()
    for c in ("frame_start", "frame_end", "team_id"):
        pp[c] = pd.to_numeric(pp[c], errors="coerce")
    pp = pp.dropna(subset=["frame_start", "frame_end", "team_id"]).sort_values("frame_start")
    if print_taxonomy:
        print(f"[taxonomy from {mid}]")
        print("  possession groups:", sorted({str(g) for g in groups if g is not None}))
        for c in ("team_in_possession_phase_type", "team_out_of_possession_phase_type",
                  "game_interruption_before", "start_type"):
            if c in pp.columns:
                print(f"  {c}:", dict(pp[c].value_counts(dropna=False).head(14)))

    ip_team = np.full(NF, np.nan)
    ip_phase = np.full(NF, None, dtype=object)
    oop_phase = np.full(NF, None, dtype=object)
    a_idx = np.searchsorted(frame, pp["frame_start"].to_numpy())
    b_idx = np.searchsorted(frame, pp["frame_end"].to_numpy(), side="right")
    for (a, b), (_, r) in zip(zip(a_idx, b_idx), pp.iterrows()):
        ip_team[a:b] = r["team_id"]
        ip_phase[a:b] = r.get("team_in_possession_phase_type")
        oop_phase[a:b] = r.get("team_out_of_possession_phase_type")
    ph = pd.DataFrame({"period": period, "ip_team": ip_team, "ip_phase": ip_phase, "oop_phase": oop_phase})
    ph[["ip_team", "ip_phase", "oop_phase"]] = ph.groupby("period")[["ip_team", "ip_phase", "oop_phase"]].ffill()
    ip_team, ip_phase, oop_phase = ph["ip_team"].to_numpy(), ph["ip_phase"].to_numpy(), ph["oop_phase"].to_numpy()

    # restart windows; taker resolved from the suffix (relative to the POSSESSING team)
    sp_type = np.full(NF, None, dtype=object)
    sp_team = np.full(NF, np.nan)
    sp_age = np.full(NF, np.nan)
    restarts = []
    for ai, (_, r) in zip(a_idx, pp.iterrows()):
        lab = r.get("game_interruption_before")
        typ = norm_restart(lab)
        if typ is None:
            continue
        pt = int(r["team_id"])
        other = away_id if pt == home_id else home_id
        s = str(lab).lower()
        taker = pt if s.endswith("_for") else (other if s.endswith("_against") else np.nan)
        restarts.append((ai, typ, taker))
    for k, (ai, typ, taker) in enumerate(restarts):
        end = min(ai + SET_PIECE_WINDOW, NF)
        if k + 1 < len(restarts):
            end = min(end, restarts[k + 1][0])
        if end <= ai:
            continue
        sp_type[ai:end] = typ
        sp_team[ai:end] = taker
        sp_age[ai:end] = np.arange(end - ai)
    mom_type = np.full(NF, None, dtype=object)
    mom_team = np.full(NF, np.nan)
    for (ai, typ, taker) in restarts:
        if typ == "corner":
            lo = max(0, ai - 100)
            seg = np.where((np.abs(bx[lo:ai + 1]) > half - 2.5) & (np.abs(by[lo:ai + 1]) > halfw - 2.5))[0]
            if len(seg):
                d = lo + seg[-1]
                a, b = max(0, d - 20), min(NF, d + 11)
                mom_type[a:b] = "corner"
                mom_team[a:b] = taker
        elif typ == "goal_kick":
            lo = max(0, ai - 150)
            seg = np.where((np.abs(bx[lo:ai + 1]) > half - 6.0) & (np.abs(by[lo:ai + 1]) < 10.0))[0]
            if len(seg):
                kf = lo + seg[-1]
                a, b = max(0, kf - 20), min(NF, kf + 6)
                mom_type[a:b] = "goal_kick"
                mom_team[a:b] = taker

    for team in (home_id, away_id):
        G = L[L["is_gk"] & (L["team"] == team)].drop_duplicates("fi")
        if len(G) < 500:
            continue
        gk_det = np.zeros(NF, bool)
        gk_present = np.zeros(NF, bool)
        gk_det[G["fi"].to_numpy()] = G["det"].to_numpy()
        gk_present[G["fi"].to_numpy()] = True
        own_side = {}
        for pv, grp in G[G["det"]].groupby("period"):
            if pv > 0 and len(grp) >= 50:
                own_side[pv] = 1.0 if np.median(grp["x"]) > 0 else -1.0
        side = np.array([own_side.get(p, np.nan) for p in period])
        ball_dist = half - side * bx
        in_box = (ball_dist < 16.5) & (np.abs(by) < 20.16)

        O = L[(~L["is_gk"]) & (L["team"] == team)].copy()
        O["d"] = half - side[O["fi"].to_numpy()] * O["x"].to_numpy()
        O = O.dropna(subset=["d"]).sort_values(["fi", "d"])
        O["rk"] = O.groupby("fi").cumcount()
        O4 = O[O["rk"] < 4].groupby("fi")["det"].agg(["all", "count"])
        out4 = np.zeros(NF, bool)
        out4[O4[(O4["count"] == 4) & O4["all"]].index.to_numpy()] = True
        unit = gk_det & out4

        live = gk_present & (period > 0) & ~np.isnan(ball_dist)
        opp = live & ~np.isnan(poss_team) & (poss_team != team)
        own = live & (poss_team == team)
        opp_ev = live & ~np.isnan(ip_team) & (ip_team != team)
        own_ev = live & (ip_team == team)
        ctx = {
            "00 baseline (all live frames)": live,
            "10 OPP poss | ball in own half": opp & (ball_dist < half),
            "11 OPP poss | ball in own def third (<35m)": opp & (ball_dist < 35),
            "12 OPP poss | ball in own penalty area": opp & in_box,
            "13 OPP poss | ball 35-52m (own half, beyond third)": opp & (ball_dist >= 35) & (ball_dist < half),
            "40 OWN poss | ball in own def third (<35m)": own & (ball_dist < 35),
            "41 OWN poss | ball in own penalty area": own & in_box,
            "42 OWN poss | ball in opp half": own & (ball_dist >= half),
        }
        for ph_val in pd.unique(ip_phase[opp_ev & (ball_dist < 35)]):
            if isinstance(ph_val, str):
                ctx[f"2x OPP phase={ph_val} | ball <35m"] = opp_ev & (ball_dist < 35) & (ip_phase == ph_val)
        for bl in pd.unique(oop_phase[opp_ev]):
            if isinstance(bl, str):
                ctx[f"3x OWN block={bl} | ball in own half"] = opp_ev & (ball_dist < half) & (oop_phase == bl)
        for ph_val in pd.unique(ip_phase[own_ev & (ball_dist < 35)]):
            if isinstance(ph_val, str):
                ctx[f"5x OWN phase={ph_val} | ball <35m"] = own_ev & (ball_dist < 35) & (ip_phase == ph_val)
        opp_sp = live & ~np.isnan(sp_team) & (sp_team != team)
        own_sp = live & (sp_team == team)
        first5 = sp_age < 50
        ctx["60 DEFENDING corner (15s)"] = opp_sp & (sp_type == "corner")
        ctx["60b DEFENDING corner (first 5s)"] = opp_sp & (sp_type == "corner") & first5
        ctx["60c DEFENDING corner | ball in own box"] = opp_sp & (sp_type == "corner") & in_box
        ctx["61 DEFENDING free kick in own half (15s)"] = opp_sp & (sp_type == "free_kick") & (ball_dist < half)
        ctx["61b DEFENDING free kick <35m (15s)"] = opp_sp & (sp_type == "free_kick") & (ball_dist < 35)
        ctx["62 DEFENDING throw-in in own third (15s)"] = opp_sp & (sp_type == "throw_in") & (ball_dist < 35)
        ctx["70 OWN goal kick (15s)"] = own_sp & (sp_type == "goal_kick")
        ctx["70b OWN goal kick (first 5s)"] = own_sp & (sp_type == "goal_kick") & first5
        ctx["71 OPP goal kick (15s) [sanity: expect ~0]"] = opp_sp & (sp_type == "goal_kick")
        opp_mom = live & ~np.isnan(mom_team) & (mom_team != team)
        own_mom = live & (mom_team == team)
        ctx["63 DEFENDING corner | delivery moment (-2s..+1s)"] = opp_mom & (mom_type == "corner")
        ctx["72 OWN goal kick | strike moment (-2s..+0.5s)"] = own_mom & (mom_type == "goal_kick")
        ctx["73 OPP goal kick | strike moment [sanity: expect ~0]"] = opp_mom & (mom_type == "goal_kick")

        for name, m in ctx.items():
            n = int(m.sum())
            if n == 0:
                continue
            acc[name] += [n, gk_det[m].sum(), unit[m].sum()]
            per_gk.append(dict(match=mid, team=team, context=name, n=n,
                               gk_rate=gk_det[m].mean(), unit_rate=unit[m].mean()))
        bins = np.digitize(ball_dist, np.arange(0, 106, 10.5)) - 1
        for key, mask in (("opp", opp), ("own", own)):
            for bi in range(10):
                sel = mask & (bins == bi)
                curve[key][bi] += [sel.sum(), gk_det[sel].sum()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["public"], default="public",
                    help="public tier only (the restricted-pilot path lives in the external workstream SSOT)")
    args = ap.parse_args()
    mj = json.loads(subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {TOK}",
                                    f"{BASE}/skillcorner/matches"], capture_output=True).stdout)
    matches = [m["id"] for m in mj["matches"] if m.get("visibility") == "public"]
    loader = load_public
    print(f"tier={args.tier}  matches={len(matches)}")
    for i, mid in enumerate(matches):
        try:
            meta, ev, t = loader(mid)
            process_match(mid, meta, ev, t, print_taxonomy=(i == 0))
            print(f"[ok {mid}]")
        except Exception as e:
            print(f"[ERR {mid}] {type(e).__name__}: {e}")

    print("\n==== GK DETECTION BY CONTEXT (pooled; unit = keeper + 4 deepest outfielders all detected) ====")
    spread = pd.DataFrame(per_gk)
    rows = []
    for name in sorted(acc):
        n, g, u = acc[name]
        s = spread[spread.context == name]
        rows.append((name, int(n), g / n, u / n, s.gk_rate.quantile(0.25) if len(s) else np.nan,
                     s.gk_rate.quantile(0.75) if len(s) else np.nan, len(s)))
    tab = pd.DataFrame(rows, columns=["context", "frames", "gk_det", "unit_det", "gk_q25", "gk_q75", "gk_matches"])
    with pd.option_context("display.width", 220, "display.max_colwidth", 60, "display.max_rows", 200):
        print(tab.round(3).to_string(index=False))
    print("\n==== GK DETECTION vs BALL DISTANCE, split by possession ====")
    for bi in range(10):
        c = 5.25 + 10.5 * bi
        no, go = curve["opp"][bi]
        nw, gw = curve["own"][bi]
        print(f"  ball {c:5.1f} m from own goal | OPP poss: {go/no if no else float('nan'):.3f} (n={int(no):,})"
              f" | OWN poss: {gw/nw if nw else float('nan'):.3f} (n={int(nw):,})")
    tab.to_csv(OUT / f"gk_observability_{args.tier}.csv", index=False)
    spread.to_csv(OUT / f"gk_observability_{args.tier}_pergk.csv", index=False)
    cv = pd.DataFrame([dict(bin_center_m=5.25 + 10.5 * bi,
                            opp_n=int(curve["opp"][bi][0]), opp_gk_det=(curve["opp"][bi][1] / curve["opp"][bi][0]) if curve["opp"][bi][0] else np.nan,
                            own_n=int(curve["own"][bi][0]), own_gk_det=(curve["own"][bi][1] / curve["own"][bi][0]) if curve["own"][bi][0] else np.nan)
                       for bi in range(10)])
    cv.to_csv(OUT / f"gk_observability_{args.tier}_curve.csv", index=False)
    print("\nCSV:", OUT / f"gk_observability_{args.tier}.csv")


if __name__ == "__main__":
    main()
