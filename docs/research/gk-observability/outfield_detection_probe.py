"""Outfield per-player detection % on the PUBLIC SkillCorner corpus (A-League 2024/25, MIT).

Sizes TF-58's exposure: its collective variables (compute_team_shape / _defensive_line) consume
OUTFIELD positions and do NOT consult `visibility`. This measures how observed the outfield actually
is, so TF-58's per-player-observation policy is a measured decision, not an assumption.

Reuses the single-source loader (gk_observability_multi.load_public) + the same flattening. Public tier
only (matches with visibility=="public"). Read-only; prints aggregates only. Public numbers = citable.
"""
import collections
import io
import json
import os
import subprocess
import sys
from pathlib import Path

# Public token, fail-closed public-only. Set BEFORE importing the loader (it reads the env at import).
os.environ.setdefault("PINING_FOR_THE_DATA_TOKEN", "test-token-pining-for-the-data")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from gk_observability_multi import BASE, TOK, TRK_COLS, load_public  # noqa: E402

# NB: gk_observability_multi already rewraps sys.stdout for UTF-8 at import; do NOT rewrap again
# (double-wrapping the same buffer → GC closes it → "I/O operation on closed file").

# pooled accumulators
pf = collections.defaultdict(float)  # scalar sums
per_match = []


def process_match_outfield(mid, meta, ev, t):
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

    period = pc.fill_null(t.column("period").combine_chunks(), -1).to_numpy(zero_copy_only=False).astype(int)
    pdl = t.column("player_data").combine_chunks()
    flat = pc.list_flatten(pdl)
    parent = pc.list_parent_indices(pdl).to_numpy()
    L = pd.DataFrame(
        {
            "fi": parent,
            "pid": pc.struct_field(flat, "player_id").to_numpy(zero_copy_only=False),
            "det": pc.fill_null(pc.struct_field(flat, "is_detected"), False)
            .to_numpy(zero_copy_only=False)
            .astype(bool),
        }
    ).dropna(subset=["pid"])
    L["pid"] = L["pid"].astype(np.int64)
    L["team"] = L["pid"].map(pid2team)
    L["is_gk"] = L["pid"].isin(gk_ids)
    L["period"] = period[L["fi"].to_numpy()]

    # in-play only (period > 0), outfield only (not GK), with a resolved team
    O = L[(~L["is_gk"]) & (L["period"] > 0) & L["team"].notna()].copy()
    if len(O) < 1000:
        return
    # (a) per-player-frame detection rate
    rate = float(O["det"].mean())
    # (b) per-(frame, team) detected-outfielder count out of those present
    grp = O.groupby(["fi", "team"])["det"].agg(["sum", "count"])
    grp = grp[grp["count"] >= 1]
    det_per_frame = grp["sum"].to_numpy()
    present_per_frame = grp["count"].to_numpy()
    nfr = len(grp)
    # coverage thresholds on DETECTED outfielders (team has 10 outfielders nominal)
    ge = {k: float((det_per_frame >= k).mean()) for k in (10, 9, 8, 7, 5)}
    per_match.append(
        dict(
            match=mid,
            outfield_det_rate=rate,
            mean_present=float(present_per_frame.mean()),
            mean_detected=float(det_per_frame.mean()),
            frac_ge10=ge[10],
            frac_ge9=ge[9],
            frac_ge8=ge[8],
            frac_ge7=ge[7],
            frac_ge5=ge[5],
            n_team_frames=nfr,
        )
    )
    pf["player_frames"] += len(O)
    pf["player_frames_det"] += float(O["det"].sum())
    pf["team_frames"] += nfr
    pf["sum_detected"] += float(det_per_frame.sum())
    pf["sum_present"] += float(present_per_frame.sum())
    for k in (10, 9, 8, 7, 5):
        pf[f"tf_ge{k}"] += float((det_per_frame >= k).sum())


def main():
    mj = json.loads(
        subprocess.run(
            ["curl", "-sL", "-H", f"Authorization: Bearer {TOK}", f"{BASE}/skillcorner/matches"],
            capture_output=True,
        ).stdout
    )
    matches = [m["id"] for m in mj["matches"] if m.get("visibility") == "public"]
    print(f"PUBLIC SkillCorner matches: {len(matches)}")
    for i, mid in enumerate(matches):
        try:
            meta, ev, t = load_public(mid)
            process_match_outfield(mid, meta, ev, t)
            print(f"[ok {mid}] ({i + 1}/{len(matches)})")
        except Exception as e:
            print(f"[ERR {mid}] {type(e).__name__}: {e}")

    if not per_match:
        print("no matches processed")
        return
    df = pd.DataFrame(per_match)
    OUT = Path(__file__).resolve().parent
    df.to_csv(OUT / "outfield_detection_public.csv", index=False)
    print("\n==== OUTFIELD PER-PLAYER DETECTION (public, in-play frames, pooled over matches) ====")
    print(f"  outfield per-player-frame detection rate : {pf['player_frames_det'] / pf['player_frames']:.3f}  "
          f"(n={int(pf['player_frames']):,} player-frames)")
    print(f"  mean outfielders PRESENT per team-frame  : {pf['sum_present'] / pf['team_frames']:.2f}")
    print(f"  mean outfielders DETECTED per team-frame : {pf['sum_detected'] / pf['team_frames']:.2f}  (of 10 nominal)")
    print(f"  team-frames total                        : {int(pf['team_frames']):,}")
    for k in (10, 9, 8, 7, 5):
        print(f"  frac team-frames with >= {k:2d} detected     : {pf[f'tf_ge{k}'] / pf['team_frames']:.3f}")
    print("\n==== PER-MATCH (spread) ====")
    with pd.option_context("display.width", 200, "display.max_rows", 40):
        print(df.round(3).to_string(index=False))
    print("\nCSV:", OUT / "outfield_detection_public.csv")


if __name__ == "__main__":
    main()
