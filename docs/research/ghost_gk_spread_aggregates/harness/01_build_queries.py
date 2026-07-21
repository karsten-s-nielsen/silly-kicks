"""Build query feature vectors from EVERY real frame fixture we can find."""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np, pandas as pd
REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux")
sys.path.insert(0, str(REPO))
OUT = pathlib.Path(__file__).resolve().parent
warnings.filterwarnings("ignore")
from silly_kicks.tracking._ghost_gk import _serve_positions_core, GHOST_GK_FEATURE_NAMES
from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

CANDS = [
    ("metrica_slim",  "tests/datasets/tracking/action_context_slim/metrica_slim.parquet"),
    ("sportec_slim",  "tests/datasets/tracking/action_context_slim/sportec_slim.parquet"),
    ("skillcorner_slim","tests/datasets/tracking/action_context_slim/skillcorner_slim.parquet"),
    ("gs_medium",     "tests/datasets/tracking/gradientsports/medium_halftime.parquet"),
    ("gs_realistic",  "tests/datasets/tracking/gradientsports/realistic.parquet"),
    ("sportec_medium","tests/datasets/tracking/sportec/medium_halftime.parquet"),
    ("sportec_realistic","tests/datasets/tracking/sportec/realistic.parquet"),
    ("sc_lakehouse",  "tests/datasets/tracking/skillcorner/lakehouse_derived.parquet"),
]
frames_all, metas_all = [], []
for name, rel in CANDS:
    p = REPO / rel
    if not p.exists():
        print(f"[skip] {name}: missing"); continue
    try:
        d = pd.read_parquet(p)
        if "__kind" in d.columns:
            d = d[d["__kind"] == "frame"].copy()
        need = {"game_id","period_id","frame_id","team_id","is_ball","is_goalkeeper","x","y"}
        if not need.issubset(d.columns):
            print(f"[skip] {name}: missing cols {sorted(need - set(d.columns))}"); continue
        ngk = int((d["is_goalkeeper"].astype(bool) & ~d["is_ball"].astype(bool)).sum())
        if ngk == 0:
            print(f"[skip] {name}: 0 GK rows"); continue
        d = derive_velocities(smooth_frames(d))
        home = sorted(pd.Series(d["team_id"].dropna().unique()).astype(str))[0]
        home = d["team_id"].dropna().unique()[0]
        _, meta, feats, _, _ = _serve_positions_core(
            d, model="default", home_team_id=home, actions=None, carrier=None, link_frame_ids=None)
        if len(feats) == 0:
            print(f"[skip] {name}: 0 queries"); continue
        f = feats[GHOST_GK_FEATURE_NAMES].copy(); f["__src"] = name
        m = meta.copy(); m["__src"] = name
        frames_all.append(f); metas_all.append(m)
        print(f"[ok]   {name}: {len(f)} queries | provider={d['source_provider'].dropna().unique()[:1]} | games={list(d['game_id'].unique())[:3]}")
    except Exception as e:
        print(f"[fail] {name}: {type(e).__name__}: {e}")

F = pd.concat(frames_all, ignore_index=True); M = pd.concat(metas_all, ignore_index=True)
print("\nTOTAL queries:", len(F))
print(F.groupby("__src").size().to_string())
F.to_parquet(OUT/"queries_all.parquet", index=False); M.to_parquet(OUT/"queries_all_meta.parquet", index=False)
print("dropna rows (any NaN feature):", int(F[GHOST_GK_FEATURE_NAMES].isna().any(axis=1).sum()))
