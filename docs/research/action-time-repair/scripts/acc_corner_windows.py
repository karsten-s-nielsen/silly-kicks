"""The corner readers (spec 5.2) on a repaired stage (read only).

For the SkillCorner matches of a stage (--stage), with the code of the clone it runs from (PYTHONPATH=.):
identify_corners -> add_corner_windows -> the run encoding (bundled RunZoneModel), the routine features (bundled
CornerRoutineModel), the chance quality, and the two drivers' snapshot rows (_snapshot_rows, _divergence_rows).
Counts, over the corners with no kick time: non-missing run / routine / chance values (each must be 0) and snapshot
rows (must be 0). Also counted: corners with a kick time and no corner_side (no encoding, no snapshot row by rule).
"""

import argparse
import glob
import json
import os
import warnings
from types import SimpleNamespace

import pandas as pd
from scripts.build_corner_routines import _snapshot_rows
from scripts.validate_corners import _divergence_rows, dominant_routine

import silly_kicks.corners
from silly_kicks.corners import (
    add_corner_windows,
    compute_corner_chance_quality,
    compute_corner_routine_features,
    identify_corners,
)
from silly_kicks.corners._encode import compute_corner_run_encoding
from silly_kicks.corners._routines import CornerRoutineModel
from silly_kicks.corners._zones import RunZoneModel
from silly_kicks.tracking._ball_contact import detect_ball_contacts

warnings.simplefilter("ignore")
ap = argparse.ArgumentParser(description="The corner readers on the corners with no kick time.")
ap.add_argument("--stage", required=True, help="stage root (holds _actions/ and shards/)")
ROOT = ap.parse_args().stage
SK = os.path.dirname(silly_kicks.corners.__file__)  # the bundled weights of the library that runs
zone = RunZoneModel.load(f"{SK}/_run_zone_weights/default")
routine = CornerRoutineModel.load(f"{SK}/_routine_weights/default")
tot = {"corners": 0, "no_kick": 0, "no_kick_direct": 0, "run_vals": 0, "routine_vals": 0, "dom_not_minus1": 0,
       "chance_vals": 0, "snapshot_rows": 0, "divergence_rows": 0, "kick_no_side": 0}  # fmt: skip
for pa in sorted(glob.glob(f"{ROOT}/_actions/skillcorner__*.parquet")):
    name = os.path.basename(pa)
    actions = pd.read_parquet(pa)
    frames = pd.read_parquet(glob.glob(f"{ROOT}/shards/*/{name}")[0])
    corners = identify_corners(actions, provider="skillcorner")
    windowed = add_corner_windows(corners, frames, detect_ball_contacts(frames))
    no_kick = windowed["t_kick"].isna().to_numpy()
    enc = compute_corner_run_encoding(windowed, frames, model=zone)
    feats = compute_corner_routine_features(enc, model=routine)
    run_cols = [c for c in enc.columns if c.startswith("run_")]
    feat_cols = [c for c in feats.columns if c.startswith("routine_feat_")]
    dom = dominant_routine(feats[feat_cols].to_numpy(dtype=float))
    q = compute_corner_chance_quality(windowed, actions)
    ids_no_kick = set(windowed.loc[no_kick, "corner_id"].astype(int))
    item = SimpleNamespace(actions=actions, frames=frames, provider="skillcorner")
    snaps = _snapshot_rows(item)
    div = _divergence_rows(item, zone)
    m = {
        "corners": len(windowed),
        "no_kick": int(no_kick.sum()),
        "no_kick_direct": int((no_kick & (windowed["corner_kind"].astype(str) == "direct").to_numpy()).sum()),
        "run_vals": int(enc.loc[enc["corner_id"].isin(ids_no_kick), run_cols].notna().to_numpy().sum()),
        "routine_vals": int(feats.loc[feats["corner_id"].isin(ids_no_kick), feat_cols].notna().to_numpy().sum()),
        "dom_not_minus1": int((dom[feats["corner_id"].isin(ids_no_kick).to_numpy()] != -1).sum()),
        "chance_vals": int(q.loc[q["corner_id"].isin(ids_no_kick), ["n_shots", "n_goals"]].notna().to_numpy().sum()),
        "snapshot_rows": int(snaps["corner_id"].isin(ids_no_kick).sum()) if len(snaps) else 0,
        "divergence_rows": len(div),  # the divergence leg is idsse-only by design; reported for completeness
        "kick_no_side": int((~no_kick & windowed["corner_side"].isna().to_numpy()).sum()),
    }
    print("MATCH", name, json.dumps(m))
    for k, v in m.items():
        tot[k] += v
print("TOTAL", json.dumps(tot))
bad = tot["run_vals"] + tot["routine_vals"] + tot["dom_not_minus1"] + tot["chance_vals"] + tot["snapshot_rows"]
print("RESULT", "PASS" if bad == 0 and tot["no_kick"] > 0 else "STOP", "(non-missing values / rows on no-kick corners:",
      bad, ")")
