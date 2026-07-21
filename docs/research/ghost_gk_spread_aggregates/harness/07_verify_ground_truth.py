"""INDEPENDENT reproduction of the saved gt.npz ground truth.

Loads the model from a TEMP COPY of the bundled weights (never the repo path),
rebuilds the exact same stratified query set the prior run used, re-runs the REAL
production path (_vectorized_leaf_indices -> predict_density(kde_backend="vectorized"))
on a random subset, and compares spread/mode/mean against the saved artifact.

Also recomputes the multimodality diagnostics with BOTH a permissive (>=) and a
STRICT (>) local-maximum rule, since the prior run's `>=` rule counts plateaus.
"""
from __future__ import annotations
import pathlib, sys, time, warnings
import numpy as np, pandas as pd

REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux")
sys.path.insert(0, str(REPO))
D = pathlib.Path(__file__).resolve().parent
AGG = D.parent / "agg"
warnings.filterwarnings("ignore")

from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, GHOST_GK_FEATURE_NAMES, _vectorized_leaf_indices,
    GRID_NX, GRID_NY, GRID_RESOLUTION)

N_CHECK = int(sys.argv[1]) if len(sys.argv) > 1 else 60

m = GhostGkModel.load(D / "weights_copy")          # TEMP COPY, not the repo
print(f"[load] from temp copy: n_train={m._training_leaves.shape[0]} "
      f"n_trees={m._training_leaves.shape[1]}", flush=True)

# --- rebuild the prior run's query set EXACTLY (same code, same TARGET, same order)
Q = pd.read_parquet(AGG / "queries_all.parquet")
Qm = pd.read_parquet(AGG / "queries_all_meta.parquet")
ok = ~Q[GHOST_GK_FEATURE_NAMES].isna().any(axis=1).values
Q = Q[ok].reset_index(drop=True); Qm = Qm[ok].reset_index(drop=True)
TARGET = 480
parts = []
for src, g in Qm.groupby("__src", sort=True):
    take = max(1, int(round(TARGET * len(g) / len(Qm))))
    parts.append(np.asarray(g.index)[np.linspace(0, len(g) - 1, min(take, len(g))).astype(int)])
keep = np.sort(np.concatenate(parts))
Q = Q.iloc[keep].reset_index(drop=True); Qm = Qm.iloc[keep].reset_index(drop=True)
NQ = len(Q)

Z = np.load(AGG / "gt.npz", allow_pickle=False)
saved_spread = Z["spread"]; saved_mode = Z["mode_xy"]; saved_mean = Z["mean_xy"]
saved_src = Z["src"].astype(str)
print(f"[q] rebuilt n_queries={NQ} | saved n={len(saved_spread)}", flush=True)
assert NQ == len(saved_spread), "query-set size mismatch -> cannot compare"
assert (Qm["__src"].values.astype(str) == saved_src).all(), "source order mismatch"
print("[q] source vector matches saved artifact exactly", flush=True)

XQ = Q[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)
QL = _vectorized_leaf_indices(m._tree_nodes, XQ)
assert (QL == Z["QL"]).all(), "leaf routing differs from saved artifact"
print(f"[route] leaf routing reproduces saved QL exactly {QL.shape}", flush=True)

rng = np.random.default_rng(11)
pick = np.sort(rng.choice(NQ, size=min(N_CHECK, NQ), replace=False))
print(f"[check] re-running production predict_density on {len(pick)} random queries", flush=True)

rep_spread = np.empty(len(pick)); rep_mode = np.empty((len(pick), 2)); rep_mean = np.empty((len(pick), 2))
nm_ge = np.empty(len(pick), dtype=int); nm_gt = np.empty(len(pick), dtype=int)
edge = np.empty(len(pick))
CH = 12
t0 = time.time()
for a in range(0, len(pick), CH):
    b = min(a + CH, len(pick))
    idx = pick[a:b]
    ds = m.predict_density(Q.iloc[idx][GHOST_GK_FEATURE_NAMES])   # default kde_backend="vectorized"
    for i, d in enumerate(ds):
        j = a + i
        rep_spread[j] = d.spread
        rep_mode[j] = (d.mode_x, d.mode_y); rep_mean[j] = (d.mean_x, d.mean_y)
        P = d.probabilities
        pk = P.max()
        Pp_ge = np.pad(P, 1, mode="constant", constant_values=-1.0)
        Pp_gt = np.pad(P, 1, mode="constant", constant_values=-1.0)
        loc_ge = np.ones_like(P, dtype=bool); loc_gt = np.ones_like(P, dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = Pp_ge[1 + dx:1 + dx + GRID_NX, 1 + dy:1 + dy + GRID_NY]
                loc_ge &= P >= nb
                loc_gt &= P > nb
        nm_ge[j] = int((loc_ge & (P >= 0.2 * pk)).sum())
        nm_gt[j] = int((loc_gt & (P >= 0.2 * pk)).sum())
        edge[j] = P[0, :].sum() + P[-1, :].sum() + P[:, 0].sum() + P[:, -1].sum()
    print(f"  [rep] {b}/{len(pick)}  {time.time()-t0:.0f}s", flush=True)

ds_ = np.abs(rep_spread - saved_spread[pick])
print("\n=== REPRODUCTION vs SAVED gt.npz ===")
print(f"  max |spread_repro - spread_saved| = {ds_.max():.6e}  (relative {np.max(ds_/saved_spread[pick]):.3e})")
print(f"  max |mode diff|  = {np.abs(rep_mode - saved_mode[pick]).max():.6e}")
print(f"  max |mean diff|  = {np.abs(rep_mean - saved_mean[pick]).max():.6e}")
print(f"  exact bitwise spread match: {int((rep_spread == saved_spread[pick]).sum())}/{len(pick)}")

print("\n=== MULTIMODALITY on the re-run subset ===")
print(f"  permissive (>=) rule : 1 mode {100*(nm_ge==1).mean():.1f}%  >=2 modes {100*(nm_ge>=2).mean():.1f}%  max {nm_ge.max()}")
print(f"  STRICT     (> ) rule : 1 mode {100*(nm_gt==1).mean():.1f}%  >=2 modes {100*(nm_gt>=2).mean():.1f}%  max {nm_gt.max()}")
saved_nm = Z["n_modes"][pick]
print(f"  saved n_modes on same subset: 1 mode {100*(saved_nm==1).mean():.1f}%  >=2 {100*(saved_nm>=2).mean():.1f}%")
print(f"  agreement saved-vs-repro (>= rule): {int((saved_nm==nm_ge).sum())}/{len(pick)}")
dist = np.hypot(rep_mode[:,0]-rep_mean[:,0], rep_mode[:,1]-rep_mean[:,1])
print(f"  |mode-mean| (m): median {np.median(dist):.3f}  p90 {np.percentile(dist,90):.3f}  max {dist.max():.3f}")
print(f"  grid-edge mass: median {np.median(edge):.4e}  max {edge.max():.4e}")

np.savez_compressed(D/"verify.npz", pick=pick, rep_spread=rep_spread, rep_mode=rep_mode,
                    rep_mean=rep_mean, nm_ge=nm_ge, nm_gt=nm_gt, edge=edge)
print("\nSAVED verify.npz", flush=True)
