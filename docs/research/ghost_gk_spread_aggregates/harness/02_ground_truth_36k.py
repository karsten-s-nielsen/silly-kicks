"""Re-measure the per-leaf Gaussian-moment approximation to ghost_gk_density_spread
at PRODUCTION configuration.

Ground truth  : GhostGkModel.predict_density (the real bundled 36k artifact, real
                _vectorized_leaf_indices routing, default kde_backend="vectorized").
Queries       : real feature vectors extracted by _serve_positions_core from real
                tracking-frame fixtures (metrica = NOT in the training corpus).
Predictor     : per-(tree,leaf) moment aggregates {n,Sx,Sy,Sxx,Sxy,Syy} ONLY.
"""
from __future__ import annotations
import pathlib, sys, time, json, warnings
import numpy as np, pandas as pd
REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux"); sys.path.insert(0,str(REPO))
D = pathlib.Path(__file__).resolve().parent
warnings.filterwarnings("ignore")
from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, GHOST_GK_FEATURE_NAMES, _vectorized_leaf_indices, _leaf_match_weights,
    _GRID_X, _GRID_Y, GRID_NX, GRID_NY, GRID_RESOLUTION)

rng = np.random.default_rng(0)
m = GhostGkModel.load(REPO/"silly_kicks/tracking/_ghost_gk_weights/default")
TL = m._training_leaves            # (36000, T)
GX = m._training_gk_x; GY = m._training_gk_y
NTR, T = TL.shape
print(f"[db] n_train={NTR} n_trees={T}", flush=True)

Q = pd.read_parquet(D/"queries_all.parquet")
Qm = pd.read_parquet(D/"queries_all_meta.parquet")
ok = ~Q[GHOST_GK_FEATURE_NAMES].isna().any(axis=1).values   # NaN rows would route by missing_go_to_left; keep it clean
Q = Q[ok].reset_index(drop=True); Qm = Qm[ok].reset_index(drop=True)
# Stratified subsample: production ground truth costs ~4 s/query (k = 36000 nonzero
# for EVERY query -- the KDE runs the full database), so take an evenly-spaced sample
# WITHIN each source to keep all three providers and span the whole time range.
TARGET = 480
parts = []
for src, g in Qm.groupby("__src", sort=True):
    take = max(1, int(round(TARGET * len(g) / len(Qm))))
    parts.append(np.asarray(g.index)[np.linspace(0, len(g)-1, min(take, len(g))).astype(int)])
keep = np.sort(np.concatenate(parts))
Q = Q.iloc[keep].reset_index(drop=True); Qm = Qm.iloc[keep].reset_index(drop=True)
NQ = len(Q); print(f"[q] n_queries={NQ} (stratified from {len(ok)})", flush=True)
print(Qm.groupby("__src").size().to_string(), flush=True)
XQ = Q[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)

# ---------------------------------------------------------------- 1. OCCUPANCY
maxleaf = int(TL.max())+1
occ = np.zeros((T, maxleaf), dtype=np.int64)
for t in range(T):
    np.add.at(occ[t], TL[:,t], 1)
occupied = occ > 0
cnts = occ[occupied]
print(f"[occ] occupied cells={occupied.sum()} | leaves/tree min={occupied.sum(1).min()} max={occupied.sum(1).max()}")
print(f"[occ] cell counts: min={cnts.min()} p1={np.percentile(cnts,1):.0f} median={np.median(cnts):.0f} "
      f"mean={cnts.mean():.2f} p99={np.percentile(cnts,99):.0f} max={cnts.max()}")
for thr in (2,5,10,20):
    print(f"[occ]   cells with n<{thr}: {(cnts<thr).sum()} ({100*(cnts<thr).mean():.3f}%)")

# ------------------------------------------------------- 2. PER-LEAF AGGREGATES
AGG = np.zeros((T, maxleaf, 6), dtype=np.float64)   # n, Sx, Sy, Sxx, Sxy, Syy
for t in range(T):
    idx = TL[:,t]
    np.add.at(AGG[t,:,0], idx, 1.0)
    np.add.at(AGG[t,:,1], idx, GX)
    np.add.at(AGG[t,:,2], idx, GY)
    np.add.at(AGG[t,:,3], idx, GX*GX)
    np.add.at(AGG[t,:,4], idx, GX*GY)
    np.add.at(AGG[t,:,5], idx, GY*GY)
agg_cells = int(occupied.sum())
agg_bytes = agg_cells*6*8
print(f"[agg] occupied cells={agg_cells} -> AGG payload {agg_bytes} bytes ({agg_bytes/1e6:.3f} MB)", flush=True)

# ----------------------------------------------------- 3. REAL-PATH LEAF ROUTING
t0=time.time(); QL = _vectorized_leaf_indices(m._tree_nodes, XQ); print(f"[route] query leaves {QL.shape} in {time.time()-t0:.1f}s", flush=True)

# aggregate-only reconstruction: sum the query's matched leaf moments over trees
sel = AGG[np.arange(T)[None,:,None], QL[:,:,None], np.arange(6)[None,None,:]]  # (NQ,T,6)
M = sel.sum(axis=1)/T          # (NQ,6) == [sum w, sum w x, sum w y, sum w xx, sum w xy, sum w yy]
Wsum = M[:,0]
mu = np.stack([M[:,1]/Wsum, M[:,2]/Wsum], axis=1)
Sxx_b = M[:,3]/Wsum - mu[:,0]**2
Sxy_b = M[:,4]/Wsum - mu[:,0]*mu[:,1]
Syy_b = M[:,5]/Wsum - mu[:,1]**2
S_b = np.stack([np.stack([Sxx_b,Sxy_b],1), np.stack([Sxy_b,Syy_b],1)], 1)   # (NQ,2,2) BIASED weighted cov
print("[agg] reconstructed biased-cov det: min %.4g median %.4g"%(np.linalg.det(S_b).min(), np.median(np.linalg.det(S_b))), flush=True)

# --------------------------------- 4. GROUND TRUTH + exactness check + true neff
CH = 48
spread = np.empty(NQ); mode_xy = np.empty((NQ,2)); mean_xy = np.empty((NQ,2))
n_modes = np.empty(NQ, dtype=int); neff_true = np.empty(NQ); ksize = np.empty(NQ, dtype=int)
mu_true = np.empty((NQ,2)); S_true_b = np.empty((NQ,2,2)); edge_mass = np.empty(NQ)
t0=time.time()
for a in range(0, NQ, CH):
    b = min(a+CH, NQ)
    ds = m.predict_density(Q.iloc[a:b][GHOST_GK_FEATURE_NAMES])
    Wt = _leaf_match_weights(TL, QL[a:b])
    for i,d in enumerate(ds):
        j = a+i
        spread[j]=d.spread; mode_xy[j]=(d.mode_x,d.mode_y); mean_xy[j]=(d.mean_x,d.mean_y)
        P = d.probabilities
        # local maxima (8-neighbourhood) above 20% of the peak
        pk = P.max(); Pp = np.pad(P,1,mode="constant",constant_values=-1)
        loc = np.ones_like(P, dtype=bool)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                loc &= P >= Pp[1+dx:1+dx+GRID_NX, 1+dy:1+dy+GRID_NY]
        n_modes[j] = int((loc & (P >= 0.2*pk)).sum())
        edge_mass[j] = P[0,:].sum()+P[-1,:].sum()+P[:,0].sum()+P[:,-1].sum()
        w = Wt[i]; nz = w>0; ww = w[nz]; ww = ww/ww.sum()
        ksize[j] = int(nz.sum()); neff_true[j] = 1.0/np.sum(ww**2)
        gx, gy = GX[nz], GY[nz]
        mx, my = float(ww@gx), float(ww@gy)
        mu_true[j]=(mx,my)
        S_true_b[j] = [[float(ww@((gx-mx)**2)), float(ww@((gx-mx)*(gy-my)))],
                       [float(ww@((gx-mx)*(gy-my))), float(ww@((gy-my)**2))]]
    del Wt
    print(f"[gt] {b}/{NQ}  {time.time()-t0:.0f}s", flush=True)

print("\n[exact] AGG-vs-true weighted moments (must be ~machine zero):")
print("  max |mu_x diff| = %.3e   max |mu_y diff| = %.3e"%(np.abs(mu[:,0]-mu_true[:,0]).max(), np.abs(mu[:,1]-mu_true[:,1]).max()))
print("  max |cov diff|  = %.3e"%np.abs(S_b-S_true_b).max())
print("[gt] k (nonzero training rows): min=%d median=%d max=%d"%(ksize.min(), np.median(ksize), ksize.max()))
print("[gt] neff_true: min=%.1f median=%.1f max=%.1f"%(neff_true.min(), np.median(neff_true), neff_true.max()))
print("[gt] spread: min=%.3f median=%.3f max=%.3f  CV=%.4f (grid area=%.0f m2)"%(
    spread.min(), np.median(spread), spread.max(), spread.std()/spread.mean(), GRID_NX*GRID_NY*GRID_RESOLUTION**2))
print("[gt] grid-edge probability mass: median=%.3e p99=%.3e max=%.3e"%(
    np.median(edge_mass), np.percentile(edge_mass,99), edge_mass.max()))

np.savez_compressed(D/"gt.npz", spread=spread, mode_xy=mode_xy, mean_xy=mean_xy, n_modes=n_modes,
    neff_true=neff_true, ksize=ksize, mu=mu, S_b=S_b, mu_true=mu_true, S_true_b=S_true_b,
    edge_mass=edge_mass, src=Qm["__src"].values.astype(str), QL=QL, occ_counts=cnts)
print("SAVED gt.npz", flush=True)
