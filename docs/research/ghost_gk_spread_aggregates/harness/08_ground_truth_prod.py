"""PRODUCTION-SCALE (Stage-B, 1,039,502 x 500) replication of the per-leaf
moment-aggregate study, on the SAME 480 queries as the 36k measurement.

Ground truth : GhostGkModel.predict_density(kde_backend="vectorized"), one query
               per call -> query_block collapses to 1 (memory-safe for 16 workers)
               while remaining the EXACT production code path.
Queries      : queries_all.parquet, the identical artifact the 36k run used, with
               the identical deterministic stratified selection -> paired design.
Checkpoint   : append-only JSONL, flush+fsync per query, resume-by-skip.
"""
from __future__ import annotations
import json, os, pathlib, sys, time, warnings
import multiprocessing as mp
import numpy as np
import pandas as pd

REPO = pathlib.Path(os.environ.get("SK_REPO", "/home/karsten/Development/silly-kicks"))
sys.path.insert(0, str(REPO))
D = pathlib.Path(__file__).resolve().parent
MODEL_DIR = pathlib.Path(os.environ["SK_MODEL"])
OUT = D / "out"
OUT.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

from silly_kicks.tracking._ghost_gk import (  # noqa: E402
    GhostGkModel, GHOST_GK_FEATURE_NAMES, _vectorized_leaf_indices,
    _leaf_match_weights, GRID_NX, GRID_NY, GRID_RESOLUTION)

NWORK = int(os.environ.get("SK_WORKERS", "16"))

# ------------------------------------------------------------------ load model
t0 = time.time()
M = GhostGkModel.load(MODEL_DIR)
TL = M._training_leaves
GX = M._training_gk_x
GY = M._training_gk_y
NTR, T = TL.shape
print(f"[db] n_train={NTR} n_trees={T}  load {time.time()-t0:.1f}s", flush=True)

# ---------------------------------------------- queries: VERBATIM 36k selection
Q = pd.read_parquet(D / "queries_all.parquet")
Qm = pd.read_parquet(D / "queries_all_meta.parquet")
ok = ~Q[GHOST_GK_FEATURE_NAMES].isna().any(axis=1).values
Q = Q[ok].reset_index(drop=True); Qm = Qm[ok].reset_index(drop=True)
TARGET = 480
parts = []
for src_, g in Qm.groupby("__src", sort=True):
    take = max(1, int(round(TARGET * len(g) / len(Qm))))
    parts.append(np.asarray(g.index)[np.linspace(0, len(g) - 1, min(take, len(g))).astype(int)])
keep = np.sort(np.concatenate(parts))
Q = Q.iloc[keep].reset_index(drop=True); Qm = Qm.iloc[keep].reset_index(drop=True)
NQ = len(Q)
SRC = Qm["__src"].values.astype(str)
print(f"[q] n_queries={NQ}", flush=True)
print(Qm.groupby("__src").size().to_string(), flush=True)
QF = Q[GHOST_GK_FEATURE_NAMES]
XQ = QF.values.astype(np.float64)

# ------------------------------------------------------- real-path leaf routing
t0 = time.time()
QL = _vectorized_leaf_indices(M._tree_nodes, XQ)
print(f"[route] query leaves {QL.shape} in {time.time()-t0:.1f}s", flush=True)

# --------------------------------------------------------------- 1. OCCUPANCY
maxleaf = int(TL.max()) + 1
t0 = time.time()
occ = np.empty((T, maxleaf), dtype=np.int64)
for t in range(T):
    occ[t] = np.bincount(TL[:, t], minlength=maxleaf)
occupied = occ > 0
cnts = occ[occupied]
print(f"[occ] occupied cells={occupied.sum()} | leaves/tree min={occupied.sum(1).min()} "
      f"max={occupied.sum(1).max()}  ({time.time()-t0:.1f}s)", flush=True)
print(f"[occ] cell counts: min={cnts.min()} p1={np.percentile(cnts,1):.0f} "
      f"median={np.median(cnts):.0f} mean={cnts.mean():.2f} p99={np.percentile(cnts,99):.0f} "
      f"max={cnts.max()}", flush=True)
for thr in (2, 5, 10, 20):
    print(f"[occ]   cells with n<{thr}: {(cnts<thr).sum()} ({100*(cnts<thr).mean():.3f}%)", flush=True)

# --------------------------------------------------- 2. PER-LEAF AGGREGATES
# bincount(weights=) is the same accumulation as the 36k run's np.add.at, but
# O(500*6) calls instead of O(500*6) python-level scatter loops over 1.04M rows.
t0 = time.time()
AGG = np.zeros((T, maxleaf, 6), dtype=np.float64)
GXX_, GYY_, GXY_ = GX * GX, GY * GY, GX * GY
for t in range(T):
    idx = TL[:, t]
    AGG[t, :, 0] = np.bincount(idx, minlength=maxleaf)
    AGG[t, :, 1] = np.bincount(idx, weights=GX, minlength=maxleaf)
    AGG[t, :, 2] = np.bincount(idx, weights=GY, minlength=maxleaf)
    AGG[t, :, 3] = np.bincount(idx, weights=GXX_, minlength=maxleaf)
    AGG[t, :, 4] = np.bincount(idx, weights=GXY_, minlength=maxleaf)
    AGG[t, :, 5] = np.bincount(idx, weights=GYY_, minlength=maxleaf)
del GXX_, GYY_, GXY_
agg_cells = int(occupied.sum())
agg_bytes = agg_cells * 6 * 8
print(f"[agg] occupied cells={agg_cells} -> AGG payload {agg_bytes} bytes "
      f"({agg_bytes/1e6:.3f} MB)  ({time.time()-t0:.1f}s)", flush=True)

sel = AGG[np.arange(T)[None, :, None], QL[:, :, None], np.arange(6)[None, None, :]]
Mo = sel.sum(axis=1) / T
Wsum = Mo[:, 0]
mu = np.stack([Mo[:, 1] / Wsum, Mo[:, 2] / Wsum], axis=1)
Sxx_b = Mo[:, 3] / Wsum - mu[:, 0] ** 2
Sxy_b = Mo[:, 4] / Wsum - mu[:, 0] * mu[:, 1]
Syy_b = Mo[:, 5] / Wsum - mu[:, 1] ** 2
S_b = np.stack([np.stack([Sxx_b, Sxy_b], 1), np.stack([Sxy_b, Syy_b], 1)], 1)
print("[agg] reconstructed biased-cov det: min %.4g median %.4g"
      % (np.linalg.det(S_b).min(), np.median(np.linalg.det(S_b))), flush=True)
del AGG, sel, occ

# ------------------------------------------- 3. UNSEEN CHECK (leaf-vector hash)
t0 = time.time()
rs = np.random.default_rng(12345)
rv = rs.integers(1, 2**63, size=T, dtype=np.int64).astype(np.uint64) | np.uint64(1)


def _hash(mat):
    h = np.zeros(mat.shape[0], dtype=np.uint64)
    for a in range(0, mat.shape[0], 100_000):
        b = min(a + 100_000, mat.shape[0])
        h[a:b] = (mat[a:b].astype(np.uint64) * rv).sum(axis=1)
    return h


h_tr = _hash(TL)
h_q = _hash(QL)
uniq_tr = len(np.unique(h_tr))
tr_set = set(h_tr.tolist())
cand = np.array([int(x) in tr_set for x in h_q])
n_coll = 0
coll_idx = []
if cand.any():
    for j in np.flatnonzero(cand):
        rows = np.flatnonzero(h_tr == h_q[j])
        if (TL[rows] == QL[j]).all(axis=1).any():
            n_coll += 1
            coll_idx.append(int(j))
print(f"[unseen] distinct training leaf vectors: {uniq_tr}/{NTR}", flush=True)
print(f"[unseen] queries whose FULL {T}-leaf vector exactly matches a training row: "
      f"{n_coll}/{NQ}  ({time.time()-t0:.1f}s)", flush=True)
for s in sorted(set(SRC)):
    k = SRC == s
    nc = sum(1 for j in coll_idx if k[j])
    print(f"[unseen]   {s:20s} n={k.sum():4d}  exact collisions={nc}", flush=True)

np.savez_compressed(OUT / "prep.npz", mu=mu, S_b=S_b, QL=QL, src=SRC,
                    occ_counts=cnts, n_train=NTR, n_trees=T,
                    n_collisions=n_coll, uniq_train=uniq_tr)
print("[prep] SAVED prep.npz", flush=True)
del h_tr, tr_set

# ------------------------------------------------------ 4. GROUND TRUTH (pool)
JSONL = OUT / "gt.jsonl"
done = set()
if JSONL.exists():
    with JSONL.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["i"])
            except Exception:
                pass
print(f"[gt] resume: {len(done)} already done", flush=True)
todo = [i for i in range(NQ) if i not in done]


def work(i):
    d = M.predict_density(QF.iloc[[i]])[0]
    P = d.probabilities
    pk = P.max()
    Pp = np.pad(P, 1, mode="constant", constant_values=-1)
    loc = np.ones_like(P, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            loc &= P >= Pp[1 + dx:1 + dx + GRID_NX, 1 + dy:1 + dy + GRID_NY]
    n_modes = int((loc & (P >= 0.2 * pk)).sum())
    edge = float(P[0, :].sum() + P[-1, :].sum() + P[:, 0].sum() + P[:, -1].sum())
    w = _leaf_match_weights(TL, QL[i:i + 1], query_block=1)[0]
    nz = w > 0
    ww = w[nz]
    max_w = float(w.max())
    ww = ww / ww.sum()
    ksize = int(nz.sum())
    neff = float(1.0 / np.sum(ww ** 2))
    gx, gy = GX[nz], GY[nz]
    mx, my = float(ww @ gx), float(ww @ gy)
    st = [[float(ww @ ((gx - mx) ** 2)), float(ww @ ((gx - mx) * (gy - my)))],
          [float(ww @ ((gx - mx) * (gy - my))), float(ww @ ((gy - my) ** 2))]]
    del w, ww, gx, gy
    return {"i": int(i), "spread": float(d.spread), "mode": [d.mode_x, d.mode_y],
            "mean": [d.mean_x, d.mean_y], "n_modes": n_modes, "edge": edge,
            "ksize": ksize, "neff": neff, "mu_true": [mx, my], "S_true_b": st,
            "max_w": max_w, "max_trees": max_w * T}


if __name__ == "__main__":
    t0 = time.time()
    n = 0
    with JSONL.open("a") as f, mp.get_context("fork").Pool(NWORK) as pool:
        for r in pool.imap_unordered(work, todo, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n += 1
            if n % 8 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[gt] {n}/{len(todo)}  {el:.0f}s  {el/n:.2f}s/query  "
                      f"eta {(len(todo)-n)*el/n/60:.1f}min", flush=True)
    print(f"[gt] DONE {n} queries in {time.time()-t0:.0f}s", flush=True)
