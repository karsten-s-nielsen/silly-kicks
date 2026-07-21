import time, os, pathlib, resource, numpy as np, pandas as pd
import multiprocessing as mp
from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, _leaf_match_weights, _kde_density_vectorized,
    _vectorized_leaf_indices, GHOST_GK_FEATURE_NAMES, _GRID_X, _GRID_Y, GRID_RESOLUTION,
)
ART = pathlib.Path.home()/"Development/sk_stageB_448/ghost_full/ghost_gk_v1"
G = {}

def init():
    m = GhostGkModel.load(ART)
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    G["tl"], G["gx"], G["gy"] = m._training_leaves, m._training_gk_x, m._training_gk_y
    G["gp"] = np.vstack([gxx.ravel(), gyy.ravel()]); G["gxx"], G["gyy"] = gxx, gyy

def work(qleaf):
    w = _leaf_match_weights(G["tl"], qleaf[None, :], query_block=1)[0]
    p = _kde_density_vectorized(G["gx"], G["gy"], w, G["gp"], train_block=1024)
    p = p/p.sum(); nz = p[p>0]
    spread = float(np.exp(float(-np.sum(nz*np.log(nz)))) * GRID_RESOLUTION**2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    return spread, float(np.sum(p*G["gxx"])), float(np.sum(p*G["gyy"])), rss

if __name__ == "__main__":
    mp.set_start_method("fork")
    init()
    feat = pd.read_parquet(ART/"_feature_cache/features.parquet")[GHOST_GK_FEATURE_NAMES]
    rng = np.random.default_rng(11)
    idx = rng.choice(len(feat), 16, replace=False)
    ql = _vectorized_leaf_indices(GhostGkModel.load(ART)._tree_nodes, feat.iloc[idx].values.astype(np.float64))
    parent_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    print(f"parent peak GB after load {parent_gb:.2f}", flush=True)
    for nw in (4, 8, 16):
        t0 = time.time()
        with mp.Pool(nw) as pool:
            res = pool.map(work, list(ql), chunksize=1)
        dt = time.time()-t0
        wmax = max(r[3] for r in res)
        print(f"[P] workers={nw}: 16 queries in {dt:.1f}s -> {dt/16:.2f}s/query wall, "
              f"throughput {16/dt*3600:.0f} q/hour, worker peak RSS {wmax:.2f} GB", flush=True)
    print("[P] sample spreads", [round(r[0],3) for r in res[:5]], flush=True)
