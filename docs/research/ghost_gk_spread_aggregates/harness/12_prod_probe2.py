import time, pathlib, resource, numpy as np, pandas as pd
from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, _leaf_match_weights, _kde_density_vectorized,
    _vectorized_leaf_indices, GHOST_GK_FEATURE_NAMES, _GRID_X, _GRID_Y,
)
def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6

art = pathlib.Path.home()/"Development/sk_stageB_448/ghost_full/ghost_gk_v1"
m = GhostGkModel.load(art)
tl = m._training_leaves
feat_all = pd.read_parquet(art/"_feature_cache/features.parquet")
feat = feat_all.iloc[:8][GHOST_GK_FEATURE_NAMES]
X = feat.values.astype(np.float64)
ql = _vectorized_leaf_indices(m._tree_nodes, X)
print("peak GB after load+cache", round(peak_gb(),2), flush=True)

# --- collision check: are cache rows SEEN by the model? ---
hits = 0
for i in range(8):
    eq = (tl == ql[i][None, :]).all(axis=1)
    hits += int(eq.any())
print(f"COLLISION: {hits}/8 query leaf-vectors exactly match a training row", flush=True)
print("peak GB after collision", round(peak_gb(),2), flush=True)

gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
gp = np.vstack([gxx.ravel(), gyy.ravel()])
W = _leaf_match_weights(tl, ql[:4], query_block=1)
for tb in (1024, 65536):
    t0=time.time()
    p = _kde_density_vectorized(m._training_gk_x, m._training_gk_y, W[0], gp, train_block=tb)
    print(f"KDE k=1039502 train_block={tb}: {time.time()-t0:.2f}s  peak GB {peak_gb():.2f}", flush=True)
del W

# --- true end-to-end predict_density, batches of 4 ---
for rep in range(2):
    t0=time.time()
    d = m.predict_density(feat.iloc[:4], kde_backend="vectorized")
    dt=time.time()-t0
    print(f"predict_density batch=4 rep{rep}: {dt:.2f}s total = {dt/4:.2f}s/query  peak GB {peak_gb():.2f}", flush=True)
print("spread[0]", d[0].spread, "mean", d[0].mean_x, d[0].mean_y, flush=True)
