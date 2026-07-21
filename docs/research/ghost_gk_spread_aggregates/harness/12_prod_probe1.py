import time, pathlib, numpy as np, pandas as pd
t0=time.time()
from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, _leaf_match_weights, _kde_density_vectorized,
    _vectorized_leaf_indices, GHOST_GK_FEATURE_NAMES, _GRID_X, _GRID_Y,
)
print("import s", round(time.time()-t0,2), flush=True)

art = pathlib.Path.home()/"Development/sk_stageB_448/ghost_full/ghost_gk_v1"
t0=time.time(); m = GhostGkModel.load(art); print("load s", round(time.time()-t0,2), flush=True)
tl = m._training_leaves
print("training_leaves", tl.shape, tl.dtype, "bytes GB", round(tl.nbytes/1e9,2), flush=True)

feat = pd.read_parquet(art/"_feature_cache/features.parquet").iloc[:8][GHOST_GK_FEATURE_NAMES]
X = feat.values.astype(np.float64)
t0=time.time(); ql = _vectorized_leaf_indices(m._tree_nodes, X); print("leaf traversal 8q s", round(time.time()-t0,3), flush=True)

for qb in (1,2,4):
    t0=time.time()
    W = _leaf_match_weights(tl, ql[:qb], query_block=qb)
    dt=time.time()-t0
    print(f"leaf_match_weights q={qb}: {dt:.2f}s  ({dt/qb:.2f}s/query)  W {W.shape} {W.nbytes/1e9:.2f}GB", flush=True)
    nz = (W[0] > 0).sum()
    print("   nonzero rows q0:", nz, "frac", round(nz/tl.shape[0],4), flush=True)
    del W
