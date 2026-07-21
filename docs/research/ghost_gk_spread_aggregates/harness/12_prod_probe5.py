import time, pathlib, resource, numpy as np, pandas as pd
from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, _leaf_match_weights, _vectorized_leaf_indices, GHOST_GK_FEATURE_NAMES)
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
art = pathlib.Path.home()/"Development/sk_stageB_448/ghost_full/ghost_gk_v1"
m = GhostGkModel.load(art); tl = m._training_leaves
n_train, n_trees = tl.shape
print(f"eq block bytes at stock query_block=64: {64*n_train*n_trees/1e9:.1f} GB", flush=True)
feat = pd.read_parquet(art/"_feature_cache/features.parquet").iloc[:64][GHOST_GK_FEATURE_NAMES]
ql = _vectorized_leaf_indices(m._tree_nodes, feat.values.astype(np.float64))
print("peak GB before", round(peak(),2), flush=True)
t0=time.time()
W = _leaf_match_weights(tl, ql)          # STOCK signature, default query_block=64
print(f"STOCK _leaf_match_weights 64 queries: {time.time()-t0:.1f}s  peak GB {peak():.2f}  W {W.nbytes/1e9:.2f}GB", flush=True)
