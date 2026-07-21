import time, os, pathlib, resource, numpy as np, pandas as pd
from silly_kicks.tracking._ghost_gk import (
    GhostGkModel, _leaf_match_weights, _kde_density_vectorized,
    _vectorized_leaf_indices, GHOST_GK_FEATURE_NAMES, _GRID_X, _GRID_Y,
    GRID_NX, GRID_NY, GRID_RESOLUTION,
)
def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6

# ---- A. DGX baseline on the 36k bundled artifact (hardware control) ----
d36 = pathlib.Path.home()/"Development/silly-kicks/silly_kicks/tracking/_ghost_gk_weights/default"
if d36.exists():
    m36 = GhostGkModel.load(d36)
    f36 = pd.read_parquet(pathlib.Path.home()/"Development/sk_stageB_448/ghost_full/ghost_gk_v1/_feature_cache/features.parquet").iloc[:4][GHOST_GK_FEATURE_NAMES]
    t0=time.time(); m36.predict_density(f36, kde_backend="vectorized"); dt=time.time()-t0
    print(f"[A] DGX 36k(243t) predict_density batch=4: {dt:.2f}s = {dt/4:.2f}s/query", flush=True)
    del m36
else:
    print("[A] 36k artifact not present on DGX at", d36, flush=True)

# ---- B. Stage-B: leave-self-out control for SEEN queries ----
art = pathlib.Path.home()/"Development/sk_stageB_448/ghost_full/ghost_gk_v1"
m = GhostGkModel.load(art)
tl, gx, gy = m._training_leaves, m._training_gk_x, m._training_gk_y
feat = pd.read_parquet(art/"_feature_cache/features.parquet")[GHOST_GK_FEATURE_NAMES]
gxx,gyy = np.meshgrid(_GRID_X,_GRID_Y,indexing="ij"); gp=np.vstack([gxx.ravel(),gyy.ravel()])

def spread_from(probs):
    p = probs/probs.sum(); nz = p[p>0]
    return float(np.exp(float(-np.sum(nz*np.log(nz)))) * GRID_RESOLUTION**2)

rng = np.random.default_rng(7)
idx = rng.choice(len(feat), 3, replace=False)
ql = _vectorized_leaf_indices(m._tree_nodes, feat.iloc[idx].values.astype(np.float64))
for j,i in enumerate(idx):
    W = _leaf_match_weights(tl, ql[j:j+1], query_block=1)[0]
    s_seen = spread_from(_kde_density_vectorized(gx, gy, W, gp, train_block=1024))
    # exclude every exact leaf-vector twin of this query
    twin = (tl == ql[j][None,:]).all(axis=1)
    keep = ~twin
    s_lso = spread_from(_kde_density_vectorized(gx[keep], gy[keep], W[keep], gp, train_block=1024))
    print(f"[B] row {i}: n_twins={int(twin.sum())} spread_seen={s_seen:.6f} spread_leave_self_out={s_lso:.6f} "
          f"rel_delta={abs(s_seen-s_lso)/s_seen:.3e}", flush=True)
    del W
print("[B] peak GB", round(peak_gb(),2), flush=True)
