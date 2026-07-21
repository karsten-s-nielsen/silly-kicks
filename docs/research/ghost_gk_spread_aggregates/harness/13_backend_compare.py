from __future__ import annotations
import pathlib, sys, time, warnings
import numpy as np, pandas as pd
REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux"); sys.path.insert(0,str(REPO))
D = pathlib.Path(__file__).resolve().parent; warnings.filterwarnings("ignore")
from silly_kicks.tracking._ghost_gk import GhostGkModel, GHOST_GK_FEATURE_NAMES
m = GhostGkModel.load(REPO/"silly_kicks/tracking/_ghost_gk_weights/default")
Q = pd.read_parquet(D/"queries_all.parquet")
# diverse probe: spread across all three sources
idx = np.concatenate([np.linspace(0,236,8).astype(int),
                      np.linspace(237,1340,12).astype(int),
                      np.linspace(1341,1592,8).astype(int)])
sub = Q.iloc[idx][GHOST_GK_FEATURE_NAMES]
for be in ("vectorized","fft-cic","fft"):
    t=time.time(); ds = m.predict_density(sub, kde_backend=be)
    s = np.array([d.spread for d in ds])
    print(f"{be:11s} {time.time()-t:7.1f}s  spread[:6]={np.round(s[:6],4)}")
    if be=="vectorized": ref=s
    else:
        rel = np.abs(s-ref)/ref
        print(f"             vs vectorized: max rel diff {rel.max():.3e}  median {np.median(rel):.3e}")
