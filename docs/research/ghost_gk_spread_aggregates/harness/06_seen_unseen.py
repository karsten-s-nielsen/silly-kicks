"""Requirement 7: were the query frames in the training set?

Two executable tests, no assumptions:
  A. EXACT 243-leaf-vector collision between a query and any training row.
     (35,892/36,000 training leaf vectors are unique -> a collision is near-proof of
      membership; absence of a collision is proof of non-membership under the model's
      own partition, since an identical feature row routes identically by construction.)
  B. Graded: the MAXIMUM number of trees (out of 243) any single training row agrees
     with, per query. A training member scores 243/243.
Also reports the leaf-match weight concentration, which controls how "conditional"
the conditional density actually is.
"""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np, pandas as pd

REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux")
sys.path.insert(0, str(REPO))
D = pathlib.Path(__file__).resolve().parent
AGG = D.parent / "agg"
warnings.filterwarnings("ignore")
from silly_kicks.tracking._ghost_gk import GhostGkModel, _leaf_match_weights

m = GhostGkModel.load(D / "weights_copy")
TL = m._training_leaves
NTR, T = TL.shape
Z = np.load(AGG / "gt.npz", allow_pickle=False)
QL = Z["QL"]; src = Z["src"].astype(str); NQ = len(QL)
print(f"n_train={NTR} n_trees={T} n_queries={NQ}")

# ---- A. exact leaf-vector collision
train_keys = {TL[i].tobytes() for i in range(NTR)}
print(f"[A] distinct training leaf vectors: {len(train_keys)}/{NTR}")
hit = np.array([QL[q].tobytes() in train_keys for q in range(NQ)])
print(f"[A] queries whose FULL 243-leaf vector exactly matches a training row: {int(hit.sum())}/{NQ}")
for s in sorted(set(src)):
    k = src == s
    print(f"      {s:18s} n={k.sum():4d}  exact collisions={int(hit[k].sum())}")

# ---- B. graded max agreement (= T * max leaf-match weight)
W = _leaf_match_weights(TL, QL)                 # (NQ, NTR) weights in [0,1], = matches/T
mx = W.max(axis=1) * T
print(f"\n[B] max trees agreeing with ANY single training row (out of {T}):")
print(f"      min={mx.min():.0f}  median={np.median(mx):.0f}  p99={np.percentile(mx,99):.0f}  max={mx.max():.0f}")
for s in sorted(set(src)):
    k = src == s
    print(f"      {s:18s} median={np.median(mx[k]):.0f}  max={mx[k].max():.0f}")

# ---- weight concentration: how conditional is the "conditional" density?
Wn = W / W.sum(axis=1, keepdims=True)
neff = 1.0 / (Wn**2).sum(axis=1)
top1 = Wn.max(axis=1)
srt = np.sort(Wn, axis=1)[:, ::-1]
top100 = srt[:, :100].sum(axis=1)
print(f"\n[C] weight concentration over the {NTR}-row database:")
print(f"      neff:            min={neff.min():.0f} median={np.median(neff):.0f} max={neff.max():.0f}  "
      f"(uniform would be {NTR})")
print(f"      neff / n_train:  median={np.median(neff)/NTR:.4f}")
print(f"      largest single weight: median={np.median(top1):.3e}  (uniform = {1/NTR:.3e})")
print(f"      mass in top-100 rows:  median={np.median(top100):.4f}  (uniform = {100/NTR:.4f})")
nz = (W > 0).sum(axis=1)
print(f"      rows with NONZERO weight: min={nz.min()} median={np.median(nz):.0f} max={nz.max()}")
