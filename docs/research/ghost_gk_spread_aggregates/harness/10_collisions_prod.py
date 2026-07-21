"""Recover the exact indices of the queries whose full 500-leaf vector collides
with a training row, so the headline can be re-scored on the UNSEEN subset only.
Reads the artifact npz directly (read-only); does not touch the model dir.
"""
from __future__ import annotations
import os, pathlib
import numpy as np

D = pathlib.Path(__file__).resolve().parent
OUT = D / "out"
P = np.load(OUT / "prep.npz", allow_pickle=False)
QL = P["QL"]
SRC = P["src"].astype(str)
T = QL.shape[1]

with np.load(pathlib.Path(os.environ["SK_MODEL"]) / "rfcde_weights.npz") as Z:
    TL = Z["training_leaves"]
print(f"[db] TL {TL.shape}  QL {QL.shape}")

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
tr_set = set(h_tr.tolist())
coll = []
for j in range(len(QL)):
    if int(h_q[j]) in tr_set:
        rows = np.flatnonzero(h_tr == h_q[j])
        if (TL[rows] == QL[j]).all(axis=1).any():
            coll.append(j)
coll = np.array(coll, dtype=np.int64)
print(f"[coll] {len(coll)}/{len(QL)} collisions at indices {coll.tolist()}")
for s in sorted(set(SRC)):
    print(f"[coll]   {s:20s} {(SRC[coll]==s).sum() if len(coll) else 0}")
np.save(OUT / "collisions.npy", coll)
print("[coll] SAVED out/collisions.npy")
