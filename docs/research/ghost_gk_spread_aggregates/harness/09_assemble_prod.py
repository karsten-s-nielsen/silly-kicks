"""Assemble the checkpointed JSONL + prep.npz into a gt.npz in the EXACT schema
the 36k scoring scripts (gapD/analyse*.py) already consume, so the production
numbers are produced by byte-identical scoring code.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

D = pathlib.Path(__file__).resolve().parent
OUT = D / "out"
# NOTE: under pandas 3.x, `Series.values.astype(str)` yields an OBJECT array, so prep.npz
# stores `src` as dtype=object and np.load(..., allow_pickle=False) raises. Read it with
# allow_pickle=True here, and re-emit `src` below as fixed-width unicode so the scoring
# scripts can keep loading gt.npz with allow_pickle=False.
P = np.load(OUT / "prep.npz", allow_pickle=True)

rows = {}
with (OUT / "gt.jsonl").open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue          # tolerate a torn final line from a kill
        rows[r["i"]] = r
idx = np.array(sorted(rows))
NQ_full = len(P["src"])
print(f"[assemble] {len(idx)}/{NQ_full} queries present")
if len(idx) < NQ_full:
    print(f"[assemble] PARTIAL: missing {NQ_full-len(idx)}")

g = lambda k: np.array([rows[i][k] for i in idx])
np.savez_compressed(
    OUT / "gt.npz",
    spread=g("spread"), mode_xy=g("mode"), mean_xy=g("mean"),
    n_modes=g("n_modes").astype(int), edge_mass=g("edge"),
    ksize=g("ksize").astype(np.int64), neff_true=g("neff"),
    mu_true=g("mu_true"), S_true_b=g("S_true_b"),
    max_w=g("max_w"), max_trees=g("max_trees"),
    mu=P["mu"][idx], S_b=P["S_b"][idx], src=P["src"][idx].astype("U32"),
    occ_counts=P["occ_counts"], n_train=P["n_train"], n_trees=P["n_trees"],
    n_collisions=P["n_collisions"], uniq_train=P["uniq_train"],
    query_index=idx,
)
print("[assemble] SAVED out/gt.npz")

NTR = int(P["n_train"]); T = int(P["n_trees"])
ks = g("ksize"); ne = g("neff"); mt = g("max_trees")
print(f"\n=== PRODUCTION-SCALE MECHANISM (n_train={NTR}, n_trees={T}) ===")
print(f"[k] nonzero-weight training rows: min={ks.min()} median={np.median(ks):.0f} max={ks.max()}")
print(f"[k] fraction of corpus at NONZERO weight: min={ks.min()/NTR:.4f} "
      f"median={np.median(ks)/NTR:.4f} max={ks.max()/NTR:.4f}")
print(f"[neff] min={ne.min():.1f} median={np.median(ne):.1f} max={ne.max():.1f}")
print(f"[neff] neff/n_train: min={ne.min()/NTR:.4f} median={np.median(ne)/NTR:.4f} "
      f"max={ne.max()/NTR:.4f}")
print(f"[w] largest single weight: median={np.median(g('max_w')):.6e} "
      f"(uniform = {1.0/NTR:.6e})")
print(f"[w] max trees agreeing with any ONE training row (of {T}): "
      f"min={mt.min():.0f} median={np.median(mt):.0f} max={mt.max():.0f}")
print(f"[unseen] exact {T}-leaf-vector collisions: {int(P['n_collisions'])}/{NQ_full}")
print(f"[unseen] distinct training leaf vectors: {int(P['uniq_train'])}/{NTR}")
c = P["occ_counts"]
print(f"[occ] occupied cells={len(c)} counts min={c.min()} median={np.median(c):.0f} max={c.max()}")
