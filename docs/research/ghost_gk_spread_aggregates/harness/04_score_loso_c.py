"""Two fairness gaps in analyse.py, closed.

(1) FREE-b was fitted by log-space OLS while b=0.5 was fitted by grid-searching the
    ACTUAL scoring metric (median relative error). Refit free-b on the same loss.
(2) The b=0.5 inflation c was fitted IN-SAMPLE on the same 480 queries it is scored on.
    Refit it OUT-OF-SAMPLE by leave-one-source-out and re-score.
"""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np, pandas as pd

REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux")
sys.path.insert(0, str(REPO))
D = pathlib.Path(__file__).resolve().parent
AGG = D.parent / "agg"
warnings.filterwarnings("ignore")
from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, GRID_NX, GRID_NY, GRID_RESOLUTION

Z = np.load(AGG / "gt.npz", allow_pickle=False)
spread = Z["spread"]; mu = Z["mu"]; S_b = Z["S_b"]; src = Z["src"].astype(str)
NQ = len(spread)
GXX, GYY = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
PTS = np.stack([GXX.ravel(), GYY.ravel()], 1)
CELL = GRID_RESOLUTION**2


def spread_of(P):
    P = P / P.sum(); nz = P[P > 0]
    return float(np.exp(-np.sum(nz * np.log(nz))) * CELL)


def gauss_grid_spread(mu_i, Sig):
    d = PTS - mu_i
    det = Sig[0, 0] * Sig[1, 1] - Sig[0, 1] ** 2
    if det <= 0:
        return np.nan
    inv = np.array([[Sig[1, 1], -Sig[0, 1]], [-Sig[0, 1], Sig[0, 0]]]) / det
    e = -0.5 * np.einsum("ij,jk,ik->i", d, inv, d)
    return spread_of(np.exp(e - e.max()).reshape(GRID_NX, GRID_NY))


BLK = 24; B = 4000
rng = np.random.default_rng(7)
src_blocks = {}
for s in sorted(set(src)):
    idx = np.flatnonzero(src == s)
    src_blocks[s] = [idx[i:i + BLK] for i in range(0, len(idx), BLK)]
BOOT_IDX = []
for _ in range(B):
    parts = []
    for s, blocks in src_blocks.items():
        pick = rng.integers(0, len(blocks), len(blocks))
        parts.append(np.concatenate([blocks[i] for i in pick]))
    BOOT_IDX.append(np.concatenate(parts))


def paired(pred, base, name):
    r = np.abs(pred - spread) / spread
    rb = np.abs(base - spread) / spread
    med = np.nanmedian(r)
    d = np.array([np.nanmedian(r[i]) - np.nanmedian(rb[i]) for i in BOOT_IDX])
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    v = "BEATS constant" if dhi < 0 else ("LOSES to constant" if dlo > 0 else "NOT DISTINGUISHABLE")
    print(f"  {name:44s} median {100*med:6.3f}%   vs constant {100*(med-np.nanmedian(rb)):+.3f} pp "
          f"[95% CI {100*dlo:+.3f},{100*dhi:+.3f}] -> {v}")


cs = np.linspace(spread.min(), spread.max(), 2001)
c_star = cs[np.argmin([np.median(np.abs(c - spread) / spread) for c in cs])]
CONST = np.full(NQ, c_star)
print(f"constant baseline c* = {c_star:.3f} m2  -> median rel err {100*np.median(np.abs(CONST-spread)/spread):.3f}%")

# ---------------- (1) FREE-b refit on the SAME loss as b=0.5 ----------------
print("\n--- (1) FREE-b refit by MINIMISING MEDIAN RELATIVE ERROR (same loss as b=0.5) ---")
det_b = np.linalg.det(S_b); ok = det_b > 0
best = None
for b in np.arange(0.0, 3.01, 0.01):
    u = det_b[ok] ** b / spread[ok]
    aa = np.linspace(0.5 / np.median(u), 2.0 / np.median(u), 400)
    e = np.array([np.median(np.abs(a * u - 1.0)) for a in aa])
    j = int(e.argmin())
    if best is None or e[j] < best[0]:
        best = (e[j], b, aa[j])
err_fb, b_fb, a_fb = best
p_fb2 = np.full(NQ, np.nan); p_fb2[ok] = a_fb * det_b[ok] ** b_fb
print(f"  loss-matched fit: a={a_fb:.6g}  b={b_fb:.3f}   (dimensionally coherent b=0.5)")
paired(p_fb2, CONST, f"FREE-b loss-matched (b={b_fb:.2f}, 2 free params)")

# ---------------- (2) b=0.5 inflation fitted OUT-OF-SAMPLE ----------------
print("\n--- (2) b=0.5 inflation c fitted LEAVE-ONE-SOURCE-OUT (honest out-of-sample) ---")
grid_c = np.arange(0.0, 0.61, 0.005)
# precompute predictions for every c once (expensive part)
P_all = np.empty((len(grid_c), NQ))
for gi, c in enumerate(grid_c):
    P_all[gi] = [gauss_grid_spread(mu[i], S_b[i] * (1 + c)) for i in range(NQ)]
p_loso = np.full(NQ, np.nan)
for s in sorted(set(src)):
    te = src == s; tr = ~te
    e = np.array([np.median(np.abs(P_all[gi][tr] - spread[tr]) / spread[tr]) for gi in range(len(grid_c))])
    gi = int(e.argmin())
    p_loso[te] = P_all[gi][te]
    print(f"    held out {s:18s}: c fitted on the other sources = {grid_c[gi]:.3f}  "
          f"-> held-out median rel err {100*np.median(np.abs(P_all[gi][te]-spread[te])/spread[te]):.3f}%")
paired(p_loso, CONST, "GAUSS-GRID b=0.5, c OUT-OF-SAMPLE (LOSO)")

# in-sample reference
e_in = np.array([np.median(np.abs(P_all[gi] - spread) / spread) for gi in range(len(grid_c))])
gi = int(e_in.argmin())
print(f"\n  (in-sample reference: c={grid_c[gi]:.3f})")
paired(P_all[gi], CONST, "GAUSS-GRID b=0.5, c IN-SAMPLE")

# ---------------- how much of the constant's own error is irreducible? ----------------
print("\n--- context: dynamic range of the target ---")
print(f"  truth min {spread.min():.2f}  max {spread.max():.2f}  -> full range is "
      f"{100*(spread.max()-spread.min())/np.median(spread):.1f}% of the median")
print(f"  a predictor scoring X% median error on a target whose ENTIRE spread is ~{100*(spread.max()-spread.min())/np.median(spread):.0f}%")
print(f"  is competing for a very small amount of explainable variation.")
