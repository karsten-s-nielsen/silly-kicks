"""Last loophole: the loss-matched FREE-b fit was still IN-SAMPLE. Refit it
leave-one-source-out, on the same loss, and re-score against the constant.
Also LOSO the constant itself, so every predictor is judged on equal terms.
"""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np

D = pathlib.Path(__file__).resolve().parent
AGG = D.parent / "agg"
warnings.filterwarnings("ignore")

Z = np.load(AGG / "gt.npz", allow_pickle=False)
spread = Z["spread"]; S_b = Z["S_b"]; src = Z["src"].astype(str)
NQ = len(spread)
det_b = np.linalg.det(S_b); ok = det_b > 0
assert ok.all(), "non-PD reconstructed covariance present"

BLK = 24; B = 4000
rng = np.random.default_rng(7)
BOOT_IDX = []
blocks_by_src = {s: [np.flatnonzero(src == s)[i:i + BLK]
                     for i in range(0, (src == s).sum(), BLK)] for s in sorted(set(src))}
for _ in range(B):
    BOOT_IDX.append(np.concatenate([np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])
                                    for bl in blocks_by_src.values()]))


def paired(pred, base, name):
    r = np.abs(pred - spread) / spread
    rb = np.abs(base - spread) / spread
    d = np.array([np.nanmedian(r[i]) - np.nanmedian(rb[i]) for i in BOOT_IDX])
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    v = "BEATS constant" if dhi < 0 else ("LOSES to constant" if dlo > 0 else "NOT DISTINGUISHABLE")
    print(f"  {name:46s} median {100*np.nanmedian(r):6.3f}%  vs constant {100*(np.nanmedian(r)-np.nanmedian(rb)):+.3f} pp "
          f"[95% CI {100*dlo:+.3f},{100*dhi:+.3f}] -> {v}")


def fit_const(mask):
    cs = np.linspace(spread[mask].min(), spread[mask].max(), 2001)
    return cs[np.argmin([np.median(np.abs(c - spread[mask]) / spread[mask]) for c in cs])]


def fit_powerlaw(mask):
    best = None
    for b in np.arange(0.0, 3.01, 0.01):
        u = det_b[mask] ** b / spread[mask]
        aa = np.linspace(0.5 / np.median(u), 2.0 / np.median(u), 400)
        e = np.array([np.median(np.abs(a * u - 1.0)) for a in aa])
        j = int(e.argmin())
        if best is None or e[j] < best[0]:
            best = (e[j], b, aa[j])
    return best[2], best[1]


# LOSO both the constant and the power law
p_const_loso = np.empty(NQ); p_fb_loso = np.empty(NQ)
print("--- leave-one-source-out refits (both predictors, same loss) ---")
for s in sorted(set(src)):
    te = src == s; tr = ~te
    c = fit_const(tr); p_const_loso[te] = c
    a, b = fit_powerlaw(tr); p_fb_loso[te] = a * det_b[te] ** b
    print(f"  held out {s:18s}: constant c={c:7.2f} | power law a={a:.5g} b={b:.3f}"
          f"  -> held-out err  const {100*np.median(np.abs(c-spread[te])/spread[te]):6.3f}%"
          f"  power {100*np.median(np.abs(a*det_b[te]**b-spread[te])/spread[te]):6.3f}%")

print("\n--- scored against the OUT-OF-SAMPLE constant baseline (equal terms) ---")
print(f"  CONSTANT (LOSO) median rel err = {100*np.median(np.abs(p_const_loso-spread)/spread):.3f}%")
paired(p_fb_loso, p_const_loso, "FREE-b power law, LOSO (2 free params)")

# and the zero-parameter oracle, which needs no fitting at all -> no LOSO needed
from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, GRID_NX, GRID_NY, GRID_RESOLUTION
import sys as _s
_s.path.insert(0, r"D:/Development/karstenskyt__silly-kicks_part-deux")
GXX, GYY = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
PTS = np.stack([GXX.ravel(), GYY.ravel()], 1); CELL = GRID_RESOLUTION**2
mu = Z["mu"]; neff = Z["neff_true"]


def gg(mu_i, Sig):
    d = PTS - mu_i
    det = Sig[0, 0] * Sig[1, 1] - Sig[0, 1] ** 2
    inv = np.array([[Sig[1, 1], -Sig[0, 1]], [-Sig[0, 1], Sig[0, 0]]]) / det
    e = -0.5 * np.einsum("ij,jk,ik->i", d, inv, d)
    P = np.exp(e - e.max()); P = P / P.sum(); nz = P[P > 0]
    return float(np.exp(-np.sum(nz * np.log(nz))) * CELL)


fac2 = neff ** (-1.0 / 3.0)
S_unb = S_b * (neff / (neff - 1.0))[:, None, None]
p_or = np.array([gg(mu[i], S_b[i] + S_unb[i] * fac2[i]) for i in range(NQ)])
paired(p_or, p_const_loso, "ORACLE exact-neff (0 free params)")
