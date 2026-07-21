"""Re-score the headline comparison on the UNSEEN subset only (drop the queries
whose full leaf vector collides with a training row). Same predictors, same
source-stratified paired block bootstrap as the 36k scorer.
"""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np

REPO = pathlib.Path("/home/karsten/Development/silly-kicks")
sys.path.insert(0, str(REPO))
D = pathlib.Path(__file__).resolve().parent
OUT = D / "out"
warnings.filterwarnings("ignore")
from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, GRID_NX, GRID_NY, GRID_RESOLUTION

Z = np.load(OUT / "gt.npz", allow_pickle=False)
coll = np.load(OUT / "collisions.npy")
qi = Z["query_index"]
mask = ~np.isin(qi, coll)
print(f"[unseen-only] dropping {(~mask).sum()} collided queries -> n={mask.sum()}")

spread = Z["spread"][mask]; mu = Z["mu"][mask]; S_b = Z["S_b"][mask]
neff_true = Z["neff_true"][mask]; src = Z["src"].astype(str)[mask]
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


BLK, B = 24, 4000
rng = np.random.default_rng(7)
BOOT_IDX = []
blocks_by_src = {s: [np.flatnonzero(src == s)[i:i + BLK]
                     for i in range(0, (src == s).sum(), BLK)] for s in sorted(set(src))}
for _ in range(B):
    BOOT_IDX.append(np.concatenate([np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])
                                    for bl in blocks_by_src.values()]))


def relerr(p):
    return np.abs(p - spread) / spread


def report(pred, name, base=None):
    r = relerr(pred); good = np.isfinite(r)
    med, p90 = np.median(r[good]), np.percentile(r[good], 90)
    boots = np.array([np.median(r[i][np.isfinite(r[i])]) for i in BOOT_IDX])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    line = f"  {name:40s} median {100*med:6.3f}%  [95% CI {100*lo:6.3f}-{100*hi:6.3f}]  p90 {100*p90:6.3f}%  n={good.sum()}"
    if base is not None:
        rb = relerr(base)
        dd = np.array([np.median(r[i][np.isfinite(r[i])]) - np.median(rb[i][np.isfinite(rb[i])]) for i in BOOT_IDX])
        dlo, dhi = np.percentile(dd, [2.5, 97.5])
        dpt = med - np.median(rb[np.isfinite(rb)])
        v = "BEATS constant" if dhi < 0 else ("LOSES to constant" if dlo > 0 else "NOT DISTINGUISHABLE")
        line += f"\n{'':44s}vs constant: delta {100*dpt:+.3f} pp  [95% CI {100*dlo:+.3f},{100*dhi:+.3f}]  -> {v}"
    print(line)
    return med


print(f"\n=== UNSEEN-ONLY n={NQ} | TRUTH median {np.median(spread):.3f} CV {spread.std()/spread.mean():.4f}")
cs = np.linspace(spread.min(), spread.max(), 2001)
c_star = cs[np.argmin([np.median(np.abs(c - spread) / spread) for c in cs])]
CONST = np.full(NQ, c_star)
print(f"  optimal constant c* = {c_star:.3f}")
report(CONST, f"CONSTANT = {c_star:.2f} m2 (zero information)")

best = None
for c in np.arange(0.0, 1.51, 0.005):
    p = np.array([gauss_grid_spread(mu[i], S_b[i] * (1 + c)) for i in range(NQ)])
    r = np.median(relerr(p))
    if best is None or r < best[0]:
        best = (r, c, p)
_, cinf, p_gg = best
print(f"\n  fitted inflation c = {cinf:.3f} (in-sample)")
report(p_gg, "GAUSS-GRID b=0.5 (truncation-exact)", base=CONST)

fac2 = neff_true ** (-1.0 / 3.0)
S_unb = S_b * (neff_true / (neff_true - 1.0))[:, None, None]
p_or = np.array([gauss_grid_spread(mu[i], S_b[i] + S_unb[i] * fac2[i]) for i in range(NQ)])
report(p_or, "GAUSS-GRID + EXACT neff (0 free params)", base=CONST)

det_b = np.linalg.det(S_b); ok = det_b > 0
A = np.stack([np.ones(ok.sum()), np.log(det_b[ok])], 1)
coef, *_ = np.linalg.lstsq(A, np.log(spread[ok]), rcond=None)
a, bexp = np.exp(coef[0]), coef[1]
p_fb = np.full(NQ, np.nan); p_fb[ok] = a * det_b[ok] ** bexp
report(p_fb, f"FREE-b power law (b={bexp:.3f})", base=CONST)
corr = np.corrcoef(np.log(det_b[ok]), np.log(spread[ok]))[0, 1]
print(f"  corr(log det(S_b), log spread) = {corr:.4f}   R2 = {corr**2:.4f}")
