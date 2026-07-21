"""Gap D analysis: can per-leaf moment aggregates reproduce ghost_gk_density_spread,
and do they beat a no-model constant?

Truth  : production predict_density(kde_backend="vectorized") spread, n=480 real queries.
Scoring: median |pred-truth|/truth, with a BLOCK bootstrap that respects source blocks
         (queries are consecutive frames -> serially correlated) and a PAIRED bootstrap
         on the DIFFERENCE vs the constant baseline, which is the decisive comparison.

Predictors (all built from per-(tree,leaf) {n,Sx,Sy,Sxx,Sxy,Syy} aggregates only,
except the oracle which is additionally handed exact neff):
  0  CONSTANT           no model at all; c grid-searched to be maximally generous
  1  GAUSS-GRID b=0.5   dimensionally correct, grid-evaluated (truncation-exact), 1 free param
  1b GAUSS-ANALYTIC     same but closed-form untruncated -> isolates the truncation correction
  2  ORACLE exact-neff  ZERO free parameters; S4b's best possible case
  3  FREE-b power law   a*det(S_b)^b, 2 free params (upper bound on the family)
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
spread = Z["spread"]; mu = Z["mu"]; S_b = Z["S_b"]; neff_true = Z["neff_true"]
mode_xy = Z["mode_xy"]; mean_xy = Z["mean_xy"]; n_modes = Z["n_modes"]
mu_true = Z["mu_true"]; S_true_b = Z["S_true_b"]; edge_mass = Z["edge_mass"]
src = Z["src"].astype(str); NQ = len(spread)

GXX, GYY = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
PTS = np.stack([GXX.ravel(), GYY.ravel()], 1)
CELL = GRID_RESOLUTION**2
GRID_AREA = GRID_NX * GRID_NY * CELL


def spread_of(P):
    """EXACTLY production's definition (_ghost_gk.py:1744-1746)."""
    P = P / P.sum()
    nz = P[P > 0]
    return float(np.exp(-np.sum(nz * np.log(nz))) * CELL)


def gauss_grid_spread(mu_i, Sig):
    d = PTS - mu_i
    det = Sig[0, 0] * Sig[1, 1] - Sig[0, 1] ** 2
    if det <= 0:
        return np.nan
    inv = np.array([[Sig[1, 1], -Sig[0, 1]], [-Sig[0, 1], Sig[0, 0]]]) / det
    e = -0.5 * np.einsum("ij,jk,ik->i", d, inv, d)
    P = np.exp(e - e.max())
    return spread_of(P.reshape(GRID_NX, GRID_NY))


def gauss_analytic_spread(Sig):
    det = Sig[0, 0] * Sig[1, 1] - Sig[0, 1] ** 2
    return 2 * np.pi * np.e * np.sqrt(det) if det > 0 else np.nan


# ---------------- block bootstrap indices, respecting source blocks ----------------
BLK = 24
B = 4000
rng = np.random.default_rng(7)
src_slices = {}
for s in sorted(set(src)):
    idx = np.flatnonzero(src == s)
    blocks = [idx[i:i + BLK] for i in range(0, len(idx), BLK)]
    src_slices[s] = blocks
BOOT_IDX = []
for _ in range(B):
    parts = []
    for s, blocks in src_slices.items():
        pick = rng.integers(0, len(blocks), len(blocks))
        parts.append(np.concatenate([blocks[i] for i in pick]))
    BOOT_IDX.append(np.concatenate(parts))
print(f"[boot] {B} block-bootstrap replicates, block={BLK}, source-stratified")


def relerr(pred):
    return np.abs(pred - spread) / spread


def report(pred, name, base=None):
    r = relerr(pred)
    good = np.isfinite(r)
    med = np.median(r[good])
    p90 = np.percentile(r[good], 90)
    boots = np.array([np.median(r[i][np.isfinite(r[i])]) for i in BOOT_IDX])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    line = (f"  {name:40s} median {100*med:6.3f}%  [95% CI {100*lo:6.3f}-{100*hi:6.3f}]  "
            f"p90 {100*p90:6.3f}%  n={good.sum()}")
    if base is not None:
        rb = relerr(base)
        d = np.array([np.median(r[i][np.isfinite(r[i])]) - np.median(rb[i][np.isfinite(rb[i])])
                      for i in BOOT_IDX])
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        dpt = med - np.median(rb[np.isfinite(rb)])
        verdict = "BEATS constant" if dhi < 0 else ("LOSES to constant" if dlo > 0 else "NOT DISTINGUISHABLE")
        line += f"\n{'':44s}vs constant: delta {100*dpt:+.3f} pp  [95% CI {100*dlo:+.3f},{100*dhi:+.3f}]  -> {verdict}"
    print(line)
    return med


print(f"\n=== n={NQ} queries | " + ", ".join(f"{k}={v}" for k, v in pd.Series(src).value_counts().items()))
print(f"=== TRUTH spread: median {np.median(spread):.3f} m2  min {spread.min():.3f}  max {spread.max():.3f}")
print(f"===   CV {spread.std()/spread.mean():.4f}   IQR/median {(np.percentile(spread,75)-np.percentile(spread,25))/np.median(spread):.4f}")
print(f"===   grid area {GRID_AREA:.0f} m2 -> truth occupies {100*np.median(spread)/GRID_AREA:.1f}% of the grid")
print(f"=== grid-EDGE probability mass: median {np.median(edge_mass):.4e}  p99 {np.percentile(edge_mass,99):.4e}  max {edge_mass.max():.4e}")

# ---- 6. exactness of the aggregate reconstruction (re-verified here, not quoted)
print("\n--- 6. PER-LEAF AGGREGATES vs TRUE WEIGHTED MOMENTS (must be ~machine zero) ---")
print(f"  max |mu_x diff| = {np.abs(mu[:,0]-mu_true[:,0]).max():.3e}")
print(f"  max |mu_y diff| = {np.abs(mu[:,1]-mu_true[:,1]).max():.3e}")
print(f"  max |cov diff|  = {np.abs(S_b-S_true_b).max():.3e}")
print(f"  relative: max |cov diff| / median |cov| = {np.abs(S_b-S_true_b).max()/np.median(np.abs(S_true_b)):.3e}")

# ---- 0. constant baseline, grid-searched to be MAXIMALLY generous
print("\n--- 0. NO-MODEL CONSTANT BASELINE (c grid-searched in-sample = maximally generous) ---")
cs = np.linspace(spread.min(), spread.max(), 2001)
errs = np.array([np.median(np.abs(c - spread) / spread) for c in cs])
c_star = cs[errs.argmin()]
CONST = np.full(NQ, c_star)
print(f"  optimal constant c* = {c_star:.3f} m2  (median(truth) = {np.median(spread):.3f})")
report(CONST, f"CONSTANT = {c_star:.2f} m2 (zero information)")

# ---- 1. b = 0.5 constrained
print("\n--- 1. b CONSTRAINED TO 0.5 (dimensionally correct), 1 free dimensionless inflation ---")
best = None
for c in np.arange(0.0, 1.51, 0.005):
    p = np.array([gauss_grid_spread(mu[i], S_b[i] * (1 + c)) for i in range(NQ)])
    r = np.median(relerr(p))
    if best is None or r < best[0]:
        best = (r, c, p)
_, cinf, p_gg = best
print(f"  fitted inflation c = {cinf:.3f}  (exponent on det(S) is exactly 0.5; 1 free param, fitted IN-SAMPLE)")
report(p_gg, "GAUSS-GRID b=0.5 (truncation-exact)", base=CONST)
p_ga = np.array([gauss_analytic_spread(S_b[i] * (1 + cinf)) for i in range(NQ)])
report(p_ga, "GAUSS-ANALYTIC b=0.5 (no truncation)", base=CONST)
print(f"  truncation correction size: median |grid-analytic|/grid = {100*np.median(np.abs(p_gg-p_ga)/p_gg):.3f}%")

# ---- 2. oracle exact-neff, ZERO free parameters
print("\n--- 2. ORACLE exact-neff, ZERO free parameters (S4b's BEST POSSIBLE case) ---")
fac2 = neff_true ** (-1.0 / 3.0)
S_unb = S_b * (neff_true / (neff_true - 1.0))[:, None, None]
p_or = np.array([gauss_grid_spread(mu[i], S_b[i] + S_unb[i] * fac2[i]) for i in range(NQ)])
print(f"  bandwidth inflation factor neff^(-1/3): min {fac2.min():.5f} median {np.median(fac2):.5f} max {fac2.max():.5f}")
print(f"    -> it varies by only {100*(fac2.max()-fac2.min())/np.median(fac2):.2f}% across the query set,")
print(f"       i.e. the exact-neff oracle is barely distinguishable from the fitted constant c={cinf:.3f}")
report(p_or, "GAUSS-GRID + EXACT neff (0 free params)", base=CONST)

# ---- 3. free-b power law
print("\n--- 3. FREE-b POWER LAW  spread = a*det(S_b)^b  (2 free params, OLS in log space) ---")
det_b = np.linalg.det(S_b); ok = det_b > 0
A = np.stack([np.ones(ok.sum()), np.log(det_b[ok])], 1)
coef, *_ = np.linalg.lstsq(A, np.log(spread[ok]), rcond=None)
a, bexp = np.exp(coef[0]), coef[1]
bb = []
for i in BOOT_IDX[:1000]:
    o = det_b[i] > 0
    Ai = np.stack([np.ones(o.sum()), np.log(det_b[i][o])], 1)
    ci, *_ = np.linalg.lstsq(Ai, np.log(spread[i][o]), rcond=None)
    bb.append(ci[1])
blo, bhi = np.percentile(bb, [2.5, 97.5])
print(f"  fitted a={a:.6g}  b={bexp:.4f}  [95% CI {blo:.4f}, {bhi:.4f}]  (dimensionally coherent b=0.5)")
p_fb = np.full(NQ, np.nan); p_fb[ok] = a * det_b[ok] ** bexp
report(p_fb, f"FREE-b power law (b={bexp:.3f})", base=CONST)
corr = np.corrcoef(np.log(det_b[ok]), np.log(spread[ok]))[0, 1]
print(f"  corr(log det(S_b), log spread) = {corr:.4f}   R2 = {corr**2:.4f}")

# ---- 5. multimodality
print("\n--- 5. MULTIMODALITY (aggregates collapse the mode onto the mean) ---")
dist = np.hypot(mode_xy[:, 0] - mean_xy[:, 0], mode_xy[:, 1] - mean_xy[:, 1])
print(f"  local maxima >=20% of peak (permissive >= rule): 1 mode {100*(n_modes==1).mean():.1f}% | "
      f">=2 modes {100*(n_modes>=2).mean():.1f}% | max {n_modes.max()}")
print(f"  |mode-mean| (m): median {np.median(dist):.3f}  p75 {np.percentile(dist,75):.3f}  "
      f"p90 {np.percentile(dist,90):.3f}  p99 {np.percentile(dist,99):.3f}  max {dist.max():.3f}")
for thr in (1, 2, 4, 6):
    print(f"    fraction with |mode-mean| > {thr} m: {100*(dist>thr).mean():.1f}%")

# ---- per-source
print("\n--- PER-SOURCE ---")
for s in sorted(set(src)):
    k = src == s
    row = f"  {s:18s} n={k.sum():4d}  truth median {np.median(spread[k]):7.2f} CV {spread[k].std()/spread[k].mean():.4f}  |"
    for nm, p in (("const", CONST), ("b=0.5", p_gg), ("oracle", p_or), ("free-b", p_fb)):
        row += f"  {nm} {100*np.nanmedian(np.abs(p[k]-spread[k])/spread[k]):6.3f}%"
    print(row)
