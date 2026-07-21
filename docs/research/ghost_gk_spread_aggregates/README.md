# Can per-leaf moment aggregates replace per-sample responses in `GhostGkModel`?

A model-engineering study of what the ghost-GK artifact has to carry in order to keep emitting
`ghost_gk_density_spread`.

Status: **Both scales complete.** The 36k and production-scale (Stage-B, 1,039,502-sample)
runs DISAGREE on the one-free-parameter arm — the conclusion is SCALE-DEPENDENT. §3.4 is the
headline result and supersedes any reading of §3.1 alone.

---

## 1. The question

`GhostGkModel` (`silly_kicks/tracking/_ghost_gk.py`) is an RFCDE-style conditional density
estimator. Its `save()` persists three per-sample arrays:

| array | shape at 36k `default` | shape at Stage-B |
|---|---|---|
| `training_gk_x` | `(36000,)` | `(1039502,)` |
| `training_gk_y` | `(36000,)` | `(1039502,)` |
| `training_leaves` | `(36000, 243)` | `(1039502, 500)` |

These three arrays dominate the artifact. The Stage-B `rfcde_weights.npz` is **208,085,291 bytes**
(208.1 MB); measured member-by-member from the zip directory:

| member | raw bytes | stored bytes | share of artifact |
|---|---|---|---|
| `training_leaves.npy` | 4,158,008,128 | 199,793,439 | 96.0% |
| `training_gk_x.npy` | 8,316,144 | 3,051,045 | 1.47% |
| `training_gk_y.npy` | 8,316,144 | 2,845,817 | 1.37% |
| everything else (tree nodes etc.) | — | 2,394,990 | 1.15% |

The three per-sample arrays are **98.85%** of the stored artifact.

At serve time the model computes, per query, a leaf-match weight vector over the whole training
database, then runs a weighted 2-D Gaussian KDE on `(training_gk_x, training_gk_y)` to get a
60×64 density grid. Exactly one shipped column is derived from that grid's *shape*:

```python
# _ghost_gk.py:1744-1746
entropy = float(-np.sum(nz * np.log(nz)))
spread  = float(np.exp(entropy) * GRID_RESOLUTION**2)
```

`ghost_gk_density_spread` is therefore an **effective area** in m² — the exponential of the
discrete entropy of the normalised density, times the cell area. The grid is `GRID_NX=60` ×
`GRID_NY=64` at `GRID_RESOLUTION=0.5` m, so the maximum representable spread (a perfectly flat
density) is 3,840 × 0.25 = **960 m²**.

`ghost_gk_x`/`ghost_gk_y` are **not** derived from this grid: since ADR-016 they are served by
`predict_mean`, the exact boosted mean. Verified by code inspection — `predict_mean` spans
`_ghost_gk.py:1615-1649`, and the first reference to any per-sample array is at `:1677`, inside
`predict_density` (`:1650-1762`). The spread column is the only shipped output that reads them.

**The question:** does the spread column actually need per-sample responses, or can a compact
per-leaf summary reproduce it? Concretely — replace the three per-sample arrays with

```
AGG[tree, leaf] = {n, Sx, Sy, Sxx, Sxy, Syy}
```

six float64 accumulators per occupied `(tree, leaf)` cell. That is enough to reconstruct the
weighted mean and weighted covariance of the leaf-matched training responses for any query,
without storing any individual response.

Payload comparison (recomputed, not copied):

| | occupied cells | AGG bytes | AGG MB |
|---|---|---|---|
| 36k `default` | 7,529 | 361,392 | 0.361 |
| Stage-B | 15,500 | 744,000 | 0.744 |

AGG would replace the three per-sample arrays (205,690,301 stored bytes at Stage-B), not the whole
artifact — the tree nodes are still needed for routing. That is a **276.5×** reduction on the part
it replaces, or 279.7× measured against the full 208 MB file.

The size case is not in doubt. The question is entirely whether the summary reproduces the number.

---

## 2. Method

### 2.1 Aggregate construction

Built by `np.add.at` over each tree's training leaf column:

```python
AGG = np.zeros((T, maxleaf, 6), dtype=np.float64)   # n, Sx, Sy, Sxx, Sxy, Syy
for t in range(T):
    idx = TL[:, t]
    np.add.at(AGG[t,:,0], idx, 1.0)
    np.add.at(AGG[t,:,1], idx, GX)
    np.add.at(AGG[t,:,2], idx, GY)
    np.add.at(AGG[t,:,3], idx, GX*GX)
    np.add.at(AGG[t,:,4], idx, GX*GY)
    np.add.at(AGG[t,:,5], idx, GY*GY)
```

A query's prediction sums its `T` matched leaf cells and divides by `T`, giving
`[Σw, Σwx, Σwy, Σwxx, Σwxy, Σwyy]`, from which the weighted mean `mu` and the **biased** weighted
covariance `S_b` follow directly.

This reconstruction is exact, not approximate. Re-verified inside the scorer rather than quoted
(`logs/36k_score.log`):

```
--- 6. PER-LEAF AGGREGATES vs TRUE WEIGHTED MOMENTS (must be ~machine zero) ---
  max |mu_x diff| = 7.816e-14
  max |mu_y diff| = 3.553e-14
  max |cov diff|  = 4.034e-12
  relative: max |cov diff| / median |cov| = 6.810e-13
```

So any failure below is a failure of *the Gaussian summary as a description of the density*, never
a failure of the moment arithmetic.

### 2.2 Why b = 0.5 is the dimensionally coherent constraint

For a 2-D Gaussian with covariance Σ, `exp(differential entropy) = 2πe·|Σ|^(1/2)`. The shipped
spread is `exp(discrete entropy) × cell_area`, which is the same quantity discretised onto the
grid. `det Σ` has units m⁴, so the exponent **0.5** is forced by dimensional analysis: it is the
only exponent that returns an area. A predictor of the form `spread ∝ det(S)^0.5` is a law with a
unit-consistent form; the only thing left free is a dimensionless scale.

A **free-b** fit `spread = a·det(S_b)^b` relaxes that. It fits better in-sample by construction,
but it is a local interpolation over the narrow range of `det` the query set happens to span, not
a description of the functional. The fitted exponent is itself the tell — measured
**b = 1.1756, 95% CI [0.7896, 1.4170]**, which *excludes* 0.5. A fit that has drifted off the
dimensionally-forced exponent is absorbing something other than the Gaussian relation.

### 2.3 The four predictors and the baseline

The four **predictors** take only `mu` and `S_b` — i.e. only what AGG can carry. The **baseline**
takes nothing at all: it is the same number for every query.

| arm | free params | form |
|---|---|---|
| CONSTANT (**baseline**, not a predictor) | 1, grid-searched | `spread = c*`, ignoring the query entirely |
| **GAUSS-GRID b=0.5** | 1 (dimensionless inflation `c`) | grid entropy of `N(mu, S_b·(1+c))` |
| **GAUSS-ANALYTIC b=0.5** | 1 (same `c`) | closed form `2πe·√det`, no grid truncation |
| **ORACLE exact-neff** | **0** | grid entropy of `N(mu, S_b + S_unb·neff^(-1/3))` |
| **FREE-b power law** | 2 | `a·det(S_b)^b`, OLS in log space |

The **ORACLE** arm is the strongest honest case for the aggregates. It reproduces the model's own
KDE convolution identity exactly — `_kde_setup` uses `factor = neff**(-1/(d+4))` and scales the
covariance by `factor²`, which for d=2 is `neff^(-1/3)` — with **zero fitted parameters**. If a
Gaussian summary can reproduce the spread at all, this is the arm that should show it.

The **CONSTANT** baseline is deliberately made as strong as possible: `c*` is grid-searched over
2001 candidates **in-sample**, i.e. it is allowed to see the test set. Any predictor that cannot
beat it is carrying no usable information about the query.

### 2.4 Bootstrap design

Queries are consecutive tracking frames, so they are not independent. The scorer uses a
**source-stratified block bootstrap**: blocks of 24 consecutive queries, resampled within each
source separately, B = 4000 replicates, `np.random.default_rng(7)`. Deltas versus the constant are
**paired** — the same replicate index set scores both arms and the difference is taken inside the
replicate — so the CI is on the difference, not a comparison of two marginal CIs.

### 2.5 Ground truth

Ground truth is the real `predict_density` at the production default `kde_backend="vectorized"`,
on real feature vectors extracted by `_serve_positions_core` from repo tracking fixtures
(`metrica_slim`, `sportec_slim`, `skillcorner_slim`) after `smooth_frames` → `derive_velocities`.
Leaf routing goes through the production `_vectorized_leaf_indices`. Nothing is injected or
stubbed.

The 36k ground truth was independently reproduced from a temp copy of the weights on a 60-query
subsample (`logs/36k_verify.log`):

```
=== REPRODUCTION vs SAVED gt.npz ===
  max |spread_repro - spread_saved| = 0.000000e+00  (relative 0.000e+00)
  exact bitwise spread match: 60/60
```

---

## 3. Results

### 3.1 36k `default` — n = 480 queries (sportec 333 / skillcorner 76 / metrica 71)

Target distribution: median **378.151 m²**, min 305.736, max 395.236, CV **0.0574**, on a 960 m²
grid (39.4% occupancy). The full min-to-max range is only **23.7%** of the median — every
predictor is competing for a small amount of explainable variation.

| predictor | median rel err | 95% CI | p90 | Δ vs constant (pp) | Δ 95% CI | verdict |
|---|---|---|---|---|---|---|
| CONSTANT `c*`=386.51 m² (in-sample) | **2.211%** | 1.664–5.322 | 9.971% | — | — | baseline |
| GAUSS-GRID b=0.5, `c`=0.070 in-sample | **1.083%** | 0.805–2.038 | 4.582% | −1.128 | [−3.288, −0.797] | BEATS |
| GAUSS-GRID b=0.5, `c` leave-one-source-out | **2.425%** | — | — | +0.214 | [−2.671, +0.871] | not distinguishable |
| GAUSS-ANALYTIC b=0.5 (untruncated) | **24.013%** | 21.698–24.540 | 31.716% | +21.802 | [+19.059, +22.040] | LOSES |
| **ORACLE exact-neff (0 free params)** | **2.210%** | 1.691–2.753 | 3.467% | **−0.001** | [−3.457, +0.658] | **not distinguishable** |
| FREE-b power law (b=1.176) | **2.681%** | 1.802–3.289 | 5.139% | +0.470 | [−2.837, +1.398] | not distinguishable |

The load-bearing row is the **ORACLE**: the zero-free-parameter arm that reproduces the model's own
KDE identity lands **0.001 pp** from a constant that ignores the query completely, with a CI
straddling zero. The best possible honest Gaussian summary is not distinguishable from no model.

Two supporting observations:

- **Grid truncation is essential, and it is doing most of the work.** The untruncated closed form
  — the pure dimensional law — is off by 24%. Median `|grid − analytic| / grid` is **21.837%**.
  Whatever the b=0.5 arm gets right, it gets right by re-running the grid, not by the closed form.
- **The in-sample b=0.5 win is a fitting artifact.** `c` is one scalar, but refitting it
  leave-one-source-out destroys the win (1.083% → 2.425%). Per-fold values were 0.075 / 0.070 /
  0.030 — the fitted inflation is not stable across sources.

Per-source detail (`logs/36k_score.log`):

```
  metrica_slim       n=  71  truth median  366.12 CV 0.0216  |  const  5.568%  b=0.5  2.134%  oracle  0.699%  free-b  1.563%
  skillcorner_slim   n=  76  truth median  382.14 CV 0.0262  |  const  1.149%  b=0.5  0.367%  oracle  2.293%  free-b  3.687%
  sportec_slim       n= 333  truth median  382.10 CV 0.0651  |  const  2.140%  b=0.5  0.990%  oracle  2.433%  free-b  2.954%
```

No arm wins on all three sources, and the ranking is unstable. Computed from the row above:

| source | rank order (best → worst) |
|---|---|
| metrica | oracle 0.699 < free-b 1.563 < b=0.5 2.134 < const 5.568 |
| skillcorner | b=0.5 0.367 < const 1.149 < oracle 2.293 < free-b 3.687 |
| sportec | b=0.5 0.990 < const 2.140 < oracle 2.433 < free-b 2.954 |

The zero-parameter oracle is the **best** arm on metrica (0.699%) and **worse than the constant**
on both skillcorner and sportec. A predictor whose rank against the do-nothing baseline flips with
the provider is not carrying a stable relationship. That instability is the subject of §4.

### 3.2 Why the Gaussian summary cannot be rescued

The density is genuinely not Gaussian. Measured on the same 480 queries:

```
--- 5. MULTIMODALITY (aggregates collapse the mode onto the mean) ---
  local maxima >=20% of peak (permissive >= rule): 1 mode 36.9% | >=2 modes 63.1% | max 4
  |mode-mean| (m): median 8.054  p75 9.318  p90 10.062  p99 10.554  max 10.673
    fraction with |mode-mean| > 2 m: 100.0%
    fraction with |mode-mean| > 4 m: 95.2%
```

Recounted from `gt.npz` rather than read off the rounded percentages: **303 of 480** queries
(63.125%) have two or more modes; `|mode − mean|` exceeds 2 m on **480 of 480** and exceeds 4 m on
**457 of 480** (95.208%). A two-moment summary has exactly one mode by construction, so it cannot
represent the shape whose entropy is being measured.

A permissive `>=` local-maximum rule can inflate mode counts on plateaus. Re-run with a strict `>`
rule on the 60-query reproduction subset, the counts are identical (`logs/36k_verify.log`):

```
  permissive (>=) rule : 1 mode 33.3%  >=2 modes 66.7%  max 4
  STRICT     (> ) rule : 1 mode 33.3%  >=2 modes 66.7%  max 4
```

### 3.3 The constant baseline on the real serving population

The 480-query fixture set is not the population the column is actually emitted on. Measured
directly on the production `fct_action_context` mart by read-only query (n = 239,240 non-null):

```
n_rows     = 239251     n_nonnull = 239240     n_distinct = 211425
mn         = 342.86753379349125
mx         = 391.6305980762241
mean       = 374.6639950588477
sd_samp    = 6.695447715491222     sd_pop = 6.695433722315516
p01        = 354.51003783055353    p50 = 376.85787193225866    p99 = 384.0861940336891
CV_samp    = 0.01787053948015363   CV_pop = 0.017870502131553574
max/min    = 1.142221293871769
```

Scoring a single stored float — the population median, 376.85787193225866 — against all 239,240
live rows:

```
med_rel_err      = 0.0075292559550203106
p90_rel_err      = 0.03419745293069797
p99_rel_err      = 0.05930043065604901
max_rel_err      = 0.09019405104765141
frac_within_2pct = 0.81409
```

**One float reproduces the live column to 0.75% median, 3.4% p90, 9.0% worst case, with 81.4% of
rows inside 2%.** The production population's CV is 0.0179 versus the fixture set's 0.0574 — the
fixtures are about 3× more dispersed than production, so §3.1 is measured on the *harder* of the
two populations and if anything overstates what any predictor could win.

This does not mean the column is information-free, and it should not be described that way. Its
variance is genuinely per-action (sd_within_match 6.6405 of sd_total 6.6954 — only 0.8488 between
match means), it is stable across matches (within-match CV across 182 matches: p05 0.014749 /
median 0.017689 / p95 0.020782 / max 0.023124), and it varies systematically with keeper-to-goal
distance (CV 0.009988 → 0.025008 across deciles). It is a small, real, structured signal — and a
constant reproduces it to 0.75%.

### 3.4 Production scale — the conclusion is SCALE-DEPENDENT (headline result)

The Stage-B run completed: **n_train=1,039,502, n_trees=500, n=480 queries**, same harness, same
queries, same block-bootstrap design (B=4000, block=24, source-stratified, paired indices).

**It reverses the 36k verdict for the one-free-parameter arm.** Read this before quoting §3.1.

| arm | 36k median | production median | production vs constant | production verdict |
|---|---|---|---|---|
| CONSTANT (zero information) | 2.211% | **1.309%** | — | — |
| b=0.5, `c` fitted **leave-one-source-out** | 2.425% | **0.844%** | -0.465 pp, CI [-1.267, -0.348] | **BEATS** |
| b=0.5, `c` fitted in-sample | 1.083% | 0.657% | -0.652 pp, CI [-1.375, -0.518] | beats |
| ORACLE exact-neff (0 free params) | 2.210% | 1.493% | +0.184 pp, CI [-0.881, +0.364] | not distinguishable |
| FREE-b power law | 2.681% | 2.485% | +1.176 pp, CI [+0.044, +1.523] | loses |

At 36k the b=0.5 arm's apparent win **vanished** once `c` was fitted out-of-sample. At production
scale it **survives** leave-one-source-out with a confidence interval excluding zero. Per-fold
held-out `c`: metrica 0.030 -> 1.065%, skillcorner 0.025 -> 0.435%, sportec 0.025 -> 0.725%.

So: **per-leaf moment aggregates plus ONE fitted dimensionless inflation constant do reproduce
`ghost_gk_density_spread` at production scale.** Anyone wanting the column back should start here,
not from the 36k result. The zero-free-parameter oracle still ties the constant at both scales.

**Mechanism at scale** (all DEMONSTRATED, pasted from the run):

    [k] nonzero-weight training rows: min=1039502 median=1039502 max=1039502
    [k] fraction of corpus at NONZERO weight: min=1.0000 median=1.0000 max=1.0000
    [neff] neff/n_train: min=0.8761 median=0.9501 max=0.9780
    [occ] occupied cells=15500 counts min=20 median=1472 max=1027314

Diffuseness **strengthens** with scale (`neff/n_train` 0.9127 -> 0.9501), and the leaf partition is
more degenerate: every tree saturates at 31 leaves and one cell holds 1,027,314 rows (98.83%, vs
96.38% at 36k).

**A universal that is scale-dependent, stated per scale rather than collapsed.** "Every query draws
on all training rows" is **false at 36k** (min 35,989 of 36,000) and **true at production scale**
(min = median = max = 1,039,502). An earlier draft asserted it unconditionally.

**Multimodality strengthens too**, reinforcing ADR-016's rejection of the mode as a served estimate:
100.0% of queries have >=2 modes (vs 63.1% at 36k), `|mode-mean|` median **9.222 m** (p99 12.259,
max 12.480), and 100.0% exceed 6 m.

**The queries are not fully unseen at this scale — measured, and it does not change the result.**
`[unseen] exact 500-leaf-vector collisions: 7/480`, against 0/480 at 36k (distinct training leaf
vectors: 1,005,753 of 1,039,502). The seven are at indices [203, 217, 218, 219, 233, 293, 385] and
are **all in `sportec_slim`** (0 in metrica, 0 in skillcorner) — 2.1% of that source, 0% of the
other two.

Rescoring on the 473 unseen queries leaves **every verdict unchanged**:

| arm | n=480 | n=473 unseen-only | verdict |
|---|---|---|---|
| CONSTANT | 1.309% | 1.297% | — |
| b=0.5 (`c` in-sample) | 0.657% | **0.667%**, -0.630 pp CI [-1.388, -0.504] | BEATS |
| ORACLE exact-neff | 1.493% | 1.503%, +0.206 pp CI [-0.930, +0.378] | ties |
| FREE-b power law | 2.485% | 2.502%, +1.204 pp CI [+0.017, +1.542] | loses |

Scope of this check, stated precisely: it rescores the **in-sample-`c`** b=0.5 arm, the oracle and
the free-b arm. The **leave-one-source-out** b=0.5 arm (0.844%) was not rescored on the unseen
subset. Since all seven collisions fall in one source, the LOSO fold most affected is the
held-out-sportec one; the in-sample arm moved 0.010 pp under the same exclusion, so a comparable
movement is expected but is **inferred, not measured**.

**Moment reconstruction stays exact at scale:** max |mu_x diff| 1.386e-13, |mu_y| 3.197e-13,
|cov| 1.864e-11 (relative 3.041e-12).

**Truth dispersion:** median 377.897 m2, CV **0.0389** (vs the live serving population's 0.0179,
§3.3) — the fixture query set is about twice as dispersed as production traffic, which makes the
constant baseline harder to beat here than it would be on the mart.

**Why the design still strips the arrays despite this result.** Aggregates require dropping
`training_leaves` too (96.0% of the artifact, and the feature-inversion channel), so they buy
**zero** reduction in what the artifact carries beyond what stripping already achieves — they only
preserve a column with no numeric consumer. That is a product decision, not a measurement one, and
the measurement above is banked precisely so it can be revisited on its merits.


## 4. The countervailing result, kept

There is a second LOSO analysis that points the *other* way, and it is not dropped here because it
is the honest framing of what the aggregates do carry.

§3.1's LOSO row refits only the predictor's `c` leave-one-source-out while the **constant keeps its
in-sample fit**. That is not equal terms. Refitting the constant leave-one-source-out too
(`logs/36k_score_loso_constant.log`):

```
--- leave-one-source-out refits (both predictors, same loss) ---
  held out metrica_slim      : constant c= 388.48 | power law a=67.193 b=0.270  -> held-out err  const  6.106%  power  4.256%
  held out skillcorner_slim  : constant c= 384.09 | power law a=0.51758 b=1.020  -> held-out err  const  1.139%  power  2.374%
  held out sportec_slim      : constant c= 367.33 | power law a=0.22762 b=1.150  -> held-out err  const  4.920%  power  3.536%

--- scored against the OUT-OF-SAMPLE constant baseline (equal terms) ---
  CONSTANT (LOSO) median rel err = 5.193%
  FREE-b power law, LOSO (2 free params)         median  3.684%  vs constant -1.509 pp [95% CI -2.283,-0.735] -> BEATS constant
  ORACLE exact-neff (0 free params)              median  2.210%  vs constant -2.983 pp [95% CI -3.978,-2.238] -> BEATS constant
```

Against a constant that is *not* allowed to see the test population, **both arms tested there win
decisively** — the zero-parameter oracle by 2.983 pp with a CI well clear of zero. (Only the free-b
and oracle arms were run in this analysis; the b=0.5 arm was not.)

This is not a contradiction; it is a statement about what the baseline is. The aggregates do carry
real, transferable information — `corr(log det(S_b), log spread) = 0.8305`, R² = 0.6897. What they
do not do is beat a constant that is allowed to see where the test population sits.

**Which framing is operative is not a matter of taste, and §3.3 settles it empirically.** The
LOSO-constant framing is the pessimistic one for the constant: it says a constant transfers badly
across populations (2.211% → 5.193%). If that were true of the *production* population, the
aggregates would have a case. It is not. A single float — the production median — reproduces the
live column at **0.75% median error over 239,240 real rows** (§3.3), better than any arm scores on
any framing here. The constant's apparent fragility is an artifact of the fold structure: three
provider-shaped folds, one game each, with per-source medians of 366.12 / 382.14 / 382.10 m²
(§7.1). Holding out a whole provider moves the target's centre in a way that holding out
production traffic does not.

So the LOSO-constant result is retained because it is true and it bounds what the aggregates carry
— but it does not overturn the conclusion, because the population it warns about is not the
population the column is served on.

---

## 5. Mechanism: the weights are diffuse, so the density is barely conditional

The reason a Gaussian summary neither helps nor hurts much is that the conditional density is
close to unconditional. Measured on the 36k model (`logs/36k_seen_unseen.log`):

```
[C] weight concentration over the 36000-row database:
      neff:            min=27918 median=32856 max=34492  (uniform would be 36000)
      neff / n_train:  median=0.9127
      largest single weight: median=6.148e-05  (uniform = 2.778e-05)
      mass in top-100 rows:  median=0.0053  (uniform = 0.0028)
      rows with NONZERO weight: min=35989 median=36000 max=36000
```

The median query draws on an effective sample size of **91.27%** of the database. The largest
single weight is 2.21× uniform and the top 100 rows hold 0.53% of the mass. The number of rows at
nonzero weight is a distribution, not a constant — **minimum 35,989, median 36,000** of 36,000 —
so for most but not all queries every row contributes.

The structural driver is leaf occupancy. From `logs/36k_ground_truth.log`:

```
[occ] occupied cells=7529 | leaves/tree min=27 max=31
[occ] cell counts: min=20 p1=21 median=139 mean=1161.91 p99=26240 max=34697
[occ]   cells with n<20: 0 (0.000%)
```

The largest single leaf holds 34,697 of 36,000 rows — **96.38%** of the corpus in one cell. A query
routing into that leaf in most trees matches nearly everything.

**Does it hold at scale?** At Stage-B the partition is *more* degenerate, not less. Every tree
saturates at exactly 31 leaves, and the dominant cell holds 1,027,314 of 1,039,502 rows =
**98.83%**, up from 96.38%:

```
[occ] occupied cells=15500 | leaves/tree min=31 max=31  (1.9s)
[occ] cell counts: min=20 p1=21 median=1472 mean=33532.32 p99=877668 max=1027314
[occ]   cells with n<20: 0 (0.000%)
```

Occupancy therefore predicts the 36k finding should hold or strengthen. The decisive statistic is
`neff / n_train` at Stage-B, which is computed per query inside the production run — reported in
§3.4 when that run completes, not inferred from occupancy here.

---

## 6. Conclusion

**At the 36k `default` scale, per-leaf Gaussian moment aggregates do not pay.** [DEMONSTRATED]

The strongest honest arm — zero free parameters, reproducing the model's own KDE convolution
identity exactly — scores 2.210% median relative error against a no-model constant's 2.211%, a
delta of −0.001 pp with a 95% CI of [−3.457, +0.658]. The only arm that beat the constant did so
with an inflation constant fitted in-sample, and refitting it leave-one-source-out removed the win.

The mechanism is understood and is not a tuning problem: leaf-match weights are diffuse
(median `neff/n_train` = 0.9127), so the conditional density is close to unconditional and its
spread is close to constant; and the density is multimodal on 63.1% of queries, which a two-moment
summary cannot represent by construction.

**What this does not say.** The aggregates are not information-free — `det(S_b)` correlates with
the target at R² = 0.69, and against a leave-one-source-out constant both arms tested win clearly
(§4). The finding is that they do not beat a constant on the population that matters.

**The independent production check agrees.** On 239,240 live rows, one stored float reproduces the
column at 0.75% median / 3.4% p90 error (§3.3) — better than any aggregate arm achieves on the
fixture set under any framing. Two measurements on different populations, by different routes,
point the same way.

**Engineering read.** Retiring the per-sample arrays removes `ghost_gk_density_spread`. Replacing
it with a per-leaf aggregate reconstruction is measured, at 36k, to be no better than replacing it
with a constant — so the aggregate is not a reason to keep a serving path. Whether the column
should be retired outright, or served as a constant, or kept as-is, is a consumer decision this
study does not make. What can be said is that a 0.36–0.74 MB aggregate buys **no measurable
accuracy** over an 8-byte float on either population tested — while carrying a serving path, a
reconstruction step and a shape assumption the density does not satisfy.

**Production scale:** see §3.4. Until that run completes, the conclusion above is scoped to the
36k artifact and is not extrapolated.

---

## 7. Limitations

1. **Query sources are repo fixtures, so LOSO folds are provider-shaped, not independent matches.**
   The three folds are `metrica_slim` (Sample_Game_3), `sportec_slim` (J03WOH) and
   `skillcorner_slim` (skillcorner_1899585) — one game each. A LOSO fold therefore measures
   *provider transfer*, not match-level generalisation, and the LOSO-constant baseline is
   correspondingly weak because per-source medians differ (366.12 / 382.14 / 382.10 m²). This is
   the direct cause of §4's disagreement with §3.1 and should not be read as a
   match-level-generalisation result.

2. **The two scales differ in trees as well as database size.** 36k `default` is 36,000 × 243;
   Stage-B is 1,039,502 × 500 — 28.9× the samples but also 2.06× the trees. Any difference between
   the two measurements confounds both axes. (Note `metadata.json`'s `n_estimators=500` is the
   *request*; the 36k artifact actually fitted 243.)

3. **The fixture query set is ~3× more dispersed than the production serving population, and *why*
   is not established.** Fixtures give CV 0.0574; production gives 0.0179 (§3.3). Two contributing
   factors were measured — in-sample queries self-match in all trees and take the maximum leaf
   weight of exactly 1.0 (in-sample CV 0.0787 vs out-of-sample 0.0638), and the two populations
   differ in range (fixtures reach down to ~305 m², while 239,240 production rows never go below
   342.867534). The residual gap is **[PLAUSIBLE], not demonstrated**: the training corpus is
   filtered to detected-keeper goal-box states while the mart scores every action-linked frame,
   most of which sit in a coarse region of the partition. Supporting gradient, measured: production
   CV rises monotonically with keeper-to-goal distance (0.009988 → 0.025008 across deciles). This
   was not closed because it needs real production feature vectors, and no proxy was substituted.
   Direction of the uncertainty is favourable — the fixture set is the harder population — but it
   is uncertainty.

4. **Cost forced n = 480 rather than the full 1,593-query pool.** Measured from the two 36k runs
   held here: 2081 s / 480 = **4.34 s/query** (`logs/36k_ground_truth.log`) and 256 s / 60 =
   **4.27 s/query** (`logs/36k_verify.log`). The 480 were selected deterministically
   (`np.linspace`, no RNG), proportional per source and evenly spaced within each source across the
   full time range.

5. **A ~300× faster FFT backend exists but was deliberately not used for headline numbers.**
   Re-measured for this note on 28 stratified queries (`logs/36k_backend_compare.log`):

   ```
   vectorized    121.4s  spread[:6]=[364.311  366.4966 366.6346 366.8681 371.335  355.7736]
   fft-cic         0.4s  vs vectorized: max rel diff 3.372e-03  median 2.720e-03
   fft             0.4s  vs vectorized: max rel diff 3.138e-03  median 3.217e-04
   ```

   That 0.03–0.3% backend discrepancy is the *same order* as a well-performing predictor's error,
   so fft is sound for screening but every headline number here is on the production `vectorized`
   backend.

6. **Not closed by this study:** whether a richer summary (e.g. per-leaf mixture components rather
   than two moments) could represent the multimodality. Only the two-moment Gaussian form was
   tested. Given §3.2's mode counts, a mixture is the obvious next candidate if the column is ever
   deemed worth preserving — but it is unmeasured, and its payload would be larger.

7. **Production-scale scoring arms are pending** (§3.4). No production-scale verdict is stated
   until they land; the occupancy result in §5 is directional evidence, not the measurement.

---

## 8. Reproduction

### 8.1 Files

```
harness/
  01_build_queries.py          # fixtures -> queries_all.parquet (+ meta)
  02_ground_truth_36k.py       # occupancy + AGG + real predict_density -> gt.npz
  03_score.py                  # HEADLINE: 4 arms, source-stratified paired block bootstrap
  04_score_loso_c.py           # LOSO on the b=0.5 inflation only
  05_score_loso_constant.py    # LOSO on the constant too (the §4 countervailing arm)
  06_seen_unseen.py            # leaf-vector collision test + weight concentration
  07_verify_ground_truth.py    # independent bit-for-bit reproduction of gt.npz
  08_ground_truth_prod.py      # Stage-B, fork pool + JSONL checkpointing
  09_assemble_prod.py          # JSONL -> gt.npz in the 36k schema
  10_collisions_prod.py        # recover colliding query indices
  11_unseen_only_prod.py       # re-score on the non-colliding subset
  12_prod_probe{1..5}.py       # Stage-B cost/memory/parallelism scoping probes
  13_backend_compare.py        # vectorized vs fft / fft-cic agreement
  logs/                        # pasted stdout for every step
```

`03/04/05` consume `gt.npz` only and run in seconds. `02` and `08` are the expensive steps.

**These scripts are banked exactly as executed — they were not rewritten for publication**, so
they carry the directory layout of the original run rather than the layout they now sit in.
Reproducing therefore means recreating that layout, not running them in place:

| script | expects |
|---|---|
| `01`, `02`, `13` | `queries_all.parquet` / `queries_all_meta.parquet` in **their own directory** |
| `03`–`07` | an `agg/` directory **as a sibling** holding `gt.npz` + the query parquets |
| `06`, `07` | a `weights_copy/` directory **in their own directory** (a copy of the bundled `default` weights — deliberately not the repo path) |
| `01`–`07`, `13` | `REPO` hardcoded to the repo root; edit that one line |
| `08`–`11` | an `out/` subdirectory; `08` reads `SK_REPO`, `10` reads `SK_MODEL` from the environment |

The `REPO` constant is a single line at the top of each file. Nothing else needs editing.

### 8.2 Seeds

| purpose | seed | where |
|---|---|---|
| block bootstrap | `np.random.default_rng(7)` | `03/04/05_*.py` |
| verification subsample | `np.random.default_rng(11)` | `07_verify_ground_truth.py` |
| leaf-vector hashing | `np.random.default_rng(12345)` | `08_ground_truth_prod.py` |
| query selection | **none — `np.linspace`, deterministic** | `02`, `08` |

### 8.3 36k run (workstation)

Recreate the two-directory layout the scripts expect, then run in order:

```bash
mkdir -p work/agg work/score
cp harness/01_build_queries.py harness/02_ground_truth_36k.py harness/13_backend_compare.py work/agg/
cp harness/0[3-7]_*.py work/score/
cp -r <repo>/silly_kicks/tracking/_ghost_gk_weights/default work/score/weights_copy

PY=.venv/Scripts/python.exe            # local: Python 3.10.19, numpy 2.2.6, pandas 2.3.3
$PY work/agg/01_build_queries.py        # ~1 min      -> queries_all.parquet
$PY work/agg/02_ground_truth_36k.py     # ~35 min     (measured 2081 s / 480 = 4.34 s/query)
$PY work/score/03_score.py              # seconds     -> the §3.1 headline table
$PY work/score/04_score_loso_c.py       # seconds     -> the §3.1 LOSO-c row
$PY work/score/05_score_loso_constant.py # seconds    -> the §4 countervailing arm
$PY work/score/06_seen_unseen.py        # ~1 min      -> §5 weight concentration, §8.5 collisions
$PY work/score/07_verify_ground_truth.py # ~5 min     (60 queries, 256 s = 4.27 s/query)
$PY work/agg/13_backend_compare.py      # ~2 min      -> §7.5 backend agreement
```

Steps `03`, `05`, `06` and `13` were re-run from these banked scripts while writing this note and
reproduced their logged values; `logs/36k_score_loso_constant.log`, `logs/36k_seen_unseen.log` and
`logs/36k_backend_compare.log` are that fresh output.

`07_verify_ground_truth.py` loads the model from a **temp copy**, never the repo path, and asserts
the rebuilt query set and leaf routing match the saved artifact before comparing.

### 8.4 Production-scale run (DGX)

Serial `predict_density` at Stage-B is **~33.0 s/query** (measured: 131.84 s for batch of 4 =
32.96 s/query; repeat 132.27 s = 33.07 s/query), of which the KDE is ~99%. `08_ground_truth_prod.py`
therefore uses a fork `multiprocessing.Pool(16)` with the model loaded **once in the parent** so the
4.16 GB `training_leaves` is copy-on-write shared.

Two settings are load-bearing:

- `query_block=1` in `_leaf_match_weights`. The stock `query_block=64` peaks at **38.32 GB** —
  that block size is sized for the 36k model. At `query_block=1` worker peak RSS is 5.25 GB.
- `train_block=1024` in the KDE. This is the stock value and is already optimal: measured 32.68 s
  at 1024 versus 36.67 s at 65536, and the larger block also costs memory (8.33 GB → 14.72 GB).

`predict_density` exposes no `query_block` parameter, so the harness drives the two primitives
directly and re-derives the spread with the shipped definition.

Checkpointing is mandatory and is the thing an earlier attempt got wrong (it wrote nothing until
the end and lost everything at 48/480). The harness writes one JSON line per query with
`f.flush(); os.fsync(f.fileno())` inside `pool.imap_unordered`, so a kill loses at most one query,
and resumes by skipping completed indices.

Working dir on the DGX was `/tmp/aggprod`; banked filenames map to the run filenames as
`08_ground_truth_prod.py` → `prod_gt.py`, `09_assemble_prod.py` → `assemble.py`,
`10_collisions_prod.py` → `collisions.py`, `11_unseen_only_prod.py` → `unseen_only.py`, and
`03/04/05_*.py` → `prod_analyse{,2,3}.py`.

```bash
cd /tmp/aggprod
nohup ~/Development/silly-kicks/.venv/bin/python prod_gt.py > prod_gt.log 2>&1 &
# resumable: re-running skips completed indices, so a kill costs at most one query
python assemble.py        # JSONL + prep.npz -> out/gt.npz, prints the mechanism block
python prod_analyse.py    # = 03_score.py, path lines only differ
python prod_analyse2.py   # = 04
python prod_analyse3.py   # = 05
python collisions.py && python unseen_only.py   # 473-query unseen sensitivity arm
```

**Two reproducibility hazards found by smoke-testing the assembly step mid-run** (both now fixed in
the banked scripts):

- Under **pandas 3.x**, `Series.values.astype(str)` returns an **object** array, so `prep.npz`
  stores `src` as `dtype=object` and `np.load(..., allow_pickle=False)` raises
  `ValueError: Object arrays cannot be loaded when allow_pickle=False`. This would have surfaced
  only *after* the full ground-truth run finished. `09_assemble_prod.py` now reads `prep.npz` with
  `allow_pickle=True` and re-emits `src` as `U32`, so the scorers keep loading `gt.npz` with
  `allow_pickle=False`. The 36k leg does not hit this — pandas 2.3.3 yields a unicode array.
- The query order is **metrica → sportec → skillcorner** (sorted on the original pool index, and
  the fixtures are enumerated in that order). A truncated run therefore loses `skillcorner`
  *entirely* rather than degrading each source evenly, which would silently break the
  source-stratified bootstrap. Partial checkpoints must not be scored.

Observed throughput on the run banked here: **6.46–8.94 s/query wall** at 16 workers, settling
near 6.5 s/query, at ~11 GB system-wide for the whole pool. That is slower than the 16-worker
scoping probe's 6.99 s/query best case would suggest for the *whole* run, because this harness runs
the full `predict_density` **plus** a second leaf-match pass to recover neff, ksize and the true
weighted moments. Expected wall-clock for n = 480: **~55–70 min**.

The production scorers differ from the 36k scorers by path lines only — verified by diff:

```
$ diff harness/03_score.py prod_analyse.py
21c21
< REPO = pathlib.Path(r"D:/Development/karstenskyt__silly-kicks_part-deux")
---
> REPO = pathlib.Path(r"/home/karsten/Development/silly-kicks")
24c24
< AGG = D.parent / "agg"
---
> AGG = D / "out"
```

Two lines, both paths. The bootstrap design, the predictor definitions, the spread function and the
reporting are byte-identical between the two scales.

DGX environment: silly_kicks 4.48.0 at commit `97c74d5`, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1,
numba absent — i.e. an **older library version than shipped 4.53.0**, so the production
measurements only transfer if the hot path is unchanged.

Verified rather than assumed: the five hot-path functions were extracted by `ast`,
whitespace-normalised and SHA-256'd on both trees. All five hashes are **identical**.

| function | sha256[:16], both trees |
|---|---|
| `_leaf_match_weights` | `e43ccc75f083dc3a` |
| `_kde_setup` | `789a45d43a5edf16` |
| `_kde_density_vectorized` | `3f33f04e9b1bc3de` |
| `predict_density` | `6a2d4931a4e3c109` |
| `_vectorized_leaf_indices` | `ab842284f068918a` |

These five cover the whole path from a query row to a spread value at the default
`kde_backend="vectorized"`: `predict_density` calls `_vectorized_leaf_indices` (routing) →
`_leaf_match_weights` (weighting) → `_kde_density_vectorized` (which uses `_kde_setup` for the
bandwidth), and then computes the entropy and spread **inline at `:1745-1746`**, i.e. inside the
already-hashed `predict_density`. A version difference elsewhere in the module therefore cannot
affect these measurements.

### 8.5 Query provenance at the two scales

The 36k queries are verified **unseen** — zero exact 243-leaf-vector collisions with any training
row, in every source (`logs/36k_seen_unseen.log`):

```
[A] distinct training leaf vectors: 35892/36000
[A] queries whose FULL 243-leaf vector exactly matches a training row: 0/480
      metrica_slim       n=  71  exact collisions=0
      skillcorner_slim   n=  76  exact collisions=0
      sportec_slim       n= 333  exact collisions=0
```

At Stage-B this no longer holds exactly: **7 of 480** queries collide, all in `sportec_slim`
(7 of 333 = 2.1% of that source; metrica and skillcorner remain at zero), because the Stage-B
corpus overlaps that fixture.

```
[unseen] distinct training leaf vectors: 1005753/1039502
[unseen] queries whose FULL 500-leaf vector exactly matches a training row: 7/480  (1.0s)
[unseen]   metrica_slim         n=  71  exact collisions=0
[unseen]   skillcorner_slim     n=  76  exact collisions=0
[unseen]   sportec_slim         n= 333  exact collisions=7
```

`11_unseen_only_prod.py` re-scores the full predictor set on the 473 non-colliding queries as a
sensitivity arm, and §3.4 reports both. Separately, a leave-self-out control measured the effect of
excluding a query's own training twin at Stage-B as **2.377e-06 / 6.087e-06 / 7.322e-07** relative
on three random rows — five orders of magnitude below the ~0.2 pp effects under test, because at
Stage-B the weights are maximally diffuse.

Note the query-selection code is byte-identical across the two scales, and reproduces the same
per-source counts (`metrica_slim 71 / skillcorner_slim 76 / sportec_slim 333`), so the two
measurements are paired by construction.
