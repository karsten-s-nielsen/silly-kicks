# Action-Context Hot-Path Acceleration — Design

**Date:** 2026-06-01
**Status:** Design (brainstorm output, pre-plan) — re-scoped after lakehouse full-chain measurement
**Feature tag:** TF-45 (provisional)
**Supersedes:** the DAS-only framing of the earlier draft (DAS validated as ~1% of the AC-1
chain — retained as Appendix A, legitimate standalone work, not the AC-1 lever).

---

## 1. Problem & motivation

The original engagement asked: re-implement DAS for performance (it delegates to the
third-party `accessible-space`). Step-0 confirmed **DAS is 70–74% of `get_das`** — but a
follow-up **full-chain** lakehouse profile reframes the *pipeline* goal.

### Measurement 1 — DAS share of `get_das` (mine, local, validated): PASSED
`get_das` on real `sportec/realistic` tracking, 1×/4×/12×: `accessible_space` self-time
**70.4 → 74.1%**; dominant cost is memory-bandwidth-bound elementwise numpy over the
`F×P×V0×PHI×T` array (51%) + einsum (12%) + motion-models (16%). DAS dominates *the DAS
function*.

### Measurement 2 — full AC-1 chain (lakehouse, serverless): different denominator
skillcorner 2011166, silly-kicks 4.1.1, 60/210-batch, wall 1405 s, all ~20 stages:

| Stage | % chain wall | Hotspot |
|---|---|---|
| `add_ghost_gk` | **74.4%** | `_ghost_gk.predict_density → scipy.stats.gaussian_kde.evaluate` (534 calls, 931 s, ~1.74 s/call) |
| `add_elastic_sync` | 6.1% | `_build_player_ball_distance_lookup` |
| `add_cover_shadows` | 4.7% | |
| `add_obso` | 3.8% | |
| **DAS** (`add_das`+sim) | **~1.2%** | |
| (hidden) pandas scalar access | ~14% | `frame.__getitem__`+`_ixs`, ~2.1 M calls (per-row `.iloc`/`iterrows`) |

**Both are right, different denominators:** DAS = 70% of `get_das` ✓; DAS = ~1% of the
chain ✓. For *"make AC-1 faster"* the chain number governs — **Amdahl caps the entire DAS
GPU engine at ~1% whole-chain.** The lever is **ghost-GK's `gaussian_kde`**.

### Validation of the lakehouse claims (per "validate all claims")
Verified against the code, not taken on faith:
- **✓ ghost-GK KDE mechanism confirmed.** `_ghost_gk.predict_density` (`_ghost_gk.py:1146`)
  is a **per-sample Python loop**: per sample it rebuilds a fresh `scipy.stats.gaussian_kde`
  (`bw_method="scott"`, weighted) and evaluates it on the fixed 60×64=3840-point grid
  (`_GRID_X×_GRID_Y`) against the nonzero-weight training set. scipy `gaussian_kde` is
  pure-Python/single-threaded → `n_samples × n_train × 3840` Gaussians in Python. Per-call
  (not JIT warm-up) → the 74% headline is robust. The loop *also* does a per-sample
  `(n_train × n_trees)` leaf-match (`self._training_leaves == query_leaves[i]`, line 1148) —
  itself heavy on the 537k-sample "full" variant.
- **✓ Nuance the review missed — `predict_mean` cannot substitute.** Consumed features are
  `ghost_gk_x = mode_x`, `ghost_gk_y = mode_y`, `ghost_gk_spread = spread`
  (`_ghost_gk.py:1474-1476`) — **all density-derived**. `predict_mean` returns the regressor
  mean (no mode/spread) and is unavailable for shipped models (regressors discarded on
  `load()`). Bypassing the KDE = a **model change requiring lakehouse sign-off**, not a free
  swap. The KDE must be *accelerated*, not skipped.
- **✗ Correction — `_build_player_ball_distance_lookup` is not O(n×m).** Distance math is
  vectorized numpy (`_elastic_sync.py:169-176`); the slow part is the **per-row
  `merged.iloc[i]["col"]` dict-build loop** (lines 178-187) — almost certainly the same root
  cause as the separately-flagged ~14% pandas scalar access. O(n) with pathological per-row
  pandas access; fixed by vectorizing the dict build. Cheap, no new deps.

**Caveats (not final):** one provider (skillcorner); IDSSE 25fps profile pending; the 74%
assumes ghost-GK runs in the chain with the slow-path bundled model.

---

## 2. Strategy — cheap wins first, GPU target gated on re-measurement

Decision: **do the high-ROI, low-risk vectorization wins first**, re-measure, *then* decide
whether the residual hot path justifies the numba/GPU multi-backend engine. This applies the
same "measure before building" discipline Step-0 used.

- **Phase 0 (this work):** pure-numpy / pandas vectorization — no new deps, behaviour-
  preserving, golden-mastered. Re-measure.
- **Phase 1+ (gated, deferred):** the multi-backend GPU playbook (§4), retargeted at whatever
  still dominates after Phase 0 (most likely the ghost-GK KDE), pending the §3.4 gate.
- **DAS:** Appendix A — validated ~1% of AC-1; deprioritized for the pipeline goal; the
  multi-backend DAS design is preserved there for direct-DAS consumers if separately
  prioritized.

---

## 3. Phase 0 — cheap wins (concrete scope)

### 3.1 Phase 0a — vectorize `_ghost_gk.predict_density` (no new deps; scipy.linalg primitives)

**Kernel shape — per-sample-over-leaf-subset (ragged), NOT one fused tensor (was P1-1).**
Each sample's KDE is built over its own leaf-matched training subset (`nonzero = weights > 0`),
so subsets are **different sizes per sample** — there is no rectangular
`(n_samples, n_train, 3840)` tensor. The design keeps a per-sample step but replaces the
`scipy.stats.gaussian_kde` *object* (whose per-call covariance + `cho_factor` + construction +
Python dispatch is most of the 1.74 s/call) with a tight kernel evaluating the weighted
Gaussian sum on the fixed 3840-point grid over that sample's subset. Verified against scipy
1.15.3 `_kde.py`, the kernel **reuses scipy's exact linear algebra** (so 1e-9 is reachable):

- weighted **Scott** bandwidth: `neff = 1/Σwᵢ²` (weights normalized), `factor = neff^(-1/(d+4))`,
  `d=2` (`_kde.py:213,493`);
- weighted covariance `np.cov(dataset, bias=False, aweights=w)` → `scipy.linalg.cholesky`;
  `covariance = data_cov·factor²`; `cho_cov = data_cho_cov·factor`;
  `log_det = 2·Σ log(diag(cho_cov))` (`_kde.py:584-591`). **Cholesky path, NOT `inv()`**
  (was P1-2 — a direct inverse is a numerically different algorithm and breaks the 1e-9 gate);
- whitening via `scipy.linalg.cho_factor`/`cho_solve` on the grid–data diffs (`_kde.py:317-322`),
  `energy = ½·Σ(diff·tdiff)`, `density = Σⱼ wⱼ·exp(-energy)`, reshape `(GRID_NX, GRID_NY)`,
  normalize to sum 1;
- preserve the `< 2` nonzero-weight and `LinAlgError` → uniform-grid fallbacks (lines 1156, 1172);
- mode (argmax), mean (`Σ probs·grid`), **spread = `exp(entropy)·resolution²` with entropy over
  `probs[probs>0]` only** (replicate the zero-prob mask exactly — no `log(0)`, was P3-10).

**Memory model — chunk over the TRAIN SET + leaf-match, with a budget (was P1-1).** For the
537k "full" variant, one sample's `leaf_subset × 3840` can be large and the leaf-match is
`537k × n_trees` per sample — chunking *over samples* does nothing for either. Accumulate
`Σⱼ wⱼ·exp(...)` in **train-set blocks** and stream the leaf-match in blocks, with a **stated
peak-memory budget compatible with the serverless 1 GB `applyInPandas` cap**. This is the #1
feasibility risk and the plan sizes it explicitly: **default `train_block=1024`** (≈150 MB
transient/block at grid m=3840), conservative under the 1 GB cap given the lakehouse's
`_FRAME_BATCH_SIZE=250` batching. **The serverless venue runs the 9 MB "default" model**
(small per-sample leaf subsets — chunking rarely binds), so the 537k-"full" pressure is an
in-memory/local concern, not serverless; the lakehouse owns the explicit UDF-memory
verification handoff (not "confirmed in passing"). Memory is guarded **structurally** (a
`cho_solve`-per-block call-count spy), not via flaky tracemalloc/RSS.

**Leaf-match:** vectorize `matches.sum(axis=1)/n_trees` (block-streamed) instead of the Python
loop; unit-tested in isolation vs the loop (§3.3, was P3-9).

**Keep the scipy path as a selectable `_reference` (was P2-4).** Do not delete the
`scipy.stats.gaussian_kde` implementation — retain it as a selectable reference (mirroring
`accessible-space` for DAS) so it can serve as a runtime oracle for the model-traveling parity
test (§3.3). `compute_ghost_gk` / `add_ghost_gk` / serialization / public surface unchanged.

### 3.2 Phase 0b — de-`iloc` per-row dict loops (profile-attributed, behaviour-preserving)
- **Attribute the ~14% `_ixs`/`__getitem__` first (was P2-8).** §1's "almost certainly the same
  root cause as elastic-sync" is an *assumption*. Use the `.pstats` caller info to attribute the
  ~2.1 M `_ixs` calls to **specific call sites** before touching anything.
- `_elastic_sync._build_player_ball_distance_lookup`: replace the `for i … merged.iloc[i]`
  dict build with vectorized key/value construction (`dict(zip(...))` over numpy columns),
  **preserving exact key tuple dtypes** (`int(frame_id)`, `str(player_id)`).
- **Then** vectorize only the **profile-attributed** sibling `.iloc[i]`/`iterrows` sites — not a
  speculative grep sweep (vectorizing a cold `.iloc` adds regression risk for no measured
  benefit — YAGNI). Each de-`iloc` is behaviour-preserving and gets its **own** golden check on
  its output (lookup values / column outputs).

### 3.3 Phase 0 correctness gate (golden master) — continuous vs discrete split

**Split the gate by output type (was P1-3).** `ghost_gk_x/y` are `argmax` over the 3840-cell
grid — a 1e-12 perturbation can flip the argmax to an adjacent cell, jumping the output by a
full grid resolution (a large *absolute* change, not small rtol). So `rtol` alone on `mode_*`
either flakes or hides a one-cell drift.
- **Continuous outputs** (`GhostGkDensity.probabilities` grid, `mean_*`, `spread`): the real
  fidelity proof — `np.testing.assert_allclose(rtol, atol, equal_nan=True)` with a **stated
  atol floor** (density/spread → 0 in degenerate frames) + a **separate NaN-mask equality
  assertion**. Realistic **~1e-9 in f64** (kernel reuses scipy's Cholesky LAPACK calls, §3.1;
  residual reduction-order absorbed by atol).
- **Discrete outputs** (`mode_x/mode_y` = `ghost_gk_x/y`): assert the **argmax index matches
  exactly**, and **quantify near-tie flakiness on real frames** (count cells within ε of the
  max); if non-trivial, document a **±1-cell tolerance**. Never gate `mode_*` on `rtol`.

**Model-traveling parity test (was P2-4).** Beyond frozen fixtures (which only validate
today's bundled model), add a test that — given **whatever** model is loaded — asserts the
vectorized kernel ≈ the retained `scipy.stats.gaussian_kde` `_reference` path on a few samples.
This auto-revalidates on retrain (new leaves/train set) and keeps scipy as a live oracle.

**Fixtures + CI coverage.**
- Use **both** bundled variants AND a **downsampled-full / synthetic large-train fixture** so
  the train-set chunking + memory path (the full-variant risk, §3.1) is **exercised in CI**, not
  just the 9 MB default (was P2-5).
- Fixture-size discipline: scalars full; grids downsampled/`.npz`/hash (silly-kicks ships a
  ~103 MB sdist).
- **Isolation unit test** for the vectorized leaf-match vs the loop (was P3-9).
- Reuse/extend existing ghost-GK + elastic-sync tests; parametrize over variant.

### 3.4 Phase 0 measurement, perf guard, and decision gate
- **Local** (mirrors Step-0): load the in-repo `"default"` ghost-GK variant, build GK features
  from a committed fixture, benchmark `predict_density` before/after (and the elastic-sync
  lookup). Capture with **no contending processes** (the `calibrate_*` sweep shares this box),
  **persist before cleanup**.
- **Perf-regression guard — structural, not wall-clock (refines P2-7).** The reviewer asks for
  pytest-benchmark; silly-kicks has a *learned* lesson that wall-clock CI budgets are flaky
  (a 500 ms ceiling failed CI at 501 ms — see `feedback_windows_ci_perf_budget`). Reconcile:
  the **hard CI gate is a structural guard** — assert the vectorized path constructs **zero
  `scipy.stats.gaussian_kde` objects** per batch (spy the construction) / bounds the per-sample
  allocation — which protects the win without wall-clock flakiness. Add a **pytest-benchmark as
  informational tracking** (saved-baseline comparison in a nightly/local job), **not** a hard
  shared-CI fail threshold.
- **Consumer correctness e2e (was P2-6) — the true Phase-0 acceptance step.** Because the
  discrete argmax (§3.3) may not be byte-identical on the consumed columns, the lakehouse runs
  AC-1 on one real match with the **vectorized** ghost-GK vs the **scipy `_reference`** path and
  diffs `fct_action_context.ghost_gk_x/y/spread` within the agreed mode tolerance — a
  *correctness* diff, distinct from the perf re-profile. Named here as the consumer acceptance
  step.
- **Lakehouse** re-profiles the full AC-1 chain (incl. the pending IDSSE 25fps delta).
- **DECISION GATE:** after Phase 0, which stage still dominates? If ghost-GK KDE remains the
  bottleneck and pure-numpy didn't close it → it justifies the numba/GPU multi-backend engine
  (§4) targeted at the KDE. If Phase 0 collapsed it → stop; no GPU engine needed.

#### Phase-0 outcome — measured 2026-06-01 (local, RTX-box, `.venv` py3.10/scipy1.15/numpy2.2)

**Vectorized KDE is correct and shipped, but the pure-numpy speedup is modest (~1.18×).**
Bundled `"default"` model, warm, 16 samples, back-to-back:

| backend | ms / sample |
|---|---|
| `scipy` (reference) | 4541 |
| `vectorized` (new default) | 3832 → **1.18×** |

**Root cause (cProfile):** 560 `cho_solve` calls = 16 samples × ~35 train-blocks → **each
sample's nonzero leaf-subset is ≈36k points** (with 500 trees nearly every training point
co-occurs in some query leaf). So the KDE is **eval-bound on ~36k-point subsets**, not
per-call-overhead-bound — vectorization only recovers scipy's object-construction overhead.
The work (36k × 3840-grid × N samples) is embarrassingly parallel → **a material win needs the
GPU engine (§4), not more CPU vectorization.** The Task-6 leaf-match vectorization is a clean
structural win (no per-sample Python loop) but small in absolute terms here.

**Gate decision:** pure-numpy did **not** collapse the ghost-GK cost → **if ghost-GK remains
the AC-1 bottleneck on the lakehouse's serverless re-profile, the GPU multi-backend engine (§4)
is justified, targeted at the KDE.** Two caveats before committing to §4: (1) the local
`"default"`-model call pattern (~36k-point subsets) may differ from the lakehouse's serverless
pattern — many small-subset calls would be more overhead-bound (bigger CPU win); the lakehouse
re-profile settles this; (2) a model-side lever (truncating the nonzero-weight subset / a weight
threshold) would cut the ~36k → big CPU win, but it **changes ghost_gk values** → a model change
requiring lakehouse sign-off, explicitly out of Phase-0 scope.

**Shipped regardless of the gate** (correctness + hygiene, independent of the speedup):
vectorized KDE (golden-master-faithful, scipy kept as `_reference`), Task-6 leaf-match
vectorization, the elastic-sync de-`iloc` (Task 10), and the full **0c DAS offside bundle**
(carrier forwarding — A/B + unit-test value-neutral, one-time warning UX, dead-ball guard).

**Deferred (YAGNI, evidence-gated):** Task 11 broader `.iloc`/`iterrows` sweep
(`_elastic_sync`/`_cover_shadows`/`_ghost_gk` have 5 `iterrows` loops) — the committed 10-action
fixture did **not** reproduce the lakehouse's ~14% / 2.1M-`_ixs` pathology (local `_ixs` traces
to pandas-internal column caching), so vectorizing them would be a speculative,
regression-risky refactor with no locally-measured benefit. **Attribute on the serverless-scale
profile first**, then vectorize only the attributed sites.

---

### 3.5 Phase 0c — DAS offside hygiene + dead-ball guard (immediate win)

`get_das`/`get_individual_das` never forward `player_in_possession_col` to
`accessible-space`, so with the DAS default `respect_offside=True` the library warns **once
per call** ("`player_in_possession_col` should be set …") — floods serverless logs — and may
mis-flag the carrier as offside. silly-kicks already computes the carrier
(`infer_ball_carrier` → `ball_carrier_player_id`) but `derive_team_in_possession` currently
**drops it** (`_ball_carrier.py:458`). **A/B (sportec realistic, 2369 possession-frames):
forwarding the carrier removes the warning and changes AS/DAS by exactly zero**
(`max_abs=0`, Spearman 1.0, NaN-mask identical) — so this is effectively pure log-noise, and
forwarding is both safe and more correct per accessible-space's contract.

Best-practice design (long-term, multi-consumer): *the data DAS needs travels with the
frames, produced once at its source; consumption is explicit and discoverable; the correct
path is the default* — **not** magic column-sniffing, **not** pure per-consumer opt-in.

1. **Source:** `derive_team_in_possession` preserves `ball_carrier_player_id` (it already
   receives it on the `carrier` df). Purely **additive** column; possession's two facets
   (team + player) now both travel with the frames for every downstream consumer.
2. **Consumption:** `get_das`/`get_individual_das` gain
   `player_in_possession_col: str | None = "ball_carrier_player_id"` (explicit, documented).
   Forward when the named column is present; **loud error** if a caller explicitly names a
   missing column; default-name-absent (old frames) degrades to current behavior.
3. **silly-kicks owns the warning UX:** suppress accessible-space's per-call offside warning;
   when no carrier column is available, emit silly-kicks' own **one-time** message
   (`stacklevel=2`, module-level flag) instead of per-call flooding.
4. **Dead-ball-subset guard:** when the `links`-restricted frame subset is all dead-ball
   (no non-NaN `team_in_possession`) the *generic* library `ValueError` bubbles instead of
   silly-kicks' clearer `_pin_attacking_direction` dead-ball message (the `_pin` guard runs on
   the full batch, which has alive frames, so the restricted-subset case escapes it). Guard the
   **restricted subset** directly in `_precompute_das_lookup` with the clear message
   (consumers still `@nan_safe_enrichment`-degrade — honest dead-ball, e.g. IDSSE ~33%
   dead-frames, is not an error condition).
5. **Golden master proves value-neutrality** on committed fixtures (A/B = zero change); any
   fixture that moves = a documented correctness improvement, announced, never silent.

This is DAS work but a genuine immediate win (independent of the deferred §4 engine); the
carrier-offside contract is carried into Appendix A.

## 4. Phase 1+ (deferred, gated) — multi-backend GPU engine playbook

> **Status update (2026-06-01):** Phase 0 shipped in silly-kicks **4.2.0**; the lakehouse A/B
> confirmed ghost-GK is still **91%** of the AC-1 chain after vectorization. Lakehouse review then
> reframed Phase 1: AC-1 runs in **CPU-only serverless `applyInPandas`** (no GPU venue) and touches
> only 84 tracking matches → **Phase 1 = numba-first (in-venue)**, GPU deferred until numba's gain is
> measured. See `docs/superpowers/specs/2026-06-01-ghost-gk-kde-numba-acceleration-design.md`
> (→ ADR-013). This GPU playbook below stays the reference for the deferred GPU track.

Reusable for whichever path wins §3.4 (likely the ghost-GK KDE; the eval is an embarrassingly-
parallel weighted Gaussian sum over samples × grid × train — same GPU profile as DAS). The
discipline is unchanged from the reviewed DAS design and applied to the target kernel:

- **Backend selector** `backend="cpu-numpy" | "gpu-jax" | "gpu-cupy" | "cpu-numba"`; canonical
  `backend=` kwarg (no `engine=` alias). Default `cpu-numpy`.
- **One array-module-parametric kernel** (numpy / jax.numpy+jit / cupy) written in **JAX
  dialect** (no in-place mutation, static shapes, explicit per-backend dtypes; NEP-50 caveat);
  `cpu-numba` only if it beats both `cpu-numpy` **and** jax-CPU by **≥1.5×** (else dropped,
  recording the number).
- **Sequencing:** cpu-numpy reference + golden master → first GPU backend (`gpu-cupy`, native
  Windows on the local **RTX 5070 Ti**, Blackwell sm_120 feasibility checked early) → **GPU
  go/no-go gate** → `gpu-jax` (datacenter artifact; lakehouse JAX precedent; WSL2/HF-Jobs) →
  conditional `cpu-numba` → benchmark report.
- **Tolerance:** f64 correctness gate (`rtol≈1e-7..1e-9` + atol + NaN-mask) vs the float64
  reference; **f32 production acceptance** by discrimination (Spearman ≥0.999 + rel-err
  ≤~1e-3) — **threshold ratified by the lakehouse** (downstream owner).
- **Deps:** new optional extras `[gpu-jax]`/`[gpu-cupy]`, lazy-guarded; **import-leak CI test**
  proving bare `import silly_kicks` (and the target subpackage) pulls no jax/cupy/numba.
- **CI:** CPU backends always in CI; GPU backends in a nightly/manual GPU job or HF Jobs L40S.
- **Constants/version discipline:** if accelerating ghost-GK, snapshot nothing third-party
  (it's our own model); if DAS, snapshot `accessible-space` 2.1.0 constants + a drift canary.
- **Default/dep end-state:** decide-after-prototype; flip default in a separate **announced** PR
  with lakehouse f32 sign-off + consumer e2e (AC-1 `backend=native` vs status-quo diff on the
  consumed columns); **keep the reference path ≥1 release after the flip** (kill-switch).
- **Benchmark hygiene:** no contention; persist-before-cleanup; **5070 Ti 16 GB memory/chunk
  numbers do NOT transfer to the 48 GB L40S** — measure prod memory on the prod GPU.

---

## 5. Contract preservation (Hyrum's Law) — all phases
- Public surfaces unchanged: `add_ghost_gk` / `compute_ghost_gk` / `ghost_gk_xfns` →
  `ghost_gk_x/y/spread`; `add_elastic_sync` / `elastic_sync_xfns`; `add_das` family →
  `AS/DAS/das_*`. No column renames, no value-scale shift beyond the announced f64 flip (§4).
- `links=` / `chunk_size=` / NaN-degradation (`@nan_safe_enrichment`, ADR-003) preserved.
- Phase 0 is behaviour-preserving (golden-mastered); any value change is a regression, not a
  feature.

---

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Vectorized KDE diverges from scipy at 1e-9 | Reuse scipy's **Cholesky** path (`cho_factor`/`cho_solve`, `log_det=2Σlog diag`), **not `inv()`** (§3.1); golden master f64 + atol + NaN-mask; uniform/LinAlgError fallbacks preserved. |
| Full-variant memory blow-up (537k train) | Per-sample-over-ragged-leaf-subset (no fused tensor); chunk over the **train set + leaf-match** (not samples) with a stated peak-memory budget ≤ serverless 1 GB UDF cap (§3.1); exercised in CI via a large-train fixture (§3.3). |
| Discrete `mode_*` (argmax) flips a cell under tiny perturbation | Gate the continuous density grid on rtol/atol; gate `mode_*` on exact argmax index or ±1-cell, with near-tie flakiness quantified (§3.3). |
| Phase 0 insufficient → GPU still needed | §3.4 gate decides; §4 playbook ready. |
| Local ghost-GK bench unrepresentative | Use in-repo `"default"` variant + real fixture; lakehouse confirms on serverless + IDSSE. |
| `predict_mean` "fix" silently changes the model | Rejected — mode/spread are consumed; KDE accelerated, not bypassed (§1 validation). |
| Benchmark contention / non-transferable GPU numbers | No-contention capture, persist-before-cleanup, prod-GPU memory (§3.4/§4). |
| De-`iloc` changes key dtypes | Preserve exact key tuple types (`int(frame_id)`, `str(player_id)`) — golden-master the lookup values. |

---

## 7. Review responses — lakehouse round 2 (2026-06-01)
| Claim | Resolution |
|---|---|
| ghost-GK KDE = 74% chain, DAS ~1% | Validated (mechanism confirmed); engagement re-scoped (§1, §2). |
| Point the playbook at ghost-GK | §3 Phase 0 cheap wins + §4 retargetable GPU playbook. |
| `predict_mean` alternative | **Pushed back** — consumed features (mode/spread) need density; can't substitute (§1). |
| elastic-sync O(n×m) | **Corrected** — O(n) with slow per-row `.iloc`; Phase 0b (§3.2). |
| ~14% pandas scalar access | Same root cause as elastic-sync; Phase 0b greps + vectorizes siblings. |
| golden-master ghost-GK at f64 / atol / NaN-mask | §3.3. |
| IDSSE delta pending / one provider | §1 caveats; §3.4 lakehouse re-measure. |

### Round 3 (2026-06-01)
| # | Item | Resolution |
|---|---|---|
| P1-1 | Ragged memory model + full-variant blow-up | §3.1: per-sample-over-leaf-subset (not fused tensor); chunk over train-set + leaf-match with a ≤1 GB budget. |
| P1-2 | Cholesky, not inverse, for 1e-9 | §3.1: reuse scipy's `cho_factor`/`cho_solve` + `log_det=2Σlog diag` (verified vs `_kde.py` 1.15.3). |
| P1-3 | Discrete argmax vs rtol | §3.3: continuous grid on rtol/atol; `mode_*` on exact-index/±1-cell + near-tie quantification. |
| P2-4 | Model-traveling scipy-parity test; keep scipy `_reference` | §3.1 retains scipy path; §3.3 parity test on whatever model is loaded. |
| P2-5 | Full variant in CI | §3.3 downsampled-full / synthetic large-train fixture exercises chunking. |
| P2-6 | Consumer correctness e2e (not just re-profile) | §3.4 lakehouse AC-1 vectorized-vs-scipy diff on consumed columns = Phase-0 acceptance. |
| P2-7 | pytest-benchmark guard | §3.4 **refined** — structural no-scipy-call guard as the hard CI gate (wall-clock CI is flaky here); pytest-benchmark informational/nightly. |
| P2-8 | Confirm ~14% attribution; bound de-iloc sweep | §3.2: `.pstats` call-site attribution first; vectorize only attributed sites; each golden-checked. |
| P3-9 | Leaf-match isolation unit test | §3.3. |
| P3-10 | Entropy/spread zero-prob masking | §3.1 spread bullet (mask `probs>0`, no `log(0)`). |
| P3-11 | `predict_mean` push-back as durable note | §1 + Appendix B (consequence note). |

### Round 4 (2026-06-01) — DAS offside warning + dead-ball
| Item | Resolution |
|---|---|
| `get_das` doesn't forward carrier → per-call offside warning + possible mis-mask | §3.5 Phase 0c: source-preserve `ball_carrier_player_id` + explicit `player_in_possession_col` default-forwarded + own the warning UX. A/B validated zero AS/DAS change. |
| Restricted-subset all-dead-ball → generic library `ValueError` | §3.5(4): guard the restricted subset in `_precompute_das_lookup` with silly-kicks' clear message. |
| Carrier-offside contract in native engine | Appendix A explicit contract. |

### Round 5 (2026-06-01) — plan review (Phase-0 plan)
| Item | Resolution (in the plan unless noted) |
|---|---|
| Degenerate fallbacks untested (`<2` uniform; singular-cov → `LinAlgError`) | Plan Task 3/4: explicit kernel singular-cov parity + predict_density degenerate vec==scipy (NaN-mask + values). |
| `peak_memory` test never measured memory | Plan Task 5: replaced with a **structural `cho_solve`-per-block call-count spy** (numpy buffers under-report in tracemalloc; RSS flaky) — spec §3.1. |
| Serverless `train_block` budget undecided | Default `train_block=1024`; serverless runs the 9 MB "default" model; explicit lakehouse UDF-memory handoff — spec §3.1, plan Task 12. |
| Kernel-1e-9 vs golden-1e-7 unexplained | Plan Task 2: documented (renorm + mode/spread amplify error ~2 orders; don't tighten). |
| "Adjust norm until parity" | Plan Task 3: `norm` derived from scipy's normalization (stated); `log_det` dead-code simplified. |
| Task 10 key dtypes / oracle | Plan Task 10: independent-oracle dict-equality + all-four-element dtype contract; key build matches loop types. |
| Golden regen on scipy/numpy bump | Plan Task 2 maintenance note. |
| Dead-ball cross-repo order | Plan Task 0c.4: comment on `_fill_possession_from_set_piece_actions` complementarity. |
| GS Int64-vs-object schema bug | **Out of Phase 0** — validated real (`schema.py:44-48`); tracked separately (Chesterton's Fence: GS `Int64` is a deliberate PR-S18 convention). |

---

## Appendix B — Durable consequence: why `predict_mean` cannot replace the KDE
`ghost_gk_x/y/spread` are `mode_x/mode_y/spread` of the **density surface**
(`_ghost_gk.py:1474-1476`). `predict_mean` returns the regressor **mean** (no mode, no spread)
and is unavailable for `load()`-ed models (regressors discarded). Swapping it in is a **model
change** (different consumed semantics) requiring lakehouse sign-off — not a performance fix.
The KDE is **accelerated, not bypassed**. (Recorded so this is not re-proposed next cycle.)

---

## Appendix A — DAS multi-backend engine (deprioritized, validated ~1% of AC-1)
The DAS design (array-module kernel + cpu-numpy/cpu-numba/gpu-jax/gpu-cupy, dual-mode
tolerance, accessible-space oracle + constants-drift canary, default-flip policy) remains
valid **standalone** work for direct-DAS consumers, but is **not** the AC-1 lever (Amdahl
~1%). The §4 reusable playbook captures the full multi-backend discipline; applied to DAS it
additionally needs the `accessible-space` float64 oracle, the per-frame `attack_poss_density`/
`player_poss_density` grid golden master, the linked-frame direction-pin bit-identicality
contract, and the `[0,105]×[0,68]` coordinate boundary.

**Carrier-offside contract (explicit, from Phase 0c §3.5).** The native DAS engine MUST accept
and honour the ball-carrier / passer identity (`player_in_possession_col`) when
`respect_offside` is on: the passer is excluded from the offside mask. The carrier travels with
the frames as `ball_carrier_player_id` (produced by `derive_team_in_possession`). The engine
forwards it the same way the adapter does; the offside masking must not silently mis-flag the
carrier. This coupling is a first-class part of the engine's contract, not an adapter detail.
If direct-DAS performance is separately prioritized, lift §4 onto the DAS kernel
(`get_individual_das` per-player path is the heavy one, 1.21× team-level).
