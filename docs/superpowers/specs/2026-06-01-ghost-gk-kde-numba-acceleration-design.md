# Ghost-GK KDE Acceleration (Phase 1) — closed-form numpy, then conditional numba

**Date:** 2026-06-01
**Status:** Implemented — silly-kicks 4.3.0 (closed-form default + `cpu-numba` backend)
**Feature tag:** TF-45 Phase 1 (provisional) — in-venue ghost-GK KDE acceleration
**Decision record:** ratified as **ADR-013** (extends ADR-008 numba precedent, ADR-012)
**Builds on:** Phase 0 — `2026-06-01-action-context-hotpath-acceleration-design.md` (shipped silly-kicks 4.2.0).
**Replaces:** the earlier GPU-first draft (withdrawn: no in-pipeline GPU venue + thin ROI). GPU deferred to §8.

---

## 1. Problem & the (refined) lever ladder

Phase 0 (4.2.0) vectorized `predict_density`'s scipy KDE into a numpy Cholesky kernel
(`_ghost_gk._kde_density_vectorized`): whole-chain **1.24×**, ghost-GK **still 91%** of the AC-1
chain, correct, offside flood gone. AC-1 runs in **Databricks serverless `applyInPandas` (CPU-only,
1 GB UDF cap, no GPU)** and touches only **84 tracking matches**.

Lakehouse kernel profile: `_kde_density_vectorized` ≈ 803 s = `cho_solve` ≈ **194 s (~24%)** +
`exp`/weighted-reduction/`(2,kb,m)` temporaries ≈ **609 s**. The 2×2 whitening is **closed-form**
(`H` is 2×2), and **eliminating `cho_solve` is a numpy optimization — not numba-specific.** So the
ladder is finer than "numpy → numba → GPU":

1. **numpy + closed-form 2×2** (kills `cho_solve`, drops the `diff`/`tdiff` temporaries) — *one kernel,
   no new dep, no threading, no new CI surface.* **Re-measure.**
2. **numba** — *only if* step 1 leaves a single-thread in-venue gap worth a second hand-maintained
   kernel (gated, §6/§9).
3. **GPU** — deferred (§8); only if numba is insufficient *and* volume + a venue re-arch warrant it.

**Each rung is gated by measurement with the same fail-fast discipline that withdrew the GPU draft.**
Don't build rung N+1 until rung N's number says it's needed.

---

## 2. Goal & non-goals

### Goal
Cut ghost-GK KDE wall time in the **existing CPU serverless venue**, cheapest-rung-first: ship the
numpy closed-form win; build a `cpu-numba` backend **only if** it clears a stated bar over that.
No public-API change; value changes governed (§5).

### Non-goals
- GPU (jax/cupy) — deferred (§8). DAS (~1% of chain). Changing the KDE model.
- Building numba "by default" — it must earn a second kernel by measurement (§6).

---

## 3. Step 1 — numpy closed-form 2×2 (do first, no new dep)

In the existing `_kde_density_vectorized`, **keep `cho_factor` for the PD-branch decision and
`log_det`** (a 2×2 factorization is ~free), and replace only the expensive `cho_solve`-over-`(2,kb·m)`
whitening with the closed-form 2×2 Mahalanobis energy:

```
det = h11*h22 - h12*h12                                     # H = covariance (2x2)
energy = 0.5/det * (h22*dx^2 - 2*h12*dx*dy + h11*dy^2)      # dx = grid_x - data_x, dy = grid_y - data_y
```

(Equivalent to `0.5·diffᵀ·cho_solve(H, diff)` — lakehouse-verified.) This removes `cho_solve` (~24%)
and the `(2,kb,m)` stacked `diff`/`tdiff` temporaries, computing `energy` directly from `dx,dy`.

**Fallback boundary stays anchored to 4.2.0 (lakehouse §1 — this is a model boundary, not a numerical
knob).** Cholesky succeeds for **any** positive-definite covariance (even ill-conditioned), failing
only on non-PD; 4.2.0 therefore computes a finite density for near-singular-but-PD samples. Because
step 1 **keeps `cho_factor`** as the branch test, the **exact same samples** take the uniform fallback
as in 4.2.0 — no `τ_singular` widening, no extra uniforms, no model change. **Do NOT raise the fallback
threshold to "protect 1/det precision"** — that would move the modeling boundary.

**Near-singular-but-PD precision** (where `1/det` amplifies rounding as `det→0`) is a *parity* concern,
not a branch concern: the §5 near-singular parity test guards it, and if the closed-form diverges from
the golden there, the fix is to **compute that zone more stably** (e.g. via the retained Cholesky
factor), **not** to move the branch.

**Honest projection (lakehouse §4):** pure numpy still materializes several `(kb,m)` temporaries
(`dx²`, `dx·dy`, `dy²`, `exp`), so step 1 ≈ the clean ~24% `cho_solve` removal + reduced memory traffic
≈ **~1.3–1.4× on the kernel**, not full fusion. Full `(kb,m)`-temporary elimination needs numba fusion
(rung 2's justification). No `numexpr`.

**Gating:** golden-gated (§4) + **governed value change** (§5) — differs from 4.2.0 only in the energy
float-ops (~1e-12..1e-9, well inside golden `rtol≈1e-7`). **Re-run the A/B (`scripts/profile_ac1_*`,
single-thread) and record the new baseline** — what numba (step 2) must beat. Ship on its own merits
regardless of step 2.

---

## 4. Step 2 — `cpu-numba` backend (conditional; gated, serial)

Build **only if** §6's number says it adds ≥~1.5× over the step-1 numpy single-thread in-venue
baseline. If so:

- Add `_ghost_gk._kde_density_numba` — a **serial `@njit`** mirror that fully fuses the per-sample loop
  (scalar `exp` + weighted accumulate over the subset × grid; closed-form 2×2 energy as in §3; no
  `(2,kb,m)` temporaries). `cache=_NUMBA_CACHE` (on-disk cache OFF — serverless read-only paths, per
  the 4.1.1 fix).
- numba can't call `scipy.linalg`, so it computes everything closed-form: `log_det = log(det)`,
  `det = h11·h22 − h12²`. It must **replicate `cho_factor`'s PD-branch** (§3) via `det`/`h11`
  thresholds calibrated to Cholesky's effective singularity boundary — so the same samples fall back to
  uniform as 4.2.0 (no model change). The §5 near-singular parity case is the guard.
- **Ship the SERIAL kernel — NOT `@njit(parallel=True)` neutered by `NUMBA_NUM_THREADS=1`** (that's the
  worst case: parallel codegen overhead, zero parallelism). In `applyInPandas`, Spark already saturates
  cores across `(period, frame_batch_id)` groups; an in-group `prange` would oversubscribe. numba's only
  in-venue levers are **fusion + cho_solve-elimination**, not parallelism. A local-only parallel variant,
  if explored, is kept separate and non-shipping.
- **Thread-knob contract (lakehouse §6):** `os.environ.setdefault("NUMBA_NUM_THREADS","1")` in the
  driver bootstrap **and both UDF closures** (set before numba import/first-compile, mirroring the
  `NUMBA_CACHE_DIR` PR). Belt-and-suspenders for a serial kernel; the lakehouse owns wiring it.
- **Thread `backend=` through** `add_ghost_gk` → `compute_ghost_gk` → `predict_density`:
  `"scipy"` (reference) · `"cpu-numpy"` (default; alias `"vectorized"`) · `"cpu-numba"`.
- **Early kill-gate (NET-OF-COMPILE, in-venue — lakehouse §2):** prototype the `@njit` kernel + bench
  single-thread vs step-1 numpy **before** building the full backend/parity/CI/threading. The decision
  number is measured **as production sees it**: `cache=OFF` → **every executor process recompiles** the
  kernel (seconds of JIT/process). If `applyInPandas` spins many short-lived tasks with few groups per
  executor, the recompile tax can **swamp** the steady-state kernel win → numba can *lose* in-venue
  despite a faster kernel (this is the original "numba ~0% whole-job" mechanism). **Ship numba only if
  it clears ≥~1.5× AFTER compile amortization, in-venue.** Below the bar → fail fast, ship step 1 only.

---

## 5. Correctness contract

- **Reuse Phase-0 assets:** scipy `_reference` + golden master + model-traveling parity — parametrized
  over `cpu-numba` (and re-validated for the step-1 numpy change).
- **Golden-anchor discipline (lakehouse §3):** the golden stays anchored to the **scipy-validated
  Cholesky values** — **do NOT regenerate it** to the closed-form or numba output (that would launder
  the change being validated). Both rungs prove equivalence to the *existing* golden within tolerance;
  step-1's ~1e-12..1e-9 drift is well inside `rtol≈1e-7`, so the existing golden passing as-is **is**
  the validation.
- **Decomposed tolerance:** Leg A (done) cpu-numpy vs scipy; **Leg B (tight): `cpu-numba`-f64 vs the
  step-1 `cpu-numpy`-f64** (same algorithm/closed-form; compiled vs interpreted) — tolerance from the
  2×2 condition number, not a blanket scipy comparison. f64 only (no f32 — that's GPU, deferred).
- **Parity test cases parametrized over every backend, including the near-singular gap (lakehouse §4):**
  clean **+ near-singular / ill-conditioned covariance** (the case most likely to break closed-form
  parity) + fully-singular→uniform + `<2`-weight→uniform.
- **`mode_*` tie-stability (lakehouse §7):** the **density-field `allclose` is the primary check**;
  `mode_x/y` (argmax) is derived — gate the mode assertion only where the **top-2 grid cells are
  separated by > ε** (genuinely unambiguous), else a different summation order can flip the argmax by
  >1 cell and flake.
- **cpu-numpy perf-regression guard:** the step-1 change (and any shared refactor) must not erode the
  shipped 4.2.0 1.24× — re-run the A/B, fail if it regresses beyond a small threshold.
- **CI:** `cpu-numpy` always; `cpu-numba` always **if it ships** (ADR-008 both-in-CI; hard parity gate,
  not skip-if-absent).

### 5.1 TDD / e2e structure (lakehouse §8)
- **Red-first:** add `cpu-numba` to the parametrized parity/golden/degenerate tests as xfail/skip →
  implement → green (golden + `_reference` exist from Phase 0).
- **Public-API e2e:** parametrize `add_ghost_gk(..., backend="cpu-numba")` (the full threaded path)
  against the CPU golden — guards the `backend=` threading, not just the kernel.
- **Cross-repo e2e (lakehouse-owned):** once `backend=` reaches `add_ghost_gk`, the lakehouse adds a
  parametrized `run_work_unit(..., backend="cpu-numba")` test against `golden.parquet` + marts. **No
  special fixture export needed** beyond the existing golden + the `backend=` kwarg (lakehouse §9-3).

---

## 6. Measurement — single-thread in-venue is the headline (lakehouse §3)

The gating number is **single-thread, in-venue (serverless)** — because production numba is serial. A
**local multi-thread `prange` figure would massively overstate the win and corrupt the gate**; it is
**informational only**, never the decision number.

- Step 1: re-run the A/B single-thread (mirror the venue) → the new numpy baseline.
- Step 2 kill-gate: numba single-thread vs that baseline, **net of per-process compile cost** (§4) —
  warm the JIT before timing for the steady-state number, **and** measure compile cost separately to
  do the amortization analysis (`cache=OFF`, realistic groups-per-executor). **Report both**; the
  decision number is net-of-compile, in-venue.
- Lakehouse measures via `scripts/profile_ac1_local.py` run single-thread + one in-venue serverless
  single-match run; persist-before-cleanup, no contention. The local box ≠ serverless CPU.

---

## 7. Decision ladder & ROI (lakehouse §9-2)

84 tracking matches; full reprocess ~1 working-day one-time; incremental per-new-match.
- **Step 1 (numpy closed-form):** ship if it's a clean dep-free win — **do it regardless.**
- **numba:** ships **only if ≥~1.5× net-of-compile** over step-1 numpy single-thread in-venue (§4/§6;
  below that, a second kernel isn't worth the permanent maintenance at this scale).
- **GPU:** scoped only if in-venue numba still leaves per-match latency unacceptable **and** tracking
  volume is trending past a few hundred matches. At ≤~few-hundred matches, step 1 (or in-venue numba)
  is almost certainly sufficient and GPU is not pursued.

---

## 8. Deferred — GPU (post-numba decision)
A future spec scopes GPU **only** if §7 triggers it: jax (XLA fusion + HF-Jobs compat; cupy = throwaway
local spike, not shipped) **plus** the venue re-architecture (ghost-GK as an offline HF-Jobs L40S batch
stage out of the serverless UDF — without which a GPU kernel has nowhere to run). Gate oracle =
jax-CPU-f64 (never consumer/L40S f64); f32 acceptance = empirical downstream propagation through the
marts (per-action rel-err + mart-contract invariance, not Spearman). Playbook: Phase-0 spec §4 + ADR-012.

---

## 9. Review status & lakehouse deliverables (round 3 — resolved; plan-ready)
All open items confirmed by the lakehouse round-3 review:
1. **`τ_singular` = Cholesky's effective PD boundary, NOT conservative** (§3/§1) — step 1 keeps
   `cho_factor` as the branch so it's automatic; numba replicates it. *Lakehouse will supply a
   representative near-singular covariance* (high 2×2 condition number, near-collinear GK candidates)
   from real IDSSE/skillcorner frames for the parity test.
2. **Single-thread in-venue benchmark, net-of-compile** (§4/§6) + the **~1.5×** numba bar confirmed.
   *Lakehouse runs `scripts/profile_ac1_local.py` single-thread + one in-venue serverless single-match
   run; reports steady-state + compile cost.*
3. **Governed value change for both rungs** confirmed — *lakehouse runs `run_work_unit → golden.parquet
   + marts` and confirms no mart value moves beyond its contract tolerance before production adoption,
   for step 1 and numba both.*

**Next:** writing-plans — **Step 1 first** (numpy closed-form: keep `cho_factor` branch+log_det, swap
`cho_solve`→closed-form energy; golden-gated; near-singular parity test; re-measure single-thread),
then the numba kill-gate (§4/§6) decides whether Step 2 is built.
