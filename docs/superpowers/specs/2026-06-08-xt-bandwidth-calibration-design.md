# Design: `calibration`-integrated xT bandwidth / resolution HPO sweep (SK-xT-3)

**Date:** 2026-06-08
**Status:** Approved (brainstorming) → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** SK-xT-1 follow-up (2026-06-07). On-Deck TODO row
"`calibration`-integrated xT bandwidth/HPO sweep (TF-24 pattern)".
**Effort label:** "SK-xT-3" (SK-xT-1 = pluggable xT, 4.17.0; SK-xT-2 = xt VAEP feature, 4.19.0).
**Decision record:** amends ADR-009 (calibration harness recommends-not-applies),
cross-references ADR-021 (pluggable xT).

## Context

SK-xT-1 (4.17.0, ADR-021) shipped the substrate this item needs:

- `ExpectedThreat(method="kde_smoothed", params=KDEParams(bandwidth, adaptive, kernel), l, w)`
  — the pluggable transition family.
- `kde_smoothed_transition_matrix(actions, grid, params)` / `singh_transition_matrix(actions, grid)`
  — the two transition builders.
- `holdout_split(actions, holdout_fraction, key_cols)` + `compute_holdout_nll(T, holdout, grid=)`
  — the deterministic, `game_id`-keyed split and the **held-out transition-model NLL** objective
  substrate (`-mean_i log T[src_zone_i, dst_zone_i]` over successful move rows). This is a
  *transition-model* likelihood, NOT an xT-quality metric (per ADR-021 / NOTICE).

The `KDEParams.bandwidth` docstring records the motivating finding: the NLL-optimal bandwidth is
**corpus-size-dependent** (adaptive Silverman shrinks per-zone `h ~ n^(-1/6)`, so larger corpora
want a larger multiplier) — ~1.0 on a 64-match sample but ≥4 on an 8.9M-action production mart.
The lakehouse SK-xT-1 reference run (at **the lakehouse's 12×8 grid — NOT silly-kicks' 16×12
default**) also saw the KDE optimum **saturate at the search-space upper edge** (bw≈2.0 of
`[0.01, 2.0]`), so the true optimum was never bracketed.

**Resolution is corpus-size-dependent in the same way bandwidth is** (symmetric to the note above,
per review M2): finer grids have more, sparser zones, so they need more data to populate, and the
KDE earns more on finer/sparser grids — but on a small sample a fine grid is dominated by the
unpopulated-zone mean-row fallback (`_transitions.py:122-128`), so the sample-optimal resolution
will differ from the production-optimal one. Both the bandwidth and the resolution recommendation
are therefore explicitly **per-corpus**, not universal. A principled per-corpus sweep over
`bandwidth` × `resolution` × `adaptive` has real, measurable value.

**silly-kicks is the product; the lakehouse is one consumer.** This sweep lives in
`silly_kicks/calibration/` next to the carrier/VAEP objectives and follows the same
recommends-an-auditable-manifest, **does-not-mutate-library-defaults** discipline (ADR-009). The
library default `KDEParams.bandwidth=1.0` is untouched by this PR.

## Decision

Add a `ruthless`/Optuna sweep that, for **one SPADL action corpus**, minimizes the **K-fold
cross-validated** held-out transition NLL over a **three-axis** search space
(`bandwidth` × `GridSpec` resolution × `adaptive`), reports `mean ± SE` against the **Singh
no-smoothing baseline**, and emits an auditable manifest recommending a `KDEParams`+`GridSpec`.
It changes no library default.

**Proxy honesty (review C1).** Held-out transition NLL is a *transition-model* likelihood, and
choosing the config that best predicts pass destinations is a reasonable but **unproven** proxy for
a better xT (xT is the value-iteration fixed point over this matrix; lower destination-NLL does not
provably yield xT values that correlate better with goals or improve VAEP Brier). This PR does NOT
establish NLL-optimal ⇒ xT-optimal. Two consequences, both binding:
(1) the manifest scopes its recommendation to *"optimal for held-out destination likelihood;
xT-quality impact unverified"* — never "the optimal xT grid";
(2) the Databricks fact-check adds one cheap **downstream xT-quality cross-check** — the
correlation of the recommended grid's `rate()` values with held-out goal outcomes vs the Singh
grid's — so the recommendation is checked against an actual xT-quality signal, not sold on NLL
alone. The cross-check is REPORTED (a manifest line), not a gating objective.

`kernel` is held at the gaussian default (not in the TODO scope). It is a frozen-dataclass field
already, so adding it as a `Choice` axis later is a one-line change — explicitly noted, not
silently dropped.

## Architecture

### 0. Refactor (National Park) — `silly_kicks/xthreat/_transitions.py`

Split `kde_smoothed_transition_matrix` so the binning (actions → per-source-zone destination
groups + zone centres) is separated from the KDE core:

```python
def _bin_destinations_by_source(actions, grid, *, max_points_per_zone=None, rng_seed=None):
    """(grouped: dict[int, np.ndarray(n_s, 2)], centres: np.ndarray(n_zones, 2)) — param-invariant.

    SINGLE vectorized grouping pass: np.argsort(start_cell) + np.split (or a groupby) — NOT the
    O(n_zones * n_actions) boolean-mask-in-loop the current code uses (review M1). At 32x20 zones x
    8.9M rows the old `for s: end_xy[start_cell == s]` is ~5.7B comparisons; the grouping pass is
    O(n_actions log n_actions) once. `grouped` is SMALL — total = (train destinations) x 2 floats
    (~64 MB on the 8.9M mart), so it is the cached invariant (review C2'). Optional deterministic
    per-zone subsample (max_points_per_zone, seeded by rng_seed) caps per-trial cdist FLOPs at the
    finest grid; default (None, None) = every row = byte-identical to the legacy binning.
    """

def _gaussian_transition_from_grouped(grouped, centres, grid, params) -> NDArray:
    """SHARED gaussian seam — called by BOTH the library core AND the objective's per-trial path
    (review M6: equivalence is DEFINITIONAL, one function, not two numerics reconciled by a gate).

    Per source zone with destinations `pts`:
      D2     = ((centres[:,None,:] - pts[None,:,:])**2).sum(-1)   # (n_zones, n_s); cdist, recomputed
      h      = params.bandwidth * silverman_2d(n_s, sigma_zone)  if params.adaptive else params.bandwidth
      logits = -D2 / (2*h**2)
      logits -= logits.max(axis=1, keepdims=True)        # SOFTMAX-STABILIZE (review M5): cancels in
      dens   = np.exp(logits).sum(axis=1)                #   the row-normalization, costs nothing,
      row    = dens / dens.sum()                         #   prevents small-h underflow -> 0/0 -> NaN
    Empty zone (n_s == 0) OR dens.sum() still 0 after stabilization -> unpopulated-zone mean-row
    fill — REPLICATES the sklearn path's `if total > 0 else mean-row` branch exactly (review M5).
    `D2` is recomputed per call (cheap cdist); the *cached* invariant is `grouped` (small), NOT D2
    (review C2' — caching D2 is (destinations x n_zones) ~ 20 GB/fold at 32x20, a memory blow-up).
    """

def _kde_transition_from_grouped(grouped, centres, grid, params) -> NDArray:
    """Dispatch on params.kernel:
      - "gaussian" (default + the only kernel this sweep uses) -> _gaussian_transition_from_grouped.
      - epanechnikov/tophat/exponential/linear/cosine -> sklearn KernelDensity fallback (unchanged).
    """
```

`kde_smoothed_transition_matrix(actions, grid, params)` becomes
`_kde_transition_from_grouped(*_bin_destinations_by_source(actions, grid), grid, params)` — so the
library gaussian path and the objective both bottom out in `_gaussian_transition_from_grouped`.

**Why vectorize the gaussian core (not just keep sklearn).** The C2 bottleneck is *per-zone Python
overhead* — ~3,200 `KernelDensity`+BallTree constructions per trial at the finest grid — **not**
per-point FLOPs. For the gaussian kernel the estimator is, after the row-normalization this code
already does, exactly a softmax of `-||c-p||^2 / 2h^2` (the `1/(n*2*pi*h^2)` constant cancels per
row). So the per-zone density is one vectorized `cdist`+`exp` — no object/tree construction. The
**cached invariant is the grouped destinations** (small, ~64 MB), and each trial recomputes the
per-zone `cdist`+`exp` from cache. That one `cdist`/zone/trial is FLOP-comparable to the `exp` we
already do, so the per-trial cost stays minutes-not-hours while memory stays ~64 MB — vs caching
`D^2`, which is `(destinations x n_zones)` ≈ 20 GB per `(grid, fold)` and unusable on any box
(review C2'). The original C2 win (kill the sklearn object overhead) is fully preserved.

**Golden re-pin (legitimate — Chesterton-verified, no consumer).** `grep` confirms
`kde_smoothed_transition_matrix` has **exactly one caller** (`_model.py:123`) and `ExpectedThreat`
defaults to `singh_counts` (`_model.py:78`) — so no shipped artifact (VAEP, lakehouse, bundled
weights) depends on the KDE numerics; the re-pin is Chesterton-safe within the repo. The gaussian
path's output is **re-pinned** to the vectorized result — not a model change, the *same* estimator
computed faster. The characterization/golden (written first, TDD) records the new vectorized output;
a separate **1e-9 sklearn-parity gate** (incl. a small-h underflow case, M5) is the safety net
against a hand-rolled-numerics mistake. The SK-xT-1 Singh byte-parity oracle
(`tests/xthreat_legacy_reference.py`) is untouched; non-gaussian kernels keep sklearn, unchanged.

### 1. `silly_kicks/calibration/_xt_bandwidth_objective.py` (new)

A duck-typed objective (`evaluate(candidate) -> Metrics`, modeled on `CarrierAccuracyObjective` —
a plain object with an internal lazy cache, NOT a `ruthless.CachedObjective`, because the resolution
axis means the invariant is keyed by resolution, which a single `prepare()` does not model cleanly).

- **Input:** one `actions` DataFrame (SPADL columns + `game_id`); `seed`; `max_points_per_zone`.
- **Direction:** MINIMIZE `xt_holdout_nll`.
- **CV:** reuse `match_cv_splits` (`silly_kicks.calibration._cv`; GroupKFold(5) for >7 games /
  leave-one-match-out ≤7) over the per-action `game_id` array. Per fold: fit the transition matrix
  on the train games, score `compute_holdout_nll` on the held-out games. Aggregate → `mean` NLL +
  `cv_standard_error`.
- **Invariant cache** keyed `(grid, fold_idx)`: the per-fold train **grouped destinations + centres**
  (`_bin_destinations_by_source(...)`, ~64 MB on the 8.9M mart) and the held-out
  `(src_zone, dst_zone)` flat-index arrays. These are invariant across `bandwidth`/`adaptive`. Each
  trial calls the **shared `_gaussian_transition_from_grouped(cached_grouped, cached_centres, grid,
  KDEParams(bw, adaptive))`** (M6) — the same function the library bottoms out in, so cache-
  equivalence is definitional. **The cache holds `grouped`, NOT `D²`** (review C2'): `D²` is
  `(destinations × n_zones)` ≈ 20 GB per `(grid, fold)` at 32×20 and would OOM any box once TPE has
  touched several grids; `grouped` is ~1000× smaller and the per-trial `cdist` it implies is
  FLOP-comparable to the `exp`. This is the TF-24 invariant-prepare lesson
  (`[[feedback_invariant_prepare_for_hpo_objectives]]`): cache the param-invariant *marshalling*
  (the binning that scales with the 8.9M corpus), not the kernel. Cache is lazy keyed by resolution
  (the discrete `grid` Choice → a handful of grids), one resolution's `grouped` resident at a time is
  trivially affordable.
- **Singh baseline** NLL is computed once per `(grid, fold)` (param-free) and cached; surfaced in
  the metrics every trial so the manifest always reports the no-smoothing reference and the relative
  KDE win/loss at the chosen resolution. **N1:** Singh uses a *different* action filter —
  `_get_move_actions` (ALL moves, for the denominator) vs KDE's `_get_successful_move_actions` — so
  it does NOT reuse the successful-only KDE grouped cache; it is computed via the library
  `singh_transition_matrix(train, grid)` directly (param-free → trivially cacheable per `(grid, fold)`).
- **Metrics returned:** `xt_holdout_nll` (mean across folds), `xt_holdout_nll_se`,
  `singh_holdout_nll` (mean, at this trial's resolution), `n_folds`, `n_holdout_moves`.

**Per-trial cost budget (review C2 / C2′).** The original C2 bottleneck was *per-zone Python
overhead* — up to 3,200 sklearn `KernelDensity`+BallTree constructions per trial at the finest grid
(32×20 = 640 zones × 5 folds), the shape that killed TF-24 Stage-2 at trial 12. The shared
vectorized gaussian seam removes it: per zone, one `cdist`+`exp` from cached `grouped`, zero object
construction. Memory stays ~64 MB (cache `grouped`, not `D²` — C2′). So:
  - **`max_points_per_zone`** is an *optional per-trial-FLOP / pathological-memory cap*, NOT
    load-bearing for the cache (which is now small regardless): a central-midfield zone on the 8.9M
    mart can hold millions of destinations, so its per-trial `cdist` (`n_s × n_zones`) is the only
    real per-trial cost at the finest grid; capping `n_s` bounds it, and is statistically principled
    (KDE on a few thousand well-sampled points ≈ on millions). Deterministic, `seed`-controlled
    (review N2 — this is what `seed` is for), applied **in the cached binning** so the cached fast
    path and the from-scratch reference subsample identically → cache-equivalence holds. Default
    `None` (no cap) for unit-test determinism; the CLI may set it for the production sweep. **N6:**
    the cap shifts per-zone `n`/`σ` and hence the NLL surface, so a capped recommendation ≠ the
    uncapped one — the manifest records the cap and states the recommendation is conditional on it;
    the library-parity gate runs at `None`. With the small `grouped` cache the cap may be unnecessary
    even at scale — the production run decides empirically.
  - **Structural perf guard** (house pattern — call-count spy, `tests/_perf_structural.py`): a test
    asserting that across N trials at fixed resolution `_bin_destinations_by_source` is invoked
    **once per (grid, fold)** (cache works) while `_gaussian_transition_from_grouped` is invoked
    **once per (trial, fold)** — i.e. the binning is NOT re-run per trial. This is the budget,
    enforced deterministically (no wall-clock assert, per the CI convention).
  - The 8.9M fact-check still runs **as a Databricks/DGX job, not inline** (loading + binning the
    full corpus is memory-heavy at ingest even though the resident `grouped` cache and per-trial CPU
    are now modest); `n_trials` default 100 local / up to 200 for the production sweep; the job
    records wall-clock + peak memory in the manifest.

Degeneracies fail loud or score as no-signal (not silent NaN-averaging): a fold whose held-out set
has zero eligible move rows is excluded from the fold mean (**N3:** if this collapses the usable
folds to 1, `cv_standard_error` returns `nan` by design — the manifest renders `xt_holdout_nll_se`
as `nan` gracefully, never crashes); a resolution+adaptive combination that yields an all-uniform
matrix still produces a finite (high) NLL and competes honestly.

### 2. Search space — extend `silly_kicks/calibration/_spaces.py`

```python
# Aspect-sane grids near the pitch's ~1.54 ratio (105x68). Resolution is SWEPT (review M2), but
# over a curated discrete set rather than two independent IntRanges — the latter is a ~475-cell
# space that wastes the trial budget and admits non-physical grids (e.g. 32x6). TPE handles a
# categorical grid axis cleanly; "16x12" is the silly-kicks default (warm-start).
_GRIDS = ("12x8", "16x12", "20x14", "24x16", "28x18", "32x20")

def xt_bandwidth_config(*, n_trials, store_path, sampler="tpe") -> OptunaConfig:
    return OptunaConfig(
        kind="optuna", metric="xt_holdout_nll", direction=Direction.MINIMIZE,
        n_trials=n_trials, sampler=sampler,
        param_space={
            "bandwidth": FloatRange(kind="float", lo=0.1, hi=20.0, log=True),
            # NOTE: ruthless Choice uses kind="choice" (NOT "categorical" — that raises a pydantic
            # literal_error) and a tuple of choices. Verified against installed ruthless.
            "adaptive":  Choice(kind="choice", choices=(True, False)),
            "grid":      Choice(kind="choice", choices=_GRIDS),
        },
        warm_start={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"},
        store=StoreConfig(kind="sqlite", path=store_path),
    )
```

- `bandwidth` range `[0.1, 20.0]` log-scale is wide enough to bracket the production optimum
  (≥4 multiplier) the SK-xT-1 lakehouse run could not (its `[0.01, 2.0]` saturated), and spans
  both interpretations of `bandwidth` (an adaptive-Silverman multiplier when `adaptive=True`; a raw
  bandwidth in SPADL metres when `adaptive=False`). The dual meaning is documented in the manifest;
  the held-out NLL is a single comparable metric across the whole joint space, so the joint
  optimization is well-posed — we report the best `(adaptive, bandwidth)` pair, not a per-mode one.
- **Resolution swept as a discrete `Choice` of ~6 aspect-sane grids** (review M2 decision). The
  objective parses `"16x12"` → `GridSpec(16, 12)` and keys its invariant cache on the grid string.
  Adding a grid to `_GRIDS` is the supported way to widen the resolution search.
- Exact `ruthless` constructor names (`Choice`, `FloatRange`) verified at implementation against the
  installed `ruthless` (`Choice`, `IntRange`, `FloatRange` are all in `dir(ruthless)`).

### 3. `scripts/calibrate_xt_bandwidth.py` (new) — I/O + manifest

Pure objective stays in `calibration/`; this script owns I/O, study orchestration, and the report.
Single corpus in → recommendation out (an xT grid is corpus/league-level, unlike TF-24's
multi-provider equal-weight fold).

- **Corpus loaders (reused):** `--source pining` loads SPADL actions across matches via
  `scripts._loader_pining`; we need only `actions`, so **load with the smallest tracking footprint**
  — pass **`tracking_limit=1`** (NOT `0`: review N8 — the GS loader gates on `if tracking_limit:`
  (`_loader_pining.py:512`), so `0` is falsy and loads *all* frames), or add an actions-only loader
  entry point that skips frame construction entirely (the cleaner route, preferred if frame-build
  cost on the full corpus matters). `--source databricks` loads
  `soccer_analytics.bronze.spadl_actions` via `scripts._loader_databricks` (the
  `_XT_COLS = [game_id, start_x, start_y, end_x, end_y, type_id, result_id]` projection already used
  by the TF-24 xT-corpus loader).
- **Orchestration (testable `run`-seam, mirrors `run_stage`):**
  ```python
  objective = XtBandwidthObjective(actions, seed=seed, max_points_per_zone=cap)
  config = xt_bandwidth_config(n_trials=n_trials, store_path=store)
  result = OptunaStrategy(config, seed=seed).run(objective, backend=InProcessBackend())
  ```
  (review N2: the pseudocode keeps both `seed=` on `OptunaStrategy` — sampler seeding — and
  `backend=InProcessBackend()`, matching `calibrate_tracking_defaults.py:72`.)
- **Manifest** (mirrors `build_manifest`): `stage="xt_bandwidth"`, source, seed,
  `max_points_per_zone`, n_trials, corpus `game_id`s, wall-clock, `silly_kicks`/`ruthless`/`xgboost`
  versions, generated_date, the recommended `KDEParams`+`GridSpec`, best `xt_holdout_nll ± se` (`se`
  may be `nan` on a 1-fold corpus — N3), the Singh baseline NLL at the best resolution, the
  bandwidth dual-meaning note, the downstream xT-quality cross-check result (C1), and an explicit
  **`applies_to_library_default: false`** line scoping the recommendation to *"optimal for held-out
  destination likelihood; xT-quality impact unverified"* (ADR-009 + review C1). Writes
  `report.json` + `report.md`.
- `--max-matches`, `--tracking-limit`, `--max-points-per-zone`, `--n-trials` caps for bounded local
  runs (the loaders already support match caps).

## Data flow

```
corpus actions (game_id + SPADL)
   │  CLI loader (pining held-out matches | databricks bronze)
   ▼
XtBandwidthObjective(actions, seed, max_points_per_zone)
   │  match_cv_splits(game_id) → K folds
   │  per (grid, fold): cache  grouped destinations + centres (~64 MB)  +  holdout (src,dst) zones
   │                    + Singh matrix (param-free, _get_move_actions filter)
   ▼  per trial (bandwidth, adaptive, grid):
        for fold: _gaussian_transition_from_grouped(cached_grouped, centres, grid, KDEParams)  ← shared seam
                  compute_holdout_nll(T, cached holdout indexes, grid)   ← cheap lookup
                  singh baseline (cached, param-free)
        → mean NLL ± SE, singh baseline
   ▼
OptunaStrategy (TPE, MINIMIZE) → best (bandwidth, adaptive, grid)
   ▼
manifest: recommended KDEParams+GridSpec, NLL±SE vs Singh, downstream xT-quality cross-check
          (REPORTED — defaults UNCHANGED)
```

## Testing (TDD)

- **sklearn-parity gate first (the safety net for the hand-rolled gaussian):**
  `_gaussian_transition_from_grouped` output == sklearn `KernelDensity(kernel="gaussian")`
  row-normalized, to 1e-9, on a small multi-game fixture, for both `adaptive` values **and including
  at least one small-`h` (underflow-regime, e.g. `adaptive=False, bandwidth=0.1`) case** (review M5):
  the softmax max-subtraction must keep the vectorized path finite where naive `exp(-D²/2h²)` would
  underflow to `0/0`, and must route a genuinely-empty/all-underflow zone to the *same* mean-row fill
  sklearn's `if total>0` branch uses. Written FIRST (TDD) — this is what makes re-pinning the golden
  safe; if the math or the stabilization is wrong, this fails before anything is pinned.
- **Characterization/golden (re-pinned to the vectorized output):** pin
  `kde_smoothed_transition_matrix` output (small fixture, multi-game synthetic SPADL) to a stored
  array set (`[[feedback_multi_hash_snapshot_sets]]` for numpy micro-version drift). The gaussian
  path's golden is the NEW vectorized result (legitimate — no ExT-v2 consumer; same estimator); the
  non-gaussian-kernel paths stay byte-identical (still sklearn). The binning vectorization (M1) is
  byte-safe because the golden pins output, not implementation.
- **Objective unit tests** (synthetic multi-game SPADL, committed fixture, NOT e2e): deterministic
  NLL for a fixed candidate; CV fold count matches `match_cv_splits`; Singh baseline surfaced;
  empty-holdout-fold excluded from the mean; resolution axis actually changes the binning.
- **Cache-equivalence gate** (1e-9, hand-written since this is a plain object, not a
  `CachedObjective`): the lazy-cached `evaluate` path (warm cache + a second call reusing it) ==
  the objective's own cold-cache recompute, for candidates spanning all three axes — including a
  repeated grid to exercise the cache hit, and both `adaptive` values. This tests **the cache**
  (warm == cold), independent of subsampling (both paths subsample identically under `seed`).
- **Library-parity gate (now a smoke test — review M6):** because the objective's per-trial path and
  the library gaussian path call the *same* `_gaussian_transition_from_grouped`, equivalence is
  definitional; the gate just confirms, with `max_points_per_zone=None`, that the objective's per-fold
  matrix == `kde_smoothed_transition_matrix(train, grid, params)` to 1e-9. No second numerics path to
  keep in lockstep — that drift surface is deleted by construction.
- **Round-trip constructibility (review M3):** feed the recommended `KDEParams`+`GridSpec` from a
  manifest back into `ExpectedThreat(method="kde_smoothed", params=..., l=, w=).fit(actions)` and
  assert a finite, non-degenerate xT grid — proves the recommendation is a *usable config*, not just
  numbers.
- **Structural perf guard (review C2):** call-count spy (`tests/_perf_structural.py`) asserting that
  over N trials at fixed resolution, `_bin_destinations_by_source` runs once per `(grid, fold)` while
  `_gaussian_transition_from_grouped` runs once per `(trial, fold)` — the invariant/per-trial split is
  the budget, enforced deterministically (no wall-clock assert).
- **`_spaces` config test** (mirrors `test_spaces.py`): metric/direction/param-space (incl. the
  `grid` Choice members)/warm-start; **and every `_GRIDS` member parses to a valid `GridSpec`**
  (review N7 — a typo'd `"32x20"` would silently become an unpopulated-zone disaster).
- **CLI smoke test** (mirrors `test_cli_smoke.py`): `run`-seam on a tiny in-memory corpus produces a
  finite best + a well-formed manifest with `applies_to_library_default: false`; no network.
- **Public-API Examples** on every new public def (CI-enforced, `[[feedback_public_api_examples]]`).
- **NaN/dtype:** the objective consumes `game_id` of mixed provider dtypes — fixtures exercise both
  numeric and string `game_id` (`[[feedback_test_fixtures_must_exercise_real_dtypes]]`); the CV
  grouping and NLL must not crash on either.

## Fact-check (Databricks)

After the harness is green locally, run it (as a **Databricks/DGX job, not inline** — review C2)
against `soccer_analytics.bronze.spadl_actions` to:
1. re-confirm the motivating claim — that the NLL-optimal bandwidth is materially larger on the
   8.9M-action mart than on a small sample, and that KDE beats Singh held-out — turning the
   `KDEParams.bandwidth` docstring assertion and this design's premise into a recorded, reproducible
   manifest;
2. run the **downstream xT-quality cross-check (review C1, pinned per review M7).** A single defined
   number per grid, reproducible:
   - **Held-out set:** the same CV held-out folds the sweep already produced (no new split).
   - **Per successful move action** in the held-out fold, compute the grid's
     `Δrate = rate(end) − rate(start)` and the binary label `scored_within_K` (the in-possession team
     scores within `K = 10` actions — the `vaep.labels.scores` window, reused for consistency).
   - **Statistic:** Spearman rank-correlation `ρ(Δrate, scored_within_K)`, computed for the
     **recommended grid** and the **Singh grid**; the manifest records both `ρ` values per grid.
   This checks the recommendation against an xT-quality signal rather than NLL alone. **Reported in
   the manifest, not a gate** — if the NLL-best grid does NOT also win `ρ` vs Singh, that is a finding
   worth surfacing (it cautions against treating the NLL recommendation as xT-optimal — exactly the
   C1 honesty concern made falsifiable).

(The adjacent *committed owner-gated lakehouse-mart NLL cross-check* is a **separate** TODO row and
is out of scope for this PR unless explicitly folded in.)

## Housekeeping

- Exports in `silly_kicks/calibration/__init__.py`: `XtBandwidthObjective`, `xt_bandwidth_config`,
  and any new `_transitions` helpers that warrant a public name (the binning split stays private `_`
  unless a test needs it importable).
- Version **4.20.0** (new feature, additive); CHANGELOG entry; remove the On-Deck TODO row
  (`[[feedback_todo_grooming_delete_dont_annotate]]`).
- Amend **ADR-009** with the xT-NLL objective; cross-ref ADR-021. The amendment must record:
  (a) the **deliberate divergence from ADR-009's `ruthless.CachedObjective` + `assert_cache_equivalence`
  pattern** (review M4) — SK-xT-3 uses a plain duck-typed object with a resolution-keyed lazy cache
  and a hand-written equivalence gate, because the resolution axis means the invariant is keyed by
  grid (which a single `prepare()` does not model), by design not oversight; and (b) the
  **gaussian-core re-pin Chesterton evidence** — `grep` shows `kde_smoothed_transition_matrix` has
  exactly one caller (`_model.py:123`) and `ExpectedThreat` defaults to `singh_counts`
  (`_model.py:78`), so no shipped artifact depends on the KDE numerics. State both so the next reader
  sees no undocumented inconsistency. No NOTICE entry needed (Singh 2018 / Silverman 1986 /
  held-out-NLL already cited).
- `/final-review` before the single commit; `ruff format --check` + `ruff check` + `pyright
  silly_kicks/` (full package) + `pytest -m "not e2e"` all green first (Shift Left).

## Non-goals (explicit, not deferred-from-this-item)

- `kernel` as a swept axis — not in the TODO scope; trivially addable (one `Choice` line). Held at
  gaussian default.
- KNN / per-source-context conditional xT — separate TODO row TF-47, pre-publication method.
- The committed owner-gated lakehouse-mart NLL e2e — separate TODO row.
- Mutating any library default — forbidden by ADR-009 (this harness recommends only).
