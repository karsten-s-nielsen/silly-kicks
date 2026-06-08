# Design: Pluggable, evaluatable xT in `silly_kicks.xthreat` (SK-xT-1)

**Date:** 2026-06-07
**Status:** Approved (brainstorming) — incorporates lakehouse spec review (2026-06-07) → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** Promotion proposal from the luxury-lakehouse session
(`luxury-lakehouse/tmp/silly_kicks_ext_v2_promotion_proposal.md`).

**Effort label:** "SK-xT-1" — deliberately distinct from the lakehouse ExT-v2 "Phase 0/1/2"
labels to avoid cross-session naming collisions. References to "Phase 0/1" below mean the
*lakehouse's* phases (the source of the empirical numbers), not this effort.

## Context

The lakehouse built an "ExT v2" Expected-Threat model (pluggable transitions + KDE
smoothing + held-out NLL evaluator) on top of silly-kicks SPADL data, and proposed
promoting the pure-model parts upstream so silly-kicks owns one canonical, first-class
xT — matching how it already owns VAEP, line-breaking, and the TF-24 calibration harness.

silly-kicks' current `xthreat.py` is the classic Singh-2018 implementation: a fixed 16×12
grid, a monolithic `fit()` with an in-method (non-pluggable) transition step, `rate()` /
`interpolator()`, and **no** KDE smoothing, **no** variable resolution as a first-class
input, and **no** held-out evaluation metric. (Verified against the source 2026-06-07.)

**silly-kicks is the product; the lakehouse is one downstream consumer.** Therefore this is
a *reshape to silly-kicks house conventions*, not a port. The lakehouse code is reference
for the math only.

### Empirical grounding (fact-checked against the lakehouse repo + committed results)

The lakehouse's "KDE improves held-out NLL" claim is backed by committed results
(`docs/evolve/ext-v2-phase-{0,1}/*.json`), run on `soccer_analytics.dev_gold.fct_action_values`
(8,809,385 actions / 5,404 matches / 22 competitions) **at the lakehouse default resolution of
12×8 = 96 zones**:

| Model (lakehouse, 12×8) | Held-out NLL (passes) | vs Singh |
|-------|----------------------|----------|
| Uniform 12×8 | ≈ 4.564 (`log 96`) | — |
| Singh counts (Phase 0) | **3.78924** | baseline |
| KDE-smoothed (Phase 1, gaussian, bw≈2.0, adaptive) | **3.74823** | **+1.082%** |

**These absolute numbers are 12×8- and corpus-specific** — they are an external *reference*,
NOT a target the silly-kicks gate must hit (silly-kicks' default is 16×12 = 192 zones, and the
gate runs on a different corpus — see Testing). Caveats carried into the validation gate: the
gain is **modest (~1.08%)**; the KDE bandwidth optimum **saturated at the search-space upper
edge** (1.99998 of `[0.01, 2.0]`), so the true optimum is unknown and the gain may be
understated; it is a **single run, no CV, no confidence interval**; **the KDE gain is
resolution-dependent** (KDE earns more on finer/sparser grids), so the magnitude at 16×12 will
differ from 12×8. KNN/conditional is **not implemented** in the lakehouse (docstring stubs
only) — so deferring it costs nothing.

## Decision

Promote the **stable, published base** as SK-xT-1, reshaped to house style:

> Pluggable transition family (Singh counts + KDE-smoothed) + variable-resolution `GridSpec`
> + standalone `value_iteration` + held-out NLL evaluator, with `ExpectedThreat` kept as a
> byte-identical back-compat facade over the Singh path at 16×12.

Defer KNN + the per-source-context conditional formulation (pre-publication method
[Salimi et al. 2026, poster] + tracking-join-dependent features) to a later phase.

## Scope

### In scope (SK-xT-1)
1. Convert the `xthreat.py` module into an `xthreat/` package (mirrors `tracking/pitch_control/`).
2. `GridSpec` — frozen dataclass carrying **only** `n_zones_x` / `n_zones_y`. Pitch dimensions
   stay in `spadlconfig` (`field_length=105`, `field_width=68`) — single source of truth; do
   **not** duplicate pitch dims into `GridSpec` (this is where the design diverges from the
   lakehouse `GridSpec`, which carried its own pitch dims).
3. Pluggable transition selection via house-style **string dispatch** (NOT ABCs):
   `Method = Literal["singh_counts", "kde_smoothed"]`, one frozen param dataclass per method
   (`SinghParams`, `KDEParams`), `_METHOD_TO_PARAMS_TYPE`, `validate_params_for_method`.
4. KDE-smoothed transition (Silverman-2D bandwidth, optional adaptive per-row sigma,
   per-source-zone `sklearn.neighbors.KernelDensity`, row-normalized, zero-row fallback).
5. Standalone `value_iteration(...)` extracted from `ExpectedThreat.__solve` — byte-identical.
6. Held-out transition-model NLL evaluator: `holdout_split` (`game_id`-keyed),
   `compute_holdout_nll` (pure function), `compute_holdout_nll_per_group`.
7. `ExpectedThreat` facade: default construction stays byte-identical to today; new
   `method=`/`params=` knob added.

### Out of scope (explicit non-goals — YAGNI)
- KNN transition + per-source-context conditional xT (pre-publication; tracking-join-dependent).
- Atomic-SPADL mirror of xthreat (it has never had one; atomic consumes a raw `xt_grid` ndarray).
- A new VAEP `xt__<method>` xfn factory (xT is a model, not a per-action feature; can be added later).
- Adopting the lakehouse `XTGrid` typed wrapper as the return type (Hyrum break on every
  consumer; xthreat keeps its raw `.xT` ndarray). [proposal §5 option (b)]
- The Optuna HPO sweep (stays lakehouse-side; the library ships the *evaluator*, the reusable
  primitive). A `calibration`-integrated xT sweep is a separate later PR, coordinated with the
  TF-24 session. [proposal §3, Q3]

## Architecture

New package `silly_kicks/xthreat/`:

| File | Responsibility |
|------|----------------|
| `_params.py` | `GridSpec`, `Method`, `SinghParams`, `KDEParams`, `XtParams` union, `_METHOD_TO_PARAMS_TYPE`, `validate_params_for_method`. All frozen dataclasses + Literal + validator — exactly the `tracking/pressure.py` + `tracking/pitch_control/_params.py` pattern. |
| `_grid.py` | Cell-index helpers generalized to `GridSpec`: `_get_cell_indexes`, `_get_flat_indexes`, `_count`, `_safe_divide`, `_scoring_prob`, `_action_prob`, `_get_move_actions`, `_get_successful_move_actions`. (Today's `xthreat.py` private helpers, parameterized by `GridSpec` instead of bare `l, w`.) |
| `_transitions.py` | `singh_transition_matrix(actions, grid)` (today's `_move_transition_matrix`), `kde_smoothed_transition_matrix(actions, grid, params)`. Both pure: DataFrame in, `np.ndarray` out. |
| `_value_iteration.py` | `value_iteration(p_scoring, p_shot, p_move, transition, *, eps, max_iter: int | None = None)` → `(xT, heatmaps)`. `max_iter=None` (default) = unbounded loop = **byte-identical to `__solve`**; a non-None bound is an opt-in safety cap for direct callers (this is now a public primitive callable on an arbitrary `T`, and a degenerate `p_move≈1` matrix would otherwise loop forever). **Extracted byte-identically from silly-kicks' `__solve`** (facade passes no `max_iter` → unbounded → parity preserved). Preserve its exact convergence test (`np.any(newxT - xT > eps)`, raw diff, NOT `max(abs(...))`) and its per-iteration `heatmaps` recording. The `(xT, heatmaps)` return is **intentional public surface** (it backs the existing `.heatmaps` attribute). **Why raw-diff is correct (document in code + ADR so nobody "fixes" it):** the iteration starts at `xT=0` and applies a monotone non-negative operator (`gs ≥ 0`, `p_move ≥ 0`, `T ≥ 0`), so the iterates increase monotonically toward the fixed point — `newxT - xT ≥ 0` always, hence raw-diff ≡ abs-diff here. **Parity trap:** the lakehouse `iterate` uses abs-convergence and returns an iteration count, not heatmaps — do NOT copy it; the Bellman step is identical but the stop condition + return differ. |
| `_model.py` | `ExpectedThreat` class (facade + dispatch). |
| `_eval.py` | `holdout_split` (`game_id`-keyed), `compute_holdout_nll` (pure: transition_matrix + holdout + grid), `compute_holdout_nll_per_group`. Transition-model destination-likelihood NLL. |
| `__init__.py` | Re-exports the **genuinely public** API only: `ExpectedThreat`, `GridSpec`, `Method`, the param dataclasses, `value_iteration`, and the eval functions. Private `_*` helpers are NOT re-exported (see back-compat note — the only importer is silly-kicks' own `tests/test_xthreat.py`, which migrates to the new `_grid.py`/`_transitions.py` paths). |

### Public API

```python
class ExpectedThreat:
    def __init__(
        self,
        l: int = 16,           # back-compat; maps to GridSpec(n_zones_x=l, ...)
        w: int = 12,           # back-compat; maps to GridSpec(n_zones_y=w)
        eps: float = 1e-5,
        method: Method = "singh_counts",
        params: XtParams | None = None,   # validated against method
    ) -> None: ...

    def fit(self, actions: pd.DataFrame) -> "ExpectedThreat": ...
    def rate(self, actions, use_interpolation=False) -> np.ndarray: ...   # unchanged
    def interpolator(self, kind="linear") -> Callable: ...               # unchanged
    # attributes preserved: .xT, .heatmaps, .scoring_prob_matrix,
    #   .shot_prob_matrix, .move_prob_matrix, .transition_matrix, .l, .w, .eps
```

`ExpectedThreat()` with all defaults → **byte-identical** to the current implementation
(Singh, 16×12). `method="kde_smoothed"` swaps **only** the transition builder inside `fit()`;
`rate()` / `interpolator()` / the grid lookup are untouched (KDE is a fit-time choice that
yields a smoother `.xT`).

**Resolution knob:** `l` / `w` remain the **only** public resolution arguments on
`ExpectedThreat` (mapped internally to `GridSpec(n_zones_x=l, n_zones_y=w)`); higher
resolution is `ExpectedThreat(l=24, w=16, ...)`. `GridSpec` is the internal representation
and the explicit argument to the **standalone** functions (`singh_transition_matrix(actions,
grid)`, `kde_smoothed_transition_matrix(...)`, `compute_holdout_nll(..., grid=...)`). No
second `grid=` constructor path — one way to set resolution on the class, no ambiguity.

### Data flow (`fit`)

```
actions ─▶ _scoring_prob / _action_prob (unchanged math, GridSpec-parameterized)
        ─▶ transition matrix  ── dispatch ──▶ singh_transition_matrix
        │                                  └▶ kde_smoothed_transition_matrix
        └▶ value_iteration(p_scoring, p_shot, p_move, transition) ─▶ .xT
```

### KDE-smoothed transition (faithful math, reshaped)

Per the lakehouse `kde.py` (numpy + `sklearn.neighbors.KernelDensity` only; no scipy):
- Same Singh move-type + `result_name=="success"` filter.
- Destination-zone centre coords derived from `GridSpec` + `spadlconfig` pitch dims.
- For each source zone `s` with `n_s > 0`:
  - `adaptive=True` (default): `sigma_s = sqrt((var_x + var_y) / 2)` over that row's end
    positions (floor `1e-6`); `h_s = bandwidth * silverman_2d(n_s, sigma_s)` where
    `silverman_2d(n, sigma) = n**(-1/6) * sigma` (d=2 leading constant simplifies to 1).
  - `adaptive=False`: `h_s = bandwidth` constant.
  - `KernelDensity(kernel=params.kernel, bandwidth=h_s).fit(end_xy).score_samples(zone_centres)`
    → exp → row-normalize.
- Zero-event source rows: fall back to the mean of populated rows (re-normalized);
  pathological all-zero → uniform `1/n_zones`.
- `KDEParams` defaults seeded from the lakehouse champion (gaussian kernel, adaptive=True);
  `bandwidth` default chosen in the plan informed by the validation gate (their optimum
  saturated at 2.0 — see gate). Param ranges and the bandwidth default are pinned in the plan.

### Held-out transition-model NLL evaluator (faithful, reshaped to house style)

This is a **transition-model NLL** — the held-out negative log-likelihood of *where the ball
goes* (pass destination zone given source zone) under the fitted transition matrix. It is NOT
an "xT-quality metric" (it does not score the value surface); name and doc it precisely.

- **Pure function, no producer/model object** (the `producer_or_xt` first arg from the rejected
  Producer concept is dropped — NLL needs only the matrix + holdout + grid):
  ```python
  def compute_holdout_nll(
      transition_matrix: np.ndarray,    # (n_zones, n_zones), row-stochastic
      holdout: pd.DataFrame,            # SPADL move rows
      *, grid: GridSpec, eps: float = 1e-10,
  ) -> float:                           # -mean_i log( T[src_zone_i, dst_zone_i] )
      # Guard the purity trade-off: going pure (no bundled model) means the matrix and grid
      # could silently disagree. Fail loud instead.
      assert transition_matrix.shape == (grid.n_zones, grid.n_zones)
  ```
  (A thin convenience overload may accept an `ExpectedThreat` and read `.transition_matrix`,
  but the pure-matrix form is primary and is what the unit tests target.)
- `holdout_split(actions, *, holdout_fraction=0.15, key_cols=("game_id",))` — **silly-kicks-
  native key** (`game_id`, the raw `convert_to_actions` match key), NOT the lakehouse
  `(competition_id, match_key)`. Deterministic `sha256(key) % 100 < threshold`; match-level
  disjoint split. `key_cols` is overridable for callers (e.g. lakehouse) that have richer keys.
- `compute_holdout_nll_per_group(transition_matrix, holdout, *, grid, group_col="game_id",
  eps=1e-10)` → `dict[str, float]` (carries the same `transition_matrix` + `grid` as
  `compute_holdout_nll`; generalized from the lakehouse `per_competition` — `competition_id` is
  just a `group_col` value a caller can pass, not a baked-in default).
- Required input columns are SPADL-native; the pass filter mirrors the classic Singh move set.

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| ABCs (`TransitionModel`/`Producer`) as proposed | silly-kicks has **zero** ABCs; ADR-005 §8 codifies string-dispatch + frozen-dataclass params (`pressure.py`, `pitch_control/`). ABCs would import a foreign architecture. |
| Parallel functional `compute_xt()` API | xthreat is class-based like VAEP; a functional API is the *tracking per-frame* pattern, wrong fit. Extend the class. |
| Adopt lakehouse `XTGrid` return type (proposal §5a) | Hyrum break on every consumer (`_xt.py`, `_gk_influence`, `_player_influence`, `_cover_shadows` read raw `.xT` / call `.interpolator()`). Keep raw ndarray (option b). |
| Include KNN-conditional now | Not implemented in the lakehouse; pre-publication method; conditional features are tracking-join-dependent (stay caller-side). |
| Keep single `xthreat.py` file | Would ~double the file to do model + transitions + eval; "a large file is doing too much". Subpackage matches the `pitch_control/` precedent. |

## Back-compat contract (non-negotiable, testable)

For `method="singh_counts"` (the default), byte-identical behavior to current `xthreat`:
- `.xT` (shape `(w, l) = (12, 16)`), `.interpolator(kind)`, `.fit()→self`, `.rate()`,
  `.transition_matrix`, `.heatmaps`, `.scoring_prob_matrix`, `.shot_prob_matrix`,
  `.move_prob_matrix`, and the `ExpectedThreat(l, w, eps)` constructor.
- The **genuinely public** import surface (`from silly_kicks.xthreat import ExpectedThreat`)
  is preserved. **Private `_*` helpers are deliberately NOT kept importable from the package
  root** — re-exporting them forever would Hyrum-lock the module layout to satisfy our own
  tests. Verified the only importer of those privates is `tests/test_xthreat.py` (all other
  consumers import the `ExpectedThreat` class); that test file **migrates** to the new
  `_grid.py` / `_transitions.py` paths as part of this change. No external consumer is affected
  (underscore-prefixed = never public API).
- `calibration/_xt.py::load_xt()` stub pattern (inject `.xT` directly, no `.fit()`) keeps working.

## Testing strategy (TDD + hexagonal + e2e)

**Parity gates (the contract):**
- New parity test using **exact** `np.testing.assert_array_equal` (not `allclose` — catches any
  float-ordering drift introduced by the `GridSpec` re-parameterization):
  `ExpectedThreat(method="singh_counts").fit(actions).xT` and `.transition_matrix` equal the
  current implementation's output on the `spadl_actions` + WC2018 fixtures.
- Existing golden gates stay green untouched: `tests/tracking/test_player_influence_snapshot.py`
  (hash `dab140505e42a94a`), `tests/calibration/test_xt.py` (sha256 save/load roundtrip),
  `tests/invariants/test_vaep_geometric_sanity.py::test_xt_grid_is_goal_monotonic`.

**Variable-resolution test (guards the headline capability — currently untested):** the refactor
GridSpec-parameterizes exactly the binning helpers (`_get_cell_indexes`/`_get_flat_indexes`/
`_count`) that variable resolution depends on — the most regression-prone code. Add an explicit
non-default-resolution end-to-end test: `ExpectedThreat(l=24, w=16).fit(WC2018)` →
`.xT.shape == (16, 24)`, `.transition_matrix.shape == (384, 384)`, `rate()` returns finite
values for successful moves, `interpolator()` runs. (rate()/interpolator() are already
resolution-aware — they read `self.l`/`self.w` + `spadlconfig` dims, not module constants — so
this exercises the real lookup path at a non-default grid.)

**KDE unit tests (bandwidth-parametric — the default is fixed later by the real-data diagnostic):**
row-stochastic; `silverman_2d` formula; adaptive per-row sigma; zero-row fallback to
populated-row mean; all-zero → uniform; smoothing converges to Singh as `bandwidth → 0`;
gaussian produces no zero-probability transitions.

**`value_iteration` tests:** Singh-path parity (covered by the parity gate); **KDE-dense-matrix
convergence** — the dense transition from KDE converges within a sane iteration bound on the
fixture (a regime `__solve` never exercised); `max_iter` bound is respected — a degenerate
`p_move≈1` matrix returns at the cap instead of looping forever; `max_iter=None` reproduces the
unbounded result.

**NLL evaluator unit tests:** synthetic-truth recovery (NLL minimized at the true matrix);
eps clipping; transition_matrix/grid shape-mismatch raises (the purity guard); `holdout_split`
determinism, disjointness, `game_id`-level stratification, key-dtype tolerance; per-group dict shape.

**Validation that KDE beats Singh — one hard gate + one diagnostic + one optional cross-check,
NO product→consumer coupling in the default suite:**

1. **Deterministic synthetic-mechanism gate — the SOLE hard merge gate.** Construct a synthetic
   corpus where Singh provably overfits: several source zones with very few (1–2) observed
   destinations, so the Singh row is a spiky empirical estimate while the held-out destinations
   land in neighboring zones. KDE smoothing must yield strictly lower held-out NLL than Singh.
   Fully deterministic (seeded), in-repo, non-flaky — proves the *mechanism* KDE exists to
   exploit. **This is the only pass/fail assertion that KDE beats Singh.** It uses an **explicit
   fixed bandwidth** (NOT the e2e-derived `KDEParams` default — so the gate is self-contained and
   independent of later tuning) and drives the **real** `kde_smoothed_transition_matrix` +
   `compute_holdout_nll` code paths (not a reimplementation).

2. **Real-data DIAGNOSTIC on silly-kicks-native data (committed, reproducible — does NOT assert).**
   Fit Singh and KDE on the **committed WC2018 StatsBomb SPADL fixture**
   (`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`, via the existing `sb_worldcup_data`
   fixture + `play_left_to_right`) and **log both held-out NLLs + their delta + the resolution**
   on a deterministic `game_id`-keyed split. **It does not fail the build.** Rationale:
   "KDE beats Singh" is empirical, not a theorem — it is corpus/resolution/bandwidth-dependent,
   and on a ~64-match fixture at 16×12 with a bandwidth tuned on a different corpus KDE can tie
   or lose. Hard-asserting `≤` here would be a latent flaky test. The diagnostic gives a real-data
   signal (and a regression tripwire if a future change tanks it) without coupling the build to
   an empirical inequality. (Promoting this to a real assertion would require proper train/holdout
   CV bandwidth tuning on this corpus — out of SK-xT-1 scope.) Self-contained: regular suite, no
   network, no Databricks, on a corpus genuinely independent of the lakehouse mart.

   **Resolution is stated explicitly in every NLL test.** Absolute NLL is resolution-dependent
   (16×12 = 192 zones ⇒ higher NLL than 12×8 = 96); an implementer must never compare a 16×12
   run against the lakehouse's 12×8 reference numbers and conclude it's "broken."

3. **Optional owner-gated lakehouse cross-check (triangulation, not a primary gate).** An
   explicitly `@pytest.mark.e2e`, owner-gated test may pull `fct_action_values` (Databricks
   warehouse `6c3b36ca64d183fe`; query recorded in the plan) to triangulate against the
   lakehouse's published 3.789→3.748 at 12×8. Clearly labeled as a consumer-infra cross-check —
   it is NOT in the default suite and NOT the gate (avoids baking product→consumer coupling into
   silly-kicks' tests).

**Bandwidth default:** widen the search past the lakehouse's saturated `[0.01, 2.0]` when fixing
`KDEParams.bandwidth`; record the chosen default + the measured improvement (and the resolution
it was measured at) in the plan/PR.

## Coordination / isolation (pre-merge)

1. **Calibration ripple** — `silly_kicks/calibration/_xt.py` (the other session's live TF-24
   area, inside this session's isolation boundary) imports + fits `ExpectedThreat`. The facade
   keeps it **untouched**; the sha256 roundtrip + snapshot gates prove it. **Coordinate with
   the calibration/lakehouse session before merge.**
2. **ADR-021** (next free; ADR-015 reserved for the causal harness): records the pluggable
   architecture + the published-method dependency surface. Follows `ADR-TEMPLATE.md`.
3. **NOTICE** — add "Mathematical / Methodological References" entries for
   Silverman, B. W. (1986) [KDE bandwidth] and Salimi, M. S. et al. (2026) [ExT poster,
   marked **pre-publication / reproduction**], cross-linked from docstrings via
   `See NOTICE for full bibliographic citations.` (ADR-005 attribution discipline.)
4. **C4** — converting a module to a subpackage and adding an xT method does not change the
   C4-enumerated `tracking` backends/models/aggregator count; confirm tokens/count unchanged →
   C4-free. (`spadl`/xthreat are not enumerated by these tokens.)
5. **Versioning** — minor bump (next free after `origin/main`; reconcile at release per the
   version-bump checklist). One feature branch, single commit (spec + ADR + NOTICE + code +
   tests bundled), PR at the end — no standalone doc commits. No commit/PR without explicit
   approval.
6. **Unrelated grid (do not touch):** `silly_kicks/tracking/_obso.py` has its own OBSO/"EPV"
   surface — a different model from xthreat's xT grid. It is out of scope and untouched by this
   change; the naming adjacency is noted only to prevent confusion.

## Success criteria

- `method="singh_counts"` byte-identical to current xthreat (exact-equality parity test + all
  existing golden gates green).
- KDE flavor implemented, bandwidth-parametric unit-tested, and validated to beat Singh on
  held-out NLL via a **deterministic synthetic-mechanism hard gate** (the sole pass/fail
  assertion), with a **non-asserting real-data diagnostic** on the committed WC2018 StatsBomb
  fixture (logs NLLs/delta/resolution) — no product→consumer-infra coupling in the default suite.
- Variable resolution exercised end-to-end (24×16 fit → shapes → rate → interpolator).
- `value_iteration` convergence verified on the KDE-dense path; optional `max_iter` guard
  (default unbounded = byte-identical facade).
- Held-out **transition-model** NLL evaluator shipped + unit-tested (pure function, `game_id`-
  keyed split) — silly-kicks' first held-out xT evaluation primitive.
- `GridSpec` enables variable resolution; pitch dims remain owned by `spadlconfig`.
- House conventions throughout (string dispatch, frozen params, `validate_params_for_method`,
  NOTICE + ADR, lazy nothing — sklearn is already a hard dep).
- ruff + ruff format + pyright clean; full suite green.
