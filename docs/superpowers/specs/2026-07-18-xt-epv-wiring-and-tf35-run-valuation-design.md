# Real-xT EPV wiring + TF-35 off-ball-run valuation — design (4.52.0)

**Date:** 2026-07-18 · **Status:** APPROVED (review round 2; R2 findings applied) — see §Review log
**Release:** 4.52.0, one PR (PR-S119, branch `pr-s119-xt-epv-and-run-values` off main @ ec543cc = 4.51.0).
One PR is the owner's stated batching preference (release/CI churn); the reviewer's two-PR
preference is recorded — mitigation adopted: commits ordered A-then-B pre-squash so a surgical
revert of Part B alone stays possible before merge.
**ADRs:** TWO, reserved at release — **ADR-041** (Part A: real-xT injection + orientation
repairs, incl. the ADR-028 amendment) and **ADR-042** (Part B: TF-35 run valuation). Shared
release, separate decisions (review Q7).
**Sources:** TODO On-Deck rows "Wire real xT into OBSO/space-creation EPV" + TF-35;
`docs/superpowers/specs/2026-07-16-soccermatics-pro-future-work-plan.md` §W3 + §W1 (Soccermatics
Pro module 16.2 — Sumpter/Twelve production-club method).
**Why these two together:** both land tracking-derived per-action columns the lakehouse will fold
into its one outstanding `fct_action_context` full recompute (add planned columns first, recompute
once), and Part B consumes the injected-xT adapter Part A ships.

All file:line anchors verified against the working tree at main `ec543cc` (4.51.0) by the
authoring session; the load-bearing claims were independently re-verified by the reviewing
session (all confirmed — see §Review log). Remaining low-stakes anchors
(`_action_orientation.py:28-90`, `test_obso.py:249`, liveness-fixture speeds,
`off_ball_context_xfns` template lines) are confirmed by the implementing session at plan time.

---

## Part A — wire real xT into OBSO / space-creation / PAUSA EPV, and repair the seam's orientation defects

### A0. Verified findings (the premise, plus two latent defects found during spec verification)

1. **Synthetic grids everywhere (the TODO premise, confirmed).** Every OBSO / pass-OBSO /
   space-creation / PAUSA value ever served used synthetic proxies: EPV =
   `_make_synthetic_epv_grid` `np.linspace(0.01, 0.3, nx)` tiled over rows
   (`_obso.py:100-103`), transition/reachability = a centre-anchored Gaussian
   (`_make_synthetic_reachability_grid`, `_obso.py:86-97`), both via `_get_default_grids`
   (`_obso.py:106-126`). The `epv_grid`/`transition_grid` ndarray kwargs exist end-to-end
   (`compute_obso_surface` `_obso.py:193`, `compute_pass_obso` `_obso.py:285`, `add_obso`
   `features.py:4795`, `obso_xfns` `features.py:4889`, `add_space_creation` `features.py:5032`,
   `space_creation_xfns` `features.py:5159`, `add_pausa`/`pausa_xfns` `features.py:5216/5275`)
   but **no call path in the repo converts a fitted `ExpectedThreat` into them** — the only
   non-None `epv_grid` call site is a synthetic array in `tests/tracking/test_obso.py:249`.
   `player_influence` is the sole real PC×xT fusion (`_player_influence.py:116-143`).

2. **DEFECT A (OBSO orientation — ADR-028 class).** OBSO performs **zero** attack-direction
   handling. `add_obso`'s `home_team_id` kwarg (`features.py:4801`) is threaded into
   `_precompute_obso_lookup` (`features.py:4666`) and **never referenced in the body** — a dead
   parameter. The synthetic EPV ramp always increases toward +x, so for away-team passes on
   canonical home-attacks-right frames the EPV multiplier rewards the **wrong goal** today.
   Compounding: `target_position` is raw SPADL `end_x`/`end_y` (**action-LTR**,
   `features.py:4725-4726`) passed against pitch-control surfaces built from
   **home-attacks-right** frames (`features.py:4772-4781`) — for away-team actions the target
   and the surface are a 180° point reflection apart, the exact bug class ADR-028 fixed
   elsewhere. ADR-028 listed obso as "self-reconciling"; that classification is **wrong** —
   obso doesn't reconcile, it simply never handled orientation. `space_creation` is
   frame-only (no action-side coordinate enters `compute_space_created`), so only its EPV
   directionality (via the shared ramp) is affected, not a target mismatch.

3. **DEFECT B (xT interpolator y-inversion; latent in `player_influence`).**
   `ExpectedThreat.xT` stores rows y-INVERTED (row 0 = top of pitch, `_grid.py:24-26,47`;
   `rate()` indexes `grid[startyc.rsub(w-1), startxc]`, `_model.py:254`). `interpolator()`
   builds `RectBivariateSpline(x, y, self.xT.T)` against **ascending physical** y centres
   (`_model.py:174-183`), pairing ascending query-y with inverted-row data — so
   `interp(xs, ys)` evaluated at physical y returns the value stored for `68 − y`. `rate()`
   cancels this with the same inverted indexing; **`player_influence` does not**
   (`_player_influence.py:117-118` multiplies the raw interp output against ascending-y PC
   surfaces, with only the away x-flip at `:121-122`) — a latent y-mirror, invisible in
   practice only because fitted xT surfaces are nearly y-symmetric. `_xt_gk._grid_value`
   handles the inversion correctly (`_xt_gk.py:210`). The `interpolator()` docstring
   (`_model.py:158-170`) does **not disclose** the inversion — a loaded public trap (§A1b).

4. **Mirror seam (4.24.0).** `compute_space_created` x-mirrors the (interpolated) transition +
   EPV grids for the opponent perspective inline — `np.flip(..., axis=1)` at
   `_space_creation.py:223-224` (no named helper; flow is `epv_grid` →
   `_get_default_grids` passthrough at `:163` → `_interpolate_grid` at `:203` → flip at
   `:224` → `obso_multiplier_opponent` at `:229`). `add_space_creation` always runs the
   opponent perspective (`features.py:5011-5019`), so an injected grid **always** traverses
   the mirror. The axis-1-only flip is exactly correct for y-symmetric grids; for a real
   (slightly y-asymmetric) xT the faithful opponent transform is a **point reflection**
   (flip both axes). Both synthetic grids are y-symmetric (row-constant ramp;
   centre-anchored Gaussian — verified `_obso.py:92-97,102-103`), so upgrading the mirror to
   a point reflection is expected byte-identical on the synthetic path (§A6 gates it).

5. **Scale semantics.** `compute_obso_surface` max-normalizes the transition grid and clips
   OBSO to [0,1] (`_obso.py:266-274`); `_interpolate_grid` treats cell-centre values as node
   values (half-cell registration offset when resampling the 12×16 xT grid). Injected-EPV
   magnitudes therefore survive *relatively*, not absolutely. Documented, not "fixed" —
   OBSO is a probability-composition surface, not an absolute EPV ledger.

### A1. New module: `tracking/_xt_adapters.py`

Small, pure, no runtime `xthreat` import at module top level (`TYPE_CHECKING`-only annotation,
duck-typed — the SK-xT-2 precedent). Two public functions, exported via `tracking.__all__`:

- **`epv_grid_from_xt(model, grid_x, grid_y)`** → `np.ndarray` of shape `(ny, nx)`,
  **ascending-y rows, ascending-x columns, attack-toward-+x (attack-LTR)**, values in xT
  probability units. `grid_x`/`grid_y` are ascending cell centres in metres (the
  `PitchControlSurface.grid_x/grid_y` convention, `_surface.py:34-49`).
  **Implementation (review V1 — flip the data, not the output):** normalize orientation at
  the data level, once — `phys = np.flipud(model.xT)` (row 0 = y=0), build
  `RectBivariateSpline(x_centres, y_centres, phys.T, kx=ky=1)` inside the adapter (cell
  centres derived from `spadlconfig` dims exactly as `_model.py:174-178` does), query at
  `(grid_x, grid_y)` directly, transpose to (ny, nx). Correct for **any** ascending
  `grid_y` — no symmetry precondition, no dependence on the defective `interpolator()`
  seam, no second compensation style. scipy guard mirrors xthreat's
  (`ImportError` when `RectBivariateSpline` is unavailable, `_model.py:171-172` — scipy is
  already required by the path being replaced). The rejected first draft
  (call `interpolator()`, row-flip the output, `ValueError` on non-mirror-symmetric
  `grid_y`) is recorded: the symmetry precondition was an artifact of the implementation
  choice, not the domain, and would have been permanent API surface.
- **`xt_at_points(model, x, y)`** → per-point xT values via **direct cell indexing with the
  `(n_rows−1)−yj` inversion**, i.e. exactly `ExpectedThreat.rate(use_interpolation=False)`
  semantics. NaN coords → NaN (ADR-003). **Cell-indexer reuse (review V3, resolved against
  the M5 freeze):** the implementation uses the sanctioned lazy function-local
  `from silly_kicks.xthreat._grid import _get_cell_indexes` — the exact pattern
  `_xt_gk._grid_value` uses and documents ("DRY (review H1): reuse xthreat's frozen
  cell-indexer (ADR-021)… this is xthreat's port, not xT-GK's to own", `_xt_gk.py:194-198`).
  The reviewer's re-home-`_grid_value`-into-the-adapter suggestion is **not** taken — as a
  **cost/benefit rejection, not an impossibility** (R2-1 correction): the M5 freeze protects
  v1 *behavior*, not file bytes — the 2026-07-10 GK-distribution-mask cycle (PR-S110)
  already shimmed `_gk_distribution_mask` inside `_xt_gk.py` as a byte-identical delegating
  shim under the golden gate, so touching the file is possible; it is simply not worth it
  here (re-pointing an internal helper buys zero behavioral gain for real golden-regen and
  review overhead on a module scheduled for removal). One story, stated: lazy private import
  of xthreat's frozen cell-indexer IS the sanctioned pattern; `_xt_adapters` is the
  canonical forward home; `_xt_gk`'s private copy dies with v1's removal (≥1 release after
  the lakehouse migrates). ADR-041 records this with the true (cost/benefit) reason.

Fail-loud guard shared by both (module-local `_require_fitted_xt`, semantics copied exactly
from `vaep/features/expected_threat.py:27-51` rather than imported — `tracking` must not grow
a `vaep` import edge): `str` → `NotImplementedError` (the reserved ADR-011 bundled-variant
door), `None` → `ValueError`, all-zero `.xT` → `NotFittedError`. **Drift sentinel (review
V4):** a cross-package parity test asserts identical exception types + messages for the
str/None/unfitted triple across `vaep.features.expected_threat._require_fitted_xt` and the
tracking copy (the version-floor-sentinel pattern).

**Golden test (the critical one):** hand-build an ASYMMETRIC fitted grid (distinct known value
in one top-band cell, another in one bottom-band cell — per
[[feedback_symmetry_test_insufficient_pin_ground_truth]]; a y-symmetric fixture would pass
under the very bug this adapter exists to avoid) and pin `epv_grid_from_xt` cell values to the
physically-correct orientation, plus `xt_at_points` == `ExpectedThreat.rate` on a synthesized
action frame. Mutation check: removing the `flipud` must go red. Additionally: a non-uniform
ascending `grid_y` case (the V1 design handles it; the rejected draft could not).

### A1b. Disarm the `interpolator()` public trap (review V2)

`ExpectedThreat.interpolator()`'s docstring is amended in this PR to state the
output-orientation contract explicitly — the returned values are y-mirrored relative to
physical pitch coordinates (evaluating at physical y returns the value stored for 68−y;
`rate()` compensates by inverted indexing) — and to point callers needing physically-oriented
grids at `tracking.epv_grid_from_xt`. Doc-only: no behavior change, the SK-xT-1 frozen-oracle
parity gate is untouched. **Deferred fork (recorded in ADR-041, 5.0-candidate):** an oriented
variant or `orientation=` kwarg on `interpolator()` with deprecation of the raw behavior —
external consumers (Hyrum) currently have no chance of getting this right.

### A2. `xt=` kwarg threading (additive, opt-in — defaults NOT flipped)

Add `xt=None` keyword to: `add_obso`, `obso_xfns`, `compute_pass_obso`, `add_space_creation`,
`space_creation_xfns`, `add_pausa`, `pausa_xfns` (+ the atomic mirrors, which are pure
re-exports of the same functions — `atomic/tracking/features.py:44,52` — so they inherit).
`compute_obso_surface` and `compute_space_created` stay grid-only (lowest-level pure engines;
policy at the edge, [[feedback_policy_at_edge_not_shared_engine]]).

- `xt=` and `epv_grid=` are **mutually exclusive** → `ValueError` if both supplied.
- When `xt=` is supplied to an **aggregator/factory**: run `_require_fitted_xt`, then build
  `epv_grid = epv_grid_from_xt(xt, obso_grid_x, obso_grid_y)` **once per call** at the
  `ObsoParams` grid geometry (`grid_nx=104`, `grid_ny=68`, `_obso.py:51-56`) and thread it
  down the existing `epv_grid` path (which `_interpolate_grid`s to whatever shape each
  consumer needs) — the aggregators never pass `xt` itself downward, so there is no
  per-action re-sampling. `compute_pass_obso`'s own `xt=` exists for standalone direct
  callers and builds the grid once per call by design (it is a single-pass entry point).
  **Recorded (R2-8):** the injected path double-resamples (spline query at 104×68, then
  node-resample to each engine's PC shape — the §A0.5 half-cell offset); the adapter could
  query the spline at each final geometry directly, but the aggregators don't know each
  engine's grid and the engines stay grid-only — policy-at-edge wins over one resample.
- **`add_pausa` signature symmetry (review sweep item):** `add_pausa` currently accepts no
  `pitch_control_cache` and forwards none to its internal `add_obso` call
  (`features.py:5250-5258`) — a shared-surface reuse miss. Since A2 touches this signature
  anyway, add `pitch_control_cache=None` and thread it through.
- `transition_grid` is untouched — the transition-from-xT adapter stays **deferred v2**
  (the xT transition family is per-source-zone `(l·w, l·w)`; OBSO's transition is a single
  ball-conditioned spatial weight — semantic overlap needs its own design; recorded, not built).

### A3. Warn on synthetic defaults (aggregator edge; review V5 mechanics)

New warning category **`SyntheticEPVWarning(UserWarning)`**, used for BOTH warnings below
so consumers can `filterwarnings("always"/"error", category=...)` precisely. **Category
homes pinned (R2-7):** `SyntheticEPVWarning` is defined in `_xt_adapters.py`,
`RunValueCoverageWarning` (§B1.6) in `_run_values.py`; BOTH are re-exported via
`tracking.__all__`, so the one stable consumer import path across releases is
`from silly_kicks.tracking import SyntheticEPVWarning, RunValueCoverageWarning`.

1. When neither `xt=` nor `epv_grid=` is supplied, the public aggregators/factories
   (`add_obso`, `add_space_creation`, `add_pausa`, and the three xfns factories at
   factory-call time) emit `warnings.warn(SyntheticEPVWarning, stacklevel=2)`: "OBSO EPV is
   the synthetic linspace(0.01, 0.3) placeholder ramp — pass xt= (fitted ExpectedThreat) or
   epv_grid= for production surfaces." Under Python's default filter this dedupes to
   once-per-process-per-callsite — **that is the designed behavior**, not an accident: quiet
   by default, and the lakehouse opts into `always`/`error` via the category. The engines
   (`_get_default_grids` and below) stay silent.
2. **PAUSA passthrough hazard (verified `features.py:5248-5258`):** if `add_pausa` receives
   `xt=`/`epv_grid=`/`transition_grid=` **and** the obso columns are already present, it
   warns (same category) that the supplied surface inputs are ignored. This stays a
   **warning, not a ValueError**: the caller who threads the SAME `xt=` uniformly through a
   chained pipeline (`add_obso(xt=m)` → `add_pausa(xt=m)`) hits this branch legitimately —
   the columns already carry the real-xT values — and the library cannot distinguish that
   from the bug case. The category makes the legitimate chainer's filter one line.

Existing tests that call the aggregators with defaults get the expected warning asserted or
filtered — sweep them in the plan ([[feedback_api_change_sweep_ci_scope]]).

### A4. DEFECT-A repair: OBSO orientation (in-scope per review; ADR-028 amendment)

Fix both sub-defects inside the obso kernel, copying the 4.31.0 `pitch_control_at_target`
seam pattern (ADR-028 re-projection at the *sampling/query* seam, never flipping the surface):

1. **Target re-projection.** Per pass action, when the acting team attacks right-to-left in
   the frame convention (`acting_team_attacks_rtl` + the involution `(105−x, 68−y)` —
   `tracking/_action_orientation.py:28-90`), map the action-LTR `end_x/end_y` target into
   frame coordinates before calling `compute_pass_obso`.
2. **EPV direction.** Per pass action, when the acting team attacks RTL, apply the EPV grid
   x-flipped (`[:, ::-1]` — column flip in frame space, the `player_influence:121-122`
   idiom). This applies to the **synthetic ramp too** — away-team obso rows change even for
   callers who never opt into `xt=`. That is the defect repair, not a side effect. The flip
   happens on the (ny, nx) grid the kernel passes down, so `compute_pass_obso` stays
   orientation-blind (engine unchanged). Its contract is thereby made explicit in the
   docstring: **grids and `target_position` are in the frames' coordinate convention** —
   that was always the implicit engine contract; `add_obso` was the caller violating it,
   and the repair lives at the aggregator edge (policy-at-edge).
3. `home_team_id` stays in the signature but orientation is keyed on the frames'
   `team_attacking_direction` via `acting_team_attacks_rtl` (id-join via `align_join_keys`,
   ADR-019), exactly like `pitch_control_at_target`. The dead-parameter status ends
   (it remains the documented LTR-consistency input for the `_validate`-style checks and any
   absolute-frame fallback the implementation needs; if it stays genuinely unused after
   implementation, document that in the docstring rather than breaking the signature).

**Consequences (flagged, Hyrum):** obso columns (`obso_actual/peak/optimal` + 9 xfns cols)
and — via the shared ramp direction — space-creation values change for **away-team rows**
(~half of all rows) relative to 4.51.0, on the synthetic path as well. obso/space-creation
xfns are in **no default list** → no forced VAEP retrain; opted-in callers self-trigger. The
lakehouse's planned single `fct_action_context` recompute absorbs it. ADR-041 records an
**ADR-028 amendment**: obso moves out of the "self-reconciling" list into the
reprojected-at-query-seam family; CLAUDE.md's ADR-028 paragraph is corrected in the same PR.

**Gates:** obso joins `tests/tracking/test_action_ltr_mirror_invariance.py` (same physical
situation under a frame mirror → identical action-LTR obso outputs) **plus** a ground-truth
asymmetric fixture (away-team pass toward a known high-EPV cell → the higher obso value must
land at the correct goal; mutation: removing the flip goes red). Mirror-invariance alone is
insufficient per [[feedback_symmetry_test_insufficient_pin_ground_truth]].
**Production-scale gate (review V11):** a new owner-gated e2e on a real WC2022/GS match —
pre-repair, home vs away obso distributions are structurally asymmetric (the away side is
valued toward the wrong goal); post-repair the away/home mean-obso ratio must land inside a
band pinned at probe time (probe-then-pin, like §B3's bands; expected ≈ [0.7, 1.4] — team
strength keeps it from 1.0). This measures the repaired defect at production scale in one
assertion.

### A5. DEFECT-B repair: `player_influence` threat grid via the adapter

Replace `_player_influence.py:117-118`'s raw `interp(pc.grid_x, pc.grid_y)` with
`epv_grid_from_xt(xt, pc.grid_x, pc.grid_y)` (the away x-flip at `:121-122` stays). This
single-sources the inversion handling and fixes the latent y-mirror. Behavior change is
small (fitted xT is near-y-symmetric) but real → `player_influence` columns
(`player_influence_xfns` 21 cols + `add_player_influence` 7 cols) shift slightly; in no
default list → no forced retrain; lakehouse re-materializes in the same recompute. Golden:
on the asymmetric fixture grid, `player_influence`'s threat weighting must match the
physically-correct orientation (red before the fix).

**Recorded observation (review, smaller note — not changed this PR):** the x-flip stays
keyed on `same_id(attacking_team_id, home_team_id)`, which is correct only under
home-attacks-right frames. That IS the contract for `convert_to_frames`-produced frames;
the implementing session verifies whether `player_influence`'s docstring states it and adds
the contract sentence if absent. Re-keying on `team_attacking_direction` (the A4 style) is
a recorded candidate for a later orientation-consistency pass, not this PR's scope.

### A6. Opponent mirror upgrade (space_creation)

Change `_space_creation.py:223-224` from `np.flip(..., axis=1)` to a point reflection
`np.flip(..., axis=(0, 1))` for both transition and EPV. Expected byte-identical on the
synthetic path (both synthetic grids y-symmetric — verified §A0.4); correct for injected
asymmetric grids. **TDD ordering (review):** write the synthetic-path byte-identity gate
(exact equality of `add_space_creation` output, no tolerance) BEFORE making the flip change —
it doubles as the registration-offset detector: if `_interpolate_grid`'s node registration
makes the resampled Gaussian not exactly y-symmetric, the gate fails first and that is a true
finding to resolve, not to paper over. Plus an injected-asymmetric-grid test showing the
opponent surface reflects both axes.

### A7. Tests & gates summary (Part A)

- Adapter golden (asymmetric, orientation-pinned; mutation-verified; non-uniform-grid_y
  case) — §A1.
- `_require_fitted_xt` cross-package parity sentinel — §A1 (V4).
- Discriminating behavioral gate: `add_obso(xt=fitted_real)` ≠ `add_obso()` (synthetic) on
  the same frames; non-vacuity: the diff must exceed tolerance on ≥1 pass row
  ([[feedback_invariance_test_needs_discriminating_power]]).
- Orientation ground-truth + mirror-invariance registration + owner-gated home/away e2e
  band — §A4.
- player_influence orientation golden — §A5.
- Synthetic-path byte-identity for the mirror upgrade (written FIRST) — §A6.
- Mutual-exclusion (`xt=`+`epv_grid=` → ValueError), fail-loud triple (str/None/unfitted),
  `SyntheticEPVWarning` category behavior (both warn sites; filterable; asserted via
  `pytest.warns(SyntheticEPVWarning)`), PAUSA ignored-inputs warning.
- Purity: add an `xt=`-supplied variant to the existing `tracking:add_obso` /
  `tracking:add_space_creation` / `tracking:add_pausa` entries (input-mode branch —
  ADR-033 contributor contract, `tests/test_add_star_purity.py:268+`).
- Existing auto-gates (liveness, dup-action_id, id-dtype, nan-safety, Examples) already
  enumerate these surfaces; no new registration beyond the purity variants.

**Flags:** C4-free (no new aggregator/model/backend). No forced retrain. NOTICE: no new
entry (Spearman 2018 already present); ADR-041 (incl. ADR-028 amendment + the V2 deferred
fork) + CLAUDE.md correction.

---

## Part B — TF-35 v1: off-ball-run valuation (value-roles arm)

### B0. Scope

Ship the **value-roles arm only** (target / disruptive / space-creation credit — the
Soccermatics 16.2 production-club method with worked anchors Rashford 0.07, Shaw 0.11).
Geometric typing (overlap/underlap/far-side/advance/support) and MSC rotation archetypes
stay deferred inside the TODO entry (Esposito et al. 2026 supplies no computable geometry;
Gradient's zones are proprietary). SkillCorner's native 10-type run labels = future
**external validation only** (native route; the kloppy gateway discards `visibility`) — not
in this PR.

Course definitions being operationalized (plan doc §W1):

- **Target run** — credit the receiving runner with MAX(pitch-control × xT) over the space
  the run opened, NOT the realized reception value.
- **Disruptive run** — a sprint coincident with a pass where the runner does not receive;
  credited by the same space-opened value. (Crediting by measured defender displacement was
  considered and **rejected by the course** — defenders must react because the pass *could*
  go there.)
- **Space-creation credit** — when the pass to a different player succeeds, each disruptive
  runner is additionally credited with the enabled pass's value.

### B1. New module `tracking/_run_values.py` — components

**`RunValuationParams`** (frozen dataclass, `__post_init__` validation):
`pre_seconds: float = 1.5` · `min_displacement_m: float = 3.0` (both TF-4-aligned) ·
`min_peak_speed_ms: float = 5.56` (20 km/h — the HSR floor from the SkillCorner/course band
constants; full sprint 25 km/h would starve detection over a 1.5 s window; review-confirmed
default, e2e band probe arbitrates) · `region_influence_floor: float | None = None` (see the
method-portability rule below) · `pitch_control_method: Literal[...] = "spearman"`.

**Floor method-portability (review V7 + R2-5 — fail-loud, not fail-documented):**
`player_surface` returns raw per-player influence whose scale is method-dependent — a
single floor calibrated on one method silently means something different under another
(vivid case: voronoi's decomposed influence is binary {0,1}, `_voronoi.py:85`, so a
universal 0.1 floor silently degenerates to "owns the cell" — a different metric). **v1
rule: `region_influence_floor=None` resolves at validation time via a per-method
calibration table (`{"spearman": 0.1}`); any other method without an explicit
caller-supplied floor RAISES** ("no calibrated floor for method X — pass
region_influence_floor explicitly"). Resolution mechanics (frozen-dataclass
`__post_init__` vs a `resolved_floor()` accessor) decided at plan time. Normalizing to
per-cell share of team influence before thresholding is the recorded scale-free v2 fork
(rejected for v1: noisy ratios where team control is low). The §B3 probe includes a
**3-point floor sensitivity check (0.05 / 0.1 / 0.2)** so the pinned band is knowingly
calibrated (review Q3).

**`detect_off_ball_runs(actions, frames, *, home_team_id, params=None)`** → long-form
DataFrame, one row per `(action_id, player_id)` qualifying run. Detection is
TF-4-semantics-compatible. **Candidacy is SHARED, not duplicated (R2-2):** the pure
candidacy predicate (same-team / non-actor / non-GK / NaN-drop / dead-ball / <2-frame) is
extracted from the TF-4 kernel into a private leaf helper in `_off_ball_runs.py`, consumed
by BOTH the TF-4 kernel and `detect_off_ball_runs` — by the house 3rd-consumer rule this PR
IS the third consumer (kernel + detect + the identity gate), so extraction triggers now
rather than being silently deferred. TF-4 is Chesterton-cautioned, not frozen; the
extraction is output-identical by construction and gated by TF-4's existing regression
suite + liveness entries. Only the leaf predicate unifies — the loop structure, windowing,
and displacement logic stay TF-4-local (kernel) vs run-values-local (detect), which is the
duplication the identity gate below sentinels. Rules:

- Window: `slice_around_event(pre_seconds, post_seconds=0.0)` anchored on the action clock
  (`_off_ball_runs.py:100`; ADR-017 period-relative).
- Candidacy (inherited exactly): same-team-as-actor, non-actor, non-goalkeeper, NaN
  positions dropped, ball rows excluded, dead-ball-tagged actions → no rows
  (`_off_ball_runs.py:105-137`). GK runs are therefore out of scope v1 — documented.
- Qualifying run: first-vs-last displacement ≥ `min_displacement_m` **AND** peak frames
  `speed` in-window ≥ `min_peak_speed_ms` (the new sprint gate; TF-4 is displacement-only).
  `speed` NaN rows are excluded from the peak; if a candidate's speeds are all-NaN, fall
  back to displacement-rate `disp / observed_span` compared against the same threshold.
  **Documented bias (review V9):** the fallback compares a MEAN-rate against a PEAK
  threshold — a strictly harsher bar, so providers with missing speed columns detect
  conservatively fewer runs. Stated in the docstring AND the lakehouse handoff (it would
  otherwise surface as a mysterious per-provider run-count skew); the threshold is NOT
  scaled for the fallback (an arbitrary factor would trade a documented bias for an
  undocumented one).
- Emitted per run (geometry + kinematics ONLY — role/value columns are added by the
  valuation stage, keeping detect standalone-testable): `game_id, period_id, action_id,
  player_id, team_id`, `run_start_x/y, run_end_x/y` (**action-LTR**, ADR-028-reprojected —
  position outputs do not get TF-4's flip-invariant-scalar exemption), `displacement_m`,
  `duration_s` (observed span, NOT the fixed window), `mean_speed_ms`
  (displacement/observed-span — deliberately NOT TF-4's fixed-window rate; different name,
  different semantics, documented), `peak_speed_ms`, `toward_goal`
  (bool; defined in action-LTR as `run_end_x > run_start_x` — attacked goal at x=105,
  orientation-consistent with the emitted positions), id dtypes per source
  (ADR-019/ADR-027: nullable-tolerant, positional joins).

**Identity gate vs TF-4 (review V10 + R2-2 — sets by construction, no test copy):** the
round-1 draft had the test reimplement the candidacy rules inline — a THIRD copy whose
kernel link was only count-pinned, reopening the swap-drift hole one level down (R2-2).
Resolved without any test-side reimplementation: the expected runner set comes from a
**hand-built ground-truth fixture** (known players, known displacements — expectation by
construction, not computation). The gate asserts, on that fixture with `min_peak_speed_ms=0`
and TF-4's `pre_seconds`/`min_displacement_m`: (i) `detect_off_ball_runs`' per-action
`(action_id, player_id)` set == the hand-written expected set, AND (ii) the TF-4 kernel's
`n_off_ball_runners_pre_window` == the expected set's size — both implementations pinned to
the same external truth, so a matched candidate-swap in either diverges from the
hand-written set. On broader generated inputs a count-pin (detect row count vs kernel
count) remains as the cheap wide-net check for the still-duplicated displacement logic.
Discriminator: with the sprint gate ON and a deliberately-slow qualifying-displacement
runner in the fixture, detect's set must differ from the expected sprint-off set
(non-vacuity).

**`value_off_ball_runs(runs, actions, frames, xt, *, links=None, pitch_control_cache=None, params=None)`**
→ the same long-form table + `run_role` (`"target"`/`"disruptive"`), `run_value`,
`enabled_pass_credit` columns. Mechanics per completed pass/cross action:

1. **Domain:** actions with `type_id ∈ {pass, cross}` AND `result_id == success` (the course
   method is defined within completed-pass phases; failed passes have no observable receiver
   in event data). Non-domain actions contribute no valued runs.
2. **Receiver:** `spadl.utils.resolve_next_touch_receiver` (`spadl/utils.py:1292` — next
   same-team touch, skips `non_action`+`foul`, positional Int64 contract, never
   float64-upcasts). Receiver ∈ detected runners → that run's `run_role = "target"` +
   `is_receiver = True`; every other detected runner on that action → `"disruptive"`.
   **Unresolved receiver on an otherwise in-domain action ⇒ the whole action is
   off-domain** (target/disruptive cannot be told apart) — its runs get no role/value and
   all four wide columns are NA.
3. **Pitch-control surface:** ONE decomposed surface per action at the linked pass frame —
   `PitchControlCache.surface(frame, attacking_team_id=acting_team, decompose=True)`
   (`pitch_control/_cache.py:46-55`; note the method is `surface`, **not** `get_surface`).
   Frame resolution honours the `links=` pre-link and a caller-supplied
   `pitch_control_cache` (house kwargs). One surface serves all runners of that action.
4. **Threat grid:** `epv_grid_from_xt(xt, pc.grid_x, pc.grid_y)` (Part A adapter — correct
   y-orientation), x-flipped `[:, ::-1]` when the acting team attacks RTL in frame
   convention (the corrected player_influence idiom). PC queries stay in **frame**
   coordinates; only *emitted positions* are action-LTR (per §B1 detect).
5. **Space opened (the pinned operationalization):** the runner's controlled region at the
   pass frame — cells where `pc.player_surface(player_id)` ≥ `region_influence_floor`
   (`_surface.py:156-170`). `run_value = max over region of (pc.surface × threat_grid)`
   where `pc.surface` is the **team** control surface (`_surface.py:34-49`, (ny, nx) in
   [0, 1]); empty region → `0.0` (ran, opened nothing).
   **Team-vs-player surface (review V8, decided explicitly):** module 16.2 says
   "pitch control × xT" without disambiguating team vs per-player control; team control is
   chosen because it keeps `run_value` probability-interpretable (bounded by threat, "P(we
   keep the ball there) × value there") where raw per-player influence is method-scaled
   (V7); the attribution concern is mitigated structurally — a cell only enters the region
   if the RUNNER's own influence exceeds the floor there, so credited cells are
   **runner-relevant** by construction (R2-6: not runner-*dominated* — a teammate can hold
   higher influence in the same cell; the §B3 probe's teammate-overlap check is the actual
   guard). `player_surface × threat` is the recorded alternative, revisited if the e2e
   probe shows teammate-overlap inflation. Cited in the docstring.
6. **Runner absent at the pass frame (review V6 — crash specified away):** detection
   candidacy is window-based, valuation is single-frame — a runner visible in the window
   but NaN/absent at the linked pass frame is NOT in the decomposed surface, and
   `player_surface()` raises `ValueError` (`_surface.py:167-169`); on SkillCorner
   visibility gaps this is a when, not an if. Behavior: membership-check first
   (ADR-019-safe id comparison against `pc.player_ids` — never try/except as flow
   control), absent runner → the run row SURVIVES with `run_value = NaN` (unmeasurable,
   not measured-zero — the NaN honesty rule) + one aggregate warning per call with the
   skipped count, under its own category **`RunValueCoverageWarning(UserWarning)`** (R2-4 —
   the V5 lesson generalized: any warning a production pipeline may filter or escalate
   gets a category; `SyntheticEPVWarning` is semantically wrong for coverage). Wide sums
   use `skipna`; counts include unvalued runs (the run happened; its value didn't
   resolve). Contract test: fixture player present in-window, missing at the event frame
   → NaN run_value, no crash, `RunValueCoverageWarning` fired.
7. **Enabled-pass credit (disruptive only):**
   `enabled_pass_credit = max(0, xt_at_points(xt, end) − xt_at_points(xt, start))` of the
   completed pass, computed entirely in action-LTR SPADL coordinates (xT is fit on SPADL
   actions — no reprojection needed on this term). Each disruptive runner carries the full
   credit in the long-form table (the course credits players individually).

**`add_off_ball_run_values(actions, frames, xt, *, home_team_id, links=None, pitch_control_cache=None, params=None)`**
(`@nan_safe_enrichment`) → wide per-action columns:

| column | dtype | semantics |
|---|---|---|
| `run_value_target` | float64 | receiver's run value; **0.0** if the completed pass's receiver made no qualifying run; NaN off-domain (failed pass, non-pass, dead-ball, unlinked, unresolved receiver) |
| `n_disruptive_runs` | Int64 | count of disruptive qualifying runs; 0 on-domain with none; NA off-domain |
| `run_value_disruptive_sum` | float64 | skipna sum of disruptive run values |
| `n_valued_disruptive_runs` | Int64 | count of disruptive runs whose `run_value` resolved (≤ `n_disruptive_runs`; the gap = coverage-skipped runs, R2-3) |
| `run_value_enabled_pass` | float64 | the enabled-pass credit **once per action** (review Q4 — NOT a k-scaled sum, which re-encodes `n_disruptive_runs` and invites downstream mis-aggregation); consumers multiply by the count if they want the total; the per-runner credit lives in the long-form table |

**Mean-bias rule (R2-3):** `run_value_disruptive_sum / n_disruptive_runs` is biased
downward exactly where visibility gaps are worst (skipna numerator over an all-runs
denominator, compounding the V9 per-provider skew) — the correct wide-only mean divides by
`n_valued_disruptive_runs`; stated in the docstring AND the lakehouse handoff.
`n_valued_disruptive_runs` is coverage provenance: it is **excluded from the xfns factory
output** (the packing provenance-exclusion precedent), keeping the factory at 4 cols.

NaN-vs-0 rule: **0 = measured absence** (domain action, nothing qualified), **NaN/NA =
unmeasurable** (off-domain) — the packing honesty convention. Provenance columns follow the
idempotent-merge rule (skip when present). Column-family prefix `run_value_*` deliberately
avoids colliding with the TF-41 `space_created_m2` family — the course's "space-creation
credit" is named `run_value_enabled_pass` for this reason.

**`off_ball_run_value_xfns(xt, *, home_team_id, params=None, ...)`** — opt-in factory,
4 cols × 3 slots = 12, `_frame_aware = True` marker + `__name__ = "off_ball_run_values"`
(the `off_ball_context_xfns` template, `features.py:1645-1694`); Int64 columns converted
explicitly to float64-with-NaN in the factory output (the `.to_numpy()`-on-Int64
object-array quirk at `features.py:1687` is NOT inherited). **Result-leakage warning
(F4-class, stronger than packing):** the whole family is conditioned on the action's OWN
`result_id == success` — the docstring carries the packing-style `.. warning::` (never into
HybridVAEP-class consumers without a0 exclusion); a result-free variant (intended-receiver
inference) is a **recorded fork, not built**. In **no** default xfn list.

**Atomic mirror:** re-export via `atomic.tracking.features` with the SK-xT-2-style
type-aware synthesized frame if needed; if the standard implementation operates purely on
columns the atomic frame also carries (`end = x+dx` synthesis, packing precedent), the
mirror is the thin re-export + synthesis wrapper. Decided at plan time against the packing
implementation; the spec requirement is: atomic mirror exists, numeric-only, and registers
in the same gates.

### B2. Gate tax (every registration this PR must wire)

- Liveness: entries for `add_off_ball_run_values` (+ atomic) in
  `tests/tracking/test_aggregator_column_liveness.py` — **requires a fixture extension**:
  the shared 5-window fixture's fastest off-ball candidate moves ≈4.95 m/s stored /
  ≈4.5 m/s displacement-rate (verified `test_aggregator_column_liveness.py:151,167,134`) —
  below any sprint gate, so the new columns would be born dead. Extend ≥2 windows with one
  sprinting non-receiver teammate (stored vx ≥ 6.5 m/s + matching displacement, varied per
  window) and make ≥1 window's pass receiver a qualifying runner (varied values → the
  float non-constant check passes). The liveness gate is structural (non-null +
  non-constant), not golden — other aggregators' entries tolerate the fixture change;
  **an explicit plan step (not a hope) verifies no golden/snapshot test reads this
  fixture** before the extension lands (review Q6). Alternative (bespoke
  `_run_shot_goalmouth`-style window builder) rejected: honestly-live columns beat a
  special case.
- Purity: `"tracking:add_off_ball_run_values"` with ≥2 variants (internal-link vs
  supplied `links`+`pitch_control_cache`) + the atomic entry.
- id-dtype invariance + dup-action_id (xfns) + nan-safety (decorated) + provenance-skip +
  Examples: auto-enumerating — registration only.
- Absence guard: extend the auto-discovering leakage-guard skeleton
  (`tests/tracking/test_packing_xfns_leakage_guard.py:31-75` pattern): forbidden substring
  `"run_value"` absent from every default list + `__name__` pin on both std/atomic
  factories + the non-vacuity floor ([[feedback_opt_in_leaky_xfn_needs_executable_absence_guard]]).
- Long-form primitive contract tests (its own — the auto-gates only cover `add_*`):
  empty-input schema, dtype contract, action-LTR orientation ground truth, dead-ball
  exclusion, <2-frame skip, sprint-gate boundary, absent-at-event-frame NaN (V6).

### B3. Validation

- **Ground-truth asymmetric fixture** (pins the whole chain): away-team completed pass;
  one sprinting receiver whose controlled region contains a known high-threat cell; hand
  computed `run_value_target` expected value; mirror the frame → identical action-LTR
  outputs (mirror-invariance necessary-not-sufficient).
- **Identity gate vs TF-4** (§B1, set-equality form) with its discriminator.
- **Owner-gated WC2022/GS e2e** (packing e2e pattern): distribution plausibility bands —
  per-match mean `n_disruptive_runs` strictly interior to a band set at spec-execution
  probe time; `run_value_*` magnitudes in the course-anchored order of magnitude
  (Rashford 0.07 / Shaw 0.11 ⇒ gate median run_value ∈ [0.005, 0.2] as the initial band;
  tightened after the first probe, before merge — bands recorded in the test, per
  [[feedback_empirical_validation_before_ship]]). The probe includes the 3-point
  `region_influence_floor` sensitivity check (0.05/0.1/0.2 — review Q3).
- SkillCorner native run-subtype label agreement: deferred (recorded next step, needs the
  native-route ingestion that is out of scope here).

### B4. Flags

New aggregator → **C4 29→30** (+ the C4 Phase-4 regeneration in /final-review). In no
default list → **no retrain trigger**. NOTICE: Sumpter/Twelve (Soccermatics Pro module
16.2, practitioner anchor — stated as such) + Esposito et al. 2026 (framing only) entries;
docstrings cross-link per ADR-005. Lakehouse: 5 new `fct_action_context` candidate columns
+ the long-form run table as an optional separate mart (their call) + the V9 per-provider
detection-bias note and the R2-3 mean-bias rule in the handoff.

### B5. Explicitly out of scope (recorded)

Geometric/rotation taxonomies; SkillCorner label ingestion; GK runs; failed-pass
(intended-receiver) valuation; defender-displacement crediting (course-rejected); TF-4
loop/windowing refactor (the candidacy LEAF predicate IS extracted this PR per R2-2 — the
3rd-consumer rule triggered; the loop structure stays duplicated); transition-grid-from-xT
(Part A v2);
Tigres "pockets" reception quality (shares the `secured_reception` seam — separate item);
per-cell share-normalized region floor (V7 v2 fork); `player_surface`-based run_value
(V8 alternative); `interpolator()` oriented variant (A1b fork, 5.0-candidate);
player_influence x-flip re-keying (A5 observation).

### B6. Space-opened operationalization — alternatives recorded

1. **Per-player-PC region at the pass frame (CHOSEN):** one decomposed surface per action
   serves all runners; faithful to "credit the potential of the opened space"; cheapest.
2. Endpoint disc of radius r: rejected — ignores control (values space the runner doesn't
   command).
3. Team-PC delta over the run window (surface at pass frame minus at run start): closest to
   literally "opened", but 2× surfaces per action + window-hysteresis sensitivity;
   **deferred as a future `method=` flavor**, ADR-005 §8 naming ready.
4. Defender displacement: **rejected by the course itself** (kept only as documentation).

---

## Part C — release mechanics & handoff

- **Version 4.52.0** (pyproject + `__init__` + uv.lock + CHANGELOG + TODO in lockstep —
  version-bump hard gate). **PR-S119**, single squash-merge branch off main @ `ec543cc`;
  commits ordered A-then-B pre-squash (§header). **ADR-041 + ADR-042** written at release;
  CLAUDE.md updated (ADR-028 self-reconciling list correction, C4 count 30, new module
  registrations).
- Order of work inside the PR: Part A adapter + repairs first (Part B consumes the adapter),
  then Part B; TDD throughout (§A6 gate-before-change ordering); /final-review incl. C4
  Phase 4 before merge.
- TODO grooming at release: delete the two shipped rows (delete-don't-annotate); the
  deferred forks recorded here live in this spec + the ADRs, not as TODO annotations.
- **Lakehouse handoff (with the recompute):** (1) new columns: `run_value_target`,
  `n_disruptive_runs`, `run_value_disruptive_sum`, `n_valued_disruptive_runs`,
  `run_value_enabled_pass`; (2) changed values on away-team rows for obso + space-creation
  (defect repair — applies even without opting into `xt=`) and small shifts in
  player_influence (y-fix); (3) to get real-xT OBSO they MUST pass `xt=` (or `epv_grid=`) —
  omitting it keeps the synthetic ramp and warns via the filterable `SyntheticEPVWarning`
  (stable import: `from silly_kicks.tracking import SyntheticEPVWarning,
  RunValueCoverageWarning`); (4) the V9 NaN-speed detection bias is per-provider — expect
  conservative run counts where `speed` is missing; (5) **mean-bias rule (R2-3): never
  compute mean disruptive run value as `run_value_disruptive_sum / n_disruptive_runs`** —
  divide by `n_valued_disruptive_runs`, or use the long-form table; (6) fold into the same
  single recompute as the already-queued 4.49–4.51 triggers.

## Review log

**Round 2 — 2026-07-18, lakehouse session.** Verdict: **APPROVE** — round-1 items faithfully
applied or coherently deviated; nothing blocking. Findings applied: **R2-1** the V3
deviation's "cannot be re-pointed" justification corrected to the true cost/benefit reason
(M5 protects v1 *behavior*, not file bytes — PR-S110's byte-identical shim inside
`_xt_gk.py` is the precedent; a fabricated fence is as bad as a removed one) — decision
unchanged; **R2-2** the round-1 set-gate had silently created a THIRD copy of the TF-4
candidacy rules while citing the 3rd-consumer rule — resolved by extracting the pure
candidacy leaf predicate (shared by kernel + detect) AND replacing the test's inline copy
with a hand-built ground-truth fixture (expectation by construction — stronger than both
options the review offered); **R2-3** new `n_valued_disruptive_runs` wide column
(xfns-excluded coverage provenance) + the mean-bias rule in docstring and handoff; **R2-4**
`RunValueCoverageWarning` category; **R2-5** floor made fail-loud (`None` default resolved
via per-method table; unlisted method without explicit floor raises — the voronoi binary
{0,1} case argued it); **R2-6** "runner-dominated" → "runner-relevant"; **R2-7** category
homes pinned with one stable `silly_kicks.tracking` import path; **R2-8** double-resample
rationale recorded in §A2. Spec is APPROVED for the writing-plans stage.

**Round 1 — 2026-07-18, lakehouse session.** Verdict: approve direction; both defect claims
independently re-verified real. All requested changes applied: **V1** adapter flips the data
(`flipud`) not the output — symmetric-grid_y precondition dropped; **V2** `interpolator()`
docstring disarmed + oriented-variant fork recorded; **V4** cross-package guard parity
sentinel; **V5** `SyntheticEPVWarning` category (answers Q5; PAUSA stays a warning — the
legitimate uniform-`xt=` chaining case makes a raise wrong); **V6** absent-at-pass-frame
runner → NaN + counted warn (membership check, not exception flow); **V7** floor pinned
spearman-calibrated + share-normalization recorded fork; **V8** team-surface choice made
explicit with the runner-dominated-region mitigation + recorded alternative; **V9** fallback
bias documented, threshold not scaled; **V10** identity gate strengthened counts→sets;
**V11** Part A owner-gated home/away e2e band added; smaller notes: `add_pausa`
`pitch_control_cache` threading, A5 flip-keying contract note, A6 TDD ordering, Q4 wide
column changed to per-action `run_value_enabled_pass`. **Deviations (2):** V3 re-homing of
`_grid_value` rejected — `_xt_gk.py` is M5-FROZEN with a byte-stability gate; the adapter
instead uses the same sanctioned lazy `_get_cell_indexes` import and becomes the canonical
forward home (one story, recorded in ADR-041). Q1 two-PRs preference not taken — the owner's
stated batching preference stands; commit-ordering mitigation adopted. Q2 (5.56 default),
Q3 (sensitivity probe), Q6 (fixture extension + explicit no-golden-reads step), Q7 (two
ADRs) adopted as recommended.
