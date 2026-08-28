# ADR-077: FOV-observability — a first-class visibility signal + single-sourced metric companions

| Field | Value |
|---|---|
| **Date** | 2026-08-28 |
| **Status** | Accepted |
| **Version** | 4.99.0 |
| **PR** | PR-S170 |
| **Deciders** | Karsten S. Nielsen (owner); lakehouse session (consumer / reviewer) |

## Context

StatsBomb-360 freeze-frames contain only the players inside the broadcast field of view, delivered as
a per-action `visible_area` polygon (`providers/statsbomb`; `parse.py` builds each frame from
`ff["freeze_frame"]` only). Every **region- or aggregate-based** tracking metric therefore *silently
under-reports* on SB360: a defender outside the FOV lowers a congestion count, shifts a
defensive-line mean, or shrinks a pressure value, and nothing in the output says so.

ADR-062 solved this for exactly **three** columns on **one** aggregator
(`add_action_context(..., visible_area=...)` → six opt-in companion columns) via a private, hand-coded
helper (`_append_visibility_companions`). Extending that pattern per-aggregator means N copies of
"build this metric's region → `classify_region_observation` → emit two columns", each with its own
purity / liveness / SB360-audit / glossary registration — scattered logic that rots, and no answer at
all for the aggregate-position metrics. Separately, `_space_creation.py`'s opponent-perspective
softening was gated on a **velocity-marker proxy** (`velocity_unavailable_by_design`) explicitly
flagged in the ADR-054 amendment as "PRAGMATIC PROXY … migrate to a real frame-level FOV signal when
one exists" — right only because velocity-less and FOV-cropped *coincide in today's providers*, not
because it measures FOV.

The forcing function is that SB360 is now an enabled first-class provider (ADR-062, 4.84.0), so the
under-reporting is live on real analysis, and the proxy is a known correctness gap waiting for exactly
this signal.

## Decision

Make FOV-observability a **first-class, single-sourced concern**: a public `validate_fov` /
`FovDiagnosis` frame-set diagnostic; a declarative observability registry
(`tracking/_fov_registry.py`) with **one** observability model — the observed-*area* fraction of a
convex region — into which ADR-062's helper is retired byte-identically; opt-in `visible_area`
companions on **seven newly-companioned aggregators this cycle** (eight companioned in total,
including `add_action_context`, whose ADR-062 companions this cycle retires into the shared engine);
retirement of the `space_creation` velocity-FOV proxy onto the real signal; and a two-axis
completeness gate. **Everything is additive and opt-in** — primary
feature columns stay byte-identical, companions appear only when the caller supplies `visible_area`,
**no VAEP retrain**, no new action-coupled aggregator (C4 count unchanged).

### S1 — one observability model: observed *area* of a convex region

There is exactly ONE companion model, because the FOV signal is *purely a polygon*. A per-contributor
"roster fraction" is a false dichotomy on a freeze-frame: a cropped-out player is **absent from the
frame entirely** (he is in neither the numerator nor the denominator), so the fraction is identically
≈1.0 — silent in exactly the biased case it was meant to flag. Every companioned metric therefore
declares a **convex region** and the companion is `region_observed_fraction(polygon, region)` — a
*tight ROI* for count/density/distance metrics (disk / triangle / oval / band) and a *broad fixed
zone* for aggregate-position metrics. A right-half crop yields 0.5, never a lie near 1.0. There is
**no `point_observed` / per-contributor path** in the companion engine.

### Component 1 — `validate_fov` / `FovDiagnosis`

The fourth member of the `validate_time_base` (ADR-017) / `validate_velocity_regime` (ADR-054) /
`validate_id_dtypes` (ADR-019) diagnostic family, split across `tracking/schema.py`
(`FovDiagnosis`, `FOV_REGIME_VALUES`) and `tracking/utils.py` (`validate_fov`). It consumes only what
it reads — the per-action `visible_area` polygon table — with `on_mismatch: Literal["warn", "raise",
"ignore"] = "raise"`, guards on key presence not row count, and never raises on empty input. Regimes:
`full_coverage` (every action's polygon ≈ the whole pitch), `fov_cropped` (polygons present but
partial — the SB360 case), `absent` (no usable polygon anywhere), `mixed` (the fail-loud case,
reachable only at `0 < n_full < n`), and `empty` (the empty-input regime — a zero-row `visible_area`
table, reported rather than smuggled into another regime and, like its diagnostic siblings, never
raising). FOV regime is a property of the *whole* frame set (verbatim
reasoning from `VelocityRegimeDiagnosis`); the per-action fractions that vary row-to-row are the
companions, not this.

### Component 2 — the declarative registry + byte-identical ADR-062 retirement

`tracking/_fov_registry.py` is the ONE engine. An `ObservabilityEntry(column, region, covers=())`
binds a raw metric column to a convex-region builder `region(i, ctx) -> (M, 2) | _NO_REGION`;
`OBSERVABILITY_REGISTRY` lists the entries per aggregator; `append_observability_companions(...)`
emits the `<column>_observed_fraction` / `_observed_source` pair for each via
`classify_region_observation` — one code path for every metric. Per-call params (radii, pressure
method) flow through `RegionCtx.extras`, so every companioned column is a STATIC registry entry.
`_NO_REGION` is an identity sentinel tested with `is`, never `==` (an array `==` raises under
`if <array>:`). The module is neutral by construction — it imports only `_visibility`, `_kernels`,
`id_compat`, `_polygon`, `spadl.config`, and reaches neither `pitch_control` nor `_das`, so those
layers may depend on it without a cycle.

ADR-062's three `add_action_context` ROIs become three registry entries and
`_append_visibility_companions` becomes a thin call into the shared engine; the emitted columns and
values are **byte-identical** (parity-gated). `REGION_OBSERVATION_SOURCE_VALUES` is reused unchanged.

### Component 3 — companion coverage across the region/aggregate family (Tasks 3–6)

The table below is the **full companioned set — eight aggregators**. `add_action_context` is the
ADR-062 incumbent (its companions are retired into the shared engine this cycle, byte-identical); the
other **seven are newly-companioned this cycle** and each *newly* gains an opt-in `visible_area:
pd.DataFrame | None = None` kwarg (default `None` → today's output byte-identical; `add_action_context`
already had the kwarg from ADR-062):

| Aggregator | Companioned column(s) | Region |
|---|---|---|
| `add_action_context` | `nearest_defender_distance`, `receiver_zone_density`, `defenders_in_triangle_to_goal` | *retired from ADR-062* — nearest-disk / receiver-disk / triangle-to-goal (tight ROI) |
| `add_pressure_on_actor` | `pressure_on_actor__andrienko_oval` | the faithful Andrienko oval, as a convex 24-gon (Trap 1) |
| `add_packing` | `packing_made`, `packing_net`, `packing_goal_threat` (the three region-COUNT members) | the full-height passer→receiver x-band |
| `add_defensive_line` | `defensive_line_x` | **fixed action-LTR zone (A)** — the defended third `x∈[70,105]` |
| `add_team_shape` | `team_shape_centroid_{x,y}_{attacking,defending}` (FOUR cols → TWO role companions via `covers`) | **fixed action-LTR zones (A)** — attacking role's own half `[0,52.5]`; defending role's own half `[52.5,105]` |
| `add_player_influence` | `off_ball_xt_team` | **fixed action-LTR zone (A)** — the attacking half `[52.5,105]` |
| `add_xt_gk` | `xt_gk_pressure`, `xt_gk_pev` | the pressure ROI **method-dispatched** on `pressure_method` (see below) |
| `add_defensive_credit` | one per-action rollup covering `defensive_credit_net`/`defensive_credit_minus`/`n_defensive_credits` | **resolution-mode-aware** per-credit region, magnitude-weighted rollup (see below) |

### Design (A) — fixed action-LTR zones for aggregate-position metrics, NOT `goal_map`

The plan originally specified a `goal_map`-keyed zone (defended goal → defended third; halfway line →
own / attacking half). During implementation this was **verified to mis-orient** and was replaced by a
**fixed action-LTR zone keyed only on the column's ROLE**. The reason is decisive and is the S1
regression this cycle exists to prevent: `defensive_line_x`, `team_shape_centroid_*`, and
`off_ball_xt_team` are emitted in the **SPADL action-LTR frame** (ADR-028 re-projects them there — the
acting team attacks x=105), which is the *same* frame the `visible_area` polygon lives in (the only
supplier, SB360, is action-LTR). So the defended end is **FIXED**: the acting team attacks the HIGH
end for every action, uniformly. A `goal_map`-keyed zone returns **FRAME-coordinate** ends and would
land on the OPPOSITE end from the action-LTR polygon for every **away-possession** action — its
observed fraction would then be measured against the wrong half, an S1 *silent* failure (a plausible
fraction from a computation that had not happened, the recurring silent-null shape). The fixed-zone
builders keep the `(i, ctx)` engine signature but consult NEITHER argument — the zone is geometry, not
player-derived, which is also the N2 frame-independence invariant (a zone drawn around the observed
players is by construction inside the FOV and collapses back to ≈1.0).

### The `xt_gk` method-dispatched ROI + composite exemption (M1, N3)

`compute_xt_gk` derives both `rho` (`xt_gk_pressure`) and the PEV forward gain (`xt_gk_pev`) through
`pressure_on_actor(method=params.pressure_method)` centred on the RESOLVED GK origin, so the
observability region is the same region that method integrates over, dispatched per action from
`ctx.extras['pressure_method']`: `andrienko_oval` → the oval; `link_zones` → the convex outer bound of
its effective support (an effective-radius disk of `max(r_hoz, r_lz, r_hz)`, since Link 2016 is
piecewise-*angular* zones with no single convex zone — N3, asserted present-and-populated on a cropped
fixture, not silently absent); `bekkers_pi` → `_NO_REGION` (a velocity-derived TTI model with no fixed
spatial ROI, honest-NaN on freeze-frames anyway). **Only** `xt_gk_pressure` / `xt_gk_pev` are
companioned — there is **no composite `xt_gk_observed_fraction`** (M1): the composite adds a
region-dependent `γ·pev` term to GK-geometry `base`/`rav`/`dzv` terms, so no honest single fraction
exists, and `xt_gk` is **exempted** in the completeness gate (its region-dependent part is covered by
the two pev/pressure companions).

### `add_defensive_credit` — magnitude-weighted rollup + a long-form Hyrum widening (T1)

Unlike every other aggregator (one convex region per emitted column), `add_defensive_credit` emits ONE
per-action rollup companion for the WHOLE credit family, because the region a credit integrates over is
per-CREDIT (its resolution mode + anchor), not per-column. The region is built BY MODE
(`anchor_actor` → `_NO_REGION`, event-resolved; `lane` → the shot→goal corridor trapezoid; every
proximity mode → the inscribed disk) and the per-action fraction is a **credit-magnitude-weighted
mean** of the per-credit observed fractions; a non-region-bearing credit is excluded from BOTH
numerator and denominator (never a fabricated 1.0). The rollup covers the three `region_support`
credit columns via `_CUSTOM_COMPANION_COVERS`. **Hyrum note:** to feed the mode-aware region rebuild,
the public `compute_defensive_credits` long-form output is widened by **three additive columns**
(`origin_x`, `origin_y`, `region_radius` — the per-credit resolution anchor + search radius,
appended at the END). Any consumer reading the long-form by fixed position/count sees three new
trailing columns; column-name readers are unaffected.

### Component 4 — retire the `space_creation` velocity-FOV proxy (M4, correctness fix)

The `velocity_unavailable_by_design(frame)` gate is replaced by the **real** signal:
`compute_space_created` / `add_space_creation` gain a `fov_cropped` parameter (resolved by
`add_space_creation` from the real `visible_area` via `add_visible_area_coverage` / `validate_fov`),
and the opponent-perspective one-team softening now keys on it. This **diverges** from the old proxy
exactly on the inputs it was wrong about — velocity-less-but-full-coverage (must now `raise`, not
soften) and velocity-bearing-but-cropped (must now soften, not `raise`) — asserted from both sides
against the old proxy (`test_space_creation_fov_migration.py`).

**Caller-facing behaviour change (intended, not additive).** Opponent-perspective softening of a
one-team frame now REQUIRES `visible_area`: a one-team frame WITHOUT it now **raises** where it
previously softened on the velocity marker. `space_created_m2` and every two-team-frame path are
unaffected (the opponent always resolves), so there is **no VAEP retrain** — the change is scoped to
the opponent-perspective one-team edge.

### Component 5 — the two-axis completeness gate (Task 8)

A registry-completeness meta-assertion (`test_fov_completeness_gate.py`) in the codebase's established
idiom: every **FOV-sensitive** column must carry a companion (`companioned_columns()`, which maps raw
columns via `ObservabilityEntry.covers`) OR appear in `_OBSERVABILITY_EXEMPT` with a stated reason.
"FOV-sensitive" (`required`) is the **union of TWO axes**, because no single tag captures it (R1): (a)
the SB360 audit's `region_support` tag — a single-player-*perturbation* axis — scoped to the tracking
`add_*` surface with the ADR-053 (4.88.0 amendment) BOUNDARY entries excluded STRUCTURALLY (they are
disjoint from `public_add_star()`); and (b) a hand-curated `_AGGREGATE_FOV_SENSITIVE` bucket for the
aggregate/region metrics the perturbation probe structurally MISSES (a mean-over-many is robust to a
single-player perturbation yet FOV-crop-biased — `defensive_line_x` / `team_shape_centroid_*` /
`packing_*` measure `no_support`). The gate derives its population structurally, asserts in both
directions, documents WHY `support_data_defined` (temporal, not area) and the boundary surface are
excluded (each excluded category proven to name a real registry population), and carries an M3
non-vacuity plant (a synthetic `region_support` column that appears in no committed entry, asserted to
hold the gate red if it were required).

**Exemptions** (`_OBSERVABILITY_EXEMPT`): `xt_gk` (composite, M1 above); `ghost_gk_x` / `ghost_gk_y`
(a learned model whose FOV dependence is its whole-frame receptive field — no single clean ROI, and a
whole-pitch fraction would over-simplify; a bespoke ghost-observability model is a later cycle).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Extend ADR-062's hand-coded helper per-aggregator | Minimal new abstraction | N copies of build-region→classify→emit, each with its own purity/liveness/audit/glossary wiring; scatters and rots; no answer for aggregate-position metrics | The scatter is the defect; a registry single-sources it |
| B. Per-contributor "roster fraction" (`point_observed`) for aggregate metrics | Intuitive "share of contributors observed" | A cropped player is absent from the freeze-frame entirely → fraction ≡ ≈1.0, silent in exactly the biased case (S1) | Structurally cannot detect the bias it exists to flag |
| C. `goal_map`-keyed aggregate zones (the plan's original design) | Reuses the ADR-055 `GoalMap`; general across conventions | Returns FRAME-coordinate ends; lands on the OPPOSITE half from the action-LTR polygon for every away-possession action — a silent S1 mis-orientation | The `visible_area` polygon + metrics are ALL action-LTR, so the end is FIXED — a fixed role-keyed zone is correct AND simpler |
| D. Make companions default-on (not opt-in) | No kwarg to remember | Full-coverage providers carry meaningless ≡1.0 companions on every action; changes default output → Hyrum/retrain | Opt-in keeps primary output byte-identical; companions are only meaningful where a polygon exists |
| E. **Registry + one region/area model + `validate_fov` + fixed action-LTR zones + real-signal `space_creation` fix (chosen)** | Single-sourced; one code path; ADR-062 retired byte-identically; correctness fix discharged; anti-rot gate | A hand-curated `_AGGREGATE_FOV_SENSITIVE` bucket is a manual-discipline surface | — |

## Consequences

### Positive

- FOV under-reporting on SB360 becomes **honestly annotated** on **eight companioned aggregators**
  (seven newly this cycle + ADR-062's `add_action_context`) through one engine, and the
  `space_creation` proxy correctness gap is discharged (not re-worded).
- A public `validate_fov` diagnostic completes the four-member frame-set diagnostic family.
- The completeness gate makes a NEW region/aggregate metric fail CI until companioned or
  exempted-with-reason — the anti-rot property ADR-062's hand-coded helper structurally lacked.

### Negative / maintenance

- `_AGGREGATE_FOV_SENSITIVE` is a hand-curated bucket (like ADR-054's `_GUARD_EXEMPT`): a new
  aggregate/region metric must be ADDED to it to be gate-forced. The M3 non-vacuity plant proves the
  enforcement fires, but the *population* of the bucket is discipline, not derivation.
- The `add_defensive_credit` long-form widening is a Hyrum surface for position/count consumers of
  `compute_defensive_credits` (three new trailing columns).
- The `space_creation` one-team-without-`visible_area` raise is a caller-facing behaviour change on
  that edge (intended; the proxy it replaces was wrong there).
- The fixed action-LTR zones are correct **because** the only `visible_area` supplier is action-LTR; a
  future FOV-bearing provider in a different convention would need the zone builders revisited (recorded
  here so a future reader meets the reasoning, not a silent assumption).

### Neutral

- **No VAEP retrain, no Hyrum break on defaults, C4-free.** Companions are opt-in; every primary
  feature column is byte-identical with and without `visible_area`; `tracking_default_xfns` and the
  per-Series functions are untouched. No new action-coupled `add_*` aggregator; `validate_fov` is a
  diagnostic and the registry is internal.
- **Glossary is NOT grown (N1).** The opt-in companion columns are absent from the default-config
  `emitted_columns()` (the glossary coverage harness never supplies `visible_area`), so authoring
  glossary entries for them would fail `test_no_stale_entries`. They are glossary-exempt by the same
  precedent that keeps ADR-062's six `add_action_context` companions out of the glossary — verified: 0
  `observed_fraction` entries in `feature_glossary.py` today. The design spec's "the `feature_glossary`
  count grows" is therefore **inaccurate**; the count stays at 352 and the C4 glossary-count string is
  unchanged (no C4 regen).
- **SB360 audit (N2/N3 scope).** Like ADR-062's companions, the new companions are opt-in and sit
  OUTSIDE the default-config SB360 audit surface (a two-leg full-coverage fixture makes a visibility
  companion vacuous — `identical → works`, the coverage-denominator-as-signal trap). Recorded at
  `tests/sb360/_registry.py::audited_surface`, verified on the real corpus by the licensed e2e +
  `scripts/validate_sb360_licensed_corpus.py` instead.

## CLAUDE.md Amendment

This ADR does not carve out an exception to a project-wide rule; it adds a durable Key-convention
(the observability registry + `validate_fov` + the single region/area model + the opt-in/no-retrain
invariant + the `space_creation` caller-facing change) and marks the ADR-054-amendment "PRAGMATIC FOV
PROXY … migrate when one exists" note as discharged by this cycle.

## Related

- **Specs:** `docs/superpowers/specs/2026-08-27-sb360-fov-observability-design.md`
- **Plans:** `docs/superpowers/plans/2026-08-27-sb360-fov-observability.md`
- **ADRs:** extends ADR-062 (visibility companions), ADR-055 (`_visibility.py`), ADR-054/ADR-063
  (velocity-availability contract), ADR-053 (SB360 audit), ADR-028 (action-LTR geometry).
- **External references:** Sutherland, I. E., & Hodgman, G. W. (1974), "Reentrant Polygon Clipping" —
  already recorded in `NOTICE` for `silly_kicks/_polygon.py` (the observed-area clipping primitive);
  Andrienko et al. 2017 (directional pressure oval); Link et al. 2016 (angular pressure zones).

## Notes

**No retrain, no re-materialize:** primary feature columns are byte-identical with and without
`visible_area`; the `space_creation` divergence is confined to the opponent-perspective one-team edge
(no two-team or `space_created_m2` change). The only persisted-output surface that moves is the
opt-in companion columns, which no bundled model or default xfn list consumes.

**A companion reports region-observability INDEPENDENTLY of the primary metric's domain** — so
`add_packing` / `add_xt_gk` / `add_defensive_credit` may emit an observed fraction on a row where the
primary metric itself is NaN (a shot row for packing, an unresolvable-destination goalkick for
`xt_gk`). This is deliberate: region observability is orthogonal to metric value, and NaN-aligning the
companion to the primary would couple the observability signal to the very quantity it exists to be
independent of. The companion answers "how much of this metric's region did the provider observe?",
which is well-defined whether or not the metric resolved a value for that row.
