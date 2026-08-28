# SB360 FOV-Observability: a first-class visibility signal + single-sourced metric companions

- **Status:** DRAFT (design, pre-implementation)
- **Date:** 2026-08-27
- **Branch:** `sb360-fov-observability`
- **Proposed ADR:** ADR-077 (next after ADR-075)
- **Prior art it extends/retires:** ADR-053 (SB360 audit), ADR-054/063 (velocity-availability contract), ADR-055 (`_visibility.py`), **ADR-062 (opt-in visibility companions on `add_action_context`)**.

## Executive summary

StatsBomb-360 freeze-frames only contain the players inside the broadcast field of view, delivered as
a per-action `visible_area` polygon. Every **region- or aggregate-based** tracking metric therefore
*silently under-reports* on SB360: a defender outside the FOV lowers a congestion count, shifts a
defensive-line mean, or shrinks a pressure value, and nothing in the output says so. ADR-062 solved
this for exactly **three** columns on **one** aggregator (`add_action_context`) via a private,
hand-coded helper, and left the general case to a *velocity-marker proxy* (`_space_creation.py:190`,
which itself flags "migrate to a real frame-level FOV signal when one exists").

This cycle makes FOV-observability a **first-class, single-sourced concern**:

1. A public **`validate_fov` / `FovDiagnosis`** frame-set diagnostic — the fourth member of the
   `validate_time_base` / `validate_velocity_regime` / `validate_id_dtypes` family.
2. A **declarative observability registry** so a metric earns FOV companions by *declaring how its
   value depends on observation*, not by another hand-copied helper. ADR-062's three-column helper is
   retired into it.
3. **One observability model — an observed-*area* fraction of a region** — because the FOV signal is
   *purely a polygon*. A per-contributor "roster fraction" is a false dichotomy: a cropped-out player
   is **absent from the freeze-frame entirely** (`parse.py:298` builds each frame only from
   `ff["freeze_frame"]`), so he is in neither the numerator nor the denominator and the fraction is
   identically ≈1.0 — silent in exactly the biased case it was meant to flag. Every companioned metric
   therefore declares a **region** (a *tight* ROI for count/density/distance metrics — disk / triangle
   / oval; a *broad zone* for aggregate-position metrics — a defensive band, a team zone, or the whole
   pitch) and the companion is `region_observed_fraction(polygon, region)`. A right-half crop yields
   0.5, never a lie near 1.0.
4. **Retirement of the `space_creation` velocity-FOV proxy** onto the real signal — a genuine
   correctness fix, not just an annotation.
5. A **completeness gate** over TWO axes — the SB360 audit's `region_support` tag PLUS a hand-curated
   `_AGGREGATE_FOV_SENSITIVE` bucket (the perturbation probe cannot detect FOV-crop bias on a
   mean-over-many) — so a new region/aggregate metric must declare a companion or be
   exempted-with-reason.

**Everything is additive and opt-in.** Primary feature columns stay byte-identical; companions appear
only when the caller supplies `visible_area`. **No VAEP retrain.** No new action-coupled aggregator
(C4 count unchanged). All FOV primitives stay in the neutral `_visibility.py` / `_polygon.py` (zero
pitch-control / DAS dependency).

## Problem

### The defect class

`visible_area` is a per-action polygon (StatsBomb 120×80 → `polygon_to_spadl` → `(N,2)` SPADL, **not
clipped** to the pitch, ADR-054 D5). Frames built from a freeze-frame contain only the in-FOV players.
Two distinct failure shapes follow:

- **Region-count metrics** (`nearest_defender_distance`, `receiver_zone_density`,
  `defenders_in_triangle_to_goal`, `pressure_on_actor__andrienko_oval`, `packing_*`): an unobserved
  player inside the metric's region-of-interest (ROI) makes the count/pressure read *lower* than
  reality. The value is not fabricated, but a consumer cannot tell a genuinely empty region from an
  unobserved one.
- **Aggregate-position metrics** (`defensive_line_x = mean(sel_x)`, `team_shape` centroids,
  `off_ball_xt_team`): a missing contributor *biases the estimate*. The share of the *true* contributor
  set observed is **unknowable from a freeze-frame** (the cropped player is not in the frame at all —
  the vocabulary has no token for "a contributor was cropped out", and a polygon-vs-point test cannot
  detect an absent point). The honest, polygon-derivable signal is instead **how much of the pitch
  zone the metric ranges over was observed** — an observed-*area* fraction (a left-half crop → 0.5),
  which correlates with "the estimate may be biased" and cannot read ≈1.0 under cropping.

### Why ADR-062 is not enough

ADR-062 is correct but narrow: `_append_visibility_companions` (`features.py:459`) hardcodes the three
`add_action_context` ROIs and emits their companions. Extending that pattern per-aggregator means N
copies of "build this metric's region → `classify_region_observation` → emit two columns", each with
its own purity / liveness / SB360-audit / glossary registration. That scatters the FOV logic and
rots. And it has no answer at all for the aggregate-position metrics.

### The recorded correctness gap

`_space_creation.py:187-192` softens its opponent-perspective abort to a per-row NaN **only** when
`velocity_unavailable_by_design(frame)` — an explicit "PRAGMATIC PROXY … migrate to a real
frame-level FOV/visibility signal when one exists" (ADR-054 amendment). The proxy is right only
because velocity-less and FOV-cropped *coincide in today's providers*; it is not a measurement of FOV.

## Non-goals

- **Correcting** the under-count (we cannot conjure unobserved players; the deliverable is honest
  annotation + a correctness fix to a proxy, not better point estimates).
- **Per-player** aggregation on SB360 (foreclosed — freeze-frames carry no player identity, ADR-062).
- Making companions **non-opt-in / default-on** (rejected below).
- Any change to `_das.py` / `pitch_control/` internals (the DAS session's area; landed as 4.97.0).

## Design

### Component 1 — `validate_fov` / `FovDiagnosis` (the real frame-level signal)

A public diagnostic mirroring `validate_velocity_regime` **exactly** in shape and discipline:

- **`FovDiagnosis`** — frozen dataclass in `tracking/schema.py`, beside `VelocityRegimeDiagnosis`.
  Fields (final set pinned in the plan):
  - `regime: str` — one of `FOV_REGIME_VALUES` (module constants, below).
  - `observed_pitch_fraction: dict[...]` or summary stats — per-action observed pitch fraction rollup.
  - `source_counts: dict[str, int]` — `visible_area_source` token → action count (reuses
    `VISIBLE_AREA_SOURCE_VALUES`).
  - `n_actions: int`, `message: str`.
- **`validate_fov(...)`** in `tracking/utils.py`, signature discipline from the family:
  - Takes only what it reads. FOV is a per-action property, so it consumes `visible_area` (the
    `action_id → polygon` table the port emits) — **no unread `frames`/`actions` parameter** (the
    dead-parameter defect the family docstring calls out).
  - `on_mismatch: Literal["warn", "raise", "ignore"] = "raise"`.
  - Guards on **column/key presence, not row count**; an **empty input never raises**.
- **Regimes** (`FOV_REGIME_VALUES`, module constants):
  - `full_coverage` — every action's polygon covers (≈) the whole pitch (fraction ≥ a pinned floor).
  - `fov_cropped` — polygons present but partial (the SB360 case).
  - `absent` — no usable polygon on any action (`no_polygon` / `degenerate_polygon` throughout).
  - `mixed` — a set that mixes full-coverage and cropped/absent actions (the fail-loud case, like
    `validate_velocity_regime`'s `mixed`).
- **Why a diagnostic, not a per-row column:** FOV regime is a property of the *whole* frame set
  (verbatim reasoning from `VelocityRegimeDiagnosis`). The *per-action* observed fractions that DO
  vary row-to-row are the companions (Components 2–3), not this.

This is the signal `space_creation` (Component 4) and any future consumer consults.

### Component 2 — the declarative observability registry (one model: region/area)

One registry, in a new private module `tracking/_fov_registry.py` (neutral; imports only
`_visibility` + `_geometry`, never pitch-control). Each entry declares the **region** the metric's
value depends on:

```
ObservabilityEntry(
    aggregator="add_pressure_on_actor",
    column="pressure_on_actor__andrienko_oval",
    region=lambda action, frame, ctx: <convex (M,2) polygon>,   # tight ROI OR broad zone
)
```

> **⚠️ SUPERSEDED (aggregate-position zones only) — the `goal_map`-keyed zone design for
> `add_defensive_line` / `add_team_shape` / `add_player_influence` (the "goal-keyed fixed zone" wording
> below and the three aggregate rows of the Component 3 table) was REJECTED during execution and
> REPLACED by ADR-077 "Design (A)": FIXED action-LTR pitch bands keyed on the column ROLE (no
> `goal_map`).** A `goal_map` lookup returns *frame-coordinate* ends, which mis-orient every
> away-possession action against the action-LTR `visible_area` polygon (the S1 silent-failure this cycle
> prevents); in action-LTR the acting team always attacks x=105, so the defended end is FIXED per role.
> The **tight-ROI** metrics (disk/triangle/oval) below are UNCHANGED. Authoritative record:
> **[ADR-077](../adrs/ADR-077-fov-observability.md)** (records `goal_map` as rejected option C). The
> `goal_map` / `GoalMap` / N2-per-`(action, goal_map)` wording in this Component is retained as the
> historical design record only.

- A single engine consumes an entry + the per-action polygon and emits the companion pair
  `<column>_observed_fraction` / `<column>_observed_source` via `classify_region_observation(polygon,
  region)` — **one code path for every metric**. The region is a *tight ROI* for count/density/distance
  metrics (disk / triangle / oval) and a *broad zone* for aggregate-position metrics — a **goal-keyed
  fixed zone** (defended third, own / attacking half), **never the whole pitch** (which would be
  redundant with the existing `visible_area_fraction`, N1); they differ only in the polygon returned,
  never in model.
- The region **must be convex** (Sutherland–Hodgman clips against a convex region only; disks are
  convex approximations, triangles / ovals / bands / the pitch are convex). A concave region is split
  or the entry is rejected at registration.
- **Frame-independence invariant (N2 — the S1 regression guard).** A region MUST be a function of
  pitch/goal geometry **only** — a tight ROI keyed on the action anchor (disk / triangle / oval), or an
  aggregate zone keyed on `goal_map` (defended goal → defended third; halfway line → own / attacking
  half). It must **never** be computed from the frame's observed player coordinates: a zone drawn around
  the in-FOV players is by construction inside the FOV, so its observed fraction collapses back to ≈1.0
  and S1 returns. A registry docstring states this; a test (below) asserts the emitted region for a
  fixed `(action, goal_map)` is independent of the frame's contents.
- **ADR-062's helper is retired into this registry**: the three `add_action_context` ROIs become three
  entries; `_append_visibility_companions` becomes a thin call into the shared engine. The emitted
  columns and their values are **byte-identical** to today (a parity gate pins this — see Testing).
- `REGION_OBSERVATION_SOURCE_VALUES` already anticipates this ("Task 4 count features"); reused
  unchanged. There is **no per-contributor / `point_observed` path** in the companion engine — S1
  proved it is identically ≈1.0 on a freeze-frame provider — so no new token is needed.

### Component 3 — companion coverage across the region/aggregate family

Entries added this cycle (each aggregator gains an opt-in `visible_area: pd.DataFrame | None = None`
kwarg, exactly as `add_action_context` has today; primary columns unchanged):

| Aggregator | Column(s) | Region (tight ROI or broad zone) |
|---|---|---|
| `add_action_context` | `nearest_defender_distance`, `receiver_zone_density`, `defenders_in_triangle_to_goal` | *retired from ADR-062* — disk r=distance / receiver disk / triangle-to-goal (tight ROI) |
| `add_pressure_on_actor` | `pressure_on_actor__andrienko_oval` | the Andrienko oval around the actor (tight ROI) |
| `add_packing` | `packing_*` (region-count members) | the packing zone, ball→goal corridor (tight ROI) |
| `add_defensive_line` | `defensive_line_x` | **fixed zone, goal-keyed** — the full-width band over the *defended* third, resolved from `goal_map` (N2); observed-*area* fraction, **not** per-contributor (S1) |
| `add_team_shape` | `team_shape_centroid_{x,y}_{attacking,defending}` (FOUR cols, two team ROLES) | **fixed zone, goal-keyed** — TWO companions (`attacking`/`defending`), each on that role's team **own half** (from `goal_map`), **not** whole pitch (redundant with `visible_area_fraction`, axis-blind; N1). Roles sit on OPPOSITE ends → separate zones (R2). Residual: per-axis directional signal is the follow-up |
| `add_player_influence` | `off_ball_xt_team` | **fixed zone, goal-keyed** — the **attacking half** the team's off-ball influence ranges over (from `goal_map`); not whole-pitch, same N1 reason |
| `add_xt_gk` | `xt_gk_pressure`, `xt_gk_pev` | the pressure ROI **dispatched on `pressure_method`** (T2): `andrienko_oval` → the oval; `link_zones` → the **convex outer bound of its effective support** (an effective-radius disk/oval derived from `LinkParams`' zone radii — Link et al. 2016 is *piecewise angular zones*, `pressure.py:36`, so there is no single "convex zone"; N3); **`bekkers_pi` → velocity-derived, already honest-NaN on SB360, no companion**. A non-convex / unsupported method emits the companion **absent** (`source` says so), never a runtime convex-`ValueError` — but the plan asserts the `link_zones` companion is **present-and-populated on a cropped fixture**, so an FOV-sensitive column is not *silently* left absent (N3) |
| `add_defensive_credit` | `defensive_credit_net`, `defensive_credit_minus`, `n_defensive_credits` | **resolution-mode-aware** per-credit region, rolled up per action (T1): `lane` / `all_within` / `all_within_beyond_nearest` → the corridor region; `nearest` / `nearest_fallback` → the inscribed disk (reuse the nearest-defender model); `anchor_actor` → **event-resolved, not FOV-sensitive** → contributes N/A |

**Companion column names** are per-column (`<column>_observed_fraction` / `_observed_source`); `xt_gk`
emits `xt_gk_pressure_observed_fraction` and `xt_gk_pev_observed_fraction` only — **no composite
`xt_gk_observed_fraction`** (M1: only the `γ·pev` term is region-dependent, so a whole-`xt_gk` fraction
would over-claim; the composite is exempted in Component 5). `defensive_credit` emits one per-action
rollup companion for the credit family.

The exact per-metric region function, the aggregate-position zones (goal-keyed fixed zones above —
defended third / own / attacking half, **never whole-pitch** per N1), and `defensive_credit`'s per-mode
rollup are pinned in the implementation plan after reading each `compute_*`. A metric whose region is not cleanly convex or is
genuinely ill-defined is **exempted-with-reason** in the registry (Component 5), never forced.

### Component 4 — retire the `space_creation` velocity-FOV proxy

Replace the `velocity_unavailable_by_design(frame)` gate at `_space_creation.py:192` with a decision
driven by the **real** signal: when a per-action polygon is present and the frame is `fov_cropped`
for that action's region, soften to the per-row NaN + `space_opponent_source == "unresolved_one_team"`;
otherwise keep the fail-loud `raise`. The velocity proxy is removed (the ADR-054-amendment comment is
discharged, not merely re-worded).

- **Contract preserved:** the softened-vs-raise *behaviour* on today's providers is unchanged where
  velocity-less ⇔ FOV-cropped actually coincide; it *diverges only* on the inputs the proxy was
  wrong about (velocity-less-but-full-coverage, or velocity-bearing-but-FOV-cropped — the exact case
  the comment warned about). This divergence is the correctness win and is asserted from both sides
  (Testing).
- ⚠️ `_space_creation.py` is pitch-control-consuming (TF-41). The DAS work landed as **4.97.0** and
  touched `gkdv/_arms.py` + `gkdv/_das_port.py` (+ packaging/docs), **none overlapping this file**
  (verified against the `aae6fdb..57004e8` file list). No coordination blocker remains.

### Component 5 — the anti-rot completeness gate

A registry-completeness meta-assertion in the codebase's established idiom (`PURITY_ENTRIES`,
`feature_glossary`, `PUBLIC_ID_SCALAR_ENTRIES`, the SB360 registry): every **FOV-sensitive** column
must **either** carry an observability companion **or** appear in `_OBSERVABILITY_EXEMPT` with a stated
reason. The gate maps a companion to the RAW metric columns it annotates via `ObservabilityEntry.covers`
(so a rollup or a role-split companion counts correctly).

- **"FOV-sensitive" is the union of TWO axes, because no single tag captures it (R1).** (a) the SB360
  audit's **`region_support`** tag (`_vocabulary.py:115`), a *single-player-perturbation* axis; and
  (b) a **hand-curated `_AGGREGATE_FOV_SENSITIVE`** bucket for the aggregate/region metrics the
  perturbation probe structurally MISSES — a mean-over-many is robust to a single-player perturbation
  (tagged `no_support`) yet is FOV-crop-biased (measured `no_support` for `defensive_line_x` /
  `team_shape_centroid_*` / `packing_*`; only `off_ball_xt_team` is `region_support`). Without bucket
  (b) the gate would leave the very aggregate metrics this cycle exists for unforced. It is a
  manual-discipline surface (like ADR-054's `_GUARD_EXEMPT`).
- **Scope justification for the audit axis (M2):** `no_support` columns are perturbation-insensitive by
  measurement (and covered via bucket (b) where FOV-relevant); **`support_data_defined`** columns
  (`actor_*_pre_window`, `elastic_confidence`) are *temporal*-support partial — NaNs about single-frame
  history, **not** observation area — so they are FOV-insensitive. This lives in the gate so a future
  auditor does not re-derive it.
- The gate derives its population structurally and asserts it in **both** directions (a new
  `region_support` column fails CI until registered or exempted) — the property ADR-062's hand-coded
  helper structurally lacked.
- **Non-vacuity plant (M3):** landing red on an *empty* registry only proves the gate detects the
  *known* population. The durable guard injects a **synthetic `region_support` column absent from every
  committed entry** (a name/shape that appears nowhere, so it cannot be satisfied by copy-paste) and
  asserts CI stays red until that synthetic column is registered or exempted — closing the recurring
  ADR-056 trap where a detector built from the known population merely reconfirms it.

## Public surface

**New (all additive):**
- `tracking.validate_fov`, `tracking.FovDiagnosis`, `FOV_REGIME_VALUES` (+ per-regime constants).
- `visible_area` kwarg on `add_pressure_on_actor`, `add_packing`, `add_defensive_line`,
  `add_team_shape`, `add_player_influence`, `add_xt_gk`, `add_defensive_credit` (default `None` →
  today's output byte-identical). Adding the kwarg to `add_xt_gk` leaves the **frozen** parametric
  model (ADR-024) untouched — it only appends companion columns.

**Changed (internal, output-preserving):**
- `_append_visibility_companions` → thin wrapper over the shared registry engine (ADR-062 columns
  byte-identical, parity-gated).

**Caller enumeration (spec discipline, ADR-051 §7.1):** the plan enumerates every caller of each
changed function. `_append_visibility_companions` has one caller (`add_action_context`); the seven
newly-companioned aggregators (`add_pressure_on_actor`, `add_packing`, `add_defensive_line`,
`add_team_shape`, `add_player_influence`, `add_xt_gk`, `add_defensive_credit`) gain an *optional* kwarg
(no existing caller is affected). `validate_fov` is new (no callers). The `space_creation` change is
behind the same public signature.

## Invariants & impact

- **No VAEP retrain / no Hyrum break on defaults.** Companions are opt-in (`visible_area is not None`);
  every primary feature column is byte-identical with and without it. `tracking_default_xfns` and the
  per-Series functions are untouched. Full-coverage providers never carry meaningless ≡1.0 companions
  (the reason opt-in beats default-on).
- **C4 aggregator count unchanged.** No new action-coupled `add_*` aggregator; `validate_fov` is a
  diagnostic and the registry is internal. The **`feature_glossary` count grows** by the new companion
  columns (a docs/glossary update, ADR-048), which is a distinct surface from the C4 aggregator count.
- **Neutral layering.** FOV primitives live in `_visibility.py` / `_polygon.py`; the registry imports
  only `_visibility` + `_geometry`. Zero pitch-control / DAS dependency — orthogonal to 4.97.0.
- **Glossary + SB360 audit.** New companion columns get `feature_glossary` entries. Like ADR-062's
  companions, they are opt-in and therefore sit **outside** the default-config SB360 audit surface (a
  two-leg full-coverage fixture makes a visibility companion vacuous) — recorded at
  `tests/sb360/_registry.py::audited_surface`, not given per-column audit verdicts.

## Testing strategy

Following the codebase's "both-sides + non-vacuity" discipline:

- **ADR-062 parity:** the retired three-column companions produce byte-identical values pre/post
  refactor (freezes the observable behaviour before restructuring — the fixture-generator discipline).
- **Tight-ROI metric, both sides:** a polygon covering the whole ROI → `observed_fraction == 1`; a
  polygon covering ~half the ROI → `observed_fraction ≈ 0.5`, with the primary count/pressure unchanged
  in both. Assert the failing side too (a mutation that should move the fraction out of band).
- **Aggregate-position metric (area), both sides — the S1 test, rewritten:** a defensive line spanning
  the pitch width with one defender in each half; a **left-half-crop** polygon → (a) the primary
  `defensive_line_x` is *biased* (the right-half defender is absent from the frame, so the mean shifts
  — this is why the annotation matters) **and** (b) `defensive_line_x_observed_fraction ≈ 0.5` for the
  full-width defensive band; a **full-pitch** polygon → fraction `== 1` and no bias. The old
  per-contributor formulation is explicitly *not* tested — it is unrepresentable (S1). Same shape for
  `team_shape` centroids and `off_ball_xt_team`.
- **`defensive_credit`, mode-aware, both sides:** a `lane`/`all_within` credit whose corridor region is
  partly outside the polygon → rollup `observed_fraction < 1`; a `nearest`/`nearest_fallback` credit
  whose inscribed disk is partly outside → `< 1`; an `anchor_actor` credit → contributes **N/A**, not a
  spurious `1.0`; a full-FOV frame → `== 1`. Assert the per-action rollup aggregates correctly across a
  mix of modes.
- **Frame-independence (N2):** for a fixed `(action, goal_map)`, the region an aggregate entry emits is
  byte-identical under permutation/removal of the frame's player rows — proving the zone is
  geometry-derived, not observed-player-derived (the S1 regression guard, per Component 2).
- **`link_zones` coverage (N3):** on a cropped fixture with `pressure_method="link_zones"`, the
  `xt_gk_pressure`/`xt_gk_pev` companions are **present and populated** (a real fraction < 1), not
  silently `absent` — so the effective-support ROI is genuinely convex and the T2 net does not mask a
  coverage hole on an FOV-sensitive column.
- **`space_creation` migration, both sides (M4 — synthetic by necessity):** today's providers cannot
  produce the decisive inputs, because velocity-less and FOV-cropped *coincide* in them; the fixtures
  are therefore **hand-crafted to decouple the two axes**. The two cases the old proxy got wrong —
  velocity-less-**but-full-coverage** (must now `raise`, not soften) and velocity-bearing-**but-cropped**
  (must now soften, not `raise`) — are each asserted to **flip behaviour versus the old
  `velocity_unavailable_by_design` proxy** (a fixture where old and new behave identically is evidence
  of nothing). Plus a non-vacuity assertion that a factual-vs-cropped pair measurably differs.
- **`validate_fov`:** each regime constructed and asserted; empty input never raises; `mixed` raises
  under `on_mismatch="raise"` and warns under `"warn"`.
- **Completeness gate:** landed **red-first** (observed failing before any entry exists), then green,
  **plus the M3 non-vacuity plant** — a synthetic `region_support` column absent from every committed
  entry, asserted to hold CI red until registered or exempted.
- **Purity / liveness / id-dtype / mirror** invariant gates auto-discover the seven newly-kwarg'd
  aggregators; each gets its companion-present and companion-absent purity variants (ADR-033 two-branch
  contract).

## Out of scope / follow-ups (surfaced, not silently deferred)

The `_OBSERVABILITY_EXEMPT` bucket — the completeness gate forces each entry, so none is silently
skipped:

- **`ghost_gk_x` / `ghost_gk_y`.** Flagged `region_support`, but `ghost_gk` is a *learned* model whose
  FOV dependence is its receptive field over the whole frame state — no single clean ROI, and a
  whole-pitch fraction would over-simplify. A bespoke ghost-observability model is a later cycle.
- **`xt_gk` (composite).** Exempt because a whole-`xt_gk` fraction would over-claim (M1): only the
  `γ·pev` term is region-dependent (base/rav/dzv are GK+goal geometry). Its region-dependent part is
  covered by the `xt_gk_pev_observed_fraction` companion; the composite carries no honest single
  fraction.

Follow-ups:
- A **real per-zone / directional FOV descriptor** (per-third / per-grid, or per-axis) on
  `FovDiagnosis` if a consumer needs finer than a single fixed-zone area fraction — this is the N1
  residual for the centroid metrics (`centroid_x` ← x-coverage, `centroid_y` ← y-coverage), which a
  single axis-agnostic fraction cannot separate.

*(`player_influence`, `xt_gk` pressure/pev, and `defensive_credit` were considered for exemption and
pulled back INTO scope after verifying they map onto the single region/area model — see Component 3.)*

## Attribution / decisions

- Decision: **ADR-077** (to be written with the implementation).
- Extends ADR-062 (visibility companions), ADR-055 (`_visibility`), ADR-054/063 (velocity contract),
  ADR-053 (SB360 audit). See `NOTICE` for the Sutherland–Hodgman clipping citation already recorded
  for `_polygon.py`.
