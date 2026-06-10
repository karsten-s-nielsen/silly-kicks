# General restart-coordinate enrichment — design spec

## Executive summary (for review)

Real Gradient Sports data ships a large fraction of **set-piece / restart events with a NaN
coordinate** (owner-tier WC2022). The xT-GK release (4.21.0) closed this *for goal-kicks only*,
inside the xT-GK valuation, via a private `resolve_gk_geometry` helper. This spec **promotes that
helper into a general, provider-agnostic coordinate-enrichment feature** that imputes coordinates for
every restart type with a Law-defined location — goal-kicks, corners, throw-ins, penalties — and emits
the derived coordinates **as new, provenance-tagged columns** (`enriched_start_x/_y`,
`enriched_end_x/_y`, `*_coord_source`, `*_coord_confidence`).

**Honest value envelope.** On the **events-only** path (`frames=None`, the common lakehouse case) the
only non-native imputation is the Law-geometry prior (`restart_prior`, confidence ≤0.5). Under the
recommended `≥0.7` position-sensitive filter that prior is excluded, so the **events-only Phase-1
value is population-aggregate / coverage, not position-sensitive**; position-sensitive imputation
value **requires tracking frames** (the `tracking_ball` / `tracking_gk` tiers). The feature is honest
about this via per-row confidence rather than overstating per-event accuracy (§5).

**Phasing is the load-bearing decision.** Changing the *canonical* `start_x/end_x` columns is a
Hyrum/retrain trigger for every trained model (VAEP, xT, calibration), shifts every golden test, and
needs a coordinated retrain. We therefore split exactly as the calibration harness does (ADR-009,
recommend-then-apply):

- **Phase 1 (this spec) — additive, zero-retrain.** Emit enriched coordinates + provenance as *new*
  columns. Canonical `start_x/start_y/end_x/end_y` are **never mutated**. No model sees different
  input → no retrain, no golden shift. Any consumer opts in by reading the enriched columns. Also
  **consolidates** the two coordinate resolvers: the new general resolver becomes the single source
  of truth and the shipped xT-GK path re-points at it (goal-kick output byte-identical → no xT-GK
  retrain, parity-gated).
- **Phase 2 (future, separate PR) — canonical promotion.** Mechanically copy `enriched_* → canonical
  *` (or flip a converter flag), *with* the coordinated VAEP/xT/calibration retrain, the geometry
  tripwire promoted to a hard gate, and golden re-baselining. Deferred to the moment a model actually
  needs the new contract. Out of scope here; Phase 1 is designed so Phase 2 is mechanical.

**Scope was grounded by a live lakehouse probe (2026-06-10), not inherited.** The TODO's framing —
"NaN coordinates degrade EVERY coordinate-consuming metric across the corpus" — is **not** what the
data shows. ~99% of `bronze.spadl_actions` (StatsBomb 73%, Wyscout 25%, SkillCorner) has **0.00% NaN
coordinates**. The gap is **almost entirely Gradient Sports**, and within GS it is **concentrated in
restart types that all have a Law-defined restart location** (goal-kick 60%, free-kick 35%, corner
30–34%, throw-in 27%, penalty 23% NaN-start). That concentration is *why* a geometric prior is
defensible — the same reason the 6-yard-box rule-point works for goal-kicks. Open-play passes (3.4%
GS NaN) have no such prior and get only the tracking-ball / next-event tiers, never a rule-point.

---

**Date:** 2026-06-10 · **Author:** Karsten (with Claude) · **Status:** Draft — awaiting maintainer review.
**Context:** follow-up from the xT-GK goal-kick-coverage work (spec
`2026-06-08-xt-gk-goalkick-coverage-design.md` §2 / D-A1; TODO "Goal-kick / event coordinate
enrichment — GENERAL"). Promotes the scoped `resolve_gk_geometry` (`silly_kicks/tracking/_gk_geometry.py`).
**ADR:** a new ADR, number reconciled against `origin/main` at PR time (no pre-reservation).

---

## 1. Evidence (live lakehouse probe, 2026-06-10)

`soccer_analytics.bronze.spadl_actions`, ~9.74M actions, via the Databricks SDK statement-execution
API (CLI SQL passthrough is dead on this box; DEFAULT-profile PAT). Throwaway probe
`scripts/_probe_coord_nan.py` (deleted after this spec).

**Per-provider NaN-coordinate prevalence (all action types):**

| provider | n actions | NaN start% | NaN end% |
|---|---|---|---|
| statsbomb | 7,151,519 (73%) | 0.00% | 0.00% |
| wyscout | 2,467,814 (25%) | 0.00% | 0.00% |
| skillcorner | 11,777 | 0.00% | 0.00% |
| **gradientsports** | 90,831 | **5.40%** | 1.59% |
| idsse | 8,430 | 1.90% | 1.90% |
| metrica | 6,159 | 3.69% | 20.31% |

(Metrica is frozen anonymized sample data — retired per TODO; not a target.)

**Gradient Sports by action type (where the gap lives):**

| type | n | NaN start% | Law-defined restart spot? |
|---|---|---|---|
| goalkick | 1001 | 60.2% | yes — 6-yard box |
| freekick_short | 1562 | 34.8% | no — variable |
| corner_short | 112 | 33.9% | yes — corner arc |
| corner_crossed | 460 | 30.4% | yes — corner arc |
| throw_in | 2600 | 27.3% | yes — touchline |
| shot_penalty | 64 | 23.4% | yes — penalty spot |
| pass | 60,772 | 3.4% | no — open play |
| (tackle/cross/clearance/…) | — | 2–4% | no |

**Conclusion.** The feature is a **set-piece / restart coordinate enrichment**, primarily serving
Gradient Sports (and any future sparse provider). It is *not* a blanket "impute every NaN coordinate."
The high-NaN types all have a Law-defined location → a defensible geometric prior. End-coordinate
(destination) NaN is small even for GS (goal-kick 6.5%), so next-event fusion is a minor tier.

## 2. Scope

**In scope (Phase 1):**
- Action types with a Law-defined restart location get a rule-point / Law-geometry tier:
  `goalkick`, `shot_penalty` (point restarts); `corner_crossed`, `corner_short`, `throw_in`
  (locus restarts — see §4).
- All other rows (`freekick_short`, open-play `pass`, etc.) with a NaN coordinate get **only** the
  tracking-ball and next-event tiers (no rule-point). They are enriched only when a non-prior source
  resolves them; otherwise `unresolved`.
- Both origin (`start_*`) and destination (`end_*`) coordinates.
- Events-only operation (`frames=None`) is the primary path; tracking frames are an optional
  accuracy-boosting input.

**Explicitly out of scope:**
- Mutating canonical `start_x/start_y/end_x/end_y` (Phase 2).
- Any model retrain, golden re-baseline, or converter-level coordinate change (Phase 2).
- A rule-point for variable-location restarts (`freekick_short`) — no Law-defined spot exists.
- Guessing a corner side / throw-in side with no disambiguating source (stays `unresolved`).

## 3. Architecture & placement

**Public surface — `silly_kicks/spadl/utils.py`:**

```python
@nan_safe_enrichment
def add_restart_coordinates(
    actions: pd.DataFrame, *,
    frames: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame: ...
```

Provider-agnostic, post-conversion, **frames-optional**, mirroring `add_pre_shot_gk_context`: returns
a sorted copy of `actions` with the enriched columns appended; **never mutates** the canonical
coordinate columns. The engine is **lazy-imported at call-time** (ADR-005 §5 — no import-time
`spadl → tracking` cycle); the import is unconditional (single-engine consolidation), but the engine
is lightweight and **touches the heavy `_kernels` linkage only when `frames` is supplied** (inside
`_tracking_ball_xy` / `_tracking_gk_xy`), so the `frames=None` path does no linkage work. NaN-safe
(ADR-003) and id-dtype-safe (ADR-019) by construction.

**Core resolver — add to `silly_kicks/tracking/_gk_geometry.py` (in place):**

```python
def resolve_restart_geometry(
    actions, *, frames=None, links=None,
) -> pd.DataFrame:  # index-aligned: enriched coords + *_source + *_confidence
```

This is the **single source of truth**. **`resolve_gk_geometry` is NOT removed or renamed** — it is
**public API** (exported in `silly_kicks.tracking.__all__`, `__init__.py:193` + re-export `:234`) with
**4 internal call sites across 3 modules** and direct test imports:

| call site | context |
|---|---|
| `_xt_gk.py:408` (`compute_xt_gk`) | RAV/base/dzv geometry + provenance |
| `_gk_completion.py:307` (`compute_gk_completion`) | standalone completion serve |
| `_gk_completion.py:338` (`add_gk_completion`) | lakehouse completion aggregator |
| `features.py:5258` (an aggregator) | tracking-feature path |

Removing or renaming it is a **Hyrum/API break** and undercounts the blast radius. Instead,
`resolve_gk_geometry` becomes a **thin, public, goal-kick-filtered delegation** to
`resolve_restart_geometry` (Chesterton's fence — the existing symbol + its goal-kick output contract
are preserved verbatim). **All 4 call sites keep calling `resolve_gk_geometry` unchanged**; none are
re-pointed in Phase 1. The module stays `_gk_geometry.py` (no rename — a rename of a module reachable
through the public export is itself an API break; see §9). The general path is reached via the new
`spadl/utils.py` helper + the new public `resolve_restart_geometry`.

**New public symbols.** `add_restart_coordinates` (in `silly_kicks.spadl` exports) and
`resolve_restart_geometry` (in `silly_kicks.tracking.__all__`, parallel to the existing public
`resolve_gk_geometry`). Both carry `Examples` docstrings (Examples-gate, §7). `RestartCoordinateReport`
is exported alongside (mirroring `XtGkReport`).

**Why the core lives under `tracking/`:** the tracking-ball / tracking-GK tiers need frames + the
linkage primitive (`resolve_frame_ids_by_position`) + `_id_compat`. `spadl/utils.py` already lazy-imports
tracking for the frames-bearing path of `add_pre_shot_gk_context`, so this matches the established
seam. The events-only tiers (native, rule-point, next-event) invoke **no linkage work** — the engine
module is imported (lightweight, cycle-free) but the heavy `_kernels` linkage is only reached when
`frames` is supplied.

**Input-ordering precondition.** `resolve_restart_geometry` is **index-aligned / positional** — the
`next_event` `shift(-1)` and any possession-depth logic assume `actions` is already in chronological
`(game_id, period_id, action_id)` order. The public `add_restart_coordinates` **sorts first** (mirroring
`add_pre_shot_gk_context`); `compute_xt_gk` etc. already pass caller-sorted actions. This precondition
is **documented on `resolve_restart_geometry`** so a future caller can't hand it shuffled rows and get
silently-wrong `next_event` / depth.

**New action↔frame join seam (ADR-019).** The new `tracking_ball` origin/dest tiers add a
**ball-position lookup** that doesn't exist today (today only `_tracking_gk_xy` joins actions↔frames).
`_tracking_ball_xy` reuses `resolve_frame_ids_by_position` for the frame id and selects the ball row.
Per ADR-019 (the object-`is_ball` `~`-no-op bug: a string `"True"`/`"False"` column makes `~is_ball` a
no-op), the ball selection **must coerce** — `is_ball.astype(bool)` / the `_id_compat` helpers — not
assume bool dtype. The id-dtype-invariance gate auto-discovers the new helper, but the seam is named
here explicitly so the `is_ball` coercion isn't omitted.

**Consolidation (single resolver, no call-site churn).** The consolidation happens *inside*
`resolve_gk_geometry`, which now delegates to `resolve_restart_geometry` (goal-kick-filtered). All 4
call sites continue to call `resolve_gk_geometry` and receive byte-identical output — tiers, clamps,
confidence values, source labels, *resolved coordinates*. This covers not just xT-GK but the two
`_gk_completion.py` call sites and the `features.py` aggregator (§7 parity scope). No output change in
any consumer → no retrain.

**Source-label + column-contract boundary (Hyrum guard).** The general `resolve_restart_geometry`
emits the **new** column names (`enriched_start_x/_y`, `start_coord_source`, …) with **generic** source
labels (`restart_prior`, `tracking_ball`, …). The existing `resolve_gk_geometry` contract is
**different**: it returns columns `origin_x/origin_y/origin_source/origin_confidence/dest_x/dest_y/
dest_source` with goal-kick-specific label values (e.g. `goalkick_prior`, `tracking_gk`). Both the
column names *and* the label values are already-released API consumed by 4 call sites — letting the
generic names/labels leak would break all of them (and shift `xt_gk_origin_source`, which `_xt_gk.py`
derives from `origin_source`).

The **general resolver is type-aware**: for `goalkick` it runs the **GK-clamp computation**
(`_tracking_gk_xy`, x ≤ 16.5) and emits the label `tracking_gk` *directly* — the ball tier never fires
for goal-kicks (§4.1 invariant); for non-goalkick restarts tier-2 is the **ball position**
(`_tracking_ball_xy`) labeled `tracking_ball`. `tracking_gk` is a first-class value in the §5 source
enum, **not** a relabeled ball position.

**The engine is parameterized by `impute_types` (the parity + perf key).** `resolve_restart_geometry`
takes `impute_types: tuple[int, ...] | None = None`. A row is imputed past `native` **only if its
`type_id ∈ impute_types`** (`None` = all types = the general default). The **shim calls the engine with
`impute_types=(goalkick,)`** — so for the frozen path the engine imputes goal-kicks *only* and does
**zero** work on corner/throw-in/open-play rows (no `_tracking_ball_xy` / `_side_y` / `_throwin_x`
calls). This matters because:

- **Parity (verified against current code).** The current `resolve_gk_geometry` imputes **goal-kicks
  only** — its `tracking_gk` / `goalkick_prior` origin tiers AND its `next_event` destination tier are
  all gated `is_goalkick`; non-goalkick rows get native-or-unresolved with no imputation. But
  `compute_xt_gk`'s in-scope set is the **GK-distribution domain** (goalkick + GK-actor pass +
  `throw_in`), so throw-ins/GK-passes flow through `resolve_gk_geometry` today and get *no* imputation.
  With `impute_types=(goalkick,)` the engine reproduces this exactly — non-goalkick rows are never
  imputed, so **no post-hoc revert is needed** (the earlier "revert non-goalkick rows" step is
  eliminated — simpler, no `.loc`-vs-positional fragility). For goal-kick rows, goalkick-mode == legacy
  *by construction* (same `_tracking_gk_xy` clamp, same `_next_event_start`, same rule-point + frozen
  `_CONF` values; the §4.1 invariant gates `tracking_ball` off for goal-kicks).
- **Perf (frozen hot path).** `compute_xt_gk` calls `resolve_gk_geometry` ~3×/slot/match; a real match
  has corners/throw-ins, so an unrestricted engine would run the new `_tracking_ball_xy` loop several
  times per call and the shim would discard it. `impute_types=(goalkick,)` keeps the frozen path's
  primitive-call profile byte-identical to today (one `_tracking_gk_xy`, zero `_tracking_ball_xy`).
  (No existing perf-budget test guards this path today — but wasted hot-path work is avoided on
  principle, and a structural guard could be added.)

So the **delegation shim** `resolve_gk_geometry` does only: (1) call the engine with
`impute_types=(goalkick,)` (+ no tripwire — see §6); (2) **rename columns** to the frozen contract
(`enriched_start_x → origin_x`, …) and **drop `end_coord_confidence`** (the frozen contract has
`origin_confidence` but **no dest-confidence column**; §7a pins this absence); (3) the **single label
transform `restart_prior → goalkick_prior`** (`tracking_gk` / `native` / `next_event` / `unresolved`
pass through unchanged — deliberately **no `tracking_ball → tracking_gk` mapping**, which would relabel
a ball coordinate as a GK coordinate and break parity). The shim stays **whole-array numpy** (no
`.loc`-mask assignment — index-independent, matching the original's style). The §7 parity test still
includes a **throw-in row** (asserts it stays native-or-unresolved through `resolve_gk_geometry` while
`resolve_restart_geometry` with default `impute_types` imputes it).

The exact frozen `resolve_gk_geometry` column set + label values + confidence numbers are read from the
**current code** (not any prior design doc) and pinned by the §7 parity test. The xT-GK
`xt_gk_origin_source` enum is consequently unchanged because its input (`origin_source`) is unchanged.

## 4. Per-type tier model

Resolution runs in confidence order; the first tier that produces a finite, tripwire-valid coordinate
wins and sets `*_coord_source`. Canonical LTR coords (own goal x=0, opponent goal x=105, 105×68,
centre y=34).

### 4.1 Origin (`start_*`) tiers

1. **`native`** — finite native `start_x/start_y`. Confidence 1.0.
2. **`tracking_ball`** — ball position at the linked restart frame (the ball *is* at the restart spot
   for a dead-ball restart). Requires `frames`. Confidence 0.8. **NOT applied to `goalkick`** (see
   invariant below).
   - **Goal-kick variant (preserved):** for `goalkick`, tier 2 is **exclusively** the existing
     **in-area tracking-GK** tier (acting-team GK clamped to `x ≤ 16.5`, off-position → falls
     through), source label `tracking_gk`, confidence **0.7** — byte-identical to today.
3. **`restart_prior`** — Law-geometry snap (only for restart types, see §4.3). Confidence per type
   (§4.4).
4. **`unresolved`** — none of the above. Confidence 0.0. Enriched coord = NaN.

> **Goal-kick tier-set invariant (parity-critical).** For `type_id == goalkick`, the origin tier set
> is **exactly** `native → tracking_gk(in-area, off-position-falls-through) → restart_prior(5.5,34)`,
> and the destination tier set is **exactly** `native → next_event → unresolved`. There is **no
> `tracking_ball` origin tier and no `tracking_ball` destination tier for goal-kicks.** This is what
> keeps `resolve_gk_geometry` byte-identical. Two concrete traps a happy-path parity fixture misses
> (both must be in the §7 red-first test): (a) an **off-position-GK** goal-kick that today falls to
> `restart_prior(0.2)` must NOT be caught by a `tracking_ball` tier (it would change the resolved
> coordinate → `xt_gk` shift); (b) a goal-kick with **no native end and no in-period next-event** is
> `unresolved` today → excluded from the `coords_ok` gate (`_xt_gk.py:421-426`) → *not scored* → a
> `tracking_ball` dest tier would resolve it → a **newly-scored row** → `xt_gk` output change → retrain
> trigger. The generalized resolver must gate the `tracking_ball` tiers off for goal-kicks by type.

### 4.2 Destination (`end_*`) tiers

1. **`native`** — finite native `end_x/end_y`. Confidence 1.0.
2. **`next_event`** — next action's `start_*` within the same `(game_id, period_id)` (the receiver
   location; standard SPADL `end ≈ next start`). Period/match-boundary-guarded (no cross-period
   leak). Confidence 0.6. **Computed positionally over the FULL, unfiltered `actions`** (matching the
   frozen `_next_event_start`'s `shift(-1)` guarded by `game_id` AND `period_id`), *then* masked to the
   rows that need it. **Do not filter to restart types first** — otherwise "next" becomes the next
   *restart*, not the next *action*, giving the wrong destination + a parity break.
3. **`tracking_ball`** — ball position a short, fixed offset after the restart frame (frames only).
   Confidence 0.5. *(Provisional — may be dropped in implementation if it adds little over
   next_event; see §9.)*
4. **`unresolved`** — NaN.

**No destination rule-point, ever.** The destination tier set has **no `restart_prior` tier** — a
restart's *destination* is where the ball is played *to* (a variable, in-play location), never a
Law-fixed spot. In particular a `shot_penalty` end is the shot's target location and gets only
`native / next_event / tracking_ball`; it is never snapped to `(94,34)` (the penalty spot is an
*origin* prior only, §4.3).

### 4.3 Law-geometry snap (`restart_prior`), per type

- **Point restarts** (fully determined, no secondary source needed):
  - `goalkick` → `(5.5, 34)` (6-yard-box centre).
  - `shot_penalty` → `(94, 34)` (penalty spot, 11 m from opponent goal line).
- **Locus restarts** (Law fixes the locus; the *side*/position needs a secondary source):
  - `corner_crossed` / `corner_short` → corner flag `(105, 0)` or `(105, 68)`. The **side** (y∈{0,68})
    is taken, in order, from: native `end_y` → `next_event` start-y → tracking-ball y at the linked
    frame. If no side source resolves → `unresolved` (never guess the side).
  - `throw_in` → nearest touchline `y ∈ {0, 68}` (side from the same precedence as corners) at
    `x` = native `start_x` → `next_event` start-x → tracking-ball x. A throw-in needs BOTH a side and
    a position-along-the-line; if either is unavailable → `unresolved`.

### 4.4 Confidence values (provisional; pinned at implementation)

| source | confidence | note |
|---|---|---|
| `native` | 1.0 | provider truth |
| `tracking_ball` | 0.8 | ball at restart spot |
| `tracking_gk` (goalkick only) | 0.7 | **preserved** — existing goalkick value |
| `restart_prior` — penalty | 0.5 | tightest Law spot |
| `restart_prior` — corner | 0.4 | exact point once side known; side adds uncertainty |
| `restart_prior` — throw_in | 0.3 | touchline known; along-line position imputed |
| `restart_prior` — goalkick | 0.2 | **preserved** — existing goalkick value |
| `next_event` (dest) | 0.6 | receiver location proxy |
| `unresolved` | 0.0 | enriched coord = NaN (never resolvable) |
| `tripwire_reverted` | 0.0 | enriched coord = NaN (resolved then reverted by the tripwire, §6) |

`tracking_ball` (0.8) ranks above `tracking_gk` (0.7) because they are **different physical
quantities**: the tracked *ball* is literally at the dead-ball restart spot, whereas the tracked *GK*
is a positional *proxy* for where the goal-kick will be taken (the GK walks up to the ball) — even
clamped in-area, it carries more error than the ball itself.

The goal-kick values (`native` 1.0, `tracking_gk` 0.7, `restart_prior` 0.2) are **frozen** to keep the
xT-GK / completion parity tests green. Other values are normative defaults, documented as such.

### 4.5 Events-only vs frames behavior matrix

| tier | `frames=None` | `frames` supplied |
|---|---|---|
| native | ✓ | ✓ |
| tracking_ball / tracking_gk | — | ✓ |
| restart_prior | ✓ (restart types) | ✓ (restart types) |
| next_event (dest) | ✓ | ✓ |

## 5. Output contract (Phase-2-promotable)

Appended columns; canonical `start_x/start_y/end_x/end_y` **untouched**:

- `enriched_start_x`, `enriched_start_y`, `enriched_end_x`, `enriched_end_y` (float) — native value
  where present, imputed where a non-native tier fired, NaN where `unresolved`.
- `start_coord_source`, `end_coord_source` (str enum):
  `native` / `tracking_ball` / `tracking_gk` / `restart_prior` / `next_event` / `unresolved` /
  `tripwire_reverted` (origin only — a resolved coord the tripwire reverted, §6).
- `start_coord_confidence`, `end_coord_confidence` (float ∈ [0,1]).

**Phase-2 promotion is then mechanical:** `start_x := enriched_start_x` (etc.) + retrain + golden
re-baseline. This column set *is* the promotion contract; nothing is re-derived in Phase 2. **Note for
the future apply-PR:** `enriched_*` is **NaN** for `unresolved` and `tripwire_reverted` rows — but
those rows had NaN *native* coordinates anyway (and `native` rows are never reverted, so their
`enriched_*` is never NaN'd), so a `coalesce(enriched, native)`-style promotion is a no-op regression
there. Don't write a guard that assumes `enriched_*` is always finite.

**Documented recommended downstream filter** (reusing the xT-GK `≥0.7` idiom): for
position-sensitive use, restrict to `start_coord_confidence ≥ 0.7` (native + tracking); use the full
set only for population aggregates.

**Honest consequence — events-only Phase-1 value is aggregate-only.** The `≥0.7` filter admits
`native(1.0)` / `tracking_ball(0.8)` / `tracking_gk(0.7)` but **excludes every `restart_prior`**
(≤0.5, incl. the 0.2 goal-kick prior). On the `frames=None` path the *only* non-native imputation is
`restart_prior`, so a position-sensitive `≥0.7` consumer gets **essentially no events-only
imputations** — the events-only feature's value is **population-aggregate / coverage**, and
position-sensitive value **requires frames**. The executive summary's "covers every restart type" is a
statement about *coverage of the imputation*, not about position-sensitive usability on events-only;
see the corrected framing in the exec summary.

Also note: **`freekick_short` is the single largest events-only-unresolved NaN bucket** (GS n=1562, 35%
NaN-start) and is **intentionally left unresolved** — it has no Law-defined restart spot, so it gets a
rule-point from nothing and resolves only when `native` / `tracking_ball` / `next_event` fire. This is
correct (§2 scope), not a coverage gap to close with a guessed prior.

**A snapped side is never position-trustworthy.** Because corner/throw-in `restart_prior` is ≤0.4, a
`≥0.7` consumer never trusts a snapped side — the tripwire (§6) bounds *gross* mis-snaps but cannot
catch a wrong-but-plausible side, so side errors are **confidence-bounded, not eliminated**, which the
≤0.4 confidence already encodes.

## 6. Geometry tripwire + report

Per the TODO's explicit ask, à la the ADR-018 own-goal own-half tripwire.

**The tripwire is a feature-policy step that lives at the EDGE (`add_restart_coordinates`), NOT in
the engine.** `resolve_restart_geometry` is a **pure** resolver — it emits no warnings and applies no
reverts. `add_restart_coordinates` calls the engine, then applies the tripwire. This keeps the frozen
`resolve_gk_geometry` path (which delegates to the engine) **silent and revert-free by construction** —
there is no flag to forget, so the engine can never leak a warning onto the hot `compute_xt_gk` path
(the current `resolve_gk_geometry` emits no warnings; this preserves that exactly). Direct callers of
the public `resolve_restart_geometry` get raw resolution; the documented enrichment behavior (incl. the
tripwire) is `add_restart_coordinates`.

- **Validation regions** (LTR, **origin only**): goal-kick `x ≤ 16.5`; penalty within ~3 m of
  `(94,34)`; corner `x ≥ 100` and (`y ≤ 5` or `y ≥ 63`); throw-in `y ≤ 3` or `y ≥ 65`. Tolerances
  pinned at implementation. **Destinations are NOT tripwire-guarded in Phase 1** (the regions are
  origin-framed; a bad `next_event`/`tracking_ball` destination is unguarded — stated honestly, a
  Phase-2 candidate).
- **On a violation of an *imputed* origin** → warn (`stacklevel=2`) + **revert to `unresolved`**
  (locked: revert-to-unresolved, **not** revert-to-next-tier — simpler, honest, no re-run loop; spec
  decision, not re-litigated at implementation). The reverted row is tagged with the distinct source
  **`tripwire_reverted`** (coord → NaN, confidence → 0.0) so it stays **distinguishable from
  `unresolved`** (never-resolvable) for QA — this is the signal the report surfaces (below).
- **Native coordinates are warn-only, never reverted** — provider data is truth (ADR-001 analog); a
  tripwire on native surfaces a data-quality signal without overriding the provider (source stays
  `native`).
- **Optional aggregate `RestartCoordinateReport`** (counts per source × type **+ `n_tripwire_reversions`**,
  derived from the `tripwire_reverted` tag), mirroring `XtGkReport` / `ConversionReport` / `LinkReport`.
  Convenience for pipeline QA (equivalently derivable via `GROUP BY`), not load-bearing.

## 7. Testing (TDD, red-first)

- **Per-type tier firing** in confidence order — synthetic fixtures: native present; NaN origin +
  rule-point; NaN origin + tracking-ball; corner/throw-in side resolved vs unresolvable; point vs
  locus restart. Assert `enriched_*` values, `*_source`, `*_confidence`.
- **Side-disambiguation precedence** (corner/throw-in): native-y > next-event-y > tracking-y; no
  source → `unresolved`.
- **Tripwire** (at the `add_restart_coordinates` edge): mis-located imputed origin → reverts to
  `unresolved` tagged `tripwire_reverted` + warns; native out-of-region → warns only, not reverted.
- **Tripwire does NOT leak onto the frozen path (Major 1):** a `resolve_gk_geometry` call on a
  native-out-of-region goal-kick (e.g. `start_x=80`) emits **no warning** (`pytest.warns(None)` /
  `recwarn`-empty) — the engine is pure, the shim never tripwires.
- **Never mutates `actions`**: input canonical columns byte-identical pre/post; returns a copy.
- **Events-only path**: `frames=None` produces the §4.5 tier subset and invokes **no linkage work**
  (the engine is imported but `_tracking_ball_xy` / `_tracking_gk_xy` are never called).
- **Linkage-fixture validity (Minor 9):** the first `_tracking_ball_xy` test asserts the resolved
  `frame_id` is finite, so a minimal-frame linkage failure is distinguishable from a selection bug.
- **Goal-kick consolidation parity — red-first, all 4 call sites, edge-cases mandatory.** Sequence:
  capture current behavior *before* the refactor, then refactor until green. Layers:
  - **(a) resolver contract — committed GOLDEN snapshot.** Capture `resolve_gk_geometry`'s **full
    output frame** on a multi-type fixture (goal-kick + corner + throw-in + open-play pass, with AND
    without frames, including the two Major-2 edge rows) on the **unmodified pre-refactor code**, commit
    it as a golden, and assert `pd.testing.assert_frame_equal(post, golden)` after the refactor. A frame
    snapshot pins **column set + order + dtypes + every cell** — per-cell assertions miss column-order /
    dtype / untested-row drift. The golden confirms the **absence of a dest-confidence column**
    (frozen contract has `origin_confidence` only; the shim drops `end_coord_confidence`). This golden
    is the real byte-identical guard; Task-10 post-refactor tests are supplementary, not the baseline.
  - **(b) edge-case rows (Major 2) — not happy-path only.** The fixture MUST include: an
    **off-position-GK** goal-kick (today → `goalkick_prior 0.2`; must NOT be caught by any
    `tracking_ball` tier) and a goal-kick with **no native end + no in-period next-event** (today →
    `unresolved`, excluded from `coords_ok` → unscored; must STAY unscored). A happy-path parity
    fixture passes while real GS goal-kicks shift — this is the gate/serve corpus-consistency trap.
    The fixture MUST also include a **NaN-coordinate `throw_in` row**: assert it stays
    **native-or-unresolved** through `resolve_gk_geometry` (the engine runs `impute_types=(goalkick,)`,
    so non-goalkicks are never imputed — §3) while `resolve_restart_geometry` (default `impute_types`)
    imputes it — guards the parity-critical type-gating that keeps `compute_xt_gk`'s throw-in/GK-pass
    in-scope rows unchanged.
  - **(c) completion-path coupling (Major 3).** Because `_completion_p` (`_xt_gk.py:455`) builds the
    GK-completion model's serve features from the resolved geometry, and the bundled GS/SkillCorner
    weights were trained against the current resolver's geometry (train==serve discipline, C1/C3),
    parity scope **explicitly includes** `compute_gk_completion` (`_gk_completion.py:307`),
    `add_gk_completion` (`:338`), the `_gk_completion_density` path, and the `features.py:5258`
    aggregator — not just `compute_xt_gk` / `add_xt_gk`. Each must produce unchanged output.
  - **(d) full-output parity** — `compute_xt_gk` / `add_xt_gk` output, incl. `xt_gk_origin_source` /
    `xt_gk_completion_source`, unchanged on the committed xT-GK fixtures.
  - Add the completion-path parity to the **owner-gated GS e2e** suite too (real-data confirmation
    beyond synthetic fixtures).
- **Cross-cutting auto-discovered gates** (the new public `add_restart_coordinates` + the new public
  `resolve_restart_geometry` will trip these — all must be satisfied):
  - **Public-API Examples docstring** (`tests/test_public_api_examples.py`) — `spadl/utils.py` is in
    that gate's `_PUBLIC_MODULE_FILES`, so `add_restart_coordinates` **needs** an `Examples` docstring
    section or CI fails. `resolve_restart_geometry` lives in `_gk_geometry.py`, which is
    underscore-prefixed and **not** in `_PUBLIC_MODULE_FILES` — so it is **not** gate-required (same as
    the existing public `resolve_gk_geometry`). Open question closed: add an `Examples` block to
    `add_restart_coordinates` only (good practice for `resolve_restart_geometry`, not mandated). (A
    pre-existing gate gap — a `__all__`-public symbol escaping the Examples gate via an underscore
    module — is noted, not this PR's to fix.)
  - **NaN-safety registry** (ADR-003 `tests/test_enrichment_nan_safety.py`) — `@nan_safe_enrichment`
    auto-discovery.
  - **id-dtype invariance** (ADR-019) and **provenance-skip idempotence** where applicable.
- **Owner-gated GS e2e** (not a CI gate — GS not in CI): real WC2022 restart coverage + tripwire
  reversion counts + source distribution sanity. The CI-ungated risks (tripwire, side-inference,
  events-only) are covered by the synthetic fixtures above.

## 8. ADR + docs + release

- **New ADR** (number reconciled against `origin/main` at PR time): records the additive-now /
  canonical-later phasing, per-type prior geometry, tripwire + provenance contract, the
  single-resolver consolidation, and the Phase-2 promotion recipe (what a future apply-PR must do:
  copy enriched→canonical, retrain VAEP/xT/calibration, re-baseline goldens, promote the tripwire to
  a hard gate).
- **CLAUDE.md** `PR-S## ships …` line + the new `add_restart_coordinates` in the spadl-enrichment
  surface description.
- **NOTICE**: no new published methodology (Law-of-the-game geometry); the goal-kick lineage already
  cites the xT-GK work. Add a one-line entry if the tripwire/prior warrants it.
- **C4**: this adds one `add_*`-family helper in `spadl/utils.py` — **not** a `tracking.__all__`
  aggregator (no `add_*` enters `tracking.__all__`), no new KDE backend / trained model / tracking
  backend. Verify the tracking aggregator count is unchanged → **C4-free** (skip regen) after
  confirming tokens/count unchanged.
- **Version bump + tag + publish** per the per-PR convention (even though Phase 1 ships no wheel-level
  behavior change to existing consumers, the new public helper is a feature → minor bump).

## 9. Open spec-level details (resolved at implementation, not blocking)

- Exact tripwire tolerances per type (§6).
- Whether the destination `tracking_ball` tier (§4.2) earns its keep over `next_event` — drop if not.
- Whether to emit a single combined `*_coord_source` per coordinate or also a finer per-tier audit.

**Resolved (not open):** the module stays `_gk_geometry.py` — **no rename**. `_gk_geometry.py` is
reachable through the public export (`resolve_gk_geometry` in `tracking.__all__`); a rename would be
an API break (Minor 8). `resolve_restart_geometry` is *added* to the same module.

## 10. Effort

**Tier-5.** Per-type tier dispatch + side-precedence + `tracking_ball` origin/dest tiers +
tripwire-with-revert + `RestartCoordinateReport` dataclass + the `resolve_gk_geometry` delegation shim
(column-rename + label-map boundary) + `add_restart_coordinates` wrapper (with Examples docstring) +
red-first parity across all 4 call sites + the Major-2 edge-case + Major-3 completion-path fixtures +
ADR + CLAUDE.md. No training, no new dependency, no retrain. Phase 2 (canonical promotion + coordinated
retrain) is a separate future spec/PR.

---

### Sources
- Live lakehouse probe (`soccer_analytics.bronze.spadl_actions`, Databricks SDK), 2026-06-10.
- xT-GK goal-kick coverage spec `docs/superpowers/specs/2026-06-08-xt-gk-goalkick-coverage-design.md`
  §2 / D-A1; `silly_kicks/tracking/_gk_geometry.py`.
- Precedent: `add_pre_shot_gk_context` (frames-optional enrichment, ADR-005 §5); calibration
  recommend-then-apply split (ADR-009); GS own-goal geometry tripwire (ADR-018); provenance idiom
  (`is_goalkeeper_source`, `XtGkReport`).
- Coordinate convention verified against `silly_kicks/spadl/orientation.py` (canonical LTR) +
  `config.py` (105×68).
```
