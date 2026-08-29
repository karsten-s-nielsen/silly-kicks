# SB360 as a first-class tracking-feature provider — resolved keeper identity, a canonical producer, and deterministic dtypes

- **Status:** DRAFT (design, pre-implementation)
- **Date:** 2026-08-28
- **Branch:** `sb360-first-class-provider` (to be created)
- **Proposed ADR:** the next free ADR (**provisional** — confirm against `main` at commit-prep; ghost-gk took ADR-076, FOV-observability took ADR-077).
- **Prior art it extends:** ADR-053 (SB360 coverage audit — the source of findings #1/#2), ADR-054 (`providers/statsbomb` pure-shaping port + the D6 identity/team resolution), ADR-055 (single-source discipline — `GoalMap`), ADR-057 (pandas-span dtype discipline), ADR-058 (nullable id schema), ADR-062 (opt-in FOV companions + the derive-what-you-can-honestly identity principle), ADR-063/067 (velocity-availability / position-only), ADR-069 (detection-aware provider visibility), TF-13 (`acting_gk_from_frames` / `defending_gk_from_frames`).

> **Scope note (owner directive):** best-practice / gold-standard / long-term. Breaking changes and scope are explicitly NOT a constraint. Lakehouse ADOPTION of the new surfaces is OUT of scope — it receives a handoff after this cycle ships.

## Executive summary

StatsBomb-360 freeze-frames are the library's most information-rich tracking source and its most awkward
to consume: they are **anonymous** (each player row carries only actor-relative flags — `teammate` /
`actor` / `keeper` — and no player identity; `snapshot_to_tracking_frames` numbers the rows), so every
GK-domain feature that needs *which keeper* has no identity to key on; and the 34 tracking `add_*`
aggregators do **not** share one call shape, so every consumer re-derives per-aggregator handling (the
`ADAPTER_MAP` in `scripts/_sb_battery.py` is the current, scripts-only, one-copy). (The third
awkwardness the original survey flagged — a pandas-version-dependent snapshot id dtype — was **already
resolved in 4.79.0** (ADR-057/058) and verified during this cycle's planning; see Component 2, retained
for traceability with **no work**.)

This cycle makes SB360 a **first-class, single-sourced tracking-feature provider**, gold-standard:

1. **Keeper-identity resolution (keystone)** — one library seam that supplies the *real* keeper
   identity onto the anonymous frames via a documented source-precedence ladder (**event › roster ›
   positional derivation › NA**), **delegating** to the existing keeper-resolution functions on native
   providers rather than duplicating them (the roster/event ladder is the SB360-specific new work). This
   is what lets `add_pre_shot_gk_*` / `add_xt_gk` / `add_ghost_gk` actually score on SB360.
2. **Deterministic snapshot id dtypes — ALREADY SHIPPED (4.79.0, ADR-057/058); NO work this cycle.**
   Retained as a numbered component only so the original survey's finding #1 stays traceable.
3. **A canonical tracking-aggregator call convention** — one signature shape, deviations normalised
   (breaking OK), pinned by a consistency gate.
4. **A first-class, provider-agnostic producer** — `tracking.run_tracking_features(...)` orchestrates
   the whole frame-consuming `add_*` family through the canonical convention (Component 3), pre-links +
   shares the pitch-control cache once, takes injected `xt`/`xg` + roster + `visible_area`, and returns
   the **enriched ACTIONS** (action-grain feature columns — the `add_*` family is action-coupled, not a
   tracking-frame producer) plus a report. Scripts/tests/audit re-point at it.
5. **The ADR-054 `_defending_goal` stale-note pointer**, folded into the single commit.

**Everything is dependency-inverted** — the library consumes injected artifacts (roster dict, fitted
`ExpectedThreat`, `xg_column`), the driver (scripts) builds them, and `providers/statsbomb` stays
pure-shaping (ADR-054). Provider-agnostic where the concern is (the aggregators are); SB360-specific
only where the data is.

## Problem

### P1 — SB360 keeper anonymity (the primary gap)

`providers/statsbomb/parse.py::shape_snapshots` resolves each player's **real `team_id`** (the ADR-062
D6 derivation: the freeze-frame gives no team labels, but `actions` names the actor team and a two-team
match makes `teammate` fix every player's real team) and sets `is_goalkeeper` from the `keeper` flag —
but **player identity is deliberately absent** (`parse.py:230`); `snapshot_to_tracking_frames`
(`tracking/_snapshot.py:24`) numbers the rows, and a row number does not recur across frames. So a
consumer knows *which row* is the defending keeper (`keeper AND NOT teammate`, `parse.py:83-92`) or the
acting keeper (`keeper AND teammate`, `:114`) — but not *who*. Every GK-domain aggregator that keys on a
keeper identity (`add_pre_shot_gk_position` *requires* a `defending_gk_player_id` column;
`add_xt_gk`/`add_ghost_gk` need the resolved keeper) therefore has nothing to key on.

The identity **exists outside the frame**, in two places, both authoritative:

- **Roster / lineup** — `scripts/_sb_raw.py::parse_roster` → `{player_id: {name, jersey, team,
  position}}`; filtering `position == "Goalkeeper"` keyed by team yields a `{team_id: gk_id}` map. This
  names the *defending* keeper (whom no event names). Driver-side, raw-JSON.
- **Goal-kick actor** — a StatsBomb goal kick maps to the SPADL `type_name == "goalkick"`
  (`spadl/statsbomb.py` → `spadlconfig`), whose `player_id` is the taker: the
  *acting/distributing* keeper, already resolved in `actions`. More authoritative than the roster for
  that row (it is the actually-recorded actor), and it beats a stale roster starter after a substitution
  for free.

**Three traps any resolution must not hit (ADR-062 D6 lessons):**
1. The synthetic `{0,1}` team fallback breaks the action↔frame join → *every* direction-dependent
   feature goes honest-NaN (the real 4.76.0→4.84.0 defect). Any re-shaping MUST preserve real-`team_id`
   resolution.
2. **Identity ⟂ velocity.** SB360 is velocity-less, so ΔDAS / accessible-space is NaN regardless of
   identity (ADR-063); zero-velocity positional models (pitch control) still score. Naming the keeper
   does NOT make a velocity-constitutive metric scoreable — do not conflate "can't name" with "can't
   score."
3. Mid-match GK substitution — the roster names the *starter*; the event (goal-kick taker) or a
   substitution-event pass is needed for per-minute correctness.

### P2 — no canonical aggregator call convention (finding #2)

The 33 `add_*` tracking aggregators do not share one call shape: six need a fitted `ExpectedThreat`
injected; `add_defensive_credit` needs an `xg_column` the library does not ship;
`add_pre_shot_gk_angle` takes `frames` **keyword-only** (`features.py:846`) while its sibling
`add_pre_shot_gk_position` takes it **positionally** (`features.py:726`); `add_sync_score` takes `links`,
not `frames` at all (TF-6). The per-aggregator adapter that absorbs this lives *scripts-only* in
`scripts/_sb_battery.py::ADAPTER_MAP` — "exactly the per-aggregator adapter layer that silently drifts"
(its own docstring). Every consumer (lakehouse, tests, scripts) re-derives it.

### P3 — pandas-version-dependent snapshot id dtype (finding #1) — ALREADY RESOLVED (4.79.0)

The original survey (ADR-053 consequences / the coverage README) recorded that
`snapshot_to_tracking_frames` returned an `Int64` id as `Int64` on pandas 2.3.3 but `Float64` on 3.0.x
(the all-NA ball row upcasting ids on the concat). **This was fixed in 4.79.0 (ADR-057/058):**
`_snapshot.py::_cast_to_declared_schema` now casts `player_id`/`team_id` to the declared nullable
`Int64` after assembly, and the fix is guarded by `tests/tracking/test_snapshot_id_dtype_across_pandas.py`
(behaviour via `id_compat`, plus the literal-`Int64` pin and a full-schema per-column check). Verified
during this cycle's planning: on pandas 3.0.3 the snapshot emits `player_id` dtype `Int64` and **zero
FutureWarnings**. This section is retained only so the survey's finding stays traceable to its
resolution — there is **no work** here.

### P4 — a stale ADR note (ADR-054)

ADR-054 records `_ghost_gk._defending_goal` as a live orientation defect "queued in `TODO.md`." It was
**resolved by ADR-055 (4.77.0)** — the fork was deleted and replaced by the canonical `GoalMap` — and no
TODO row ever existed. The stale note nearly re-sent a reviewer chasing a non-defect.

## Non-goals

- **Fetching or parsing raw StatsBomb data in the library.** `providers/statsbomb` stays pure-shaping
  (ADR-054); the roster is built driver-side and injected.
- **Shipping an xG or xT model.** The library ships neither (ADR-009/024); the producer takes them
  injected.
- **Making velocity-constitutive metrics score on SB360.** DAS/pre-window/actor-speed stay honest-NaN
  (ADR-063) — foreclosed, not deferred.
- **Per-player aggregation on SB360** beyond the keeper — freeze-frames carry no non-keeper identity
  (numbered rows); foreclosed.
- **Lakehouse adoption** of the new producer/resolver — a documented handoff, not this cycle.

## Design

### Component 1 — keeper-identity resolution (keystone)

**One resolution with a documented source-precedence ladder.** It has two paths, split on `identity`, and
**neither reimplements what already resolves keeper identity** (the ONE-resolution invariant, N1):

- **Native path (`identity="native"`) DELEGATES to the existing TF-13 functions.**
  `_gk_resolve.defending_gk_from_frames` and `acting_gk_from_frames` **already return the keeper's real
  `player_id`** — `defending_gk_from_frames` reads the opposing team's `is_goalkeeper` row's `player_id`
  (`_gk_resolve.py:216`), and `acting_gk_from_frames` adds a roster-stable `is_goalkeeper` **identity
  fallback** plus sub-nearest-in-time resolution (`:256-265`). They resolve *identity*, not merely
  location, on providers whose frames carry a real `player_id` (kloppy/sportec/GS). `resolve_keeper_identities`
  therefore **calls them** on the native path and does not re-derive — otherwise it is a second
  native-identity path, exactly the fork the invariant forbids. (`_gk_identification.derive_goalkeepers`
  remains the positional *location* fallback the TF-13 functions already lean on.) Those functions stay
  **public and per-action**, so a consumer needing per-action sub resolution keeps using them directly.
- **SB360 path (`identity="roster"`) is the NEW work** — SB360 frames carry numbered rows (no real
  `player_id`), so the frame-based functions would return row numbers; the roster/event ladder supplies
  the identity the anonymous frame lacks. This is the only genuinely new identity resolution in the
  cycle.

**Precedence (per keeper role, most-authoritative first):**

> **event-carried id › roster (`{team_id: gk_id}`) › positional derivation › NA (dropped-and-counted).**

- **Defending keeper** (needed by `add_pre_shot_gk_*`, `add_xt_gk` defended, `add_ghost_gk` on a
  shot/save): no event names the opponent's keeper → **roster** is the top available rung → derivation →
  NA.
- **Acting / distributing keeper** (a goal kick): the `type_name == "goalkick"` action carries the taker
  → **event** is the top rung → roster → derivation → NA. The event beating the roster is what makes
  mid-match subs correct for that row for free.

**Cross-validation, not silent override.** Where two rungs both name a team's keeper (a goal-kick taker
*and* the roster keeper for that team), a disagreement is a roster/sub error — **surfaced** (a warning +
a provenance token), never silently reconciled.

**Public surface (return shape DECIDED here — it is the keystone's contract):**
- `tracking.resolve_keeper_identities(actions, frames, *, identity, roster=None) ->
  tuple[KeeperIdentityMap, KeeperIdentityReport]`
  — a **PURE resolved-identity MAPPING** (+ report), not an enriched grain. It returns
  `{(game_id, period_id, team_id) → (gk_id, keeper_id_source, conflict)}`: the best-available keeper
  identity for each team-per-period from the precedence ladder (the `conflict` flag is F2, below), plus a
  `KeeperIdentityReport` (per-source counts + the unresolved and conflict counts). It **mutates neither `actions` nor `frames`** — the *caller* owns
  placement (F1, hexagonal, one source of truth), applied via two **pure placement helpers** shipped
  alongside the resolver so every consumer bridges the SAME way: `add_defending_gk_player_id(actions,
  map)` (stamps `defending_gk_player_id` on the actions — `add_pre_shot_gk_position`'s required input)
  and `apply_keeper_identities_to_frames(frames, map)` (stamps the real id onto the frames' keeper rows —
  the **R1 identity→frame bridge**, applied on the anonymous-frame/roster path). A `(game, period, team)`
  key uniquely names a team's keeper, so "role" is the caller's *lookup selector*, not a key dimension:
  the producer looks up the **defending** team for the action stamp; the gkdv driver (A3) uses the same
  helpers to stamp per-frame arm values. One mapping + two helpers serve every consumer.
  - `identity: Literal["native", "roster"]` is **explicit per provider, never guessed** ("roster
    present" ≠ "use roster" — the ADR-069 detection-aware discipline). `native` = trust an id the frames
    already carry (non-SB360 providers); `roster` = SB360's injected-roster path.
  - `roster: dict | None` — the plain injected `{team_id: gk_id}` map. `None` under `identity="roster"`
    with no event fallback → that team is `unresolved` → NA, counted.
  - **`keeper_id_source` vocabulary** (F2): `{event, roster, native, derived, unresolved}` records the
    **winning rung** (per precedence), NOT the disagreement. The roster path emits `event` (goal-kick
    actor) / `roster` (injected); the native path emits `native` / `derived` — the value it inherits from
    the resolved keeper's frame `is_goalkeeper_source` (a sportec/GS keeper carrying a real provider id is
    `native`, a positionally-derived one is `derived`), so the resolver never loses the provider-vs-inferred
    distinction. A roster-vs-event disagreement is a *separate durable diagnostic*: the report carries a
    `keeper_id_conflict` count and the mapping value flags it (a `conflict: bool` alongside
    `(gk_id, source)`), so a corpus run recovers disagreements from the output, not only a transient warning.
  - **Granularity:** the mapping is per-`(game, period, team)` — a mid-period substitution (two keeper
    ids for one team within a period) is the documented refinement (substitution-event resolution, out
    of scope), surfaced as a conflict, never silently coalesced.
- A thin accompanying helper (or a documented driver recipe) that turns `parse_roster(...)` output into
  the `{team_id: gk_id}` map (`position == "Goalkeeper"`, keyed by team) — **driver-side**, so the
  library never parses raw JSON.

**Single-source coordination (ADR-055 + ADR-037) — load-bearing.** This resolver is THE ONE
keeper-identity seam; the in-flight TF-19 A+2 (GKDV) cycle **drops its planned gkdv-local resolver and
consumes this one**. It lives in `tracking/` because it must serve the tracking GK families
(`add_pre_shot_gk_*` / `add_xt_gk` / `add_ghost_gk`) — a gkdv-local one could not. **ADR-037 confines
gkdv to importing only `tracking._das` (via `_das_port`), so a gkdv-domain consumer reaches this resolver
DRIVER-side (scripts may import `tracking` freely), NEVER via a gkdv library import.** The mapping return
shape (above) is what serves that driver consumer.

**Real-team-id preservation (trap 1):** the resolver operates on frames whose `team_id` is already the
real ADR-062 D6 value; under `identity="roster"` it must refuse (raise) rather than proceed on the
synthetic `{0,1}` fallback pair, because a synthetic team makes the roster key meaningless. **This raise
is a NEW guard, not a mirror of `shape_snapshots`** — `shape_snapshots` itself does NOT raise on the
synthetic fallback; it silently emits the `{0,1}` pair per-action (`parse.py:297`) when the two teams are
unresolvable. The resolver detects the synthetic signature the robust way: under `identity="roster"`, if
NONE of the frames' non-ball `team_id`s intersect the supplied `roster`'s keys, the frames are the
synthetic fallback (or the roster is for the wrong match) → raise with a message naming both. (This is
also why the roster is keyed on the *real* team ids — the ADR-062 D6 derivation is the precondition.)

**Honest boundaries (traps 2/3):** the resolver names keepers; it does not compute velocity — a
velocity-constitutive metric stays NaN whether or not the keeper is named. The roster is the *starter*;
the event rung supplies mid-match-sub correctness where an event names the keeper, and a
substitution-event source is the documented refinement (out of scope here).

### Component 2 — deterministic snapshot id dtypes — ALREADY SHIPPED (4.79.0); NO WORK

This component is **already complete** and is retained only for traceability. `_cast_to_declared_schema`
(`_snapshot.py:203`, landed 4.79.0 / ADR-057/058) casts `player_id`/`team_id` to the declared nullable
`Int64` after assembly; genuinely-string ids keep `object` (per the caller's domain, decided per column).
Verified during planning on pandas 3.0.3: the snapshot emits `player_id` dtype `Int64` and **zero
FutureWarnings** (the concat's all-NA columns no longer warn, so there is nothing to "eliminate at its
source"). The F8 two-contract pinning the review asked for **already exists**:
`tests/tracking/test_snapshot_id_dtype_across_pandas.py::test_output_dtypes_match_the_declared_frames_schema`
derives its expectation from `TRACKING_FRAMES_COLUMNS`, so an ADR-058 schema edit moves this test — the
nullable-schema contract and the determinism contract fail together or hold together. **The plan carries
no task for this component.** (If the reviewer wants belt-and-suspenders, a one-line `pytest.warns`-free
assertion on the snapshot path is a cheap optional micro-hardening — but it guards a warning that does
not currently fire.)

### Component 3 — a canonical tracking-aggregator call convention

**One canonical signature shape** for the frame-consuming `add_*` family:
`add_*(actions, frames, [xt], *, links=None, <other injected models keyword>, visible_area=None, ...)`.
Concretely, the gate enforces exactly two rules per frame-consuming aggregator:
1. **`frames` is NEVER keyword-only** — it is positional-or-keyword, next to `actions` (both required
   core inputs).
2. **Every OPTIONAL parameter is keyword-only** (behind the `*`).
A single **REQUIRED fitted model** (`xt: ExpectedThreat`) MAY sit as the 3rd positional parameter — an
allowed, documented convention, not a deviation.

**Why a required model is allowed positional, not forced keyword-only (a REFINEMENT of the spec's earlier
"everything keyword-only", measured).** The mis-wiring class a canonical convention should foreclose is
*optional* args (`links`, `visible_area`, config) — those are already keyword-only behind the `*`. A
required `xt` is a distinctively-typed `ExpectedThreat`; passing the wrong thing 3rd (e.g. a `links`
DataFrame) fails immediately on `xt.rate(...)`, so it is not a silent bug class. It is also the
pandas/sklearn convention for required inputs (`fit(X, y, sample_weight)`). **Measured cost of forcing
keyword-only:** the five positional-`xt` aggregators (`add_gk_influence`, `add_cover_shadows`,
`add_off_ball_run_values`, `add_player_influence`, `add_xt_gk`) have **108 call sites across 24 files**
(production, the atomic mirrors, calibration, tests). Churning all of them buys uniformity with **no
correctness payoff** (type-guarded) and barely simplifies the producer (which routes models per-family
either way, F3). YAGNI: not worth it. The convention therefore admits the required-model-positional
shape, and the ONE genuine inconsistency — a `frames` that is keyword-only in one sibling and positional
in the other, which makes `add_pre_shot_gk_angle(a, f)` raise while `add_pre_shot_gk_position(a, f)`
works — is fixed:
- `add_pre_shot_gk_angle`: `frames` keyword-only → positional-or-keyword, matching
  `add_pre_shot_gk_position` and the canonical shape above. **This is the only signature change.**
- `add_sync_score` is the **link-consumer family** (TF-6: `add_sync_score(actions, links, *, ...)`, no
  frames) — it is *classified* (exempt), not force-fit into the frame shape.
- `add_visible_area_coverage` (no `frames`; requires `visible_area`) and
  `add_gradientsports_player_ids` (a jersey/roster helper over different inputs) are likewise
  *classified* exempt.
- **Consistency gate:** a `tests/` registry gate (the mirror/purity/id-scalar idiom) that derives the
  frame-consuming `add_*` surface from `tracking.__all__` and asserts each conforms to the two rules
  above (or is listed in a small `_CALL_SHAPE_EXEMPT` dict with a reason — `add_sync_score`,
  `add_visible_area_coverage`, `add_gradientsports_player_ids`). Two meta-assertions pin the registry to
  the public surface in both directions (a new aggregator with a deviating shape, or a stale exemption,
  fails CI). Lands **red-first** on `add_pre_shot_gk_angle`'s current keyword-only `frames`.

### Component 4 — a first-class, provider-agnostic producer

`tracking.run_tracking_features(actions, frames, *, links=None, xt=None, xg_column=None, roster=None,
identity="native", visible_area=None, home_team_id=None, families=None, pitch_control_cache=None) ->
tuple[pd.DataFrame, TrackingFeaturesReport]` — the canonical way to run the full frame-consuming `add_*`
family and return the **enriched ACTIONS** (action-grain feature columns; the family is action-coupled —
this is not a tracking-frame producer) **plus a structured report**.

- **Its dispatch is model-injection routing + family classification, NOT a port of today's per-aggregator
  ADAPTER_MAP (F3).** Component 3 normalises the frame-consumer *signatures* to one shape, which
  dissolves most of what `scripts/_sb_battery.py::ADAPTER_MAP` does today (absorbing per-aggregator call
  *shapes*). What remains for the producer after normalisation is (a) **which injected model each family
  needs** (`xt` vs `xg_column` vs neither), and (b) the **family classification** that keeps the
  link-consumer (`add_sync_score`, TF-6) and any velocity-constitutive families on their own path. The
  scripts-only ADAPTER_MAP is retired *into* this normalised dispatch (not lifted verbatim);
  `scripts/_sb_battery.py` + `tests/sb360/_calls.py` re-point at the producer (tests → scripts → lib
  layering; the audit's ADAPTER_MAP becomes a re-export, as it already did once). One copy.
- **Provider-agnostic** — the aggregators are; SB360 is the first consumer, sportec/skillcorner/metrica
  get it for free. SB360-specific concerns enter as *parameters*: `roster`/`identity` (Component 1),
  `visible_area` (the 4.99.0 opt-in companions).
- **Performance by construction** — pre-link once (`link_actions_to_frames`) and share one
  `PitchControlCache`, passing `links`/`pitch_control_cache` to every family (the existing lakehouse
  pattern, now library-owned).
- **Dependency-inverted models** — `xt` (a fitted `ExpectedThreat`) and `xg_column` are injected; where
  an aggregator needs one and it is absent, that family's columns are honest-NaN (the ADR-063 discipline),
  not a fabricated value.
- **Keeper identity threaded onto BOTH grains** — the producer runs Component 1 first, then applies the
  map with two pure placement helpers: `add_defending_gk_player_id` stamps `defending_gk_player_id` on
  the actions, and — **on the roster path (anonymous SB360 frames) only** — `apply_keeper_identities_to_frames`
  bridges the resolved id onto the frames' `is_goalkeeper` rows. **The frame bridge is load-bearing (R1):**
  `add_pre_shot_gk_position` locates the keeper by matching `frame.player_id == defending_gk_player_id`
  (`tracking/utils.py:1034`), and SB360 frames carry synthetic numbered ids, so without the bridge the
  roster id matches no frame row and every GK-position feature is NaN — the cycle's headline deliverable
  would silently not happen. The bridge is NOT applied on the native path (frames already carry real ids;
  stamping the per-period consensus would clobber a mid-period sub). `add_ghost_gk` is unaffected (it
  finds the keeper by `is_goalkeeper`, not by identity match).
- **`families` selector** — run all, or a named subset. The producer runs each family under a per-family
  guard (the `run_add_star_battery` precedent) and NEVER crashes: a family that self-degrades on
  declared-velocity-less frames emits its NaN value columns with a provenance token (`add_das` catches
  `DasUnscoreableError` → `das_source="unscoreable_frame"`; the ADR-063 four lift Tier-1, NaN Tier-2); a
  family that would raise is caught and recorded as `skipped`. Either way, a velocity-constitutive metric
  stays NaN — naming the keeper does not make it score (ADR-063).
- Returns the **enriched ACTIONS** (the caller's actions with all emitted feature columns) plus a
  structured report (per-family status + the drop/unresolved counts — the ADR-052 conservation idiom).

### Component 5 — ADR-054 stale-note pointer

Append one line to ADR-054's `_defending_goal` bullet: "→ RESOLVED by ADR-055 (4.77.0): the fork was
deleted and replaced by the canonical `GoalMap`." Folded into the single commit. No code.

## Public surface

**New (library):**
- `tracking.resolve_keeper_identities(actions, frames, *, identity, roster=None) ->
  tuple[KeeperIdentityMap, KeeperIdentityReport]` (the pure
  `{(game, period, team) → (gk_id, source, conflict)}` mapping + report) + the `keeper_id_source`
  provenance vocabulary `{event, roster, native, derived, unresolved}` and the separate `keeper_id_conflict`
  diagnostic. The native path delegates to the existing (unchanged) `defending_gk_from_frames` /
  `acting_gk_from_frames`.
- `tracking.add_defending_gk_player_id(actions, keeper_map)` and
  `tracking.apply_keeper_identities_to_frames(frames, keeper_map)` — the two pure placement helpers that
  apply the map to the actions and (the R1 bridge) the frames' keeper rows respectively.
- `tracking.run_tracking_features(...)` (+ its result/report type).
- Possibly a small `tracking`/`providers.statsbomb` helper for the roster-map recipe (or a documented
  driver recipe only — decided in the plan).

**Changed (breaking, intended):**
- `add_pre_shot_gk_angle` (and any other normalised aggregator) signature.
- (The `snapshot_to_tracking_frames` id-dtype determinism was a breaking change, but it **already
  shipped in 4.79.0** — not this cycle.)
- `scripts/_sb_battery.py::ADAPTER_MAP` moves to the library (re-export left behind).

## Invariants & impact

- **Retrain / Hyrum (real, intended).** Resolving keeper identity turns SB360 GK-domain features from
  honest-NaN into VALUES — a retrain trigger for any SB360 GK-feature consumer, and a Hyrum change for
  the snapshot dtype and the normalised signatures. This is the deliverable, recorded, not a regression.
  Non-GK / non-SB360 output is unchanged.
- **`providers/statsbomb` stays pure-shaping (ADR-054).** No fetch/parse added; the roster is injected.
- **Single-source (ADR-055).** Keeper identity has ONE resolution: the native path **delegates** to the
  existing TF-13 `*_gk_from_frames` functions (which already return the keeper `player_id`) and only the
  SB360 roster/event ladder is genuinely new — no second native-identity path (N1). The call shapes have
  ONE copy (the producer). No new fork.
- **Honest degradation (ADR-063/054).** Velocity-constitutive families stay NaN; unresolved keepers →
  NA-counted; unavailable models → NaN family, never fabricated.
- **C4.** The producer is a new public seam (a container/interface); `run_tracking_features` is an
  orchestrator, not a new action-coupled aggregator — the aggregator count is unchanged, but the C4
  model gains the producer + resolver surface (regen if the documented count/surface changes).

## Testing strategy

Following the codebase's "both-sides + non-vacuity + registry-completeness" discipline:
- **Keeper identity (the keystone — each case lands RED-first, F7):** every resolution case is written as
  a failing assertion against the not-yet-existing resolver first, then made green — a fixture where the
  defending keeper resolves from **roster**, a goal-kick where the acting keeper resolves from the
  **event** (and *overrides* a deliberately-wrong roster starter — the mid-match-sub case), an
  **unresolvable** case (no roster, no event → NA + `unresolved`, counted, never fabricated), and a
  **cross-validation disagreement** (roster vs event → a surfaced warning **and** a durable
  `keeper_id_conflict` count + `conflict=True` on the mapping value, so the disagreement is recoverable
  from the output, not only the transient warning — F2). Assert the synthetic `{0,1}` team pair
  **raises** (trap 1). Assert a velocity-constitutive metric stays NaN *even when the keeper is named*
  (trap 2 — the non-vacuity that names ⟂ velocity).
- **Dtype determinism (ADR-057/058) — no new test; already covered.** The existing
  `tests/tracking/test_snapshot_id_dtype_across_pandas.py` asserts the *behaviour* (ids compare equal
  through `id_compat` across the CI pandas-span), the literal `Int64` pin, and the full-schema per-column
  check, and its `test_output_dtypes_match_the_declared_frames_schema` derives from
  `TRACKING_FRAMES_COLUMNS` — the F8 two-contract pinning is therefore already met. The plan does not
  add a Component-2 test.
- **Call convention:** the consistency gate lands **red-first** (observed failing on the current
  deviations), then green; a non-vacuity plant (a synthetic deviating signature is flagged).
- **Producer:** run the full family on a paired fixture; assert (a) the enriched output equals composing
  the individual `add_*` calls **applied after the SAME keeper-identity resolution AND frame bridge the
  producer threads first** — the composition baseline is `resolve_keeper_identities` →
  `add_defending_gk_player_id` → `apply_keeper_identities_to_frames` (the R1 bridge) → `add_*`, NOT the
  naive `add_pre_shot_gk_position(actions, frames)` on identity-less actions or unbridged frames, or the
  GK families are not byte-equal and the test is wrong (F6). **Plus a MANDATORY non-vacuity assertion
  (R1): `pre_shot_gk_x.notna().any()`** — without the bridge the geometry is NaN on both sides and the
  equality passes VACUOUSLY (NaN == NaN); the assertion proves the SB360 GK feature actually unlocked.
  The producer adds no behaviour, only orchestration + single-sourcing. Also assert (b) the
  shared-cache/pre-link path is byte-identical to per-family calls, (c) the report conserves (families
  run + skipped == families in), (d) injected-model-absent → honest-NaN family, (e) the `visible_area`
  companions appear when supplied and the primary columns are byte-identical without it.
- **Placement helpers (R1):** a dedicated control-and-treatment test — `add_pre_shot_gk_position` on
  unbridged SB360 frames is all-NaN (control), and non-NaN after `apply_keeper_identities_to_frames`
  (treatment) — proving the bridge is load-bearing, plus purity (both helpers return copies).
- The existing SB360 audit (`tests/sb360/`) re-points at the library producer and stays green (the
  ADAPTER_MAP move is output-preserving).

## Out of scope / follow-ups (surfaced, not silently deferred)

- **Native-path `defending_gk_player_id` reconciliation with `add_pre_shot_gk_context`.** On the native
  path the producer stamps `defending_gk_player_id` from the resolver MAP (a per-`(game, period, team)`
  consensus of `defending_gk_from_frames`). The established standalone flow
  `spadl.utils.add_pre_shot_gk_context` populates the same column differently: EVENT-derived keeper first
  (a recorded keeper action), then a `defending_gk_from_frames` FALLBACK for NaN rows (`utils.py:732`).
  The two share the TF-13 source only on the frame-fallback portion, so they can differ where the
  event-primary keeper wins, and at a mid-period sub (per-action vs per-period consensus). This is **not
  a regression** — the producer is a NEW entry point and does not change `add_pre_shot_gk_context`; a
  native consumer wanting the event-primary richer stamp keeps using it directly. Reconciling the two
  native paths (e.g. the producer delegating to `add_pre_shot_gk_context` on the native path) is a
  follow-up. This cycle is SB360-focused, where the roster path is the only option and no such second
  path exists.
- **Lakehouse adoption** of `run_tracking_features` / `resolve_keeper_identities` — documented handoff.
- **Substitution-event keeper resolution** (per-minute correctness beyond the roster starter + the
  goal-kick-event rung) — the mid-match-sub refinement.
- **Non-keeper SB360 identity** — foreclosed (numbered rows).
- **Velocity-constitutive metrics on SB360** — foreclosed (ADR-063).
- **The ADR-077 per-zone/axis FOV descriptor** and **a bespoke ghost-observability model** — separate
  future items.

## Attribution / decisions

- Decision: the next free ADR (**provisional**), written with the implementation.
- Extends ADR-053/054/055/057/058/062/063/067/069 and TF-13; see `NOTICE` for any new methodological
  citation the plan introduces (none anticipated — this is orchestration + identity plumbing, no new
  algorithm).
