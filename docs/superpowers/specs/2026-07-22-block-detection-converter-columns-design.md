# Block-detection converter columns (`shot_blocked` + `cross_blocked`) — design

**Date:** 2026-07-22
**Target:** silly-kicks (next free release — confirm with owner at commit-prep) / PR-S`<NN>` / ADR-0`XX`
**Status:** design, awaiting owner review
**Relationship:** **prerequisite** for TF-51 (`docs/superpowers/specs/2026-07-22-tf51-defensive-credit-design.md`) — ships FIRST as its own PR. TF-51's `shot_block` rule consumes `shot_blocked`; bravery consumes **both** (numerator = blocked shots + blocked crosses).

---

## 1. Problem

Canonical SPADL has **no "blocked" `result_id`** — a blocked shot or blocked cross flattens to
`fail`, indistinguishable from off-target / saved / overhit / intercepted. Yet the blocked signal is
**present in the raw stream of most providers** and dropped during conversion.

TF-51 needs it in two places: the **`shot_block`** credit rule (per-defender attribution) and the
event-only **bravery** metric (% of opposition *final actions* blocked — Tigres 32/40, where final
actions = **shots + crosses**). This PR surfaces the already-parsed signal as two first-class,
nullable, cross-provider columns — **`shot_blocked`** and **`cross_blocked`** — so TF-51 (and any
future consumer) reads clean booleans instead of re-deriving them.

## 2. Feasibility (fixture- AND real-data-verified)

Two signals, investigated separately (two provider audits each) and **probed on real pining data**
where the audits were inconclusive.

### 2.1 `shot_blocked` — present-but-dropped in 6 of 8

| provider | source | cost | verification (BD-1) |
|---|---|---|---|
| Gradient Sports | `shot_outcome_type=="B"` (`gradientsports.py:180/244`) | trivial | **pining-probed** (real WC2022) |
| StatsBomb | `shot.outcome.name=="Blocked"` (`statsbomb.py:90/530`) | trivial | **fixture-verified** (8/12/11 blocked shots in the 3 committed fixtures) |
| Metrica | `subtype.str.endswith("BLOCKED")` (`metrica.py:472`) | trivial; sparse | **fixture-verified** (n=1, sparse) |
| Kloppy gateway | `ShotResult.BLOCKED` (`kloppy.py:34/437`) | trivial | enum-verified (live) |
| Sportec/DFL | `shot_outcome_type=="blocked"` (`sportec.py:915`) | trivial | parse-port maps `BlockedShot`; **probe instances at impl** |
| Wyscout | tag 2101 → `df_events["blocked"]` | moderate | **mechanism-only** (no fixture, not in pining) — **probe at impl** |
| Opta | qualifier (id 82?) — **unverified in-repo** | **deferred → `pd.NA`** | — |
| SkillCorner | absent (public `dynamic_events.csv` ≡ owner-tier RM `events.parquet`, identical 294-col Game Intelligence schema; no shot-outcome granularity) | **infeasible → `pd.NA`** | **real-data verified absent, both tiers** |

### 2.2 `cross_blocked` — present in 3 of 8 (incl. GS, the owner's primary data)

| provider | source | cost | verification (BD-1/BD-2) |
|---|---|---|---|
| **Gradient Sports** | `crossOutcomeType=="B"` (⟺ `incompletionReasonType=="BL"`) | **trivial** | **pining-probed** (real WC2022, 6/39 crosses) |
| Wyscout | tag 2101 (shot+pass-shared) scoped to `type_id==8 & subtype_id==80` (cross) | moderate | **mechanism-only** (no fixture) — **probe at impl** |
| StatsBomb | `Block` event via `related_events` + `pass.cross==true` — native only (kloppy collapses it) | **deferred → `pd.NA`** | **n=1 only → deferred** (BD-2; gate on a StatsBomb-corpus probe, like Opta) |
| Opta | "blocked pass" event exists but real feed + team-attribution unverified | **deferred → `pd.NA`** | low |
| Kloppy gateway | **no `PassResult.BLOCKED`** (verified: `{COMPLETE, INCOMPLETE, OUT, OFFSIDE}`) | **infeasible → `pd.NA`** | high |
| Sportec/DFL | no blocked-cross field (`play_evaluation` is binary; `GoalKeeperInterference` ≠ block) | **infeasible → `pd.NA`** | high |
| Metrica | structural (failed passes are separate `BALL LOST`; `BLOCKED` only pairs with SHOT) | **infeasible → `pd.NA`** | high |
| SkillCorner | absent — real-data verified (`pass_outcome` ∈ {successful, unsuccessful, offside}; `unsuccessful` conflates blocked) | **infeasible → `pd.NA`** | high |

**The GS real-data probe (2026-07-22, match 10502) is load-bearing.** The provider audit reported GS
`cross_blocked` "unverified / no observed B" — but that was a **synthetic-fixture artifact** (the GS
test generator only emits `crossOutcomeType ∈ {null, C, F}`). The real WC2022 feed shows
`crossOutcomeType ∈ {C:7, D:24, B:6, O:2}` — **`B` = blocked, 6 of 39 crosses** — and a cross-tab
confirms `crossOutcomeType=="B"` and `incompletionReasonType=="BL"` are **perfectly aligned** (all 6,
nothing else). This mirrors the shot `"B"` pattern exactly (same 1-line-mask cost). *(Same
synthetic-fixture trap that nearly mis-verdicted SkillCorner — real data is the arbiter.)*

Where a signal exists, providers **separate blocked from saved/overhit**, so both columns mean an
**outfield-defender block**, not a keeper save or a miscontrol.

**Verification discipline (BD-1).** The GS lesson — a synthetic fixture *hid* `crossOutcomeType=="B"`
and real data was the arbiter — generalizes: **every feasible provider × column not `pining-probed`
or `fixture-verified` for actual blocked *instances* gets a real-data instance probe at
implementation** (Wyscout both columns; DFL/kloppy blocked-instance counts), or is explicitly
recorded as mechanism-only with the risk accepted. Never ship a `True`/`False` column on a provider
whose blocked instances have never been seen in real data — that is exactly the near-miss GS escaped.

## 3. Column contract

Two new **nullable-boolean** columns (pandas `boolean` dtype), **emitted consistently by every
converter** so consumers get stable columns regardless of provider:

| column | scope | `True` | `False` | `pd.NA` |
|---|---|---|---|---|
| `shot_blocked` | shot action types | shot blocked by an opponent | shot, known not blocked | not a shot **or** unknown-provider shot |
| `cross_blocked` | `cross` action type | cross blocked by an opponent | cross, known not blocked | not a cross **or** unknown-provider cross |

- **`True`/`False` vs `pd.NA` is the load-bearing distinction.** A provider that *can* encode blocked
  emits `True`/`False` on the in-scope actions; a provider that cannot emits `pd.NA`. This is the
  seam TF-51's bravery "unknown ≠ 0 %" guard (R2-2) reads — all-`NA` in-scope actions for a (team,
  game) → bravery `NaN`, not `0 %`.
- **Opponent-block semantics.** Where a provider flags own-team deflections (DFL
  `shot_outcome_blocked_by_own_team`, `sportec.py:225`) exclude them → `False`. Otherwise the
  provider's blocked flag is taken as an opponent block (the common case).
- **`cross_blocked` scope + set-piece crosses (BD-3).** `cross_blocked` covers the **open-play
  `cross`** action type. Set-piece crosses (`corner_crossed`, `freekick_crossed`) are a distinct
  phase and are **out of v1 scope** (`pd.NA` on those types); TF-51 bravery's "crosses faced"
  denominator counts open-play `cross` only in v1 (set-piece crosses documented-deferred).
- **`False` reliability on tag-based providers (BD-3).** For Wyscout (`False` = "no 2101 tag"), the
  accuracy of the `False` class depends on tag completeness — a dropped tag reads as a
  false-negative. Documented; the BD-1 real-data probe bounds it.
- **Placement.** Threaded through each converter's output + `_finalize_output` dtype contract
  (nullable `boolean`). Whether the columns register in shared `SPADL_COLUMNS` or as documented
  consistently-emitted extensions (like `is_synthetic` / `result_source` / `tackle_winner_*`) is a
  plan detail; the contract is "present on every converter's output, `pd.NA` where unavailable."
  Additive — existing columns/values unchanged.

## 4. Per-provider implementation

Each is additive next to the existing outcome derivation; no existing column/value changes.

| provider | `shot_blocked` | `cross_blocked` |
|---|---|---|
| **Gradient Sports** | `is_shot & (shot_outcome_type=="B")` | `is_cross & (cross_outcome_type=="B")` — the converter reads `cross_outcome_type` for `"C"` only (`gradientsports.py:240`); add the `"B"` mask |
| **Wyscout** | restrict `df_events["blocked"]` to shots + wire into `_create_df_actions` | restrict to `cross` (type 8/subtype 80) + wire into `_create_df_actions` |
| **StatsBomb** | `is_shot & (_shot_outcome=="Blocked")` (native) | **`pd.NA`** — deferred (BD-2: n=1-verified + fragile `related_events` join; gate a real StatsBomb-corpus probe in the plan before populating) |
| **Sportec/DFL** | `is_shot & (shot_outcome=="blocked")`, minus `blocked_by_own_team` | `pd.NA` — no blocked-cross field |
| **Metrica** | `is_shot & subtype.str.endswith("BLOCKED")` (exact-token per BD-3; sparse) | `pd.NA` — structural (failed crosses are untyped `BALL LOST`) |
| **Kloppy gateway** | `event.result == ShotResult.BLOCKED` | `pd.NA` — kloppy has no `PassResult.BLOCKED` |
| **Opta** | `pd.NA` — unverified qualifier (don't ship unverified) | `pd.NA` — unverified + team-attribution unknown |
| **SkillCorner** | `pd.NA` — confirmed absent (both tiers) | `pd.NA` — confirmed absent |

Note the StatsBomb wrinkle (relevant when `cross_blocked` is un-deferred post-probe): it needs the
**native** `statsbomb.py` path (the `Block` event + `related_events`); a StatsBomb match routed
through the **kloppy gateway** cannot carry it (`shot_blocked` via `ShotResult.BLOCKED` still works,
but kloppy drops the `Block` event so `cross_blocked = pd.NA`).

## 5. Bravery numerator coverage (resolves the former open decision)

Both columns ship → TF-51 bravery's numerator (blocked shots + blocked crosses) by provider:

- **Complete** (both columns present): **Gradient Sports** (both pining-probed), **Wyscout** (both
  mechanism-only — BD-1 probe at impl).
- **Shots-known, crosses-NA** (mixed): **StatsBomb** (cross deferred, BD-2), **Sportec/DFL, Metrica,
  Kloppy-gateway.** Bravery handles this **per final-action type** — a (team, game) with all-`NA`
  `cross_blocked` reports its cross component as `NaN` (unknown), never a fabricated 0; the shot
  component still computes. TF-51 §9.3's R2-2 guard extends naturally to per-type (shots vs crosses).
- **NaN** (both NA): **SkillCorner, Opta.**

**Handoff to TF-51 (folded in).** TF-51 §9.3 bravery consumes **both** `shot_blocked` and
`cross_blocked` (signature `compute_bravery(actions, *, shot_blocked_column, cross_blocked_column)`),
with the R2-2 unknown-≠-0 guard applied **per final-action type** (a mixed provider yields a
shots-only bravery + an explicit NaN cross component, not a silently-low combined rate). **No
individual `cross_block` credit rule ships in v1** (TF-1: blocked crosses are bravery-only — no clean
sizing; the per-defender rule is deferred to the v2 DPA work).

## 6. Validation

- **Per-provider unit fixtures** — for each feasible converter × column: a blocked in-scope action →
  `True`; a non-blocked in-scope action → `False`; an out-of-scope action → `pd.NA`. Reuse committed
  raw fixtures (StatsBomb `3754058`/`7584`, Metrica `per_period_match`, the DFL/GS/Wyscout slices).
- **GS real-data e2e (owner-gated)** — assert the WC2022 GS `crossOutcomeType=="B"` maps to
  `cross_blocked==True` on real data (the 6/39 finding), mirroring the GS shot e2e.
- **Unknown-provider fixtures** — Opta + SkillCorner shots/crosses → all-`pd.NA` (the honest-unknown
  contract TF-51's R2-2 bravery guard depends on).
- **Own-team deflection** — a DFL `blocked_by_own_team` shot → `shot_blocked==False`.
- **Golden / parity updates** — each converter's golden/parity fixture gains the two columns
  (additive; existing columns byte-unchanged). DFL parse-port parity + the SC/Metrica builder gates
  unaffected (events-converter output columns).
- **Additivity guard** — existing output columns/values byte-identical before/after (→ **no retrain**).

## 7. Deferred

- **Opta `shot_blocked` / `cross_blocked`** — verify the blocked qualifier + team attribution against
  a live F24/MA1 feed, then populate (currently `pd.NA`).
- **SkillCorner** — no event-stream path (confirmed both tiers); only a tracking-based block detection
  (TF-48 goalmouth trajectory) could recover it; out of scope.
- **Blocked *passes* (general)** — GS `incompletionReasonType=="BL"` covers passes too (33 on the
  probe match), but TF-51 needs only shots + crosses; not built.

## 8. Attribution, C4, retrain

- **C4** — **C4-free** (converter output columns, not aggregators; count unchanged).
- **Retrain** — **none.** Purely additive columns; no existing value changes (additivity-guarded).
- **ADR** — ADR-worthy: a new **cross-provider output-column convention with a downstream consumer**
  (TF-51). Draft at commit-prep (ADR-0`XX`); may fold into the TF-51 ADR.
- **NOTICE** — none (data-surfacing columns, not an algorithm).

## 9. Open questions

- Whether the two columns register in `SPADL_COLUMNS` or as documented extensions (plan detail).
- Opta qualifier + team attribution (deferred; `pd.NA` until live-feed verified).
- The GS probe used one WC2022 match (match 10502); a multi-match GS probe at implementation time
  confirms the `crossOutcomeType=="B"` rate is stable (expected — same field family as the verified
  shot `"B"`).
