# `pitch_control_at_target` — retire the dead at-ball column, sample the destination (design)

**Date:** 2026-06-16 · **Status:** draft for review · **Decision:** ADR-032 (new) · **Base:** main @ v4.30.0 → target **4.31.0**

## Problem

`silly_kicks.tracking.features.pitch_control_at_action` samples the pitch-control (PPCF) surface at
`(action.start_x, action.start_y)` — explicitly "proxy for ball position" (features.py:1932-1938). Because
the action *happens at the ball*, the query is always ~0 m from the frame ball, where the **Spearman PPCF is
the 0.5 reaction-time fallback** (the ball reaches near cells before any player's reaction time, so no player
accrues control). The emitted `pitch_control_at_ball__spearman` is therefore ~0.5 for every well-linked
action — **informationally dead**. It is declared a `STRUCTURAL_CONSTANTS` entry in the liveness gate
(`tests/tracking/test_aggregator_column_liveness.py`) and was flagged to the lakehouse in the 4.24.0 changelog.

### Two findings that shape the fix

1. **Scope.** The degeneracy is specific to *sampling at the ball*. The other PPCF consumers (`obso`,
   `cover_shadows`, `gk_influence`, `player_influence`, `space_creation`) sample the surface at their own
   relevant points and are unaffected. Re-aiming the at-ball sample point is **localized to
   `pitch_control_at_action`** — it does NOT touch the surface model or the other consumers. (Only a
   "potential-control model variant" would touch every consumer; that option is rejected as disproportionate.)

2. **Latent ADR-028 bug, currently masked.** `pitch_control_at_action` samples the **action-LTR**
   `start_x/start_y` against a surface built from **absolute-frame** (home-attacks-right) frame positions,
   with **no per-action re-projection**. For away-team actions those conventions are a 180° point reflection
   apart, so the query hits the wrong cell. It is harmless *today* only because near-ball is 0.5 in both
   conventions — i.e. the degeneracy is exactly why ADR-028 could leave pitch_control "untouched." **Any
   meaningful (non-near-ball) sample point resurrects this as a real away-team bug**, so the fix MUST include
   the re-projection.

## Decision (Option A — retire and replace)

Retire the dead `pitch_control_at_ball__<method>` column and replace it with
`pitch_control_at_target__<method>`, sampled at the action **destination** with the ADR-028 re-projection.

1. **Re-aim to the destination `(end_x, end_y)`.** PPCF at where the ball is *going* (pass target, carry/
   dribble end, shot target). Ball-travel-time to the destination is strictly positive, so players can
   contest it → non-degenerate, and the value answers a real question: *does the acting team control the
   destination of this action?*
2. **Re-project the query into the frame convention (ADR-028).** Compute the per-action flip via
   `acting_team_attacks_rtl(actions, frames)` and re-project the query point with `reproject_to_action_ltr`
   (the 180° reflection is involutive, so applying it to the action-LTR `(end_x, end_y)` yields the
   absolute-frame point to sample). The cached per-frame surface stays absolute-frame (cache key unchanged);
   only the **query point** is flipped — so `PitchControlCache` reuse across consumers is preserved.
3. **Retire `pitch_control_at_ball__<method>`** entirely (the established 4.24.0 lean-contract pattern for
   dead columns): drop the column, the `STRUCTURAL_CONSTANTS` entry, and the near-ball-degeneracy invariant
   test. `add_pitch_control` now emits a **live** column, so it needs no structural-constant exemption.
4. **All three methods** (`spearman` / `fernandez_bornn` / `voronoi`) move to `at_target`. The 0.5 degeneracy
   was Spearman-only, but (a) sampling at the ball is a poor anchor for *every* method and (b) the ADR-028
   mis-projection affected all three (masked only for Spearman) — so re-aim + re-project uniformly.

### Honest edge: in-place actions

Actions with `end ≈ start` (tackles, fouls, some events) sample near the ball again → ~0.5 (Spearman). That
is the *honest* value: an action with no spatial destination has no control-of-destination signal. The column
is meaningful for ball-progression actions (pass/cross/carry); near-ball for in-place ones, by construction.
Documented, not special-cased. NaN `end_x/end_y` → NaN (unchanged nan-safe behavior).

### Semantic edge: shots (review)

For a **shot**, `end_x/end_y` is the shot target (typically the goal mouth / inside the box), so `at_target`
measures **contestation of the target cell — GK/defender-dominated** — NOT open-play team control of a pass
destination. The two readings differ inside the box. **Decision: keep shots in, do not exclude them.** The
column is a *uniform raw feature* = "the acting team's PPCF at this action's destination cell"; its
*interpretation* legitimately varies by action type (open-play control for passes/carries; target-cell
contestation for shots), and a VAEP/calibration model conditions on that via the existing `type_id`
interaction. Excluding shots would (a) introduce a `NaN`-for-shots special case absent from the original
all-action `at_ball` column and (b) discard genuinely informative shot-target-contestation signal (a
wide-open target vs a crowded one). The docstring states the per-type interpretation explicitly so the
feature is not mis-read as "team control" for shots.

## Surface touched (consumers of the at-ball column)

| File | Change |
|------|--------|
| `silly_kicks/tracking/features.py` | `pitch_control_at_action` (re-aim to `end_*` + ADR-028 re-projection), `add_pitch_control`, `pitch_control_xfns`, `pitch_control_default_xfns`, `__name__`/`col_name`/docstrings → `at_target` |
| `silly_kicks/atomic/tracking/features.py` | atomic `pitch_control_at_action` must synthesize `end_x=x+dx, end_y=y+dy` (mirror `_structural_pass_atomic_endpoints`), not just rename `x→start_x`; `add_pitch_control`, `atomic_pitch_control_xfns`, `atomic_pitch_control_default_xfns`, docstrings → `at_target` |
| `silly_kicks/calibration/_features.py:69-71` | rename the 3 `pitch_control_at_ball__*` feature names → `pitch_control_at_target__*` |
| `silly_kicks/tracking/__init__.py`, `silly_kicks/atomic/tracking/__init__.py` | re-export names unchanged (function names `pitch_control_xfns` etc. stay; only the emitted column string changes) |
| `tests/tracking/test_aggregator_column_liveness.py` | DELETE the `add_pitch_control` `STRUCTURAL_CONSTANTS` entry + `test_pitch_control_at_ball_near_ball_degeneracy`; `add_pitch_control` now produces a live column under the standard liveness check |
| `tests/tracking/pitch_control/test_action_coupled.py`, `tests/tracking/pitch_control/test_atomic_pitch_control.py` | column-name assertions → `pitch_control_at_target__*` |
| CLAUDE.md | update the PR-S31/TF-7 + the liveness-gate convention mention (the `STRUCTURAL_CONSTANTS` example) to reflect the retired→live change |

## Testing strategy

- **Ground-truth-anchored correctness test (THE correctness core — review B).** Mirror-symmetry alone is
  necessary but NOT sufficient: a reprojection that is wrong *symmetrically* (sampling the reflected-but-wrong
  cell, or a double-flip that cancels) makes both mirrored frames agree on the same wrong value → a
  symmetry-only test passes green. So pin an **absolute, hand-computable** value: construct a frame where the
  action's destination cell is controlled by the **acting team** → assert `at_target ≈ 1.0`.
  - **The fixture MUST be ASYMMETRIC (review R1 — the one place the gate can still be defeated).** If the
    destination cell and its 180° reflection have the *same* PPCF, a wrong-direction flip samples the
    reflection — which also reads ≈1.0 — and the test passes green on a broken projection (the exact
    symmetric-wrong mode this test exists to catch, sneaking back via a symmetric fixture). So the
    fixture-construction constraint is: **`PPCF(destination)` and `PPCF(its absolute-frame 180° reflection)`
    must differ by ≥Δ** — e.g. destination acting-team-controlled (≈1.0) while its reflection is
    opponent/empty-controlled (≈0.0). Then a wrong-direction flip lands on the ≈0.0 cell and the test fails
    RED as intended.
  - **Geometry must be EXTREME enough to hit the PPCF asymptote (review R3).** `≈1.0`/`≈0.0` only holds if the
    surrounding players push Spearman PPCF to its asymptote within tolerance; a marginal geometry could land
    at 0.62/0.41 and flake against the sigmoid-steepness / reaction-time params. Construct it with extreme
    separation — acting players ~adjacent to the destination cell, opponents ~a pitch-length away.
  - Run for BOTH a home-team and an away-team action (the away case is where the reprojection must fire) →
    pins the value AND the flip direction (action-LTR → absolute vs absolute → action-LTR). RED before the
    re-projection fix (away case lands on the reflected ≈0.0 cell), GREEN after.
- **Away-team mirror-invariance** (symmetry companion to the above, not a substitute): a physical situation
  mirrored between a home-attacking and away-attacking frame yields the **same** `at_target`. Mirrors the
  ADR-028 `test_action_ltr_mirror_invariance.py` pattern.
- **Liveness gate — hard fixture precondition (review A; do NOT leave as "if needed").** The gate's teeth
  rest on the fixture having off-ball destinations. So the test FIRST asserts a precondition —
  `≥K` actions with `dist((end_x,end_y), frame_ball) > R` (off-ball destinations) — and ONLY THEN asserts the
  emitted `pitch_control_at_target__spearman` is non-NaN + **non-constant**. This way a fixture refactor that
  accidentally makes every action in-place fails loudly on the precondition rather than silently neutering the
  gate (a near-ball-only fixture would leave `at_target ≈ 0.5` everywhere and pass weakly). Pick `K`/`R` from
  the existing 5-window fixture (pass/shot/GK/attacking-third/wide-area) — confirm + pin, don't assume.
- **Retire the degeneracy test** (`test_pitch_control_at_ball_near_ball_degeneracy`) + the `add_pitch_control`
  `STRUCTURAL_CONSTANTS` entry — the column is now live, so the standard non-constant liveness check applies.
- **Atomic mirror parity:** atomic `at_target` equals the standard column on geometry-matched actions
  (`end = x+dx, y+dy`).
- Full `ruff` + bare `pyright` (incl `tests/`) + full suite; reproduce on a pandas-3 env if any parity/dtype
  surface is touched (none expected here).

## Blast radius / retrain

- **VAEP/tracking + calibration retrain trigger.** The column changes from a dead ~0.5 constant to a live
  destination-control signal (and away-team values are corrected). Any consumer that fed
  `pitch_control_at_ball__*` (the silly-kicks calibration `_features.py` augmented-VAEP Brier objective + the
  lakehouse feature set) re-materializes against the new `pitch_control_at_target__*`.
- **C4-free:** no new aggregator/container; the action-coupled aggregator count stays 28.
- **Lakehouse handoff = a full ADR-013-style column-lifecycle MIGRATION, not a find-replace (review).** The
  `pitch_control_at_ball__*` → `pitch_control_at_target__*` rename lands in ~17 files across **two** lakehouse
  subsystems — **AC** (`schema.py`, `enrich.py`, `action_context.py`, `oracle_map.py`, `tracking_context.py`)
  and **DEFCON** (`defcon_lite.py`, `defcon_lite_tracking.py`, `defcon_lite_360.py`, `defcon_lite_common.py` +
  `test_defcon_schema_parity.py` / `test_defcon_projection_parity.py`) — the latter NOT obvious from the
  silly-kicks side, so flag it explicitly. Adoption is: a bronze column drop+add (`scripts/migrations/*.sql`
  + operator-applied runner), dbt staging/mart rename, Lakebase synced-table reshape, HF dataset republish,
  the DEFCON schema-parity tests, and a **forced AC recompute** (the column goes dead-0.5 → live + away-team
  correction → real values change). **Batch this recompute with the parked Metrica y-fix recompute — don't
  recompute AC three times.** And **validate, don't blind-adopt:** a dead constant → live feature can hurt a
  model as easily as help it, so the lakehouse retrain should A/B the new column's contribution (augmented-
  VAEP Brier improves, or at least does not regress) before shipping — not re-materialize on faith.
- **4.31.0 is BREAKING — NOT a currency pin-bump; atomic with the migration (review R2).** Unlike 4.29.0
  (gateway fix) and 4.30.0 (additive parse port), which were safe to adopt as currency and defer, the moment
  the lakehouse runs AC enrichment against 4.31.0, `add_pitch_control` emits `pitch_control_at_target__*`
  while the schema/oracle/DEFCON still expect `pitch_control_at_ball__*` → **schema-mismatch / KeyError in the
  AC pipeline**. So the lakehouse **cannot bump the pin to 4.31.0 until the full AC+DEFCON column migration +
  recompute lands in the same change** — 4.31.0 is gated behind the batched recompute, not a quick pin bump.
  The handoff must say "do not adopt 4.31.0 as a currency bump — it is atomic with the column migration," so
  nobody casually moves the pin and breaks AC. (This is also why batching with the parked Metrica recompute is
  right — they share the one AC recompute window.)

## ADR-032

Records: the retire-and-replace decision; Option A (re-aim) over the rejected potential-control model variant
(disproportionate, every-consumer blast radius); the destination anchor + ball-travel-time rationale; the
ADR-028 re-projection that the degeneracy had been masking (with the ground-truth-anchored test that pins the
involution direction, not just symmetry); the in-place-action + shots (target-cell-contestation, kept,
per-type interpretation) semantics; and the retrain trigger framed as a lakehouse column-lifecycle migration
(AC + DEFCON), batched with the Metrica y-fix recompute and A/B-validated.

## Alternatives rejected

- **Potential-control / near-ball PPCF model variant** — changes the surface for every consumer
  (obso/das/cover_shadows/…), needs broad re-validation; disproportionate to reviving one column.
- **Integration window past the reaction time** — more machinery, fuzzier meaning than "the destination."
- **Redefine `at_ball` in place** — the name would be a lie if it sampled the destination; retiring dead
  columns is the established pattern.

## Self-review

- **Placeholders:** none — every touched file + test is named.
- **Internal consistency:** the ADR-028 re-projection is required (not optional) because re-aiming
  resurrects the masked away-team bug; the liveness gate flips from structural-constant-exempt to live.
- **Scope:** single subsystem (the at-ball action-coupled column + its atomic mirror + the calibration
  feature-name list). Other PPCF consumers are explicitly out of scope.
- **Ambiguity:** "destination" = SPADL `end_x/end_y` (standard) / `x+dx, y+dy` (atomic); in-place actions
  near-ball-by-construction is documented, not a bug.
