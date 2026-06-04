# Design: Gradient Sports goal-capture correctness — own goals (RE+G), cross-goals (CR+G), voided-event exclusion (nonEvent), and VAEP own-goal labeling

**Date:** 2026-06-04
**Status:** Draft — revised after lakehouse **round-2** review (all 8 round-1 + 6 round-2 items
dispositioned; Component 4 `nonEvent` exclusion in scope). Reviewer verdict: implementation-ready. See
round-1 + round-2 dispositions near end.
**Author:** Claude Opus 4.8 (1M) + Karsten S. Nielsen
**Reviewers:** cross-session review (mediated by maintainer)

---

## Context — why this change

The Gradient Sports (PFF FC) SPADL converter (`silly_kicks/spadl/gradientsports.py`) reasons about
goals **only** from `SH` (shot) events. A lakehouse bug report + an independent full-catalog
empirical verification surfaced that this is wrong in two directions, and exposed a third,
codebase-wide issue in the VAEP labeller:

1. **Own goals are never captured.** In the PFF FC feed, an own goal is a `possessionEventType == "RE"`
   (rebound) event with `shotOutcomeType == "G"`. These never reach the shot-result branch, so they
   currently dispatch to the `RE` default (`keeper_save`) with the default `fail` result — captured as
   neither goal nor own goal.
2. **Cross-goals are undercounted.** A goal scored directly from a cross is a
   `possessionEventType == "CR"` event with `shotOutcomeType == "G"` (crosser = scorer). The converter
   maps these to `cross` / `freekick_crossed` with `success`/`fail` — never as a goal.
3. **Own goals are uncounted in VAEP for *every* provider.** `vaep/labels.py` detects goals **and**
   own goals via `type_name.str.contains("shot") & result==…`. But every converter (StatsBomb
   `statsbomb.py:508`, opta `_fix_owngoals`, sportec `sportec.py:861-863`) maps own goals to
   `bad_touch` + `owngoal` — which fails the `"shot"` gate. So no own goal, from any provider, has ever
   registered in `scores`/`concedes`/xG labels, even though the label logic explicitly *intends* to
   handle own goals (the team-attribution branches reference `owngoal`).
4. **Voided ("annulled") events are emitted as real actions.** PFF marks an event that occurred but
   did not count with `possessionEvents.nonEvent == True` (play called back for a foul/advantage,
   offside, or a disallowed goal). The converter currently ignores this flag and emits all such events
   as real SPADL actions — **1081** across the catalog (PA 530, CH 197, CL 130, SH 89, RE 75, CR 30,
   TC 25, BC 5), of which **21** are disallowed `shotOutcomeType=="G"` goals. This over-states passes/
   challenges/shots and inflates goal counts. (Investigated: 53% are immediately followed by a stoppage
   and 48% sit adjacent to a foul; the events are real-looking but annulled — directly analogous to the
   `initialNonEvent` empty-PE markers the converter **already** excludes.)

The predecessor fix (silly-kicks **4.12.2**, shipped) removed a *false* own-goal mapping
(`SH` + `shotOutcomeType == "O"` → `owngoal`; `"O"` is off-target). This spec covers the *positive*
capture of the real own goals and cross-goals, plus the VAEP-label correction.

### Empirical evidence (authoritative)

Full GS catalog, all 64 WC2022 matches, 144,541 events (queried directly via the pining provider API):

| Signal | Count | Notes |
|---|---|---|
| `SH` + `shotOutcomeType=="O"` | **563** | the 4.12.2 phantom-owngoal class (off-target) — confirmed exact |
| `shotOutcomeType=="G"` by event type | `{SH: 216, RE: 3, CR: 3}` = 222 | goals live under SH (normal), RE (own), CR (cross) |
| of which **real** (`nonEvent != True`) | **201** (`SH:195, RE:3, CR:3`) | the true goal population |
| of which **voided** (`nonEvent == True`) | **21** (all `SH`) | disallowed/annulled goals — currently over-counted |
| `nonEvent == True` (all events) | **1081** | annulled plays; currently emitted as real actions |
| `RE` + `G` (real) | **3**, all own goals | Enzo Fernández (g10503, team 364), Nayef Aguerd (g3853, team 374), Manuel Neuer (g3855, team 368) |
| `CR` + `G` (real) | **3**, all real cross-goals | Ziyech (g3837, FK), Sabiri (g3837, FK), Bruno Fernandes (g3843, open play) |

Key structural facts for `RE`+`G` own goals:
- `gameEvents.playerId` is **populated** and equals `possessionEvents.rebounderPlayerId` = the OG
  scorer (Enzo/Aguerd/Neuer). `shooterPlayerId` is null (RE is not a shot).
- `gameEvents.teamId` = the **conceding** team (the OG scorer's side).
- There are **zero** `RE`+`G` events that are *not* own goals → the `RE`+`G` rule is empirically
  airtight (n=3, 100%, no counterexamples). A "NULL-shooter" guard is unnecessary and would misread
  these (the scorer is known via `rebounderPlayerId`).

**Codebook status:** the only available PFF documentation (the "PFF FC Change Log" in the WC2022 data
release) is an incremental change log, not a semantic data dictionary — it does not define `RE` or
own-goal encoding. The full-catalog empirical evidence is therefore the authority; it is unanimous.

---

## Goals / non-goals

**Goals**
- Exclude voided (`nonEvent == True`) events from the GS converter output (incl. the 21 disallowed
  goals), with a `ConversionReport` tally — consistent with the existing `initialNonEvent` exclusion.
- Capture `RE`+`G` own goals as `bad_touch` + `owngoal`, attributed to the conceding team / OG scorer.
- Capture `CR`+`G` cross-goals as goals while preserving the cross event.
- Make own goals register in VAEP `scores`/`concedes`/xG labels — for **all** providers.

**Non-goals**
- Re-typing own goals or crosses to `shot` (rejected: pollutes shot/xG features with non-shots).
- Changing detection of normal (`SH`) goals.
- Retraining VAEP models (flagged as a downstream consequence, not done here).

---

## Design

### Component 1 — GS converter: `RE`+`G` → own goal

In `_dispatch_actiontype_resultid` (`gradientsports.py`), add a refinement applied **after** the
`np.select` type dispatch and the existing `keeper_pick_up` refinement, so it takes priority over the
`RE` → `keeper_save`/`keeper_pick_up` handling:

```python
is_owngoal = (pe == "RE") & (shot_outcome == "G")
type_id_arr = np.where(is_owngoal, at_ids["bad_touch"], type_id_arr)
# result handled in the result dispatch: add is_owngoal -> rs_ids["owngoal"]
```

- **Detection rule:** `RE` + `shotOutcomeType=="G"`, **gated by a coordinate sanity check** (below).
  No NULL-shooter guard (the scorer is *known* via `rebounderPlayerId` = `gameEvents.playerId`).
- **Attribution (unchanged, ADR-001 compliant):** `team_id` = `gameEvents.teamId` (conceding team);
  `player_id` = `gameEvents.playerId` (= `rebounderPlayerId` = OG scorer). No `team_id` flip — the
  `owngoal` result carries the credit-the-opponent semantics downstream.
- **Coordinates:** `end == start` (the GS converter never derives shot-class end-coords; consistent).
- **Precedent:** StatsBomb/opta/sportec all emit own goals as `bad_touch` + `owngoal`.

**Semantic rationale (why `RE`+`G` ≡ own goal — beyond n=3).** `RE` is an *uncontrolled rebound*
possession event. An attacker who buries a rebound plays a NEW shot → that is an `SH` event, not `RE`.
So a rebound that itself ends in a goal is, almost by construction, a deflection into the rebounder's
own net. Empirical support (full catalog): of 2544 `RE` events only 10 carry *any* shot outcome
(`G:3, S:5, O:1, F:1`); the 3 `G` are all own goals whose rebounders are defenders/keeper (Enzo
Fernández, Nayef Aguerd, Manuel Neuer), with `reboundOutcomeType` D/A and the ball in the goal area.
**Honest limitation:** `originateType` is `None` on all 1559 `SH` events, so I could not mechanically
confirm "attacking rebound-finishes are typed `SH`" via that field — the rule rests on n=3 +
rebounder-identity + near-goal geometry. This is *why* the coordinate tripwire below is load-bearing,
not optional.

**Defensive tripwire (do not silently trust n=3).** Gate the own-goal classification on a coordinate
sanity check: the rebound must occur in the *conceding* (acting) team's own half. The `bad_touch`+
`owngoal` type/result is assigned provisionally in the dispatch stage; the geometry check runs as a
**post-`to_spadl_ltr` validation pass** (round-2 C) so it reads the **LTR-canonical frame** — where the
acting team always attacks toward `x = field_length`, so its own goal is at `x = 0` and a true own goal's
ball sits in the **own half (`start_x < field_length / 2`)**. Threshold = own half, deliberately the
**looser** bound (round-2 C): the tripwire's job is catching gross-wrong geometry (a rebound *goal* at
the attacking end), not precise classification, so it must not false-`WARN` on a legitimate deflected OG
from outside the box. If an `RE`+`G` row **passes**, keep `bad_touch`+`owngoal` (no noise — own goals are
legitimate and recur). If it **fails**, emit a loud `warnings.warn(..., stacklevel=2)` and **revert** the
row to the default `RE` handling (`keeper_save` / `fail`) — a genuine rebound-goal-as-`RE`+`G` or a
future-feed anomaly surfaces as a logged event, never a silent mislabel. All 3 known OGs pass the own-half
bound (and the tighter defensive-third bound too), so the looser threshold costs nothing on known data
and is safer for unseen feeds. **Boundary test required** for the chosen threshold (round-2 C).

### Component 2 — GS converter: `CR`+`G` → cross + synthesized shot

Keep the existing `cross` / `freekick_crossed` / `corner_crossed` action (its result stays per the
existing `cross_outcome` dispatch). **Synthesize an additional `shot` action** immediately after it,
reusing the established foul-synthesis pattern (`gradientsports.py:536-549`, the `0.5`-offset
sort-key insertion + dense `action_id` renumber):

- **Detection:** `(pe == "CR") & (shot_outcome == "G")`.
- **Synthetic shot type:** mirrors the `SH` set-piece dispatch — `setpiece=="F"` → `shot_freekick`,
  `"P"` → `shot_penalty`, else `shot`.
- **Result:** `success`. **player_id:** crosser (= `gameEvents.playerId`). **team_id:** crossing team.
  **coords:** same as the cross event. **bodypart:** the cross's bodypart (else default `foot`).
- **Provenance:** match the existing foul-synthesis precedent — synthesized rows are `.copy()` of the
  parent, so they **inherit the cross's `original_event_id`** (they are NOT given a fresh provider id).
  There is no dedicated `synthetic`/`derived` flag column in `GRADIENTSPORTS_SPADL_COLUMNS` today, and
  the foul-synthesis path doesn't add one — so this PR matches that precedent rather than introducing a
  schema column. **Hyrum flag:** the synthetic shot inflates shot counts by 3 (these cross-goals now
  also appear as shots) and joins back to the same source `original_event_id` as the cross. Anyone
  counting shots or de-duplicating on `original_event_id` should be aware. **Concrete trap (round-2 E):**
  a consumer de-duplicating actions on `original_event_id` will collapse the cross and its synthetic shot
  into one row and **silently drop the goal**. **Adopted (maintainer pulled into scope):** an
  `is_synthetic` bool column on `GRADIENTSPORTS_SPADL_COLUMNS` is `True` on the synthetic shot AND the
  synthesized foul rows, `False` on real rows — so consumers can keep synthesized rows on a dedup.
- **Geometry note:** `coords = cross event` gives the synthetic shot a wide/byline origin. For the two
  direct free-kicks (FK spot = shot spot) this is fine; for the open-play cross (Bruno Fernandes) it's a
  wide-angle "shot." Negligible at n=3, but flagged for any xG-feature consumer if this rule ever scales
  to a busier feed.
- Rationale: SPADL records a normal goal only as `shot`+`success`; `cross`+`success` is
  indistinguishable from an ordinary completed cross. Synthesizing the shot preserves the cross event
  *and* records the goal (scoreline + VAEP correctness), without reinterpreting PFF's `CR` type.

### Component 3 — VAEP labels: count own goals regardless of action type

The bug being fixed *is* a duplicated, copy-pasted predicate (`type_name.contains("shot") & result==…`)
that is wrong in all ~6 sites. **Do not re-paste the corrected predicate** into 6 functions — that
repeats the exact anti-pattern. Instead, extract single-source module helpers and route every site
through them:

```python
_SHOT_TYPE_IDS = {spadl.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")}

def _is_goal(actions):                                          # explicit id-set, not str.contains (round-2 F)
    return actions["type_id"].isin(_SHOT_TYPE_IDS) & (actions["result_id"] == spadl.result_id["success"])

def _is_owngoal(actions):
    return actions["result_id"] == spadl.result_id["owngoal"]   # result is unambiguous — no "shot" gate
```

(Verified the three names are exactly the `"shot"`-containing types in `spadl.config`, so the id-set is
behavior-identical to the former `str.contains("shot")` but robust to a future type name like
`"shoot_out"` — round-2 F.)

- Replace the inline `goal = …` / `owngoal = …` definitions in all ~6 `vaep/labels.py` functions
  (`scores`, `concedes`, the xG variants, helpers around lines 112/189/206/226/246/291) with calls to
  `_is_goal` / `_is_owngoal`. One definition, one place to change next time.
- The `owngoal` result is unambiguous (only own goals carry it), so dropping the `"shot"` gate fixes
  the codebase-wide undercount; `_is_goal` (shot+success) is unchanged — cross-goals register via the
  Component-2 synthetic shot.

**Atomic-SPADL path: already correct — no change (resolves former Open item 3).** Verified:
`atomic/spadl/base.py:167` converts *any* `result_id == owngoal` action into a dedicated atomic
`owngoal` action type (independent of the source type), and `atomic/vaep/labels.py:46` /
`features.py:357` detect own goals by that `owngoal` type, not by a `"shot"` gate. So the atomic
labeller never had this bug, and once the GS converter emits `bad_touch`+`owngoal` (Component 1), the
atomic path counts it correctly with no edit. The fix is therefore **isolated to `vaep/labels.py`**.

This is a **cross-cutting behavior change** (regular-SPADL VAEP labels, all providers) → see ADR (below)
+ the concrete golden re-baseline (Testing).

### Component 4 — GS converter: exclude voided events (`nonEvent == True`) — *executes first*

Although listed last, this runs **first** in `convert_to_actions`, in the existing exclusion-filtering
stage (`gradientsports.py:341-385`), so Components 1-2 dispatch only over real events and the disallowed
goals never reach the goal logic.

- **Rule:** drop rows where `possessionEvents.nonEvent == True`, alongside the existing `initialNonEvent`
  / excluded-`game_event_type` / excluded-pair filtering. Tally them in `ConversionReport.excluded_counts`
  (e.g. a `nonEvent` key) so the drop is auditable, matching the existing exclusion bookkeeping.
- **Input contract (backward-compatible BUT observable — round-2 A):** `nonEvent` is **optional**. If the
  input DataFrame carries a `nonEvent` column, exclude its `True` rows; if the column is absent, no-op so
  existing callers keep working — **but the no-op must not be silent.** A silent no-op reproduces the
  exact silent-undercount failure mode this spec exists to eliminate (a caller keeps emitting 1081 voided
  events incl. 21 phantom goals, with no signal). So when the column is absent: emit a **one-time**
  `warnings.warn("nonEvent column not supplied — voided events not excluded", stacklevel=2)` **and** record
  it distinguishably in `ConversionReport` — `excluded_counts["nonEvent"] = <int>` when applied vs an
  explicit unavailable sentinel (e.g. omit the key / set `None`) when the column is absent, so "0 voided"
  (column present, none found) is never confused with "not checked." The GS flatten (`_gs_flatten_events`
  in the pining loader) and the committed test fixture must add `nonEvent` (mapped from
  `possessionEvents.nonEvent`) to exercise + benefit from the exclusion. **Lakehouse note:** the
  bronze→input mapping must surface `possessionEvents.nonEvent`, or the absent-column warning fires and
  the fix is silently missed.
- **Effect:** removes 1081 voided events (~0.75% of GS events), incl. the 21 disallowed goals. The 3 real
  own goals and 3 real cross-goals are `nonEvent == False` → unaffected. End-coordinate chaining
  (`_derive_end_coordinates`) reforms over the remaining real events.
- **Why exclude (not just suppress goal-ness):** the voided *build-up* touches (post-whistle passes/
  challenges) are also not real actions; dropping the whole voided event is the faithful, consistent
  representation (the converter already drops `initialNonEvent`).

### Canonical pipeline ordering (invariant — round-2 B)

`convert_to_actions` now has several row-mutating stages; the interactions are where regressions hide, so
the order is a stated invariant (and an e2e test asserts the composition):

1. **Exclusion stage** (`gradientsports.py:341-385`): drop excluded `game_event_type` / excluded pairs /
   `initialNonEvent` **and `nonEvent==True` (Component 4)** → every later stage sees only real events.
2. **Dispatch** (`_dispatch_actiontype_resultid`): `np.select` type+result; then refinements —
   `keeper_pick_up`, and the **provisional `RE`+`G` → `bad_touch`+`owngoal` (Component 1)**.
3. Build actions frame → NaN `time_seconds` impute → tackle winner/loser passthrough.
4. **`_derive_end_coordinates`** — MUST precede synthesis (existing comment `:485-489`: synthesized rows
   interleave via the `0.5`-offset and would intercept the `shift(-1)` chain).
5. **Synthesis stage** (`0.5`-offset insert, then dense `action_id` renumber): foul-synthesis **and
   Component-2 cross-goal synthetic shot**.
6. **`to_spadl_ltr`** per-period direction normalization.
7. **Component-1 owngoal geometry tripwire** — runs **here, post-LTR** (validate own-half geometry;
   WARN + revert on failure).
8. Clip coordinates → `_finalize_output`.

A regression in any single stage's interaction with the next is caught by the composition e2e test
(below), not just the per-stage unit tests.

---

## Edge cases & risks

- **`RE`+`G` non-OG meaning:** empirically none (n=3, all OG). If a future (non-WC2022) feed encodes a
  rebound *goal* (not own goal) as `RE`+`G`, this would mislabel it. Documented as a known assumption;
  the empirical corpus is unanimous. (Codebook confirmation unavailable.)
- **Cross result semantics:** the cross keeps its existing result (Ziyech/Bruno `crossOutcome=="I"` →
  `fail`; Sabiri `"C"` → `success`). A `fail` cross immediately followed by a synthetic `shot`+`success`
  is acceptable — the cross *as a pass* was incomplete; the goal is the shot. (Open item: force the
  cross to `success`? Proposed: no — keep per `cross_outcome`.)
- **Synthetic-shot interactions:** insert post-dispatch (like foul synthesis) so the synthetic row is
  not re-dispatched; ensure it interleaves correctly with the foul-synthesis ordering and the
  `_derive_end_coordinates` / `to_spadl_ltr` passes (both run before/after synthesis as appropriate).
- **VAEP label blast radius (Hyrum) — concrete in-PR re-baseline, not "audit":** own goals now count
  in regular-SPADL VAEP labels for **all** providers → `scores`/`concedes`/xG distributions shift.
  Implementation step (in THIS PR): grep the committed fixtures/goldens for actions with
  `result_id==owngoal` whose enclosing test asserts `scores`/`concedes`/xG counts; **enumerate the exact
  goldens that move, regenerate them in this PR, and document the expected delta** (+N scores / +M
  concedes per fixture) in the PR body. This prevents both a CI break on merge and a rubber-stamped
  re-baseline that could hide a real regression inside the expected shift. Also flag that VAEP models
  trained on these labels would shift if retrained (not done here). (xS/xCross use *occurrence* labels,
  not VAEP scores — unaffected; atomic labels already counted own goals — unaffected.)
- **`nonEvent` exclusion blast radius + input contract (Hyrum):** dropping 1081 events lowers GS action
  counts ~0.75% (and removes 21 goals) → GS golden/e2e fixtures with voided events shift; fold into the
  same concrete in-PR re-baseline. The `nonEvent` input column is **optional** (absent → no-op) so
  existing callers don't break, BUT a caller only gets the fix once they supply it — the lakehouse
  bronze→input mapping must add `possessionEvents.nonEvent`. This is a soft contract addition; flag in
  the PR body + lakehouse copy/paste.
- **Atomic-SPADL:** the converter change is inherited via the shared converter; verify atomic
  decomposition of a `bad_touch` own goal and of the synthesized shot.

---

## Testing (TDD + realistic data)

- **GS converter unit:** `RE`+`G` → `bad_touch`+`owngoal` (correct team/player); `CR`+`G` → existing
  cross/freekick_crossed **plus** a synthetic `shot`/`shot_freekick`+`success` by the crosser; guards:
  `SH`+`O` → `fail` (no 4.12.2 regression), `RE` without `G` → `keeper_save`.
- **`nonEvent` exclusion unit:** a `nonEvent==True` event (incl. an `SH`+`G` disallowed goal, and a
  voided `PA`) is dropped from the output and tallied in `ConversionReport.excluded_counts`; a
  `nonEvent==False`/absent event is kept (backward-compat: input without the column → no-op).
  Coordinate-tripwire unit: an `RE`+`G` whose ball is NOT in the conceding team's defensive area → `WARN`
  + not classified `owngoal`.
- **GS realistic regression (production-shaped, not redistributed):** extend the committed
  `tests/datasets/gradientsports/synthetic_match.json` with `RE`+`G` and `CR`+`G` events that match the
  *real* data shape — null `shooterPlayerId`, populated `rebounderPlayerId`, the `RE` dispatch path,
  set-piece `CR` (`F`) — modeled on g3853 (Aguerd) / g3837 (Sabiri/Ziyech) but with synthetic
  ids/names. Also add a `nonEvent==True` disallowed `SH`+`G` (the over-count case). (We do **not** commit
  a redacted real PFF slice: the GS feed is owner-tier, so redistribution — even redacted — risks the
  license; a faithfully-shaped synthetic fixture gives the same e2e value. It stays committed → runs in
  the regular suite.) Assert: exactly one `owngoal` on the conceding team, one synthetic shot-goal by the
  crosser, and the disallowed goal **absent** from the output. RED-prove against current code.
- **VAEP labels:** a `bad_touch`+`owngoal` registers in `concedes` (acting team) / `scores` (opponent)
  / xG; RED-prove (currently uncounted). Also test the **`cross`+`fail` → `shot`+`success` adjacency**
  for the cross-goals (esp. the FK cases `freekick_crossed`+`fail` then `shot_freekick`+`success`): the
  preceding failed cross must not cancel/pervert the goal's VAEP credit.
- **Composition e2e (round-2 B):** one synthetic match containing **all of** {a `nonEvent==True` voided
  event, an `RE`+`G` own goal, a `CR`+`G` cross-goal, a foul} together; assert the **final** action list
  and the **dense `action_id` sequence** (0..N-1, contiguous) — proving the exclusion → dispatch →
  derive-end → synthesis → LTR → tripwire stages compose without interaction regressions. (Committed
  fixture → regular suite.)
- **Catalog scoreline guard (round-2 D) — owner-gated e2e + documented verification:** assert the
  **real** (`nonEvent==False`) `G` population reproduces the official final scorelines for the matches
  that previously over-counted (at minimum g3853 4→3). This needs the full GS catalog (owner-tier), which
  public CI cannot access — so it ships as an **e2e test gated on the pining owner token / data
  availability** (mark `e2e`, per the repo convention that data-dependent tests aren't in the default
  suite), **plus** the verification result recorded in the PR body (already established: 222 → 201 real /
  21 voided; g3853 confirmed 4→3 by both sessions).
- **Atomic mirror** tests (own goal → atomic `owngoal` type; already result-based — guards no
  regression).
- Full suite + the concrete golden re-baseline (above); `ruff` + `pyright` clean.

---

## ADR

The VAEP-label change (Component 3) is a cross-cutting behavior change with downstream consumers
(all providers' VAEP labels; trained-model targets) → an ADR is warranted: *"Own goals are counted in
VAEP scores/concedes/xG labels by result, independent of action type."* Records the rationale, the
Hyrum impact, and the golden-test/retraining consequences. ADR number: next free at authoring time.

---

## Version / release

Cross-cutting behavior change (VAEP labels affect all providers) → **minor** bump. Reconcile the exact
number against `origin/main` at release time per the version-bump checklist (next free minor after
what is tagged; `4.12.2` is currently tagged). Coordinate with any other in-flight session at release
(no numbers are pre-reserved — compare against `main`).

---

## Cross-session review — round 1 dispositions (lakehouse, 2026-06-04)

| # | Concern | Disposition |
|---|---------|-------------|
| 1 | RE+G ≡ OG: semantic backing + tripwire | **Accepted + refined.** Recorded semantic rationale; **honest correction** — `originateType` is null on all 1559 `SH`, so "rebound-finishes are `SH`" is not mechanically verifiable; rule rests on n=3 + rebounder-identity + geometry → the **coordinate sanity-check tripwire is now a gating check + anomaly `WARN`** (Component 1). |
| 2 | Don't paste predicate into ~6 sites — extract helper | **Accepted fully.** `_is_goal` / `_is_owngoal` module helpers; all `vaep/labels.py` sites call them (Component 3). |
| 3 | Resolve atomic-labels path | **Pushback (verified):** atomic is **already** result-based (`base.py:167` → dedicated `owngoal` type; `labels.py:46`) — no bug, no change. Fix isolated to `vaep/labels.py`. |
| 4 | Concrete in-PR golden re-baseline | **Accepted.** Enumerate exact moving goldens, regenerate in-PR, document +N/+M delta in PR body (Risks + Testing). |
| 5 | Real-data-derived fixture | **Accepted w/ pushback:** build a **production-shaped synthetic** fixture (null `shooterPlayerId`, populated `rebounderPlayerId`, `RE` path, set-piece `CR`) — do **not** redistribute a redacted real PFF slice (owner-tier license). Same e2e value (Testing). |
| 6 | Synthetic-shot geometry + provenance | **Accepted.** Inherit `original_event_id` (matches foul-synth, which has no synthetic flag); Hyrum shot-count flag; wide-origin xG note. Dedicated provenance flag = possible follow-up, not adopted (Component 2). |
| 7 | cross+fail→shot+success adjacency test | **Accepted.** Added (Testing). Keep cross per `cross_outcome` (Open item 1). |
| 8 | `SH`+`G` over-count (disallowed/VAR goals) | **Pulled into scope (maintainer).** Investigated: the root cause is the general `nonEvent==True` voided-event flag (1081 events, 21 of them goals), not just disallowed goals → **Component 4** excludes all `nonEvent==True`. |

## Cross-session review — round 2 dispositions (lakehouse, 2026-06-04)

All Component-4 numbers independently reproduced by the lakehouse (1081 by type; 201 real / 21 voided;
g3853 4→3; atomic pushback confirmed). Residual items:

| # | Concern | Disposition |
|---|---------|-------------|
| A | Optional `nonEvent` no-op is silent | **Accepted.** Absent column → one-time `warnings.warn` + distinguishable `ConversionReport` signal (unavailable ≠ "0 voided"). Component 4 updated. |
| B | State + test the stage-ordering invariant | **Accepted.** Added the canonical pipeline-ordering invariant + a composition e2e ({voided, RE+G, CR+G, foul} together, asserting final list + dense `action_id`). Corrected: `_derive_end_coordinates` precedes synthesis; tripwire is post-LTR. |
| C | Tripwire: looser bound, post-LTR frame, boundary test | **Accepted.** Threshold = own half (looser); anchored to the LTR-canonical frame as a post-`to_spadl_ltr` validation/revert pass; boundary test required. |
| D | Catalog-wide scoreline assertion | **Accepted w/ mechanism clarified.** Owner-tier data → ships as an owner-gated `e2e` test + documented PR verification (public CI can't reach the catalog). |
| E | `original_event_id` dedup trap | **Accepted + shipped (maintainer pulled into scope).** Added an `is_synthetic` bool column to `GRADIENTSPORTS_SPADL_COLUMNS`, `True` on the cross-goal shot + synthesized foul rows; consumers can keep synthesized rows instead of collapsing them on `original_event_id`. |
| F | `_is_goal` substring vs id-set | **Accepted (verified).** Helper uses explicit `{shot, shot_penalty, shot_freekick}` id-set (the exact `"shot"`-containing types). |

## Open items

1. Cross result for `CR`+`G`: keep per `cross_outcome` (proposed — the cross *as a pass to a teammate*
   genuinely failed for Ziyech/Bruno) vs force `success`. Proposed: keep; the synthetic shot carries
   the goal. (Reviewer concurs; covered by the adjacency test.)
2. Whether to retrain VAEP models after the label change (flagged; not in this PR).
3. ~~Atomic-labels location/shape~~ **RESOLVED:** atomic is already result-based / dedicated `owngoal`
   type — no change, no parallel bug. Fix isolated to `vaep/labels.py`.
4. ~~Tracked follow-up (round-2 E): a dedicated provenance flag~~ **RESOLVED — shipped in this PR:**
   `is_synthetic` bool column on `GRADIENTSPORTS_SPADL_COLUMNS`, `True` on synthesized rows (cross-goal
   shot + foul). Maintainer pulled it into scope.

### Disallowed-goal over-count — root-caused and IN SCOPE (Component 4)

Review #8 noted `shotOutcomeType=="G"` over-counts goals (g3853 CAN–MAR carries a 2nd En-Nesyri @47:17,
distinct `gameEventId 6616921` / `possessionEventId 6497312`, NOT row duplication). Investigation found
the general mechanism: that event has `possessionEvents.nonEvent == True` (the real goals have `False`),
and the restart confirms it (real goal → kickoff `setpiece=K`; disallowed → free kick `setpiece=F`).
This is the same `nonEvent` voided-event flag covered by **Component 4**, which excludes all 21 voided
goals (and the 1060 voided non-goal events) — so the disallowed-goal over-count is fixed here, not
deferred.
