# TF-51 Item 4 — Defensive-credit atomic-SPADL mirror — Design

| Field | Value |
|---|---|
| **Date** | 2026-09-15 |
| **Status** | Approved — independent re-review APPROVE (2026-09-16, r2); ready for implementation plan |
| **Author session** | silly-kicks (Part Deux) |
| **Feature** | TF-51 Item 4 |
| **Source** | ADR-047 §Consequences/Neutral (v1 deferral); spec `2026-07-24-tf51-defensive-credit-v2-design.md` §Item 4 (owner split, 2026-07-24) |
| **New ADR** | Yes — TF-51 v2 Item 4 gets its own ADR at implementation (representation-port decision + the atomic `type_id` renumber migration) |

---

## Executive summary

Ship the **atomic-SPADL mirror** of the TF-51 defensive-credit family — the last piece deferred out
of v1 (ADR-047) and split to its own spec by owner decision on 2026-07-24. Three public entry points
gain an atomic sibling in `silly_kicks/atomic/tracking/features.py`, beside the shipped
packing / run-values / xt-gk mirrors:

- `compute_defensive_credits` — long-form per-(action, credited player, rule) credit rows;
- `add_defensive_credit` — the per-action defending-team aggregate (a new action-coupled aggregator
  in the `silly_kicks.atomic` container);
- `compute_bravery` — event-only per-team bravery.

The design is a genuine **representation adapter**, not a rename (unlike the thin `add_press_commitment`
atomic mirror that rode along in v2, N11). It is a pure transform + direct delegation — nothing is
inverted, so "adapter" (the `_adapter` naming below), not a hexagonal "port"; the v2 split note's
"representation-port" wording is kept only where it quotes that note. Standard defensive-credit consumes
columns the atomic projection strips — `result_id` and the derived `possession_id` — and the credit rules consume both
action endpoints (`start_x/y` **and** `end_x/y`) plus injected analytics (`xg`, `shot_blocked`,
`cross_blocked`). The mirror therefore ships a dedicated `_defensive_credit_atomic_adapter` (a richer
sibling of the shipped `_packing_atomic_adapter`) that synthesizes std `type_id` + a **per-type**
`result_id` from the atom stream and both endpoints from `x,y,dx,dy`, plus a `preserve_native`
required-column contract with a loud raise.

The mirror is **faithful**, not std-parity-seeking (owner decision, 2026-09-15): it delegates on the
atom stream exactly as every existing atomic tracking mirror does, accepts atom-dense window
semantics, reads results from the following atoms, and honest-NaNs the quantities atomic cannot
observe. It carries a **co-shipped bug fix** (bundled per owner decision A; not a hard prerequisite —
§13) — the duplicate `"interception"` entry in
`atomic/spadl/config.py`, whose reverse lookup silently resolves `interception → 24` and shadows std
interception (idx 10) — whose fix renumbers the atomic type-id tail and is a serialized-id migration.

**The defensive-credit mirror itself is additive** — **no `*_xfns`** (F4 result-leakage, inherited
from v1), in **no** default xfn list, **no defensive-credit VAEP feature change, no retrain of the
credit family**. But the co-shipped **config dedup (§9) is NOT retrain-free**: it
renumbers the atomic `type_id` tail, and the **default** `AtomicVAEP.xfns_default` consumes those ids
directly (`fs.actiontype` = raw `type_id`; `fs.actiontype_onehot` = position-enumerated per-type
columns — `atomic/vaep/base.py:19`, `atomic/vaep/features.py:65,202`). So any persisted **champion
AtomicVAEP** trained on the old encoding must be **retrained** (or proven not deployed), else it serves
old-id weights against new-id features with no gate — the exact ADR-043 fail-loud class this spec
otherwise honors (§9.3). Re-materialize/retrain obligations: atomic-SPADL re-materialize + **AtomicVAEP
retrain owed** (§9.3).

---

## 1. Context

TF-51 v1 (ADR-047, `silly_kicks/tracking/defensive_credit/`) ships signed per-defender defensive
credit over ten named rules, sized by shot xG or turnover `xT(origin)`, with three public entry
points (`compute_defensive_credits`, `add_defensive_credit`, `compute_bravery`) and no atomic mirror.
TF-51 v2 (spec `2026-07-24-tf51-defensive-credit-v2-design.md`) added the pressing lens (Item 1), the
lane-geometry `shot_block` blocker (Item 2), the through-ball line-break rule (Item 3) and the
`press_commitment` cue + its thin atomic mirror (Item 5). The v2 owner review **split Item 4 (the
defensive-credit atomic mirror) to its own later spec** (review B1), because it is a representation
port — not a refinement — with a co-shipped repo bug fix (v2's "prerequisite"; §13 records it is not a
hard dependency of the mirror). This document is that spec.

**Why an adapter, not a rename.** `atomic/spadl/base.py::_convert_columns` projects to a
fixed 13-column `base_cols` set; neither `result_id` nor `possession_id` is in `ATOMIC_SPADL_COLUMNS`,
so both are stripped by `convert_to_atomic`. The credit rules (`tracking/defensive_credit/_rules.py`)
read std `type_id` + `result_id` + `start_x/y` + `end_x/y`, and the possession-scoped resolvers
(`_chaining.py`) read a `possession_id` that `with_possessions` = `spadl.utils.add_possessions`
derives from `result_id`. So the atom stream must be re-lifted into a std-shaped stream before the
existing engine can run — the `_packing_atomic_adapter` pattern (`atomic/tracking/features.py:198`),
extended to a larger type domain and a per-type result table.

---

## 2. Scope

**In scope:**

1. `_defensive_credit_atomic_adapter(actions, params)` in `atomic/tracking/features.py` (§3).
2. Atomic `compute_defensive_credits`, `_aggregate_defensive_credit`, `add_defensive_credit` (§4, §5).
3. Atomic `compute_bravery` with honest-NaN set-piece degradation (§6).
4. `preserve_native` required-column contract + loud raise at the mirror edge (§7).
5. Per-representation (atom-dense) window semantics, documented (§8).
6. **Co-shipped fix:** dedup `"interception"` in `atomic/spadl/config.py` + the tail renumber
   migration (§9).
7. Gate registrations (purity, no-xfns absence, C4, glossary) + the documented gate-coverage limits
   (§10).
8. TDD test plan (§11).
9. **Extract `_rollup_defending_aggregate` in std `tracking/defensive_credit/_orchestration.py`**
   (§5) — the long-form → aggregate-columns step, shared by the std and atomic aggregates. This is the
   ONE shipped-std-production-code touch in this change; **owner-approved (2026-09-16, option (a))** and
   **parity-gated** so the std `add_defensive_credit` output is byte-identical (a pure DRY refactor, no
   behavior change).

**Faithful, not std-parity (owner, 2026-09-15).** The mirror delegates on the atom stream and does
not attempt to reproduce the std-SPADL numbers. This matches every shipped atomic tracking mirror
(`add_packing`, `add_off_ball_run_values`, `add_xt_gk`, `add_structural_pass`), none of which promises
byte-parity with its std sibling.

## 2.1 Non-goals

- **No std-parity.** Not a goal; not asserted; not tested for byte-equality. (§2.)
- **No window rescale.** `DefensiveCreditParams` stays frozen and shared; atomic reuses the same
  numeric defaults with documented atom-dense semantics (§8). No `for_provider`, no atomic-specific
  window fields.
- **No `*_xfns`** for either representation (F4 result-leakage; ADR-047 alt D, ADR-039/042). The
  absence guard stays green.
- **No `add_press_commitment` atomic mirror** — shipped in v2 (N11).
- **Track B** (the DPA / role-responsibility model, arXiv:2606.19931) is a separate later spec.
- **No new methodology** → `NOTICE` unchanged (a representation mirror of a cited method).

---

## 3. The adapter `_defensive_credit_atomic_adapter(actions, params)`

A dedicated sibling of `_packing_atomic_adapter` (not a reuse — the type domain and the result table
are larger). Pure; returns a NEW synthesized std-shaped frame; the caller's atomic frame is never
mutated.

### 3.1 Endpoint synthesis

Reuse `_structural_pass_atomic_endpoints(actions)` (`atomic/tracking/features.py:143`): `start_x ← x`,
`start_y ← y`, `end_x ← x + dx`, `end_y ← y + dy`. Both endpoints are required —
`rule_failed_cross_block` reads `a["end_x"]/a["end_y"]` (the cross receipt point,
`_rules.py:445`); every other rule reads `start_x/start_y`.

### 3.2 Std `type_id` map (atomic → std, symbolic ids)

Map only the types the rules or the chained resolvers test; everything else → std `non_action`
(off-domain, mirroring packing). Symbolic lookups only (`spadlconfig.actiontype_id[...]` /
`atomicconfig.actiontype_id[...]`) — never raw ints (a future config renumber must not silently
break the map; packing's D2 discipline).

| Atomic type | Std type | Why the rules need it |
|---|---|---|
| `pass` | `pass` | pressure-pass-fail, synchronized-pressure, recovery, through-ball anchors |
| `cross` | `cross` | failed-cross-block anchor |
| `shot` | `shot` | shot rules + goal detection + resulting-shot |
| `shot_penalty` | `shot_penalty` | resulting-shot (`_chaining._SHOT_TYPE_IDS`) |
| `take_on` | `take_on` | beaten-1v1 anchor |
| `bad_touch` | `bad_touch` | forced-bad-touch anchor |
| everything else | `non_action` | off-domain (recovery keys on team-change, not type) |

`tackle` / `interception` / `clearance` are deliberately **not** in the domain map: no rule tests
those std types (recovery is resolved by the first opponent regain via team-change in
`_chaining.recovery_after_pass`, not by type).

### 3.3 Per-type `result_id` synthesis (the faithful table)

Read the outcome from the **following atom** (same game+period), per anchor type. This is the atom
stream's own encoding of the result — the faithful reading. `result_id` feeds three consumers:
`with_possessions`/`add_possessions` (possession chaining), `_ensure_on_target` (goal → on-target),
and the rules' `result_id == success/fail` gates.

| Anchor type | Synthesized `result_id` | Rule |
|---|---|---|
| `pass`, `cross` | `success` iff next atom = `receival` **or** a same-team `keeper_pick_up`/`keeper_claim`; else `fail` | packing's completion rule (`_packing_atomic_adapter:250`); atomic inserts no `receival` before a keeper collection |
| `take_on` | `success` iff next atom is **same-team** and **not** `interception`/`out` (ball retained by the attacker); else `fail` | a beaten-1v1 is a *retained* dribble-past, NOT a pass reception — the receival rule is wrong here |
| `shot`, `shot_penalty` | `success` (= goal) iff next atom = `goal`; else `fail` | shot rules gate on block/on-target, but `result==success` is the goal → on-target signal (`_rules._GOAL_RESULT`) |
| `bad_touch` | `fail` (result unread by `rule_forced_bad_touch`) | anchor is `type==bad_touch` only |

The last row of a game+period (no following same-gp atom) synthesizes `fail` for pass/cross/take_on
(no observed completion) and `fail` for shot (no goal atom) — the honest atom-stream reading.

**Reviewer note — `take_on` is the one non-trivial synthesis.** `convert_to_atomic` inserts no
outcome atom after a `take_on` (`_compute_pass_extras` handles only pass-like actions), so its result
is inferred from the *next real action's* team (retained vs lost), not from an inserted
`receival`/`interception`. pass/cross (receival probe), shot (goal probe) and bad_touch (unread) are
unambiguous atom readings; `take_on` is a heuristic on ball retention. It gates exactly one rule
(`beaten_1v1`, itself further gated on a resulting shot ≥ `beaten_1v1_min_shot_xg`), so a
mis-inference is bounded to that rule. Pinned by a dedicated adapter test (§11) on both a retained and
a dispossessed take_on. The faithful alternative — `preserve_native=["result_id"]` for take_on only —
is rejected as inconsistent (mixing preserved std results with atom-read results in one stream).

### 3.4 Assembly

The mirror runs the adapter, delegates to the std sub-package
(`silly_kicks.tracking.defensive_credit`), and **assembles the enrichment on a COPY of the caller's
atomic frame** — the synthesized `type_id` / `result_id` / `start_*` / `end_*` are mirror-internal and
never leak into the returned columns (packing's D3 discipline, `atomic/tracking/features.py:294`).

---

## 4. Atomic `compute_defensive_credits`

Signature mirrors the std entry
(`tracking/defensive_credit/_orchestration.py:73`):

```python
def compute_defensive_credits(actions, frames, *, xg_column, xt,
                              blocked_column="shot_blocked",
                              on_target_column="shot_on_target_derived",
                              links=None, params=None) -> pd.DataFrame:
```

Body: `adapted = _defensive_credit_atomic_adapter(actions, params)` → delegate to the std
`compute_defensive_credits(adapted, frames, ...)`. Possession chaining is **not** re-invoked by the
mirror — the std entry already calls `with_possessions` at its top
(`_orchestration.py:94`), and it now succeeds because `result_id` is present on the adapted stream
(§3.3). The long-form output schema (`_LONG_COLS`) is unchanged — it already carries `game_id/period_id/action_id/player_id/team_id`
which survive the atomic projection natively; `frame_id` comes from the same
`link_actions_to_frames`/`links` path (frames are representation-agnostic).

`_ensure_on_target` works unchanged: it reads the synthesized `result_id` for the goal case and falls
back to the TF-48 `add_shot_goalmouth` frame-based `shot_on_target_derived` otherwise (frames identical
across representations).

## 5. Atomic `add_defensive_credit` (+ `_aggregate_defensive_credit`)

Mirrors `tracking/features.py:7210`:

```python
def add_defensive_credit(actions, frames, *, xg_column, xt,
                         blocked_column="shot_blocked",
                         on_target_column="shot_on_target_derived",
                         links=None, params=None, visible_area=None) -> pd.DataFrame:
```

**Correction (review r1, 2026-09-16): the std `_aggregate_defensive_credit` is NOT reusable verbatim.**
It re-computes the long-form internally (`_orchestration.py:205`, `long = compute_defensive_credits(
actions, ...)`) and assembles on `actions.copy()` — so calling it on the atomic actions would re-run
the STD compute (which the atom stream cannot feed) and, on the adapted stream, leak the synthesized
`type_id`/`result_id`/`start_*` columns. What IS representation-agnostic is the **rollup STEP** alone
(the long-form → aggregate-columns body, `_orchestration.py:215-246`): it groups the long-form on
`action_id`, splits defending vs acting by `ids_differ(team_id, acting-team)`, emits
`defensive_credit_net/_plus/_minus` + `n_defensive_credits`, and appends the ADR-077 `visible_area`
companions — all consuming only the long-form `resolution/origin_x/origin_y/region_radius` + the
action's polygon.

So the atomic mirror computes its OWN atomic long-form (atomic `compute_defensive_credits`) and applies
that rollup step, **assembling on the caller's ATOMIC frame** (never the adapted stream — packing's D3
no-leak discipline). The rollup step is shared with the std path.

**Scope decision (surfaced, owner-gated) — how the rollup step is shared:**

- **(a) Extract `_rollup_defending_aggregate(actions, long, *, params, visible_area, links)` in std
  `_orchestration.py`**, called by both the std `_aggregate_defensive_credit` and the atomic mirror.
  Single-source (DRY, matches the codebase's single-truth discipline); parity-gated so std output is
  byte-identical. **Touches shipped std production code** → expands this spec's §2 scope (owner
  sign-off required; recorded here 2026-09-16).
- **(b) Atomic-local rollup** (duplicate ~30 lines in `atomic/tracking/features.py`); std
  `_orchestration.py` untouched, no §2 expansion. Costs a second copy that can drift.

**Owner decision (2026-09-16): (a) — extract the shared `_rollup_defending_aggregate` in std
`_orchestration.py`.** The std `_orchestration.py` refactor is thereby owner-approved scope (§2 item 9);
it is parity-gated so the std `add_defensive_credit` output stays byte-identical. Plan Task 3 implements
it, Task 5 consumes it.

`add_defensive_credit` is the **new action-coupled aggregator** in the `silly_kicks.atomic` container
(the 16th `atomic.tracking` feature mirror).

---

## 6. Atomic `compute_bravery` (honest-NaN set-piece)

Event-only; mirrors `tracking/defensive_credit/_bravery.py:36`:

```python
def compute_bravery(actions, *, shot_blocked_column="shot_blocked",
                    cross_blocked_column="cross_blocked") -> pd.DataFrame:
```

**Survives atomic intact** (`_simplify` does not touch these types): `shot` and open-play `cross`, so
`bravery_shots`, `bravery_open_play_crosses`, `bravery_pct_known_domain` (the headline — built on
shots + open-play crosses only), `n_shots_faced`, `n_open_play_crosses_faced`, `n_blocks_known` are
computed exactly.

**Honest-NaN degradation.** `atomic/spadl/base.py::_simplify` collapses `corner_crossed` +
`corner_short` → atomic `corner`, and `freekick_crossed` + `freekick_short` + `shot_freekick` →
atomic `freekick`. Atomic cannot distinguish a crossed set-piece from a short one, so the two
set-piece columns emit **honest-NaN / `pd.NA`** — never a conflated crossed+short overcount
(ADR-027 — a fabricated count is worse than a declared absence):

- `bravery_set_piece_crosses` → `NaN` (v1 already NaNs this as a column limitation; atomic NaNs it
  for a *representation* reason, documented).
- `n_set_piece_crosses_faced` → `pd.NA` (Int64).

The mechanism: the atomic mirror computes bravery over the collapsed atomic types and, because the
set-piece-crossed types carry zero atomic rows, sets the two set-piece columns to NaN/NA directly
rather than counting the ambiguous `corner`/`freekick` collapsed rows.

---

## 7. `preserve_native` required-column contract + loud raise

`convert_to_atomic(std, *, preserve_native=[...])` (`atomic/spadl/base.py:13`) already surfaces
caller-chosen std columns unchanged through the atomic projection, with `_validate_preserve_native`
raising on a missing or schema-overlapping name **at conversion time**. That validates the columns
exist when converting; it does **not** know which columns a *downstream* metric needs. So the mirror
adds its own **required-column raise at its edge**:

- Atomic `compute_defensive_credits` / `add_defensive_credit`: require `xg_column` and
  `blocked_column` (default `shot_blocked`) to be present on the atomic input.
- Atomic `compute_bravery`: require `shot_blocked_column` and `cross_blocked_column` to be present
  (else the block signal is silently absent and every rate is NaN — the R2-2 defect the std guard
  already avoids by NaN-not-zero, but on atomic the *column itself* is the thing that goes missing).

The raise is a `ValueError` that names the missing column **and** the remedy, e.g.:

```
add_defensive_credit (atomic): required column 'xg' is absent. Injected analytics are not
atom-derivable — thread them through the conversion:
convert_to_atomic(std_actions, preserve_native=['xg', 'shot_blocked', 'cross_blocked']).
```

Rationale (ADR-043 discipline): a forgotten injected column must fail LOUD, not degrade into an
all-NaN credit column indistinguishable from "no credit was earned".

**`possession_id` needs no preservation** — it is not a native input; `with_possessions` =
`add_possessions` derives it from the synthesized `result_id`, so §3.3 fixes it transitively.
`shot_on_target_derived` needs no preservation — `_ensure_on_target` recomputes it from frames.

---

## 8. Window semantics (atom-dense, documented)

`resulting_shot_max_actions` (default 10) and `recovery_max_actions` (default 3) count **rows** of the
stream. On atomic the stream is denser (a completed pass = pass atom + receival atom; a scored shot =
shot atom + goal atom), so N rows spans fewer real events than on std. Consequences, all documented in
the mirror docstrings (faithful — no rescale):

- **Recovery gains precision.** A failed pass's first opponent regain on atomic is typically the
  inserted `interception` atom at distance 1 — the regain *is* the atom, tighter than the std scan.
- **Resulting-shot window is effectively tighter.** 10 atoms ≈ 5 std actions, so the chained rules
  (`beaten_1v1`, `failed_cross_block`, `failed_marking_through_ball`) reach fewer forward events.

The params stay frozen and shared (§2.1); the atomic docstrings state the atom-count semantics
explicitly so a consumer reads the meaning rather than assuming std-action counts.

---

## 9. Co-shipped fix — `interception` dedup + tail renumber (migration)

### 9.1 The bug

`atomic/spadl/config.py::actiontypes` re-appends `"interception"` even though it is already inherited
from `_spadl.actiontypes` at index 10:

```python
actiontypes = [*_spadl.actiontypes, "receival", "interception", "out", ...]
#                                                  ^ duplicate — std idx 10 already has it
```

The reverse lookup `actiontype_id = {name: i for i, name in enumerate(actiontypes)}` keeps the LAST
occurrence, so `actiontype_id["interception"] == 24`. Effect: synthetic pass-interception atoms
(inserted by `_compute_pass_extras`, tagged via the reverse lookup) are written as **24**, while
converted std interception actions keep their std id **10** (`_convert_columns` does not remap them).
`actiontypes_df()` renders both as `"interception"` so `add_names` looks fine — but any code filtering
atomic interceptions via `actiontype_id["interception"]` matches only the synthetic (24) rows and
**silently excludes** the converted std ones (10).

### 9.2 The fix

Remove the duplicate append. `"interception"` is then inherited once at idx 10; the synthetic atoms —
written via the symbolic `actiontype_id["interception"]` — automatically become 10, unifying the two
readings. The tail renumbers down by one:

| type | before | after |
|---|---|---|
| `receival` | 23 | 23 |
| `interception` (appended) | 24 | *(removed)* |
| `out` | 25 | 24 |
| `offside` | 26 | 25 |
| `goal` | 27 | 26 |
| `owngoal` | 28 | 27 |
| `yellow_card` | 29 | 28 |
| `red_card` | 30 | 29 |
| `corner` | 31 | 30 |
| `freekick` | 32 | 31 |

### 9.3 Blast radius (verified)

- **Code auto-follows (sweep is the floor, not the check).** The brainstorm grep over
  `silly_kicks/**/atomic/**` found every consumer resolving the id symbolically at runtime
  (`actiontype_id["goal"]`, `["owngoal"]`, `["receival"]`, …; `vaep/labels.py`, `vaep/features.py`,
  `spadl/utils.py`, `tracking/features.py`) — no hardcoded 24/25/27. The implementation **re-runs the
  sweep over `tests/` and the whole tree** for any hardcoded atomic tail-id int or any
  `atomicconfig`/`_atomicspadl.actiontype_id[...]` tail read before concluding "no code edit beyond
  the config" (a name sweep cannot see a fixture or a number recorded in prose).
- **Serialized data must be regenerated.** Atomic `type_id` is a serialized value. Two committed
  artifacts are the known candidates and must be checked + rebaselined:
  - `tests/datasets/spadl/atomic_spadl.json`
  - `tests/atomic/_golden_atomic_pre_shot_gk_context_v280.parquet`
  The implementation enumerates every committed `.json`/`.parquet`/`.csv` fixture derived from
  `convert_to_atomic` (not only these two), regenerates only those carrying ids in the 24–32 tail, and
  proves each regen is purely the renumber (no other column moves).
- **Trained-model surface — AtomicVAEP retrain owed (the load-bearing consequence).** The renumber
  changes the **default** AtomicVAEP feature encoding, not just the feature *code*. `AtomicVAEP`'s
  `xfns_default` (`atomic/vaep/base.py:19`) includes `fs.actiontype` — the raw integer `type_id` as a
  categorical feature (`atomic/vaep/features.py:65`, `actiontype_categorical`) — and `fs.actiontype_onehot`,
  which builds one boolean column per type by **position** (`features.py:202`,
  `for type_id, type_name in enumerate(atomicspadl.actiontypes)`). The tail shift (25→24 … 32→31) and
  the interception unification (24-only → 10-unified, one fewer column) therefore move both features.
  "Code auto-follows symbolically" covers the extractor; it does **not** cover a *trained* model, whose
  weights are keyed to the old ids. Consequence if missed: an operator re-materializes atomic SPADL but
  keeps a champion AtomicVAEP → old-id weights served against new-id features → every tail/interception
  atomic VAEP value silently wrong, with **no gate** (the ADR-043 fail-loud class this spec cites). So:
  **any persisted/champion AtomicVAEP consuming the default xfns must be retrained on the new encoding,
  or proven not deployed.** silly-kicks ships no bundled AtomicVAEP weights (the model is fit by the
  consumer), so this obligation lands on the lakehouse; it is recorded here, in the CHANGELOG, and in
  the owner handoff — not silently assumed away. Also fix the now-stale
  `actiontype_onehot` docstring "33 boolean columns" → 32 (`features.py:199`) in the same §9 change.
- **Lakehouse atomic re-materialize owed** (any persisted atomic-SPADL table encoding tail ids),
  **paired with the AtomicVAEP retrain above** — a re-materialize WITHOUT the retrain is the exact
  silent-skew failure mode. Recorded for the CHANGELOG + owner handoff.

### 9.4 Guard

Red-first test asserting `actiontype_id["interception"] == 10`, that `"interception"` appears exactly
once in `actiontypes`, and that the tail ids match §9.2. This catches *reintroduction* of the
duplicate (a deletion alone cannot).

---

## 10. Gate registrations + documented gate-coverage limits

- **Purity (`tests/test_add_star_purity.py`).** Register `atomic.tracking:add_defensive_credit` with
  **two variants** (with / without `visible_area` — the ADR-033 conditional-column contributor
  contract, exactly as `tracking:add_packing` and the std `add_defensive_credit` are registered). Bump
  the header prose "15 feature mirrors" → 16.
- **No `*_xfns` absence guard.** Ships none in either representation; the auto-discovering absence
  guard (anchored on the transformer NAME) stays green.
- **id-dtype / nan-safety / liveness gates are tracking-only** (they iterate `tracking.__all__`, not
  `atomic.tracking`). Per the v2 N11 finding, the atomic mirror is **gate-covered by purity alone** —
  an honest, documented limitation, not a silent assumption. The correctness of the credit VALUES on
  atomic is pinned by the §11 behavioral tests instead.
- **C4 — the *mirror* needs no DSL edit; the *dedup* does.** The `silly_kicks.atomic` container
  (`docs/c4/architecture.dsl:24`) carries **no numeric aggregator count** (unlike `tracking`'s "33"),
  and adding one more mirror leaves its text unchanged — the mirror alone is DSL-free. **But that same
  line says "continuous 33-type action representation", and `33 = len(atomicconfig.actiontypes)`,
  which §9 drops to 32.** No C4 gate binds it: `test_c4_aggregator_count` matches "action-coupled
  aggregators" and `test_c4_feature_column_count` matches "derived feature columns", so the completeness
  gate stays green while the prose rots. So the §9 change **edits `architecture.dsl:24` "33-type" →
  "32-type" and regenerates `architecture.html`** (Graphviz `dot`, per the C4 pipeline) — not "no DSL
  edit". Std `add_defensive_credit` is already inside tracking's "33".
- **feature_glossary.** The mirror emits only already-documented column names
  (`defensive_credit_net/_plus/_minus`, `n_defensive_credits`, the `bravery_*` set) — `FEATURE_GLOSSARY`
  keys on column NAME and `emitting_module` is documentation, not gate-verified — so **no new glossary
  rows** and `test_no_undocumented_columns` / `test_no_stale_entries` stay green.

---

## 11. Test plan (TDD, red-first)

Each behavior lands as a failing test first, then the implementation.

**Adapter (`_defensive_credit_atomic_adapter`):**

- Endpoint synthesis: `start ← x/y`, `end ← x+dx / y+dy`.
- Std type map: `pass/cross/shot/shot_penalty/take_on/bad_touch` mapped; all else → `non_action`.
- Per-type result table (§3.3), one assertion each:
  - pass/cross completed (next = `receival`) → `success`; failed (next = `interception`/`out`/none) →
    `fail`; completed back-pass to keeper (next = same-team `keeper_pick_up`) → `success`.
  - take_on retained (next = same-team, not interception/out) → `success`; dispossessed → `fail`.
  - shot scored (next = `goal`) → `success`; missed/saved (no goal atom) → `fail`.
  - bad_touch → `fail` (unread).
  - game/period-last atom → `fail`.
- No caller mutation; synth columns absent from the delegated-through return.

**Mirrors (structure, NOT byte-parity — asserted through the PUBLIC return, not only the adapter):**
The adapter tests above pin the private transform; these pin the delegated end-to-end behavior on the
public entry points, so a regression in the delegation/assembly (not just the adapter) is caught.

- `compute_defensive_credits` on a hand-built atomic fixture produces the expected credit ROWS
  (rule / sign / credited team) for a scripted scene (a pressured failed pass → recovery; a blocked
  shot; a beaten 1v1 → resulting shot). Assert structure + signs, never std byte-equality.
- `add_defensive_credit` aggregate: net/plus/minus/n over the same fixture; the `visible_area`
  companion path on a polygon fixture (opt-in additive; primary columns byte-identical with/without).
- Faithful-limitation pins: `shot_freekick` collapsed → NOT counted as a resulting shot; atom-dense
  recovery fires on the distance-1 interception atom.

**`preserve_native` raise:**

- Atomic `add_defensive_credit` with `xg` absent → `ValueError` naming `xg` + the remedy.
- Atomic `compute_bravery` with `cross_blocked` absent → `ValueError` naming `cross_blocked`.

**Bravery:**

- Honest-NaN: `bravery_set_piece_crosses` = NaN, `n_set_piece_crosses_faced` = `pd.NA` on atomic;
  `bravery_shots` / `bravery_open_play_crosses` / `bravery_pct_known_domain` computed exactly.

**Config fix (§9.4):** red-first id-map assertions + fixture-regen parity.

**Purity:** the two-variant `atomic.tracking:add_defensive_credit` entry.

Run the full non-e2e suite green (`python -m pytest tests/ -m "not e2e"`), lint at CI scope
(`python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`), and `python -m pyright`
before proposing the commit.

---

## 12. Delivery

- **One feature branch** off `main` (per-cycle; no worktree).
- **One fully-tested commit** — the mirror + the prereq config fix + fixture regen + gate
  registrations + tests as a single coherent, green state. No micro-commits.
- **Explicit human-approval gate before the commit.** The implementation session stops at the point
  of committing, shows the diff / file list, and waits for Karsten's explicit yes. No spec/plan step
  authorizes the commit.
- **Per-PR version bump** + CHANGELOG entry (new `PR-Snnn`, a new ADR number) recording: the atomic
  `type_id` renumber (breaking; lakehouse atomic re-materialize owed), the faithful-representation
  limitations, and "additive to VAEP, no retrain, no `*_xfns`".
- **Independent review** of this implementation runs in a separate session Karsten starts — not this
  author session.

---

## 13. Open questions

Resolved during brainstorming:

- Success criterion — **faithful, not std-parity** (owner, 2026-09-15).
- Bravery set-piece — **honest-NaN** degradation, not a forced `preserve_native` discriminator
  (owner-confirmed; follows faithful + ADR-027).
- `result_id` — **synthesized** from the next atom per §3.3, not preserved (owner split note;
  faithful-atomic reading).

Resolved during independent review (2026-09-15, report
`D:\Development\_reviews\2026-09-15-tf51-item4-defensive-credit-atomic-mirror-spec.md`):

- **The config dedup changes the DEFAULT AtomicVAEP encoding → AtomicVAEP retrain owed** (§9.3);
  "no VAEP retrain" narrowed to the defensive-credit family only. Verified against
  `atomic/vaep/base.py:19` + `atomic/vaep/features.py:65,202`.
- **The dedup edits `architecture.dsl:24` "33-type" → "32-type"** (§10); "no DSL edit" was false.

**One owner decision surfaced (new information, re-raising a prior call).** The config dedup is a
**co-shipped independent bug fix, NOT a hard prerequisite** for the mirror — the adapter maps
`interception → non_action` and never reads its id, so the mirror is correct whether interception is
10 or 24 (verified §3.2/§3.3). The dedup, not the mirror, carries the AtomicVAEP-retrain blast radius.
The v2 split note said "this spec owns the fix", but that predates the discovered retrain obligation.
Options were **(A) keep bundled** (honors the v2 decision; this spec ships mirror + dedup + the
AtomicVAEP-retrain handoff as one commit) or **(B) split the dedup + its AtomicVAEP migration into its
own PR**. **Owner decision (2026-09-16): (A) — keep bundled.** The spec is written for A; the single
commit carries the mirror + the dedup + fixture regen + the `dsl:24` "32-type" edit + the CHANGELOG
AtomicVAEP-retrain/re-materialize handoff.

---

## 14. References

- ADR-047 (TF-51 v1; §Consequences/Neutral defers the atomic mirror).
- Spec `docs/superpowers/specs/2026-07-24-tf51-defensive-credit-v2-design.md` §Item 4 (owner split).
- Precedent: `_packing_atomic_adapter` + `_structural_pass_atomic_endpoints`
  (`silly_kicks/atomic/tracking/features.py`); atomic `add_off_ball_run_values` / `add_xt_gk`.
- ADR-027 (honest-NaN, never a fabricated sentinel), ADR-033 (add_* purity + conditional-column
  variants), ADR-039/042 (F4 xfn result-leakage), ADR-043 (missing ≠ 0; fail-loud), ADR-077 (FOV
  `visible_area` companions), ADR-019 (id-compat), ADR-005 (attribution — unchanged here).
- `NOTICE` — Sumpter, Soccermatics Pro module 16.3; Bischofberger/Bauer/Baca, arXiv:2606.19931
  (unchanged; no new methodology).
