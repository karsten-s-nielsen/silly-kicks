# TF-51 v2 — Defensive-credit refinements (design)

**Date:** 2026-07-24
**Status:** rev 3 — addresses analysis-review-1 + analysis-review-2 (both verified against code). Review-2
close-out (N1–N12) applied; reviewer noted those are edits, not decisions → **ready to plan.**
**Version / PR / ADR:** placeholders until commit-prep (register next-free is provisional
`4.61.0 / PR-S132 / ADR-049` as of 2026-07-24 — the parallel session took the earlier `4.60.0/PR-S131`
reservation via #175; re-confirm at commit-prep; do not pre-claim — [[feedback_no_version_number_until_commit_prep]]).
**Supersedes / amends:** TF-51 v1 (`ADR-047`, `silly-kicks 4.57.0`,
`docs/superpowers/specs/2026-07-22-tf51-defensive-credit-design.md`, §11 deferrals) and the block-detection
`ADR-046` **Opta** status (§10).

---

## 1. Scope

Four bounded refinements to the **shipped** v1 defensive-credit family
(`silly_kicks/tracking/defensive_credit/`, 10 rules + bravery), **plus one bundled v1 bug fix** (B2). This is
the "refine what we shipped" half of the v1 §11 backlog ("Track A").

**Item 4 (atomic-SPADL mirror) is SPLIT into its own later spec** (owner decision, 2026-07-24, on analysis-
review-1 B1). It is not a refinement — it is a new representation port whose v1-spec justification ("the D3
hard part dissolves via possession-preserved chaining") was **factually wrong**: `_convert_columns`
(`atomic/spadl/base.py:248-274`) projects to a fixed `base_cols`, and neither `possession_id` nor `result_id`
is in `ATOMIC_SPADL_COLUMNS`, so both are stripped. It needs the `_packing_atomic_adapter`-class lookahead
bridge (synthesize std `type_id` **and** `result_id` from the next atom — `atomic/tracking/features.py:192+`),
a documented `preserve_native=[…]` caller contract + loud raise, per-representation window semantics, and a
**prerequisite fix** for a real repo bug: duplicate `"interception"` in `atomic/spadl/config.py` (std idx 10 +
atomic idx 24 → the dict-comprehension keeps 24, silently excluding std interception events). That spec owns
the interception-id fix. **The Paper-2 DPA/role model ("Track B") remains a separate later spec.**

The four items + the fix:

| # | Item | Kind | Output effect |
|---|------|------|---------------|
| 1 | Reverse-xT "position won" pressing lens | opt-in sizing param | none by default (opt-in) |
| 2 | Lane-geometry `shot_block` blocker | replace-in-place | `shot_block` attribution changes |
| 3 | Line-break-gated through-ball | replace-in-place | `failed_marking_through_ball` firing changes |
| 5 | Pressure-commitment cue | additive (new feature) | new columns only |
| B2 | `recovery_after_pass` game/period-boundary fix | bug fix | fewer false cross-game recoveries |

**Internal implementation order: 3 → 1 → 2 → 5** (Item 3 first — it removes `rule_failed_marking_through_ball`'s
`_xt_at` calls, which cleans Item 1's sizing seam; see §3/§5). The B2 fix lands first of all (it is a
prerequisite for correctness regardless of item order).

**Explicitly OUT (see §10):** Opta block-detection (dropped from roadmap entirely); Item 4 (split spec);
the Paper-2 DPA/role model; the passing-lane *blocking-credit* rule; the individual `cross_block` rule; a
general `ax/ay` acceleration primitive (TF-50).

---

## 2. Architecture placement & C4

- **Items 1–3 refine `tracking/defensive_credit/`** in place — edits to `_rules.py` (policy), `_resolution.py`
  (the *who* — Item 2's lane geometry belongs here, **not** in `_rules.py`; H1), `_sizing.py` (Item 1's lens),
  `_orchestration.py` (Item 3's precomputed line-break signal + the B2 fix), `_params.py` (new params +
  constants). **No new aggregator** in this sub-package.
- **Item 5 is separate**: a new `tracking/_press_commitment.py` primitive + `add_press_commitment` aggregator
  in `tracking/features.py` + atomic mirror. **New action-coupled aggregator → re-derive the C4 count at
  commit-prep** (ADR-043 says *re-derive*, do not copy a written number; the convention excludes
  `add_gradientsports_player_ids`). Expected +1 in `silly_kicks.tracking`, plus the thin atomic mirror (§6,
  N11) in the **separate** `silly_kicks.atomic.tracking` container — re-derive both.
- **`features.py` monolith note (H2):** `add_press_commitment` joins `add_defensive_credit` in the ~6,400-line
  `tracking/features.py`. That matches precedent (every public producer in the namespace lives there), so it
  isn't a blocker — recorded so the emitting-module attribution limitation (the same one the merged glossary
  spec hit) is acknowledged, not rediscovered.
- **No new `*_xfns` anywhere.** The credit rules stay out of every default xfn list (F4 result-leakage, ADR-039/
  042 — unchanged). The commitment feature ships **aggregator-only**; the deferral is **enforced by a test**
  (T4: extend `tests/tracking/test_defensive_credit_xfns_absence_guard.py` to assert
  `not hasattr(T, "press_commitment_xfns")`), not left as prose. → **No VAEP retrain from any item.**

---

## 3. Item 1 — Reverse-xT "position won" pressing lens (opt-in)

**What.** The four xT-sized turnover rules (`pressure_pass_fail`, `forced_bad_touch`,
`synchronized_final_third_pressure`, `recovery_double_credit`) size by `xT(origin)` today. Add an **optional**
opponent-perspective "position won" lens that sizes by `xT(105−x, 68−y)` — rewarding regains near the
opponent's goal (high pressing) over danger-prevented.

**Where — per-call-site, NOT at the `_xt_at` seam (B6).** `_xt_at` (`_rules.py:141`) has **five** callers, and
`rule_failed_marking_through_ball` calls it **twice** to compute a ΔxT gate (`:405`) — that rule is not a
turnover rule, so reflecting inside `_xt_at` would silently corrupt its gate. The lens is therefore applied at
each turnover rule's own sizing call-site via a **primitives-only** helper `sized_xt(x, y, xt, *,
pressing_lens: bool) -> float` — **it takes primitives, NOT `ctx` (N2).** A `RuleContext`-typed helper in
`_sizing.py` would import `RuleContext` from `_rules.py`, closing a `_sizing → _rules → _sizing` cycle
(`_rules.py:34` already imports from `_sizing`; `_sizing.py` deliberately has *no* first-party imports — its
xthreat import is function-local for exactly this reason; `tests/test_no_import_cycles.py` gates it). Keeping
the helper primitives-only preserves the value **port** (H1: `_sizing` is pure functions over points + an
injected surface, with no knowledge of rule orchestration); each turnover rule reads
`ctx.params.pressing_lens` and passes the bool. (Equivalently the helper may live in `_rules.py` beside
`_xt_at` — either preserves the port.) This leaves `_xt_at` — and the through-ball gate — untouched. **Ordering coupling:** after **Item 3** lands, `rule_failed_marking_through_ball` no longer calls
`_xt_at` at all (it gates on the line-break, §5), so the only remaining `_xt_at` callers are the four turnover
rules. Per-call-site gating is still the stated mechanism (correct independent of order); the 3→1 order simply
makes the seam naturally clean.

**Which point reflects (the two-anchor rules).** The lens reflects **each row at the exact point it is sized
at**: `pressure_pass_fail` sizes both rows at the passer origin (they stay equal-and-opposite under the lens);
`recovery_double_credit` sizes the `+recoverer` at the recovery location and the `−passer` at the passer
origin, so each reflects its own point. **Applies to `−` (debit) rows too**, uniformly — a `−` row sized by
"position won" is the mirror-threat the loss conceded at that spot; the `−` rows are excluded from the
defending-team aggregate anyway (R2-1) and live only in the long-form for per-player rollup. The alternative
("+ rows only") is recorded in Open Questions (§12).

**Decision (c): a single global param `pressing_lens: bool = False`** on `DefensiveCreditParams`, not per-rule.
The lens most affects `synchronized_final_third_pressure` (a deep press sized ≈0 under origin becomes
magnitude-valuable). Per-rule split is YAGNI — deferred.

**Provenance + the `sizing`/`anchor_type` closed vocabularies (B5).** There is **no** `sizing` closed vocab
today — `SIZING_XG`/`SIZING_XT` are two loose constants and the only guard is a hardcoded literal
`set(out["sizing"]) <= {"xg","xt"}` (`test_defensive_credit_orchestration.py:43`, a subset check that can't
even detect a *missing* token). Build the structure first, in this order:
1. Add `SIZING_VALUES: tuple[str, ...] = (SIZING_XG, SIZING_XT, SIZING_XT_PRESSING)` in `_params.py`, mirroring
   `DEFENSIVE_CREDIT_RULES`.
2. Repoint the orchestration test at the constant and make it **exhaustive when the lens is on** (assert the
   set *equals* the expected subset, not merely `<=`).
3. *Then* add `xt_pressing` (turnover rows only, lens on). **Keep `xt` = origin** (default byte-identical); no
   token is renamed.
While in the file, close the same gap for `anchor_type` — five inline literals (`"shot"`, `"pass"`,
`"bad_touch"`, `"cross"`, `"take_on"`) in `_rules.py` with no constant and no test: add
`ANCHOR_TYPE_VALUES: tuple[str, ...]` + a closed-set test.

**Output effect.** Default (`pressing_lens=False`) → **byte-identical to v1** (no token rename; `xt_pressing`
never appears). Fully opt-in → **not a retrain and not a re-materialize** unless a consumer turns it on.

**Attribution / caveat.** `xT(origin)` remains the validated default (arXiv:2606.19931). Carry v1's **full**
caveat into the docstring + NOTICE (v2 rev 1 had softened it): the lens "diverges from the validated standard
**and under-values last-ditch defending**." A worked numeric example goes in the docstring (see T1 on why).

---

## 4. Item 2 — Lane-geometry `shot_block` blocker (replace-in-place)

**What.** `rule_shot_block` credits the block to the defender **nearest the shot origin** today
(`_shot_credit(..., mode="nearest")`) — it can credit a defender near the ball but not in the shot's path.
Replace it with the defender geometrically **in the shot→goal lane**.

**Where (H1): in `_resolution.py`, not `_rules.py`.** Widen the resolution `Mode` Literal
(`_resolution.py:15`, currently `Literal["nearest", "all_within", "all_within_beyond_nearest"]`) to add
`"lane_blocker"`; add a closed-set test on the Literal (mirrors B5). `rule_shot_block` stays a thin policy
line that asks for `mode="lane_blocker"`; all geometry lives in the resolver.

**Candidate set (B7 — the load-bearing decisions).**
- **The origin-proximity threshold does NOT apply to `lane_blocker`.** v1's `resolve_responsible_defenders`
  filters to defenders within the box-aware 4.5 m / 3.0 m radius **of the shot origin** — which would filter
  out a real lane blocker 10 m from the origin before the lane test runs (and if the threshold stayed, the
  item would be near-vacuous). So `lane_blocker` candidates are **not** origin-distance-filtered; they are
  filtered by the lane corridor + in-front-of-shot instead.
- **Exclude the goalkeeper — by BOTH mechanisms (N5, defence-in-depth).** The GK stands on the shot→goal
  segment near `(105, 34)` on essentially every shot, so a naive lane minimiser credits the keeper — a
  category error (a keeper stop is a *save*, not a block; `shot_blocked` from the converters means an outfield
  block). Exclude `is_goalkeeper` defenders from the candidate set **and** cap candidates by distance-along-lane
  (a blocker must be materially in front of the goal-mouth, not on the line). The `is_goalkeeper` flag alone is
  **not trustworthy on the e2e provider**: Gradient Sports sets it **all-False with only a `UserWarning`** when
  the roster lacks `position_group_type` (`gradientsports.py:315-321`; Metrica derives positionally,
  `metrica.py:140`; kloppy overwrites its native flag, `kloppy.py:189-213`) — so a flag-only exclusion can
  silently no-op, and the distance cap is the flag-independent backstop. Red-first test with a GK on the line
  that **first asserts `is_goalkeeper` is genuinely set** on the fixture (not a vacuous all-False frame; T5).
- **Corridor = the repo's distance-scaled cone, not a fixed metre (B7).** Match `_cover_shadows`'s existing
  lane convention (`cone_width_factor = 0.2`, half-width scales with distance — `_cover_shadows.py:48,558`)
  rather than introducing a second, fixed-metre lane-width convention. Param `shot_lane_cone_width_factor:
  float` on `DefensiveCreditParams`, validated positive in `__post_init__` (`_params.py:78-89` — which the
  rev-1 spec forgot to mention).
- **Blocker = the in-corridor, in-front, non-GK defender with minimum perpendicular distance to the lane**
  (the tightest to the ball's path).

**Fallback discipline (B8, N10).** Fires only on `shot_blocked == True`, tracking-gated. If no frame links or
no defender falls in the corridor, **fall back to `mode="nearest"`** — a lane-unresolvable block is still a
real, xG-sizable block, and the family's `signed_value = NaN` convention is for *unsizable* rows, not
*misattributed* ones, so a real attributee (approximate) beats emitting NaN or no credit. **Three outcomes,
not two:** `lane` (resolved on the lane) / `nearest_fallback` (fell back) / **no row at all** — when `nearest`
*also* finds nobody within its origin threshold, `_shot_credit` emits no row (v1 behaviour too). The fallback
is **not silent** (it is recorded in the `resolution` provenance column, below); its **rate is a mandatory e2e
acceptance number**, and §9 also reports the total `shot_block` **row count** vs v1 so the third outcome (a
drop in rows) is visible — if fallback dominates or rows collapse, the item is near-vacuous and gets
reconsidered before ship.

**Long-form schema change (B8a, N8 — make it actually generic).** `_LONG_COLS` is 10 columns pinned by exact
equality (`test_defensive_credit_orchestration.py:40`). Add one generic column `resolution` recording **the
resolution mode that actually produced the row** — `∈ {"nearest", "all_within", "all_within_beyond_nearest",
"lane", "nearest_fallback"}`. **All ten rules resolve a defender** via `resolve_responsible_defenders` with a
`mode` (`_resolution.py:74-78`), so a `{lane, nearest_fallback, None}` column would be shot-block-specific with
a `None` hole on 9/10 rows; the mode is already known at emit time, costs nothing, and gives honest free
provenance for every rule (e.g. which `synchronized_final_third_pressure` rows came from
`all_within_beyond_nearest`). Update `_LONG_COLS` to 11 + repoint the exact-equality test; `resolution` gets a
real `RESOLUTION_VALUES` closed-set constant + test (the B5 pattern), not a two-value column with a null hole.

**Output effect.** `shot_block` may credit a **different player** than v1 → re-materialize note (v1 unadopted).
The lighter-than-`_cover_shadows` call is confirmed correct: `LaneControlResult` is corridor-aggregate with no
per-defender field (`_cover_shadows.py:83-102`), so direct point-to-segment geometry is both lighter and the
only path that yields per-defender attribution.

---

## 5. Item 3 — Line-break-gated through-ball (replace-in-place)

**What.** `rule_failed_marking_through_ball` fires on `ΔxT ≥ through_ball_delta_xt_min` (0.02, provisional /
never calibrated) — a threat proxy, not specifically a ball threaded *through* the line. Replace the ΔxT gate
with a genuine **line-break test** using the TF-32 ward straddle geometry (`"between_lines"` — the straddle of
two adjacent same-line defenders; `_line_breaking.py:288-333`; TF-4's `end_x > defensive_line_x` is the cruder
proxy).

**home_team_id-free — respect the family's P-2 principle (B3a).** The family **deliberately refuses
`home_team_id`** — the P-2 note lives in the docstring of `_aggregate_defensive_credit`
(`_orchestration.py:151-155`, the function that *calls* `compute_defensive_credits`; N12), and both it and
`compute_defensive_credits` omit the param — using `team_id != acting-team` + `acting_team_attacks_rtl` for
orientation. `detect_line_breaking` **requires**
`home_team_id` (`_line_breaking.py:65`). So Item 3 **does not call `detect_line_breaking`.**

**Mandate ONE straddle implementation — extract, do not re-write (N3).** The line-break output is already
consumed twice, one of them baked into trained models: `add_line_break(method="ward")`
(`features.py:1568-1621`) and **`line_breaking_ward_xfns`** (`features.py:2332-2384`, a second independent
`detect_line_breaking` call emitting 9 VAEP feature columns). A second local straddle implementation would
give "did this pass break the line?" two answers in one library. And extraction is clean: `home_team_id`
appears in `_line_breaking.py` at exactly one **functional** site — a coordinate flip (`:250-260`) that a
one-convention input collapses to a no-op (opponent selection is already done from `action_team` alone,
`:158-161,225`). So **extract a `home_team_id`-free straddle core** from `_line_breaking.py` and **re-point
TF-32's own tests + `detect_line_breaking` at it** (one implementation, and it removes the coordinate-flip
branch from TF-32 too). **No new required kwarg on the two public defensive-credit functions** (that would be a
breaking API change and violate P-2).

**Feed it action-LTR positions via the family's scalar-flip idiom (N4), NOT `reproject_to_action_ltr`.** That
function reflects **named columns** of a one-row-per-action frame with a `flip_mask` row-aligned to *that
frame's* index — it does not fit a tracking frame (many player rows per action). The family's own idiom is the
inline **scalar** flip on a single action's frame slice (`_resolution.py:53-54`:
`px = _FIELD_LENGTH - opp["x"].to_numpy() if flip else opp["x"].to_numpy()`), using the precomputed per-action
`flip` the orchestrator already threads. Use the family constants `_FIELD_LENGTH` / `_FIELD_WIDTH` (do not add
a fourth hard-coded `105`/`68` site).

**Precompute once + perf (B3b, H1).** The line-break signal is a **firing-condition input**, not rule logic:
compute it **once** in `_orchestration.py` (threading the family's existing `links=` so
`link_actions_to_frames` is called once — the perf gate `test_defensive_credit_perf_budget.py` asserts exactly
one link call per 100 actions, and one Ward clustering per action inside the rule loop would blow both) and
expose it as a precomputed boolean column on `RuleContext`. Rules stay geometry-free.

**Fire/no-fire mapping (B3c).** `detect_line_breaking`'s core returns four distinct states: computed `True`;
computed `False`; computed `0` via the `min_pass_length` / `min_opponents` / `min_x_spread` short-circuits
(`_line_breaking.py:213-248`); and `<NA>` for unlinked actions (`:183-185`). The firing rule is: **fire iff
the straddle is `True` AND `line_breaking_type == "between_lines"`; every other state (False, short-circuit 0,
`<NA>`) → no fire.** These four states collapse to one firing decision. **There is no row provenance for the
non-fire reasons** — a rule that doesn't fire emits **no row** (family convention, §8), so there is nowhere to
carry it (N9). Instead the distinction is observed **in aggregate on the e2e**: §9's table carries an
**Item-3 state distribution** line (counts of `True` / `False` / short-circuit-0 / `<NA>`) so "the gate got
stricter" is distinguishable from "the frames didn't link."

**Anchor unchanged.** The debit still anchors to the beaten marker (nearest defender to the pass origin, sized
by `xT(origin)`); only the **firing condition** changes.

**Retire `through_ball_delta_xt_min` (B4).** The field is removed from the `frozen=True`
`DefensiveCreditParams` (`_params.py:57`). Because the dataclass is frozen, a caller passing the removed kwarg
gets Python's **standard `TypeError`** *before* `__post_init__` runs — the rev-1 spec's "`__post_init__` note"
mechanism is impossible and is dropped. The bare `TypeError` is accepted as the honest, standard failure
(no `**_deprecated` catcher). Any test/fixture referencing the field is updated in the same commit.

**Column-name precision (MINOR).** The detector emits `line_breaking_type__ward` (not `line_breaking_type`);
the `"between_lines"` token means the straddle of two adjacent **same-line** defenders (not "received in the
space between two lines") — pin this meaning in a call-site comment so the next reader doesn't "fix" it.

**Output effect.** Fires on a **different set of passes** → re-materialize note.

---

## 6. Item 5 — Pressure-commitment cue (descriptive feature)

**What.** A per-action descriptor of whether the defender pressing the actor **commits** (drives in / does not
brake) versus **contains** (decelerates to jockey). Role **(A)**: a press style/quality descriptor, not signed
credit — it composes with the credit rules but is not itself a value. Home: `tracking/_press_commitment.py`
primitive + `add_press_commitment` aggregator + atomic mirror. **Not** in `defensive_credit/`.

**Why this atomic mirror is thin (N11 — unlike the split-out Item 4).** `press_commitment` needs no
`result_id`, no `possession_id`, no synthesized `end_*`, and no chaining — it reads the linked frame's velocities
and the actor's position — so the atomic mirror is a **rename-only bridge** (the `add_cover_shadows` shape,
`atomic/tracking/features.py:1190-1211`), not the `_packing_atomic_adapter` lookahead port that made Item 4 a
representation port. That is why Item 5's mirror rides along here while Item 4 was split.

**Pressing-defender resolution — lift to a shared core with the dependency INVERTED (B10, N6).** Do **not**
write a second nearest-opponent resolver, and do **not** import `resolve_responsible_defenders` from
`defensive_credit/` either: it takes a `params: DefensiveCreditParams` and derives its radius from the private
box-aware `params._proximity_threshold(...)` (`_resolution.py:21-32,57`), so importing it would make a generic
tracking primitive depend on a specific feature sub-package (backwards), drag `DefensiveCreditParams` into a
feature unrelated to defensive credit, and bind press detection to a box-aware threshold when this cue wants a
**flat `press_max_distance_m`**. Instead **lift the resolver to a shared home taking `threshold_m: float`**
(or a threshold callable), and make `defensive_credit/_resolution.py` a thin adapter that supplies
`params._proximity_threshold(...)`. One producer of "who is the nearest opponent," testable without the
feature's params. **Press distance gate:** if the nearest opponent is beyond `press_max_distance_m` there is no
press → `press_commitment = NaN`,
`press_commitment_source = "no_pressing_defender"`. Without it the column is never-NaN-when-it-should-be and
its distribution is polluted by non-presses.

**The metric — pinned (B9, N7).** Let `axis` = the unit defender→actor vector **fixed at the action frame**.
Closing-speed at a window frame = that frame's `(vx, vy) · axis`. **Axis-timing is a deliberate choice with its
own number (N7):** projecting the window's earlier velocities onto the *action-frame* axis measures closing
along the **final** approach direction (the coaching-relevant "did they commit into *this* press"), accepting
that a defender who has rotated during the window contributes its earlier speed along the later axis; the
alternative (per-frame axis) is a different number and is recorded in Open Questions. **Commitment = the
least-squares slope of closing-speed over the window frames** (m/s²; positive = committing, negative =
braking) — a **least-squares slope, NOT a two-point endpoint difference** (N7: a two-point diff is the noisiest
estimator meeting the ≥0.1 s rule and is wrecked by a single bad endpoint frame; the slope is a few lines more
and far more robust). The window is `W = commitment_window_seconds` (provisional **0.5 s**, Open Questions). It
is computed over positions/velocities on a **fixed ≥0.1 s baseline — NOT a per-frame second savgol derivative**
on the already-`deriv=1` `vx/vy` (`preprocess/_velocity.py:120`, which would multiply jitter by `1/dt` again;
[[feedback_highfps_kinematics_baseline_velocities]]: **strictly — no sub-baseline fallback, skip at segment
edges**). If the window is not spanned by enough frames meeting the baseline → `press_commitment = NaN`,
`press_commitment_source = "window_too_short"`.
- **Degenerate-axis guard (N7):** if defender and actor are within a `min_separation_m` floor the unit axis is
  ill-conditioned (and flips sign if the defender overruns the actor) → `press_commitment = NaN`,
  `press_commitment_source = "degenerate_axis"`.
- **Emitted columns:** `press_commitment` (float, m/s² — positive = committing / accelerating in; negative =
  containing / braking), `press_commitment_closing_speed` (float, m/s, context), `press_commitment_source`.
- **`press_commitment_source` is a structural-raise closed vocab (B10 minor),** matching `DAS_SOURCE_VALUES` +
  `DasUnscoreableError.__init__` (`_das.py:42-69`), not merely a test-side subset assert:
  `{"computed", "no_pressing_defender", "velocity_unavailable", "window_too_short", "degenerate_axis",
  "unlinked"}`.

**Velocity contract — share the helper, don't copy it (B/MINOR speed_source).** The feature requires `vx/vy`.
The "structurally unavailable vs caller-forgot" distinction lives in exactly one place —
`_das._velocity_unavailable_by_design` (`_das.py:249-265`). **Extract and share** it (every other velocity
consumer either raises unconditionally or silently fills zeros — do neither): `speed_source == "unavailable"`
on **all** rows → `press_commitment = NaN` + `"velocity_unavailable"`; a **partially**-marked frame set is the
caller bug → **raise loud** (preserve the helper's "ALL, not any" rule); `vx/vy` absent and not marked
unavailable → **raise loud** ([[feedback_loud_raise_for_required_input_columns]]).

**Orientation.** The defender→actor axis and the closing projection are a **relative** vector between two
players in one frame — direction-agnostic, no ADR-028 reprojection needed for the scalar. The aggregator links
per ADR-017 and resolves the actor via the acting-team row.

**Glossary + NOTICE (B11).** The three new columns must get `FEATURE_GLOSSARY` entries (with
`higher_is_better` direction) **and** the practitioner-concept `attribution` string must appear verbatim in
`NOTICE` **in the same commit** — `test_feature_glossary_notice_linkage.py` and the emitted-columns coverage
gate both auto-enumerate the new surface. (The glossary spec **has since merged as 4.59.0**, so this is a
straight add on `main`, not a branch collision.)

**Aggregator discipline.** `@nan_safe_enrichment` + pure (ADR-003/033), idempotent provenance-merge, accepts
`links=`, registered in all four auto-gates (purity / nan-safety / liveness / id-dtype) + the C4 aggregator-
liveness surface. Atomic mirror follows the shipped precedent. **No `press_commitment_xfns`** (T4 guard).

---

## 7. B2 — `recovery_after_pass` game/period-boundary fix (bundled)

`recovery_after_pass` (`_chaining.py:34-41`) is a bare positional slice
`actions.iloc[pass_idx+1 : pass_idx+1+max_actions]` with **no** `game_id` / `period_id` / `possession_id`
guard — unlike `resulting_shot_in_possession` right above it (`:21-25`), which filters all three. In a
multi-game batch, a failed pass within `recovery_max_actions` (default 3) rows of a game/period boundary can
"recover" into the **next game** (the `pd.isna(team_id)` guard doesn't help — foreign team ids read as a real
opponent regain). This is a **v1 latent defect**, but v2 is the cycle that claimed possession scoping as a
justification, so fix it here: scope the recovery scan to the passer's **`(game_id, period_id)` only**.

**Do NOT clamp to `possession_id` (N1 — the wrong-layer fix that would ship broken).** A recovery is
*definitionally a possession change*: `add_possessions` sets `boundary = team_change & ~set_piece_carve_out`
(`spadl/utils.py:905,913`) — a team change **is** a possession boundary, landing on the opponent's row, and
`with_possessions` calls `add_possessions` with **no overrides** (`_chaining.py:16`). `recovery_after_pass`
searches for an **opponent** action (`not same_id(r["team_id"], passer_team)`); clamping that scan to the
passer's possession makes the two conditions mutually exclusive (every row in the passer's possession shares
the passer's team), so the rule would return `None` **every time — silently**. (The sibling
`resulting_shot_in_possession` scopes by possession correctly only because it looks for a **same-team** shot
inside a single-team possession — copying its guard here is exactly wrong.) A `# why not possession-scoped`
code comment goes at the fix so the next reader doesn't "fix" it back.

**Red-first, with a two-game fixture** (the existing chaining test is single-game/single-period only) **plus a
non-vacuity assertion that `recovery_double_credit` still fires on a normal single-game fixture after the
change** — the failure mode is silent emptiness, so a boundary-only test would pass on a rule that fires never.
~3-line change; it also removes a false statement the rev-1 spec made about the family.

---

## 8. Cross-cutting

**Output / retrain summary.** No VAEP retrain from any item (no xfns). **Re-materialize notes (for whenever the
lakehouse adopts TF-51 — v1 not yet adopted):** item 2's `shot_block` attribution; item 3's through-ball firing
set; B2 removes some false cross-game recoveries. Items 1 (opt-in) and 5 (additive) don't change existing
outputs. **NaN discipline (ADR-043)** unchanged: fired-but-unsizable → long-form row `signed_value = NaN`; not
fired → no row; the `add_*` aggregate stays always-finite; item 5 uses its own `_source` provenance.

**Auto-gates.** Every new/edited aggregator + the new closed-vocab constants register in the meta-pinned gate
surfaces (purity / nan-safety / id-dtype / liveness + the C4 liveness surface + the xfns absence guard), so a
missing registration fails CI ([[feedback_auto_enumerating_gates_new_surface]]). **Gate-surface caveat:** only
`test_add_star_purity.py` iterates `atomic.tracking`; nan-safety / id-dtype / liveness are **tracking-only**, so
the thin `add_press_commitment` **atomic mirror is gate-covered by purity alone** — an honest limitation, not a
silent assumption (needs a `tracking:` **and** an `atomic.tracking:` purity entry).

**NOTICE.** Reverse-xT (arXiv:2606.19931), cover-shadows (Cascioli 2025), TF-32 ward already cited; the
commitment cue is a practitioner concept (PSG / Luis Enrique; Sumpter coaching literature) attributed as such
in NOTICE + docstring ([[feedback_academic_attribution_discipline]]).

**C4.** Re-derive the aggregator count at commit-prep per ADR-043 (do not copy a number); expected +1
(`add_press_commitment`) in `silly_kicks.tracking`.

---

## 9. Validation plan (red-first / TDD) — structure THEN numbers

The owner-gated GS e2e (`test_defensive_credit_e2e.py`, match 10502) is **plumbing/sanity smoke** — it proves
the code *runs*, not that attribution is *right* (B12). "Ran and exited 0" is not verification. So the e2e's
real deliverable is a **quantitative acceptance table**, each number deciding whether an item shipped:

| Item | Number the e2e must report | Fails if |
|---|---|---|
| 2 | `resolution` breakdown: `lane` / `nearest_fallback` counts **AND total `shot_block` row count vs v1** | fallback dominates, or rows collapse → near-vacuous (N10 third outcome) |
| 2 | % of `shot_block` rows whose credited player **changed** vs v1 | ≈0% → item did nothing |
| 2 | count of blocks credited to a GK | > 0 → B7 GK exclusion failed |
| 2 | **`is_goalkeeper` row count per team on 10502 (N5 coverage)** | 0 → the GK-credit=0 above is vacuous (flag all-False), not a working exclusion |
| 3 | passes firing in v1-not-v2 and v2-not-v1 | ≈0 both ways → gate is the old gate / did nothing |
| 3 | **line-break state distribution: `True` / `False` / short-circuit-0 / `<NA>` counts (N9)** | can't tell "gate got stricter" from "frames didn't link" |
| 5 | distribution of `press_commitment_source` | `computed` ≈100% → B10 distance gate missing |
| 5 | sign split of `press_commitment` on real data | all one sign → axis or derivative wrong |

**Unit / behavioral (red-first, mutate→RED):**
- **B2:** two-game fixture — a failed pass near a game boundary must **not** recover into the next game
  (constructed so v1 fires and the fix doesn't).
- **Item 1 (T1):** pin the **absolute** `xT` at the origin AND at the 180°-reflected point on an **asymmetric,
  extreme** fixture (deep regain x≈20 → reflects to x≈85 = high xT; last-ditch x≈100 → x≈5 ≈ 0) — a directional
  "moves the right way" assertion is insufficient because the reflection is trivially easy to invert
  ([[feedback_symmetry_test_insufficient_pin_ground_truth]]). Default `pressing_lens=False` → byte-identical to
  v1 (no exception — the rev-1 §9 "except the token rename" text was stale from a superseded draft and is
  removed). Closed-set exhaustive-when-on guard on `SIZING_VALUES`.
- **Item 2 (T5):** three-fixture discriminator — (a) defender near origin but off-lane vs (b) on-lane but
  farther from origin → v1 credits (a), v2 credits (b); (c) a **GK on the lane** → never credited; a defender
  10 m from origin but dead on the lane → credited (proves the threshold was dropped). Frame-absent →
  `nearest_fallback` provenance.
- **Item 3:** a `between_lines` through-ball fires; an `around_line` / same-ΔxT non-line-break pass that v1
  *would* have fired on does **not** fire (both directions; [[feedback_invariance_test_needs_discriminating_power]]);
  short-circuit-0 and `<NA>` both → no fire.
- **Item 5:** committed press (defender accelerating in) → positive; containing press (braking) → negative; no
  opponent within `press_max_distance_m` → `no_pressing_defender` NaN; window unspanned → `window_too_short`
  NaN; `speed_source="unavailable"` (all rows) → `velocity_unavailable` NaN; partially-marked or missing
  `vx/vy` → loud raise; high-fps baseline-velocity care.
- **Auto-gates + the xfns absence guard** auto-discover `add_press_commitment` / assert `press_commitment_xfns`
  absent (T4).

---

## 10. Deferred / out of scope

- **Item 4 — atomic-SPADL mirror — SPLIT to its own later spec** (owner, 2026-07-24). A representation port,
  not a refinement (§1 / review B1): needs the `_packing_atomic_adapter`-class lookahead bridge, a
  `preserve_native=[…]` caller contract + loud raise, per-representation window semantics (T3: atomic is
  denser, so `recovery_max_actions` means something different), and a **prerequisite** duplicate-`interception`-
  id fix (`atomic/spadl/config.py`) — that spec owns the fix. `compute_bravery`'s atomic mirror rides with it.
- **Opta block-detection — DROPPED from the roadmap entirely (owner, 2026-07-24), not deferred.**
  `shot_blocked`/`cross_blocked` stay **permanent `pd.NA`** for Opta (the terminal status ADR-046 gave
  SkillCorner). No public analysis-grade Opta data (Stats Perform fully paywalled, no StatsBomb-open
  equivalent); the project never had an Opta corpus (only socceraction 1–2-match parser fixtures, deleted with
  `data/opta/` in commit `0d30771`). **This spec's ADR supersedes the ADR-046/047 "Opta deferred" wording.**
- **Paper-2 DPA / role-responsibility model ("Track B")** — its own later spec; read the reference code first.
- **Passing-lane *blocking-credit* rule** + **individual `cross_block` rule** — Track B.
- **General `ax/ay` acceleration primitive (TF-50)** — item 5 computes closing-acceleration inline; the shared
  primitive is TF-50's scope.
- **`press_commitment_xfns`** — aggregator-only in v2; xfns is a later opt-in (T4 guards the deferral).

---

## 11. Open Questions (unresolved — flagged, not hidden)

- **Item 2 corridor value** — `shot_lane_cone_width_factor` starting value (proposal: match `_cover_shadows`'s
  `0.2`); confirm the cone convention transfers to shot lanes vs a bespoke value.
- **Item 2 GK exclusion** — settled (N5): use **both** `is_goalkeeper` exclusion **and** the distance-along-lane
  cap (defence-in-depth, since the GS flag can be all-False); the distance-cap threshold value is open.
- **Item 2 fallback rate** — unknown until the e2e runs; it is the item's acceptance criterion (§9).
- **Item 1 lens on `−` debit rows** — spec picks "reflect each row at its own point, uniformly (+ and −)";
  the alternative "+ rows only" is open.
- **Item 5 window length + params** — `commitment_window_seconds` (proposal 0.5 s), `press_max_distance_m`, and
  `min_separation_m` (N7 degenerate-axis floor) are intent-set / never calibrated; pin provisional values, mark
  them so.
- **Item 5 axis timing (N7)** — spec picks the **action-frame-fixed** axis (measures closing along the final
  approach direction); the per-frame-axis alternative is a different number and is open.
- **Long-form schema** — adding the generic `resolution` column is an 11th column on a pinned surface; confirm
  the exact-equality test repoint is acceptable vs returning resolution separately.

---

## 12. Rollback boundaries

Four items + one fix in one commit, two of which change shipped attribution. Per-item revert boundaries so a
single item can be backed out without unwinding the others:
- **B2 fix** — self-contained in `_chaining.py` + its two-game test.
- **Item 1** — `pressing_lens` param + the `sized_xt` call-sites + `SIZING_VALUES`/`ANCHOR_TYPE_VALUES`
  constants; opt-in, so reverting restores byte-identical behaviour.
- **Item 2** — the `_resolution.py` `lane_blocker` mode + `resolution` column + params; reverting restores
  `mode="nearest"` and the 10-column schema.
- **Item 3** — the `_orchestration.py` precomputed line-break signal + the `rule_failed_marking_through_ball`
  gate swap + the removed param; reverting restores the ΔxT gate.
- **Item 5** — the new `_press_commitment.py` + aggregator + glossary/NOTICE entries; fully additive, isolable.

---

## 13. Decisions log

| id | Decision | Choice |
|----|----------|--------|
| Scope | Which of TF-51 v2 | Track A refinements; Track B separate |
| Item 4 | Atomic mirror | **Split to its own representation-port spec** (owner, on review B1) |
| Opta | Block-detection | **Dropped from roadmap** (permanent `pd.NA`), not deferred |
| Bundle | This cycle | Items 1, 2, 3, 5 + the B2 fix, one PR |
| Order | Internal | **3 → 1 → 2 → 5** (Item 3 cleans Item 1's seam); B2 first |
| (a) | Line-break gate | **TF-32 ward `between_lines`**, computed home_team_id-free in action-LTR, precomputed on `RuleContext` |
| (b) | Shot-block blocker | **Direct point-to-segment lane geometry** in `_resolution.py`; GK excluded; origin-threshold dropped; distance-scaled cone corridor |
| (c) | Reverse-xT lens | **Single global opt-in param**, per-call-site gating, default byte-identical |
| (e) | Pressure-commitment | **(A)** descriptive feature; **least-squares slope** of closing-speed over the window (≥0.1 s baseline, not a two-point diff); shared resolver + `speed_source` helper; aggregator-only |
| Behaviour | Items 2 & 3 | **Replace in place** (re-materialize note; v1 unadopted) |
| B2 | recovery scan | **Scope to `(game_id, period_id)`**, red-first two-game fixture |
| Schema | long-form | **+1 generic `resolution` column** (11 cols), repoint the exact-equality test |
| Retrain | VAEP | **None** (no xfns; deferral test-enforced) |
