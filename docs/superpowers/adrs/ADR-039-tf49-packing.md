# ADR-039: TF-49 packing — Impect-faithful surface over the existing LBS kernel

| Field | Value |
|---|---|
| **Date** | 2026-07-17 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner), Claude (PR-S117) |

## Context

The Soccermatics Pro + Modern Soccer Coach course ingestion (2026-07-16 plan, §W5) named
packing — Impect's opponents-bypassed count — as the highest-value course-derived candidate
(TF-49). The repo already ships the identical bypass inequality as TF-45's `structural_lbs`
(`start_x < d_x <= end_x`, Goes et al. 2019), but structural-pass is a frozen research kernel
with no completion gate, no set-piece/dribble domain, no goal-threat restriction, no receiver
identity, and no reception-quality label. Practitioner packing (MSC lesson; Twelve course)
additionally wants a back-line-restricted "goal threat" count, a direction-signed "net" variant
(`football-packing` multipliers), and a "secured reception" rule (the bounce-pass caveat).

Verified probe grounding (2026-07-16, read-only Databricks + repo): next-same-team-touch
receiver resolution is structurally sound at 90.8–99.8% across all six providers under the
shipping non_action-skip rule; GS dribbles were 100% zero-displacement (fixed GS-locally as
PR-S116/4.49.0 — this PR builds on corrected geometry); the 45°/135° bands are provider-stable.
Spec: `docs/superpowers/specs/2026-07-16-tf49-packing-design.md` (2×2 cross-session review
rounds; embedded reference code execution-verified).

## Decision

Ship packing as a NEW frozen-param surface (`tracking/_packing.py` + `_kernels._packing_at_actions`
+ `add_packing`/`packing_xfns` + atomic mirror) that duplicates the ~15-line defender-extraction/
mirror block from `_structural_pass.py` rather than refactoring it, resolves receivers through a
new PUBLIC packing-agnostic `spadl.utils.resolve_next_touch_receiver`, and labels secured
receptions on the `retains()` skeleton with a REQUIRED foul-skip. In no default xfn list — no
retrain trigger. C4 action-coupled-aggregator count 28 → 29.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Extend `_structural_pass.py` with the packing rules | zero duplication | mutates a frozen research kernel (TF-45 golden gates, arXiv attribution surface); Chesterton + Hyrum | frozen-kernel isolation is the cheaper risk |
| B. Consumer-side packing in the lakehouse | no library change | duplicates geometry three ways (the TF-23 lesson); no cross-provider gates | library-owned per the delete-and-depend doctrine |
| C. New `_packing.py` + duplicated extraction block (chosen) | frozen kernel untouched; golden identity gate pins the copies byte-equivalent | ~15 duplicated lines | — |

## Decisions recorded (spec §8 list + execution-time findings)

1. **Canon vs variants.** `packing_made` is the canon Impect count (completion-gated, forward-only,
   outfield-only by default). `packing_net` (direction multipliers +1/+0.5/−1) and
   `packing_goal_threat` (back-`n` restriction via `select_back_line_players`) are parameterized
   variants, not the canon. There is NO subtraction rule (a backward completed pass scores
   negative only in the NET view — the canon count has no negative points; probe finding).
   `eligibility="tti"` (MSC's "recovering defender is not packed") is DEFERRED — needs TTI wiring
   + its own real-data validation.
2. **Secured design = `retains()` skeleton + foul-skip (corrected rationale).** The round-1
   "SPADL has no possession column" premise was FALSE — `add_possessions` exists and self-heals
   (`_retention_labels.py` does exactly this). The skeleton is possession-aware; the foul-skip is
   load-bearing ON TOP because the heuristic emits a possession boundary AT the foul row
   (verified 2026-07-16: A-pass P0 → B-foul P1 → A-freekick P1 via the rule-4 carve-out) — a bare
   boundary rule would flip loss at every foul won. Coupling secured to the `add_possessions`
   heuristic is ACCEPTED for ρ-label-family consistency.
3. **Reception-is-shot decides True (execution-time clarification, spec §3 amended).** In the
   literal pass → shot → keeper_save shape the shot IS the next same-team touch; the scan starts
   after the reception, so without this rule the save's possession boundary would read the
   first-time shot as a loss — exactly the misread finding 1 forbids. The reception row's
   `start_x` is still never tested (blocker 4). Mutation-probed: emptying the shot set flips the
   keystone fixture.
4. **`packing_xfns` result-leakage warning (F4) — docstring warning + EXECUTABLE guard + recorded
   fork.** Every packing column gates on the action's OWN `result_id`; as an a0-slot feature that
   is the exact leakage class HybridVAEP strips. The factory docstring carries the warning, AND —
   for consistency with the two precedents (`shot_goalmouth`'s `test_no_shot_goalmouth_in_any_
   default_xfn_list`, `xt_xfns`'s `test_not_in_any_default_list`) — an executable guard
   (`tests/tracking/test_packing_xfns_leakage_guard.py`) auto-discovers every default/union xfn
   list across the tracking + atomic + VAEP surfaces and asserts no `packing`-named transformer
   enters one, with a discriminating-power anchor (the transformer name is pinned so a rename
   fails loudly instead of neutering the guard) and a mutation-verified red path. Added during
   the /final-review consistency pass — the F4 decision previously rested on docs alone, unlike
   both precedents. A result-free-a0 variant remains a RECORDED FORK, not built (YAGNI until a
   consumer asks; that is where a caller who genuinely needs packing in a default list goes).
   `packing_xfns` REJECTS `require_secured=True` (`ValueError`) — receiver/secured resolution
   needs true next-row relationships, which shifted gamestate slots do not have.
5. **Receiver seam placement.** `resolve_next_touch_receiver` is PUBLIC in `spadl/utils.py`
   (event-only, frames-free, TF-35-reusable; `add_pre_shot_gk_context` precedent), packing-agnostic
   (resolves every action type; the dribble mask lives in `add_packing`'s assembly), skipping
   `non_action` AND `foul` rows — neither is a ball touch (GS emits non-touch rows; the foul-skip
   is execution-review fix D1: the fouler must never resolve as receiver, and an advantage-played
   opponent foul must not block resolution — live-reproduced fouler-as-receiver + mis-anchored
   secured before the fix). Fully positional implementation (ADR-019 /
   PR-S110 `ids_equal`-positional-world lesson). The private
   `spadl.utils._resolve_next_touch_positions` is a **STABLE INTERNAL SEAM** consumed by
   `tracking._packing.secured_reception` (reception-anchored window) — the public export defaults
   `receiver_pos=None` and computes internally, so external callers never need the private helper.
6. **F5 dtype rule, extended at execution time.** Int64/object id sources pass through; plain
   int64 AND float64 (NaN-coded-int convention — h5 fixtures, `defending_gk_player_id`-style
   columns) pre-convert to Int64; the receiver column NEVER float64-upcasts. The float64 branch
   was added when the ADR-028-style mirror fixture (float ids) crashed the seam — a real caller
   shape the int64/Int64/object trio missed.
7. **Duplication trigger.** The defender-extraction/mirror block now exists ×2
   (`_structural_pass.py`, `_packing.py`; packing mirrors x ONLY — its counts are 1-D interval
   tests, the direction angle lives in action coords). A THIRD consumer (TF-35 is the named
   candidate) triggers extraction of a shared `_eligible_defenders_action_ltr` with a
   byte-identity gate on structural_pass. Until then the golden identity gate
   (`test_packing_golden_identity.py`, non-vacuity meta-asserted) pins the copies.
8. **Degenerate geometry scoping.** `start == end` → NaN for DRIBBLES ONLY (placeholder-
   indistinguishable: pre-PR-S116 corpora, post-fix period-last carries); pass-class `start == end`
   is recorded data → honest geometric 0. Both branches unit-tested.
9. **Atomic mirror synthesizes a type-aware result (SK-xT-2 precedent).** Atomic SPADL has no
   `result_id`; a bare delegate would silently emit all-NaN. The adapter maps domain types by NAME
   (never raw id passthrough) and synthesizes success = dribble-intrinsic OR next-atom-receival;
   non-domain atoms map to std `non_action`. The atomic mirror emits the three numeric columns
   only (receival atoms already carry receiver identity) and REJECTS `require_secured=True` —
   gating counts on a dropped, atom-stream secured label would be a silent semantic trap.

## Consequences

### Positive

- Coach-facing canon metric (packing) on every tracking provider, gated by golden-identity,
  mirror-invariance, liveness, purity, id-dtype, dup-action-id, NaN-safety and perf-spy CI.
- The receiver + secured seams are the shared substrate TF-35 (off-ball-run valuation) needs.

### Negative

- ~15 duplicated defender-extraction lines until a third consumer forces extraction.
- Secured inherits `add_possessions`' precision profile (~0.44) at possession boundaries — the
  foul-skip removes the worst bias class; the rest is accepted heuristic noise.

### Neutral

- Opt-in surface: no VAEP retrain, no default-list change; lakehouse adoption notes in spec §9
  (receiver materialization needs `COALESCE(player_id, player_id_native)` or gold `player_key`).

## Execution-review fixes (2026-07-17 — 12-agent adversarial review, all six live-reproduced)

A refute-prompted multi-agent review of the finished diff confirmed six defects, all fixed
in-PR (spec §9.5 carries the same list): **D1** receiver seam treated fouls as touches (fixed:
foul-skip in `_resolve_next_touch_positions`); **D2** the atomic adapter never matched
`convert_to_atomic`'s collapsed `corner`/`freekick` atoms — 4 of 9 default domain types silently
dead on the atomic path, caller-unfixable (fixed: collapsed-atom bridging; shot-shaped freekicks
stay out via the result synthesis); **D3** atomic `add_packing` returned the ADAPTED frame —
rewritten `type_id` + phantom `result_id` leaked to the caller (fixed: output assembled on a
copy of the caller's frame); **D4** `secured_reception` scanned positionally while anchors
resolve in `action_id` order — time-tied swapped rows flipped labels (fixed: scan in `action_id`
order, rank-map anchor lookup); **D5** the atomic result synthesis marked completed same-team
keeper receptions (`keeper_pick_up`/`keeper_claim`) as fail — atomic never inserts `receival`
before keeper collections (fixed: same-team keeper-reception next-atoms count as received);
**D6** the secured possession-boundary test was a raw `!=` — a caller-supplied `possession_id`
with NA decided false losses, the ADR-027 class (fixed: both sides must be attested).

## Relay-back observations (recorded, OUT of scope)

1. **`retains()` shares the foul-row / raw-`!=` bias classes D1+D6 fixed in packing's secured
   seam — but a 2026-07-17 read-only probe over the LIVE ρ training cohorts measured ZERO label
   flips (0/3451 GS rows, 0/5483 SkillCorner rows; probe replica parity-asserted against the
   library `retains()`).** The mechanism is structurally dead on the mart path: the gold
   `fct_action_values.possession_id` (unlike silly-kicks' `add_possessions`) keeps the
   possession id continuous through foul rows (2115/2122 GS, 1228/1260 SC fouls share their
   predecessor's id), foul-row teams are fully resolved (0 NaN), and possession ids carry 0
   NAs — so of the 135 (GS) / 136 (SC) anchor windows that DO reach a foul row first, ZERO
   satisfy the diff-team ∧ diff-poss precondition. **Consequence: the bundled ρ weights are NOT
   trained on biased labels; no retrain urgency.** The bias IS live only for a hypothetical
   caller feeding `retains()` an `add_possessions`-healed stream (the self-heal path, currently
   unused by any production consumer) — so the hardening was APPLIED IN-PR at the owner's
   direction, red-first, with a POST-FIX gate re-verifying the shipped function == the probed
   variant on all 223,718 cohort rows and ZERO training-label changes vs pre-fix (full-stream
   diffs confined outside the training mask; e.g. 7 GS rows at the non-continuous foul
   possessions). Weights + recorded metrics untouched — no retrain. Full record: ADR-036
   amendment (2026-07-17). The course-audit TODO item keeps only the course-rules diff.
2. **GS dribbles were structurally off-domain for packing (e2e finding, 2026-07-17) — FIXED
   IN-PR at the owner's direction.** The GS result dispatch had NO success condition for
   `OTB`+`BC` carries — every GS dribble fell to the `fail` default (statsbomb: 100% success).
   Initially recorded as out-of-scope; the owner lifted the scope constraint mid-review, so the
   fix shipped in PR-S117 after its own probe (native `ballCarryOutcome` {R, L} on 100% of BC
   rows; R→success, L→fail; L ~86% opponent-next cross-check). Full record: **ADR-018 amendment
   (2026-07-17)**. GS-only retrain trigger folding into the 4.49.0-queued re-fit. The packing
   e2e now GATES the in-domain dribble share strictly interior.

## Related

- **Specs:** `docs/superpowers/specs/2026-07-16-tf49-packing-design.md`,
  `docs/superpowers/plans/2026-07-16-tf49-packing.md`
- **ADRs:** builds on ADR-018 amendment (PR-S116 GS dribble ends, 4.49.0); ADR-019 (id dtypes);
  ADR-028 (action-LTR); ADR-033 (purity); ADR-036 (retains()/ρ family)
- **External references:** Reinartz & Hegeler (Impect packing); Goes et al. 2019
  doi:10.1089/big.2018.0067; `football-packing` (S. K. Varadharajan); MSC "Packing Data" lesson;
  Twelve/Soccermatics Pro course. See NOTICE.
