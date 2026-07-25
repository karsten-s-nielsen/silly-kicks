# ADR-049: TF-51 v2 — defensive-credit refinements + pressure-commitment cue

| Field | Value |
|---|---|
| **Date** | 2026-07-25 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |
| **Supersedes / amends** | ADR-047 (TF-51 v1); ADR-046/047 **Opta** block-detection status |

## Context

TF-51 v1 (ADR-047, 4.57.0) shipped the per-event defensive credit/debit family with ten rules and a
per-team bravery rollup. Its §11 backlog carried a "Track A" of bounded refinements to what shipped,
plus one latent v1 bug (B2). This ADR ships that Track A — four refinements + the bug fix — in one PR.
Item 4 (the atomic-SPADL mirror of the credit family) was split to its own later spec, and Opta
block-detection was dropped from the roadmap entirely (no public analysis-grade Opta feed exists).

The design was locked across two spec-review rounds and two plan-review rounds, all verified against
code (spec `docs/superpowers/specs/2026-07-24-tf51-defensive-credit-v2-design.md`, plan
`docs/superpowers/plans/2026-07-24-tf51-defensive-credit-v2.md`).

## Decision

Ship five changes to `silly_kicks/tracking/defensive_credit/` (Items 1–3 + B2) plus one new descriptive
feature outside the family (Item 5):

- **B2 — `recovery_after_pass` game/period boundary.** Scope the forward opponent scan to the passer's
  own `(game_id, period_id)` before the search (NOT possession-scoped — a recovery *is* a possession
  change, which would make the opponent search vacuous and the rule fire never).
- **Item 3 — line-break-gated through-ball.** Replace `rule_failed_marking_through_ball`'s provisional
  `ΔxT ≥ through_ball_delta_xt_min` gate with a genuine TF-32 ward `between_lines` line-break, computed
  `home_team_id`-free (family P-2) in action-LTR. Mandate ONE straddle implementation: extract
  `_straddle_core` from `_line_breaking.py` (re-pointing `detect_line_breaking` + TF-32's tests at it)
  and precompute the per-action signal once on `RuleContext`, candidate-gated (successful passes only)
  so no Ward clustering runs for other actions. Remove `through_ball_delta_xt_min` (frozen dataclass →
  standard `TypeError` on the removed kwarg).
- **Item 2 — lane-geometry `shot_block` blocker.** In `_resolution.py`, add a `lane_blocker` mode:
  credit the in-corridor, in-front, non-GK defender with minimum perpendicular offset to the shot→goal
  lane (distance-scaled `_cover_shadows`-style cone, floored so it does not pinch to 0 at the shooter);
  no origin-proximity threshold; GK excluded by BOTH the `is_goalkeeper` flag AND a distance-along-lane
  cap (the GS flag can be all-False); nearest-to-origin fallback. Add a generic `resolution` column to
  the long-form (10→11) recording how each credited player was determined, with a `RESOLUTION_VALUES`
  closed vocabulary.
- **Item 1 — reverse-xT "position won" pressing lens (opt-in).** A single global
  `DefensiveCreditParams(pressing_lens=True)` sizes the four xT-sized turnover rules by
  `xT(105−x, 68−y)` at each row's own sized point, via a primitives-only `sized_xt` helper gated
  per-call-site (never at the shared `_xt_at` seam, which the through-ball's ex-gate also used). Default
  off → byte-identical. Turnover rows tag `sizing="xt_pressing"`. New `SIZING_VALUES` /
  `ANCHOR_TYPE_VALUES` closed vocabularies.
- **Item 5 — pressure-commitment cue.** A per-action descriptor (`tracking/_press_commitment.py` +
  `add_press_commitment` + atomic mirror) of whether the pressing defender COMMITS (drives in) vs
  CONTAINS (jockeys): the least-squares slope of the defender's closing-speed (projected onto the
  action-frame defender→actor axis) over a ≥0.1 s pre-action window, with a closed
  `press_commitment_source` provenance vocabulary. Role (A) — descriptive, NOT signed credit — so it
  lives outside `defensive_credit/` and ships aggregator-only (no `*_xfns`).

Shared plumbing: `tracking._opponent_resolution.opponents_within` is lifted as the ONE nearest-opponent
resolver (dependency inverted to take a `threshold_m: float`), consumed by defensive-credit resolution
(the box-aware threshold) and the press cue (a flat `press_max_distance_m`); `lane_blocker` bypasses it
by design. `velocity_unavailable_by_design` is extracted from `_das.py` into
`tracking/_velocity_availability.py` and shared with the press cue's velocity contract.

## Consequences

- **No VAEP retrain from any item** — no new `*_xfns`, and the credit family stays out of every default
  xfn list (F4 result-leakage; test-enforced). Items 1 (opt-in) and 5 (additive) change no existing
  output.
- **Re-materialize notes** (for whenever the lakehouse adopts TF-51 — v1 not yet adopted): Item 2's
  `shot_block` attribution; Item 3's through-ball firing set; B2 removes some false cross-game recoveries.
- **C4** action-coupled aggregator count 31→32 (`add_press_commitment`); its atomic mirror is API
  symmetry (documentation count), not new capability.
- **Opta** `shot_blocked`/`cross_blocked` are terminal `pd.NA` (this ADR supersedes the ADR-046/047
  "deferred" wording); do not re-add Opta block-detection unless a real Opta feed is licensed.
- **`_xt_at` removed** (execution note): the plan directed leaving it "intact as the raw lookup", but
  after Item 1's per-call-site swap it retained zero callers; per the repo's no-dead-code discipline it
  was removed and its role subsumed by the lens-aware `_sized_xt` (default byte-identical).

## Attribution

`xT(origin)` turnover sizing + the reverse-xT lens: Bischofberger/Bauer/Baca (arXiv:2606.19931). The
pressure-commitment cue is a practitioner concept (PSG / Luis Enrique pressing style; Sumpter coaching
literature), attributed as such in NOTICE — no numeric result reproduced.
