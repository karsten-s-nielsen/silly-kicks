# ADR-096: TF-51 Item 4 -- defensive-credit atomic-SPADL mirror

| Field | Value |
|---|---|
| **Date** | 2026-09-16 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

TF-51 v1 (ADR-047) shipped the defensive-credit family -- `compute_defensive_credits` (long-form),
`add_defensive_credit` (per-action defending-team aggregate) and `compute_bravery` (event-only) -- for
**standard SPADL** only, deferring the atomic-SPADL mirror. TF-51 v2 (spec
`2026-07-24-tf51-defensive-credit-v2-design.md`) split Item 4 (the mirror) to its own later spec because
it is a genuine representation port, not a rename: the credit rules read std `type_id` + `result_id` +
`start_x/y` + `end_x/y`, and `convert_to_atomic` strips `result_id` and the derived `possession_id`.
This is that work (spec `2026-09-15-tf51-item4-defensive-credit-atomic-mirror-design.md`).

While mapping the mirror, a real repo bug surfaced: `atomic/spadl/config.py::actiontypes` re-appended
`"interception"` (already inherited from std at idx 10), so the reverse dict resolved
`actiontype_id["interception"] == 24`, silently shadowing the std interception events that
`convert_to_atomic` keeps unremapped.

## Decision

Ship the atomic mirror in `silly_kicks/atomic/tracking/features.py` -- `compute_defensive_credits`,
`add_defensive_credit`, `compute_bravery` -- **faithful** to the atom stream (delegate on the atom
stream, no std-parity), matching the shipped `_packing_atomic_adapter` / `add_off_ball_run_values`
precedents.

- **`_defensive_credit_atomic_adapter`** re-lifts the atom stream: endpoints from `x,y,dx,dy`; std
  `type_id` for the six rule-anchor types (`pass`/`cross`/`shot`/`shot_penalty`/`take_on`/`bad_touch`,
  else `non_action`); a **per-type next-atom `result_id`** (pass/cross succeed on a following
  `receival` or same-team keeper reception; take_on on a retained same-team next atom that is not
  `interception`/`out`; shot/shot_penalty on a following `goal`; else fail). `possession_id` is not
  synthesized -- `with_possessions` derives it downstream from time/team, not `result_id`.
- **The shared rollup `_rollup_defending_aggregate`** is extracted into std `_orchestration.py` and
  called by both representations, each assembling on its OWN frame (so the adapter's synthesized
  columns never leak). This is the ONE shipped-std-production-code touch; it is parity-gated so std
  output stays byte-identical (owner-approved scope, spec §2 item 9).
- **`preserve_native` required-column raise** at the mirror edge: `xg` + `shot_blocked` (credits),
  `shot_blocked` + `cross_blocked` (bravery) must be threaded through the conversion, else a loud
  `ValueError` naming the missing column and the remedy (ADR-043 -- never an all-NaN credit column).
- **Bravery honest-NaN set-piece**: `_simplify` collapses `corner_crossed`/`freekick_crossed` into
  `corner`/`freekick`, so `bravery_set_piece_crosses` = NaN and `n_set_piece_crosses_faced` = `<NA>`
  (ADR-027 -- never a conflated crossed+short overcount); the shots + open-play-cross headline is exact.
- **Co-shipped fix**: remove the duplicate `"interception"` from `atomic/spadl/config.py`. It unifies
  at idx 10 and renumbers the tail (`out` 25->24 ... `freekick` 32->31); all code consumers resolve ids
  symbolically and auto-follow.

Ships **no `*_xfns`** (F4 result-leakage, inherited from v1); in no default xfn list.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Std-parity (reproduce v1 numbers on atomic) | Not achievable across the full type domain without de-atomizing; no atomic mirror promises it. Owner chose faithful (2026-09-15). |
| Reuse std `_aggregate_defensive_credit` verbatim on atomic | Infeasible -- it re-computes the long internally and copies `actions`, so it would re-run the std compute on the atom stream and leak synth columns. Hence the rollup extraction. |
| Preserve `result_id` via `preserve_native` instead of synthesizing | Rejected -- the atom stream re-expresses results as atoms; mixing preserved std results with atom-read results is inconsistent. |
| Atomic-local rollup duplicate (leave std untouched) | Rejected (owner, 2026-09-16, option a): a second copy can drift; single-truth discipline is repo-wide. |
| Force a `preserve_native` set-piece discriminator for bravery | Rejected -- honest-NaN is the faithful answer for an unobservable quantity (ADR-027). |

## Consequences

### Positive

- Atomic-SPADL consumers get the full defensive-credit surface; the shared rollup means std and atomic
  aggregates cannot drift.
- The co-shipped config fix closes a silent atomic-interception-shadowing bug.
- Additive to VAEP for the credit family -- no `*_xfns`, no defensive-credit feature change, no
  retrain of the credit family.

### Negative (faithful-representation limitations, documented)

- Set-piece crossed/short are indistinguishable on atomic -> bravery set-piece columns are NaN.
- A `shot_freekick` collapses to `freekick` -> not detected as a resulting shot (`{shot, shot_penalty}`
  only on atomic). Rare.
- Window params (`resulting_shot_max_actions` / `recovery_max_actions`) count atom rows (denser).

### Migration (breaking)

The atomic `type_id` tail renumber changes the **default** `AtomicVAEP` feature encoding
(`xfns_default` = `fs.actiontype` raw `type_id` + `fs.actiontype_onehot` position-enumerated). Any
persisted/champion AtomicVAEP consuming the default xfns must be **retrained** on the new encoding (or
proven not deployed) -- paired with an atomic-SPADL re-materialize; a re-materialize WITHOUT the retrain
is a silent old-id-weights-vs-new-id-features skew (no gate). silly-kicks ships no bundled AtomicVAEP
weights (the model is fit by the consumer), so this obligation lands on the lakehouse.

### Neutral

- The C4 `silly_kicks.atomic` container prose "33-type" -> "32-type" (no aggregator count on that
  container; the tracking "33" is unchanged). html regenerated via Graphviz `dot`.
- Deferred to Track B (its own spec): the DPA / role-responsibility model (arXiv:2606.19931).

## References

- Spec: `docs/superpowers/specs/2026-09-15-tf51-item4-defensive-credit-atomic-mirror-design.md`
- Plan: `docs/superpowers/plans/2026-09-16-tf51-item4-defensive-credit-atomic-mirror.md`
- ADR-047 (TF-51 v1), spec `2026-07-24-tf51-defensive-credit-v2-design.md` (owner split).
- ADR-027 (honest-NaN), ADR-033 (add_* purity + conditional-column variants), ADR-039/042 (F4 xfn
  result-leakage), ADR-043 (missing != 0; fail-loud), ADR-077 (FOV `visible_area` companions),
  ADR-019 (id-compat), ADR-005 (attribution -- unchanged).
- `NOTICE` -- Sumpter, Soccermatics Pro module 16.3; Bischofberger/Bauer/Baca, arXiv:2606.19931
  (unchanged; no new methodology).
