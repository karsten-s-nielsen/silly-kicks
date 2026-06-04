# ADR-018: Own goals are counted in VAEP labels by result, independent of action type

| Field | Value |
|---|---|
| **Date** | 2026-06-04 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.8 (1M); luxury-lakehouse session (two-round cross-session review) |

## Context

`silly_kicks/vaep/labels.py` detected goals **and own goals** with the same gate:
`actions["type_name"].str.contains("shot") & (result_id == ...)`. But every SPADL converter
(StatsBomb `statsbomb.py:508`, opta `_fix_owngoals`, sportec `sportec.py:861-863`) emits an own goal as
a **`bad_touch`** action with `result_id == owngoal` — which never matches a `"shot"` type gate. So
**no own goal, from any provider, ever registered** in `scores` / `concedes` / xG labels, even though the
label functions explicitly intend to handle own goals (their team-attribution branches reference
`owngoal`). The predicate was copy-pasted across **8** label functions (a missed copy is how the bug
hid). This surfaced while adding Gradient Sports own-goal (`RE`+`G`) and cross-goal (`CR`+`G`) capture
(spec 2026-06-04): the new own goals would have been captured into SPADL but remained invisible to VAEP.

## Decision

Own goals are detected by **result** (`result_id == owngoal`), with no action-type gate, via a
single-source `_is_owngoal(actions)` helper that all label functions call; goal detection uses a sibling
`_is_goal(actions)` (explicit `{shot, shot_penalty, shot_freekick}` name-set on `type_name`). Own goals
now count in `scores`/`concedes`/xG for **every** provider.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Type own goals as `shot`+`owngoal` per-provider so the existing `"shot"` gate matches | No label change | Pollutes shot/xG features with a defender "shooting" at their own net; diverges from the bad_touch precedent in all 3 converters | Wrong representation; doesn't fix the codebase-wide gap |
| B. Re-paste the corrected `result==owngoal` predicate into the 8 sites | Local | Repeats the exact copy-paste anti-pattern that hid the bug; a future missed site silently regresses | Maintainability |
| C. (chosen) Extract `_is_goal`/`_is_owngoal` helpers; detect own goals by result, no type gate | One definition, one place to change; fixes all providers; a guard test forbids the old pattern | Cross-cutting label-distribution shift (see Consequences) | — |

## Consequences

### Positive

- Own goals count in VAEP `scores`/`concedes`/xG for all providers (≈3–5% of goals previously invisible).
- The goal/own-goal predicates live in one place; a guard test (`test_no_shot_gated_owngoal_predicate_survives`) fails CI if the old shot-gated owngoal pattern reappears.
- The atomic-SPADL label path already detected own goals by result/dedicated type — now consistent with the regular path.

### Negative

- **Hyrum / behavior change:** `scores`/`concedes`/xG label distributions shift for every provider whose
  data contains own goals. Golden/e2e tests asserting these counts must be re-baselined (the shift per
  fixture must equal that fixture's own-goal count — a larger delta is a real regression). VAEP models
  trained on these labels would shift if retrained (not done in this change).

### Neutral

- Goal detection for normal shots is unchanged (`type_name.isin({shot,shot_penalty,shot_freekick})` is
  behavior-identical to the former `str.contains("shot")` for the current type vocabulary).
- Shipped alongside this change: an `is_synthetic` bool column on `GRADIENTSPORTS_SPADL_COLUMNS` marking
  converter-injected rows (the cross-goal synthetic shot + synthesized foul rows) that share their
  parent's `original_event_id`, so consumers can avoid collapsing them on a `original_event_id` dedup.

## Related

- **Specs:** `docs/superpowers/specs/2026-06-04-gs-goal-capture-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-04-gs-goal-capture.md`
- **External references:** own-goal encoding investigated empirically against the full PFF FC / Gradient Sports WC2022 catalog.

## Notes

Shipped alongside the Gradient Sports converter changes (own-goal `RE`+`G` capture, cross-goal `CR`+`G`
synthetic shot, `nonEvent` voided-event exclusion). ADR number provisional — reconcile against
`origin/main` at merge (no pre-reserved numbers).
