# GS input-convention shot distribution (item 23, step 2a)

Produced by `scripts/measure_gs_shot_distribution.py` over the **full owner-tier Gradient Sports
corpus (64 matches, uncapped)**.

## Why this exists

The GS converter warns `declared=per_period_absolute but detector inferred possession_perspective`
on the pining loader path, and **CI cannot see the case**: the committed fixture
`tests/datasets/gradientsports/synthetic_match.json` DEFERS, so `detect_input_convention` never
classifies and the disagreement never surfaces.

Reshaping that fixture requires knowing what real GS actually looks like. A fixture shaped to an
unrecorded number rebuilds the exact failure this cycle removes, so the number is measured here and
committed with provenance rather than assumed.

## What it found

| | |
|---|---|
| matches | 64, 0 failed |
| shots per (team, period) group | min 1, **median 5.0**, max 20 |
| reliable groups per match, `high` (>= 10 shots) | min 0, **median 0.0**, max 3 |
| reliable groups per match, `medium` (>= 5 shots) | min 0, **median 2.5**, max 6 |
| matches with >= 2 reliable groups at `high` | **6 of 64** |
| matches with >= 2 reliable groups at `medium` | **50 of 64** |

**The binding constraint is the two-reliable-groups clause, not per-group shot count.** The committed
fixture has 10 shots in one group -- AT the `high` threshold -- but only ONE team with shots at all,
so it defers on *fewer than 2 reliable groups* (`silly_kicks/spadl/orientation.py`). Raising
per-group counts would not have made CI see the case.

**The median real match has ZERO groups at `high`.** Only 6 of 64 clear two; 50 of 64 clear two at
`medium`. So a fixture built to trigger at `high` would exercise a tier real data almost never
reaches. **The reshape target is two groups of >= 5 shots.**

Independently corroborated during the same corpus pass: `validate_xcross_causal` logged the live
disagreement at `confidence=medium`, with the diagnostic *"every (match, team, period) group attacks
high-x"* -- classification rule 1. Two measurements from opposite directions agreeing.

## What travels, and what does not

Gradient Sports is owner-tier. This artifact carries **counts only** -- shots per group, reliability
tallies, thresholds. `team_id` is replaced by a per-match dense rank, so no real identifier travels
and no position is reconstructible. Enforced by a test
(`tests/scripts/test_measure_gs_shot_distribution.py::test_team_ids_are_dense_ranked_so_no_identifier_travels`).

`scope` in `metrics.json` records `tracking_limit: 1` -- which caps FRAMES per match, not actions.
Shot counts come from events and are unaffected by it; the field is recorded anyway, because "it does
not affect this measurement" is exactly the claim a reader must be able to check rather than take on
trust.
