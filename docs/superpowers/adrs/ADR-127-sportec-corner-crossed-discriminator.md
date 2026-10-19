# ADR-127: Sportec/DFL corner-crossed discriminator via `play_height`

| Field | Value |
|---|---|
| **Date** | 2026-10-17 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

`silly_kicks.spadl.sportec.convert_to_actions` classified a DFL `CornerKick` as `corner_crossed` when
`corner_target_area ∈ {box, penaltyBox, sixYardBox}` OR `corner_placing ∈ {aerial, high}`, else
`corner_short`. **Those token values appear in no real DFL data.** They originated in a self-referential
unit fixture (`tests/spadl/test_sportec.py::_df_corner_crossed` sets `corner_target_area=["box"]`, and the
converter checks for `"box"`), so the vocabulary was never grounded in a real feed.

Measured on the pining IDSSE corpus (all 7 matches, 57 `CornerKick` events): the real
`corner_target_area` vocabulary is `{center, nearPost, farPost, short, outOfPlay, centerOutsideBox}` and
`play_height ∈ {high, flat}`. Under the old rule **0 of 57** corners were typed `corner_crossed` — every
DFL corner was mistyped `corner_short`. Downstream (TF-46 corner-offence) this made IDSSE contribute
**zero** direct corners; more broadly any sportec consumer distinguishing crossed vs short saw a corpus
with no crossed corners.

The real discriminator is the DFL `play_height` Play qualifier, which on `CornerKick` events takes
`{high, flat}` (distinct from the `{cross, head}` vocabulary it carries on open-play passes). Measured
partition: `play_height=="high"` ⟺ `corner_target_area ∈ {center, nearPost, farPost, centerOutsideBox,
outOfPlay}` (53 crossed); `play_height=="flat"` ⟺ `corner_target_area=="short"` (4 short) — a clean split.

## Decision

Type a DFL corner `corner_crossed` when `play_height=="high"` (the aerial-delivery qualifier), else
`corner_short`; retain the prior `{box, penaltyBox, sixYardBox}` / `{aerial, high}` tokens in the union
for backward-compatibility with any pre-normalized bronze input. This change lands as **its own base-code
commit** (separate from the TF-46 corners feature commit) with **its own independent A/B review**.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep the existing `{box,penaltyBox,sixYardBox}`/`{aerial,high}` vocabulary | no code change | matches no real DFL token → 0/57 crossed, every DFL corner mistyped | leaves the measured defect in place |
| B. `corner_target_area` NOT in `{short, outOfPlay}` ⇒ crossed | uses the target field directly | two-field edge cases (`centerOutsideBox`, `intoTouch`, `backcourt`); less direct than the aerial signal | `play_height` is the single direct aerial discriminator and partitions cleanly |
| C. `play_height=="high"` ∪ legacy tokens (chosen) | single direct DFL signal; clean measured partition; backward-compatible | widens the OR set | — |

## Consequences

### Positive
- DFL/IDSSE corners are correctly typed; IDSSE contributes direct corners to TF-46 (was zero).
- Any sportec consumer distinguishing crossed vs short now sees correct DFL corner types.
- `test_sportec.py` gains real-DFL-token cases (`play_height=="high"`→crossed, `flat`+`short`→short).

### Negative
- **Value-changing** on `spadl.sportec` output: ~93% of DFL corners flip `corner_short`→`corner_crossed`.
  It is therefore NOT an additive change (SemVer **minor**, number pinned at commit-prep).
- Measured consumer blast radius (sportec/DFL data only — SkillCorner and StatsBomb type corners
  natively and are unaffected):
  - **No consumer breaks.** Consumers that group both corner types as one set are unaffected:
    `vaep/formula.py`, `coordination/_windows`, `tracking/_elastic_sync`, `tracking/_ghost_gk`,
    `tracking/_packing`, `team_metrics/_possession`, and `atomic/_simplify` (both → atomic `corner`).
  - Consumers that distinguish crossed vs short shift toward correctness, no retrain required (feature
    schemas unchanged): `tracking/defensive_credit/_bravery` (`_SET_PIECE_CROSSES` now includes the DFL
    corners that are genuinely crosses) and the `type_corner_crossed`/`type_corner_short` one-hot features
    used by xSuccess / VAEP (the DFL-corner feature bucket moves; predictions on DFL corners drift).
- **Re-materialization (owner-ruled out of scope 2026-10-18).** xSuccess and VAEP are TRAINED models, so
  their DFL/IDSSE-corner predictions drift toward correctness under this fix (no retrain — feature schemas
  are unchanged). Any already-materialized IDSSE/DFL mart carrying those predictions is therefore stale and
  WOULD need re-materialization to propagate the corrected corner types. The owner has ruled the lakehouse
  **not a concern for this cycle** (2026-10-18): no re-materialization is performed here. Live
  (non-materialized) consumers pick up the corrected types immediately; the deferral is explicit and
  recorded, not silent.

### Neutral
- Legacy `{box, penaltyBox, sixYardBox}` / `{aerial, high}` tokens remain accepted (union), so any
  pre-normalized bronze that used them keeps mapping to `corner_crossed`.

## Related

- **Specs:** `docs/superpowers/specs/2026-10-16-tf46-corner-offence-design.md` (§C amendment, measured numbers)
- **Issues / PRs:** `#287` (TF-46 branch; C lands as its own commit on it)
- **Tests:** `tests/spadl/test_sportec.py` (`test_corner_real_dfl_high_maps_to_corner_crossed`,
  `test_corner_real_dfl_flat_short_maps_to_corner_short`)

## Notes

Measured crosstab, pining IDSSE corpus (7 matches, 57 CornerKick events), `play_height` × `corner_target_area`:

```
('flat', 'short'): 4
('high', 'center'): 15
('high', 'centerOutsideBox'): 2
('high', 'farPost'): 9
('high', 'nearPost'): 26
('high', 'outOfPlay'): 1
```
