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

## Amendment (2026-07-16, silly-kicks 4.49.0, PR-S116): GS dribbles derive real end coordinates

**Found** during TF-49 packing spec probes (owner prompt "could this be an existing bug?"): every
GS dribble shipped `end == start` — verified 850/850 zero-displacement (0 m) on the live corpus,
while IDSSE/metrica/skillcorner/wyscout dribbles carry real geometry (median 6.2–13.7 m) and
statsbomb is 89% distinct (11% = genuine stationary carries).

**Root cause chain (all three legs required):** (1) the GS converter maps `OTB`+`BC` ball-carries
to SPADL `dribble`; (2) it initializes `end = start = ball_x/y` for EVERY event and only
`_derive_end_coordinates` writes real ends — whose shared `_DERIVE_END_TYPE_IDS` excludes
`dribble`; (3) GS is the only event converter that never calls `_add_dribbles` (whose synthesized
dribbles get `start = prev.end`, `end = next.start`).

**Decision — GS-local opt-in, shared set untouched:** `_derive_end_coordinates` gains a
keyword-only `extra_type_ids: frozenset[int] = frozenset()`; ONLY `gradientsports.py` passes
`{dribble}`. A global set change was rejected (cross-session review): the `placeholder_end` guard
cannot distinguish statsbomb's genuine stationary carries from placeholders, so all eight
converter paths would silently rewrite recorded data. Default-path byte-identity is
regression-locked; period-last carries honestly keep the placeholder (no successor to derive
from). Owner-gated e2e asserts >90% of real WC2022 dribbles derive an end.

**Consequences:** GS-only retrain trigger — xT/xtgk move-sets include dribbles (GS previously
contributed zero-displacement transitions to GS-fitted transition matrices) and VAEP features
consume dribble ends; GS-fitted artifacts re-fit on next touch. Zero delta for the other seven
providers. Lakehouse re-materializes GS-derived marts on adoption. Precursor to TF-49 packing
(PR-S117): the dribble-packing channel needs real GS carry geometry, and TF-49's
degenerate-geometry NaN policy remains as the residual guard (period-last carries).

## Amendment (2026-07-17, silly-kicks 4.50.0, PR-S117): GS ball-carry results from `ballCarryOutcome`

**Found** by the TF-49 packing owner-gated e2e (0/12 dribbles in the packing domain on match
10503): the GS result dispatch (`_derive_type_result` `result_conds`) had NO success condition
for `OTB`+`BC` carries — every GS dribble fell through to the `fail` default, structurally
excluding GS carries from every completion-gated consumer (packing's completion gate, xT/xtgk
success-filtered move-sets, VAEP result features). StatsBomb dribbles are 100% success — GS was
the outlier. The 4.49.0 amendment fixed dribble GEOMETRY; this fixes dribble RESULTS — the two
legs of the same "GS dribbles are invisible" defect.

**Decision — map the native `ballCarryOutcome` (owner-directed in-PR fix, PR-S117):** the field
was already flattened into `EXPECTED_INPUT_COLUMNS` (`ball_carry_outcome` ← `ballCarryOutcome`)
but never consulted. Live WC2022 vocabulary probed 2026-07-17 (4 matches, 66 BC rows, field
present on 100%): **R (retained) → `success`, L (lost) → `fail`**; unknown/absent tokens keep the
`fail` default — this converter's exact-token allowlist style (pass/cross `"C"`, shot `"G"`).
Cross-checked empirically on the converted stream: R carries resolve a same-team next touch
43/43 (100%); L carries are 19/22 opponent-next (the 3 same-team-next L rows are quick regains);
the fixed converter agrees with the outcome field exactly (success↔R 43/43, fail↔L 22/22).
Lowest `np.select` priority (card results keep precedence by design). The owner-gated packing e2e gates the in-domain dribble share strictly
interior — a future feed that drops or renames the field fails loudly, never mass-fails silently.

**Consequences:** GS-only retrain trigger that **folds into the SAME pending re-fit the 4.49.0
amendment queued** (no additional retrain): GS-fitted xT/xtgk KDE move-sets now include retained
carries; GS VAEP result features shift. Zero delta for the other seven providers. Lakehouse: GS
`spadl_actions.result_id` changes on dribble rows — re-materialize on adoption. Cross-ref:
ADR-039 (packing) records the discovery sequence.
