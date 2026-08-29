# ADR-054 — Honest degradation on freeze-frames, and the StatsBomb 360 parse port

**Status:** Accepted
**Date:** 2026-08-06
**Version:** silly-kicks 4.76.0 (PR-S144)
**Spec:** `docs/superpowers/specs/2026-08-05-sb360-degradation-and-statsbomb-port-design.md` (rev 4)
**Supersedes nothing. Successor to** ADR-053, which audited and deliberately did not repair.

## Context

ADR-053 measured what every `add_*` does on StatsBomb 360 freeze-frames and found exactly one
fabrication: `add_ghost_gk` served a fitted model a feature vector it had not been given. The audit
reported rather than repaired, on purpose. This is the repair, plus the parse port that removes the
glue every SB360 consumer was otherwise writing.

## Decision 1 — the velocity guard goes at the SHARED SERVING SEAM

`_ghost_gk._serve_positions_core` refuses when frames declare velocity structurally unavailable, and
raises when they are unmarked and carry no `vx`/`vy`.

**Why there and not at the aggregator.** There are THREE public ghost entry points, and two bypass
`add_ghost_gk` entirely: `ghost_gk_xfns` reaches `compute_ghost_gk` (the VAEP path) and
`gkdv/_engine.py` calls `serve_ghost_gk_positions` (TF-19). A guard at the aggregator would have
fixed one caller in four. All three funnel through `_serve_positions_core`, which is also where the
4.22.1 physical-pitch clamp lives — the policy CLAUDE.md cites for *policy lives at the edge*.

**Neither direction is new policy.** CLAUDE.md's `speed_source` bullet already required both; the
ghost path violated a rule `_das.py` and `_press_commitment.py` already obeyed.

**The mechanism matters, because it determined the fix.** The extractor yields NaN, and
`predict_mean`'s HGBR reconstruction routes NaN down each split's LEARNED missing-value direction —
fitted where NaN meant an occasional dropped measurement, applied where 5 of 26 features are absent
on 100% of rows. Measured: `NaN -> [6.795, 33.522]` vs `zero-fill -> [6.888, 33.362]`. It is an
imputation POLICY, not a zero-fill, so "fill the zeros correctly" was never the fix.

## Decision 2 — the seams degrade DIFFERENTLY, and that asymmetry is forced

| seam | on a marked frame set |
|---|---|
| `add_ghost_gk`, `compute_ghost_gk` | NaN positions + `ghost_gk_source` |
| `serve_ghost_gk_positions` | **no rows** |

Not a style choice. `gkdv/_engine.py:557-562` RAISES on a non-finite ghost on a scored frame —
"pitch control silently DROPS NaN-coordinate rows, so a NaN ghost would make the keeper vanish
rather than error." Returning NaN rows there would BREAK TF-19 rather than degrade it; returning
nothing routes into its existing counted-drop path. The refusal reuses the seam's existing
`len(positions) == 0` branch rather than building a second empty frame, preserving the
input-derived join-key dtypes an ADR-019 comment there records.

## Decision 3 — a provenance COLUMN where the value changes, a DIAGNOSTIC where only the reading does

`ghost_gk_source` is a closed vocabulary (`computed` / `velocity_unavailable` / `no_keeper` /
`unlinked`) with a post-condition guard, because the ghost VALUE changes — a number becomes NaN, a
per-row fact.

Measured from the committed registry, **5** other aggregators produce output that moves with
velocity but stays honest and usable: `add_elastic_sync`, `add_obso`, `add_pausa`,
`add_pitch_control`, `add_space_creation`. Pitch control at zero velocity is a well-defined
positional model, not a fabrication. What a consumer cannot otherwise tell is that the value is
positional-only — and that is a property of the whole frame set, so a per-row column would carry a
constant five times over.

Hence `validate_velocity_regime` / `VelocityRegimeDiagnosis`, a third member of the
`validate_time_base` (ADR-017) / `validate_id_dtypes` (ADR-019) family, split across `schema.py` and
`utils.py` the way that family already is. It carries FOUR regimes plus `empty`, because
"forgot `derive_velocities()`" is not a variant of `mixed`: nothing is structurally missing and
labelling it `mixed` would raise with a false explanation, in the case a user is most likely to hit.

**The number is a floor, not a census.** `add_xshot_occurrence` is `not_exercised` on that axis —
unknown, not negative.

## Decision 4 — the port SHAPES, never fetches, and is EXTRACTED not written

`providers/statsbomb` takes already-loaded payloads and returns the
`snapshot_to_tracking_frames` contract plus `visible_area`. It adds no runtime dependency —
verified by AST over the subpackage, which imports only stdlib, numpy/pandas and `silly_kicks`.
`providers/sportec/parse.py` is the precedent and fetches nothing.

It is EXTRACTED from `scripts/build_sb360_coverage.py`, which had already grown most of the parse
half. A parallel implementation would be the fork `tracking.defended_goal_x`'s docstring names as a
defect class. Verified an identity move (`mod.defending_gk_visible is defending_gk_visible`), which
also answers ADR-037's clause-(e) hop: `coverage.md` needs no re-run, because the script calls the
objects the library exposes.

## Decision 5 — the clip is EVENT semantics and does not follow the affine

`spadl/_sb_coordinates.sb_xy_to_spadl` holds the affine; `_convert_locations` keeps the clip and the
3-element shot `y_offset`. This is ADR-038's split (`_scale_to_spadl` vs `_transform_coords`) applied
to a new provider, and it is what lets `visible_area` extend past the touchline: a broadcast camera
legitimately sees beyond it, so clipping would silently shrink the observed region — the entire
quantity the column carries.

**The cell-centre correction IS applied to polygons, for a measured reason that is not the obvious
one.** The feared conflict with `visible_fraction` does not exist — that returns an area RATIO and
`crc` is a translation, invisible to it (0.625 either way). What binds is player/polygon alignment:
players reach SPADL through the same affine WITH `crc`, so omitting it would offset the polygon
**0.4375 m** from the players it bounds.

## Consequences

* **No retrain, no re-materialize.** Ghost positions on velocity-bearing frames are unchanged,
  asserted by an oracle captured before the change. Tolerance CHOSEN per ADR-050 (`atol=1e-6`,
  `rtol=0`), not inherited from the sibling golden that scopes itself to one machine — confirmed by
  CI across ubuntu 3.10/3.11/3.12 and windows 3.12.
* **Schema (Hyrum):** `add_ghost_gk` and `compute_ghost_gk` gain one column. Values unchanged.
* **Breaking, narrowly:** unmarked velocity-less frames now RAISE. What breaks is a fabricated
  coordinate.
* **The audit re-derives to ZERO fabrications** — by RULE, not hand-edit. That is ADR-053's
  locked-observation / reviewable-adjudication split working as designed.
* **`player_id` does not recur across SB360 frames**, which forecloses per-player aggregation. In
  the port's published contract, not a downstream surprise.

## What this cycle deliberately did NOT do

* **The `visible_area` consuming seam.** The polygon is carried as raw data (ADR-009); the API waits
  for a consumer.
* **`_defending_goal`.** It re-derives the pinned public `defended_goal_x`, whose docstring forbids
  exactly that, and the two ALREADY disagree on the no-GK-rows fallback. It is also an ADR-028 D3
  orientation defect, but `_ghost_gk.py` is not in the pinned D3 unit
  (`tests/tracking/test_mirror_registry.py:294-311` asserts exactly three files), so it is queued in
  `TODO.md` rather than assumed covered. → RESOLVED by ADR-055 (4.77.0): the fork was deleted and
  replaced by the canonical `GoalMap`; no TODO row ever existed.
* **The `snapshot_to_tracking_frames` dtype pin** and the four unauditable boundary entry points.

## Three defects this cycle found in its own guards

Recorded because each is a reusable shape, not a one-off.

1. **A non-vacuity gate coupled to a defect existing.**
   `test_at_least_one_column_was_adjudicated_a_fabrication` asserted the registry contained a
   `silent_degrade`; repairing the only one broke it. Split into a PLANTED case against the rule
   engine and a companion asserting the repaired state.
2. **Prose read as code, twice.** A `statsbombpy` import check grepped source text while the
   docstring explained the absence; and the script's lazy-import guard matched the SUBSTRING
   `"statsbomb"`, so the new first-party `providers.statsbomb` tripped a check meant for the
   optional package. Both now AST- or package-root-matched.
3. **A checksummed fixture without a `binary` pin.** `SOURCE_SHA` recorded a CRLF working-copy
   digest describing bytes that exist only on Windows; CI read the LF blob. The repo had solved this
   three times already for model artifacts — this is the fourth `.gitattributes` entry.

## Amendment (4.90.0 / PR-S160 — xShot guard, self-enforcing contract, space_creation softening)

Three additions extend the velocity-availability contract this ADR established.

**1. xShot joins the contract (a third fitted model), and the fabrication was LIVE.**
`add_xshot_occurrence` had no velocity guard: `XSHOT_FEATURE_NAMES_FAITHFUL` includes `speed`, which
is NaN on a velocity-less SB360 freeze-frame, so XGBoost's missing-value routing produced an
ungrounded probability (measured: `0.239811` on the audit's declared frame). `xshot_occurrence_xfns()`
is a member of `pre_shot_gk_full_default_xfns`, so any consumer building that GK-union bundle on SB360
was carrying fabricated xShot. `compute_xshot_occurrence` now carries the same two-prong guard as
`compute_xcross_attempt` (declared-unavailable → honest NaN; undeclared-missing-velocity → loud raise
naming `derive_velocities()`), at the shared compute seam all three public entry points reach.

**2. The contract becomes CI-ENFORCED, not merely documented — and the enforcement is anchored
STRUCTURALLY.** Nothing structurally required a velocity-feature model to route through the guard,
which is exactly why xShot slipped through releases while the SB360 audit recorded it `not_exercised`
(a `not_exercised` model is invisible to a runtime audit — the audit provably cannot police the class).
`tests/tracking/test_velocity_feature_contract.py` turns the prose contract into a static invariant.
Its population is anchored on the STRUCTURAL property "takes a `frames` parameter AND scores a fitted
model", NOT on the `*_FEATURE_NAMES*` naming convention — anchoring on a convention would re-commit the
original sin one level up (a future frame-served model that declares features differently would
escape). Each frame-scoring entry with a velocity feature must CALL the guard in its body (a module
import is not enough — the refactor-forgot-to-call bug); `_GUARD_EXEMPT` holds verified wrappers
(delegate to a guarded entry), eval probes, and velocity-free-model scorers; `test_population_is_exact`
(keyed on `(module, name)` pairs) catches a new frame-served model whatever it names its features.
**Named limitation:** the guard check requires a direct in-body call, so the two-prong block stays
intentionally duplicated across the serve modules; if it is ever DRY'd into a shared helper,
`_fn_calls_guard` must be taught to follow one call level (or the helper added to its alias set) —
the gate trades refactorability for seam-visibility, on purpose.

**3. `add_space_creation` opponent-perspective softening — and the marker is a PRAGMATIC FOV PROXY.**
`_resolve_opponent_team_id` and `_compute_space_creation_for_action`'s two-team guard now soften a
legitimate one-team SB360 FOV crop to a per-row NaN opponent value (with a `space_opponent_source`
provenance token: `resolved` / `unresolved_one_team`) instead of aborting the whole batch, while a
full-tracking one-team frame — or a 0/3+-team frame in any mode — still raises. **The coupling this
ADR must own:** the gate keys on `velocity_unavailable_by_design`, which answers "does this frame carry
kinematics?", NOT "is this a legitimate FOV crop?" — two orthogonal properties that COINCIDE only in
today's provider set (SB360 is both velocity-less and FOV-cropped). There is no frame-level FOV signal
today (SB360's `visible_area` is an action-level polygon; the schema visibility column is unpopulated),
so the velocity marker is the pragmatic proxy — but the softening should migrate to a real FOV signal
the day a frame-level one exists. `space_opponent_source` is the mitigation: a consumer can count
softened rows rather than conflating an unresolvable opponent with a genuine zero. Additive; no VAEP
retrain. Downstream materializers gain the new column (flagged in `docs/PRIVATE_CONSUMERS.md`).
