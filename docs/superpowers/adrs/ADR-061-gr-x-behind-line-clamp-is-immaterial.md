# ADR-061: The `gr_x >= 0` behind-line clamp is immaterial (~0.2%); parked, not shipped

| Field | Value |
|---|---|
| **Date** | 2026-08-17 |
| **Status** | Accepted (DGX measurement landed 2026-08-17; run_commit aa34017) |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8) |

## Context

`_geometry.in_penalty_area_goal_relative*` has no lower bound on `gr_x`, so a point *behind* the
defended goal line (`gr_x < 0`) counts as in-box. ADR-050 §6 parked the question "should a
behind-the-line point count as in-box?" with a measured frame-row population but no answer. The signed
`gr_x` reaches two trained models — `_ghost_gk` (`attackers_in_box`) and `_xcross_attempt`
(`box_off_def_ratio`, feature #6) — so a clamp is a two-model re-fit + republish, not a one-line
change.

**Basis A** re-frames the question as: does a `gr_x >= 0` clamp materially change the *training
examples* each model sees? Measured on the full 179-match pining corpus
(`docs/research/box_constant_delta/`, `run_commit aa34017`, clean):

| model | box feature | `changed_fraction` | off-pitch fraction |
|---|---|---:|---:|
| ghost | `attackers_in_box` | 0.213 % | 0.268 |
| xcross | `box_off_def_ratio` | 0.193 % | — |

The clamp moves ~0.2 % of examples corpus-wide (the pre-commit 2-match probe gave 0.12 % / 0.06 %).
Separately, **26.8 %** of the ghost behind-line box points sit > 2 m off-pitch — the broadcast/detection
artifacts `_loader_pining` already warns about, not real keepers behind their line.

## Decision

**Do not ship the clamp; park it, doc-only.** The `gr_x >= 0` change is immaterial (~0.2 %) and does
not warrant a two-model re-fit + republish. The 26.8 % off-pitch fraction is an **upstream ingestion
(D-data)** signal — recorded here as a data-quality observation — not a geometry bug a predicate clamp should paper over.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Ship the `gr_x >= 0` clamp now (declared constant + ghost/xCross re-fit) | removes behind-line points | ~0.2 % effect; re-fits two models + republishes weights for a noise-level change; papers over a data-quality issue | effect immaterial; would re-fit for the wrong reason |
| B. Doc-only / park + record the off-pitch observation (chosen) | honest about materiality; keeps attribution clean; points at the real fix (ingestion) | the unbounded predicate remains; discipline is manual | — |
| C. Silently leave unbounded, no record | zero work | the parked question rots; the off-pitch signal is lost | ADR-050 §6 already showed "no record" fails |

## Consequences

### Positive
- The behind-line predicate stays unbounded, so ghost/xCross weights are unchanged — no retrain, no republish, no train/serve skew.
- The off-pitch signal is captured in this ADR + `docs/research/box_constant_delta/`, pointing a future maintainer at the ingestion layer rather than the geometry.

### Negative
- The unbounded predicate still admits behind-line points; a future keeper-domain analysis may reopen it.
- **If the clamp is ever warranted, ADR-050's feature-contract will NOT catch its omission — and that is measured.** A lower bound declares no new constant and the probe frame carries no behind-the-line player, so `_feature_contract_block()` is byte-identical with and without the clamp. Shipping it would require a declared `penalty_area_min_gr_x` constant + a `cache_token()` bump, gated red-first (spec §5.4); the discipline is manual until then.

### Neutral
- The measurement driver (`measure_box_constant_delta.py`) gained a `training_flip` block and adopted the ADR-052 `for_each` shard seam; both additive.

## Revisit trigger

Take this up as its own two-model re-fit + republish cycle if either holds: (a) a keeper-domain
analysis shows behind-line rows concentrating in the GK box rather than spreading across the pitch, or
(b) the off-pitch population grows materially on a higher-noise corpus. The first fix to reach for is
the ingestion cleanup the 26.8 % implies, not the predicate clamp.
