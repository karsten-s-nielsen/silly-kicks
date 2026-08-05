# Covariate invariance under the ADR-051 geometry change

Which causal covariates the 4.74.0 geometry correction moved, and **which axis moved them**.

Produced by `scripts/measure_covariate_invariance.py` — provenanced, re-runnable, and the reusable
instrument for the same question in PR 6, PR 7 and the artifact-contract cycle.

    run_commit  0343eddca6ad   run_tree_dirty  false   status  ok
    corpus      1,360 frame groups from the committed slim fixtures (sportec, skillcorner, metrica)

## Why three arms and not two

The change had **two independent geometry axes**, and they interact:

* **Axis A** — `to_goal_relative_y` was added, so the two goal ends became a 180-degree rotation apart
  instead of opposite handedness.
* **Axis B** — `_dominant_region_area`'s y grid was re-anchored from centre 34.50 to 34.00.

The **new** grid is closed under the ADR-028 point reflection (`1.0 -> 67.0` is a grid centre); the
**old** one is not (`1.5 -> 66.5` is not). So measuring axis A against current code forces
`space_controlled`'s axis-A delta to zero **by construction**, while the baseline — which carries the
old grid — does move under it. Two arms cannot separate that:

| | old grid | new grid |
|---|---|---|
| **old transform** | `parent` | axis-B leg |
| **new transform** | **the interaction** | `current` |

Measured, and this is the result the design exists for:

    space_controlled  goal_x_0     axis A  0.0000   axis B  70.9565            B
    space_controlled  goal_x_105   axis A 97.5652   axis B  70.9565  A+interaction

A two-arm design would have reported axis A as **0.0** at `goal_x=105` and published a clean
decomposition that does not exist.

## The two structural invariants

These are what the decision *not* to rebuild `tf19_signoff_power` rests on, and they are exact:

    GK_r        axis A 0.0   axis B 0.0      hypot(a, -b) == hypot(a, b)
    gk_depth_x  axis A 0.0   axis B 0.0      cos is even; treatment = GK_r * cos(GK_theta)

Layer 2's treatment is `gk_depth_x >= 16.5`, so the treated set is identical and every count,
prevalence, and the `N_MIN_MATCHED` estimability verdict are unchanged.

## What moved (19 of 88 rows)

Only bearings, only at `goal_x=105`, plus `space_controlled` under the grid:

| covariate | arm | axis A @105 | axis |
|---|---|---|---|
| `theta` | shot, layer2-build, layer2-analysis | 2.6202 | A |
| `GK_theta` | shot | 0.7858 | A |
| `gk_theta` | cross | 0.7858 | A |
| `gk_lateral_offset` | cross | 11.3210 | A |
| `ball_theta` | model-feature-only | 2.6202 | A |
| `space_controlled` | cross | 97.5652 | **A+interaction** |
| `*Angle_*` (10 cols) | model-feature-only | ~6.28 | A |

The `~6.28` on the angle columns is ~2π — an `atan2` branch wraparound rather than a physical move.
They are model features, not causal covariates, so they sit outside the scope this artifact is cited
for, but they are emitted rather than filtered so the reader can see them.

## Limits — read these before reusing the instrument

**Four `LAYER2_CONFOUNDERS` are not measurable here**, and are emitted as data rather than omitted:
`defensive_line_height`, `defensive_line_compactness`, `pressure_on_actor__bekkers_pi`,
`time_remaining_s`. They are per-spell **joins** (`causal/_confounders.py`), not extractor features.
Two of them are **PR 6's own mechanism**, so an instrument billed as reusable for PR 6 must not be
read as covering them.

**`score_differential` is `not-measurable`, not axis B.** It is all-NaN by construction — the slim
fixtures carry no score context. An earlier revision of the classifier let NaN fall through to `"B"`,
asserting that the grid re-anchor moved a confounder that was never compared.

**The design is covariate-keyed, not model-feature-keyed.** Reuse for a model-feature question needs a
different key.

## Isolation

The `parent` arm runs in a **separate interpreter** against a `git archive` of `6e3a132~1`, not an
in-process monkeypatch. Both extractors bind geometry absolutely, so importing a baseline copy under
another name still resolves `_geo` to the *current* module. That is inert for this diff — the
`_geometry` change is purely additive — but it would silently measure zero for any future change that
alters an existing function's behaviour. The baseline arm's `GEOMETRY_VERSION` is asserted to differ
from current (`goal-relative-1` vs `goal-relative-2`), which is the control that catches a failed
isolation.
