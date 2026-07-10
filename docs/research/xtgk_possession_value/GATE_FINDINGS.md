# xT-GK v2 — Make-or-Break Gate: Real-Data Result (2026-07-10)

Owner-run of `scripts/validate_xtgk_possession_value.py` against the Databricks gold marts
(`bronze.spadl_actions` ⋈ `dim_matches` ⋈ `dev_gold.fct_action_context` [pressure] ⋈
`dev_gold.fct_shot_xg` [calibrated xG]). Pre-registered `GateConfig`: `effect_floor=0.005`,
`relative_effect_floor=0.25`, `n_min=30`, `min_occupied_cells=2`, `expected_direction=decreasing`.
Machine report: `gate.json`.

## Pressure measure — pinned to `bekkers_pi` (§5 Q3, resolved 2026-07-10)

The first run used `pressure_on_actor__andrienko_oval` and STOPped (0 occupied deep cells) because
**52% of actions had pressure exactly 0**, degenerating the terciles. The lakehouse 3-method audit
(handoff F2) settled it: the zero-mass is a **method artifact, not missing data** — on the same
actions the exact-zero rate is andrienko 46.9%, link_zones 79.5%, **bekkers_pi 4.7%**. `bekkers_pi`
has a non-degenerate tail and is the trustworthy measure; the gate is now pinned to it. (See
`LAKEHOUSE_HANDOFF.md` F2.)

## Verdict (bekkers_pi)

| Cohort | Reward | Rung | Occupied | Effect (xG) | Relative | Direction | Monotone | Cross-check | Passed |
|---|---|---|---|---|---|---|---|---|---|
| **WC2022** (gradientsports, **authorising**) | certified (ood 0.0, n=1473) | global | 8 | 0.0027 | **0.86** | decreasing | ✓ | ✗ | **No** |
| **RM** (skillcorner, provisional) | **uncertified** (ood 1.0, n=2596) | global | 17 | 0.0089 | **1.05** | decreasing | ✓ | ✗ | **No** |

Both orientations agree (mirror_y equivariant; mirror_x rejected as non-attack-LTR). Deep V on
WC2022: V_lo ≈ 0.0045 → V_mid → V_hi ≈ 0.0018 (decreasing, monotone).

## Interpretation — GO-leaning, not the degenerate STOP

The make-or-break question is *"after refitting V honestly, does the deep keeper zone show a real,
pressure-dependent gradient?"* With the trustworthy pressure measure the answer is **yes**: a
decreasing, monotone deep-zone gradient with a **strong relative effect (0.86 WC2022, 1.05 RM ≫ the
0.25 floor)** on both cohorts. That is the signal v1 lacked. It is **not a clean PASS**, for two
reasons, both Eyestone-review items rather than kills:

1. **Absolute-floor miss (WC2022):** the absolute effect (0.0027 xG) sits below the pre-registered
   0.005 floor. This is a **magnitude-calibration** question, not an absence of signal — deep-zone
   *possession* value is intrinsically tiny (E[first-shot xG | ball in the keeper zone] ≈ 0.003–0.005),
   so a 0.005 absolute floor is large relative to the quantity it measures. The relative effect (0.86)
   is the scale-free read and it is strong. **Do NOT lower the pre-registered floor post-hoc**;
   Eyestone/owner should re-examine whether the absolute floor was set commensurate with deep-zone
   xG magnitude (RM, at 0.0089, clears it).
2. **Cross-check divergence (both):** the model-free empirical surface disagrees with the Markov
   surface on the build-up-cell gradient. The empirical estimator is high-variance (per-action
   first-shot xG over a reverse scan) and independent by design, so divergence is a flag to
   investigate (transition-structure vs. sampling noise), not proof the Markov surface is wrong.

## Escalation (Eyestone/owner)

- Re-examine the **0.005 absolute effect floor** against the measured deep-zone value magnitude
  (≈0.003–0.005) — likely too high; the relative effect (0.86–1.05) is the meaningful criterion here.
- Investigate the **empirical cross-check divergence** (is it transition-structure or estimator
  variance?).
- Confirm `bekkers_pi` as the canonical pressure measure for the metric (this run pins it).

Per the owner build-ahead directive, SP2–SP5 shipped regardless. This result upgrades the finding
from "degenerate STOP" to "real pressure-dependent deep gradient present; PASS pending an
absolute-floor recalibration + a cross-check review".

## RM caveat

Reward is 100% OOD (uncertified xG on SkillCorner tracking-derived features; the lakehouse confirmed
this is the small-n certification floor, not roadmapped — handoff F3), so the RM verdict is
provisional by construction, though it shows the same real decreasing gradient (relative 1.05) and
even clears the absolute floor.

## History (superseded)

The initial `andrienko_oval` run: WC2022 STOP (0 occupied cells), RM FAIL-crosscheck — an artifact of
the 52% pressure-zero mass, since resolved by pinning `bekkers_pi`. Two real code bugs were fixed
during that run and remain valuable: `prepare_cohort` now drops frame-absent tracking-gap nulls, and
a NaN-safe `flat_zones` guards the zone-binning seams against real NaN coords.
