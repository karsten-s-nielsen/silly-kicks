# ADR-081: Rest-defense Layer-2 danger-behind-line valuation + additive tracking seams (TF-60 PR2)

| Field | Value |
|---|---|
| **Date** | 2026-08-30 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

TF-60 PR1 (ADR-080) shipped the descriptive Layer-1 rest-defense structure KPIs in a new
`silly_kicks/restdefense/` package. PR2 adds **Layer 2 — danger-behind-the-line valuation**: five
additive Tier-1 columns on the `compute_rest_defense` samples table quantifying the opponent's
counter-danger in the zone behind the in-possession team's rearguard line. The columns
(`rd_attacker_space_control`, `rd_danger_behind_line`, `rd_danger_behind_line_gk`,
`rd_gk_coverage_behind_line`, `rd_gk_reachable_coverage_m2`) reuse the shipped pitch-control (TF-7)
and GK-influence (TF-15) engines, oriented via `resolve_defended_goals` → `GoalMap` (ADR-055, never
team identity). Spec: `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md`
§7.2 / §13 / §14.

Two of the metrics could not be expressed through the existing *public* tracking seams as they stood:
the OBPV `w_field` re-weighting of `rd_danger_behind_line` (`compute_threat_pc` returns a collapsed
scalar with no per-cell weight hook) and `rd_gk_reachable_coverage_m2 = reachable-area ∩ Z`
(`compute_gk_influence` returns only the whole-pitch area). Both `compute_tti` / `select_back_line_players`
/ `physical_grid` are public, so either extending the seams or realizing the physics inside
`restdefense/` was feasible.

## Decision

Ship Layer 2 in `restdefense/_danger.py` (+ `restdefense/_wfield.py` for the OBPV `w_field`), reusing
`compute_threat_pc` / `compute_pitch_control` / `PitchControlSurface.control_in_region` /
`compute_gk_influence`. Close the two seam gaps with **small, additive, default-`None`-byte-identical
extensions to the one shared engine each** (owner-approved 2026-08-30):

1. `compute_threat_pc(..., field_weight=None)` — an optional per-cell weight callable multiplied into
   the oriented threat grid before the Voronoi partition. Powers `w_field`; default None is
   byte-identical to today.
2. `compute_gk_influence(..., region=None)` — an optional `(x_min, x_max, y_min, y_max)` restricting
   the reachable-area sum. Powers `∩ Z`; default None is byte-identical (velocity-suppression applies
   first).
3. Two additive public re-exports so `restdefense/` imports only public tracking seams:
   `zero_velocity_if_unavailable` (for its own `compute_pitch_control` velocity handling) and
   `compute_gk_influence` (which was not in `tracking.__all__`).

Additive throughout — no existing column changes, **no VAEP retrain**, in no default xfn list, C4
unchanged.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Restdefense-local realization (tracking byte-frozen) | no tracking change | reimplements TF-7/TF-15 physics + params in restdefense (drift), and `rd_danger_behind_line` default no longer equals `compute_threat_pc` | rejected — reimplementation + drift is the failure the reuse discipline exists to prevent |
| B. `compute_threat_pc` as danger engine + `field_weight` hook, `compute_gk_influence` + `region` (chosen) | one threat/reachable engine; default byte-identical; receiver-attribution is the *more correct* counter-danger measure (a counter needs a receiver); spec-faithful ("realised by compute_threat_pc") | touches two tracking seams | — |
| Hybrid (plain ∫_Z via physical_grid + region param) | tracking frozen for the integrals | a second threat definition in restdefense; diverges from the compute_threat_pc anchor | rejected — DRY + semantic correctness favour the one engine |

## Consequences

### Positive

- A threat-weighted counter-danger surface with the in-possession keeper folded in as a first-class
  control agent (`lambda_gk`) — the literature gap TF-60 names.
- Two general, reusable tracking capabilities (a per-cell threat weight; a region-restricted reachable
  area) land in `tracking/`, not in a restdefense-private helper.

### Neutral

- **GK-inclusion mechanism:** GK-blind base = A's keeper row DROPPED; GK-included = keeper kept, which
  under the default `SpearmanParams(lambda_gk=3.0)` is *exactly equivalent* to the spec's literal
  `SpearmanParams(lambda_gk)` framing (the surface scales the kept keeper by `lambda_gk`; restdefense
  never constructs a `SpearmanParams`). The GK deterrent contribution `= base − gk` is derivable from
  the two emitted columns, not a sixth column.
- **`w_field` is opt-in, off by default, un-tuned** (`RestDefenseParams.danger_field_weight=False`,
  `WFieldParams` un-tuned spec-time defaults, empty `for_provider`; ADR-009/066). With the default,
  `rd_danger_behind_line` is byte-identical to a plain `compute_threat_pc`.
- **`rd_gk_reachable_coverage_m2` is Tier-2** (velocity-constitutive) → honest-NaN on velocity-less
  providers, inherited from `compute_gk_influence`'s existing suppression (ADR-063). `#1`–`#4` are
  Tier-1 lifts (positional / zero-velocity model on SB360).
- **The Layer-2 family is gated on a fitted `xt` (P2-02, owner-approved):** without one, all five
  columns are NaN before any pitch-control call, so a Layer-1-only caller is byte-identical to PR1 (no
  pitch-control cost, no velocity precondition). The cost — `#1`/`#4` (space control, gk coverage)
  require an `xt` despite not using its values — is the accepted price of a single opt-in signal and
  no caller break. SB360 loses nothing provider-specific: it is velocity-declared-absent so it never
  hits the fail-loud raise, and `#5` is Tier-2 NaN regardless of the gate. An unfitted `xt` raises
  (fail-closed, 4.62.0).
- **SB360 audit (ADR-053):** the four pitch-control metrics are `differs_by_design` (Tier-1 lift), the
  reachable m² is `honest_nan` (Tier-2), and on the keeper-less `gk_absent` roster the danger/keeper
  metrics are `not_exercised` (no GoalMap orientation / no keeper). The boundary entry's
  `verdict_provenance` becomes `substantive` (the differs cells respond to velocity).

## Related

- **Spec:** `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md` (§7.2 / §13 / §14 / §16.2)
- **Plan:** `docs/superpowers/plans/2026-08-30-tf60-rest-defense-pr2-layer2.md`
- **Review:** `D:\Development\_reviews\2026-08-30-tf60-rest-defense-pr2-layer2-plan.md` (+ `-r2`), findings P2-01..P2-09
- **ADRs:** builds on ADR-080 (PR1), ADR-055 (`GoalMap`), ADR-019 (`id_compat`), ADR-063 (velocity
  tiers), ADR-062/077 (FOV companions), ADR-066/009 (calibratable defaults), ADR-043 (pitch-control
  cache moved-player trap), ADR-053 (SB360 audit), ADR-048 (glossary), ADR-005 (attribution).
- **External references:** Novillo et al. (2025) Chaos Solitons & Fractals (λ_GK behind the line);
  Ogawa et al. (2025) arXiv:2505.14711 (OBPV `w_field`); Spearman/Shaw & Sudarshan (pitch control).
  See `NOTICE`.
