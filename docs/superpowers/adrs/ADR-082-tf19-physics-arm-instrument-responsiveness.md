# ADR-082: TF-19 A+2 — physics-arm instrument-validity (Layer 0) + responsiveness (Layer 1) probes

| Field | Value |
|---|---|
| **Date** | 2026-08-29 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

The shipped GKDV physics arms (`delta_das`, `delta_threat_suppression`; ADR-043, ADR-075) are a
counterfactual valuation of keeper positioning. Before their per-keeper aggregates can be interpreted,
two prior questions must be answered on the corpus, and answered honestly: **is the instrument alive**
(does the arm respond at all when the keeper is moved a lot?) and **is it responsive to a realistic,
keeper-specific displacement more than to a generic one?** The parent §6.4 validation stack
(`docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`) decomposes these along its own
layer boundaries. This cycle is the owner-chosen **A+2** slice: **A** (named-keeper expected-sign
validation of the shipped arms) **+ Layer 0** (instrument validity) **+ Layer 1** (responsiveness),
**physics arms only**. Layers 2 (causal ATT — measured underpowered on WC2022, 151 treated spells) and
3 (feature-headroom geometry) and the composing `gkdv_discrimination_verdict` stay deferred.

Three constraints shaped the design:

- The physics arms are **model-free** (they difference an accessible-space / pitch-control surface),
  so the parent's S1 training-support (OOD-flatness) diagnostic has no confound to guard here.
- The two arms are ATTACKER-value (`actual − ghost`), so a deterrent keeper reads **negative** on both
  — but the threat arm's OUTPUT column is `delta_threat_suppression` while its registered direction key
  is `delta_threat`, a bridge every sign check must cross or silently skip.
- The keeper identity a named-keeper validation needs already has ONE resolver — the ADR-078
  `tracking.resolve_keeper_identities`. gkdv must NOT grow a second one (ADR-055 single-source), and
  ADR-037 confines gkdv to `tracking._das`, so it reaches that resolver DRIVER-side.

## Decision

Ship the probes as new pure functions in `gkdv/_probe.py`, the arm-direction-key seam in
`gkdv/_validate.py`, and an owner-run corpus driver:

1. **Dose imposer** — `impose_defending_keeper_dose(frames, *, home_team_id, dose, displacement=None,
   model=None)` substitutes ONLY the defending keeper at a dose position (`realistic` = the ghost
   model's own position filtered to `|displacement| ≥ REALISTIC_MIN_DISP_M`; `ladder` = the actual
   position displaced toward the defended goal; `saturating_goalline` / `saturating_x30` = fixed
   landmarks), PURE, reusing the engine's domain/provenance so the scored set matches
   `build_ghost_frames`.
2. **Layer-0 instrument validity** — `layer0_instrument_verdict(...)` over ALREADY-POOLED corpus
   statistics: `arm_unscoreable` short-circuits FIRST (domain `< MIN_DOMAIN_FRAMES`, or any all-NaN leg),
   then the parent void condition (void iff the saturating median clears NEITHER `SATURATING_MULTIPLE ×`
   the realistic median NOR the placebo 95th percentile).
3. **Layer-1 responsiveness** — `layer1_responsiveness_verdict(...)` = `gk_med ≥ PHYSICS_ARM_PROBE_RATIO
   × max(nd_med, placebo_p95)`, same short-circuits, no absolute floor (Layer 1 is comparable-not-
   decisive). `paired_vector_controls(...)` follows the parent `_model_eval` idiom: it displaces **ONE**
   defending-team outfielder per control — the NEAREST alone (`nd`) plus R **single**-outfielder placebo
   replicates — by the SAME per-frame vector, so `nd_med` and `placebo_p95` are DISTINCT single-player
   quantities and the `max` is meaningful (a combined R+1-player control would collapse them and compare
   a 1-player keeper move against a many-player control). The **Layer-0 multiple leg requires
   `real_med > 0`** — ΔDAS is zero-dominated, so `sat ≥ 5·0` would vacuously validate a dead instrument;
   the reduce records the pooled medians for auditability.
4. **Arm-direction-key seam** — `_ARM_DIRECTION_KEY` + `expected_direction_for_arm(arm_column)` bridge
   the arm's OUTPUT column to its `EXPECTED_DIRECTION` key (an unmapped arm raises, never silently skips).
5. **Owner-run driver** — `scripts/build_tf19_instrument_responsiveness.py`: a `for_each` map (ADR-052
   per-match shards, resumable, conserving) of the per-frame dose magnitudes, then a REDUCE computing
   the POOLED Layer-0/1 verdicts (with the medians they rest on), the §6.1 `gate_eligible` census, the
   §6.2 per-KEEPER sign table (`named_keeper_signs.parquet` — the "A" face-validity deliverable) against
   a PRE-REGISTERED named-keeper prior (`NAMED_KEEPER_PRIOR = {Alisson, Neuer} → deterrent`, locked in
   code before the run + stamped into `metrics.json`; an owner-injected `--keeper-names-json` runs the
   confirmatory `meets_prior` check, dependency-inverted so name resolution is never parsed here) and the
   Layer-4 `behavioural_anchoring_verdict` (on the gate-eligible subset). Keeper identity is resolved
   ONCE per match via `tracking.resolve_keeper_identities` and threaded driver-side (ADR-037), with the
   resolution dropped-AND-counted (`keeper_identity` totals). The authoritative artifact is `metrics.json`
   (ADR-056 staleness-detector-visible; the driver is enrolled in `_DECLARING` and declares every
   registered threshold), also recording the `registered_constants` + `provider_support` matrix. A
   `--match-ids-json` partition worker writes shards + a per-worker manifest ONLY; the pooled reduce runs
   on a final UNPARTITIONED pass (a partition worker's shard set is a race-dependent partial).

Everything is **reported-not-gated**: no gate flip (`TF19_PROBE_ABS_FLOOR` / `regate_verdict` /
`EXPECTED_DIRECTION` untouched), no retrain, no re-materialize, C4-free.

## Alternatives considered

| Option | Why rejected |
|---|---|
| A gkdv-local `resolve_defending_keeper_id` (the original A+2 plan) | A second native-identity path violates ADR-055 single-source, and ADR-037 confines gkdv to `tracking._das`. **Dropped** — the driver consumes ADR-078's `tracking.resolve_keeper_identities` instead; only the gkdv-specific `_ARM_DIRECTION_KEY` / `expected_direction_for_arm` stays local. |
| Port the parent S1 training-support diagnostic | The physics arms are model-free, so there is no OOD-flatness confound to guard — a Chesterton's-Fence clearance, not an inheritance. |
| Reuse `TF19_PROBE_RATIO` / `XS_PROBE_RATIO` for the Layer-1 threshold | Both are model-specific (xCross / xShot). `PHYSICS_ARM_PROBE_RATIO = 2.0` is a NEW registration; the parent Layer-1 idiom fixes the FORM, not the value. |
| Compute the verdicts per shard | Layer-0/1 are POOLED-corpus statistics; a per-shard implementation reads a thin per-match domain as `arm_unscoreable`. The verdicts are a REDUCE over all shards (guarded non-vacuously by the driver test). |

## Consequences

### Positive
- The shipped physics arms get a demonstrable, reported-not-gated named-keeper validation on the
  velocity-bearing (Gradient Sports) corpus.
- The probe seams (dose imposer, verdicts, paired-vector controls, arm-key bridge) are reusable and
  publicly exported with worked Examples.

### Negative / Neutral
- The threat arm (`delta_threat_suppression`) is reported **`arm_unscoreable`** here: it needs a fitted
  `ExpectedThreat` and the package ships no loader (the same constraint `build_gkdv_arm_values` records;
  fitting one in-process is a leakage decision for its own cycle). This is a first-class verdict,
  reported rather than silently omitted.
- ΔDAS is velocity-constitutive, so it is NaN on velocity-less SB360 freeze frames regardless of keeper
  identity (ADR-063). The identity resolver's SB360 roster path is wired through the same seam but the
  runnable pass is the velocity-bearing corpus.

## Related

- **Spec:** `docs/superpowers/specs/2026-08-28-tf19-a2-physics-arm-instrument-responsiveness-design.md`
- **Plan:** `docs/superpowers/plans/2026-08-28-tf19-a2-physics-arm-instrument-responsiveness.md`
- **Parent spec (§6.4 layer defs):** `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`
- **ADRs:** ADR-043 (gkdv physics arms + import confinement), ADR-075 (batched arms), ADR-078
  (keeper-identity resolver this driver consumes), ADR-037 (gkdv → `tracking._das` only), ADR-052
  (`for_each` corpus-driver seam), ADR-056 (artifact provenance), ADR-063 (velocity-availability
  honest-NaN).
