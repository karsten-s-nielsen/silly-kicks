# ADR-100: `ExpectedThreat` JSON serialize seam (`to_dict`/`from_dict`/`save`/`load`)

| Field | Value |
|---|---|
| **Date** | 2026-09-20 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner), luxury-lakehouse session (consumer), silly-kicks session |

## Context

`silly_kicks.xthreat.ExpectedThreat` had `__init__`, `fit`, `interpolator`, `rate`, and plain attributes — **no serialization**. A fitted model could not cross a process boundary except by re-`fit()`.

The luxury-lakehouse ExT-v2 fold needs exactly that: `wf-xt-grids` fits `ExpectedThreat` on HF Jobs (CPU) and persists it to the Hub; `territory_writer` reconstructs it on Databricks serverless to inject into the TF-54b `territory` counterfactual (`compute_territorial_dominance(method="counterfactual", ...)`) and `xthreat.destination_profiles`. Both require an injected, **fitted** model (`require_fitted_xt` raises `NotFittedError` on `not np.any(model.xT)`); silly-kicks ships no xT, so the consumer fits it once. Re-fitting at the consumer is not acceptable — a full-fact `.toPandas()` fit violates the consumer's boundedness gate. So `ExpectedThreat` needs a serialize/deserialize round-trip. The consumer policy forbids `pickle.loads`.

## Decision

Add four additive public methods to `ExpectedThreat`: `to_dict()` / `from_dict()` (the JSON-round-trippable serialization primitive) plus `save(path)` / `load(path)` (thin JSON-file wrappers over them). The serialized state is `l`/`w`/`eps`/`method`/`params` + the five fitted arrays (`xT`, `scoring_prob_matrix`, `shot_prob_matrix`, `move_prob_matrix`, `transition_matrix`) + `heatmaps`; `grid` is derived and rebuilt in the constructor. A top-level `format_version` gate makes the reader fail-closed on any unknown schema.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. `pickle` the model | trivial | executes arbitrary code on load; consumer policy forbids `pickle.loads`; opaque, not cross-language | security + policy |
| B. `to_dict`/`from_dict` only (no `save`/`load`) | minimal surface | every caller re-writes the same `json.dump`/`json.load` file I/O | consistency — sk models carry `save`/`load`; owner chose the complete surface |
| C. `save`/`load` with a SHA256SUMS sidecar + chirality/feature-contract (the bundled-model precedent, ADR-011/040/050) | matches DependenceModel/GhostGk load discipline | that machinery detects tampering of **wheel-bundled** artifacts loaded by pip users; `ExpectedThreat` is consumer-fitted + consumer-persisted (to HF/UC, the consumer owns integrity) — there is no sk bundle to tamper | cargo-culting a bundled-artifact control onto a non-bundled model |
| D. **chosen** — `to_dict`/`from_dict` primitive + thin `save`/`load` JSON wrappers + `format_version` fail-closed | one serialization source, JSON-safe, inspectable, cross-process, forward-compatible | a `format_version` bump is a coordinated cross-repo change | — |

## Consequences

### Positive
- A fitted `ExpectedThreat` crosses a process boundary as JSON; the lakehouse ExT-v2/territory-counterfactual leg is unblocked without a consumer-side re-fit.
- `format_version` gives a loud forward-compat door: an older reader rejects a newer schema rather than silently mis-parsing.
- Single serialization source (`to_dict`/`from_dict`); `save`/`load` cannot drift from it.

### Negative
- The serialized dict is a cross-repo consumer contract — a future schema change is a coordinated `format_version` bump + a consumer bump.
- No sk-side integrity checksum on `save`/`load` (Option C declined); the consumer owns the integrity of its own persisted artifact (its ADR-072 upload seam).
- **Blast radius — the TF-19 gkdv threat-arm refusal is relocated from a package impossibility to a driver policy.** `scripts/build_gkdv_arm_values.py` refused the `threat`/`both` arm on the *fact* that "`ExpectedThreat` has no save/load" (the validation-harness "make the outcome UNREPRESENTABLE" discipline). That fact is now false, so the refusal is re-stated as an explicit DRIVER policy (the driver wires no xT loader and never fits one in-process). The refusal stays loud and test-pinned (`test_threat_arm_is_refused_not_silently_defaulted`), but the guarantee is now driver-scoped, not package-scoped — a future `--xt` wiring must go through its own registered leakage cycle. The absence-pin `test_expected_threat_really_has_no_loader` was retired (its docstring pre-authorised exactly this revisit) → `test_serialization_exists_and_the_refusal_is_now_a_driver_policy`.

### Neutral
- Purely additive: `fit`/`rate`/`interpolator`/`values_at_points`/`destination_profiles` and the SK-xT-1 `singh_counts` frozen-oracle parity are byte-identical. No VAEP/tracking retrain, no re-materialize, C4-free. No new bundled artifact (silly-kicks still ships no xT).
- `xT` is stored y-inverted (ADR-041); the round-trip is verbatim on both legs — no orientation normalization — so `rate`/`destination_profiles` stay bit-identical.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-20-expectedthreat-serialize-seam-design.md`
- **Plans:** `docs/superpowers/plans/2026-09-20-expectedthreat-serialize-seam-plan.md`
- **ADRs:** ADR-021 (xthreat package / pluggable transition family), ADR-041 (raw-orientation storage; `_physical.py` neutralizes it), ADR-011/040/050 (the bundled-artifact fail-closed load discipline whose SHA/chirality prong is deliberately NOT applied here — Option C).

## Notes

Fail-closed order in `from_dict` (spec §5): `format_version != 1` → `ValueError` first; a missing required key → `KeyError`; a method/params mismatch → `TypeError` (via `validate_params_for_method` in the constructor); an all-zero `xT` payload → `NotFittedError` (via `require_fitted_xt`). Guarded by `tests/xthreat/test_serialize.py` (round-trip fidelity, functional equivalence incl. `destination_profiles`, unfitted-raises, both methods, JSON boundary, fail-closed non-vacuity, `save`/`load` file round-trip). Both `singh_counts` and `kde_smoothed` fit non-degenerate on the committed `spadl.json` fixture (400 actions), so the round-trip guard is non-vacuous without an e2e fixture.
