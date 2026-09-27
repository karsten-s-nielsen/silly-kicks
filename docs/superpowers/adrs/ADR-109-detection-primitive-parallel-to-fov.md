# ADR-109: per-player detection is a primitive PARALLEL to polygon-FOV, not unified

**(Workstream label "D1"; placed as ADR-109 in `docs/superpowers/adrs/` 2026-09-27, owner-approved. Written in commit 1 of the detection-primitive cycle.)**

**Status:** Proposed (accompanies `docs/superpowers/specs/2026-09-27-detection-primitive-design.md`, rev 1 APPROVE).

## Context

silly-kicks has two distinct observability signals, answering different questions on different providers:

- **Polygon-FOV** (`tracking/_visibility.py`, `_fov_registry.py`, `restdefense/_fov.py`): a per-frame *region* observed-fraction from a `visible_area` polygon. Produced **only** by StatsBomb-360 (`providers/statsbomb/parse.py`); a continuous threshold `fov_min_observed_fraction=0.7` (in `gk_decision/_config.py:45`, ADR-077 — not the FOV modules; D1-SPEC-02). Answers "was this REGION of the pitch observed?"
- **Per-player detection** (`visibility` / `is_detected` flag): a boolean per player per frame on **detection-aware** providers (`_DETECTION_AWARE_PROVIDERS={skillcorner}`). Answers "was this PLAYER actually detected, or is the position extrapolated?" Consulted today at exactly one site — ghost-GK training, via `_ghost_gk.keeper_detection_mask`.

Generalising per-player detection (TF-58's outfield collective variables and the deferred GK detection-gate both need it) forces a choice: fold it into the polygon-FOV abstraction, or keep it a separate seam.

## Decision

Keep them **parallel**. Add the per-player detection primitive `detected_mask(visibility, *, provider, assume_observed=False)` to the neutral `_provider_visibility.py` (its inputs — the provider sets, `validate_provider`, `assert_detection_aware_visibility`, `_detection_discarded_message` — already live there). Do **not** touch `_visibility.py` / `_fov_registry.py`. Semantics are the existing `keeper_detection_mask` doctrine, generalised: **fail-closed** on detection-aware providers (extrapolated → `False`; entirely-null → raise; unknown provider → raise), with a **general opt-out** (`assume_observed=True` → all-`True`, logged, never the default, never consumer-specific). `keeper_detection_mask` becomes a byte-identical delegating wrapper.

## Alternatives considered

- **Unify into one observability abstraction.** Rejected. The two answer structurally different questions (region-crop fraction vs per-player detection boolean) and apply to **disjoint** providers (SB360 polygon vs SkillCorner flag). A forced union would conflate them, add coupling, and gain nothing — no provider carries both, and the polygon's continuous 0.7 threshold does not transfer to a boolean detected/not gate.
- **Keep detection ghost-GK-private** (leave `keeper_detection_mask` in `_ghost_gk.py`). Rejected. TF-58 (outfield) and the GK detection-gate (keeper wiring) both need it; a general tc3/coordination consumer importing a ghost-model-private keeper function is the exact layering smell the `_provider_visibility` neutral-home precedent (the module's own docstring) removes.

## Consequences

**Positive**
- One neutral primitive serves every detection consumer — ghost-GK training (via the wrapper), TF-58's outfield collective variables, and the future keeper-metric gate — with the fail-closed doctrine single-sourced.
- Byte-identical to the shipped ghost-GK gate (`assume_observed=False`); parity-tested → no retrain, no re-materialise, no Hyrum event from this change.

**Negative**
- Two observability systems coexist; a consumer must know which applies (polygon-FOV = SB360 region cropping; detection = SkillCorner per-player). Mitigated by the clear provider disjointness and the neutral home's docstring.

**Neutral**
- `assume_observed` is the sole escape from fail-closed (per-consumer carve-outs are banned). The keeper wrapper preserves the ghost-GK training path exactly.
- The keeper-metric wiring (the ~13 GK outputs) and TF-58's own wiring are separate, later changes that consume this primitive; this ADR records only the primitive + the parallel-not-unified decision.

## References
- `spec-detection-primitive.md` (this workstream); the parent `spec-gk-detection-gate.md`; `docs/superpowers/specs/2026-07-14-skillcorner-corpus-and-visibility-design.md` (the per-player gate's origin, ghost-GK-scoped).
- `silly_kicks/tracking/_provider_visibility.py`; `silly_kicks/tracking/_ghost_gk.py::keeper_detection_mask`.
