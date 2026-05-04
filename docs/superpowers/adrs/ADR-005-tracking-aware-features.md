# ADR-005: Tracking-aware feature integration contract

| Field | Value |
|---|---|
| **Date** | 2026-05-01 |
| **Status** | Accepted (silly-kicks 2.8.0) |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks 2.7.0 (PR-S19) shipped the `silly_kicks.tracking` namespace
primitive layer (ADR-004): 19-column long-form schema, 4 provider
adapters, `link_actions_to_frames` + `slice_around_event` linkage
primitives. ADR-004 explicitly DEFERRED tracking-aware *features*
(`action_context`, `pressure_on_carrier`, pitch control, etc.) to
follow-up scoping cycles.

silly-kicks 2.8.0 (PR-S20) ships the **first** tracking-aware feature
set: 4 features (`nearest_defender_distance`, `actor_speed`,
`receiver_zone_density`, `defenders_in_triangle_to_goal`) for both
standard SPADL and atomic SPADL, plus full HybridVAEP / AtomicVAEP
integration. The integration shape sets precedent for PR-S21+
tracking-aware features. Locking the cross-cutting decisions here
keeps the next ~10 features (TF-1..TF-10 in TODO.md On-Deck) inside
the same architectural envelope without re-litigation.

## Decision

PR-S20 introduces seven cross-cutting decisions that bind every
PR-S21+ tracking-aware feature.

### 1. Frame-aware xfn protocol — marker decorator + signature

A tracking-aware feature transformer is any callable
`(states, frames) -> Features` marked with the `_frame_aware = True`
attribute via the `@frame_aware` decorator (in
`silly_kicks.vaep.feature_framework`). The mirror predicate
`is_frame_aware(fn) -> bool` is used by VAEP dispatch.

The marker convention parallels the existing `@nan_safe_enrichment`
contract (ADR-003). Setting an attribute on the function object is
preferred over a separate registry because (a) it follows the function
through wrappers / partial application, (b) it requires no global
state, (c) it's introspectable from a debugger.

### 2. VAEP base-class extension via composition, not subclass

`VAEP.compute_features` and `VAEP.rate` gain an additive `frames=None`
keyword-only parameter. When `frames is None`, behavior is bit-identical
to historical (regression-tested in `tests/vaep/test_compute_features_frames_kwarg.py`).
When `frames` is supplied, the iteration over `self.xfns` dispatches
each xfn either as `fn(states)` (regular) or `fn(states, frames)`
(frame-aware), keyed on `is_frame_aware(fn)`. A `ValueError` is raised
if a frame-aware xfn appears in the list but `frames is None`, so the
mismatch surfaces loudly at the boundary rather than as a silent
degradation.

`HybridVAEP` and `AtomicVAEP` inherit the extension without code
changes — both call into `VAEP.compute_features` via `super()` /
inheritance and gain frames-pass-through automatically. This avoids a
class-per-feature-kind explosion (no `TrackingVAEP`, `TrackingHybridVAEP`,
`TrackingAtomicVAEP` subclasses) and gives atomic-SPADL parity for free.

### 3. Kernel-extraction pattern — `silly_kicks/tracking/_kernels.py`

The compute logic of all 4 PR-S20 features lives in a single private
module, `silly_kicks.tracking._kernels`. Each kernel takes
`(anchor_x, anchor_y, ctx)` (or just `ctx` for `actor_speed`) and
returns a `pd.Series` indexed by the input actions. The kernels are
schema-agnostic — they don't read SPADL column names directly; the
caller supplies the right anchor columns.

Per-schema wrapper modules (`silly_kicks.tracking.features` for
standard SPADL, `silly_kicks.atomic.tracking.features` for atomic SPADL)
read the right columns and call the same kernel. PR-S21+ features
follow the same pattern: kernel in `_kernels.py`, schema wrappers in
the per-schema `features.py`.

### 4. Schema-aware-via-per-namespace-wrapper

Standard SPADL has `start_x, start_y, end_x, end_y`; atomic SPADL has
`x, y, dx, dy`. Rather than a runtime schema dispatcher inside each
public function, silly-kicks publishes two parallel public surfaces
(`silly_kicks.tracking.features` and `silly_kicks.atomic.tracking.features`)
that mirror the existing `silly_kicks.spadl.utils` /
`silly_kicks.atomic.spadl.utils` precedent.

Rationale over the dispatcher alternative: typed signatures stay
clean, discoverability is per-namespace, and the wrapper layer is
~10 lines per feature — adding a schema dispatcher would cost more
LOC and lose static-typing fidelity for atomic-specific edge cases
(e.g., the `dx == dy == 0` instantaneous-action degenerate case
documented in `atomic.tracking.features.receiver_zone_density`).

### 5. LTR-normalization symmetric internal call

When `VAEP.compute_features(frames=...)` is invoked, both `states` and
`frames` are normalized to left-to-right via the matching pair of
`play_left_to_right` calls — `silly_kicks.spadl.utils.play_left_to_right`
on the actions side (already happened pre-PR-S20) and
`silly_kicks.tracking.utils.play_left_to_right` on the frames side
(new in PR-S20). The latter is **lazy-imported** inside the
`if frames is not None` branch so importing
`silly_kicks.vaep.base` alone (without tracking) never triggers a
tracking-namespace import — no module-import-time vaep ↔ tracking
cycle. This is regression-tested in
`test_no_module_import_cycle_when_frames_is_none`.

The opposite direction (tracking → vaep) is a regular import — every
tracking feature depends on `frame_aware` / `is_frame_aware` from
`silly_kicks.vaep.feature_framework`. The cycle would close only if
vaep eagerly imported tracking; the lazy import keeps it open.

### 6. Action-anchor for geometric features

Geometric features (`nearest_defender_distance`,
`receiver_zone_density`, `defenders_in_triangle_to_goal`) anchor on
the **action's** start/end coordinates, NOT on the actor's position
in the linked frame. Only `actor_speed` reads the actor's frame row.

Rationale:

- **Coordinate alignment.** Provider event coords and tracking coords
  may differ slightly (~0.5-2m typical drift) due to capture geometry
  and clock skew. Anchoring on the action's start_x/y keeps geometric
  features consistent with VAEP's existing event-driven feature space.
- **Preserves event-side determinism.** Two callers with identical
  actions but different tracking sources get identical geometric
  features (modulo defender positions). Anchoring on the actor's frame
  position would couple geometric features to actor's frame coord —
  which has provider-dependent jitter.
- **Documented degenerate case.** Atomic-SPADL actions with
  `dx == dy == 0` (e.g. shots) have end == start; `receiver_zone_density`
  in that case computes density at the anchor (covered by
  `test_atomic_zero_dx_dy_is_degenerate_density`).

### 7. Linkage-provenance pass-through — audit by default

`add_action_context` (the standard + atomic public aggregator) joins
four provenance columns from `link_actions_to_frames` pointers into
its output:

- `frame_id` (Int64; NaN if unlinked)
- `time_offset_seconds` (float64; NaN if unlinked)
- `n_candidate_frames` (int64)
- `link_quality_score` (float64; NaN if unlinked)

Every consumer of the aggregator gets full audit-by-default — they
can filter on `link_quality_score >= 0.9` for high-confidence
analysis, count unlinked actions via `frame_id.isna()`, or trace any
feature value back to its source frame. Without this convention, downstream
pipelines would re-compute the linkage to recover the same audit trail.

### 8. Multi-flavor xfn column naming convention

Added in PR-S25 (silly-kicks 3.2.0). When a single feature concept admits
multiple methodologies (multi-flavor xfns — first concrete instance:
PR-S25 `pressure_on_actor` with three methods `andrienko_oval` / `link_zones`
/ `bekkers_pi`):

- Each flavor MUST emit a flavor-suffixed column name `<feature>__<method>`
  (double-underscore separator) so consumers registering parallel
  `functools.partial(fn, method="X")` xfns in `VAEP.xfns` do not silently
  overwrite each other inside `VAEP.compute_features`.
- The default xfn list (`<feature>_default_xfns`) ships exactly ONE flavor
  (the default method) to keep the VAEP feature space stable across
  silly-kicks versions. Consumers wanting additional flavors register
  additional `functools.partial` xfns explicitly.
- Per-method parameters are passed via a flavor-specific frozen dataclass
  (e.g., `AndrienkoParams`, `LinkParams`, `BekkersParams`) on the `params=`
  kwarg, not as a flat keyword bag — keeps each flavor's parameter
  surface discoverable and statically typed (pyright-friendly), and
  allows `__post_init__` validation per flavor.
- If `params=` is supplied with a type not matching the chosen `method`,
  the public function raises TypeError loudly (no silent default fallback).

This pattern applies only to **VAEP-consumed xfns** that emit per-action
scalar features. Preprocessing utilities that produce canonical per-row
columns (e.g., `derive_velocities` emits `vx`, `vy`, `speed`) keep their
canonical-single-column names regardless of method; QA/inspection helpers
case-by-case (per `feedback_multi_flavor_xfn_column_names`).

Rationale recap: the suffix convention surfaces the methodology choice in
the column name (downstream debugging is easier — column-name self-documents
which formula produced each value), and the default-xfn-list-ships-one-flavor
rule prevents "oh, did adding Link to the default break my model artefact?"
regression after consumer updates.

## Consequences

### Positive

- **PR-S21+ inherits the contract.** Every TF-1..TF-10 feature in
  TODO.md On-Deck plugs into the framework with the same shape:
  per-feature kernel + per-schema wrapper + `lift_to_states` for VAEP
  integration.
- **Atomic-SPADL parity is free.** `AtomicVAEP` inherits the
  `compute_features(*, frames=None)` extension verbatim (zero code
  changes in `silly_kicks/atomic/vaep/base.py`); only the per-namespace
  wrapper module is new.
- **No vaep ↔ tracking module cycle.** Lazy import keeps the
  dependency direction correct: tracking depends on vaep
  (`feature_framework.frame_aware`), vaep does not depend on tracking
  at module-import time.
- **Backward-compat preserved.** Every existing call site
  (`v.compute_features(game, actions)`) is bit-identical to today —
  `frames=None` is the default and walks the same code path.
- **Audit by default.** The 4 provenance columns mean any analytical
  downstream gets the linkage trail for free.

### Negative

- **Two parallel public surfaces** (`silly_kicks.tracking.features` +
  `silly_kicks.atomic.tracking.features`) mean documentation duplication
  for each feature's docstring. Mitigated: both surfaces share the
  same kernel + reference the same NOTICE entries; the wrapper layer
  is mechanical.
- **Marker-attribute dispatch is not statically typed.** A future
  contributor could write a frame-aware xfn but forget the decorator;
  it would silently run as a regular xfn (passing `gamestates` only,
  with the second-positional argument bound to whatever happened to be
  next). Mitigated: a frame-aware xfn that doesn't receive frames raises
  `TypeError` from inside the xfn (it's expected to take 2 positional
  args), surfacing the bug at first call.

### Bundled refactor

`silly_kicks._nan_safety` gained an `is_nan_safe_enrichment(fn)` peer
predicate to the existing `nan_safe_enrichment` decorator. This
mirrors the new `is_frame_aware(fn)` introduced in PR-S20 and gives
a clean introspection API for both contracts. ~5 LOC; pure refactor.

## References

- `docs/superpowers/specs/2026-04-30-action-context-pr1-design.md` — full PR-S20 design.
- `docs/superpowers/plans/2026-04-30-action-context-pr1.md` — implementation plan.
- ADR-001 — converter identifier conventions (cross-namespace consistency precedent).
- ADR-003 — NaN-safety contract for enrichment helpers (sets the marker-decorator precedent).
- ADR-004 — `silly_kicks.tracking` namespace charter (the 9 invariants PR-S20 inherits).
- Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). "Actions Speak Louder Than Goals." Proc. KDD '19. (Foundational VAEP.)
- Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2014). "Quality vs Quantity." MIT Sloan SAC. (Defenders-in-triangle origin.)
- Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on synchronized positional and event data." Frontiers in Sports and Active Living, 3, 624475.
- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan SAC.
- Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017). "Not all passes are created equal." KDD '17 (OBSO).
- Pollard, R., & Reep, C. (1997). "Measuring the effectiveness of playing strategies at soccer." J. Royal Statistical Society Series D, 46(4), 541-550.
- See `NOTICE` for full bibliographic citations.
