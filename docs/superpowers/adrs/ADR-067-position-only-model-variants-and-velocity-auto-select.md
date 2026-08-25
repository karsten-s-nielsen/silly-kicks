# ADR-067: Position-only model variants + velocity-keyed auto-select

> Spec: `docs/superpowers/specs/2026-08-23-position-only-model-variants-and-velocity-auto-select-design.md`.
> Plan: `docs/superpowers/plans/2026-08-23-position-only-model-variants-and-velocity-auto-select.md`.

## Status

Accepted (implemented). Depends on ADR-054 (velocity-availability contract), ADR-063 (velocity-less
positional lift), ADR-011/016/040/044/050 (parameters-only fail-closed artifacts), ADR-033 (`add_*`
purity), ADR-048 (feature glossary), ADR-052 (corpus-driver shard seam).

## Context

`XShotOccurrenceModel`, `XCrossAttemptModel` and `GhostGkModel` each carry a minority of
velocity-derived features. As of 4.90.0 (ADR-054) they **honest-NaN** on velocity-less StatsBomb-360
freeze-frames: the contract stopped them *fabricating* a value from a structurally-absent feature, but
left them producing *nothing*. The unlock — making them *produce* a value on SB360 — requires
position-only variants (the same models re-fit with velocity features dropped), which the 4.90.0 spec
explicitly deferred to "a separate future cycle with DGX training."

Velocity is a small minority of each feature set: xShot drops **1 of 27** (`speed`), xCross **1 of 16**
(`ball_speed`), ghost **5 of 26** (`ball_vx`/`ball_vy`/`ball_speed` + the two cross-frame temporal
derivatives `defensive_line_speed`/`defending_centroid_vx`). So position-only variants are viable —
most signal is positional.

## Decision

1. **Position-only variants, feature-set-selected from ONE extractor.** Each extractor's `feature_set`
   literal is EXTENDED to `Literal["faithful", "extended", "position_only"]` (ghost, which had no
   `feature_set`, gains `Literal["faithful", "position_only"]`); `"extended"` keeps its
   `NotImplementedError` (Chesterton's Fence — a reserved roadmap slot we were not asked to touch).
   Velocity features are **dropped** (a shorter vector), never NaN-filled — the feature contract raises
   on non-finite. Ghost's position-only path is genuinely **single-frame-capable** (it skips the
   `prev_state` cross-frame bookkeeping), which is what a lone freeze frame needs.

2. **Velocity-keyed auto-select at the serve seam** (mirrors the proven provider-keyed
   `variant_key_for_provider` / `_resolve_completion_for_frames` pattern). Two layers: a pure 2-way key
   `variant_key_for_velocity(frames)` (Layer A, in `_velocity_availability.py`), and a per-model
   `(model, variant_key)` resolver (Layer B). Resolution: explicit `model=` override → `"custom"`;
   declared-velocity-unavailable → `position_only`; else → `default`.

3. **The missing-variant fallback goes to NaN, NEVER to default** — the load-bearing asymmetry with the
   completion template. The default velocity model is *invalid* on velocity-less frames (running it
   fabricates from `speed=NaN`), so an unbundled `position_only` degrades to NaN, not to `default`.

4. **Mixed-availability frame sets RAISE.** `velocity_unavailable_by_design` is an all-rows predicate
   (False on a partially-marked set), so a new `velocity_availability_is_mixed` predicate catches a
   mixed set and the serve seam raises — otherwise a mixed set would resolve to `default` and fabricate
   `speed=NaN` on the marked rows (the ADR-054 defect reappearing on mixed frames).

5. **The ADR-054 undeclared-missing-velocity RAISE is preserved**, and lives in `compute_*` (not the
   resolver): a forgotten `derive_velocities()` still fails loud; only the *declared* marker triggers
   auto-select.

6. **Provenance is a closed vocabulary `{default, position_only, custom}`** on the `add_*` path
   (`xshot_occurrence_variant` / `xcross_attempt_variant` / `ghost_gk_variant`), re-resolved
   deterministically from the frames (the `das_source` stamped-column shape, NOT the open
   `xt_gk_completion_variant`). `compute_*` output stays byte-identical (the column is added by
   `add_*`, which re-resolves) — so velocity-bearing direct callers are unchanged, and the `*_xfns` /
   VAEP path stays numeric.

## Consequences

- **Behavior change (disclosed retrain / Hyrum trigger):** on declared velocity-less (SB360) frames,
  `xshot_occurrence` / `xcross_attempt` / `ghost_gk_x`/`_y` move from NaN to a real position-only value
  (once the variants are bundled); before bundling they honest-NaN with the chosen-variant provenance.
  **Velocity-bearing frames are byte-identical.** gkdv's ghost arm — which drops SB360 frames today —
  begins to work there (it consumes `serve_ghost_gk_positions`, which now serves position-only).
- **`add_ghost_gk`'s own guard was restructured to preserve a measured fence.** Its documented
  fabrication guard (the `ghost_gk_x = 52.5` defect) is preserved by disabling the precompute
  short-circuit on declared frames (never trust a pre-computed ghost that bypassed the serving seam)
  rather than by the old NaN short-circuit; the `ghost_gk_source` finalization gains a
  `velocity_unavailable` branch for the declared-unbundled degrade.
- **Per-model asymmetries, recorded:** xShot has 3 guard sites (extract/init/prepare), xCross 2
  (extract/init — no prepare guard); ghost never had a `feature_set` (it is ADDED here, incl. save/load
  serialization). The load-guard blocks are threaded by `feature_set` (`_chirality_block` reads
  `model.feature_set`; `_feature_contract_block` gains a `feature_set` parameter).
- **Trainers gain `--feature-set position_only`**, threaded to `prepare_*` + the model + the
  `for_each` **shard-generation token** (feature_set changes the X columns, so it must key the shards or
  a faithful run's shards get reused — the 4.77.1 stale-shard trap). A reported (not gated)
  comparability artifact quantifies the velocity-vs-position-only skill cost.
- **`feature_set` threads through EVERY feature-selection site, not just the model.** Execution (the
  trainer-subprocess smokes below) surfaced that a shorter position-only vector `KeyError`s wherever the
  FAITHFUL name list was hardcoded: `prepare_*`'s final column-select and width-check, the model
  `save()` metadata `feature_names`, ghost's `fit`/`predict_mean`/`predict_density` positional selects
  (routed through a new `GhostGkModel._feature_names()`) and its `_extract_all` empty fallback, AND the
  **xcross evaluator + substitution probe** — `_xcross_eval.gk_block_ablation` /
  `permutation_importance_report` (now derive columns from `X.columns`, feature-set-agnostic) and
  `_model_eval._extract_kwargs` (now threads the scored `model.feature_set`, a no-op for faithful so the
  golden pin holds). xShot's fit/predict are already agnostic (they key on `features.columns`); ghost's
  predict was positional and needed the helper.
- **Each variant path is covered by a trainer-subprocess `@slow` smoke** (`train_xshot_occurrence` /
  `train_xcross_attempt` / `train_ghost_gk` with `--feature-set position_only`), plus per-model
  `fit→save→load→predict` round-trips and `prepare`/empty-fallback unit tests. The subprocess smokes are
  what caught the eval/probe hardcoded-FAITHFUL sites the model-level tests could not see.

## Alternatives considered

- **Explicit opt-in (no auto-select).** Rejected: the position-only variant is the *only* valid scorer
  on a freeze frame, so serving NaN when it exists is the wrong default; auto-select is correctness,
  not convenience, and mirrors the house pattern.
- **Tuple return / internal `__variant` column on `compute_*`.** Rejected: both change `compute_*`
  output for velocity-bearing direct callers (a Hyrum break). `add_*` re-resolving the key
  (deterministic; `from_variant` cached) leaves `compute_*` byte-identical.
- **SB360-native training.** Rejected: 30 licensed matches, goalkick frame-existence ~44% — far too
  small; the variants train on the full-tracking corpus with velocity dropped and serve on SB360.

## Amendment — Phase B bundle: gkdv on SB360, fixture realism, the ADR-053 gate

Bundling the position-only ghost made gkdv's counterfactual arms *reachable* on SB360 for the first
time (`serve_ghost_gk_positions` now serves via the position_only variant), which surfaced three
follow-ons, all folded into the bundle commit:

- **`delta_das` degrades honestly; `delta_threat_suppression` computes.** Accessible space
  STRUCTURALLY requires velocity, so on a velocity-less freeze frame `delta_das` now catches
  `DasUnscoreableError` → NaN (the ADR-043 consumer-degrade `add_das` uses via `das_source`) instead
  of propagating a crash. `delta_threat_suppression` needs no such degrade — pitch control has a valid
  zero-velocity positional model (ADR-063) — so it produces a positional-only value. An interim "gkdv
  declines velocity-less frames wholesale" workaround was **rejected and reverted**: it dodged the
  audit symptom instead of giving each arm its correct behaviour.
- **The SB360 audit fixture gained a realistic striker (`sb360-fixture-2`).** With no attacking
  receiver ahead of the ball the threat arm is legitimately 0 on both legs, so `delta_threat` read a
  masked `0==0` `identical` that certified nothing. A central striker ahead of the ball makes the arm
  non-vacuous — the keeper's threat-suppression now measurably moves the value across legs — which is
  the property the audit exists to check.
- **The ADR-053 provenance gate was refined for the first MIXED boundary entry.** `delta_threat` is
  substantive where there is threat but a coincidental `0==0` `works` on the no-threat goalkick roster
  — a mix the gate's per-cell `works→structural` lock could not express. The refinement: an entry is
  `substantive` if ANY cell moves (`differs_by_design`/`silent_degrade`), exempting coincidental
  `works` cells; an entry with NO substantive cell still locks `works→structural`, so genuinely
  frame-blind entries (`add_restart_coordinates`, `xt_gk_v2`) are unaffected. See ADR-053.
