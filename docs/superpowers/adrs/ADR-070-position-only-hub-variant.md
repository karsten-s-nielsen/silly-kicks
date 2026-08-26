# ADR-070: Position-only Hub variant for xShot / xCross (`sc_extended_position_only`)

> Follows ADR-067 (position-only variants + velocity-keyed auto-select), ADR-011/016/040/050
> (parameters-only fail-closed artifacts).

## Status

Accepted (implemented, 4.94.0).

## Context

ADR-067 added **position-only** variants (velocity features dropped) that serve velocity-less
StatsBomb-360 freeze frames, and bundled them in the wheel trained on the PUBLIC corpus. The
higher-quality **owner-tier** models cannot be bundled (licensing), so they live on the HuggingFace
Hub: `sc_extended` for xShot/xCross (repos `xshot-occurrence-v1` / `xcross-attempt-v1`), `full` for
ghost. Those Hub artifacts are **faithful** (velocity-bearing) and reachable ONLY by an explicit
`from_variant("sc_extended")` / `model="sc_extended"` call — the velocity-keyed auto-select
(`variant_key_for_velocity`) can NEVER produce a Hub key.

We want the owner-tier **position-only** xShot/xCross models on the Hub as well (a stronger
position-only model than the bundled public one, for owner-tier SB360 serving).

## Decision

**A position-only Hub model is a SEPARATE variant key + SEPARATE repo, never a re-fit of the existing
`sc_extended` slot.** `_HUB_VARIANTS` gains `sc_extended_position_only`, and a new `_HUB_REPOS` dict
maps each Hub key to its repo (`sc_extended` → `*-v1`; `sc_extended_position_only` →
`silly-kicks/{xshot-occurrence,xcross-attempt}-position-only-v1`). `from_variant` routes each key to
`from_hub(_HUB_REPOS[variant])`.

The separation is the whole safety argument, and it is forced by the resolver's shape:

1. **`from_variant("sc_extended")` still returns the FAITHFUL model.** Every existing consumer asked
   for `sc_extended` by name, relying on the model-card faithful PR-AUC/Brier. Overwriting that slot
   with a position-only fit would silently feed a velocity-bearing caller a velocity-less model —
   `speed`/`ball_speed` dropped from the extracted vector, no error, no warning (the explicit-override
   path does no velocity check). A separate key leaves that caller untouched.

2. **The position-only Hub model is reachable ONLY by its own explicit name.** Auto-select never
   routes to a Hub key, so a consumer receives `sc_extended_position_only` only by asking for it —
   an informed choice, documented in the model card as velocity-less.

3. **The publish scripts are feature-set-aware.** `publish_xshot_occurrence.py` /
   `publish_xcross_attempt.py` previously hard-coded the faithful feature list for their verify
   sample; a position-only artifact (26/15 cols vs 27/16) would raise an xgboost feature-count
   mismatch. They now pick the column set from the artifact's `feature_set`, and call
   `create_repo(exist_ok=True)` before upload so the new repo is created on first publish.

## Consequences

- **Additive; no retrain trigger; no change to any bundled or existing Hub artifact.** `sc_extended` /
  `full` keep serving faithful; the new key is inert until the position-only artifacts are published.
- **Ghost is unchanged.** Its position-only variant is bundled (owner-tier, in the wheel); its Hub
  variant `full` stays faithful. There is no `sc_extended_position_only` for ghost.
- Model cards gain a `sc_extended_position_only` section documenting the velocity-less intent.
- Guarded by `tests/tracking/test_from_variant_serve_identity.py` (per-key routing + the feature-set-
  aware publish verify on the bundled position-only artifact).

## Alternatives considered

- **Overwrite the `sc_extended` slot with a position-only fit.** Rejected — silently breaks every
  explicit consumer of the faithful model (Decision 1). This is the interpretation the earlier
  investigation flagged as a consumer-contract hazard.
- **A velocity-availability check on the explicit-override path.** Deferred — a separate key already
  removes the silent-substitution hazard, and an explicit `model="sc_extended_position_only"` request
  is by definition informed. A general "warn if a position-only model is served on velocity-bearing
  frames" guard is a broader change than this variant needs.
