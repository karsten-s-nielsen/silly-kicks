# ADR-121: Retire xT-GK v1

| Field | Value |
|---|---|
| **Date** | 2026-10-13 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner, 2026-10-10) |

## Context

xT-GK v1 (ADR-024) is a FROZEN, intent-set parametric goalkeeper-distribution value surface
(`tracking/_xt_gk.py` + `tracking/_gk_completion.py` + the `GkCompletionModel` bundled weights, exposed
as `compute_xt_gk` / `add_xt_gk` / `xt_gk_xfns` / `XtGkParams` / `native_origin_is_trusted` / the
`resolve_gk_geometry` goal-kick shim). xT-GK v2 (ADR-036, `xtgk/`) is a Markov possession-value surface
that supersedes it; both shipped side-by-side pending the lakehouse's column migration.

The forcing function is the TF-65 velocity-gap re-fit (ADR-120, 4.134.0): the dense-grid fix moved
broadcast-provider velocities, so every velocity-dependent bundled model needed a re-fit. `GkCompletionModel`
reads frame-derived features via `resolve_gk_geometry`, so v1 would have required a wasted retrain purely to
stay on the corrected velocities — a model we intend to delete. v2 reads no `vx`/`vy`/`speed`, so it is
unaffected by the velocity change. Retiring v1 instead of retraining it is the cheaper, honest path.

v1 has no bundled downstream it must keep alive inside the repo (it is in no default xfn list); the only
external consumption is the lakehouse `xt_gk*` columns, whose migration to v2 the owner coordinates
cross-repo and has stated is not a blocker.

## Decision

Remove xT-GK v1 entirely — the `_xt_gk.py` / `_gk_completion.py` modules, the `_gk_completion_weights`
bundle, the v1-exclusive `_gk_geometry.py` shims (`native_origin_is_trusted` / `resolve_gk_geometry` /
`flag_native_goalkick_out_of_region`), all re-exports, the v1 trainer/validator/comparability scripts, the
v1 tests, and the `feature_glossary` / `NOTICE` / `docs/context` / `AGENTS.md` / `PRIVATE_CONSUMERS` entries
— and keep xT-GK v2 as the sole xT-GK surface. The ADR-025 restart-coordinate engine
(`resolve_restart_geometry` and its helpers in `_gk_geometry.py`) is a separate, live feature and is
preserved untouched.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Retrain v1 on the corrected velocities | Keeps v1 available | A full re-fit + re-bundle + card push for a model we intend to delete; carries v1 debt forward | Wasted compute + provenance churn on a superseded model |
| B. Freeze v1 un-retrained (leave stale weights) | No work now | v1 weights silently out-of-step with the 4.134.0 velocity floor — a stale-artifact trap; v1 debt persists | Honest-staleness hazard; defers the removal indefinitely |
| C. Retire v1 (chosen) | Deletes a superseded surface + its debt; no wasted retrain; v2 unaffected (velocity-blind) | A public-surface removal (Hyrum); trips repo-wide enumeration gates; cross-repo lakehouse migration owed | — |

## Consequences

### Positive

- One xT-GK surface (v2), no frozen parametric duplicate to maintain or explain.
- No wasted v1 retrain in the TF-65 cluster; the velocity re-fit touches only the 5 live frame families.
- Removes the `_xt_gk` / `_gk_completion` module-path pins from the lakehouse drift guard once migrated.

### Negative

- **Hyrum's Law:** `compute_xt_gk` / `add_xt_gk` / `xt_gk_xfns` / `XtGkParams` / `XtGkReport` /
  `GkCompletionModel` / `compute_gk_completion` / `resolve_gk_geometry` / `native_origin_is_trusted` and the
  `xt_gk*` / `gk_completion` glossary columns are gone — any out-of-repo consumer breaks until it moves to v2.
- The cross-repo lakehouse `xt_gk*`-column migration to v2 is now owed (owner-coordinated, not a repo task).
- The v1 trainer/validator/comparability scripts and their corpus-driver/provenance gate rows are removed,
  shrinking the enumerated script population (ADR-052/056 gates reconciled in the same change).

### Neutral

- `resolve_restart_geometry` (ADR-025) keeps its `distrust_native_origin` parameter and the
  `_tracking_gk_xy_detected` helper; their removal, if ever warranted, is a separate refactor.
- The `agents_md_invariant_inventory.json` `xt-gk-v1` entry is marked `dropped_with_owner_approval: true`
  rather than deleted, so the frozen `@77286f4` snapshot check stays honest while the live-findability check
  is released.
