# ADR-002: Shared VAEP feature framework boundary

| Field | Value |
|---|---|
| **Date** | 2026-04-30 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks ships two VAEP feature stacks: the standard SPADL stack
(`silly_kicks.vaep.features`) and the atomic SPADL stack
(`silly_kicks.atomic.vaep.features`). Both build on a small set of
genuinely-shared framework primitives: 4 type aliases (`Actions`,
`Features`, `FeatureTransfomer`, `GameStates`), the `gamestates`
gamestate-construction helper, the `simple` decorator that lifts an
action-level feature function to a gamestate-level transformer, and a
SPADL-config-parameterized helper for categorical action-type encoding.

Through silly-kicks 2.3.0, those primitives lived inside
`silly_kicks.vaep.features.core`. Atomic-VAEP imported them via
`from silly_kicks.vaep.features.core import (...)`, including a
leading-underscore-private symbol `_actiontype`. The cross-package
private import was a structural smell: documented-private surface used
by another package's public-facing module.

PR-S15 (silly-kicks 2.3.0) decomposed the standard `vaep/features.py`
monolith into an 8-submodule package and explicitly deferred A9 closure
with the trigger condition: "extracting truly-shared framework into a
cross-package module." With 2.3.0 stable, the framework primitives'
boundary had crystallized and the trigger condition became actionable
as a small, well-bounded PR.

## Decision

Introduce a new public module `silly_kicks/vaep/feature_framework.py`
as the named cross-package boundary that both standard and atomic VAEP
feature stacks build upon. It holds 7 symbols: 4 type aliases,
`gamestates`, `simple`, and the promoted helper
`actiontype_categorical(actions, spadl_cfg)` (was `_actiontype`).

Atomic-VAEP imports framework primitives directly from
`silly_kicks.vaep.feature_framework`. Standard-VAEP's
`silly_kicks.vaep.features.core` keeps its standard-SPADL-specific
helpers (`play_left_to_right`, `feature_column_names`) and re-exports
the framework primitives so existing
`from silly_kicks.vaep.features.core import gamestates` paths continue
to resolve. Atomic-VAEP's per-concern coupling to `bodypart`, `context`,
`temporal` submodules is preserved — that coupling is intentional
verbatim feature reuse, not framework leak. T-C is rewritten to forbid
reaching into `vaep.features.core` for framework primitives and to
require import from `vaep.feature_framework`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Cosmetic flatten — collapse atomic's 4 per-concern imports to one package-root import (`from silly_kicks.vaep.features import ...`) | minimal churn | T-C added in 2.3.0 actively forbids package-root import to prevent monolith-coupling regression; this option fights an existing test on purpose | rejected: the codified rule is correct, not the option |
| B. Keep `_actiontype` private but move to framework module | smaller surface | doesn't resolve the cross-package private-import smell — atomic still reaches into another package's documented-private surface | rejected: solves the location problem without the naming problem |
| C. Document the deferral as ADR'd design decision; close A9 as won't-do until atomic divergence triggers | honest about YAGNI | leaves the cross-package private-import smell in place; the trigger condition's stated resolution mechanism (framework extraction) is itself low-cost, so deferring it indefinitely is harder to justify than doing it once | rejected: the framework-extraction work is small and yields a real architectural boundary |
| D (chosen). Extract framework to public `silly_kicks.vaep.feature_framework`; promote `_actiontype` to public `actiontype_categorical(actions, spadl_cfg)` | clean cross-package boundary; named module; atomic stops reaching into `core`; future atomic divergence has a stable seam | introduces one new public module + one new public function — additive Hyrum's-Law surface; `_actiontype` rename is a true breaking change for any consumer who imported the private symbol | — |

## Consequences

### Positive

- Cross-package framework boundary has a name (`silly_kicks.vaep.feature_framework`) — easier to reason about, easier to extend.
- Atomic-VAEP no longer reaches into another package's documented-private surface for `_actiontype`.
- `actiontype_categorical(actions, spadl_cfg)` is now a discoverable public helper. External consumers can use it to build custom categorical actiontype features against any SPADL config (standard, atomic, or hypothetical future variants).
- Future atomic-VAEP divergence has a stable framework seam to lean on. When atomic's `bodypart` / `context` / `temporal` features eventually need to differ from standard's, the framework dependency is already isolated and named.
- Closes A9 — the last open Architecture row in TODO.md.

### Negative

- `gamestates.__module__` (and `simple.__module__`) flips from `silly_kicks.vaep.features.core` to `silly_kicks.vaep.feature_framework`. Any consumer introspecting via `inspect.getmodule(gamestates)` would see the new value (Hyrum's Law exposure). Mitigation: T-D codifies the new canonical home; the `__module__` flip is explicitly noted in the 2.4.0 CHANGELOG.
- `_actiontype` rename is a true breaking change for any consumer who imported the leading-underscore symbol directly. Mitigation: the function was never in `__all__` and never documented as public; consumers who imported it accepted instability per Python convention. The functional replacement is `actiontype_categorical(actions, spadl_cfg)`.

### Neutral

- The `vaep/features` package's external surface (`__all__`) gains exactly one symbol: `actiontype_categorical`. Net additive — every previously-public symbol stays accessible at the same path.
- ADR-002 follows ADR-001's lakehouse-vendored template; future ADRs on silly-kicks architectural decisions continue this pattern.

## CLAUDE.md Amendment

None. ADR-002 documents the framework-boundary decision; CLAUDE.md "Key conventions" already covers ML naming and no-pandera. The framework boundary doesn't rise to the level of a project-wide rule — it's the natural consequence of "name shared things and locate them where the share is named."

## Related

- **Spec:** `docs/superpowers/specs/2026-04-30-A9-feature-framework-extraction-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-30-A9-feature-framework-extraction.md`
- **Predecessor spec:** `docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md` (PR-S15, 2.3.0 — established the trigger condition this ADR resolves)
- **ADRs:** ADR-001 set the silly-kicks ADR pattern (vendored from luxury-lakehouse)
- **Issues / PRs:** silly-kicks PR-S16 (this PR — closes A9)

## Notes

### `_actiontype` → `actiontype_categorical` rename rationale

The rename does three things at once:

1. **Drops the leading underscore** — the helper is now genuinely public and used cross-package, so it deserves a stable public name.
2. **Tightens the parameter contract** — old signature was `_actiontype(actions, _spadl_cfg=None)` with module-level fallback to `silly_kicks.spadl.config`. New signature is `actiontype_categorical(actions, spadl_cfg)` (positional, required). The function is meaningless without a config; the implicit None fallback was hiding that.
3. **Names the output** — `categorical` in the new name describes what the function returns (a Categorical column). Descriptive function names are cheaper to understand than ones that just describe the input domain.
