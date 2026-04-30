# ADR-003: NaN-safety contract for enrichment helpers

| Field | Value |
|---|---|
| **Date** | 2026-04-30 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks's post-conversion enrichment helpers (`add_possessions`,
`add_names`, `add_gk_role`, `add_gk_distribution_metrics`,
`add_pre_shot_gk_context`, plus atomic counterparts — 10 functions total)
operate on caller-supplied SPADL DataFrames. ADR-001 (silly-kicks 2.0.0)
commits silly-kicks to caller's identifier conventions: converters never
override the caller's `team_id` / `player_id`. That sacred-pass-through
implies enrichment helpers must be robust to NaN identifiers — caller
data shapes vary by provider, and StatsBomb-style dense attribution is
not the universal case.

On 2026-04-30, the luxury-lakehouse daily ingestion job's first
end-to-end run on real IDSSE bronze data hit
`ValueError: cannot convert float NaN to integer` at
`silly_kicks/spadl/utils.py:543` in `add_pre_shot_gk_context`:

```python
gk_id = int(player_id[window_start + relative_indices[-1]])
```

When the most-recent defending-keeper-action's row has NaN `player_id`,
`int(NaN)` raises. A symmetric latent crash exists in
`add_gk_distribution_metrics` at `silly_kicks/spadl/utils.py:374-377`
(zone-binning `.astype(int)` on possibly-NaN coordinates). The atomic
package mirrors both bugs at `silly_kicks/atomic/spadl/utils.py:826`
and `665-668`. A defensive variant lives in `coverage_metrics`
(`int(tid)` at line 1074 std / 1036 atomic) — same crash class
on NaN type_id.

The deeper problem: silly-kicks did not have an explicit, codified,
CI-enforced NaN-safety contract for enrichment helpers. Each helper
made ad-hoc decisions about NaN handling; some were accidentally
NaN-safe (`add_gk_role` uses `==` comparison, which is NaN-safe),
some were accidentally NaN-unsafe. Without a contract + a
self-enforcing perimeter, the next helper a contributor adds is just
as likely to repeat the bug.

## Decision

silly-kicks codifies a **NaN-safety contract for public enrichment
helpers**. A NaN-safe enrichment helper is a public function
`add_*(actions: pd.DataFrame, ...) -> pd.DataFrame` that satisfies all of:

1. **No crash on NaN identifiers.** For every column in the input that
   is a caller-supplied identifier (`player_id`, `team_id`, `game_id`,
   `period_id`, `action_id`), NaN values do not raise. Internal logic
   that relies on the value detects NaN and routes to the per-row
   default.
2. **No crash on NaN numerics.** For every numeric column the helper
   internally casts to integer (e.g. coordinate columns flowing into
   zone-binning), NaN values do not raise. Affected rows are excluded
   from the cast-dependent computation; their output column receives
   NaN/default.
3. **NaN preservation in identifier outputs.** When the helper outputs
   an identifier-derived column (e.g. `defending_gk_player_id`), NaN
   inputs that prevent identification produce NaN outputs at those rows.
4. **Documented NaN-input semantics.** The helper's docstring contains
   an explicit sentence describing what happens when an input row has
   NaN in an identifier column.
5. **`@nan_safe_enrichment` decorator applied.** The decorator from
   `silly_kicks._nan_safety` sets `fn._nan_safe = True`; CI gates
   auto-discover decorated helpers via this attribute.

Enforcement is two-pronged:

- **Auto-discovered fuzz** (`tests/test_enrichment_nan_safety.py`)
  parametrizes over every decorated helper × a synthetic NaN-laced
  fixture; asserts no crash + sensible defaults.
- **Cross-provider e2e** (`tests/test_enrichment_provider_e2e.py`)
  parametrizes over every decorated helper × vendored production-shape
  fixtures from each supported provider (StatsBomb / IDSSE / Metrica);
  asserts no crash on real production data shapes.

Both gates include **registry-floor sanity assertions** that fail if the
auto-discovery silently regresses (e.g. marker name typo, decoration
removed). The fence has bulletproof posts.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Documentation-only — write a CONVENTIONS doc; rely on PR review | minimal change | no automated enforcement; future contributors miss the convention | rejected: lakehouse failure today shows convention-without-enforcement is insufficient |
| B. AST-based gate forbidding `int(...)` / `.astype(int)` patterns on caller-DataFrame values | catches the pattern at PR time | high false-positive rate (legitimate uses); hard to specify the "caller-data" predicate precisely; complex to maintain | rejected: auto-discovered fuzz catches the same class with much simpler implementation |
| C. Full nullable-Int64 dtype migration (player_id / team_id → pandas Int64) | type-level NaN-safety enforced by pyright | massive blast radius — every converter, every helper, every downstream consumer (lakehouse) needs to handle nullable Int64; multi-week migration | rejected for now: future direction; this PR addresses immediate pain |
| D. Per-helper hand-written NaN tests | explicit, easy to understand | requires test author to remember on every new helper; future helpers without NaN tests slip through | rejected: auto-discovered fuzz is the same effort with self-enforcing perimeter |
| E (chosen). Contract + decorator + auto-discovered fuzz + cross-provider e2e + registry-floor sanity | self-enforcing perimeter; decoration is explicit opt-in; future helpers auto-covered when decorated | requires opt-in via decoration; helpers that forget the decorator aren't tested by ADR-003 (mitigated by code review against CLAUDE.md rule + registry-floor sanity catching mass-decoration breakage) | — |

## Consequences

### Positive

- The 2 confirmed crash sites (`add_pre_shot_gk_context` × 2) and
  2 latent crash sites (`add_gk_distribution_metrics` × 2) are fixed.
  The `coverage_metrics` `int(NaN)` defensive guard ships alongside
  (not strictly under ADR-003 — `coverage_metrics` returns a TypedDict,
  not an enriched DataFrame — but same crash class, fixed while we're here).
- The contract is explicit, codified, and CI-enforced.
- New enrichment helpers added in future PRs auto-extend the fuzz
  coverage when decorated. ADR-003 + the decorator together create a
  self-enforcing perimeter: contributors must explicitly opt in (which
  forces them to think about NaN-safety) but the test infrastructure
  is automatic.
- The cross-provider e2e gate catches the "first time helper X meets
  provider Y data" bug class — exactly the class that surfaced today.
- Every existing call site (`goalkeeper_ids=None` on `add_gk_role`,
  unchanged signatures elsewhere) preserves byte-for-byte behavior.

### Negative

- Decoration is opt-in. A new helper that's NaN-unsafe AND undecorated
  AND has no dedicated NaN test would satisfy CI but fail ADR-003 in
  spirit. Mitigation: CLAUDE.md "Key conventions" amendment makes
  decoration a project-wide rule reviewers check during PR review.
- Adding `goalkeeper_ids: set | None = None` parameter to `add_gk_role`
  expands the function's `inspect.signature(add_gk_role)` surface.
  Consumers using `inspect.signature(add_gk_role)` (rare) would see the
  new keyword-only param. Documented in CHANGELOG.
- Rule (b) NaN-team fallback in `add_gk_role` may over-count distribution
  rows on data with multiple NaN-player_id non-keeper actions following
  a keeper action by the same team. Caller's opt-in via `goalkeeper_ids`
  is the explicit signal that they accept the coarser heuristic.

### Neutral

- `gamestates.__module__` (and `simple.__module__`) flips noted in
  ADR-002 (silly-kicks 2.4.0) is a similar Hyrum's-Law class of change;
  ADR-003's `_nan_safe` attribute on helpers is a parallel additive
  surface.
- ADR-003 follows the ADR-001/ADR-002 precedent. The vendored
  `ADR-TEMPLATE.md` is the canonical structural source.

## CLAUDE.md Amendment

Adds one rule to the project's `CLAUDE.md` "Key conventions" section:

> Public enrichment helpers (post-conversion `add_*` family) tolerate NaN
> in caller-supplied identifier columns. NaN identifiers route to the
> documented per-row default (typically NaN-output / False / 0); helpers
> never crash on NaN input. Decoration with `@nan_safe_enrichment` from
> `silly_kicks._nan_safety` is the formal opt-in. Decision: ADR-003.

Scope: every public `add_*` enrichment helper in `silly_kicks.spadl.utils`,
`silly_kicks.atomic.spadl.utils`, and any future post-conversion enrichment
module.

## Related

- **Spec:** `docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-30-nan-safety-enrichment-helpers.md`
- **ADRs:** ADR-001 (caller-pass-through identifier convention — implies the NaN-safety requirement at the helper layer); ADR-002 (PR-S16 framework extraction precedent for the cross-package private-module pattern used here for `_nan_safety.py`).
- **Issues / PRs:** silly-kicks PR-S17 (this PR — primary failure surfaced 2026-04-30 by luxury-lakehouse `compute_spadl_vaep` task on IDSSE bronze data; lakehouse memo references PR-LL3-era observation now superseded).

## Notes

### `_nan_safe` attribute name and discoverability

The marker `fn._nan_safe = True` is set as a function attribute by the
`@nan_safe_enrichment` decorator. Tests discover via
`inspect.getmembers(module, inspect.isfunction)` filtered on
`getattr(fn, "_nan_safe", False)`. Trade-off vs. an explicit registry
list:

- **Attribute-based** (chosen): no central list to maintain; marker
  travels with the function across refactors / module moves; failure
  mode is "helper not in registry" which is benign (helper just not
  fuzz-tested) and caught by registry-floor sanity if it happens
  en-masse.
- **Central registry list**: explicit and auditable, but creates a
  second maintenance point and an import-order subtlety (helpers must
  be imported for the registry to populate).

### `goalkeeper_ids` rule semantics on `add_gk_role`

The new `goalkeeper_ids: set | None = None` keyword-only parameter
extends distribution-detection with two additional matching rules
when provided:

- **Rule (a) — known-GK match.** When the current row's `player_id`
  is in `goalkeeper_ids` AND the preceding action (within
  `distribution_lookback_actions` k-step lookback, same `team_id` and
  `game_id`) was a keeper-type action, tag as distribution. Use case:
  caller has clean player attribution and knows the GK player_id set
  (clean StatsBomb-style data with explicit GK metadata).
- **Rule (b) — NaN-team fallback.** When both the current row's and
  the preceding action's `player_id` are NaN AND the `team_id` matches
  AND the preceding action was keeper-type, tag as distribution. Use
  case: caller's data has NaN player_id (sparse provider attribution
  like IDSSE / Metrica) but the team and sequence imply the GK
  distributed the ball.

When `goalkeeper_ids` is `None` (default), neither rule fires;
behavior is byte-for-byte compatible with pre-2.5.0. Caller's opt-in
via passing the parameter is the signal that they accept the coarser
heuristic of rule (b) (which can over-count if multiple NaN-player_id
non-keeper actions follow a keeper action by the same team within
the lookback window).

### Future direction

Long-term, migrating `player_id` / `team_id` columns to pandas nullable
`Int64` dtype throughout SPADL would push NaN-safety to the type
system. That's a multi-week migration with major Hyrum's Law surface;
ADR-003 explicitly defers it. When the migration happens, the
`@nan_safe_enrichment` decorator may be retired (or its semantic
shifted to "helper supports nullable-Int64 identifier columns").
