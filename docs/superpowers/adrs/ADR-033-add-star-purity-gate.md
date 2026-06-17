# ADR-033: `add_*` input-purity CI gate + gk_distribution mutation fix + pitch_control_at_target rename

| Field | Value |
|---|---|
| **Date** | 2026-06-16 |
| **Status** | Accepted (PR-S97, silly-kicks 4.32.0) |
| **Deciders** | Karsten (with Claude); lakehouse session (cross-session spec + plan review, 4 rounds) |

## Context

`silly_kicks.spadl.utils.add_gk_distribution_metrics` (and its atomic mirror) was documented to return a
"Sorted copy" but only copied+sorted on the `gk_role`-**absent** path. When `gk_role` was already present
the helper assigned its four columns straight onto the caller's frame — an in-place mutation of a
caller-supplied DataFrame, violating the hexagonal pure-function contract (pandas-in / pandas-out, zero
mutation). The pre-existing `test_does_not_mutate_input` never caught it because it exercised only the
absent path (which always copied).

This is a *class* of defect, not a one-off: nothing structurally prevented any of the ~58 public `add_*`
enrichers (spadl/atomic.spadl/tracking/atomic.tracking) from mutating an input. The auto-enumerating gates
already established for other contracts (nan-safety ADR-003, aggregator-liveness, dup-`action_id` ADR-020,
id-dtype ADR-019) are the proven pattern for closing a class repo-wide.

Separately, `pitch_control_at_action` (renamed in spirit by ADR-032 to sample the action *destination*) kept
its old name while emitting `pitch_control_at_target__<method>` — a function/column-name mismatch. The
column rename shipped in 4.31.0 has no released consumer yet, so the window to align the function name is now.

Spec: `docs/superpowers/specs/2026-06-16-add-star-purity-gate-design.md`.

## Decision

**Part A — fix the mutation (scoped: identity + order ONLY, no value miscompute, no recompute).**
Both `add_gk_distribution_metrics` implementations hoist `out = actions.sort_values([...]).reset_index(drop=True)`
to the top and operate on `out`. The sort key matches `add_gk_role`'s internal sort, so the
`require_gk_role` path is value/order-identical; only the `gk_role`-present path changes — it now returns a
sorted copy (the documented contract) instead of mutating in place. Derivation is per-row vectorized, so
rows carry their values under the sort — no value miscompute. **This no-recompute claim is scoped to this
helper**; any other helper a future audit flags needs its own per-helper value analysis.

**Part B — the auto-enumerating no-mutation gate** (`tests/test_add_star_purity.py`). One canonical
`PURITY_ENTRIES` registry keyed `"<package>:<add_name>"`; everything (parametrization, per-package
registered subsets, `_resolve_fn`, the heuristic) derives from it. For every registered `add_*`, the gate
builds fresh OWNED inputs once, snapshots every array-like arg, invokes, and asserts (a) every caller
DataFrame/Series/ndarray is value-unchanged and (b) the result is a NEW object. Helpers that branch on
column presence register one variant per branch. Five rigor requirements:

1. **Build-once, hold the reference** — not the liveness gate's rebuild-and-cache (the object the function
   gets must be the one the test snapshots); cached builders are `.copy(deep=True)`'d to avoid poisoning the
   shared liveness cache.
2. **ndarray `equal_nan` guarded by `np.issubdtype(dtype, np.inexact)`** (int/object raise otherwise).
3. **Frames + xt + links are snapshotted too** (they're caller inputs).
4. **Two meta-assertions pin the surface to the public export** — `__all__` UNION `.features.__all__`
   (the 15 `atomic.tracking.features` mirrors export via the submodule, not the package), so a new public
   `add_*` cannot land unregistered; and every public `def add_*` in a defining submodule must be in that
   export surface.
5. **Best-effort branch-conditional heuristic** — an AST check flags a single-variant helper whose `if`
   tests a STRING-LITERAL column name against `*.columns` with a non-`raise` body (the bug shape), while
   excluding the provenance-skip genexp guard (`Name`-left), validation list-comps (`comprehension`), and
   input-validation `raise` guards.

**Known limit.** The gate closes *default-path* mutation for every registered helper and *both* branches for
multi-variant ones. It does NOT prove a future branch-conditional mutation in an unregistered shape — the AST
heuristic recognizes only the one known shape and is explicitly a NUDGE, not a proof. The real backstop is
the **contributor contract** (CLAUDE.md): any `add_*` that conditionally adds columns MUST register ≥2 purity
variants.

**Audit result.** Running the gate over all ~58 public `add_*` (incl. the 15 `atomic.tracking.features`
mirrors, which the plan initially missed and the meta-assertion caught) found the mutation class is otherwise
**clean** — only `add_gk_distribution_metrics` (standard + atomic) mutated. So Parts A/B/C/D bundle into one
4.32.0 release (the spec's "audit == identity/order-only" cut).

**Part C — docstring tightening.** Enumerate emitted columns + dtypes for `add_off_ball_runs`,
`add_off_ball_context`, `add_shot_goalmouth`; add the `gk_pass_length_class` Categorical/Spark-`StringType`
note and the `gk_xt_delta` caller-supplied-`(12,8)`-SPADL-grid (never self-fit) note to
`add_gk_distribution_metrics` (standard + atomic). A doc-accuracy test pins each exhaustive-claim helper's
emitted feature set to an explicit `frozenset` and asserts the docstring names every column.

**Part D — rename `pitch_control_at_action` → `pitch_control_at_target`** (standard + atomic; `__all__`,
imports, callers, tests). The emitted column base is unchanged (`pitch_control_at_target__<method>`, already
correct since 4.31.0) — only the function name aligns. Breaking, window-justified (no released consumer of
the 4.31.0 column rename yet).

## Consequences

- **VAEP/tracking:** no retrain. The gk_distribution fix is identity/order only; the rename is name-only
  (column base byte-unchanged); the gate + docstrings are test/doc-only.
- **Hyrum:** a consumer that passed `gk_role`-present, **unsorted** actions to `add_gk_distribution_metrics`
  and relied on the (buggy) unsorted, mutated-in-place output sees sorted output in a new object. The
  documented contract ("Sorted copy") always promised this. Lakehouse handoff: pin 4.32.0, rename the
  `pitch_control_at_action` call sites (keep the DEFCON `pitch_control_at_action` mart *column* — different
  semantics), gk_distribution fix needs no re-materialization.
- **CI:** the purity gate joins the auto-enumerating-gate family; a new `add_*` must register or CI fails.

## Alternatives considered

- **Scope the gate to package `__all__` only** (leave the 15 `atomic.tracking.features` mirrors uncovered,
  documented as "covered transitively via delegation"): rejected — the delegation/synthesis wrapper is itself
  unverified for purity, and leaving 15 public functions ungated re-creates the blind spot the gate exists to
  remove. Owner decision: cover all 16 atomic-tracking public mirrors.
- **Allowlist the heuristic-flagged links-optimization helpers** instead of adding `supplied_links` variants:
  rejected — a `supplied_links` variant verifies a real path (the function does not mutate caller-supplied
  `links`), which is strictly more coverage than an allowlist entry.
- **Docstring-backtick parsing for the doc-accuracy test:** rejected — fragile; the explicit per-helper
  `frozenset` IS the contract a doc edit must keep in sync.
