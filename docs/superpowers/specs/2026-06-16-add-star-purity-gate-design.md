# `add_*` input-purity CI gate + `add_gk_distribution_metrics` mutation fix + pitch-control rename (design)

**Date:** 2026-06-16 · **Status:** draft for review · **Decision:** ADR-033 (new) · **Base:** main @ v4.31.0 → **target 4.32.0** (minor, breaking)

## Context

Lakehouse 4.31.0-adoption feedback surfaced one functional defect and several doc/ergonomics gaps. The defect
is a *class*, not an instance, so the long-term fix closes the class.

**The defect (confirmed + reproduced).** `silly_kicks.spadl.utils.add_gk_distribution_metrics` **mutates the
caller's DataFrame in place** when `gk_role` is already present: the `.copy()`/`.sort_values()` lives only
inside the `if "gk_role" not in actions.columns` branch, so the `gk_role`-present path assigns the four new
columns straight onto the input. Repro: with `gk_role` pre-set, the original frame gains all 4 columns and
`out is actions` is `True`. This violates (a) the docstring's explicit **"Sorted copy of `actions`"**
contract, (b) the hexagonal **zero-mutation** principle (CLAUDE.md: "all core functions are pure … zero global
state mutation"), and (c) **sort-consistency** — the `gk_role`-present path also skips the sort the other
paths do. The atomic mirror (`atomic/spadl/utils.py`) has the same structure → same bug.

**Why it reached a consumer instead of CI:** there is no gate guarding `add_*` input-purity. The existing
auto-enumerating gates (nan-safety ADR-003, aggregator-liveness, dup-action_id ADR-020, id-dtype ADR-019) each
discover the `add_*`/`*_xfns` surface and assert a contract — but none asserts "the input is not mutated."

## Decision

Four parts, shipped together in **4.32.0** (minor; breaking due to Part D). Decision ADR-033.

### A — fix the mutation + sort-consistency (`add_gk_distribution_metrics`, standard + atomic)
**This is an identity + row-order defect ONLY — NOT a value miscompute.** In the `gk_role`-present path the
four columns are derived from `actions` via positional `.to_numpy()` and assigned back to the same object
positionally, so row alignment is internally consistent regardless of sort order — the emitted VALUES were
always correct. The defect is exactly (a) in-place mutation of the caller's frame and (b) output row-order
inconsistency vs the other paths. **No downstream emergency recompute is warranted** (ADR-033 + the commit
must state this so consumers don't fear silently-wrong historical numbers).

Fix: `out = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)`
at the **top** and operate on `out` (do NOT rebind the `actions` parameter — match the family idiom
`out = actions.copy()`, the very convention Part B's gate enforces), so **every** path returns a sorted copy.
The `gk_role`-absent branch keeps calling `add_gk_role` (idempotent re-sort on the already-sorted copy).
Same in the atomic mirror.

### B — the long-term solution: an auto-enumerating `add_*` no-mutation CI gate
A new gate (mirrors the established pattern) that auto-discovers every public `add_*` in `silly_kicks.spadl`,
`silly_kicks.tracking`, `silly_kicks.atomic.spadl`, `silly_kicks.atomic.tracking`, invokes each on a valid
fixture **with kwargs that make it actually do its work** (column-adding path, not an early return), and
asserts purity. A **meta-assertion** pins the gate surface to the registered `add_*` __all__ (a new `add_*`
that isn't wired — or that mutates — fails CI). **Five rigor requirements (without these the gate false-greens
on the motivating bug):**

1. **Equality by VALUE, not column-set.** Snapshot `before = arg.copy(deep=True)` for each input and assert
   `before.equals(arg)` after the call (full shape + dtype + value equality). Column-set equality misses
   in-place value mutation (`df["x"] *= 2` adds no column).
2. **Per-function input VARIANTS that hit the mutating path.** A single generic `actions` fixture exercises
   only one branch — `add_gk_distribution_metrics` mutates ONLY on the `gk_role`-**present** path, so a
   no-`gk_role` fixture goes GREEN on the bug. The gate runs each helper over its relevant input variants
   (minimum here: `gk_role` present AND absent). This variant is what makes "red-first" true.
3. **Snapshot EVERY array-like argument**, not just the primary `actions` — tracking `add_*` take `frames`,
   `xt`, etc.; purity = none of them mutated.
4. **`out is not actions` is UNCONDITIONAL.** No "…unless a helper returns the same object" escape hatch
   (that reintroduces the laxity that let the bug through). An `add_*` adds columns → it must return a NEW
   object. If a specific helper genuinely cannot, it goes on an **explicit allowlist with a reason** (the
   nan-safety-gate exemption pattern), never a blanket clause.
5. **Exercise the real work** — pass valid kwargs (`require_gk_role=True`, `pre_shot_gk_context` with
   `frames`, etc.) so each helper runs its column-adding path; tie fixtures to the liveness-gate inputs which
   are already valid. An early-return-on-missing-kwargs helper trivially "doesn't mutate" = false green.

- **tracking side:** reuse the `tests/tracking/test_aggregator_column_liveness.py` `ENTRIES` infrastructure
  (it already invokes every tracking `add_*` with valid frames/xt/home_team_id) — wrap it to snapshot all
  array args + assert value-equality + `out is not <each input>`.
- **spadl side** (7: `add_game_state`, `add_gk_distribution_metrics`, `add_gk_role`, `add_names`,
  `add_possessions`, `add_pre_shot_gk_context`, `add_restart_coordinates`; + the 5 atomic mirrors): a shared
  SPADL / atomic-SPADL actions fixture + the per-function variants from (2).
- **Discovery introspects `def add_*`, not just `__all__` (review).** A public `def add_*` accidentally
  omitted from `__all__` would be invisible to both an `__all__`-scan and the meta-assertion. Discovery
  enumerates actual module-level `add_*` defs and asserts `__all__ ⊇ {public add_* defs}` — a helper missing
  from `__all__` fails CI rather than escaping the gate. (Confirm whether the liveness/nan-safety gates already
  do this; align with them.)
- **`before.equals(arg)` on Categorical + NaN (review — verify, don't assume).** `DataFrame.equals` treats
  co-located NaN as equal (NaN-heavy fixtures won't false-RED) and is strict on dtype. The `gk_role`-present
  variant carries a **Categorical** `gk_role`; confirm `equals` behaves as expected on Categorical on BOTH the
  pandas-2 and pandas-3 parity envs before relying on it.
- **Audit-and-fix:** run the gate; fix every `add_*` it flags (each fix = `out = actions.copy()` up front).
  The gate — not an eyeball — is the authority on the audit scope (expected small; `add_gk_distribution_metrics`
  is the known outlier).

#### Known limit of the gate + the heuristic that targets it (review, strongest point)
The gate closes "mutates on the **default** path." It does NOT, by construction, close "mutates only on a
**non-default branch**" for FUTURE helpers: the per-function variants (req #2) are **manual**, and the
meta-assertion pins only that each `add_*` is *wired*, not that its mutating *branches* are covered. So the
exact class that produced this bug (a branch-conditional mutation — `gk_role`-present only) is closed for
`add_gk_distribution_metrics` today, but a future branch-conditional `add_*` run with a single default variant
would false-green — the same human-memory gap, one level up. To target this class (not just document it):
- **A heuristic meta-check (CI nudge):** flag any `add_*` whose signature has a behavior-toggling
  `bool`/`Optional` kwarg (`require_gk_role`, `frames=None`, …) OR whose body references
  `in <param>.columns` (a branch on column presence) but has **< 2 registered gate variants** → fail with
  "register ≥2 input variants for the branching helper, or allowlist with a reason." Heuristic (AST/signature
  inspection, not airtight) but it hits exactly this class; false positives go on an explicit allowlist (the
  nan-safety-exemption pattern). ADR-033 Consequences names this as the gate's known boundary.

#### Audit-scope contingency inside a breaking release (review)
Part B's "fix every `add_*` the gate flags" is unbounded going in, and it rides with a breaking public rename
(Part D). **Decision rule:** if the audit surfaces ONLY identity/row-order mutations (expected; the
`add_gk_distribution_metrics` class), bundle everything into 4.32.0. If it surfaces a **value-affecting**
mutation (see scoping below) OR an unexpectedly large set, **split**: land B (gate + audit-fixes) on its own
validation/release and let D (the rename) ride separately — do not let an unexpectedly large audit silently
expand a breaking release. The 4.32.0 cut is gated on "audit result == identity/order-only."

### C — doc/ergonomics (non-functional)
- `add_off_ball_context` + `add_off_ball_runs` + `add_line_break`: enumerate emitted columns **+ dtypes**;
  add a "for all N off-ball-context columns, use `add_off_ball_context`" cross-ref (stops double-wiring).
- `add_shot_goalmouth`: enumerate the 11 emitted columns + dtypes (currently prose).
- `add_gk_distribution_metrics`: docstring note that `gk_pass_length_class` is `category` dtype
  (Arrow/Spark → `.astype(object)` before `createDataFrame`); document `gk_xt_delta`'s expected
  **(12×8) SPADL-convention** grid (rows=x-bins, cols=y-bins) + that it is intentionally silly-kicks' own grid
  (a consumer with a different grid/coord-system derives its own delta, as the lakehouse chose — avoids a
  second xT source of truth).
- **Self-policing (review): a doc-accuracy assertion** so the enumerations can't silently drift — the
  liveness gate already computes each helper's emitted column set; add a cheap test asserting the
  docstring-enumerated names are a subset of the actually-emitted set for the touched helpers. Makes Part C
  a guarded invariant instead of "cross-check by eye."

### D — rename `pitch_control_at_action` → `pitch_control_at_target` (standard + atomic)
The Series-level function's base name never matched its column (`at_action` vs `at_ball` pre-4.31.0, `at_target`
now); every other action-coupled Series feature is named for what it emits. Rename to align. **Now is the
window:** it's a breaking rename, and 4.31.0 is already a breaking change the lakehouse hasn't adopted, so it
folds into the one break (the lakehouse pins 4.32.0, skips 4.31.0). The function rename's blast radius is contained
— the emitted **column** (`pitch_control_at_target__<method>`) and the `add_pitch_control` /
`pitch_control_xfns` / `pitch_control_default_xfns` names are **unchanged**. **Guarded invariant (review):**
the rename must touch ONLY the function / `__all__` / callers and leave the emitted column base
`pitch_control_at_target__<method>` byte-identical — the existing column-name tests
(`test_action_coupled.py` / `test_atomic_pitch_control.py`, asserting `pitch_control_at_target__*`) stay as
the guard; confirm they remain green post-rename (the rename does not re-touch `col_name`).

**Lakehouse impact (correcting the earlier "near-zero"):** lakehouse *column*-consumers are unaffected, but
the lakehouse is a **direct caller** of the Series function (`enrich.py`, `tracking_context.py` + a test patch
target) — so the rename IS on them, though small + column-stable. Their DEFCON mart column literally named
`pitch_control_at_action` is **intentionally KEPT** (their schema; semantically correct — DEFCON samples PC at
the defender's action location, not a pass target — Chesterton + Hyrum). silly-kicks renaming its function
does not obligate that mart-column rename; the two `pitch_control_at_action` strings are distinct concerns.

**No deprecation alias** (the codebase's clean-break convention — cf. the 4.14.0
`ghost_gk_spread`→`ghost_gk_density_spread` rename and 4.31.0's column break). Surface: the 2 Series functions,
the 4 `__all__` exports, internal callers (`add_pitch_control`, `pitch_control_xfns`, atomic equivalents),
tests, docstrings, CLAUDE.md.

## Testing strategy

**Explicit TDD ordering (review):** (1) build Part B's gate WITH the `gk_role`-present variant wired → run it
→ **RED** on `add_gk_distribution_metrics` (proves the gate catches the motivating bug, not just decorates).
(2) Apply Part A → gate **GREEN**. (3) Add Part A's targeted standard+atomic regression (belt-and-suspenders).
(4) Run the gate over the full surface → audit-fix anything else it flags. (5) Part C doc-accuracy assertion +
Part D rename. The present-variant prerequisite in step 1 is what makes "red-first" genuine.

- **Part A:** regression test asserting `add_gk_distribution_metrics` does NOT mutate its input for BOTH the
  `gk_role`-present and `gk_role`-absent cases (the exact gap that let it through; value-equality snapshot),
  and that the output is a sorted copy. Standard + atomic.
- **Part B:** the gate IS the test — RED-first on the `gk_role`-present variant, GREEN after Part A, with the
  five rigor requirements above (value-equality, per-function variants, all-arg snapshot, unconditional
  `out is not input`, work-exercising kwargs). Meta-assertion (gate surface == registered `add_*`).
- **Part C:** doc-only — `tests/test_public_api_examples.py` already enforces Examples; the column-enumeration
  is prose (not separately gated), but accuracy is checkable by running each `add_*` (the liveness gate
  already produces the column sets — cross-check the docstring lists against them).
- **Part D:** rename — `git grep pitch_control_at_action` is the authority, but the "zero remaining" assertion
  is scoped to **code** (`silly_kicks/**`, callers, `__all__`, imports, test code) — **exclude historical prose**
  (`CHANGELOG.md`, `docs/superpowers/adrs/**`, prose examples describing 4.31.0): blanket-renaming those would
  rewrite the record. The emitted column base `pitch_control_at_target__<method>` stays byte-identical (the
  existing column-name tests guard it); full suite confirms the xfn/calibration wiring.
- Full gate: `ruff` + bare `pyright` (incl `tests/`) + full suite; reproduce on a pandas-3 env if any
  parity/dtype surface is touched.

## Blast radius / version

- **4.32.0 (minor, breaking).** Part D is a public-API rename. Part A is a behavior fix (stops mutation; the
  `gk_role`-present path now also sorts — a Hyrum-observable row-order change for callers who fed unsorted,
  `gk_role`-present data, but it honors the documented "Sorted copy"). Part B audit-fixes may change other
  `add_*` from mutating-input to returning-a-copy (behavior fix; flag any that affect a retrain-relevant
  feature).
- **Lakehouse:** pins 4.32.0 directly (skips 4.31.0). The `pitch_control` **column** migration (`at_ball` →
  `at_target`, AC+DEFCON, atomic-with-pin) is unchanged by the function rename. The mutation fix means any
  lakehouse code relying on `add_gk_distribution_metrics` mutating its input in place must read the returned
  frame instead (it should already — the docstring always said "Sorted copy").
- **C4-free:** no new aggregator/container (count stays 28). New CI test only.

## ADR-033

Records: the `add_*` input-purity contract is now CI-gated (auto-enumerating no-mutation gate, five rigor
requirements so it can't false-green) — the hexagonal zero-mutation principle made enforceable, motivated by
the `add_gk_distribution_metrics` gk_role-present in-place-mutation defect found in lakehouse adoption.
**The motivating defect (`add_gk_distribution_metrics`) is identity + row-order ONLY — NO value miscompute**
(positionally-consistent `.to_numpy()` derivation), so no downstream emergency recompute is warranted; ADR-033
states this explicitly so consumers don't fear silently-wrong historical numbers. **This proof is scoped to
`add_gk_distribution_metrics` and does NOT generalize** — for ANY other helper the Part-B audit flags, value
impact is assessed per-helper (an index-aligned assignment onto an unsorted frame, or mutation of a shared
input array, COULD be value-affecting); the no-recompute conclusion is inherited only after that per-helper
analysis (and triggers the split-release contingency above if value-affecting). Consequences note the bundled
mutation fix + sort-consistency, the gate's known branch-coverage limit + the heuristic nudge, the
`pitch_control_at_action`→`pitch_control_at_target` rename (breaking, window-justified; the emitted column base
is byte-unchanged + guarded), and the doc clarifications.

## Alternatives rejected

- **`.copy()`-only (no forced sort)** for Part A — leaves the sort-inconsistency latent + the docstring's
  "Sorted copy" only half-true. Honoring the documented contract is the long-term-correct choice.
- **Docstring note instead of the rename** (Part D) — papers over permanent function/column name debt; the
  pre-adoption window makes the clean rename free of a second migration.
- **Deprecation alias for the rename** — against the codebase's clean-break convention; adds a surface to
  later remove; the lakehouse hasn't adopted, so no soft-transition value.
- **Fix only `add_gk_distribution_metrics`, skip the gate** — band-aids the instance, leaves the class
  unguarded (the next mutating `add_*` reaches a consumer again). Not "long-term."

## Self-review

- **Spec coverage:** A (mutation + sort) · B (gate + audit) · C (4 doc items) · D (rename) — all four
  lakehouse-feedback items mapped, plus the class-level gate.
- **Placeholder scan:** the audit scope is "whatever the gate surfaces" (a genuine gate-driven result, not a
  vague TODO); every other surface is named.
- **Consistency:** `pitch_control_at_target` (function == column base) consistent across standard/atomic/
  tests; the no-mutation gate follows the existing auto-enumerating meta-assertion pattern.
- **Scope:** one coherent theme (`add_*` purity + the pitch-control naming it touches); no unrelated refactor.
