# `add_*` input-purity gate + gk_dist mutation fix + pitch-control rename (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the `add_*` in-place-mutation class with an auto-enumerating no-mutation CI gate (+ a
heuristic nudge for branch-conditional mutation), fix the motivating `add_gk_distribution_metrics` defect,
rename `pitch_control_at_action`→`pitch_control_at_target`, and tighten four docstrings.

**Architecture:** A new auto-enumerating gate (`tests/test_add_star_purity.py`) that builds fresh inputs ONCE
per `add_*`, snapshots every array-like arg, invokes with work-exercising kwargs over per-function variants,
and asserts value-equality (`before.equals(after)`) + unconditional `out is not <input>`. Fixes are
`out = actions.copy()` up front. Pure-function discipline; no runtime/behaviour change beyond the rename.

**Tech stack:** Python 3.10–3.14, pandas/numpy, pytest, ruff, pyright. Spec:
`docs/superpowers/specs/2026-06-16-add-star-purity-gate-design.md`. **Decision:** ADR-033. **Base:** main @
v4.31.0 → **target 4.32.0** (minor, breaking — Part D rename).

**Owner-policy adaptations:** feature branch `pr-s97-add-star-purity-gate` off `main`, **no worktree**. **NO
intermediate commits** — each task ends in a verify checkpoint + `git add` (staging); the **single commit** is
the last task, after `/final-review` + explicit owner approval. RED-first = run + observe + capture, NOT a
commit. Never tag before CI green.

**Audit-scope contingency (spec):** Part B's audit fixes "every `add_*` the gate flags." If it surfaces ONLY
identity/row-order mutations (expected), bundle all parts into 4.32.0. If it surfaces a **value-affecting**
mutation or an unexpectedly large set, STOP and flag the owner — split B onto its own release, let D ride
separately. The 4.32.0 cut is gated on "audit == identity/order-only."

---

## File structure

| Path | Responsibility | Action |
|------|----------------|--------|
| `tests/test_add_star_purity.py` | NEW — the auto-enumerating no-mutation gate (Part B): build-once harness, per-function variants, value-equality, `out is not input`, def-introspection discovery + `__all__ ⊇ defs` meta-assertion, the branch-conditional heuristic meta-check | Create |
| `silly_kicks/spadl/utils.py` | Part A: `add_gk_distribution_metrics` → `out = actions.sort_values(...).reset_index(drop=True)` at top (no param rebind); docstring category + grid notes (Part C) | Modify |
| `silly_kicks/atomic/spadl/utils.py` | Part A atomic mirror; Part C doc notes | Modify |
| `silly_kicks/tracking/features.py` | Part D rename `pitch_control_at_action`→`pitch_control_at_target` + `__all__` + internal callers (`add_pitch_control`, `pitch_control_xfns`); Part C column enumeration for `add_off_ball_context`/`add_off_ball_runs`/`add_line_break`/`add_shot_goalmouth` | Modify |
| `silly_kicks/atomic/tracking/features.py` | Part D atomic rename + `__all__` | Modify |
| `silly_kicks/tracking/__init__.py` | Part D `__all__` + import rename | Modify |
| any other `add_*` the gate flags | Part B audit-fix (`out = actions.copy()` up front) — scope = gate result | Modify |
| `tests/tracking/pitch_control/test_*` | Part D: rename function refs (column assertions stay `pitch_control_at_target__*` — guarded invariant) | Modify |
| `docs/superpowers/adrs/ADR-033-add-star-purity-gate.md` | NEW ADR | Create |
| `CLAUDE.md` | Part B gate convention + Part D rename note | Modify |
| `pyproject.toml`/`__init__.py`/`CHANGELOG.md`/`TODO.md`/`uv.lock` | version 4.32.0 | Modify |

---

## Phase 1 — the no-mutation gate (Part B), authored FIRST → RED on the motivating bug

### Task 1.1: ONE canonical registry + purity harness + the `add_gk_distribution_metrics` entry → RED

**Files:** Create `tests/test_add_star_purity.py`.

- [ ] **Step 0: Define the ONE canonical registry (review #3 — single source of truth).** Everything derives
      from it: the parametrization, `REGISTERED_NAMES`, `_resolve_fn`, the heuristic's variant-count, and 4.2's
      emitted-column sets. A `Variant = (variant_name: str, build_inputs: () -> list, invoke: (inputs) ->
      result)`; `PURITY_ENTRIES: dict[str, list[Variant]]`; `REGISTERED_NAMES = set(PURITY_ENTRIES)`.
- [ ] **Step 1: Write the harness + the gk_dist entry.** Build inputs ONCE via `build_inputs`, snapshot EVERY
      array-like arg deep, `invoke(inputs)`, assert value-equality + unconditional `out is not <input>`.

```python
"""add_* input-purity gate (ADR-033). Every public add_* must be PURE: it must not mutate any caller-supplied
DataFrame/ndarray, and (it adds columns) must return a NEW object. Auto-enumerating + single-source: a new
add_* that isn't registered -- or that mutates -- fails CI. Build inputs ONCE and hold the reference (the
liveness gate's _std rebuilds the input + caches _frames -- unusable here)."""
from __future__ import annotations
import numpy as np, pandas as pd, pytest

def _assert_pure(name, variant, inputs, invoke):
    df_snaps = [(x, x.copy(deep=True)) for x in inputs if isinstance(x, (pd.DataFrame, pd.Series))]
    arr_snaps = [(x, x.copy()) for x in inputs if isinstance(x, np.ndarray)]
    out = invoke(inputs)
    for orig, snap in df_snaps:
        assert snap.equals(orig), f"{name}[{variant}] MUTATED a caller DataFrame/Series in place"
    for orig, snap in arr_snaps:
        # review #2: equal_nan requires an inexact dtype; int/object ndarrays raise -> guard it.
        eq = (np.array_equal(snap, orig, equal_nan=True) if np.issubdtype(orig.dtype, np.inexact)
              else np.array_equal(snap, orig))
        assert eq, f"{name}[{variant}] MUTATED a caller ndarray in place"
    for x in inputs:
        assert out is not x, f"{name}[{variant}] returned the SAME object as an input (must return a copy)"

def _spadl_actions(*, with_gk_role):
    df = pd.DataFrame({
        "game_id": [1, 1], "period_id": [1, 1], "action_id": [0, 1], "team_id": [10, 10],
        "player_id": [1, 2], "type_id": [0, 0], "result_id": [1, 1],
        "start_x": [5.0, 50.0], "start_y": [34.0, 34.0], "end_x": [60.0, 70.0], "end_y": [34.0, 40.0]})
    if with_gk_role:
        from silly_kicks.spadl.utils import _GK_ROLE_CATEGORIES
        # review #6: use REAL category members ("none" is not one -> would coerce to NaN).
        df["gk_role"] = pd.Categorical(["distribution", "shot_stopping"], categories=list(_GK_ROLE_CATEGORIES))
    return df

def _gk_dist_invoke(inputs):
    from silly_kicks.spadl.utils import add_gk_distribution_metrics
    return add_gk_distribution_metrics(inputs[0])

PURITY_ENTRIES = {
    "add_gk_distribution_metrics": [
        ("gk_role_present", lambda: [_spadl_actions(with_gk_role=True)], _gk_dist_invoke),
        ("gk_role_absent", lambda: [_spadl_actions(with_gk_role=False)], _gk_dist_invoke),
    ],
    # ... Task 1.2 fills the rest ...
}
REGISTERED_NAMES = set(PURITY_ENTRIES)

@pytest.mark.parametrize("name,variant_name", [(n, v[0]) for n, vs in PURITY_ENTRIES.items() for v in vs])
def test_add_star_does_not_mutate_input(name, variant_name):
    variant = next(v for v in PURITY_ENTRIES[name] if v[0] == variant_name)
    _vname, build_inputs, invoke = variant
    _assert_pure(name, variant_name, build_inputs(), invoke)
```

- [ ] **Step 2: Run → RED.** `python -m pytest tests/test_add_star_purity.py -v` → the
      `add_gk_distribution_metrics[gk_role_present]` case FAILS (`snap.equals` false + `out is actions`);
      `[gk_role_absent]` passes. Red-first proof the gate catches the motivating bug. Capture the output.

### Task 1.2: Wire the full surface (build-once, fresh frames) — reveal the audit scope

**Files:** `tests/test_add_star_purity.py`.

- [ ] **Step 1: Add spadl + atomic.spadl entries** (`add_game_state`, `add_gk_role`, `add_names`,
      `add_possessions`, `add_pre_shot_gk_context`, `add_restart_coordinates` + the 5 atomic mirrors) over the
      `_spadl_actions` fixture (+ an atomic-SPADL fixture with `x,y,dx,dy`). Each with the kwargs that exercise
      its column-adding path.
- [ ] **Step 2: Add tracking + atomic.tracking entries** as `PURITY_ENTRIES` variants. Lift the call
      signatures from `tests/tracking/test_aggregator_column_liveness.py::ENTRIES` (`_std(fn, **kw)` =
      `fn(actions, frames, **kw)`, `_xtf(fn, **kw)` = `fn(actions, frames, xt, home_team_id=5, **kw)`, the
      custom `_run_*`), BUT **build-once with FRESH, OWNED inputs**: each `build_inputs` constructs the inputs,
      and the harness snapshots + passes THOSE references. Do NOT reuse `_std`/`_xtf` (double-`make_actions()`
      + the returned input ≠ the one the fn got). **Cached-builder rule (review #4):** `make_actions`/
      `make_frames` are uncached (fresh each call), but `_frames_with_possession` / the jersey-roster / links /
      shot-goalmouth fixtures are `@functools.cache`d with NO uncached sibling — passing a cached object into a
      (pre-fix) mutating helper would **poison the shared cache** (cross-test contamination + nondeterministic
      purity result). So **any input sourced from a cached builder MUST be `cached_builder().copy(deep=True)`**
      in `build_inputs` — owned + fresh, cache stays clean. Snapshot frames + xt too (req #3 — they're in the
      `inputs` list).
- [ ] **Step 3: Discovery + meta-assertions — TWO checks at the RIGHT module (review #1, BLOCKER).** The
      naive `o.__module__ == package.__name__` filter returns **∅** for our re-export layout (add_* are DEFINED
      in `*.features`/`*.utils` submodules, re-exported through the package `__init__`/`__all__`), making both
      meta-asserts **vacuously true** — the gate would find nothing and a missing/mutating `add_*` would pass.
      Two distinct, correctly-targeted checks:
  - **(a) Registration completeness — `__all__`-based, MIRROR the proven liveness pattern**
    (`test_aggregator_column_liveness.py:390` does `{n for n in tracking.__all__ if n.startswith("add_")} ==
    set(ENTRIES)`). Per PACKAGE: the registered subset for that package == its public `add_*` exports. (Keep a
    per-package registered subset; their union == `REGISTERED_NAMES`.)
  - **(b) `__all__` completeness — introspect the DEFINING SUBMODULES** (`silly_kicks.tracking.features`,
    `silly_kicks.spadl.utils`, + atomic), the only place `o.__module__ == module.__name__` is valid; assert
    each public `def add_*` there is in the owning package's `__all__` (catches a public def omitted from
    `__all__`).

```python
def _defined_add_defs(submodule):
    import inspect
    return {n for n, o in inspect.getmembers(submodule, inspect.isfunction)
            if n.startswith("add_") and o.__module__ == submodule.__name__ and not n.startswith("_")}

def test_meta_registration_complete_per_package():
    import silly_kicks.spadl as sp, silly_kicks.tracking as tr
    import silly_kicks.atomic.spadl as asp, silly_kicks.atomic.tracking as atr
    for pkg, registered_subset in ((sp, SPADL_REGISTERED), (tr, TRACKING_REGISTERED),
                                   (asp, ATOMIC_SPADL_REGISTERED), (atr, ATOMIC_TRACKING_REGISTERED)):
        exported = {n for n in pkg.__all__ if n.startswith("add_")}
        assert exported == registered_subset, (
            f"{pkg.__name__}: purity-gate surface != public add_* exports "
            f"(unwired: {exported - registered_subset}, stale: {registered_subset - exported})")

def test_meta_all_public_add_defs_are_exported():
    import silly_kicks.spadl.utils, silly_kicks.tracking.features
    import silly_kicks.atomic.spadl.utils, silly_kicks.atomic.tracking.features
    import silly_kicks.spadl as sp, silly_kicks.tracking as tr
    import silly_kicks.atomic.spadl as asp, silly_kicks.atomic.tracking as atr
    for submod, pkg in ((silly_kicks.spadl.utils, sp), (silly_kicks.tracking.features, tr),
                        (silly_kicks.atomic.spadl.utils, asp), (silly_kicks.atomic.tracking.features, atr)):
        missing = _defined_add_defs(submod) - set(pkg.__all__)
        assert not missing, f"{submod.__name__}: public add_* defs missing from {pkg.__name__}.__all__: {missing}"
```

      (`SPADL_REGISTERED` etc. are the per-package keys of `PURITY_ENTRIES`; `REGISTERED_NAMES` is their union.)

- [ ] **Step 4: Run → capture the AUDIT RESULT.** Run the full gate. Record which `add_*` (besides
      `add_gk_distribution_metrics`) fail. **Apply the contingency:** if any failure is value-affecting or the
      set is unexpectedly large, STOP and flag the owner (split B from D). Else proceed.

### Task 1.3: The branch-conditional heuristic meta-check (precise AST `if`-guard detection)

**Files:** `tests/test_add_star_purity.py`.

- [ ] **Step 1: Add the heuristic — detect the ACTUAL pattern, via AST (review #1).** The regex + kwarg-toggle
      version over-fired catastrophically: `re.search(r"in \w+\.columns")` matched **validation list-comps**
      (`[c for c in REQ if c not in actions.columns]` at `spadl/utils.py:386` — present in nearly every helper)
      and `p.default in (None, True, False)` matched **every optional-input kwarg** (`frames=None`/`links=None`)
      *and* numeric `0`/`1` via `==` coercion → most of the surface flagged → a huge allowlist → meaningless
      ceremony. **Drop the kwarg half entirely** (an optional-input kwarg can't be distinguished from a
      behavior toggle by signature alone). Detect ONLY the bug's actual shape: an **`if`-statement whose test
      is a `Compare` with `In`/`NotIn` against an `Attribute` `.columns`** (the `if "gk_role" not in
      actions.columns:` branch) — which an AST walk catches but a list-comp validation does NOT (that's a
      `comprehension`, not an `ast.If`):

```python
import ast, inspect

def _resolve_fn(name):
    """Resolve a registered add_* by NAME via getattr across the four package namespaces (review #3 -- the
    registry's `invoke` is an OPAQUE closure, NOT the bound fn, so it can't yield the source object). First
    namespace that has it wins; the meta-asserts already pin name<->package, so collisions are pre-excluded."""
    import silly_kicks.spadl as sp, silly_kicks.tracking as tr
    import silly_kicks.atomic.spadl as asp, silly_kicks.atomic.tracking as atr
    for pkg in (sp, tr, asp, atr):
        fn = getattr(pkg, name, None)
        if fn is not None:
            return fn
    raise AssertionError(f"{name} registered in PURITY_ENTRIES but not exported by any package")

def _branches_on_column_presence(fn) -> bool:
    """True iff fn has an `if`-statement whose test compares (In/NotIn) against a `*.columns` Attribute --
    the branch-conditional-mutation shape. Excludes validation list-comps (those are `comprehension`, not `If`).
    Best-effort (review #6): a white-box source heuristic, NOT a proof -- see the test docstring."""
    fn = inspect.unwrap(fn)  # review #2: getsource on a DECORATED fn returns the wrapper, not the body
    try:
        tree = ast.parse(inspect.getsource(fn))
    except (OSError, TypeError):
        # review #2: dynamically-created / C-level / source-less helper -> cannot inspect; skip (don't crash).
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for cmp in ast.walk(node.test):
            if isinstance(cmp, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in cmp.ops):
                if any(isinstance(c, ast.Attribute) and c.attr == "columns" for c in cmp.comparators):
                    return True
    return False

def test_meta_column_branching_helpers_have_multiple_variants():
    """BEST-EFFORT nudge (review #6), NOT a guarantee. The AST heuristic only recognizes the ONE shape the
    motivating bug took (`if <col> [not] in <df>.columns:`); a helper that branches a different way (a
    `.get`, a try/except, a precomputed-mask flag) and mutates only on one branch will NOT be flagged. The
    real guarantee is per-variant coverage in PURITY_ENTRIES. Contributor contract (CLAUDE.md): any add_* that
    conditionally adds columns MUST register >=2 purity variants (the present AND absent branch)."""
    _SINGLE_VARIANT_OK: dict[str, str] = {}  # name: reason (justified single-variant; e.g. branch is non-mutating)
    for name, variants in PURITY_ENTRIES.items():
        if name in _SINGLE_VARIANT_OK or len(variants) >= 2:
            continue
        if _branches_on_column_presence(_resolve_fn(name)):
            raise AssertionError(
                f"{name} has an `if ... in <df>.columns` branch but <2 purity variants -- register the "
                f"branch's variant (the gate closes default-path mutation only), or allowlist with a reason"
            )
```
- [ ] **Step 2: Run** → it fires precisely (≈the `gk_role`-branching helpers), not the whole surface; add the
      missing variant or allowlist-with-reason for each. Capture.

---

## Phase 2 — Part A: fix the mutation + sort-consistency → gate GREEN

### Task 2.1: `add_gk_distribution_metrics` returns a sorted copy in every path (standard + atomic)

**Files:** `silly_kicks/spadl/utils.py` (~404), `silly_kicks/atomic/spadl/utils.py` (~mirror).

- [ ] **Step 1: Hoist the sort-copy to the top** (operate on `out`, do NOT rebind `actions`):

```python
    out = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)
    if "gk_role" not in out.columns:
        if require_gk_role:
            out = add_gk_role(out)
        else:
            out["gk_role"] = pd.Categorical([None] * len(out), categories=list(_GK_ROLE_CATEGORIES))
    # ... all subsequent reads + the 4 assignments use `out`, return `out` ...
```

      Replace every `actions[...]`/`actions.columns`/`len(actions)` in the body below the hoist with `out`.
      Mirror in atomic.
- [ ] **Step 2: Run the gate → GREEN** on `add_gk_distribution_metrics` (both variants). + run
      `tests/spadl/` + `tests/atomic/` gk-distribution tests for collateral.

### Task 2.2: Targeted regression (belt-and-suspenders)

**Files:** `tests/spadl/test_*gk*` (the gk-distribution test module) + atomic equivalent.

- [ ] **Step 1:** Add a test: `add_gk_distribution_metrics` on a `gk_role`-present input — assert the input is
      byte-unchanged (`before.equals(actions)`), `out is not actions`, and the output is sorted by
      `(game_id, period_id, action_id)`. Standard + atomic. Run → PASS. Stage.

### Task 2.3: Audit-fixes (any other `add_*` the gate flagged in Task 1.2)

- [ ] **Step 1:** For each flagged helper: assess value-impact (positionally-consistent → identity/order only;
      index-aligned-onto-unsorted or shared-array mutation → value-affecting → contingency). Fix with
      `out = actions.copy()` up front. Re-run the gate → GREEN. (If none flagged, this task is a no-op — record
      "audit clean except gk_distribution".) Stage.

---

## Phase 3 — Part D: rename `pitch_control_at_action` → `pitch_control_at_target`

### Task 3.1: Rename the function (standard + atomic) + exports + callers

**Files:** `silly_kicks/tracking/features.py`, `silly_kicks/atomic/tracking/features.py`,
`silly_kicks/tracking/__init__.py`.

- [ ] **Step 1:** Rename `def pitch_control_at_action` → `def pitch_control_at_target` (standard + atomic);
      update the `__all__` entry in `tracking/features.py`, `tracking/__init__.py` (+ its import), and the
      atomic module's `__all__`/import. Update internal callers: `add_pitch_control` + `pitch_control_xfns`
      (and atomic equivalents) call the renamed function. **Do NOT touch `col_name`** — the emitted column
      stays `pitch_control_at_target__<method>` (already correct since 4.31.0).
- [ ] **Step 2: Code-scoped grep-to-zero.** `git grep -n pitch_control_at_action -- silly_kicks tests scripts`
      → update every CODE reference (callers/imports/`__all__`/test calls + the `test_action_context`-style
      patch targets if any in this repo) → re-run → **zero remaining in code**. **Do NOT** rewrite
      `CHANGELOG.md` / `docs/superpowers/adrs/**` / historical prose (preserve the record).
- [ ] **Step 3: Run** `tests/tracking/pitch_control/` → the column-name assertions (`pitch_control_at_target__*`)
      stay GREEN (guarded invariant: the rename didn't change the column); the function-ref updates resolve.
      Stage.

---

## Phase 4 — Part C: docstrings + the doc-accuracy assertion

### Task 4.1: Column enumeration + dtypes + the ergonomics notes

**Files:** `silly_kicks/tracking/features.py` (off_ball + shot_goalmouth), `silly_kicks/spadl/utils.py`
(+ atomic) gk_distribution.

- [ ] **Step 1:** Enumerate emitted columns + dtypes in the docstrings of `add_off_ball_context`,
      `add_off_ball_runs`, `add_line_break` (+ the "for all N, use `add_off_ball_context`" cross-ref) and
      `add_shot_goalmouth` (the 11 columns). Add the `gk_pass_length_class` category-dtype Spark note +
      `gk_xt_delta` (12×8) SPADL-grid / own-grid note to `add_gk_distribution_metrics` (standard + atomic).

### Task 4.2: Doc-accuracy self-policing assertion

**Files:** `tests/test_add_star_purity.py` (or a sibling) — reuse the registry's per-helper emitted-column sets
(emitted = `set(out.columns) - set(input.columns)` from a `PURITY_ENTRIES` variant).

- [ ] **Step 1 (review #5 — set-equality, not subset, for exhaustive claims; review #4 — explicit set, NOT
      docstring parsing):** ⊆ only catches a docstring naming a non-existent column; it does NOT catch the
      Part-C goal (an emitted column the docstring FORGOT). For the helpers Part C touches whose docstring
      claims exhaustiveness (`add_gk_distribution_metrics`, `add_shot_goalmouth`, `add_off_ball_context`),
      assert **SET EQUALITY** between the columns the gate observed emitted (`emitted = set(out.columns) -
      set(input.columns)` from the helper's `PURITY_ENTRIES` variant) and an **explicit per-helper
      expected-set pinned in the test** — NOT a docstring-backtick parse (review #4: parsing is fragile — a
      stray backtick, a `` `link_quality_score` `` provenance mention, or a reflowed line silently breaks it;
      the literal `frozenset` is the contract and IS the thing a doc edit must keep in sync):

```python
_EXHAUSTIVE_EMITTED: dict[str, frozenset[str]] = {
    "add_gk_distribution_metrics": frozenset({...}),  # the gk_dist columns
    "add_shot_goalmouth": frozenset({...}),           # the 11 shot-goalmouth columns
    "add_off_ball_context": frozenset({...}),
}
```

      Then, in the SAME test, assert each helper's docstring actually NAMES every column in its pinned set
      (`assert col in fn.__doc__` per column) — this is the doc-accuracy half (the explicit set guards the
      gate; the docstring-membership check guards the prose). Helpers with non-exhaustive prose stay ⊆. Run →
      PASS. Stage.

### Task 4.3: Chain-purity e2e (review #7 — documents the real consumer contract)

**Files:** `tests/test_add_star_purity.py` (or `tests/spadl/`).

- [ ] **Step 1 (review #5 — no `convert`):** One test mirroring the real enrichment chain, starting from the
      ALREADY-BUILT actions + frames fixtures the gate uses (NOT from `convert` — a converter takes raw
      provider JSON/DataFrames, not SPADL actions; threading a real converter in here would be a provider e2e,
      out of scope and fixture-heavy). Chain only the enrichers: `add_gk_role → add_gk_distribution_metrics →
      add_off_ball_context → add_shot_goalmouth` (passing `frames` where needed). Hold a reference to the
      ORIGINAL `actions` (and `frames`) and assert byte-unchanged end-to-end (`before.equals(actions)`).
      Per-function purity implies chain purity, so this is belt-and-suspenders, but it documents the chained
      consumer contract the lakehouse actually exercises. Run → PASS. Stage.

---

## Phase 5 — full gate

### Task 5.1: Full verification

- [ ] **Step 1:** `ruff format --check . && ruff check .` → clean.
- [ ] **Step 2:** `python -m pyright` (bare, full incl `tests/`) → 0 errors.
- [ ] **Step 3:** `python -m pytest tests/ -m "not e2e and not slow" --benchmark-skip -q` (whole `tests/`,
      background per the >30s rule) → all pass.
- [ ] **Step 4: pandas-3 confirm (spec req #6):** verify `DataFrame.equals` behaves on the Categorical
      `gk_role`-present variant on a py3.12 + pandas-3 venv (`uv venv --python 3.12`; the cross-version reflex
      from PR-S95/S96) — the purity gate must pass there, not just on local pandas-2.

---

## Phase 6 — ADR + docs + version + handoff + commit

### Task 6.1: ADR-033

**Files:** Create `docs/superpowers/adrs/ADR-033-add-star-purity-gate.md`.

- [ ] Record: the auto-enumerating no-mutation gate (5 rigor reqs); the gate's **known limit** (closes
      default-path mutation, not future branch-conditional — the heuristic nudge targets it); the
      `add_gk_distribution_metrics` defect is **identity + order ONLY, no value miscompute, no recompute** —
      **scoped to that helper** (other audit-flagged helpers need per-helper value analysis); the
      `pitch_control_at_action`→`pitch_control_at_target` rename (breaking, window-justified, column-base
      byte-unchanged + guarded); the doc clarifications. Status → Accepted (PR-S97, 4.32.0).

### Task 6.2: CLAUDE.md

- [ ] Add the `add_*` input-purity gate to the auto-enumerating-gates convention list (alongside nan-safety /
      liveness / dup-action_id / id-dtype). **Contributor contract (review #6):** state that any `add_*` which
      *conditionally* adds columns (a present/absent branch) MUST register **≥2 purity variants** (both
      branches) — the AST heuristic is a best-effort nudge for the one known shape, NOT a guarantee, so the
      contract is the real backstop. Note the `pitch_control_at_action`→`at_target` rename + that the lakehouse
      keeps its own DEFCON `pitch_control_at_action` mart column (different semantics).

### Task 6.3: Version bump 4.32.0 (5-file gate)

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`.

- [ ] Bump `4.31.0 → 4.32.0`; CHANGELOG entry (the purity gate; the gk_dist mutation fix — identity/order only,
      no recompute; the breaking `pitch_control_at_action`→`pitch_control_at_target` rename; doc tightening);
      TODO (no shipped-row to remove — this is reactive to lakehouse feedback, not a TODO item); `uv lock` (or
      hand-edit the version line). Verify all five agree.

### Task 6.4: `/final-review`

- [ ] Run `/final-review` (code + docs + C4 drift). **C4 expected no-op** (no new aggregator/container; count
      28) — regenerate `docs/c4/architecture.*` and confirm no diff. Phase 2.5 ADR check → ADR-033 present.

### Task 6.5: Lakehouse handoff (copy/paste — NOT a silly-kicks TODO)

- [ ] Draft the handoff: pin 4.32.0 (skip 4.31.0); rename the 2 direct call sites + 1 test patch target
      (`enrich.py`, `tracking_context.py`); KEEP the DEFCON `pitch_control_at_action` mart column; the column
      migration (`at_ball`→`at_target`, AC+DEFCON) is unchanged from 4.31.0; gk_dist fix is identity/order only
      → no recompute; the gk_xt_delta-own-grid + gk_pass_length_class-category notes confirm their choices.

### Task 6.6: Single commit (ONLY after explicit owner approval)

- [ ] `git add -A`; one commit (subject:
      `feat(tracking)!: add_* input-purity CI gate + gk_distribution mutation fix + pitch_control_at_target rename -- silly-kicks 4.32.0 (ADR-033, PR-S97)`)
      + `Co-Authored-By` trailer. Do NOT tag. Wait for CI green (owner monitors), then tag `v4.32.0`.

---

## Self-review

- **Spec coverage:** Part A (2.1/2.2) · Part B gate + 5 rigor + discovery + AST heuristic + audit (1.1–1.3,
  2.3) · Part C docs + set-equality accuracy assertion + chain-purity (4.1/4.2/4.3) · Part D rename +
  column-base guard + code-scoped grep (3.1) · contingency (plan header + 1.2 Step 4 + 2.3) · scoped
  no-recompute (ADR 6.1) · pandas-3 equals check (5.1).
- **Round-2 review folded:** ONE canonical `PURITY_ENTRIES` registry, everything derived (Task 1.1 Step 0,
  #3); `array_equal` inexact-dtype guard (#2); cached-builder `.copy(deep=True)` rule (1.2 Step 2, #4); AST
  `if … in <df>.columns` heuristic, kwarg-toggle dropped (1.3, #1); set-equality for exhaustive docstrings
  (4.2, #5); real `_GK_ROLE_CATEGORIES` member in the fixture (#6); chain-purity e2e (4.3, #7).
- **Round-4 review folded:** discovery filter fixed — two correctly-targeted meta-asserts
  (`__all__`-based per-package + defining-submodule introspection; 1.2 Step 3, #1 BLOCKER); `inspect.unwrap`
  + `try/except (OSError, TypeError)` in `_branches_on_column_presence` (1.3 Step 1, #2); `_resolve_fn` resolves
  via `getattr` across the four package namespaces, NOT from the opaque `invoke` closure (1.3 Step 1, #3);
  explicit per-helper `frozenset` expected-set + docstring-membership check, NO backtick parse (4.2, #4); chain
  e2e drops `convert`, starts from built actions+frames (4.3, #5); AST heuristic explicitly framed best-effort
  in both docstrings + the ≥2-variants contributor contract added to CLAUDE.md (1.3 + 6.2, #6).
- **Placeholder scan:** harness + gk_dist entry + AST heuristic + meta-asserts are concrete; the 28 tracking
  entries are "lift from ENTRIES into `PURITY_ENTRIES`, build-once-fresh, `.copy(deep=True)` cached fixtures"
  (precise mechanical instruction with the caveats spelled out); the audit scope is gate-driven by design.
- **Type consistency:** `pitch_control_at_target` (function == column base) across standard/atomic/tests; the
  registry `Variant = (name, build_inputs, invoke)` + `_assert_pure(name, variant, inputs, invoke)` consistent;
  `REGISTERED_NAMES`/`_resolve_fn`/parametrization/heuristic/4.2 all derive from `PURITY_ENTRIES`.
- **RED-first under single-commit:** Task 1.1 Step 2 run-RED on the `gk_role`-present variant before Part A
  (evidence, not a commit); 1.2 Step 4 captures the audit; 2.2 is the green-confirming regression.
