# Docstring `Examples` coverage + CI guardrail (PR-S13)

**Status:** Approved (design)
**Target release:** silly-kicks 2.1.1
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.1.0 (`docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`)

---

## 1. Problem

`silly-kicks` 2.1.0's public API has ~50 functions / classes / methods. ~22 of them have `Examples` sections in their docstrings (added incrementally during PR-S8 / S10 / S11 / S12). The remaining ~28 have signatures, parameter descriptions, return-type docs — but no usage example. Consumers reading `help(...)` output, IDE hover-cards, or rendered Sphinx docs see "this is what the function takes, this is what it returns" but not "this is how you call it in practice."

`TODO.md` D-8 has tracked this gap since the PR-S8 era ("49 public functions, zero examples — start with the 10 most-used"). The "10 most-used" framing was a phase-1 hedge; the user's intent (LinkedIn-public motivation, gold-standard library positioning) is to close D-8 entirely.

A separate `TODO.md` entry, D-9, tracks "5 xthreat module-level functions (`scoring_prob`, `get_move_actions`, etc.) not underscore-prefixed but not re-exported." Inspection of the current `silly_kicks/xthreat.py` shows all 9 module-level helpers are already underscore-prefixed (`_scoring_prob`, `_get_move_actions`, `_get_cell_indexes`, etc.); `ExpectedThreat` is the only public top-level symbol. The TODO entry is stale — code is already correct.

## 2. Goals

1. **Close D-8 entirely.** Every public function / class / method in the silly-kicks public API has an `Examples` section in its docstring. Style matches the established PR-S8/S12 illustrative pattern (3-7 lines, context-free, no doctest verification).
2. **Add a CI guardrail to prevent drift.** A new test walks the public API via AST and asserts every public symbol's docstring contains an `Examples` section. Failure message points to canonical-style references. Future PRs that add a public function without Examples fail CI; the test message provides the remediation hint.
3. **Close D-9.** The TODO entry is stale (code already correct). Remove the entry; no code change required.
4. **Patch release 2.1.1.** Pure documentation + test addition. No API changes, no behavior changes.

## 3. Non-goals

1. **No doctest verification.** Examples are illustrative, not runnable in CI. (Same as PR-S8/S12 established style; doctest infrastructure cost not justified for the marginal gain.)
2. **No ADR.** The CI gate enforces the convention mechanically; an ADR would be belt-redundant. ADR-002 namespace stays unallocated for a future cross-cutting decision.
3. **No CLAUDE.md amendment.** The CI gate's failure message + canonical-style references are self-documenting enough.
4. **No Examples for private (underscore-prefixed) symbols.** Private API doesn't surface in `help()` for consumers.
5. **No Examples for pure-type symbols** (`BoundaryMetrics`, `CoverageMetrics`, `ConversionReport`, etc.) where the data shape IS the documentation. Explicit `SKIP_SYMBOLS` allowlist in the test.
6. **No example refactoring of existing examples.** PR-S8/S10/S11/S12 examples stay as written. Style consistency is a goal for new examples; rewriting old ones is out of scope.
7. **No vendored fixture changes.** Existing fixtures (committed StatsBomb JSONs, HDF5, IDSSE/Metrica parquets) are sufficient for any spot-check rendering verification.

## 4. Architecture

### 4.1 File structure

```
silly_kicks/spadl/utils.py                       MOD  +60 / 0     Examples on add_pre_shot_gk_context, add_names, validate_spadl, play_left_to_right
silly_kicks/spadl/sportec.py                     MOD  +10 / 0     Examples on convert_to_actions
silly_kicks/spadl/metrica.py                     MOD  +10 / 0     Examples on convert_to_actions
silly_kicks/spadl/kloppy.py                      MOD  +10 / 0     Examples on convert_to_actions
silly_kicks/atomic/spadl/utils.py                MOD  +60 / 0     Examples on 6 atomic mirror helpers
silly_kicks/vaep/base.py                         MOD  +60 / 0     Examples on VAEP class + 5 methods
silly_kicks/vaep/hybrid.py                       MOD  +15 / 0     Examples on HybridVAEP class
silly_kicks/atomic/vaep/base.py                  MOD  +15 / 0     Examples on AtomicVAEP class
silly_kicks/xthreat.py                           MOD  +30 / 0     Examples on ExpectedThreat class + fit/rate/interpolator
silly_kicks/vaep/features.py                     MOD  +150 / 0    Examples on gamestates, feature_column_names, ~26 feature extractors
tests/test_public_api_examples.py                NEW  ~80 lines   CI guardrail
pyproject.toml                                   MOD  +1 / -1     version 2.1.0 → 2.1.1
CHANGELOG.md                                     MOD  +15 / 0     [2.1.1] entry
TODO.md                                          MOD  +0 / -2     close D-8 + delete D-9
```

Total: ~500 lines added across 10 source files + 1 new test file. No existing tests touched.

### 4.2 Example style — illustrative

Match the PR-S8/S12 pattern (canonical references: `add_possessions`, `boundary_metrics`, `coverage_metrics`, `convert_to_atomic`, `use_tackle_winner_as_actor`).

Each Example:
- 3-7 lines of code
- Context-free: assumes `actions` is "an existing SPADL DataFrame" without showing its full creation lineage (unless the function being documented IS the entry point that produces `actions`, in which case the convert call is shown inline)
- Variable-naming consistency: `actions` (SPADL DataFrame), `states` (gamestates output), `xt` (fitted ExpectedThreat), `v` (fitted VAEP), `feats` (feature DataFrame)
- No imports inside the Example block unless the example IS demonstrating an import (e.g., `from silly_kicks.spadl import statsbomb`)

### 4.3 Per-group templates

**Group A — Standard SPADL (3 missing converters + 4 helpers).**

Converter template (matches existing `statsbomb.convert_to_actions`):
```python
Examples
--------
Convert events from <provider>::

    from silly_kicks.spadl import <provider>
    actions, report = <provider>.convert_to_actions(events, home_team_id=100)
```

Helper template (e.g., `add_names`):
```python
Examples
--------
Append name columns for human-readable diagnostics::

    actions = add_names(actions)
    actions[["type_name", "result_name", "bodypart_name"]].head()
```

**Group B — Atomic-SPADL (6 mirror helpers).**

Mirror standard-side template, with atomic-specific notes:
```python
Examples
--------
Append name columns for atomic-SPADL diagnostics::

    from silly_kicks.atomic.spadl.utils import add_names
    atomic = add_names(atomic)
```

**Group C — VAEP family.**

`VAEP` class docstring shows the full lifecycle:
```python
Examples
--------
Train and rate actions::

    from silly_kicks.vaep import VAEP

    v = VAEP()
    v.fit(games, actions)
    rated = v.rate(actions)
    # rated["vaep_value"] is the per-action value estimate.
```

`VAEP.fit`, `VAEP.rate`, `VAEP.score`, `VAEP.compute_features`, `VAEP.compute_labels` get shorter examples that show their specific call shape.

`HybridVAEP` and `AtomicVAEP` mirror the `VAEP` class example with their respective entry pattern.

**Group D — xT family.**

```python
Examples
--------
Fit an Expected Threat grid and rate actions::

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat()
    xt.fit(actions)
    values = xt.rate(actions)
```

Methods (`fit`, `rate`, `interpolator`) get shorter call-shape examples.

**Group E — vaep/features helpers.**

Shared preamble: every example assumes `states = gamestates(actions)` is available.

```python
Examples
--------
Compute action-type features for the gamestate stream::

    states = gamestates(actions, nb_prev_actions=3)
    feats = actiontype_onehot(states)
```

`gamestates` itself shows its production:
```python
Examples
--------
Build a 3-step gamestate stream::

    states = gamestates(actions, nb_prev_actions=3)
```

`feature_column_names` shows how to query expected output columns:
```python
Examples
--------
List the column names a feature function will produce::

    cols = feature_column_names([actiontype_onehot, bodypart_onehot])
```

### 4.4 CI guardrail — `tests/test_public_api_examples.py`

```python
"""Enforce: every public function / class / method docstring includes an Examples section.

Closes D-8 (PR-S13). Backstops the discipline by failing CI when a future
PR adds a public symbol without an Examples section.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files containing public API. Excludes underscore-prefixed modules,
# tests, and scripts.
_PUBLIC_MODULE_FILES = (
    "silly_kicks/spadl/utils.py",
    "silly_kicks/spadl/statsbomb.py",
    "silly_kicks/spadl/opta.py",
    "silly_kicks/spadl/wyscout.py",
    "silly_kicks/spadl/sportec.py",
    "silly_kicks/spadl/metrica.py",
    "silly_kicks/spadl/kloppy.py",
    "silly_kicks/atomic/spadl/utils.py",
    "silly_kicks/atomic/spadl/base.py",
    "silly_kicks/vaep/base.py",
    "silly_kicks/vaep/hybrid.py",
    "silly_kicks/vaep/features.py",
    "silly_kicks/atomic/vaep/base.py",
    "silly_kicks/xthreat.py",
)

# Pure-type symbols that don't fit the illustrative-example pattern.
# Adding a new entry here is a deliberate documentation-policy decision —
# the additive-only nature is a forcing function.
_SKIP_SYMBOLS = frozenset({
    "BoundaryMetrics",       # TypedDict — fields are the documentation
    "CoverageMetrics",       # TypedDict
    "ConversionReport",      # TypedDict
})


def _has_examples_section(docstring: str | None) -> bool:
    """True if the docstring contains a NumPy-style Examples section or doctest."""
    if not docstring:
        return False
    # NumPy-style: "Examples\n--------"
    if "Examples\n    --------" in docstring or "Examples\n--------" in docstring:
        return True
    # Doctest-style: ">>> "
    return ">>> " in docstring


def _walk_public_definitions(tree: ast.AST) -> list[tuple[str, int, str, ast.AST]]:
    """Yield (kind, lineno, qualified_name, node) for top-level public defs + public methods."""
    out: list[tuple[str, int, str, ast.AST]] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("function", node.lineno, node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("class", node.lineno, node.name, node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") or child.name in _SKIP_SYMBOLS:
                        continue
                    out.append(("method", child.lineno, f"{node.name}.{child.name}", child))
    return out


@pytest.mark.parametrize("file_path", _PUBLIC_MODULE_FILES)
def test_public_definitions_have_examples_section(file_path: str):
    """Every public function / class / method in <file_path> has an Examples section.

    See ``silly_kicks.spadl.add_possessions`` and ``silly_kicks.spadl.boundary_metrics``
    for canonical illustrative-style examples. Add a 3-7 line example showing typical
    usage; no doctest verification is required.
    """
    abs_path = REPO_ROOT / file_path
    assert abs_path.exists(), f"public-API module file does not exist: {file_path}"

    tree = ast.parse(abs_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for kind, lineno, name, node in _walk_public_definitions(tree):
        doc = ast.get_docstring(node)
        if not _has_examples_section(doc):
            missing.append(f"  {file_path}:{lineno}  {kind}  {name}")

    assert not missing, (
        f"Public symbols in {file_path} missing 'Examples' section in docstring:\n"
        + "\n".join(missing)
        + "\n\nAdd a 3-7 line illustrative Examples section. See "
        "`silly_kicks.spadl.add_possessions` or `silly_kicks.spadl.boundary_metrics` "
        "for the canonical style. Pure-type symbols (TypedDict / dataclass) that don't "
        "fit the example pattern can be added to `_SKIP_SYMBOLS` in this test file — "
        "but only with a clear documentation-policy justification."
    )
```

**Design choices:**
- **One parametrized test per file**, not one global. Per-file failures localize the diagnosis: when CI fails, the failing test name tells you which file to look at.
- **`_walk_public_definitions` is top-level only** — it doesn't recurse into nested classes or nested functions. Public API surfaces in silly-kicks are always top-level or one level deep (class methods); no nested public defs exist in the codebase.
- **`_has_examples_section` accepts both NumPy-style and doctest-style** so the test doesn't lock out future doctest-runnable additions if we ever change the style.
- **`_SKIP_SYMBOLS` is intentionally minimal.** Three entries today (the existing TypedDicts). Adding a new entry requires a deliberate code change — that's the forcing function: TypedDicts are easy to add accidentally; this gate makes "skip Examples for X" a visible decision rather than a silent omission.
- **Failure message points to canonical references** — `add_possessions` and `boundary_metrics` already have illustrative examples in the established style. New developers don't need to read an ADR; they read those two functions.

### 4.5 D-9 close-out

`TODO.md` currently has:

```markdown
| D-9 | Low | 5 xthreat module-level functions (`scoring_prob`, `get_move_actions`, etc.) not underscore-prefixed but not re-exported | Implementation helpers technically public API. Audit-source: DEFERRED.md (migrated 1.9.0). |
```

Reality check (verified via `grep -nE "^def [a-z]" silly_kicks/xthreat.py`): zero non-underscore-prefixed module-level functions. All helpers (`_get_cell_indexes`, `_get_flat_indexes`, `_count`, `_safe_divide`, `_scoring_prob`, `_get_move_actions`, `_get_successful_move_actions`, `_action_prob`, `_move_transition_matrix`) are already private. `ExpectedThreat` is the only public top-level symbol.

The TODO entry is stale. Action: delete the row from `TODO.md`'s "Tech Debt" table.

## 5. Implementation order

TDD-first: write the gate test before populating any examples. The test fails initially against ~28 missing surfaces; each group lands and shrinks the failure list until the test goes green. Six groups, each verified before moving to the next.

1. **Group F.1 — Failing CI gate.** Write `tests/test_public_api_examples.py` with the AST walker, `_PUBLIC_MODULE_FILES` list, and `_SKIP_SYMBOLS` allowlist. Run it locally; confirm it fails with a precise list of the ~28 currently-missing public symbols across the 14 module files. The failure list itself becomes the implementation checklist for groups A-E.
2. **Group A — Standard SPADL** (~7 examples, 4 files). Re-run the gate; verify the failure list shrinks by ~7 entries.
3. **Group B — Atomic mirrors** (~6 examples, 1 file). Re-run; failure list down by ~6.
4. **Group C — VAEP family** (~8 examples, 3 files). Re-run; down by ~8.
5. **Group D — xT family** (~4 examples, 1 file). Re-run; down by ~4.
6. **Group E — vaep/features** (~28 examples, 1 file). Largest by count; smallest per-example since each is templated. Re-run; failure list reaches 0.
7. **Group F.2 — Close-out.** Final verification (gate green); CHANGELOG `[2.1.1]`, TODO close D-8 + delete D-9, pyproject version bump 2.1.0 → 2.1.1.

Order rationale: TDD throughout matches the established discipline (PR-S12 used red-then-green for unit tests). The gate's failure-list output IS the canonical "where am I?" checklist, removing the risk of forgetting a surface. By the end, "test passes" is equivalent to "all 50 surfaces have Examples."

## 6. Verification gates (before commit)

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/

# Spot-check rendering on a representative new example per group
uv run python -c "from silly_kicks.spadl.utils import add_names; help(add_names)" | head -30

# Full suite — should remain 774 passed (4 skipped) baseline + ~14 new test cases
# (one per file in _PUBLIC_MODULE_FILES via parametrize)
uv run pytest tests/ --tb=short
```

Expected: 774 + 14 = 788 passing tests. 4 skipped (unchanged from 2.1.0).

`/final-review` runs after the gates pass.

## 7. Commit cycle

Per `feedback_commit_policy`. Branch: `feat/docstring-examples-coverage`. Single commit. Patch release.

```
1. All gates green + /final-review pass
2. User approves → git add + git commit -s (one commit)
3. User approves → git push -u origin feat/...
4. User approves → gh pr create
5. User approves → gh pr merge --admin --squash --delete-branch
6. User approves → git tag v2.1.1 + git push origin v2.1.1  # auto-fires PyPI publish
```

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Style inconsistency across 50 examples written in one cycle | Five per-group templates locked in §4.3. After Group A lands, spot-check by running `help()` on 3 examples; correct any drift before continuing. |
| Ruff formatter reflows long Example code blocks unexpectedly | Run `ruff format silly_kicks/` after each group; if reflows are unexpected, the per-group commit reveals which line lengths are over budget. |
| `_has_examples_section` matches false positives (e.g., the word "Examples" in a Notes section) | Detection requires `Examples\n    --------` (the NumPy heading separator), not just the word. Verified by inspection of the sentinel string against the existing canonical examples. |
| Some `vaep/features.py` helpers have signatures that resist 3-line examples (e.g., `feature_column_names` takes a list of fns) | Templates in §4.3 cover the awkward cases. Worst-case: a 5-line example showing setup + call + result-shape comment. |
| The CI gate over-fires on first introduction (any of the 50 new examples have wrong indentation) | Group F's test runs locally before commit; it'll fail with a precise list of misplaced examples that we fix in the same cycle. |
| New parametrized test slows CI by parsing 14 files × AST walk on every run | Negligible. Pure AST parsing on ~5000 lines total; runs in <100ms. |
| `_SKIP_SYMBOLS` becomes a dumping ground for "I don't want to write Examples" | The list lives in the test file under version control. Each addition is a code-review gate. The required justification (in code review or in the PR description) is the social forcing function. |

## 9. Acceptance criteria

1. Every public function / class / method in `_PUBLIC_MODULE_FILES` has an `Examples` section in its docstring (verified by the new parametrized test passing).
2. `tests/test_public_api_examples.py` exists, passes, and serves as a CI gate.
3. Style consistent with PR-S8/S12 illustrative pattern (verified by a manual style review during Group F).
4. `TODO.md` D-8 entry closed (moved out of "Documentation" table); D-9 entry deleted.
5. `CHANGELOG.md` has a `[2.1.1]` entry: brief, doc-only ("Added: Examples sections on all public API surfaces. New CI guardrail asserts every future public symbol has an Examples section.").
6. `pyproject.toml` version is `2.1.1`.
7. Verification gates (ruff, pyright with exact pins, full pytest suite, /final-review) all pass before commit.
8. Tag `v2.1.1` pushed → PyPI auto-publish workflow fires successfully.
9. `silly_kicks.spadl.add_possessions` / `boundary_metrics` / `coverage_metrics` examples are unchanged from 2.1.0 (no scope creep into existing examples).
10. Total test count: 788 passed (774 baseline + 14 new parametrized cases).
