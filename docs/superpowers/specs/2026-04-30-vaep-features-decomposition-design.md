# `vaep/features.py` decomposition (PR-S15)

**Status:** Approved (design)
**Target release:** silly-kicks 2.3.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.2.0 (`docs/superpowers/specs/2026-04-30-public-api-examples-coverage-completion-design.md`)

---

## 1. Problem

`silly_kicks/vaep/features.py` is 1170 lines and houses 30+ public functions spanning seven distinct concerns: framework (gamestates / `simple` decorator / `play_left_to_right` / `feature_column_names`), action-type features, result features, bodypart features, spatial features, temporal features, context features (team / score / possession), and silly-kicks-original specialty features (`cross_zone`, `assist_type`). The unnumbered TODO.md "Architecture" entry has flagged this since 1.9.0 with the trigger condition "do when next adding features to this file". No new features are imminent, but the file's size now actively hampers cognition: a reader has to scroll past 14 unrelated functions to compare bodypart features against each other, the AST gate's parametrized case for this file lists 28+ public symbols in a single failure message, and atomic-VAEP's 12-import dependency on the monolith (TODO A9) cannot be sharpened without first making the monolith into something more navigable.

This PR decomposes the monolith into a `vaep/features/` package with 8 focused submodules, updates atomic-VAEP to import per-concern, and bundles closure of three stale Tech Debt rows (A19, O-M1, O-M6).

## 2. Goals

1. **Decompose `silly_kicks/vaep/features.py` (1170 lines) into a package** `silly_kicks/vaep/features/` with 8 submodules grouped by concern, each ~50–250 lines. Closes the unnumbered "vaep/features.py 809-line decomposition" Architecture entry. (Note: was 809 in 1.9.0; grew to 1170 across PR-S4..PR-S13 GK and Examples additions.)
2. **Update `silly_kicks/atomic/vaep/features.py` to import per-concern.** 12-import monolith dependency becomes 4 grouped imports against specific submodules (`core`, `bodypart`, `context`, `temporal`). Mark A9 as **partially addressed** in TODO.md with a tightened description.
3. **Bundle TODO closes (National Park).** A19, O-M1, O-M6 reviewed and closed as stale-or-by-design — see § 4.6.
4. **Preserve full backwards compatibility.** Every existing import of `silly_kicks.vaep.features.X` keeps working via package `__init__.py` re-exports. Hyrum's-Law surface unchanged for consumers.
5. **Lock the new structure with TDD-first tests.** Three new test files written BEFORE decomposition begins (T-A backcompat, T-B layout, T-C atomic-per-concern); each submodule extraction is a red→green cycle on T-B.
6. **Minor release 2.3.0.** Per pragmatic semver: adding 8 new public submodule paths (`silly_kicks.vaep.features.core`, `.spatial`, etc.) is additive public API. Same precedent as 2.2.0's `coverage_metrics` minor bump. Backwards compat held — no opt-out needed.

## 3. Non-goals

1. **No behavior change.** Pure structural refactor; every test should pass before AND after every step.
2. **No new features.** Symbols moved verbatim, signatures unchanged.
3. **No public API removed.** Every name currently in `silly_kicks.vaep.features.__all__` (or accessible via attribute) remains accessible.
4. **No A9 closure.** Atomic still depends on standard for 12 symbols — the per-concern grouping is an improvement, not a dissolution. Full A9 closure (extracting truly-shared framework into a cross-package module) waits for the original trigger condition: atomic features genuinely needing to diverge.
5. **No CLAUDE.md amendment, no ADR.** The decomposition is internal restructure with explicit backwards-compat; the spec captures the design.
6. **No vaep/learners.py touch.** That's A19 (closed in this cycle's bundled doc cleanup but not via code change).
7. **Submodules are not underscore-prefixed.** They're public-but-not-canonical: importable, but the documented entry point remains `silly_kicks.vaep.features` (package). See § 4.2.

## 4. Architecture

### 4.1 Package layout

```
silly_kicks/vaep/features/
├── __init__.py        # re-exports the same __all__ as today's vaep/features.py
├── core.py            # Type aliases (Actions, GameStates, Features,
│                      # FeatureTransfomer), gamestates, simple decorator,
│                      # play_left_to_right, feature_column_names,
│                      # _actiontype helper                          (~250 lines)
├── actiontype.py      # actiontype, actiontype_onehot               (~50)
├── result.py          # result, result_onehot, actiontype_result_onehot,
│                      # result_onehot_prev_only,
│                      # actiontype_result_onehot_prev_only          (~150)
├── bodypart.py        # bodypart, bodypart_detailed,
│                      # bodypart_onehot, bodypart_detailed_onehot   (~150)
├── spatial.py         # startlocation, endlocation, startpolar,
│                      # endpolar, movement, space_delta             (~250)
├── temporal.py        # time, time_delta, speed                     (~100)
├── context.py         # team, player_possession_time, goalscore     (~150)
└── specialty.py       # cross_zone, assist_type                     (~100)
```

Total stays at ~1170 lines (no logic deletion — pure restructure).

**Inter-submodule dependencies:** `core` is the only submodule imported by siblings. Each non-core submodule imports the type aliases (`Actions`, `Features`, `GameStates`) and `simple` decorator from `core`, plus standard libs (`pandas`, `numpy`, `silly_kicks.spadl.config`). No sibling-to-sibling imports — flat tree under `core`. Verified by `T-D` (see § 4.5).

**Original `silly_kicks/vaep/features.py` is deleted** at the end of the migration. Python's import system disambiguates: `silly_kicks/vaep/features/` (the package directory) takes precedence once the original `.py` file is gone.

### 4.2 Visibility — Hybrid

Submodules are technically importable (`from silly_kicks.vaep.features.spatial import startlocation` works), but **the documented canonical entry point remains `silly_kicks.vaep.features`** via package re-exports.

- `silly_kicks/vaep/features/__init__.py` does `from .core import *; from .actiontype import *; ...` (or explicit re-export of `__all__` from each).
- The package's `__all__` matches the union of all submodules' `__all__` plus the existing names.
- Consumers using `from silly_kicks.vaep.features import startlocation` see no change.
- Atomic-VAEP imports per-concern (advanced/internal use). New submodule paths exist but are expected to be used sparingly outside the standard library boundary.
- Hyrum's-Law minimization: if we later split `bodypart.py` into `bodypart_id.py` + `bodypart_onehot.py`, consumers using the package re-exports never notice. Atomic would update.

### 4.3 Atomic update (A9 partial)

`silly_kicks/atomic/vaep/features.py` updates:

```python
# Before (current — 12-import monolith dependency):
from silly_kicks.vaep.features import (
    _actiontype, bodypart, bodypart_detailed, bodypart_detailed_onehot,
    bodypart_onehot, gamestates, player_possession_time, simple, speed,
    team, time, time_delta,
)

# After (12 symbols across 4 grouped imports):
from silly_kicks.vaep.features.core import _actiontype, gamestates, simple
from silly_kicks.vaep.features.bodypart import (
    bodypart, bodypart_detailed, bodypart_detailed_onehot, bodypart_onehot,
)
from silly_kicks.vaep.features.context import team, player_possession_time
from silly_kicks.vaep.features.temporal import time, time_delta, speed
```

Atomic's local type alias duplicates (lines 46-49):

```python
# Before:
Actions = pd.DataFrame
GameStates = list[pd.DataFrame]
Features = pd.DataFrame
FeatureTransfomer = Callable[[GameStates], Features]
```

become a single import:

```python
from silly_kicks.vaep.features.core import Actions, GameStates, Features, FeatureTransfomer
```

A9 in TODO.md/memory updated to: *"`atomic/vaep/features.py` per-concern coupling to `vaep/features` (12 symbols across 4 submodules — was monolith). Full decoupling deferred until atomic features need to diverge."* — open-but-narrowed.

### 4.4 CI gate (`tests/test_public_api_examples.py`)

`_PUBLIC_MODULE_FILES` updates:
- **Drop:** `"silly_kicks/vaep/features.py"` (file no longer exists)
- **Add (8 entries):**
  - `"silly_kicks/vaep/features/core.py"`
  - `"silly_kicks/vaep/features/actiontype.py"`
  - `"silly_kicks/vaep/features/result.py"`
  - `"silly_kicks/vaep/features/bodypart.py"`
  - `"silly_kicks/vaep/features/spatial.py"`
  - `"silly_kicks/vaep/features/temporal.py"`
  - `"silly_kicks/vaep/features/context.py"`
  - `"silly_kicks/vaep/features/specialty.py"`

Net: 19 → 26 entries. All 8 new entries pass immediately because every public function already has Examples sections (preserved through the move; PR-S13 / PR-S14 work intact).

The package's `__init__.py` is NOT added to the gate — it contains only re-exports (no top-level function definitions to walk via AST). The gate's `_walk_public_definitions` would find zero items in `__init__.py`, so adding it would be a no-op.

### 4.5 TDD strategy — three new test files

**Written BEFORE any decomposition begins.** Each step of the migration is a red→green cycle on T-B.

**T-A — `tests/vaep/test_features_backcompat.py`** (safety net)

Parametrized over every currently-public symbol of `silly_kicks.vaep.features`. Asserts each is importable via the package path:

```python
import importlib
import pytest

_PUBLIC_SYMBOLS: tuple[str, ...] = (
    "Actions", "Features", "FeatureTransfomer", "GameStates",
    "actiontype", "actiontype_onehot",
    "actiontype_result_onehot", "actiontype_result_onehot_prev_only",
    "assist_type", "bodypart", "bodypart_detailed", "bodypart_detailed_onehot",
    "bodypart_onehot", "cross_zone", "endlocation", "endpolar",
    "feature_column_names", "gamestates", "goalscore", "movement",
    "play_left_to_right", "player_possession_time", "result",
    "result_onehot", "result_onehot_prev_only", "simple", "space_delta",
    "speed", "startlocation", "startpolar", "team", "time", "time_delta",
)


@pytest.mark.parametrize("symbol_name", _PUBLIC_SYMBOLS)
def test_symbol_importable_from_package_path(symbol_name: str) -> None:
    """Every currently-public symbol stays importable from the package path."""
    mod = importlib.import_module("silly_kicks.vaep.features")
    assert hasattr(mod, symbol_name), (
        f"{symbol_name} no longer importable from silly_kicks.vaep.features. "
        f"Decomposition must preserve every public symbol via __init__.py re-exports."
    )
```

Initial state: PASSES (monolith has all symbols). Throughout migration: must KEEP PASSING after every move.

**T-B — `tests/vaep/test_features_submodule_layout.py`** (structure lock)

Parametrized over `(symbol_name, expected_submodule)` tuples — the design contract. Asserts `__module__` for each symbol:

```python
import importlib
import pytest

_LAYOUT: tuple[tuple[str, str], ...] = (
    # core
    ("gamestates", "core"),
    ("simple", "core"),
    ("play_left_to_right", "core"),
    ("feature_column_names", "core"),
    ("Actions", "core"),
    ("GameStates", "core"),
    ("Features", "core"),
    ("FeatureTransfomer", "core"),
    # actiontype
    ("actiontype", "actiontype"),
    ("actiontype_onehot", "actiontype"),
    # result
    ("result", "result"),
    ("result_onehot", "result"),
    ("actiontype_result_onehot", "result"),
    ("result_onehot_prev_only", "result"),
    ("actiontype_result_onehot_prev_only", "result"),
    # bodypart
    ("bodypart", "bodypart"),
    ("bodypart_detailed", "bodypart"),
    ("bodypart_onehot", "bodypart"),
    ("bodypart_detailed_onehot", "bodypart"),
    # spatial
    ("startlocation", "spatial"),
    ("endlocation", "spatial"),
    ("startpolar", "spatial"),
    ("endpolar", "spatial"),
    ("movement", "spatial"),
    ("space_delta", "spatial"),
    # temporal
    ("time", "temporal"),
    ("time_delta", "temporal"),
    ("speed", "temporal"),
    # context
    ("team", "context"),
    ("player_possession_time", "context"),
    ("goalscore", "context"),
    # specialty
    ("cross_zone", "specialty"),
    ("assist_type", "specialty"),
)


@pytest.mark.parametrize("symbol_name, expected_submodule", _LAYOUT)
def test_symbol_lives_in_expected_submodule(symbol_name: str, expected_submodule: str) -> None:
    """Each symbol is defined in its expected submodule.

    Locks the package structure so accidental moves between submodules fail fast.
    """
    mod = importlib.import_module("silly_kicks.vaep.features")
    symbol = getattr(mod, symbol_name)
    expected_full = f"silly_kicks.vaep.features.{expected_submodule}"
    actual = getattr(symbol, "__module__", None)
    assert actual == expected_full, (
        f"{symbol_name} should be defined in {expected_full} "
        f"(actually defined in {actual})"
    )
```

Initial state: 33 cases FAIL (every symbol has `__module__ = "silly_kicks.vaep.features"` — the monolith). After step-by-step decomposition: 33 cases PASS.

**T-C — `tests/atomic/test_features_per_concern_import.py`** (A9-coupling lock)

Uses `inspect.getsource()` on `silly_kicks.atomic.vaep.features` to assert imports come from per-concern submodules:

```python
import inspect

import silly_kicks.atomic.vaep.features as atomic_features


def test_atomic_imports_per_concern_not_from_monolith() -> None:
    """Atomic VAEP features imports per-concern submodules, not the monolith.

    Locks the A9-partial-closure win against future regressions where someone
    consolidates atomic's imports back to the package root.
    """
    source = inspect.getsource(atomic_features)
    forbidden_pattern = "from silly_kicks.vaep.features import"
    assert forbidden_pattern not in source, (
        f"silly_kicks.atomic.vaep.features should import from per-concern "
        f"submodules (e.g. 'from silly_kicks.vaep.features.core import ...'), "
        f"not from the package root '{forbidden_pattern}'."
    )

    # Specifically verify the 4 expected per-concern imports are present.
    expected_modules = (
        "silly_kicks.vaep.features.core",
        "silly_kicks.vaep.features.bodypart",
        "silly_kicks.vaep.features.context",
        "silly_kicks.vaep.features.temporal",
    )
    for mod_path in expected_modules:
        expected_line = f"from {mod_path} import"
        assert expected_line in source, (
            f"silly_kicks.atomic.vaep.features should import from {mod_path} "
            f"(expected line containing '{expected_line}')"
        )
```

Initial state: FAILS (atomic still imports `from silly_kicks.vaep.features import (...)`). After atomic update step: PASSES.

**T-D (folded into T-B) — no-circular-import.** Implicit in T-B passing: each submodule is importable without sibling pre-import (Python would surface circular-import errors during T-B's `importlib.import_module`). No separate test needed.

### 4.6 Bundled doc cleanup — close A19, O-M1, O-M6

National Park principle: TODO.md rows for the three Tech Debt items removed in this PR's commit. CHANGELOG entry notes each was reviewed and closed without code change:

- **A19** (`learners.py` default hyperparameters): already centralized as `_XGBOOST_DEFAULTS` / `_CATBOOST_DEFAULTS` / `_LIGHTGBM_DEFAULTS` module-level constants since 1.9.0. The audit description ("scattered across 3 functions") predates that extraction. Stale.
- **O-M1** (`statsbomb.py:209` `events.copy()`): defensive copy is correct by design. `_flatten_extra` (line 212) mutates the DataFrame by adding ~22 underscore columns; without the copy, caller's events would be mutated in place. The cost is microseconds for typical 3k-event matches. By-design.
- **O-M6** (`statsbomb.py:341-356` n×3 fidelity-check DataFrame): ~50 KB peak per match. Could be numpy-fied for marginal gain (~25 KB savings). No measurable impact.

No code changes for these items — TODO row deletion + CHANGELOG note is the work.

## 5. Implementation order (TDD-first)

1. **T1 — Branch setup.** `feat/vaep-features-decomposition` from main.

2. **T2 — Add T-A (backcompat test).** Currently passes (monolith intact). `tests/vaep/test_features_backcompat.py`.

3. **T3 — Add T-B (layout test).** Currently fails 33/33 (RED BAR). `tests/vaep/test_features_submodule_layout.py`.

4. **T4 — Convert monolith to package + create `core.py`.** This is the most delicate step.
   - Rename `silly_kicks/vaep/features.py` → `silly_kicks/vaep/features/__init__.py` (atomic git mv via stage + delete + add).
   - Extract framework symbols (type aliases, `gamestates`, `simple`, `play_left_to_right`, `feature_column_names`, `_actiontype`) into a new `silly_kicks/vaep/features/core.py`.
   - In `__init__.py`: replace those defs with `from .core import *` (or explicit re-exports).
   - Verify: T-A still passes; T-B now passes 8/33 (the core symbols); existing 807 tests still pass.

5. **T5–T11 — Extract one submodule per task.** For each of `actiontype`, `result`, `bodypart`, `spatial`, `temporal`, `context`, `specialty`:
   - Move public functions from `__init__.py` to the new submodule file.
   - Add re-export to `__init__.py` (`from .<submodule> import *`).
   - Submodule's `__all__` lists its public names.
   - Run T-B → expect more cases pass; run full pytest → expect 807 still passing.

   Atomic features.py unchanged at this stage (still imports from package root via re-exports).

6. **T12 — `__init__.py` cleanup.** After all 7 non-core submodules extracted, `__init__.py` is just re-exports. Verify it's a clean re-export hub:

   ```python
   """Public-API package for silly_kicks.vaep.features.

   Decomposed in 2.3.0 from a 1170-line monolith. Submodules are importable
   directly (`silly_kicks.vaep.features.spatial.startlocation`) but the
   canonical entry point is the package itself
   (`silly_kicks.vaep.features.startlocation`).
   """

   from .actiontype import *  # noqa: F401, F403
   from .bodypart import *  # noqa: F401, F403
   from .context import *  # noqa: F401, F403
   from .core import *  # noqa: F401, F403
   from .result import *  # noqa: F401, F403
   from .spatial import *  # noqa: F401, F403
   from .specialty import *  # noqa: F401, F403
   from .temporal import *  # noqa: F401, F403

   from .actiontype import __all__ as _at
   from .bodypart import __all__ as _bp
   from .context import __all__ as _cx
   from .core import __all__ as _cr
   from .result import __all__ as _rs
   from .spatial import __all__ as _sp
   from .specialty import __all__ as _spec
   from .temporal import __all__ as _tm

   __all__ = sorted({*_at, *_bp, *_cx, *_cr, *_rs, *_sp, *_spec, *_tm})
   ```

   T-A passes; T-B passes 33/33.

7. **T13 — Atomic update.** Update `silly_kicks/atomic/vaep/features.py`:
   - Replace single multi-line monolith import with 4 grouped per-concern imports.
   - Remove local `Actions = ... ; GameStates = ...` aliases; import from `core`.
   - Run T-B + T-A still passing; T-C now passes; existing atomic tests still passing.

8. **T14 — Add T-C (atomic-per-concern test).** `tests/atomic/test_features_per_concern_import.py`. Should PASS (T13 satisfied it).

9. **T15 — CI gate update.** `_PUBLIC_MODULE_FILES`: drop `vaep/features.py`, add 8 submodule paths. Run gate → 26 cases pass.

10. **T16 — Verification gates.** ruff check, ruff format --check, pyright, full pytest (-m "not e2e"), spot-check rendered docstrings.

11. **T17 — Bundled doc cleanup.** TODO.md: delete A19, O-M1, O-M6 rows; update A9 entry to "partially addressed". CHANGELOG entry for 2.3.0 covers decomposition + atomic update + closed Tech Debt.

12. **T18 — Version bump.** pyproject.toml 2.2.0 → 2.3.0.

13. **T19 — `/final-review`.**

14. **T20 — Single-commit gate (USER APPROVAL).**

15. **T21 — Push, PR, CI watch, squash-merge --admin, tag v2.3.0, verify PyPI.**

## 6. Verification gates (before commit)

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/

# Smoke test — package import + a representative symbol from each submodule
uv run python -c "
from silly_kicks.vaep.features import gamestates, startlocation, time, bodypart, team, cross_zone
from silly_kicks.vaep.features.core import gamestates as gs2
from silly_kicks.vaep.features.spatial import startlocation as sl2
assert gamestates is gs2 and startlocation is sl2
print('smoke OK')
"

# Full pytest suite (-m 'not e2e')
uv run pytest tests/ -m "not e2e" --tb=short
```

Expected: ruff/pyright clean; smoke prints `smoke OK`; pytest = **881 passing, 4 deselected** (807 baseline + 33 T-A parametrized cases + 33 T-B parametrized cases + 1 T-C + 7 net new gate parametrize cases (gate drops `vaep/features.py` and adds 8 submodule paths, net +7)). Test count is exact: 807 + 33 + 33 + 1 + 7 = 881.

## 7. Commit cycle

Same as PR-S14. Single commit. Branch: `feat/vaep-features-decomposition`. Version bump 2.2.0 → 2.3.0 (minor — additive public API via 8 new submodule paths).

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Circular imports between submodules | Design constraint: only `core` is imported by siblings. T-B passing implies all submodules importable in isolation. Smoke test (§ 6) catches regression. |
| `from .X import *` skipping symbols not in `__all__` | Each submodule MUST declare `__all__` listing its public names. Spec self-review verifies this. T-A catches dropped symbols. |
| Atomic test breakage from import change | T-C verifies the new shape, but the substantive verification is the existing atomic test suite (~80 tests) staying green at T13. |
| pyright + pandas-stubs reaction to type aliases moving | `Actions = pd.DataFrame` etc. resolve identically when imported from the new path. Verified by pyright run at T16. |
| Public submodule paths leaking into Hyrum's-Law contract | Spec § 4.2 documents the canonical entry as the package; atomic uses submodule paths but is internal. Future consumers expected to follow `__all__`-via-package convention. Doc note in package `__init__.py` (T12) makes this explicit. |
| The git mv of `features.py` → `features/__init__.py` showing as delete + add in `git diff` | Use `git mv` explicitly so history is preserved; verify `git log --follow silly_kicks/vaep/features/__init__.py` shows the prior history. |
| ruff/isort sorting `from .X import *` lines into a non-canonical order | Single-line `from .X import *` per submodule; ruff sorts alphabetically. Confirmed in T12's `__init__.py` template. |

## 9. Acceptance criteria

1. `silly_kicks/vaep/features/` exists as a package; `silly_kicks/vaep/features.py` no longer exists.
2. 8 submodules created (`core`, `actiontype`, `result`, `bodypart`, `spatial`, `temporal`, `context`, `specialty`), each with appropriate `__all__` and Examples-on-every-public-fn.
3. `__init__.py` re-exports every previously-public symbol; `__all__` is the union.
4. `silly_kicks/atomic/vaep/features.py` imports per-concern from 4 submodules; local type alias duplicates removed.
5. T-A, T-B, T-C all pass (33 + 33 + 1 = 67 new tests).
6. CI gate `_PUBLIC_MODULE_FILES` lists 8 submodule paths (was: 1 monolith path); gate count 26 (was 19).
7. TODO.md: A19, O-M1, O-M6 rows deleted; A9 row updated to "partially addressed (per-concern coupling — full decoupling deferred)".
8. CHANGELOG.md `[2.3.0]` entry covers decomposition, atomic update, A9-partial / A19/O-M1/O-M6 close.
9. `pyproject.toml` version is `2.3.0`.
10. Verification gates pass (ruff/format/pyright/pytest); test count exactly 881 passing (807 baseline + 33 T-A + 33 T-B + 1 T-C + 7 net gate parametrize delta), 4 deselected.
11. `/final-review` clean.
12. Tag `v2.3.0` pushed → PyPI auto-publish workflow fires successfully.
13. Public API surface unchanged from consumer perspective: `from silly_kicks.vaep.features import X` works for every X that worked at 2.2.0.
