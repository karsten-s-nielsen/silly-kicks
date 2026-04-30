# `vaep/features.py` decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 1170-line `silly_kicks/vaep/features.py` monolith into a `silly_kicks/vaep/features/` package with 8 concern-focused submodules, update atomic-VAEP to import per-concern, and ship as silly-kicks 2.3.0.

**Architecture:** Pure structural refactor — no behavior change. Hybrid visibility: submodule paths are importable but the canonical entry point remains `silly_kicks.vaep.features` (package re-exports preserve every existing import). TDD-first: three new test files (T-A backcompat, T-B layout, T-C atomic-per-concern) added BEFORE decomposition; each submodule extraction is a red→green cycle on T-B.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113, uv.

**Commit policy:** Per user's standing rule, **literally one commit per branch, gated on explicit user approval at the end**. No WIP commits, no per-task commits. All changes accumulate on `feat/vaep-features-decomposition`. Spec + plan ride with the implementation commit.

**Spec:** `docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md`.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `silly_kicks/vaep/features.py` | **Delete** | Monolith — content distributed to package submodules. |
| `silly_kicks/vaep/features/__init__.py` | **Create** | Package re-export hub; `__all__` is union of all submodules'. |
| `silly_kicks/vaep/features/core.py` | **Create** | Type aliases, `gamestates`, `simple` decorator, `play_left_to_right`, `feature_column_names`, `_actiontype` helper. |
| `silly_kicks/vaep/features/actiontype.py` | **Create** | `actiontype`, `actiontype_onehot`. |
| `silly_kicks/vaep/features/result.py` | **Create** | `result`, `result_onehot`, `actiontype_result_onehot`, `result_onehot_prev_only`, `actiontype_result_onehot_prev_only`. |
| `silly_kicks/vaep/features/bodypart.py` | **Create** | `bodypart`, `bodypart_detailed`, `bodypart_onehot`, `bodypart_detailed_onehot`. |
| `silly_kicks/vaep/features/spatial.py` | **Create** | `startlocation`, `endlocation`, `startpolar`, `endpolar`, `movement`, `space_delta`. |
| `silly_kicks/vaep/features/temporal.py` | **Create** | `time`, `time_delta`, `speed`. |
| `silly_kicks/vaep/features/context.py` | **Create** | `team`, `player_possession_time`, `goalscore`. |
| `silly_kicks/vaep/features/specialty.py` | **Create** | `cross_zone`, `assist_type`. |
| `silly_kicks/atomic/vaep/features.py` | **Modify** | Imports update from monolith → per-concern submodules; remove DRY type alias duplicates. |
| `tests/vaep/test_features_backcompat.py` | **Create** | T-A: parametrized over 33 public symbols; assert each is importable via `silly_kicks.vaep.features`. |
| `tests/vaep/test_features_submodule_layout.py` | **Create** | T-B: parametrized over `(symbol, expected_submodule)`; assert `__module__` matches. |
| `tests/atomic/test_features_per_concern_import.py` | **Create** | T-C: assert atomic imports from per-concern submodules, not the package root. |
| `tests/test_public_api_examples.py` | **Modify** | `_PUBLIC_MODULE_FILES`: drop `vaep/features.py`, add 8 submodule paths (19 → 26 entries). |
| `TODO.md` | **Modify** | Update A9 (partially addressed); delete A19, O-M1, O-M6, vaep/features-decomposition rows. |
| `CHANGELOG.md` | **Modify** | New `[2.3.0]` entry. |
| `pyproject.toml` | **Modify** | Version 2.2.0 → 2.3.0. |

---

### Task 0: Branch setup

**Files:** Working tree (no file changes)

- [ ] **Step 1: Confirm clean main + spec/plan present**

```bash
git status --short
```

Expected:
```
?? README.md.backup
?? docs/superpowers/plans/2026-04-30-vaep-features-decomposition.md
?? docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md
?? uv.lock
```

If anything else appears, STOP and check with the user.

- [ ] **Step 2: Create the feature branch**

```bash
git switch -c feat/vaep-features-decomposition
```

Expected: `Switched to a new branch 'feat/vaep-features-decomposition'`.

- [ ] **Step 3: Confirm on the new branch**

```bash
git branch --show-current
```

Expected: `feat/vaep-features-decomposition`.

---

### Task 1: Add T-A (backcompat test) — passes initially

**Files:**
- Create: `tests/vaep/test_features_backcompat.py`

- [ ] **Step 1: Write the test file**

Use Write to create `tests/vaep/test_features_backcompat.py`:

```python
"""Backwards-compat: every currently-public symbol of silly_kicks.vaep.features
remains importable via the package path through and after the 2.3.0 decomposition.
"""

from __future__ import annotations

import importlib

import pytest

_PUBLIC_SYMBOLS: tuple[str, ...] = (
    "Actions",
    "Features",
    "FeatureTransfomer",
    "GameStates",
    "actiontype",
    "actiontype_onehot",
    "actiontype_result_onehot",
    "actiontype_result_onehot_prev_only",
    "assist_type",
    "bodypart",
    "bodypart_detailed",
    "bodypart_detailed_onehot",
    "bodypart_onehot",
    "cross_zone",
    "endlocation",
    "endpolar",
    "feature_column_names",
    "gamestates",
    "goalscore",
    "movement",
    "play_left_to_right",
    "player_possession_time",
    "result",
    "result_onehot",
    "result_onehot_prev_only",
    "simple",
    "space_delta",
    "speed",
    "startlocation",
    "startpolar",
    "team",
    "time",
    "time_delta",
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

- [ ] **Step 2: Verify test PASSES (monolith intact)**

```bash
uv run pytest tests/vaep/test_features_backcompat.py -v 2>&1 | tail -10
```

Expected: `33 passed in ...`. If any FAIL, STOP — that means a public symbol assumption in `_PUBLIC_SYMBOLS` is wrong (e.g., spelled wrong, or never existed). Investigate before continuing.

---

### Task 2: Add T-B (layout test) — fails 33/33 (TDD red bar)

**Files:**
- Create: `tests/vaep/test_features_submodule_layout.py`

- [ ] **Step 1: Write the test file**

Use Write to create `tests/vaep/test_features_submodule_layout.py`:

```python
"""Submodule layout lock: each public symbol of silly_kicks.vaep.features
is defined in its expected submodule. Locks the 2.3.0 decomposition contract.
"""

from __future__ import annotations

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

    For function/class symbols, ``__module__`` is the canonical source. For
    type aliases (``Actions = pd.DataFrame``), the alias does not carry a
    ``__module__`` attribute — we instead verify that the submodule's source
    actually defines the alias.
    """
    mod = importlib.import_module("silly_kicks.vaep.features")
    symbol = getattr(mod, symbol_name)
    expected_full = f"silly_kicks.vaep.features.{expected_submodule}"
    actual = getattr(symbol, "__module__", None)
    if actual is not None:
        assert actual == expected_full, (
            f"{symbol_name} should be defined in {expected_full} "
            f"(actually defined in {actual})"
        )
    else:
        # Type aliases (e.g. Actions = pd.DataFrame) don't have __module__.
        # Verify the alias is reachable via the expected submodule path.
        sub = importlib.import_module(expected_full)
        assert hasattr(sub, symbol_name), (
            f"Type alias {symbol_name} should be present in {expected_full} "
            f"(not found via module attribute)"
        )
        assert getattr(sub, symbol_name) is symbol, (
            f"{symbol_name} accessible via package and submodule but they "
            f"resolve to different objects"
        )
```

- [ ] **Step 2: Verify test FAILS 33/33 (RED BAR)**

```bash
uv run pytest tests/vaep/test_features_submodule_layout.py -v 2>&1 | tail -10
```

Expected: `33 failed in ...`. The error messages should all be the form `<symbol> should be defined in silly_kicks.vaep.features.<submodule> (actually defined in silly_kicks.vaep.features)`.

This is the TDD red bar for the entire decomposition. Each subsequent submodule extraction will flip a parametrize-case from FAIL → PASS.

---

### Task 3: Convert monolith to package + create `core.py`

**Files:**
- Delete: `silly_kicks/vaep/features.py`
- Create: `silly_kicks/vaep/features/__init__.py` (initial form: contains everything except core symbols)
- Create: `silly_kicks/vaep/features/core.py`

This is the most delicate task. The monolith file is converted to a package directory, and the framework symbols (type aliases + `gamestates` + `simple` + `play_left_to_right` + `feature_column_names` + `_actiontype`) move into the new `core.py`.

- [ ] **Step 1: Read the current monolith content**

```bash
ls -la silly_kicks/vaep/features.py
```

Confirm 1170 lines, single file. We'll use this as the source for split content.

- [ ] **Step 2: Use git mv to begin the rename**

```bash
mkdir -p silly_kicks/vaep/features
git mv silly_kicks/vaep/features.py silly_kicks/vaep/features/__init__.py
```

Expected: `git status --short` now shows
```
R  silly_kicks/vaep/features.py -> silly_kicks/vaep/features/__init__.py
```

The file is now at `silly_kicks/vaep/features/__init__.py` with the same 1170-line content as before. Python imports continue to work because Python prefers a package (`features/__init__.py`) over an absent `features.py`.

- [ ] **Step 3: Verify imports still work**

```bash
uv run python -c "from silly_kicks.vaep.features import gamestates, startlocation; print('OK')" 2>&1
```

Expected: `OK`. If `ImportError`, STOP — the package conversion failed.

- [ ] **Step 4: Create `silly_kicks/vaep/features/core.py` with framework symbols**

Use Write to create `silly_kicks/vaep/features/core.py`. This file contains:

1. The header docstring + imports identical to the original `__init__.py:1-10`
2. The four type aliases (`Actions`, `GameStates`, `Features`, `FeatureTransfomer`) from `__init__.py:12-15`
3. The `feature_column_names` function (verbatim, currently at `__init__.py:18-66`)
4. The `gamestates` function (verbatim, currently at `__init__.py:69-126`)
5. The `play_left_to_right` function (verbatim, currently at `__init__.py:128-169`)
6. The `simple` decorator (verbatim, currently at `__init__.py:171-209`)
7. The `_actiontype` helper (verbatim, currently at `__init__.py:211-237`)
8. An `__all__` list at the top declaring the public-export contract

Concrete file content template:

```python
"""Framework primitives for VAEP feature transformers — type aliases,
``gamestates``, ``simple`` decorator, ``play_left_to_right``,
``feature_column_names``, and the ``_actiontype`` helper used by both
the standard and atomic feature stacks.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, no_type_check

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

__all__ = [
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "feature_column_names",
    "gamestates",
    "play_left_to_right",
    "simple",
]

Actions = pd.DataFrame
GameStates = list[pd.DataFrame]
Features = pd.DataFrame
FeatureTransfomer = Callable[[GameStates], Features]


# === feature_column_names — copy verbatim from features/__init__.py:18-66 ===
def feature_column_names(fs: list[FeatureTransfomer], nb_prev_actions: int = 3) -> list[str]:
    """Return the names of the features generated by a list of transformers.
    ... (full docstring + body copied verbatim from __init__.py:18-66) ...
    """
    # ... full implementation ...


# === gamestates — copy verbatim from features/__init__.py:69-126 ===
def gamestates(actions: Actions, nb_prev_actions: int = 3) -> GameStates:
    # ... full implementation ...


# === play_left_to_right — copy verbatim from features/__init__.py:128-169 ===
def play_left_to_right(gamestates: GameStates, home_team_id: int) -> GameStates:
    # ... full implementation ...


# === simple — copy verbatim from features/__init__.py:171-209 ===
def simple(actionfn: Callable) -> FeatureTransfomer:
    # ... full implementation ...


# === _actiontype — copy verbatim from features/__init__.py:211-237 ===
def _actiontype(actions: Actions, _spadl_cfg: Any = None) -> Features:
    # ... full implementation ...
```

**Implementation note:** Read `silly_kicks/vaep/features/__init__.py` lines 1-237, copy the type aliases + 5 functions verbatim into `core.py` after the `__all__` declaration. Note that `_actiontype` is private (underscore-prefixed) and is intentionally excluded from `__all__` — it's accessed by atomic via `from silly_kicks.vaep.features.core import _actiontype` (works for direct submodule imports even when not in `__all__`).

- [ ] **Step 5: Update `__init__.py` — delete the moved code, add the re-export**

Use Edit to:
1. Delete the entire block from `__init__.py:1-237` (header + imports + type aliases + the 5 framework functions). Replace with:

```python
"""Public-API package for ``silly_kicks.vaep.features``.

Decomposed in 2.3.0 from a 1170-line monolith into 8 concern-focused submodules.
Submodules are importable directly (``silly_kicks.vaep.features.spatial.startlocation``),
but the canonical entry point remains the package itself
(``silly_kicks.vaep.features.startlocation``). All 33 previously-public symbols
remain importable via the package path; new code is encouraged to use the
package import for backwards-compat.

See ``docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md``
for the rationale and submodule layout.
"""

from .core import *  # noqa: F401, F403

from .core import __all__ as _core_all

__all__ = sorted({*_core_all})
```

(More submodules will be added to the re-export list in subsequent tasks.)

The remaining ~933 lines of `__init__.py` (the non-core feature transformers) stay in place for now; subsequent tasks will progressively extract them.

**Implementation note:** Use Edit with `old_string` matching the original file's first ~237 lines and `new_string` being the new package docstring + re-export block above. Be precise with line breaks and trailing newlines.

- [ ] **Step 6: Verify T-B partial pass + T-A still pass + smoke test**

```bash
uv run python -c "from silly_kicks.vaep.features import gamestates, startlocation, simple; print('OK')" 2>&1
```

Expected: `OK`.

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -v 2>&1 | tail -50
```

Expected: T-A 33/33 pass; T-B 8/33 pass (`Actions`, `FeatureTransfomer`, `Features`, `GameStates`, `feature_column_names`, `gamestates`, `play_left_to_right`, `simple` — the 8 core symbols), 25/33 still failing.

- [ ] **Step 7: Run full pytest — confirm no regression in existing tests**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -10
```

Expected: 807 baseline tests still passing (plus T-A's 33 pass + T-B's 8 pass + 25 fail). Roughly: `840 passed, 25 failed, 4 deselected`. No regressions in pre-existing tests; the only failures are T-B parametrize cases that subsequent tasks will close.

If any pre-existing test fails, STOP — investigate. Most likely cause: a function in `core.py` was incorrectly extracted or differs from the original.

---

### Task 4: Extract `actiontype.py`

**Files:**
- Create: `silly_kicks/vaep/features/actiontype.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `actiontype` (currently at `__init__.py:239-262`), `actiontype_onehot` (currently at `__init__.py:264-291`).

- [ ] **Step 1: Write `actiontype.py`**

Use Write to create `silly_kicks/vaep/features/actiontype.py`:

```python
"""Action-type feature transformers.

Two transformers: integer-typed and one-hot-encoded action-type per gamestate
slot. Both delegate the standard / atomic split via the ``_actiontype`` helper
in ``core``.
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, _actiontype, simple

__all__ = ["actiontype", "actiontype_onehot"]


# === actiontype — copy verbatim from features/__init__.py:239-262 ===
@simple
def actiontype(actions: Actions) -> Features:
    """..."""
    # ... copy body from __init__.py:239-262 ...


# === actiontype_onehot — copy verbatim from features/__init__.py:264-291 ===
@simple
def actiontype_onehot(actions: Actions) -> Features:
    """..."""
    # ... copy body from __init__.py:264-291 ...
```

**Implementation note:** Read `silly_kicks/vaep/features/__init__.py` lines 239-291, copy the `@simple def actiontype(...)` and `@simple def actiontype_onehot(...)` definitions verbatim. Both functions reference `spadlcfg.actiontypes` so the `import silly_kicks.spadl.config as spadlcfg` is required.

- [ ] **Step 2: Edit `__init__.py` — delete the moved functions**

Use Edit to delete the block at `__init__.py:238-291` (the two `@simple def actiontype...` definitions plus the empty line before them). Match on the exact lines (use Read first if needed).

- [ ] **Step 3: Update `__init__.py` re-exports**

Use Edit to expand the re-export block. Replace:

```python
from .core import *  # noqa: F401, F403

from .core import __all__ as _core_all

__all__ = sorted({*_core_all})
```

with:

```python
from .actiontype import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .core import __all__ as _core_all

__all__ = sorted({*_at_all, *_core_all})
```

- [ ] **Step 4: Verify T-B progress + T-A intact + full pytest passing**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -10
```

Expected: T-A 33/33 pass; T-B 10/33 pass (added: `actiontype`, `actiontype_onehot`).

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 807 + 33 (T-A) + 10 (T-B) = 850 passing, 23 failing (remaining T-B), 4 deselected.

---

### Task 5: Extract `result.py`

**Files:**
- Create: `silly_kicks/vaep/features/result.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `result` (`__init__.py:293-322`), `result_onehot` (`:324-351`), `actiontype_result_onehot` (`:353-380`), `result_onehot_prev_only` (`:381-415`), `actiontype_result_onehot_prev_only` (`:416-448`).

Note: `result_onehot_prev_only` and `actiontype_result_onehot_prev_only` are NOT decorated with `@simple`; they consume `GameStates` directly.

- [ ] **Step 1: Write `result.py`**

Use Write to create `silly_kicks/vaep/features/result.py`:

```python
"""Result feature transformers.

Five transformers: integer-typed result, one-hot result, joined actiontype +
result one-hot, and prev-only variants (zero-out the current action's result
to avoid leakage in models like HybridVAEP).
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, GameStates, simple

__all__ = [
    "actiontype_result_onehot",
    "actiontype_result_onehot_prev_only",
    "result",
    "result_onehot",
    "result_onehot_prev_only",
]


# === result — copy verbatim from features/__init__.py:293-322 ===
@simple
def result(actions: Actions) -> Features:
    # ...


# === result_onehot — copy verbatim from features/__init__.py:324-351 ===
@simple
def result_onehot(actions: Actions) -> Features:
    # ...


# === actiontype_result_onehot — copy verbatim from features/__init__.py:353-380 ===
@simple
def actiontype_result_onehot(actions: Actions) -> Features:
    # ...


# === result_onehot_prev_only — copy verbatim from features/__init__.py:381-415 ===
def result_onehot_prev_only(gamestates: GameStates) -> Features:
    # ...


# === actiontype_result_onehot_prev_only — copy verbatim from features/__init__.py:416-448 ===
def actiontype_result_onehot_prev_only(gamestates: GameStates) -> Features:
    # ...
```

**Implementation note:** Read original lines 293-448, copy 5 functions verbatim. Note the last two consume `GameStates` (no `@simple` decorator).

- [ ] **Step 2: Edit `__init__.py` — delete moved functions**

Use Edit to delete the block at `__init__.py:292-448` (the 5 result functions; preserve any blank lines between sections).

- [ ] **Step 3: Update `__init__.py` re-exports**

Use Edit to add `result` to the re-export block:

```python
from .actiontype import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403
from .result import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .core import __all__ as _core_all
from .result import __all__ as _rs_all

__all__ = sorted({*_at_all, *_core_all, *_rs_all})
```

- [ ] **Step 4: Verify T-B progress + full pytest passing**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -5
```

Expected: T-A 33/33; T-B 15/33 (added 5 result symbols).

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 855 passing, 18 failing, 4 deselected.

---

### Task 6: Extract `bodypart.py`

**Files:**
- Create: `silly_kicks/vaep/features/bodypart.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `bodypart` (`__init__.py:450-491`), `bodypart_detailed` (`:493-530`), `bodypart_onehot` (`:532-579`), `bodypart_detailed_onehot` (`:581-627`).

- [ ] **Step 1: Write `bodypart.py`**

Use Write to create:

```python
"""Bodypart feature transformers.

Four transformers: integer-typed bodypart, integer-typed detailed bodypart
(``head/other`` split), one-hot bodypart, one-hot detailed bodypart.
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, simple

__all__ = [
    "bodypart",
    "bodypart_detailed",
    "bodypart_detailed_onehot",
    "bodypart_onehot",
]


# === bodypart — copy verbatim from features/__init__.py:450-491 ===
@simple
def bodypart(actions: Actions) -> Features:
    # ...


# === bodypart_detailed — copy verbatim from features/__init__.py:493-530 ===
@simple
def bodypart_detailed(actions: Actions) -> Features:
    # ...


# === bodypart_onehot — copy verbatim from features/__init__.py:532-579 ===
@simple
def bodypart_onehot(actions: Actions) -> Features:
    # ...


# === bodypart_detailed_onehot — copy verbatim from features/__init__.py:581-627 ===
@simple
def bodypart_detailed_onehot(actions: Actions) -> Features:
    # ...
```

**Implementation note:** Read original lines 450-627, copy 4 functions verbatim.

- [ ] **Step 2: Edit `__init__.py` — delete moved functions**

Delete the block at `__init__.py:449-627` (the 4 bodypart functions).

- [ ] **Step 3: Update `__init__.py` re-exports**

Add `from .bodypart import *` and update `__all__`:

```python
from .actiontype import *  # noqa: F401, F403
from .bodypart import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403
from .result import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .bodypart import __all__ as _bp_all
from .core import __all__ as _core_all
from .result import __all__ as _rs_all

__all__ = sorted({*_at_all, *_bp_all, *_core_all, *_rs_all})
```

- [ ] **Step 4: Verify T-B + full pytest**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -5
```

Expected: T-B 19/33 (added 4 bodypart).

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 859 passing, 14 failing, 4 deselected.

---

### Task 7: Extract `spatial.py`

**Files:**
- Create: `silly_kicks/vaep/features/spatial.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `startlocation` (`__init__.py:667-690`), `endlocation` (`:692-715`), `startpolar` (`:717-748`), `endpolar` (`:750-781`), `movement` (`:783-811`), `space_delta` (`:913-946`).

Note: `space_delta` is at line 913, NOT contiguous with the others. It consumes `GameStates` (no `@simple`).

- [ ] **Step 1: Write `spatial.py`**

Use Write to create:

```python
"""Spatial feature transformers.

Six transformers: start/end location coordinates, start/end polar (distance +
angle to opponent goal), movement (per-action delta), and space_delta
(gamestate-to-gamestate spatial change).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, GameStates, simple

__all__ = [
    "endlocation",
    "endpolar",
    "movement",
    "space_delta",
    "startlocation",
    "startpolar",
]


_goal_x: float = spadlcfg.field_length
_goal_y: float = spadlcfg.field_width / 2


# === startlocation — copy verbatim from features/__init__.py:667-690 ===
@simple
def startlocation(actions: Actions) -> Features:
    # ...


# === endlocation — copy verbatim from features/__init__.py:692-715 ===
@simple
def endlocation(actions: Actions) -> Features:
    # ...


# === startpolar — copy verbatim from features/__init__.py:717-748 ===
@simple
def startpolar(actions: Actions) -> Features:
    # ...


# === endpolar — copy verbatim from features/__init__.py:750-781 ===
@simple
def endpolar(actions: Actions) -> Features:
    # ...


# === movement — copy verbatim from features/__init__.py:783-811 ===
@simple
def movement(actions: Actions) -> Features:
    # ...


# === space_delta — copy verbatim from features/__init__.py:913-946 ===
def space_delta(gamestates: GameStates) -> Features:
    # ...
```

**Implementation note:** Read original lines 667-811 (5 contiguous functions) AND lines 913-946 (`space_delta`, non-contiguous). Verify if the original file has any `_goal_x` / `_goal_y` module-level constants near these functions — if so, hoist them to `spatial.py`. Confirm by Reading lines around `startpolar`/`endpolar` in the original.

If `_goal_x` and `_goal_y` exist in the original file, they belong with the polar transformers (used to compute `dist_to_goal` / `angle_to_goal`). Move them to `spatial.py`. Verify by checking the polar functions' implementations.

- [ ] **Step 2: Edit `__init__.py` — delete moved functions**

Two Edit operations needed (non-contiguous moves):
1. Delete block at `__init__.py:666-811` (5 spatial functions + their `_goal_x`/`_goal_y` constants if present)
2. Delete block at the post-extraction-shifted location of `space_delta` (~line 913 originally, will have shifted by ~146 lines after the bodypart extraction in Task 6, so calculate fresh — use Read to find current location)

**Implementation note:** After each prior extraction, line numbers shift. Always Read `__init__.py` to find current line ranges before each Edit.

- [ ] **Step 3: Update `__init__.py` re-exports**

Add `from .spatial import *` and update `__all__`:

```python
from .actiontype import *  # noqa: F401, F403
from .bodypart import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403
from .result import *  # noqa: F401, F403
from .spatial import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .bodypart import __all__ as _bp_all
from .core import __all__ as _core_all
from .result import __all__ as _rs_all
from .spatial import __all__ as _sp_all

__all__ = sorted({*_at_all, *_bp_all, *_core_all, *_rs_all, *_sp_all})
```

- [ ] **Step 4: Verify T-B + full pytest**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -5
```

Expected: T-B 25/33 (added 6 spatial).

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 865 passing, 8 failing, 4 deselected.

---

### Task 8: Extract `temporal.py`

**Files:**
- Create: `silly_kicks/vaep/features/temporal.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `time` (`__init__.py:629-665` originally), `time_delta` (`:884-912`), `speed` (`:947-985`).

- [ ] **Step 1: Write `temporal.py`**

```python
"""Temporal feature transformers.

Three transformers: ``time`` (period + minute + second from time_seconds),
``time_delta`` (gamestate-to-gamestate dt), ``speed`` (movement / dt).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .core import Actions, Features, GameStates, simple

__all__ = ["speed", "time", "time_delta"]


# === time — copy verbatim from features/__init__.py:629-665 ===
@simple
def time(actions: Actions) -> Features:
    # ...


# === time_delta — copy verbatim from features/__init__.py:884-912 ===
def time_delta(gamestates: GameStates) -> Features:
    # ...


# === speed — copy verbatim from features/__init__.py:947-985 ===
def speed(gamestates: GameStates) -> Features:
    # ...
```

**Implementation note:** `time` is `@simple`-decorated; `time_delta` and `speed` consume `GameStates` directly. Confirm by reading the original.

- [ ] **Step 2: Edit `__init__.py` — delete moved functions**

Three non-contiguous Edits (Read first to find current line ranges; substantial shifts have occurred from earlier tasks).

- [ ] **Step 3: Update `__init__.py` re-exports**

Add `from .temporal import *`:

```python
from .actiontype import *  # noqa: F401, F403
from .bodypart import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403
from .result import *  # noqa: F401, F403
from .spatial import *  # noqa: F401, F403
from .temporal import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .bodypart import __all__ as _bp_all
from .core import __all__ as _core_all
from .result import __all__ as _rs_all
from .spatial import __all__ as _sp_all
from .temporal import __all__ as _tm_all

__all__ = sorted({*_at_all, *_bp_all, *_core_all, *_rs_all, *_sp_all, *_tm_all})
```

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -5
```

Expected: T-B 28/33.

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 868 passing, 5 failing, 4 deselected.

---

### Task 9: Extract `context.py`

**Files:**
- Create: `silly_kicks/vaep/features/context.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `player_possession_time` (`__init__.py:813-850`), `team` (`:851-883`), `goalscore` (`:986-1026`).

- [ ] **Step 1: Write `context.py`**

```python
"""Context feature transformers.

Three transformers: ``team`` (same-team-as-actor mask per gamestate slot),
``player_possession_time`` (per-player rolling possession seconds),
``goalscore`` (cumulative score / opponent / diff at each action).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .core import Actions, Features, GameStates, simple

__all__ = ["goalscore", "player_possession_time", "team"]


# === player_possession_time — copy verbatim from features/__init__.py:813-850 ===
@simple
def player_possession_time(actions: Actions) -> Features:
    # ...


# === team — copy verbatim from features/__init__.py:851-883 ===
def team(gamestates: GameStates) -> Features:
    # ...


# === goalscore — copy verbatim from features/__init__.py:986-1026 ===
def goalscore(gamestates: GameStates) -> Features:
    # ...
```

**Implementation note:** `player_possession_time` is `@simple`-decorated; `team` and `goalscore` consume `GameStates` directly.

- [ ] **Step 2: Edit `__init__.py` — delete moved functions**

Three non-contiguous Edits.

- [ ] **Step 3: Update `__init__.py` re-exports**

```python
from .actiontype import *  # noqa: F401, F403
from .bodypart import *  # noqa: F401, F403
from .context import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403
from .result import *  # noqa: F401, F403
from .spatial import *  # noqa: F401, F403
from .temporal import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .bodypart import __all__ as _bp_all
from .context import __all__ as _cx_all
from .core import __all__ as _core_all
from .result import __all__ as _rs_all
from .spatial import __all__ as _sp_all
from .temporal import __all__ as _tm_all

__all__ = sorted({*_at_all, *_bp_all, *_cx_all, *_core_all, *_rs_all, *_sp_all, *_tm_all})
```

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -5
```

Expected: T-B 31/33.

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 871 passing, 2 failing, 4 deselected.

---

### Task 10: Extract `specialty.py` (final extraction)

**Files:**
- Create: `silly_kicks/vaep/features/specialty.py`
- Modify: `silly_kicks/vaep/features/__init__.py`

Symbols to move: `cross_zone` (`__init__.py:1028-1081`), `assist_type` (`:1082-1170`).

- [ ] **Step 1: Write `specialty.py`**

```python
"""Specialty feature transformers — silly-kicks's own additions beyond the
upstream socceraction feature set.

Two transformers: ``cross_zone`` (categorical zone classification for crosses
into the box), ``assist_type`` (categorical type-of-assist preceding a shot).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, GameStates, simple

__all__ = ["assist_type", "cross_zone"]


# === cross_zone — copy verbatim from features/__init__.py:1028-1081 ===
@simple
def cross_zone(actions: Actions) -> Features:
    # ...


# === assist_type — copy verbatim from features/__init__.py:1082-1170 ===
def assist_type(gamestates: GameStates) -> Features:
    # ...
```

**Implementation note:** `cross_zone` is `@simple`-decorated; `assist_type` consumes `GameStates`.

- [ ] **Step 2: Edit `__init__.py` — delete moved functions**

Two contiguous Edits at end-of-file.

- [ ] **Step 3: Update `__init__.py` re-exports — final form**

After this task, `__init__.py` is JUST the re-export hub:

```python
"""Public-API package for ``silly_kicks.vaep.features``.

Decomposed in 2.3.0 from a 1170-line monolith into 8 concern-focused submodules.
Submodules are importable directly (``silly_kicks.vaep.features.spatial.startlocation``),
but the canonical entry point remains the package itself
(``silly_kicks.vaep.features.startlocation``). All 33 previously-public symbols
remain importable via the package path; new code is encouraged to use the
package import for backwards-compat.

See ``docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md``
for the rationale and submodule layout.
"""

from .actiontype import *  # noqa: F401, F403
from .bodypart import *  # noqa: F401, F403
from .context import *  # noqa: F401, F403
from .core import *  # noqa: F401, F403
from .result import *  # noqa: F401, F403
from .spatial import *  # noqa: F401, F403
from .specialty import *  # noqa: F401, F403
from .temporal import *  # noqa: F401, F403

from .actiontype import __all__ as _at_all
from .bodypart import __all__ as _bp_all
from .context import __all__ as _cx_all
from .core import __all__ as _core_all
from .result import __all__ as _rs_all
from .spatial import __all__ as _sp_all
from .specialty import __all__ as _spec_all
from .temporal import __all__ as _tm_all

__all__ = sorted({*_at_all, *_bp_all, *_cx_all, *_core_all, *_rs_all, *_sp_all, *_spec_all, *_tm_all})
```

The file is now ~30 lines — pure re-export hub.

- [ ] **Step 4: Verify T-B fully green**

```bash
uv run pytest tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -q 2>&1 | tail -5
```

Expected: **T-A 33/33 + T-B 33/33** all passing. 66 total cases passing, zero failing.

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 873 passing, 0 failing, 4 deselected. The decomposition is structurally complete.

---

### Task 11: Atomic update — per-concern imports + remove DRY type aliases

**Files:**
- Modify: `silly_kicks/atomic/vaep/features.py`

- [ ] **Step 1: Read current atomic features.py**

```bash
head -50 silly_kicks/atomic/vaep/features.py
```

Confirm the imports at lines 9-22 (single multi-line import from `silly_kicks.vaep.features`) and the local type alias duplicates at lines 46-49 (`Actions`, `GameStates`, `Features`, `FeatureTransfomer`).

- [ ] **Step 2: Update imports — replace monolith import with 4 grouped per-concern imports**

Use Edit to replace:

```python
from silly_kicks.vaep.features import (
    _actiontype,
    bodypart,
    bodypart_detailed,
    bodypart_detailed_onehot,
    bodypart_onehot,
    gamestates,
    player_possession_time,
    simple,
    speed,
    team,
    time,
    time_delta,
)
```

with:

```python
from silly_kicks.vaep.features.bodypart import (
    bodypart,
    bodypart_detailed,
    bodypart_detailed_onehot,
    bodypart_onehot,
)
from silly_kicks.vaep.features.context import player_possession_time, team
from silly_kicks.vaep.features.core import (
    Actions,
    FeatureTransfomer,
    Features,
    GameStates,
    _actiontype,
    gamestates,
    simple,
)
from silly_kicks.vaep.features.temporal import speed, time, time_delta
```

This both updates to per-concern imports AND adds `Actions`, `FeatureTransfomer`, `Features`, `GameStates` to the import (removing the need for local duplicates).

- [ ] **Step 3: Delete the local type alias duplicates**

Use Edit to delete lines 46-49 (currently):

```python
Actions = pd.DataFrame
GameStates = list[pd.DataFrame]
Features = pd.DataFrame
FeatureTransfomer = Callable[[GameStates], Features]
```

Replace the block with nothing (delete) — the type aliases are now imported from `core`.

- [ ] **Step 4: Verify atomic still works**

```bash
uv run python -c "from silly_kicks.atomic.vaep.features import gamestates, location, polar; print('OK')" 2>&1
```

Expected: `OK`.

- [ ] **Step 5: Run full pytest including atomic tests**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 873 passing, 0 failing, 4 deselected. Atomic changes are import-restructure only; behavior unchanged.

If atomic tests fail, the most likely cause is a missed type alias usage somewhere in atomic/vaep/features.py (use grep to find any remaining bare `Actions`/`Features` references that aren't covered by the new import).

---

### Task 12: Add T-C (atomic-per-concern import test)

**Files:**
- Create: `tests/atomic/test_features_per_concern_import.py`

- [ ] **Step 1: Write the test file**

Use Write to create `tests/atomic/test_features_per_concern_import.py`:

```python
"""Locks the A9-partial-closure pattern: silly_kicks.atomic.vaep.features
imports per-concern from vaep.features submodules, NOT from the package root.

If a future PR consolidates atomic's imports back to the package root (a
monolith-coupling regression), this test fails fast.
"""

from __future__ import annotations

import inspect

import silly_kicks.atomic.vaep.features as atomic_features


def test_atomic_imports_per_concern_not_from_monolith() -> None:
    """Atomic VAEP features imports per-concern submodules, not the package root."""
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

- [ ] **Step 2: Verify T-C passes**

```bash
uv run pytest tests/atomic/test_features_per_concern_import.py -v 2>&1 | tail -10
```

Expected: `1 passed in ...`. If FAIL, the atomic update in Task 11 didn't establish the expected per-concern import structure — re-check the atomic file's imports.

- [ ] **Step 3: Full pytest sweep**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 874 passing (873 + T-C), 0 failing, 4 deselected.

---

### Task 13: CI gate update — `_PUBLIC_MODULE_FILES` 19 → 26

**Files:**
- Modify: `tests/test_public_api_examples.py`

- [ ] **Step 1: Update `_PUBLIC_MODULE_FILES` tuple**

Use Edit to replace the existing 19-entry tuple. The entry `"silly_kicks/vaep/features.py"` is dropped; 8 submodule entries are added.

Match on the existing tuple (use Read first to confirm exact current shape) and replace with:

```python
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
    "silly_kicks/atomic/vaep/base.py",
    "silly_kicks/xthreat.py",
    "silly_kicks/vaep/labels.py",
    "silly_kicks/vaep/formula.py",
    "silly_kicks/atomic/vaep/features.py",
    "silly_kicks/atomic/vaep/labels.py",
    "silly_kicks/atomic/vaep/formula.py",
    "silly_kicks/vaep/features/core.py",
    "silly_kicks/vaep/features/actiontype.py",
    "silly_kicks/vaep/features/result.py",
    "silly_kicks/vaep/features/bodypart.py",
    "silly_kicks/vaep/features/spatial.py",
    "silly_kicks/vaep/features/temporal.py",
    "silly_kicks/vaep/features/context.py",
    "silly_kicks/vaep/features/specialty.py",
)
```

The 8 new entries are at the end (preserves original-order readability for the diff).

- [ ] **Step 2: Run gate — expect 26 passing**

```bash
uv run pytest tests/test_public_api_examples.py -v 2>&1 | tail -10
```

Expected: `26 passed in ...`. Each new submodule passes immediately because every public function already has its Examples section (preserved through the move; PR-S13/PR-S14 work intact).

If any FAIL, the failure message lists the offending symbol — most likely `__all__` is missing from the submodule (gate's AST walk finds the def but the module doesn't declare it as public). Add `__all__` to the submodule.

- [ ] **Step 3: Full pytest sweep**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 881 passing (874 + 7 net gate delta), 0 failing, 4 deselected.

---

### Task 14: Verification gates (full local equivalent of CI)

**Files:** None — read-only verification.

- [ ] **Step 1: Pin tooling versions**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: install/upgrade output. Zero error lines.

- [ ] **Step 2: ruff check**

```bash
uv run ruff check silly_kicks/ tests/
```

Expected: `All checks passed!`. Common likely issues:
- Unused import warnings on the submodule re-export `from .X import *` lines (suppressed by `# noqa: F401, F403`).
- F821 forward-references to type aliases — should not occur since `Actions`/`Features`/etc. are imported normally.

- [ ] **Step 3: ruff format check**

```bash
uv run ruff format --check silly_kicks/ tests/
```

Expected: `... files already formatted`.

If files would be reformatted, run `uv run ruff format silly_kicks/ tests/` to apply, then re-run `--check`.

- [ ] **Step 4: pyright**

```bash
uv run pyright silly_kicks/
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 5: Full pytest suite**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 881 passing, 0 failing, 4 deselected.

- [ ] **Step 6: Smoke check — package import + per-concern submodule imports**

```bash
uv run python -c "
from silly_kicks.vaep.features import (
    gamestates, startlocation, time, bodypart, team, cross_zone,
    actiontype, result, Actions, GameStates, FeatureTransfomer,
)
from silly_kicks.vaep.features.core import gamestates as gs2
from silly_kicks.vaep.features.spatial import startlocation as sl2
from silly_kicks.vaep.features.specialty import cross_zone as cz2
assert gamestates is gs2
assert startlocation is sl2
assert cross_zone is cz2
print('smoke OK')
"
```

Expected: `smoke OK`. If `AssertionError` or `ImportError`, the package re-exports or submodule paths are wrong.

- [ ] **Step 7: Spot-check rendered docstring on a representative new submodule**

```bash
uv run python -c "from silly_kicks.vaep.features import startlocation; help(startlocation)" 2>&1 | head -20
```

Expected: docstring renders correctly with the existing Examples section preserved through the move.

---

### Task 15: Admin updates — TODO.md, CHANGELOG, version bump

**Files:**
- Modify: `TODO.md`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update TODO.md — A9 narrowed, A19/O-M1/O-M6 closed, vaep-features decomposition closed**

Read current `TODO.md` to confirm current text, then Edit:

Replace:
```markdown
| A9 | Medium | Reduce `atomic/vaep/features.py` coupling to `vaep/features` (12 imports) | Legitimate delegation today, but tight coupling will fight if atomic features need to diverge independently |
| — | Medium | Decompose `vaep/features.py` (809 lines) | Natural split: spatial features, temporal features, categorical features. Do when next adding features to this file |
```

with:

```markdown
| A9 | Low | `atomic/vaep/features.py` per-concern coupling to `vaep/features` (12 symbols across 4 submodules — was monolith) | Partially addressed in 2.3.0 via vaep/features decomposition. Full decoupling deferred until atomic features genuinely need to diverge independently — extracting truly-shared framework into a cross-package module is the trigger condition. |
```

(`A9` severity drops from Medium to Low; the unnumbered "Decompose vaep/features.py" row is deleted because that work is now done.)

Replace the Tech Debt table — delete the A19, O-M1, O-M6 rows. Final table should contain only:

```markdown
| # | Sev | Item | Context |
|---|-----|------|---------|
| D-9 | Low | 5 xthreat module-level functions naming (`scoring_prob`, `get_move_actions`, etc.) | Pre-2.0.0 work already underscore-prefixed all module-level helpers; this row tracks legacy concern. |
```

Wait — D-9 was closed in 2.1.1 per the spec / memory. Verify by reading current TODO.md. Actually D-9 doesn't appear in the current TODO.md (already closed). Current Tech Debt rows are A19, O-M1, O-M6, C-1 (now closed). After the deletes, the Tech Debt table is empty.

If the Tech Debt table is empty after deletions, replace it with:

```markdown
## Tech Debt

(none currently queued — A19 / O-M1 / O-M6 reviewed and closed in 2.3.0 as stale-or-by-design; D-9 closed in 2.1.1; C-1 closed in 2.2.0)
```

- [ ] **Step 2: Update CHANGELOG.md — add `[2.3.0]` entry**

Use Edit to add the new entry above `[2.2.0]`. Match on `## [2.2.0] — 2026-04-30` and prepend:

```markdown
## [2.3.0] — 2026-04-30

### Changed

- **`silly_kicks.vaep.features` decomposed from a 1170-line monolith into a
  package** of 8 concern-focused submodules (`core`, `actiontype`, `result`,
  `bodypart`, `spatial`, `temporal`, `context`, `specialty`). Hybrid visibility:
  every previously-public symbol remains importable via the package path
  (`from silly_kicks.vaep.features import startlocation` keeps working
  unchanged); submodule paths are also importable for advanced/atomic-internal
  use. Closes the long-standing TODO architecture entry. **Pure structural
  refactor — zero behavior change; every existing test passes through every
  step.**
- **`silly_kicks.atomic.vaep.features` updated to import per-concern.** 12
  symbols imported across 4 grouped statements against
  `vaep.features.{core,bodypart,context,temporal}` (was: single 12-symbol
  monolith import). TODO A9 partially addressed — full decoupling deferred
  until atomic features need to diverge independently. Local type alias
  duplicates (`Actions = pd.DataFrame` etc.) replaced by a single import
  from `vaep.features.core` (DRY cleanup).

### Added

- **8 new public-API submodule paths** (`silly_kicks.vaep.features.core`,
  `.actiontype`, `.result`, `.bodypart`, `.spatial`, `.temporal`, `.context`,
  `.specialty`). Documented as implementation detail of where each symbol
  lives — the canonical entry point remains the package itself.
- **3 new test files locking the structure:** `T-A` backcompat (33 parametrized
  cases asserting every public symbol stays importable from the package path),
  `T-B` submodule layout (33 parametrized cases asserting each symbol's
  `__module__` matches the design contract), `T-C` atomic-per-concern (1 test
  asserting atomic imports from per-concern submodules, not the package root).
- **CI gate (`tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`)
  widened from 19 → 26 entries** to cover all 8 new submodule paths. Net +7
  parametrize cases.

### Closed

- **TODO A19** (default hyperparameters scattered across 3 learner functions):
  reviewed and closed without code change. Already centralized as
  `_XGBOOST_DEFAULTS` / `_CATBOOST_DEFAULTS` / `_LIGHTGBM_DEFAULTS` module-level
  constants since 1.9.0; the audit description ("scattered across 3 functions")
  predates that extraction.
- **TODO O-M1** (full `events.copy()` at top of StatsBomb `convert_to_actions`):
  reviewed and closed without code change. The defensive copy is correct by
  design — `_flatten_extra` mutates the DataFrame by adding ~22 underscore
  columns; without the copy, caller's events would be mutated in place.
- **TODO O-M6** (temporary n×3 DataFrame for StatsBomb fidelity version check):
  reviewed and closed without code change. ~50 KB peak per match; could be
  numpy-fied for marginal gain (~25 KB savings); no measurable impact.

No API breakage. 881 tests passing (807 baseline + 33 T-A + 33 T-B + 1 T-C +
7 net gate delta), 4 deselected.

```

- [ ] **Step 3: Bump pyproject.toml 2.2.0 → 2.3.0**

Use Edit to replace `version = "2.2.0"` → `version = "2.3.0"`.

- [ ] **Step 4: Re-run pytest after admin updates**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: 881 passing, 0 failing, 4 deselected. Same as Task 14 Step 5.

---

### Task 16: /final-review

**Files:** None — read-only review pass.

- [ ] **Step 1: Invoke /final-review skill**

Use the Skill tool: `Skill(skill="mad-scientist-skills:final-review")`.

The skill runs Phase 1-5: codebase discovery, code quality review, architectural decision review, documentation review, architecture diagram check.

Expected outcomes:
- Code quality: clean (no critical/high issues from the structural refactor; existing test suite is the safety net).
- Architectural decisions: the cross-package import pattern (`atomic/vaep/features.py` → `vaep.features.{core,bodypart,...}`) was already established in 2.2.0 (atomic CoverageMetrics reuse). Not a new ADR-worthy decision.
- Documentation: README's API section uses `silly_kicks.vaep.features` as one example — verify it still works (it does — package re-export).
- Architecture diagram: `docs/c4/architecture.html` exists, last regenerated 2026-04-29. PR-S15 is internal restructure — no new container, component, or actor. No regen needed.

- [ ] **Step 2: Re-run verification gates after any /final-review fixes**

```bash
uv run ruff check silly_kicks/ tests/ && uv run pyright silly_kicks/ && uv run pytest tests/ -m "not e2e" --tb=short -q 2>&1 | tail -5
```

Expected: all green. 881 passing.

---

### Task 17: Single-commit gate (USER APPROVAL REQUIRED)

**Files:** No file changes — commit decision point.

- [ ] **Step 1: Show git status + diff stat to user**

```bash
git status --short
git diff --stat
```

Expected output:
```
M  CHANGELOG.md
M  TODO.md
M  pyproject.toml
M  silly_kicks/atomic/vaep/features.py
R  silly_kicks/vaep/features.py -> silly_kicks/vaep/features/__init__.py
M  tests/test_public_api_examples.py
?? docs/superpowers/plans/2026-04-30-vaep-features-decomposition.md
?? docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md
?? silly_kicks/vaep/features/actiontype.py
?? silly_kicks/vaep/features/bodypart.py
?? silly_kicks/vaep/features/context.py
?? silly_kicks/vaep/features/core.py
?? silly_kicks/vaep/features/result.py
?? silly_kicks/vaep/features/spatial.py
?? silly_kicks/vaep/features/specialty.py
?? silly_kicks/vaep/features/temporal.py
?? tests/atomic/test_features_per_concern_import.py
?? tests/vaep/test_features_backcompat.py
?? tests/vaep/test_features_submodule_layout.py
```

Plus pre-existing untracked items (`README.md.backup`, `uv.lock`).

- [ ] **Step 2: Ask user for explicit approval to commit**

Per memory `feedback_commit_policy.md`: "no commits or PRs without explicit approval".

User must say "yes commit" / "approved" / equivalent before proceeding.

- [ ] **Step 3: Stage explicit files**

Avoid `git add -A`. Stage explicitly:

```bash
git add CHANGELOG.md TODO.md pyproject.toml \
  silly_kicks/atomic/vaep/features.py \
  silly_kicks/vaep/features/__init__.py \
  silly_kicks/vaep/features/core.py \
  silly_kicks/vaep/features/actiontype.py \
  silly_kicks/vaep/features/result.py \
  silly_kicks/vaep/features/bodypart.py \
  silly_kicks/vaep/features/spatial.py \
  silly_kicks/vaep/features/temporal.py \
  silly_kicks/vaep/features/context.py \
  silly_kicks/vaep/features/specialty.py \
  tests/test_public_api_examples.py \
  tests/atomic/test_features_per_concern_import.py \
  tests/vaep/test_features_backcompat.py \
  tests/vaep/test_features_submodule_layout.py \
  docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md \
  docs/superpowers/plans/2026-04-30-vaep-features-decomposition.md
```

(The `R` rename `silly_kicks/vaep/features.py` → `silly_kicks/vaep/features/__init__.py` is auto-detected when `__init__.py` is staged — no separate action needed.)

- [ ] **Step 4: Create the single commit**

Use a HEREDOC for the commit message:

```bash
git commit -m "$(cat <<'EOF'
feat(vaep): decompose vaep/features.py monolith into 8-submodule package — silly-kicks 2.3.0 (PR-S15)

Pure structural refactor — zero behavior change; 807 baseline tests pass
through every step.

silly_kicks.vaep.features (1170-line monolith) → package with 8 concern-focused
submodules:
- core: type aliases (Actions/GameStates/Features/FeatureTransfomer),
  gamestates, simple decorator, play_left_to_right, feature_column_names,
  _actiontype helper
- actiontype: actiontype, actiontype_onehot
- result: result, result_onehot, actiontype_result_onehot,
  result_onehot_prev_only, actiontype_result_onehot_prev_only
- bodypart: bodypart, bodypart_detailed, bodypart_onehot,
  bodypart_detailed_onehot
- spatial: startlocation, endlocation, startpolar, endpolar, movement,
  space_delta
- temporal: time, time_delta, speed
- context: team, player_possession_time, goalscore
- specialty: cross_zone, assist_type

Hybrid visibility: every previously-public symbol remains importable via the
package path (canonical entry); submodule paths are also importable for
advanced / atomic-internal use.

silly_kicks.atomic.vaep.features updated to import per-concern (12 symbols
across 4 grouped statements against vaep.features.{core,bodypart,context,
temporal} — was: single multi-line monolith import). Local type alias
duplicates removed. TODO A9 partially addressed (severity Medium → Low).

3 new test files lock the structure:
- T-A backcompat (33 parametrized cases): every public symbol stays
  importable from the package path
- T-B submodule layout (33 parametrized cases): each symbol's __module__
  matches the design contract
- T-C atomic-per-concern (1 test): atomic imports from per-concern
  submodules, not the package root

CI gate widened from 19 → 26 entries to cover all 8 new submodule paths.

Bundled close (National Park principle): TODO A19, O-M1, O-M6 reviewed and
closed without code change (stale-or-by-design — see CHANGELOG).

Version bump 2.2.0 → 2.3.0 (minor — additive public API via 8 new submodule
paths). 881 tests passing (807 baseline + 33 T-A + 33 T-B + 1 T-C + 7 net
gate delta), 4 deselected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If a pre-commit hook fails: fix the issue, re-stage, create a NEW commit (do NOT use `--amend`).

- [ ] **Step 5: Verify commit**

```bash
git log --oneline -1
git show --stat HEAD | head -25
```

Expected: single commit on the branch with the full file list.

---

### Task 18: Push, PR, CI watch, squash-merge --admin, tag, PyPI verify

**Files:** None — git/gh operations.

Gated on Task 17 commit being clean.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/vaep-features-decomposition 2>&1 | tail -5
```

Expected: branch published.

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "feat(vaep): decompose vaep/features.py into 8-submodule package — silly-kicks 2.3.0 (PR-S15)" --body "$(cat <<'EOF'
## Summary
- Decomposes \`silly_kicks/vaep/features.py\` (1170 lines) into a \`silly_kicks/vaep/features/\` package with 8 concern-focused submodules. Pure structural refactor — zero behavior change.
- Updates \`silly_kicks/atomic/vaep/features.py\` to import per-concern (12 symbols across 4 grouped statements). TODO A9 partially addressed.
- Bundles National-Park close of TODO A19 / O-M1 / O-M6 (reviewed and closed as stale-or-by-design, no code change).
- Version bump 2.2.0 → 2.3.0 (minor — 8 new public submodule paths).

## Test plan
- [x] T-A (33 parametrized cases) — every previously-public symbol stays importable from the package path.
- [x] T-B (33 parametrized cases) — each symbol's \`__module__\` matches the design contract.
- [x] T-C (1 test) — atomic imports per-concern, not from package root.
- [x] CI gate widened 19 → 26 entries (\`_PUBLIC_MODULE_FILES\`).
- [x] Full pytest suite: 881 passing, 4 deselected.
- [x] ruff check + format clean. Pyright 0 errors.
- [x] Smoke test: \`from silly_kicks.vaep.features import gamestates, startlocation, time, bodypart, team, cross_zone\` round-trips identical to direct submodule imports.
- [x] /final-review clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 3: Watch CI checks**

```bash
gh pr checks --watch
```

Expected: all checks green (5 checks: lint + ubuntu 3.10/3.11/3.12 + windows 3.12). Total ~3-5 min.

- [ ] **Step 4: Squash-merge with --admin override**

Branch protection on main blocks unreviewed merges (sole-maintainer pattern from prior cycles requires admin override).

```bash
gh pr merge --squash --delete-branch --admin
```

Expected: PR merged, branch auto-deleted on remote.

- [ ] **Step 5: Sync local main**

```bash
git switch main && git pull origin main 2>&1 | tail -5 && git log --oneline -2
```

Expected: latest commit on main is the squashed PR-S15.

- [ ] **Step 6: Tag v2.3.0 and push**

```bash
git tag -a v2.3.0 -m "silly-kicks 2.3.0 — vaep/features decomposition + atomic per-concern imports + closed A19/O-M1/O-M6 (PR-S15)"
git push origin v2.3.0
```

Expected: tag pushed; PyPI auto-publish workflow fires from the tag push.

- [ ] **Step 7: Verify PyPI publish workflow**

```bash
gh run list --limit=2
```

Expected: most recent workflow is `Publish to PyPI` with status `queued` or `in_progress`. Watch:

```bash
gh run watch <run-id> --exit-status
```

Expected: workflow completes successfully (~40s for build + publish jobs).

- [ ] **Step 8: Final state confirmation**

```bash
git log --oneline -3
git status --short
git tag --list | tail -5
```

Expected:
- main has the squashed PR-S15 commit at HEAD.
- Working tree clean (only pre-existing untracked: `README.md.backup`, `uv.lock`).
- `v2.3.0` in tag list.

---

## Self-Review

Re-checking the plan against the spec section-by-section:

**Spec §1 (Problem):** ✅ vaep/features.py decomposition (Tasks 3-10), atomic per-concern (Task 11), bundled A19/O-M1/O-M6 closes (Task 15).

**Spec §2 (Goals):**
1. ✅ 8 submodules created (Tasks 3-10).
2. ✅ Atomic per-concern imports (Task 11).
3. ✅ Bundled doc cleanup (Task 15).
4. ✅ Backwards compat preserved via `__init__.py` re-exports + T-A.
5. ✅ TDD-first tests (T-A in Task 1, T-B in Task 2, T-C in Task 12); each submodule extraction is a red→green cycle on T-B (Tasks 3-10 step 4).
6. ✅ Minor release 2.3.0 (Task 15 Step 3).

**Spec §3 (Non-goals):** ✅ no behavior change (807 baseline preserved through every task), no new features, no public API removed (T-A enforces), no A9 closure (TODO updated to "partially addressed"), no CLAUDE.md/ADR, no learners.py touch, submodules NOT underscore-prefixed.

**Spec §4.1 (Package layout):** ✅ Tasks 3-10 create the 8 submodules + `__init__.py` exactly as specified.

**Spec §4.2 (Visibility — Hybrid):** ✅ `__init__.py` docstring (Task 10 Step 3) explicitly documents the hybrid visibility.

**Spec §4.3 (Atomic update):** ✅ Task 11 implements the exact code transformation specified.

**Spec §4.4 (CI gate):** ✅ Task 13 implements the 19 → 26 widening.

**Spec §4.5 (TDD strategy):** ✅ T-A, T-B, T-C all implemented with the exact code from the spec.

**Spec §4.6 (Bundled close):** ✅ Task 15 closes A19, O-M1, O-M6 (TODO row deletion + CHANGELOG note).

**Spec §5 (Implementation order):** ✅ Tasks T0-T18 align with spec T1-T21 (off by one for branch setup).

**Spec §6 (Verification gates):** ✅ Task 14 implements all gates.

**Spec §7 (Commit cycle):** ✅ Task 17 = single commit with HEREDOC + Co-Authored-By trailer.

**Spec §8 (Risks + mitigations):** ✅ Each risk has a mitigation embedded in the relevant task. Circular imports caught by smoke test (Task 14 Step 6) and T-B passing (Task 10 Step 4). Missing `__all__` caught by gate (Task 13 Step 2). git mv preserves history (Task 3 Step 2).

**Spec §9 (Acceptance criteria):**
1. ✅ Tasks 3-10 produce the package + 8 submodules.
2. ✅ Tasks 3-10 add `__all__` to each submodule.
3. ✅ Tasks 3-10 maintain `__init__.py` re-exports.
4. ✅ Task 11 updates atomic imports.
5. ✅ T-A in Task 1, T-B in Task 2, T-C in Task 12 (33 + 33 + 1 = 67 new tests).
6. ✅ Task 13 widens gate to 26 entries.
7. ✅ Task 15 Step 1 updates TODO.
8. ✅ Task 15 Step 2 adds CHANGELOG entry.
9. ✅ Task 15 Step 3 bumps version.
10. ✅ Task 14 verification gates; expected 881 passing.
11. ✅ Task 16 /final-review.
12. ✅ Task 18 Step 6+7 tags v2.3.0 and verifies PyPI.
13. ✅ T-A enforces public-API-stability.

**Placeholder scan:** No "TBD" / "implement later" / "similar to Task N". Each step has explicit code or commands; the per-submodule "copy verbatim from features/__init__.py:line-line" instructions are concrete (every line-range references a specific source file and is verifiable via Read).

**Type / signature consistency:** All Examples and signatures use the established names. `Actions`, `GameStates`, `Features`, `FeatureTransfomer` are defined in `core.py` (Task 3) and imported by all subsequent submodules. `_actiontype` lives in `core.py` and is imported by `actiontype.py` (Task 4) and atomic (Task 11). `simple` decorator in `core.py`.

No issues found. Plan is ready.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-30-vaep-features-decomposition.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session via `superpowers:executing-plans`, batch execution with checkpoints at natural gates (post-T2 / post-T10 / post-T14 / pre-commit / pre-merge).

Per memory and prior cycle preference, **Inline Execution** is the standing default. PR-S15 is largely mechanical Edit-tool work (extracting per-submodule), so subagent parallelism doesn't help much.

Which approach?
