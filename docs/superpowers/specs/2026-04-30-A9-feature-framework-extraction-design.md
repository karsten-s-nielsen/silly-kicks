# A9 close — VAEP feature framework extraction (PR-S16)

**Status:** Approved (design)
**Target release:** silly-kicks 2.4.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.3.0 (`docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md`)
**Closes:** TODO A9 (the last open Architecture row)

---

## 1. Problem

`silly_kicks/atomic/vaep/features.py` reaches into `silly_kicks.vaep.features.core` to import 7 framework primitives — `Actions`, `Features`, `FeatureTransfomer`, `GameStates`, `gamestates`, `simple`, and the *private* `_actiontype` helper. The first 6 are conceptually framework infrastructure (type aliases, gamestate construction, the `simple` decorator), genuinely shared across both standard-VAEP and atomic-VAEP feature stacks. The 7th — `_actiontype` — is a SPADL-config-parameterized helper used cross-package via a leading-underscore symbol, i.e. an explicit reach into another package's documented-private surface.

PR-S15 (silly-kicks 2.3.0) decomposed the standard `vaep/features.py` monolith into an 8-submodule package and updated atomic to import per-concern. That PR explicitly deferred A9 with the trigger condition: *"extracting truly-shared framework into a cross-package module."* The deferral was correct at the time — the immediate decomposition win was the per-concern grouping, not the cross-package framework extraction. With 2.3.0 stable, the framework primitives' boundary has crystallized and the trigger condition is now actionable as a small, well-bounded PR.

This PR introduces a new public module `silly_kicks/vaep/feature_framework.py`, moves the 7 cross-package primitives into it, promotes `_actiontype` to public `actiontype_categorical(actions, spadl_cfg)`, refits atomic-VAEP to depend on the framework module + the per-concern feature submodules (and not on `vaep/features/core.py` for framework primitives), and refactors the lock-in tests (T-A, T-B, T-C) to reflect the new boundary plus a new T-D framework layout test. A9 closes.

## 2. Goals

1. **Introduce `silly_kicks/vaep/feature_framework.py` (public)** as the named cross-package boundary that both standard and atomic VAEP feature stacks build upon. Holds 4 type aliases (`Actions`, `Features`, `FeatureTransfomer`, `GameStates`), 2 framework helpers (`gamestates`, `simple`), and the promoted helper `actiontype_categorical(actions, spadl_cfg)` (was `_actiontype`).
2. **Promote `_actiontype` → `actiontype_categorical(actions, spadl_cfg)`.** Drop the leading underscore — cross-package use of a documented-private symbol is a smell that gold-standard demands we fix when we touch this code. Positional `spadl_cfg` parameter (was keyword-only `_spadl_cfg=None` with module-level fallback). Examples-section docstring per the public-API discipline. Wrapped in `@simple` by both standard and atomic to produce their respective `actiontype` feature transformers.
3. **Slim `vaep/features/core.py`** to its genuinely standard-SPADL-specific contents: `play_left_to_right` (uses `start_x`/`end_x`) and `feature_column_names` (uses `result_id`/`start_x`/`end_x`/`result_name`). Both stay in `core` because they hardcode standard SPADL columns and are re-implemented separately in atomic with different schemas. `core.py` adds `from silly_kicks.vaep.feature_framework import *` so existing imports of `from silly_kicks.vaep.features.core import gamestates` continue to resolve (Hyrum's Law preservation).
4. **Refit `silly_kicks/atomic/vaep/features.py`** to import framework primitives from `silly_kicks.vaep.feature_framework` directly. Drop the `vaep.features.core` import. Replace `_actiontype` call with `actiontype_categorical`. Per-concern imports from `bodypart`, `context`, `temporal` stay (verbatim feature reuse — intentional code-share, not framework leak).
5. **Update lock-in tests T-B, T-C; add T-D.** T-B drops the 6 rows now living outside the features package (4 type aliases + `gamestates` + `simple`). T-C rewritten: forbids `vaep.features.core` import for framework primitives, requires `vaep.feature_framework`, retains the 3 per-concern requirements. T-D NEW: 7-case parametrized test asserting framework symbols' canonical home is `silly_kicks.vaep.feature_framework`. T-A gains one row (`actiontype_categorical`) and remains a pure backwards-compat lock.
6. **ADR-002 — *Shared VAEP feature framework boundary*** captures: PR-S15 deferral context, the cross-package coupling problem, chosen module + naming, `_actiontype` → public rename rationale, revisit conditions. Future maintainers benefit from understanding why `feature_framework.py` exists as a sibling of `features/`.
7. **Minor release 2.4.0.** Additive public API (one new module path, one new public function). Backward-compat preserved — every previously-public symbol stays accessible from the same paths it was at in 2.3.0. Same precedent as 2.2.0's `coverage_metrics` minor bump.

## 3. Non-goals

1. **No behavior change.** Pure structural refactor + one symbol rename. Every existing test passes before AND after.
2. **No atomic feature divergence.** Atomic still reuses `bodypart`, `context`, `temporal` verbatim — that coupling is by design (genuine code-share, not framework leak). Atomic divergence is a future trigger condition for that relaxation, not part of this PR.
3. **No `play_left_to_right` / `feature_column_names` move.** Both stay in `vaep/features/core.py` because they hardcode standard SPADL columns. Atomic has its own equivalents in `silly_kicks/atomic/vaep/features.py`. Treating them as framework would force a config-parameterized rewrite which is out of scope.
4. **No removal of `_actiontype` from public-import compatibility surface.** It was never in `__all__` and never documented as public; the rename is a true rename, not a deprecate-and-remove. Promoting to `actiontype_categorical` and the disappearance of `_actiontype` are simultaneous. The ADR notes this as an acceptable Hyrum's Law exposure (leading-underscore = never contract).
5. **No CLAUDE.md amendment.** ADR-002 documents the decision; CLAUDE.md "Key conventions" section already covers ML naming and no-pandera; framework-boundary discipline doesn't rise to the level of a CLAUDE.md rule (it's the natural consequence of "name shared things").
6. **No subagent dispatching.** Single-session, single-commit per the user commit policy.

## 4. Architecture

### 4.1 Module layout (after this PR)

```
silly_kicks/vaep/
├── feature_framework.py       # NEW — 7 framework primitives shared across
│                              # standard + atomic VAEP feature stacks
├── features/                  # (existing 2.3.0 package — surface unchanged)
│   ├── __init__.py            # adds `from ..feature_framework import *`
│   │                          # __all__ adds `actiontype_categorical`
│   ├── core.py                # slimmed: play_left_to_right + feature_column_names
│   │                          # only; re-exports framework primitives via
│   │                          # `from ..feature_framework import *` (Hyrum's Law)
│   ├── actiontype.py          # uses actiontype_categorical (was _actiontype)
│   ├── bodypart.py            # unchanged
│   ├── context.py             # unchanged
│   ├── result.py              # unchanged
│   ├── spatial.py             # unchanged
│   ├── specialty.py           # unchanged
│   └── temporal.py            # unchanged
└── (other vaep modules unchanged)

silly_kicks/atomic/vaep/
└── features.py                # imports framework from `vaep.feature_framework`,
                               # per-concern features from `vaep.features.{bodypart,
                               # context, temporal}` (no longer reaches into core)
```

### 4.2 Framework module surface

`silly_kicks/vaep/feature_framework.py`:

```python
"""Shared VAEP feature framework primitives.

Both ``silly_kicks.vaep.features`` (standard VAEP) and
``silly_kicks.atomic.vaep.features`` (atomic VAEP) build on this module's
type aliases and helpers. Extracted in 2.4.0 to give the cross-package
boundary a named home and to close TODO A9. See ADR-002.

Public surface:

- Type aliases: ``Actions``, ``Features``, ``FeatureTransfomer``, ``GameStates``
- Gamestate construction: ``gamestates``
- Decorator: ``simple``
- SPADL-config-parameterized helper: ``actiontype_categorical``
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, no_type_check

import numpy as np
import pandas as pd

__all__ = [
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "actiontype_categorical",
    "gamestates",
    "simple",
]

Actions = pd.DataFrame
GameStates = list[pd.DataFrame]
Features = pd.DataFrame
FeatureTransfomer = Callable[[GameStates], Features]


def gamestates(actions: Actions, nb_prev_actions: int = 3) -> GameStates:
    # body unchanged from 2.3.0 vaep/features/core.gamestates
    ...


@no_type_check
def simple(actionfn: Callable) -> FeatureTransfomer:
    # body unchanged from 2.3.0 vaep/features/core.simple
    ...


def actiontype_categorical(actions: Actions, spadl_cfg: Any) -> Features:
    """SPADL-config-parameterized categorical actiontype helper.

    Both standard-VAEP and atomic-VAEP wrap this with ``@simple`` to
    produce their respective ``actiontype`` feature transformers.

    Parameters
    ----------
    actions : Actions
        The actions of a game.
    spadl_cfg : module
        A SPADL config module exposing ``actiontypes`` (list of type names)
        and ``actiontypes_df()`` (DataFrame with ``type_id`` / ``type_name``).
        Pass ``silly_kicks.spadl.config`` for standard SPADL or
        ``silly_kicks.atomic.spadl.config`` for atomic SPADL.

    Returns
    -------
    Features
        A single-column DataFrame with column ``actiontype`` of dtype
        ``pd.Categorical`` whose categories are the SPADL config's
        action-type names in original order.

    Examples
    --------
    Build a per-action categorical actiontype feature for standard SPADL::

        from silly_kicks.vaep.feature_framework import actiontype_categorical
        import silly_kicks.spadl.config as spadlcfg

        feats = actiontype_categorical(actions, spadlcfg)
        # feats has one column 'actiontype' of dtype Categorical.
    """
    X = pd.DataFrame(index=actions.index)
    categories = list(dict.fromkeys(spadl_cfg.actiontypes))  # dedupe, preserve order
    X["actiontype"] = pd.Categorical(
        actions["type_id"].replace(spadl_cfg.actiontypes_df().type_name.to_dict()),
        categories=categories,
        ordered=False,
    )
    return X
```

Notes:

- Framework module file is **not** underscore-prefixed; it's a public boundary, named accordingly.
- The 7 symbols are the irreducible cross-package framework. Anything more (e.g. `play_left_to_right`, `feature_column_names`) is schema-specific and stays in the per-package feature module.
- `gamestates` and `simple` bodies are identical to the 2.3.0 `vaep/features/core` versions — pure relocation, no semantic change.
- `actiontype_categorical` is `_actiontype` with: leading underscore dropped, parameter renamed (`_spadl_cfg=None` keyword → `spadl_cfg` positional), implicit-None fallback dropped (caller now always passes the config explicitly — which is the actually-correct contract since the function is meaningless without one), Examples docstring added.

### 4.3 `vaep/features/core.py` after slim-down

```python
"""Standard-SPADL-specific framework helpers — ``play_left_to_right`` and
``feature_column_names``. Both hardcode standard SPADL columns
(``start_x`` / ``end_x`` / ``result_id`` / ``result_name``) so they don't
generalize to atomic SPADL; atomic has its own equivalents in
``silly_kicks.atomic.vaep.features``.

Re-exports the framework primitives from ``silly_kicks.vaep.feature_framework``
so existing ``from silly_kicks.vaep.features.core import gamestates`` paths
continue to resolve (Hyrum's Law preservation).
"""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.vaep.feature_framework import (
    Actions,
    FeatureTransfomer,
    Features,
    GameStates,
    actiontype_categorical,
    gamestates,
    simple,
)

__all__ = [
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "actiontype_categorical",
    "feature_column_names",
    "gamestates",
    "play_left_to_right",
    "simple",
]


def feature_column_names(fs: list[FeatureTransfomer], nb_prev_actions: int = 3) -> list[str]:
    # body unchanged from 2.3.0
    ...


def play_left_to_right(gamestates: GameStates, home_team_id: int) -> GameStates:
    # body unchanged from 2.3.0
    ...
```

The `_actiontype` helper is gone (was private, no consumer outside silly-kicks).

### 4.4 `vaep/features/__init__.py` change

One added re-export block (absolute first-party, separated from the relative re-exports per project isort convention) + one `__all__` row:

```diff
 # ruff: noqa: F405
 """Public-API package for ``silly_kicks.vaep.features``.
 …
 """

+from silly_kicks.vaep.feature_framework import *  # noqa: F403
+
 from .actiontype import *  # noqa: F403
 from .bodypart import *  # noqa: F403
 from .context import *  # noqa: F403
 from .core import *  # noqa: F403
 …

 __all__ = [
     "Actions",
     "FeatureTransfomer",
     "Features",
     "GameStates",
+    "actiontype_categorical",
     "actiontype",
     …
 ]
```

`Actions` etc. are already in `__all__` and continue to resolve identically (the `core` re-export still provides them via `from silly_kicks.vaep.feature_framework import *` inside `core.py`; the new package-level re-export is the explicit canonical-source binding). No external import path breaks.

### 4.5 `vaep/features/actiontype.py` change

The existing `import silly_kicks.spadl.config as spadlcfg` (line 10) stays — already needed by `actiontype_onehot`. Two changes:

```diff
-from .core import Actions, Features, _actiontype, simple
+from silly_kicks.vaep.feature_framework import (
+    Actions,
+    Features,
+    actiontype_categorical,
+    simple,
+)


 @simple
 def actiontype(actions: Actions) -> Features:
     """…"""
-    return _actiontype(actions)
+    return actiontype_categorical(actions, spadlcfg)
```

The current body `_actiontype(actions)` relies on the implicit-None fallback (`_spadl_cfg=None` → resolves to `spadlcfg` inside `_actiontype`). The new body passes `spadlcfg` explicitly — same resolved behavior, no implicit fallback. Module docstring header should also be updated to drop the reference to the now-deleted `_actiontype` helper.

### 4.6 Atomic update — `silly_kicks/atomic/vaep/features.py`

The `vaep.features.core` import is removed and replaced with a `vaep.feature_framework` import. By isort/ruff alphabetical ordering, the new import slots between `silly_kicks.atomic.spadl.config` and `silly_kicks.vaep.features.bodypart` (`vaep.feature_framework` < `vaep.features.bodypart` because `_` < `s` in ASCII).

```diff
 import silly_kicks.atomic.spadl.config as atomicspadl
+from silly_kicks.vaep.feature_framework import (
+    Actions,
+    FeatureTransfomer,
+    Features,
+    GameStates,
+    actiontype_categorical,
+    gamestates,
+    simple,
+)
 from silly_kicks.vaep.features.bodypart import (
     bodypart,
     bodypart_detailed,
     bodypart_detailed_onehot,
     bodypart_onehot,
 )
 from silly_kicks.vaep.features.context import player_possession_time, team
-from silly_kicks.vaep.features.core import (
-    Actions,
-    Features,
-    FeatureTransfomer,
-    GameStates,
-    _actiontype,
-    gamestates,
-    simple,
-)
 from silly_kicks.vaep.features.temporal import speed, time, time_delta
```

Body change for `actiontype`:

```diff
 @simple
 def actiontype(actions: Actions) -> Features:
     """…"""
-    return _actiontype(actions, _spadl_cfg=atomicspadl)
+    return actiontype_categorical(actions, atomicspadl)
```

The 4-import-from-vaep.features structure becomes 3-import (`bodypart`, `context`, `temporal`) + 1-import-from-`vaep.feature_framework`. Atomic no longer reaches into `vaep.features.core` for framework primitives; the only remaining `vaep.features` dependency is on per-concern feature submodules where the implementation is genuinely identical (verbatim reuse).

## 5. Test plan

### 5.1 T-A — `tests/vaep/test_features_backcompat.py` (existing, +1 row)

`_PUBLIC_SYMBOLS` adds `"actiontype_categorical"`. 33 → 34 parametrized cases. Each asserts the symbol is importable from `silly_kicks.vaep.features` (the package). Locks the `actiontype_categorical` symbol into the package-path Hyrum's Law surface from day one.

### 5.2 T-B — `tests/vaep/test_features_submodule_layout.py` (existing, refit)

`_LAYOUT` drops the 6 rows now living outside the `vaep/features/` package:

```diff
 _LAYOUT: tuple[tuple[str, str], ...] = (
     # core (slimmed: only standard-SPADL-specific helpers stay)
-    ("gamestates", "core"),
-    ("simple", "core"),
     ("play_left_to_right", "core"),
     ("feature_column_names", "core"),
-    ("Actions", "core"),
-    ("GameStates", "core"),
-    ("Features", "core"),
-    ("FeatureTransfomer", "core"),
     # actiontype
     ("actiontype", "actiontype"),
     ("actiontype_onehot", "actiontype"),
     …
 )
```

33 → 27 parametrized cases. The dropped 6 are now covered by T-D (§ 5.4). T-B remains "every public symbol still in the features package lives in its expected submodule."

### 5.3 T-C — `tests/atomic/test_features_per_concern_import.py` (existing, rewritten)

The 2.3.0 version forbids package-root import and requires per-concern imports from `core` / `bodypart` / `context` / `temporal`. Rewrite:

```python
def test_atomic_imports_framework_and_per_concern() -> None:
    """Atomic VAEP features imports framework from the dedicated cross-package
    module, and per-concern feature reuse from the appropriate submodules.

    Codifies the A9-closure shape (silly-kicks 2.4.0). If a future PR
    consolidates atomic's imports back to the package root or reaches into
    `vaep.features.core` for framework primitives, this test fails fast.
    """
    source = inspect.getsource(atomic_features)

    # Forbid: package-root re-export reach (would mask which submodules atomic
    # actually depends on, eroding the per-concern boundary).
    assert "from silly_kicks.vaep.features import" not in source, (
        "atomic.vaep.features must import per-concern, not from package root."
    )

    # Forbid: reaching into vaep.features.core for framework primitives
    # (post-2.4.0 those live in silly_kicks.vaep.feature_framework).
    assert "from silly_kicks.vaep.features.core import" not in source, (
        "atomic.vaep.features must import framework primitives from "
        "silly_kicks.vaep.feature_framework, not from vaep.features.core."
    )

    # Forbid: the now-deleted `_actiontype` symbol.
    assert "_actiontype" not in source, (
        "atomic.vaep.features must use actiontype_categorical (public) — "
        "the private _actiontype was promoted in silly-kicks 2.4.0."
    )

    # Require: framework module import + 3 per-concern submodule imports.
    expected_lines = (
        "from silly_kicks.vaep.feature_framework import",
        "from silly_kicks.vaep.features.bodypart import",
        "from silly_kicks.vaep.features.context import",
        "from silly_kicks.vaep.features.temporal import",
    )
    for line in expected_lines:
        assert line in source, f"atomic.vaep.features missing required import line: {line!r}"
```

Still 1 test case (single function); test body is the load-bearing change.

### 5.4 T-D — `tests/vaep/test_feature_framework_layout.py` (NEW)

Locks the framework module's surface and the canonical `__module__` of each symbol.

```python
"""Framework module layout lock: every framework primitive's canonical
home is silly_kicks.vaep.feature_framework. Closes A9 (silly-kicks 2.4.0).
"""

from __future__ import annotations

import importlib

import pytest

_FRAMEWORK_SYMBOLS: tuple[str, ...] = (
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "actiontype_categorical",
    "gamestates",
    "simple",
)

_FRAMEWORK_MODULE = "silly_kicks.vaep.feature_framework"


@pytest.mark.parametrize("symbol_name", _FRAMEWORK_SYMBOLS)
def test_symbol_lives_in_framework_module(symbol_name: str) -> None:
    """Each framework primitive is canonically defined in feature_framework.

    For function/class symbols, ``__module__`` is the canonical check. For
    type aliases (``Actions = pd.DataFrame``), ``__module__`` points to the
    alias target's defining module (``pandas.core.frame``); attribute
    presence on the framework module is the binding check.
    """
    mod = importlib.import_module(_FRAMEWORK_MODULE)
    assert hasattr(mod, symbol_name), (
        f"{symbol_name} not exposed by {_FRAMEWORK_MODULE}"
    )
    symbol = getattr(mod, symbol_name)
    actual = getattr(symbol, "__module__", None)
    if actual is not None and actual.startswith("silly_kicks."):
        # Function/class defined within our package — __module__ is canonical.
        assert actual == _FRAMEWORK_MODULE, (
            f"{symbol_name} should be defined in {_FRAMEWORK_MODULE} "
            f"(actually defined in {actual})"
        )
    # Else: type alias. Attribute-presence check above is sufficient.
```

7 parametrized cases.

### 5.5 Examples gate — `tests/test_public_api_examples.py`

Add `silly_kicks/vaep/feature_framework.py` to `_PUBLIC_MODULE_FILES` (alphabetically slotted between `silly_kicks/vaep/features/specialty.py` and a hypothetical later vaep file). 26 → 27 parametrized cases. Within the new file's case, 3 public functions need Examples sections (`gamestates`, `simple`, `actiontype_categorical`); the 4 type aliases are skipped by the `_walk_public_definitions` AST walk (only `FunctionDef` / `ClassDef` nodes are checked; type-alias `Actions = pd.DataFrame` is an `Assign` node).

`gamestates` and `simple` already have Examples sections in the 2.3.0 `core.py` (verified — they were given Examples in PR-S15). The Examples sections move with the function bodies. `actiontype_categorical` gets a new Examples section as drafted in § 4.2.

### 5.6 Test count delta

| Test file | 2.3.0 | 2.4.0 | Δ |
|---|---|---|---|
| `tests/vaep/test_features_backcompat.py` (T-A) | 33 | 34 | +1 |
| `tests/vaep/test_features_submodule_layout.py` (T-B) | 33 | 27 | -6 |
| `tests/atomic/test_features_per_concern_import.py` (T-C) | 1 | 1 | 0 |
| `tests/vaep/test_feature_framework_layout.py` (T-D NEW) | 0 | 7 | +7 |
| `tests/test_public_api_examples.py` (Examples gate) | 26 | 27 | +1 |
| **Net delta** | | | **+3** |

Baseline: 881 passing, 4 deselected (per 2.3.0 spec § 6). Expected after this PR: **884 passing, 4 deselected**.

## 6. Implementation order (TDD-first)

Following the structural-refactor pattern from PR-S15 (memory: *"for any module decomposition / large-scale move, write T-A backcompat + T-B layout + T-C coupling-shape parametrized tests upfront"*).

1. **T1 — Add T-D (framework layout test).** RED — `silly_kicks.vaep.feature_framework` doesn't exist; all 7 cases fail with `ModuleNotFoundError`.

2. **T2 — Update T-C (atomic-coupling test).** RED — atomic still imports from `vaep.features.core`; the new "forbid `vaep.features.core` for framework primitives" assertion fails.

3. **T3 — Update T-A (backcompat test).** RED — `actiontype_categorical` not yet importable from `silly_kicks.vaep.features`.

4. **T4 — Update T-B (layout test).** Drop the 6 rows. After this update T-B is GREEN (the remaining 27 rows still hold against the 2.3.0 layout).

5. **T5 — Create `silly_kicks/vaep/feature_framework.py`.** Move:
   - 4 type aliases verbatim from `vaep/features/core.py`
   - `gamestates` (entire body + docstring) from `vaep/features/core.py`
   - `simple` (entire body + docstring) from `vaep/features/core.py`
   - Adapt `_actiontype` → `actiontype_categorical(actions, spadl_cfg)`: drop leading underscore, change parameter to positional, drop None fallback, add Examples docstring per § 4.2.

   T-D now passes 7/7. T-A still fails (1 case — `actiontype_categorical` not yet re-exported from features package).

6. **T6 — Slim `vaep/features/core.py`.** Replace the 6 framework definitions with `from silly_kicks.vaep.feature_framework import *`. Keep `play_left_to_right`, `feature_column_names`. Update `__all__` to reflect the new union (lists the framework re-exports + the 2 native helpers + `actiontype_categorical` re-export). Delete the old `_actiontype` definition.

7. **T7 — Update `vaep/features/__init__.py`.** Add `from silly_kicks.vaep.feature_framework import *` (alphabetically first, before `.actiontype`). Add `"actiontype_categorical"` to the static `__all__` (alphabetically before `"actiontype"`). T-A now passes 34/34.

8. **T8 — Update `vaep/features/actiontype.py`.** Replace `from silly_kicks.vaep.features.core import _actiontype, simple, Actions, Features` with `from silly_kicks.vaep.feature_framework import actiontype_categorical, simple, Actions, Features` (and add `import silly_kicks.spadl.config as spadlcfg`). Update body to `return actiontype_categorical(actions, spadlcfg)`.

9. **T9 — Refit `silly_kicks/atomic/vaep/features.py`.** Per § 4.6 diff. T-C now passes.

10. **T10 — Examples gate update.** `_PUBLIC_MODULE_FILES` adds `silly_kicks/vaep/feature_framework.py`. Run gate → 27/27 passing.

11. **T11 — Verification gates.** ruff check, ruff format --check, pyright, full pytest (-m "not e2e"), smoke-import test (§ 7).

12. **T12 — Write ADR-002.** `docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md`. Format from `ADR-TEMPLATE.md`. Status: Accepted. References this spec, ADR-001 pattern, PR-S15 deferral context.

13. **T13 — Update TODO.md.** Delete the A9 row from "Architecture". The "Architecture" section becomes empty (note: "(none currently queued)" or similar, matching the convention used in "Documentation" and "Tech Debt" sections).

14. **T14 — Update CHANGELOG.md.** New `[2.4.0]` entry covering: new module, A9 close, `_actiontype` → `actiontype_categorical` rename, ADR-002.

15. **T15 — Version bump.** `pyproject.toml` 2.3.0 → 2.4.0.

16. **T16 — Update memory.** Refresh: `project_release_state.md` (current version 2.4.0, last PR PR-S16, A9 closed); `project_followup_prs.md` (PR-S16 SHIPPED, no follow-up queued — last known TODO closed); `project_phase_roadmap.md` if applicable.

17. **T17 — `/final-review`.**

18. **T18 — Single-commit gate (USER APPROVAL).**

19. **T19 — Push, PR, CI watch, squash-merge --admin, tag v2.4.0, verify PyPI.**

## 7. Verification gates (before commit)

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/

# Smoke test — every backwards-compat path resolves identically AND the new
# framework path resolves identically.
uv run python -c "
from silly_kicks.vaep.feature_framework import (
    Actions, Features, FeatureTransfomer, GameStates,
    actiontype_categorical, gamestates, simple,
)
from silly_kicks.vaep.features import (
    Actions as A2, gamestates as gs2, simple as s2,
    actiontype_categorical as ac2,
)
from silly_kicks.vaep.features.core import gamestates as gs3, simple as s3
assert Actions is A2
assert gamestates is gs2 is gs3
assert simple is s2 is s3
assert actiontype_categorical is ac2
print('smoke OK')
"

# Atomic-side smoke
uv run python -c "
import silly_kicks.atomic.vaep.features as af
import silly_kicks.atomic.spadl.config as ac
import pandas as pd
# actiontype_categorical reachable through atomic transitively
df = pd.DataFrame({'type_id': [0]})
result = af.actiontype_categorical(df, ac)
assert 'actiontype' in result.columns
print('atomic smoke OK')
"

# Full pytest suite (-m 'not e2e')
uv run pytest tests/ -m "not e2e" --tb=short
```

Expected: ruff/pyright clean; both smoke prints succeed; pytest = **884 passing, 4 deselected** (881 baseline + 1 T-A + (-6) T-B + 0 T-C + 7 T-D + 1 Examples gate parametrize = +3 net).

## 8. Risks + Hyrum's Law audit

| Risk | Mitigation |
|---|---|
| `gamestates.__module__` flips from `silly_kicks.vaep.features.core` to `silly_kicks.vaep.feature_framework` — anyone introspecting via `inspect.getmodule(gamestates)` would see the new value. | Documented in ADR-002 as accepted Hyrum's Law exposure. T-D codifies the new canonical home. The same precedent was set in PR-S15 (4 type aliases moved from monolith to `core` then; same `__module__`-flip class of change). |
| `_actiontype` rename is a true breaking change for any consumer importing the leading-underscore symbol directly. | Leading-underscore is documented-private by Python convention; consumers who imported it accepted instability. ADR-002 records this. The path forward — `actiontype_categorical` — is functionally equivalent and public. |
| Circular import between `vaep/features/core.py` and `vaep/feature_framework.py`. | None possible: `feature_framework` imports nothing from `vaep.features`; `vaep/features/core.py` imports from `feature_framework`. Linear dependency, no cycle. Smoke test (§ 7) catches regressions. |
| Standard-VAEP `actiontype` body change (`_actiontype(actions)` → `actiontype_categorical(actions, spadlcfg)`) is observable behavior at the boundary. | The two are exactly equivalent: `_actiontype(actions)` defaulted `_spadl_cfg=None` then fell back to `spadlcfg`, which is what `actiontype_categorical(actions, spadlcfg)` passes explicitly. Smoke + full pytest verify. The "drop None fallback" is intentional API tightening — the function is meaningless without a config. |
| pyright + pandas-stubs reaction to type aliases moving (memory: "type-alias `__module__` gotcha"). | `Actions = pd.DataFrame` etc. are pure type aliases; pyright resolves identically when imported from `feature_framework` vs `core` (the alias target is `pd.DataFrame` either way). Verified at T11 pyright run. T-D's type-alias-handling fallback (object identity check, not `__module__`) mirrors T-B's existing handling for the same case. |
| Examples gate adding a new file shifts test parametrize order, causing flaky-CI noise. | `_PUBLIC_MODULE_FILES` is a tuple — alphabetical insertion is stable. Each parametrized test runs independently. No flakiness. |
| `from silly_kicks.vaep.features import *` star-import semantics with the new framework re-export. | The framework module declares `__all__` covering its 7 symbols; `from .feature_framework import *` re-exports exactly those into the features package namespace, where they co-exist with the rest. No clobbering — names are unique. T-A (34 cases) verifies every public symbol resolves. |
| Atomic `actiontype_categorical(actions, atomicspadl)` vs prior `_actiontype(actions, _spadl_cfg=atomicspadl)` — semantically equivalent? | Yes. The keyword-only `_spadl_cfg=atomicspadl` and the positional `spadl_cfg=atomicspadl` produce identical function calls. Atomic test suite (~80 tests, untouched) is the substantive verification. |
| ADR-002 placement next to ADR-001 — does the doc set imply a numbered series consumers depend on? | ADR-001 is an architectural-decisions log; sequential numbering is conventional. Adding ADR-002 follows the established pattern (memory: "ADR pattern adopted 2.0.0"). No external Hyrum's Law exposure. |

## 9. Acceptance criteria

1. `silly_kicks/vaep/feature_framework.py` exists with the 7 framework symbols (`Actions`, `Features`, `FeatureTransfomer`, `GameStates`, `gamestates`, `simple`, `actiontype_categorical`) per § 4.2.
2. `silly_kicks/vaep/features/core.py` slimmed to `play_left_to_right` + `feature_column_names` + framework re-exports per § 4.3. The old `_actiontype` definition is removed.
3. `silly_kicks/vaep/features/__init__.py` re-exports the framework module's symbols; `__all__` includes `actiontype_categorical`.
4. `silly_kicks/vaep/features/actiontype.py` uses `actiontype_categorical` from `vaep.feature_framework` (not `_actiontype` from `vaep.features.core`).
5. `silly_kicks/atomic/vaep/features.py` imports framework primitives from `silly_kicks.vaep.feature_framework`; no longer imports `_actiontype`; per-concern imports from `bodypart`, `context`, `temporal` retained verbatim.
6. T-A passes 34/34; T-B passes 27/27; T-C passes (rewritten); T-D passes 7/7; Examples gate passes 27/27.
7. Total pytest count: 884 passing, 4 deselected.
8. `docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md` exists and follows the ADR-TEMPLATE format.
9. `TODO.md` "Architecture" section no longer lists A9.
10. `CHANGELOG.md` `[2.4.0]` entry covers framework extraction, A9 close, `_actiontype` → `actiontype_categorical`, ADR-002.
11. `pyproject.toml` version is `2.4.0`.
12. ruff / ruff format / pyright clean.
13. `/final-review` clean.
14. Memory files refreshed (release state, follow-up PRs).

## 10. Commit cycle

Same as PR-S15. Single commit. Branch: `feat/A9-feature-framework-extraction`. Version bump 2.3.0 → 2.4.0.
