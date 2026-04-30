# A9 — VAEP Feature Framework Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close TODO A9 by extracting the 7 framework primitives that both standard-VAEP and atomic-VAEP build on into a new public module `silly_kicks.vaep.feature_framework`, promoting `_actiontype` to the public `actiontype_categorical(actions, spadl_cfg)`, refitting both feature stacks to depend on the framework module, and refreshing the lock-in tests.

**Architecture:** New module `silly_kicks/vaep/feature_framework.py` (sibling of the `vaep/features/` package) holds 4 type aliases + `gamestates` + `simple` + `actiontype_categorical`. `vaep/features/core.py` slims to `play_left_to_right` + `feature_column_names` and re-exports the framework primitives for Hyrum's-Law preservation. Atomic-VAEP imports framework primitives directly from `feature_framework`; per-concern feature reuse from `bodypart` / `context` / `temporal` is preserved.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113.

**Spec:** `docs/superpowers/specs/2026-04-30-A9-feature-framework-extraction-design.md`

**Commit policy:** Per project CLAUDE.md memory — *literally ONE commit per branch; explicit user approval before that one commit*. This plan therefore does NOT have intermediate `git commit` steps. All changes accumulate in the working tree; a single commit at the very end (Task 20) gathers everything after user approval.

**Test count target:** 884 passing, 4 deselected (881 baseline + 1 T-A + (-6) T-B + 0 T-C + 7 T-D + 1 Examples-gate parametrize = +3 net).

---

## Phase 1 — Setup

### Task 1: Create feature branch

**Files:**
- (none — git operation only)

- [ ] **Step 1.1: Verify clean working tree**

Run:
```bash
git status --short
```
Expected output:
```
?? README.md.backup
?? uv.lock
```
(Or empty. The two untracked files from the conversation start are pre-existing — not introduced by this PR.)

- [ ] **Step 1.2: Create and switch to the feature branch**

Run:
```bash
git checkout -b feat/A9-feature-framework-extraction
```
Expected output: `Switched to a new branch 'feat/A9-feature-framework-extraction'`

- [ ] **Step 1.3: Verify branch position**

Run:
```bash
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
```
Expected: branch name `feat/A9-feature-framework-extraction`; last commit is `dbca13c feat(vaep): decompose vaep/features.py monolith into 8-submodule package — silly-kicks 2.3.0 (PR-S15) (#20)`.

### Task 2: Capture pytest baseline before any changes

**Files:**
- (none — verification only)

- [ ] **Step 2.1: Run baseline pytest count**

Run:
```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: ends with `881 passed, 4 deselected` (or substring matching that pattern). Note the **exact** number — this is the baseline. If different from 881, write it down; subsequent expectations adjust by `+3`.

---

## Phase 2 — TDD setup (RED tests written first)

### Task 3: Add T-D — framework module layout test (RED — module does not exist)

**Files:**
- Create: `tests/vaep/test_feature_framework_layout.py`

- [ ] **Step 3.1: Create the T-D test file**

Create `tests/vaep/test_feature_framework_layout.py` with this exact content:

```python
"""Framework module layout lock: every framework primitive's canonical
home is silly_kicks.vaep.feature_framework. Closes A9 (silly-kicks 2.4.0).

For function/class symbols, ``__module__`` is the canonical check. For
type aliases (``Actions = pd.DataFrame``), ``__module__`` points to the
alias target's defining module (``pandas.core.frame``); attribute
presence on the framework module is the binding check.
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
    """Each framework primitive is canonically defined in feature_framework."""
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

- [ ] **Step 3.2: Run T-D — verify all 7 cases FAIL**

Run:
```bash
uv run pytest tests/vaep/test_feature_framework_layout.py -v --tb=short
```
Expected: 7 cases FAIL with `ModuleNotFoundError: No module named 'silly_kicks.vaep.feature_framework'`. This is the desired RED state.

### Task 4: Update T-C — atomic coupling test (RED — atomic still imports from `core`)

**Files:**
- Modify: `tests/atomic/test_features_per_concern_import.py`

- [ ] **Step 4.1: Replace the T-C file with the rewritten version**

Replace the entire contents of `tests/atomic/test_features_per_concern_import.py` with:

```python
"""Atomic-coupling lock: silly_kicks.atomic.vaep.features imports framework
from the dedicated cross-package module (silly_kicks.vaep.feature_framework)
and per-concern feature reuse from the appropriate submodules.

Codifies the A9-closure shape (silly-kicks 2.4.0). If a future PR consolidates
atomic's imports back to the package root or reaches into vaep.features.core
for framework primitives, this test fails fast.
"""

from __future__ import annotations

import inspect

import silly_kicks.atomic.vaep.features as atomic_features


def test_atomic_imports_framework_and_per_concern() -> None:
    """Atomic VAEP features imports framework from feature_framework + per-concern features."""
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

- [ ] **Step 4.2: Run T-C — verify it FAILS on the new "forbid `vaep.features.core`" rule**

Run:
```bash
uv run pytest tests/atomic/test_features_per_concern_import.py -v --tb=short
```
Expected: 1 case FAILS with `AssertionError: atomic.vaep.features must import framework primitives from silly_kicks.vaep.feature_framework, not from vaep.features.core.` (atomic still imports `from silly_kicks.vaep.features.core import` — that's the RED state we want).

### Task 5: Update T-A — add `actiontype_categorical` to backwards-compat parametrize tuple (RED — symbol doesn't exist yet)

**Files:**
- Modify: `tests/vaep/test_features_backcompat.py`

- [ ] **Step 5.1: Add `actiontype_categorical` to `_PUBLIC_SYMBOLS`**

In `tests/vaep/test_features_backcompat.py`, the `_PUBLIC_SYMBOLS` tuple is sorted ASCII-first (uppercase block, then lowercase block). `actiontype_categorical` slots into the lowercase block between `actiontype` and `actiontype_onehot` (c < o). Apply this edit:

```diff
     "actiontype",
+    "actiontype_categorical",
     "actiontype_onehot",
     "actiontype_result_onehot",
```

- [ ] **Step 5.2: Run T-A — verify the new row FAILS, the other 33 PASS**

Run:
```bash
uv run pytest tests/vaep/test_features_backcompat.py -v --tb=short
```
Expected: **33 passed, 1 failed** — the failing case is `test_symbol_importable_from_package_path[actiontype_categorical]` with `AssertionError: actiontype_categorical no longer importable from silly_kicks.vaep.features. ...`. RED.

### Task 6: Refit T-B — drop the 6 framework rows (passes 27/27 against pre-extraction state)

**Files:**
- Modify: `tests/vaep/test_features_submodule_layout.py`

- [ ] **Step 6.1: Remove the 6 framework rows from `_LAYOUT`**

In `tests/vaep/test_features_submodule_layout.py`, the `_LAYOUT` tuple currently has 33 rows starting with these 8 under the `# core` comment:

```python
    # core
    ("gamestates", "core"),
    ("simple", "core"),
    ("play_left_to_right", "core"),
    ("feature_column_names", "core"),
    ("Actions", "core"),
    ("GameStates", "core"),
    ("Features", "core"),
    ("FeatureTransfomer", "core"),
```

Replace those 8 lines with these 2 lines + a comment:

```python
    # core (slimmed: only standard-SPADL-specific helpers stay; the 6 framework
    # primitives moved to silly_kicks.vaep.feature_framework — see T-D)
    ("play_left_to_right", "core"),
    ("feature_column_names", "core"),
```

- [ ] **Step 6.2: Run T-B — verify 27/27 PASS against pre-extraction state**

Run:
```bash
uv run pytest tests/vaep/test_features_submodule_layout.py -v --tb=short
```
Expected: **27 passed**. None fail because all 27 remaining symbols are still in their expected submodules in the 2.3.0 layout. T-B is now ready for the post-extraction state.

### Task 7: Phase-2 verification — confirm all RED states are exactly as expected

**Files:**
- (none — verification only)

- [ ] **Step 7.1: Run the four targeted test files**

Run:
```bash
uv run pytest tests/vaep/test_feature_framework_layout.py tests/atomic/test_features_per_concern_import.py tests/vaep/test_features_backcompat.py tests/vaep/test_features_submodule_layout.py -v --tb=short -q
```
Expected:
- T-D (`test_feature_framework_layout.py`): 7 failed (ModuleNotFoundError)
- T-C (`test_features_per_concern_import.py`): 1 failed (AssertionError on `vaep.features.core` rule)
- T-A (`test_features_backcompat.py`): 1 failed, 33 passed (`actiontype_categorical` not yet importable)
- T-B (`test_features_submodule_layout.py`): 27 passed
- Total: **27 + 33 = 60 passed, 9 failed**

This is the desired RED-state baseline. Proceed to implementation.

---

## Phase 3 — Framework module

### Task 8: Create `silly_kicks/vaep/feature_framework.py`

**Files:**
- Create: `silly_kicks/vaep/feature_framework.py`

- [ ] **Step 8.1: Read source bodies from current `core.py`**

Run:
```bash
uv run python -c "
from silly_kicks.vaep.features.core import gamestates, simple, _actiontype
import inspect
print('--- gamestates ---'); print(inspect.getsource(gamestates))
print('--- simple ---'); print(inspect.getsource(simple))
print('--- _actiontype ---'); print(inspect.getsource(_actiontype))
"
```
Use the printed sources for the next step. (Alternative: open `silly_kicks/vaep/features/core.py` and read directly — same content.)

- [ ] **Step 8.2: Write `silly_kicks/vaep/feature_framework.py`**

Create the file with this exact content. Imports list `pandas as pd` only — `numpy` is **not** needed at module level (the moved bodies use only pandas methods like `.shift`, `.groupby`, `.transform`):

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

import pandas as pd  # type: ignore

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
    r"""Convert a dataframe of actions to gamestates.

    Each gamestate is represented as the <nb_prev_actions> previous actions.

    The list of gamestates is internally represented as a list of actions
    dataframes :math:`[a_0,a_1,\ldots]` where each row in the a_i dataframe contains the
    previous action of the action in the same row in the :math:`a_{i-1}` dataframe.

    Parameters
    ----------
    actions : Actions
        A DataFrame with the actions of a game.
    nb_prev_actions : int, default=3  # noqa: DAR103
        The number of previous actions included in the game state.

    Raises
    ------
    ValueError
        If the number of actions is smaller 1.

    Returns
    -------
    GameStates
         The <nb_prev_actions> previous actions for each action.

    Examples
    --------
    Build a 3-step gamestate stream from a SPADL action DataFrame::

        from silly_kicks.vaep.feature_framework import gamestates

        states = gamestates(actions, nb_prev_actions=3)
        # ``states`` is a list of 3 DataFrames — states[0] is the current action,
        # states[1] is the previous action aligned by row, states[2] is the one before.
    """
    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    states = [actions]
    group_keys = ["game_id", "period_id"]
    # Precompute group-first values once for boundary filling (excludes groupby keys)
    first_in_group = actions.groupby(group_keys, sort=False).transform("first")
    for i in range(1, nb_prev_actions):
        prev = actions.shift(i)
        # Detect period/game boundaries: where shifted row crosses a group
        boundary = (actions["game_id"] != actions["game_id"].shift(i)) | (
            actions["period_id"] != actions["period_id"].shift(i)
        )
        # At boundaries, fill groupby-key columns with current row values
        for col in group_keys:
            prev.loc[boundary, col] = actions.loc[boundary, col]
        # At boundaries, fill remaining columns with the first row of the current group
        for col in first_in_group.columns:
            prev.loc[boundary, col] = first_in_group.loc[boundary, col]
        prev.index = actions.index.copy()
        states.append(prev)
    return states


@no_type_check
def simple(actionfn: Callable) -> FeatureTransfomer:
    """Make a function decorator to apply actionfeatures to game states.

    Parameters
    ----------
    actionfn : Callable
        A feature transformer that operates on actions.

    Returns
    -------
    FeatureTransfomer
        A feature transformer that operates on game states.

    Examples
    --------
    Lift an action-level feature function to a gamestate-level transformer::

        from silly_kicks.vaep.feature_framework import simple

        @simple
        def my_action_feature(actions):
            return actions[['x']]

        feats = my_action_feature(states)
    """

    @wraps(actionfn)
    def _wrapper(gamestates: list[Actions]) -> pd.DataFrame:
        if not isinstance(gamestates, (list,)):
            gamestates = [gamestates]
        X = []
        for i, a in enumerate(gamestates):
            Xi = actionfn(a)
            Xi.columns = [c + "_a" + str(i) for c in Xi.columns]
            X.append(Xi)
        return pd.concat(X, axis=1)  # type: ignore[reportReturnType]

    return _wrapper


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

- [ ] **Step 8.3: Run T-D — verify all 7 cases now PASS**

Run:
```bash
uv run pytest tests/vaep/test_feature_framework_layout.py -v --tb=short
```
Expected: **7 passed**. T-D is now GREEN.

- [ ] **Step 8.4: Smoke-import the new module manually**

Run:
```bash
uv run python -c "
from silly_kicks.vaep.feature_framework import (
    Actions, Features, FeatureTransfomer, GameStates,
    actiontype_categorical, gamestates, simple,
)
print('framework imports OK')
print('actiontype_categorical signature:', actiontype_categorical.__doc__.split(chr(10))[0])
"
```
Expected output:
```
framework imports OK
actiontype_categorical signature: SPADL-config-parameterized categorical actiontype helper.
```

---

## Phase 4 — Slim core.py

### Task 9: Refactor `silly_kicks/vaep/features/core.py` to delegate framework primitives

**Files:**
- Modify: `silly_kicks/vaep/features/core.py`

- [ ] **Step 9.1: Replace `core.py` with the slim version**

Replace the **entire contents** of `silly_kicks/vaep/features/core.py` with:

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

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

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
    """Return the names of the features generated by a list of transformers.

    Parameters
    ----------
    fs : list(callable)
        A list of feature transformers.
    nb_prev_actions : int, default=3  # noqa: DAR103
        The number of previous actions included in the game state.

    Returns
    -------
    list(str)
        The name of each generated feature.

    Examples
    --------
    Discover the feature column names a feature-fn list will produce::

        from silly_kicks.vaep import features as fs

        cols = fs.feature_column_names([fs.actiontype_onehot, fs.bodypart_onehot])
        # cols includes the cartesian product of feature x previous-action index.
    """
    spadlcolumns = [
        "game_id",
        "original_event_id",
        "action_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "result_id",
        "result_name",
        "bodypart_id",
        "bodypart_name",
        "type_id",
        "type_name",
    ]
    dummy_actions = pd.DataFrame(np.zeros((10, len(spadlcolumns))), columns=spadlcolumns)
    for c in spadlcolumns:
        if "name" in c:
            dummy_actions[c] = dummy_actions[c].astype(str)
    gs = gamestates(dummy_actions, nb_prev_actions)  # type: ignore
    return list(pd.concat([f(gs) for f in fs], axis=1).columns.values)


def play_left_to_right(gamestates: GameStates, home_team_id: int) -> GameStates:
    """Perform all actions in a gamestate in the same playing direction.

    This changes the start and end location of each action in a gamestate,
    such that all actions are performed as if the team that performs the first
    action in the gamestate plays from left to right.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    GameStates
        The game states with all actions performed left to right.

    See Also
    --------
    silly_kicks.spadl.play_left_to_right : For transforming actions.

    Examples
    --------
    Mirror gamestates to a single direction (per home_team_id)::

        from silly_kicks.vaep.features import play_left_to_right

        ltr_states = play_left_to_right(states, home_team_id=100)
        # Every away-team gamestate now has flipped (start_x, start_y) / (end_x, end_y).
    """
    a0 = gamestates[0]
    away_idx = a0.team_id != home_team_id
    for actions in gamestates:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = spadlcfg.field_length - actions[away_idx][col].values  # type: ignore[reportAttributeAccessIssue]
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = spadlcfg.field_width - actions[away_idx][col].values  # type: ignore[reportAttributeAccessIssue]
    return gamestates
```

Notes:
- The old `_actiontype` definition (and its module-level fallback to `spadlcfg`) is **deleted** — replaced by the public `actiontype_categorical` in framework, imported above.
- The 4 type aliases + `gamestates` + `simple` are imported from framework, not redefined.
- `__all__` includes the framework re-exports + the 2 native helpers + `actiontype_categorical` (re-exported via the import block).
- `np` and `spadlcfg` are still used inside `feature_column_names` and `play_left_to_right` respectively — keep both imports.

- [ ] **Step 9.2: Verify standard-VAEP still imports cleanly**

Run:
```bash
uv run python -c "
from silly_kicks.vaep.features.core import (
    Actions, Features, FeatureTransfomer, GameStates,
    actiontype_categorical, feature_column_names,
    gamestates, play_left_to_right, simple,
)
print('core re-exports OK')
"
```
Expected output: `core re-exports OK`.

- [ ] **Step 9.3: Run vaep tests — ensure nothing in the standard feature stack broke**

Run:
```bash
uv run pytest tests/vaep/ -v --tb=short -q
```
Expected: All previously-passing vaep tests still pass. T-A still has 1 fail (`actiontype_categorical` not yet in features package); T-B passes 27/27; T-D passes 7/7. (Other VAEP tests — feature transformer tests, etc. — should pass because `core`'s public surface is preserved through re-exports.)

If anything else fails, STOP and diagnose before proceeding.

---

## Phase 5 — Wire up the package

### Task 10: Update `silly_kicks/vaep/features/__init__.py` to expose framework symbols at the package level

**Files:**
- Modify: `silly_kicks/vaep/features/__init__.py`

- [ ] **Step 10.1: Add framework re-export and `actiontype_categorical` to `__all__`**

Current file (lines 1-23):
```python
# ruff: noqa: F405
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

from .actiontype import *  # noqa: F403
from .bodypart import *  # noqa: F403
from .context import *  # noqa: F403
from .core import *  # noqa: F403
from .result import *  # noqa: F403
from .spatial import *  # noqa: F403
from .specialty import *  # noqa: F403
from .temporal import *  # noqa: F403
```

Apply two edits:

(a) Insert the framework re-export as a separate import block above the relative-import block (per project isort convention — absolute first-party imports, then a blank line, then relative imports). Insert after the docstring closing `"""` and BEFORE the `from .actiontype` line:

```diff
 """

+from silly_kicks.vaep.feature_framework import *  # noqa: F403
+
 from .actiontype import *  # noqa: F403
```

(b) Add `"actiontype_categorical"` to the static `__all__` list. The list is alphabetised ASCII-first. `actiontype_categorical` slots between `actiontype` and `actiontype_onehot` (c < o):

```diff
     "actiontype",
+    "actiontype_categorical",
     "actiontype_onehot",
     "actiontype_result_onehot",
     "actiontype_result_onehot_prev_only",
```

- [ ] **Step 10.2: Run T-A — verify all 34 cases PASS**

Run:
```bash
uv run pytest tests/vaep/test_features_backcompat.py -v --tb=short
```
Expected: **34 passed**. T-A is now GREEN.

- [ ] **Step 10.3: Smoke-import via the package path**

Run:
```bash
uv run python -c "
from silly_kicks.vaep.features import (
    actiontype_categorical, gamestates, simple,
    Actions, Features, FeatureTransfomer, GameStates,
)
import silly_kicks.spadl.config as spadlcfg
import pandas as pd
df = pd.DataFrame({'type_id': [0, 1, 2]})
result = actiontype_categorical(df, spadlcfg)
print('result columns:', list(result.columns))
print('result dtype:', result['actiontype'].dtype)
print('package re-export OK')
"
```
Expected output:
```
result columns: ['actiontype']
result dtype: category
package re-export OK
```

---

## Phase 6 — Update standard-VAEP `actiontype.py`

### Task 11: Refactor `silly_kicks/vaep/features/actiontype.py` to use `actiontype_categorical`

**Files:**
- Modify: `silly_kicks/vaep/features/actiontype.py`

- [ ] **Step 11.1: Replace the imports block**

In `silly_kicks/vaep/features/actiontype.py`, replace lines 12 (the `from .core` import) with the framework import:

Current line 12:
```python
from .core import Actions, Features, _actiontype, simple
```

Replace with:
```python
from silly_kicks.vaep.feature_framework import (
    Actions,
    Features,
    actiontype_categorical,
    simple,
)
```

The existing `import silly_kicks.spadl.config as spadlcfg` (line 10) and `import pandas as pd` (line 8) are kept verbatim.

After this edit, ruff/isort will want the imports in this order (top of file unchanged, then):

```python
import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.vaep.feature_framework import (
    Actions,
    Features,
    actiontype_categorical,
    simple,
)
```

(The `from silly_kicks.vaep.feature_framework import` block goes after `import silly_kicks.spadl.config as spadlcfg` because both are first-party absolute, sorted alphabetically: `silly_kicks.spadl.config` < `silly_kicks.vaep.feature_framework`.)

Also update the module-level docstring (lines 1-6). Current:
```python
"""Action-type feature transformers.

Two transformers: integer-typed and one-hot-encoded action-type per gamestate
slot. Both delegate the standard / atomic split via the ``_actiontype`` helper
in :mod:`silly_kicks.vaep.features.core`.
"""
```

Replace with:
```python
"""Action-type feature transformers.

Two transformers: integer-typed and one-hot-encoded action-type per gamestate
slot. The integer-typed variant delegates to ``actiontype_categorical`` in
:mod:`silly_kicks.vaep.feature_framework` so standard and atomic VAEP share
the same SPADL-config-parameterized helper.
"""
```

- [ ] **Step 11.2: Replace the `actiontype` body**

Current body (line 39):
```python
    return _actiontype(actions)
```

Replace with:
```python
    return actiontype_categorical(actions, spadlcfg)
```

The `actiontype_onehot` body (lines 42-68) is **unchanged** — it uses `spadlcfg.actiontypes` directly, no helper change needed.

- [ ] **Step 11.3: Run actiontype-related tests**

Run:
```bash
uv run pytest tests/vaep/ -v --tb=short -k "actiontype"
```
Expected: All actiontype-related tests pass. The `actiontype` feature transformer's behavior is unchanged (the new code path passes the same `spadlcfg` that the old `_actiontype(actions)` defaulted to).

If failures occur, run:
```bash
uv run python -c "
from silly_kicks.vaep.features import actiontype, gamestates
import silly_kicks.spadl.config as spadlcfg
import pandas as pd
df = pd.DataFrame({
    'game_id': [1]*3, 'period_id': [1]*3,
    'type_id': [0, 1, 2],
    'start_x': [0.0]*3, 'start_y': [0.0]*3,
    'end_x': [0.0]*3, 'end_y': [0.0]*3,
})
states = gamestates(df, nb_prev_actions=1)
print(actiontype(states))
"
```
This should print a DataFrame with one column `actiontype_a0` of dtype Categorical. If that works but the test suite fails, inspect the failing test for context-specific issues.

---

## Phase 7 — Refit atomic-VAEP

### Task 12: Refit `silly_kicks/atomic/vaep/features.py`

**Files:**
- Modify: `silly_kicks/atomic/vaep/features.py`

- [ ] **Step 12.1: Replace the imports block**

In `silly_kicks/atomic/vaep/features.py`, replace the existing `from silly_kicks.vaep.features.core import (...)` block (lines 14-22) with a `from silly_kicks.vaep.feature_framework import (...)` block. Per ruff/isort alphabetical sort, the framework import goes BEFORE `from silly_kicks.vaep.features.bodypart import` (because `vaep.feature_framework` < `vaep.features.bodypart` — `_` precedes `s` in ASCII).

Current imports (lines 6-23):
```python
import silly_kicks.atomic.spadl.config as atomicspadl
from silly_kicks.vaep.features.bodypart import (
    bodypart,
    bodypart_detailed,
    bodypart_detailed_onehot,
    bodypart_onehot,
)
from silly_kicks.vaep.features.context import player_possession_time, team
from silly_kicks.vaep.features.core import (
    Actions,
    Features,
    FeatureTransfomer,
    GameStates,
    _actiontype,
    gamestates,
    simple,
)
from silly_kicks.vaep.features.temporal import speed, time, time_delta
```

Replace with:
```python
import silly_kicks.atomic.spadl.config as atomicspadl
from silly_kicks.vaep.feature_framework import (
    Actions,
    FeatureTransfomer,
    Features,
    GameStates,
    actiontype_categorical,
    gamestates,
    simple,
)
from silly_kicks.vaep.features.bodypart import (
    bodypart,
    bodypart_detailed,
    bodypart_detailed_onehot,
    bodypart_onehot,
)
from silly_kicks.vaep.features.context import player_possession_time, team
from silly_kicks.vaep.features.temporal import speed, time, time_delta
```

(Note on the Actions / FeatureTransfomer / Features / GameStates order inside the import block: ASCII `T` (0x54) precedes lowercase `s` (0x73), so `FeatureTransfomer` comes before `Features`. The block above already has this order; keep it.)

- [ ] **Step 12.2: Replace the `actiontype` body**

Current body (line ~62):
```python
    return _actiontype(actions, _spadl_cfg=atomicspadl)
```

Replace with:
```python
    return actiontype_categorical(actions, atomicspadl)
```

- [ ] **Step 12.3: Run T-C — verify GREEN**

Run:
```bash
uv run pytest tests/atomic/test_features_per_concern_import.py -v --tb=short
```
Expected: **1 passed**. T-C is now GREEN.

- [ ] **Step 12.4: Run atomic-VAEP feature tests**

Run:
```bash
uv run pytest tests/atomic/ -v --tb=short -q
```
Expected: All atomic tests pass (the `actiontype_categorical(actions, atomicspadl)` call is semantically equivalent to the old `_actiontype(actions, _spadl_cfg=atomicspadl)`).

---

## Phase 8 — Examples gate

### Task 13: Add `feature_framework.py` to the Examples-gate file list

**Files:**
- Modify: `tests/test_public_api_examples.py`

- [ ] **Step 13.1: Insert `feature_framework.py` into `_PUBLIC_MODULE_FILES`**

In `tests/test_public_api_examples.py`, the `_PUBLIC_MODULE_FILES` tuple (lines 18-45) lists 26 module files. Insert `"silly_kicks/vaep/feature_framework.py"` alphabetically. Looking at the current ordering, the relevant region is:

```python
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
    ...
```

The current list is **not strictly alphabetical** — it groups by package (vaep, atomic/vaep, vaep/features) with topic ordering inside each group. Follow the existing convention: insert `feature_framework.py` next to other top-level `vaep/` modules. Concretely, place it after `silly_kicks/vaep/formula.py` and before `silly_kicks/atomic/vaep/features.py` — keeping the "all `silly_kicks/vaep/*.py` files together" pattern:

```diff
     "silly_kicks/vaep/labels.py",
     "silly_kicks/vaep/formula.py",
+    "silly_kicks/vaep/feature_framework.py",
     "silly_kicks/atomic/vaep/features.py",
```

- [ ] **Step 13.2: Run the Examples gate — verify all 27 cases PASS**

Run:
```bash
uv run pytest tests/test_public_api_examples.py -v --tb=short
```
Expected: **27 passed**. The new case `test_public_definitions_have_examples_section[silly_kicks/vaep/feature_framework.py]` walks the framework module's public defs (`gamestates`, `simple`, `actiontype_categorical`); each has an Examples section per Step 8.2. Type aliases are skipped by the AST walk (they're `Assign` nodes, not `FunctionDef`/`ClassDef`).

---

## Phase 9 — Verification gates

### Task 14: Run all verification gates before commit

**Files:**
- (none — verification only)

- [ ] **Step 14.1: ruff check**

Run:
```bash
uv run ruff check silly_kicks/ tests/
```
Expected: `All checks passed!` (or empty stdout with exit code 0). If failures: read the output, fix in the reported file, re-run.

- [ ] **Step 14.2: ruff format check**

Run:
```bash
uv run ruff format --check silly_kicks/ tests/
```
Expected: `XX files already formatted` with exit code 0. If failures: run `uv run ruff format silly_kicks/ tests/` to apply, then re-run --check.

- [ ] **Step 14.3: pyright type check**

Run:
```bash
uv run pyright silly_kicks/
```
Expected: `0 errors, 0 warnings, 0 informations`. If errors: read the output, fix in the reported file. Most-likely concern is the `Any` parameter type on `actiontype_categorical(actions, spadl_cfg: Any)` — pyright should accept `Any` cleanly since SPADL config modules don't have a uniform typed Protocol in this repo.

- [ ] **Step 14.4: Smoke test — backwards-compat resolution**

Run:
```bash
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
```
Expected output: `smoke OK`.

- [ ] **Step 14.5: Smoke test — atomic side**

Run:
```bash
uv run python -c "
import silly_kicks.atomic.vaep.features as af
import silly_kicks.atomic.spadl.config as ac
import pandas as pd
df = pd.DataFrame({'type_id': [0]})
result = af.actiontype_categorical(df, ac)
assert 'actiontype' in result.columns
assert str(result['actiontype'].dtype) == 'category'
print('atomic smoke OK')
"
```
Expected output: `atomic smoke OK`.

- [ ] **Step 14.6: Full pytest suite (-m "not e2e")**

Run:
```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: ends with `884 passed, 4 deselected` (or `<baseline + 3> passed, 4 deselected`). If a different number, audit:
- T-A: 34 passed (was 33 baseline)
- T-B: 27 passed (was 33 baseline)
- T-C: 1 passed (unchanged count, rewritten)
- T-D: 7 passed (NEW)
- Examples gate: 27 passed (was 26 baseline)
- Net delta: +1 - 6 + 0 + 7 + 1 = **+3**

If the count is off by more than +3, find the unexpected failure or skip and resolve before proceeding.

---

## Phase 10 — Documentation

### Task 15: Write ADR-002

**Files:**
- Create: `docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md`

- [ ] **Step 15.1: Write the ADR**

Create the file with this exact content:

```markdown
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
```

### Task 16: Update TODO.md — delete A9 row

**Files:**
- Modify: `TODO.md`

- [ ] **Step 16.1: Replace the Architecture section**

Current `TODO.md` (lines 10-14):
```markdown
## Architecture

| # | Size | Item | Context |
|---|------|------|---------|
| A9 | Low | `atomic/vaep/features.py` per-concern coupling to `vaep/features` (12 symbols across 4 submodules — was monolith) | Partially addressed in 2.3.0 via vaep/features decomposition. Full decoupling deferred until atomic features genuinely need to diverge independently — extracting truly-shared framework into a cross-package module is the trigger condition. |
```

Replace with:
```markdown
## Architecture

(none currently queued — A9 closed in silly-kicks 2.4.0 via `silly_kicks.vaep.feature_framework` extraction; see ADR-002)
```

- [ ] **Step 16.2: Verify TODO.md is empty across all sections**

Run:
```bash
uv run python -c "
from pathlib import Path
content = Path('TODO.md').read_text(encoding='utf-8')
assert 'A9' not in content, 'A9 row still present'
assert '(none currently queued — A9 closed' in content, 'A9-closure note missing'
print('TODO.md OK — Architecture section closed')
"
```
Expected output: `TODO.md OK — Architecture section closed`.

### Task 17: Update CHANGELOG.md — new `[2.4.0]` entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 17.1: Read current CHANGELOG header**

Run:
```bash
uv run python -c "
from pathlib import Path
text = Path('CHANGELOG.md').read_text(encoding='utf-8')
print(text[:600])
"
```
Note the existing format style. (Use the same `## [X.Y.Z] - YYYY-MM-DD` heading pattern as 2.3.0.)

- [ ] **Step 17.2: Insert the 2.4.0 entry**

Insert this block immediately above the existing `## [2.3.0]` heading:

```markdown
## [2.4.0] - 2026-04-30

### Added

- New public module `silly_kicks.vaep.feature_framework` holding the 7 framework primitives both standard and atomic VAEP feature stacks build on: 4 type aliases (`Actions`, `Features`, `FeatureTransfomer`, `GameStates`), `gamestates`, `simple`, and the promoted helper `actiontype_categorical(actions, spadl_cfg)`.
- `actiontype_categorical(actions, spadl_cfg)` — promoted from the previously-private `_actiontype` to a public, SPADL-config-parameterized framework helper. Both standard-VAEP and atomic-VAEP wrap it with `@simple` to produce their respective `actiontype` feature transformers.
- `tests/vaep/test_feature_framework_layout.py` — 7-case framework-layout lock (T-D).
- `docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md` — captures the framework-extraction decision, alternatives, and `_actiontype → actiontype_categorical` rename rationale.

### Changed

- `silly_kicks.vaep.features.core` slimmed to its standard-SPADL-specific helpers (`play_left_to_right`, `feature_column_names`); re-exports the framework primitives from `silly_kicks.vaep.feature_framework` so existing `from silly_kicks.vaep.features.core import gamestates` paths continue to resolve (Hyrum's-Law preservation).
- `silly_kicks.atomic.vaep.features` imports framework primitives directly from `silly_kicks.vaep.feature_framework` (no longer reaches into `vaep.features.core`); per-concern feature reuse from `bodypart` / `context` / `temporal` is preserved.
- T-A (`tests/vaep/test_features_backcompat.py`) gains one row (`actiontype_categorical`) — 33 → 34 cases.
- T-B (`tests/vaep/test_features_submodule_layout.py`) drops the 6 framework rows now living outside the features package — 33 → 27 cases.
- T-C (`tests/atomic/test_features_per_concern_import.py`) rewritten to forbid `vaep.features.core` import for framework primitives and require import from `vaep.feature_framework`.
- Examples-gate file list (`tests/test_public_api_examples.py`) adds `silly_kicks/vaep/feature_framework.py` — 26 → 27 cases.

### Removed

- `silly_kicks.vaep.features.core._actiontype` — promoted to public `actiontype_categorical(actions, spadl_cfg)` in the new framework module. Was a leading-underscore-private symbol; never in `__all__`, never documented as public surface.

### Closed

- TODO A9 — `atomic/vaep/features.py` per-concern coupling — closed via framework extraction (the trigger-condition resolution from PR-S15's deferral). See ADR-002.

### Notes

- Hyrum's-Law surface: `gamestates.__module__` (and `simple.__module__`) flips from `silly_kicks.vaep.features.core` to `silly_kicks.vaep.feature_framework`. Consumers introspecting via `inspect.getmodule(...)` would see the new value. Documented in ADR-002 as accepted exposure.
- Test count: 881 → 884 passing (+3 net: +1 T-A, -6 T-B, +7 T-D, +1 Examples gate).

```

(Verify via `git diff CHANGELOG.md` that the 2.4.0 block appears above the 2.3.0 block.)

### Task 18: Bump version

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 18.1: Find the version line**

Run:
```bash
uv run python -c "
from pathlib import Path
for i, line in enumerate(Path('pyproject.toml').read_text(encoding='utf-8').splitlines(), 1):
    if 'version' in line.lower() and '=' in line:
        print(f'{i}: {line}')
"
```
Note the line and the current version string (expected: `version = "2.3.0"`).

- [ ] **Step 18.2: Apply the version bump**

Edit `pyproject.toml`:

```diff
-version = "2.3.0"
+version = "2.4.0"
```

- [ ] **Step 18.3: Verify the import-time version reflects the bump**

Run:
```bash
uv run python -c "
import importlib.metadata
print('silly-kicks version:', importlib.metadata.version('silly-kicks'))
"
```
Expected output: `silly-kicks version: 2.4.0`.

(If the installed metadata still shows 2.3.0 because the package was installed in editable mode from a previous version, that's fine — the source-of-truth is `pyproject.toml`. CI installs fresh.)

---

## Phase 11 — Memory updates

### Task 19: Refresh memory files

**Files:**
- Modify: `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\project_release_state.md`
- Modify: `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\project_followup_prs.md`
- Modify (likely): `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\MEMORY.md` (the index — update one-line hooks if relevant)

- [ ] **Step 19.1: Read current state of `project_release_state.md`**

Run:
```bash
uv run python -c "
from pathlib import Path
p = Path.home() / '.claude' / 'projects' / 'D--Development-karstenskyt--silly-kicks' / 'memory' / 'project_release_state.md'
print(p.read_text(encoding='utf-8'))
"
```
Note the current version (expected: 2.3.0) and the trajectory list. The PR-S16 entry needs to be added; the "current version" updated to 2.4.0.

- [ ] **Step 19.2: Update `project_release_state.md`**

Update the file:
- Bump current version line (e.g. `**Current version:** 2.4.0` if that field exists).
- Add a 2.4.0 trajectory entry: "2.4.0 (PR-S16) — VAEP feature framework extracted to `silly_kicks.vaep.feature_framework`. Closes A9 (the last open Architecture TODO). `_actiontype` promoted to public `actiontype_categorical(actions, spadl_cfg)`. ADR-002 captures the decision."
- Update the description front-matter to reflect 2.4.0 if the front-matter mentions a version.

- [ ] **Step 19.3: Update `project_followup_prs.md`**

The current memory says "PR-S9..S15 all SHIPPED ... No follow-up queued."

Update to:
- Add PR-S16 to the SHIPPED list with the 2.4.0 + A9-closure note.
- The "No follow-up queued" line stays — A9 was the last open TODO, and this PR closes it.
- Optionally add a one-line victory note: "TODO.md Architecture/Documentation/Tech Debt sections all empty as of 2.4.0."

- [ ] **Step 19.4: Update `MEMORY.md` index**

Run:
```bash
uv run python -c "
from pathlib import Path
p = Path.home() / '.claude' / 'projects' / 'D--Development-karstenskyt--silly-kicks' / 'memory' / 'MEMORY.md'
for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    print(f'{i}: {line}')
"
```
Look at the existing one-liner for `[Release state]`, `[Follow-up PRs]`, `[ADR pattern]`. Update each one-liner if the new info changes the hook (e.g. "current version (2.4.0, PR-S16 ...)"). Keep each line ≤ 150 characters.

---

## Phase 12 — Final review

### Task 20: Run /final-review and address findings

**Files:**
- (none — review pass)

- [ ] **Step 20.1: Run /final-review**

Type `/final-review` in the Claude Code session (or otherwise invoke the skill). The skill will pre-commit audit code + docs + diagram. Address any findings inline.

If no findings, proceed. If findings: STOP, fix, re-run /final-review until clean. Do NOT commit until /final-review is clean.

- [ ] **Step 20.2: Run the full verification gate sequence one more time**

Run:
```bash
uv run ruff check silly_kicks/ tests/ && \
uv run ruff format --check silly_kicks/ tests/ && \
uv run pyright silly_kicks/ && \
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: all four pass; pytest shows `884 passed, 4 deselected` (or `<baseline + 3>`).

---

## Phase 13 — Commit + ship

### Task 21: Single-commit gate (USER APPROVAL REQUIRED)

**Files:**
- (none — git operation)

- [ ] **Step 21.1: Show user the file list and diff scope**

Run (in parallel):
```bash
git status --short
git diff --stat
```

Show the output to the user. Wait for explicit user approval ("yes commit", "approved", "proceed with commit") before continuing.

The expected file list:
- New: `silly_kicks/vaep/feature_framework.py`
- New: `tests/vaep/test_feature_framework_layout.py`
- New: `docs/superpowers/specs/2026-04-30-A9-feature-framework-extraction-design.md`
- New: `docs/superpowers/plans/2026-04-30-A9-feature-framework-extraction.md`
- New: `docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md`
- Modified: `silly_kicks/vaep/features/core.py`
- Modified: `silly_kicks/vaep/features/__init__.py`
- Modified: `silly_kicks/vaep/features/actiontype.py`
- Modified: `silly_kicks/atomic/vaep/features.py`
- Modified: `tests/vaep/test_features_backcompat.py`
- Modified: `tests/vaep/test_features_submodule_layout.py`
- Modified: `tests/atomic/test_features_per_concern_import.py`
- Modified: `tests/test_public_api_examples.py`
- Modified: `TODO.md`
- Modified: `CHANGELOG.md`
- Modified: `pyproject.toml`
- (Memory files outside the repo — track via memory tooling, not git.)

If user requests changes, address them, re-run verification gates, and re-request approval.

- [ ] **Step 21.2: Stage all files and commit (user-approved only)**

Once approved, run:
```bash
git add silly_kicks/vaep/feature_framework.py
git add tests/vaep/test_feature_framework_layout.py
git add docs/superpowers/specs/2026-04-30-A9-feature-framework-extraction-design.md
git add docs/superpowers/plans/2026-04-30-A9-feature-framework-extraction.md
git add docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md
git add silly_kicks/vaep/features/core.py
git add silly_kicks/vaep/features/__init__.py
git add silly_kicks/vaep/features/actiontype.py
git add silly_kicks/atomic/vaep/features.py
git add tests/vaep/test_features_backcompat.py
git add tests/vaep/test_features_submodule_layout.py
git add tests/atomic/test_features_per_concern_import.py
git add tests/test_public_api_examples.py
git add TODO.md
git add CHANGELOG.md
git add pyproject.toml
```

Then commit with this message (use HEREDOC for formatting):
```bash
git commit -m "$(cat <<'EOF'
feat(vaep)!: extract VAEP feature framework to silly_kicks.vaep.feature_framework — silly-kicks 2.4.0 (PR-S16)

Closes TODO A9 — the last open Architecture row.

The 7 framework primitives shared across standard-VAEP and atomic-VAEP
feature stacks (4 type aliases + gamestates + simple + the promoted
actiontype_categorical helper) move to a new public module
silly_kicks.vaep.feature_framework. Atomic-VAEP imports framework
primitives directly from there (no longer reaches into
silly_kicks.vaep.features.core for cross-package private symbols).
silly_kicks.vaep.features.core slims to its standard-SPADL-specific
helpers (play_left_to_right, feature_column_names) and re-exports the
framework primitives so existing import paths continue to resolve
(Hyrum's Law preservation).

The previously-private _actiontype is promoted to public
actiontype_categorical(actions, spadl_cfg) — drops the leading
underscore (cross-package use deserves a real public name), tightens
the contract (positional spadl_cfg, no implicit None fallback), and
gains an Examples docstring per the public-API discipline.

Lock-in tests refreshed: T-A gains one row (actiontype_categorical);
T-B drops the 6 framework rows now living outside the features
package; T-C rewritten to forbid vaep.features.core import for
framework primitives and require vaep.feature_framework; T-D NEW
(7 cases) locks the framework module's symbol layout. Examples gate
adds the new module — 27 cases.

ADR-002 captures the framework-extraction decision, alternatives, and
_actiontype → actiontype_categorical rename rationale.

Test count: 881 → 884 (+3 net).

BREAKING (private-surface): silly_kicks.vaep.features.core._actiontype
removed. Was leading-underscore-private; never in __all__; never
documented as public. Functional replacement:
silly_kicks.vaep.feature_framework.actiontype_categorical(actions, spadl_cfg).

Spec: docs/superpowers/specs/2026-04-30-A9-feature-framework-extraction-design.md
ADR:  docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 21.3: Verify the commit landed cleanly**

Run:
```bash
git log -1 --stat
git status --short
```
Expected: commit shows the 16 file changes; `git status` shows working tree clean (or only the pre-existing untracked files `README.md.backup`, `uv.lock`).

### Task 22: Push, PR, CI watch, merge, tag, verify

**Files:**
- (none — git/GitHub operations)

- [ ] **Step 22.1: Push the branch**

Run:
```bash
git push -u origin feat/A9-feature-framework-extraction
```
Expected: branch published; PR-create URL printed in stderr.

- [ ] **Step 22.2: Open the PR**

Run:
```bash
gh pr create --title "feat(vaep)!: extract VAEP feature framework — silly-kicks 2.4.0 (PR-S16)" --body "$(cat <<'EOF'
## Summary

- Closes TODO A9 (the last open Architecture row) by extracting the 7 cross-package VAEP feature framework primitives into a new public module `silly_kicks.vaep.feature_framework`.
- Promotes the previously-private `_actiontype` helper to public `actiontype_categorical(actions, spadl_cfg)`.
- Refits atomic-VAEP to import framework primitives directly from `vaep.feature_framework`; per-concern coupling to `bodypart` / `context` / `temporal` is preserved (intentional verbatim feature reuse).
- ADR-002 captures the framework-extraction decision and alternatives.

## Test plan

- [ ] T-A backcompat lock: 34 cases (was 33; +1 for `actiontype_categorical`)
- [ ] T-B features-package layout lock: 27 cases (was 33; -6 framework rows now outside the package)
- [ ] T-C atomic-coupling lock: rewritten — forbids `vaep.features.core` for framework primitives, requires `vaep.feature_framework`
- [ ] T-D framework-module layout lock: 7 NEW cases
- [ ] Examples gate: 27 cases (was 26; +1 for the new framework module)
- [ ] Net delta: +3 tests; full pytest expected at 884 passing, 4 deselected
- [ ] ruff / ruff format / pyright: clean

## Hyrum's Law notes

`gamestates.__module__` (and `simple.__module__`) flip from `silly_kicks.vaep.features.core` to `silly_kicks.vaep.feature_framework`. Consumers introspecting via `inspect.getmodule(...)` would see the new value. Documented in ADR-002 as accepted exposure.

`silly_kicks.vaep.features.core._actiontype` removed. Was leading-underscore-private; never in `__all__`; never documented as public. Functional replacement: `actiontype_categorical(actions, spadl_cfg)`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL printed.

- [ ] **Step 22.3: Watch CI**

Run:
```bash
gh pr checks --watch
```
Wait for all CI checks to complete (typically 3-7 minutes). Expected: green across all matrix entries (Python 3.10, 3.11, 3.12).

If CI fails: investigate via `gh run view <run-id> --log-failed`. Common issues for cross-version pyright/ruff: pinned package versions in `uv pip install` block resolve differently in CI; pandas-stubs may differ. Refer to the memory note on cross-version CI for context.

- [ ] **Step 22.4: Squash-merge with --admin override**

Once CI is green, run:
```bash
gh pr merge --squash --admin --delete-branch
```
Expected: PR merged; branch auto-deleted.

- [ ] **Step 22.5: Tag and push the release**

Run:
```bash
git checkout main
git pull
git tag -a v2.4.0 -m "silly-kicks 2.4.0 — VAEP feature framework extraction (PR-S16, closes A9)"
git push origin v2.4.0
```

- [ ] **Step 22.6: Verify PyPI publication**

The repo's release workflow should auto-publish to PyPI on tag push. Verify after ~5 minutes:
```bash
uv run python -c "
import urllib.request, json
with urllib.request.urlopen('https://pypi.org/pypi/silly-kicks/json') as r:
    data = json.load(r)
print('Latest PyPI version:', data['info']['version'])
print('Released:', data['releases'].get('2.4.0', ['(not yet)'])[:1])
"
```
Expected: `Latest PyPI version: 2.4.0`. If `2.4.0` is not yet present, wait 2-3 minutes and re-check (PyPI propagation has small lag).

---

## Self-review checklist (already applied during writing)

- [x] **Spec coverage:** Every section of the spec has a corresponding task. § 1-3 (problem/goals/non-goals) inform the plan but don't need tasks; § 4 architecture → Tasks 8-13; § 5 test plan → Tasks 3-7 + 13; § 6 implementation order → Tasks 8-19; § 7 verification → Task 14; § 8 risks → covered by smoke tests + Hyrum's Law audit in CHANGELOG; § 9 acceptance criteria → all 14 criteria are touched by Tasks 8-19.
- [x] **Placeholder scan:** No "TBD" / "TODO" / "implement later" placeholders. The "TODO" string occurs only in references to TODO.md and the TODO A9 entry.
- [x] **Type consistency:** `actiontype_categorical(actions, spadl_cfg)` signature is consistent across spec, plan, ADR, framework module, and `__all__` entries.
- [x] **Commit policy:** No intermediate commits — single commit gated on user approval at Task 21, per CLAUDE.md memory.
