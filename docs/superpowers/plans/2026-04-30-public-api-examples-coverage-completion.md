# Public-API Examples coverage completion + atomic `coverage_metrics` (PR-S14) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the PR-S13 documentation coverage gap (25 missing Examples sections across 5 public-API module files) and bundle TODO C-1 (atomic `coverage_metrics` parity), shipping as silly-kicks 2.2.0.

**Architecture:** Mechanical, TDD-first extension of the PR-S13 CI guardrail (`tests/test_public_api_examples.py`). The gate's `_PUBLIC_MODULE_FILES` is widened from 14 → 19 to make 5 parametrize cases fail (25 missing surfaces enumerated). Each function then gains an illustrative Examples section in PR-S13 style. C-1 mirrors `silly_kicks.spadl.coverage_metrics` verbatim, **reusing the standard `CoverageMetrics` TypedDict** as the single source of truth (atomic `__init__.py` re-exports both function and type for atomic-side discoverability).

**Tech Stack:** Python 3.10+ (project floor), pandas / numpy, pytest, ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113, uv.

**Commit policy:** Per user's standing rule (memory: `feedback_commit_policy.md`), **literally one commit per branch, gated on explicit user approval at the end**. No WIP commits, no per-task commits. All file changes accumulate on `feat/public-api-examples-coverage-completion` and are squashed into a single commit at Task 11.

**Spec reference:** `docs/superpowers/specs/2026-04-30-public-api-examples-coverage-completion-design.md` (post-edit revision: TypedDict reuse + corrected test class names).

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `tests/test_public_api_examples.py` | Modify | Extend `_PUBLIC_MODULE_FILES` 14 → 19. No other changes. |
| `silly_kicks/vaep/labels.py` | Modify | Add Examples to 5 public functions (`scores`, `concedes`, `goal_from_shot`, `save_from_shot`, `claim_from_cross`). |
| `silly_kicks/vaep/formula.py` | Modify | Add Examples to 3 public functions (`offensive_value`, `defensive_value`, `value`). |
| `silly_kicks/atomic/vaep/labels.py` | Modify | Add Examples to 5 public functions (same names as standard; uses atomic `type_id` directly). |
| `silly_kicks/atomic/vaep/formula.py` | Modify | Add Examples to 3 public functions (uses atomic `type_name` via `add_names`). |
| `silly_kicks/atomic/vaep/features.py` | Modify | Add Examples to 9 public functions (`actiontype`, `feature_column_names`, `play_left_to_right`, `actiontype_onehot`, `location`, `polar`, `movement_polar`, `direction`, `goalscore`). |
| `silly_kicks/atomic/spadl/utils.py` | Modify | Add `coverage_metrics` function reusing standard `CoverageMetrics` TypedDict. |
| `silly_kicks/atomic/spadl/__init__.py` | Modify | Re-export `coverage_metrics` (from `.utils`) and `CoverageMetrics` (from `silly_kicks.spadl.utils`). Update `__all__`. |
| `tests/atomic/test_atomic_coverage_metrics.py` | Create | ~10 tests in 3 classes (`TestAtomicCoverageMetricsContract`, `TestAtomicCoverageMetricsCorrectness`, `TestAtomicCoverageMetricsDegenerate`). |
| `pyproject.toml` | Modify | Version 2.1.1 → 2.2.0. |
| `CHANGELOG.md` | Modify | New `[2.2.0]` entry. |
| `TODO.md` | Modify | Close C-1 entry. |

Total: 1 new test file + 8 modified source files + 3 admin files. ~250 lines added.

**Style anchor:** `silly_kicks/spadl/utils.py:add_possessions` and `silly_kicks/spadl/utils.py:boundary_metrics` are the canonical illustrative-Examples references. 3-7 lines per Example, no doctest verification, no ` >>> ` prefix — just an indented `::` block.

---

### Task 0: Branch setup

**Files:**
- Modify: working tree (branch only, no file changes)

- [ ] **Step 1: Confirm clean main**

```bash
git status --short
```

Expected (only pre-existing untracked):
```
?? README.md.backup
?? docs/superpowers/specs/2026-04-30-public-api-examples-coverage-completion-design.md
?? uv.lock
```

If anything else appears (staged/modified files), STOP and check with the user before proceeding.

- [ ] **Step 2: Create the feature branch**

```bash
git switch -c feat/public-api-examples-coverage-completion
```

Expected: `Switched to a new branch 'feat/public-api-examples-coverage-completion'`.

- [ ] **Step 3: Confirm on the new branch**

```bash
git branch --show-current
```

Expected: `feat/public-api-examples-coverage-completion`.

---

### Task 1: Extend the failing CI gate (TDD red bar)

**Files:**
- Modify: `tests/test_public_api_examples.py:18-33`

- [ ] **Step 1: Replace `_PUBLIC_MODULE_FILES` with the 19-entry tuple**

In `tests/test_public_api_examples.py`, replace the existing 14-entry tuple with:

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
    "silly_kicks/vaep/features.py",
    "silly_kicks/atomic/vaep/base.py",
    "silly_kicks/xthreat.py",
    "silly_kicks/vaep/labels.py",
    "silly_kicks/vaep/formula.py",
    "silly_kicks/atomic/vaep/features.py",
    "silly_kicks/atomic/vaep/labels.py",
    "silly_kicks/atomic/vaep/formula.py",
)
```

The 5 new entries are appended (preserves ordering for review-diff readability). No other changes to the file.

- [ ] **Step 2: Run the gate locally — expect 5 NEW failing parametrize cases**

```bash
uv run pytest tests/test_public_api_examples.py -v --tb=short
```

Expected (truncated):
```
tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/spadl/utils.py] PASSED
... (13 more PASSED — existing 14 entries) ...
tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/vaep/labels.py] FAILED
tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/vaep/formula.py] FAILED
tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/atomic/vaep/features.py] FAILED
tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/atomic/vaep/labels.py] FAILED
tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/atomic/vaep/formula.py] FAILED

= 5 failed, 14 passed in ...
```

The 5 failure messages enumerate 25 missing public surfaces total (5+3+9+5+3). Save the failure output mentally — Tasks 2-6 will close them one file at a time.

If a different number of failures (e.g., 6 instead of 5), STOP — investigate; some symbol may have been added to a target file without an Examples section but in scope of an unrelated PR.

---

### Task 2: Add Examples to `silly_kicks/vaep/labels.py` (5 functions)

**Files:**
- Modify: `silly_kicks/vaep/labels.py`

These functions consume a SPADL DataFrame that has `type_name` / `result_id` / `team_id` (i.e., output of `silly_kicks.spadl.add_names()` on a converter result). The Examples reflect this prerequisite.

- [ ] **Step 1: Add Examples to `scores()` (~line 29)**

Use Edit with `old_string` matching the existing closing block of the `scores` docstring:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.
    """
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)
```

Replace with:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "scores" labels for VAEP training::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import scores

        actions_with_names = add_names(actions)
        y_scores = scores(actions_with_names, nr_actions=10)
        # y_scores["scores"] is bool: True iff the team in possession scores
        # within the next 10 actions.
    """
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)
```

- [ ] **Step 2: Add Examples to `concedes()` (~line 67)**

Match on:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.
    """
    if xg_column is not None:
        return _concedes_xg(actions, nr_actions, xg_column)
```

Replace with:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "concedes" labels (the dual of ``scores``) for VAEP training::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import concedes

        actions_with_names = add_names(actions)
        y_concedes = concedes(actions_with_names, nr_actions=10)
        # y_concedes["concedes"] is bool: True iff the team in possession
        # concedes within the next 10 actions.
    """
    if xg_column is not None:
        return _concedes_xg(actions, nr_actions, xg_column)
```

- [ ] **Step 3: Add Examples to `goal_from_shot()` (~line 141)**

Match on:

```python
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.
    """
    goals = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
```

Replace with:

```python
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.

    Examples
    --------
    Build per-action goal labels for an xG model::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import goal_from_shot

        actions_with_names = add_names(actions)
        y = goal_from_shot(actions_with_names)
        # y["goal_from_shot"] is True only on shot rows that resulted in a goal.
    """
    goals = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
```

- [ ] **Step 4: Add Examples to `save_from_shot()` (~line 162)**

Match on:

```python
        A dataframe with a column 'save_from_shot' and a row for each action
        set to True if the action is a keeper save; otherwise False.
    """
    saves = actions["type_name"].str.contains("keeper_save") & (actions["result_id"] == spadl.result_id["success"])
```

Replace with:

```python
        A dataframe with a column 'save_from_shot' and a row for each action
        set to True if the action is a keeper save; otherwise False.

    Examples
    --------
    Build per-action keeper-save labels for an Expected Saves (xS) model::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import save_from_shot

        actions_with_names = add_names(actions)
        y = save_from_shot(actions_with_names)
        # y["save_from_shot"] is True only on successful keeper_save rows.
    """
    saves = actions["type_name"].str.contains("keeper_save") & (actions["result_id"] == spadl.result_id["success"])
```

- [ ] **Step 5: Add Examples to `claim_from_cross()` (~line 183)**

Match on:

```python
        A dataframe with a column 'claim_from_cross' and a row for each action
        set to True if the action is a keeper claim; otherwise False.
    """
    claims = actions["type_name"].str.contains("keeper_claim") & (actions["result_id"] == spadl.result_id["success"])
```

Replace with:

```python
        A dataframe with a column 'claim_from_cross' and a row for each action
        set to True if the action is a keeper claim; otherwise False.

    Examples
    --------
    Build per-action keeper-claim labels for an Expected Claims (xC) model::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import claim_from_cross

        actions_with_names = add_names(actions)
        y = claim_from_cross(actions_with_names)
        # y["claim_from_cross"] is True only on successful keeper_claim rows.
    """
    claims = actions["type_name"].str.contains("keeper_claim") & (actions["result_id"] == spadl.result_id["success"])
```

- [ ] **Step 6: Verify the `vaep/labels.py` parametrize case now passes**

```bash
uv run pytest "tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/vaep/labels.py]" -v
```

Expected: PASS.

If it fails, the error message lists which symbol is still missing — re-check that the `Examples` header is present and indented correctly (4 spaces, NumPy underline `--------`).

---

### Task 3: Add Examples to `silly_kicks/vaep/formula.py` (3 functions)

**Files:**
- Modify: `silly_kicks/vaep/formula.py`

These functions consume actions with `type_name` / `result_name` (i.e., output of `add_names()`) plus a pair of probability series. Naming convention `p_scores` / `p_concedes` distinguishes the probability inputs from the `scores` / `concedes` label-generator functions.

- [ ] **Step 1: Add Examples to `offensive_value()` (~line 44)**

Match on:

```python
    Returns
    -------
    pd.Series
        The offensive value of each action.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = (_prev(scores) * sameteam + _prev(concedes) * (~sameteam)).astype(float)
```

Replace with:

```python
    Returns
    -------
    pd.Series
        The offensive value of each action.

    Examples
    --------
    Compute the offensive component of VAEP from estimated probabilities::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.formula import offensive_value

        actions_with_names = add_names(actions)
        # p_scores, p_concedes: pd.Series, one row per action, e.g. from
        # VAEP._estimate_probabilities or any binary classifier of choice.
        ov = offensive_value(actions_with_names, p_scores, p_concedes)
        # Returns a pd.Series of per-action offensive values.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = (_prev(scores) * sameteam + _prev(concedes) * (~sameteam)).astype(float)
```

- [ ] **Step 2: Add Examples to `defensive_value()` (~line 96)**

Match on:

```python
    Returns
    -------
    pd.Series
        The defensive value of each action.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = (_prev(concedes) * sameteam + _prev(scores) * (~sameteam)).astype(float)
```

Replace with:

```python
    Returns
    -------
    pd.Series
        The defensive value of each action.

    Examples
    --------
    Compute the defensive component of VAEP from estimated probabilities::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.formula import defensive_value

        actions_with_names = add_names(actions)
        dv = defensive_value(actions_with_names, p_scores, p_concedes)
        # Returns a pd.Series of per-action defensive values (sign convention:
        # a successful defensive action that lowers conceding probability is
        # positive).
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = (_prev(concedes) * sameteam + _prev(scores) * (~sameteam)).astype(float)
```

- [ ] **Step 3: Add Examples to `value()` (~line 140)**

Match on:

```python
    See Also
    --------
    :func:`~silly_kicks.vaep.formula.offensive_value`: The offensive value
    :func:`~silly_kicks.vaep.formula.defensive_value`: The defensive value
    """
    v = pd.DataFrame()
```

Replace with:

```python
    See Also
    --------
    :func:`~silly_kicks.vaep.formula.offensive_value`: The offensive value
    :func:`~silly_kicks.vaep.formula.defensive_value`: The defensive value

    Examples
    --------
    Compute per-action VAEP values directly from probabilities (without going
    through ``VAEP.rate()``)::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.formula import value

        actions_with_names = add_names(actions)
        v = value(actions_with_names, p_scores, p_concedes)
        # v has columns 'offensive_value', 'defensive_value', 'vaep_value';
        # vaep_value = offensive_value + defensive_value per row.
    """
    v = pd.DataFrame()
```

- [ ] **Step 4: Verify the `vaep/formula.py` parametrize case now passes**

```bash
uv run pytest "tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/vaep/formula.py]" -v
```

Expected: PASS.

---

### Task 4: Add Examples to `silly_kicks/atomic/vaep/labels.py` (5 functions)

**Files:**
- Modify: `silly_kicks/atomic/vaep/labels.py`

**Atomic-specific:** these functions consume `type_id` directly (NOT `type_name`), so `add_names()` is NOT a prerequisite. The fixture is simply an atomic-SPADL DataFrame from `convert_to_atomic`.

- [ ] **Step 1: Add Examples to `scores()` (~line 29)**

Match on:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.
    """
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)
```

Replace with:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "scores" labels on an atomic-SPADL stream for VAEP training::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import scores

        atomic = convert_to_atomic(actions)
        y_scores = scores(atomic, nr_actions=10)
        # y_scores["scores"] is bool: True iff the team in possession scores
        # within the next 10 atomic actions.
    """
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)
```

- [ ] **Step 2: Add Examples to `concedes()` (~line 67)**

Match on:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.
    """
    if xg_column is not None:
        return _concedes_xg(actions, nr_actions, xg_column)
```

Replace with:

```python
        the next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "concedes" labels on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import concedes

        atomic = convert_to_atomic(actions)
        y_concedes = concedes(atomic, nr_actions=10)
        # y_concedes["concedes"] is bool: True iff the team in possession
        # concedes within the next 10 atomic actions.
    """
    if xg_column is not None:
        return _concedes_xg(actions, nr_actions, xg_column)
```

- [ ] **Step 3: Add Examples to `goal_from_shot()` (~line 141)**

Match on:

```python
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.
    """
    goals = (actions["type_id"] == atomicspadl.actiontype_id["shot"]) & (
```

Replace with:

```python
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.

    Examples
    --------
    Build per-action goal labels on an atomic-SPADL stream for an xG model.
    Atomic decomposes a successful shot into a ``shot`` row immediately
    followed by a ``goal`` row, so the label is True only on the ``shot``
    row whose successor is ``goal``::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import goal_from_shot

        atomic = convert_to_atomic(actions)
        y = goal_from_shot(atomic)
        # y["goal"] is True only on shot rows followed by a goal row.
    """
    goals = (actions["type_id"] == atomicspadl.actiontype_id["shot"]) & (
```

- [ ] **Step 4: Add Examples to `save_from_shot()` (~line 164)**

Match on:

```python
        A dataframe with a column 'save_from_shot' and a row for each action
        set to True if the action is a keeper save; otherwise False.
    """
    saves = actions["type_id"] == atomicspadl.actiontype_id["keeper_save"]
```

Replace with:

```python
        A dataframe with a column 'save_from_shot' and a row for each action
        set to True if the action is a keeper save; otherwise False.

    Examples
    --------
    Build per-action keeper-save labels on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import save_from_shot

        atomic = convert_to_atomic(actions)
        y = save_from_shot(atomic)
        # y["save_from_shot"] is True on every keeper_save row.
    """
    saves = actions["type_id"] == atomicspadl.actiontype_id["keeper_save"]
```

- [ ] **Step 5: Add Examples to `claim_from_cross()` (~line 184)**

Match on:

```python
        A dataframe with a column 'claim_from_cross' and a row for each action
        set to True if the action is a keeper claim; otherwise False.
    """
    claims = actions["type_id"] == atomicspadl.actiontype_id["keeper_claim"]
```

Replace with:

```python
        A dataframe with a column 'claim_from_cross' and a row for each action
        set to True if the action is a keeper claim; otherwise False.

    Examples
    --------
    Build per-action keeper-claim labels on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import claim_from_cross

        atomic = convert_to_atomic(actions)
        y = claim_from_cross(atomic)
        # y["claim_from_cross"] is True on every keeper_claim row.
    """
    claims = actions["type_id"] == atomicspadl.actiontype_id["keeper_claim"]
```

- [ ] **Step 6: Verify the `atomic/vaep/labels.py` parametrize case now passes**

```bash
uv run pytest "tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/atomic/vaep/labels.py]" -v
```

Expected: PASS.

---

### Task 5: Add Examples to `silly_kicks/atomic/vaep/formula.py` (3 functions)

**Files:**
- Modify: `silly_kicks/atomic/vaep/formula.py`

**Atomic-specific:** these functions DO consume `type_name` (line 50: `_prev(actions.type_name).isin(["goal", "owngoal"])`), so `add_names()` from `silly_kicks.atomic.spadl` IS a prerequisite.

- [ ] **Step 1: Add Examples to `offensive_value()` (~line 39)**

Match on:

```python
    Returns
    -------
    pd.Series
        The offensive value of each action.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = _prev(scores) * sameteam + _prev(concedes) * (~sameteam)
```

Replace with:

```python
    Returns
    -------
    pd.Series
        The offensive value of each action.

    Examples
    --------
    Compute the offensive component of Atomic-VAEP from probabilities::

        from silly_kicks.atomic.spadl import add_names, convert_to_atomic
        from silly_kicks.atomic.vaep.formula import offensive_value

        atomic = add_names(convert_to_atomic(actions))
        # p_scores, p_concedes: pd.Series, one row per atomic action.
        ov = offensive_value(atomic, p_scores, p_concedes)
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = _prev(scores) * sameteam + _prev(concedes) * (~sameteam)
```

- [ ] **Step 2: Add Examples to `defensive_value()` (~line 83)**

Match on:

```python
    Returns
    -------
    pd.Series
        The defensive value of each action.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = _prev(concedes) * sameteam + _prev(scores) * (~sameteam)
```

Replace with:

```python
    Returns
    -------
    pd.Series
        The defensive value of each action.

    Examples
    --------
    Compute the defensive component of Atomic-VAEP from probabilities::

        from silly_kicks.atomic.spadl import add_names, convert_to_atomic
        from silly_kicks.atomic.vaep.formula import defensive_value

        atomic = add_names(convert_to_atomic(actions))
        dv = defensive_value(atomic, p_scores, p_concedes)
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = _prev(concedes) * sameteam + _prev(scores) * (~sameteam)
```

- [ ] **Step 3: Add Examples to `value()` (~line 128)**

Match on:

```python
    See Also
    --------
    :func:`~silly_kicks.vaep.formula.offensive_value`: The offensive value
    :func:`~silly_kicks.vaep.formula.defensive_value`: The defensive value
    """
    v = pd.DataFrame()
```

Replace with:

```python
    See Also
    --------
    :func:`~silly_kicks.vaep.formula.offensive_value`: The offensive value
    :func:`~silly_kicks.vaep.formula.defensive_value`: The defensive value

    Examples
    --------
    Compute per-action Atomic-VAEP value DataFrame from probabilities::

        from silly_kicks.atomic.spadl import add_names, convert_to_atomic
        from silly_kicks.atomic.vaep.formula import value

        atomic = add_names(convert_to_atomic(actions))
        v = value(atomic, p_scores, p_concedes)
        # v has columns 'offensive_value', 'defensive_value', 'vaep_value'.
    """
    v = pd.DataFrame()
```

- [ ] **Step 4: Verify the `atomic/vaep/formula.py` parametrize case now passes**

```bash
uv run pytest "tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/atomic/vaep/formula.py]" -v
```

Expected: PASS.

---

### Task 6: Add Examples to `silly_kicks/atomic/vaep/features.py` (9 functions)

**Files:**
- Modify: `silly_kicks/atomic/vaep/features.py`

**Atomic-specific:** all `@simple`-decorated transformers (`actiontype`, `actiontype_onehot`, `location`, `polar`, `movement_polar`, `direction`) take `Actions` per-row but are typically applied via `gamestates()` (re-exported from `silly_kicks.vaep.features` — see line 15). `play_left_to_right` and `goalscore` consume `GameStates`. `feature_column_names` introspects a list of transformers.

The `gamestates(atomic, nb_prev_actions=3)` motif from spec §4.3 is the standard fixture motif.

- [ ] **Step 1: Add Examples to `actiontype()` (~line 54)**

Match on:

```python
@simple
def actiontype(actions: Actions) -> Features:
    """Actiontype feature using atomic SPADL config (33 action types)."""
    return _actiontype(actions, _spadl_cfg=atomicspadl)
```

Replace with:

```python
@simple
def actiontype(actions: Actions) -> Features:
    """Actiontype feature using atomic SPADL config (33 action types).

    Examples
    --------
    Extract the atomic action-type integer feature per gamestate slot::

        from silly_kicks.atomic.vaep.features import actiontype, gamestates

        states = gamestates(atomic, nb_prev_actions=3)
        feats = actiontype(states)
        # feats has columns 'type_id_a0', 'type_id_a1', 'type_id_a2'.
    """
    return _actiontype(actions, _spadl_cfg=atomicspadl)
```

- [ ] **Step 2: Add Examples to `feature_column_names()` (~line 72)**

Match on:

```python
    Returns
    -------
    list(str)
        The name of each generated feature.
    """
    spadlcolumns = [
```

Replace with:

```python
    Returns
    -------
    list(str)
        The name of each generated feature.

    Examples
    --------
    Enumerate the column names a feature stack will produce, without
    actually computing features on real data::

        from silly_kicks.atomic.vaep.features import (
            actiontype_onehot, location, feature_column_names,
        )

        names = feature_column_names([actiontype_onehot, location], nb_prev_actions=3)
        # names is e.g. ['actiontype_pass_a0', ..., 'x_a0', 'y_a0', ...].
    """
    spadlcolumns = [
```

- [ ] **Step 3: Add Examples to `play_left_to_right()` (~line 114)**

Match on:

```python
    Returns
    -------
    list(pd.DataFrame)
        The game states with all actions performed left to right.
    """
    a0 = gamestates[0]
```

Replace with:

```python
    Returns
    -------
    list(pd.DataFrame)
        The game states with all actions performed left to right.

    Examples
    --------
    Mirror gamestates so all actions are performed left-to-right (per home team)::

        from silly_kicks.atomic.vaep.features import gamestates, play_left_to_right

        states = gamestates(atomic, nb_prev_actions=3)
        states = play_left_to_right(states, home_team_id=100)
        # All away-team rows in each slot now have flipped (x, y) and (dx, dy).
    """
    a0 = gamestates[0]
```

- [ ] **Step 4: Add Examples to `actiontype_onehot()` (~line 138)**

Match on:

```python
    Returns
    -------
    Features
        A one-hot encoding of each action's type.
    """
    X = {}
```

Replace with:

```python
    Returns
    -------
    Features
        A one-hot encoding of each action's type.

    Examples
    --------
    One-hot encode atomic action types per gamestate slot::

        from silly_kicks.atomic.vaep.features import actiontype_onehot, gamestates

        states = gamestates(atomic, nb_prev_actions=3)
        feats = actiontype_onehot(states)
        # feats has 33 boolean columns per slot (one per atomic action type).
    """
    X = {}
```

- [ ] **Step 5: Add Examples to `location()` (~line 160)**

Match on:

```python
    Returns
    -------
    Features
        The 'x' and 'y' location of each action.
    """
    return actions[["x", "y"]]  # type: ignore[reportReturnType]
```

Replace with:

```python
    Returns
    -------
    Features
        The 'x' and 'y' location of each action.

    Examples
    --------
    Extract atomic location features per gamestate slot::

        from silly_kicks.atomic.vaep.features import gamestates, location

        states = gamestates(atomic, nb_prev_actions=3)
        feats = location(states)
        # feats has columns 'x_a0', 'y_a0', 'x_a1', 'y_a1', 'x_a2', 'y_a2'.
    """
    return actions[["x", "y"]]  # type: ignore[reportReturnType]
```

- [ ] **Step 6: Add Examples to `polar()` (~line 183)**

Match on:

```python
    Returns
    -------
    Features
        The 'dist_to_goal' and 'angle_to_goal' of each action.
    """
    polardf = pd.DataFrame(index=actions.index)
```

Replace with:

```python
    Returns
    -------
    Features
        The 'dist_to_goal' and 'angle_to_goal' of each action.

    Examples
    --------
    Compute polar features (distance + angle to opponent goal) per slot::

        from silly_kicks.atomic.vaep.features import gamestates, polar

        states = gamestates(atomic, nb_prev_actions=3)
        feats = polar(states)
        # feats has columns 'dist_to_goal_a0', 'angle_to_goal_a0', etc.
    """
    polardf = pd.DataFrame(index=actions.index)
```

- [ ] **Step 7: Add Examples to `movement_polar()` (~line 206)**

Match on:

```python
    Returns
    -------
    Features
        The distance covered ('mov_d') and direction ('mov_angle') of each action.
    """
    mov = pd.DataFrame(index=actions.index)
    mov["mov_d"] = np.sqrt(actions.dx**2 + actions.dy**2)
```

Replace with:

```python
    Returns
    -------
    Features
        The distance covered ('mov_d') and direction ('mov_angle') of each action.

    Examples
    --------
    Compute per-action movement magnitude + angle features per slot::

        from silly_kicks.atomic.vaep.features import gamestates, movement_polar

        states = gamestates(atomic, nb_prev_actions=3)
        feats = movement_polar(states)
        # feats has 'mov_d_a0' (metres) and 'mov_angle_a0' (radians) per slot.
    """
    mov = pd.DataFrame(index=actions.index)
    mov["mov_d"] = np.sqrt(actions.dx**2 + actions.dy**2)
```

- [ ] **Step 8: Add Examples to `direction()` (~line 229)**

Match on:

```python
    Returns
    -------
    Features
        The x-component ('dx') and y-compoment ('mov_angle') of the unit
        vector of each action.
    """
    mov = pd.DataFrame(index=actions.index)
    totald = np.sqrt(actions.dx**2 + actions.dy**2)
```

Replace with:

```python
    Returns
    -------
    Features
        The x-component ('dx') and y-compoment ('mov_angle') of the unit
        vector of each action.

    Examples
    --------
    Compute per-action unit-direction vector features per slot::

        from silly_kicks.atomic.vaep.features import direction, gamestates

        states = gamestates(atomic, nb_prev_actions=3)
        feats = direction(states)
        # feats has 'dx_a0' and 'dy_a0' (unit-length components, 0 if static).
    """
    mov = pd.DataFrame(index=actions.index)
    totald = np.sqrt(actions.dx**2 + actions.dy**2)
```

- [ ] **Step 9: Add Examples to `goalscore()` (~line 254)**

Match on:

```python
    Returns
    -------
    Features
        The number of goals scored by the team performing the last action of the
        game state ('goalscore_team'), by the opponent ('goalscore_opponent'),
        and the goal difference between both teams ('goalscore_diff').
    """
    actions = gamestates[0]
```

Replace with:

```python
    Returns
    -------
    Features
        The number of goals scored by the team performing the last action of the
        game state ('goalscore_team'), by the opponent ('goalscore_opponent'),
        and the goal difference between both teams ('goalscore_diff').

    Examples
    --------
    Compute the cumulative-goalscore context feature on a gamestate sequence::

        from silly_kicks.atomic.vaep.features import gamestates, goalscore

        states = gamestates(atomic, nb_prev_actions=3)
        feats = goalscore(states)
        # feats has 'goalscore_team', 'goalscore_opponent', 'goalscore_diff'.
    """
    actions = gamestates[0]
```

- [ ] **Step 10: Verify the `atomic/vaep/features.py` parametrize case now passes**

```bash
uv run pytest "tests/test_public_api_examples.py::test_public_definitions_have_examples_section[silly_kicks/atomic/vaep/features.py]" -v
```

Expected: PASS.

- [ ] **Step 11: Run the full gate to confirm all 19 cases now pass (no regressions)**

```bash
uv run pytest tests/test_public_api_examples.py -v
```

Expected: `19 passed in ...`. Zero failures.

---

### Task 7: Atomic `coverage_metrics` (closes TODO C-1) — TDD

**Files:**
- Modify: `silly_kicks/atomic/spadl/utils.py`
- Modify: `silly_kicks/atomic/spadl/__init__.py`
- Create: `tests/atomic/test_atomic_coverage_metrics.py`

The function mirrors `silly_kicks.spadl.coverage_metrics` exactly. The atomic version:
- **Reuses** the standard `CoverageMetrics` TypedDict (from `silly_kicks.spadl.utils`) — single source of truth.
- Resolves `type_id` against `silly_kicks.atomic.spadl.config.actiontypes` (atomic vocabulary, 33 types).
- Same behavior on unknown `type_id` (reports as `"unknown"`).
- Same behavior on unknown names in `expected_action_types` — reports as missing, does NOT raise. (Verified by re-reading standard's `tests/spadl/test_coverage_metrics.py:79` — `expected_action_types` is NOT validated against the vocabulary.)

The standard test classes are: `TestCoverageMetricsContract`, `TestCoverageMetricsCorrectness`, `TestCoverageMetricsDegenerate` (3 classes, ~12 tests). Atomic version mirrors with `TestAtomicCoverageMetrics*` prefix.

- [ ] **Step 1: Write the failing tests**

Create `tests/atomic/test_atomic_coverage_metrics.py` with this exact content:

```python
"""Tests for ``silly_kicks.atomic.spadl.coverage_metrics`` (added in 2.2.0).

Mirrors ``tests/spadl/test_coverage_metrics.py`` (PR-S6 / 1.10.0). Coverage
is measured on an atomic-SPADL action stream, resolving ``type_id`` to
action-type name via ``atomicspadl.actiontypes_df`` and reporting per-type
counts plus any ``expected_action_types`` that produced zero rows. The
``CoverageMetrics`` TypedDict is the standard one (single source of truth);
the atomic-side ``coverage_metrics`` function uses the atomic-vocabulary
33-type alphabet for resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl import config as atomicspadl
from silly_kicks.atomic.spadl.utils import coverage_metrics
from silly_kicks.spadl.utils import CoverageMetrics

_ACT = atomicspadl.actiontype_id


def _df_one(action_type: str = "pass") -> pd.DataFrame:
    """Single-row atomic-SPADL-shaped DataFrame with one action of the given type."""
    return pd.DataFrame({"type_id": [_ACT[action_type]]})


def _df_many(action_types: list[str]) -> pd.DataFrame:
    """Atomic-SPADL-shaped DataFrame with one row per supplied action type name."""
    return pd.DataFrame({"type_id": [_ACT[t] for t in action_types]})


class TestAtomicCoverageMetricsContract:
    def test_returns_dict_with_required_keys(self):
        m = coverage_metrics(actions=_df_one("pass"))
        assert set(m.keys()) == {"counts", "missing", "total_actions"}

    def test_counts_value_type_is_int(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "shot"]))
        for v in m["counts"].values():
            assert isinstance(v, int)

    def test_total_actions_is_int(self):
        m = coverage_metrics(actions=_df_many(["pass", "shot"]))
        assert isinstance(m["total_actions"], int)
        assert m["total_actions"] == 2

    def test_keyword_only_args_required(self):
        with pytest.raises(TypeError):
            coverage_metrics(_df_one("pass"))  # type: ignore[misc]

    def test_missing_type_id_column_raises_value_error(self):
        actions = pd.DataFrame({"team_id": [1, 2]})
        with pytest.raises(ValueError, match=r"type_id"):
            coverage_metrics(actions=actions)

    def test_returns_typeddict_shape(self):
        # The atomic coverage_metrics returns the standard CoverageMetrics
        # TypedDict (single source of truth — re-exported from
        # silly_kicks.spadl.utils).
        m: CoverageMetrics = coverage_metrics(actions=_df_one("pass"))
        assert m["counts"]["pass"] == 1


class TestAtomicCoverageMetricsCorrectness:
    def test_atomic_only_action_type_counted(self):
        # 'receival' is atomic-only — not present in standard SPADL vocabulary.
        m = coverage_metrics(actions=_df_many(["pass", "receival", "receival"]))
        assert m["counts"] == {"pass": 1, "receival": 2}
        assert m["total_actions"] == 3

    def test_collapsed_freekick_name_counted(self):
        # Atomic collapses freekick_short / freekick_crossed → 'freekick'.
        # Verify the post-collapse name resolves correctly.
        m = coverage_metrics(actions=_df_many(["freekick", "freekick", "shot"]))
        assert m["counts"] == {"freekick": 2, "shot": 1}

    def test_collapsed_corner_name_counted(self):
        # Atomic collapses corner_short / corner_crossed → 'corner'.
        m = coverage_metrics(actions=_df_many(["corner", "corner"]))
        assert m["counts"] == {"corner": 2}

    def test_expected_partially_absent_missing_sorted(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot"]),
            expected_action_types={"pass", "shot", "tackle", "keeper_save"},
        )
        assert m["missing"] == ["keeper_save", "tackle"]

    def test_unknown_type_id_reported_as_unknown(self):
        # type_id = 999 is not in the atomic actiontype_id reverse map.
        actions = pd.DataFrame({"type_id": [_ACT["pass"], 999, 999]})
        m = coverage_metrics(actions=actions)
        assert m["counts"].get("pass") == 1
        assert m["counts"].get("unknown") == 2


class TestAtomicCoverageMetricsDegenerate:
    def test_empty_dataframe_returns_zeros(self):
        actions = pd.DataFrame({"type_id": pd.Series([], dtype=np.int64)})
        m = coverage_metrics(actions=actions)
        assert m["counts"] == {}
        assert m["missing"] == []
        assert m["total_actions"] == 0

    def test_empty_with_expected_returns_all_missing_sorted(self):
        actions = pd.DataFrame({"type_id": pd.Series([], dtype=np.int64)})
        m = coverage_metrics(
            actions=actions,
            expected_action_types={"pass", "shot", "receival"},
        )
        assert m["counts"] == {}
        assert m["missing"] == ["pass", "receival", "shot"]
        assert m["total_actions"] == 0

    def test_does_not_mutate_input(self):
        actions = _df_many(["pass", "receival"])
        cols_before = list(actions.columns)
        len_before = len(actions)
        coverage_metrics(actions=actions, expected_action_types={"pass"})
        assert list(actions.columns) == cols_before
        assert len(actions) == len_before
```

The test count is 12 (4 contract + 5 correctness + 3 degenerate).

- [ ] **Step 2: Verify the tests fail (no implementation yet)**

```bash
uv run pytest tests/atomic/test_atomic_coverage_metrics.py -v
```

Expected: All 12 tests fail at collection or with `ImportError: cannot import name 'coverage_metrics' from 'silly_kicks.atomic.spadl.utils'`.

- [ ] **Step 3: Implement `coverage_metrics` in `silly_kicks/atomic/spadl/utils.py`**

Use Edit to add the new function. Match on the very last function in the file (after `play_left_to_right`'s closing brace). The cleanest anchor is the trailing newline at end-of-file.

First, find the current EOF of `silly_kicks/atomic/spadl/utils.py`. The last function is `play_left_to_right`, ending at:

```python
    ltr_actions.loc[away_idx, "dy"] = -actions[away_idx]["dy"].values  # type: ignore[reportAttributeAccessIssue]
    return ltr_actions
```

Use Edit with `old_string`:

```python
    ltr_actions.loc[away_idx, "dy"] = -actions[away_idx]["dy"].values  # type: ignore[reportAttributeAccessIssue]
    return ltr_actions
```

Replace with (appends new function below):

```python
    ltr_actions.loc[away_idx, "dy"] = -actions[away_idx]["dy"].values  # type: ignore[reportAttributeAccessIssue]
    return ltr_actions


def coverage_metrics(
    *,
    actions: pd.DataFrame,
    expected_action_types: set[str] | None = None,
) -> "CoverageMetrics":
    """Compute Atomic-SPADL action-type coverage for an atomic action DataFrame.

    Atomic-SPADL counterpart to :func:`silly_kicks.spadl.utils.coverage_metrics`.
    Resolves ``type_id`` to action-type name via the **atomic** vocabulary
    (:func:`silly_kicks.atomic.spadl.config.actiontypes_df` — 33 types) and
    counts each action type present. When ``expected_action_types`` is
    provided, returns any of those types with zero rows under ``missing``.

    The return type is the standard :class:`silly_kicks.spadl.utils.CoverageMetrics`
    TypedDict — single source of truth across standard and atomic.

    Use cases:

    1. Test discipline — assert atomic converter X emits action types Y on a
       fixture (parity with standard's cross-provider coverage gate).
    2. Downstream validation — consumers calling silly-kicks-converted bronze
       atomic data can verify expected coverage before downstream aggregation.

    Parameters
    ----------
    actions : pd.DataFrame
        Atomic-SPADL action stream. Must contain ``type_id``.
    expected_action_types : set[str] or None, default ``None``
        Action type names expected to be present. Returned (sorted) as
        ``missing`` if absent. ``None`` skips the expectation check; an empty
        list is then returned for ``missing``. Atomic vocabulary differs from
        standard on collapsed names — pass ``"corner"`` (not ``"corner_short"``)
        and ``"freekick"`` (not ``"freekick_short"``).

    Returns
    -------
    CoverageMetrics
        ``{"counts": {...}, "missing": [...], "total_actions": N}`` — the
        standard TypedDict from ``silly_kicks.spadl.utils``.

    Raises
    ------
    ValueError
        If the ``type_id`` column is missing.

    Examples
    --------
    Validate an atomic converter's output covers all expected action types::

        from silly_kicks.atomic.spadl import convert_to_atomic, coverage_metrics

        atomic = convert_to_atomic(actions)
        m = coverage_metrics(
            actions=atomic,
            expected_action_types={"pass", "shot", "receival", "interception"},
        )
        assert not m["missing"], f"Missing atomic action types: {m['missing']}"
    """
    if "type_id" not in actions.columns:
        raise ValueError(
            f"coverage_metrics: actions missing required 'type_id' column. Got: {sorted(actions.columns)}"
        )

    n = len(actions)
    counts: dict[str, int] = {}
    if n > 0:
        # Reverse map: id -> name. Built from atomic actiontypes (single source
        # of truth for atomic). Out-of-vocab ids report as "unknown".
        id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
        type_id_arr = actions["type_id"].to_numpy()
        for tid in type_id_arr:
            name = id_to_name.get(int(tid), "unknown")
            counts[name] = counts.get(name, 0) + 1

    expected = set(expected_action_types) if expected_action_types else set()
    missing = sorted(expected - set(counts.keys())) if expected else []

    from silly_kicks.spadl.utils import CoverageMetrics

    return CoverageMetrics(counts=counts, missing=missing, total_actions=n)
```

The `from silly_kicks.spadl.utils import CoverageMetrics` is **inside** the function body (deferred) to keep the standard module import cheap and avoid surfacing unused-import warnings at module load. The forward-reference string `"CoverageMetrics"` in the return-annotation suppresses the same.

- [ ] **Step 4: Run the new tests — expect all 12 to pass**

```bash
uv run pytest tests/atomic/test_atomic_coverage_metrics.py -v
```

Expected: `12 passed in ...`. Zero failures.

If `test_returns_typeddict_shape` fails (the `m: CoverageMetrics = ...` test), it means the import path or annotation isn't being recognized — re-check the deferred-import pattern.

- [ ] **Step 5: Re-export `coverage_metrics` and `CoverageMetrics` from `silly_kicks.atomic.spadl.__init__.py`**

Edit `silly_kicks/atomic/spadl/__init__.py`. Replace:

```python
"""Implementation of the Atomic-SPADL language."""

__all__ = [
    "ATOMIC_SPADL_COLUMNS",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "convert_to_atomic",
    "play_left_to_right",
    "validate_atomic_spadl",
]

from .base import convert_to_atomic
from .config import actiontypes_df, bodyparts_df
from .schema import ATOMIC_SPADL_COLUMNS
from .utils import (
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    play_left_to_right,
    validate_atomic_spadl,
)
```

With:

```python
"""Implementation of the Atomic-SPADL language."""

__all__ = [
    "ATOMIC_SPADL_COLUMNS",
    "CoverageMetrics",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "convert_to_atomic",
    "coverage_metrics",
    "play_left_to_right",
    "validate_atomic_spadl",
]

from silly_kicks.spadl.utils import CoverageMetrics

from .base import convert_to_atomic
from .config import actiontypes_df, bodyparts_df
from .schema import ATOMIC_SPADL_COLUMNS
from .utils import (
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    coverage_metrics,
    play_left_to_right,
    validate_atomic_spadl,
)
```

Notes:
- `CoverageMetrics` and `coverage_metrics` inserted in alphabetical position in `__all__` (matches the existing alphabetised pattern).
- `from silly_kicks.spadl.utils import CoverageMetrics` placed **above** the `from .base import ...` block (cross-package import comes before relative imports per ruff/isort default).
- `coverage_metrics` added to the relative-import tuple in alphabetical position.

- [ ] **Step 6: Verify the public re-export round-trips**

```bash
uv run python -c "from silly_kicks.atomic.spadl import coverage_metrics, CoverageMetrics; print(coverage_metrics, CoverageMetrics)"
```

Expected output (one line, with the actual addresses):

```
<function coverage_metrics at 0x...> <class 'silly_kicks.spadl.utils.CoverageMetrics'>
```

Confirms `CoverageMetrics` resolves to the standard module's class (single source of truth) and `coverage_metrics` resolves to the new atomic function.

- [ ] **Step 7: Re-run atomic coverage tests + the public-API examples gate to confirm no regressions**

```bash
uv run pytest tests/atomic/test_atomic_coverage_metrics.py tests/test_public_api_examples.py -v
```

Expected: All 12 atomic-coverage tests + all 19 gate parametrize cases pass. Zero failures.

The gate's `_SKIP_SYMBOLS` already excludes `CoverageMetrics` (it's a TypedDict — fields are the documentation). `coverage_metrics` (the new function in `atomic/spadl/utils.py`) gets an Examples section in its docstring as part of Step 3 — verify the Examples-block matches PR-S13 style (one trailing `::` block, no `>>> ` prefix).

---

### Task 8: Verification gates (full local equivalent of CI)

**Files:** None — read-only verification.

These are the same gates CI runs. Catching anything here is cheaper than catching it in CI.

- [ ] **Step 1: Pin tooling versions (matches CI exactly)**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: install/upgrade output. Zero error lines.

- [ ] **Step 2: Run ruff check**

```bash
uv run ruff check silly_kicks/ tests/
```

Expected: `All checks passed!`. Zero issues.

If issues appear, they will be in the new test file or new function — most commonly: import ordering, unused imports, or line length. Fix in place.

- [ ] **Step 3: Run ruff format check**

```bash
uv run ruff format --check silly_kicks/ tests/
```

Expected: `... files already formatted`. No re-format suggestions.

If files would be reformatted, run `uv run ruff format silly_kicks/ tests/` to apply, then re-run `--check`.

- [ ] **Step 4: Run pyright**

```bash
uv run pyright silly_kicks/
```

Expected: `0 errors, 0 warnings, 0 informations`.

If any type errors appear in the atomic `coverage_metrics` function or the new test file, common culprits:
- The deferred `from silly_kicks.spadl.utils import CoverageMetrics` inside the function body — pyright should accept this; if not, move it to module top with a `# noqa` or refactor to a top-level import.
- The forward-reference string `"CoverageMetrics"` in the return annotation — ensure the import is resolvable at type-check time.

- [ ] **Step 5: Run the full pytest suite (excluding e2e tests that need external fixtures)**

```bash
uv run pytest tests/ -m "not e2e" --tb=short
```

Expected baseline pre-PR-S14 was **788 passed, 4 skipped**. Post-PR-S14 expected: **~803 passed, 4 skipped** (788 baseline + 5 new gate parametrize cases now passing + 12 new atomic coverage tests = 805 — minor variance acceptable; the spec rounded to 803). Zero failures.

If any non-related test fails, STOP and investigate — this is exactly the "silently-skipping-tests hide breakage" case (memory: `feedback_silently_skipping_tests.md`). Some other test may have been latent-broken and now surfaces.

- [ ] **Step 6: Spot-check the rendered docstring for one representative new Example**

```bash
uv run python -c "from silly_kicks.vaep.formula import value; help(value)" | head -40
```

Expected: the Examples block renders correctly, with the `::` block treated as code by `help()` (i.e., shown verbatim with proper indentation).

Same spot-check for atomic coverage_metrics:

```bash
uv run python -c "from silly_kicks.atomic.spadl import coverage_metrics; help(coverage_metrics)" | head -40
```

---

### Task 9: Admin updates (CHANGELOG, TODO, version)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current `CHANGELOG.md` head + locate the `[2.1.1]` entry**

```bash
head -40 CHANGELOG.md
```

(Use the Read tool, or `head -40` is fine since it's a small file.) Note the format/style of the existing entries — match it.

- [ ] **Step 2: Add `[2.2.0]` entry above `[2.1.1]`**

Use Edit with `old_string` matching the line just above `## [2.1.1]` (likely `## [2.1.1] - 2026-04-...` or the section header). Add the new entry. Sample structure (adjust to actual CHANGELOG style):

```markdown
## [2.2.0] - 2026-04-30

### Added
- `silly_kicks.atomic.spadl.coverage_metrics` — atomic counterpart to the standard
  `coverage_metrics` utility (added in 1.10.0). Resolves `type_id` against the
  atomic 33-type vocabulary; reuses the standard `CoverageMetrics` TypedDict for
  single-source-of-truth typing. Closes TODO C-1.
- Examples sections on 25 previously-uncovered public-API surfaces across
  `silly_kicks/vaep/labels.py`, `silly_kicks/vaep/formula.py`,
  `silly_kicks/atomic/vaep/features.py`, `silly_kicks/atomic/vaep/labels.py`,
  and `silly_kicks/atomic/vaep/formula.py`. Closes the PR-S13 documentation
  coverage gap.

### Changed
- `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` widened from 14 →
  19 entries. The CI gate now mechanically enforces Examples coverage across
  the entire public API surface.

```

- [ ] **Step 3: Close C-1 entry in `TODO.md`**

The current C-1 row at `TODO.md:28` will be moved out of the "Tech Debt" table (or marked closed in-line, depending on the file's convention).

First, read TODO.md head + the C-1 row to understand the format:

```bash
head -40 TODO.md
```

Then Edit: remove the C-1 row from its current table. If the file has a "Closed" / "Resolved" section, move it there with a "Resolved in 2.2.0" note. If not, simply delete the row — `git log` is the historical record.

- [ ] **Step 4: Bump version in `pyproject.toml` from 2.1.1 → 2.2.0**

Edit `pyproject.toml`. Match on:

```toml
version = "2.1.1"
```

Replace with:

```toml
version = "2.2.0"
```

(There should be exactly one occurrence of `version = "2.1.1"` in the file.)

- [ ] **Step 5: Re-run pytest to confirm version bump didn't break anything**

```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```

Expected: same green result as Task 8 Step 5.

---

### Task 10: /final-review

**Files:** None — read-only review pass.

- [ ] **Step 1: Invoke /final-review skill**

This runs the project's pre-commit quality gate (mad-scientist-skills:final-review). It reviews all code/doc changes for consistency, best practices, and completeness.

Use the Skill tool: `Skill(skill="mad-scientist-skills:final-review")`.

The skill will report findings. Fix anything it flags as actionable in the working tree (no commit yet — that's Task 11).

- [ ] **Step 2: Re-run verification gates after any /final-review fixes**

```bash
uv run ruff check silly_kicks/ tests/ && uv run pyright silly_kicks/ && uv run pytest tests/ -m "not e2e" --tb=short -q
```

Expected: all green. If anything regresses, fix and re-run.

---

### Task 11: Single-commit gate (USER APPROVAL REQUIRED)

**Files:** No file changes — this is a commit decision point.

- [ ] **Step 1: Show git status and diff summary to the user**

```bash
git status --short
git diff --stat
```

Expected output:

```
M  CHANGELOG.md
M  TODO.md
M  pyproject.toml
M  silly_kicks/atomic/spadl/__init__.py
M  silly_kicks/atomic/spadl/utils.py
M  silly_kicks/atomic/vaep/features.py
M  silly_kicks/atomic/vaep/formula.py
M  silly_kicks/atomic/vaep/labels.py
M  silly_kicks/vaep/formula.py
M  silly_kicks/vaep/labels.py
M  tests/test_public_api_examples.py
?? docs/superpowers/specs/2026-04-30-public-api-examples-coverage-completion-design.md
?? docs/superpowers/plans/2026-04-30-public-api-examples-coverage-completion.md
?? tests/atomic/test_atomic_coverage_metrics.py
```

(Plus `?? README.md.backup` and `?? uv.lock` from the pre-existing untracked items — these stay untracked.)

- [ ] **Step 2: Ask the user for explicit approval to commit**

User must say "yes commit" / "approved" / equivalent before proceeding. Per memory `feedback_commit_policy.md`, "no commits or PRs without explicit approval".

- [ ] **Step 3: Stage exactly the intended files**

Avoid `git add -A` / `git add .` (per system rule — could accidentally include untracked items). Stage explicitly:

```bash
git add CHANGELOG.md TODO.md pyproject.toml \
  silly_kicks/atomic/spadl/__init__.py \
  silly_kicks/atomic/spadl/utils.py \
  silly_kicks/atomic/vaep/features.py \
  silly_kicks/atomic/vaep/formula.py \
  silly_kicks/atomic/vaep/labels.py \
  silly_kicks/vaep/formula.py \
  silly_kicks/vaep/labels.py \
  tests/test_public_api_examples.py \
  tests/atomic/test_atomic_coverage_metrics.py \
  docs/superpowers/specs/2026-04-30-public-api-examples-coverage-completion-design.md \
  docs/superpowers/plans/2026-04-30-public-api-examples-coverage-completion.md
```

(The spec + plan files ride with this PR for posterity, mirroring PR-S13's pattern.)

- [ ] **Step 4: Create the single commit**

Use a HEREDOC for the commit message:

```bash
git commit -m "$(cat <<'EOF'
feat(api): public-API Examples coverage completion + atomic coverage_metrics — silly-kicks 2.2.0 (PR-S14)

Closes the PR-S13 documentation coverage gap by widening the
tests/test_public_api_examples.py CI gate from 14 to 19 module files
and adding Examples sections to 25 previously-uncovered public surfaces:

- silly_kicks/vaep/labels.py: 5 functions
- silly_kicks/vaep/formula.py: 3 functions
- silly_kicks/atomic/vaep/features.py: 9 functions
- silly_kicks/atomic/vaep/labels.py: 5 functions
- silly_kicks/atomic/vaep/formula.py: 3 functions

Also closes TODO C-1 (atomic coverage_metrics parity, deferred from 1.10.0):

- silly_kicks.atomic.spadl.coverage_metrics — atomic counterpart that
  resolves type_id against the atomic 33-type vocabulary
  (silly_kicks.atomic.spadl.config.actiontypes), reusing the standard
  CoverageMetrics TypedDict from silly_kicks.spadl.utils as the single
  source of truth.
- Re-exported from silly_kicks.atomic.spadl alongside CoverageMetrics
  for atomic-side discoverability.
- 12 new tests in tests/atomic/test_atomic_coverage_metrics.py mirror
  the standard tests/spadl/test_coverage_metrics.py shape, with
  atomic-vocabulary-specific assertions (collapsed corner / freekick
  names, atomic-only receival type, etc.).

Version bump 2.1.1 → 2.2.0 (minor — new public API).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit created with the SHA printed. Pre-commit hooks (if any are configured) run and pass.

If a pre-commit hook fails:
- **Do NOT use `--amend`** (memory rule + system policy).
- Fix the underlying issue, re-stage, create a NEW commit. Then squash interactively (or via `git reset --soft HEAD~N` + re-commit) to maintain the one-commit-per-branch invariant.

- [ ] **Step 5: Verify the commit**

```bash
git log --oneline -1
git show --stat HEAD
```

Expected: single commit on the branch with all 14 files (11 modified + 3 new — including spec and plan).

---

### Task 12: Push, PR, merge, tag

**Files:** None — git operations.

This task is **gated on Task 11 commit existing and being clean**.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/public-api-examples-coverage-completion
```

Expected: branch published to GitHub remote.

- [ ] **Step 2: Open PR with gh CLI**

```bash
gh pr create --title "feat(api): Examples coverage completion + atomic coverage_metrics — silly-kicks 2.2.0 (PR-S14)" --body "$(cat <<'EOF'
## Summary
- Closes the PR-S13 documentation coverage gap by extending the public-API Examples gate from 14 → 19 module files (covering 25 previously-uncovered public surfaces).
- Closes TODO C-1 (atomic `coverage_metrics` parity, deferred from 1.10.0). Atomic version reuses the standard `CoverageMetrics` TypedDict (single source of truth) and resolves against the atomic 33-type vocabulary.
- Version bump 2.1.1 → 2.2.0 (minor — new public API via `silly_kicks.atomic.spadl.coverage_metrics`).

## Test plan
- [x] `uv run pytest tests/test_public_api_examples.py -v` — 19 passed (was 14).
- [x] `uv run pytest tests/atomic/test_atomic_coverage_metrics.py -v` — 12 passed (new file).
- [x] `uv run pytest tests/ -m "not e2e"` — full suite green (~803 passed, 4 skipped).
- [x] `uv run ruff check silly_kicks/ tests/` — clean.
- [x] `uv run ruff format --check silly_kicks/ tests/` — clean.
- [x] `uv run pyright silly_kicks/` — 0 errors.
- [x] `from silly_kicks.atomic.spadl import coverage_metrics, CoverageMetrics` round-trips.
- [x] `/final-review` — clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 3: Confirm CI passes on the PR**

Wait for CI to fire on the PR (typically ~3-5 min for this repo's matrix). Use:

```bash
gh pr checks --watch
```

Expected: all checks green. If any check fails, investigate — same matrix CI memory caveat (Python 3.10/3.11/3.12 vs local 3.14) applies.

- [ ] **Step 4: Squash-merge (per repo policy — squash-only, auto-delete branches)**

```bash
gh pr merge --squash --delete-branch
```

Expected: PR merged, branch deleted on remote.

- [ ] **Step 5: Sync local and tag the release**

```bash
git switch main
git pull origin main
git tag -a v2.2.0 -m "silly-kicks 2.2.0 — public-API Examples coverage completion + atomic coverage_metrics (PR-S14)"
git push origin v2.2.0
```

Expected: tag pushed; PyPI auto-publish workflow fires (see `.github/workflows/`).

- [ ] **Step 6: Verify PyPI publish succeeded**

After ~5 min, check:

```bash
gh run list --workflow=publish.yml --limit=1
```

Expected: workflow status `completed` / `success`. If failed, investigate workflow logs — most common cause is PyPI token expiry or transient network.

- [ ] **Step 7: Final state confirmation**

```bash
git log --oneline -3
git status --short
```

Expected: latest commit on main is the squash-merged PR-S14, working tree clean (or with the same pre-existing untracked items: `?? README.md.backup`, `?? uv.lock`).

---

## Self-Review

Re-checking the plan against the spec section-by-section (per writing-plans skill):

**Spec §1 (Problem):** ✅ All 5 files + 25 surfaces covered (Tasks 2-6). C-1 covered (Task 7).

**Spec §2 (Goals):** ✅ Goal 1 (close gap, widen gate to 19) → Tasks 1-6, 8. Goal 2 (atomic coverage_metrics) → Task 7. Goal 3 (2.2.0 minor release) → Tasks 9, 12.

**Spec §3 (Non-goals):** ✅ No doctests (style is illustrative only). No examples on private symbols. No examples on `learners.py` (verified fully private). No new tests beyond the gate-extension + atomic coverage_metrics tests. No CLAUDE.md / ADR.

**Spec §4.1 (File structure):** ✅ All 11+1 files in the file-structure table covered by Tasks 1-9.

**Spec §4.2 (Style):** ✅ Variable-naming conventions (`actions`, `atomic`, `states`, `feats`, `p_scores`, `p_concedes`, `actions_with_names`) used consistently in plan Examples.

**Spec §4.3 (Per-group templates):** ✅ Each Example matches the template structure for its group.

**Spec §4.4 (Atomic coverage_metrics design):** ✅ Reuse-not-duplicate pattern in Task 7. Three test classes (post-edit names). Atomic vocabulary resolved from `atomicspadl.actiontypes`.

**Spec §4.5 (Updated `_PUBLIC_MODULE_FILES`):** ✅ Task 1 Step 1 extends to 19 entries with the 5 named files.

**Spec §5 (Implementation order):** ✅ Tasks 1 (T1) → 2-6 (T2-T4) → 7 (T5) → 8 (T6) → 9-10 (T7-T8) → 11 (T9) → 12 (T10).

**Spec §6 (Verification gates):** ✅ Task 8 covers ruff check, ruff format check, pyright, pytest, spot-check.

**Spec §7 (Commit cycle):** ✅ Single commit, gated on user approval. Branch name `feat/public-api-examples-coverage-completion`. Version bump.

**Spec §8 (Risks + mitigations):** ✅ Probability inputs (`p_scores`, `p_concedes`) used as abstract Series. Atomic test cases lock vocabulary contract. Atomic feature comments clarify atomic-specific behavior. Spot-check in Task 8 Step 6.

**Spec §9 (Acceptance criteria):**
1. ✅ 19 entries in `_PUBLIC_MODULE_FILES` (Task 1).
2. ✅ All 25 surfaces have Examples (Tasks 2-6).
3. ✅ `silly_kicks.atomic.spadl.coverage_metrics` importable, returns `CoverageMetrics`, validates against atomic vocabulary, ~12 unit tests (Task 7).
4. ✅ TODO C-1 closed (Task 9 Step 3).
5. ✅ CHANGELOG `[2.2.0]` entry (Task 9 Step 2).
6. ✅ pyproject version 2.2.0 (Task 9 Step 4).
7. ✅ Verification gates green (Task 8).
8. ✅ /final-review clean (Task 10).
9. ✅ Tag v2.2.0 pushed → PyPI publish (Task 12 Steps 5-6).
10. ✅ Style consistent with PR-S13.

**Placeholder scan:** No "TBD" / "implement later" / "similar to Task N" / "add error handling" anywhere. Each step has explicit code, exact commands, expected output.

**Type / signature consistency:** All Examples use the same import paths, the same variable names. `coverage_metrics` signature in plan Step 3 matches the standard's signature exactly (keyword-only args). `CoverageMetrics` TypedDict is the standard one throughout.

No issues found. Plan is ready.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-30-public-api-examples-coverage-completion.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review at natural gates (post-T1 / post-T6 / post-T8 / pre-commit / pre-merge).

Per memory `feedback_commit_policy.md` and prior session preference ("Inline execution preferred (vs subagent flow) for batched approval gates at the end"), **Inline Execution** is the standing default unless you say otherwise.

Which approach?
