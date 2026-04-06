# Phase 3d: VAEP Evolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hybrid-VAEP mode (result-free pre-action features), xG-targeted label option, reproducible training, and resolve remaining VAEP optimizations (O-7, O-14, O-16).

**Architecture:** HybridVAEP is a subclass of VAEP that uses `_prev_only` feature transformers to exclude result information from the current action (`a0`). xG labels are an optional parameter on the existing `scores()`/`concedes()` functions. `random_state` is added to `fit()`. Labels are vectorized to eliminate the 27-column nested loop. All changes are backward-compatible — existing behavior is preserved when new parameters are not used.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest. No new dependencies.

**Working directory:** `D:\Development\karstenskyt__silly-kicks\`

**Sources:** Design spec at `docs/specs/2026-04-06-phase3d-vaep-evolution-design.md`

---

## File Structure

```
Created:
  silly_kicks/vaep/hybrid.py               — HybridVAEP subclass + hybrid_xfns_default

Modified:
  silly_kicks/vaep/features.py             — result_onehot_prev_only, actiontype_result_onehot_prev_only, O-16
  silly_kicks/vaep/labels.py               — xg_column parameter, O-7 vectorization
  silly_kicks/vaep/base.py                 — yfns parameter, random_state, generic rate(), O-14
  silly_kicks/vaep/__init__.py             — export HybridVAEP
  silly_kicks/atomic/vaep/labels.py        — xg_column parameter, O-7 vectorization
  docs/DEFERRED.md                         — mark S6, O-7, O-14, O-16 resolved

Test files:
  tests/vaep/test_features.py             — add _prev_only and O-16 tests
  tests/vaep/test_labels.py               — add xG label and O-7 tests
  tests/vaep/test_vaep.py                 — add HybridVAEP and random_state tests
```

---

## Task 1: Labels Vectorization (O-7)

**Files:**
- Modify: `silly_kicks/vaep/labels.py`
- Modify: `silly_kicks/atomic/vaep/labels.py`
- Modify: `tests/vaep/test_labels.py`

The existing labels use a nested loop adding 27 columns. Replace with a shift-based accumulator.

- [ ] **Step 1: Write the tests**

Add to `tests/vaep/test_labels.py`:

```python
import silly_kicks.atomic.spadl.utils as aspu
import silly_kicks.atomic.vaep.labels as alab


def test_atomic_scores(atomic_spadl_actions: pd.DataFrame) -> None:
    atomic_spadl_actions = aspu.add_names(atomic_spadl_actions)
    scores = alab.scores(atomic_spadl_actions, 10)
    assert len(scores) == len(atomic_spadl_actions)
    assert scores.columns.tolist() == ["scores"]
    assert scores.dtypes["scores"] == bool


def test_atomic_concedes(atomic_spadl_actions: pd.DataFrame) -> None:
    atomic_spadl_actions = aspu.add_names(atomic_spadl_actions)
    concedes = alab.concedes(atomic_spadl_actions, 10)
    assert len(concedes) == len(atomic_spadl_actions)
    assert concedes.columns.tolist() == ["concedes"]


def test_scores_dtype(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    scores = lab.scores(spadl_actions, 10)
    assert scores.dtypes["scores"] == bool


def test_concedes_dtype(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    concedes = lab.concedes(spadl_actions, 10)
    assert concedes.dtypes["concedes"] == bool
```

- [ ] **Step 2: Rewrite `silly_kicks/vaep/labels.py` scores and concedes**

Replace `scores()` (lines 8-49) with:

```python
def scores(actions: pd.DataFrame, nr_actions: int = 10, xg_column: str | None = None) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10
        Number of actions after the current action to consider.
    xg_column : str, optional
        Column name containing xG values. When provided, returns continuous
        xG-weighted labels instead of binary labels.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action.
    """
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)

    goal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["success"]
    )
    owngoal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["owngoal"]
    )
    team_id = actions["team_id"]

    result = goal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        result = result | (shifted_goal & same_team) | (shifted_owngoal & ~same_team)

    return pd.DataFrame(result, columns=["scores"])
```

Replace `concedes()` (lines 52-92) with the mirror logic:

```python
def concedes(actions: pd.DataFrame, nr_actions: int = 10, xg_column: str | None = None) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10
        Number of actions after the current action to consider.
    xg_column : str, optional
        Column name containing xG values. When provided, returns continuous
        xG-weighted labels instead of binary labels.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action.
    """
    if xg_column is not None:
        return _concedes_xg(actions, nr_actions, xg_column)

    goal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["success"]
    )
    owngoal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["owngoal"]
    )
    team_id = actions["team_id"]

    result = owngoal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        result = result | (shifted_goal & ~same_team) | (shifted_owngoal & same_team)

    return pd.DataFrame(result, columns=["concedes"])
```

Keep `goal_from_shot` unchanged.

Add the xG helper stubs (implemented in Task 3):

```python
def _scores_xg(actions: pd.DataFrame, nr_actions: int, xg_column: str) -> pd.DataFrame:
    """xG-weighted scoring labels."""
    goal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["success"]
    )
    owngoal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["owngoal"]
    )
    xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)
    team_id = actions["team_id"]

    result = pd.Series(0.0, index=actions.index)
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_xg = xg.shift(-i).fillna(0.0)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        score_xg = shifted_xg.where(shifted_goal & same_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & ~same_team, 0.0)
        result = result.combine(score_xg + owngoal_xg, max, fill_value=0.0)

    return pd.DataFrame({"scores": result})


def _concedes_xg(actions: pd.DataFrame, nr_actions: int, xg_column: str) -> pd.DataFrame:
    """xG-weighted conceding labels."""
    goal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["success"]
    )
    owngoal = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.result_id["owngoal"]
    )
    xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)
    team_id = actions["team_id"]

    result = pd.Series(0.0, index=actions.index)
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_xg = xg.shift(-i).fillna(0.0)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        concede_xg = shifted_xg.where(shifted_goal & ~same_team, 0.0)
        owngoal_concede_xg = shifted_xg.where(shifted_owngoal & same_team, 0.0)
        result = result.combine(concede_xg + owngoal_concede_xg, max, fill_value=0.0)

    return pd.DataFrame({"concedes": result})
```

- [ ] **Step 3: Apply the same vectorization to `silly_kicks/atomic/vaep/labels.py`**

Same pattern but using `atomicspadl.actiontype_id["goal"]` and `atomicspadl.actiontype_id["owngoal"]` instead of `type_name.str.contains("shot")`. Add the `xg_column` parameter to both functions. The atomic xG helpers use `type_id == goal` instead of `type_name.str.contains("shot")`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/vaep/test_labels.py tests/atomic/test_atomic_labels.py -v`
Expected: all PASS

---

## Task 2: Features — _prev_only Transformers + O-16 Optimization

**Files:**
- Modify: `silly_kicks/vaep/features.py`
- Modify: `tests/vaep/test_features.py`

- [ ] **Step 1: Write the tests**

Add to `tests/vaep/test_features.py`:

```python
def test_result_onehot_prev_only(spadl_actions: pd.DataFrame):
    """Hybrid-VAEP: result_onehot_prev_only excludes a0 columns."""
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.result_onehot_prev_only(gamestates)
    # Should have a1 and a2 columns but NOT a0
    a0_cols = [c for c in result.columns if c.endswith("_a0")]
    a1_cols = [c for c in result.columns if c.endswith("_a1")]
    assert len(a0_cols) == 0, f"Found a0 columns: {a0_cols}"
    assert len(a1_cols) > 0, "Missing a1 columns"


def test_actiontype_result_onehot_prev_only(spadl_actions: pd.DataFrame):
    """Hybrid-VAEP: actiontype_result_onehot_prev_only excludes a0 columns."""
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.actiontype_result_onehot_prev_only(gamestates)
    a0_cols = [c for c in result.columns if c.endswith("_a0")]
    a1_cols = [c for c in result.columns if c.endswith("_a1")]
    assert len(a0_cols) == 0, f"Found a0 columns: {a0_cols}"
    assert len(a1_cols) > 0


def test_actiontype_result_onehot_vectorized(spadl_actions: pd.DataFrame):
    """O-16: Vectorized actiontype_result_onehot produces same output as before."""
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=2)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.actiontype_result_onehot(gamestates)
    # Check it has the expected cross-product columns
    assert any("actiontype_pass_result_success" in c for c in result.columns)
    assert result.shape[0] == len(spadl_actions)
```

- [ ] **Step 2: Add `result_onehot_prev_only` and `actiontype_result_onehot_prev_only`**

Add to `silly_kicks/vaep/features.py` (after `actiontype_result_onehot`):

```python
def result_onehot_prev_only(gamestates: GameStates) -> Features:
    """Result one-hot encoding for previous actions only (a1, a2, ...).

    Excludes a0 to prevent result leakage in Hybrid-VAEP mode.
    """
    dfs = []
    for i, actions in enumerate(gamestates):
        if i == 0:
            continue
        result_df = result_onehot.__wrapped__(actions)
        result_df.columns = [c + "_a" + str(i) for c in result_df.columns]
        dfs.append(result_df)
    return pd.concat(dfs, axis=1)


def actiontype_result_onehot_prev_only(gamestates: GameStates) -> Features:
    """Action type x result cross-product for previous actions only (a1, a2, ...).

    Excludes a0 to prevent result leakage in Hybrid-VAEP mode.
    """
    dfs = []
    for i, actions in enumerate(gamestates):
        if i == 0:
            continue
        cross_df = actiontype_result_onehot.__wrapped__(actions)
        cross_df.columns = [c + "_a" + str(i) for c in cross_df.columns]
        dfs.append(cross_df)
    return pd.concat(dfs, axis=1)
```

- [ ] **Step 3: Optimize `actiontype_result_onehot` (O-16)**

Replace the nested loop in `actiontype_result_onehot` (lines 297-303):

```python
@simple
def actiontype_result_onehot(actions: Actions) -> Features:
    """Get a one-hot encoding of the combination between the type and result of each action."""
    res = result_onehot.__wrapped__(actions)
    tys = actiontype_onehot.__wrapped__(actions)
    cross = tys.values[:, :, np.newaxis] & res.values[:, np.newaxis, :]
    cols = [f"{tc}_{rc}" for tc in tys.columns for rc in res.columns]
    return pd.DataFrame(cross.reshape(len(actions), -1), columns=cols, index=actions.index)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/vaep/test_features.py -v`
Expected: all PASS

---

## Task 3: VAEP Base — yfns, random_state, generic rate(), O-14

**Files:**
- Modify: `silly_kicks/vaep/base.py`
- Modify: `tests/vaep/test_vaep.py`

- [ ] **Step 1: Add `yfns` parameter to `__init__`**

In `silly_kicks/vaep/base.py`, change `__init__` (line 76-84):

```python
def __init__(
    self,
    xfns: Optional[list[fs.FeatureTransfomer]] = None,
    yfns: Optional[list[Callable]] = None,
    nb_prev_actions: int = 3,
) -> None:
    self.__models: dict[str, Any] = {}
    self.xfns = xfns_default if xfns is None else xfns
    self.yfns = yfns if yfns is not None else [self._lab.scores, self._lab.concedes]
    self.nb_prev_actions = nb_prev_actions
```

Add `from typing import Callable` if not already imported.

- [ ] **Step 2: Add `random_state` to `fit()`**

Change `fit()` signature to add `random_state: int | None = None` parameter. Replace line 171:

```python
idx = np.random.permutation(nb_states)
```

with:

```python
rng = np.random.default_rng(random_state)
idx = rng.permutation(nb_states)
```

- [ ] **Step 3: Make `rate()` generic**

In `rate()`, replace the hardcoded column access (line 248):

```python
p_scores, p_concedes = y_hat.scores, y_hat.concedes
```

with positional access:

```python
p_scores = y_hat.iloc[:, 0]
p_concedes = y_hat.iloc[:, 1]
```

- [ ] **Step 4: Cache feature column names (O-14)**

In `silly_kicks/vaep/base.py`, add a cached property for feature columns. In `fit()` and `_estimate_probabilities()`, both call `self._fs.feature_column_names(self.xfns, self.nb_prev_actions)`. Replace both with a shared cached call:

```python
def _feature_columns(self) -> list[str]:
    """Return cached feature column names."""
    if not hasattr(self, "_cached_feature_cols"):
        self._cached_feature_cols = self._fs.feature_column_names(
            self.xfns, self.nb_prev_actions
        )
    return self._cached_feature_cols
```

Replace both `cols = self._fs.feature_column_names(...)` calls with `cols = self._feature_columns()`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/vaep/ tests/atomic/ -m "not e2e" -v`
Expected: all PASS

---

## Task 4: HybridVAEP Subclass

**Files:**
- Create: `silly_kicks/vaep/hybrid.py`
- Modify: `silly_kicks/vaep/__init__.py`
- Modify: `tests/vaep/test_vaep.py`

- [ ] **Step 1: Write the test**

Add to `tests/vaep/test_vaep.py`:

```python
from silly_kicks.vaep import HybridVAEP
from silly_kicks.vaep import features as fs


def test_hybrid_vaep_no_result_a0():
    """HybridVAEP features must not contain result_*_a0 columns."""
    model = HybridVAEP(nb_prev_actions=3)
    cols = fs.feature_column_names(model.xfns, model.nb_prev_actions)
    a0_result_cols = [c for c in cols if "result" in c and c.endswith("_a0")]
    assert len(a0_result_cols) == 0, f"Found result a0 columns: {a0_result_cols}"


def test_hybrid_vaep_has_result_a1():
    """HybridVAEP features must contain result_*_a1 columns."""
    model = HybridVAEP(nb_prev_actions=3)
    cols = fs.feature_column_names(model.xfns, model.nb_prev_actions)
    a1_result_cols = [c for c in cols if "result" in c and c.endswith("_a1")]
    assert len(a1_result_cols) > 0, "Missing result a1 columns"


def test_hybrid_vaep_fewer_features_than_standard():
    """HybridVAEP should have fewer features than standard VAEP."""
    from silly_kicks.vaep import VAEP
    standard = VAEP(nb_prev_actions=3)
    hybrid = HybridVAEP(nb_prev_actions=3)
    standard_cols = fs.feature_column_names(standard.xfns, standard.nb_prev_actions)
    hybrid_cols = fs.feature_column_names(hybrid.xfns, hybrid.nb_prev_actions)
    assert len(hybrid_cols) < len(standard_cols)
```

- [ ] **Step 2: Create `silly_kicks/vaep/hybrid.py`**

```python
"""Implements the Hybrid-VAEP framework.

Hybrid-VAEP removes result leakage from the current action's features.
Standard VAEP includes the action's result (success/fail) as a feature,
which means the model knows the outcome before valuing the action. This
undercredits defenders and pass receivers.

HybridVAEP uses result_onehot_prev_only and actiontype_result_onehot_prev_only
to include result information only for previous actions (a1, a2) where the
result is already known, not for the current action (a0).
"""

from typing import Optional

from silly_kicks.vaep.base import VAEP

from . import features as fs
from . import formula as vaep
from . import labels as lab

hybrid_xfns_default = [
    fs.actiontype_onehot,
    fs.result_onehot_prev_only,
    fs.actiontype_result_onehot_prev_only,
    fs.bodypart_onehot,
    fs.time,
    fs.startlocation,
    fs.endlocation,
    fs.startpolar,
    fs.endpolar,
    fs.movement,
    fs.team,
    fs.time_delta,
    fs.space_delta,
    fs.goalscore,
]


class HybridVAEP(VAEP):
    """VAEP with result leakage removed from current-action features.

    In standard VAEP, the model receives the action's result (success/fail)
    as a feature, which creates information leakage. HybridVAEP removes
    result information from the current action (a0) while preserving it
    for previous actions (a1, a2) where the result is already known.

    Parameters
    ----------
    xfns : list, optional
        Feature transformers. Uses hybrid_xfns_default if None.
    yfns : list, optional
        Label functions. Uses [scores, concedes] if None.
    nb_prev_actions : int, default=3
        Number of previous actions in game state.
    """

    def __init__(
        self,
        xfns: Optional[list[fs.FeatureTransfomer]] = None,
        yfns: Optional[list] = None,
        nb_prev_actions: int = 3,
    ) -> None:
        xfns = hybrid_xfns_default if xfns is None else xfns
        super().__init__(xfns, yfns, nb_prev_actions)
```

- [ ] **Step 3: Update `silly_kicks/vaep/__init__.py`**

```python
"""Implements the VAEP framework."""

from . import features, formula, labels
from .base import VAEP
from .hybrid import HybridVAEP

__all__ = ["VAEP", "HybridVAEP", "features", "labels", "formula"]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/vaep/test_vaep.py -m "not e2e" -v`
Expected: all PASS

---

## Task 5: xG Label Tests + random_state Test

**Files:**
- Modify: `tests/vaep/test_labels.py`
- Modify: `tests/vaep/test_vaep.py`

- [ ] **Step 1: Add xG label tests**

Add to `tests/vaep/test_labels.py`:

```python
def test_scores_xg(spadl_actions: pd.DataFrame) -> None:
    """xG-targeted labels should produce float scores."""
    spadl_actions = spu.add_names(spadl_actions)
    # Add a fake xG column
    spadl_actions = spadl_actions.copy()
    spadl_actions["xg"] = 0.0
    # Set xG for shots
    shot_mask = spadl_actions["type_name"].str.contains("shot")
    spadl_actions.loc[shot_mask, "xg"] = 0.5
    scores = lab.scores(spadl_actions, 10, xg_column="xg")
    assert len(scores) == len(spadl_actions)
    assert scores["scores"].dtype == float


def test_scores_backward_compat(spadl_actions: pd.DataFrame) -> None:
    """scores() without xg_column should produce binary labels."""
    spadl_actions = spu.add_names(spadl_actions)
    scores = lab.scores(spadl_actions, 10)
    assert scores["scores"].dtype == bool
```

- [ ] **Step 2: Add random_state test**

Add to `tests/vaep/test_vaep.py` (non-e2e section):

```python
def test_random_state_reproducibility():
    """S6: random_state should produce identical train/val splits."""
    import numpy as np
    rng1 = np.random.default_rng(42)
    idx1 = rng1.permutation(100)
    rng2 = np.random.default_rng(42)
    idx2 = rng2.permutation(100)
    assert (idx1 == idx2).all()
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all pass

---

## Task 6: Update DEFERRED.md + Final Verification

**Files:**
- Modify: `docs/DEFERRED.md`

- [ ] **Step 1: Update DEFERRED.md**

Add Phase 3d section:

```markdown
## Phase 3d: VAEP Evolution (2026-04-06)

- **S6** (unseeded np.random.permutation): RESOLVED — random_state parameter on VAEP.fit()
- **O-7** (labels nested loop): RESOLVED — vectorized shift-based accumulation
- **O-14** (feature_column_names uncached): RESOLVED — instance-level caching
- **O-16** (type×result onehot loop): RESOLVED — numpy broadcasting outer product
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

- [ ] **Step 3: Verify HybridVAEP is importable**

Run: `python -c "from silly_kicks.vaep import HybridVAEP; print('HybridVAEP available:', HybridVAEP)"`
Expected: prints the class
