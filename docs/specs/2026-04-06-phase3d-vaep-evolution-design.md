# Phase 3d: VAEP Evolution — Design Spec

> **Goal:** Add Hybrid-VAEP mode (result-free pre-action features), xG-targeted
> label option, reproducible training via random_state, and resolve remaining
> VAEP optimization deferrals (O-7, O-14, O-16). Resolves S6, O-7, O-14, O-16.

## Context

Phase 3c vectorized all converter internals. Phase 3d evolves the VAEP
framework itself — the model training, feature engineering, and label generation
layer. These changes add research-grade capabilities that the academic community
has identified but no open-source library has implemented.

VAEP has three known design limitations:
1. **Result leakage** — pre-action features include action result, undercrediting
   defenders (fixed by Hybrid-VAEP)
2. **Target variable** — trains on raw goals; xG-based labels are statistically
   superior (fixed by xG-targeted labels)
3. **Team-strength bias** — no debiasing (out of scope)

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hybrid-VAEP API | `HybridVAEP(VAEP)` subclass | Clean separation, no flag sprawl |
| Result suppression | `_prev_only` feature transformers that skip `a0` | Surgical — reuses existing `@simple` infrastructure |
| xG labels | Optional `xg_column` parameter on `scores()`/`concedes()` | Backward-compatible, no new label functions needed |
| Label customization | `yfns` parameter on `VAEP.__init__` | Lets users inject custom label functions without subclassing |
| random_state | `np.random.default_rng(seed)` in `fit()` | Modern numpy API, reproducible |
| Optimizations | All three (O-7, O-14, O-16) in this phase | Clean up everything VAEP-related before Phase 4 |

## 1. HybridVAEP Subclass

### Problem

Standard VAEP includes `result_onehot` and `actiontype_result_onehot` as
features for the current action `a0`. The `@simple` decorator applies each
feature function across all gamestates `a0, a1, a2`, producing columns like
`result_success_a0`, `result_fail_a0`, etc. The model sees the action's outcome
before valuing it — result leakage.

Atomic-VAEP avoids this by design (no result column). Regular VAEP does not.

### Solution

Two new feature transformers in `silly_kicks/vaep/features.py`:

```python
def result_onehot_prev_only(gamestates: GameStates) -> Features:
    """Result one-hot encoding for previous actions only (a1, a2, ...).

    Excludes a0 to prevent result leakage in Hybrid-VAEP mode.
    """
    dfs = []
    for i, actions in enumerate(gamestates):
        if i == 0:
            continue  # skip a0
        result_df = result_onehot.__wrapped__(actions)
        result_df.columns = [c + "_a" + str(i) for c in result_df.columns]
        dfs.append(result_df)
    return pd.concat(dfs, axis=1)


def actiontype_result_onehot_prev_only(gamestates: GameStates) -> Features:
    """Action type × result cross-product for previous actions only (a1, a2, ...).

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

These use `__wrapped__` to access the underlying function before the `@simple`
decorator, then manually apply the suffix logic for `a1+` only.

### HybridVAEP class

New file `silly_kicks/vaep/hybrid.py`:

```python
class HybridVAEP(VAEP):
    """VAEP with result leakage removed from current-action features.

    In standard VAEP, the model receives the action's result (success/fail)
    as a feature, which creates information leakage — the model knows the
    outcome before valuing the action. HybridVAEP removes result information
    from the current action (a0) while preserving it for previous actions
    (a1, a2) where the result is already known.
    """
```

With `hybrid_xfns_default` replacing `result_onehot` and
`actiontype_result_onehot` with their `_prev_only` variants.

### Public API

```python
from silly_kicks.vaep import HybridVAEP

model = HybridVAEP()
model.fit(X, y)
ratings = model.rate(game, game_actions)
```

Export from `silly_kicks/vaep/__init__.py`.

## 2. xG-Targeted Labels

### Problem

Standard labels (`scores`, `concedes`) are binary — "did a goal happen in
the next 10 actions?" A wide-open shot that misses produces `scores=0` for
the preceding pass, even though the chance was created. xG-based labels
weight by shot quality.

### Solution

Add optional `xg_column: str | None = None` to `scores()` and `concedes()`.

**When `xg_column` is None** (default): existing binary behavior unchanged.

**When `xg_column` is provided**: instead of binary goal detection, the label
is the maximum xG value of any goal-scoring shot in the next `nr_actions`.
Non-shot actions contribute 0. This makes the label continuous (0.0 to 1.0)
instead of binary.

```python
def scores(actions, nr_actions=10, xg_column=None):
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)
    return _scores_binary(actions, nr_actions)
```

The `_scores_xg` variant looks ahead `nr_actions` and takes the max xG
value from same-team shots (or opposing-team own goals).

### yfns parameter

Add `yfns` to `VAEP.__init__`:

```python
def __init__(self, xfns=None, yfns=None, nb_prev_actions=3):
    self.xfns = xfns_default if xfns is None else xfns
    self.yfns = yfns if yfns is not None else [self._lab.scores, self._lab.concedes]
    self.nb_prev_actions = nb_prev_actions
```

### Make `rate()` generic

Replace the hardcoded `y_hat.scores, y_hat.concedes` with positional access:

```python
p_scores = y_hat.iloc[:, 0]
p_concedes = y_hat.iloc[:, 1]
```

The formula still expects two probability columns (scoring, conceding) —
the column *names* don't matter, only the order.

### Both VAEP and Atomic-VAEP

Apply the same changes to `silly_kicks/atomic/vaep/labels.py` (scores and
concedes functions). `AtomicVAEP` inherits `yfns` from `VAEP.__init__`.

## 3. Reproducible Training (S6)

Add `random_state: int | None = None` to `VAEP.fit()`:

```python
def fit(self, X, y, learner="xgboost", val_size=0.25,
        tree_params=None, fit_params=None, random_state=None):
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(nb_states)
    ...
```

`AtomicVAEP` inherits this via `super().fit()`.

## 4. Labels Vectorization (O-7)

### Problem

`scores()` and `concedes()` use a nested loop adding 27 columns
(`team_id+1`, `goal+1`, `owngoal+1`, ..., `team_id+9`, `goal+9`,
`owngoal+9`) then OR-ing them together. Same pattern in atomic labels.

### Solution

Replace with a shift-based loop that accumulates into a single boolean
Series:

```python
def _scores_binary(actions, nr_actions=10):
    dominated = actions.type_name.str.contains("shot")
    goal = dominated & (actions.result_id == spadl.result_id["success"])
    owngoal = actions.result_id == spadl.result_id["owngoal"]
    team_id = actions.team_id

    scores = pd.Series(False, index=actions.index)
    for i in range(1, nr_actions + 1):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        scores |= (shifted_goal & same_team) | (shifted_owngoal & ~same_team)
    return pd.DataFrame({"scores": scores}, index=actions.index)
```

Same pattern for `concedes`. Apply to both regular and atomic label modules.

## 5. Feature Column Names Caching (O-14)

`feature_column_names(fs, nb_prev_actions)` is called redundantly in the
fit/rate cycle. Cache at the instance level:

```python
def compute_features(self, game, game_actions):
    ...
    if not hasattr(self, "_cached_feature_cols"):
        self._cached_feature_cols = self._fs.feature_column_names(
            self.xfns, self.nb_prev_actions
        )
    ...
```

Or simpler: since `feature_column_names` creates a dummy DataFrame and runs
all transformers, cache the result with `@functools.lru_cache` using a
`tuple(fs)` key.

## 6. Type×Result Onehot Optimization (O-16)

Replace the 138-iteration nested loop in `actiontype_result_onehot`:

```python
@simple
def actiontype_result_onehot(actions):
    res = result_onehot.__wrapped__(actions)
    tys = actiontype_onehot.__wrapped__(actions)
    # Vectorized outer product via numpy broadcasting
    cross = tys.values[:, :, np.newaxis] & res.values[:, np.newaxis, :]
    cols = [f"{tc}_{rc}" for tc in tys.columns for rc in res.columns]
    return pd.DataFrame(cross.reshape(len(actions), -1), columns=cols, index=actions.index)
```

## 7. Files Modified/Created

```
Created:
  silly_kicks/vaep/hybrid.py              — HybridVAEP subclass

Modified:
  silly_kicks/vaep/features.py            — result_onehot_prev_only, actiontype_result_onehot_prev_only, O-16 optimization
  silly_kicks/vaep/labels.py              — xg_column parameter, O-7 vectorization
  silly_kicks/vaep/base.py                — yfns parameter, random_state, generic rate(), O-14 caching
  silly_kicks/vaep/formula.py             — no changes (formula unchanged)
  silly_kicks/vaep/__init__.py            — export HybridVAEP
  silly_kicks/atomic/vaep/labels.py       — xg_column parameter, O-7 vectorization
  silly_kicks/atomic/vaep/base.py         — inherits changes via super()
  docs/DEFERRED.md                        — mark S6, O-7, O-14, O-16 resolved
```

## 8. What This Does NOT Change

- **SPADL vocabulary** — no changes to action types, results, bodyparts
- **Converter code** — no changes (Phase 3c is done)
- **Standard VAEP behavior** — all existing defaults remain; HybridVAEP is
  opt-in, xG labels are opt-in, random_state defaults to None (non-reproducible,
  matching current behavior)
- **Atomic-VAEP features** — already result-free; no Hybrid variant needed
- **Formula** — the value computation is unchanged; only the training inputs
  (features, labels) evolve

## 9. Testing Strategy

- **HybridVAEP features**: Verify `result_*_a0` columns absent, `result_*_a1`
  present. Verify column count is less than standard VAEP.
- **xG labels**: Test with a DataFrame that has an `xg` column. Verify
  `scores(actions, xg_column="xg")` produces float labels. Verify backward
  compat with `scores(actions)`.
- **random_state**: Two calls to `fit(X, y, random_state=42)` produce identical
  train/val splits.
- **O-7**: Labels output matches between old and new on the existing test
  fixtures.
- **O-16**: `actiontype_result_onehot` output matches between old loop and
  vectorized version.
- **All existing tests** must pass unchanged.
