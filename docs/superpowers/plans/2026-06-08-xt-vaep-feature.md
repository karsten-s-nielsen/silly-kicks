# xT-as-a-VAEP-feature (`xt__<method>` xfn factory) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire fitted xT ratings into the VAEP feature framework as an opt-in, frame-free `xt__<method>` feature-transformer factory for both standard and atomic SPADL.

**Architecture:** A `xt_xfns(*, model)` factory closes over a caller-supplied **fitted** `ExpectedThreat` and emits one column `xt__<model.method>` per gamestate slot (NaN for non-move / failed-move actions). Standard rides `model.rate()` directly under `@simple`. Atomic reuses the **same** `model.rate()` by synthesizing a standard-SPADL-shaped frame with a **type-aware** `result_id` (dribble intrinsic; pass/cross next-atom-`receival`), computed once on `states[0]` and mapped to slots by the composite `(game_id, period_id, action_id)` key. The factories only duck-type the model, so the `ExpectedThreat` import is `TYPE_CHECKING`-only (no new runtime dependency edge). The feature is in **no** default xfn list (opt-in ⇒ zero forced retrain).

**Tech Stack:** Python, pandas, numpy, scikit-learn (`NotFittedError`); pytest + `pytest-mock`. Spec: `docs/superpowers/specs/2026-06-08-xt-vaep-feature-design.md`.

---

## File Structure

- **Create** `silly_kicks/vaep/features/expected_threat.py` — standard `xt_xfns` factory + fail-closed validation.
- **Modify** `silly_kicks/vaep/features/__init__.py` — import + `__all__` for `xt_xfns`.
- **Modify** `silly_kicks/vaep/__init__.py` — re-export `xt_xfns`.
- **Modify** `silly_kicks/atomic/vaep/features.py` — atomic `xt_xfns` + `_atomic_xt_delta_map` helper + `__all__`.
- **Modify** `silly_kicks/atomic/vaep/__init__.py` — re-export `xt_xfns`.
- **Create** `tests/vaep/test_xt_feature.py` — standard unit + integration + default-list guard + standard e2e.
- **Create** `tests/atomic/test_atomic_xt_feature.py` — atomic unit cases + universal symmetry oracle + composite-key guard + boundary oracle + atomic e2e.
- **Create** `docs/superpowers/adrs/ADR-<next>-xt-vaep-feature.md` — records the 5 decisions (final task).
- **Modify** `CLAUDE.md`, `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` — ship metadata (final task).

**Commit policy (overrides skill default):** per the user's workflow this branch produces **one** squashed feature commit at the end (spec + plan + ADR + code + tests + metadata bundled). Do **not** commit per task. Each task ends at a green-test checkpoint; the single commit (Task 9) is gated on explicit user approval. Use `.venv/Scripts/python -m pytest` for all test runs.

---

### Task 1: Standard `xt_xfns` factory + fail-closed validation

**Files:**
- Create: `silly_kicks/vaep/features/expected_threat.py`
- Test: `tests/vaep/test_xt_feature.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/vaep/test_xt_feature.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import silly_kicks.xthreat as xt
from silly_kicks.vaep import features as fs
from silly_kicks.vaep.features.expected_threat import xt_xfns
from tests._xthreat_helpers import _moves


@pytest.fixture(scope="module")
def fitted_xt() -> xt.ExpectedThreat:
    """A small fitted singh_counts xT model (fast, deterministic)."""
    return xt.ExpectedThreat().fit(_moves(n_per_zone=40, seed=0))


def test_factory_returns_single_transformer(fitted_xt: xt.ExpectedThreat) -> None:
    transformers = xt_xfns(model=fitted_xt)
    assert isinstance(transformers, list)
    assert len(transformers) == 1


def test_column_names_track_method(fitted_xt: xt.ExpectedThreat) -> None:
    cols = fs.feature_column_names(xt_xfns(model=fitted_xt), nb_prev_actions=3)
    assert cols == ["xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"]


def test_values_equal_model_rate(fitted_xt: xt.ExpectedThreat) -> None:
    actions = _moves(n_per_zone=10, seed=1)
    states = fs.gamestates(actions, 1)
    out = xt_xfns(model=fitted_xt)[0](states)
    expected = fitted_xt.rate(states[0])
    np.testing.assert_array_equal(out["xt__singh_counts_a0"].to_numpy(), expected)


def test_fail_closed_none() -> None:
    with pytest.raises(ValueError, match="fitted ExpectedThreat"):
        xt_xfns(model=None)


def test_fail_closed_str() -> None:
    with pytest.raises(NotImplementedError, match="bundled"):
        xt_xfns(model="default")


def test_fail_closed_unfitted() -> None:
    with pytest.raises(NotFittedError):
        xt_xfns(model=xt.ExpectedThreat())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.vaep.features.expected_threat'`.

- [ ] **Step 3: Write the factory**

```python
# silly_kicks/vaep/features/expected_threat.py
"""xT-as-a-VAEP-feature transformer factory (``xt__<method>``).

Wraps a *fitted, caller-supplied* ``ExpectedThreat`` (see NOTICE for citations) as a
frame-free VAEP feature. Train/serve consistency is the caller's responsibility: fit +
freeze the grid on the VAEP training corpus (or a disjoint exogenous corpus) and reuse
the identical object at serve time (mirrors FrozenXt / ADR-009). NaN for non-move /
failed-move actions, matching ``ExpectedThreat.rate``. Opt-in: NOT in any default xfn
list — adding it to a caller's xfns is a deliberate, self-triggered VAEP retrain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from silly_kicks.vaep.feature_framework import Actions, Features, FeatureTransfomer, simple

if TYPE_CHECKING:  # ExpectedThreat is only duck-typed at runtime (model.xT/.method/.rate)
    from silly_kicks.xthreat import ExpectedThreat

__all__ = ["xt_xfns"]


def _require_fitted_xt(model: "ExpectedThreat | str | None") -> None:
    """Fail closed unless ``model`` is a fitted ExpectedThreat. See NOTICE for citations."""
    if isinstance(model, str):
        raise NotImplementedError(
            "xt_xfns: bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat."
        )
    if model is None:
        raise ValueError("xt_xfns requires a fitted ExpectedThreat (model=...).")
    if not np.any(model.xT):  # same fitted-check ExpectedThreat.rate() uses
        raise NotFittedError("xt_xfns requires a fitted ExpectedThreat; call model.fit(actions) first.")


def xt_xfns(*, model: "ExpectedThreat | str | None" = None) -> list[FeatureTransfomer]:
    """Factory: one frame-free transformer emitting ``xt__<model.method>_a{0,1,2}``.

    Parameters
    ----------
    model : ExpectedThreat
        A fitted xT model. ``str`` (a future bundled variant name) and ``None`` raise.

    Returns
    -------
    list[FeatureTransfomer]
        A one-element list holding the transformer.

    Raises
    ------
    ValueError, NotImplementedError, NotFittedError
        See ``_require_fitted_xt``.

    Examples
    --------
    Opt in to xT as a VAEP feature::

        from silly_kicks.vaep import VAEP, features as fs
        from silly_kicks.vaep.features import xt_xfns
        v = VAEP(xfns=fs.xfns_default + xt_xfns(model=frozen_xt))
    """
    _require_fitted_xt(model)
    col = f"xt__{model.method}"  # type: ignore[union-attr]

    def _xt(actions: Actions) -> Features:
        return pd.DataFrame({col: model.rate(actions)}, index=actions.index)  # type: ignore[union-attr]

    transformer = simple(_xt)
    transformer.__name__ = col
    return [transformer]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Checkpoint** — tests green; do not commit (batched into Task 9).

---

### Task 2: Wire standard exports + default-list guard

**Files:**
- Modify: `silly_kicks/vaep/features/__init__.py` (add import + `__all__` entry)
- Modify: `silly_kicks/vaep/__init__.py` (re-export)
- Test: `tests/vaep/test_xt_feature.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/vaep/test_xt_feature.py`)

```python
def test_exported_from_features_package() -> None:
    from silly_kicks.vaep import features as fs
    assert hasattr(fs, "xt_xfns")
    assert "xt_xfns" in fs.__all__


def test_exported_from_vaep_package() -> None:
    import silly_kicks.vaep as v
    assert hasattr(v, "xt_xfns")
    assert "xt_xfns" in v.__all__


def test_not_in_any_default_list(fitted_xt: xt.ExpectedThreat) -> None:
    """Opt-in invariant: the produced transformer is in no default/union xfn list."""
    from silly_kicks.vaep import base, hybrid
    produced = xt_xfns(model=fitted_xt)[0]
    for lst in (
        base.xfns_default,
        base.xfns_default_no_goalscore,
        hybrid.hybrid_xfns_default,
        hybrid.hybrid_xfns_default_no_goalscore,
    ):
        assert produced not in lst
        # also: no xfn already in the defaults emits an xt__ column
        names = [getattr(f, "__name__", "") for f in lst]
        assert not any(n.startswith("xt__") for n in names)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py -k "exported or not_in_any_default" -v`
Expected: FAIL — `AttributeError: module 'silly_kicks.vaep.features' has no attribute 'xt_xfns'`.

- [ ] **Step 3: Add the import + `__all__` entry in `silly_kicks/vaep/features/__init__.py`**

After the existing `from .temporal import *  # noqa: F403` line (currently line 24), add:

```python
from .expected_threat import xt_xfns
```

And add `"xt_xfns",` to the static `__all__` list (keep alphabetised — insert after `"team",` near the end, i.e. between `"team"` and `"time"` is wrong alphabetically; place after `"time_delta",` as the last entry, then re-sort: it sorts after `"time_delta"`). Final tail of `__all__`:

```python
    "team",
    "time",
    "time_delta",
    "xt_xfns",
]
```

- [ ] **Step 4: Re-export from `silly_kicks/vaep/__init__.py`**

Change the import block and `__all__`:

```python
from . import features, formula, labels
from .base import VAEP, xfns_default_no_goalscore
from .features import xt_xfns
from .hybrid import HybridVAEP, hybrid_xfns_default_no_goalscore

__all__ = [
    "VAEP",
    "HybridVAEP",
    "features",
    "formula",
    "hybrid_xfns_default_no_goalscore",
    "labels",
    "xfns_default_no_goalscore",
    "xt_xfns",
]
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py -v`
Expected: PASS (9 passed).

- [ ] **Step 6: Checkpoint** — green; no commit.

---

### Task 3: Standard VAEP integration + e2e

**Files:**
- Test: `tests/vaep/test_xt_feature.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_vaep_integration_adds_column(fitted_xt: xt.ExpectedThreat) -> None:
    """compute_features with xt_xfns appended produces the xt__ columns, dtype float."""
    from silly_kicks.vaep import VAEP, features as fs
    actions = _moves(n_per_zone=15, seed=2)
    game = pd.Series({"game_id": 1, "home_team_id": 1})
    v = VAEP(xfns=fs.xfns_default + xt_xfns(model=fitted_xt), nb_prev_actions=3)
    X = v.compute_features(game, actions)
    for c in ("xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"):
        assert c in X.columns
        assert X[c].dtype == np.float64


@pytest.mark.filterwarnings("ignore")
def test_e2e_worldcup_finite_for_moves(sb_worldcup_data) -> None:
    """On the committed WC2018 fixture: finite xT for successful moves, NaN for shots."""
    from tests._xthreat_helpers import _worldcup_ltr
    import silly_kicks.spadl.config as cfg
    ltr = _worldcup_ltr(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(ltr)
    one_game = ltr[ltr.game_id == ltr.game_id.iloc[0]].copy()
    out = xt_xfns(model=model)[0](fs.gamestates(one_game, 1))
    col = out["xt__singh_counts_a0"]
    is_succ_move = one_game.type_id.isin(
        [cfg.actiontype_id[t] for t in ("pass", "dribble", "cross")]
    ) & (one_game.result_id == cfg.result_id["success"])
    assert np.isfinite(col[is_succ_move.to_numpy()]).any()
    is_shot = (one_game.type_id == cfg.actiontype_id["shot"]).to_numpy()
    assert col[is_shot].isna().all()
```

- [ ] **Step 2: Run to verify failure (expected: passes immediately if Tasks 1-2 correct)**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py -k "integration or e2e" -v`
Expected: these are behavioral confirmations of already-built code — they should PASS. If `test_vaep_integration_adds_column` fails on a column-name or dtype mismatch, fix the factory (Task 1) before proceeding; if it passes, the integration is confirmed.

- [ ] **Step 3: (only if a test failed) triage then fix.** These are behavioral confirmations of code built in Tasks 1-2; a failure most likely points at the factory (column name / dtype / NaN handling), not the test. Fix the factory, not the assertion. No new implementation expected.

- [ ] **Step 4: Run full standard file**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Checkpoint** — green; no commit.

---

### Task 4: Atomic `xt_xfns` factory + `_atomic_xt_delta_map` helper

**Files:**
- Modify: `silly_kicks/atomic/vaep/features.py`
- Test: `tests/atomic/test_atomic_xt_feature.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/atomic/test_atomic_xt_feature.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import silly_kicks.atomic.spadl.config as acfg
import silly_kicks.xthreat as xt
from silly_kicks.atomic.vaep import features as afs
from silly_kicks.atomic.vaep.features import xt_xfns
from tests._xthreat_helpers import _moves


@pytest.fixture(scope="module")
def fitted_xt() -> xt.ExpectedThreat:
    return xt.ExpectedThreat().fit(_moves(n_per_zone=40, seed=0))


def _atomic_row(action_id, type_id, x, y, dx, dy, *, game_id=1, period_id=1, team_id=1):
    return dict(
        game_id=game_id, period_id=period_id, action_id=action_id, original_event_id=str(action_id),
        time_seconds=float(action_id), team_id=team_id, player_id=1,
        x=x, y=y, dx=dx, dy=dy, bodypart_id=0, bodypart_name="foot",
        type_id=type_id, type_name=acfg.actiontype_name[type_id], result_id=-1,
    )


def test_atomic_fail_closed() -> None:
    with pytest.raises(ValueError):
        xt_xfns(model=None)
    with pytest.raises(NotImplementedError):
        xt_xfns(model="default")
    with pytest.raises(NotFittedError):
        xt_xfns(model=xt.ExpectedThreat())


def test_dribble_is_always_finite(fitted_xt: xt.ExpectedThreat) -> None:
    """A dribble atom (never followed by receival) must still get a finite delta."""
    atomic = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["dribble"], 30.0, 34.0, 15.0, 0.0),
        _atomic_row(1, acfg.actiontype_id["pass"], 45.0, 34.0, 20.0, 0.0),
    ])
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(atomic, 1))
    assert np.isfinite(out["xt__singh_counts_a0"].iloc[0])


def test_pass_success_iff_next_receival(fitted_xt: xt.ExpectedThreat) -> None:
    succ = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["pass"], 30.0, 34.0, 20.0, 0.0),
        _atomic_row(1, acfg.actiontype_id["receival"], 50.0, 34.0, 0.0, 0.0),
    ])
    fail = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["pass"], 30.0, 34.0, 20.0, 0.0),
        _atomic_row(1, acfg.actiontype_id["interception"], 50.0, 34.0, 0.0, 0.0, team_id=2),
    ])
    out_s = xt_xfns(model=fitted_xt)[0](afs.gamestates(succ, 1))
    out_f = xt_xfns(model=fitted_xt)[0](afs.gamestates(fail, 1))
    assert np.isfinite(out_s["xt__singh_counts_a0"].iloc[0])
    assert np.isnan(out_f["xt__singh_counts_a0"].iloc[0])


def test_non_move_and_period_last_are_nan(fitted_xt: xt.ExpectedThreat) -> None:
    df = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["shot"], 100.0, 34.0, 5.0, 0.0),   # non-move
        _atomic_row(1, acfg.actiontype_id["pass"], 30.0, 34.0, 20.0, 0.0),   # last action -> no follow-up
    ])
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(df, 1))
    assert np.isnan(out["xt__singh_counts_a0"].iloc[0])  # shot
    assert np.isnan(out["xt__singh_counts_a0"].iloc[1])  # period-last pass


def test_column_name_symmetry(fitted_xt: xt.ExpectedThreat) -> None:
    cols = afs.feature_column_names(xt_xfns(model=fitted_xt), nb_prev_actions=3)
    assert cols == ["xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py -v`
Expected: FAIL — `ImportError: cannot import name 'xt_xfns' from 'silly_kicks.atomic.vaep.features'`.

- [ ] **Step 3: Add imports to `silly_kicks/atomic/vaep/features.py`**

In the import block (after `import silly_kicks.atomic.spadl.config as atomicspadl`, currently line 6), add:

```python
import silly_kicks.spadl.config as stdspadl
from sklearn.exceptions import NotFittedError
```

(Keep ruff import-ordering: `from sklearn.exceptions import NotFittedError` goes with third-party imports above the first-party `import silly_kicks...` block; run `ruff check --fix` in Task 9 to settle ordering.)

- [ ] **Step 4: Implement the atomic factory + helper** (append to `silly_kicks/atomic/vaep/features.py`)

```python
def _atomic_xt_delta_map(a0: pd.DataFrame, model) -> dict[tuple, float]:
    """Per-action xT delta on the unshifted atomic stream, keyed by (game,period,action_id).

    Type-aware success (dribble intrinsic; pass/cross next-atom-``receival``); reuses
    ``model.rate`` via a synthesized standard-SPADL frame. See NOTICE for citations.
    """
    move_names = ("pass", "dribble", "cross")
    move_ids = [atomicspadl.actiontype_id[n] for n in move_names]          # atomic ids
    type_id = a0["type_id"].to_numpy()
    is_move = np.isin(type_id, move_ids)
    if not is_move.any():
        return {}

    # Next-atom success for pass/cross (within same game+period); dribble always successful.
    n = len(a0)
    next_type = np.full(n, -1, dtype=type_id.dtype)
    next_type[:-1] = type_id[1:]
    game = a0["game_id"].to_numpy()
    period = a0["period_id"].to_numpy()
    next_game = np.full(n, -1, dtype=game.dtype)
    next_game[:-1] = game[1:]
    next_period = np.full(n, -1, dtype=period.dtype)
    next_period[:-1] = period[1:]
    same_gp = (next_game == game) & (next_period == period)

    receival_id = atomicspadl.actiontype_id["receival"]
    dribble_id = atomicspadl.actiontype_id["dribble"]
    is_dribble = type_id == dribble_id
    is_passcross = is_move & ~is_dribble
    success = is_dribble | (is_passcross & same_gp & (next_type == receival_id))

    move_idx = np.flatnonzero(is_move)
    sub = a0.iloc[move_idx]
    synth = pd.DataFrame(
        {
            # Map atomic move type -> standard move type by NAME. The ids currently coincide
            # (pass/dribble/cross == 0/21/1 in both configs), but keep the name-map deliberate:
            # do NOT "simplify" to raw ids — that would silently break if a future config (GS /
            # SkillCorner) ever renumbers. (The test oracle's _geo_key relies on the id-equality
            # only as a representation-stable test key, a separate concern from this conversion.)
            "type_id": [stdspadl.actiontype_id[atomicspadl.actiontype_name[int(t)]] for t in type_id[move_idx]],
            "result_id": np.where(success[move_idx], stdspadl.result_id["success"], stdspadl.result_id["fail"]),
            "start_x": sub["x"].to_numpy(dtype=float),
            "start_y": sub["y"].to_numpy(dtype=float),
            "end_x": (sub["x"] + sub["dx"]).to_numpy(dtype=float),
            "end_y": (sub["y"] + sub["dy"]).to_numpy(dtype=float),
        }
    )
    # NOTE: synth feeds atomic (x, y) / (x+dx, y+dy) into model.rate(), which bins via
    # silly_kicks.spadl.config field dims. Correct because atomic SPADL shares the standard
    # 105x68 pitch frame (atomicspadl.field_length/width == spadl.config.field_length/width).
    deltas = model.rate(synth)  # ndarray len(move_idx); NaN for failed/NaN-coord rows
    # int-cast the key parts: gamestates' shift() can upcast id columns int->float on slots
    # a1/a2, so building/looking-up keys as ints keeps them comparable regardless of dtype.
    keys = [(int(g), int(p), int(a)) for g, p, a in zip(sub["game_id"], sub["period_id"], sub["action_id"], strict=False)]
    # dict() collapses duplicate keys (e.g. the all-zero dummy used by feature_column_names),
    # so this never raises on non-unique keys the way a reindex/merge would.
    return dict(zip(keys, deltas, strict=False))


def xt_xfns(*, model=None) -> list[FeatureTransfomer]:
    """Atomic mirror of the standard ``xt_xfns``: emits ``xt__<model.method>_a{i}``.

    Type-aware success + ``model.rate`` reuse via a synthesized standard frame; maps to
    slots by the composite (game_id, period_id, action_id) key. Opt-in (not in any
    default list). See ``silly_kicks.vaep.features.expected_threat.xt_xfns`` and NOTICE.

    Raises
    ------
    ValueError, NotImplementedError, NotFittedError
        If ``model`` is not a fitted ExpectedThreat.

    Examples
    --------
    >>> # xfns = xt_xfns(model=fitted_xt)
    """
    if isinstance(model, str):
        raise NotImplementedError(
            "xt_xfns: bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat."
        )
    if model is None:
        raise ValueError("xt_xfns requires a fitted ExpectedThreat (model=...).")
    if not np.any(model.xT):
        raise NotFittedError("xt_xfns requires a fitted ExpectedThreat; call model.fit(actions) first.")

    col = f"xt__{model.method}"

    def _xt(states: GameStates) -> Features:
        a0 = states[0]
        delta_map = _atomic_xt_delta_map(a0, model)
        out = pd.DataFrame(index=a0.index)
        for i, slot in enumerate(states):
            # int-cast to match the int keys built in _atomic_xt_delta_map (shift() may have
            # upcast these id columns to float on slots a1/a2; boundary rows are filled, never NaN).
            keys = [(int(g), int(p), int(a)) for g, p, a in zip(slot["game_id"], slot["period_id"], slot["action_id"], strict=False)]
            out[f"{col}_a{i}"] = pd.Series([delta_map.get(k, np.nan) for k in keys], index=slot.index)
        return out

    _xt.__name__ = col
    return [_xt]
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Checkpoint** — green; no commit.

---

### Task 5: Atomic exports

**Files:**
- Modify: `silly_kicks/atomic/vaep/features.py` (`__all__`)
- Modify: `silly_kicks/atomic/vaep/__init__.py` (re-export)
- Test: `tests/atomic/test_atomic_xt_feature.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_atomic_exports() -> None:
    from silly_kicks.atomic.vaep import features as afs2
    assert "xt_xfns" in afs2.__all__
    import silly_kicks.atomic.vaep as av
    assert hasattr(av, "xt_xfns")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py::test_atomic_exports -v`
Expected: FAIL — `assert 'xt_xfns' in [...]`.

- [ ] **Step 3: Add `"xt_xfns"` to `__all__` in `silly_kicks/atomic/vaep/features.py`**

Insert alphabetically into the static `__all__` (after `"time_delta",` as the last entry):

```python
    "time",
    "time_delta",
    "xt_xfns",
]
```

- [ ] **Step 4: Re-export from `silly_kicks/atomic/vaep/__init__.py`**

```python
"""Implements the Atomic-VAEP framework."""

from . import features, formula, labels
from .base import AtomicVAEP
from .features import xt_xfns

__all__ = ["AtomicVAEP", "features", "formula", "labels", "xt_xfns"]
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Checkpoint** — green; no commit.

---

### Task 6: Universal cross-representation symmetry oracle + composite-key + boundary

**Files:**
- Test: `tests/atomic/test_atomic_xt_feature.py` (append)

- [ ] **Step 1: Write the tests** (append)

```python
def _converted_pair(sb_worldcup_data):
    """One WC2018 game as standard-LTR SPADL and its atomic conversion."""
    from tests._xthreat_helpers import _worldcup_ltr
    from silly_kicks.atomic.spadl.base import convert_to_atomic
    ltr = _worldcup_ltr(sb_worldcup_data)
    one = ltr[ltr.game_id == ltr.game_id.iloc[0]].copy().reset_index(drop=True)
    return one, convert_to_atomic(one)


def _geo_key(type_id, sx, sy, ex, ey):
    """Representation-stable key. The xT delta is a pure function of (start_zone, end_zone),
    and atomic (x, x+dx) == standard (start_x, end_x) by construction, so geometry matches
    across representations even though convert_to_atomic RENUMBERS action_id (so action_id is
    NOT a valid cross-representation key). pass/dribble/cross share ids 0/21/1 across the
    standard and atomic configs, so the raw type_id is also stable in the key."""
    return (int(type_id), round(float(sx), 3), round(float(sy), 3), round(float(ex), 3), round(float(ey), 3))


# pass/dribble/cross ids (identical in standard and atomic configs)
_MOVE_IDS = {acfg.actiontype_id[n] for n in ("pass", "dribble", "cross")}


@pytest.mark.filterwarnings("ignore")
def test_symmetry_oracle_value_agreement(sb_worldcup_data) -> None:
    """For any move action present-and-finite in BOTH representations (matched by GEOMETRY),
    the atomic delta equals the standard rate() delta — across slots a0/a1/a2. Validates the
    coordinate frame, the y-flip, and the grid lookup. Robust by design: it compares only the
    intersection of finite move-deltas, so inherent success-encoding edges (period-last
    pass/cross, out/offside follow-ups) simply fall out of the intersection rather than
    spuriously failing the build."""
    from silly_kicks.vaep import features as sfs
    from silly_kicks.vaep.features import xt_xfns as std_xt_xfns
    std, atomic = _converted_pair(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(std)

    std_states, atomic_states = sfs.gamestates(std, 3), afs.gamestates(atomic, 3)
    std_out = std_xt_xfns(model=model)[0](std_states)
    atomic_out = xt_xfns(model=model)[0](atomic_states)

    for i in range(3):
        s, a = std_states[i], atomic_states[i]
        s_geo = {
            _geo_key(t, sx, sy, ex, ey): d
            for t, sx, sy, ex, ey, d in zip(
                s.type_id, s.start_x, s.start_y, s.end_x, s.end_y, std_out[f"xt__singh_counts_a{i}"], strict=False
            )
            if int(t) in _MOVE_IDS and np.isfinite(d)
        }
        a_geo = {
            _geo_key(t, x, y, x + dx, y + dy): d
            for t, x, y, dx, dy, d in zip(
                a.type_id, a.x, a.y, a.dx, a.dy, atomic_out[f"xt__singh_counts_a{i}"], strict=False
            )
            if int(t) in _MOVE_IDS and np.isfinite(d)
        }
        common = set(s_geo) & set(a_geo)
        # K>=3 (not just non-empty): a0's intersection is large; a1/a2 are thinner (the atomic
        # "previous atom" is often a filtered receival). On a full WC2018 game all three slots
        # clear this easily — a future fixture swap that shrinks the game should fail LOUDLY here
        # rather than pass on a one-element intersection.
        assert len(common) >= 3, f"slot a{i}: too few shared finite move geometries ({len(common)})"
        for k in common:
            # atol only — a bin-flip would give a non-tiny diff, not a 1-ULP one. Theoretical
            # edge: atomic end = x + (end-start) is within ~1 ULP of standard end_x, so a coord
            # sitting EXACTLY on a cell boundary (k*105/16) could bin differently. StatsBomb
            # coords (1-2 decimals) never land on those irrational edges, so this is inert here.
            assert np.isclose(s_geo[k], a_geo[k], rtol=0, atol=1e-9), f"slot a{i} delta mismatch at {k}"


@pytest.mark.filterwarnings("ignore")
def test_symmetry_oracle_dribbles_finite(sb_worldcup_data) -> None:
    """KEYSTONE (round-1 critical invariant on real data): every standard dribble has a finite,
    equal atomic counterpart (geometry-matched). A blanket next-atom-receival predicate would
    NaN all atomic dribbles and fail here. Dribbles have no success-encoding ambiguity (always
    successful both representations), so this is a clean hard gate — do NOT weaken it."""
    import silly_kicks.spadl.config as scfg
    from silly_kicks.vaep import features as sfs
    from silly_kicks.vaep.features import xt_xfns as std_xt_xfns
    std, atomic = _converted_pair(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(std)
    dribble_std = scfg.actiontype_id["dribble"]
    dribble_atom = acfg.actiontype_id["dribble"]

    std_out = std_xt_xfns(model=model)[0](sfs.gamestates(std, 1))["xt__singh_counts_a0"].to_numpy()
    atomic_out = xt_xfns(model=model)[0](afs.gamestates(atomic, 1))["xt__singh_counts_a0"].to_numpy()
    a_geo = {
        _geo_key(t, x, y, x + dx, y + dy): d
        for t, x, y, dx, dy, d in zip(atomic.type_id, atomic.x, atomic.y, atomic.dx, atomic.dy, atomic_out, strict=False)
        if int(t) == dribble_atom
    }
    n_checked = 0
    for idx in np.flatnonzero((std.type_id == dribble_std).to_numpy()):
        d_std = std_out[idx]
        if not np.isfinite(d_std):
            continue
        k = _geo_key(dribble_std, std.start_x.iloc[idx], std.start_y.iloc[idx], std.end_x.iloc[idx], std.end_y.iloc[idx])
        assert k in a_geo and np.isfinite(a_geo[k]), f"atomic dribble missing/NaN at {k}"
        assert np.isclose(a_geo[k], d_std, rtol=0, atol=1e-9)
        n_checked += 1
    assert n_checked > 0, "fixture had no dribbles to check"


def test_multi_game_composite_key(fitted_xt: xt.ExpectedThreat) -> None:
    """Two games with overlapping action_id ranges must not cross-contaminate (the composite
    key includes game_id; bare-action_id keying would alias game-1 and game-2 row 0)."""
    g1 = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["dribble"], 20.0, 34.0, 15.0, 0.0, game_id=1),
        _atomic_row(1, acfg.actiontype_id["pass"], 35.0, 34.0, 20.0, 0.0, game_id=1),
    ])
    g2 = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["dribble"], 80.0, 60.0, 5.0, 0.0, game_id=2),
        _atomic_row(1, acfg.actiontype_id["pass"], 85.0, 60.0, 5.0, 0.0, game_id=2),
    ])
    both = pd.concat([g1, g2], ignore_index=True)
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(both, 1))["xt__singh_counts_a0"].to_numpy()
    assert not np.isclose(out[0], out[2])  # different start zones -> different deltas


def test_boundary_a1_is_map_hit_not_nan(fitted_xt: xt.ExpectedThreat) -> None:
    """A boundary a1 row is filled with the first-in-group action (gamestates), so its composite
    key is present -> the atomic loop must emit that action's finite delta, NOT NaN. Guards the
    int/float composite-key dtype handling AND the no-boundary-NaN decision (symmetry vs standard)."""
    df = pd.DataFrame([
        _atomic_row(0, acfg.actiontype_id["dribble"], 25.0, 34.0, 15.0, 0.0),
        _atomic_row(1, acfg.actiontype_id["pass"], 40.0, 34.0, 20.0, 0.0),
        _atomic_row(2, acfg.actiontype_id["receival"], 60.0, 34.0, 0.0, 0.0),
    ])
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(df, 2))
    # row 0 is a group boundary; a1 is filled with the first-in-group action (the dribble) -> finite
    assert np.isfinite(out["xt__singh_counts_a1"].iloc[0])
```

- [ ] **Step 2: Run the oracles**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py -k "symmetry or composite or boundary" -v`
Expected: PASS. **Triage discipline if a symmetry oracle fails:** first determine whether the failure is a *test-key* problem (geometry rounding, a type-id assumption) or a *real production asymmetry* (a genuine atomic-vs-standard delta difference). NEVER loosen the assertion to make it pass — `test_symmetry_oracle_dribbles_finite` is the keystone for the round-1 dribble bug, and weakening it silently re-admits that bug. A real mismatch points at `_atomic_xt_delta_map` (success predicate, type-id map, or coordinate frame).

- [ ] **Step 3: (only if a real production asymmetry is confirmed) fix `_atomic_xt_delta_map` and re-run.**

- [ ] **Step 4: Run full atomic file**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Checkpoint** — green; no commit.

---

### Task 7: Atomic VAEP integration + atomic e2e

**Files:**
- Test: `tests/atomic/test_atomic_xt_feature.py` (append)

- [ ] **Step 1: Write the tests** (append)

```python
@pytest.mark.filterwarnings("ignore")
def test_atomic_vaep_integration(sb_worldcup_data) -> None:
    """AtomicVAEP.compute_features with xt_xfns appended produces the columns."""
    from silly_kicks.atomic.vaep import AtomicVAEP
    from silly_kicks.atomic.vaep import features as afs2
    std, atomic = _converted_pair(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(std)
    game = pd.Series({"game_id": int(atomic.game_id.iloc[0]), "home_team_id": int(atomic.team_id.iloc[0])})
    v = AtomicVAEP(xfns=[afs2.location, afs2.actiontype_onehot] + xt_xfns(model=model), nb_prev_actions=3)
    X = v.compute_features(game, atomic)
    for c in ("xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"):
        assert c in X.columns
        assert X[c].dtype == np.float64
    # prove the pipeline actually RATES something — guards against an all-NaN integration
    # (e.g. a column-name drift or unexpected LTR mirror through compute_features that the
    # raw-transformer oracle wouldn't catch).
    assert np.isfinite(X["xt__singh_counts_a0"]).any()
```

- [ ] **Step 2: Run to verify**

Run: `.venv/Scripts/python -m pytest tests/atomic/test_atomic_xt_feature.py::test_atomic_vaep_integration -v`
Expected: PASS. (If `AtomicVAEP(xfns=...)` rejects the custom list or a default xfn is required, consult `silly_kicks/atomic/vaep/base.py::AtomicVAEP.__init__` and use its accepted xfns parameter shape; adjust the supporting xfns in the list, not `xt_xfns`.)

- [ ] **Step 3: Run both new test files**

Run: `.venv/Scripts/python -m pytest tests/vaep/test_xt_feature.py tests/atomic/test_atomic_xt_feature.py -v`
Expected: PASS (all).

- [ ] **Step 4: Checkpoint** — green; no commit.

---

### Task 8: Full-suite + lint/type green

**Files:** none (verification gate)

- [ ] **Step 1: Run the full non-e2e suite**

Run: `.venv/Scripts/python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Expected: all pass (no regressions). In particular `tests/test_xthreat.py` and the SK-xT-1 parity gates stay green (we did not touch `rate()`).

- [ ] **Step 2: Lint (replicate CI exactly)**

Run: `.venv/Scripts/ruff check silly_kicks/ tests/ scripts/`
Run: `.venv/Scripts/ruff format --check silly_kicks/ tests/ scripts/`
Expected: clean. If import-order (I001) flags `expected_threat.py` / atomic imports, run `.venv/Scripts/ruff check --fix silly_kicks/ tests/`.

- [ ] **Step 3: Type-check the whole package**

Run: `.venv/Scripts/pyright silly_kicks/`
Expected: clean. The duck-typed `model` uses `# type: ignore[union-attr]` on `.method`/`.rate` (already in the factory code); if pyright flags the atomic `model` param, annotate it `model: "ExpectedThreat | str | None" = None` with a `TYPE_CHECKING` import of `ExpectedThreat` in `atomic/vaep/features.py` mirroring the standard module.

- [ ] **Step 4: Checkpoint** — all gates green; no commit.

---

### Task 9: Ship metadata + single commit (gated on explicit approval)

**Files:**
- Create: `docs/superpowers/adrs/ADR-<next>-xt-vaep-feature.md`
- Modify: `CLAUDE.md`, `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`

- [ ] **Step 0: Confirm the spec matches the shipped code (reconcile if drifted)**

Run: `grep -n "_rate_cells\|parity" docs/superpowers/specs/2026-06-08-xt-vaep-feature-design.md`
Confirm every hit is in the **rejected/"dropped"** framing (Scope §3 is titled "No `rate()` refactor / no `_rate_cells` extraction"; the only Alternatives row marks it rejected; Success criteria says "rate() left unchanged"). This was reconciled during brainstorming round 2 — it should already be clean. If any hit frames `_rate_cells` as *in-scope* or describes the atomic mechanism around it, fix it to the synthesized-`result_id` reuse so the committed design record cannot contradict the shipped code in the single squash.

- [ ] **Step 1: Determine the next free ADR number**

Run: `ls docs/superpowers/adrs/`
Pick the next unused integer (e.g. if highest is `ADR-021`, use `ADR-022`). Create `docs/superpowers/adrs/ADR-<next>-xt-vaep-feature.md` following `docs/superpowers/adrs/ADR-TEMPLATE.md`, recording the 5 decisions: (a) caller-frozen-model train/serve contract + fail-closed; (b) opt-in, not in any default list; (c) atomic type-aware success (dribble intrinsic; pass/cross next-atom-`receival`) reusing **`model.rate()` via a synthesized `result_id`** — explicitly NOT a `_rate_cells` extraction (`rate()` is left untouched; this is the architecturally interesting choice and supersedes the brainstorming-era draft); (d) the documented last-action-of-period NaN edge; (e) boundary map-by-composite-key symmetry.

- [ ] **Step 2: Reconcile the version**

Run: `git fetch origin && git describe --tags --abbrev=0 origin/main`
Set the next free **minor** version (e.g. `4.19.0` if `4.18.0` is latest) in: `pyproject.toml` (`version =`), `silly_kicks/__init__.py` (`__version__`), top of `CHANGELOG.md` (new section with the feature summary + "opt-in; no forced retrain; opting in is a self-triggered VAEP retrain"), and the relevant `TODO.md` line (mark this SK-xT-1 follow-up done). Then run `.venv/Scripts/python -m pip install -q uv` (if needed) and `uv lock` to refresh `uv.lock`.

- [ ] **Step 3: Add the CLAUDE.md feature line**

Under the xT (`xthreat/`) architecture bullet, append a sentence: SK-xT-N ships the opt-in `xt__<method>` VAEP xfn factory (`vaep.features.xt_xfns` + atomic mirror), caller-passes-fitted-model, not in any default list (additive, no forced retrain). Confirm **C4-free** (no tracking backend/model/aggregator change → tokens + count unchanged; skip regen).

- [ ] **Step 4: Final full verification before commit**

Run: `.venv/Scripts/python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Run: `.venv/Scripts/ruff check silly_kicks/ tests/ scripts/ && .venv/Scripts/ruff format --check silly_kicks/ tests/ scripts/ && .venv/Scripts/pyright silly_kicks/`
Expected: all green.

- [ ] **Step 5: Present diff + commit command to the user and HOLD for explicit approval**

Do NOT create the commit sentinel or commit without an explicit per-commit "yes". Present `git status` + `git diff --stat` + the proposed commit message (written to a temp file, committed via `git commit -F`), then wait. After approval and commit, offer push + PR (chat-approval gated).

---

## Self-Review

**Spec coverage:** standard factory (Task 1) ✓; atomic mirror with type-aware success + synthesized-frame `rate()` reuse + composite-key map (Tasks 4,6) ✓; fail-closed (Tasks 1,4) ✓; column derived from `model.method` (Tasks 1,4) ✓; NaN pass-through (Tasks 1,4) ✓; opt-in + default-list guard (Task 2) ✓; geometry-keyed value-agreement oracle (a0/a1/a2) + dribble keystone oracle + multi-game composite-key + boundary-map-hit (Task 6) ✓; standard + atomic e2e (Tasks 3,7) ✓; `rate()` untouched / no `_rate_cells` ✓ (no task modifies `rate()`); ADR records 5 decisions (Task 9) ✓; C4-free + version + CLAUDE.md (Task 9) ✓; pitch-frame: explicit comment in `_atomic_xt_delta_map` + covered by the value-agreement oracle (fails on coordinate drift) ✓.

**Oracle keying note:** cross-representation matching is by **geometry** (`type_id` + rounded start/end coords), NOT `action_id` — `convert_to_atomic` renumbers `action_id` (inserts synthetic extras then `action_id = range(len)`), so `action_id` is not stable across representations. The xT delta is a pure function of (start_zone, end_zone), so geometry is the correct stable key. The dribble oracle is the keystone and must never be loosened.

**Placeholder scan:** ADR number + version number are resolved by explicit commands at implementation time (not placeholders); all code/test steps contain complete content.

**Type consistency:** `xt_xfns(*, model)` signature identical across standard/atomic; column `xt__<model.method>`; `_atomic_xt_delta_map(a0, model)` returns `dict[tuple,float]`; helper names match between Task 4 definition and Task 6 usage.
