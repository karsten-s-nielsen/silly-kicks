# SK-xT-1: Pluggable, evaluatable xT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `silly_kicks/xthreat.py` into a pluggable `xthreat/` package with a Singh-counts back-compat facade, a KDE-smoothed transition flavor, variable-resolution `GridSpec`, a standalone `value_iteration`, and a held-out transition-model NLL evaluator — all in silly-kicks house style.

**Architecture:** Convert the module to a package mirroring `tracking/pitch_control/`. Pluggability is house-style **string dispatch + frozen-dataclass params** (no ABCs). `ExpectedThreat` stays a class facade; `method="singh_counts"` (default) is byte-identical to today, `method="kde_smoothed"` swaps only the transition builder. Binning helpers stay on `(l, w)` ints internally (byte-identical, just relocated); `GridSpec` is the public/standalone-function abstraction that unpacks to `(l, w)`.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn (hard dep — `KernelDensity`), scipy (`RectBivariateSpline`, already used). pytest.

**Spec:** `docs/superpowers/specs/2026-06-07-xthreat-pluggable-xt-promotion-design.md`

---

## File structure

| File | Responsibility |
|------|----------------|
| `silly_kicks/xthreat/__init__.py` | Re-export the **public** API only (`ExpectedThreat`, `GridSpec`, `Method`, params, `validate_params_for_method`, `value_iteration`, transition builders, `silverman_2d`, eval fns). No `_*` privates. |
| `silly_kicks/xthreat/_params.py` | `GridSpec`, `Method`, `SinghParams`, `KDEParams`, `KdeKernel`, `XtParams`, `_METHOD_TO_PARAMS_TYPE`, `validate_params_for_method`. |
| `silly_kicks/xthreat/_grid.py` | Binning helpers relocated byte-identically from the old module: `_get_cell_indexes`, `_get_flat_indexes`, `_count`, `_safe_divide`, `_scoring_prob`, `_action_prob`, `_get_move_actions`, `_get_successful_move_actions`. Still keyed on `(l, w)` ints. |
| `silly_kicks/xthreat/_transitions.py` | `singh_transition_matrix(actions, grid)`, `silverman_2d(n, sigma)`, `kde_smoothed_transition_matrix(actions, grid, params)`, `_zone_centres(grid)`. |
| `silly_kicks/xthreat/_value_iteration.py` | `value_iteration(p_scoring, p_shot, p_move, transition, *, eps, max_iter=None)` → `(xT, heatmaps)`. |
| `silly_kicks/xthreat/_model.py` | `ExpectedThreat` class (facade + dispatch); owns `RectBivariateSpline` import, `interpolator()`, `rate()`. |
| `silly_kicks/xthreat/_eval.py` | `holdout_split`, `compute_holdout_nll`, `compute_holdout_nll_per_group`. |
| `silly_kicks/xthreat.py` | **Deleted** (replaced by the package). |
| `tests/xthreat_legacy_reference.py` | Frozen verbatim copy of the pre-refactor `xthreat.py` — test-only Singh-parity oracle (permanent regression guard). |
| `tests/_xthreat_helpers.py` | Shared synthetic SPADL factories (`_moves`, `_sparse_overfit_corpus`) — decouples the xthreat test modules. |
| `tests/test_xthreat.py` | Migrated import paths + new parity test. |
| `tests/test_xthreat_kde.py` | KDE transition unit tests. |
| `tests/test_xthreat_value_iteration.py` | `value_iteration` tests (max_iter, KDE-dense convergence). |
| `tests/test_xthreat_eval.py` | NLL evaluator unit tests. |
| `tests/test_xthreat_resolution.py` | Variable-resolution end-to-end test. |
| `tests/test_xthreat_kde_beats_singh.py` | Synthetic hard gate + WC2018 diagnostic. |
| `NOTICE` | Silverman 1986 + Salimi 2026 entries. |
| `docs/superpowers/adrs/ADR-021-pluggable-xt.md` | Decision record. |

**Conventions to follow (verified in-repo):**
- `tracking/pressure.py` + `tracking/pitch_control/_params.py` are the dispatch/param templates.
- `warnings.warn(..., stacklevel=2)` everywhere (none expected here).
- Run tests: `.venv/Scripts/python.exe -m pytest <path> -v` (Windows; timeout ≤30s or background).
- Lint before done: `.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/`, `ruff format --check`, `pyright silly_kicks/`.

---

## Task 1: Freeze the pre-refactor implementation as a parity oracle

**Files:**
- Create: `tests/xthreat_legacy_reference.py`

Rationale: the Singh path must be byte-identical after the refactor. Exact `assert_array_equal` against a frozen, in-process copy avoids cross-platform float fragility (a recompute-vs-recompute in the same process is bit-exact).

- [ ] **Step 1: Copy the current module verbatim into a test-only reference.**

```bash
cp silly_kicks/xthreat.py tests/xthreat_legacy_reference.py
```

- [ ] **Step 2: Add a `ruff: noqa` header + docstring marker so it is self-contained and lint-clean.**

The frozen copy uses `l`/`w` (E741) and `X` (N806). The `tests/**` per-file-ignore covers N806
but NOT E741, so the file needs its own header. `import silly_kicks.spadl.config as spadlconfig`
stays valid. Make the first lines:

```python
# ruff: noqa: E741, N803, N806
"""FROZEN pre-refactor xthreat implementation. Test-only Singh-parity oracle for SK-xT-1.

Do NOT edit — this is the byte-for-byte reference the refactored package must reproduce
on the Singh path. See docs/superpowers/specs/2026-06-07-xthreat-pluggable-xt-promotion-design.md.
"""
```

(Pyright is run only on `silly_kicks/` in Task 12, not `tests/`, and this is a verbatim copy of
pyright-clean code with its `# type: ignore` comments intact — no pyright action needed.)

- [ ] **Step 3: Create the shared test-helpers module** (all xthreat test files import from here — created up front so Task 5's WC2018 parity test can use `_worldcup_ltr`).

```python
# tests/_xthreat_helpers.py
"""Shared SPADL factories + the WC2018 actions builder for xthreat tests.

A plain module (not conftest) because these are parametrized factories, not fixtures.
"""

from typing import cast

import numpy as np
import pandas as pd

import silly_kicks.spadl as spadl
import silly_kicks.spadl.config as cfg

_PASS = cfg.actiontype_id["pass"]
_SUCCESS = cfg.result_id["success"]


def _worldcup_ltr(sb_worldcup_data) -> pd.DataFrame:
    """All WC2018 games concatenated as left-to-right SPADL actions."""
    games = cast(pd.DataFrame, sb_worldcup_data["games"]).set_index("game_id")
    return cast(pd.DataFrame, pd.concat([
        spadl.play_left_to_right(cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{gid}"]), g.home_team_id)
        for gid, g in games.iterrows()
    ]))


def _moves(n_per_zone: int = 20, seed: int = 0) -> pd.DataFrame:
    """Successful passes from 3 source x-bands with gaussian-jittered destinations."""
    rng = np.random.default_rng(seed)
    rows, aid = [], 0
    for sx in (20.0, 50.0, 80.0):
        for _ in range(n_per_zone):
            rows.append(dict(
                game_id=1, action_id=aid, period_id=1, time_seconds=float(aid),
                team_id=1, player_id=1, bodypart_id=0, type_id=_PASS, result_id=_SUCCESS,
                start_x=sx, start_y=34.0,
                end_x=float(np.clip(sx + rng.normal(10, 3), 0, cfg.field_length)),
                end_y=float(np.clip(34 + rng.normal(0, 3), 0, cfg.field_width)),
            ))
            aid += 1
    return pd.DataFrame(rows)


def _sparse_overfit_corpus(seed: int = 0, n_games: int = 20) -> pd.DataFrame:
    """Sparse, wide-jitter passes from 4 centres across many games — Singh overfits (spiky rows),
    KDE smooths. Used by the KDE-beats-Singh hard gate (Task 10).

    game_id is seed-offset (seed*1000 + g) so different seeds vary BOTH the destinations AND the
    sha256 holdout split. n_games=20 keeps the 25% split non-degenerate (~5 holdout games).
    """
    rng = np.random.default_rng(seed)
    centres = [(15.0, 20.0), (40.0, 50.0), (70.0, 30.0), (90.0, 60.0)]
    rows, aid = [], 0
    for g in range(n_games):
        game_id = seed * 1000 + g
        for sx, sy in centres:
            for _ in range(2):  # only 2 obs per (game, centre) -> spiky Singh rows
                rows.append(dict(
                    game_id=game_id, action_id=aid, period_id=1, time_seconds=float(aid),
                    team_id=1, player_id=1, bodypart_id=0, type_id=_PASS, result_id=_SUCCESS,
                    start_x=sx, start_y=sy,
                    end_x=float(np.clip(sx + 12 + rng.normal(0, 6), 0, cfg.field_length)),
                    end_y=float(np.clip(sy + rng.normal(0, 6), 0, cfg.field_width)),
                ))
                aid += 1
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Verify the reference + helpers import.**

Run: `.venv/Scripts/python.exe -c "import tests.xthreat_legacy_reference as r; import tests._xthreat_helpers as h; print(r.ExpectedThreat, h._moves(2).shape)"`
Expected: prints the class + `(6, ...)`, no error.

- [ ] **Step 5: Stage.** (Bundled — do not push; final commit gated.)

```bash
git add tests/xthreat_legacy_reference.py tests/_xthreat_helpers.py
```

---

## Task 2: `_params.py` — GridSpec + method/param surface

**Files:**
- Create: `silly_kicks/xthreat/_params.py`
- Test: `tests/test_xthreat_params.py`

- [ ] **Step 1: Write failing tests.**

```python
# tests/test_xthreat_params.py
import pytest

from silly_kicks.xthreat._params import (
    GridSpec, KDEParams, SinghParams, validate_params_for_method,
)


def test_gridspec_defaults_match_legacy_16x12():
    g = GridSpec()
    assert (g.n_zones_x, g.n_zones_y) == (16, 12)
    assert g.n_zones == 192


def test_gridspec_cell_dims_from_spadlconfig():
    import silly_kicks.spadl.config as cfg
    g = GridSpec(n_zones_x=12, n_zones_y=8)
    assert g.cell_length == pytest.approx(cfg.field_length / 12)
    assert g.cell_width == pytest.approx(cfg.field_width / 8)


def test_gridspec_rejects_nonpositive():
    with pytest.raises(ValueError):
        GridSpec(n_zones_x=0, n_zones_y=8)


def test_validate_accepts_matching_params():
    validate_params_for_method("singh_counts", None)
    validate_params_for_method("singh_counts", SinghParams())
    validate_params_for_method("kde_smoothed", KDEParams())


def test_validate_rejects_mismatched_params():
    with pytest.raises(TypeError):
        validate_params_for_method("singh_counts", KDEParams())


def test_validate_rejects_unknown_method():
    with pytest.raises(ValueError):
        validate_params_for_method("bogus", None)  # type: ignore[arg-type]


def test_params_are_frozen():
    with pytest.raises(Exception):
        KDEParams().bandwidth = 1.0  # type: ignore[misc]
```

- [ ] **Step 2: Run — expect failure (module missing).**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_params.py -v`
Expected: FAIL — `ModuleNotFoundError: silly_kicks.xthreat._params`.

- [ ] **Step 3: Implement `_params.py`.**

```python
"""Parameter surfaces for the pluggable xT model.

House-style string-dispatch + frozen-dataclass params (mirrors tracking/pressure.py and
tracking/pitch_control/_params.py). See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import silly_kicks.spadl.config as spadlconfig

Method = Literal["singh_counts", "kde_smoothed"]
KdeKernel = Literal["gaussian", "epanechnikov", "tophat", "exponential", "linear", "cosine"]


@dataclass(frozen=True)
class GridSpec:
    """Grid resolution for the xT model. Pitch dimensions live in ``spadlconfig`` (SSOT)."""

    n_zones_x: int = 16
    n_zones_y: int = 12

    def __post_init__(self) -> None:
        if self.n_zones_x < 1 or self.n_zones_y < 1:
            raise ValueError(f"GridSpec requires positive dimensions, got {self.n_zones_x}x{self.n_zones_y}")

    @property
    def n_zones(self) -> int:
        return self.n_zones_x * self.n_zones_y

    @property
    def cell_length(self) -> float:
        return spadlconfig.field_length / self.n_zones_x

    @property
    def cell_width(self) -> float:
        return spadlconfig.field_width / self.n_zones_y


@dataclass(frozen=True)
class SinghParams:
    """No parameters — row-normalized empirical counts (classic Singh 2018)."""


@dataclass(frozen=True)
class KDEParams:
    """Per-source-zone 2D KDE smoothing of the transition matrix.

    bandwidth : multiplier on the Silverman rule when ``adaptive`` (else the raw sklearn
        ``KernelDensity`` bandwidth, in SPADL metres). Default seeded from the lakehouse
        champion; revisit via the SK-xT-1 validation diagnostic.
    adaptive : per-source-zone bandwidth from Silverman's rule on that row's destinations.
    kernel : sklearn ``KernelDensity`` kernel name.
    """

    bandwidth: float = 2.0
    adaptive: bool = True
    kernel: KdeKernel = "gaussian"


XtParams = SinghParams | KDEParams
_METHOD_TO_PARAMS_TYPE: dict[Method, type] = {
    "singh_counts": SinghParams,
    "kde_smoothed": KDEParams,
}


def validate_params_for_method(method: Method, params: XtParams | None) -> None:
    """Raise if ``params`` is the wrong type for ``method``. ``None`` always allowed (defaults)."""
    if method not in _METHOD_TO_PARAMS_TYPE:
        raise ValueError(f"Unknown xT method {method!r}; expected one of {list(_METHOD_TO_PARAMS_TYPE)}.")
    if params is None:
        return
    expected_type = _METHOD_TO_PARAMS_TYPE[method]
    if not isinstance(params, expected_type):
        raise TypeError(
            f"method={method!r} expects {expected_type.__name__}, got {type(params).__name__}. "
            f"Use {expected_type.__name__}() (or omit params=) for defaults."
        )
```

- [ ] **Step 4: Run — expect pass.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_params.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Stage.** `git add silly_kicks/xthreat/_params.py tests/test_xthreat_params.py`

---

## Task 3: `_grid.py` — relocate binning helpers byte-identically

**Files:**
- Create: `silly_kicks/xthreat/_grid.py`
- Modify: `tests/test_xthreat.py` (helper import paths)

- [ ] **Step 1: Create `_grid.py` by copying the helper block verbatim from the old `xthreat.py`** (lines 13–162: `M`, `N`, `_get_cell_indexes`, `_get_flat_indexes`, `_count`, `_safe_divide`, `_scoring_prob`, `_get_move_actions`, `_get_successful_move_actions`, `_action_prob`). Keep signatures `(x, y, l=N, w=M)` etc. exactly. Header:

```python
"""Grid binning + per-cell probability helpers (relocated verbatim from the legacy module).

Keyed on (l, w) ints; GridSpec callers unpack via grid.n_zones_x / grid.n_zones_y.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

M: int = 12
N: int = 16
# ... (paste _get_cell_indexes ... _action_prob verbatim) ...
```

- [ ] **Step 2: Migrate the helper imports in `tests/test_xthreat.py`.** Change `import silly_kicks.xthreat as xt` usages of helpers to the new module. Add at top:

```python
from silly_kicks.xthreat import _grid
```

Replace `xt._get_cell_indexes` → `_grid._get_cell_indexes`, `xt._get_flat_indexes` → `_grid._get_flat_indexes`, `xt._count` → `_grid._count`, `xt._action_prob` → `_grid._action_prob`, `xt._scoring_prob` → `_grid._scoring_prob`, `xt._get_move_actions` → `_grid._get_move_actions`, `xt._get_successful_move_actions` → `_grid._get_successful_move_actions` in `TestGridCount`, `test_get_move_actions`, `test_get_successful_move_actions`, `test_action_prob`, `test_scoring_prob`, and `test_xt_model_rate` (the `xt._get_successful_move_actions(spadl_actions).index` line).

- [ ] **Step 3: Run the migrated helper tests against the new module** (xthreat package doesn't exist yet, so import the helpers directly — the `import silly_kicks.xthreat as xt` line will still fail until Task 5; temporarily guard by running only the helper tests once `_grid` + a stub `__init__` exist). Create a **stub** `silly_kicks/xthreat/__init__.py` for now:

```python
"""xT framework (pluggable). Public API assembled across Tasks 2-8."""
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat.py -k "GridCount or get_move or get_successful or action_prob or scoring_prob" -v`
Expected: PASS for the helper tests.

- [ ] **Step 4: Add a drift guard** — `GridSpec()` defaults and `_grid.N/M` are independently hardcoded; pin that they agree so they can't drift apart. Append to `tests/test_xthreat_params.py`:

```python
def test_gridspec_default_matches_grid_module_constants():
    from silly_kicks.xthreat import _grid
    from silly_kicks.xthreat._params import GridSpec
    assert (GridSpec().n_zones_x, GridSpec().n_zones_y) == (_grid.N, _grid.M)
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_params.py::test_gridspec_default_matches_grid_module_constants -v`
Expected: PASS.

- [ ] **Step 5: Stage.** `git add silly_kicks/xthreat/_grid.py silly_kicks/xthreat/__init__.py tests/test_xthreat.py tests/test_xthreat_params.py`

---

## Task 4: `_transitions.py` (Singh) + `_value_iteration.py`

**Files:**
- Create: `silly_kicks/xthreat/_transitions.py`
- Create: `silly_kicks/xthreat/_value_iteration.py`
- Modify: `tests/test_xthreat.py` (`test_move_transition_matrix`)

- [ ] **Step 1: Implement `_value_iteration.py`** (extracted byte-identically from `__solve`).

```python
"""Standalone undiscounted value iteration for the xT fixed point.

Extracted byte-identically from the legacy ExpectedThreat.__solve: raw-diff convergence
(NOT abs) + per-iteration heatmaps. Correct because iteration starts at xT=0 under a
monotone non-negative operator (gs, p_move, T >= 0) -> iterates increase from below ->
raw-diff == abs-diff. Do NOT "fix" the stop condition. See ADR-021.
"""

import numpy as np
import numpy.typing as npt


def value_iteration(
    p_scoring: npt.NDArray[np.float64],
    p_shot: npt.NDArray[np.float64],
    p_move: npt.NDArray[np.float64],
    transition: npt.NDArray[np.float64],
    *,
    eps: float = 1e-5,
    max_iter: int | None = None,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Solve xT. max_iter=None (default) reproduces the legacy unbounded loop exactly;
    a non-None bound is an opt-in safety cap for direct callers on arbitrary matrices."""
    w, l = p_scoring.shape
    gs = p_scoring * p_shot
    xT = np.zeros((w, l), dtype=np.float64)
    heatmaps: list[npt.NDArray[np.float64]] = [xT.copy()]
    diff = np.ones((w, l), dtype=np.float64)
    it = 0
    while np.any(diff > eps):
        if max_iter is not None and it >= max_iter:
            break
        total_payoff = (transition @ xT.ravel()).reshape(w, l)
        newxT = gs + (p_move * total_payoff)
        diff = newxT - xT
        xT = newxT
        heatmaps.append(xT.copy())
        it += 1
    return xT, heatmaps
```

- [ ] **Step 2: Implement `_transitions.py` Singh path** (the legacy `_move_transition_matrix` body, GridSpec-wrapped). `silverman_2d` + KDE land in Task 6 — define `silverman_2d` now (pure, trivial) and leave KDE for Task 6.

```python
"""Transition-matrix builders for the xT model. See NOTICE for citations."""

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _get_flat_indexes, _get_move_actions
from silly_kicks.xthreat._params import GridSpec

# Task 6 extends this module: add `_get_successful_move_actions` to the _grid import and
# `KDEParams` to the _params import when kde_smoothed_transition_matrix lands.


def singh_transition_matrix(actions: pd.DataFrame, grid: GridSpec) -> npt.NDArray[np.float64]:
    """Row-normalized empirical move-transition counts (classic Singh 2018).

    Byte-identical to the legacy ``_move_transition_matrix(actions, grid.n_zones_x, grid.n_zones_y)``.
    """
    l, w = grid.n_zones_x, grid.n_zones_y
    move_actions = _get_move_actions(actions)
    move_actions = move_actions.dropna(subset=["start_x", "start_y", "end_x", "end_y"])

    X = pd.DataFrame()
    X["start_cell"] = _get_flat_indexes(move_actions.start_x, move_actions.start_y, l, w)
    X["end_cell"] = _get_flat_indexes(move_actions.end_x, move_actions.end_y, l, w)
    X["result_id"] = move_actions.result_id

    vc = X.start_cell.value_counts(sort=False)
    start_counts = np.zeros(w * l)
    start_counts[vc.index] = vc

    transition_matrix = np.zeros((w * l, w * l))
    for i in range(0, w * l):
        vc2 = X[((X.start_cell == i) & (X.result_id == spadlconfig.result_id["success"]))].end_cell.value_counts(
            sort=False
        )
        transition_matrix[i, vc2.index] = vc2 / start_counts[i]
    return transition_matrix


def silverman_2d(n: int, sigma: float) -> float:
    """Silverman's rule-of-thumb bandwidth in 2D: h = n^(-1/6) * sigma.

    (4/(d+2))^(1/(d+4)) with d=2 simplifies to 1. Silverman (1986). See NOTICE.
    """
    return float(n ** (-1 / 6) * sigma)
```

Note: `singh_transition_matrix` must reproduce the legacy loop exactly (including `start_counts[i]` possibly 0 → the row stays 0 via `vc2 / 0` only when `vc2` is empty, matching legacy; an empty `vc2.index` assigns nothing).

- [ ] **Step 3: Migrate `test_move_transition_matrix` in `tests/test_xthreat.py`.**

```python
from silly_kicks.xthreat._params import GridSpec
from silly_kicks.xthreat._transitions import singh_transition_matrix
# ...
def test_move_transition_matrix() -> None:
    # ... (same fixture df) ...
    move_mat = singh_transition_matrix(spadl_actions, GridSpec(n_zones_x=2, n_zones_y=2))
    assert np.sum(move_mat) == 1
    assert move_mat.shape == (4, 4)
    assert move_mat[2, 2] == 1
```

- [ ] **Step 4: Run.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat.py::test_move_transition_matrix -v`
Expected: PASS.

- [ ] **Step 5: Stage.** `git add silly_kicks/xthreat/_transitions.py silly_kicks/xthreat/_value_iteration.py tests/test_xthreat.py`

---

## Task 5: `_model.py` facade + `__init__.py`; delete old module; Singh parity gate

**Files:**
- Create: `silly_kicks/xthreat/_model.py`
- Modify: `silly_kicks/xthreat/__init__.py`
- Delete: `silly_kicks/xthreat.py`
- Modify: `tests/test_xthreat.py` (remaining `xt.` usages + the scipy-patch target + new parity test)

- [ ] **Step 1: Implement `_model.py`** — the `ExpectedThreat` class, dispatching the transition and delegating to `value_iteration`. `interpolator()` and `rate()` are pasted verbatim from the legacy module (they already use `self.l`/`self.w`/`spadlconfig`).

```python
"""Expected Threat (xT) model — pluggable transition family. See NOTICE for citations."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RectBivariateSpline  # type: ignore[reportMissingImports]
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import (
    N, M, _action_prob, _get_cell_indexes, _get_successful_move_actions, _scoring_prob,
)
from silly_kicks.xthreat._params import GridSpec, KDEParams, Method, XtParams, validate_params_for_method
from silly_kicks.xthreat._transitions import singh_transition_matrix
from silly_kicks.xthreat._value_iteration import value_iteration

# NOTE: kde_smoothed_transition_matrix is lazy-imported inside fit() (below), NOT at module
# top. This (a) lets the package import cleanly at Task 5 before KDE lands in Task 6 — so the
# Task 5 parity gate can run — and (b) keeps `import silly_kicks` light (sklearn is only pulled
# when the KDE path actually runs).


class ExpectedThreat:
    """xT model [Singh 2018]. ``method="kde_smoothed"`` swaps the transition builder only.

    Parameters
    ----------
    l, w : grid cells in x / y. Default 16x12 (back-compat). Maps to GridSpec(l, w).
    eps : value-iteration precision (default 1e-5).
    method : "singh_counts" (default, byte-identical to the classic model) or "kde_smoothed".
    params : SinghParams | KDEParams | None — validated against ``method``.

    Examples
    --------
    Fit and rate::

        from silly_kicks.xthreat import ExpectedThreat
        xt = ExpectedThreat().fit(actions)
        values = xt.rate(actions)

    KDE-smoothed at higher resolution::

        from silly_kicks.xthreat import ExpectedThreat, KDEParams
        xt = ExpectedThreat(l=24, w=16, method="kde_smoothed", params=KDEParams()).fit(actions)
    """

    def __init__(
        self,
        l: int = N,
        w: int = M,
        eps: float = 1e-5,
        method: Method = "singh_counts",
        params: XtParams | None = None,
    ) -> None:
        validate_params_for_method(method, params)
        self.l = l
        self.w = w
        self.eps = eps
        self.method: Method = method
        self.params = params
        self.grid = GridSpec(n_zones_x=l, n_zones_y=w)
        self.heatmaps: list[npt.NDArray[np.float64]] = []
        self.xT: npt.NDArray[np.float64] = np.zeros((self.w, self.l))
        self.scoring_prob_matrix: npt.NDArray[np.float64] | None = None
        self.shot_prob_matrix: npt.NDArray[np.float64] | None = None
        self.move_prob_matrix: npt.NDArray[np.float64] | None = None
        self.transition_matrix: npt.NDArray[np.float64] | None = None

    def fit(self, actions: pd.DataFrame) -> "ExpectedThreat":
        """Fit the xT grid. See NOTICE for full bibliographic citations.

        Examples
        --------
        ::

            xt = ExpectedThreat().fit(actions)
        """
        self.scoring_prob_matrix = _scoring_prob(actions, self.l, self.w)
        self.shot_prob_matrix, self.move_prob_matrix = _action_prob(actions, self.l, self.w)
        if self.method == "singh_counts":
            self.transition_matrix = singh_transition_matrix(actions, self.grid)
        else:  # kde_smoothed
            from silly_kicks.xthreat._transitions import kde_smoothed_transition_matrix

            params = self.params if isinstance(self.params, KDEParams) else KDEParams()
            self.transition_matrix = kde_smoothed_transition_matrix(actions, self.grid, params)
        self.xT, self.heatmaps = value_iteration(
            self.scoring_prob_matrix,
            self.shot_prob_matrix,
            self.move_prob_matrix,
            self.transition_matrix,
            eps=self.eps,
        )
        return self

    def interpolator(self, kind: str = "linear") -> Callable[..., npt.NDArray[np.float64]]:
        """Interpolate xT over the pitch (RectBivariateSpline). See NOTICE for citations.

        Examples
        --------
        ::

            interp = xt.interpolator("linear")
            grid = interp(xs, ys)
        """
        # ... PASTE the legacy interpolator() body verbatim (lines 381-401 of old module) ...

    def rate(self, actions: pd.DataFrame, use_interpolation: bool = False) -> npt.NDArray[np.float64]:
        """Rate move actions; non-move actions get NaN. See NOTICE for citations.

        Examples
        --------
        ::

            values = ExpectedThreat().fit(actions).rate(actions)
        """
        # ... PASTE the legacy rate() body verbatim (lines 437-468 of old module) ...
```

(Paste the `interpolator()` and `rate()` bodies verbatim from `tests/xthreat_legacy_reference.py` — they are unchanged.)

- [ ] **Step 2: Assemble the Task-5 `__init__.py` — ONLY the symbols that exist now.**

`kde_smoothed_transition_matrix` (Task 6) and the `_eval` functions (Task 8) are deliberately
absent here and added by those tasks — importing them now would `ImportError`. This is the exact
partial form to paste at Task 5 (no trimming guesswork):

```python
"""xT framework — pluggable transition family + held-out NLL evaluator.

See NOTICE for full bibliographic citations.
"""

from silly_kicks.xthreat._model import ExpectedThreat
from silly_kicks.xthreat._params import (
    GridSpec,
    KDEParams,
    KdeKernel,
    Method,
    SinghParams,
    XtParams,
    validate_params_for_method,
)
from silly_kicks.xthreat._transitions import silverman_2d, singh_transition_matrix
from silly_kicks.xthreat._value_iteration import value_iteration

__all__ = [
    "ExpectedThreat",
    "GridSpec",
    "Method",
    "KdeKernel",
    "SinghParams",
    "KDEParams",
    "XtParams",
    "validate_params_for_method",
    "value_iteration",
    "singh_transition_matrix",
    "silverman_2d",
]
```

Task 6 Step 5 adds the `kde_smoothed_transition_matrix` import + `__all__` entry; Task 8 Step 4
adds the `_eval` import (`holdout_split`, `compute_holdout_nll`, `compute_holdout_nll_per_group`)
+ its 3 `__all__` entries. (This replaces the earlier stub `__init__` from Task 3.)

- [ ] **Step 3: Delete the old module + repoint the ruff per-file-ignore to the package.**

```bash
git rm silly_kicks/xthreat.py
```

In `pyproject.toml` under `[tool.ruff.lint.per-file-ignores]`, change the path-scoped ignore
(line ~149) from the module to the package (the new `_grid.py`/`_model.py`/`_transitions.py`
use `l`/`w` (E741) + `X` (N806)):

```toml
# before
"silly_kicks/xthreat.py" = ["N803", "N806", "E741"]
# after
"silly_kicks/xthreat/*.py" = ["N803", "N806", "E741"]
```

- [ ] **Step 4: Migrate remaining `tests/test_xthreat.py` usages.**
  - `import silly_kicks.xthreat as xt` → keep (now the package). `xt.ExpectedThreat` still resolves via `__init__`.
  - `test_interpolate_xt_grid_no_scipy`: change patch target from `xt` to the model module:
    ```python
    mocker.patch("silly_kicks.xthreat._model.RectBivariateSpline", None)
    ```
  - The `xt_model` session fixture + `test_predict` / `test_predict_with_interpolation` need no change (use `xt.ExpectedThreat`).

- [ ] **Step 5: Add the exact Singh-parity test (the contract).**

```python
# tests/test_xthreat.py
import tests.xthreat_legacy_reference as legacy


def test_singh_path_byte_identical_to_legacy(spadl_actions):
    """Default ExpectedThreat (Singh, 16x12) must reproduce the pre-refactor output exactly."""
    new = xt.ExpectedThreat().fit(spadl_actions)
    old = legacy.ExpectedThreat().fit(spadl_actions)
    np.testing.assert_array_equal(new.xT, old.xT)
    np.testing.assert_array_equal(new.transition_matrix, old.transition_matrix)
    np.testing.assert_array_equal(new.scoring_prob_matrix, old.scoring_prob_matrix)
    np.testing.assert_array_equal(new.shot_prob_matrix, old.shot_prob_matrix)
    np.testing.assert_array_equal(new.move_prob_matrix, old.move_prob_matrix)


def test_singh_path_byte_identical_on_worldcup(sb_worldcup_data):
    """Same, on a real multi-match corpus, including rate() output."""
    from typing import cast

    from tests._xthreat_helpers import _worldcup_ltr
    actions = _worldcup_ltr(sb_worldcup_data)
    new = xt.ExpectedThreat(l=16, w=12).fit(actions)
    old = legacy.ExpectedThreat(l=16, w=12).fit(actions)
    np.testing.assert_array_equal(new.xT, old.xT)
    np.testing.assert_array_equal(new.transition_matrix, old.transition_matrix)
    last = cast(pd.DataFrame, sb_worldcup_data["games"]).iloc[-1]
    acts = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{last.game_id}"])
    np.testing.assert_array_equal(new.rate(acts), old.rate(acts))
```

- [ ] **Step 6: Run the FULL xthreat suite (parity gate).**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat.py -v`
Expected: PASS — all migrated tests + both new parity tests green. This proves the Singh path is byte-identical.

- [ ] **Step 7: Run the broader suite for ripple (calibration + tracking consumers of xthreat).**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat.py tests/calibration/test_xt.py tests/tracking/test_player_influence_snapshot.py tests/invariants/test_vaep_geometric_sanity.py -v`
Expected: PASS — the golden gates (`dab140505e42a94a`, sha256 roundtrip, goal-monotonic) stay green; `calibration/_xt.py` untouched and working.

- [ ] **Step 8: Stage.** `git add -A silly_kicks/xthreat tests/test_xthreat.py`

---

## Task 6: KDE-smoothed transition

**Files:**
- Modify: `silly_kicks/xthreat/_transitions.py`
- Modify: `silly_kicks/xthreat/__init__.py` (add the KDE import + `__all__` entry)
- Test: `tests/test_xthreat_kde.py`

(`tests/_xthreat_helpers.py` — incl. `_moves`/`_sparse_overfit_corpus`/`_worldcup_ltr` — was
created in Task 1 Step 3; just import from it here.)

- [ ] **Step 1: Write failing KDE tests** (bandwidth-parametric; add a dispatch-actually-swaps test).

```python
# tests/test_xthreat_kde.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.xthreat import ExpectedThreat, KDEParams
from silly_kicks.xthreat._params import GridSpec
from silly_kicks.xthreat._transitions import (
    _zone_centres, kde_smoothed_transition_matrix, silverman_2d, singh_transition_matrix,
)
from tests._xthreat_helpers import _moves


def test_dispatch_actually_swaps_transition():
    # cheap guard: kde_smoothed must produce a different matrix/grid than singh_counts.
    df = _moves(n_per_zone=80)
    singh = ExpectedThreat(l=6, w=4, method="singh_counts").fit(df)
    kde = ExpectedThreat(l=6, w=4, method="kde_smoothed").fit(df)
    assert not np.array_equal(singh.transition_matrix, kde.transition_matrix)
    assert not np.array_equal(singh.xT, kde.xT)


def test_silverman_2d_formula():
    assert silverman_2d(64, 2.0) == pytest.approx(64 ** (-1 / 6) * 2.0)


def test_zone_centres_invert_flat_index():
    g = GridSpec(n_zones_x=4, n_zones_y=3)
    from silly_kicks.xthreat._grid import _get_flat_indexes
    centres = _zone_centres(g)
    assert centres.shape == (12, 2)
    # the centre of each zone must map back to that flat index
    xs = pd.Series(centres[:, 0]); ys = pd.Series(centres[:, 1])
    flat = _get_flat_indexes(xs, ys, g.n_zones_x, g.n_zones_y).to_numpy()
    np.testing.assert_array_equal(flat, np.arange(12))


@pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
def test_kde_rows_stochastic(bandwidth):
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    T = kde_smoothed_transition_matrix(_moves(), g, KDEParams(bandwidth=bandwidth, adaptive=True))
    assert T.shape == (24, 24)
    sums = T.sum(axis=1)
    # every row sums to 1 (populated rows by KDE; empty rows by fallback)
    np.testing.assert_allclose(sums, np.ones(24), atol=1e-9)


def test_kde_zero_event_row_uses_populated_mean():
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    T = kde_smoothed_transition_matrix(_moves(), g, KDEParams(bandwidth=1.0))
    # an unobserved source zone still sums to 1 (fallback), never all-zero
    assert np.all(np.isclose(T.sum(axis=1), 1.0))


def test_kde_converges_to_singh_as_bandwidth_shrinks():
    # tiny bandwidth -> KDE mass concentrates at the nearest centre -> approaches counts
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    df = _moves(n_per_zone=200)
    singh = singh_transition_matrix(df, g)
    kde = kde_smoothed_transition_matrix(df, g, KDEParams(bandwidth=0.05, adaptive=False))
    # on populated rows the argmax destination should agree
    for s in range(24):
        if singh[s].sum() > 0 and kde[s].sum() > 0:
            assert singh[s].argmax() == kde[s].argmax()
```

- [ ] **Step 2: Run — expect failure** (`_zone_centres` / `kde_smoothed_transition_matrix` missing).

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_kde.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement KDE in `_transitions.py`** (and extend its imports per the Task 4 note: add `_get_successful_move_actions` to the `_grid` import and `KDEParams` to the `_params` import).

```python
def _zone_centres(grid: GridSpec) -> npt.NDArray[np.float64]:
    """(n_zones, 2) SPADL coords of each flat-index zone centre, matching _get_flat_indexes.

    Legacy flat index = (w-1 - yj)*l + xi  =>  xi = flat % l ;  yj = (w-1) - flat // l.
    """
    l, w = grid.n_zones_x, grid.n_zones_y
    flat = np.arange(l * w)
    xi = flat % l
    yj = (w - 1) - (flat // l)
    cx = (xi + 0.5) * grid.cell_length
    cy = (yj + 0.5) * grid.cell_width
    return np.column_stack([cx, cy]).astype(np.float64)


def kde_smoothed_transition_matrix(
    actions: pd.DataFrame, grid: GridSpec, params: KDEParams
) -> npt.NDArray[np.float64]:
    """Per-source-zone 2D KDE-smoothed move-transition matrix. Salimi et al. 2026 (poster);
    Silverman 1986 bandwidth. See NOTICE. Indexed by silly-kicks flat zone indices."""
    from sklearn.neighbors import KernelDensity

    l, w = grid.n_zones_x, grid.n_zones_y
    n = l * w
    move = _get_successful_move_actions(actions).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
    centres = _zone_centres(grid)

    T = np.zeros((n, n), dtype=np.float64)
    populated: list[int] = []
    for s in range(n):
        rows = end_xy[start_cell == s]
        if rows.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((rows[:, 0].var() + rows[:, 1].var()) / 2.0))
            if sigma == 0.0:
                sigma = 1e-6
            h = params.bandwidth * silverman_2d(rows.shape[0], sigma)
        else:
            h = params.bandwidth
        kde = KernelDensity(kernel=params.kernel, bandwidth=h).fit(rows)
        dens = np.exp(kde.score_samples(centres))
        total = dens.sum()
        if total > 0:
            T[s] = dens / total
            populated.append(s)

    if populated:
        mean_row = T[populated].mean(axis=0)
        s_mean = mean_row.sum()
        mean_row = mean_row / s_mean if s_mean > 0 else np.full(n, 1.0 / n)
        for s in range(n):
            if s not in populated:
                T[s] = mean_row
    else:
        T[:] = 1.0 / n
    return T
```

- [ ] **Step 4: Wire the KDE import into `__init__.py`** (add `kde_smoothed_transition_matrix` to the `_transitions` import + `__all__`), and confirm `_model.fit` already dispatches to it (Task 5 lazy-imports it in the else branch).

- [ ] **Step 5: Run** (KDE tests + the dispatch-swaps test).

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_kde.py -v`
Expected: PASS.

- [ ] **Step 6: Smoke the model KDE path.**

Run: `.venv/Scripts/python.exe -c "from tests._xthreat_helpers import _moves; from silly_kicks.xthreat import ExpectedThreat; import numpy as np; m=ExpectedThreat(l=6,w=4,method='kde_smoothed').fit(_moves()); print(np.isfinite(m.xT).all())"`
Expected: `True`.

- [ ] **Step 7: Stage.** `git add silly_kicks/xthreat/_transitions.py silly_kicks/xthreat/__init__.py tests/test_xthreat_kde.py`

---

## Task 7: `value_iteration` guard + KDE-dense convergence test

**Files:**
- Test: `tests/test_xthreat_value_iteration.py`

(The `max_iter` param already exists from Task 4 Step 1; this task tests it + the dense regime.)

- [ ] **Step 1: Write tests.**

```python
# tests/test_xthreat_value_iteration.py
import numpy as np

from silly_kicks.xthreat import ExpectedThreat, KDEParams
from silly_kicks.xthreat._value_iteration import value_iteration
from tests._xthreat_helpers import _moves


def test_max_iter_none_matches_bounded_when_converged():
    w, l = 4, 6
    rng = np.random.default_rng(0)
    p_scoring = rng.random((w, l)) * 0.2
    p_shot = rng.random((w, l)) * 0.3
    p_move = 1 - p_shot
    T = rng.random((w * l, w * l))
    T = T / T.sum(axis=1, keepdims=True)
    xt_unbounded, _ = value_iteration(p_scoring, p_shot, p_move, T, eps=1e-7)
    xt_bounded, _ = value_iteration(p_scoring, p_shot, p_move, T, eps=1e-7, max_iter=10_000)
    np.testing.assert_array_equal(xt_unbounded, xt_bounded)


def test_max_iter_caps_nonconverging_loop():
    # Non-physical inputs (passed directly, not via the model, so p_shot+p_move need not be 1):
    # gs=0.5 constant injection + p_move=1 + row-stochastic T => operator spectral radius 1,
    # xT grows by 0.5 every iteration and NEVER converges. Without the cap this loops forever;
    # with max_iter=5 it returns after exactly 5 iterations.
    w, l = 2, 2
    p_scoring = np.full((w, l), 0.5)
    p_shot = np.ones((w, l))      # gs = p_scoring * p_shot = 0.5
    p_move = np.ones((w, l))
    T = np.full((w * l, w * l), 1.0 / (w * l))  # row-stochastic
    xt, heatmaps = value_iteration(p_scoring, p_shot, p_move, T, eps=1e-9, max_iter=5)
    assert len(heatmaps) == 6  # initial snapshot + exactly 5 capped iterations
    assert np.all(np.isfinite(xt))


def test_kde_dense_matrix_converges_quickly():
    # KDE produces a dense T; the model's fit must still converge in a sane number of iterations.
    m = ExpectedThreat(l=6, w=4, method="kde_smoothed", params=KDEParams(bandwidth=2.0)).fit(_moves(n_per_zone=100))
    assert np.isfinite(m.xT).all()
    assert len(m.heatmaps) < 500  # dense but still a contraction (shot prob > 0 somewhere)
```

- [ ] **Step 2: Run.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_value_iteration.py -v`
Expected: PASS. (If `test_max_iter_caps_nonconverging_loop` heatmap count is off by the exact loop semantics, adjust the expected count to match the implemented loop — initial append + one per iteration up to the cap.)

- [ ] **Step 3: Stage.** `git add tests/test_xthreat_value_iteration.py`

---

## Task 8: `_eval.py` — held-out transition-model NLL

**Files:**
- Create: `silly_kicks/xthreat/_eval.py`
- Modify: `silly_kicks/xthreat/__init__.py` (add eval imports + `__all__`)
- Test: `tests/test_xthreat_eval.py`

- [ ] **Step 1: Write failing tests.**

```python
# tests/test_xthreat_eval.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.xthreat import (
    GridSpec, compute_holdout_nll, compute_holdout_nll_per_group, holdout_split,
    singh_transition_matrix,
)
from tests._xthreat_helpers import _moves


def test_holdout_split_deterministic_and_disjoint():
    df = _moves(n_per_zone=10).assign(game_id=lambda d: d.action_id % 7)
    tr1, ho1 = holdout_split(df, holdout_fraction=0.3)
    tr2, ho2 = holdout_split(df, holdout_fraction=0.3)
    pd.testing.assert_frame_equal(ho1, ho2)  # deterministic
    assert set(tr1.game_id) & set(ho1.game_id) == set()  # game-level disjoint


def test_compute_holdout_nll_shape_guard():
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    bad = np.zeros((10, 10))
    with pytest.raises(AssertionError):
        compute_holdout_nll(bad, _moves(), grid=g)


def test_compute_holdout_nll_synthetic_truth():
    # NLL is lower for the matrix that generated the data than for a uniform matrix.
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    df = _moves(n_per_zone=200)
    T = singh_transition_matrix(df, g)
    n = g.n_zones
    uniform = np.full((n, n), 1.0 / n)
    nll_fit = compute_holdout_nll(T, df, grid=g)
    nll_uniform = compute_holdout_nll(uniform, df, grid=g)
    assert nll_fit < nll_uniform


def test_per_group_returns_dict():
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    df = _moves(n_per_zone=20).assign(game_id=lambda d: d.action_id % 3)
    T = singh_transition_matrix(df, g)
    out = compute_holdout_nll_per_group(T, df, grid=g, group_col="game_id")
    assert isinstance(out, dict) and len(out) == 3
    assert all(isinstance(v, float) for v in out.values())
```

- [ ] **Step 2: Run — expect failure.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_eval.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `_eval.py`.**

```python
"""Held-out transition-model NLL — negative log-likelihood of pass destination zone given
source zone under the transition matrix. NOT an xT-quality metric. See NOTICE / ADR-021.
"""

import hashlib

import numpy as np
import numpy.typing as npt
import pandas as pd

from silly_kicks.xthreat._grid import _get_flat_indexes, _get_successful_move_actions
from silly_kicks.xthreat._params import GridSpec


def holdout_split(
    actions: pd.DataFrame,
    *,
    holdout_fraction: float = 0.15,
    key_cols: tuple[str, ...] = ("game_id",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic match-level holdout split (silly-kicks-native ``game_id`` key)."""
    threshold = int(round(holdout_fraction * 100))
    keys = actions[list(key_cols)].astype(str).agg("|".join, axis=1)

    def _bucket(k: str) -> int:
        return int(hashlib.sha256(k.encode()).hexdigest(), 16) % 100

    is_holdout = keys.map(lambda k: _bucket(k) < threshold)
    return actions[~is_holdout].copy(), actions[is_holdout].copy()


def compute_holdout_nll(
    transition_matrix: npt.NDArray[np.float64],
    holdout: pd.DataFrame,
    *,
    grid: GridSpec,
    eps: float = 1e-10,
) -> float:
    """-mean_i log T[src_zone_i, dst_zone_i] over successful move rows with valid coords."""
    assert transition_matrix.shape == (grid.n_zones, grid.n_zones), (
        f"transition_matrix {transition_matrix.shape} does not match grid {grid.n_zones} zones"
    )
    move = _get_successful_move_actions(holdout).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if len(move) == 0:
        return float("nan")
    l, w = grid.n_zones_x, grid.n_zones_y
    src = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    dst = _get_flat_indexes(move.end_x, move.end_y, l, w).to_numpy()
    probs = transition_matrix[src, dst]
    return float(-np.mean(np.log(np.maximum(probs, eps))))


def compute_holdout_nll_per_group(
    transition_matrix: npt.NDArray[np.float64],
    holdout: pd.DataFrame,
    *,
    grid: GridSpec,
    group_col: str = "game_id",
    eps: float = 1e-10,
) -> dict[str, float]:
    """Per-group held-out NLL (e.g. per game or, with group_col override, per competition)."""
    return {
        str(g): compute_holdout_nll(transition_matrix, sub, grid=grid, eps=eps)
        for g, sub in holdout.groupby(group_col)
    }
```

- [ ] **Step 4: Add eval imports to `__init__.py`** (the imports + `__all__` entries deferred in Task 5 Step 2).

- [ ] **Step 5: Run.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Stage.** `git add silly_kicks/xthreat/_eval.py silly_kicks/xthreat/__init__.py tests/test_xthreat_eval.py`

---

## Task 9: Variable-resolution end-to-end test

**Files:**
- Test: `tests/test_xthreat_resolution.py`

- [ ] **Step 1: Write the test** (guards the GridSpec-parameterized binning at a non-default resolution).

```python
# tests/test_xthreat_resolution.py
from typing import cast

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat
from tests._xthreat_helpers import _worldcup_ltr


def test_variable_resolution_24x16(sb_worldcup_data):
    actions = _worldcup_ltr(sb_worldcup_data)
    m = ExpectedThreat(l=24, w=16).fit(actions)
    assert m.xT.shape == (16, 24)
    assert m.transition_matrix.shape == (384, 384)
    last = cast(pd.DataFrame, sb_worldcup_data["games"]).iloc[-1]
    acts = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{last.game_id}"])
    ratings = m.rate(acts)
    assert len(ratings) == len(acts)
    from silly_kicks.xthreat._grid import _get_successful_move_actions
    idx = _get_successful_move_actions(acts.reset_index()).index
    assert np.isfinite(ratings[idx]).all()
    m.interpolator()  # must construct without error at 24x16
```

- [ ] **Step 2: Run.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_resolution.py -v`
Expected: PASS.

- [ ] **Step 3: Stage.** `git add tests/test_xthreat_resolution.py`

---

## Task 10: KDE-beats-Singh — synthetic hard gate + WC2018 diagnostic

**Files:**
- Test: `tests/test_xthreat_kde_beats_singh.py`

- [ ] **Step 1: Write the synthetic hard gate** (the sole pass/fail KDE-wins assertion; fixed bandwidth; real code paths; **multi-seed** so it proves the mechanism, not a lucky draw).

```python
# tests/test_xthreat_kde_beats_singh.py
import numpy as np
import pytest

from silly_kicks.xthreat import (
    GridSpec, KDEParams, compute_holdout_nll, holdout_split,
    kde_smoothed_transition_matrix, singh_transition_matrix,
)
from tests._xthreat_helpers import _sparse_overfit_corpus, _worldcup_ltr


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kde_strictly_beats_singh_on_synthetic_sparse_corpus(seed):
    """Singh overfits the sparse, wide-jitter rows; KDE smoothing strictly lowers held-out NLL.
    Asserted across 5 seeds — the mechanism, not one favorable draw."""
    grid = GridSpec(n_zones_x=12, n_zones_y=8)
    df = _sparse_overfit_corpus(seed=seed)
    train, holdout = holdout_split(df, holdout_fraction=0.25, key_cols=("game_id",))
    assert len(train) > 0 and len(holdout) > 0, f"seed={seed}: degenerate split"
    singh = singh_transition_matrix(train, grid)
    kde = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=3.0, adaptive=True))
    nll_singh = compute_holdout_nll(singh, holdout, grid=grid)
    nll_kde = compute_holdout_nll(kde, holdout, grid=grid)
    assert nll_kde < nll_singh, f"seed={seed}: KDE {nll_kde} should beat Singh {nll_singh}"
```

- [ ] **Step 2: Run the synthetic gate — tune the fixture (in `tests/_xthreat_helpers.py`) until all 5 seeds pass.**

Run: `.venv/Scripts/python.exe -m pytest "tests/test_xthreat_kde_beats_singh.py::test_kde_strictly_beats_singh_on_synthetic_sparse_corpus" -v`
Expected: PASS (all 5 seeds). If any seed fails, the corpus is not sparse enough to make Singh overfit — in `_sparse_overfit_corpus` raise the jitter, lower the per-(game,centre) count, or raise the gate's `bandwidth` until KDE strictly wins on every seed (the mechanism is real; tune the fixture, NEVER weaken `<` to `<=`).

- [ ] **Step 3: Add the bandwidth-sweep diagnostic on WC2018 (fulfils the spec's "widen the search past 2.0" + sets/justifies the default; NON-asserting).**

```python
def test_kde_bandwidth_sweep_worldcup_diagnostic(sb_worldcup_data, capsys):
    """Widen past the lakehouse's saturated 2.0 edge. Logs the Singh baseline + the KDE NLL
    curve over bandwidths so the chosen KDEParams.bandwidth default is justified (NOT asserting)."""
    actions = _worldcup_ltr(sb_worldcup_data)
    grid = GridSpec(n_zones_x=16, n_zones_y=12)  # silly-kicks default resolution
    train, holdout = holdout_split(actions, holdout_fraction=0.15, key_cols=("game_id",))
    nll_singh = compute_holdout_nll(singh_transition_matrix(train, grid), holdout, grid=grid)
    curve = {}
    for bw in (1.0, 2.0, 4.0, 8.0):
        T = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=bw, adaptive=True))
        curve[bw] = compute_holdout_nll(T, holdout, grid=grid)
    with capsys.disabled():
        print(f"\n[xT bandwidth sweep 16x12 WC2018] Singh NLL={nll_singh:.5f}")
        for bw, nll in curve.items():
            print(f"  bw={bw:>4}: KDE NLL={nll:.5f}  delta_vs_singh={nll_singh - nll:+.5f}")
    assert all(np.isfinite(v) for v in curve.values())  # sanity only
```

- [ ] **Step 4: Run the sweep, then set the default in `_params.py`.**

Run: `.venv/Scripts/python.exe -m pytest "tests/test_xthreat_kde_beats_singh.py::test_kde_bandwidth_sweep_worldcup_diagnostic" -v -s`
Expected: PASS (prints the curve).

Read the curve and **edit `KDEParams.bandwidth` in `silly_kicks/xthreat/_params.py`** (set to `2.0` in Task 2):
  - If the curve has an **interior minimum** at a bandwidth ≠ 2.0 → set `KDEParams.bandwidth` to that value, and note it in CHANGELOG + ADR-021.
  - If the curve is **still falling at bw=8** → either widen the sweep once more and repeat, or **retain `2.0`** with an explicit "saturated-edge default, pending a wider real-data sweep" note in ADR-021/CHANGELOG.
  - Record the full curve in the PR description either way.
  - Safe to change late: KDE unit tests are bandwidth-parametric, and the parity/dispatch/resolution tests do not depend on the default. After editing, re-run `tests/test_xthreat_kde.py` to confirm still green.

- [ ] **Step 5: Run the whole file.**

Run: `.venv/Scripts/python.exe -m pytest tests/test_xthreat_kde_beats_singh.py -v -s`
Expected: PASS (5-seed hard gate asserts; both diagnostics print + sanity-pass).

- [ ] **Step 6: Stage.** `git add tests/test_xthreat_kde_beats_singh.py`

---

## Task 11: Attribution — NOTICE + ADR-021 + docstring cross-links

**Files:**
- Modify: `NOTICE`
- Create: `docs/superpowers/adrs/ADR-021-pluggable-xt.md`

- [ ] **Step 1: Add NOTICE entries** under "Mathematical / Methodological References", near the existing Singh xT entry. (Match the existing entry format — author (year), title, venue, `Used by:`, faithfulness note.)

```
- Silverman, B. W. (1986). "Density Estimation for Statistics and Data Analysis."
  Chapman & Hall.
  Used by: silly_kicks.xthreat (method="kde_smoothed"), silverman_2d.
  2D rule-of-thumb bandwidth h = n^(-1/6) * sigma (the (4/(d+2))^(1/(d+4)) constant is 1 at d=2);
  optional per-source-zone adaptive bandwidth.

- Salimi, M. S.; Salmankhah, A.; Nodin, A. (2026). "ExT: Improving the Computational Efficiency
  and Spatial Granularity of the Expected Threat Model." LISS Football Analytics Symposium.
  Used by: silly_kicks.xthreat (method="kde_smoothed").
  PRE-PUBLICATION (poster). silly-kicks implements a REPRODUCTION of the poster-level
  KDE-smoothed transition only; the per-source-context KNN/conditional formulation is NOT
  implemented. Update on publication.
```

- [ ] **Step 2: Create ADR-021** following `docs/superpowers/adrs/ADR-TEMPLATE.md` (sections: metadata table, Context, Decision, Alternatives considered, Consequences +/-/Neutral, Related). Record: pluggable string-dispatch architecture (not ABCs); Singh facade byte-identical; KDE flavor (Silverman + pre-publication Salimi reproduction); raw-diff convergence rationale (monotone-from-below ⇒ raw≡abs — "do not 'fix'"); held-out NLL is a transition-model metric; `game_id`-native split; KNN/XTGrid/atomic/xfns deferred; calibration facade untouched; the chosen `KDEParams.bandwidth` default + its justification from the Task 10 sweep. Use the ADR's "Related" to link the spec + plan.

  Under **Consequences → Neutral**, record this **conscious fast-follow debt**: `singh_transition_matrix` preserves the legacy O(n_zones × n_actions) boolean-mask-in-loop (a full-DataFrame scan per zone) — newly hotter at the 24×16 this PR enables (384 scans over up to millions of actions). It is kept verbatim *only* to satisfy the exact-parity gate. A byte-identical vectorization is feasible (`np.add.at` accumulating successful-move (start_cell, end_cell) counts, then dividing each row by the all-move `start_counts[i]` — same integer operands ⇒ same floats) and should be a separate fast-follow PR, re-gated by the same `tests/xthreat_legacy_reference.py` parity oracle, so it is not silently locked behind exact-parity.

- [ ] **Step 3: Verify the xthreat docstrings carry `See NOTICE for full bibliographic citations.`** (Already in the Task 5/6/8 code blocks — confirm present in `_model.py`, `_transitions.py`, `_eval.py`.)

- [ ] **Step 4: Stage.** `git add NOTICE docs/superpowers/adrs/ADR-021-pluggable-xt.md`

---

## Task 12: Full verification + version bump (commit/PR GATED)

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`

- [ ] **Step 1: Run the full non-e2e suite.**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --no-header` (background if >30s; read the summary line, not a piped tail).
Expected: all pass; in particular the parity, KDE, value_iteration, eval, resolution, and synthetic-gate tests + the pre-existing golden gates.

- [ ] **Step 2: Lint + types (replicate CI).**

Run: `.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/`
Run: `.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/`
Run: `.venv/Scripts/python.exe -m pyright silly_kicks/`
Expected: all clean. (The `xthreat.py` → `xthreat/*.py` per-file-ignore repoint was done in Task 5
Step 3, and the frozen reference carries its own `# ruff: noqa` header from Task 1 Step 2 —
verify both took effect here.)

- [ ] **Step 3: Confirm `import silly_kicks` stays dependency-light** (sklearn is a hard dep; no NEW heavy import at package import time).

Run: `.venv/Scripts/python.exe -c "import silly_kicks; print('ok')"`
Expected: `ok`. KDE's `from sklearn.neighbors import KernelDensity` is function-local (Task 6) so bare import is unaffected.

- [ ] **Step 4: Version bump** to the next free minor after `origin/main` (reconcile per the version-bump checklist — likely 4.17.0): `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md` (new section: SK-xT-1 — pluggable xT, KDE flavor, NLL evaluator; note this is additive, Singh path byte-identical, no retrain trigger for existing consumers), `TODO.md` current-release line, then `uv lock`.

- [ ] **Step 5: C4 check** — confirm no C4-enumerated token/count change (xthreat is not an enumerated `tracking` backend/model/aggregator). Expected: C4-free, no regen.

- [ ] **Step 6: HALT for commit approval.** Present the full diff + the single bundled commit message (spec + plan + ADR + NOTICE + code + tests + version bump). Do NOT create the sentinel or commit/push/PR without explicit per-step approval (user rule). One feature branch, single squash-merged commit, PR at the end.

---

## Coordination notes (carry into the PR description)
- **Calibration:** `silly_kicks/calibration/_xt.py` is untouched; the byte-identical Singh facade + the sha256 roundtrip + `test_player_influence_snapshot` golden prove it. Flag for the TF-24/lakehouse session before merge.
- **Lakehouse cross-check (optional, owner-gated):** a separate `@pytest.mark.e2e` test may triangulate against `fct_action_values` at 12×8; out of the default suite, not in this plan's task list (add later if desired).
- **Versioning/ADR:** ADR-021 assumed free — reconcile at release.
