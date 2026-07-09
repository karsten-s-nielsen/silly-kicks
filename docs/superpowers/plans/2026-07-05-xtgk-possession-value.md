# xT-GK v2 — Sub-project 1: Honest Possession-Value Surface `V(z,p)` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (owner prefers INLINE execution, NO subagents — see repo memory) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a pressure-stratified Markov possession-value surface `V(z,p)` = expected xG of the possession's first shot given the ball in grid-zone `z` under pressure `p∈{1,2,3}`, plus a model-free empirical cross-check and a pre-registered deep-zone go/no-go gate — the value function that replaces v1 xT-GK's flat destination-only surface.

**Architecture:** New hexagonal `silly_kicks/xtgk/` package. One `PossessionValue` Protocol, two adapters: `MarkovPossessionValue` (production — reuses the tested `xthreat` value-iteration solver + its low-level KDE/grid seams with an xG-calibrated immediate reward, a goal-kick-inclusive move-set, and pressure stratification) and `EmpiricalPossessionValue` (cross-check only, not shipped). xG enters as an **injected `xg_column`**; silly-kicks ships no xG model.

**Tech Stack:** pandas, numpy, existing `silly_kicks.xthreat` internals (`value_iteration`, `_grid` primitives, `_transitions` low-level KDE seams), pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-xtgk-v2-possession-value-design.md` (rev 3).

**Plan revision:** rev 2 — incorporates the analysis-session plan review (B1 3-band fixtures, B2 occupied-cell gate, B3/M1/M2 xtgk-local builders reusing low-level seams (no `xthreat` public-API edits), M3 split orientation tests, M4 configurable direction, M5 serialization in SP1, m1–m5).

---

## Policy overrides (read before executing)

- **ONE commit per branch.** Per-task steps produce green checkpoints but **do NOT commit per task**. Stage nothing until the whole plan is green, `/final-review` has run, and the owner has approved. The single commit is the last task.
- **Branch:** `pr-s107-xtgk-possession-value` off `main` (no worktree).
- **Session start:** `pip install -e ".[test]"` before anything.
- **Target release:** silly-kicks **4.40.0**, **PR-S107**, **ADR-036**. Confirm the current version in `pyproject.toml` at execution start; if it moved past 4.39.0, bump to the next minor and keep the PR-S/ADR numbers unless taken.
- **Blocked phases:** Phases 0–10 (the entire library + CI suite) are **fully unblocked** — synthetic fixtures only. **Phase 11 (real-data owner-run gate) is BLOCKED** on collaboration-call answers Q3 (canonical `xg_column` source) and Q4 (locked gate numbers). Build 0–10 to done; leave Phase 11 wired-but-not-run.

---

## Key design decision — the extended move-set seam (was "D1"; revised per plan-review M1/M2/B3)

The surface must include **goal-kicks** (and throw-ins) in the transition law (spec §G2) — but classic xT's move-set (`xthreat._grid._get_move_actions`) is `pass∪dribble∪cross` only, and, critically, the KDE builder bins via `_get_successful_move_actions` (**successful** moves only, `_transitions.py:107`), while the Singh builder consumes **all** moves and computes success internally (`_transitions.py:36–45`).

**Decision:** implement **xtgk-local** transition + action-probability builders in `silly_kicks/xtgk/_moves.py` that reuse `xthreat`'s **shared low-level seams** and **do NOT modify any `xthreat` public function**:
- Singh path → replicate the ~10-line count-and-normalize (byte-identical to `singh_transition_matrix`) over the **all-results** extended move-set.
- KDE path → bin the **success-filtered** extended move-set by source (reusing `_zone_centres` + `_get_flat_indexes`), then call the shared `_kde_transition_from_grouped` kernel verbatim.
- `p_shot`/`p_move` → replicate the ~6-line `_action_prob` count over the extended move-set, reusing `_count`/`_safe_divide`.

**Why this over widening the public builders:** (1) the KDE kernel is *already* shared at the correct low level — no need to touch `singh_transition_matrix`/`kde_smoothed_transition_matrix`/`_action_prob`; (2) Singh and KDE need **different** populations (all vs successful) — a single injected `move_actions` DataFrame can't serve both correctly; (3) zero regression surface on classic xT (the public builders are untouched), so no oracle-widening; (4) honors the spec's "xtgk-local reuse, don't edit xthreat" boundary. Correctness is proven by **parity tests**: on a `pass∪dribble∪cross`-only cohort the xtgk builders must be **byte-identical** to the stock `xthreat` builders (Task 2), across N random cohorts.

**No open deviations.** (The earlier D2 flag — per-action vs per-possession-origin empirical conditioning — was accepted on the merits and **reconciled into the spec** on 2026-07-05: spec §M1 and its reuse-map row now state per-action conditioning explicitly. Plan and spec now agree.)

**D2 (now spec-aligned) — Empirical cross-check conditions PER ACTION (ball-at-`z`).** Per action, outcome = xG of the first shot *after* that action within the possession (0 if none), averaged per `(cell, tercile)` — the like-for-like estimator of the Markov `V(z,p)` (the value of the ball being at `z` *now*, not of a possession that originated at `z`). Implemented in `_empirical.py::_possession_outcomes` (Task 9).

---

## File structure

```
silly_kicks/xtgk/
  __init__.py               # public exports
  _possession_value.py      # PressureLevel, State, DeltaV, PossessionValue Protocol, zone_of
  _moves.py                 # extended move-set + xtgk-local singh/kde transition + action_prob builders
  _xg_reward.py             # xg_scoring_prob(): per-cell E[xG|shot] from injected xg_column
  _pressure_levels.py       # PressureLevels: fit/apply terciles + occupancy report
  _validate.py              # validate_possession_value_input() + diagnosis dataclass
  _markov.py                # MarkovPossessionValue (fit/value/surface/delta_v/save/load)
  _empirical.py             # EmpiricalPossessionValue (cross-check; surface/value only)
  _diagnostics.py           # occupied-cell deep-zone gate + support/occupancy reports
  _serialize.py             # pickle-free npz/JSON + SHA256 artifact I/O (G4)

tests/xtgk/
  __init__.py  conftest.py
  test_moves.py  test_xg_reward.py  test_pressure_levels.py  test_validate.py
  test_types.py  test_markov.py  test_empirical.py  test_diagnostics.py
  test_honesty_property.py  test_serialize.py  test_regression_boundary.py
tests/
  test_xtgk_builder_parity.py   # xtgk builders == stock xthreat on pass-only cohorts (flat dir; the
                                # xthreat suite lives flat in tests/, e.g. tests/test_xthreat.py)

scripts/
  validate_xtgk_possession_value.py           # owner-run real-data gate (Phase 11)

docs/superpowers/adrs/ADR-036-xtgk-possession-value-surface.md
```

**No `xthreat` files are modified.** The regression boundary (§10) is therefore satisfied by construction; a guard test still asserts it.

---

## Conventions used throughout

- Grid `w×l = 12×16` (`GridSpec(n_zones_x=16, n_zones_y=12)`), `field_length=105.0`, `field_width=68.0`.
- **Flat zone index** matches `xthreat._grid._get_flat_indexes` and `value_iteration`'s `ravel()`: cell `(xi∈[0,16), yj∈[0,12))` → `flat=(w-1-yj)*l+xi`; a surface `S` of shape `(w,l)` is accessed `S.ravel()[flat]`. A y-reflection (`y→68−y`) is a **row reversal** `S[::-1, :]` (used by the equivariance test). Always index a cell via `.ravel()[flat]`, never hand-computed `[row,col]`.
- Action-type ids (verified): `pass=0, cross=1, throw_in=2, shot=11, dribble=21, goalkick=22`. `result success=1`.
- All `warnings.warn(..., stacklevel=2)`.

---

## Phase 0 — Branch, scaffold, shared fixtures

### Task 0: Branch + empty package + conftest

**Files:** Create `silly_kicks/xtgk/__init__.py`, `tests/xtgk/__init__.py`, `tests/xtgk/conftest.py`

- [ ] **Step 1:** Branch + install.

```bash
git checkout main && git pull && git checkout -b pr-s107-xtgk-possession-value
pip install -e ".[test]"
```

- [ ] **Step 2:** `silly_kicks/xtgk/__init__.py` (temporary docstring only):

```python
"""xT-GK v2 — honest possession-value surface V(z,p). See NOTICE / ADR-036."""
```

- [ ] **Step 3:** `tests/xtgk/__init__.py` (empty) and `tests/xtgk/conftest.py`:

```python
"""Synthetic SPADL cohorts for xtgk tests. No real data — CI-safe.

Pressure fixtures use THREE well-separated bands (0.1 / 0.5 / 0.9) so all three terciles
populate (plan-review B1). Deep goal-kicks are SPREAD across several deep cells so the
occupied-cell gate (B2) has >1 populated deep cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

PASS = spadlconfig.actiontype_id["pass"]
DRIBBLE = spadlconfig.actiontype_id["dribble"]
CROSS = spadlconfig.actiontype_id["cross"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
THROW_IN = spadlconfig.actiontype_id["throw_in"]
SHOT = spadlconfig.actiontype_id["shot"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]

# three pressure bands -> terciles {1,2,3}; xg decreases with pressure (a real gradient)
BANDS = ((0.1, 0.5), (0.5, 0.25), (0.9, 0.05))  # (pressure, shot_xg)
DEEP_YS = (12.0, 24.0, 34.0, 44.0, 56.0)         # spread deep goal-kick origins across cells


def _row(action_id, type_id, result_id, sx, sy, ex, ey, *, game_id=1, period_id=1,
         team_id=10, player_id=100, possession_id=0, time_seconds=0.0, xg=np.nan,
         pressure=0.5):
    return dict(
        game_id=game_id, period_id=period_id, action_id=action_id, time_seconds=time_seconds,
        team_id=team_id, player_id=player_id, type_id=type_id, result_id=result_id,
        bodypart_id=0, start_x=sx, start_y=sy, end_x=ex, end_y=ey,
        possession_id=possession_id, xg=xg, pressure=pressure,
    )


def make_cohort(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)


def build_up_to_shot_possession(possession_id, *, pressure, xg, deep_x=3.0, deep_y=34.0,
                                buildup_x=30.0, mid_x=55.0, shot_x=100.0, base_action_id=0,
                                game_id=1):
    """deep GOALKICK -> BUILD-UP pass -> mid pass -> forward pass -> shot(xg). Attack-LTR.

    The goal-kick lands at buildup_x=30 (grid xi=4, in BUILD_UP_CELLS xi 2..6) and the next pass
    STARTS there, so the build-up band is populated with real support and value — WITHOUT this the
    gate's cross-check comparison (graded on the build-up band) is vacuously 0==0 (plan-review R2).
    """
    a = base_action_id
    return [
        _row(a + 0, GOALKICK, SUCCESS, deep_x, deep_y, buildup_x, 34.0, game_id=game_id,
             possession_id=possession_id, pressure=pressure, time_seconds=a + 0.0),
        _row(a + 1, PASS, SUCCESS, buildup_x, 34.0, mid_x, 34.0, game_id=game_id,
             possession_id=possession_id, pressure=pressure, time_seconds=a + 1.0),
        _row(a + 2, PASS, SUCCESS, mid_x, 34.0, 80.0, 34.0, game_id=game_id,
             possession_id=possession_id, pressure=pressure, time_seconds=a + 2.0),
        _row(a + 3, PASS, SUCCESS, 80.0, 34.0, shot_x, 34.0, game_id=game_id,
             possession_id=possession_id, pressure=pressure, time_seconds=a + 3.0),
        _row(a + 4, SHOT, FAIL, shot_x, 34.0, 105.0, 34.0, game_id=game_id,
             possession_id=possession_id, pressure=pressure, xg=xg, time_seconds=a + 4.0),
    ]


def three_band_cohort(n_per_band=40) -> pd.DataFrame:
    """Honest cohort across 3 pressure bands and several deep cells (B1 + B2)."""
    rows: list[dict] = []
    pid = 0
    for k in range(n_per_band):
        for bi, (pressure, xg) in enumerate(BANDS):
            deep_y = DEEP_YS[(k + bi) % len(DEEP_YS)]
            base = 1000 * pid
            rows += build_up_to_shot_possession(pid, pressure=pressure, xg=xg, deep_y=deep_y,
                                                base_action_id=base)
            pid += 1
    return make_cohort(rows)


def flat_no_shot_cohort(n_per_band=40) -> pd.DataFrame:
    """Negative control (G7): deep possessions that NEVER reach a shot -> deep V ~ 0, flat."""
    rows: list[dict] = []
    pid = 0
    for k in range(n_per_band):
        for pressure, _xg in BANDS:
            base = 1000 * pid
            rows += [
                _row(base, GOALKICK, SUCCESS, 3.0, DEEP_YS[k % len(DEEP_YS)], 40.0, 34.0,
                     possession_id=pid, pressure=pressure),
                _row(base + 1, PASS, SUCCESS, 40.0, 34.0, 55.0, 34.0,
                     possession_id=pid, pressure=pressure),
            ]
            pid += 1
    return make_cohort(rows)


def mirror_y(actions: pd.DataFrame) -> pd.DataFrame:
    """Vertical reflection y->68-y ONLY. Attack direction (x) is PRESERVED -> still attack-LTR.
    Used for the surface y-equivariance test."""
    out = actions.copy()
    out["start_y"] = spadlconfig.field_width - actions["start_y"]
    out["end_y"] = spadlconfig.field_width - actions["end_y"]
    return out


def mirror_x(actions: pd.DataFrame) -> pd.DataFrame:
    """Horizontal reflection x->105-x ONLY. REVERSES attack direction -> NOT attack-LTR.
    Used ONLY for the orientation-rejection test (fit must refuse this)."""
    out = actions.copy()
    out["start_x"] = spadlconfig.field_length - actions["start_x"]
    out["end_x"] = spadlconfig.field_length - actions["end_x"]
    return out
```

- [ ] **Step 4:** `python -m pytest tests/xtgk/ -q` — no tests collected, no import errors.

---

## Phase 1 — Extended move-set + xtgk-local builders (reuse low-level seams)

### Task 1: `_moves.py` — extended move-set + transition/action-prob builders

**Files:** Create `silly_kicks/xtgk/_moves.py`, `tests/xtgk/test_moves.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_moves.py`:

```python
import numpy as np

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat import GridSpec
from silly_kicks.xtgk._moves import (
    MOVE_TYPE_IDS, extended_move_actions, xtgk_action_prob, xtgk_transition_matrix,
)
from tests.xtgk.conftest import make_cohort, _row, PASS, GOALKICK, THROW_IN, SHOT, SUCCESS, FAIL

GRID = GridSpec(n_zones_x=16, n_zones_y=12)


def test_extended_move_set_includes_goalkick_and_throw_in_not_shots():
    rows = [
        _row(0, PASS, SUCCESS, 10, 34, 20, 34),
        _row(1, GOALKICK, SUCCESS, 5, 34, 50, 34),
        _row(2, THROW_IN, SUCCESS, 40, 0, 45, 10),
        _row(3, SHOT, SUCCESS, 100, 34, 105, 34),
    ]
    out = extended_move_actions(make_cohort(rows))
    assert set(out["type_id"]) == {PASS, GOALKICK, THROW_IN}
    assert spadlconfig.actiontype_id["goalkick"] in MOVE_TYPE_IDS


def test_singh_transition_includes_goalkick_rows():
    rows = [_row(0, GOALKICK, SUCCESS, 5, 34, 60, 34), _row(1, PASS, SUCCESS, 60, 34, 80, 34)]
    T = xtgk_transition_matrix(make_cohort(rows), GRID, method="singh_counts")
    assert T.sum() > 0  # goal-kick produced a transition row (excluded by classic xT)


def test_kde_path_uses_successful_moves_only():
    # a FAILED goal-kick must NOT create a destination row under KDE (success-filtered)
    rows = [
        _row(0, GOALKICK, FAIL, 5, 34, 60, 34),   # failed -> no destination
        _row(1, GOALKICK, SUCCESS, 5, 34, 62, 34),  # success -> destination
    ]
    T = xtgk_transition_matrix(make_cohort(rows), GRID, method="kde_smoothed")
    assert np.isfinite(T).all()
```

- [ ] **Step 2: Run — FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_moves.py`:

```python
"""Extended move-set + xtgk-local transition/action-prob builders (ADR-036 §G2).

Classic xT's move-set excludes goal-kicks/throw-ins; this GK surface needs them in the
transition law and the p_shot/p_move split. These builders REUSE xthreat's shared low-level
seams (grid binning, the KDE kernel) and never modify xthreat's public functions. On a
pass-only cohort they are byte-identical to the stock builders (proven in test_xtgk_builder_parity).

Singh consumes ALL extended moves (success computed internally); KDE consumes SUCCESSFUL
extended moves only (mirrors _get_successful_move_actions) — the two paths need different
populations, which is exactly why a single injected DataFrame on the public builders would
be wrong (plan-review M1).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _count, _get_flat_indexes, _safe_divide
from silly_kicks.xthreat._params import GridSpec, KDEParams
from silly_kicks.xthreat._transitions import _kde_transition_from_grouped, _zone_centres

Method = Literal["singh_counts", "kde_smoothed"]

MOVE_TYPE_IDS: tuple[int, ...] = (
    spadlconfig.actiontype_id["pass"],
    spadlconfig.actiontype_id["dribble"],
    spadlconfig.actiontype_id["cross"],
    spadlconfig.actiontype_id["goalkick"],
    spadlconfig.actiontype_id["throw_in"],
)
_SHOT = spadlconfig.actiontype_id["shot"]
_SUCCESS = spadlconfig.result_id["success"]


def extended_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """All ball-progressing actions incl. goal-kicks/throw-ins (any result)."""
    return actions[actions["type_id"].isin(MOVE_TYPE_IDS)]


def _successful(move_actions: pd.DataFrame) -> pd.DataFrame:
    return move_actions[move_actions["result_id"] == _SUCCESS]


def xtgk_action_prob(
    actions: pd.DataFrame, l: int, w: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(p_shot, p_move) per cell over the EXTENDED move-set. Mirrors xthreat _action_prob."""
    move = extended_move_actions(actions)
    shots = actions[actions["type_id"] == _SHOT]
    movematrix = _count(move.start_x, move.start_y, l, w)
    shotmatrix = _count(shots.start_x, shots.start_y, l, w)
    total = movematrix + shotmatrix
    return _safe_divide(shotmatrix, total), _safe_divide(movematrix, total)


def _singh_transition(actions: pd.DataFrame, grid: GridSpec) -> npt.NDArray[np.float64]:
    """Byte-identical to xthreat.singh_transition_matrix but over the extended move-set."""
    l, w = grid.n_zones_x, grid.n_zones_y
    n = w * l
    move = extended_move_actions(actions).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_cell = _get_flat_indexes(move.end_x, move.end_y, l, w).to_numpy()
    is_success = (move.result_id == _SUCCESS).to_numpy()
    start_counts = np.zeros(n)
    np.add.at(start_counts, start_cell, 1.0)
    counts = np.zeros((n, n))
    np.add.at(counts, (start_cell[is_success], end_cell[is_success]), 1.0)
    T = np.zeros((n, n))
    nz = start_counts > 0
    T[nz] = counts[nz] / start_counts[nz, None]
    return T


def _bin_extended_successful(actions: pd.DataFrame, grid: GridSpec):
    """Group SUCCESSFUL extended-move destinations by source zone (mirrors
    _bin_destinations_by_source, defaults: keep every row)."""
    l, w = grid.n_zones_x, grid.n_zones_y
    centres = _zone_centres(grid)
    move = _successful(extended_move_actions(actions)).dropna(
        subset=["start_x", "start_y", "end_x", "end_y"]
    )
    if len(move) == 0:
        return {}, centres
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
    order = np.argsort(start_cell, kind="stable")
    sc_sorted = start_cell[order]
    end_sorted = end_xy[order]
    boundaries = np.flatnonzero(np.diff(sc_sorted)) + 1
    zone_per_group = sc_sorted[np.concatenate(([0], boundaries))]
    groups = np.split(end_sorted, boundaries)
    grouped = {int(s): pts for s, pts in zip(zone_per_group, groups, strict=True)}
    return grouped, centres


def _kde_transition(actions: pd.DataFrame, grid: GridSpec, params: KDEParams) -> npt.NDArray[np.float64]:
    grouped, centres = _bin_extended_successful(actions, grid)
    return _kde_transition_from_grouped(grouped, centres, grid, params)


def xtgk_transition_matrix(
    actions: pd.DataFrame, grid: GridSpec, *, method: Method = "singh_counts",
    params: KDEParams | None = None,
) -> npt.NDArray[np.float64]:
    if method == "kde_smoothed":
        return _kde_transition(actions, grid, params or KDEParams())
    return _singh_transition(actions, grid)
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_moves.py -v`

### Task 2: parity — xtgk builders == stock xthreat on pass-only cohorts

**Files:** Create `tests/test_xtgk_builder_parity.py`

- [ ] **Step 1: Failing test** — `tests/test_xtgk_builder_parity.py`:

```python
"""xtgk-local builders must be byte-identical to the stock xthreat builders when the move-set
is restricted to pass∪dribble∪cross (no goal-kicks/throw-ins). Property test over N cohorts."""
import numpy as np
import pandas as pd
import pytest

from silly_kicks.xthreat import GridSpec, kde_smoothed_transition_matrix, singh_transition_matrix
from silly_kicks.xthreat._grid import _action_prob
from silly_kicks.xthreat._params import KDEParams
from silly_kicks.xtgk._moves import xtgk_action_prob, xtgk_transition_matrix
from tests.xtgk.conftest import make_cohort, _row, PASS, DRIBBLE, CROSS, SHOT, SUCCESS, FAIL

GRID = GridSpec(n_zones_x=16, n_zones_y=12)


def _random_pass_only_cohort(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(60):
        t = rng.choice([PASS, DRIBBLE, CROSS])
        res = SUCCESS if rng.random() < 0.7 else FAIL
        rows.append(_row(i, int(t), int(res),
                         float(rng.uniform(0, 105)), float(rng.uniform(0, 68)),
                         float(rng.uniform(0, 105)), float(rng.uniform(0, 68))))
    rows.append(_row(60, SHOT, SUCCESS, 100, 34, 105, 34))
    return make_cohort(rows)


@pytest.mark.parametrize("seed", range(8))
def test_singh_parity(seed):
    a = _random_pass_only_cohort(seed)
    assert np.array_equal(xtgk_transition_matrix(a, GRID, method="singh_counts"),
                          singh_transition_matrix(a, GRID))


@pytest.mark.parametrize("seed", range(8))
def test_kde_parity(seed):
    a = _random_pass_only_cohort(seed)
    assert np.allclose(xtgk_transition_matrix(a, GRID, method="kde_smoothed", params=KDEParams()),
                       kde_smoothed_transition_matrix(a, GRID, KDEParams()), atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_action_prob_parity(seed):
    a = _random_pass_only_cohort(seed)
    s0, m0 = xtgk_action_prob(a, 16, 12)
    s1, m1 = _action_prob(a, 16, 12)
    assert np.array_equal(s0, s1) and np.array_equal(m0, m1)


def _one_zone_mixed_success_cohort():
    """Many mixed-success passes piled into ONE source zone -> exercises the denominator=all /
    numerator=success row-normalization edge (plan-review m6), not just the sparse average case."""
    rows = [_row(i, PASS, SUCCESS if i % 3 else FAIL, 10, 34, 60 + (i % 4), 34) for i in range(30)]
    rows.append(_row(30, SHOT, SUCCESS, 100, 34, 105, 34))
    return make_cohort(rows)


def test_singh_parity_normalization_edge():
    a = _one_zone_mixed_success_cohort()
    assert np.array_equal(xtgk_transition_matrix(a, GRID, method="singh_counts"),
                          singh_transition_matrix(a, GRID))


def test_kde_parity_normalization_edge():
    a = _one_zone_mixed_success_cohort()
    assert np.allclose(xtgk_transition_matrix(a, GRID, method="kde_smoothed", params=KDEParams()),
                       kde_smoothed_transition_matrix(a, GRID, KDEParams()), atol=1e-12)


def test_action_prob_parity_normalization_edge():
    a = _one_zone_mixed_success_cohort()
    s0, m0 = xtgk_action_prob(a, 16, 12)
    s1, m1 = _action_prob(a, 16, 12)
    assert np.array_equal(s0, s1) and np.array_equal(m0, m1)
```

- [ ] **Step 2: Run — PASS** (the Task-1 builders already satisfy parity; if KDE parity fails, the `_bin_extended_successful` replication has drifted from `_bin_destinations_by_source` — reconcile line-for-line, do not weaken the tolerance).

Run: `python -m pytest tests/test_xtgk_builder_parity.py -v`

- [ ] **Step 3: Confirm classic xThreat still green** (untouched, but prove it — these paths are verified to exist; `xthreat_legacy_reference.py` is CLAUDE.md's frozen oracle, and the KDE files cover the seam our builder reuses):

Run: `python -m pytest tests/xthreat_legacy_reference.py tests/test_xthreat.py tests/test_xthreat_kde.py tests/test_xthreat_kde_vectorized.py -v`
Expected: PASS.

---

## Phase 2 — xG reward surface

### Task 3: `xg_scoring_prob` (per-cell `E[xG|shot]`)

**Files:** Create `silly_kicks/xtgk/_xg_reward.py`, `tests/xtgk/test_xg_reward.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_xg_reward.py`:

```python
import numpy as np
import pandas as pd

from silly_kicks.xthreat._grid import _get_flat_indexes
from silly_kicks.xtgk._xg_reward import xg_scoring_prob
from tests.xtgk.conftest import make_cohort, _row, SHOT, PASS, SUCCESS, FAIL


def _flat(x, y):
    return int(_get_flat_indexes(pd.Series([float(x)]), pd.Series([float(y)]), 16, 12).iloc[0])


def test_mean_xg_over_shots_per_cell_not_goal_gated():
    # two shots same cell (100,34), xg 0.2 (goal) and 0.4 (miss) -> E[xG|shot]=0.3, NOT 0.2.
    rows = [
        _row(0, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.2),
        _row(1, SHOT, FAIL, 100, 34, 105, 34, xg=0.4),
        _row(2, PASS, SUCCESS, 10, 34, 20, 34, xg=np.nan),
    ]
    surf = xg_scoring_prob(make_cohort(rows), xg_column="xg", l=16, w=12)
    assert surf.shape == (12, 16)
    assert np.isclose(surf.ravel()[_flat(100, 34)], 0.3)


def test_empty_cell_is_zero_not_nan():
    surf = xg_scoring_prob(make_cohort([_row(0, PASS, SUCCESS, 10, 34, 20, 34, xg=np.nan)]),
                           xg_column="xg", l=16, w=12)
    assert np.all(np.isfinite(surf)) and np.all(surf == 0.0)


def test_nan_coord_shots_excluded():
    rows = [_row(0, SHOT, SUCCESS, np.nan, 34, 105, 34, xg=0.9),
            _row(1, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.3)]
    surf = xg_scoring_prob(make_cohort(rows), xg_column="xg", l=16, w=12)
    assert np.isclose(surf.ravel()[_flat(100, 34)], 0.3)


def test_weighted_sum_layout_matches_count():
    from silly_kicks.xthreat._grid import _count
    from silly_kicks.xtgk._xg_reward import _weighted_cell_sum
    rows = [_row(i, SHOT, SUCCESS, 10 + 7 * i, 20 + 3 * i, 105, 34, xg=1.0) for i in range(8)]
    a = make_cohort(rows)
    assert np.array_equal(_count(a.start_x, a.start_y, 16, 12).astype(float),
                          _weighted_cell_sum(a.start_x, a.start_y, a["xg"], 16, 12))


def test_missing_xg_column_raises():
    import pytest
    with pytest.raises(ValueError, match="xg_column"):
        xg_scoring_prob(make_cohort([_row(0, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.3)]),
                        xg_column="nope", l=16, w=12)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_xg_reward.py`:

```python
"""Per-cell immediate reward E[xG | shot in cell] from an injected xg_column (ADR-036 §4.1).

The xG analogue of xthreat._grid._scoring_prob: goal COUNTS -> an xG SUM over shots, / shot
counts. NOT goal-gated (that is the v1 degeneracy). Own goals never appear (not the
possessing team's shot xG — N4). Same (w,l) layout as _scoring_prob; access via .ravel()[flat].
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import M, N, _count, _get_flat_indexes, _safe_divide


def _weighted_cell_sum(x: pd.Series, y: pd.Series, w: pd.Series, l: int, ww: int) -> npt.NDArray[np.float64]:
    # Layout MUST match xthreat._grid._count exactly: reshape the flat-indexed vector to
    # (ww, l) with NO extra flip (the y-flip already lives in _get_flat_indexes).
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(w))
    x, y, w = x[mask], y[mask], w[mask]
    flat = _get_flat_indexes(x, y, l, ww).to_numpy()
    out = np.zeros(ww * l, dtype=np.float64)
    np.add.at(out, flat, w.to_numpy(dtype=np.float64))
    return out.reshape((ww, l))


def xg_scoring_prob(actions: pd.DataFrame, *, xg_column: str, l: int = N, w: int = M) -> npt.NDArray[np.float64]:
    """E[xG|shot] per grid cell, shape (w, l), same layout as xthreat _scoring_prob."""
    if xg_column not in actions.columns:
        raise ValueError(
            f"xg_column {xg_column!r} not found. Supply a calibrated per-shot xG column "
            f"(silly-kicks ships no xG model; see ADR-036 §6)."
        )
    shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]].dropna(
        subset=["start_x", "start_y"]
    )
    shotmatrix = _count(shots.start_x, shots.start_y, l, w)
    xgsum = _weighted_cell_sum(shots.start_x, shots.start_y, shots[xg_column], l, w)
    return _safe_divide(xgsum, shotmatrix)
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_xg_reward.py -v`

---

## Phase 3 — Pressure discretization

### Task 4: `PressureLevels` (terciles + occupancy)

**Files:** Create `silly_kicks/xtgk/_pressure_levels.py`, `tests/xtgk/test_pressure_levels.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_pressure_levels.py`:

```python
import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._pressure_levels import PressureLevels


def test_global_terciles_partition_roughly_thirds():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 900)))
    counts = pd.Series(pl.apply(pd.Series(np.linspace(0, 1, 900)))).value_counts()
    assert set(counts.index) == {1, 2, 3}
    assert all(280 <= counts[k] <= 320 for k in (1, 2, 3))


def test_three_band_input_populates_all_levels():
    # the fixture bands 0.1/0.5/0.9 must land in distinct terciles (guards B1 regression)
    pl = PressureLevels(mode="global").fit(pd.Series([0.1, 0.5, 0.9] * 100))
    lv = pl.apply(pd.Series([0.1, 0.5, 0.9]))
    assert set(lv.tolist()) == {1, 2, 3}


def test_apply_stable_to_persisted_cutpoints():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    pl2 = PressureLevels.from_cutpoints(pl.cutpoints)
    assert np.array_equal(pl.apply(pd.Series([0.1, 0.5, 0.9])), pl2.apply(pd.Series([0.1, 0.5, 0.9])))


def test_missing_pressure_raises():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    with pytest.raises(ValueError, match="missing pressure"):
        pl.apply(pd.Series([0.1, np.nan, 0.9]))


def test_occupancy_report_counts_per_level():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    rep = pl.occupancy(pd.Series(np.linspace(0, 1, 300)))
    assert all(rep[k] == pytest.approx(100, abs=5) for k in (1, 2, 3))
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_pressure_levels.py`:

```python
"""Continuous pressure -> {1,2,3} tercile quantizer (ADR-036 §5).

fit() learns cutpoints on the fit cohort; apply() maps new actions; cutpoints persist with the
surface. Occupancy is reported so a degenerate deep-zone stratification (M3) is visible.
NOTE (plan-review m5): heavy ties skew tercile fill; continuous tracking pressure is fine, but
a 2-value input collapses to two levels — always feed >=3 well-separated bands in fixtures.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Mode = Literal["global", "zone_conditional"]


class PressureLevels:
    def __init__(self, *, mode: Mode = "global") -> None:
        self.mode: Mode = mode
        self.cutpoints: tuple[float, float] | None = None

    def fit(self, pressure: "pd.Series[float]") -> "PressureLevels":
        p = pressure.dropna().to_numpy(dtype=float)
        if p.size == 0:
            raise ValueError("cannot fit pressure terciles on empty/all-NaN pressure")
        lo, hi = np.quantile(p, [1 / 3, 2 / 3])
        self.cutpoints = (float(lo), float(hi))
        return self

    @classmethod
    def from_cutpoints(cls, cutpoints: tuple[float, float], *, mode: Mode = "global") -> "PressureLevels":
        obj = cls(mode=mode)
        obj.cutpoints = (float(cutpoints[0]), float(cutpoints[1]))
        return obj

    def apply(self, pressure: "pd.Series[float]") -> np.ndarray:
        if self.cutpoints is None:
            raise ValueError("PressureLevels not fitted")
        if pressure.isna().any():
            raise ValueError("missing pressure value(s); never default a level (ADR-036 §5)")
        lo, hi = self.cutpoints
        p = pressure.to_numpy(dtype=float)
        return np.where(p <= lo, 1, np.where(p <= hi, 2, 3)).astype(int)

    def occupancy(self, pressure: "pd.Series[float]") -> dict[int, int]:
        lv = self.apply(pressure)
        return {k: int((lv == k).sum()) for k in (1, 2, 3)}
```

> `mode="zone_conditional"` is reserved (M3): terciles-within-zone-band, built only if the owner-run occupancy report shows a degenerate deep stratum. Its unit test is added in the owner script; keep the partition assertion **mode-conditional** (G6) — never assert global-⅓ under zone-conditional mode.

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_pressure_levels.py -v`

---

## Phase 4 — Input validator (G5)

### Task 5: `validate_possession_value_input`

**Files:** Create `silly_kicks/xtgk/_validate.py`, `tests/xtgk/test_validate.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_validate.py`:

```python
import numpy as np

from silly_kicks.xtgk._validate import validate_possession_value_input
from tests.xtgk.conftest import make_cohort, _row, GOALKICK, SHOT, SUCCESS, mirror_x, three_band_cohort


def _ok():
    return make_cohort([
        _row(0, GOALKICK, SUCCESS, 5, 34, 60, 34, xg=np.nan, pressure=0.2),
        _row(1, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.3, pressure=0.8),
    ])


def test_ok_input_passes():
    diag = validate_possession_value_input(_ok(), xg_column="xg", pressure_column="pressure")
    assert diag.ok is True and diag.problems == []


def test_missing_columns_flagged():
    diag = validate_possession_value_input(_ok(), xg_column="missing", pressure_column="pressure")
    assert diag.ok is False and any("missing" in p for p in diag.problems)


def test_attack_reversed_orientation_flagged():
    diag = validate_possession_value_input(mirror_x(three_band_cohort(20)),
                                           xg_column="xg", pressure_column="pressure")
    assert diag.ok is False and any("orientation" in p.lower() for p in diag.problems)


def test_require_possession_id_for_crosscheck():
    diag = validate_possession_value_input(_ok().drop(columns=["possession_id"]),
                                           xg_column="xg", pressure_column="pressure",
                                           require_possession_id=True)
    assert diag.ok is False and any("possession_id" in p for p in diag.problems)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_validate.py`:

```python
"""Opt-in loud-guard for MarkovPossessionValue.fit inputs (ADR-036 §11, G5).

House style: one diagnosis object (cf. validate_time_base ADR-017, validate_id_dtypes ADR-019),
the natural home for the §M4 attack-orientation guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import silly_kicks.spadl.config as spadlconfig

_REQUIRED = ["game_id", "period_id", "action_id", "type_id", "result_id",
             "start_x", "start_y", "end_x", "end_y"]


@dataclass(frozen=True)
class PossessionValueInputDiagnosis:
    ok: bool
    problems: list[str] = field(default_factory=list)


def validate_possession_value_input(
    actions: pd.DataFrame, *, xg_column: str, pressure_column: str,
    require_possession_id: bool = False,
) -> PossessionValueInputDiagnosis:
    problems: list[str] = []
    for c in _REQUIRED + [xg_column, pressure_column]:
        if c not in actions.columns:
            problems.append(f"missing required column: {c!r}")
    if require_possession_id and "possession_id" not in actions.columns:
        problems.append("missing 'possession_id' (call spadl.add_possessions first)")
    if "type_id" in actions.columns and "start_x" in actions.columns:
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        if len(shots) >= 10:
            frac_far = (shots["start_x"] > spadlconfig.field_length / 2).mean()
            if frac_far < 0.5:
                problems.append(
                    f"orientation: only {frac_far:.0%} of shots are in the attacking half; "
                    f"actions must be attack-LTR (attack toward x=105) — ADR-028/§M4"
                )
    return PossessionValueInputDiagnosis(ok=(len(problems) == 0), problems=problems)
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_validate.py -v`

---

## Phase 5 — Port + shared types

### Task 6: `PressureLevel`, `State`, `DeltaV`, `PossessionValue` Protocol, `zone_of`

**Files:** Create `silly_kicks/xtgk/_possession_value.py`, `tests/xtgk/test_types.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_types.py`:

```python
from silly_kicks.xtgk._possession_value import DeltaV, State, zone_of


def test_zone_of_matches_flat_convention():
    # x~3 -> xi=0, y=34 -> yj=6 -> flat=(12-1-6)*16+0 = 80
    assert zone_of(3.0, 34.0) == (11 - 6) * 16 + 0


def test_deltav_identity_holds_by_construction():
    dv = DeltaV(delta=0.5, pressure_component=0.2, position_component=0.3)
    assert abs((dv.pressure_component + dv.position_component) - dv.delta) < 1e-12


def test_state_carries_zone_and_pressure():
    s = State(zone=80, pressure_level=1)
    assert s.zone == 80 and s.pressure_level == 1
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_possession_value.py`:

```python
"""PossessionValue port + shared value types (ADR-036 §3, §7, §9)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

PressureLevel = Literal[1, 2, 3]


@dataclass(frozen=True)
class State:
    zone: int
    pressure_level: PressureLevel


@dataclass(frozen=True)
class DeltaV:
    delta: float
    pressure_component: float
    position_component: float


@runtime_checkable
class PossessionValue(Protocol):
    def value(self, zone: int, p: PressureLevel) -> float: ...
    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]: ...
    def delta_v(self, s: State, s_next: State) -> DeltaV: ...


def zone_of(x: float, y: float, l: int = N, w: int = M) -> int:
    """Flat grid index for a coordinate, matching value_iteration's ravel()."""
    return int(_get_flat_indexes(pd.Series([float(x)]), pd.Series([float(y)]), l, w).iloc[0])
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_types.py -v`

---

## Phase 6 — `MarkovPossessionValue` (production)

### Task 7: `fit` builds three pressure surfaces

**Files:** Create `silly_kicks/xtgk/_markov.py`, `tests/xtgk/test_markov.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_markov.py`:

```python
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import State, zone_of
from tests.xtgk.conftest import three_band_cohort


def _fit():
    return MarkovPossessionValue().fit(three_band_cohort(), xg_column="xg", pressure_column="pressure")


def test_fit_returns_three_surfaces_of_grid_shape():
    m = _fit()
    for p in (1, 2, 3):
        assert m.surface(p).shape == (12, 16)


def test_all_three_levels_populated():
    # guards B1: every tercile must have move support (else the gradient is measured on a void)
    m = _fit()
    for p in (1, 2, 3):
        assert m.support(p).sum() > 0


def test_value_before_fit_raises():
    with pytest.raises(NotFittedError):
        MarkovPossessionValue().value(0, 1)


def test_deep_value_nonzero_and_pressure_ordered():
    m = _fit()
    z = zone_of(3.0, 34.0)
    v1, v2, v3 = m.value(z, 1), m.value(z, 2), m.value(z, 3)
    assert v1 > 0.0                    # deep build-up carries value via propagation
    assert v1 >= v2 >= v3              # xg decreases with pressure in the fixture
    assert v3 < v1                     # a REAL gap (not because a stratum is empty — see above)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_markov.py`:

```python
"""MarkovPossessionValue — production possession-value surface (ADR-036 §4).

Reuses xthreat.value_iteration verbatim with (i) an xG-calibrated immediate reward, (ii) a
goal-kick-inclusive move-set, (iii) pressure stratification. V(z,p) = E[xG of the possession's
FIRST shot | ball at z under pressure p] — the shoot branch is terminal, so the recursion
values the first shot; deep-zone value is pure forward propagation. See NOTICE / ADR-036 §4.2.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

from silly_kicks.xthreat import GridSpec, value_iteration
from silly_kicks.xthreat._grid import M, N, _count
from silly_kicks.xthreat._params import KDEParams
from silly_kicks.xtgk._moves import extended_move_actions, xtgk_action_prob, xtgk_transition_matrix
from silly_kicks.xtgk._possession_value import DeltaV, PressureLevel, State
from silly_kicks.xtgk._validate import validate_possession_value_input
from silly_kicks.xtgk._xg_reward import xg_scoring_prob

Method = Literal["singh_counts", "kde_smoothed"]
_LEVELS: tuple[PressureLevel, ...] = (1, 2, 3)


class MarkovPossessionValue:
    def __init__(self, *, l: int = N, w: int = M, eps: float = 1e-5,
                 method: Method = "singh_counts") -> None:
        self.l, self.w, self.eps, self.method = l, w, eps, method
        self.grid = GridSpec(n_zones_x=l, n_zones_y=w)
        self._surfaces: dict[PressureLevel, npt.NDArray[np.float64]] = {}
        self._support: dict[PressureLevel, npt.NDArray[np.int_]] = {}
        self._fitted = False
        self.xg_column: str | None = None
        self.pressure_levels = None
        self.provenance: dict = {}

    def fit(self, actions: pd.DataFrame, *, xg_column: str, pressure_column: str,
            pressure_levels=None) -> "MarkovPossessionValue":
        diag = validate_possession_value_input(actions, xg_column=xg_column,
                                               pressure_column=pressure_column)
        if not diag.ok:
            raise ValueError("invalid fit input: " + "; ".join(diag.problems))
        from silly_kicks.xtgk._pressure_levels import PressureLevels
        pl = pressure_levels or PressureLevels().fit(actions[pressure_column])
        levels = pl.apply(actions[pressure_column])
        actions = actions.assign(_p_level=levels)
        for p in _LEVELS:
            sub = actions[actions["_p_level"] == p]
            if len(sub) == 0:
                warnings.warn(
                    f"pressure tercile {p} has zero actions at fit; its surface is all-zero "
                    f"(check pressure distribution / cutpoints — ADR-036 §5/B1)",
                    stacklevel=2,
                )
            self._surfaces[p] = self._solve_level(sub, xg_column)
            self._support[p] = self._support_counts(sub)
        self.xg_column, self.pressure_levels, self._fitted = xg_column, pl, True
        self.provenance = {"xg_column": xg_column, "method": self.method,
                           "grid": (self.l, self.w), "cutpoints": pl.cutpoints,
                           "n_actions": int(len(actions))}
        return self

    def _solve_level(self, sub: pd.DataFrame, xg_column: str) -> npt.NDArray[np.float64]:
        xg_scoring = xg_scoring_prob(sub, xg_column=xg_column, l=self.l, w=self.w)
        p_shot, p_move = xtgk_action_prob(sub, self.l, self.w)
        T = xtgk_transition_matrix(sub, self.grid, method=self.method,
                                   params=KDEParams() if self.method == "kde_smoothed" else None)
        xt, _ = value_iteration(xg_scoring, p_shot, p_move, T, eps=self.eps)
        return xt

    def _support_counts(self, sub: pd.DataFrame) -> npt.NDArray[np.int_]:
        moves = extended_move_actions(sub).dropna(subset=["start_x", "start_y"])
        return _count(moves.start_x, moves.start_y, self.l, self.w)

    def _check(self) -> None:
        if not self._fitted:
            raise NotFittedError("MarkovPossessionValue.fit not called")

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        self._check(); return self._surfaces[p]

    def value(self, zone: int, p: PressureLevel) -> float:
        self._check(); return float(self._surfaces[p].ravel()[zone])

    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]:
        self._check(); return self._support[p]

    def delta_v(self, s: State, s_next: State) -> DeltaV:
        self._check()
        z, p, zp, pp = s.zone, s.pressure_level, s_next.zone, s_next.pressure_level
        v_zp, v_zpp, v_zpp_, v_zp_pp = self.value(z, p), self.value(z, pp), self.value(zp, p), self.value(zp, pp)
        delta = v_zp_pp - v_zp
        pressure = 0.5 * ((v_zpp - v_zp) + (v_zp_pp - v_zpp_))
        position = 0.5 * ((v_zpp_ - v_zp) + (v_zp_pp - v_zpp))
        return DeltaV(delta=delta, pressure_component=pressure, position_component=position)
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_markov.py -v`

### Task 8: `delta_v` Shapley identity + unsupported-corner (characterization tests — not red-first)

> **Note (m3):** the `delta_v` from Task 7 already satisfies these; these are **characterization** tests that pin current behaviour, not red→green.

**Files:** append to `tests/xtgk/test_markov.py`

- [ ] **Step 1: Add tests:**

```python
def test_delta_v_shapley_identity():
    m = _fit()
    dv = m.delta_v(State(zone_of(3.0, 34.0), 1), State(zone_of(100.0, 34.0), 3))
    assert abs((dv.pressure_component + dv.position_component) - dv.delta) < 1e-12


def test_delta_v_on_unsupported_corner_is_finite():
    m = _fit()
    dv = m.delta_v(State(191, 1), State(0, 2))  # empty cells solve to 0.0 (absorbing), never NaN
    assert np.isfinite(dv.delta) and np.isfinite(dv.pressure_component)
```

- [ ] **Step 2: Run — PASS.** `python -m pytest tests/xtgk/test_markov.py -v`

---

## Phase 7 — `EmpiricalPossessionValue` (cross-check, D2)

### Task 9: per-action first-shot empirical surface (O(n) reverse scan)

**Files:** Create `silly_kicks/xtgk/_empirical.py`, `tests/xtgk/test_empirical.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_empirical.py`:

```python
import numpy as np

from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._possession_value import zone_of
from silly_kicks.xtgk._pressure_levels import PressureLevels
from tests.xtgk.conftest import make_cohort, _row, PASS, SUCCESS, three_band_cohort


def test_first_shot_value_matches_terminal_xg_per_tercile():
    a = three_band_cohort()
    pl = PressureLevels().fit(a["pressure"])
    m = EmpiricalPossessionValue().fit(a, xg_column="xg", pressure_column="pressure",
                                       aggregation="first_shot", pressure_levels=pl)
    z = zone_of(3.0, 34.0)
    assert np.isclose(m.value(z, 1), 0.5, atol=1e-9)   # low-pressure band shot xg
    assert np.isclose(m.value(z, 3), 0.05, atol=1e-9)  # high-pressure band shot xg


def test_no_shot_possession_contributes_zero():
    rows = [_row(0, PASS, SUCCESS, 3, 34, 40, 34, possession_id=0, pressure=0.1)]
    m = EmpiricalPossessionValue().fit(make_cohort(rows), xg_column="xg",
                                       pressure_column="pressure", aggregation="first_shot")
    assert m.value(zone_of(3.0, 34.0), 1) == 0.0


def test_reverse_scan_matches_naive_first_shot():
    from silly_kicks.xtgk._empirical import _possession_outcomes
    from tests.xtgk.conftest import SHOT
    rows = [
        _row(0, PASS, SUCCESS, 3, 34, 40, 34, possession_id=0, pressure=0.1),
        _row(1, PASS, SUCCESS, 40, 34, 80, 34, possession_id=0, pressure=0.1),
        _row(2, SHOT, SUCCESS, 90, 34, 105, 34, possession_id=0, pressure=0.1, xg=0.3),
        _row(3, SHOT, SUCCESS, 95, 34, 105, 34, possession_id=0, pressure=0.1, xg=0.7),
    ]
    out = _possession_outcomes(make_cohort(rows), "xg", "first_shot")
    # actions 0,1 -> first shot xg 0.3; action 2 -> next shot 0.7; action 3 -> none 0.0
    assert list(out) == [0.3, 0.3, 0.7, 0.0]
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_empirical.py`:

```python
"""EmpiricalPossessionValue — model-free cross-check (ADR-036 §M1). NOT shipped.

Per action, outcome = xG of the FIRST shot after it in the same possession (0 if none),
averaged per (cell, tercile). This 'first_shot' aggregation is the like-for-like estimator of
the Markov target (D2). Independent of the Markov estimator (no shared transitions), which is
what makes disagreement diagnostic (§8.5). Partial port: surface/value only (G9).

Coincidence band (m1): first_shot EXCLUDES the action's own shot, so at shot-origin (final-
third) cells it undercounts vs the Markov gs immediate term — harmless BECAUSE the gate
compares only on BUILD-UP cells, exactly the region where the two coincide. Never compare on
final-third cells.

O(n) per possession via a right-to-left scan (m4).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import M, N, _get_flat_indexes
from silly_kicks.xtgk._possession_value import PressureLevel

Aggregation = Literal["first_shot", "noisy_or", "sum"]
_LEVELS: tuple[PressureLevel, ...] = (1, 2, 3)
_SHOT = spadlconfig.actiontype_id["shot"]


def _possession_outcomes(a: pd.DataFrame, xg_column: str, aggregation: Aggregation) -> np.ndarray:
    """For each action, the aggregated xG of shots strictly AFTER it within its possession."""
    out = np.zeros(len(a), dtype=float)
    pos = {ix: i for i, ix in enumerate(a.index)}
    group_cols = ["game_id", "possession_id"] if "game_id" in a.columns else ["possession_id"]
    for _key, grp in a.groupby(group_cols, sort=False):
        idx = list(grp.index)
        is_shot = (grp["type_id"] == _SHOT).to_numpy()
        xg = grp[xg_column].fillna(0.0).to_numpy(dtype=float)
        acc_first = 0.0          # first-shot value for the current position
        acc_sum = 0.0            # sum of shot xg after
        acc_prod = 1.0           # product of (1-xg) after -> noisy_or = 1 - prod
        for i in range(len(idx) - 1, -1, -1):
            if aggregation == "first_shot":
                out[pos[idx[i]]] = acc_first
            elif aggregation == "sum":
                out[pos[idx[i]]] = acc_sum
            else:
                out[pos[idx[i]]] = 1.0 - acc_prod
            if is_shot[i]:
                acc_first = xg[i]
                acc_sum += xg[i]
                acc_prod *= (1.0 - xg[i])
    return out


class EmpiricalPossessionValue:
    def __init__(self, *, l: int = N, w: int = M) -> None:
        self.l, self.w = l, w
        self._surfaces: dict[PressureLevel, npt.NDArray[np.float64]] = {}
        self._fitted = False

    def fit(self, actions: pd.DataFrame, *, xg_column: str, pressure_column: str,
            aggregation: Aggregation = "first_shot", pressure_levels=None) -> "EmpiricalPossessionValue":
        from silly_kicks.xtgk._pressure_levels import PressureLevels
        pl = pressure_levels or PressureLevels().fit(actions[pressure_column])
        a = actions.reset_index(drop=True).copy()
        a["_p_level"] = pl.apply(a[pressure_column])
        a["_outcome"] = _possession_outcomes(a, xg_column, aggregation)
        for p in _LEVELS:
            sub = a[a["_p_level"] == p].dropna(subset=["start_x", "start_y"])
            flat = _get_flat_indexes(sub.start_x, sub.start_y, self.l, self.w).to_numpy()
            num = np.zeros(self.w * self.l); den = np.zeros(self.w * self.l)
            np.add.at(num, flat, sub["_outcome"].to_numpy(dtype=float))
            np.add.at(den, flat, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                surf = np.where(den > 0, num / den, 0.0)
            self._surfaces[p] = surf.reshape((self.w, self.l))
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted:
            raise NotFittedError("EmpiricalPossessionValue.fit not called")

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        self._check(); return self._surfaces[p]

    def value(self, zone: int, p: PressureLevel) -> float:
        self._check(); return float(self._surfaces[p].ravel()[zone])
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_empirical.py -v`

---

## Phase 8 — Diagnostics / occupied-cell gate (B2, M4, m2)

### Task 10: pre-registered gate on OCCUPIED deep cells

**Files:** Create `silly_kicks/xtgk/_diagnostics.py`, `tests/xtgk/test_diagnostics.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_diagnostics.py`:

```python
from silly_kicks.xtgk._diagnostics import DEEP_ZONE_CELLS, GateConfig, run_deep_zone_gate
from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._pressure_levels import PressureLevels
from tests.xtgk.conftest import flat_no_shot_cohort, three_band_cohort


def _fit_pair(a):
    pl = PressureLevels().fit(a["pressure"])          # m2: ONE tercile fit shared by both
    mk = MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    emp = EmpiricalPossessionValue().fit(a, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    return mk, emp


def test_deep_zone_cells_are_first_two_columns():
    assert len(DEEP_ZONE_CELLS) == 24 and all((c % 16) in (0, 1) for c in DEEP_ZONE_CELLS)


def test_gate_passes_on_honest_cohort():
    mk, emp = _fit_pair(three_band_cohort())
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2))
    assert rep.passed is True
    assert rep.n_occupied_cells >= 2
    assert rep.effect_size > 0.005
    assert rep.observed_direction == "decreasing"


def test_gate_stops_on_too_few_occupied_cells():
    mk, emp = _fit_pair(three_band_cohort())
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=10_000_000,
                                                 min_occupied_cells=2))
    assert rep.passed is False and "support" in rep.stop_reason.lower()


def test_gate_fails_on_flat_negative_control():
    mk, emp = _fit_pair(flat_no_shot_cohort())
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2))
    assert rep.passed is False


def test_gate_fails_when_crosscheck_disagrees_on_buildup():
    # real mk passes effect + monotonicity; a stub empirical surface with an OPPOSITE build-up
    # gradient must flip crosscheck_agrees to False and fail the gate — this pins G1's mechanism
    # in the failing direction (plan-review R2). Requires the build-up band to be populated (the
    # R2 fixture routing), else mk_grad would be 0 and the sign comparison vacuous.
    mk, _ = _fit_pair(three_band_cohort())

    class _DisagreeingEmp:
        def value(self, zone, p):
            return {1: 0.0, 2: 0.05, 3: 0.2}[p]  # rises with pressure -> opposite sign to mk

    rep = run_deep_zone_gate(mk, _DisagreeingEmp(),
                             GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2))
    assert rep.crosscheck_agrees is False and rep.passed is False


def test_expected_direction_configurable_and_reported():
    mk, emp = _fit_pair(three_band_cohort())
    # forcing 'increasing' must FAIL an actually-decreasing gradient, but still report it
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=3,
                                                 min_occupied_cells=2, expected_direction="increasing"))
    assert rep.observed_direction == "decreasing" and rep.passed is False
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_diagnostics.py`:

```python
"""Pre-registered deep-zone go/no-go gate (ADR-036 §8, BLOCKING).

Numbers (effect_floor, n_min, min_occupied_cells, direction) are LOCKED by owner/Eyestone
before fitting (Q4); GateConfig carries them so the STRUCTURE is testable on synthetic data now.

Occupied-cell semantics (plan-review B2, m8): a keeper populates only a HANDFUL of deep cells,
so the gate operates on deep cells with >= n_min support in ALL THREE terciles (the effect check
reads level 2 as well, so every averaged tercile must be supported); it requires at least
min_occupied_cells such cells (else STOP — the gate cannot run) and computes effect /
monotonicity over ONLY those cells. Direction is configurable (M4, Q2 still open) and always
reported. Cross-check agreement is graded on BUILD-UP cells, not deep cells (G1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue

_L, _W = 16, 12
DEEP_ZONE_CELLS: tuple[int, ...] = tuple((_W - 1 - yj) * _L + xi for yj in range(_W) for xi in (0, 1))
BUILD_UP_CELLS: tuple[int, ...] = tuple((_W - 1 - yj) * _L + xi for yj in range(_W) for xi in range(2, 7))

Direction = Literal["decreasing", "increasing", "either"]


@dataclass(frozen=True)
class GateConfig:
    effect_floor: float
    n_min: int
    min_occupied_cells: int = 2
    crosscheck_rel_tol: float = 0.5
    expected_direction: Direction = "either"


@dataclass(frozen=True)
class DeepZoneGateReport:
    passed: bool
    effect_size: float
    observed_direction: str
    monotone_ok: bool
    crosscheck_agrees: bool
    n_occupied_cells: int
    stop_reason: str


def _occupied(mk: MarkovPossessionValue, cfg: GateConfig) -> list[int]:
    s = {p: mk.support(p).ravel() for p in (1, 2, 3)}  # all three terciles (m8)
    return [c for c in DEEP_ZONE_CELLS if all(s[p][c] >= cfg.n_min for p in (1, 2, 3))]


def _mean(fn, cells, p) -> float:
    return float(np.mean([fn(c, p) for c in cells])) if cells else 0.0


def run_deep_zone_gate(mk: MarkovPossessionValue, emp: EmpiricalPossessionValue,
                       cfg: GateConfig) -> DeepZoneGateReport:
    occ = _occupied(mk, cfg)
    if len(occ) < cfg.min_occupied_cells:
        return DeepZoneGateReport(False, 0.0, "n/a", False, False, len(occ),
                                  f"insufficient support: {len(occ)} occupied deep cells "
                                  f"(>= n_min in both terciles) < {cfg.min_occupied_cells}")
    v1, v2, v3 = _mean(mk.value, occ, 1), _mean(mk.value, occ, 2), _mean(mk.value, occ, 3)
    effect = abs(v1 - v3)
    nonincreasing = v1 >= v2 >= v3
    nondecreasing = v1 <= v2 <= v3
    observed = "decreasing" if v1 > v3 else ("increasing" if v1 < v3 else "flat")
    if cfg.expected_direction == "decreasing":
        monotone_ok = nonincreasing
    elif cfg.expected_direction == "increasing":
        monotone_ok = nondecreasing
    else:
        monotone_ok = nonincreasing or nondecreasing
    mk_grad = _mean(mk.value, BUILD_UP_CELLS, 1) - _mean(mk.value, BUILD_UP_CELLS, 3)
    emp_grad = _mean(emp.value, BUILD_UP_CELLS, 1) - _mean(emp.value, BUILD_UP_CELLS, 3)
    same_sign = np.sign(mk_grad) == np.sign(emp_grad)
    rel_ok = abs(mk_grad - emp_grad) <= cfg.crosscheck_rel_tol * max(abs(mk_grad), abs(emp_grad), 1e-9)
    crosscheck = bool(same_sign and rel_ok)
    passed = bool(effect >= cfg.effect_floor and monotone_ok and crosscheck)
    reason = "" if passed else "; ".join(s for s, ok in [
        (f"effect {effect:.4f}<{cfg.effect_floor}", effect >= cfg.effect_floor),
        (f"direction {observed}!={cfg.expected_direction}/non-monotone", monotone_ok),
        ("cross-check divergent", crosscheck)] if not ok)
    return DeepZoneGateReport(passed, effect, observed, monotone_ok, crosscheck, len(occ), reason)
```

- [ ] **Step 4: Run — PASS.** `python -m pytest tests/xtgk/test_diagnostics.py -v`

---

## Phase 9 — Orientation properties (M3) + negative control (G7)

### Task 11: y-equivariance + attack-reversal rejection + negative control

**Files:** Create `tests/xtgk/test_honesty_property.py`

- [ ] **Step 1: Add tests** — `tests/xtgk/test_honesty_property.py`:

```python
import numpy as np
import pytest

from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import zone_of
from tests.xtgk.conftest import flat_no_shot_cohort, mirror_x, mirror_y, three_band_cohort


def _fit(a):
    return MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure")


def test_surface_is_y_reflection_equivariant():
    # y->68-y preserves attack-LTR; a fit on y-mirrored data must be the row-reversed surface.
    a = three_band_cohort()
    S = _fit(a).surface(1)
    Sm = _fit(mirror_y(a)).surface(1)
    assert np.allclose(Sm, S[::-1, :], atol=1e-9)


def test_attack_reversed_input_is_rejected_not_fit():
    # x->105-x reverses attack; the fit MUST refuse it (the orientation guard), never value it.
    with pytest.raises(ValueError, match="orientation"):
        _fit(mirror_x(three_band_cohort()))


def test_negative_control_flat_cohort_gives_flat_deep_value():
    m = _fit(flat_no_shot_cohort())
    z = zone_of(3.0, 34.0)
    assert m.value(z, 1) < 1e-6 and m.value(z, 3) < 1e-6


def test_honest_cohort_deep_gradient_positive():
    m = _fit(three_band_cohort())
    z = zone_of(3.0, 34.0)
    assert m.value(z, 1) > 0.0 and m.value(z, 1) > m.value(z, 3)
```

- [ ] **Step 2: Run.**

Run: `python -m pytest tests/xtgk/test_honesty_property.py -v`
Expected: PASS. If `test_surface_is_y_reflection_equivariant` fails, there is a hidden orientation dependence in the grid indexing — fix the code, not the test (pin-the-ground-truth). If `test_attack_reversed_input_is_rejected` fails, the §M4 guard is too weak.

---

## Phase 10 — Serialization (M5), wiring, regression boundary

### Task 12: pickle-free artifact `save`/`load` (G4)

**Files:** Create `silly_kicks/xtgk/_serialize.py`, `tests/xtgk/test_serialize.py`; modify `silly_kicks/xtgk/_markov.py`

- [ ] **Step 1: Failing test** — `tests/xtgk/test_serialize.py`:

```python
import numpy as np

from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import zone_of
from tests.xtgk.conftest import three_band_cohort


def test_save_load_roundtrip_is_exact(tmp_path):
    a = three_band_cohort()
    m = MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure")
    out = tmp_path / "surface"
    m.save(out)
    m2 = MarkovPossessionValue.load(out)
    for p in (1, 2, 3):
        assert np.array_equal(m.surface(p), m2.surface(p))
        assert np.array_equal(m.support(p), m2.support(p))
    assert m2.provenance["xg_column"] == "xg"
    assert m2.pressure_levels.cutpoints == m.pressure_levels.cutpoints
    z = zone_of(3.0, 34.0)
    assert m.value(z, 1) == m2.value(z, 1)


def test_load_detects_tampering(tmp_path):
    import pytest
    a = three_band_cohort()
    m = MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure")
    out = tmp_path / "surface"
    m.save(out)
    (out / "SHA256SUMS").write_text("deadbeef  surfaces.npz\n")
    with pytest.raises(ValueError, match="checksum"):
        MarkovPossessionValue.load(out)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — `silly_kicks/xtgk/_serialize.py`:

```python
"""Pickle-free artifact I/O for the fitted possession-value surfaces (ADR-036 §4/G4).

Mirrors the house convention (ghost-GK / xShot / GkCompletionModel): npz for arrays, JSON for
metadata, SHA256SUMS for integrity. No pickle. A fitted grid, not an ADR-011 weights lifecycle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_ARRAYS = "surfaces.npz"
_META = "metadata.json"
_SUMS = "SHA256SUMS"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_surface(directory, *, surfaces: dict, support: dict, metadata: dict) -> None:
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for p in (1, 2, 3):
        arrays[f"surface_{p}"] = surfaces[p]
        arrays[f"support_{p}"] = support[p]
    np.savez(d / _ARRAYS, **arrays)
    (d / _META).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    lines = [f"{_sha256(d / f)}  {f}\n" for f in (_ARRAYS, _META)]
    (d / _SUMS).write_text("".join(lines))


def load_surface(directory):
    d = Path(directory)
    expected = {}
    for line in (d / _SUMS).read_text().splitlines():
        h, f = line.split("  ", 1)
        expected[f.strip()] = h.strip()
    for f in (_ARRAYS, _META):
        if _sha256(d / f) != expected.get(f):
            raise ValueError(f"checksum mismatch for {f} — artifact tampered or corrupt")
    npz = np.load(d / _ARRAYS)
    surfaces = {p: npz[f"surface_{p}"] for p in (1, 2, 3)}
    support = {p: npz[f"support_{p}"] for p in (1, 2, 3)}
    metadata = json.loads((d / _META).read_text())
    return surfaces, support, metadata
```

- [ ] **Step 4: Add `save`/`load` to `MarkovPossessionValue`** (append methods in `_markov.py`):

```python
    def save(self, directory) -> None:
        self._check()
        from silly_kicks.xtgk._serialize import save_surface
        meta = dict(self.provenance)
        meta["cutpoints"] = list(self.pressure_levels.cutpoints)
        save_surface(directory, surfaces=self._surfaces, support=self._support, metadata=meta)

    @classmethod
    def load(cls, directory) -> "MarkovPossessionValue":
        from silly_kicks.xtgk._pressure_levels import PressureLevels
        from silly_kicks.xtgk._serialize import load_surface
        surfaces, support, meta = load_surface(directory)
        l, w = meta["grid"]
        obj = cls(l=int(l), w=int(w), method=meta.get("method", "singh_counts"))
        obj._surfaces = {int(p): surfaces[p] for p in (1, 2, 3)}
        obj._support = {int(p): support[p] for p in (1, 2, 3)}
        obj.provenance = meta
        obj.xg_column = meta.get("xg_column")
        obj.pressure_levels = PressureLevels.from_cutpoints(tuple(meta["cutpoints"]))
        obj._fitted = True
        return obj
```

> **Note:** `np.load`/`np.savez` do not pickle for numeric arrays (no `allow_pickle`); keep it that way. `metadata["grid"]` round-trips through JSON as a list — the `load` casts back to ints.

- [ ] **Step 5: Run — PASS.** `python -m pytest tests/xtgk/test_serialize.py -v`

### Task 13: `__init__` exports + regression boundary

**Files:** Modify `silly_kicks/xtgk/__init__.py`; create `tests/xtgk/test_regression_boundary.py`

- [ ] **Step 1:** Fill `silly_kicks/xtgk/__init__.py`:

```python
"""xT-GK v2 — honest possession-value surface V(z,p). See ADR-036 / NOTICE."""
from silly_kicks.xtgk._diagnostics import DeepZoneGateReport, GateConfig, run_deep_zone_gate
from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import DeltaV, PossessionValue, PressureLevel, State, zone_of
from silly_kicks.xtgk._pressure_levels import PressureLevels
from silly_kicks.xtgk._validate import PossessionValueInputDiagnosis, validate_possession_value_input

__all__ = [
    "DeepZoneGateReport", "DeltaV", "EmpiricalPossessionValue", "GateConfig",
    "MarkovPossessionValue", "PossessionValue", "PossessionValueInputDiagnosis",
    "PressureLevel", "PressureLevels", "State", "run_deep_zone_gate",
    "validate_possession_value_input", "zone_of",
]
```

- [ ] **Step 2:** Create `tests/xtgk/test_regression_boundary.py`:

```python
"""xtgk touches NO xthreat source; importing it must not change any classic xthreat output."""
import numpy as np

import silly_kicks.xtgk  # noqa: F401
from silly_kicks.xthreat import GridSpec, singh_transition_matrix
from silly_kicks.xthreat._grid import _action_prob
from tests.xtgk.conftest import make_cohort, _row, PASS, SHOT, SUCCESS

GRID = GridSpec(n_zones_x=16, n_zones_y=12)


def test_classic_xt_unaffected_by_xtgk_import():
    rows = [_row(i, PASS, SUCCESS, 10 + i, 34, 50 + i, 40) for i in range(12)]
    rows += [_row(12, SHOT, SUCCESS, 100, 34, 105, 34)]
    a = make_cohort(rows)
    T = singh_transition_matrix(a, GRID)
    assert T.shape == (192, 192) and np.isfinite(T).all()
    s, m = _action_prob(a, 16, 12)
    assert np.isfinite(s).all() and np.isfinite(m).all()
```

- [ ] **Step 3: Full xtgk + xthreat suites — all PASS** (all paths verified to exist):

Run: `python -m pytest tests/xtgk/ tests/test_xtgk_builder_parity.py tests/xthreat_legacy_reference.py tests/test_xthreat.py tests/test_xthreat_kde.py tests/test_xthreat_kde_vectorized.py -v`

- [ ] **Step 4: Repo-wide fast suite.**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q`
Expected: PASS (pre-existing xfails only).

### Task 14: NOTICE + ADR-036 + CLAUDE.md

**Files:** Modify `NOTICE`; create `docs/superpowers/adrs/ADR-036-xtgk-possession-value-surface.md`; modify `CLAUDE.md`

- [ ] **Step 1:** Write `ADR-036` from `docs/superpowers/adrs/ADR-TEMPLATE.md`. Record: pressure-stratified Markov value iteration + xG-calibrated first-shot reward; the extended goal-kick-inclusive move-set (§G2) via **xtgk-local builders reusing xthreat low-level seams (no public-API edits; parity-gated)**; per-cohort surfaces + pooling comparability gate (G3); injected-`xg_column` boundary (ships no xG model); the pre-registered occupied-cell gate; pickle-free artifact convention (G4); v1/v2 coexistence + freeze/deprecation end-state (M5). Cross-link the spec.

- [ ] **Step 2:** Add a `xtgk/` bullet to `CLAUDE.md`'s Architecture section (house style), noting v1 (`tracking/_xt_gk.py`) coexists and the freeze end-state.

- [ ] **Step 3:** NOTICE — add the methodological entry (Singh 2018 xT lineage; Eyestone xT-GK v2 possession-value formulation); cross-link from `_markov.py` docstring.

### Task 15: Version bump (hard gate) — 4.40.0 / PR-S107

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1:** Confirm current version; bump all four in lockstep to `4.40.0`.

- [ ] **Step 2:** `CHANGELOG.md` `4.40.0`: "xT-GK v2 sub-project 1: honest possession-value surface `V(z,p)` (`silly_kicks/xtgk/`) — pressure-stratified Markov value iteration with an xG-calibrated first-shot reward + goal-kick-inclusive move-set (xtgk-local builders reusing xthreat low-level seams; classic xT byte-unchanged); model-free empirical cross-check; pre-registered occupied-cell deep-zone gate; injected `xg_column` (ships no xG model); pickle-free surface artifact. ADR-036, PR-S107. No production/lakehouse wiring yet (gated on the real-data gate)."

- [ ] **Step 3:** `TODO.md` — On Deck: the follow-on sub-projects (V_opp, ρ/xR-GK, metric+lakehouse migration, validation) + the blocked real-data gate (Phase 11). Groom per memory rules.

### Task 16: `/final-review` + SINGLE commit

- [ ] **Step 1:** `ruff format --check . && ruff check . && pyright silly_kicks/xtgk && python -m pytest tests/ -m "not e2e and not slow" -q` — fix all findings.

- [ ] **Step 2:** Invoke the mandatory `/final-review` gate.

- [ ] **Step 3:** After owner approval, ONE commit:

```bash
git add -A
git commit  # feat(xtgk): honest possession-value V(z,p) ... 4.40.0 (ADR-036, PR-S107)
```

Do **not** tag or push until the owner confirms CI is green (memory: never tag before CI green; don't poll CI).

---

## Phase 11 — Owner-run real-data gate (BLOCKED on Q3 + Q4 — do NOT run in this PR)

### Task 17: wire `scripts/validate_xtgk_possession_value.py` (build, don't run)

**Files:** Create `scripts/validate_xtgk_possession_value.py`

- [ ] **Step 1:** Owner-run script (mirrors existing `scripts/validate_*`): load WC2022 (gradientsports) + RM (skillcorner) via `scripts/_loader_pining.py`; require the owner-confirmed `xg_column` (Q3) + pinned pressure measure; **fit `PressureLevels` ONCE per cohort and inject into both `MarkovPossessionValue` and `EmpiricalPossessionValue`** (m2 — identical strata); fit **per cohort** (G3); before the gate emit the **deep-zone pressure-coverage / drop-rate** report (G8) and **tercile-occupancy** report (M3; switch `PressureLevels(mode="zone_conditional")` if the deep stratum is degenerate); run `run_deep_zone_gate` with the **locked `GateConfig` numbers (Q4)** in **both orientations** (fit on the cohort, and on its `mirror_y`, asserting equivariance; the `mirror_x` reversal is the rejection check, not a gate run); **persist the fitted surfaces** via `MarkovPossessionValue.save` plus a JSON verdict under `docs/research/xtgk_possession_value/`. Add an `@pytest.mark.e2e` smoke wrapper that self-skips without pining data.

- [ ] **Step 2:** Header: **DO NOT RUN until Q3 (xg_column source) and Q4 (locked gate numbers) are answered; results are the go/no-go for v2 sub-projects 2–5.**

- [ ] **Step 3:** `TODO.md` On-Deck row: "xT-GK v2 SP1 real-data gate — run `scripts/validate_xtgk_possession_value.py` once Q3/Q4 land; PASS authorises SP2–5, FAIL ⇒ STOP+escalate."

---

## Self-review (rev 2, against spec rev 3 + plan-review)

- **B1 (3-band fixtures)** → conftest `three_band_cohort` + `test_all_three_levels_populated` (Task 7) + `test_three_band_input_populates_all_levels` (Task 4). ✓
- **B2 (occupied-cell gate)** → `_occupied` + `min_occupied_cells`; fixtures spread deep goal-kicks across `DEEP_YS`; STOP-on-too-few test. ✓
- **B3/M1/M2 (seam + KDE)** → xtgk-local builders reusing `_kde_transition_from_grouped`/`_zone_centres`; success-filtered KDE population; **no xthreat edits**; parity property tests (Task 2). ✓
- **M3 (orientation)** → split: `test_surface_is_y_reflection_equivariant` (equivariance) + `test_attack_reversed_input_is_rejected` (rejection). ✓
- **M4 (direction)** → `GateConfig.expected_direction` + `observed_direction` reported; default `either`. ✓
- **M5 (serialization)** → `_serialize.py` + `save`/`load` + tamper test (Task 12). ✓
- **m1** coincidence band documented in `_empirical` docstring. **m2** shared `PressureLevels` in gate tests + owner script. **m3** Task 8 relabelled characterization. **m4** O(n) reverse scan + `test_reverse_scan_matches_naive_first_shot`. **m5** tie note in `_pressure_levels`. ✓
- **Spec §2–§13 coverage** unchanged from rev 1 and still mapped (reward Task 3; move-set Task 1; pressure Task 4; validator Task 5; port Task 6; markov Task 7/8; empirical Task 9; gate Task 10; honesty+neg-control Task 11; serialize Task 12; regression boundary Task 13; artifact/home ADR Task 14; version Task 15; owner gate Task 17). ✓
- **Type consistency:** `xtgk_transition_matrix(method=)`, `xtgk_action_prob`, `GateConfig(effect_floor,n_min,min_occupied_cells,crosscheck_rel_tol,expected_direction)`, `DeepZoneGateReport` fields, `MarkovPossessionValue.save/load` — consistent across tasks.

**No open deviations.** D2 (per-action empirical conditioning) was reconciled into spec §M1 on 2026-07-05 — plan and spec now agree.

### Plan-review rev 3 (2026-07-05)
- **R1 (verified FALSE)** — `tests/test_xthreat.py` and `tests/xthreat_legacy_reference.py` both exist and resolve (confirmed via `ls`); the review's glob missed them. Paths kept and **broadened** to the KDE seam files (`test_xthreat_kde.py`, `test_xthreat_kde_vectorized.py`) our builder reuses (Task 2 Step 3, Task 13 Step 3).
- **R2 (accepted, real gap)** — the fixture possession now routes through the build-up band (goal-kick → `buildup_x=30`, xi=4; next pass starts there), so `mk_grad`/`emp_grad` are non-zero and the gate's cross-check is genuinely exercised. Added `test_gate_fails_when_crosscheck_disagrees_on_buildup` (stub empirical with an opposite build-up gradient ⇒ `crosscheck_agrees=False` ⇒ gate fails), pinning G1 in the failing direction.
- **m6** — `test_{singh,kde,action_prob}_parity_normalization_edge` (many mixed-success passes in one source zone) exercise the denominator=all/numerator=success normalization edge.
- **m7** — `MarkovPossessionValue.fit` warns on a zero-action tercile at fit (fail-loud before the gate).
- **m8** — `_occupied` now requires support in all three terciles (level 2 is read by the effect check).
- **m9** (minor, noted) — folded to execution: the fixture's deep goal-kicks all start at `xi=0`; a couple at `x≈9` (`xi=1`) would exercise the full deep column but are not required for correctness.
