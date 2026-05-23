# TF-36 + TF-33: Per-Player Influence + Off-Ball xT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `compute_player_influence` (per-frame primitive) + `add_player_influence` (action-coupled aggregator) + `player_influence_xfns` (VAEP factory) emitting 7 columns per action and 21 VAEP columns.

**Architecture:** New `_player_influence.py` module with a single per-frame primitive that computes both off-ball xT and uniquely reachable area for all outfield players in one pass. One `compute_pitch_control(decompose=True)` call per frame serves both metrics. Action-coupled layer aggregates per-team following the DAS `_team`/`_opponent`/`_diff` convention.

**Tech Stack:** pandas, numpy, scipy (via existing `compute_tti`/`compute_pitch_control`), pytest, pytest-benchmark

**Spec:** `docs/superpowers/specs/2026-05-23-tf36-tf33-player-influence-off-ball-xt-design.md`

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `silly_kicks/tracking/_player_influence.py` | Per-frame primitive: `PlayerInfluence` dataclass + `compute_player_influence` |
| `tests/tracking/test_player_influence.py` | Unit tests: primitive correctness, invariants, edge cases, TTI optimization parity |
| `tests/tracking/test_player_influence_aggregator.py` | Action-coupled aggregator + per-Series helpers + VAEP xfns smoke |
| `tests/tracking/test_player_influence_snapshot.py` | Snapshot hash test with multi-hash set |
| `tests/tracking/test_player_influence_perf_budget.py` | pytest-benchmark per-frame budget |
| `tests/tracking/test_player_influence_e2e.py` | Per-provider committed-fixture E2E tests |

### Modified files
| File | Changes |
|------|---------|
| `silly_kicks/tracking/features.py` | Batch kernel `_player_influence_at_actions`, aggregator `add_player_influence`, 5 per-Series helpers, `player_influence_xfns` factory |
| `silly_kicks/tracking/__init__.py` | Re-export `compute_player_influence`, `PlayerInfluence`, `add_player_influence`, per-Series helpers, `player_influence_xfns` |
| `silly_kicks/atomic/tracking/features.py` | Atomic mirror: `add_player_influence`, `player_influence_xfns` |
| `tests/tracking/test_provenance_skip_guard.py` | Add `add_player_influence` to chain |
| `tests/test_enrichment_nan_safety.py` | Bump tracking floor from ≥9 to ≥10 |
| `NOTICE` | Academic attribution entry |
| `CHANGELOG.md` | New feature entry |
| `TODO.md` | Delete TF-33 and TF-36 rows |

---

## Task 1: Per-frame primitive — `_player_influence.py` (TDD)

**Files:**
- Create: `silly_kicks/tracking/_player_influence.py`
- Create: `tests/tracking/test_player_influence.py`

### Step 1: Write the test fixture helper

- [ ] **Step 1a: Create test file with fixture**

```python
# tests/tracking/test_player_influence.py
"""Unit tests for compute_player_influence (TF-36 + TF-33)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_frame(
    *,
    n_home_outfield: int = 10,
    n_away_outfield: int = 10,
    home_team_id: int = 1,
    away_team_id: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic 22-player frame for testing."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    # Ball at center
    rows.append(
        dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=0, team_id=np.nan,
            is_ball=True, is_goalkeeper=False,
            x=50.0, y=34.0, vx=5.0, vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Home GK at x=3
    rows.append(
        dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=1, team_id=home_team_id,
            is_ball=False, is_goalkeeper=True,
            x=3.0, y=34.0, vx=0.0, vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Home outfield
    for i in range(n_home_outfield):
        rows.append(
            dict(
                game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
                frame_rate=25.0, player_id=10 + i, team_id=home_team_id,
                is_ball=False, is_goalkeeper=False,
                x=float(rng.uniform(10, 60)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    # Away GK at x=102
    rows.append(
        dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=50, team_id=away_team_id,
            is_ball=False, is_goalkeeper=True,
            x=102.0, y=34.0, vx=0.0, vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Away outfield
    for i in range(n_away_outfield):
        rows.append(
            dict(
                game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
                frame_rate=25.0, player_id=60 + i, team_id=away_team_id,
                is_ball=False, is_goalkeeper=False,
                x=float(rng.uniform(45, 95)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def frame_22():
    return _make_frame()


@pytest.fixture
def xt_grid():
    """Pre-fit xT with linear gradient (high xT near x=105)."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt
```

### Step 2: Write failing tests for the primitive

- [ ] **Step 2a: Basic correctness test**

Add to `tests/tracking/test_player_influence.py`:

```python
def test_compute_player_influence_returns_outfield_only(frame_22, xt_grid):
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    # GKs (player_id 1 and 50) excluded; ball excluded
    assert 1 not in result
    assert 50 not in result
    assert 0 not in result
    # 20 outfield players present
    assert len(result) == 20


def test_off_ball_xt_positive_for_outfield(frame_22, xt_grid):
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    for pid, pi in result.items():
        assert pi.off_ball_xt >= 0.0, f"Player {pid} has negative off_ball_xt"


def test_reachable_area_positive_for_outfield(frame_22, xt_grid):
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    for pid, pi in result.items():
        assert pi.reachable_area_m2 >= 0.0, f"Player {pid} has negative area"
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_player_influence.py -v --tb=short`
Expected: FAIL with `ModuleNotFoundError: No module named 'silly_kicks.tracking._player_influence'`

### Step 3: Implement the primitive

- [ ] **Step 3a: Create `_player_influence.py`**

```python
# silly_kicks/tracking/_player_influence.py
"""Per-player influence primitives (TF-36 + TF-33).

Per-frame computation of off-ball xT (threat-weighted pitch control share)
and uniquely reachable area for all outfield players. Both metrics share a
single compute_pitch_control(decompose=True) call per frame.

See docs/superpowers/specs/2026-05-23-tf36-tf33-player-influence-off-ball-xt-design.md.
See NOTICE for full bibliographic citations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .pitch_control import PitchControlParams, PitchControlSurface, SpearmanParams, compute_pitch_control
from .pitch_control._spearman import compute_tti

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat


@dataclass(frozen=True)
class PlayerInfluence:
    """Per-player per-frame influence measurement.

    Examples
    --------
    >>> pi = PlayerInfluence(off_ball_xt=0.35, reachable_area_m2=120.0)
    """

    off_ball_xt: float
    reachable_area_m2: float


def compute_player_influence(
    frame: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
    surface: PitchControlSurface | None = None,
    tau_seconds: float = 1.0,
    reaction_time: float | None = None,
    max_acceleration: float | None = None,
) -> dict[int | str, PlayerInfluence]:
    """Per-frame influence for all outfield players.

    Computes off-ball xT (threat-weighted pitch control share) and uniquely
    reachable area for every non-GK, non-ball player in the frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data (TRACKING_FRAMES_COLUMNS schema).
    xt : ExpectedThreat
        Pre-fit xT model for threat weighting.
    attacking_team_id : int | str
        Team currently in possession.
    home_team_id : int | str
        Home team identifier for goal-end / xT orientation.
    method : str, default "spearman"
        Pitch control model (ignored when ``surface`` is provided).
    params : PitchControlParams | None
        Pitch control params override (ignored when ``surface`` is provided).
    surface : PitchControlSurface | None
        Pre-computed decomposed pitch control surface. When provided,
        ``method`` and ``params`` are ignored. Must have been computed with
        ``decompose=True``.
    tau_seconds : float, default 1.0
        TTI threshold for uniquely reachable area.
    reaction_time : float | None
        Outfield reaction time. Defaults to ``SpearmanParams().reaction_time``.
    max_acceleration : float | None
        Outfield max acceleration. Defaults to ``SpearmanParams().max_acceleration``.

    Returns
    -------
    dict[int | str, PlayerInfluence]
        Mapping from player_id to influence metrics. GKs and ball excluded.

    Examples
    --------
    >>> from silly_kicks.tracking._player_influence import compute_player_influence
    >>> result = compute_player_influence(
    ...     frame, xt, attacking_team_id=1, home_team_id=1,
    ... )

    See NOTICE for full bibliographic citations.
    """
    sp_defaults = SpearmanParams()
    rt = reaction_time if reaction_time is not None else sp_defaults.reaction_time
    ma = max_acceleration if max_acceleration is not None else sp_defaults.max_acceleration

    # --- Pitch control surface ---
    if surface is not None:
        pc = surface
    else:
        pc = compute_pitch_control(
            frame,
            attacking_team_id=attacking_team_id,
            method=method,
            params=params,
            decompose=True,
        )

    # --- xT interpolation ---
    interp = xt.interpolator(kind="linear")
    threat_grid = interp(pc.grid_x, pc.grid_y)  # (ny, nx)

    # xT flip for away-team attack
    if attacking_team_id != home_team_id:
        threat_grid = threat_grid[:, ::-1]

    cell_area = pc.cell_area

    # --- Identify outfield players ---
    players = frame[~frame["is_ball"].astype(bool)].copy()
    players = players[~players["is_goalkeeper"].astype(bool)]
    players = players.dropna(subset=["x", "y", "team_id"])

    if len(players) == 0:
        return {}

    # --- Off-ball xT (from decomposed PC surface) ---
    off_ball_xt_map: dict[int | str, float] = {}
    for _, row in players.iterrows():
        pid = row["player_id"]
        try:
            ps = pc.player_surface(pid)  # (ny, nx)
        except ValueError:
            off_ball_xt_map[pid] = 0.0
            continue
        off_ball_xt_map[pid] = float((ps * threat_grid * cell_area).sum())

    # --- Uniquely reachable area (team-TTI-matrix optimization) ---
    reachable_map: dict[int | str, float] = {}

    # Build grid targets
    gx, gy = np.meshgrid(pc.grid_x, pc.grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])  # (n_cells, 2)

    # Process per-team
    for team_id in players["team_id"].unique():
        team_players = players[players["team_id"] == team_id]
        n_team = len(team_players)
        pids = team_players["player_id"].values

        # Build position + velocity arrays, NaN vx/vy -> 0.0
        pos = team_players[["x", "y"]].to_numpy(dtype="float64")
        vx_arr = team_players["vx"].to_numpy(dtype="float64") if "vx" in team_players.columns else np.zeros(n_team)
        vy_arr = team_players["vy"].to_numpy(dtype="float64") if "vy" in team_players.columns else np.zeros(n_team)
        vel = np.column_stack([np.nan_to_num(vx_arr), np.nan_to_num(vy_arr)])

        # Compute full-team TTI matrix: (n_team, n_cells)
        tti_matrix = compute_tti(pos, vel, targets, rt, ma)

        if n_team == 1:
            # Single player: every cell within tau is uniquely reachable
            unique_cells = tti_matrix[0] <= tau_seconds
            reachable_map[pids[0]] = float(unique_cells.sum() * cell_area)
        else:
            # argmin/second-min optimization
            global_argmin = np.argmin(tti_matrix, axis=0)  # (n_cells,)
            global_min = tti_matrix.min(axis=0)            # (n_cells,) — explicit, no partition ambiguity
            # np.partition: kth=1 gives second-smallest in position [1]
            partitioned = np.partition(tti_matrix, kth=1, axis=0)
            second_min = partitioned[1, :]   # (n_cells,)

            for idx in range(n_team):
                pid = pids[idx]
                player_tti = tti_matrix[idx]  # (n_cells,)
                # min TTI of teammates excluding this player
                min_excluding = np.where(
                    global_argmin == idx,
                    second_min,
                    global_min,
                )
                unique_cells = (player_tti <= tau_seconds) & (min_excluding > tau_seconds)
                reachable_map[pid] = float(unique_cells.sum() * cell_area)

    # --- Assemble result ---
    result: dict[int | str, PlayerInfluence] = {}
    for _, row in players.iterrows():
        pid = row["player_id"]
        result[pid] = PlayerInfluence(
            off_ball_xt=off_ball_xt_map.get(pid, 0.0),
            reachable_area_m2=reachable_map.get(pid, 0.0),
        )
    return result
```

- [ ] **Step 3b: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_player_influence.py -v --tb=short`
Expected: 3 PASSED

### Step 4: Add invariant and edge-case tests

- [ ] **Step 4a: Add invariant tests**

Append to `tests/tracking/test_player_influence.py`:

```python
def test_reachable_area_sum_lte_pitch_area(frame_22, xt_grid):
    """Sum of same-team uniquely reachable areas <= total pitch area."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    # Get team membership from frame
    players = frame_22[
        ~frame_22["is_ball"].astype(bool)
        & ~frame_22["is_goalkeeper"].astype(bool)
    ]
    for tid in players["team_id"].unique():
        team_pids = players[players["team_id"] == tid]["player_id"].values
        team_area = sum(result[pid].reachable_area_m2 for pid in team_pids if pid in result)
        assert team_area <= 105.0 * 68.0, (
            f"Team {tid} total area {team_area:.1f} > pitch area {105*68}"
        )


def test_tau_zero_all_areas_zero(frame_22, xt_grid):
    """tau=0 -> nobody reaches anywhere -> all areas 0."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
        tau_seconds=0.0,
    )
    for pid, pi in result.items():
        assert pi.reachable_area_m2 == 0.0, f"Player {pid} area={pi.reachable_area_m2} with tau=0"


@pytest.mark.parametrize("method", ["spearman", "voronoi", "fernandez_bornn"])
def test_off_ball_xt_conservation(frame_22, xt_grid, method):
    """Sum of all players' off_ball_xt <= total xT-weighted pitch area."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
        method=method,
    )
    total_player_xt = sum(pi.off_ball_xt for pi in result.values())
    # Upper bound: sum of xT * cell_area over entire grid
    interp = xt_grid.interpolator("linear")
    from silly_kicks.tracking.pitch_control import compute_pitch_control

    pc = compute_pitch_control(frame_22, attacking_team_id=1, method=method)
    threat_grid = interp(pc.grid_x, pc.grid_y)
    total_xt_area = float((threat_grid * pc.cell_area).sum())
    assert total_player_xt <= total_xt_area * 1.01, (
        f"Total player xT {total_player_xt:.2f} > total xT area {total_xt_area:.2f}"
    )
```

- [ ] **Step 4b: Add edge-case tests**

Append to `tests/tracking/test_player_influence.py`:

```python
def test_single_outfield_player_per_team(xt_grid):
    """1 outfield player -> all cells within tau are uniquely reachable."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    frame = _make_frame(n_home_outfield=1, n_away_outfield=1)
    result = compute_player_influence(
        frame, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    assert len(result) == 2
    for pid, pi in result.items():
        # Isolated player should have substantial unique area
        assert pi.reachable_area_m2 > 0.0


def test_all_players_same_position(xt_grid):
    """All outfield at same position -> all uniquely reachable = 0."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    rows: list[dict] = []
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
        frame_rate=25.0, player_id=0, team_id=np.nan,
        is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # 2 GKs
    for gk_pid, gk_tid, gk_x in [(1, 1, 3.0), (50, 2, 102.0)]:
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=gk_pid, team_id=gk_tid,
            is_ball=False, is_goalkeeper=True,
            x=gk_x, y=34.0, vx=0.0, vy=0.0,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    # 4 outfield all at (50, 34) with zero velocity
    for i in range(4):
        tid = 1 if i < 2 else 2
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=10 + i, team_id=tid,
            is_ball=False, is_goalkeeper=False,
            x=50.0, y=34.0, vx=0.0, vy=0.0,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    frame = pd.DataFrame(rows)
    result = compute_player_influence(
        frame, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    for pid, pi in result.items():
        assert pi.reachable_area_m2 == 0.0, f"Player {pid} area={pi.reachable_area_m2} (expected 0)"


def test_nan_velocity_defaults_to_zero(xt_grid):
    """NaN vx/vy should not produce NaN results."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    frame = _make_frame(n_home_outfield=3, n_away_outfield=3)
    # Set some velocities to NaN
    frame.loc[frame["player_id"] == 10, "vx"] = np.nan
    frame.loc[frame["player_id"] == 10, "vy"] = np.nan
    result = compute_player_influence(
        frame, xt_grid,
        attacking_team_id=1, home_team_id=1,
    )
    pi = result[10]
    assert not np.isnan(pi.reachable_area_m2), "NaN velocity should not produce NaN area"
    assert not np.isnan(pi.off_ball_xt), "NaN velocity should not produce NaN off_ball_xt"


@pytest.mark.parametrize("method", ["voronoi", "fernandez_bornn"])
def test_non_spearman_method_non_degenerate(frame_22, xt_grid, method):
    """Non-spearman methods should produce non-zero results."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
        method=method,
    )
    total_xt = sum(pi.off_ball_xt for pi in result.values())
    assert total_xt > 0.0, f"Method {method} produced zero total off_ball_xt"
```

- [ ] **Step 4c: Add TTI optimization parity test**

Append to `tests/tracking/test_player_influence.py`:

```python
def test_tti_optimization_matches_naive(xt_grid):
    """The argmin/second-min trick must be numerically equivalent to naive loop."""
    from silly_kicks.tracking._player_influence import compute_player_influence
    from silly_kicks.tracking.pitch_control import SpearmanParams, compute_pitch_control
    from silly_kicks.tracking.pitch_control._spearman import compute_tti

    frame = _make_frame(n_home_outfield=5, n_away_outfield=5)
    sp = SpearmanParams()
    tau = 1.0

    # Get optimized result
    optimized = compute_player_influence(
        frame, xt_grid,
        attacking_team_id=1, home_team_id=1,
        tau_seconds=tau,
    )

    # Compute naive result for comparison
    pc = compute_pitch_control(frame, attacking_team_id=1, decompose=True)
    gx, gy = np.meshgrid(pc.grid_x, pc.grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])

    players = frame[
        ~frame["is_ball"].astype(bool)
        & ~frame["is_goalkeeper"].astype(bool)
    ].dropna(subset=["x", "y"])

    for tid in players["team_id"].unique():
        team = players[players["team_id"] == tid]
        n = len(team)
        pos = team[["x", "y"]].to_numpy(dtype="float64")
        vx = team["vx"].to_numpy(dtype="float64")
        vy = team["vy"].to_numpy(dtype="float64")
        vel = np.column_stack([np.nan_to_num(vx), np.nan_to_num(vy)])

        tti_all = compute_tti(pos, vel, targets, sp.reaction_time, sp.max_acceleration)

        for i in range(n):
            pid = team.iloc[i]["player_id"]
            player_tti = tti_all[i]
            # Naive: min of ALL OTHER teammates
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if mask.any():
                min_others = tti_all[mask].min(axis=0)
            else:
                min_others = np.full(len(targets), np.inf)
            naive_unique = (player_tti <= tau) & (min_others > tau)
            naive_area = float(naive_unique.sum() * pc.cell_area)

            assert optimized[pid].reachable_area_m2 == pytest.approx(naive_area, abs=1e-10), (
                f"Player {pid}: optimized={optimized[pid].reachable_area_m2}, naive={naive_area}"
            )
```

- [ ] **Step 4d: Run all tests**

Run: `python -m pytest tests/tracking/test_player_influence.py -v --tb=short`
Expected: All PASSED

---

## Task 2: Pre-computed surface parameter test

**Files:**
- Modify: `tests/tracking/test_player_influence.py`

- [ ] **Step 1: Add test for surface parameter**

Append to `tests/tracking/test_player_influence.py`:

```python
def test_surface_parameter_skips_pc_call(frame_22, xt_grid):
    """When surface is provided, method/params are ignored."""
    from silly_kicks.tracking._player_influence import compute_player_influence
    from silly_kicks.tracking.pitch_control import compute_pitch_control

    # Pre-compute surface with spearman
    surface = compute_pitch_control(
        frame_22, attacking_team_id=1, method="spearman", decompose=True,
    )

    # Pass surface + method="voronoi" — voronoi should be ignored
    result_with_surface = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
        surface=surface,
        method="voronoi",  # should be ignored
    )

    # Compare with direct spearman call (no surface param)
    result_direct = compute_player_influence(
        frame_22, xt_grid,
        attacking_team_id=1, home_team_id=1,
        method="spearman",
    )

    for pid in result_with_surface:
        assert result_with_surface[pid].off_ball_xt == pytest.approx(
            result_direct[pid].off_ball_xt, abs=1e-10
        )
        assert result_with_surface[pid].reachable_area_m2 == pytest.approx(
            result_direct[pid].reachable_area_m2, abs=1e-10
        )
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/tracking/test_player_influence.py::test_surface_parameter_skips_pc_call -v --tb=short`
Expected: PASSED

---

## Task 3: Action-coupled layer — batch kernel + aggregator + per-Series helpers

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Create: `tests/tracking/test_player_influence_aggregator.py`

### Step 1: Write aggregator tests

- [ ] **Step 1a: Create aggregator test file**

```python
# tests/tracking/test_player_influence_aggregator.py
"""Tests for add_player_influence aggregator + per-Series helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions


@pytest.fixture
def xt_grid():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@pytest.fixture
def sportec_data(xt_grid):
    frames = load_provider_frames("sportec")
    actions = synthesize_actions(frames, n_actions=5)
    return actions, frames, xt_grid


_OUTPUT_COLS = [
    "actor_reachable_area_m2",
    "off_ball_xt_team",
    "off_ball_xt_opponent",
    "off_ball_xt_diff",
    "reachable_area_team",
    "reachable_area_opponent",
    "reachable_area_diff",
]


def test_add_player_influence_output_columns(sportec_data):
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    result = add_player_influence(
        actions, frames, xt, home_team_id=actions["team_id"].dropna().iloc[0],
    )
    for col in _OUTPUT_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_diff_identity(sportec_data):
    """_diff = _team - _opponent (exact equality)."""
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    home = actions["team_id"].dropna().iloc[0]
    result = add_player_influence(actions, frames, xt, home_team_id=home)

    valid = result["off_ball_xt_team"].notna()
    pd.testing.assert_series_equal(
        result.loc[valid, "off_ball_xt_diff"],
        (result.loc[valid, "off_ball_xt_team"] - result.loc[valid, "off_ball_xt_opponent"]).rename("off_ball_xt_diff"),
    )
    pd.testing.assert_series_equal(
        result.loc[valid, "reachable_area_diff"],
        (result.loc[valid, "reachable_area_team"] - result.loc[valid, "reachable_area_opponent"]).rename("reachable_area_diff"),
    )


def test_provenance_columns_added(sportec_data):
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    home = actions["team_id"].dropna().iloc[0]
    result = add_player_influence(actions, frames, xt, home_team_id=home)

    provenance = {"frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"}
    assert provenance.issubset(result.columns)


def test_provenance_skip_guard(sportec_data):
    """Calling add_player_influence twice doesn't create _x/_y suffixed columns."""
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    home = actions["team_id"].dropna().iloc[0]
    result = add_player_influence(actions, frames, xt, home_team_id=home)
    result2 = add_player_influence(result, frames, xt, home_team_id=home)

    for col in ["frame_id", "time_offset_seconds"]:
        bad_x = f"{col}_x"
        bad_y = f"{col}_y"
        assert bad_x not in result2.columns, f"Found {bad_x} — skip guard failed"
        assert bad_y not in result2.columns, f"Found {bad_y} — skip guard failed"
```

- [ ] **Step 1b: Add per-Series helper tests**

Append to `tests/tracking/test_player_influence_aggregator.py`:

```python
@pytest.mark.parametrize(
    "helper_name",
    [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "reachable_area_team",
        "reachable_area_opponent",
    ],
)
def test_per_series_helper_returns_series(sportec_data, helper_name):
    from silly_kicks.tracking import features

    actions, frames, xt = sportec_data
    home = actions["team_id"].dropna().iloc[0]
    fn = getattr(features, helper_name)
    result = fn(actions, frames, xt, home_team_id=home)
    assert isinstance(result, pd.Series)
    assert len(result) == len(actions)


@pytest.mark.parametrize(
    "helper_name",
    [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "reachable_area_team",
        "reachable_area_opponent",
    ],
)
def test_per_series_helper_none_frames(helper_name, xt_grid):
    """frames=None -> all NaN (column-name probing tolerance)."""
    from silly_kicks.tracking import features

    actions = pd.DataFrame({
        "action_id": [1, 2],
        "game_id": [1, 1],
        "period_id": [1, 1],
        "time_seconds": [1.0, 2.0],
        "team_id": [1, 2],
        "player_id": [10, 60],
    })
    fn = getattr(features, helper_name)
    result = fn(actions, None, xt_grid, home_team_id=1)
    assert result.isna().all()
```

- [ ] **Step 1c: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_player_influence_aggregator.py -v --tb=short`
Expected: FAIL (functions don't exist yet)

### Step 2: Implement batch kernel + aggregator + per-Series helpers

- [ ] **Step 2a: Add to `features.py`**

Add the following at the end of `silly_kicks/tracking/features.py` (before the final atomic-mirror imports if any, or at the bottom):

```python
# ---------------------------------------------------------------------------
# TF-36 + TF-33: Per-player influence + Off-ball xT
# ---------------------------------------------------------------------------


def _player_influence_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
    links: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch kernel: compute player influence for all actions.

    Cache key: (period_id, frame_id, attacking_team_id). Returns
    (result_df, pointers).
    """
    from ._player_influence import PlayerInfluence, compute_player_influence

    col_names = [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "off_ball_xt_diff",
        "reachable_area_team",
        "reachable_area_opponent",
        "reachable_area_diff",
    ]

    result = pd.DataFrame(
        {col: np.full(len(actions), np.nan) for col in col_names},
        index=actions.index,
    )

    if len(frames) == 0:
        return result, pd.DataFrame()

    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    # Cache: (period_id, frame_id, attacking_team_id) -> dict | None
    cache: dict[tuple, dict[int | str, PlayerInfluence] | None] = {}

    # Build player -> team_id lookup from PC surface (populated on first call)
    player_team_lookup: dict[int | str, int | str] = {}

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        aid = action_row["action_id"]
        tid = action_row["team_id"]
        actor_pid = action_row["player_id"]
        if pd.isna(tid):
            continue
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue

        pid = action_row["period_id"]
        fid = int(float(fid_raw))
        cache_key = (pid, fid, tid)

        if cache_key not in cache:
            try:
                frame_data = frame_groups.get_group((pid, fid))
            except KeyError:
                cache[cache_key] = None
                continue

            try:
                pi_dict = compute_player_influence(
                    frame_data,
                    xt,
                    attacking_team_id=tid,
                    home_team_id=home_team_id,
                    method=method,
                    tau_seconds=tau_seconds,
                )
            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"compute_player_influence failed for frame {fid}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                pi_dict = None

            cache[cache_key] = pi_dict

            # Populate player->team lookup from frame data
            if pi_dict is not None:
                outfield = frame_data[
                    ~frame_data["is_ball"].astype(bool)
                    & ~frame_data["is_goalkeeper"].astype(bool)
                ]
                for _, prow in outfield.iterrows():
                    p_id = prow["player_id"]
                    if p_id not in player_team_lookup:
                        player_team_lookup[p_id] = prow["team_id"]

        pi_dict = cache[cache_key]
        if pi_dict is None:
            continue

        # Aggregate per-team
        actor_team = tid
        team_xt = 0.0
        opponent_xt = 0.0
        actor_area = 0.0
        team_area = 0.0
        opponent_area = 0.0

        for p_id, pi in pi_dict.items():
            p_team = player_team_lookup.get(p_id)
            if p_team is None:
                continue
            is_same_team = (str(p_team) == str(actor_team))
            is_actor = (str(p_id) == str(actor_pid))

            if is_same_team:
                team_area += pi.reachable_area_m2
                if is_actor:
                    actor_area = pi.reachable_area_m2
                else:
                    team_xt += pi.off_ball_xt
            else:
                opponent_xt += pi.off_ball_xt
                opponent_area += pi.reachable_area_m2

        result.iloc[i, result.columns.get_loc("actor_reachable_area_m2")] = actor_area
        result.iloc[i, result.columns.get_loc("off_ball_xt_team")] = team_xt
        result.iloc[i, result.columns.get_loc("off_ball_xt_opponent")] = opponent_xt
        result.iloc[i, result.columns.get_loc("off_ball_xt_diff")] = team_xt - opponent_xt
        result.iloc[i, result.columns.get_loc("reachable_area_team")] = team_area
        result.iloc[i, result.columns.get_loc("reachable_area_opponent")] = opponent_area
        result.iloc[i, result.columns.get_loc("reachable_area_diff")] = team_area - opponent_area

    return result, pointers


def actor_reachable_area_m2(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """Actor's uniquely reachable area (m^2) at the linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import actor_reachable_area_m2
    >>> area = actor_reachable_area_m2(actions, frames, xt, home_team_id=1)
    """
    col_name = "actor_reachable_area_m2"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id, method=method, tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


def off_ball_xt_team(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
) -> pd.Series:
    """Sum of teammates' off-ball xT (excluding actor) at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import off_ball_xt_team
    >>> val = off_ball_xt_team(actions, frames, xt, home_team_id=1)
    """
    col_name = "off_ball_xt_team"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id, method=method,
    )
    return batch[col_name].rename(col_name)


def off_ball_xt_opponent(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
) -> pd.Series:
    """Sum of opponents' off-ball xT at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import off_ball_xt_opponent
    >>> val = off_ball_xt_opponent(actions, frames, xt, home_team_id=1)
    """
    col_name = "off_ball_xt_opponent"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id, method=method,
    )
    return batch[col_name].rename(col_name)


def reachable_area_team(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """Sum of acting team's uniquely reachable area (m^2) at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import reachable_area_team
    >>> val = reachable_area_team(actions, frames, xt, home_team_id=1)
    """
    col_name = "reachable_area_team"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id, method=method, tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


def reachable_area_opponent(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """Sum of opponent team's uniquely reachable area (m^2) at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import reachable_area_opponent
    >>> val = reachable_area_opponent(actions, frames, xt, home_team_id=1)
    """
    col_name = "reachable_area_opponent"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id, method=method, tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


@nan_safe_enrichment
def add_player_influence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> pd.DataFrame:
    """Enrich actions with 7 player-influence columns + 4 provenance.

    Columns: actor_reachable_area_m2, off_ball_xt_team, off_ball_xt_opponent,
    off_ball_xt_diff, reachable_area_team, reachable_area_opponent,
    reachable_area_diff.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_player_influence
    >>> enriched = add_player_influence(actions, frames, xt, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    out = actions.copy()
    batch, pointers = _player_influence_at_actions(
        actions, frames, xt,
        links=links,
        home_team_id=home_team_id,
        method=method,
        tau_seconds=tau_seconds,
    )
    for col in batch.columns:
        out[col] = batch[col].values

    # Provenance (idempotent skip-guard)
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(
            ptr_cols,
            left_on="action_id",
            right_index=True,
            how="left",
        )

    return out
```

Also add the new names to `__all__` in `features.py` (in alphabetical order among the existing entries):
- `"actor_reachable_area_m2"`
- `"add_player_influence"`
- `"off_ball_xt_opponent"`
- `"off_ball_xt_team"`
- `"player_influence_xfns"` (added in Task 4)
- `"reachable_area_opponent"`
- `"reachable_area_team"`

And add the import at the top of the `features.py` import block (with existing `_warnings` alias):

```python
import warnings as _warnings  # already exists — verify
```

- [ ] **Step 2b: Run aggregator tests**

Run: `python -m pytest tests/tracking/test_player_influence_aggregator.py -v --tb=short`
Expected: All PASSED

---

## Task 4: VAEP xfns factory

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `tests/tracking/test_player_influence_aggregator.py`

### Step 1: Write xfns test

- [ ] **Step 1a: Add xfns smoke test**

Append to `tests/tracking/test_player_influence_aggregator.py`:

```python
def test_player_influence_xfns_column_names(xt_grid):
    """feature_column_names probing (empty frames) returns 21 columns."""
    from silly_kicks.tracking.features import player_influence_xfns

    xfns = player_influence_xfns(xt_grid, home_team_id=1)
    assert len(xfns) == 1

    # Simulate VAEP probing: 10-row dummy actions, no frames
    dummy = pd.DataFrame({
        "action_id": range(10),
        "game_id": 1,
        "period_id": 1,
        "time_seconds": np.arange(10, dtype=float),
        "team_id": 1,
        "player_id": 10,
        "type_id": 0,
        "result_id": 0,
        "bodypart_id": 0,
        "start_x": 50.0,
        "start_y": 34.0,
        "end_x": 55.0,
        "end_y": 34.0,
    })
    states = [dummy, dummy.copy(), dummy.copy()]

    transformer = xfns[0]
    result = transformer(states, None)

    # 7 base columns x 3 slots = 21
    assert result.shape[1] == 21
    assert result.isna().all().all()

    # Verify column naming pattern
    for col_base in _OUTPUT_COLS:
        for slot in range(3):
            expected = f"{col_base}_a{slot}"
            assert expected in result.columns, f"Missing VAEP column: {expected}"
```

- [ ] **Step 1b: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_player_influence_aggregator.py::test_player_influence_xfns_column_names -v --tb=short`
Expected: FAIL

### Step 2: Implement xfns factory

- [ ] **Step 2a: Add factory to `features.py`**

Add after `add_player_influence`:

```python
def player_influence_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> list:
    """Factory returning a FrameAwareTransformer for player influence.

    Emits 7 columns x 3 gamestate slots = 21 VAEP columns.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, player_influence_xfns
        xfns = tracking_default_xfns + player_influence_xfns(xt, home_team_id=1)
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._player_influence import PlayerInfluence, compute_player_influence

    col_names = [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "off_ball_xt_diff",
        "reachable_area_team",
        "reachable_area_opponent",
        "reachable_area_diff",
    ]

    def _player_influence_transformer(states, frames):
        """Multi-column player influence xfn with frame precomputation cache."""
        out = pd.DataFrame(index=states[0].index)

        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Shared cache across all 3 slots
        cache: dict[tuple, dict[int | str, PlayerInfluence] | None] = {}
        frame_groups = frames.groupby(["period_id", "frame_id"])
        player_team_lookup: dict[int | str, int | str] = {}

        def _get_pi(period_id, frame_id_int, team_id):
            key = (period_id, frame_id_int, team_id)
            if key in cache:
                return cache[key]

            try:
                frame_data = frame_groups.get_group((period_id, frame_id_int))
            except KeyError:
                cache[key] = None
                return None

            try:
                pi_dict = compute_player_influence(
                    frame_data,
                    xt,
                    attacking_team_id=team_id,
                    home_team_id=home_team_id,
                    method=method,
                    tau_seconds=tau_seconds,
                )
            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"compute_player_influence failed for frame {frame_id_int}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                pi_dict = None

            cache[key] = pi_dict

            # Populate player->team lookup
            if pi_dict is not None:
                outfield = frame_data[
                    ~frame_data["is_ball"].astype(bool)
                    & ~frame_data["is_goalkeeper"].astype(bool)
                ]
                for _, prow in outfield.iterrows():
                    p_id = prow["player_id"]
                    if p_id not in player_team_lookup:
                        player_team_lookup[p_id] = prow["team_id"]

            return pi_dict

        def _aggregate(pi_dict, actor_team, actor_pid):
            """Aggregate per-player values into 7-column dict."""
            vals = {col: np.nan for col in col_names}
            if pi_dict is None:
                return vals

            team_xt = 0.0
            opp_xt = 0.0
            actor_area = 0.0
            team_area = 0.0
            opp_area = 0.0

            for p_id, pi in pi_dict.items():
                p_team = player_team_lookup.get(p_id)
                if p_team is None:
                    continue
                is_same = (str(p_team) == str(actor_team))
                is_actor = (str(p_id) == str(actor_pid))

                if is_same:
                    team_area += pi.reachable_area_m2
                    if is_actor:
                        actor_area = pi.reachable_area_m2
                    else:
                        team_xt += pi.off_ball_xt
                else:
                    opp_xt += pi.off_ball_xt
                    opp_area += pi.reachable_area_m2

            vals["actor_reachable_area_m2"] = actor_area
            vals["off_ball_xt_team"] = team_xt
            vals["off_ball_xt_opponent"] = opp_xt
            vals["off_ball_xt_diff"] = team_xt - opp_xt
            vals["reachable_area_team"] = team_area
            vals["reachable_area_opponent"] = opp_area
            vals["reachable_area_diff"] = team_area - opp_area
            return vals

        for i, slot in enumerate(states[:3]):
            slot_results = {col: np.full(len(slot), np.nan) for col in col_names}

            pointers, _ = link_actions_to_frames(slot, frames)
            pointer_lookup = pointers.set_index("action_id")

            for j, (_idx, row) in enumerate(slot.iterrows()):
                aid = row["action_id"]
                tid = row["team_id"]
                if pd.isna(tid):
                    continue
                if aid not in pointer_lookup.index:
                    continue
                fid_raw = pointer_lookup.at[aid, "frame_id"]
                if pd.isna(fid_raw):
                    continue

                period = row["period_id"]
                fid = int(float(fid_raw))
                actor_pid = row["player_id"]

                pi_dict = _get_pi(period, fid, tid)
                agg = _aggregate(pi_dict, tid, actor_pid)
                for col in col_names:
                    slot_results[col][j] = agg[col]

            for col in col_names:
                out[f"{col}_a{i}"] = slot_results[col]

        return out

    _player_influence_transformer._frame_aware = True
    return [_player_influence_transformer]
```

Add `"player_influence_xfns"` to `__all__` in `features.py`.

- [ ] **Step 2b: Run xfns test**

Run: `python -m pytest tests/tracking/test_player_influence_aggregator.py::test_player_influence_xfns_column_names -v --tb=short`
Expected: PASSED

---

## Task 5: Exports — `__init__.py` + atomic mirror

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `silly_kicks/atomic/tracking/features.py`

- [ ] **Step 1: Add exports to `tracking/__init__.py`**

Add to `__all__` (in alphabetical order):
- `"PlayerInfluence"`
- `"actor_reachable_area_m2"`
- `"add_player_influence"`
- `"compute_player_influence"`
- `"off_ball_xt_opponent"`
- `"off_ball_xt_team"`
- `"player_influence_xfns"`
- `"reachable_area_opponent"`
- `"reachable_area_team"`

Add import block:

```python
from ._player_influence import PlayerInfluence, compute_player_influence
```

Add to the `from .features import (...)` block:

```python
    add_player_influence,
    actor_reachable_area_m2,
    off_ball_xt_opponent,
    off_ball_xt_team,
    player_influence_xfns,
    reachable_area_opponent,
    reachable_area_team,
```

- [ ] **Step 2: Add atomic mirror**

Add to `silly_kicks/atomic/tracking/features.py`:

```python
# Import from standard features (player influence is schema-agnostic)
from silly_kicks.tracking.features import (
    add_player_influence,
    actor_reachable_area_m2,
    off_ball_xt_opponent,
    off_ball_xt_team,
    player_influence_xfns,
    reachable_area_opponent,
    reachable_area_team,
)
```

Add all 7 names to `__all__` in the atomic features module.

- [ ] **Step 3: Verify imports**

Run: `python -c "from silly_kicks.tracking import compute_player_influence, PlayerInfluence, add_player_influence, player_influence_xfns; print('OK')"`
Run: `python -c "from silly_kicks.atomic.tracking.features import add_player_influence, player_influence_xfns; print('OK')"`
Expected: Both print `OK`

---

## Task 6: Snapshot test

**Files:**
- Create: `tests/tracking/test_player_influence_snapshot.py`

- [ ] **Step 1: Write snapshot test**

```python
# tests/tracking/test_player_influence_snapshot.py
"""Snapshot hash test for compute_player_influence.

Multi-hash set pattern per feedback_multi_hash_snapshot_sets.md.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest


def _make_deterministic_frame():
    """Fixed-seed frame for snapshot reproducibility."""
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
        frame_rate=25.0, player_id=0, team_id=np.nan,
        is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, vx=5.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    for gk_pid, gk_tid, gk_x in [(1, 1, 3.0), (50, 2, 102.0)]:
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=gk_pid, team_id=gk_tid,
            is_ball=False, is_goalkeeper=True,
            x=gk_x, y=34.0, vx=0.0, vy=0.0,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    # Asymmetric 3v2 to exercise team-decomposition asymmetry
    for i in range(5):
        tid = 1 if i < 3 else 2
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=10 + i, team_id=tid,
            is_ball=False, is_goalkeeper=False,
            x=float(rng.uniform(10, 90)),
            y=float(rng.uniform(5, 63)),
            vx=float(rng.uniform(-2, 2)),
            vy=float(rng.uniform(-2, 2)),
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    return pd.DataFrame(rows)


def test_compute_player_influence_snapshot():
    from silly_kicks.tracking._player_influence import compute_player_influence
    from silly_kicks.xthreat import ExpectedThreat

    frame = _make_deterministic_frame()
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))

    result = compute_player_influence(
        frame, xt,
        attacking_team_id=1, home_team_id=1,
    )

    # Build a deterministic string representation
    parts = []
    for pid in sorted(result.keys(), key=str):
        pi = result[pid]
        parts.append(f"{pid}:{pi.off_ball_xt:.8f}:{pi.reachable_area_m2:.8f}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    # Multi-hash set for numpy runner drift
    # FIRST RUN: run once locally to capture the hash, then add it here.
    # Update this set if numpy/scipy micro-versions cause ULP drift.
    valid_hashes: set[str] = set()

    if not valid_hashes:
        # Bootstrap: print hash for first capture
        pytest.skip(
            f"BOOTSTRAP: captured hash = {digest!r}. "
            f"Add it to valid_hashes and remove this skip."
        )

    assert digest in valid_hashes, (
        f"Snapshot hash {digest!r} not in valid set {valid_hashes}. "
        f"If numpy/scipy changed, add the new hash."
    )
```

- [ ] **Step 2: Run to bootstrap hash**

Run: `python -m pytest tests/tracking/test_player_influence_snapshot.py -v --tb=short`
Expected: SKIP with captured hash. Copy the hash into `valid_hashes`.

- [ ] **Step 3: Update hash and re-run**

Edit the `valid_hashes` set to include the captured hash. Remove the bootstrap skip block.

Run: `python -m pytest tests/tracking/test_player_influence_snapshot.py -v --tb=short`
Expected: PASSED

---

## Task 7: Performance budget test

**Files:**
- Create: `tests/tracking/test_player_influence_perf_budget.py`

- [ ] **Step 1: Write benchmark test**

```python
# tests/tracking/test_player_influence_perf_budget.py
"""Performance budget for compute_player_influence (TF-36 + TF-33).

Uses pytest-benchmark. Budget set from first CI observation + 1.5x headroom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Flat ceiling — no platform ternary per feedback_windows_ci_perf_budget.md.
# FIRST RUN: set from worst observed CI timing with 1.5x headroom.
_BUDGET = 0.100  # 100ms — generous initial budget, tighten after first CI run


def _make_22_player_frame():
    """Standard 22-player frame for benchmarking."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
        frame_rate=25.0, player_id=0, team_id=np.nan,
        is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, vx=5.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    for gk_pid, gk_tid, gk_x in [(1, 1, 3.0), (50, 2, 102.0)]:
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=gk_pid, team_id=gk_tid,
            is_ball=False, is_goalkeeper=True,
            x=gk_x, y=34.0, vx=0.0, vy=0.0,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    for i in range(10):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=10 + i, team_id=1,
            is_ball=False, is_goalkeeper=False,
            x=float(rng.uniform(10, 60)),
            y=float(rng.uniform(5, 63)),
            vx=float(rng.uniform(-3, 3)),
            vy=float(rng.uniform(-3, 3)),
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    for i in range(10):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0,
            frame_rate=25.0, player_id=60 + i, team_id=2,
            is_ball=False, is_goalkeeper=False,
            x=float(rng.uniform(45, 95)),
            y=float(rng.uniform(5, 63)),
            vx=float(rng.uniform(-3, 3)),
            vy=float(rng.uniform(-3, 3)),
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_22():
    from silly_kicks.xthreat import ExpectedThreat

    frame = _make_22_player_frame()
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return frame, xt


def test_compute_player_influence_perf_budget(benchmark, fixture_22):
    """compute_player_influence on 22-player frame within budget."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    frame, xt = fixture_22

    result = benchmark(
        compute_player_influence,
        frame,
        xt,
        attacking_team_id=1,
        home_team_id=1,
    )
    assert result is not None
    assert len(result) == 20  # 20 outfield players
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"compute_player_influence mean "
            f"{benchmark.stats.stats.mean * 1000:.1f}ms > "
            f"budget {_BUDGET * 1000:.0f}ms"
        )
```

- [ ] **Step 2: Run benchmark**

Run: `python -m pytest tests/tracking/test_player_influence_perf_budget.py -v --tb=short --benchmark-disable`
Expected: PASSED (benchmark disabled for quick check; enable for actual timing)

---

## Task 8: Provenance skip-guard integration

**Files:**
- Modify: `tests/tracking/test_provenance_skip_guard.py`

- [ ] **Step 1: Add `add_player_influence` to the chain test**

In `tests/tracking/test_provenance_skip_guard.py`, add the import:

```python
from silly_kicks.tracking.features import (
    add_action_context,
    add_actor_pre_window,
    add_player_influence,  # NEW
    add_pre_shot_gk_position,
    add_pressure_on_actor,
)
```

Add a new step after step 4 in `test_chained_enrichments_no_duplicate_provenance`:

```python
    # Step 5: add_player_influence => should SKIP provenance
    xt = _make_xt()
    home = actions["team_id"].dropna().iloc[0]
    actions = add_player_influence(actions, frames, xt, home_team_id=home)
    _assert_no_suffix_duplicates(actions, "add_player_influence")
```

Add the `_make_xt` helper at module level:

```python
def _make_xt():
    import numpy as np
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt
```

- [ ] **Step 2: Run provenance test**

Run: `python -m pytest tests/tracking/test_provenance_skip_guard.py -v --tb=short`
Expected: All PASSED

---

## Task 9: NOTICE + CHANGELOG + TODO updates

**Files:**
- Modify: `NOTICE`
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`

- [ ] **Step 1: Add NOTICE entry**

Add after the GK influence section in `NOTICE`:

```
The per-player influence primitives (silly_kicks/tracking/_player_influence.py,
TF-36 + TF-33) compose:

- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics
  Conference.
  (pitch control decomposition, TTI kinematic model for uniquely reachable area)

- Singh, K. (2018). "Introducing Expected Threat (xT)." karun.in/blog/
  expected-threat
  (threat surface for per-player off-ball xT weighting)

The per-player composition (off-ball xT via PC share × xT, uniquely reachable
area generalized from GK-specific to all outfield) is novel to silly-kicks.
```

- [ ] **Step 2: Update CHANGELOG.md**

Add to the `## [Unreleased]` section (or create it if needed):

```markdown
### Added
- `compute_player_influence`: per-frame primitive computing off-ball xT and uniquely reachable area for all outfield players (TF-36 + TF-33)
- `add_player_influence`: action-coupled aggregator emitting 7 columns (`actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `off_ball_xt_diff`, `reachable_area_team`, `reachable_area_opponent`, `reachable_area_diff`)
- `player_influence_xfns`: VAEP factory (21 columns across 3 gamestate slots)
- 5 per-Series helpers: `actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `reachable_area_team`, `reachable_area_opponent`
- `PlayerInfluence` frozen dataclass return type
```

- [ ] **Step 3: Bump nan_safe_enrichment registry floor**

In `tests/test_enrichment_nan_safety.py`, update the tracking floor from ≥9 to ≥10:

```python
def test_registry_nonempty_tracking() -> None:
    """At least 10 @nan_safe_enrichment helpers in silly_kicks.tracking.features."""
    assert len(TRACKING_ENRICHMENTS) >= 10, (
        f"Expected ≥10 @nan_safe_enrichment helpers in silly_kicks.tracking.features; "
        f"found {len(TRACKING_ENRICHMENTS)}: {[fn.__name__ for fn in TRACKING_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )
```

- [ ] **Step 4: Update TODO.md**

Delete the TF-33 row (line starting with `**TF-33`) and the TF-36 row (line starting with `**TF-36`) from the `## Research & Future Work` section. Update the `**Last updated**` date to today and update `**Current release**` to the new version.

---

## Task 10: E2E provider fixture tests

**Files:**
- Create: `tests/tracking/test_player_influence_e2e.py`

Not marked `@pytest.mark.e2e` — all fixtures are committed to the repo. Follows the `test_gk_influence_e2e.py` pattern exactly.

- [ ] **Step 1: Create E2E provider test**

```python
# tests/tracking/test_player_influence_e2e.py
"""Provider fixture tests for player influence primitives (TF-36 + TF-33).

Uses committed slim-parquet fixtures (Sportec, Metrica, SkillCorner) and Gradient Sports
synthetic fixtures for real-data validation of add_player_influence. Exercises
string-typed IDs (Sportec/Metrica/SkillCorner), partial NaN velocities
(Metrica), and the full smooth→derive→link→compute pipeline.

Not marked @pytest.mark.e2e: all fixtures are committed to the repo.
"""
from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking import play_left_to_right
from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
from tests.tracking._provider_inputs import (
    GRADIENTSPORTS_DIR,
    SLIM_DIR,
    load_provider_frames,
    synthesize_actions,
)

_SLIM_PROVIDERS = sorted(p.stem.replace("_slim", "") for p in SLIM_DIR.glob("*_slim.parquet"))
_PROVIDERS = _SLIM_PROVIDERS + (["gradientsports"] if GRADIENTSPORTS_DIR.exists() else [])

_OUTPUT_COLS = [
    "actor_reachable_area_m2",
    "off_ball_xt_team",
    "off_ball_xt_opponent",
    "off_ball_xt_diff",
    "reachable_area_team",
    "reachable_area_opponent",
    "reachable_area_diff",
]


def _prepare(provider: str) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Load frames, add velocities, determine home_team_id, synthesize actions."""
    frames = load_provider_frames(provider)
    if "vx" not in frames.columns:
        frames = derive_velocities(smooth_frames(frames))
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    actions = synthesize_actions(frames)
    return actions, frames, home_team_id


@pytest.fixture(params=_PROVIDERS)
def provider_data(request, fitted_xt):
    """Load + preprocess real provider data for player influence tests."""
    provider = request.param
    actions, frames, home_team_id = _prepare(provider)
    return provider, actions, frames, home_team_id, fitted_xt


class TestPlayerInfluenceProviders:
    """Per-provider: add_player_influence runs on real data and produces valid output."""

    def test_adds_expected_columns(self, provider_data):
        """All 7 player influence columns present, output length matches input."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt, home_team_id=home_team_id)
        for col in _OUTPUT_COLS:
            assert col in result.columns, f"{provider}: missing column {col}"
        assert len(result) == len(actions)

    def test_non_nan_coverage(self, provider_data):
        """At least 1 non-NaN value per output column on real data."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt, home_team_id=home_team_id)
        for col in _OUTPUT_COLS:
            n_valid = result[col].notna().sum()
            assert n_valid >= 1, (
                f"{provider}: {col} has 0 non-NaN values out of {len(result)}"
            )

    def test_physical_invariants(self, provider_data):
        """Physical invariants: areas >= 0, off_ball_xt >= 0, diff identity."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt, home_team_id=home_team_id)

        # Areas >= 0
        for col in ["actor_reachable_area_m2", "reachable_area_team", "reachable_area_opponent"]:
            vals = result[col].dropna()
            assert (vals >= 0.0).all(), f"{provider}: {col} has negative values"

        # Off-ball xT >= 0
        for col in ["off_ball_xt_team", "off_ball_xt_opponent"]:
            vals = result[col].dropna()
            assert (vals >= 0.0).all(), f"{provider}: {col} has negative values"

        # Diff identity: _diff = _team - _opponent
        valid = result["off_ball_xt_team"].notna()
        if valid.any():
            pd.testing.assert_series_equal(
                result.loc[valid, "off_ball_xt_diff"],
                (
                    result.loc[valid, "off_ball_xt_team"]
                    - result.loc[valid, "off_ball_xt_opponent"]
                ).rename("off_ball_xt_diff"),
                check_exact=False,
                atol=1e-10,
            )
            pd.testing.assert_series_equal(
                result.loc[valid, "reachable_area_diff"],
                (
                    result.loc[valid, "reachable_area_team"]
                    - result.loc[valid, "reachable_area_opponent"]
                ).rename("reachable_area_diff"),
                check_exact=False,
                atol=1e-10,
            )

    def test_team_area_lte_pitch_area(self, provider_data):
        """Per-action team reachable area <= total pitch area."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt, home_team_id=home_team_id)

        pitch_area = 105.0 * 68.0
        for col in ["reachable_area_team", "reachable_area_opponent"]:
            vals = result[col].dropna()
            assert (vals <= pitch_area).all(), (
                f"{provider}: {col} exceeds pitch area (max={vals.max():.1f})"
            )
```

- [ ] **Step 2: Run E2E tests**

Run: `python -m pytest tests/tracking/test_player_influence_e2e.py -v --tb=short`
Expected: All PASSED across all available providers.

---

## Task 11: Pre-commit quality gates

- [ ] **Step 1: Run ruff check**

Run: `python -m ruff check silly_kicks/tracking/_player_influence.py silly_kicks/tracking/features.py`
Expected: No errors. Fix any issues.

- [ ] **Step 2: Run ruff format check**

Run: `python -m ruff format --check silly_kicks/tracking/_player_influence.py silly_kicks/tracking/features.py`
Expected: No reformatting needed. If needed, run `ruff format` on the files.

- [ ] **Step 3: Run pyright**

Run: `python -m pyright silly_kicks/tracking/_player_influence.py`
Expected: 0 errors.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short -x`
Expected: All PASSED, no regressions.

---

## Task 12: Final review

- [ ] **Step 1: Run `/final-review` skill**

Invoke the `mad-scientist-skills:final-review` skill for pre-commit quality gate.

- [ ] **Step 2: Address any findings**

Fix any issues found by the final review.

- [ ] **Step 3: Await explicit commit approval**

Present the summary of changes and await user approval before creating the single commit.
