# TF-15: GK Influence Primitives — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three per-frame GK influence primitives (threat-weighted pitch control share, reachable area, zone closing time) as the Layer 1 foundation for the GKDV metric (TF-19), with full action-coupled VAEP integration and bundled TF-31/TF-32 bugfixes.

**Architecture:** One new private module `_gk_influence.py` paralleling `_defensive_line.py` / `_team_shape.py`. Two prerequisite API extractions (public `compute_tti`, `select_back_line_players`) enable the primitives. Three frozen dataclasses (`Zone`, `ZoneClosingTime`, `GkInfluence`) model the return types. Action-coupled integration follows the standard three-tier pattern: per-frame primitive -> `add_gk_influence` aggregator with `@nan_safe_enrichment` -> `gk_influence_xfns` VAEP factory with `_frame_aware` marker and frame-precomputation cache. Atomic SPADL mirror delegates to standard computation.

**Tech Stack:** pandas, numpy, scipy (all already in dependency tree)

**Commit policy:** One commit per branch (user policy). All tasks produce working code; commit happens at the final task.

**Spec:** `docs/superpowers/specs/2026-05-09-tf15-gk-influence-primitives-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `silly_kicks/tracking/pitch_control/_spearman.py` (MODIFY) | Rename `_compute_tti` -> `compute_tti` (PR-1) |
| `silly_kicks/tracking/pitch_control/__init__.py` (MODIFY) | Export `compute_tti` |
| `silly_kicks/tracking/_defensive_line.py` (MODIFY) | Extract `select_back_line_players` (PR-2) |
| `silly_kicks/tracking/_gk_influence.py` (CREATE) | `Zone`, `ZoneClosingTime`, `GkInfluence` dataclasses + `compute_gk_influence` per-frame entry point |
| `silly_kicks/tracking/features.py` (MODIFY) | 4 per-Series helpers + `add_gk_influence` + `gk_influence_xfns`; `__all__` updates |
| `silly_kicks/tracking/__init__.py` (MODIFY) | Re-exports for new public API |
| `silly_kicks/atomic/tracking/features.py` (MODIFY) | Atomic SPADL mirror |
| `silly_kicks/tracking/_line_breaking.py` (MODIFY) | Bundled fixes H1, H2, H3, M4 |
| `silly_kicks/tracking/_team_shape.py` (MODIFY) | Bundled docs M2, M3 |
| `tests/tracking/test_gk_influence.py` (CREATE) | T-1..T-6 + T-PR1..T-PR2 unit tests |
| `tests/tracking/test_gk_influence_action_coupled.py` (CREATE) | T-7..T-10 action-coupled tests |
| `tests/tracking/test_gk_influence_perf_budget.py` (CREATE) | T-PB1 performance benchmark |
| `tests/tracking/test_gk_influence_e2e.py` (CREATE) | T-11 provider e2e tests |
| `tests/invariants/test_gk_influence_invariants.py` (CREATE) | T-12 physical invariants |
| `tests/tracking/test_team_shape_perf_budget.py` (CREATE) | M1 bundled perf benchmark |
| `tests/tracking/test_line_breaking_perf_budget.py` (CREATE) | M1 bundled perf benchmark |
| `tests/tracking/test_line_breaking_bundled_fixes.py` (CREATE) | H1, H2, M4 bundled fix tests |
| `tests/invariants/test_invariant_line_breaking.py` (MODIFY) | L2 provider sweep invariants |
| `tests/invariants/test_invariant_team_shape.py` (MODIFY) | L2 provider sweep invariants |
| `NOTICE` (MODIFY) | Add academic attribution entries |
| `TODO.md` (MODIFY) | Delete TF-15 row |
| `CHANGELOG.md` (MODIFY) | Add new features |
| `tests/conftest.py` (MODIFY) | Shared `fitted_xt` fixture (M-5, C-2 v2) |
| `tests/tracking/_gk_test_helpers.py` (CREATE) | Shared `_make_two_team_frame` builder (M-4 v2) |

---

## Lakehouse Review v1 Amendments

Applied fixes from lakehouse review (2026-05-10). See review disposition in session notes.

**C-1 (xfns links states[0] only):** Fixed in Task 6. xfns factory now calls `link_actions_to_frames` per slot inside the `for i, slot` loop, not once outside.

**C-2 (closing-time waste):** Added `compute_zone_closing_times` lightweight function in `_gk_influence.py` that calls `compute_tti` directly without pitch control. Per-Series closing-time helpers use it instead of `compute_gk_influence`.

**H-1 (silent exception swallowing):** All `except (ValueError, KeyError): continue` blocks now emit `warnings.warn(f"...: {exc}", UserWarning, stacklevel=2)` before continuing. Matches DAS pattern in `features.py:1844`.

**H-2 (xt: object):** Changed to `xt: ExpectedThreat` with `TYPE_CHECKING` guard. Verified: no circular import exists (no tracking module imports `xthreat`).

**H-3 (manual timing):** All perf tests now use `pytest-benchmark` (`benchmark` fixture), matching `test_pressure_perf_budget.py` and `pitch_control/test_perf_budget.py` patterns.

**H-4 (no batch kernel):** Added `_gk_influence_at_actions` batch kernel that computes once per unique `(period_id, frame_id, team_id)`. Per-Series helpers extract single columns from batch result. Aggregator and xfns call batch kernel.

**M-1 (is_goalkeeper):** All comparisons use `.astype(bool)` consistently.

**M-2 (string IDs):** Added one invariant test with string-typed team_id/player_id.

**M-3 (ball_y wiring):** Aggregator resolves ball position from frame's ball row. `add_gk_influence` and xfns accept `zone_names: list[str] | None` instead of `list[Zone]`; construct `Zone` per-action with resolved `goal_x` and `ball_y`. `compute_gk_influence` (per-frame) keeps `list[Zone] | None`.

**M-4 (back-line only):** Documented as intentional Layer 1 approximation in `compute_gk_influence` docstring.

**M-5 (duplicate fixture):** `fitted_xt` moved to `tests/conftest.py` (root test conftest — visible to both `tests/tracking/` and `tests/invariants/`).

**M-6 (e2e stubs):** Removed `pytest.skip()` bodies. Tests are `@pytest.mark.e2e`-only (excluded from regular suite). Implementor fills in loader using existing e2e patterns. StatsBomb correctly absent (no tracking data).

**M-7 (contradictory approaches):** Task 2 now specifies one canonical approach.

**L-1 (function-body imports):** Moved to module-level since no circular import exists.

**L-3 (loose cache assertion):** Tightened to `assert call_count[0] == 1`.

**Question 4 (zone goal_x):** `add_gk_influence`/xfns accept `zone_names: list[str] | None` (strings), not pre-built `Zone` instances. Zones constructed per-action with correct `goal_x` + `ball_y`.

### Lakehouse Review v2 Amendments

**C-1 v2 (test_additional_zones `zones=` kwarg):** Fixed. Test now passes `zone_names=["six_yard_box", "near_post", "far_post"]` matching the updated API.

**C-2 v2 (`fitted_xt` conftest scope):** Fixed. Moved to `tests/conftest.py` (parent of both `tracking/` and `invariants/`), not `tests/tracking/conftest.py` which is invisible to `tests/invariants/`.

**H-1 v2 (`is_ball` missing `.astype(bool)`):** Fixed. All `is_ball` comparisons in xfns `_get_gi` now use `.astype(bool)` consistently (3 instances).

**H-2 v2 (empty e2e test bodies):** Fixed. Restored `pytest.skip()` with descriptive reason strings. Empty bodies that silently pass are worse than skips.

**M-1 v2 (`benchmark.stats.stats.mean` access path):** Verified against `tests/tracking/pitch_control/test_perf_budget.py` (lines 70-71, 77-78, 84-85) — the double `.stats.stats` is correct. Added `if benchmark.stats is not None:` guard matching the existing pattern.

**M-2 v2 (double `link_actions_to_frames`):** Fixed. `_gk_influence_at_actions` now returns `(result_df, pointers)` tuple. `add_gk_influence` reuses pointers from batch kernel.

**M-3 v2 (redundant frame lookup for `ball_y`):** Fixed. `ball_y` resolution moved inside `_get_gi` where `frame_data` is already available. `ball_y` parameter removed from `_get_gi` signature (it's a per-frame property deterministic from the frame, not per-action).

**M-4 v2 (cross-file test helper coupling):** Fixed. `_make_two_team_frame` moved to `tests/tracking/_gk_test_helpers.py`. All test files import from there instead of from `test_gk_influence.py`.

**L-1 v2 (unnecessary `.copy()`):** Fixed. Removed `.copy()` from player filtering in `compute_gk_influence` — the filtered DataFrame is read-only.

**L-2 v2 (no per-frame cache in `_closing_time_per_series`):** Acknowledged. Acceptable for now since `compute_zone_closing_times` is very lightweight (just `compute_tti` on ~9 zone points). Noted for future profiling.

**VQ-1 (`cache: dict[tuple, object]`):** Fixed to `dict[tuple, GkInfluence | None]` (matching xfns version).

**VQ-2 (`gk_reachable_area_m2` requires `xt`):** Intentional architectural constraint. `_gk_influence_at_actions` computes all 3 primitives together (pitch control needed for share + reachable area). Per-Series closing-time helpers use the lightweight `compute_zone_closing_times` path that doesn't need `xt`. Per-Series `gk_reachable_area_m2` goes through the batch kernel (which needs `xt` for share computation) — acceptable tradeoff for cache efficiency.

**VQ-3 (TF-19 backlog status):** Confirmed filed. TF-19 is in TODO.md (Tier 6, line 46) with full spec, dependencies, and validation strategy. The "Layer 2" reference in the docstring is not dangling.

---

### Task 1: Prerequisite PR-1 — Export `compute_tti` from pitch control

**Files:**
- Modify: `silly_kicks/tracking/pitch_control/_spearman.py:28-81`
- Modify: `silly_kicks/tracking/pitch_control/__init__.py`
- Create: `tests/tracking/test_gk_influence.py` (T-PR1 tests only at this stage)

- [ ] **Step 1: Write failing tests for public `compute_tti` import**

Create `tests/tracking/test_gk_influence.py`:

```python
"""Tests for TF-15 GK influence primitives."""

from __future__ import annotations

import numpy as np
import pytest


# === T-PR1: compute_tti public export ===


class TestComputeTtiExport:
    """T-PR1: compute_tti is importable from pitch_control and produces correct results."""

    def test_importable_from_pitch_control(self):
        """compute_tti is importable from the public pitch_control namespace."""
        from silly_kicks.tracking.pitch_control import compute_tti

        assert callable(compute_tti)

    def test_regression_parity_with_private(self):
        """Public compute_tti produces identical results to the private _compute_tti."""
        from silly_kicks.tracking.pitch_control import compute_tti

        pos = np.array([[0.0, 0.0], [10.0, 10.0]])
        vel = np.array([[3.0, 0.0], [0.0, -2.0]])
        targets = np.array([[5.0, 0.0], [10.0, 5.0], [50.0, 34.0]])

        result = compute_tti(pos, vel, targets, 0.7, 7.0)

        assert result.shape == (2, 3)
        # Player at origin moving right: TTI to (5,0) should be less than to (50,34)
        assert result[0, 0] < result[0, 2]
        # All TTI values >= reaction_time
        assert np.all(result >= 0.7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_influence.py::TestComputeTtiExport -v`
Expected: FAIL with `ImportError: cannot import name 'compute_tti'`

- [ ] **Step 3: Rename `_compute_tti` to `compute_tti` in `_spearman.py`**

In `silly_kicks/tracking/pitch_control/_spearman.py`:
- Rename `def _compute_tti(` -> `def compute_tti(`
- Update the internal call at line 178: `tti_all = _compute_tti(` -> `tti_all = compute_tti(`
- Update docstring Examples to remove the leading underscore

- [ ] **Step 4: Export from `pitch_control/__init__.py`**

Add to imports:
```python
from ._spearman import compute_tti
```

Add `"compute_tti"` to `__all__` (alphabetical order, after `"compute_pitch_control_at_points"`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_gk_influence.py::TestComputeTtiExport -v`
Expected: 2 PASSED

---

### Task 2: Prerequisite PR-2 — Extract `select_back_line_players`

**Files:**
- Modify: `silly_kicks/tracking/_defensive_line.py:132-150`
- Test: `tests/tracking/test_gk_influence.py` (append T-PR2)

- [ ] **Step 1: Write failing tests for `select_back_line_players`**

Append to `tests/tracking/test_gk_influence.py`:

```python
import pandas as pd


def _make_outfield_frame(
    *,
    positions: list[tuple[float, float]],
    team_id: int = 1,
    home_team_id: int = 1,
    velocities: list[tuple[float, float]] | None = None,
    frame_id: int = 1,
    period_id: int = 1,
    game_id: int = 1,
) -> pd.DataFrame:
    """Build a minimal tracking frame with outfield players + ball + GK."""
    rows = []
    # Ball row
    rows.append(dict(
        game_id=game_id, period_id=period_id, frame_id=frame_id,
        time_seconds=1.0, frame_rate=25.0,
        player_id=np.nan, team_id=np.nan,
        is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # GK
    gk_x = 3.0 if team_id == home_team_id else 102.0
    rows.append(dict(
        game_id=game_id, period_id=period_id, frame_id=frame_id,
        time_seconds=1.0, frame_rate=25.0,
        player_id=99, team_id=team_id,
        is_ball=False, is_goalkeeper=True,
        x=gk_x, y=34.0, vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # Outfield
    for i, (px, py) in enumerate(positions):
        vx_val = velocities[i][0] if velocities else 0.0
        vy_val = velocities[i][1] if velocities else 0.0
        rows.append(dict(
            game_id=game_id, period_id=period_id, frame_id=frame_id,
            time_seconds=1.0, frame_rate=25.0,
            player_id=100 + i, team_id=team_id,
            is_ball=False, is_goalkeeper=False,
            x=px, y=py, vx=vx_val, vy=vy_val,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    return pd.DataFrame(rows)


class TestSelectBackLinePlayers:
    """T-PR2: select_back_line_players returns individual player rows."""

    def test_returns_player_rows_with_coordinates(self):
        """Returns DataFrame with x, y, vx, vy preserved per player."""
        from silly_kicks.tracking._defensive_line import select_back_line_players

        frames = _make_outfield_frame(
            positions=[(10, 20), (15, 30), (20, 40), (50, 34), (60, 25)],
            velocities=[(1, 0), (2, 0), (-1, 0), (0, 1), (3, -1)],
            team_id=1, home_team_id=1,
        )
        result = select_back_line_players(frames, team_id=1, home_team_id=1, n=4)

        assert len(result) == 4
        assert set(result.columns) >= {"x", "y", "vx", "vy", "player_id"}
        # Home team defends x=0, so back line = lowest x values
        assert result["x"].max() <= 50.0

    def test_defensive_line_unchanged_after_refactor(self):
        """compute_defensive_line produces identical output after refactor."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_outfield_frame(
            positions=[(10, 20), (15, 30), (20, 40), (50, 34), (60, 25)],
            team_id=1, home_team_id=1,
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)

        assert len(result) == 1
        assert result["defensive_line_x"].notna().all()
        assert result["back_n_count"].iloc[0] == 4

    def test_away_team_selects_highest_x(self):
        """Away team (defends x=105) selects players closest to x=105."""
        from silly_kicks.tracking._defensive_line import select_back_line_players

        frames = _make_outfield_frame(
            positions=[(40, 20), (50, 30), (80, 40), (90, 34), (95, 25)],
            team_id=2, home_team_id=1,
        )
        result = select_back_line_players(frames, team_id=2, home_team_id=1, n=4)

        assert len(result) == 4
        # Away team defends x=105, so back line = highest x values
        assert result["x"].min() >= 50.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_influence.py::TestSelectBackLinePlayers -v`
Expected: FAIL with `ImportError: cannot import name 'select_back_line_players'`

- [ ] **Step 3: Implement `select_back_line_players` in `_defensive_line.py`**

Add new function before `compute_defensive_line`:

```python
def select_back_line_players(
    frames: pd.DataFrame,
    team_id: int | str,
    home_team_id: int | str,
    *,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Select the N outfield players closest to their own goal.

    Returns a DataFrame of player rows (preserving x, y, vx, vy, player_id,
    etc.) sorted by proximity to own goal. Operates on a single frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frame (single frame expected, but multi-frame
        is tolerated — groups by game_id/period_id/frame_id).
    team_id : int | str
        Team to select back-line players for.
    home_team_id : int | str
        Home team identifier for goal-end resolution.
    n : int | Literal["adaptive"], default 4
        Target back-line player count. Clamped to available outfield.
    adaptive_max_n : int, default 5
        Upper bound for adaptive N.

    Returns
    -------
    pd.DataFrame
        Player rows with all original columns preserved, sorted by
        proximity to own goal. Length = min(n_effective, available_outfield).

    Examples
    --------
    >>> from silly_kicks.tracking._defensive_line import select_back_line_players
    >>> back_line = select_back_line_players(frame, team_id=1, home_team_id=1)
    >>> back_line[["player_id", "x", "y"]].head()

    See NOTICE for full bibliographic citations.
    """
    outfield = frames[
        (~frames["is_ball"].astype(bool))
        & (~frames["is_goalkeeper"].astype(bool))
        & (frames["team_id"] == team_id)
        & frames["x"].notna()
    ].copy()

    if len(outfield) < 3:
        return outfield

    defends_x0 = team_id == home_team_id
    xs = outfield["x"].to_numpy(dtype="float64")

    if defends_x0:
        order = np.argsort(xs)
    else:
        order = np.argsort(-xs)

    xs_sorted = xs[order]
    p = len(outfield)
    n_effective = _select_n(xs_sorted, n, adaptive_max_n, p)

    return outfield.iloc[order[:n_effective]]
```

**Canonical refactor approach (one path only):** `select_back_line_players` does its own filtering from raw frames (is_ball, is_goalkeeper, team_id, x.notna). `compute_defensive_line` calls it per group, passing the full per-(game, period, frame) frame slice. In the groupby loop, replace lines 106-150:

```python
    # Filter once for groupby (keep existing performance)
    outfield = frames[
        (~frames["is_ball"].astype(bool))
        & (~frames["is_goalkeeper"].astype(bool))
        & frames["x"].notna()
    ].copy()

    groups = outfield.groupby(
        ["game_id", "period_id", "frame_id", "team_id"], dropna=False,
    )

    for (game_id, period_id, frame_id, team_id), group in groups:
        p = len(group)
        if p < 3:
            rows.append({...NaN row...})
            continue

        # Delegate sort+select to shared helper.
        # group is already filtered to one team's outfield — pass it directly.
        # select_back_line_players will re-apply filters (idempotent on pre-filtered data).
        frame_slice = frames[
            (frames["game_id"] == game_id)
            & (frames["period_id"] == period_id)
            & (frames["frame_id"] == frame_id)
        ]
        selected = select_back_line_players(
            frame_slice, team_id=team_id, home_team_id=home_team_id,
            n=n, adaptive_max_n=adaptive_max_n,
        )
        n_effective = len(selected)
        if n_effective < 3:
            rows.append({...NaN row...})
            continue
        sel_x = selected["x"].to_numpy(dtype="float64")
        sel_y = selected["y"].to_numpy(dtype="float64")
        defends_x0 = team_id == home_team_id

        # ... rest of computation unchanged (lines 153-184) ...
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/tracking/test_gk_influence.py -v`
Expected: 5 PASSED (2 T-PR1 + 3 T-PR2)

Also run existing defensive line tests for regression:
Run: `python -m pytest tests/tracking/ -k "defensive_line" -v`
Expected: All existing tests PASS

---

### Task 3: Zone dataclass + ZoneClosingTime + GkInfluence return types

**Files:**
- Create: `silly_kicks/tracking/_gk_influence.py` (dataclasses only — no `compute_gk_influence` yet)
- Test: `tests/tracking/test_gk_influence.py` (append T-1 zone geometry tests)

- [ ] **Step 1: Write failing T-1 zone geometry tests**

Append to `tests/tracking/test_gk_influence.py`:

```python
class TestZoneGeometry:
    """T-1: Zone dataclass factory methods produce correct geometry."""

    def test_six_yard_box_goal_x_0(self):
        from silly_kicks.tracking._gk_influence import Zone

        zone = Zone.six_yard_box(goal_x=0.0)
        assert zone.name == "six_yard_box"
        assert zone.points.shape[1] == 2
        assert len(zone.points) >= 6
        # All x in [0, 5.5], y in [24.84, 43.16]
        assert np.all(zone.points[:, 0] >= 0.0)
        assert np.all(zone.points[:, 0] <= 5.5)
        assert np.all(zone.points[:, 1] >= 24.84)
        assert np.all(zone.points[:, 1] <= 43.16)

    def test_six_yard_box_goal_x_105(self):
        from silly_kicks.tracking._gk_influence import Zone

        zone = Zone.six_yard_box(goal_x=105.0)
        assert np.all(zone.points[:, 0] >= 99.5)
        assert np.all(zone.points[:, 0] <= 105.0)

    def test_near_far_post_corridors(self):
        from silly_kicks.tracking._gk_influence import Zone

        near = Zone.near_post(goal_x=0.0, ball_y=25.0)
        far = Zone.far_post(goal_x=0.0, ball_y=25.0)
        assert near.name == "near_post"
        assert far.name == "far_post"
        # Near post should be closer to ball_y=25 than far post
        near_mean_y = near.points[:, 1].mean()
        far_mean_y = far.points[:, 1].mean()
        assert abs(near_mean_y - 25.0) < abs(far_mean_y - 25.0)

    def test_frozen_immutability(self):
        from silly_kicks.tracking._gk_influence import Zone

        zone = Zone.six_yard_box(goal_x=0.0)
        with pytest.raises(ValueError, match="read-only"):
            zone.points[0, 0] = 999.0

    def test_ball_relative_near_post_different_sides(self):
        """near_post gives different point sets for ball_y=25 vs ball_y=40."""
        from silly_kicks.tracking._gk_influence import Zone

        near_low = Zone.near_post(goal_x=0.0, ball_y=25.0)
        near_high = Zone.near_post(goal_x=0.0, ball_y=40.0)
        # Different ball positions should select different goalposts
        assert not np.allclose(near_low.points, near_high.points)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_influence.py::TestZoneGeometry -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement dataclasses in `_gk_influence.py`**

Create `silly_kicks/tracking/_gk_influence.py`:

```python
"""GK influence primitives (TF-15, GKDV Layer 1).

Three per-frame primitives measuring distinct aspects of GK spatial
contribution: threat-weighted pitch control share, uniquely reachable area,
and zone closing time.

See docs/superpowers/specs/2026-05-09-tf15-gk-influence-primitives-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._defensive_line import select_back_line_players
from .pitch_control import PitchControlParams, SpearmanParams, compute_pitch_control
from .pitch_control._spearman import compute_tti

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat


# ---------------------------------------------------------------------------
# Goal geometry constants
# ---------------------------------------------------------------------------

_FIELD_WIDTH = spadlconfig.field_width  # 68.0
_POST_LEFT_Y = (_FIELD_WIDTH - 7.32) / 2  # 30.34
_POST_RIGHT_Y = (_FIELD_WIDTH + 7.32) / 2  # 37.66


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneClosingTime:
    """GK closing time to a single zone.

    Examples
    --------
    >>> zct = ZoneClosingTime(min_s=0.8, mean_s=1.2)
    """

    min_s: float
    mean_s: float


@dataclass(frozen=True)
class GkInfluence:
    """Per-frame GK influence measurement (all three primitives).

    Examples
    --------
    >>> gi = GkInfluence(
    ...     pitch_control_share_weighted=0.12,
    ...     reachable_area_m2=150.0,
    ...     closing_times={"six_yard_box": ZoneClosingTime(0.8, 1.2)},
    ... )
    """

    pitch_control_share_weighted: float
    reachable_area_m2: float
    closing_times: dict[str, ZoneClosingTime]


# ---------------------------------------------------------------------------
# Zone dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Zone:
    """A named set of target points for GK closing-time computation.

    Examples
    --------
    >>> zone = Zone.six_yard_box(goal_x=0.0)
    >>> zone.points.shape
    (9, 2)
    """

    name: str
    points: np.ndarray  # (N, 2) — x, y in meters, LTR-normalized

    def __post_init__(self) -> None:
        """Enforce array immutability."""
        self.points.flags.writeable = False

    @staticmethod
    def six_yard_box(goal_x: float) -> Zone:
        """~9 evenly-spaced points covering the six-yard box.

        Examples
        --------
        >>> Zone.six_yard_box(goal_x=0.0).points.shape
        (9, 2)
        """
        if goal_x == 0.0:
            xs = np.linspace(0.0, 5.5, 3)
        else:
            xs = np.linspace(99.5, 105.0, 3)
        ys = np.linspace(_POST_LEFT_Y, _POST_RIGHT_Y, 3)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return Zone(name="six_yard_box", points=pts)

    @staticmethod
    def near_post(goal_x: float, ball_y: float | None = None) -> Zone:
        """~4 points near the goalpost closest to the ball.

        Examples
        --------
        >>> Zone.near_post(goal_x=0.0, ball_y=25.0).name
        'near_post'
        """
        near_y, far_y = _resolve_near_far_post_y(ball_y)
        return _build_post_zone("near_post", goal_x, near_y)

    @staticmethod
    def far_post(goal_x: float, ball_y: float | None = None) -> Zone:
        """~4 points near the goalpost farthest from the ball.

        Examples
        --------
        >>> Zone.far_post(goal_x=0.0, ball_y=25.0).name
        'far_post'
        """
        near_y, far_y = _resolve_near_far_post_y(ball_y)
        return _build_post_zone("far_post", goal_x, far_y)


def _resolve_near_far_post_y(ball_y: float | None) -> tuple[float, float]:
    """Determine near-post and far-post y based on ball position."""
    if ball_y is not None:
        d_left = abs(_POST_LEFT_Y - ball_y)
        d_right = abs(_POST_RIGHT_Y - ball_y)
        if d_left <= d_right:
            return _POST_LEFT_Y, _POST_RIGHT_Y
        else:
            return _POST_RIGHT_Y, _POST_LEFT_Y
    else:
        # Fixed proxy: left half -> near = left post
        return _POST_LEFT_Y, _POST_RIGHT_Y


def _build_post_zone(name: str, goal_x: float, post_y: float) -> Zone:
    """Build a ~4-point zone around one goalpost."""
    center_y = (_POST_LEFT_Y + _POST_RIGHT_Y) / 2
    if goal_x == 0.0:
        xs = np.array([0.0, 2.75])
    else:
        xs = np.array([102.25, 105.0])
    # Y corridor: from post_y toward center, 2 points
    mid_y = (post_y + center_y) / 2
    ys = np.array([post_y, mid_y])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return Zone(name=name, points=pts)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/tracking/test_gk_influence.py::TestZoneGeometry -v`
Expected: 5 PASSED

---

### Task 4: `compute_gk_influence` core logic — tests + implementation

**Files:**
- Modify: `silly_kicks/tracking/_gk_influence.py` (add `compute_gk_influence`)
- Test: `tests/tracking/test_gk_influence.py` (append T-2, T-3, T-5, T-6)

This is the largest task. It implements the per-frame entry point with all three primitives.

- [ ] **Step 1: Write T-2 core logic tests**

First, create `tests/tracking/_gk_test_helpers.py` with the shared frame builder (M-4 v2 fix — avoids fragile cross-file test imports). Define `_make_two_team_frame` in that module:

```python
"""Shared test helpers for TF-15 GK influence tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_two_team_frame(
    *,
    home_positions: list[tuple[float, float]],
    away_positions: list[tuple[float, float]],
    home_gk_pos: tuple[float, float] = (3.0, 34.0),
    away_gk_pos: tuple[float, float] = (102.0, 34.0),
    home_team_id: int = 1,
    away_team_id: int = 2,
    home_velocities: list[tuple[float, float]] | None = None,
    away_velocities: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build a two-team frame with GKs, outfield players, and ball."""
    rows = []
    # Ball
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=np.nan, team_id=np.nan,
        is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # Home GK
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=1, team_id=home_team_id,
        is_ball=False, is_goalkeeper=True,
        x=home_gk_pos[0], y=home_gk_pos[1], vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # Away GK
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=50, team_id=away_team_id,
        is_ball=False, is_goalkeeper=True,
        x=away_gk_pos[0], y=away_gk_pos[1], vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # Home outfield
    for i, (px, py) in enumerate(home_positions):
        vx_v = home_velocities[i][0] if home_velocities else 0.0
        vy_v = home_velocities[i][1] if home_velocities else 0.0
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=10 + i, team_id=home_team_id,
            is_ball=False, is_goalkeeper=False,
            x=px, y=py, vx=vx_v, vy=vy_v,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    # Away outfield
    for i, (px, py) in enumerate(away_positions):
        vx_v = away_velocities[i][0] if away_velocities else 0.0
        vy_v = away_velocities[i][1] if away_velocities else 0.0
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=60 + i, team_id=away_team_id,
            is_ball=False, is_goalkeeper=False,
            x=px, y=py, vx=vx_v, vy=vy_v,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    return pd.DataFrame(rows)
```

Then in `tests/tracking/test_gk_influence.py`, import from the helper module:

```python
from tests.tracking._gk_test_helpers import _make_two_team_frame

# fitted_xt fixture inherited from tests/conftest.py (M-5, C-2 v2)


@pytest.fixture
def standard_frame():
    """Standard 10v10 + 2 GK frame."""
    home_pos = [(20, 15), (25, 25), (30, 40), (35, 55),
                (45, 10), (50, 30), (55, 45), (60, 55), (70, 30), (75, 40)]
    away_pos = [(85, 15), (80, 25), (75, 40), (70, 55),
                (60, 10), (55, 30), (50, 45), (45, 55), (35, 30), (30, 40)]
    return _make_two_team_frame(
        home_positions=home_pos,
        away_positions=away_pos,
    )


class TestComputeGkInfluenceCore:
    """T-2: Core logic of compute_gk_influence."""

    def test_weighted_share_less_than_raw(self, standard_frame, fitted_xt):
        """Threat-weighted share < raw player_share when GK near own goal."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        # Raw share for comparison
        surface = compute_pitch_control(
            standard_frame, attacking_team_id=2, method="spearman",
            decompose=True,
        )
        raw_share = surface.player_share(1)
        assert gi.pitch_control_share_weighted < raw_share

    def test_share_in_range(self, standard_frame, fitted_xt):
        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_reachable_area_no_defenders(self, fitted_xt):
        """With no outfield defenders, reachable area ~ full GK circle."""
        frame = _make_two_team_frame(
            home_positions=[], away_positions=[],
            home_gk_pos=(3.0, 34.0), away_gk_pos=(102.0, 34.0),
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert gi.reachable_area_m2 > 0.0

    def test_reachable_area_decreases_with_defenders(self, fitted_xt):
        """Adding defenders reduces reachable area."""
        frame_no_def = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[],
        )
        frame_with_def = _make_two_team_frame(
            home_positions=[(5, 20), (8, 30), (6, 40), (7, 50)],
            away_positions=[],
        )
        gi_far = compute_gk_influence(
            frame_no_def, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        gi_near = compute_gk_influence(
            frame_with_def, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert gi_far.reachable_area_m2 > gi_near.reachable_area_m2

    def test_reachable_area_non_negative(self, standard_frame, fitted_xt):
        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert gi.reachable_area_m2 >= 0.0

    def test_closing_time_near_vs_far(self, fitted_xt):
        """GK at six-yard box -> low min_s; GK at halfway -> high min_s."""
        frame_near = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
            home_gk_pos=(3.0, 34.0),
        )
        frame_far = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
            home_gk_pos=(52.5, 34.0),
        )
        gi_near = compute_gk_influence(
            frame_near, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        gi_far = compute_gk_influence(
            frame_far, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        ct_near = gi_near.closing_times["six_yard_box"]
        ct_far = gi_far.closing_times["six_yard_box"]
        assert ct_near.min_s < ct_far.min_s
```

- [ ] **Step 2: Write T-5 xT orientation tests**

Append:

```python
class TestXtOrientation:
    """T-5: xT interpolation and flip logic."""

    def test_xt_interpolated_sum_positive(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        # If we got a valid share, the threat sum was positive
        assert not np.isnan(gi.pitch_control_share_weighted)

    def test_xt_all_zeros_returns_nan(self, standard_frame):
        from silly_kicks.tracking._gk_influence import compute_gk_influence
        from silly_kicks.xthreat import ExpectedThreat

        xt_zero = ExpectedThreat(l=16, w=12)
        xt_zero.xT = np.zeros((12, 16))
        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=xt_zero, home_team_id=1,
        )
        assert np.isnan(gi.pitch_control_share_weighted)

    def test_home_attack_no_flip(self, fitted_xt):
        """When home team attacks (toward x=105), xT is NOT flipped."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(80, 30), (85, 40), (70, 20), (75, 50)],
            away_positions=[(20, 30), (25, 40), (15, 20), (10, 50)],
            away_gk_pos=(3.0, 34.0),
        )
        # Home attacks -> away defends -> away GK (id=50) at x=3
        gi = compute_gk_influence(
            frame, attacking_team_id=1, gk_player_id=50,
            xt=fitted_xt, home_team_id=1,
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_away_attack_flip(self, fitted_xt):
        """When away team attacks (toward x=0), xT is x-flipped."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        # Same frame but away attacks → home GK (id=1) at x=3 defends x=0
        frame = _make_two_team_frame(
            home_positions=[(20, 30), (25, 40), (15, 20), (10, 50)],
            away_positions=[(80, 30), (85, 40), (70, 20), (75, 50)],
        )
        gi_away = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        gi_home = compute_gk_influence(
            frame, attacking_team_id=1, gk_player_id=50,
            xt=fitted_xt, home_team_id=1,
        )
        # Different attacking directions should produce different shares
        assert gi_away.pitch_control_share_weighted != gi_home.pitch_control_share_weighted

    def test_flip_is_x_only_not_y(self, fitted_xt):
        """Flip is [:, ::-1] not [::-1, ::-1] — y-axis preserved."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        # Asymmetric xT in y to detect y-flip
        xt_asym = fitted_xt
        # Make top row different from bottom row
        xt_asym.xT[0, :] = 0.1
        xt_asym.xT[-1, :] = 0.9

        frame = _make_two_team_frame(
            home_positions=[(20, 30), (25, 40), (15, 20), (10, 50)],
            away_positions=[(80, 30), (85, 40), (70, 20), (75, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=xt_asym, home_team_id=1,
        )
        # Should produce a valid result (no crash from y-flip mismatch)
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_interpolated_grid_shape(self, fitted_xt):
        """xT interpolated onto pitch control grid has correct shape."""
        interp = fitted_xt.interpolator(kind="linear")
        from silly_kicks.tracking.pitch_control import SpearmanParams

        p = SpearmanParams()
        grid_x = np.linspace(0, 105.0, p.grid_cells_x)
        grid_y = np.linspace(0, 68.0, p.grid_cells_y)
        threat_grid = interp(grid_x, grid_y)
        assert threat_grid.shape == (p.grid_cells_y, p.grid_cells_x)
```

- [ ] **Step 3: Write T-6 edge case tests**

Append:

```python
class TestGkInfluenceEdgeCases:
    """T-6: Edge cases for compute_gk_influence."""

    def test_gk_not_in_frame_raises(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        with pytest.raises(ValueError, match="not found"):
            compute_gk_influence(
                standard_frame, attacking_team_id=2, gk_player_id=999,
                xt=fitted_xt, home_team_id=1,
            )

    def test_min_s_leq_mean_s(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        for zct in gi.closing_times.values():
            assert zct.min_s <= zct.mean_s

    def test_near_zero_denominator_share_zero(self, fitted_xt):
        """Near-zero team_influence -> share = 0, not infinity."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        # Frame where defending team is extremely weak (GK only, no outfield)
        frame = _make_two_team_frame(
            home_positions=[],
            away_positions=[(50, 30), (55, 40), (60, 20), (65, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert not np.isinf(gi.pitch_control_share_weighted)

    def test_custom_reaction_time_lower_closing(self, standard_frame, fitted_xt):
        """Lower reaction_time -> lower closing times."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi_default = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, gk_reaction_time=0.4,
        )
        gi_fast = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, gk_reaction_time=0.3,
        )
        ct_default = gi_default.closing_times["six_yard_box"]
        ct_fast = gi_fast.closing_times["six_yard_box"]
        assert ct_fast.min_s < ct_default.min_s

    def test_custom_reaction_time_larger_reachable(self, standard_frame, fitted_xt):
        """Lower reaction_time -> larger reachable area (monotonicity)."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi_default = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, gk_reaction_time=0.4,
        )
        gi_fast = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, gk_reaction_time=0.3,
        )
        assert gi_fast.reachable_area_m2 >= gi_default.reachable_area_m2

    def test_ball_relative_near_post_zones(self, fitted_xt):
        """near_post(ball_y=25) vs near_post(ball_y=40) give different zones."""
        from silly_kicks.tracking._gk_influence import Zone

        z1 = Zone.near_post(goal_x=0.0, ball_y=25.0)
        z2 = Zone.near_post(goal_x=0.0, ball_y=40.0)
        assert not np.allclose(z1.points, z2.points)

    def test_no_outfield_defenders_full_reachable(self, fitted_xt):
        """No outfield defenders -> reachable area = full GK circle."""
        frame = _make_two_team_frame(
            home_positions=[],
            away_positions=[(50, 30), (55, 40), (60, 20), (65, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert gi.reachable_area_m2 > 0.0
```

- [ ] **Step 4: Write T-3 method dispatch tests**

Append:

```python
class TestMethodDispatch:
    """T-3: All pitch control methods produce valid GkInfluence."""

    @pytest.mark.parametrize("method", ["spearman", "fernandez_bornn", "voronoi"])
    def test_method_produces_valid_result(self, method, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, method=method,
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0
        assert gi.reachable_area_m2 >= 0.0
        assert "six_yard_box" in gi.closing_times
```

- [ ] **Step 5: Write T-4 zone-parameterized tests**

Append:

```python
class TestZoneParameterized:
    """T-4: Zone-parameterized closing time."""

    def test_closing_times_keys(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import Zone, compute_gk_influence

        zones = [
            Zone.six_yard_box(goal_x=0.0),
            Zone.near_post(goal_x=0.0, ball_y=34.0),
            Zone.far_post(goal_x=0.0, ball_y=34.0),
        ]
        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, zones=zones,
        )
        assert set(gi.closing_times.keys()) == {"six_yard_box", "near_post", "far_post"}

    def test_gk_near_near_post_physical_invariant(self, fitted_xt):
        """GK near near-post -> near_post min_s < far_post min_s."""
        from silly_kicks.tracking._gk_influence import Zone, compute_gk_influence

        # GK at (3, 31) — near the left post (y=30.34)
        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
            home_gk_pos=(3.0, 31.0),
        )
        zones = [
            Zone.near_post(goal_x=0.0, ball_y=25.0),
            Zone.far_post(goal_x=0.0, ball_y=25.0),
        ]
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, zones=zones,
        )
        assert gi.closing_times["near_post"].min_s < gi.closing_times["far_post"].min_s

    @pytest.mark.parametrize("zone_factory,goal_x", [
        ("six_yard_box", 0.0),
        ("near_post", 0.0),
        ("far_post", 105.0),
    ])
    def test_zone_produces_valid_closing_time(self, zone_factory, goal_x, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import Zone, compute_gk_influence

        zone_fn = getattr(Zone, zone_factory)
        zone = zone_fn(goal_x=goal_x) if zone_factory == "six_yard_box" else zone_fn(goal_x=goal_x, ball_y=34.0)
        gi = compute_gk_influence(
            standard_frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, zones=[zone],
        )
        zct = gi.closing_times[zone_factory]
        assert zct.min_s >= 0.0
        assert zct.mean_s >= zct.min_s
```

- [ ] **Step 6: Run all tests to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_influence.py -v -k "not Export and not BackLine"`
Expected: FAIL with `ImportError: cannot import name 'compute_gk_influence'`

- [ ] **Step 7: Implement `compute_zone_closing_times` + `compute_gk_influence` in `_gk_influence.py`**

Append to `silly_kicks/tracking/_gk_influence.py` after the Zone class.

First, the lightweight closing-times-only function (C-2 fix — avoids pitch control computation):

```python
def compute_zone_closing_times(
    frame: pd.DataFrame,
    gk_player_id: int | str,
    zones: list[Zone],
    *,
    gk_reaction_time: float = 0.4,
    gk_max_acceleration: float = 7.0,
) -> dict[str, ZoneClosingTime]:
    """Compute GK closing time to zones WITHOUT pitch control overhead.

    Lightweight path for callers who only need closing times (not share
    or reachable area). Calls compute_tti directly.

    Examples
    --------
    >>> from silly_kicks.tracking._gk_influence import compute_zone_closing_times, Zone
    >>> cts = compute_zone_closing_times(frame, gk_player_id=1,
    ...     zones=[Zone.six_yard_box(goal_x=0.0)])
    """
    players = frame[~frame["is_ball"].astype(bool)].dropna(subset=["x", "y"])
    gk_mask = players["player_id"] == gk_player_id
    if not gk_mask.any():
        raise ValueError(
            f"gk_player_id={gk_player_id!r} not found in frame"
        )
    gk_row = players[gk_mask].iloc[0]
    gk_pos = np.array([[float(gk_row["x"]), float(gk_row["y"])]])
    gk_vel_x = float(gk_row.get("vx", 0.0)) if pd.notna(gk_row.get("vx")) else 0.0
    gk_vel_y = float(gk_row.get("vy", 0.0)) if pd.notna(gk_row.get("vy")) else 0.0
    gk_vel = np.array([[gk_vel_x, gk_vel_y]])

    result: dict[str, ZoneClosingTime] = {}
    for zone in zones:
        zone_tti = compute_tti(
            gk_pos, gk_vel, zone.points,
            gk_reaction_time, gk_max_acceleration,
        )[0]
        result[zone.name] = ZoneClosingTime(
            min_s=float(zone_tti.min()),
            mean_s=float(zone_tti.mean()),
        )
    return result
```

Then the full `compute_gk_influence`:

```python
def compute_gk_influence(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    gk_player_id: int | str,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
    zones: list[Zone] | None = None,
    tau_seconds: float = 1.0,
    gk_reaction_time: float = 0.4,
    gk_max_acceleration: float = 7.0,
) -> GkInfluence:
    """Per-frame GK influence measurement (all three primitives).

    Uses back-line defenders only for reachable area (primitive b). This is
    an intentional Layer 1 approximation; full-team outfield coverage is
    deferred to GKDV Layer 2 (TF-19).

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data (TRACKING_FRAMES_COLUMNS schema).
    attacking_team_id : int | str
        Team currently in possession / attacking.
    gk_player_id : int | str
        The defending GK's player_id.
    xt : ExpectedThreat
        Pre-fit xT model for threat weighting.
    home_team_id : int | str
        Home team identifier (REQUIRED). Determines goal-end resolution.
    method : str, default "spearman"
        Pitch control model for primitive (a). Primitives (b) and (c)
        always use the Spearman kinematic TTI model regardless.
    params : PitchControlParams | None
        Optional pitch control params override.
    zones : list[Zone] | None
        Target zones for closing time. Default: [Zone.six_yard_box(goal_x)].
    tau_seconds : float, default 1.0
        TTI threshold for reachable area (primitive b).
    gk_reaction_time : float, default 0.4
        GK-specific reaction time (seconds).
    gk_max_acceleration : float, default 7.0
        GK-specific max acceleration (m/s^2).

    Returns
    -------
    GkInfluence

    Raises
    ------
    ValueError
        If gk_player_id is not found in the frame.

    Examples
    --------
    >>> from silly_kicks.tracking._gk_influence import compute_gk_influence
    >>> gi = compute_gk_influence(
    ...     frame, attacking_team_id=2, gk_player_id=1,
    ...     xt=fitted_xt, home_team_id=1,
    ... )

    See NOTICE for full bibliographic citations.
    """
    # --- Validate GK presence ---
    players = frame[~frame["is_ball"].astype(bool)]
    players = players.dropna(subset=["x", "y"])
    gk_mask = players["player_id"] == gk_player_id
    if not gk_mask.any():
        raise ValueError(
            f"gk_player_id={gk_player_id!r} not found in frame "
            f"(available: {players['player_id'].tolist()})"
        )

    gk_row = players[gk_mask].iloc[0]
    gk_pos = np.array([[float(gk_row["x"]), float(gk_row["y"])]])
    gk_vel_x = float(gk_row.get("vx", 0.0)) if pd.notna(gk_row.get("vx")) else 0.0
    gk_vel_y = float(gk_row.get("vy", 0.0)) if pd.notna(gk_row.get("vy")) else 0.0
    gk_vel = np.array([[gk_vel_x, gk_vel_y]])

    # --- Goal-end resolution ---
    defending_team_id = gk_row["team_id"]
    if defending_team_id == home_team_id:
        goal_x = 0.0  # home defends x=0
    else:
        goal_x = 105.0  # away defends x=105

    # --- Default zones ---
    if zones is None:
        zones = [Zone.six_yard_box(goal_x)]

    # --- Primitive (a): threat-weighted pitch control share ---
    surface = compute_pitch_control(
        frame, attacking_team_id=attacking_team_id,
        method=method, params=params, decompose=True,
    )

    gk_surface = surface.player_surface(gk_player_id)  # (ny, nx)

    # Team influence: sum over teammates (same team_id as GK)
    team_surface = np.zeros_like(gk_surface)
    if surface.player_ids is not None and surface.player_team_ids is not None:
        team_mask_arr = surface.player_team_ids == defending_team_id
        for idx in np.flatnonzero(team_mask_arr):
            team_surface += surface.per_player_influence[idx]

    # Per-cell share with threshold guard
    safe_team = np.where(team_surface < 1e-8, np.inf, team_surface)
    share_grid = np.where(team_surface < 1e-8, 0.0, gk_surface / safe_team)

    # Interpolate xT onto pitch control grid
    interp = xt.interpolator(kind="linear")
    threat_grid = interp(surface.grid_x, surface.grid_y)  # (ny, nx)

    # xT flip for away-team attack
    if attacking_team_id != home_team_id:
        # Away attacks toward x=0 in LTR frames
        # Defending team is home -> goal at x=0 -> high threat near x=0
        threat_grid = threat_grid[:, ::-1]

    # Weighted average
    cell_area = surface.cell_area
    threat_weight = threat_grid * cell_area
    total_weight = threat_weight.sum()

    if total_weight < 1e-8:
        pitch_control_share_weighted = float("nan")
    else:
        pitch_control_share_weighted = float(
            (share_grid * threat_weight).sum() / total_weight
        )

    # --- Primitive (b): reachable area ---
    sp = SpearmanParams() if params is None else params

    # GK TTI to all grid cells
    grid_x = surface.grid_x
    grid_y = surface.grid_y
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])

    tti_gk = compute_tti(gk_pos, gk_vel, targets, gk_reaction_time, gk_max_acceleration)
    tti_gk = tti_gk[0]  # (n_targets,)

    # Back-line defenders TTI
    back_line = select_back_line_players(
        frame, team_id=defending_team_id, home_team_id=home_team_id,
    )

    if len(back_line) > 0:
        def_pos = back_line[["x", "y"]].to_numpy(dtype="float64")
        vx_col = back_line["vx"].to_numpy(dtype="float64") if "vx" in back_line.columns else np.zeros(len(back_line))
        vy_col = back_line["vy"].to_numpy(dtype="float64") if "vy" in back_line.columns else np.zeros(len(back_line))
        def_vel = np.column_stack([np.nan_to_num(vx_col), np.nan_to_num(vy_col)])

        tti_defenders = compute_tti(
            def_pos, def_vel, targets,
            sp.reaction_time, sp.max_acceleration,
        )
        min_tti_def = tti_defenders.min(axis=0)  # (n_targets,)

        # Cells where GK can reach within tau but no defender can
        gk_reachable = tti_gk <= tau_seconds
        def_not_reachable = min_tti_def > tau_seconds
        unique_cells = gk_reachable & def_not_reachable
    else:
        unique_cells = tti_gk <= tau_seconds

    reachable_area_m2 = float(unique_cells.sum() * cell_area)

    # --- Primitive (c): zone closing times ---
    closing_times: dict[str, ZoneClosingTime] = {}
    for zone in zones:
        zone_tti = compute_tti(
            gk_pos, gk_vel, zone.points,
            gk_reaction_time, gk_max_acceleration,
        )[0]  # (n_zone_points,)
        closing_times[zone.name] = ZoneClosingTime(
            min_s=float(zone_tti.min()),
            mean_s=float(zone_tti.mean()),
        )

    return GkInfluence(
        pitch_control_share_weighted=pitch_control_share_weighted,
        reachable_area_m2=reachable_area_m2,
        closing_times=closing_times,
    )
```

- [ ] **Step 8: Run all tests**

Run: `python -m pytest tests/tracking/test_gk_influence.py -v`
Expected: All ~35 tests PASS

---

### Task 5: Action-coupled per-Series helpers + aggregator

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Create: `tests/tracking/test_gk_influence_action_coupled.py` (T-7, T-8)

- [ ] **Step 1: Write T-7 per-Series helper tests**

Create `tests/tracking/test_gk_influence_action_coupled.py`:

```python
"""Tests for TF-15 GK influence action-coupled features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._gk_test_helpers import _make_two_team_frame

# fitted_xt fixture inherited from tests/conftest.py (M-5, C-2 v2)


def _make_actions_and_frames():
    """Build a minimal action+frame pair for action-coupled testing."""
    frame = _make_two_team_frame(
        home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
        away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
    )
    actions = pd.DataFrame({
        "action_id": [0, 1],
        "game_id": [1, 1],
        "period_id": [1, 1],
        "time_seconds": [1.0, 999.0],  # second action unlinked
        "team_id": [2, 2],
        "type_id": [0, 0],
        "result_id": [1, 1],
        "start_x": [80.0, 50.0],
        "start_y": [30.0, 34.0],
        "end_x": [85.0, 55.0],
        "end_y": [35.0, 40.0],
        "bodypart_id": [0, 0],
        "player_id": [60, 61],
    })
    return actions, frame


class TestPerSeriesHelpers:
    """T-7: Per-Series helper functions."""

    def test_known_gk_produces_scalar(self, fitted_xt):
        from silly_kicks.tracking.features import gk_pitch_control_share_weighted

        actions, frames = _make_actions_and_frames()
        result = gk_pitch_control_share_weighted(
            actions, frames, fitted_xt, home_team_id=1,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        # First action linked, should have valid value
        assert not np.isnan(result.iloc[0])

    def test_unlinked_action_nan(self, fitted_xt):
        from silly_kicks.tracking.features import gk_reachable_area_m2

        actions, frames = _make_actions_and_frames()
        result = gk_reachable_area_m2(
            actions, frames, fitted_xt, home_team_id=1,
        )
        # Second action time_seconds=999 -> no matching frame -> NaN
        assert np.isnan(result.iloc[1])

    def test_nan_team_id_nan(self, fitted_xt):
        from silly_kicks.tracking.features import gk_closing_time_min_s

        actions, frames = _make_actions_and_frames()
        actions.loc[0, "team_id"] = np.nan
        result = gk_closing_time_min_s(
            actions, frames, home_team_id=1,
        )
        assert np.isnan(result.iloc[0])

    def test_introspection_all_nan(self, fitted_xt):
        from silly_kicks.tracking.features import gk_pitch_control_share_weighted

        actions, _ = _make_actions_and_frames()
        result = gk_pitch_control_share_weighted(
            actions, None, fitted_xt, home_team_id=1,
        )
        assert result.isna().all()
        assert result.name == "gk_pitch_control_share_weighted"


class TestAggregator:
    """T-8: add_gk_influence aggregator."""

    def test_correct_column_set(self, fitted_xt):
        from silly_kicks.tracking.features import add_gk_influence

        actions, frames = _make_actions_and_frames()
        result = add_gk_influence(
            actions, frames, fitted_xt, home_team_id=1,
        )
        expected_cols = {
            "gk_pitch_control_share_weighted",
            "gk_reachable_area_m2",
            "gk_closing_time_min_s__six_yard_box",
            "gk_closing_time_mean_s__six_yard_box",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_idempotent_provenance(self, fitted_xt):
        """Provenance columns skipped if already present."""
        from silly_kicks.tracking.features import add_action_context, add_gk_influence

        actions, frames = _make_actions_and_frames()
        enriched = add_action_context(actions, frames)
        result = add_gk_influence(enriched, frames, fitted_xt, home_team_id=1)
        # Should not duplicate provenance columns
        assert result.columns.duplicated().sum() == 0

    def test_nan_safe_decorator(self):
        from silly_kicks.tracking.features import add_gk_influence

        assert hasattr(add_gk_influence, "_nan_safe_enrichment")

    def test_additional_zones(self, fitted_xt):
        from silly_kicks.tracking.features import add_gk_influence

        actions, frames = _make_actions_and_frames()
        result = add_gk_influence(
            actions, frames, fitted_xt, home_team_id=1,
            zone_names=["six_yard_box", "near_post", "far_post"],
        )
        assert "gk_closing_time_min_s__near_post" in result.columns
        assert "gk_closing_time_mean_s__far_post" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_influence_action_coupled.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement batch kernel + per-Series helpers + aggregator in `features.py`**

Add to `silly_kicks/tracking/features.py` (after the DAS section, before end of file).

**H-4 fix: batch kernel computes once per unique frame, not per action.**
**C-2 fix: closing-time helpers use `compute_zone_closing_times` (no pitch control).**
**H-1 fix: all exception handlers use `warnings.warn`.**
**M-1 fix: all `is_goalkeeper` comparisons use `.astype(bool)`.**
**M-3 fix: `add_gk_influence` accepts `zone_names: list[str] | None`, constructs zones per-action with resolved `goal_x` + `ball_y`.**

```python
# ---------------------------------------------------------------------------
# TF-15 -- GK influence primitives
# ---------------------------------------------------------------------------

import warnings as _gk_warnings  # noqa: E402


def _gk_influence_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    zone_names: list[str] | None = None,
    tau_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch kernel: compute GK influence for all actions at once.

    Caches compute_gk_influence per unique (period_id, frame_id, team_id)
    to avoid redundant pitch control computation.

    Returns (result_df, pointers) — result_df aligned with actions.index
    containing gk_pitch_control_share_weighted, gk_reachable_area_m2,
    gk_closing_time_{min,mean}_s__<zone_name>; pointers from
    link_actions_to_frames for caller reuse (M-2 v2 fix).
    """
    from ._gk_influence import Zone, compute_gk_influence

    _zone_names = zone_names or ["six_yard_box"]

    # Initialize output columns
    col_names = ["gk_pitch_control_share_weighted", "gk_reachable_area_m2"]
    for zn in _zone_names:
        col_names.extend([
            f"gk_closing_time_min_s__{zn}",
            f"gk_closing_time_mean_s__{zn}",
        ])

    result = pd.DataFrame(
        {col: np.full(len(actions), np.nan) for col in col_names},
        index=actions.index,
    )

    if len(frames) == 0:
        return result, pd.DataFrame()

    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    # Cache: (period_id, frame_id, team_id) -> GkInfluence | None
    cache: dict[tuple, GkInfluence | None] = {}

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        aid = action_row["action_id"]
        tid = action_row["team_id"]
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

            gk_rows = frame_data[
                frame_data["is_goalkeeper"].astype(bool)
                & (~frame_data["is_ball"].astype(bool))
                & (frame_data["team_id"] != tid)
            ]
            if gk_rows.empty:
                cache[cache_key] = None
                continue

            gk_pid = gk_rows.iloc[0]["player_id"]
            gk_team = gk_rows.iloc[0]["team_id"]
            goal_x = 0.0 if gk_team == home_team_id else 105.0

            # Resolve ball position for near/far post zones (M-3)
            ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
            ball_y = float(ball_rows.iloc[0]["y"]) if len(ball_rows) > 0 and pd.notna(ball_rows.iloc[0]["y"]) else None

            # Build Zone instances per-action with resolved goal_x + ball_y
            zones = []
            for zn in _zone_names:
                if zn == "six_yard_box":
                    zones.append(Zone.six_yard_box(goal_x))
                elif zn == "near_post":
                    zones.append(Zone.near_post(goal_x, ball_y=ball_y))
                elif zn == "far_post":
                    zones.append(Zone.far_post(goal_x, ball_y=ball_y))
                else:
                    _gk_warnings.warn(
                        f"Unknown zone name '{zn}'; skipping",
                        UserWarning, stacklevel=2,
                    )

            try:
                gi = compute_gk_influence(
                    frame_data, attacking_team_id=tid,
                    gk_player_id=gk_pid, xt=xt,
                    home_team_id=home_team_id, method=method,
                    zones=zones, tau_seconds=tau_seconds,
                )
                cache[cache_key] = gi
            except (ValueError, KeyError) as exc:
                _gk_warnings.warn(
                    f"compute_gk_influence failed for frame=({pid},{fid}), "
                    f"team={tid}: {exc}",
                    UserWarning, stacklevel=2,
                )
                cache[cache_key] = None

        gi = cache[cache_key]
        if gi is None:
            continue

        idx = actions.index[i]
        result.at[idx, "gk_pitch_control_share_weighted"] = gi.pitch_control_share_weighted
        result.at[idx, "gk_reachable_area_m2"] = gi.reachable_area_m2
        for zn, zct in gi.closing_times.items():
            result.at[idx, f"gk_closing_time_min_s__{zn}"] = zct.min_s
            result.at[idx, f"gk_closing_time_mean_s__{zn}"] = zct.mean_s

    return result, pointers


def gk_pitch_control_share_weighted(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
) -> pd.Series:
    """Threat-weighted GK pitch control share at the linked frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_pitch_control_share_weighted
    >>> share = gk_pitch_control_share_weighted(actions, frames, xt, home_team_id=1)
    """
    col_name = "gk_pitch_control_share_weighted"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _gk_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id, method=method,
    )
    return batch[col_name].rename(col_name)


def gk_reachable_area_m2(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """GK uniquely reachable area (m^2) at the linked frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_reachable_area_m2
    >>> area = gk_reachable_area_m2(actions, frames, xt, home_team_id=1)
    """
    col_name = "gk_reachable_area_m2"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _gk_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id,
        method=method, tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


def gk_closing_time_min_s(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str,
    zone_name: str = "six_yard_box",
) -> pd.Series:
    """GK minimum closing time (seconds) to the specified zone.

    Lightweight: uses compute_zone_closing_times directly (no pitch
    control computation). See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_closing_time_min_s
    >>> ct = gk_closing_time_min_s(actions, frames, home_team_id=1)
    """
    col_name = f"gk_closing_time_min_s__{zone_name}"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    return _closing_time_per_series(
        actions, frames, home_team_id=home_team_id,
        zone_name=zone_name, extract="min_s", col_name=col_name,
    )


def gk_closing_time_mean_s(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str,
    zone_name: str = "six_yard_box",
) -> pd.Series:
    """GK mean closing time (seconds) to the specified zone.

    Lightweight: uses compute_zone_closing_times directly (no pitch
    control computation). See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_closing_time_mean_s
    >>> ct = gk_closing_time_mean_s(actions, frames, home_team_id=1)
    """
    col_name = f"gk_closing_time_mean_s__{zone_name}"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    return _closing_time_per_series(
        actions, frames, home_team_id=home_team_id,
        zone_name=zone_name, extract="mean_s", col_name=col_name,
    )


def _closing_time_per_series(
    actions, frames, *, home_team_id, zone_name, extract, col_name,
) -> pd.Series:
    """Lightweight closing-time path — calls compute_zone_closing_times directly."""
    from ._gk_influence import Zone, compute_zone_closing_times

    pointers, _ = link_actions_to_frames(actions, frames)
    results = np.full(len(actions), np.nan)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for i, (_idx, row) in enumerate(actions.iterrows()):
        aid = row["action_id"]
        tid = row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue

        pid = row["period_id"]
        fid = int(float(fid_raw))
        try:
            frame_data = frame_groups.get_group((pid, fid))
        except KeyError:
            continue

        gk_rows = frame_data[
            frame_data["is_goalkeeper"].astype(bool)
            & (~frame_data["is_ball"].astype(bool))
            & (frame_data["team_id"] != tid)
        ]
        if gk_rows.empty:
            continue
        gk_pid = gk_rows.iloc[0]["player_id"]
        gk_team = gk_rows.iloc[0]["team_id"]
        goal_x = 0.0 if gk_team == home_team_id else 105.0

        ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
        ball_y = float(ball_rows.iloc[0]["y"]) if len(ball_rows) > 0 and pd.notna(ball_rows.iloc[0]["y"]) else None

        if zone_name == "six_yard_box":
            zone = Zone.six_yard_box(goal_x)
        elif zone_name == "near_post":
            zone = Zone.near_post(goal_x, ball_y=ball_y)
        elif zone_name == "far_post":
            zone = Zone.far_post(goal_x, ball_y=ball_y)
        else:
            continue

        try:
            cts = compute_zone_closing_times(
                frame_data, gk_player_id=gk_pid, zones=[zone],
            )
            zct = cts.get(zone_name)
            if zct is not None:
                results[i] = getattr(zct, extract)
        except (ValueError, KeyError) as exc:
            _gk_warnings.warn(
                f"compute_zone_closing_times failed for action_id={aid}: {exc}",
                UserWarning, stacklevel=2,
            )

    return pd.Series(results, index=actions.index, name=col_name)


@nan_safe_enrichment
def add_gk_influence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    zone_names: list[str] | None = None,
    tau_seconds: float = 1.0,
) -> pd.DataFrame:
    """Enrich actions with GK influence columns.

    Default zone_names (["six_yard_box"]) emit 4 columns. Additional zone
    names ("near_post", "far_post") add closing-time columns. Zones are
    constructed per-action with the correct goal_x and ball_y.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_gk_influence
    >>> enriched = add_gk_influence(actions, frames, xt, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    out = actions.copy()
    # M-2 v2 fix: reuse pointers from batch kernel (avoids double link_actions_to_frames)
    batch, pointers = _gk_influence_at_actions(
        actions, frames, xt, home_team_id=home_team_id,
        method=method, zone_names=zone_names, tau_seconds=tau_seconds,
    )
    for col in batch.columns:
        out[col] = batch[col].values

    # Provenance (reuse pointers from batch kernel)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")

    return out
```

Also update `__all__` in `features.py` to include:
- `"add_gk_influence"`
- `"gk_closing_time_mean_s"`
- `"gk_closing_time_min_s"`
- `"gk_influence_xfns"` (added in Task 6)
- `"gk_pitch_control_share_weighted"`
- `"gk_reachable_area_m2"`

Add at top of features.py (in the TYPE_CHECKING block):
```python
if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/tracking/test_gk_influence_action_coupled.py -v`
Expected: 8 PASSED (T-7: 4 + T-8: 4)

---

### Task 6: xfns VAEP factory with frame-precomputation cache

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Test: `tests/tracking/test_gk_influence_action_coupled.py` (append T-9)

- [ ] **Step 1: Write T-9 xfns factory tests**

Append to `tests/tracking/test_gk_influence_action_coupled.py`:

```python
from unittest.mock import patch


class TestXfnsFactory:
    """T-9: gk_influence_xfns factory."""

    def test_returns_frame_aware_transformer(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, home_team_id=1)
        assert len(xfns) == 1
        assert hasattr(xfns[0], "_frame_aware")
        assert xfns[0]._frame_aware is True

    def test_introspection_column_names(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, home_team_id=1)
        transformer = xfns[0]

        # Create dummy states (3 slots of 2 actions)
        dummy_actions = pd.DataFrame({
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 2.0],
            "team_id": [1, 1],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "start_x": [50.0, 60.0],
            "start_y": [34.0, 34.0],
            "end_x": [55.0, 65.0],
            "end_y": [34.0, 34.0],
            "bodypart_id": [0, 0],
            "player_id": [10, 11],
        })
        states = [dummy_actions.copy() for _ in range(3)]

        result = transformer(states, None)  # frames=None -> introspection
        assert result.isna().all().all()
        # 4 cols x 3 states = 12 columns
        assert len(result.columns) == 12

    def test_full_mode_column_count(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        actions, frames = _make_actions_and_frames()
        xfns = gk_influence_xfns(fitted_xt, home_team_id=1)
        transformer = xfns[0]
        states = [actions.copy() for _ in range(3)]
        result = transformer(states, frames)
        assert len(result.columns) == 12

    def test_column_naming_convention(self, fitted_xt):
        from silly_kicks.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, home_team_id=1)
        states = [pd.DataFrame({"action_id": [0], "team_id": [1],
                                "period_id": [1], "time_seconds": [1.0],
                                "game_id": [1], "type_id": [0],
                                "result_id": [1], "start_x": [50.0],
                                "start_y": [34.0], "end_x": [55.0],
                                "end_y": [34.0], "bodypart_id": [0],
                                "player_id": [10]}) for _ in range(3)]
        result = xfns[0](states, None)
        for col in result.columns:
            assert col.endswith(("_a0", "_a1", "_a2"))

    def test_cache_avoids_redundant_calls(self, fitted_xt):
        """Frame precomputation: same frame_id shared across states -> single call."""
        from silly_kicks.tracking.features import gk_influence_xfns

        # Two actions sharing same frame
        actions = pd.DataFrame({
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 1.0],
            "team_id": [2, 2],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "start_x": [80.0, 82.0],
            "start_y": [30.0, 35.0],
            "end_x": [85.0, 87.0],
            "end_y": [35.0, 40.0],
            "bodypart_id": [0, 0],
            "player_id": [60, 61],
        })
        _, frames = _make_actions_and_frames()
        xfns = gk_influence_xfns(fitted_xt, home_team_id=1)
        states = [actions.copy() for _ in range(3)]

        call_count = [0]
        original_fn = None

        from silly_kicks.tracking import _gk_influence

        original_fn = _gk_influence.compute_gk_influence

        def counting_wrapper(*args, **kwargs):
            call_count[0] += 1
            return original_fn(*args, **kwargs)

        with patch.object(_gk_influence, "compute_gk_influence", side_effect=counting_wrapper):
            xfns[0](states, frames)

        # L-3: Both actions share same frame_id=1, team_id=2 -> exactly 1 call
        # (not 6 = 2 actions x 3 states). Cache key is (period_id, frame_id, team_id).
        assert call_count[0] == 1

    def test_different_params_no_stale_cache(self, fitted_xt):
        """Different method produces different results (no stale cache)."""
        from silly_kicks.tracking.features import gk_influence_xfns

        actions, frames = _make_actions_and_frames()
        xfns_s = gk_influence_xfns(fitted_xt, home_team_id=1, method="spearman")
        xfns_v = gk_influence_xfns(fitted_xt, home_team_id=1, method="voronoi")

        states = [actions.copy() for _ in range(3)]
        result_s = xfns_s[0](states, frames)
        result_v = xfns_v[0](states, frames)

        # Different methods -> different share values (at least for non-NaN)
        share_col = "gk_pitch_control_share_weighted_a0"
        if share_col in result_s.columns and share_col in result_v.columns:
            s_vals = result_s[share_col].dropna()
            v_vals = result_v[share_col].dropna()
            if len(s_vals) > 0 and len(v_vals) > 0:
                # They should differ (different models produce different values)
                assert not np.allclose(s_vals.values, v_vals.values, equal_nan=True)
```

- [ ] **Step 2: Implement `gk_influence_xfns` in `features.py`**

**C-1 fix:** Each game-state slot has different `action_id` values, so `link_actions_to_frames` must be called per-slot inside the loop (not once for `states[0]`). Reusing `states[0]`'s pointer lookup for `states[1]`/`states[2]` silently produces all-NaN because their `action_id` values are absent from the lookup.

**H-1 fix:** `except (ValueError, KeyError)` blocks emit `warnings.warn` before continuing.

**H-2 fix:** `xt` parameter typed as `ExpectedThreat` with `TYPE_CHECKING` guard.

Append to `silly_kicks/tracking/features.py`:

```python
def gk_influence_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    zone_names: list[str] | None = None,
    tau_seconds: float = 1.0,
) -> list:
    """Factory returning a list with one FrameAwareTransformer for GK influence.

    Default zones (six_yard_box only): 4 columns x 3 game states = 12 VAEP columns.
    With near_post + far_post: 8 columns x 3 states = 24 columns.

    The transformer precomputes compute_gk_influence per unique
    (period_id, frame_id, team_id), avoiding redundant pitch control computation
    across 3 game-state slots and repeated actions.

    Parameters
    ----------
    xt : ExpectedThreat
        Fitted xT model for threat weighting.
    home_team_id : int | str
        Home team identifier for goal-end orientation.
    method : {"spearman", "fernandez_bornn", "voronoi"}
        Pitch control model, default "spearman".
    zone_names : list[str] | None
        Zone factory names (e.g. ["six_yard_box", "near_post"]).
        Defaults to ["six_yard_box"]. Zones are constructed per-action
        with resolved goal_x + ball_y.
    tau_seconds : float
        TTI tau parameter, default 1.0.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, gk_influence_xfns
        xfns = tracking_default_xfns + gk_influence_xfns(xt, home_team_id=1)
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._gk_influence import Zone, compute_gk_influence

    resolved_zone_names = zone_names if zone_names is not None else ["six_yard_box"]

    col_names = [
        "gk_pitch_control_share_weighted",
        "gk_reachable_area_m2",
    ]
    for zn in resolved_zone_names:
        col_names.append(f"gk_closing_time_min_s__{zn}")
        col_names.append(f"gk_closing_time_mean_s__{zn}")

    def _gk_influence_transformer(states, frames):
        """Multi-column GK influence xfn with frame precomputation cache."""
        out = pd.DataFrame(index=states[0].index)

        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Shared cache across all 3 slots: (period_id, frame_id, team_id) -> GkInfluence
        cache: dict[tuple, GkInfluence | None] = {}
        frame_groups = frames.groupby(["period_id", "frame_id"])

        def _get_gi(period_id, frame_id_int, team_id):
            """M-3 v2 fix: ball_y resolved inside (avoids redundant frame lookup)."""
            key = (period_id, frame_id_int, team_id)
            if key in cache:
                return cache[key]

            try:
                frame_data = frame_groups.get_group((period_id, frame_id_int))
            except KeyError:
                cache[key] = None
                return None

            gk_rows = frame_data[
                frame_data["is_goalkeeper"].astype(bool)
                & (~frame_data["is_ball"].astype(bool))
                & (frame_data["team_id"] != team_id)
            ]
            if gk_rows.empty:
                cache[key] = None
                return None
            gk_pid = gk_rows.iloc[0]["player_id"]
            gk_team = gk_rows.iloc[0]["team_id"]
            goal_x = 0.0 if gk_team == home_team_id else 105.0

            # Resolve ball_y from frame (per-frame property, not per-action)
            ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
            ball_y = (
                float(ball_rows.iloc[0]["y"])
                if not ball_rows.empty and pd.notna(ball_rows.iloc[0]["y"])
                else 34.0
            )

            # Build zones per-action with resolved goal_x + ball_y
            action_zones = [
                getattr(Zone, zn)(goal_x, ball_y=ball_y)
                if zn in ("near_post", "far_post")
                else getattr(Zone, zn)(goal_x)
                for zn in resolved_zone_names
            ]

            try:
                gi = compute_gk_influence(
                    frame_data, attacking_team_id=team_id,
                    gk_player_id=gk_pid, xt=xt,
                    home_team_id=home_team_id, method=method,
                    zones=action_zones, tau_seconds=tau_seconds,
                )
            except (ValueError, KeyError) as exc:
                warnings.warn(
                    f"compute_gk_influence failed for frame {frame_id_int}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                gi = None

            cache[key] = gi
            return gi

        # C-1 fix: link_actions_to_frames per-slot (each slot has different action_ids)
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

                pid = row["period_id"]
                fid = int(float(fid_raw))

                gi = _get_gi(pid, fid, tid)
                if gi is None:
                    continue

                slot_results["gk_pitch_control_share_weighted"][j] = gi.pitch_control_share_weighted
                slot_results["gk_reachable_area_m2"][j] = gi.reachable_area_m2
                for zn, zct in gi.closing_times.items():
                    if f"gk_closing_time_min_s__{zn}" in slot_results:
                        slot_results[f"gk_closing_time_min_s__{zn}"][j] = zct.min_s
                        slot_results[f"gk_closing_time_mean_s__{zn}"][j] = zct.mean_s

            for col in col_names:
                out[f"{col}_a{i}"] = slot_results[col]

        return out

    _gk_influence_transformer._frame_aware = True  # type: ignore[attr-defined]
    _gk_influence_transformer.__name__ = "gk_influence"
    return [_gk_influence_transformer]
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/tracking/test_gk_influence_action_coupled.py -v`
Expected: 14 PASSED (T-7: 4 + T-8: 4 + T-9: 6)

---

### Task 7: Atomic SPADL mirror + exports

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `silly_kicks/tracking/features.py` (`__all__`)
- Test: `tests/tracking/test_gk_influence_action_coupled.py` (append T-10)

- [ ] **Step 1: Write T-10 atomic mirror tests**

Append to `tests/tracking/test_gk_influence_action_coupled.py`:

```python
class TestAtomicMirror:
    """T-10: Atomic SPADL produces same values via x/y anchor."""

    def test_atomic_share_matches_standard(self, fitted_xt):
        """Standard and atomic paths produce identical share values."""
        from silly_kicks.atomic.tracking.features import gk_pitch_control_share_weighted as atomic_share
        from silly_kicks.tracking.features import gk_pitch_control_share_weighted as std_share

        actions, frames = _make_actions_and_frames()
        # Build atomic-shaped actions (x/y instead of start_x/start_y)
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]

        std_result = std_share(actions, frames, fitted_xt, home_team_id=1)
        atomic_result = atomic_share(atomic_actions, frames, fitted_xt, home_team_id=1)
        pd.testing.assert_series_equal(std_result, atomic_result, check_names=False)

    def test_atomic_closing_time_matches(self, fitted_xt):
        from silly_kicks.atomic.tracking.features import gk_closing_time_min_s as atomic_ct
        from silly_kicks.tracking.features import gk_closing_time_min_s as std_ct

        actions, frames = _make_actions_and_frames()
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]

        std_result = std_ct(actions, frames, home_team_id=1)
        atomic_result = atomic_ct(atomic_actions, frames, home_team_id=1)
        pd.testing.assert_series_equal(std_result, atomic_result, check_names=False)

    def test_atomic_aggregator_column_set(self, fitted_xt):
        from silly_kicks.atomic.tracking.features import add_gk_influence

        actions, frames = _make_actions_and_frames()
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]

        result = add_gk_influence(atomic_actions, frames, fitted_xt, home_team_id=1)
        assert "gk_pitch_control_share_weighted" in result.columns
        assert "gk_reachable_area_m2" in result.columns

    def test_atomic_xfns_column_count(self, fitted_xt):
        from silly_kicks.atomic.tracking.features import gk_influence_xfns

        xfns = gk_influence_xfns(fitted_xt, home_team_id=1)
        assert len(xfns) == 1
        assert xfns[0]._frame_aware is True
```

- [ ] **Step 2: Add atomic mirror to `silly_kicks/atomic/tracking/features.py`**

The atomic mirror simply re-exports from standard features since GK influence
primitives don't depend on start_x/start_y vs x/y (they operate on frame data,
not action coordinates). Add imports:

```python
from silly_kicks.tracking.features import (
    add_gk_influence,
    gk_closing_time_mean_s,
    gk_closing_time_min_s,
    gk_influence_xfns,
    gk_pitch_control_share_weighted,
    gk_reachable_area_m2,
)
```

Add to atomic `__all__`:
```python
"add_gk_influence",
"gk_closing_time_mean_s",
"gk_closing_time_min_s",
"gk_influence_xfns",
"gk_pitch_control_share_weighted",
"gk_reachable_area_m2",
```

- [ ] **Step 3: Update `tracking/__init__.py` exports**

Add to imports from `.features`:
```python
add_gk_influence,
gk_closing_time_mean_s,
gk_closing_time_min_s,
gk_influence_xfns,
gk_pitch_control_share_weighted,
gk_reachable_area_m2,
```

Add to `__all__`:
```python
"add_gk_influence",
"gk_closing_time_mean_s",
"gk_closing_time_min_s",
"gk_influence_xfns",
"gk_pitch_control_share_weighted",
"gk_reachable_area_m2",
```

Also add `compute_tti` export from pitch_control:
```python
from .pitch_control import (
    ...,
    compute_tti,
)
```
And `"compute_tti"` to `__all__`.

Also add `select_back_line_players` export from `._defensive_line`:
```python
from ._defensive_line import compute_defensive_line, select_back_line_players
```
And `"select_back_line_players"` to `__all__`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/tracking/test_gk_influence_action_coupled.py -v`
Expected: 18 PASSED (T-7: 4 + T-8: 4 + T-9: 6 + T-10: 4)

---

### Task 8: Performance benchmark

**Files:**
- Create: `tests/tracking/test_gk_influence_perf_budget.py`

- [ ] **Step 1: Write T-PB1 performance benchmark**

Create `tests/tracking/test_gk_influence_perf_budget.py`:

```python
"""Performance budget for compute_gk_influence (TF-15).

Uses pytest-benchmark (H-3 fix), matching test_pressure_perf_budget.py
and pitch_control/test_perf_budget.py patterns.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

_BUDGET = 0.015 if sys.platform == "win32" else 0.010


def _make_22_player_frame():
    """Standard 22-player frame for benchmarking."""
    rows = []
    # Ball
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=0, team_id=np.nan, is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, vx=5.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # Home team: GK + 10 outfield
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=1, team_id=1, is_ball=False, is_goalkeeper=True,
        x=3.0, y=34.0, vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=10 + i, team_id=1, is_ball=False, is_goalkeeper=False,
            x=float(rng.uniform(10, 60)),
            y=float(rng.uniform(5, 63)),
            vx=float(rng.uniform(-3, 3)),
            vy=float(rng.uniform(-3, 3)),
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    # Away team: GK + 10 outfield
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=50, team_id=2, is_ball=False, is_goalkeeper=True,
        x=102.0, y=34.0, vx=0.0, vy=0.0,
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    for i in range(10):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=60 + i, team_id=2, is_ball=False, is_goalkeeper=False,
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


def test_compute_gk_influence_perf_budget(benchmark, fixture_22):
    """compute_gk_influence on 22-player frame within budget."""
    from silly_kicks.tracking._gk_influence import compute_gk_influence

    frame, xt = fixture_22

    result = benchmark(
        compute_gk_influence,
        frame, attacking_team_id=2, gk_player_id=1,
        xt=xt, home_team_id=1,
    )
    assert result is not None
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"compute_gk_influence mean {benchmark.stats.stats.mean*1000:.1f}ms "
            f"> budget {_BUDGET*1000:.0f}ms"
        )
```

- [ ] **Step 2: Run benchmark**

Run: `python -m pytest tests/tracking/test_gk_influence_perf_budget.py -v --benchmark-disable`
Expected: 1 PASSED (benchmark stats not collected with --benchmark-disable, assertion on mean skipped)

For full benchmark: `python -m pytest tests/tracking/test_gk_influence_perf_budget.py -v`

---

### Task 9: Bundled TF-31/TF-32 fixes

**Files:**
- Modify: `silly_kicks/tracking/_line_breaking.py` (H1, H2, H3, M4)
- Modify: `silly_kicks/tracking/_team_shape.py` (M2, M3)
- Create: `tests/tracking/test_line_breaking_bundled_fixes.py`
- Create: `tests/tracking/test_team_shape_perf_budget.py` (M1)
- Create: `tests/tracking/test_line_breaking_perf_budget.py` (M1)

- [ ] **Step 1: Write H1/H2/M4 failing tests**

Create `tests/tracking/test_line_breaking_bundled_fixes.py`:

```python
"""Bundled fixes for TF-31/TF-32 (National Park Principle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_line_breaking_fixture(
    *,
    action_team_id: int = 1,
    opp_positions: list[tuple[float, float]],
    start_xy: tuple[float, float] = (40.0, 34.0),
    end_xy: tuple[float, float] = (70.0, 34.0),
    home_team_id: int = 1,
    action_type_id: int = 0,  # 0 = pass in SPADL
):
    """Build minimal action + frame for line-breaking testing."""
    from silly_kicks.spadl import config as spadlconfig

    rows = []
    # Ball
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=0, team_id=np.nan, is_ball=True, is_goalkeeper=False,
        x=start_xy[0], y=start_xy[1],
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    # Opponents
    opp_team = 2 if action_team_id == 1 else 1
    for i, (ox, oy) in enumerate(opp_positions):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=50 + i, team_id=opp_team, is_ball=False, is_goalkeeper=False,
            x=ox, y=oy,
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    # Action-team player (passer)
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=10, team_id=action_team_id, is_ball=False, is_goalkeeper=False,
        x=start_xy[0], y=start_xy[1],
        source_provider="synthetic", team_attacking_direction="ltr",
    ))
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame({
        "action_id": [0],
        "game_id": [1],
        "period_id": [1],
        "time_seconds": [1.0],
        "team_id": [action_team_id],
        "type_id": [action_type_id],
        "result_id": [1],
        "start_x": [start_xy[0]],
        "start_y": [start_xy[1]],
        "end_x": [end_xy[0]],
        "end_y": [end_xy[1]],
        "bodypart_id": [0],
        "player_id": [10],
    })
    return actions, frames


class TestH1DropnaMisalignment:
    """H1: Joint dropna for opponent x/y prevents misalignment."""

    def test_partial_nan_no_crash(self):
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        # One opponent has valid x but NaN y
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[(50.0, 20.0), (55.0, 30.0), (60.0, 40.0)],
        )
        # Inject NaN y on one opponent
        frames.loc[frames["player_id"] == 52, "y"] = np.nan
        result = detect_line_breaking(actions, frames, home_team_id=1)
        # Should not crash
        assert len(result) == 1


class TestH2ExtensionPoisoning:
    """H2: between_lines dominates when both extension + through intersect."""

    def test_between_lines_dominates(self):
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        # Setup: opponents arranged so pass intersects BOTH an extension
        # segment AND a between-players segment of the same cluster.
        # Arrange 6 opponents in 2 clusters (x ~50 and x ~70):
        # Cluster 1 at x~50: players at (50,10), (50,30), (50,50)
        # Cluster 2 at x~70: players at (70,10), (70,30), (70,50)
        # Pass from (40,34) to (80,34) goes through both clusters.
        # The pass trajectory y=34 goes between y=30 and y=50 (between_lines)
        # AND may hit extension (sideline segment at y=0 or y=68)
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[
                (50, 10), (50, 30), (50, 50),
                (70, 10), (70, 30), (70, 50),
            ],
            start_xy=(40.0, 34.0),
            end_xy=(80.0, 34.0),
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        # When both extension AND between-players segments intersect,
        # type should be "between_lines" (not "around_line")
        if result["line_break__ward"].iloc[0]:
            assert result["line_breaking_type__ward"].iloc[0] == "between_lines"


class TestM4NonPassFiltering:
    """M4: Non-pass actions produce pd.NA."""

    def test_shot_produces_na(self):
        from silly_kicks.spadl import config as spadlconfig
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        shot_type_id = spadlconfig.actiontype_id["shot"]
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[(50, 20), (55, 30), (60, 40)],
            action_type_id=shot_type_id,
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        assert pd.isna(result["line_break__ward"].iloc[0])

    def test_dribble_produces_na(self):
        from silly_kicks.spadl import config as spadlconfig
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        dribble_type_id = spadlconfig.actiontype_id["dribble"]
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[(50, 20), (55, 30), (60, 40)],
            action_type_id=dribble_type_id,
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        assert pd.isna(result["line_break__ward"].iloc[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_line_breaking_bundled_fixes.py -v`
Expected: H1 and M4 tests FAIL (H2 may fail or pass depending on current behavior)

- [ ] **Step 3: Fix H1 — joint dropna in `_line_breaking.py`**

Replace lines 178-179:
```python
        opp_x = opp_df["x"].dropna().to_numpy(dtype="float64")
        opp_y = opp_df["y"].dropna().to_numpy(dtype="float64")
```
With:
```python
        valid_mask = opp_df["x"].notna() & opp_df["y"].notna()
        valid_opp = opp_df[valid_mask]
        opp_x = valid_opp["x"].to_numpy(dtype="float64")
        opp_y = valid_opp["y"].to_numpy(dtype="float64")
```

- [ ] **Step 4: Fix H2 — track `cluster_has_through` independently**

Replace the segment-testing block (lines 240-266):
```python
            cluster_broken = False
            broke_on_extension = False
            n_segments = len(points_x) - 1

            for si in range(n_segments):
                ...
                if _segments_intersect(...):
                    cluster_broken = True
                    if si == 0 or si == n_segments - 1:
                        broke_on_extension = True

            if cluster_broken:
                lines_broken += 1
                if not broke_on_extension:
                    any_through = True
```
With:
```python
            cluster_broken = False
            cluster_has_through = False
            n_segments = len(points_x) - 1

            for si in range(n_segments):
                ax, ay = points_x[si], points_y[si]
                bx, by = points_x[si + 1], points_y[si + 1]

                if _segments_intersect(
                    track_start_x, track_start_y,
                    track_end_x, track_end_y,
                    ax, ay, bx, by,
                ):
                    cluster_broken = True
                    # Extension segments are first and last
                    if si != 0 and si != n_segments - 1:
                        cluster_has_through = True

            if cluster_broken:
                lines_broken += 1
                if cluster_has_through:
                    any_through = True
```

- [ ] **Step 5: Fix M4 — action-type filter**

Add module-level constant (L-2 fix: compute once, not per-call):

```python
from silly_kicks.spadl import config as spadlconfig

_PASS_CROSS_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[n]
    for n in ("pass", "cross")
    if n in spadlconfig.actiontype_id
)
```

Then in the loop:
```python
        action_type = row.get("type_id")
        if pd.notna(action_type) and int(action_type) not in _PASS_CROSS_TYPE_IDS:
            continue  # Non-pass/cross -> leave as pd.NA
```

This requires `actions` to have a `type_id` column. Add it to the merge:
```python
    linked = linked.merge(
        actions[["action_id", "team_id", "start_x", "start_y",
                 "end_x", "end_y", "period_id", "game_id", "type_id"]],
        on="action_id", how="left",
    )
```

- [ ] **Step 6: Add H3 algorithm divergence docstring**

Add to `_line_breaking.py` module docstring (after the existing note):
```python
Deviations from reference: Karakus & Arkadas (2025) use centroid +
vertical-span intersection test. This implementation uses polyline +
cross-product straddle test, which captures actual defensive geometry
(player positions form the line segments, not cluster centroids). The
straddle test is more geometrically precise and handles non-vertical
lines correctly.
```

- [ ] **Step 7: Add M2/M3 docstring clarifications to `_team_shape.py`**

Add to `compute_team_shape` docstring:
```
Frames with zero visible outfield players are omitted from output
(consumers should LEFT JOIN and fill NaN).
```

Add to `_line_breaking.py` `LineBreakingParams` docstring for `n_clusters`:
```
n_clusters is a design choice (defense/midfield/attack partition),
not from the reference paper. Configurable here.
```

Add L1 note to `_line_breaking.py` module docstring or `detect_line_breaking` docstring:
```
Out-of-scope paper metrics (Karakus & Arkadas 2025): SBR (Successful
Ball Recovery), LBPCh1 (Line-Breaking Pass Chance 1st-half),
LBPCh2 (Line-Breaking Pass Chance 2nd-half). These are game-level
aggregates computed from the per-pass detection output and are not
implemented here.
```

- [ ] **Step 8: Write M1 perf benchmark tests (pytest-benchmark, H-3 pattern)**

Create `tests/tracking/test_team_shape_perf_budget.py`:
```python
"""Performance budget for compute_team_shape (TF-31)."""
import sys

import numpy as np
import pandas as pd
import pytest

_BUDGET = 0.0015 if sys.platform == "win32" else 0.001


@pytest.fixture
def team_shape_frame():
    rows = []
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=i + 10, team_id=1, is_ball=False, is_goalkeeper=False,
            x=float(rng.uniform(10, 90)),
            y=float(rng.uniform(5, 63)),
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=0, team_id=np.nan, is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, source_provider="synthetic", team_attacking_direction="ltr",
    ))
    return pd.DataFrame(rows)


def test_team_shape_perf_budget(benchmark, team_shape_frame):
    from silly_kicks.tracking._team_shape import compute_team_shape

    result = benchmark(compute_team_shape, team_shape_frame, team_id=1)
    assert result is not None
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"compute_team_shape: {benchmark.stats.stats.mean*1000:.1f}ms > {_BUDGET*1000:.0f}ms"
        )
```

Create `tests/tracking/test_line_breaking_perf_budget.py`:
```python
"""Performance budget for detect_line_breaking (TF-32)."""
import sys

import numpy as np
import pandas as pd
import pytest

_BUDGET = 0.003 if sys.platform == "win32" else 0.002


@pytest.fixture
def line_breaking_fixture():
    rows = []
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=0, team_id=np.nan, is_ball=True, is_goalkeeper=False,
        x=50.0, y=34.0, source_provider="synthetic", team_attacking_direction="ltr",
    ))
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=50 + i, team_id=2, is_ball=False, is_goalkeeper=False,
            x=float(rng.uniform(40, 80)), y=float(rng.uniform(5, 63)),
            source_provider="synthetic", team_attacking_direction="ltr",
        ))
    rows.append(dict(
        game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
        player_id=10, team_id=1, is_ball=False, is_goalkeeper=False,
        x=40.0, y=34.0, source_provider="synthetic", team_attacking_direction="ltr",
    ))
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame({
        "action_id": [0], "game_id": [1], "period_id": [1],
        "time_seconds": [1.0], "team_id": [1], "type_id": [0],
        "result_id": [1], "start_x": [40.0], "start_y": [34.0],
        "end_x": [70.0], "end_y": [34.0], "bodypart_id": [0], "player_id": [10],
    })
    return actions, frames


def test_line_breaking_perf_budget(benchmark, line_breaking_fixture):
    from silly_kicks.tracking._line_breaking import detect_line_breaking

    actions, frames = line_breaking_fixture
    result = benchmark(detect_line_breaking, actions, frames, home_team_id=1)
    assert result is not None
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"detect_line_breaking: {benchmark.stats.stats.mean*1000:.1f}ms > {_BUDGET*1000:.0f}ms"
        )
```

- [ ] **Step 9: Run all bundled fix tests**

Run: `python -m pytest tests/tracking/test_line_breaking_bundled_fixes.py tests/tracking/test_team_shape_perf_budget.py tests/tracking/test_line_breaking_perf_budget.py -v`
Expected: All PASS

Also run existing line-breaking tests for regression:
Run: `python -m pytest tests/tracking/ -k "line_breaking" -v`
Expected: All existing tests PASS

---

### Task 10: Physical invariant tests + provider sweep invariants

**Files:**
- Create: `tests/invariants/test_gk_influence_invariants.py`
- Modify: `tests/invariants/test_invariant_line_breaking.py` (L2)
- Modify: `tests/invariants/test_invariant_team_shape.py` (L2)

- [ ] **Step 0: Move `fitted_xt` fixture to `tests/conftest.py` (M-5, C-2 v2 fix)**

Add to `tests/conftest.py` (the root test conftest — visible to both `tests/tracking/` and `tests/invariants/`):

```python
import numpy as np
import pytest


@pytest.fixture
def fitted_xt():
    """Shared fitted ExpectedThreat fixture for all test subdirectories."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt
```

**Important:** `tests/tracking/conftest.py` is NOT visible to `tests/invariants/`. The fixture must live in `tests/conftest.py` (parent of both directories).

Remove any duplicate `fitted_xt` fixtures from `test_gk_influence.py`, `test_gk_influence_action_coupled.py`, and `test_gk_influence_perf_budget.py` — all inherit from `tests/conftest.py`.

- [ ] **Step 1: Write T-12 physical invariant tests**

Create `tests/invariants/test_gk_influence_invariants.py`:

```python
"""Physical invariants for GK influence primitives (TF-15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._gk_test_helpers import _make_two_team_frame


# fitted_xt inherited from tests/conftest.py (C-2 v2 fix)


@pytest.fixture(params=["spearman", "fernandez_bornn", "voronoi"])
def method(request):
    return request.param


class TestGkInfluenceInvariants:
    """Physical invariants that must hold across all configurations."""

    def test_share_in_unit_interval(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, method=method,
        )
        if not np.isnan(gi.pitch_control_share_weighted):
            assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_reachable_area_bounded(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, method=method,
        )
        assert 0.0 <= gi.reachable_area_m2 <= 7140.0

    def test_min_leq_mean_closing_time(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, method=method,
        )
        for zct in gi.closing_times.values():
            assert zct.min_s <= zct.mean_s

    def test_closing_time_non_negative(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1, method=method,
        )
        for zct in gi.closing_times.values():
            assert zct.min_s >= 0.0

    def test_closer_gk_lower_closing_time(self, fitted_xt):
        """GK closer to zone -> lower closing time (monotonicity)."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        outfield = [(20, 20), (25, 30), (30, 40), (35, 50)]
        away_pos = [(80, 20), (85, 30), (80, 40), (85, 50)]

        frame_close = _make_two_team_frame(
            home_positions=outfield, away_positions=away_pos,
            home_gk_pos=(3.0, 34.0),
        )
        frame_far = _make_two_team_frame(
            home_positions=outfield, away_positions=away_pos,
            home_gk_pos=(30.0, 34.0),
        )
        gi_close = compute_gk_influence(
            frame_close, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        gi_far = compute_gk_influence(
            frame_far, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, home_team_id=1,
        )
        assert (
            gi_close.closing_times["six_yard_box"].min_s
            < gi_far.closing_times["six_yard_box"].min_s
        )

    def test_string_typed_ids(self, fitted_xt):
        """M-2: String-typed team_id/player_id (DFL-OBJ-* style) must work."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        # Convert IDs to strings (Sportec/kloppy DFL-OBJ-* style)
        frame["team_id"] = frame["team_id"].astype(str)
        frame["player_id"] = frame["player_id"].astype(str)
        frame.loc[frame["is_ball"], "team_id"] = pd.NA

        gi = compute_gk_influence(
            frame, attacking_team_id="2", gk_player_id="1",
            xt=fitted_xt, home_team_id="1",
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0
        assert gi.reachable_area_m2 >= 0.0
```

- [ ] **Step 2: Add L2 provider sweep invariants**

Add to existing `tests/invariants/test_invariant_line_breaking.py`:
```python
def test_lines_broken_in_valid_range(line_breaking_result):
    """lines_broken__ward in {0, 1, 2, 3} (bounded by n_clusters)."""
    valid = line_breaking_result["lines_broken__ward"].dropna()
    if len(valid) > 0:
        assert valid.min() >= 0
        assert valid.max() <= 3
```

Add to existing `tests/invariants/test_invariant_team_shape.py`:
```python
def test_convex_hull_area_non_negative(team_shape_result):
    valid = team_shape_result["convex_hull_area"].dropna()
    if len(valid) > 0:
        assert (valid >= 0).all()

def test_stretch_index_non_negative(team_shape_result):
    valid = team_shape_result["stretch_index"].dropna()
    if len(valid) > 0:
        assert (valid >= 0).all()

def test_n_outfield_positive(team_shape_result):
    valid = team_shape_result["n_outfield_players"].dropna()
    if len(valid) > 0:
        assert (valid > 0).all()
```

The exact fixtures/parametrization will depend on the existing structure of those invariant test files — read them first, then follow the same pattern.

- [ ] **Step 3: Run invariant tests**

Run: `python -m pytest tests/invariants/test_gk_influence_invariants.py -v`
Expected: ~15 PASSED (5 tests x 3 methods)

---

### Task 11: Provider e2e tests

**Files:**
- Create: `tests/tracking/test_gk_influence_e2e.py`

- [ ] **Step 1: Write T-11 e2e tests**

Create `tests/tracking/test_gk_influence_e2e.py`:

```python
"""Provider e2e tests for GK influence primitives (TF-15).

Tests are @pytest.mark.e2e, excluded from regular suite by ``-m "not e2e"``.
Each test body uses pytest.skip() with a reason string so running with
``-m e2e`` shows them as SKIPPED (not falsely green). Implementor replaces
skip calls with actual loaders. StatsBomb correctly absent (no tracking data).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.e2e
class TestGkInfluenceE2e:
    """Per-provider e2e: all 3 primitives produce non-NaN on real data."""

    @pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
    def test_provider_all_primitives_non_nan(self, provider, fitted_xt):
        """All 3 primitives produce at least 1 non-NaN per provider."""
        pytest.skip(
            f"e2e dataset fixture for {provider} not yet wired — "
            "implementor: load actions + frames using existing e2e loader, "
            "call add_gk_influence, assert n_valid >= 1 per output column"
        )

    @pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
    def test_provider_physical_invariants(self, provider, fitted_xt):
        """Physical invariants hold on real data."""
        pytest.skip(
            f"e2e dataset fixture for {provider} not yet wired — "
            "implementor: load actions + frames, compute GK influence, "
            "assert share in [0,1], area >= 0, min <= mean closing time"
        )
```

**Note:** These tests are marked `@pytest.mark.e2e` and require dataset fixtures not committed to the repo. They are excluded from the regular suite by `-m "not e2e"`. Bodies use `pytest.skip()` with a descriptive reason (not empty bodies that silently pass). Implementor replaces `pytest.skip` calls with actual loaders using existing e2e patterns.

- [ ] **Step 2: Verify e2e tests are skipped in regular suite**

Run: `python -m pytest tests/tracking/test_gk_influence_e2e.py -v -m "not e2e"`
Expected: 0 tests collected (all skipped by mark filter)

---

### Task 12: NOTICE + TODO + CHANGELOG + docs

**Files:**
- Modify: `NOTICE`
- Modify: `TODO.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add NOTICE entries**

Append to the "Mathematical / Methodological References" section:

```
The GK influence primitives (silly_kicks/tracking/_gk_influence.py, TF-15,
GKDV Layer 1) implement threat-weighted pitch control decomposition using:
- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics
  Conference. (pitch control foundation, TTI kinematic model)
- Fernandez, J., & Bornn, L. (2018). "Wide Open Spaces: A Statistical
  Technique for Measuring Space Creation in Professional Soccer." MIT Sloan
  SAC. (alternate pitch control formulation)
- Singh, K. (2018). "Introducing Expected Threat (xT)." karun.in/blog/
  expected-threat (threat surface for weighting)
The Get Goalside critique of raw pitch control GK over-crediting motivated
the threat-weighting correction.
```

- [ ] **Step 2: Update TODO.md**

Delete the TF-15 row. Bump header date.

- [ ] **Step 3: Update CHANGELOG.md**

Add under `## [Unreleased]` (or new version heading):

```markdown
### Added
- **TF-15: GK influence primitives** (GKDV Layer 1):
  - `compute_gk_influence()` per-frame entry point with 3 primitives:
    threat-weighted pitch control share, uniquely reachable area, zone closing time
  - `Zone` dataclass with `six_yard_box()`, `near_post()`, `far_post()` factories
  - `GkInfluence` + `ZoneClosingTime` frozen return dataclasses
  - GK-specific kinematic parameters (`gk_reaction_time`, `gk_max_acceleration`)
  - Action-coupled: `add_gk_influence`, `gk_influence_xfns`, 4 per-Series helpers
  - Atomic SPADL mirror
  - Frame-precomputation cache in xfns factory (~3x speedup)
- **Prerequisite: `compute_tti`** exported as public API from `pitch_control`
- **Prerequisite: `select_back_line_players`** extracted from `_defensive_line.py`

### Fixed
- **TF-32 H1:** Independent dropna misalignment in `_line_breaking.py` (joint
  dropna prevents silent data corruption when opponent has valid x but NaN y)
- **TF-32 H2:** Extension-poisoning on `line_breaking_type` — `between_lines`
  now correctly dominates when both extension and through-player intersections
  occur in the same cluster
- **TF-32 M4:** Non-pass actions (shots, dribbles, etc.) now correctly produce
  pd.NA instead of being analyzed for line-breaking
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: All tests PASS, no regressions

---

### Task 13: Final review + commit

- [ ] **Step 1: Run pre-commit quality gates**

```bash
ruff check silly_kicks/ tests/
ruff format --check silly_kicks/ tests/
python -m pytest tests/ -m "not e2e" -v --tb=short
```

All must pass.

- [ ] **Step 2: Run `/final-review`**

Invoke the `mad-scientist-skills:final-review` skill. Follow all phases including C4 architecture diagram update.

- [ ] **Step 3: Commit**

Single commit per branch policy:
```bash
git checkout -b pr-s34-tf15-gk-influence-primitives
git add -A
git commit -m "feat(tracking): TF-15 GK influence primitives -- silly-kicks 3.10.0 (PR-S34)"
```

---

## Test Count Summary

| Category | Tests |
|----------|-------|
| T-PR1 compute_tti export | 2 |
| T-PR2 select_back_line_players | 3 |
| T-1 Zone geometry | 5 |
| T-2 Core logic | 6 |
| T-3 Method dispatch | 3 (parametrized) |
| T-4 Zone-parameterized | 3 + parametrized |
| T-5 xT interpolation + orientation | 6 |
| T-6 Edge cases | 7 |
| T-7 Per-Series helpers | 4 |
| T-8 Aggregator | 4 |
| T-9 xfns factory | 6 |
| T-10 Atomic mirror | 4 |
| T-11 Provider e2e | 4 (parametrized, e2e-marked) |
| T-PB1 Perf benchmark | 1 |
| T-12 Physical invariants | 5 x 3 methods = 15 |
| Bundled: H1/H2/M4 | 4 |
| Bundled: M1 perf | 2 |
| Bundled: L2 invariants | ~5 |
| **Total** | **~84** |
