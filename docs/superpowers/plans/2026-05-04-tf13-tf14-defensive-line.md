# PR-S27: TF-13 + TF-14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship frame-based defending-GK resolution (TF-13) and per-frame defensive-line geometry with action-coupled VAEP features (TF-14).

**Architecture:** Two new private modules (`_gk_resolve.py`, `_defensive_line.py`) provide the core algorithms. `features.py` extends with action-coupled public API (6 per-Series functions + aggregator + xfn factory). A batch kernel in `_kernels.py` bridges the per-frame compute to action-level dispatch efficiently (3 calls, not 18).

**Tech Stack:** pandas, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md`

---

## Task 1: TF-13 — Tests for `defending_gk_from_frames`

**Files:**
- Create: `tests/tracking/test_gk_resolve.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for silly_kicks.tracking._gk_resolve.defending_gk_from_frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_frames(
    *,
    home_team_id=1,
    away_team_id=2,
    gk_player_id_home=100,
    gk_player_id_away=200,
    frame_id=10,
    period_id=1,
    time_seconds=5.0,
    include_away_gk=True,
):
    """Minimal 1-frame tracking fixture with both teams + GKs."""
    rows = [
        # Ball
        dict(game_id=1, period_id=period_id, frame_id=frame_id, time_seconds=time_seconds,
             frame_rate=25.0, player_id=np.nan, team_id=np.nan, is_ball=True,
             is_goalkeeper=False, x=50.0, y=34.0, source_provider="sportec",
             team_attacking_direction="ltr"),
        # Home GK
        dict(game_id=1, period_id=period_id, frame_id=frame_id, time_seconds=time_seconds,
             frame_rate=25.0, player_id=gk_player_id_home, team_id=home_team_id,
             is_ball=False, is_goalkeeper=True, x=5.0, y=34.0,
             source_provider="sportec", team_attacking_direction="ltr"),
        # Home outfield
        dict(game_id=1, period_id=period_id, frame_id=frame_id, time_seconds=time_seconds,
             frame_rate=25.0, player_id=101, team_id=home_team_id,
             is_ball=False, is_goalkeeper=False, x=40.0, y=30.0,
             source_provider="sportec", team_attacking_direction="ltr"),
        # Away outfield
        dict(game_id=1, period_id=period_id, frame_id=frame_id, time_seconds=time_seconds,
             frame_rate=25.0, player_id=201, team_id=away_team_id,
             is_ball=False, is_goalkeeper=False, x=60.0, y=34.0,
             source_provider="sportec", team_attacking_direction="ltr"),
    ]
    if include_away_gk:
        rows.append(
            dict(game_id=1, period_id=period_id, frame_id=frame_id, time_seconds=time_seconds,
                 frame_rate=25.0, player_id=gk_player_id_away, team_id=away_team_id,
                 is_ball=False, is_goalkeeper=True, x=100.0, y=34.0,
                 source_provider="sportec", team_attacking_direction="ltr"),
        )
    return pd.DataFrame(rows)


def _make_actions(team_id=1, time_seconds=5.0, period_id=1):
    """Single-action DataFrame."""
    return pd.DataFrame({
        "action_id": [1],
        "period_id": [period_id],
        "time_seconds": [time_seconds],
        "team_id": [team_id],
        "player_id": [101],
        "start_x": [40.0],
        "start_y": [30.0],
        "type_id": [0],  # pass
    })


class TestDefendingGkFromFrames:
    def test_resolves_opposing_gk(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)  # home team acts → should get away GK
        result = defending_gk_from_frames(actions, frames)
        assert result.iloc[0] == 200

    def test_all_actions_not_just_shots(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        # 3 actions: pass, dribble, tackle — all should resolve
        actions = pd.DataFrame({
            "action_id": [1, 2, 3],
            "period_id": [1, 1, 1],
            "time_seconds": [5.0, 5.0, 5.0],
            "team_id": [1, 1, 2],
            "player_id": [101, 101, 201],
            "start_x": [40.0, 40.0, 60.0],
            "start_y": [30.0, 30.0, 34.0],
            "type_id": [0, 5, 8],
        })
        result = defending_gk_from_frames(actions, frames)
        assert result.iloc[0] == 200  # home acts → away GK
        assert result.iloc[1] == 200
        assert result.iloc[2] == 100  # away acts → home GK

    def test_nan_when_no_gk_in_frame(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(include_away_gk=False)
        actions = _make_actions(team_id=1)  # wants away GK, but none exists
        result = defending_gk_from_frames(actions, frames)
        assert pd.isna(result.iloc[0])

    def test_nan_when_unlinked(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(time_seconds=100.0)  # far from action time
        actions = _make_actions(time_seconds=5.0)
        result = defending_gk_from_frames(actions, frames, tolerance_seconds=0.2)
        assert pd.isna(result.iloc[0])

    def test_nan_when_team_id_nan(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions()
        actions["team_id"] = pd.array([pd.NA], dtype="Int64")
        result = defending_gk_from_frames(actions, frames)
        assert pd.isna(result.iloc[0])

    def test_dtype_matches_frames_object(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(gk_player_id_away="DFL-OBJ-200", away_team_id="team_b",
                              home_team_id="team_a", gk_player_id_home="DFL-OBJ-100")
        # Fix outfield player IDs to strings too
        frames["player_id"] = frames["player_id"].astype(object)
        frames["team_id"] = frames["team_id"].astype(object)
        actions = _make_actions(team_id="team_a")
        actions["player_id"] = actions["player_id"].astype(object)
        actions["team_id"] = actions["team_id"].astype(object)
        result = defending_gk_from_frames(actions, frames)
        assert result.dtype == object
        assert result.iloc[0] == "DFL-OBJ-200"

    def test_multi_gk_deterministic(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(gk_player_id_away=200)
        # Add second GK on away team (substitution overlap)
        extra = pd.DataFrame([dict(
            game_id=1, period_id=1, frame_id=10, time_seconds=5.0,
            frame_rate=25.0, player_id=199, team_id=2,
            is_ball=False, is_goalkeeper=True, x=100.0, y=34.0,
            source_provider="sportec", team_attacking_direction="ltr",
        )])
        frames = pd.concat([frames, extra], ignore_index=True)
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        # Lowest player_id wins
        assert result.iloc[0] == 199

    def test_tolerance_respected(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(time_seconds=5.3)  # 0.3s offset
        actions = _make_actions(time_seconds=5.0)
        # tolerance=0.2 should miss
        result = defending_gk_from_frames(actions, frames, tolerance_seconds=0.2)
        assert pd.isna(result.iloc[0])
        # tolerance=0.5 should hit
        result = defending_gk_from_frames(actions, frames, tolerance_seconds=0.5)
        assert result.iloc[0] == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_resolve.py -v --tb=short 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._gk_resolve'`

---

## Task 2: TF-13 — Implement `defending_gk_from_frames`

**Files:**
- Create: `silly_kicks/tracking/_gk_resolve.py`
- Modify: `silly_kicks/tracking/features.py` (add to `__all__` + import)
- Modify: `silly_kicks/tracking/__init__.py` (add to `__all__` + import)

- [ ] **Step 3: Implement the module**

```python
"""Frame-based defending-GK resolution (TF-13).

Resolves the defending team's goalkeeper player_id from tracking frames
for every action. Standalone composable utility — callers use for fillna
on events-based defending_gk_player_id or as direct lookup.

See spec: docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md §2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import link_actions_to_frames


def defending_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action defending-GK player_id resolved from tracking frames.

    For each action, links to the nearest frame (within tolerance), finds
    the opposing team's is_goalkeeper=True row, and returns that player_id.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype
        (object for kloppy/sportec, int64/Int64 for PFF).
        NaN where action couldn't link, no opposing-team GK in linked frame,
        or action.team_id is NaN.

    Examples
    --------
    Fill NaN from events-based GK resolution::

        from silly_kicks.tracking.features import defending_gk_from_frames
        gk_series = defending_gk_from_frames(actions, frames)
        actions["defending_gk_player_id"] = (
            actions["defending_gk_player_id"].fillna(gk_series)
        )

    See NOTICE for full bibliographic citations.
    """
    # Determine output dtype from frames' player_id
    pid_dtype = frames["player_id"].dtype

    n = len(actions)
    out = pd.Series(np.full(n, np.nan), index=actions.index, dtype="object")

    if n == 0 or len(frames) == 0:
        return out

    pointers, _report = link_actions_to_frames(actions, frames, tolerance_seconds=tolerance_seconds)

    # Build lookup: for each (period_id, frame_id), the GK player_id per team
    gk_rows = frames[(frames["is_goalkeeper"] == True) & (~frames["is_ball"])].copy()  # noqa: E712
    if gk_rows.empty:
        return out

    # Join pointers with actions to get action team_id + period_id
    ptr = pointers.merge(
        actions[["action_id", "team_id", "period_id"]],
        on="action_id",
        how="left",
    )
    # Filter to linked actions only
    linked = ptr[ptr["frame_id"].notna()].copy()
    if linked.empty:
        return out

    # Join with GK rows on (period_id, frame_id) to find GKs in linked frame
    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    gk_in_frame = linked.merge(
        gk_rows[["period_id", "frame_id", "team_id", "player_id"]].rename(
            columns={"team_id": "gk_team_id", "player_id": "gk_player_id"}
        ),
        left_on=["period_id", "frame_id_int"],
        right_on=["period_id", "frame_id"],
        how="inner",
    )

    # Filter to opposing team's GK (gk_team_id != action team_id)
    # Handle NaN team_id on actions: comparison with NaN is False → filtered out
    opposing = gk_in_frame[gk_in_frame["gk_team_id"] != gk_in_frame["team_id"]]

    if opposing.empty:
        return out

    # Deterministic tiebreak: lowest player_id per action
    opposing_sorted = opposing.sort_values("gk_player_id")
    best = opposing_sorted.drop_duplicates("action_id", keep="first")

    # Map back to actions index
    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in best.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = row["gk_player_id"]

    # Cast to match frames dtype if numeric
    if pid_dtype == np.dtype("int64") or str(pid_dtype) == "Int64":
        out = pd.to_numeric(out, errors="coerce")
        if str(pid_dtype) == "Int64":
            out = out.astype("Int64")

    return out
```

- [ ] **Step 4: Add to features.py `__all__` and import**

In `silly_kicks/tracking/features.py`, add to `__all__` list:
```python
"defending_gk_from_frames",
```

And add import at top of file (after existing `from .` imports):
```python
from ._gk_resolve import defending_gk_from_frames
```

- [ ] **Step 5: Add to `__init__.py` exports**

In `silly_kicks/tracking/__init__.py`, add `"defending_gk_from_frames"` to `__all__` list and add to the `from .features import (...)` block.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_gk_resolve.py -v --tb=short`
Expected: 9 PASSED

---

## Task 3: TF-14 — Tests for `compute_defensive_line` (per-frame kernel)

**Files:**
- Create: `tests/tracking/test_defensive_line.py`

- [ ] **Step 7: Write failing tests for the per-frame kernel**

```python
"""Tests for silly_kicks.tracking._defensive_line.compute_defensive_line."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_frame_rows(
    *,
    home_team_id=1,
    away_team_id=2,
    home_outfield_xs: list[float],
    home_outfield_ys: list[float],
    away_outfield_xs: list[float],
    away_outfield_ys: list[float],
    home_gk_pos=(3.0, 34.0),
    away_gk_pos=(102.0, 34.0),
    frame_id=1,
    period_id=1,
    time_seconds=1.0,
):
    """Build a single-frame fixture with specified outfield positions."""
    rows = []
    pid = 1
    # Ball
    rows.append(dict(
        game_id=1, period_id=period_id, frame_id=frame_id,
        time_seconds=time_seconds, frame_rate=25.0,
        player_id=np.nan, team_id=np.nan, is_ball=True,
        is_goalkeeper=False, x=50.0, y=34.0,
        source_provider="sportec", team_attacking_direction="ltr",
    ))
    # Home GK
    rows.append(dict(
        game_id=1, period_id=period_id, frame_id=frame_id,
        time_seconds=time_seconds, frame_rate=25.0,
        player_id=pid, team_id=home_team_id, is_ball=False,
        is_goalkeeper=True, x=home_gk_pos[0], y=home_gk_pos[1],
        source_provider="sportec", team_attacking_direction="ltr",
    ))
    pid += 1
    # Home outfield
    for x, y in zip(home_outfield_xs, home_outfield_ys):
        rows.append(dict(
            game_id=1, period_id=period_id, frame_id=frame_id,
            time_seconds=time_seconds, frame_rate=25.0,
            player_id=pid, team_id=home_team_id, is_ball=False,
            is_goalkeeper=False, x=x, y=y,
            source_provider="sportec", team_attacking_direction="ltr",
        ))
        pid += 1
    # Away GK
    rows.append(dict(
        game_id=1, period_id=period_id, frame_id=frame_id,
        time_seconds=time_seconds, frame_rate=25.0,
        player_id=pid, team_id=away_team_id, is_ball=False,
        is_goalkeeper=True, x=away_gk_pos[0], y=away_gk_pos[1],
        source_provider="sportec", team_attacking_direction="ltr",
    ))
    pid += 1
    # Away outfield
    for x, y in zip(away_outfield_xs, away_outfield_ys):
        rows.append(dict(
            game_id=1, period_id=period_id, frame_id=frame_id,
            time_seconds=time_seconds, frame_rate=25.0,
            player_id=pid, team_id=away_team_id, is_ball=False,
            is_goalkeeper=False, x=x, y=y,
            source_provider="sportec", team_attacking_direction="ltr",
        ))
        pid += 1
    return pd.DataFrame(rows)


class TestFixedN4:
    def test_basic_4_defenders(self):
        """4 home defenders at known positions → exact values."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        # Home defends x=0; back line at x=10,12,11,13 sorted → 10,11,12,13
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 11.0, 13.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            away_outfield_xs=[95.0, 93.0, 94.0, 92.0, 60.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)

        # Home team row
        home = result[result["team_id"] == 1].iloc[0]
        # Back 4 (lowest x): 10, 11, 12, 13
        assert home["defensive_line_x"] == pytest.approx(11.5)  # mean(10,11,12,13)
        assert home["back_line_high_x"] == pytest.approx(13.0)  # max
        assert home["compactness_x"] == pytest.approx(3.0)  # 13-10
        # y values for back 4: 20, 40, 25, 48 → sorted: 20, 25, 40, 48
        assert home["lateral_width"] == pytest.approx(28.0)  # 48-20
        # gaps: 5, 15, 8 → max = 15
        assert home["max_lateral_gap"] == pytest.approx(15.0)
        assert home["back_n_count"] == 4

    def test_both_teams_computed(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 11.0, 13.0, 50.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 94.0, 92.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        teams = result["team_id"].unique()
        assert set(teams) == {1, 2}

    def test_away_team_defends_x105(self):
        """Away team's back line = highest-x outfield players."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 11.0, 13.0, 50.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 94.0, 92.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        away = result[result["team_id"] == 2].iloc[0]
        # Away defends x=105; back 4 (highest x): 95, 94, 93, 92
        assert away["defensive_line_x"] == pytest.approx(93.5)
        # back_line_high_x = min(x) for away (furthest from x=105)
        assert away["back_line_high_x"] == pytest.approx(92.0)
        assert away["compactness_x"] == pytest.approx(3.0)
        assert away["back_n_count"] == 4


class TestFixedN3N5:
    def test_n3(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 40.0, 50.0],
            home_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 60.0, 50.0],
            away_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=3)
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3
        assert home["defensive_line_x"] == pytest.approx(12.0)  # mean(10,12,14)

    def test_n5(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 18.0, 50.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 87.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=5)
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 5
        assert home["defensive_line_x"] == pytest.approx(14.0)  # mean(10,12,14,16,18)


class TestEdgeCases:
    def test_fewer_than_3_outfield_nan(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0],  # only 2
            home_outfield_ys=[20.0, 40.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        assert pd.isna(home["defensive_line_x"])
        assert pd.isna(home["back_n_count"])
        # Away should still work
        away = result[result["team_id"] == 2].iloc[0]
        assert not pd.isna(away["defensive_line_x"])

    def test_fixed_n_clamped_to_available(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0],  # 3 available, n=4 requested
            home_outfield_ys=[20.0, 34.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3  # clamped

    def test_gk_excluded(self):
        """Even if GK is at x=2 (lower than outfield), it shouldn't be in back line."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_gk_pos=(2.0, 34.0),  # GK very close to goal
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        # Back 4 should be 10,12,14,16 — NOT include GK at x=2
        assert home["defensive_line_x"] == pytest.approx(13.0)

    def test_nan_coordinates_excluded(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, float("nan"), 14.0, 50.0],
            home_outfield_ys=[20.0, 25.0, float("nan"), 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        # Only 4 valid outfield (NaN excluded); n=4 takes all 4: 10,12,14,50
        assert home["back_n_count"] == 4

    def test_empty_frames(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = pd.DataFrame(columns=[
            "game_id", "period_id", "frame_id", "time_seconds", "frame_rate",
            "player_id", "team_id", "is_ball", "is_goalkeeper", "x", "y",
            "source_provider", "team_attacking_direction",
        ])
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        assert len(result) == 0

    def test_invalid_n_raises(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        with pytest.raises(ValueError, match="n must be"):
            compute_defensive_line(frames, home_team_id=1, n=2)
        with pytest.raises(ValueError, match="n must be"):
            compute_defensive_line(frames, home_team_id=1, n=6)

    def test_invalid_adaptive_max_n_raises(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        with pytest.raises(ValueError, match="adaptive_max_n"):
            compute_defensive_line(frames, home_team_id=1, n="adaptive", adaptive_max_n=10)

    def test_ltr_guard_raises(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        frames["team_attacking_direction"] = "rtl"
        with pytest.raises(ValueError, match="LTR-normalized"):
            compute_defensive_line(frames, home_team_id=1, n=4)

    def test_ltr_guard_allows_nan(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        frames["team_attacking_direction"] = None  # all NaN
        # Should not raise
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        assert len(result) > 0


class TestAdaptive:
    def test_detects_4_back(self):
        """4 defenders clustered at x=10-13, big gap, then midfielders at x=40+."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 11.0, 12.0, 13.0, 40.0, 42.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[15.0, 25.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0, 48.0, 40.0, 35.0, 30.0, 25.0],
            away_outfield_ys=[15.0, 25.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4

    def test_detects_5_back(self):
        """5 clustered at x=10-14, big gap at [4]→[5], then midfield at x=45+."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 11.0, 12.0, 13.0, 14.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 87.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 5

    def test_detects_3_back(self):
        """3 clustered at x=10-12, big gap, then rest at x=30+."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 11.0, 12.0, 30.0, 35.0, 50.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 70.0, 65.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3

    def test_no_dominant_gap_defaults_to_4(self):
        """Evenly spaced players → no dominant gap → N=4."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        # Evenly spaced at 10, 15, 20, 25, 30, 35, ... (gap=5 everywhere)
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            home_outfield_ys=[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            away_outfield_xs=[95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4

    def test_all_same_x_defaults_to_4(self):
        """Degenerate: all at same x → all gaps 0 → N=4."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[20.0] * 10,
            home_outfield_ys=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            away_outfield_xs=[80.0] * 10,
            away_outfield_ys=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4

    def test_exactly_3_outfield(self):
        """Only 3 outfield → N=3 (no cuts to examine)."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0],
            home_outfield_ys=[20.0, 34.0, 48.0],
            away_outfield_xs=[95.0, 85.0, 75.0],
            away_outfield_ys=[20.0, 34.0, 48.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3

    def test_exactly_4_outfield_defaults_to_4(self):
        """P=4, single cut [2]→[3] → defaults to N=4."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4


class TestMultiPeriod:
    def test_period_isolation(self):
        """Two periods don't bleed into each other."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        f1 = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            period_id=1, frame_id=1,
        )
        f2 = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[85.0, 83.0, 81.0, 79.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            period_id=2, frame_id=1,
        )
        frames = pd.concat([f1, f2], ignore_index=True)
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        # Period 1 home line
        p1_home = result[(result["period_id"] == 1) & (result["team_id"] == 1)].iloc[0]
        assert p1_home["defensive_line_x"] == pytest.approx(13.0)
        # Period 2 home line (different positions)
        p2_home = result[(result["period_id"] == 2) & (result["team_id"] == 1)].iloc[0]
        assert p2_home["defensive_line_x"] == pytest.approx(23.0)
```

- [ ] **Step 8: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_defensive_line.py -v --tb=short 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._defensive_line'`

---

## Task 4: TF-14 — Implement `compute_defensive_line`

**Files:**
- Create: `silly_kicks/tracking/_defensive_line.py`

- [ ] **Step 9: Implement the per-frame kernel**

```python
"""Per-frame defensive-line geometry (TF-14).

Computes back-line geometry for both teams per frame. Foundational primitive
consumed by action-coupled VAEP features, GKDV stack, and line-break detection.

See spec: docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md §3.
See NOTICE for full bibliographic citations.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def compute_defensive_line(
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Per-(period_id, frame_id, team_id): 6 back-line geometry columns.

    Computes for BOTH teams. home_team_id determines goal assignment
    (must match the value used in play_left_to_right).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
        Must be LTR-normalized (play_left_to_right applied).
    home_team_id : int | str
        Home team identifier. After LTR normalization:
        - home_team_id defends goal at x=0 (back-line = lowest-x outfield)
        - other team defends goal at x=105 (back-line = highest-x outfield)
    n : int | Literal["adaptive"], default 4
        Target back-line player count (3, 4, or 5), clamped to available
        outfield players (minimum 3). Or "adaptive" for x-gap clustering.
    adaptive_max_n : int, default 5
        Upper bound for adaptive N. Must be in {3, 4, 5}.

    Returns
    -------
    pd.DataFrame
        Columns: period_id, frame_id, team_id, defensive_line_x,
        back_line_high_x, compactness_x, lateral_width, max_lateral_gap,
        back_n_count.

    Raises
    ------
    ValueError
        If n is an int outside {3, 4, 5}, adaptive_max_n outside {3, 4, 5},
        frames missing required columns, or non-LTR direction values found.

    Examples
    --------
    Compute defensive-line geometry for both teams::

        from silly_kicks.tracking.features import compute_defensive_line
        dl = compute_defensive_line(frames, home_team_id=1, n=4)

    See NOTICE for full bibliographic citations.
    """
    # --- Validation ---
    if isinstance(n, int) and n not in (3, 4, 5):
        raise ValueError(f"n must be 3, 4, or 5 (got {n})")
    if adaptive_max_n not in (3, 4, 5):
        raise ValueError(f"adaptive_max_n must be in {{3, 4, 5}} (got {adaptive_max_n})")

    required_cols = {"period_id", "frame_id", "team_id", "player_id", "is_ball", "is_goalkeeper", "x", "y"}
    missing = required_cols - set(frames.columns)
    if missing:
        raise ValueError(f"compute_defensive_line: frames missing columns {sorted(missing)}")

    # LTR guard
    if "team_attacking_direction" in frames.columns:
        directions = frames["team_attacking_direction"].dropna().unique()
        non_ltr = [d for d in directions if d != "ltr"]
        if non_ltr:
            raise ValueError(
                "compute_defensive_line: frames must be LTR-normalized "
                "(play_left_to_right). Found non-'ltr' values in "
                f"team_attacking_direction: {non_ltr}"
            )

    # --- Short-circuit ---
    result_cols = ["period_id", "frame_id", "team_id", "defensive_line_x",
                   "back_line_high_x", "compactness_x", "lateral_width",
                   "max_lateral_gap", "back_n_count"]
    if len(frames) == 0:
        return pd.DataFrame(columns=result_cols)

    # --- Core computation ---
    # Filter to outfield players with valid coordinates
    outfield = frames[
        (~frames["is_ball"]) & (~frames["is_goalkeeper"]) & frames["x"].notna()
    ].copy()

    # Group by (period_id, frame_id, team_id)
    rows: list[dict] = []
    groups = outfield.groupby(["period_id", "frame_id", "team_id"], dropna=False)

    for (period_id, frame_id, team_id), group in groups:
        p = len(group)
        if p < 3:
            rows.append({
                "period_id": period_id, "frame_id": frame_id, "team_id": team_id,
                "defensive_line_x": np.nan, "back_line_high_x": np.nan,
                "compactness_x": np.nan, "lateral_width": np.nan,
                "max_lateral_gap": np.nan, "back_n_count": pd.NA,
            })
            continue

        # Sort by proximity to own goal
        defends_x0 = (team_id == home_team_id)
        xs = group["x"].to_numpy(dtype="float64")
        ys = group["y"].to_numpy(dtype="float64")

        if defends_x0:
            order = np.argsort(xs)  # ascending: closest to x=0 first
        else:
            order = np.argsort(-xs)  # descending: closest to x=105 first

        xs_sorted = xs[order]
        ys_sorted = ys[order]

        # Determine N
        n_effective = _select_n(xs_sorted, n, adaptive_max_n, p)

        # Select back-line players
        sel_x = xs_sorted[:n_effective]
        sel_y = ys_sorted[:n_effective]

        # Compute 6 columns
        defensive_line_x = float(np.mean(sel_x))
        compactness_x = float(np.max(sel_x) - np.min(sel_x))

        if defends_x0:
            back_line_high_x = float(np.max(sel_x))  # furthest from x=0
        else:
            back_line_high_x = float(np.min(sel_x))  # furthest from x=105

        lateral_width = float(np.max(sel_y) - np.min(sel_y))

        # max_lateral_gap: sort by y, compute adjacent gaps
        y_sorted = np.sort(sel_y)
        if len(y_sorted) >= 2:
            y_gaps = np.diff(y_sorted)
            max_lateral_gap = float(np.max(y_gaps))
        else:
            max_lateral_gap = 0.0

        rows.append({
            "period_id": period_id, "frame_id": frame_id, "team_id": team_id,
            "defensive_line_x": defensive_line_x,
            "back_line_high_x": back_line_high_x,
            "compactness_x": compactness_x,
            "lateral_width": lateral_width,
            "max_lateral_gap": max_lateral_gap,
            "back_n_count": n_effective,
        })

    result = pd.DataFrame(rows, columns=result_cols)
    result["back_n_count"] = result["back_n_count"].astype("Int64")
    return result


def _select_n(
    xs_sorted: np.ndarray,
    n: int | Literal["adaptive"],
    adaptive_max_n: int,
    p: int,
) -> int:
    """Determine how many players form the back line.

    Parameters
    ----------
    xs_sorted : sorted x-positions (closest to own goal first)
    n : target N or "adaptive"
    adaptive_max_n : upper bound for adaptive
    p : total available outfield players

    Returns
    -------
    int : effective N (3..5, clamped to available)
    """
    if isinstance(n, int):
        return min(n, p)

    # --- Adaptive algorithm ---
    if p == 3:
        return 3
    if p == 4:
        # Single cut-point; no relative comparison possible → default N=4
        return 4

    # Examine cut-points: gaps between positions [2]→[3], [3]→[4], [4]→[5]
    gaps = np.diff(xs_sorted)  # gaps[i] = xs_sorted[i+1] - xs_sorted[i]

    # Available cut indices (0-indexed into gaps array):
    # cut at [2]→[3] means gaps[2]; corresponds to N=3
    # cut at [3]→[4] means gaps[3]; corresponds to N=4
    # cut at [4]→[5] means gaps[4]; corresponds to N=5
    cut_indices = []
    cut_ns = []
    for candidate_n in (3, 4, 5):
        gap_idx = candidate_n - 1  # gaps[2] = gap between sorted[2] and sorted[3] → N=3
        if gap_idx < len(gaps) and candidate_n <= adaptive_max_n:
            cut_indices.append(gap_idx)
            cut_ns.append(candidate_n)

    if not cut_indices:
        return min(4, p)

    cut_gaps = [abs(float(gaps[i])) for i in cut_indices]

    # Degenerate: all gaps are 0
    if max(cut_gaps) == 0.0:
        return min(4, p)

    # Find dominant gap
    sorted_gaps = sorted(cut_gaps, reverse=True)
    max_gap = sorted_gaps[0]
    second_gap = sorted_gaps[1] if len(sorted_gaps) > 1 else 0.0

    if second_gap == 0.0 or max_gap >= 1.5 * second_gap:
        # Dominant gap found
        best_idx = cut_gaps.index(max_gap)
        return cut_ns[best_idx]

    # No dominant gap → default to 4
    return min(4, p)
```

- [ ] **Step 10: Run per-frame kernel tests**

Run: `python -m pytest tests/tracking/test_defensive_line.py -v --tb=short`
Expected: All PASSED (21 tests)

---

## Task 5: TF-14 — Action-Coupled Batch Kernel + Per-Series Functions + Aggregator + Factory

**Files:**
- Modify: `silly_kicks/tracking/_kernels.py` (add `_defensive_line_at_actions`)
- Modify: `silly_kicks/tracking/features.py` (add 6 per-Series + aggregator + factory)
- Modify: `silly_kicks/tracking/__init__.py` (extend exports)

- [ ] **Step 11: Write action-coupled tests**

Create `tests/tracking/test_defensive_line_features.py`:

```python
"""Tests for action-coupled defensive-line features (TF-14 §4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking.test_defensive_line import _make_frame_rows


def _make_actions_for_defensive_line(team_id=1, time_seconds=1.0, period_id=1):
    """Actions by home team at known time."""
    return pd.DataFrame({
        "action_id": [1, 2],
        "period_id": [period_id, period_id],
        "time_seconds": [time_seconds, time_seconds],
        "team_id": [team_id, team_id],
        "player_id": [50, 51],
        "start_x": [50.0, 55.0],
        "start_y": [34.0, 34.0],
        "type_id": [0, 0],
    })


class TestActionCoupledFeatures:
    def test_action_gets_opposing_team_line(self):
        from silly_kicks.tracking.features import defensive_line_x

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        # Home team acts → should get AWAY team's defensive line
        actions = _make_actions_for_defensive_line(team_id=1)
        result = defensive_line_x(actions, frames, home_team_id=1)
        # Away back 4 (highest x): 95, 93, 91, 89 → mean = 92.0
        assert result.iloc[0] == pytest.approx(92.0)

    def test_unlinked_action_nan(self):
        from silly_kicks.tracking.features import defensive_line_x

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            time_seconds=100.0,  # far from action
        )
        actions = _make_actions_for_defensive_line(time_seconds=1.0)
        result = defensive_line_x(actions, frames, home_team_id=1)
        assert pd.isna(result.iloc[0])

    def test_aggregator_column_count(self):
        from silly_kicks.tracking.features import add_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        original_cols = len(actions.columns)
        result = add_defensive_line(actions, frames, home_team_id=1)
        new_cols = len(result.columns) - original_cols
        assert new_cols == 10  # 6 feature + 4 provenance

    def test_aggregator_provenance_skip_if_exists(self):
        """Provenance cols already present → not duplicated."""
        from silly_kicks.tracking.features import add_action_context, add_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        enriched = add_action_context(actions, frames)
        # Now add defensive line on top — provenance already exists
        result = add_defensive_line(enriched, frames, home_team_id=1)
        # Should have exactly 6 new feature cols, no _x/_y suffixes
        assert "frame_id_x" not in result.columns
        assert "frame_id_y" not in result.columns

    def test_xfns_factory_produces_valid(self):
        from silly_kicks.tracking.features import defensive_line_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        xfns = defensive_line_xfns(home_team_id=1)
        assert len(xfns) == 1
        assert is_frame_aware(xfns[0])

    def test_xfns_factory_has_name(self):
        from silly_kicks.tracking.features import defensive_line_xfns

        xfns = defensive_line_xfns(home_team_id=1)
        assert xfns[0].__name__ == "defensive_line"

    def test_xfns_column_count(self):
        """Factory transformer emits 6 × 3 = 18 columns."""
        from silly_kicks.tracking.features import defensive_line_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        states = gamestates(actions, nb_prev_actions=3)
        xfn = defensive_line_xfns(home_team_id=1)[0]
        result = xfn(states, frames)
        assert result.shape[1] == 18  # 6 cols × 3 states

    def test_batch_kernel_called_once(self):
        """Verify compute_defensive_line is called 3× (once per state), not 18×."""
        from unittest.mock import patch

        from silly_kicks.tracking._defensive_line import compute_defensive_line
        from silly_kicks.tracking.features import defensive_line_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        states = gamestates(actions, nb_prev_actions=3)
        xfn = defensive_line_xfns(home_team_id=1)[0]
        with patch(
            "silly_kicks.tracking._defensive_line.compute_defensive_line",
            wraps=compute_defensive_line,
        ) as mock_cdl:
            xfn(states, frames)
            assert mock_cdl.call_count == 3  # once per state slot
```

- [ ] **Step 12: Add batch kernel to `_kernels.py`**

First, add `Literal` to the imports at the top of `silly_kicks/tracking/_kernels.py`:
```python
from typing import TYPE_CHECKING, Literal
```

Then append to `silly_kicks/tracking/_kernels.py`:

```python
def _defensive_line_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.DataFrame:
    """All 6 defensive-line columns for the defending team at each action's linked frame.

    Calls compute_defensive_line ONCE on the full frames DataFrame, then
    joins on (period_id, frame_id, opposing_team_id) per action.

    Returns DataFrame aligned with actions.index.
    """
    from ._defensive_line import compute_defensive_line
    from .utils import link_actions_to_frames

    feature_cols = [
        "defensive_line_x", "back_line_high_x", "compactness_x",
        "lateral_width", "max_lateral_gap", "back_n_count",
    ]
    n_actions = len(actions)
    empty = pd.DataFrame(
        {col: np.full(n_actions, np.nan) for col in feature_cols},
        index=actions.index,
    )
    empty["back_n_count"] = pd.array([pd.NA] * n_actions, dtype="Int64")

    if n_actions == 0 or len(frames) == 0:
        return empty

    # Compute defensive line for all frames (ONCE)
    dl = compute_defensive_line(frames, home_team_id=home_team_id, n=n)
    if dl.empty:
        return empty

    # Link actions to frames
    pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    # Determine opposing team per action
    action_info = actions[["action_id", "team_id", "period_id"]]
    linked = linked.merge(action_info, on="action_id", how="left")

    # For each action, we want the OPPOSING team's defensive line at the linked frame
    # Get unique teams in the data
    linked["frame_id_int"] = linked["frame_id"].astype("int64")

    # Join with defensive-line data: match on (period_id, frame_id) then filter to opposing team
    merged = linked.merge(
        dl,
        left_on=["period_id", "frame_id_int"],
        right_on=["period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_dl"),
    )
    # Keep only rows where dl team != action team (opposing team's line)
    opposing = merged[merged["team_id_dl"] != merged["team_id_action"]]

    # Drop duplicates (one per action_id)
    opposing = opposing.drop_duplicates("action_id", keep="first")

    # Map back to actions index
    out = empty.copy()
    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in opposing.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            idx = action_to_idx.loc[aid]
            for col in feature_cols:
                out.at[idx, col] = row[col]

    out["back_n_count"] = out["back_n_count"].astype("Int64")
    return out
```

- [ ] **Step 13: Add per-Series functions, aggregator, and factory to `features.py`**

Append to `silly_kicks/tracking/features.py`:

```python
# ---------------------------------------------------------------------------
# PR-S27 -- TF-14: defensive-line features
# ---------------------------------------------------------------------------


def defensive_line_x(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """Mean x of the defending team's back-line at the linked frame (m).

    NaN where action is unlinked or defending team has <3 valid outfield players.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import defensive_line_x
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["defensive_line_x"].rename("defensive_line_x")


def back_line_high_x(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """x of the most advanced back-line player on the defending team (m).

    Approximates the offside line when the GK is behind the defensive line
    (typical case); NOT law-compliant for sweeper-keeper scenarios.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import back_line_high_x
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["back_line_high_x"].rename("back_line_high_x")


def compactness_x(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """x-spread of defending team's back-line (max - min, meters).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import compactness_x
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["compactness_x"].rename("compactness_x")


def lateral_width(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """y-spread of defending team's back-line (max - min, meters).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import lateral_width
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["lateral_width"].rename("lateral_width")


def max_lateral_gap(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """Largest y-gap between adjacent y-sorted back-line players (m).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import max_lateral_gap
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["max_lateral_gap"].rename("max_lateral_gap")


def back_n_count(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """Number of players in the defending team's back line (3/4/5).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import back_n_count
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["back_n_count"].rename("back_n_count")


@nan_safe_enrichment
def add_defensive_line(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.DataFrame:
    """Enrich actions with 6 defensive-line columns + 4 linkage-provenance columns.

    Provenance columns (frame_id, time_offset_seconds, link_quality_score,
    n_candidate_frames) are skipped if they already exist on the input DataFrame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_defensive_line
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    out = actions.copy()
    for col in ("defensive_line_x", "back_line_high_x", "compactness_x",
                "lateral_width", "max_lateral_gap"):
        out[col] = df[col]
    out["back_n_count"] = df["back_n_count"].astype("Int64")

    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing_provenance = [c for c in provenance_cols if c in out.columns]
    if not existing_provenance:
        pointers, _report = link_actions_to_frames(actions, frames)
        pointer_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


def defensive_line_xfns(
    home_team_id: int | str,
    *,
    n: int | Literal["adaptive"] = 4,
) -> list:
    """Build VAEP xfn list bound to a specific home_team_id.

    Returns a list with ONE FrameAwareTransformer that emits all 6
    defensive-line columns × 3 game-states = 18 columns total. This ensures
    compute_defensive_line is called 3x (once per state), not 18x.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, defensive_line_xfns
        xfns = tracking_default_xfns + defensive_line_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    col_names = [
        "defensive_line_x", "back_line_high_x", "compactness_x",
        "lateral_width", "max_lateral_gap", "back_n_count",
    ]

    def _defensive_line_transformer(states, frames):
        """Multi-column defensive-line xfn (6 cols x nb_states)."""
        out = pd.DataFrame(index=states[0].index)
        for i, slot in enumerate(states[:3]):
            batch = _kernels._defensive_line_at_actions(
                slot, frames, home_team_id=home_team_id, n=n
            )
            for col in col_names:
                out[f"{col}_a{i}"] = batch[col].to_numpy()
        return out

    _defensive_line_transformer._frame_aware = True  # type: ignore[attr-defined]
    _defensive_line_transformer.__name__ = "defensive_line"
    return [_defensive_line_transformer]
```

Also add `from typing import Literal` to the imports at the top of `features.py` and add all new names to `__all__`.

- [ ] **Step 14: Update `__init__.py` exports**

Add to `silly_kicks/tracking/__init__.py` `__all__` and the `from .features import (...)` block:
```python
"add_defensive_line",
"back_line_high_x",
"back_n_count",
"compactness_x",
"compute_defensive_line",
"defensive_line_x",
"defensive_line_xfns",
"lateral_width",
"max_lateral_gap",
```

Add to imports:
```python
from .features import (
    ...existing...,
    add_defensive_line,
    back_line_high_x,
    back_n_count,
    compactness_x,
    defensive_line_xfns,
    defensive_line_x,
    lateral_width,
    max_lateral_gap,
)
from ._defensive_line import compute_defensive_line
```

- [ ] **Step 15: Run action-coupled tests**

Run: `python -m pytest tests/tracking/test_defensive_line_features.py -v --tb=short`
Expected: All PASSED (8 tests)

- [ ] **Step 16: Run full tracking test suite**

Run: `python -m pytest tests/tracking/ -v --tb=short -m "not e2e" 2>&1 | tail -20`
Expected: All existing + new tests PASS

---

## Task 6: Invariant Tests

**Files:**
- Create: `tests/tracking/conftest.py` (shared fixture helpers)
- Create: `tests/invariants/test_invariant_defensive_line.py`
- Create: `tests/invariants/test_invariant_gk_resolve.py`

> **Note for implementer:** Tasks 3 and 5 define `_make_frame_rows`, `_make_frames`, and
> `_make_actions` in their respective test files. This task extracts those shared helpers
> into `tests/tracking/conftest.py` and updates ALL imports (test_defensive_line.py,
> test_defensive_line_features.py, plus the invariant test files below) to import from
> conftest. This avoids fragile cross-file test imports.

- [ ] **Step 17: Write invariant tests**

`tests/invariants/test_invariant_defensive_line.py`:

```python
"""Physical invariants for defensive-line geometry (TF-14)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking.conftest import _make_frame_rows


@pytest.fixture
def defensive_line_both_teams():
    """Multi-frame fixture with both teams having valid back lines."""
    from silly_kicks.tracking._defensive_line import compute_defensive_line

    frames = _make_frame_rows(
        home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
        home_outfield_ys=[10.0, 20.0, 40.0, 55.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0],
        away_outfield_ys=[10.0, 20.0, 40.0, 55.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    return compute_defensive_line(frames, home_team_id=1, n=4)


class TestRangeInvariants:
    def test_defensive_line_x_in_pitch(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["defensive_line_x"].dropna()
        assert (valid >= 0).all() and (valid <= 105).all()

    def test_back_line_high_x_in_pitch(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["back_line_high_x"].dropna()
        assert (valid >= 0).all() and (valid <= 105).all()

    def test_compactness_non_negative(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["compactness_x"].dropna()
        assert (valid >= 0).all()

    def test_lateral_width_in_range(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["lateral_width"].dropna()
        assert (valid >= 0).all() and (valid <= 68).all()

    def test_max_gap_bounded_by_width(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl[dl["max_lateral_gap"].notna()]
        assert (valid["max_lateral_gap"] <= valid["lateral_width"] + 1e-9).all()

    def test_back_n_count_domain(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["back_n_count"].dropna()
        assert set(valid.unique()).issubset({3, 4, 5})


class TestTriangleInequality:
    def test_home_back_line_high_minus_mean_le_compactness(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        home = dl[dl["team_id"] == 1].dropna(subset=["defensive_line_x"])
        # back_line_high_x - defensive_line_x <= compactness_x
        diff = home["back_line_high_x"] - home["defensive_line_x"]
        assert (diff <= home["compactness_x"] + 1e-9).all()

    def test_away_mean_minus_back_line_high_le_compactness(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        away = dl[dl["team_id"] == 2].dropna(subset=["defensive_line_x"])
        # For away: defensive_line_x - back_line_high_x <= compactness_x
        diff = away["defensive_line_x"] - away["back_line_high_x"]
        assert (diff <= away["compactness_x"] + 1e-9).all()


class TestCrossTeamSanity:
    def test_lines_not_both_near_same_goal(self, defensive_line_both_teams):
        """Both teams' lines shouldn't cluster near the same goal."""
        dl = defensive_line_both_teams
        home = dl[dl["team_id"] == 1]["defensive_line_x"].iloc[0]
        away = dl[dl["team_id"] == 2]["defensive_line_x"].iloc[0]
        # Home line near x=0, away near x=105 → sum ~ 105
        # If both near x=50, that's suspicious but possible (both pressing)
        # Invariant: they shouldn't BOTH be < 20 or BOTH be > 85
        assert not (home < 20 and away < 20)
        assert not (home > 85 and away > 85)
```

`tests/invariants/test_invariant_gk_resolve.py`:

```python
"""Physical invariants for defending_gk_from_frames (TF-13)."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking.conftest import _make_frames, _make_actions


class TestGkResolveInvariants:
    def test_resolved_player_is_goalkeeper(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        resolved_pid = result.iloc[0]
        if pd.notna(resolved_pid):
            gk_pids = set(frames[frames["is_goalkeeper"] == True]["player_id"].dropna())  # noqa: E712
            assert resolved_pid in gk_pids

    def test_resolved_player_is_opposing_team(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        resolved_pid = result.iloc[0]
        if pd.notna(resolved_pid):
            # Find which team this player belongs to
            player_team = frames.loc[
                frames["player_id"] == resolved_pid, "team_id"
            ].iloc[0]
            # Must be different from action team
            assert player_team != actions["team_id"].iloc[0]
```

- [ ] **Step 18: Run invariant tests**

Run: `python -m pytest tests/invariants/test_invariant_defensive_line.py tests/invariants/test_invariant_gk_resolve.py -v --tb=short`
Expected: All PASSED

---

## Task 7: NOTICE File + Public API Examples CI Gate

**Files:**
- Modify: `NOTICE` (add attribution section)
- Modify: `silly_kicks/tracking/features.py` (ensure all new public defs have Examples sections)

- [ ] **Step 19: Add academic attribution to NOTICE**

Append to the NOTICE file's "Mathematical / Methodological References" section:

```
## Defensive Line Geometry (TF-14)

Hernandez-Rodriguez, A., et al. (2025). "Prediction-based evaluation of
    back-four defense with spatial control in soccer." arXiv:2511.06191.
    (Fixed N=4 selection; defensive line height absolute/relative;
    stretch index; minimum-4-defenders threshold.)

Nakashima, Y., et al. (2025). "Analysis of Line Break prediction models for
    detecting defensive breakthrough in football." arXiv:2511.00121.
    (Second-last-defender offside-line definition; line-break detection.)

FIFA Enhanced Football Intelligence (2022). "Defensive line height and team
    length." FIFA Training Centre, World Cup 2022.
    (Deepest-outfield-player definition; GK exclusion convention.)

Forcher, L., Altmann, S., Forcher, L., Jekauc, D., & Kempe, M. (2022).
    "The use of player tracking data to analyze defensive play in
    professional soccer - A scoping review." International Journal of
    Sports Science & Coaching, 17(6), 1567-1592.
    (Survey of defensive tracking metrics; line height definition.)
```

- [ ] **Step 20: Run public API examples CI gate**

Run: `python -m pytest tests/test_public_api_examples.py -v --tb=short`
Expected: PASS (all new public functions have Examples sections)

---

## Task 8: Full Test Suite + Lint + Type Check

**Files:** None (verification only)

- [ ] **Step 21: Run ruff lint**

Run: `uv run ruff check silly_kicks/tracking/_gk_resolve.py silly_kicks/tracking/_defensive_line.py silly_kicks/tracking/features.py silly_kicks/tracking/_kernels.py`
Expected: No errors

- [ ] **Step 22: Run ruff format check**

Run: `uv run ruff format --check silly_kicks/tracking/ tests/tracking/test_gk_resolve.py tests/tracking/test_defensive_line.py tests/tracking/test_defensive_line_features.py tests/invariants/test_invariant_defensive_line.py tests/invariants/test_invariant_gk_resolve.py`
Expected: No reformatting needed (or fix and re-run)

- [ ] **Step 23: Run pyright**

Run: `uv run pyright silly_kicks/tracking/_gk_resolve.py silly_kicks/tracking/_defensive_line.py silly_kicks/tracking/features.py silly_kicks/tracking/_kernels.py`
Expected: 0 errors

- [ ] **Step 24: Run full test suite (non-e2e)**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short 2>&1 | tail -10`
Expected: All PASSED, no regressions

---

## Task 9: Final Review + Commit

**Files:** All modified/created files

- [ ] **Step 25: Run /final-review skill**

Invoke the `mad-scientist-skills:final-review` skill to run the pre-commit quality gate.

- [ ] **Step 26: Single commit**

```bash
git add silly_kicks/tracking/_gk_resolve.py \
        silly_kicks/tracking/_defensive_line.py \
        silly_kicks/tracking/features.py \
        silly_kicks/tracking/_kernels.py \
        silly_kicks/tracking/__init__.py \
        tests/tracking/conftest.py \
        tests/tracking/test_gk_resolve.py \
        tests/tracking/test_defensive_line.py \
        tests/tracking/test_defensive_line_features.py \
        tests/invariants/test_invariant_defensive_line.py \
        tests/invariants/test_invariant_gk_resolve.py \
        NOTICE \
        docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md

git commit -m "$(cat <<'EOF'
feat(tracking): TF-13 defending-GK resolution + TF-14 defensive-line geometry -- silly-kicks 3.4.0 (PR-S27)

TF-13: defending_gk_from_frames(actions, frames) resolves defending GK
player_id from tracking is_goalkeeper=True for all actions (not just shots).
Standalone composable utility for fillna on events-based defending_gk_player_id.

TF-14: compute_defensive_line(frames, home_team_id, n=4) computes per-frame
back-line geometry (6 columns) for both teams. Action-coupled layer adds
6 per-Series functions + add_defensive_line aggregator + defensive_line_xfns
factory (single multi-column transformer, 3x compute not 18x).

Columns: defensive_line_x, back_line_high_x, compactness_x, lateral_width,
max_lateral_gap, back_n_count. Supports fixed N=3/4/5 and adaptive x-gap
clustering. Minimum 3 outfield players threshold; NaN below.

Academic references: arXiv:2511.06191, arXiv:2511.00121, FIFA EFI 2022,
Forcher et al. 2022.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Spec Coverage Verification

| Spec Section | Task |
|---|---|
| §2 TF-13 API + implementation | Tasks 1-2 |
| §3 Per-frame kernel + algorithm | Tasks 3-4 |
| §3.5.1 Adaptive algorithm | Task 4 (Step 9) |
| §3.6 LTR guard | Task 4 (Step 9) |
| §4 Action-coupled layer | Task 5 |
| §4.4 Factory (no partial, single transformer) | Task 5 (Step 13) |
| §4.5 Aggregator (provenance skip-if-exists) | Task 5 (Step 13) |
| §6.1-6.3 Unit tests | Tasks 1, 3, 5 |
| §6.4 Invariant tests | Task 6 |
| §7 NOTICE attribution | Task 7 |
| Lint + type check | Task 8 |
| Final review + commit | Task 9 |
