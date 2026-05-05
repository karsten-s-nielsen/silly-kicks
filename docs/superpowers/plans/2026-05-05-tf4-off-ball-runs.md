# TF-4: Off-Ball Runs + Line-Break Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 6 action-coupled tracking features (4 off-ball-run + 2 line-break) with VAEP factory integration and full test coverage.

**Architecture:** Two kernels in a new `_off_ball_runs.py` module (paralleling `_ball_carrier.py` / `_defensive_line.py`). Off-ball-runs uses `slice_around_event` with game_id partitioning; line-break uses `compute_defensive_line` + `link_actions_to_frames`. Both resolve coordinates via `home_team_id`. Public surface: 3 aggregators + 1 VAEP factory in `features.py`.

**Tech Stack:** pandas, numpy (no new dependencies)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `silly_kicks/tracking/_off_ball_runs.py` (CREATE) | Two kernel functions: `_off_ball_runs_kernel` + `_line_break_kernel` |
| `silly_kicks/tracking/features.py` (MODIFY) | 3 aggregators + 1 xfn factory + `__all__` updates |
| `silly_kicks/tracking/__init__.py` (MODIFY) | Re-exports for new public API |
| `tests/tracking/test_off_ball_runs.py` (CREATE) | Unit + aggregator + VAEP integration tests |
| `tests/tracking/test_off_ball_runs_providers.py` (CREATE) | Provider fixture parametrized tests |
| `tests/invariants/test_invariant_off_ball_runs.py` (CREATE) | Physical invariant tests |
| `tests/tracking/conftest.py` (MODIFY) | Re-export new shared helper |
| `NOTICE` (MODIFY) | Add Spearman 2018 entry for off-ball-runs/line-break |
| `TODO.md` (MODIFY) | Delete TF-4 row, update TF-24 notes |
| `CHANGELOG.md` (MODIFY) | Add 6 columns + 3 aggregators + 1 factory |

---

### Task 1: Write the off-ball-runs kernel — failing tests first

**Files:**
- Create: `tests/tracking/test_off_ball_runs.py`
- Create: `silly_kicks/tracking/_off_ball_runs.py`

- [ ] **Step 1: Write failing unit tests for `_off_ball_runs_kernel`**

Create `tests/tracking/test_off_ball_runs.py` with a multi-frame fixture builder and core tests:

```python
"""Tests for off-ball runs + line-break features (TF-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_multi_frame_fixture(
    *,
    home_team_id=1,
    away_team_id=2,
    n_frames=5,
    frame_rate=25.0,
    period_id=1,
    game_id=1,
    players: list[dict] | None = None,
):
    """Build a multi-frame tracking fixture with controlled player movement.

    Each player dict: {player_id, team_id, is_goalkeeper, positions: [(x, y), ...]}
    where positions[i] is the (x, y) at frame i. len(positions) must == n_frames.
    """
    if players is None:
        players = []
    rows = []
    for fi in range(n_frames):
        time_s = fi * (1.0 / frame_rate)
        # Ball row
        rows.append(
            dict(
                game_id=game_id,
                period_id=period_id,
                frame_id=fi + 1,
                time_seconds=time_s,
                frame_rate=frame_rate,
                player_id=np.nan,
                team_id=np.nan,
                is_ball=True,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                ball_state="alive",
                team_attacking_direction="ltr",
                source_provider="synthetic",
            )
        )
        for p in players:
            pos = p["positions"][fi]
            rows.append(
                dict(
                    game_id=game_id,
                    period_id=period_id,
                    frame_id=fi + 1,
                    time_seconds=time_s,
                    frame_rate=frame_rate,
                    player_id=p["player_id"],
                    team_id=p["team_id"],
                    is_ball=False,
                    is_goalkeeper=p.get("is_goalkeeper", False),
                    x=pos[0],
                    y=pos[1],
                    ball_state="alive",
                    team_attacking_direction="ltr",
                    source_provider="synthetic",
                )
            )
    return pd.DataFrame(rows)


def _make_action_at(
    *,
    time_seconds: float,
    player_id: int,
    team_id: int,
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    period_id: int = 1,
    game_id: int = 1,
    action_id: int = 1,
    type_id: int = 0,
):
    """Create a single-row actions DataFrame."""
    return pd.DataFrame(
        {
            "game_id": [game_id],
            "action_id": [action_id],
            "period_id": [period_id],
            "time_seconds": [time_seconds],
            "team_id": [team_id],
            "player_id": [player_id],
            "start_x": [start_x],
            "start_y": [start_y],
            "end_x": [end_x],
            "end_y": [end_y],
            "type_id": [type_id],
        }
    )


class TestOffBallRunsKernel:
    def test_basic_two_qualifying_runners(self):
        """Two teammates move >=3m, one doesn't -> count=2."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        # Actor is player 10 on team 1 (home). Action at the last frame.
        # Teammate 11 moves 5m rightward (toward goal for home team).
        # Teammate 12 moves 4m upward (not toward goal).
        # Teammate 13 moves 1m (below threshold).
        n_frames = 5
        dt = 1.5 / (n_frames - 1)  # total window = 1.5s
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 11, "team_id": 1, "positions": [(50 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 12, "team_id": 1, "positions": [(30, 34 + i * 4.0 / (n_frames - 1)) for i in range(n_frames)]},
            {"player_id": 13, "team_id": 1, "positions": [(40, 34 + i * 1.0 / (n_frames - 1)) for i in range(n_frames)]},
            {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},  # opponent
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(
            players=players, n_frames=n_frames, frame_rate=n_frames / 1.5,
        )
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)

        assert result["n_off_ball_runners_pre_window"].iloc[0] == 2
        assert result["max_off_ball_run_displacement_pre_window"].iloc[0] == pytest.approx(5.0, abs=0.01)
        # mean speed: (5.0/1.5 + 4.0/1.5) / 2
        assert result["mean_off_ball_run_speed_pre_window"].iloc[0] == pytest.approx(3.0, abs=0.1)
        # Only player 11 moves toward goal (positive dx for home team)
        assert result["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 1

    def test_actor_excluded(self):
        """Actor's own movement is not counted."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50 + i * 10.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

    def test_opponent_excluded(self):
        """Opponents' movement is not counted."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 20, "team_id": 2, "positions": [(80 + i * 10.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

    def test_below_threshold_all_nan(self):
        """All teammates move < min_displacement_m -> 0 runners, NaN max/mean."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 11, "team_id": 1, "positions": [(30 + i * 1.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0
        assert pd.isna(result["max_off_ball_run_displacement_pre_window"].iloc[0])
        assert pd.isna(result["mean_off_ball_run_speed_pre_window"].iloc[0])

    def test_toward_goal_home_team(self):
        """Home-team runners: positive dx = toward goal."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            # Moves LEFT (negative dx) - NOT toward goal for home team
            {"player_id": 11, "team_id": 1, "positions": [(60 - i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1  # qualifies by displacement
        assert result["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 0  # but NOT toward goal

    def test_toward_goal_away_team(self):
        """Away-team runners: negative dx = toward goal (x=0 is their attacking direction)."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 20, "team_id": 2, "positions": [(60, 34)] * n_frames},
            # Away teammate moves LEFT (negative dx) -> toward x=0 = toward goal for away
            {"player_id": 22, "team_id": 2, "positions": [(40 - i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=20, team_id=2)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1
        assert result["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 1

    def test_dead_ball_at_action_time_nan(self):
        """Dead ball at action timestamp -> entire action NaN."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 11, "team_id": 1, "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        # Mark the last frame (action time) as dead ball
        last_frame_mask = frames["frame_id"] == n_frames
        frames.loc[last_frame_mask, "ball_state"] = "dead"

        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert pd.isna(result["n_off_ball_runners_pre_window"].iloc[0])

    def test_no_teammates_zero(self):
        """Actor is only outfield player on team -> 0 runners."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
            {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

    def test_custom_params(self):
        """Non-default pre_seconds and min_displacement_m are respected."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        # Player moves 2m total
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 11, "team_id": 1, "positions": [(30 + i * 2.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        # default threshold=3.0 -> 0 runners
        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

        # lower threshold=1.5 -> 1 runner
        result = _off_ball_runs_kernel(actions, frames, home_team_id=1, min_displacement_m=1.5)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1

    def test_multi_game_no_cross_contamination(self):
        """Two games with same period_id=1 don't cross-contaminate."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        # Game 1: teammate moves 5m
        players_g1 = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 11, "team_id": 1, "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        # Game 2: NO teammates for actor (different player set)
        players_g2 = [
            {"player_id": 30, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 2, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 40, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames_g1 = _make_multi_frame_fixture(players=players_g1, n_frames=n_frames, frame_rate=n_frames / 1.5, game_id=1)
        frames_g2 = _make_multi_frame_fixture(players=players_g2, n_frames=n_frames, frame_rate=n_frames / 1.5, game_id=2)
        frames = pd.concat([frames_g1, frames_g2], ignore_index=True)

        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions_g1 = _make_action_at(time_seconds=action_time, player_id=10, team_id=1, game_id=1, action_id=1)
        actions_g2 = _make_action_at(time_seconds=action_time, player_id=30, team_id=1, game_id=2, action_id=2)
        actions = pd.concat([actions_g1, actions_g2], ignore_index=True)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        # Game 1 should see 1 runner (player 11); game 2 should see 0 runners
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1
        assert result["n_off_ball_runners_pre_window"].iloc[1] == 0

    def test_ltr_guard_raises(self):
        """Non-LTR frames raise ValueError."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 3
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=25.0)
        frames["team_attacking_direction"] = "rtl"
        actions = _make_action_at(time_seconds=0.08, player_id=10, team_id=1)

        with pytest.raises(ValueError, match="LTR"):
            _off_ball_runs_kernel(actions, frames, home_team_id=1)

    def test_empty_frames_returns_columns(self):
        """Empty frames -> result with correct columns, all NaN."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        actions = _make_action_at(time_seconds=1.0, player_id=10, team_id=1)
        frames = pd.DataFrame(columns=[
            "game_id", "period_id", "frame_id", "time_seconds", "frame_rate",
            "player_id", "team_id", "is_ball", "is_goalkeeper", "x", "y",
            "ball_state", "team_attacking_direction", "source_provider",
        ])
        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        expected_cols = {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py::TestOffBallRunsKernel -v --tb=short 2>&1 | head -40`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._off_ball_runs'`

- [ ] **Step 3: Implement `_off_ball_runs_kernel`**

Create `silly_kicks/tracking/_off_ball_runs.py`:

```python
"""Off-ball runs + line-break kernels (TF-4).

Novel implementations inspired by the OBSO framework (Spearman 2018).
Off-ball-runs: per-attacking-teammate displacement in the pre-action window.
Line-break: action destination vs opposing team's defensive line geometry.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_OFF_BALL_RUNS_COLS = [
    "n_off_ball_runners_pre_window",
    "max_off_ball_run_displacement_pre_window",
    "mean_off_ball_run_speed_pre_window",
    "n_off_ball_runners_toward_goal_pre_window",
]

_LINE_BREAK_COLS = [
    "line_break",
    "n_attackers_behind_line",
]


def _validate_ltr(frames: pd.DataFrame) -> None:
    """Raise ValueError if frames contain non-LTR direction values."""
    if "team_attacking_direction" in frames.columns:
        directions = frames["team_attacking_direction"].dropna().unique()
        non_ltr = [d for d in directions if d != "ltr"]
        if non_ltr:
            raise ValueError(
                "_off_ball_runs: frames must be LTR-normalized "
                "(play_left_to_right). Found non-'ltr' values in "
                f"team_attacking_direction: {non_ltr}"
            )


def _off_ball_runs_kernel(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Per-action off-ball-run metrics for attacking teammates.

    Returns DataFrame aligned with actions.index, columns: _OFF_BALL_RUNS_COLS.
    """
    from .utils import slice_around_event

    n_actions = len(actions)
    empty = pd.DataFrame(
        {
            "n_off_ball_runners_pre_window": pd.array([pd.NA] * n_actions, dtype="Int64"),
            "max_off_ball_run_displacement_pre_window": np.full(n_actions, np.nan),
            "mean_off_ball_run_speed_pre_window": np.full(n_actions, np.nan),
            "n_off_ball_runners_toward_goal_pre_window": pd.array([pd.NA] * n_actions, dtype="Int64"),
        },
        index=actions.index,
    )

    if n_actions == 0 or len(frames) == 0:
        return empty

    _validate_ltr(frames)

    # Partition by game_id to prevent period_id collisions across games
    results = []
    for game_id, game_actions in actions.groupby("game_id", sort=False):
        game_frames = frames[frames["game_id"] == game_id]
        if game_frames.empty:
            results.append(
                pd.DataFrame(
                    {col: empty[col].dtype.type(pd.NA) if "Int64" in str(empty[col].dtype) else np.nan for col in _OFF_BALL_RUNS_COLS},
                    index=game_actions.index,
                )
            )
            continue

        sliced = slice_around_event(game_actions, game_frames, pre_seconds=pre_seconds, post_seconds=0.0)
        if sliced.empty:
            results.append(empty.loc[game_actions.index])
            continue

        # Extract dead-ball state from sliced BEFORE removing ball rows
        action_end_ball_state: dict = {}
        if "ball_state" in game_frames.columns:
            ball_in_window = sliced[sliced["is_ball"].astype(bool)]
            if not ball_in_window.empty:
                # Closest to action time = smallest |time_offset_seconds| per action
                abs_offset = ball_in_window["time_offset_seconds"].abs()
                closest_idx = abs_offset.groupby(ball_in_window["action_id"]).idxmin()
                closest = ball_in_window.loc[closest_idx]
                dead_actions = closest.loc[closest["ball_state"] == "dead", "action_id"]
                action_end_ball_state = {aid: "dead" for aid in dead_actions}

        # NOW remove ball rows
        sliced = sliced[~sliced["is_ball"].astype(bool)].copy()

        # Build per-action teammate displacements
        actor_id_per_action = game_actions[["action_id", "player_id", "team_id"]].rename(
            columns={"player_id": "actor_player_id", "team_id": "action_team_id"}
        )
        sliced = sliced.merge(actor_id_per_action, on="action_id", how="left")

        # Keep only same-team, non-actor, non-goalkeeper teammates
        teammates = sliced[
            (sliced["team_id"] == sliced["action_team_id"])
            & (sliced["player_id"] != sliced["actor_player_id"])
            & (~sliced["is_goalkeeper"].astype(bool))
        ].copy()

        # Drop NaN positions
        teammates = teammates.dropna(subset=["x", "y"])

        # Exclude dead-ball frames within window
        if "ball_state" in teammates.columns:
            teammates = teammates[teammates["ball_state"] != "dead"]

        # Compute per (action_id, player_id): first and last position
        game_out = pd.DataFrame(
            {
                "n_off_ball_runners_pre_window": pd.array([pd.NA] * len(game_actions), dtype="Int64"),
                "max_off_ball_run_displacement_pre_window": np.full(len(game_actions), np.nan),
                "mean_off_ball_run_speed_pre_window": np.full(len(game_actions), np.nan),
                "n_off_ball_runners_toward_goal_pre_window": pd.array([pd.NA] * len(game_actions), dtype="Int64"),
            },
            index=game_actions.index,
        )

        action_to_idx = pd.Series(game_actions.index.values, index=game_actions["action_id"].values)
        # O(1) team lookup per action (H1 fix — avoids O(n²) boolean mask scan)
        action_team_lookup = game_actions.set_index("action_id")["team_id"]

        for aid, action_group in teammates.groupby("action_id", sort=False):
            # Check dead-ball at action time
            if action_end_ball_state.get(aid) == "dead":
                continue  # stays NaN

            # Get action's team_id for toward-goal direction — O(1) lookup
            action_team = action_team_lookup.loc[aid]
            is_home = action_team == home_team_id

            per_player = action_group.sort_values("time_seconds").groupby("player_id", sort=False)

            runners = 0
            toward_goal = 0
            displacements = []

            for _pid, player_frames in per_player:
                if len(player_frames) < 2:
                    continue
                x_start = float(player_frames["x"].iloc[0])
                y_start = float(player_frames["y"].iloc[0])
                x_end = float(player_frames["x"].iloc[-1])
                y_end = float(player_frames["y"].iloc[-1])
                disp = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
                if disp >= min_displacement_m:
                    runners += 1
                    displacements.append(disp)
                    dx = x_end - x_start
                    # Home team: toward goal = positive dx
                    # Away team: toward goal = negative dx
                    if (is_home and dx > 0) or (not is_home and dx < 0):
                        toward_goal += 1

            if aid in action_to_idx.index:
                idx = action_to_idx.loc[aid]
                game_out.at[idx, "n_off_ball_runners_pre_window"] = runners
                game_out.at[idx, "n_off_ball_runners_toward_goal_pre_window"] = toward_goal
                if displacements:
                    game_out.at[idx, "max_off_ball_run_displacement_pre_window"] = max(displacements)
                    # Note: this is mean(displacement) / window_duration, not
                    # mean(displacement_i / observed_duration_i). For continuous
                    # tracking data (all players visible throughout) the two are
                    # equivalent. The denominator is the fixed window, making the
                    # metric a "displacement rate across the pre-window."
                    game_out.at[idx, "mean_off_ball_run_speed_pre_window"] = (
                        np.mean(displacements) / pre_seconds
                    )

        results.append(game_out)

    if not results:
        return empty

    return pd.concat(results).loc[actions.index]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py::TestOffBallRunsKernel -v --tb=short`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_off_ball_runs.py tests/tracking/test_off_ball_runs.py
git commit -m "feat(tracking): TF-4 off-ball-runs kernel with unit tests"
```

---

### Task 2: Write the line-break kernel — failing tests first

**Files:**
- Modify: `tests/tracking/test_off_ball_runs.py`
- Modify: `silly_kicks/tracking/_off_ball_runs.py`

- [ ] **Step 1: Write failing unit tests for `_line_break_kernel`**

Append to `tests/tracking/test_off_ball_runs.py`:

```python
class TestLineBreakKernel:
    def test_crosses_line_home_team(self):
        """Home-team action end_x past away team's defensive line -> True."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Away back 4 at x=90,92,94,96 -> mean defensive_line_x = 93
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        # Home-team action with end_x=95 > 93 -> line_break
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=1,
            end_x=95.0, end_y=34.0,
        )
        # Need action_id matching; fix player_id to one in frames
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        assert result["line_break"].iloc[0] is True

    def test_does_not_cross_line(self):
        """Home-team action end_x short of line -> False."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=1,
            end_x=80.0, end_y=34.0,
        )
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        assert result["line_break"].iloc[0] is False

    def test_crosses_line_away_team(self):
        """Away-team action: coordinate flip applied correctly."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Home back 4 at x=10,12,14,16 -> mean defensive_line_x = 13
        # For away team action: spadl_def_line_x = 105 - 13 = 92
        # Away-team action with end_x=95 > 92 -> line_break=True
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=2,
            end_x=95.0, end_y=34.0,
        )
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 2) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        assert result["line_break"].iloc[0] is True

    def test_no_defensive_line_returns_na(self):
        """< 3 outfield opponents -> pd.NA."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Only 2 away outfield players (need 3 for defensive line)
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0],
            away_outfield_ys=[30.0, 40.0],
        )
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=1,
            end_x=95.0, end_y=34.0,
        )
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        assert pd.isna(result["line_break"].iloc[0])

    def test_n_attackers_behind_line_home(self):
        """Home-team action: count attackers with tracking x > defensive_line_x."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Away back 4 at x=70,72,74,76 -> defensive_line_x = 73
        # Home outfield at x=10,12,14,16,75 -> player at x=75 is behind away line
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 75.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[70.0, 72.0, 74.0, 76.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=1,
            end_x=80.0, end_y=34.0,
        )
        # Use actor from home team
        home_outfield = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]
        actions["player_id"] = home_outfield["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        # One home player (x=75) is behind away defensive line (73)
        assert result["n_attackers_behind_line"].iloc[0] == 1

    def test_n_attackers_behind_line_away(self):
        """Away-team action: count attackers with tracking x < defensive_line_x."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Home back 4 at x=30,32,34,36 -> defensive_line_x = 33
        # Away outfield at x=90,92,94,96,25 -> player at x=25 is behind home line
        frames = _make_frame_rows(
            home_outfield_xs=[30.0, 32.0, 34.0, 36.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 25.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=2,
            end_x=95.0, end_y=34.0,
        )
        away_outfield = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 2) & (~frames["is_goalkeeper"])
        ]
        actions["player_id"] = away_outfield["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        # One away player (x=25) is behind home defensive line (33) -- x < 33
        assert result["n_attackers_behind_line"].iloc[0] == 1

    def test_line_break_dtype_is_boolean(self):
        """line_break column is nullable boolean, not object."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(
            time_seconds=1.0, player_id=50, team_id=1,
            end_x=95.0, end_y=34.0,
        )
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        assert str(result["line_break"].dtype) == "boolean"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py::TestLineBreakKernel -v --tb=short 2>&1 | head -20`
Expected: FAIL — `ImportError: cannot import name '_line_break_kernel'`

- [ ] **Step 3: Implement `_line_break_kernel`**

Append to `silly_kicks/tracking/_off_ball_runs.py`:

```python
def _line_break_kernel(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int = 4,
) -> pd.DataFrame:
    """Per-action line-break detection and attacker count behind defensive line.

    Returns DataFrame aligned with actions.index, columns: _LINE_BREAK_COLS.
    """
    from ._defensive_line import compute_defensive_line
    from .utils import link_actions_to_frames

    n_actions = len(actions)
    empty = pd.DataFrame(
        {
            "line_break": pd.array([pd.NA] * n_actions, dtype="boolean"),
            "n_attackers_behind_line": pd.array([pd.NA] * n_actions, dtype="Int64"),
        },
        index=actions.index,
    )

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

    linked["frame_id_int"] = linked["frame_id"].astype("int64")

    # Join with actions to get team_id, end_x, period_id, game_id
    linked = linked.merge(
        actions[["action_id", "team_id", "end_x", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )

    # Join with defensive-line data: match on (game_id, period_id, frame_id)
    merged = linked.merge(
        dl,
        left_on=["game_id", "period_id", "frame_id_int"],
        right_on=["game_id", "period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_dl"),
    )
    # Keep only rows where dl team != action team (opposing team's line)
    opposing = merged[merged["team_id_dl"] != merged["team_id_action"]].copy()
    opposing = opposing.drop_duplicates("action_id", keep="first")

    out = empty.copy()
    action_to_idx = pd.Series(actions.index.values, index=actions["action_id"].values)

    # Pre-build grouped dict for O(1) frame-player lookups (C1 fix — avoids
    # O(n_actions × n_frame_rows) boolean mask scan inside the loop)
    non_ball_non_gk = frames[
        (~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))
    ]
    frame_groups = dict(iter(
        non_ball_non_gk.groupby(["game_id", "period_id", "frame_id", "team_id"], sort=False)
    ))

    for _, row in opposing.iterrows():
        aid = row["action_id"]
        if aid not in action_to_idx.index:
            continue
        idx = action_to_idx.loc[aid]

        def_line_x = row["defensive_line_x"]
        if pd.isna(def_line_x):
            continue  # stays pd.NA

        action_team = row["team_id_action"]
        end_x = row["end_x"]

        # Coordinate-frame resolution
        if action_team == home_team_id:
            spadl_def_line_x = def_line_x
        else:
            spadl_def_line_x = 105.0 - def_line_x

        # Line-break: end_x > spadl_def_line_x (both in action-team SPADL frame)
        out.at[idx, "line_break"] = bool(end_x > spadl_def_line_x)

        # Count attackers behind line — O(1) lookup from pre-built dict
        frame_id = int(row["frame_id_int"])
        period_id = row["period_id"]
        game_id = row["game_id"]
        key = (game_id, period_id, frame_id, action_team)
        frame_players = frame_groups.get(key, pd.DataFrame())

        if frame_players.empty:
            out.at[idx, "n_attackers_behind_line"] = 0
            continue

        # In tracking coords:
        # Home-team attackers "behind" away line: tracking x > defensive_line_x
        # Away-team attackers "behind" home line: tracking x < defensive_line_x
        if action_team == home_team_id:
            behind_mask = frame_players["x"] > def_line_x
        else:
            behind_mask = frame_players["x"] < def_line_x

        out.at[idx, "n_attackers_behind_line"] = int(behind_mask.sum())

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py::TestLineBreakKernel -v --tb=short`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_off_ball_runs.py tests/tracking/test_off_ball_runs.py
git commit -m "feat(tracking): TF-4 line-break kernel with unit tests"
```

---

### Task 3: Public aggregators + xfn factory

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `tests/tracking/test_off_ball_runs.py`

- [ ] **Step 1: Write failing tests for aggregators and factory**

Append to `tests/tracking/test_off_ball_runs.py`:

```python
class TestAggregators:
    def test_add_off_ball_runs_columns(self):
        from silly_kicks.tracking.features import add_off_ball_runs

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 11, "team_id": 1, "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
            {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = add_off_ball_runs(actions, frames, home_team_id=1)
        new_cols = set(result.columns) - set(actions.columns)
        assert "n_off_ball_runners_pre_window" in new_cols
        assert "max_off_ball_run_displacement_pre_window" in new_cols
        assert "mean_off_ball_run_speed_pre_window" in new_cols
        assert "n_off_ball_runners_toward_goal_pre_window" in new_cols
        assert len(new_cols) == 4

    def test_add_line_break_columns(self):
        from silly_kicks.tracking.features import add_line_break
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(time_seconds=1.0, player_id=50, team_id=1, end_x=95.0)
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = add_line_break(actions, frames, home_team_id=1)
        new_cols = set(result.columns) - set(actions.columns)
        assert "line_break" in new_cols
        assert "n_attackers_behind_line" in new_cols
        assert len(new_cols) == 2
        assert str(result["line_break"].dtype) == "boolean"
        assert str(result["n_attackers_behind_line"].dtype) == "Int64"

    def test_add_off_ball_context_all_six(self):
        from silly_kicks.tracking.features import add_off_ball_context
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(time_seconds=1.0, player_id=50, team_id=1, end_x=95.0)
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = add_off_ball_context(actions, frames, home_team_id=1)
        expected = {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
            "line_break",
            "n_attackers_behind_line",
        }
        new_cols = set(result.columns) - set(actions.columns)
        assert expected.issubset(new_cols)


class TestXfnFactory:
    def test_factory_returns_frame_aware(self):
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        xfns = off_ball_context_xfns(home_team_id=1)
        assert len(xfns) == 1
        assert is_frame_aware(xfns[0])

    def test_factory_column_count(self):
        """Factory transformer emits 6 x 3 = 18 columns."""
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.feature_framework import gamestates
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(time_seconds=1.0, player_id=50, team_id=1, end_x=95.0)
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]
        # gamestates needs enough rows; duplicate for 3 actions
        actions = pd.concat([actions] * 4, ignore_index=True)
        actions["action_id"] = list(range(1, 5))
        states = gamestates(actions, nb_prev_actions=3)

        xfn = off_ball_context_xfns(home_team_id=1)[0]
        result = xfn(states, frames)
        assert result.shape[1] == 18

    def test_vaep_introspection_no_crash(self):
        """feature_column_names with off_ball_context_xfns -> no crash, NaN-filled."""
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.features.core import feature_column_names

        xfns = off_ball_context_xfns(home_team_id=1)
        cols = feature_column_names(xfns)
        assert len(cols) == 18
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py::TestAggregators -v --tb=short 2>&1 | head -20`
Expected: FAIL — `ImportError: cannot import name 'add_off_ball_runs' from 'silly_kicks.tracking.features'`

- [ ] **Step 3: Implement aggregators and factory in `features.py`**

Append to `silly_kicks/tracking/features.py` (before the final line, after the `defensive_line_xfns` section):

```python
# ---------------------------------------------------------------------------
# PR-S30 -- TF-4: off-ball runs + line-break features
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_off_ball_runs(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Enrich actions with 4 off-ball-run columns.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_off_ball_runs
    >>> # See tests/tracking/test_off_ball_runs.py for runnable examples.
    """
    from ._off_ball_runs import _off_ball_runs_kernel

    df = _off_ball_runs_kernel(
        actions, frames,
        home_team_id=home_team_id,
        pre_seconds=pre_seconds,
        min_displacement_m=min_displacement_m,
    )
    out = actions.copy()
    for col in (
        "n_off_ball_runners_pre_window",
        "max_off_ball_run_displacement_pre_window",
        "mean_off_ball_run_speed_pre_window",
        "n_off_ball_runners_toward_goal_pre_window",
    ):
        out[col] = df[col]
    return out


@nan_safe_enrichment
def add_line_break(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int = 4,
) -> pd.DataFrame:
    """Enrich actions with 2 line-break columns.

    Provenance columns are NOT emitted by this aggregator. Use
    ``add_defensive_line`` or ``add_action_context`` first if linkage
    provenance is needed — they append provenance with skip-if-present guard.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_line_break
    >>> # See tests/tracking/test_off_ball_runs.py for runnable examples.
    """
    from ._off_ball_runs import _line_break_kernel

    df = _line_break_kernel(actions, frames, home_team_id=home_team_id, n=n)
    out = actions.copy()
    out["line_break"] = df["line_break"]
    out["n_attackers_behind_line"] = df["n_attackers_behind_line"]
    return out


@nan_safe_enrichment
def add_off_ball_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int = 4,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Umbrella: add all 6 off-ball-run + line-break columns.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_off_ball_context
    >>> # See tests/tracking/test_off_ball_runs.py for runnable examples.
    """
    from ._off_ball_runs import _line_break_kernel, _off_ball_runs_kernel

    runs = _off_ball_runs_kernel(
        actions, frames,
        home_team_id=home_team_id,
        pre_seconds=pre_seconds,
        min_displacement_m=min_displacement_m,
    )
    lb = _line_break_kernel(actions, frames, home_team_id=home_team_id, n=n)
    out = actions.copy()
    for col in runs.columns:
        out[col] = runs[col]
    for col in lb.columns:
        out[col] = lb[col]
    return out


def off_ball_context_xfns(
    home_team_id: int | str,
    *,
    n: int = 4,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> list:
    """Build VAEP xfn list bound to home_team_id for TF-4 features.

    Returns a list with ONE FrameAwareTransformer that emits all 6
    off-ball-run + line-break columns x 3 game-states = 18 columns total.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, off_ball_context_xfns
        xfns = tracking_default_xfns + off_ball_context_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._off_ball_runs import _LINE_BREAK_COLS, _OFF_BALL_RUNS_COLS, _line_break_kernel, _off_ball_runs_kernel

    col_names = _OFF_BALL_RUNS_COLS + _LINE_BREAK_COLS

    def _off_ball_context_transformer(states, frames):
        """Multi-column off-ball-context xfn (6 cols x nb_states).

        Known optimization target (M1): _line_break_kernel calls
        compute_defensive_line per slot, but defensive-line depends only
        on frames (not actions) — result is identical across all 3 slots.
        Acceptable for v1; hoist into shared pre-computation if profiling
        shows this as a bottleneck.
        """
        out = pd.DataFrame(index=states[0].index)
        for i, slot in enumerate(states[:3]):
            runs = _off_ball_runs_kernel(
                slot, frames,
                home_team_id=home_team_id,
                pre_seconds=pre_seconds,
                min_displacement_m=min_displacement_m,
            )
            lb = _line_break_kernel(slot, frames, home_team_id=home_team_id, n=n)
            for col in _OFF_BALL_RUNS_COLS:
                out[f"{col}_a{i}"] = runs[col].to_numpy()
            for col in _LINE_BREAK_COLS:
                out[f"{col}_a{i}"] = lb[col].to_numpy()
        return out

    _off_ball_context_transformer._frame_aware = True  # type: ignore[attr-defined]
    _off_ball_context_transformer.__name__ = "off_ball_context"
    return [_off_ball_context_transformer]
```

Also update `__all__` in `features.py` — add these names in alphabetical position:

```python
"add_line_break",
"add_off_ball_context",
"add_off_ball_runs",
"off_ball_context_xfns",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py::TestAggregators tests/tracking/test_off_ball_runs.py::TestXfnFactory -v --tb=short`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/features.py tests/tracking/test_off_ball_runs.py
git commit -m "feat(tracking): TF-4 aggregators + VAEP xfn factory"
```

---

### Task 4: Module re-exports + `__init__.py`

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`

- [ ] **Step 1: Update `__init__.py` re-exports**

Add to `__all__` list (alphabetical):
```python
"add_line_break",
"add_off_ball_context",
"add_off_ball_runs",
"off_ball_context_xfns",
```

Add to the `from .features import (...)` block:
```python
add_line_break,
add_off_ball_context,
add_off_ball_runs,
off_ball_context_xfns,
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from silly_kicks.tracking import add_off_ball_runs, add_line_break, add_off_ball_context, off_ball_context_xfns; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add silly_kicks/tracking/__init__.py
git commit -m "feat(tracking): TF-4 public re-exports in __init__.py"
```

---

### Task 5: Provider fixture tests

**Files:**
- Create: `tests/tracking/test_off_ball_runs_providers.py`

- [ ] **Step 1: Write provider parametrized tests**

```python
"""Provider fixture tests for off-ball runs + line-break (TF-4)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_PROVIDERS = ["sportec", "metrica", "pff"]


@pytest.fixture(params=_PROVIDERS)
def provider_data(request):
    """Load frames and synthesize actions for a provider."""
    provider = request.param
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    # Determine home_team_id from majority team in frames
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    return actions, frames, home_team_id


class TestOffBallRunsProviders:
    def test_off_ball_runs_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_off_ball_runs

        actions, frames, home_team_id = provider_data
        result = add_off_ball_runs(actions, frames, home_team_id=home_team_id)
        assert "n_off_ball_runners_pre_window" in result.columns
        assert len(result) == len(actions)

    def test_line_break_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_line_break

        actions, frames, home_team_id = provider_data
        result = add_line_break(actions, frames, home_team_id=home_team_id)
        assert "line_break" in result.columns
        assert "n_attackers_behind_line" in result.columns
        assert len(result) == len(actions)

    def test_off_ball_context_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_off_ball_context

        actions, frames, home_team_id = provider_data
        result = add_off_ball_context(actions, frames, home_team_id=home_team_id)
        expected_cols = {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
            "line_break",
            "n_attackers_behind_line",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == len(actions)
```

- [ ] **Step 2: Run provider tests**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs_providers.py -v --tb=short`
Expected: All 9 tests (3 providers x 3 tests) PASS

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_off_ball_runs_providers.py
git commit -m "test(tracking): TF-4 provider fixture tests"
```

---

### Task 6: Invariant tests

**Files:**
- Create: `tests/invariants/test_invariant_off_ball_runs.py`
- Modify: `tests/tracking/conftest.py`

- [ ] **Step 1: Write invariant tests**

```python
"""Physical invariants for off-ball runs + line-break (TF-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking.test_off_ball_runs import _make_action_at, _make_multi_frame_fixture


@pytest.fixture
def off_ball_fixture():
    """Multi-player fixture with known off-ball movement."""
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

    n_frames = 10
    players = [
        {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
        {"player_id": 11, "team_id": 1, "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
        {"player_id": 12, "team_id": 1, "positions": [(40, 20 + i * 4.0 / (n_frames - 1)) for i in range(n_frames)]},
        {"player_id": 13, "team_id": 1, "positions": [(60 + i * 6.0 / (n_frames - 1), 34) for i in range(n_frames)]},
        {"player_id": 14, "team_id": 1, "positions": [(45, 34 + i * 1.0 / (n_frames - 1)) for i in range(n_frames)]},
        {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
        {"player_id": 21, "team_id": 2, "positions": [(85, 34)] * n_frames},
        {"player_id": 22, "team_id": 2, "positions": [(90, 34)] * n_frames},
        {"player_id": 23, "team_id": 2, "positions": [(95, 34)] * n_frames},
        {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
        {"player_id": 24, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
    ]
    frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
    action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
    actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

    return _off_ball_runs_kernel(actions, frames, home_team_id=1)


class TestOffBallRunsInvariants:
    def test_n_runners_non_negative(self, off_ball_fixture):
        valid = off_ball_fixture["n_off_ball_runners_pre_window"].dropna()
        assert (valid >= 0).all()

    def test_toward_goal_subset_of_runners(self, off_ball_fixture):
        df = off_ball_fixture.dropna(subset=["n_off_ball_runners_pre_window"])
        assert (
            df["n_off_ball_runners_toward_goal_pre_window"] <= df["n_off_ball_runners_pre_window"]
        ).all()

    def test_max_displacement_exceeds_threshold(self, off_ball_fixture):
        has_runners = off_ball_fixture[off_ball_fixture["n_off_ball_runners_pre_window"] > 0]
        if not has_runners.empty:
            assert (has_runners["max_off_ball_run_displacement_pre_window"] >= 3.0 - 1e-9).all()

    def test_mean_speed_non_negative(self, off_ball_fixture):
        valid = off_ball_fixture["mean_off_ball_run_speed_pre_window"].dropna()
        assert (valid >= 0).all()


class TestLineBreakInvariants:
    def test_n_attackers_non_negative(self):
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(time_seconds=1.0, player_id=50, team_id=1, end_x=95.0)
        actions["player_id"] = frames[
            (~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        result = _line_break_kernel(actions, frames, home_team_id=1)
        valid = result["n_attackers_behind_line"].dropna()
        assert (valid >= 0).all()
```

- [ ] **Step 2: Update `tests/tracking/conftest.py`**

Add re-export so invariant tests can import shared fixtures:

```python
from tests.tracking.test_off_ball_runs import _make_action_at, _make_multi_frame_fixture
```

Update `__all__` to include `"_make_action_at"` and `"_make_multi_frame_fixture"`.

- [ ] **Step 3: Run invariant tests**

Run: `uv run python -m pytest tests/invariants/test_invariant_off_ball_runs.py -v --tb=short`
Expected: All 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/invariants/test_invariant_off_ball_runs.py tests/tracking/conftest.py
git commit -m "test(tracking): TF-4 physical invariant tests"
```

---

### Task 7: NOTICE + TODO + CHANGELOG updates

**Files:**
- Modify: `NOTICE`
- Modify: `TODO.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update NOTICE**

Add after the existing Spearman 2018 entry (which is already there for zone-based defender intensity). Find the existing entry and append a clarification, OR add a new paragraph referencing TF-4:

```
The off-ball-runs and line-break detection features in
silly_kicks/tracking/_off_ball_runs.py (PR-S30, TF-4) are novel
implementations inspired by:

- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics
  Conference.
  (OBSO framework — Off-Ball Scoring Opportunity; off-ball-runs and
  line-break concepts.)

- Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017). "Not all passes are
  created equal: Objectively measuring the risk and reward of passes in
  soccer from tracking data." KDD '17.
  (Contextual passing risk/reward; §4 qualitatively mentions line-breaking
  passes inside formation clustering.)
```

- [ ] **Step 2: Update TODO.md**

- Delete the TF-4 row from Tier 4.
- Update TF-24 notes to append: `plus TF-4's off_ball_runs parameters (pre_seconds, min_displacement_m)`.
- Bump header date to today if needed and update current release to `3.6.0`.

- [ ] **Step 3: Update CHANGELOG.md**

Under an `## [Unreleased]` or `## [3.6.0]` section, add:

```markdown
### Added
- `add_off_ball_runs(actions, frames, *, home_team_id)` — 4 off-ball-run columns: `n_off_ball_runners_pre_window`, `max_off_ball_run_displacement_pre_window`, `mean_off_ball_run_speed_pre_window`, `n_off_ball_runners_toward_goal_pre_window`
- `add_line_break(actions, frames, *, home_team_id)` — 2 line-break columns: `line_break` (nullable boolean), `n_attackers_behind_line` (Int64)
- `add_off_ball_context(actions, frames, *, home_team_id)` — umbrella aggregator adding all 6 columns
- `off_ball_context_xfns(home_team_id)` — VAEP factory (6 features x 3 states = 18 columns)
```

- [ ] **Step 4: Commit**

```bash
git add NOTICE TODO.md CHANGELOG.md
git commit -m "docs: TF-4 NOTICE + TODO + CHANGELOG updates"
```

---

### Task 8: Full test suite + lint + pyright

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite for TF-4 tests**

Run: `uv run python -m pytest tests/tracking/test_off_ball_runs.py tests/tracking/test_off_ball_runs_providers.py tests/invariants/test_invariant_off_ball_runs.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run ruff check + format**

Run: `uv run ruff check silly_kicks/tracking/_off_ball_runs.py silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py && uv run ruff format --check silly_kicks/tracking/_off_ball_runs.py silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py`
Expected: No errors

- [ ] **Step 3: Run pyright**

Run: `uv run pyright silly_kicks/tracking/_off_ball_runs.py silly_kicks/tracking/features.py`
Expected: 0 errors

- [ ] **Step 4: Run broader test suite to check for regressions**

Run: `uv run python -m pytest tests/ -m "not e2e" --tb=short -q`
Expected: All pass, no regressions

- [ ] **Step 5: Final review**

Invoke `/final-review` skill before the single commit.

---

### Task 9: Squash into single commit + push

**Files:** All modified/created files

- [ ] **Step 1: Interactive rebase to squash all task commits**

```bash
git rebase -i main
```

Squash all commits into one with message:

```
feat(tracking): TF-4 off-ball runs + line-break detection -- silly-kicks 3.6.0 (PR-S30)

Six new action-coupled tracking features (4 off-ball-run + 2 line-break)
with coordinate-frame resolution, VAEP factory integration, provider
fixture tests, and physical invariant coverage.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

- [ ] **Step 2: Push and create PR**

```bash
git push -u origin pr-s30-tf4-off-ball-runs
gh pr create --title "feat(tracking): TF-4 off-ball runs + line-break detection (PR-S30)" --body "..."
```
