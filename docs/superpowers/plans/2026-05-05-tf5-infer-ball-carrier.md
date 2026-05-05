# TF-5 `infer_ball_carrier` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship per-frame ball-carrier inference with hysteresis, action-coupled wrapper, and compute_defensive_line game_id consistency fix.

**Architecture:** Per-frame standalone primitive in `_ball_carrier.py` (sequential within game/period groups, vectorized distance/velocity per frame). Action-coupled wrapper in `features.py` follows `defending_gk_from_frames` pattern. Atomic SPADL re-exports standard version. Consistency fix adds `game_id` to `compute_defensive_line` groupby.

**Tech Stack:** pandas, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-05-tf5-infer-ball-carrier-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `silly_kicks/tracking/_ball_carrier.py` | Create | `infer_ball_carrier` per-frame primitive |
| `silly_kicks/tracking/features.py` | Modify | `ball_carrier_at_action` wrapper + `__all__` |
| `silly_kicks/tracking/__init__.py` | Modify | Re-exports |
| `silly_kicks/atomic/tracking/features.py` | Modify | Re-export `ball_carrier_at_action` |
| `silly_kicks/tracking/_defensive_line.py` | Modify | Add `game_id` to groupby + return schema |
| `silly_kicks/tracking/_kernels.py` | Modify | Update `_defensive_line_at_actions` merge keys |
| `tests/tracking/test_ball_carrier.py` | Create | TDD unit tests |
| `tests/tracking/test_defensive_line.py` | Modify | Assert `game_id` in output + multi-game test |
| `tests/invariants/test_invariant_ball_carrier.py` | Create | Physical invariants |
| `tests/tracking/conftest.py` | Modify | Re-export new fixture helpers |
| `NOTICE` | Modify | Bauer & Anzer 2021 + Vidal-Codina 2022 citations |
| `TODO.md` | Modify | Move TF-5 to shipped |

---

## Task 1: `compute_defensive_line` `game_id` consistency fix

Do this first — it's a prerequisite for the consistent `game_id`-in-groupby pattern that `infer_ball_carrier` inherits.

**Files:**
- Modify: `silly_kicks/tracking/_defensive_line.py`
- Modify: `silly_kicks/tracking/_kernels.py:835-841`
- Modify: `tests/tracking/test_defensive_line.py`

- [ ] **Step 1: Write test for multi-game collision**

Add to `tests/tracking/test_defensive_line.py` at the end of class `TestMultiPeriod`:

```python
class TestMultiGame:
    def test_game_id_in_output_columns(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        assert "game_id" in result.columns

    def test_multi_game_no_collision(self):
        """Two games with same (period_id, frame_id) produce separate rows."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        f1 = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        f2 = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[85.0, 83.0, 81.0, 79.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        f2["game_id"] = 2  # different game, same period_id=1, frame_id=1
        frames = pd.concat([f1, f2], ignore_index=True)
        result = compute_defensive_line(frames, home_team_id=1, n=4)

        # Should have 4 rows: 2 games x 2 teams
        assert len(result) == 4
        g1_home = result[(result["game_id"] == 1) & (result["team_id"] == 1)].iloc[0]
        g2_home = result[(result["game_id"] == 2) & (result["team_id"] == 1)].iloc[0]
        assert g1_home["defensive_line_x"] == pytest.approx(13.0)
        assert g2_home["defensive_line_x"] == pytest.approx(23.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_defensive_line.py::TestMultiGame -v --tb=short`
Expected: FAIL — `game_id` not in result columns.

- [ ] **Step 3: Update `_defensive_line.py`**

In `silly_kicks/tracking/_defensive_line.py`:

(a) Add `"game_id"` to `required_cols` (line 73):
```python
required_cols = {"game_id", "period_id", "frame_id", "team_id", "player_id", "is_ball", "is_goalkeeper", "x", "y"}
```

(b) Add `"game_id"` as first entry in `result_cols` (line 90):
```python
    result_cols = [
        "game_id",
        "period_id",
        "frame_id",
        ...
    ]
```

(c) Add `"game_id"` to the groupby (line 110):
```python
    groups = outfield.groupby(["game_id", "period_id", "frame_id", "team_id"], dropna=False)
```

(d) Update the loop unpacking (line 112) and row dicts to include `game_id`:
```python
    for (game_id, period_id, frame_id, team_id), group in groups:
```

And in both the `< 3 players` NaN row dict and the normal row dict, add `"game_id": game_id` as the first entry.

- [ ] **Step 4: Update `_kernels.py:_defensive_line_at_actions` merge keys**

In `silly_kicks/tracking/_kernels.py`, update the merge at line 835-841.

The actions DataFrame carries `game_id` from the SPADL schema. Add it to the linked projection and merge keys:

(a) Update projection at line 827 to include `game_id`:
```python
    linked = linked.merge(
        actions_with_idx[["action_id", "_row_idx", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )
```

(b) Update the merge keys at line 835:
```python
    merged = linked.merge(
        dl,
        left_on=["game_id", "period_id", "frame_id_int"],
        right_on=["game_id", "period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_dl"),
    )
```

- [ ] **Step 5: Run all defensive-line tests**

Run: `python -m pytest tests/tracking/test_defensive_line.py tests/tracking/test_defensive_line_features.py tests/invariants/test_invariant_defensive_line.py -v --tb=short`
Expected: ALL PASS.

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -m "not e2e" --tb=short -q`
Expected: ALL PASS, no regressions.

---

## Task 2: TDD — `infer_ball_carrier` unit tests

Write all tests before implementation.

**Files:**
- Create: `tests/tracking/test_ball_carrier.py`

- [ ] **Step 1: Create test file with fixture helper and all tests**

Create `tests/tracking/test_ball_carrier.py`:

```python
"""Tests for silly_kicks.tracking._ball_carrier.infer_ball_carrier."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _make_carrier_frame(
    *,
    game_id=1,
    period_id=1,
    frame_id=1,
    ball_x=50.0,
    ball_y=34.0,
    ball_state="alive",
    players: list[dict],
) -> pd.DataFrame:
    """Build a single-frame fixture for ball carrier tests.

    Each player dict: {pid, tid, x, y} + optional {vx, vy, is_goalkeeper}.
    Ball row omits vx/vy keys — pandas fills them with NaN via dict-union,
    so ``has_velocity`` (column-existence check) is True when any player
    dict includes vx/vy. Ball-row NaN velocity is intentional and doesn't
    affect carrier inference (ball rows are filtered out before scoring).
    """
    rows = []
    # Ball row
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=frame_id * 0.04,
            frame_rate=25.0,
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=ball_x,
            y=ball_y,
            ball_state=ball_state,
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
    )
    for p in players:
        row = dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=frame_id * 0.04,
            frame_rate=25.0,
            player_id=p["pid"],
            team_id=p["tid"],
            is_ball=False,
            is_goalkeeper=p.get("is_goalkeeper", False),
            x=p["x"],
            y=p["y"],
            ball_state=ball_state,
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
        if "vx" in p:
            row["vx"] = p["vx"]
            row["vy"] = p["vy"]
        rows.append(row)
    return pd.DataFrame(rows)


def _concat_frames(*frame_dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frame_dfs, ignore_index=True)


_RESULT_COLS = [
    "game_id",
    "period_id",
    "frame_id",
    "ball_carrier_player_id",
    "ball_carrier_distance_m",
    "ball_carrier_team_id",
]


class TestVelocityAwareScoring:
    def test_velocity_breaks_distance_tie(self):
        """Player farther away but moving toward ball wins over closer stationary player."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                # Player 10: closer (1.5m) but stationary
                dict(pid=10, tid=1, x=51.5, y=34.0, vx=0.0, vy=0.0),
                # Player 20: farther (2.5m) but moving toward ball at 4 m/s
                # velocity toward ball = 4.0 m/s; score = 2.5 - 0.5*4 = 0.5
                # vs player 10: score = 1.5 - 0.5*0 = 1.5
                dict(pid=20, tid=1, x=52.5, y=34.0, vx=-4.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 20


class TestHysteresis:
    def test_incumbent_retained_within_gamma(self):
        """Incumbent carrier kept when new candidate is only slightly better."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        # Frame 1: player 10 is closest (1.0m), becomes carrier
        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: player 20 is now closer (1.8m vs 2.0m for player 10)
        # But difference (0.2m) < gamma (1.0m), so incumbent (10) stays
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.8, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 10]

    def test_incumbent_overridden_when_exceeds_gamma(self):
        """New carrier wins when score difference exceeds gamma."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        # Frame 1: player 10 at 2.0m
        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=55.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: player 20 now at 0.5m, player 10 at 2.5m
        # Difference = 2.0m > gamma(1.0m), so 20 takes over
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 20]

    def test_hysteresis_resets_on_dead_ball(self):
        """Dead-ball gap clears incumbent; next alive frame uses pure scoring."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: dead ball
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            ball_state="dead",
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 3: alive again, player 20 is closer (0.8m vs 1.0m for 10)
        # No incumbent -> 20 wins on pure scoring (difference < gamma but no incumbent)
        f3 = _make_carrier_frame(
            frame_id=3,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.8, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2, f3)
        result = infer_ball_carrier(frames, gamma=1.0)
        r = result.sort_values("frame_id")
        assert r.iloc[0]["ball_carrier_player_id"] == 10  # frame 1
        assert pd.isna(r.iloc[1]["ball_carrier_player_id"])  # frame 2 dead
        assert r.iloc[2]["ball_carrier_player_id"] == 20  # frame 3 no incumbent

    def test_hysteresis_resets_on_nan_carrier(self):
        """No-candidate frame clears incumbent."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: player too far (> tolerance 3m)
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=60.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 3: player 20 close, should win without hysteresis (incumbent was reset)
        f3 = _make_carrier_frame(
            frame_id=3,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2, f3)
        result = infer_ball_carrier(frames, tolerance_m=3.0, gamma=1.0)
        r = result.sort_values("frame_id")
        assert r.iloc[0]["ball_carrier_player_id"] == 10
        assert pd.isna(r.iloc[1]["ball_carrier_player_id"])
        assert r.iloc[2]["ball_carrier_player_id"] == 20

    def test_first_frame_no_hysteresis(self):
        """First frame of period: pure scoring, no gamma bonus."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames, gamma=1.0)
        assert result["ball_carrier_player_id"].iloc[0] == 20  # closer wins


class TestDistanceOnlyFallback:
    def test_fallback_when_no_velocity_columns(self):
        """Correct carrier + UserWarning when vx/vy absent."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0),  # no vx/vy
                dict(pid=20, tid=1, x=53.0, y=34.0),
            ],
        )
        # Columns vx/vy not present
        assert "vx" not in frames.columns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = infer_ball_carrier(frames)
            assert any("vx/vy columns not found" in str(x.message) for x in w)
        assert result["ball_carrier_player_id"].iloc[0] == 10

    def test_distance_only_with_hysteresis(self):
        """Hysteresis applies even in distance-only mode."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0),
                dict(pid=20, tid=1, x=53.0, y=34.0),
            ],
        )
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0),  # 2.0m
                dict(pid=20, tid=1, x=51.8, y=34.0),  # 1.8m, but delta < gamma
            ],
        )
        frames = _concat_frames(f1, f2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 10]  # incumbent retained


class TestEdgeCases:
    def test_dead_ball_nan(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_state="dead",
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_ball_state_nan_treated_as_alive(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_state=None,
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 10

    def test_no_ball_row_nan(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        # Remove ball row — player rows remain, so unique_frames still has
        # this (game_id, period_id, frame_id), but ball_pos merge yields NaN.
        frames = frames[~frames["is_ball"]].reset_index(drop=True)
        result = infer_ball_carrier(frames)
        assert len(result) == 1
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_ball_coords_nan(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=np.nan,
            ball_y=np.nan,
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_no_candidates_within_tolerance(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=10, tid=1, x=60.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames, tolerance_m=3.0)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_gk_as_carrier(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=5.0,
            ball_y=34.0,
            players=[
                dict(pid=1, tid=1, x=5.5, y=34.0, vx=0.0, vy=0.0, is_goalkeeper=True),
                dict(pid=10, tid=1, x=15.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 1

    def test_tiebreak_determinism(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=20, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 10  # lowest pid

    def test_empty_frames(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = pd.DataFrame(columns=[
            "game_id", "period_id", "frame_id", "time_seconds", "frame_rate",
            "player_id", "team_id", "is_ball", "is_goalkeeper", "x", "y",
            "ball_state", "source_provider", "team_attacking_direction",
        ])
        result = infer_ball_carrier(frames)
        assert list(result.columns) == _RESULT_COLS
        assert len(result) == 0

    def test_set_piece_transition(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        # Frame 1: dead ball
        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            ball_state="dead",
            players=[
                dict(pid=10, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.3, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: alive — kicker (20) closer
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            ball_state="alive",
            players=[
                dict(pid=10, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.2, y=34.0, vx=1.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames)
        r = result.sort_values("frame_id")
        assert pd.isna(r.iloc[0]["ball_carrier_player_id"])
        assert r.iloc[1]["ball_carrier_player_id"] == 20

    def test_multiple_ball_rows_mean(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=10, tid=1, x=52.0, y=35.0, vx=0.0, vy=0.0)],
        )
        # Add a second ball row with different position
        extra_ball = pd.DataFrame([dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=0.04,
            frame_rate=25.0, player_id=np.nan, team_id=np.nan,
            is_ball=True, is_goalkeeper=False,
            x=54.0, y=36.0,
            ball_state="alive", source_provider="sportec",
            team_attacking_direction="ltr", vx=np.nan, vy=np.nan,
        )])
        frames = pd.concat([frames, extra_ball], ignore_index=True)
        result = infer_ball_carrier(frames)
        # Mean ball pos = (52, 35); player at (52, 35) -> distance ~0
        # Just verify it produces a result without error
        assert result["ball_carrier_player_id"].iloc[0] == 10

    def test_multi_game_batch(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            game_id=1,
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        f2 = _make_carrier_frame(
            game_id=2,
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=20, tid=2, x=51.5, y=34.0, vx=0.0, vy=0.0)],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames)
        assert len(result) == 2
        g1 = result[result["game_id"] == 1].iloc[0]
        g2 = result[result["game_id"] == 2].iloc[0]
        assert g1["ball_carrier_player_id"] == 10
        assert g2["ball_carrier_player_id"] == 20


class TestReturnSchema:
    def test_output_columns(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert list(result.columns) == _RESULT_COLS

    def test_distance_bounded_by_tolerance(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames, tolerance_m=3.0)
        valid = result["ball_carrier_distance_m"].dropna()
        assert (valid <= 3.0).all()

    def test_team_id_matches_carrier(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_team_id"].iloc[0] == 1

    def test_fresh_range_index(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert list(result.index) == list(range(len(result)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_ball_carrier.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._ball_carrier'`

---

## Task 3: Implement `infer_ball_carrier` primitive

**Files:**
- Create: `silly_kicks/tracking/_ball_carrier.py`

- [ ] **Step 1: Create the implementation**

Create `silly_kicks/tracking/_ball_carrier.py`:

```python
"""Per-frame ball-carrier inference (TF-5).

Heuristic: composite scoring of distance + velocity-toward-ball with
hysteresis to prevent flickering. Operates on tracking frames (long-form
TRACKING_FRAMES_COLUMNS shape). Returns one row per (game_id, period_id,
frame_id).

See spec: docs/superpowers/specs/2026-05-05-tf5-infer-ball-carrier-design.md
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def infer_ball_carrier(
    frames: pd.DataFrame,
    *,
    tolerance_m: float = 3.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Per-frame ball-carrier inference with hysteresis.

    For each frame where ``ball_state`` is not ``"dead"`` (NaN/None treated
    as alive), identifies the player most likely carrying the ball via a
    composite score of distance and velocity-toward-ball, with an incumbent
    bonus (``gamma``) to prevent flickering.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_m : float, default 3.0
        Maximum ball-to-player distance for candidacy (meters).
        Carrier-attribution radius, not dribbling-contact threshold.
    beta : float, default 0.5
        Distance advantage per m/s of velocity toward ball (seconds).
    gamma : float, default 1.0
        Hysteresis bonus for incumbent carrier (meters). The incumbent's
        score is reduced by ``gamma``, so a new candidate must be
        ``gamma`` meters better in composite score to take over.
        ``gamma=0`` gives stateless per-frame behaviour.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, period_id, frame_id, ball_carrier_player_id,
        ball_carrier_distance_m, ball_carrier_team_id. One row per unique
        (game_id, period_id, frame_id). Fresh RangeIndex.

    Examples
    --------
    Infer ball carrier for a full match::

        from silly_kicks.tracking import infer_ball_carrier
        carriers = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.5, gamma=1.0)

    References
    ----------
    Bauer & Anzer (2021). "Data-driven detection of counterpressing in
        professional football." Data Mining and Knowledge Discovery.
    Vidal-Codina et al. (2022). "Automatic Event Detection in Football
        Using Tracking Data." Sports Engineering.

    See NOTICE for full bibliographic citations.
    """
    result_cols = [
        "game_id",
        "period_id",
        "frame_id",
        "ball_carrier_player_id",
        "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]

    if len(frames) == 0:
        return pd.DataFrame(columns=result_cols)

    has_velocity = "vx" in frames.columns and "vy" in frames.columns
    if not has_velocity:
        warnings.warn(
            "vx/vy columns not found; falling back to distance-only carrier "
            "inference. Call derive_velocities() first for velocity-aware scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Detect output dtypes from frames
    pid_dtype = frames["player_id"].dtype
    tid_dtype = frames["team_id"].dtype

    # Split ball and player rows
    ball_mask = frames["is_ball"] == True  # noqa: E712
    ball_rows = frames[ball_mask]
    player_rows = frames[~ball_mask & frames["x"].notna()]

    # Per-frame ball position: mean of non-NaN ball x/y
    ball_pos = (
        ball_rows.groupby(["game_id", "period_id", "frame_id"], dropna=False)
        .agg(bx=("x", "mean"), by=("y", "mean"), bs=("ball_state", "first"))
        .reset_index()
    )

    # Unique frames (from all rows, not just ball rows)
    unique_frames = (
        frames[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "period_id", "frame_id"])
        .reset_index(drop=True)
    )

    # Merge ball position onto unique frames
    frame_ball = unique_frames.merge(
        ball_pos, on=["game_id", "period_id", "frame_id"], how="left"
    )

    # Pre-build grouped dict for O(1) per-frame candidate lookup.
    # Avoids O(n×m) boolean-mask filtering inside the sequential loop.
    _empty_player_df = player_rows.iloc[:0]
    player_groups: dict[tuple, pd.DataFrame] = dict(
        iter(player_rows.groupby(["game_id", "period_id", "frame_id"]))
    )

    # Process sequentially within each (game_id, period_id) group
    results: list[dict] = []
    groups = frame_ball.groupby(["game_id", "period_id"], dropna=False)

    for (_gid, _pid), group in groups:
        incumbent_pid = None
        group_sorted = group.sort_values("frame_id")

        for _, frow in group_sorted.iterrows():
            gid = frow["game_id"]
            pid_val = frow["period_id"]
            fid = frow["frame_id"]
            bx = frow["bx"]
            by = frow["by"]
            bs = frow["bs"]

            # Dead ball -> NaN, reset incumbent
            if bs == "dead":
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            # No ball position -> NaN, reset incumbent
            if pd.isna(bx) or pd.isna(by):
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            # O(1) candidate lookup via pre-built dict
            cands = player_groups.get((gid, pid_val, fid), _empty_player_df)

            if cands.empty:
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            # Compute distances
            cx = cands["x"].to_numpy(dtype=float)
            cy = cands["y"].to_numpy(dtype=float)
            dx = cx - float(bx)
            dy = cy - float(by)
            dists = np.sqrt(dx * dx + dy * dy)

            # Filter to tolerance
            within = dists <= tolerance_m
            if not within.any():
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            cand_idx = np.flatnonzero(within)
            cand_dists = dists[cand_idx]
            cand_pids = cands["player_id"].to_numpy()[cand_idx]
            cand_tids = cands["team_id"].to_numpy()[cand_idx]

            # Compute scores
            scores = cand_dists.copy()

            if has_velocity:
                vx_vals = cands["vx"].to_numpy(dtype=float)[cand_idx]
                vy_vals = cands["vy"].to_numpy(dtype=float)[cand_idx]

                # Direction from player to ball
                dir_x = -dx[cand_idx]
                dir_y = -dy[cand_idx]
                dir_norm = np.sqrt(dir_x * dir_x + dir_y * dir_y)
                # Avoid division by zero
                safe_norm = np.where(dir_norm > 0, dir_norm, 1.0)
                unit_x = dir_x / safe_norm
                unit_y = dir_y / safe_norm

                # Velocity toward ball (dot product)
                v_toward = vx_vals * unit_x + vy_vals * unit_y
                # Clamp negative, handle NaN
                v_toward = np.where(np.isnan(v_toward), 0.0, np.maximum(v_toward, 0.0))

                scores = cand_dists - beta * v_toward

            # Apply hysteresis bonus to incumbent
            if incumbent_pid is not None and gamma > 0:
                inc_mask = cand_pids == incumbent_pid
                if inc_mask.any():
                    inc_idx = np.flatnonzero(inc_mask)
                    scores[inc_idx] -= gamma

            # Select best: lowest score, tiebreak by lowest player_id
            best_idx = _select_best(scores, cand_pids)
            winner_pid = cand_pids[best_idx]
            winner_dist = float(cand_dists[best_idx])
            winner_tid = cand_tids[best_idx]

            incumbent_pid = winner_pid
            results.append({
                "game_id": gid,
                "period_id": pid_val,
                "frame_id": fid,
                "ball_carrier_player_id": winner_pid,
                "ball_carrier_distance_m": winner_dist,
                "ball_carrier_team_id": winner_tid,
            })

    if not results:
        return pd.DataFrame(columns=result_cols)

    out = pd.DataFrame(results, columns=result_cols)

    # Preserve dtype for player_id and team_id
    if str(pid_dtype) == "Int64":
        out["ball_carrier_player_id"] = pd.to_numeric(
            out["ball_carrier_player_id"], errors="coerce"
        ).astype("Int64")
    if str(tid_dtype) == "Int64":
        out["ball_carrier_team_id"] = pd.to_numeric(
            out["ball_carrier_team_id"], errors="coerce"
        ).astype("Int64")

    return out


def _nan_row(game_id, period_id, frame_id) -> dict:
    return {
        "game_id": game_id,
        "period_id": period_id,
        "frame_id": frame_id,
        "ball_carrier_player_id": np.nan,
        "ball_carrier_distance_m": np.nan,
        "ball_carrier_team_id": np.nan,
    }


def _select_best(scores: np.ndarray, pids: np.ndarray) -> int:
    """Index of lowest score; tiebreak by lowest player_id.

    Uses Python-level ``<`` for tiebreak comparison so both int and
    string player_ids (e.g. Sportec DFL-OBJ-*) work safely across
    numpy versions.
    """
    min_score = np.nanmin(scores)
    tied = np.flatnonzero(np.abs(scores - min_score) < 1e-12)
    if len(tied) == 1:
        return int(tied[0])
    # Tiebreak: lowest player_id via Python comparison (safe for
    # both int and object/string dtypes).
    best_idx = tied[0]
    best_pid = pids[tied[0]]
    for i in tied[1:]:
        if pids[i] < best_pid:
            best_idx = i
            best_pid = pids[i]
    return int(best_idx)
```

- [ ] **Step 2: Run unit tests**

Run: `python -m pytest tests/tracking/test_ball_carrier.py -v --tb=short`
Expected: ALL PASS.

- [ ] **Step 3: Run ruff + pyright**

Run: `uv run ruff check silly_kicks/tracking/_ball_carrier.py && uv run ruff format --check silly_kicks/tracking/_ball_carrier.py && uv run pyright silly_kicks/tracking/_ball_carrier.py`
Expected: Clean.

---

## Task 4: Action-coupled wrapper + re-exports

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `silly_kicks/atomic/tracking/features.py`

- [ ] **Step 1: Add test for `ball_carrier_at_action`**

Append to `tests/tracking/test_ball_carrier.py`:

```python
class TestActionCoupledWrapper:
    def _make_actions(self, n=2):
        return pd.DataFrame({
            "game_id": [1] * n,
            "action_id": list(range(n)),
            "period_id": [1] * n,
            "time_seconds": [0.04 * (i + 1) for i in range(n)],
            "team_id": [1] * n,
            "player_id": [10] * n,
        })

    def test_linked_carrier_matches(self):
        from silly_kicks.tracking.features import ball_carrier_at_action

        frames = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        actions = self._make_actions(n=1)
        actions["time_seconds"] = [0.04]  # matches frame_id=1
        result = ball_carrier_at_action(actions, frames)
        assert len(result) == 1
        assert result.iloc[0] == 10

    def test_unlinked_action_nan(self):
        from silly_kicks.tracking.features import ball_carrier_at_action

        frames = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        actions = self._make_actions(n=1)
        actions["time_seconds"] = [999.0]  # no matching frame
        result = ball_carrier_at_action(actions, frames)
        assert pd.isna(result.iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_carrier.py::TestActionCoupledWrapper -v --tb=short`
Expected: FAIL — `ball_carrier_at_action` not found.

- [ ] **Step 3: Implement `ball_carrier_at_action` in `features.py`**

In `silly_kicks/tracking/features.py`:

(a) Add to `__all__` (after `"back_n_count"` in alphabetical position):
```python
    "ball_carrier_at_action",
```

(b) Add the import at the top with the other private-module imports (after line 38 `from ._gk_resolve import defending_gk_from_frames`):
```python
from ._ball_carrier import infer_ball_carrier
```

(c) Add the function body after the `defending_gk_from_frames` re-export block and before the existing `nearest_defender_distance` function. Place it after the existing private imports section:

```python
def ball_carrier_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
    tolerance_m: float = 3.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> pd.Series:
    """Per-action ball carrier player_id resolved from tracking frames.

    Links actions to frames via ``link_actions_to_frames``, then looks up
    the ``infer_ball_carrier`` result at the linked frame.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.
    tolerance_m : float, default 3.0
        Carrier-attribution radius passed to ``infer_ball_carrier``.
    beta : float, default 0.5
        Velocity weight passed to ``infer_ball_carrier``.
    gamma : float, default 1.0
        Hysteresis bonus passed to ``infer_ball_carrier``.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype.
        NaN where action couldn't link or no carrier found.

    Examples
    --------
    Get the ball carrier at each action::

        from silly_kicks.tracking.features import ball_carrier_at_action
        carrier = ball_carrier_at_action(actions, frames)

    See NOTICE for full bibliographic citations.
    """
    import numpy as np

    pid_dtype = frames["player_id"].dtype
    n = len(actions)
    out = pd.Series(np.full(n, np.nan), index=actions.index, dtype="object")

    if n == 0 or len(frames) == 0:
        return out

    # Compute per-frame carriers
    carriers = infer_ball_carrier(
        frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma
    )
    if carriers.empty:
        return out

    # Link actions to frames
    pointers, _report = link_actions_to_frames(
        actions, frames, tolerance_seconds=tolerance_seconds
    )

    # Join pointers with actions to get period_id
    ptr = pointers.merge(
        actions[["action_id", "period_id"]],
        on="action_id",
        how="left",
    )
    linked = ptr[ptr["frame_id"].notna()].copy()
    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")

    # Join with carriers on (period_id, frame_id)
    merged = linked.merge(
        carriers[["period_id", "frame_id", "ball_carrier_player_id"]],
        left_on=["period_id", "frame_id_int"],
        right_on=["period_id", "frame_id"],
        how="left",
    )

    # Deduplicate: one carrier per action_id (take first)
    merged = merged.drop_duplicates("action_id", keep="first")

    # Map back to actions index
    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in merged.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = row["ball_carrier_player_id"]

    # Cast to match frames dtype if numeric
    if pid_dtype == np.dtype("int64") or str(pid_dtype) == "Int64":
        out = pd.to_numeric(out, errors="coerce")
        if str(pid_dtype) == "Int64":
            out = out.astype("Int64")

    return out
```

- [ ] **Step 4: Update `tracking/__init__.py`**

In `silly_kicks/tracking/__init__.py`:

(a) Add to `__all__` (alphabetical, after `"add_sync_score"`):
```python
    "ball_carrier_at_action",
    "infer_ball_carrier",
```

(b) Add to the imports from `._ball_carrier`:
```python
from ._ball_carrier import infer_ball_carrier
```

(c) Add to the imports from `.features`:
```python
    ball_carrier_at_action,
```

- [ ] **Step 5: Update `atomic/tracking/features.py`**

In `silly_kicks/atomic/tracking/features.py`:

(a) Add to `__all__` (alphabetical, after `"add_pressure_on_actor"`):
```python
    "ball_carrier_at_action",
```

(b) Add the re-export import at the top of the file (after the existing imports):
```python
from silly_kicks.tracking.features import ball_carrier_at_action  # noqa: F401
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/tracking/test_ball_carrier.py -v --tb=short`
Expected: ALL PASS.

- [ ] **Step 7: Run ruff + pyright on all modified files**

Run: `uv run ruff check silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py silly_kicks/atomic/tracking/features.py && uv run ruff format --check silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py silly_kicks/atomic/tracking/features.py`
Expected: Clean.

---

## Task 5: Invariant tests

**Files:**
- Create: `tests/invariants/test_invariant_ball_carrier.py`
- Modify: `tests/tracking/conftest.py`

- [ ] **Step 1: Create invariant test file**

Create `tests/invariants/test_invariant_ball_carrier.py`:

```python
"""Physical invariants for ball-carrier inference (TF-5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking.test_ball_carrier import _make_carrier_frame, _concat_frames


@pytest.fixture
def carrier_multi_frame():
    """Multi-frame fixture with varied carrier scenarios."""
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier

    f1 = _make_carrier_frame(
        frame_id=1,
        ball_x=50.0,
        ball_y=34.0,
        players=[
            dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            dict(pid=20, tid=2, x=52.0, y=35.0, vx=-1.0, vy=0.0),
            dict(pid=30, tid=1, x=48.0, y=34.0, vx=0.0, vy=0.0),
        ],
    )
    f2 = _make_carrier_frame(
        frame_id=2,
        ball_x=50.0,
        ball_y=34.0,
        players=[
            dict(pid=10, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
            dict(pid=20, tid=2, x=51.0, y=34.5, vx=-2.0, vy=0.0),
        ],
    )
    frames = _concat_frames(f1, f2)
    return infer_ball_carrier(frames, tolerance_m=3.0), frames


class TestCarrierInvariants:
    def test_distance_bounded_by_tolerance(self, carrier_multi_frame):
        result, _ = carrier_multi_frame
        valid = result["ball_carrier_distance_m"].dropna()
        assert (valid <= 3.0 + 1e-9).all()

    def test_carrier_is_never_ball_row(self, carrier_multi_frame):
        result, frames = carrier_multi_frame
        ball_pids = frames[frames["is_ball"] == True]["player_id"].unique()  # noqa: E712
        carrier_pids = result["ball_carrier_player_id"].dropna().unique()
        for cpid in carrier_pids:
            assert cpid not in ball_pids or pd.isna(cpid)

    def test_team_id_matches_carrier_player(self, carrier_multi_frame):
        result, frames = carrier_multi_frame
        valid = result[result["ball_carrier_player_id"].notna()]
        player_teams = (
            frames[~frames["is_ball"]]
            .drop_duplicates("player_id")[["player_id", "team_id"]]
            .set_index("player_id")["team_id"]
        )
        for _, row in valid.iterrows():
            cpid = row["ball_carrier_player_id"]
            expected_tid = player_teams.loc[cpid]
            assert row["ball_carrier_team_id"] == expected_tid
```

- [ ] **Step 2: Update conftest.py to re-export new helpers**

In `tests/tracking/conftest.py`, add the import (matching existing pattern
where conftest re-exports `_make_frame_rows` from `test_defensive_line`):

```python
from tests.tracking.test_ball_carrier import _make_carrier_frame

__all__ = ["_make_actions", "_make_carrier_frame", "_make_frame_rows", "_make_frames"]
```

- [ ] **Step 3: Run invariant tests**

Run: `python -m pytest tests/invariants/test_invariant_ball_carrier.py -v --tb=short`
Expected: ALL PASS.

---

## Task 6: NOTICE + public-API-examples coverage

**Files:**
- Modify: `NOTICE`
- Modify: `tests/test_public_api_examples.py` (if `_ball_carrier.py` needs adding — check if auto-discovered)

- [ ] **Step 1: Add NOTICE entries**

In `NOTICE`, after the existing FIFA (2022) entry (around line 133, before the "The implementations are independent..." paragraph), add:

```
- Bauer, P., & Anzer, G. (2021). "Data-driven detection of counterpressing in
  professional football." Data Mining and Knowledge Discovery, 35(5), 2009-2049.
  (Section 3 describes a velocity-toward-ball heuristic for carrier
  identification, used as input to their counterpressing classifier.
  Adapted for infer_ball_carrier primitive in silly_kicks.tracking.)

- Vidal-Codina, F., Evans, N., El Fakir, B., & Billingham, J. (2022).
  "Automatic Event Detection in Football Using Tracking Data."
  Sports Engineering, 25, 18.
  (Inertia/hysteresis recommendation for ball-possession algorithms;
  motivates the gamma hysteresis parameter in infer_ball_carrier.)
```

- [ ] **Step 2: Verify public-API examples CI coverage**

The `tests/test_public_api_examples.py` scans `_PUBLIC_MODULE_FILES`. `_ball_carrier.py` is a private module (underscore-prefixed) so it is not in the scan list. The public surface is `features.py` and `atomic/tracking/features.py`, both already in the list. `ball_carrier_at_action` docstring has an Examples section. Verify:

Run: `python -m pytest tests/test_public_api_examples.py -v --tb=short`
Expected: ALL PASS.

---

## Task 7: TODO.md + CHANGELOG.md update

**Files:**
- Modify: `TODO.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update TODO.md**

(a) Move TF-5 row from On Deck to Active Cycle (or delete it — per feedback, shipped rows are deleted).

Since TF-5 will ship with this PR: delete the TF-5 row from the Tier 3 table.

(b) Update the header date and current release:
```
**Last updated**: 2026-05-05. **Current release**: silly-kicks 3.5.0.
```

(c) Move TF-5 to Active Cycle:
```
## Active Cycle

| # | Task | PR | Status |
|---|------|----|--------|
| TF-5 | `infer_ball_carrier` per-frame primitive + `ball_carrier_at_action` wrapper | PR-S28 | In progress |
```

- [ ] **Step 2: Add CHANGELOG.md entry**

Insert a new `## [3.5.0]` section at the top of CHANGELOG.md (after the header, before `## [3.4.0]`):

```markdown
## [3.5.0] — 2026-05-XX

### Added

#### TF-5: Per-frame ball-carrier inference

- `silly_kicks.tracking._ball_carrier.infer_ball_carrier(frames, *, tolerance_m, beta, gamma)` — per-frame ball-carrier identification via composite distance + velocity-toward-ball scoring with hysteresis. Returns one row per (game_id, period_id, frame_id) with carrier player_id, distance, and team_id. Distance-only fallback when vx/vy columns absent.
- `silly_kicks.tracking.features.ball_carrier_at_action(actions, frames, ...)` — action-coupled wrapper resolving ball carrier at each linked frame.

#### Consistency: `compute_defensive_line` game_id groupby

- `compute_defensive_line` now includes `game_id` in groupby + return schema, preventing cross-game collisions when processing multi-game batches.

#### Academic references (NOTICE)

- Bauer & Anzer 2021 (Data Mining and Knowledge Discovery) — velocity-toward-ball carrier identification heuristic.
- Vidal-Codina et al. 2022 (Sports Engineering) — hysteresis recommendation for ball-possession algorithms.
```

---

## Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: ALL PASS.

- [ ] **Step 2: Run ruff + pyright across all changed files**

Run: `uv run ruff check silly_kicks/tracking/_ball_carrier.py silly_kicks/tracking/_defensive_line.py silly_kicks/tracking/_kernels.py silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py silly_kicks/atomic/tracking/features.py && uv run ruff format --check silly_kicks/tracking/_ball_carrier.py silly_kicks/tracking/_defensive_line.py silly_kicks/tracking/_kernels.py silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py silly_kicks/atomic/tracking/features.py`
Expected: Clean.

Run: `uv run pyright silly_kicks/tracking/_ball_carrier.py silly_kicks/tracking/features.py`
Expected: Clean.

- [ ] **Step 3: Invoke `/final-review`**

Per feedback: `/final-review` is mandatory before the single commit.
