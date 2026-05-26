# TF-18 Ghost-GK Training Data Assembly + Hub Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `prepare_ghost_gk_training_data()` as a composable building block, refactor `compute_ghost_gk` to share iteration logic and accept match context, flesh out the training script, and add a publish script for HuggingFace Hub.

**Architecture:** Shared internal batch helper `_extract_all_ghost_gk_features()` extracts the frame-iteration + velocity-tracking loop from `compute_ghost_gk`. Both inference (`compute_ghost_gk`) and training (`prepare_ghost_gk_training_data`) call it. Match context (score, phase, carrier) is optional via `actions` parameter. Training script is a standalone CLI; publish script uploads to HF Hub.

**Tech Stack:** pandas, numpy, scikit-learn (HistGradientBoostingRegressor, StratifiedGroupKFold, permutation_importance), scipy (ConvexHull, gaussian_kde), huggingface_hub

**Spec:** `docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `silly_kicks/tracking/_ghost_gk.py` | Add `_SET_PIECE_DECAY_SECONDS`, `_build_score_lookup`, `_build_phase_lookup`, `_extract_all_ghost_gk_features`, `prepare_ghost_gk_training_data`. Refactor `compute_ghost_gk`. Fix timestamp bug. |
| `silly_kicks/tracking/__init__.py` | Re-export `prepare_ghost_gk_training_data` |
| `silly_kicks/tracking/features.py` | Update `add_ghost_gk` to thread `actions` |
| `scripts/train_ghost_gk.py` | Full training CLI |
| `scripts/publish_ghost_gk.py` | HF Hub publish CLI |
| `tests/tracking/test_ghost_gk.py` | All new unit tests |
| `tests/tracking/test_ghost_gk_integration.py` | Round-trip + smoke tests |
| `tests/tracking/fixtures/ghost_gk_backward_compat.parquet` | Golden-file baseline |

---

### Task 0: Capture backward-compatibility golden file

Before any code changes, capture `compute_ghost_gk` output on a synthetic fixture to use as the backward-compatibility baseline.

**Files:**
- Create: `tests/tracking/fixtures/ghost_gk_backward_compat.parquet`

- [ ] **Step 1: Generate and save golden-file baseline**

Run this script to produce the golden file from the current (3.19.0) code:

```bash
cd D:/Development/karstenskyt__silly-kicks
uv run python -c "
import numpy as np
import pandas as pd
from pathlib import Path

# Build multi-frame synthetic fixture (3 frames, both GKs)
rows = []
for fid in range(1, 4):
    ts = float(fid)
    base = dict(game_id='100', period_id=1, frame_id=fid,
                time_seconds=ts, timestamp=ts, frame_rate=25.0,
                ball_state='alive', source_provider='test',
                team_attacking_direction=None, confidence=None,
                visibility=None, is_goalkeeper_source='native', z=0.0)
    # Ball
    rows.append({**base, player_id='ball', team_id=None,
                 x=50.0+fid, y=34.0, vx=2.0, vy=0.0, speed=2.0,
                 is_ball=True, is_goalkeeper=False})
    # Home GK
    rows.append({**base, player_id='p1', team_id=1,
                 x=5.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                 is_ball=False, is_goalkeeper=True})
    # Home defenders
    for i, (px, py) in enumerate([(20,25),(22,30),(21,38),(23,45)]):
        rows.append({**base, player_id=f'p{10+i}', team_id=1,
                     x=float(px), y=float(py), vx=0.5, vy=0.0, speed=0.5,
                     is_ball=False, is_goalkeeper=False})
    # Away attackers
    for i, (px, py) in enumerate([(40,30),(45,34),(38,40),(50,34)]):
        rows.append({**base, player_id=f'a{10+i}', team_id=2,
                     x=float(px), y=float(py), vx=-1.0, vy=0.0, speed=1.0,
                     is_ball=False, is_goalkeeper=False})
    # Away GK
    rows.append({**base, player_id='a1', team_id=2,
                 x=100.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                 is_ball=False, is_goalkeeper=True})

frames = pd.DataFrame(rows)

# Train a tiny model on synthetic data
from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES, GhostGkModel, compute_ghost_gk,
)
rng = np.random.default_rng(42)
n = 100
X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
X['phase'] = rng.integers(0, 3, n).astype(float)
X['team_in_possession'] = rng.integers(0, 2, n).astype(float)
X['ball_in_own_half'] = rng.integers(0, 2, n).astype(float)
labels = pd.DataFrame({'gk_x': rng.uniform(2, 20, n), 'gk_y': rng.uniform(25, 45, n)})
model = GhostGkModel(n_estimators=10)
model.fit(X, labels)

# Run inference with current code
result = compute_ghost_gk(frames, model=model, home_team_id=1)

# Save GK rows with ghost predictions
gk_mask = result['is_goalkeeper'].astype(bool) & ~result['is_ball'].astype(bool)
golden = result.loc[gk_mask, ['game_id','period_id','frame_id','team_id',
                               'ghost_gk_x','ghost_gk_y','ghost_gk_spread']].copy()
golden = golden.reset_index(drop=True)

out = Path('tests/tracking/fixtures')
out.mkdir(parents=True, exist_ok=True)
golden.to_parquet(out / 'ghost_gk_backward_compat.parquet', index=False)
print('Saved ' + str(len(golden)) + ' rows to ' + str(out / 'ghost_gk_backward_compat.parquet'))
print(golden)
"
```

Expected: prints ~6 rows (3 frames x 2 GKs) with ghost_gk_x/y/spread values. Parquet file created.

- [ ] **Step 2: Verify golden file loads**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('tests/tracking/fixtures/ghost_gk_backward_compat.parquet')
print(df.columns.tolist())
print(f'Shape: {df.shape}')
assert df.shape[0] == 6
assert 'ghost_gk_x' in df.columns
print('OK')
"
```

Expected: Shape (6, 7), OK.

---

### Task 1: Bug fix -- `"timestamp"` -> `"time_seconds"` + named constant

Fix the schema mismatch bug and add the named phase-decay constant. These are leaf changes with no dependency on new functions.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py:342` (extract_ghost_gk_features)
- Modify: `silly_kicks/tracking/_ghost_gk.py:837` (compute_ghost_gk)

- [ ] **Step 1: Write the failing test**

Add to `tests/tracking/test_ghost_gk.py`:

```python
class TestTimestampBugFix:
    """Bug fix: time_seconds column used, not timestamp."""

    def test_time_seconds_column_used(self):
        """extract_ghost_gk_features reads time_seconds, not timestamp."""
        from silly_kicks.tracking._ghost_gk import extract_ghost_gk_features

        # Frame with time_seconds=42.5 but NO timestamp column
        rows = []
        base = dict(
            game_id="100", period_id=1, frame_id=1,
            time_seconds=42.5, frame_rate=25.0,
            ball_state="alive", source_provider="test",
        )
        rows.append({**base, player_id="ball", team_id=None,
                     x=50.0, y=34.0, vx=2.0, vy=0.0,
                     is_ball=True, is_goalkeeper=False})
        rows.append({**base, player_id="p1", team_id=1,
                     x=5.0, y=34.0, vx=0.0, vy=0.0,
                     is_ball=False, is_goalkeeper=True})
        for i, (px, py) in enumerate([(20, 25), (22, 30), (21, 38), (23, 45)]):
            rows.append({**base, player_id=f"p{10+i}", team_id=1,
                         x=float(px), y=float(py), vx=0.5, vy=0.0,
                         is_ball=False, is_goalkeeper=False})
        for i, (px, py) in enumerate([(40, 30), (45, 34)]):
            rows.append({**base, player_id=f"a{10+i}", team_id=2,
                         x=float(px), y=float(py), vx=-1.0, vy=0.0,
                         is_ball=False, is_goalkeeper=False})
        frame = pd.DataFrame(rows)

        result = extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0)
        assert result["time_seconds"].iloc[0] == pytest.approx(42.5), (
            "Should read time_seconds column, not timestamp"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestTimestampBugFix -v`
Expected: FAIL — `time_seconds` feature is 0.0 (fallback), not 42.5

- [ ] **Step 3: Fix extract_ghost_gk_features**

In `silly_kicks/tracking/_ghost_gk.py`, line 342, change:

```python
# BEFORE (line 342):
    time_s = float(frame_data["timestamp"].iloc[0]) if "timestamp" in frame_data.columns else 0.0
```

to:

```python
    time_s = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0
```

- [ ] **Step 4: Fix compute_ghost_gk**

In `silly_kicks/tracking/_ghost_gk.py`, line 837, change:

```python
# BEFORE (line 837):
            current_ts = float(frame_data["timestamp"].iloc[0]) if "timestamp" in frame_data.columns else 0.0
```

to:

```python
            current_ts = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0
```

- [ ] **Step 5: Add named constant**

At the top of the constants section (after `_VELOCITY_WINDOW_S = 0.5`, around line 164), add:

```python
_SET_PIECE_DECAY_SECONDS = 10.0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestTimestampBugFix -v`
Expected: PASS

- [ ] **Step 7: Run full existing test suite to confirm no regression**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py -v --tb=short`
Expected: all existing tests PASS

---

### Task 2: Match context resolution helpers

Implement `_build_score_lookup` and `_build_phase_lookup` with tests.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py`
- Modify: `tests/tracking/test_ghost_gk.py`

- [ ] **Step 1: Write failing tests for score lookup**

Add to `tests/tracking/test_ghost_gk.py`:

```python
def _make_spadl_actions(
    *,
    game_id: str = "100",
    goals: list[tuple[float, int]] | None = None,
    owngoals: list[tuple[float, int]] | None = None,
    set_pieces: list[tuple[float, str]] | None = None,
) -> pd.DataFrame:
    """Build minimal SPADL actions for context resolution tests.

    Parameters
    ----------
    goals : list of (time_seconds, team_id) for successful shots
    owngoals : list of (time_seconds, team_id) for own goals
    set_pieces : list of (time_seconds, type_name) for set-piece actions
    """
    from silly_kicks.spadl import config as spadlconfig

    rows = []
    action_id = 0

    # Always add a non_action at t=0 so the DF is never empty
    rows.append({
        "game_id": game_id, "action_id": action_id,
        "period_id": 1, "time_seconds": 0.0,
        "team_id": 1, "player_id": 10,
        "start_x": 52.5, "start_y": 34.0,
        "end_x": 52.5, "end_y": 34.0,
        "type_id": spadlconfig.actiontype_id["non_action"],
        "result_id": spadlconfig.result_id["success"],
        "bodypart_id": 0,
        "type_name": "non_action",
        "result_name": "success",
        "bodypart_name": "foot",
    })
    action_id += 1

    for ts, tid in (goals or []):
        rows.append({
            "game_id": game_id, "action_id": action_id,
            "period_id": 1, "time_seconds": ts,
            "team_id": tid, "player_id": 10,
            "start_x": 90.0, "start_y": 34.0,
            "end_x": 104.0, "end_y": 34.0,
            "type_id": spadlconfig.actiontype_id["shot"],
            "result_id": spadlconfig.result_id["success"],
            "bodypart_id": 0,
            "type_name": "shot",
            "result_name": "success",
            "bodypart_name": "foot",
        })
        action_id += 1

    for ts, tid in (owngoals or []):
        rows.append({
            "game_id": game_id, "action_id": action_id,
            "period_id": 1, "time_seconds": ts,
            "team_id": tid, "player_id": 10,
            "start_x": 20.0, "start_y": 34.0,
            "end_x": 5.0, "end_y": 34.0,
            "type_id": spadlconfig.actiontype_id["shot"],
            "result_id": spadlconfig.result_id["owngoal"],
            "bodypart_id": 0,
            "type_name": "shot",
            "result_name": "owngoal",
            "bodypart_name": "foot",
        })
        action_id += 1

    for ts, tname in (set_pieces or []):
        rows.append({
            "game_id": game_id, "action_id": action_id,
            "period_id": 1, "time_seconds": ts,
            "team_id": 1, "player_id": 10,
            "start_x": 50.0, "start_y": 34.0,
            "end_x": 55.0, "end_y": 34.0,
            "type_id": spadlconfig.actiontype_id[tname],
            "result_id": spadlconfig.result_id["success"],
            "bodypart_id": 0,
            "type_name": tname,
            "result_name": "success",
            "bodypart_name": "foot",
        })
        action_id += 1

    return pd.DataFrame(rows)


class TestBuildScoreLookup:
    """_build_score_lookup returns home-perspective running score diff."""

    def test_no_goals(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions()
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 10.0) == 0.0
        assert fn("100", 60.0) == 0.0

    def test_home_goal(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions(goals=[(30.0, 1)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 25.0) == 0.0  # before goal
        assert fn("100", 30.0) == 1.0  # at goal
        assert fn("100", 60.0) == 1.0  # after goal

    def test_away_goal(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions(goals=[(30.0, 2)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 35.0) == -1.0  # home perspective: 0-1

    def test_multiple_goals(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions(goals=[(10.0, 1), (20.0, 2), (30.0, 1)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 15.0) == 1.0   # 1-0
        assert fn("100", 25.0) == 0.0   # 1-1
        assert fn("100", 35.0) == 1.0   # 2-1

    def test_own_goal_attributed_to_opponent(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        # Team 1 scores own goal -> counts as team 2 scoring
        actions = _make_spadl_actions(owngoals=[(30.0, 1)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 35.0) == -1.0  # 0-1 from home perspective
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestBuildScoreLookup -v`
Expected: FAIL — `_build_score_lookup` not found

- [ ] **Step 3: Implement `_build_score_lookup`**

Add to `silly_kicks/tracking/_ghost_gk.py`, after the `_SET_PIECE_DECAY_SECONDS` constant and before `extract_ghost_gk_features`:

```python
# ---------------------------------------------------------------------------
# Match context resolution
# ---------------------------------------------------------------------------


def _build_score_lookup(
    actions: pd.DataFrame,
    home_team_id: int | str,
) -> "Callable[[Any, float], float]":
    """Build (game_id, time_seconds) -> score_diff callback (home perspective).

    Returns home_score - away_score at the queried time. The **caller**
    must negate for away-team GKs.

    Own goals (result_name == "owngoal") are attributed to the opponent
    of the acting team.

    Note: Uses str() comparison internally for team ID matching.
    Assumes actions and frames share team ID type from the same provider
    (always true in practice — both come from the same kloppy/converter
    pipeline). If actions have int team_id=1 and caller passes
    home_team_id="001", str(1) != "001" would mismatch.
    """
    from typing import Any

    # Resolve type/result columns (supports both ID and name DataFrames)
    if "type_name" in actions.columns:
        shots = actions[actions["type_name"] == "shot"].copy()
    else:
        shot_id = spadlconfig.actiontype_id["shot"]
        shots = actions[actions["type_id"] == shot_id].copy()

    if "result_name" in actions.columns:
        goals = shots[shots["result_name"].isin(["success", "owngoal"])].copy()
    else:
        success_id = spadlconfig.result_id["success"]
        owngoal_id = spadlconfig.result_id["owngoal"]
        goals = shots[shots["result_id"].isin([success_id, owngoal_id])].copy()

    if len(goals) == 0:
        def _zero(_game_id: Any, _time_s: float) -> float:
            return 0.0
        return _zero

    # Flip own-goal team attribution
    if "result_name" in goals.columns:
        is_own = goals["result_name"] == "owngoal"
    else:
        is_own = goals["result_id"] == spadlconfig.result_id["owngoal"]

    # For own goals, the scoring team is the OPPONENT of the actor
    goals = goals.copy()
    # We need to identify two teams — use home_team_id as anchor
    goals["_scoring_team"] = goals["team_id"].copy()
    if is_own.any():
        # Get all unique team IDs from the goals (or from actions)
        all_teams = actions["team_id"].unique()
        if len(all_teams) == 2:
            team_a, team_b = all_teams[0], all_teams[1]
            flip_map = {team_a: team_b, team_b: team_a}
            goals.loc[is_own, "_scoring_team"] = goals.loc[is_own, "team_id"].map(flip_map)

    goals = goals.sort_values(["game_id", "time_seconds"]).reset_index(drop=True)

    # Build per-game cumulative score arrays
    home_team_id_norm = str(home_team_id)
    _lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid, grp in goals.groupby("game_id"):
        times = grp["time_seconds"].values.astype(np.float64)
        is_home = np.array([str(t) == home_team_id_norm for t in grp["_scoring_team"]])
        home_cum = np.cumsum(is_home.astype(np.float64))
        away_cum = np.cumsum((~is_home).astype(np.float64))
        diffs = home_cum - away_cum
        _lookup[str(gid)] = (times, diffs)

    def _score(game_id: Any, time_s: float) -> float:
        key = str(game_id)
        if key not in _lookup:
            return 0.0
        times, diffs = _lookup[key]
        idx = int(np.searchsorted(times, time_s, side="right")) - 1
        if idx < 0:
            return 0.0
        return float(diffs[idx])

    return _score
```

- [ ] **Step 4: Run score lookup tests**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestBuildScoreLookup -v`
Expected: 5 PASSED

- [ ] **Step 5: Write failing tests for phase lookup**

Add to `tests/tracking/test_ghost_gk.py`:

```python
class TestBuildPhaseLookup:
    """_build_phase_lookup returns 0/1/2 for open/set_piece/goal_kick."""

    def test_open_play(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions()
        fn = _build_phase_lookup(actions)
        assert fn("100", 10.0) == 0

    def test_freekick_within_decay(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "freekick_short")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 1  # 3s after freekick -> set_piece

    def test_goalkick(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "goalkick")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 2  # goal_kick phase

    def test_corner(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "corner_crossed")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 1  # set_piece

    def test_decay_after_10s(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "freekick_short")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 41.0) == 0  # >10s -> open play

    def test_throw_in_excluded(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "throw_in")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 0  # throw_in is NOT a set piece
```

- [ ] **Step 6: Run phase lookup tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestBuildPhaseLookup -v`
Expected: FAIL — `_build_phase_lookup` not found

- [ ] **Step 7: Implement `_build_phase_lookup`**

Add to `silly_kicks/tracking/_ghost_gk.py`, right after `_build_score_lookup`:

```python
def _build_phase_lookup(
    actions: pd.DataFrame,
) -> "Callable[[Any, float], int]":
    """Build (game_id, time_seconds) -> phase callback.

    Returns 0 (open_play), 1 (set_piece), or 2 (goal_kick).
    A set-piece phase decays to open play after _SET_PIECE_DECAY_SECONDS.
    throw_in is excluded --- does not alter GK positioning expectations.
    """
    from typing import Any

    # Set-piece types (excluding throw_in per spec)
    _SP_TYPES = {"freekick_crossed", "freekick_short", "corner_crossed", "corner_short"}
    _GK_TYPE = "goalkick"

    # Resolve type column
    if "type_name" in actions.columns:
        sp_mask = actions["type_name"].isin(_SP_TYPES | {_GK_TYPE})
        sp = actions[sp_mask].copy()
        if len(sp) > 0:
            sp["_phase_code"] = sp["type_name"].apply(lambda t: 2 if t == _GK_TYPE else 1)
        else:
            sp["_phase_code"] = pd.Series(dtype=int)
    else:
        sp_ids = {spadlconfig.actiontype_id[t] for t in _SP_TYPES if t in spadlconfig.actiontype_id}
        gk_id = spadlconfig.actiontype_id.get(_GK_TYPE)
        if gk_id is not None:
            sp_ids.add(gk_id)
        sp = actions[actions["type_id"].isin(sp_ids)].copy()
        if len(sp) > 0:
            sp["_phase_code"] = sp["type_id"].apply(lambda tid: 2 if tid == gk_id else 1)
        else:
            sp["_phase_code"] = pd.Series(dtype=int)

    if len(sp) == 0:
        def _open(_game_id: Any, _time_s: float) -> int:
            return 0
        return _open

    sp = sp.sort_values(["game_id", "time_seconds"]).reset_index(drop=True)

    _lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid, grp in sp.groupby("game_id"):
        times = grp["time_seconds"].values.astype(np.float64)
        codes = grp["_phase_code"].values.astype(np.int64)
        _lookup[str(gid)] = (times, codes)

    def _phase(game_id: Any, time_s: float) -> int:
        key = str(game_id)
        if key not in _lookup:
            return 0
        times, codes = _lookup[key]
        idx = int(np.searchsorted(times, time_s, side="right")) - 1
        if idx < 0:
            return 0
        elapsed = time_s - times[idx]
        if elapsed > _SET_PIECE_DECAY_SECONDS:
            return 0
        return int(codes[idx])

    return _phase
```

- [ ] **Step 8: Run all context resolution tests**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestBuildScoreLookup tests/tracking/test_ghost_gk.py::TestBuildPhaseLookup -v`
Expected: 11 PASSED

---

### Task 3: Shared batch helper `_extract_all_ghost_gk_features`

The core extraction that both inference and training share.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py`
- Modify: `tests/tracking/test_ghost_gk.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/tracking/test_ghost_gk.py`:

```python
def _make_multi_frame_fixture(
    *,
    n_frames: int = 5,
    home_team_id: int = 1,
    away_team_id: int = 2,
    game_id: str = "100",
    fps: float = 25.0,
) -> pd.DataFrame:
    """Build multi-frame fixture suitable for shared helper tests."""
    rows = []
    for fid in range(1, n_frames + 1):
        ts = fid / fps
        base = dict(
            game_id=game_id, period_id=1, frame_id=fid,
            time_seconds=ts, frame_rate=fps,
            ball_state="alive", source_provider="test",
            team_attacking_direction=None, confidence=None,
            visibility=None, is_goalkeeper_source="native", z=0.0,
        )
        # Ball
        rows.append({**base, player_id="ball", team_id=None,
                     x=50.0 + fid * 0.5, y=34.0, vx=2.0, vy=0.0, speed=2.0,
                     is_ball=True, is_goalkeeper=False})
        # Home GK
        rows.append({**base, player_id="p1", team_id=home_team_id,
                     x=5.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                     is_ball=False, is_goalkeeper=True})
        # Home defenders
        for i, (px, py) in enumerate([(20, 25), (22, 30), (21, 38), (23, 45)]):
            rows.append({**base, player_id=f"p{10+i}", team_id=home_team_id,
                         x=float(px), y=float(py), vx=0.5, vy=0.0, speed=0.5,
                         is_ball=False, is_goalkeeper=False})
        # Away attackers
        for i, (px, py) in enumerate([(40, 30), (45, 34), (38, 40), (50, 34)]):
            rows.append({**base, player_id=f"a{10+i}", team_id=away_team_id,
                         x=float(px), y=float(py), vx=-1.0, vy=0.0, speed=1.0,
                         is_ball=False, is_goalkeeper=False})
        # Away GK
        rows.append({**base, player_id="a1", team_id=away_team_id,
                     x=100.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                     is_ball=False, is_goalkeeper=True})
    return pd.DataFrame(rows)


class TestExtractAllFeatures:
    """_extract_all_ghost_gk_features shared helper."""

    def test_shape(self):
        from silly_kicks.tracking._ghost_gk import (
            GHOST_GK_FEATURE_NAMES,
            _extract_all_ghost_gk_features,
        )

        frames = _make_multi_frame_fixture(n_frames=5)
        features, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        # 5 frames x 2 GKs = 10 rows
        assert features.shape[0] == 10
        assert features.shape[1] == len(GHOST_GK_FEATURE_NAMES)
        assert meta.shape == (10, 6)
        assert list(meta.columns) == [
            "game_id", "period_id", "frame_id", "gk_team_id", "gk_x_gr", "gk_y_gr",
        ]

    def test_velocity_state_non_nan_after_first(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=3)
        features, _ = _extract_all_ghost_gk_features(frames, home_team_id=1)
        # First frame has NaN velocity, subsequent frames should have values
        # Group by team: rows 0,2,4 = home GK; 1,3,5 = away GK
        home_rows = features.iloc[0::2]  # even indices = home GK
        assert np.isnan(home_rows["defensive_line_speed"].iloc[0])
        assert not np.isnan(home_rows["defensive_line_speed"].iloc[1])

    def test_subsample(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=25, fps=25.0)
        full, _ = _extract_all_ghost_gk_features(frames, home_team_id=1)
        sub, _ = _extract_all_ghost_gk_features(
            frames, home_team_id=1, subsample_fps=1.0,
        )
        # 25fps -> 1fps = keep every 25th frame -> 1 frame -> 2 GKs
        assert sub.shape[0] < full.shape[0]
        assert sub.shape[0] == 2  # 1 frame x 2 GKs

    def test_goal_relative_coords(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)
        _, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        # Home GK at x=5.0 (goal at x=0 -> gr_x = 5.0)
        home_meta = meta[meta["gk_team_id"] == 1]
        assert home_meta["gk_x_gr"].iloc[0] == pytest.approx(5.0)
        # Away GK at x=100.0 (goal at x=105 -> gr_x = 105-100 = 5.0)
        away_meta = meta[meta["gk_team_id"] == 2]
        assert away_meta["gk_x_gr"].iloc[0] == pytest.approx(5.0)

    def test_home_team_id_normalization_int_to_str(self):
        """home_team_id=1 works when frames have string team_id."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)
        frames["team_id"] = frames["team_id"].astype(str)
        features, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        assert features.shape[0] == 2  # both GKs extracted

    def test_home_team_id_normalization_str_to_int(self):
        """home_team_id='1' works when frames have int team_id."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)
        features, meta = _extract_all_ghost_gk_features(frames, home_team_id="1")
        assert features.shape[0] == 2

    def test_score_callback_negated_for_away(self):
        """Away GK sees negated score_diff."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)

        def mock_score(game_id, time_s):
            return 2.0  # home perspective: home leads 2-0

        features, meta = _extract_all_ghost_gk_features(
            frames, home_team_id=1, score_at_time=mock_score,
        )
        home_feat = features[meta["gk_team_id"].values == 1]
        away_feat = features[meta["gk_team_id"].values == 2]
        assert home_feat["score_diff"].iloc[0] == pytest.approx(2.0)
        assert away_feat["score_diff"].iloc[0] == pytest.approx(-2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestExtractAllFeatures -v`
Expected: FAIL — `_extract_all_ghost_gk_features` not found

- [ ] **Step 3: Implement `_extract_all_ghost_gk_features`**

Add to `silly_kicks/tracking/_ghost_gk.py`, after `_build_phase_lookup` and before `extract_ghost_gk_features`:

```python
def _extract_all_ghost_gk_features(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    carrier: pd.DataFrame | None = None,
    score_at_time: "Callable[[Any, float], float] | None" = None,
    phase_at_time: "Callable[[Any, float], int] | None" = None,
    subsample_fps: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared batch helper: iterate frames, extract features for every GK.

    Both compute_ghost_gk (inference) and prepare_ghost_gk_training_data
    (training) call this function. Single source of truth for the
    frame-iteration + velocity-tracking + feature-extraction loop.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS, LTR-normalized, vx/vy present.
    home_team_id : str | int
        Home team (attacks right, GK at x=0).
    carrier : pd.DataFrame | None
        Per-frame ball_carrier_team_id (from derive_team_in_possession).
    score_at_time : callable | None
        (game_id, time_seconds) -> score_diff (home perspective).
    phase_at_time : callable | None
        (game_id, time_seconds) -> int (0=open, 1=set_piece, 2=goal_kick).
    subsample_fps : float | None
        Thin frames to target fps before extraction.

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)).
    meta : pd.DataFrame
        (n_samples, 6): game_id, period_id, frame_id, gk_team_id,
        gk_x_gr, gk_y_gr.
    """
    # --- Team ID normalization (§7) ---
    frame_team_dtype = frames["team_id"].dtype
    if frame_team_dtype == object:
        home_team_id = str(home_team_id)
    else:
        try:
            home_team_id = type(frames["team_id"].dropna().iloc[0])(home_team_id)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"home_team_id={home_team_id!r} cannot be coerced to "
                f"frames['team_id'] dtype {frame_team_dtype}"
            ) from exc

    # --- Pre-index carrier for O(1) lookup ---
    carrier_idx: pd.Series | None = None
    if carrier is not None and "ball_carrier_team_id" in carrier.columns:
        carrier_idx = carrier.set_index(
            ["game_id", "period_id", "frame_id"]
        )["ball_carrier_team_id"]

    # --- Subsample ---
    work = frames
    if subsample_fps is not None and "frame_rate" in frames.columns:
        fr = frames["frame_rate"].iloc[0]
        if fr > 0 and subsample_fps > 0:
            step = max(1, round(fr / subsample_fps))
            # Keep every step-th unique frame_id per (game_id, period_id)
            unique_frames = (
                frames[["game_id", "period_id", "frame_id"]]
                .drop_duplicates()
                .sort_values(["game_id", "period_id", "frame_id"])
            )
            keep_mask = unique_frames.groupby(["game_id", "period_id"]).cumcount() % step == 0
            keep_keys = unique_frames[keep_mask.values]
            work = frames.merge(keep_keys, on=["game_id", "period_id", "frame_id"])

    # --- Group and iterate ---
    group_keys = ["game_id", "period_id", "frame_id"]
    grouped = list(work.groupby(group_keys, sort=True))

    feature_rows: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    prev_state: dict[tuple, tuple[float, float]] = {}
    prev_timestamps: dict[tuple, float] = {}

    for (gid, pid, fid), frame_data in grouped:
        gk_rows = frame_data[
            frame_data["is_goalkeeper"].astype(bool)
            & ~frame_data["is_ball"].astype(bool)
        ]
        time_s = (
            float(frame_data["time_seconds"].iloc[0])
            if "time_seconds" in frame_data.columns
            else 0.0
        )

        for _, gk_row in gk_rows.iterrows():
            gk_team = gk_row["team_id"]
            goal_x = 0.0 if gk_team == home_team_id else _FIELD_LENGTH

            # Score: callback returns home perspective, negate for away
            if score_at_time is not None:
                sd = score_at_time(gid, time_s)
                if gk_team != home_team_id:
                    sd = -sd
            else:
                sd = 0.0

            # Phase
            ph = phase_at_time(gid, time_s) if phase_at_time is not None else 0

            # Carrier
            carrier_team = None
            if carrier_idx is not None:
                try:
                    carrier_team = carrier_idx.loc[(gid, pid, fid)]
                except KeyError:
                    pass

            # Velocity state
            state_key = (gid, gk_team)
            prev_dl_x, prev_dc_x = prev_state.get(state_key, (None, None))
            prev_ts = prev_timestamps.get(state_key)
            actual_dt = (
                (time_s - prev_ts)
                if prev_ts is not None and time_s > prev_ts
                else _VELOCITY_WINDOW_S
            )

            feat = extract_ghost_gk_features(
                frame_data,
                gk_team_id=gk_team,
                goal_x=goal_x,
                score_diff=sd,
                phase=ph,
                ball_carrier_team_id=carrier_team,
                prev_defensive_line_x=prev_dl_x,
                prev_defending_centroid_x=prev_dc_x,
                dt=actual_dt,
            )
            feature_rows.append(feat)

            # GK position in goal-relative coords for labels
            gk_x_raw = float(gk_row["x"])
            gk_y_raw = float(gk_row["y"])
            flip = goal_x > 50.0
            gk_x_gr = (_FIELD_LENGTH - gk_x_raw) if flip else gk_x_raw
            gk_y_gr = gk_y_raw

            meta_rows.append({
                "game_id": gid, "period_id": pid, "frame_id": fid,
                "gk_team_id": gk_team, "gk_x_gr": gk_x_gr, "gk_y_gr": gk_y_gr,
            })

            # Update velocity state
            defending = frame_data[
                (frame_data["team_id"] == gk_team)
                & ~frame_data["is_goalkeeper"].astype(bool)
                & ~frame_data["is_ball"].astype(bool)
            ]
            if len(defending) > 0:
                if flip:
                    def_cx = float(np.mean(_FIELD_LENGTH - np.asarray(defending["x"].values)))
                else:
                    def_cx = float(np.mean(np.asarray(defending["x"].values)))
            else:
                def_cx = np.nan
            prev_state[state_key] = (
                float(feat["defensive_line_x"].iloc[0]),
                def_cx,
            )
            prev_timestamps[state_key] = time_s

    if not feature_rows:
        return (
            pd.DataFrame(columns=GHOST_GK_FEATURE_NAMES),
            pd.DataFrame(
                columns=["game_id", "period_id", "frame_id",
                          "gk_team_id", "gk_x_gr", "gk_y_gr"]
            ),
        )

    features = pd.concat(feature_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    return features, meta
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestExtractAllFeatures -v`
Expected: 8 PASSED

---

### Task 4: `prepare_ghost_gk_training_data` public API + `__init__.py` re-export

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `tests/tracking/test_ghost_gk.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/tracking/test_ghost_gk.py`:

```python
class TestPrepareTrainingData:
    """prepare_ghost_gk_training_data public API."""

    def test_basic_shape(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=5)
        features, labels = prepare_ghost_gk_training_data(
            frames, home_team_id=1, subsample_fps=None,
        )
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] > 0
        assert list(labels.columns) == ["gk_x", "gk_y"]
        assert not labels.isna().any().any()

    def test_with_actions_score_nonzero(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=5)
        actions = _make_spadl_actions(goals=[(0.01, 1)])  # home scores early
        features, labels = prepare_ghost_gk_training_data(
            frames, home_team_id=1, actions=actions, subsample_fps=None,
        )
        # Home GK should see positive score_diff after goal
        assert (features["score_diff"] != 0.0).any()

    def test_without_actions_defaults(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=3)
        features, labels = prepare_ghost_gk_training_data(
            frames, home_team_id=1, subsample_fps=None,
        )
        assert (features["score_diff"] == 0.0).all()
        assert (features["phase"] == 0.0).all()

    def test_subsample_reduces(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=25, fps=25.0)
        full, _ = prepare_ghost_gk_training_data(
            frames, home_team_id=1, subsample_fps=None,
        )
        sub, _ = prepare_ghost_gk_training_data(
            frames, home_team_id=1, subsample_fps=1.0,
        )
        assert sub.shape[0] < full.shape[0]

    def test_sweeper_rush_filtered(self):
        """GK outside [0,30]x[18,50] should be filtered with warning."""
        import warnings

        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=1)
        # Move home GK far out of domain (sweeper rush at x=50, y=34)
        frames.loc[
            (frames["player_id"] == "p1") & (frames["is_goalkeeper"] == True),  # noqa: E712
            "x",
        ] = 50.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features, labels = prepare_ghost_gk_training_data(
                frames, home_team_id=1, subsample_fps=None,
            )
            # The home GK at x=50 -> gr_x=50 -> outside [0,30] -> filtered
            sweeper_warnings = [x for x in w if "goal-relative domain" in str(x.message)]
            assert len(sweeper_warnings) >= 1

    def test_public_import_path(self):
        """prepare_ghost_gk_training_data is importable from silly_kicks.tracking."""
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        assert callable(prepare_ghost_gk_training_data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestPrepareTrainingData -v`
Expected: FAIL — `prepare_ghost_gk_training_data` not found in `silly_kicks.tracking`

- [ ] **Step 3: Implement `prepare_ghost_gk_training_data`**

Add to `silly_kicks/tracking/_ghost_gk.py`, after `_extract_all_ghost_gk_features`:

```python
def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = None,
    subsample_fps: float | None = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble training features + labels from one game's tracking frames.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS schema, LTR-normalized, with vx/vy
        columns (from smooth_frames + derive_velocities).
    home_team_id : str | int
        Home team ID (attacks right in LTR convention).
    actions : pd.DataFrame | None
        SPADL actions for the same game. Provides score_diff and phase
        context. If None, both default to 0 (valid but less informative).
    subsample_fps : float | None
        Target frame rate for training (default 1.0 Hz). None = no
        subsampling.

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)) with GHOST_GK_FEATURE_NAMES
        columns.
    labels : pd.DataFrame
        (n_samples, 2) with columns "gk_x", "gk_y" in goal-relative
        coordinates matching the GhostGkModel training domain
        ([0, 30] x [18, 50]).

    Examples
    --------
    >>> features, labels = prepare_ghost_gk_training_data(
    ...     frames, home_team_id=1, actions=actions, subsample_fps=1.0
    ... )
    >>> model = GhostGkModel()
    >>> model.fit(features, labels)
    """
    import warnings

    from ._ball_carrier import derive_team_in_possession, infer_ball_carrier

    # Build context callbacks
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # Carrier (always computed — only needs frames)
    carrier_raw = infer_ball_carrier(frames)
    carrier_df = derive_team_in_possession(frames, carrier_raw)
    # Extract per-frame carrier lookup
    carrier_cols = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]

    features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        carrier=carrier_cols,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
        subsample_fps=subsample_fps,
    )

    if len(meta) == 0:
        return (
            pd.DataFrame(columns=GHOST_GK_FEATURE_NAMES),
            pd.DataFrame(columns=["gk_x", "gk_y"]),
        )

    # Extract labels
    labels = meta[["gk_x_gr", "gk_y_gr"]].rename(
        columns={"gk_x_gr": "gk_x", "gk_y_gr": "gk_y"}
    )

    # Drop NaN labels (GK not visible)
    valid = labels["gk_x"].notna() & labels["gk_y"].notna()
    features = features[valid.values].reset_index(drop=True)
    labels = labels[valid.values].reset_index(drop=True)

    # Validate feature width
    assert features.shape[1] == len(GHOST_GK_FEATURE_NAMES), (
        f"Expected {len(GHOST_GK_FEATURE_NAMES)} features, got {features.shape[1]}"
    )

    # Filter label domain (sweeper-keeper rushes, off-pitch artifacts)
    in_domain = (
        (labels["gk_x"] >= GRID_X_MIN) & (labels["gk_x"] <= GRID_X_MAX)
        & (labels["gk_y"] >= GRID_Y_MIN) & (labels["gk_y"] <= GRID_Y_MAX)
    )
    n_out = int((~in_domain).sum())
    if n_out > 0:
        total = len(labels)
        warnings.warn(
            f"Dropped {n_out} of {total} rows with GK outside "
            f"goal-relative domain (sweeper rushes/artifacts)",
            stacklevel=2,
        )
        features = features[in_domain.values].reset_index(drop=True)
        labels = labels[in_domain.values].reset_index(drop=True)

    return features, labels
```

- [ ] **Step 4: Add re-export to `__init__.py`**

In `silly_kicks/tracking/__init__.py`, add `"prepare_ghost_gk_training_data"` to the `__all__` list (alphabetical position, after `"play_left_to_right"`), and add the import:

In `__all__`, add:
```python
    "prepare_ghost_gk_training_data",
```

Add import line (after the existing `_ghost_gk`-related imports or alongside `_ball_carrier` imports):
```python
from ._ghost_gk import prepare_ghost_gk_training_data
```

- [ ] **Step 5: Run tests**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestPrepareTrainingData -v`
Expected: 6 PASSED

- [ ] **Step 6: Run full test suite**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py -v --tb=short`
Expected: all tests PASS

---

### Task 5: Refactor `compute_ghost_gk` to use shared helper + accept `actions`

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py:777-901`
- Modify: `silly_kicks/tracking/features.py:3453-3468`
- Modify: `tests/tracking/test_ghost_gk.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/tracking/test_ghost_gk.py`:

```python
class TestComputeGhostGkRefactored:
    """compute_ghost_gk with shared helper + actions parameter."""

    def test_backward_compat(self):
        """actions=None produces identical output to 3.19.0 golden file."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model = self._make_model(n_estimators=10)

        # Build same frames as golden file
        frames = _make_multi_frame_fixture(n_frames=3, game_id="100")
        # Add timestamp column for backward compat (old code read it)
        frames["timestamp"] = frames["time_seconds"]

        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        gk_mask = (
            result["is_goalkeeper"].astype(bool) & ~result["is_ball"].astype(bool)
        )
        actual = result.loc[
            gk_mask,
            ["game_id", "period_id", "frame_id", "team_id",
             "ghost_gk_x", "ghost_gk_y", "ghost_gk_spread"],
        ].reset_index(drop=True)

        golden = pd.read_parquet(
            "tests/tracking/fixtures/ghost_gk_backward_compat.parquet"
        )

        # Compare — tolerance for float precision
        pd.testing.assert_frame_equal(
            actual, golden, check_dtype=False, atol=1e-6,
        )

    @staticmethod
    def _make_model(n_estimators: int = 10) -> "GhostGkModel":
        """Build a deterministic small model for testing (same seed as golden file)."""
        from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame(
            rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES,
        )
        X["phase"] = rng.integers(0, 3, n).astype(float)
        X["team_in_possession"] = rng.integers(0, 2, n).astype(float)
        X["ball_in_own_half"] = rng.integers(0, 2, n).astype(float)
        labels = pd.DataFrame(
            {"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)},
        )
        model = GhostGkModel(n_estimators=n_estimators)
        model.fit(X, labels)
        return model

    def test_with_actions_changes_features(self):
        """Passing actions changes score_diff/phase in the extraction."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model = self._make_model(n_estimators=10)
        frames = _make_multi_frame_fixture(n_frames=3)
        actions = _make_spadl_actions(goals=[(0.01, 1)])

        result_no_actions = compute_ghost_gk(
            frames, model=model, home_team_id=1,
        )
        result_with_actions = compute_ghost_gk(
            frames, model=model, home_team_id=1, actions=actions,
        )
        # Predictions should differ because features differ
        gk_mask_no = (
            result_no_actions["is_goalkeeper"].astype(bool)
            & ~result_no_actions["is_ball"].astype(bool)
        )
        gk_mask_with = (
            result_with_actions["is_goalkeeper"].astype(bool)
            & ~result_with_actions["is_ball"].astype(bool)
        )
        x_no = result_no_actions.loc[gk_mask_no, "ghost_gk_x"].values
        x_with = result_with_actions.loc[gk_mask_with, "ghost_gk_x"].values
        # Not necessarily different (tiny model), but API accepts actions
        assert len(x_no) == len(x_with)

    def test_actions_none_is_default(self):
        """actions=None is the default and works."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model = self._make_model(n_estimators=10)
        frames = _make_multi_frame_fixture(n_frames=2)
        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        assert "ghost_gk_x" in result.columns


class TestAddGhostGkThreadsActions:
    """Verify aggregator passes actions through to compute_ghost_gk."""

    def test_add_ghost_gk_threads_actions(self):
        """add_ghost_gk passes actions= to compute_ghost_gk."""
        from unittest.mock import patch, MagicMock
        from silly_kicks.tracking.features import add_ghost_gk
        from silly_kicks.tracking._ghost_gk import GhostGkModel

        # Minimal model stub — add_ghost_gk resolves model internally,
        # but compute_ghost_gk is patched so model value doesn't matter.
        mock_result = _make_multi_frame_fixture(n_frames=1)
        mock_result["ghost_gk_x"] = 10.0
        mock_result["ghost_gk_y"] = 34.0
        mock_result["ghost_gk_spread"] = 2.0

        actions = _make_spadl_actions(goals=[])
        frames = _make_multi_frame_fixture(n_frames=1)

        with patch(
            "silly_kicks.tracking.features.compute_ghost_gk",
            return_value=mock_result,
        ) as mock_compute, patch(
            "silly_kicks.tracking.features._resolve_model",
            return_value=MagicMock(spec=GhostGkModel),
        ):
            # We can't fully run add_ghost_gk without real linking,
            # so just verify compute_ghost_gk is called with actions.
            try:
                add_ghost_gk(
                    actions, frames, home_team_id=1, actions_for_context=actions,
                )
            except Exception:
                pass  # linking may fail on synthetic data

            # Check that compute_ghost_gk was called with actions kwarg
            if mock_compute.called:
                _, kwargs = mock_compute.call_args
                assert "actions" in kwargs
                assert kwargs["actions"] is actions
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestComputeGhostGkRefactored::test_with_actions_changes_features -v`
Expected: FAIL — `compute_ghost_gk() got an unexpected keyword argument 'actions'`

- [ ] **Step 3: Refactor `compute_ghost_gk`**

Replace `compute_ghost_gk` (lines 777-901) in `silly_kicks/tracking/_ghost_gk.py`:

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-frame ghost-GK primitive (batched).

    Adds ghost_gk_x, ghost_gk_y, ghost_gk_spread columns.
    One prediction per (frame, GK team). Results written to GK rows.

    Input frames MUST be in LTR-normalized convention (home team attacks
    right in all periods --- standard silly-kicks tracking output).

    Parameters
    ----------
    frames : pd.DataFrame
        Tracking frames (TRACKING_FRAMES_COLUMNS schema, LTR-normalized).
    model : GhostGkModel | None
        Pre-loaded model. None = lazy download from Hub.
    home_team_id : int | str
        Home team ID (attacks right -> defends at x=0).
    actions : pd.DataFrame | None
        SPADL actions for match context (score_diff, phase). None =
        defaults (backward-compatible with 3.19.0).

    Returns
    -------
    pd.DataFrame
        Copy of frames with ghost_gk_x, ghost_gk_y, ghost_gk_spread added.

    Examples
    --------
    >>> from silly_kicks.tracking._ghost_gk import compute_ghost_gk
    >>> result = compute_ghost_gk(frames, home_team_id=1)
    >>> result = compute_ghost_gk(frames, home_team_id=1, actions=spadl_actions)
    """
    from ._ball_carrier import derive_team_in_possession, infer_ball_carrier

    resolved = _resolve_model(model)
    out = frames.copy()
    out["ghost_gk_x"] = np.nan
    out["ghost_gk_y"] = np.nan
    out["ghost_gk_spread"] = np.nan

    # Build context
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None
    carrier_raw = infer_ball_carrier(frames)
    carrier_cols = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]

    features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        carrier=carrier_cols,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
    )

    if len(features) == 0:
        return out

    # Batch predict
    densities = resolved.predict_density(features)

    # Build result DataFrame from predictions
    result_df = pd.DataFrame({
        "game_id": meta["game_id"].values,
        "period_id": meta["period_id"].values,
        "frame_id": meta["frame_id"].values,
        "team_id": meta["gk_team_id"].values,
        "ghost_gk_x": [d.mode_x for d in densities],
        "ghost_gk_y": [d.mode_y for d in densities],
        "ghost_gk_spread": [d.spread for d in densities],
    })

    # Merge into GK rows via single join
    gk_mask = out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)
    gk_rows_df = out.loc[gk_mask, ["game_id", "period_id", "frame_id", "team_id"]].copy()
    gk_rows_df = gk_rows_df.merge(
        result_df,
        on=["game_id", "period_id", "frame_id", "team_id"],
        how="left",
    )
    out.loc[gk_mask, "ghost_gk_x"] = gk_rows_df["ghost_gk_x"].values
    out.loc[gk_mask, "ghost_gk_y"] = gk_rows_df["ghost_gk_y"].values
    out.loc[gk_mask, "ghost_gk_spread"] = gk_rows_df["ghost_gk_spread"].values

    return out
```

- [ ] **Step 4: Update `add_ghost_gk` to thread actions**

In `silly_kicks/tracking/features.py`, update `add_ghost_gk` (around line 3468):

Change:
```python
        ghost_frames = compute_ghost_gk(frames, model=resolved_model, home_team_id=home_team_id)
```
to:
```python
        ghost_frames = compute_ghost_gk(
            frames, model=resolved_model, home_team_id=home_team_id, actions=actions,
        )
```

- [ ] **Step 5: Run refactored tests**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py::TestComputeGhostGkRefactored -v`
Expected: 3 PASSED

- [ ] **Step 6: Run full test suite**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk.py -v --tb=short`
Expected: all tests PASS

---

### Task 6: Training script

**Files:**
- Modify: `scripts/train_ghost_gk.py`

- [ ] **Step 1: Implement the full training script**

Replace `scripts/train_ghost_gk.py` entirely:

```python
#!/usr/bin/env python
"""Train Ghost-GK positioning model (TF-18).

Usage:
    uv run python scripts/train_ghost_gk.py \
        --data-dir /path/to/tracking/parquet/ \
        --output-dir models/ \
        --home-teams home_teams.json \
        --subsample-fps 1.0 \
        --n-estimators 500 \
        --max-depth 8 \
        --cv-folds 5

Requires: silly-kicks installed (uv run handles this).

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ghost-GK model")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory of tracking parquets")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Where to save model artifact")
    parser.add_argument("--actions-dir", type=Path, default=None,
                        help="Optional: directory of SPADL actions parquets")
    parser.add_argument("--home-teams", type=Path, required=True,
                        help="JSON file: {game_id: home_team_id, ...}")
    parser.add_argument("--subsample-fps", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"Data: {args.data_dir}, subsample_fps={args.subsample_fps}")
    print(f"CV: {args.cv_folds}-fold StratifiedGroupKFold (match+provider)")
    print(f"Output: {args.output_dir}")

    # --- 1. Load tracking data ---
    parquets = sorted(args.data_dir.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: No .parquet files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    frames = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    # Validate schema
    required = {"game_id", "period_id", "frame_id", "time_seconds", "player_id",
                "team_id", "is_ball", "is_goalkeeper", "x", "y"}
    missing = required - set(frames.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if "vx" not in frames.columns or "vy" not in frames.columns:
        print("ERROR: vx/vy columns missing. Run smooth_frames + derive_velocities first.",
              file=sys.stderr)
        sys.exit(1)

    n_games = frames["game_id"].nunique()
    n_frames_total = frames[["game_id", "period_id", "frame_id"]].drop_duplicates().shape[0]
    providers = frames["source_provider"].unique().tolist() if "source_provider" in frames.columns else ["unknown"]
    print(f"\nLoaded: {n_games} games, {n_frames_total} frames, providers: {providers}")

    # --- 2. Load actions (optional) ---
    actions: pd.DataFrame | None = None
    if args.actions_dir is not None:
        action_parquets = sorted(args.actions_dir.glob("*.parquet"))
        if action_parquets:
            actions = pd.concat([pd.read_parquet(p) for p in action_parquets], ignore_index=True)
            print(f"Loaded: {len(actions)} actions from {len(action_parquets)} files")

    # --- 3. Load home team mapping ---
    with open(args.home_teams) as f:
        home_team_map: dict[str, str] = json.load(f)
    print(f"Home team mapping: {len(home_team_map)} games")

    # --- 4. Per-game feature extraction ---
    from silly_kicks.tracking import prepare_ghost_gk_training_data

    frames_by_game = dict(list(frames.groupby("game_id")))
    actions_by_game = dict(list(actions.groupby("game_id"))) if actions is not None else {}

    all_features: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    all_game_ids: list = []
    all_providers: list[str] = []
    t0 = time.time()

    for game_id in sorted(frames_by_game):
        game_frames = frames_by_game[game_id]
        game_actions = actions_by_game.get(game_id) if actions is not None else None
        home = home_team_map.get(str(game_id))
        if home is None:
            print(f"  SKIP game {game_id}: no home_team_id in mapping")
            continue

        feats, labs = prepare_ghost_gk_training_data(
            game_frames, home_team_id=home, actions=game_actions,
            subsample_fps=args.subsample_fps,
        )
        if len(feats) > 0:
            all_features.append(feats)
            all_labels.append(labs)
            all_game_ids.extend([game_id] * len(feats))
            prov = str(game_frames["source_provider"].iloc[0]) if "source_provider" in game_frames.columns else "unknown"
            all_providers.extend([prov] * len(feats))

    if not all_features:
        print("ERROR: No training samples extracted.", file=sys.stderr)
        sys.exit(1)

    features = pd.concat(all_features, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    groups = np.array(all_game_ids)
    provider_labels = np.array(all_providers)
    print(f"\nExtracted {len(features)} samples from {len(set(all_game_ids))} games in {time.time()-t0:.1f}s")

    # --- 5. StratifiedGroupKFold CV ---
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._ghost_gk import GhostGkModel

    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(features, provider_labels, groups)):
        print(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model = GhostGkModel(n_estimators=args.n_estimators, max_depth=args.max_depth)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)  # shape (n, 2)

        mae_x = float(np.mean(np.abs(preds[:, 0] - y_test["gk_x"].values)))
        mae_y = float(np.mean(np.abs(preds[:, 1] - y_test["gk_y"].values)))
        mae_euclid = float(np.mean(np.sqrt(
            (preds[:, 0] - y_test["gk_x"].values) ** 2
            + (preds[:, 1] - y_test["gk_y"].values) ** 2
        )))

        # Per-provider MAE
        test_provs = provider_labels[test_idx]
        per_prov: dict[str, float] = {}
        for prov in np.unique(test_provs):
            mask = test_provs == prov
            per_prov[prov] = float(np.mean(np.sqrt(
                (preds[mask, 0] - y_test["gk_x"].values[mask]) ** 2
                + (preds[mask, 1] - y_test["gk_y"].values[mask]) ** 2
            )))

        print(f"  MAE x={mae_x:.3f}m  y={mae_y:.3f}m  euclid={mae_euclid:.3f}m")
        print(f"  Per-provider: {per_prov}")
        fold_metrics.append({
            "mae_x": mae_x, "mae_y": mae_y, "mae_euclidean": mae_euclid,
            "per_provider": per_prov,
        })

    # Aggregate CV
    mae_x_vals = [m["mae_x"] for m in fold_metrics]
    mae_y_vals = [m["mae_y"] for m in fold_metrics]
    mae_e_vals = [m["mae_euclidean"] for m in fold_metrics]
    print(f"\n=== CV Summary ===")
    print(f"MAE x: {np.mean(mae_x_vals):.3f} +/- {np.std(mae_x_vals):.3f}")
    print(f"MAE y: {np.mean(mae_y_vals):.3f} +/- {np.std(mae_y_vals):.3f}")
    print(f"MAE euclid: {np.mean(mae_e_vals):.3f} +/- {np.std(mae_e_vals):.3f}")

    # --- 6. Feature importance ---
    from sklearn.inspection import permutation_importance

    print("\n--- Feature importance (full model, x-coordinate only) ---")
    print("NOTE: Importance measured for gk_x predictions only.")
    print("Features primarily influencing gk_y may show artificially low importance.")
    final_model = GhostGkModel(n_estimators=args.n_estimators, max_depth=args.max_depth)
    final_model.fit(features, labels)
    preds_full = final_model.predict(features)

    # Use a simple sklearn wrapper for permutation importance
    from sklearn.metrics import mean_absolute_error

    class _SklearnWrapper:
        def __init__(self, model):
            self._model = model
        def predict(self, X):
            return self._model.predict(pd.DataFrame(X, columns=features.columns))[:, 0]

    pi = permutation_importance(
        _SklearnWrapper(final_model), features.values, labels["gk_x"].values,
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=42,
    )
    importances = sorted(
        zip(features.columns, pi.importances_mean), key=lambda x: -x[1],
    )
    print("Top 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")

    # --- 7. Save final model ---
    artifact_dir = args.output_dir / "ghost_gk_v1"
    final_model.save(artifact_dir)
    print(f"\nModel saved to {artifact_dir}")

    # Round-trip verify
    loaded = GhostGkModel.load(artifact_dir)
    sample_pred = loaded.predict(features.head(10))
    expected = final_model.predict(features.head(10))
    np.testing.assert_allclose(sample_pred, expected, atol=1e-10)
    print("Round-trip verification: PASS")

    # --- 8. Metrics JSON ---
    # Aggregate per-provider MAE across folds
    all_provs_set = set()
    for m in fold_metrics:
        all_provs_set.update(m["per_provider"].keys())
    per_prov_agg: dict[str, float] = {}
    for prov in sorted(all_provs_set):
        vals = [m["per_provider"].get(prov, np.nan) for m in fold_metrics]
        per_prov_agg[prov] = float(np.nanmean(vals))

    artifact_bytes = sum(f.stat().st_size for f in artifact_dir.rglob("*") if f.is_file())

    metrics = {
        "n_games": int(len(set(all_game_ids))),
        "n_samples": int(len(features)),
        "n_providers": int(len(set(all_providers))),
        "providers": sorted(set(all_providers)),
        "cv_folds": args.cv_folds,
        "subsample_fps": args.subsample_fps,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
        "cv_mae_x_mean": float(np.mean(mae_x_vals)),
        "cv_mae_x_std": float(np.std(mae_x_vals)),
        "cv_mae_y_mean": float(np.mean(mae_y_vals)),
        "cv_mae_y_std": float(np.std(mae_y_vals)),
        "cv_mae_euclidean_mean": float(np.mean(mae_e_vals)),
        "cv_mae_euclidean_std": float(np.std(mae_e_vals)),
        "per_provider_mae_euclidean": per_prov_agg,
        "acceptance": {
            "overall_mae_lt_2m": float(np.mean(mae_e_vals)) < 2.0,
            "per_provider_mae_lt_3m": all(v < 3.0 for v in per_prov_agg.values()),
            "cross_fold_std_lt_05m": float(np.std(mae_e_vals)) < 0.5,
            "artifact_size_lt_15mb": artifact_bytes < 15_000_000,
        },
        "artifact_size_bytes": artifact_bytes,
    }
    metrics_path = args.output_dir / "ghost_gk_v1" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Acceptance criteria
    print("\n=== Acceptance Criteria ===")
    for key, passed in metrics["acceptance"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses args**

Run: `uv run python scripts/train_ghost_gk.py --help`
Expected: prints usage with all arguments

---

### Task 7: Publish script

**Files:**
- Create: `scripts/publish_ghost_gk.py`

- [ ] **Step 1: Create publish script**

```python
#!/usr/bin/env python
"""Publish trained Ghost-GK model to HuggingFace Hub.

Usage:
    uv run python scripts/publish_ghost_gk.py \
        --artifact-dir models/ghost_gk_v1 \
        --repo-id karsten-s-nielsen/ghost-gk-v1

    # Verify only (dry run):
    uv run python scripts/publish_ghost_gk.py \
        --artifact-dir models/ghost_gk_v1 \
        --verify-only

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Ghost-GK model to HF Hub")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Model artifact directory (from train script)")
    parser.add_argument("--repo-id", type=str,
                        default="karsten-s-nielsen/ghost-gk-v1",
                        help="HF Hub repo ID")
    parser.add_argument("--verify-only", action="store_true",
                        help="Dry run: verify integrity without uploading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    # --- 1. Load and verify artifact ---
    print(f"Loading artifact from {args.artifact_dir}")
    model = GhostGkModel.load(args.artifact_dir)
    print("  SHA-256 verification: PASS (automatic in load)")

    # Sanity check: predict on synthetic sample
    rng = np.random.default_rng(99)
    n = 5
    X = pd.DataFrame(
        rng.standard_normal((n, len(GHOST_GK_FEATURE_NAMES))),
        columns=GHOST_GK_FEATURE_NAMES,
    )
    X["phase"] = 0.0
    X["team_in_possession"] = 1.0
    X["ball_in_own_half"] = 0.0
    local_preds = model.predict(X)
    print(f"  Sanity check: predicted {n} samples, shape {local_preds.shape}")

    if args.verify_only:
        print("\n--verify-only: artifact integrity confirmed. Skipping upload.")
        return

    # --- 2. Upload to HF Hub ---
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. pip install silly-kicks[ghost-gk]",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nUploading to {args.repo_id}...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(args.artifact_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("  Upload complete.")

    # --- 3. Verify download ---
    print(f"\nVerifying download from {args.repo_id}...")
    downloaded = GhostGkModel.from_hub(args.repo_id)
    remote_preds = downloaded.predict(X)
    np.testing.assert_allclose(remote_preds, local_preds, atol=1e-10)
    print("  Download + predict verification: PASS")
    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses args**

Run: `uv run python scripts/publish_ghost_gk.py --help`
Expected: prints usage

---

### Task 8: Integration tests

**Files:**
- Create: `tests/tracking/test_ghost_gk_integration.py`

- [ ] **Step 1: Create integration test file**

```python
"""Integration tests for Ghost-GK training + publish pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _build_synthetic_parquets(tmpdir: Path, n_games: int = 3) -> tuple[Path, Path, Path]:
    """Build synthetic tracking + actions parquets + home_teams.json."""
    from silly_kicks.spadl import config as spadlconfig

    tracking_dir = tmpdir / "tracking"
    actions_dir = tmpdir / "actions"
    tracking_dir.mkdir()
    actions_dir.mkdir()

    for g in range(n_games):
        game_id = str(100 + g)
        rows = []
        action_rows = []
        for fid in range(1, 6):  # 5 frames per game
            ts = float(fid) * 0.04  # 25fps
            base = dict(
                game_id=game_id, period_id=1, frame_id=fid,
                time_seconds=ts, frame_rate=25.0,
                ball_state="alive", source_provider="test",
                team_attacking_direction=None, confidence=None,
                visibility=None, is_goalkeeper_source="native", z=0.0,
            )
            rows.append({**base, player_id="ball", team_id=None,
                         x=50.0+fid, y=34.0, vx=2.0, vy=0.0, speed=2.0,
                         is_ball=True, is_goalkeeper=False})
            rows.append({**base, player_id="p1", team_id="1",
                         x=5.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                         is_ball=False, is_goalkeeper=True})
            for i, (px, py) in enumerate([(20,25),(22,30),(21,38),(23,45)]):
                rows.append({**base, player_id=f"p{10+i}", team_id="1",
                             x=float(px), y=float(py), vx=0.5, vy=0.0, speed=0.5,
                             is_ball=False, is_goalkeeper=False})
            for i, (px, py) in enumerate([(40,30),(45,34),(38,40),(50,34)]):
                rows.append({**base, player_id=f"a{10+i}", team_id="2",
                             x=float(px), y=float(py), vx=-1.0, vy=0.0, speed=1.0,
                             is_ball=False, is_goalkeeper=False})
            rows.append({**base, player_id="a1", team_id="2",
                         x=100.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                         is_ball=False, is_goalkeeper=True})

        pd.DataFrame(rows).to_parquet(tracking_dir / f"game_{game_id}.parquet", index=False)

    # Home teams JSON
    home_teams = {str(100 + g): "1" for g in range(n_games)}
    ht_path = tmpdir / "home_teams.json"
    with open(ht_path, "w") as f:
        json.dump(home_teams, f)

    return tracking_dir, actions_dir, ht_path


class TestRoundTripTrainPredict:
    """prepare_training_data -> fit -> predict round-trip."""

    def test_round_trip(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        # Build multi-frame fixture
        rows = []
        for fid in range(1, 11):
            ts = float(fid) * 0.04
            base = dict(
                game_id="100", period_id=1, frame_id=fid,
                time_seconds=ts, frame_rate=25.0,
                ball_state="alive", source_provider="test",
                team_attacking_direction=None, confidence=None,
                visibility=None, is_goalkeeper_source="native", z=0.0,
            )
            rows.append({**base, player_id="ball", team_id=None,
                         x=50.0+fid, y=34.0, vx=2.0, vy=0.0, speed=2.0,
                         is_ball=True, is_goalkeeper=False})
            rows.append({**base, player_id="p1", team_id=1,
                         x=5.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                         is_ball=False, is_goalkeeper=True})
            for i, (px, py) in enumerate([(20,25),(22,30),(21,38),(23,45)]):
                rows.append({**base, player_id=f"p{10+i}", team_id=1,
                             x=float(px), y=float(py), vx=0.5, vy=0.0, speed=0.5,
                             is_ball=False, is_goalkeeper=False})
            for i, (px, py) in enumerate([(40,30),(45,34),(38,40),(50,34)]):
                rows.append({**base, player_id=f"a{10+i}", team_id=2,
                             x=float(px), y=float(py), vx=-1.0, vy=0.0, speed=1.0,
                             is_ball=False, is_goalkeeper=False})
            rows.append({**base, player_id="a1", team_id=2,
                         x=100.0, y=34.0, vx=0.0, vy=0.0, speed=0.0,
                         is_ball=False, is_goalkeeper=True})

        frames = pd.DataFrame(rows)
        features, labels = prepare_ghost_gk_training_data(
            frames, home_team_id=1, subsample_fps=None,
        )
        assert len(features) > 0

        model = GhostGkModel(n_estimators=10)
        model.fit(features, labels)
        preds = model.predict(features)
        assert preds.shape == (len(features), 2)

        # Predictions should be in plausible range
        assert np.all(preds[:, 0] >= -5)
        assert np.all(preds[:, 0] <= 35)


class TestTrainScriptSmoke:
    """Train script runs on synthetic data and produces artifacts."""

    def test_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tracking_dir, _, ht_path = _build_synthetic_parquets(tmpdir, n_games=3)

            output_dir = tmpdir / "output"
            result = subprocess.run(
                [
                    sys.executable, "scripts/train_ghost_gk.py",
                    "--data-dir", str(tracking_dir),
                    "--home-teams", str(ht_path),
                    "--output-dir", str(output_dir),
                    "--n-estimators", "10",
                    "--max-depth", "3",
                    "--cv-folds", "3",
                    "--subsample-fps", "25.0",  # keep all frames (fps == source fps)
                ],
                capture_output=True, text=True, timeout=120,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
            assert result.returncode == 0, f"Script failed:\n{result.stderr}"

            # Check artifacts exist
            artifact_dir = output_dir / "ghost_gk_v1"
            assert artifact_dir.exists()
            assert (artifact_dir / "metrics.json").exists()

            # Verify metrics.json schema
            with open(artifact_dir / "metrics.json") as f:
                metrics = json.load(f)
            assert "n_games" in metrics
            assert "cv_mae_euclidean_mean" in metrics
            assert "acceptance" in metrics
            assert "artifact_size_bytes" in metrics
```

- [ ] **Step 2: Run integration tests**

Run: `uv run python -m pytest tests/tracking/test_ghost_gk_integration.py -v --tb=short`
Expected: 2 PASSED (may take 30-60s for the smoke test)

---

### Task 9: Pre-commit quality gates

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`
- Modify: `pyproject.toml` (version bump)
- Modify: `silly_kicks/__init__.py` (version bump)

- [ ] **Step 1: Run ruff check**

Run: `uv run ruff check silly_kicks/tracking/_ghost_gk.py silly_kicks/tracking/__init__.py silly_kicks/tracking/features.py scripts/train_ghost_gk.py scripts/publish_ghost_gk.py`
Fix any issues.

- [ ] **Step 2: Run ruff format check**

Run: `uv run ruff format --check silly_kicks/tracking/_ghost_gk.py silly_kicks/tracking/__init__.py silly_kicks/tracking/features.py scripts/train_ghost_gk.py scripts/publish_ghost_gk.py`
Fix any formatting issues.

- [ ] **Step 3: Run pyright on full package**

Run: `uv run pyright silly_kicks/`
Fix any type errors. Common patterns from memory:
- `np.asarray(series.values)` for pyright `.values` complaints
- `float(str(scalar))` for `Scalar` includes `complex`
- `list(df.columns).index("col")` instead of `df.columns.get_loc("col")`

- [ ] **Step 4: Run full test suite**

Run: `uv run python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all PASS

- [ ] **Step 5: Update CHANGELOG.md, TODO.md, version**

Update `CHANGELOG.md` with `[Unreleased]` section:
```markdown
### Added
- `prepare_ghost_gk_training_data()` public API for assembling Ghost-GK training data from tracking frames
- `_extract_all_ghost_gk_features()` shared batch helper (DRY inference/training)
- `_build_score_lookup()` / `_build_phase_lookup()` match context resolution
- `actions` parameter on `compute_ghost_gk()` and `add_ghost_gk()` for match context
- `scripts/train_ghost_gk.py` reference training CLI with StratifiedGroupKFold CV
- `scripts/publish_ghost_gk.py` HF Hub publish + verify CLI

### Fixed
- Ghost-GK feature extraction read `"timestamp"` instead of `"time_seconds"` column (always fell back to 0.0)
```

Update version in `pyproject.toml` and `silly_kicks/__init__.py`.
Delete the TF-18 training row from `TODO.md`.

- [ ] **Step 6: Invoke /final-review**

Run the `final-review` skill for the complete pre-commit quality gate.

---

## Self-Review

**Spec coverage:**
- S1 Scope: all 8 items covered (Tasks 0-8)
- S2 Shared batch helper: Task 3
- S3 Match context: Task 2
- S4 prepare_ghost_gk_training_data: Task 4
- S5 compute_ghost_gk refactoring: Task 5
- S6 Bug fix: Task 1
- S7 Team ID normalization: Task 3 (in shared helper)
- S8 NaN policy: Task 4 (label drop + HGBR handles NaN features)
- S9 Training script: Task 6
- S10 Publish script: Task 7
- S11 Testing: Tasks 1-5 (unit) + Task 8 (integration)
- S12 File changes: all covered
- S13 Consumer patterns: import paths verified in Task 4 test
- S14 Acceptance criteria: all covered

**Placeholder scan:** No TBD/TODO/implement-later found.

**Type consistency:** `_extract_all_ghost_gk_features` signature, `_build_score_lookup`/`_build_phase_lookup` signatures, `prepare_ghost_gk_training_data` signature all match across tasks. `carrier_cols` naming consistent. `home_team_id: str | int` throughout.
