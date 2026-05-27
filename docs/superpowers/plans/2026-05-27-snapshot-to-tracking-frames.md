# Snapshot-to-Tracking-Frames Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `snapshot_to_tracking_frames` public API that converts per-event player-position snapshots into the 20-column tracking frame schema + pre-built linkage pointers, enabling all single-frame `add_*` enrichment functions on StatsBomb 360 freeze-frame data.

**Architecture:** Single private module `_snapshot.py` with one public function returning `(frames, links)` tuple. No hardcoded pitch constants; coordinates pass through untouched. Schema registration adds `"snapshot"` to `TRACKING_CATEGORICAL_DOMAINS["source_provider"]`.

**Tech Stack:** pandas, numpy (no new dependencies)

**Spec:** `docs/superpowers/specs/2026-05-27-snapshot-to-tracking-frames-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `silly_kicks/tracking/_snapshot.py` | Converter logic: build frames + links from snapshots + actions |
| Modify | `silly_kicks/tracking/schema.py:68` | Add `"snapshot"` to `TRACKING_CATEGORICAL_DOMAINS["source_provider"]` |
| Modify | `silly_kicks/tracking/__init__.py` | Re-export `snapshot_to_tracking_frames` |
| Create | `tests/tracking/test_snapshot.py` | All unit tests for the converter |

---

### Task 0: Schema registration

**Files:**
- Modify: `silly_kicks/tracking/schema.py:68`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_snapshot.py` with a schema domain test:

```python
"""Tests for snapshot_to_tracking_frames converter."""

from __future__ import annotations

from silly_kicks.tracking.schema import TRACKING_CATEGORICAL_DOMAINS


def test_snapshot_in_source_provider_domain():
    """H1: 'snapshot' must be a valid source_provider value."""
    assert "snapshot" in TRACKING_CATEGORICAL_DOMAINS["source_provider"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_snapshot.py::test_snapshot_in_source_provider_domain -v`
Expected: FAIL with `AssertionError` (the frozenset currently only contains `{"gradientsports", "sportec", "metrica", "skillcorner"}`)

- [ ] **Step 3: Add "snapshot" to the domain**

In `silly_kicks/tracking/schema.py`, change line 68 from:

```python
    "source_provider": frozenset({"gradientsports", "sportec", "metrica", "skillcorner"}),
```

to:

```python
    "source_provider": frozenset({"gradientsports", "sportec", "metrica", "skillcorner", "snapshot"}),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_snapshot.py::test_snapshot_in_source_provider_domain -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `python -m pytest tests/ -m "not e2e" -x -q`
Expected: all pass (no existing test asserts the exact set of source_provider values)

---

### Task 1: Core converter — frames construction

**Files:**
- Create: `silly_kicks/tracking/_snapshot.py`
- Modify: `tests/tracking/test_snapshot.py`

- [ ] **Step 1: Write the failing tests for frames output**

Append to `tests/tracking/test_snapshot.py`:

```python
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS


@pytest.fixture()
def actions_3() -> pd.DataFrame:
    """3-action SPADL fixture.

    action 0: has 6 snapshot players (3v3, one GK per side)
    action 1: has 0 snapshot players (partial coverage test)
    action 2: has 4 snapshot players (2v2, no player_id — synthetic ID test)
    """
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "action_id": [10, 11, 12],
            "period_id": [1, 1, 1],
            "time_seconds": [5.0, 10.0, 15.0],
            "team_id": [100, 200, 100],
            "player_id": [1, 2, 3],
            "start_x": [50.0, 60.0, 70.0],
            "start_y": [34.0, 20.0, 40.0],
            "end_x": [55.0, 65.0, 75.0],
            "end_y": [30.0, 25.0, 35.0],
            "type_id": [0, 0, 0],
            "result_id": [1, 1, 1],
            "bodypart_id": [0, 0, 0],
        }
    )


@pytest.fixture()
def snapshots_3v3() -> pd.DataFrame:
    """6 players for action_id=10 (3v3 with one GK per side)."""
    return pd.DataFrame(
        {
            "action_id": [10, 10, 10, 10, 10, 10],
            "team_id": [100, 100, 100, 200, 200, 200],
            "player_id": [1, 2, 3, 4, 5, 6],
            "is_goalkeeper": [True, False, False, True, False, False],
            "x": [5.0, 40.0, 50.0, 100.0, 60.0, 55.0],
            "y": [34.0, 20.0, 40.0, 34.0, 50.0, 15.0],
        }
    )


@pytest.fixture()
def snapshots_2v2_no_pid() -> pd.DataFrame:
    """4 players for action_id=12 — no player_id column (synthetic ID test)."""
    return pd.DataFrame(
        {
            "action_id": [12, 12, 12, 12],
            "team_id": [100, 100, 200, 200],
            "is_goalkeeper": [True, False, True, False],
            "x": [5.0, 45.0, 100.0, 65.0],
            "y": [34.0, 30.0, 34.0, 40.0],
        }
    )


@pytest.fixture()
def snapshots_combined(snapshots_3v3) -> pd.DataFrame:
    """action 10 has 6 players w/ player_id. Actions 11, 12 have no snapshots.

    The combined fixture tests the partial-coverage path (actions without
    snapshot data excluded). The no-player_id path is tested separately
    via snapshots_2v2_no_pid.
    """
    return snapshots_3v3


def test_frames_schema(actions_3, snapshots_combined):
    """Output frames have exactly the 20 TRACKING_FRAMES_COLUMNS."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert list(frames.columns) == list(TRACKING_FRAMES_COLUMNS.keys())


def test_frames_row_count(actions_3, snapshots_combined):
    """6 player rows + 1 ball row = 7 rows for the single action with data."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert len(frames) == 7  # 6 players + 1 ball


def test_ball_row(actions_3, snapshots_combined):
    """One ball row per frame, position from action start_x/start_y."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    ball = frames[frames["is_ball"]]
    assert len(ball) == 1
    row = ball.iloc[0]
    assert row["x"] == 50.0  # action 10's start_x
    assert row["y"] == 34.0  # action 10's start_y
    assert pd.isna(row["player_id"])
    assert pd.isna(row["team_id"])


def test_frame_metadata(actions_3, snapshots_combined):
    """game_id, period_id, time_seconds derived from actions join."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert (frames["game_id"] == 1).all()
    assert (frames["period_id"] == 1).all()
    assert (frames["time_seconds"] == 5.0).all()
    assert (frames["frame_id"] == 10).all()  # frame_id = action_id


def test_constant_columns(actions_3, snapshots_combined):
    """Verify NaN/constant columns per spec."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    player_rows = frames[~frames["is_ball"]]
    assert player_rows["frame_rate"].isna().all()
    assert player_rows["z"].isna().all()
    assert player_rows["speed"].isna().all()
    assert player_rows["speed_source"].isna().all()
    assert player_rows["confidence"].isna().all()
    assert player_rows["visibility"].isna().all()
    assert (player_rows["ball_state"] == "alive").all()
    assert (player_rows["team_attacking_direction"] == "ltr").all()
    assert (player_rows["source_provider"] == "snapshot").all()
    assert (player_rows["is_goalkeeper_source"] == "native").all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_snapshot.py -v -k "not domain"`
Expected: FAIL with `ModuleNotFoundError: No module named 'silly_kicks.tracking._snapshot'`

- [ ] **Step 3: Implement `_snapshot.py`**

Create `silly_kicks/tracking/_snapshot.py`:

```python
"""Convert per-event player-position snapshots to tracking frame schema.

Public API: snapshot_to_tracking_frames
Module: silly_kicks.tracking._snapshot
Spec: docs/superpowers/specs/2026-05-27-snapshot-to-tracking-frames-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import TRACKING_FRAMES_COLUMNS


def snapshot_to_tracking_frames(
    snapshots: pd.DataFrame,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert per-event player-position snapshots to tracking frame schema.

    Parameters
    ----------
    snapshots : pd.DataFrame
        One row per player per event. Required columns: action_id, team_id,
        is_goalkeeper, x, y. Optional: player_id (synthetic sequential int
        if absent). Coordinates must be in the current SPADL coordinate system.
    actions : pd.DataFrame
        SPADL actions DataFrame. Used to derive game_id, period_id,
        time_seconds, and ball position (start_x, start_y) per frame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (frames, links) where:
        - frames: 20-column TRACKING_FRAMES_COLUMNS schema, one synthetic
          frame per action that has snapshot data.
        - links: Pre-built pointer DataFrame matching the
          link_actions_to_frames output contract (action_id, frame_id,
          time_offset_seconds=0.0, n_candidate_frames=1,
          link_quality_score=1.0).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking import snapshot_to_tracking_frames
    >>> # See tests/tracking/test_snapshot.py for runnable examples.
    """
    # --- empty input ---
    if len(snapshots) == 0:
        return _empty_frames(), _empty_links()

    # --- action metadata lookup ---
    action_meta = actions[["action_id", "game_id", "period_id", "time_seconds", "start_x", "start_y"]].copy()
    action_ids_with_data = snapshots["action_id"].unique()
    action_meta = action_meta[action_meta["action_id"].isin(action_ids_with_data)]

    if len(action_meta) == 0:
        return _empty_frames(), _empty_links()

    # --- player rows ---
    has_player_id = "player_id" in snapshots.columns
    player = snapshots.merge(
        action_meta[["action_id", "game_id", "period_id", "time_seconds"]],
        on="action_id",
        how="inner",
    )

    if not has_player_id:
        # Synthetic sequential int per frame
        player = player.copy()
        player["player_id"] = np.arange(len(player))

    player_frames = pd.DataFrame(
        {
            "game_id": player["game_id"],
            "period_id": player["period_id"],
            "frame_id": player["action_id"],
            "time_seconds": player["time_seconds"],
            "frame_rate": np.nan,
            "player_id": player["player_id"],
            "team_id": player["team_id"],
            "is_ball": False,
            "is_goalkeeper": player["is_goalkeeper"],
            "x": player["x"],
            "y": player["y"],
            "z": np.nan,
            "speed": np.nan,
            "speed_source": np.nan,
            "ball_state": "alive",
            "team_attacking_direction": "ltr",
            "confidence": np.nan,
            "visibility": np.nan,
            "source_provider": "snapshot",
            "is_goalkeeper_source": "native",
        }
    )

    # --- ball rows (one per frame) ---
    ball_frames = pd.DataFrame(
        {
            "game_id": action_meta["game_id"].values,
            "period_id": action_meta["period_id"].values,
            "frame_id": action_meta["action_id"].values,
            "time_seconds": action_meta["time_seconds"].values,
            "frame_rate": np.nan,
            "player_id": np.nan,
            "team_id": np.nan,
            "is_ball": True,
            "is_goalkeeper": False,
            "x": action_meta["start_x"].values,
            "y": action_meta["start_y"].values,
            "z": np.nan,
            "speed": np.nan,
            "speed_source": np.nan,
            "ball_state": "alive",
            "team_attacking_direction": "ltr",
            "confidence": np.nan,
            "visibility": np.nan,
            "source_provider": "snapshot",
            "is_goalkeeper_source": "native",
        }
    )

    # --- combine and enforce column order ---
    frames = pd.concat([player_frames, ball_frames], ignore_index=True)
    frames = frames[list(TRACKING_FRAMES_COLUMNS.keys())]

    # --- links ---
    links = pd.DataFrame(
        {
            "action_id": action_meta["action_id"].values,
            "frame_id": action_meta["action_id"].values,
            "time_offset_seconds": 0.0,
            "n_candidate_frames": 1,
            "link_quality_score": 1.0,
        }
    )

    return frames, links


def _empty_frames() -> pd.DataFrame:
    """Return an empty DataFrame with TRACKING_FRAMES_COLUMNS schema."""
    return pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in TRACKING_FRAMES_COLUMNS.items()})


def _empty_links() -> pd.DataFrame:
    """Return an empty links DataFrame matching link_actions_to_frames contract.

    Dtypes default to int64 for the empty case (no input to infer from).
    Matches the empty-return pattern in link_actions_to_frames (utils.py:163-170).
    """
    return pd.DataFrame(
        {
            "action_id": pd.Series([], dtype="int64"),
            "frame_id": pd.Series([], dtype="int64"),
            "time_offset_seconds": pd.Series([], dtype="float64"),
            "n_candidate_frames": pd.Series([], dtype="int64"),
            "link_quality_score": pd.Series([], dtype="float64"),
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_snapshot.py -v`
Expected: all 6 tests PASS (1 domain + 5 frames)

---

### Task 2: Empty input, partial coverage, synthetic player_id

**Files:**
- Modify: `tests/tracking/test_snapshot.py`

- [ ] **Step 1: Write the empty + partial + synthetic tests**

Append to `tests/tracking/test_snapshot.py`:

```python
def test_empty_snapshots(actions_3):
    """0 snapshots -> empty frames + empty links with correct columns."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    empty_snap = pd.DataFrame(
        {"action_id": pd.Series([], dtype="int64"),
         "team_id": pd.Series([], dtype="int64"),
         "player_id": pd.Series([], dtype="int64"),
         "is_goalkeeper": pd.Series([], dtype="bool"),
         "x": pd.Series([], dtype="float64"),
         "y": pd.Series([], dtype="float64")}
    )
    frames, links = snapshot_to_tracking_frames(empty_snap, actions_3)

    assert len(frames) == 0
    assert list(frames.columns) == list(TRACKING_FRAMES_COLUMNS.keys())
    assert len(links) == 0
    assert list(links.columns) == ["action_id", "frame_id", "time_offset_seconds",
                                    "n_candidate_frames", "link_quality_score"]


def test_partial_coverage(actions_3, snapshots_combined):
    """Actions without snapshot data excluded from both outputs."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    # Only action_id=10 has snapshot data
    assert set(frames["frame_id"].unique()) == {10}
    assert set(links["action_id"].unique()) == {10}
    # Action 11 and 12 not in outputs
    assert 11 not in links["action_id"].values
    assert 12 not in links["action_id"].values


def test_synthetic_player_id(actions_3, snapshots_2v2_no_pid):
    """When player_id absent, synthetic sequential IDs are generated."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    frames, links = snapshot_to_tracking_frames(snapshots_2v2_no_pid, actions_3)
    player_rows = frames[~frames["is_ball"]]
    assert len(player_rows) == 4
    # Synthetic IDs are sequential integers
    pids = player_rows["player_id"].tolist()
    assert pids == [0, 1, 2, 3]


def test_links_contract(actions_3, snapshots_combined):
    """Links have exact-match values per spec."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    _frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    assert (links["time_offset_seconds"] == 0.0).all()
    assert (links["n_candidate_frames"] == 1).all()
    assert (links["link_quality_score"] == 1.0).all()
    assert (links["frame_id"] == links["action_id"]).all()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_snapshot.py -v`
Expected: all 10 tests PASS (6 from Tasks 0-1 + 4 new)

---

### Task 3: Public API re-export + downstream integration tests

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `tests/tracking/test_snapshot.py`

- [ ] **Step 1: Write the import and downstream tests**

Append to `tests/tracking/test_snapshot.py`:

```python
def test_public_import():
    """snapshot_to_tracking_frames is importable from silly_kicks.tracking."""
    from silly_kicks.tracking import snapshot_to_tracking_frames  # noqa: F401


def test_downstream_line_break_works(actions_3, snapshots_combined):
    """Downstream works: add_line_break(method='ward') produces valid output."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_line_break

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    # Only pass the action that has snapshot data
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    result = add_line_break(actions_with_data, frames, links=links, home_team_id=100, method="ward")
    assert "line_break__ward" in result.columns
    assert "lines_broken__ward" in result.columns
    assert "line_breaking_type__ward" in result.columns
    assert len(result) == len(actions_with_data)
    # Verify Ward actually computed something (not just all-NaN)
    assert result["line_break__ward"].notna().any()


def test_downstream_line_break_missing_home_team_id_raises(actions_3, snapshots_combined):
    """M1: Calling add_line_break without home_team_id raises TypeError."""
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_line_break

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    with pytest.raises(TypeError):
        add_line_break(actions_with_data, frames, links=links, method="ward")


def test_downstream_action_context_actor_speed_nan(actions_3, snapshots_combined):
    """actor_speed degrades to NaN; other 3 action_context columns have values.

    add_action_context returns 4 columns total: nearest_defender_distance,
    actor_speed, receiver_zone_density, defenders_in_triangle_to_goal.
    Only actor_speed reads the speed column from the linked frame (NaN on
    snapshots); the other 3 are purely positional.
    """
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_action_context

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    result = add_action_context(actions_with_data, frames, links=links)
    assert result["actor_speed"].isna().all()
    # Position-only columns should have values (not all NaN)
    assert result["nearest_defender_distance"].notna().any()


def test_downstream_cover_shadows_degrades(actions_3, snapshots_combined):
    """Velocity-dependent add_cover_shadows returns NaN columns, not raises."""
    from unittest.mock import MagicMock

    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_cover_shadows

    frames, links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    actions_with_data = actions_3[actions_3["action_id"].isin(links["action_id"])]
    # xt is a required positional arg (ExpectedThreat). MagicMock suffices
    # because cover_shadows returns None before reaching xt when vx/vy are
    # absent (_cover_shadows.py:792-794).
    mock_xt = MagicMock()
    result = add_cover_shadows(actions_with_data, frames, mock_xt, links=links, home_team_id=100)
    # Cover shadows requires vx/vy — should degrade to NaN, not raise
    assert "blocking_score" in result.columns
    assert result["blocking_score"].isna().all()
```

- [ ] **Step 2: Run the import test to verify it fails**

Run: `python -m pytest tests/tracking/test_snapshot.py::test_public_import -v`
Expected: FAIL with `ImportError: cannot import name 'snapshot_to_tracking_frames'`

- [ ] **Step 3: Add re-export to `__init__.py`**

In `silly_kicks/tracking/__init__.py`, add to `__all__` (in alphabetical position, after `"smooth_frames"`):

```python
    "snapshot_to_tracking_frames",
```

Add to the imports (after the `_shape_graph` import line):

```python
from ._snapshot import snapshot_to_tracking_frames
```

- [ ] **Step 4: Run all snapshot tests**

Run: `python -m pytest tests/tracking/test_snapshot.py -v`
Expected: all 15 tests PASS (10 from Tasks 0-2 + 5 new)

- [ ] **Step 5: Run full test suite for regressions**

Run: `python -m pytest tests/ -m "not e2e" -x -q`
Expected: all pass

---

### Task 4: Lint + type check + final gates

**Files:**
- Possibly modify: `silly_kicks/tracking/_snapshot.py` (lint fixes)

- [ ] **Step 1: Run ruff check**

Run: `python -m ruff check silly_kicks/tracking/_snapshot.py tests/tracking/test_snapshot.py`
Expected: no errors. If errors, fix them.

- [ ] **Step 2: Run ruff format check**

Run: `python -m ruff format --check silly_kicks/tracking/_snapshot.py tests/tracking/test_snapshot.py`
Expected: no reformatting needed. If needed, run `python -m ruff format` on the files.

- [ ] **Step 3: Run pyright on full package**

Run: `uv run pyright silly_kicks/`
Expected: 0 errors. Fix any type errors (common: `np.nan` vs `float("nan")`, `.values` needing `np.asarray()` wrap).

- [ ] **Step 4: Run full test suite one final time**

Run: `python -m pytest tests/ -m "not e2e" -x -q`
Expected: all pass

---

### Task 5: Version bump + changelog + commit

**Files:**
- Modify: `pyproject.toml` (version)
- Modify: `silly_kicks/__init__.py` (version)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change `version = "3.22.2"` to `version = "3.23.0"` (new feature = minor bump).

In `silly_kicks/__init__.py`, change `__version__ = "3.22.2"` to `__version__ = "3.23.0"`.

- [ ] **Step 2: Update CHANGELOG.md**

Add a new section at the top of the changelog (below the header, above the previous release):

```markdown
## [3.23.0] - 2026-05-27

### Added

- `snapshot_to_tracking_frames` public API in `silly_kicks.tracking` — converts per-event player-position snapshots (e.g. StatsBomb 360 freeze-frames) into the 20-column `TRACKING_FRAMES_COLUMNS` schema + pre-built linkage pointers. Enables all single-frame `add_*` enrichment functions on freeze-frame data without modification. (PR-S61)
- `"snapshot"` added to `TRACKING_CATEGORICAL_DOMAINS["source_provider"]` domain set.
```

- [ ] **Step 3: Invoke /final-review**

Run the final-review skill before committing. Address any findings.

- [ ] **Step 4: Single commit**

```bash
git add -A
git commit -m "feat(tracking): snapshot_to_tracking_frames converter for freeze-frame data -- silly-kicks 3.23.0 (PR-S61)"
```
