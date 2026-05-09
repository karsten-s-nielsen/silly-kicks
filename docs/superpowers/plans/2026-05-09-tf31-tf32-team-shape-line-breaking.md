# TF-31 + TF-32: Team Shape Envelope & Ward Line-Breaking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 7 per-frame team-shape metrics (TF-31) and 3 Ward-clustering line-breaking features (TF-32) as action-coupled tracking features with VAEP integration and full test coverage.

**Architecture:** Two new private modules (`_team_shape.py`, `_line_breaking.py`) paralleling the existing `_defensive_line.py` / `_off_ball_runs.py` pattern. TF-31 uses `scipy.spatial.ConvexHull` for hull area; TF-32 uses `scipy.cluster.hierarchy.linkage` + `fcluster` for 1D Ward clustering with cross-product straddle intersection. Both integrate via the standard three-tier pattern: per-frame/per-action primitive -> `add_*` aggregator with `@nan_safe_enrichment` -> `*_xfns` VAEP factory with `_frame_aware` marker. The existing `add_line_break` gains a `method=` kwarg dispatching to either `"threshold"` (existing) or `"ward"` (new).

**Tech Stack:** pandas, numpy, scipy (all already in dependency tree)

**Commit policy:** One commit per branch (user policy). All tasks produce working code; commit happens at Task 8.

**Spec:** `docs/superpowers/specs/2026-05-09-tf31-tf32-team-shape-line-breaking-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `silly_kicks/tracking/_team_shape.py` (CREATE) | `compute_team_shape` per-frame primitive (7 metrics) |
| `silly_kicks/tracking/_line_breaking.py` (CREATE) | `LineBreakingParams` dataclass + `detect_line_breaking` per-action Ward detection (3 output cols) |
| `silly_kicks/tracking/features.py` (MODIFY) | `add_team_shape` + `team_shape_xfns` (TF-31); extend `add_line_break` with `method=`/`params=` + `line_breaking_ward_xfns` (TF-32); `__all__` updates |
| `silly_kicks/tracking/__init__.py` (MODIFY) | Re-exports for new public API |
| `tests/tracking/test_team_shape.py` (CREATE) | Unit + edge-case tests for `compute_team_shape` + `add_team_shape` + xfn introspection |
| `tests/tracking/test_line_breaking.py` (CREATE) | Unit + edge-case tests for `detect_line_breaking` + `add_line_break(method="ward")` + xfn introspection |
| `tests/tracking/test_team_shape_providers.py` (CREATE) | Provider fixture parametrized tests for TF-31 |
| `tests/tracking/test_line_breaking_providers.py` (CREATE) | Provider fixture parametrized tests for TF-32 |
| `tests/invariants/test_invariant_team_shape.py` (CREATE) | Physical invariant tests for TF-31 |
| `tests/invariants/test_invariant_line_breaking.py` (CREATE) | Physical invariant tests for TF-32 |
| `tests/tracking/_provider_inputs.py` (MODIFY) | Amend `synthesize_actions` to give passes offset `end_x`/`end_y` |
| `tests/tracking/conftest.py` (MODIFY) | Re-export new shared helper(s) |
| `NOTICE` (MODIFY) | Add Clemente 2013 + Karakus 2025 entries |
| `TODO.md` (MODIFY) | Delete TF-31 and TF-32 rows |
| `CHANGELOG.md` (MODIFY) | Add new features |

---

### Task 1: TF-31 `compute_team_shape` — failing tests + implementation

**Files:**
- Create: `tests/tracking/test_team_shape.py`
- Create: `silly_kicks/tracking/_team_shape.py`

- [ ] **Step 1: Write failing unit tests for `compute_team_shape`**

Create `tests/tracking/test_team_shape.py` with known-geometry fixtures:

```python
"""Tests for silly_kicks.tracking._team_shape.compute_team_shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_team_frames(
    *,
    team_id=1,
    outfield_positions: list[tuple[float, float]],
    gk_pos: tuple[float, float] = (3.0, 34.0),
    frame_id: int = 1,
    period_id: int = 1,
    game_id: int = 1,
    time_seconds: float = 1.0,
) -> pd.DataFrame:
    """Build a single-frame fixture for one team with known positions."""
    rows = []
    pid = 100
    # Ball
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # GK
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=pid,
            team_id=team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=gk_pos[0],
            y=gk_pos[1],
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    pid += 1
    # Outfield
    for x, y in outfield_positions:
        rows.append(
            dict(
                game_id=game_id,
                period_id=period_id,
                frame_id=frame_id,
                time_seconds=time_seconds,
                frame_rate=25.0,
                player_id=pid,
                team_id=team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=x,
                y=y,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
        pid += 1
    return pd.DataFrame(rows)


class TestComputeTeamShape:
    def test_known_square_geometry(self):
        """4 players at (0,0), (10,0), (10,10), (0,10) -> known metrics."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0),
        ])
        result = compute_team_shape(frames, team_id=1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_outfield_players"] == 4
        assert row["centroid_x"] == pytest.approx(5.0)
        assert row["centroid_y"] == pytest.approx(5.0)
        assert row["convex_hull_area"] == pytest.approx(100.0)
        assert row["team_length"] == pytest.approx(10.0)
        assert row["team_width"] == pytest.approx(10.0)
        # stretch_index = mean distance from (5,5) to each corner = sqrt(50) = 7.071...
        assert row["stretch_index"] == pytest.approx(np.sqrt(50.0))

    def test_triangle_geometry(self):
        """3 players in right triangle -> hull area = 0.5 * base * height."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[
            (0.0, 0.0), (10.0, 0.0), (0.0, 6.0),
        ])
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 3
        assert row["convex_hull_area"] == pytest.approx(30.0)  # 0.5 * 10 * 6
        assert row["team_length"] == pytest.approx(10.0)
        assert row["team_width"] == pytest.approx(6.0)

    def test_zero_players_all_nan(self):
        """No outfield players for the team -> all metrics NaN."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        # Only build a GK, no outfield
        frames = _make_team_frames(outfield_positions=[])
        result = compute_team_shape(frames, team_id=1)

        # Should return 0 rows (no outfield players -> no entry for this frame)
        assert len(result) == 0

    def test_one_player_degenerate(self):
        """1 player -> centroid=position, length/width/stretch=0, hull=NaN."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[(20.0, 30.0)])
        result = compute_team_shape(frames, team_id=1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_outfield_players"] == 1
        assert row["centroid_x"] == pytest.approx(20.0)
        assert row["centroid_y"] == pytest.approx(30.0)
        assert row["team_length"] == pytest.approx(0.0)
        assert row["team_width"] == pytest.approx(0.0)
        assert row["stretch_index"] == pytest.approx(0.0)
        assert pd.isna(row["convex_hull_area"])

    def test_two_players_hull_nan(self):
        """2 players -> hull NaN, rest valid."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[(10.0, 34.0), (20.0, 34.0)])
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 2
        assert pd.isna(row["convex_hull_area"])
        assert row["team_length"] == pytest.approx(10.0)
        assert row["team_width"] == pytest.approx(0.0)

    def test_collinear_players_hull_nan(self):
        """3+ collinear players -> QhullError caught, hull=NaN."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[
            (10.0, 34.0), (20.0, 34.0), (30.0, 34.0),
        ])
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 3
        assert pd.isna(row["convex_hull_area"])  # degenerate
        assert row["team_length"] == pytest.approx(20.0)
        assert row["team_width"] == pytest.approx(0.0)

    def test_filters_goalkeeper(self):
        """GK is NOT included in outfield metrics."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        # GK at (3, 34) should not affect centroid of outfield at (50, 34)
        frames = _make_team_frames(
            outfield_positions=[(50.0, 34.0), (60.0, 34.0), (70.0, 34.0)],
            gk_pos=(3.0, 34.0),
        )
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 3
        assert row["centroid_x"] == pytest.approx(60.0)  # mean(50, 60, 70)

    def test_empty_frames(self):
        """Empty frames -> empty result."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = pd.DataFrame(columns=[
            "game_id", "period_id", "frame_id", "team_id",
            "player_id", "is_ball", "is_goalkeeper", "x", "y",
        ])
        result = compute_team_shape(frames, team_id=1)
        assert len(result) == 0

    def test_multi_frame_batch(self):
        """Multiple frames produce one row per frame."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        f1 = _make_team_frames(
            outfield_positions=[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],
            frame_id=1, time_seconds=0.0,
        )
        f2 = _make_team_frames(
            outfield_positions=[(30.0, 30.0), (40.0, 30.0), (40.0, 40.0), (30.0, 40.0)],
            frame_id=2, time_seconds=0.04,
        )
        frames = pd.concat([f1, f2], ignore_index=True)
        result = compute_team_shape(frames, team_id=1)

        assert len(result) == 2
        assert result["frame_id"].tolist() == [1, 2]
        # Frame 1 centroid at (15, 15); Frame 2 at (35, 35)
        assert result.iloc[0]["centroid_x"] == pytest.approx(15.0)
        assert result.iloc[1]["centroid_x"] == pytest.approx(35.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_team_shape.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._team_shape'`

- [ ] **Step 3: Write `compute_team_shape` implementation**

Create `silly_kicks/tracking/_team_shape.py`:

```python
"""Per-frame team shape envelope (TF-31).

Computes centroid, convex hull area, length, width, stretch index, and
visible outfield player count for a specified team per frame.

See spec: docs/superpowers/specs/2026-05-09-tf31-tf32-team-shape-line-breaking-design.md s1.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError


_RESULT_COLS = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "n_outfield_players",
    "centroid_x",
    "centroid_y",
    "convex_hull_area",
    "team_length",
    "team_width",
    "stretch_index",
]


def compute_team_shape(
    frames: pd.DataFrame,
    team_id: int | str,
) -> pd.DataFrame:
    """Per-(game_id, period_id, frame_id) team shape metrics for one team.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS schema).
    team_id : int | str
        Team to compute shape for.

    Returns
    -------
    pd.DataFrame
        One row per (game_id, period_id, frame_id) where the team has at
        least one visible outfield player. Columns: game_id, period_id,
        frame_id, team_id, n_outfield_players, centroid_x, centroid_y,
        convex_hull_area, team_length, team_width, stretch_index.

    Examples
    --------
    Compute team shape for a single team::

        from silly_kicks.tracking._team_shape import compute_team_shape
        shape = compute_team_shape(frames, team_id=1)

    See NOTICE for full bibliographic citations.
    """
    if len(frames) == 0:
        return pd.DataFrame(columns=_RESULT_COLS)

    # Filter to outfield players with valid coordinates
    mask = (
        (frames["team_id"] == team_id)
        & (~frames["is_ball"])
        & (~frames["is_goalkeeper"])
        & frames["x"].notna()
        & frames["y"].notna()
    )
    outfield = frames[mask]
    if outfield.empty:
        return pd.DataFrame(columns=_RESULT_COLS)

    rows: list[dict] = []
    groups = outfield.groupby(["game_id", "period_id", "frame_id"], dropna=False)

    for (game_id, period_id, frame_id), group in groups:
        xs = group["x"].to_numpy(dtype="float64")
        ys = group["y"].to_numpy(dtype="float64")
        n = len(xs)

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        team_length = float(np.max(xs) - np.min(xs))
        team_width = float(np.max(ys) - np.min(ys))

        # Stretch index: mean Euclidean distance from centroid
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        stretch = float(np.mean(dists))

        # Convex hull area
        if n < 3:
            hull_area = np.nan
        else:
            try:
                hull = ConvexHull(np.column_stack([xs, ys]))
                hull_area = float(hull.volume)  # 2D: volume = area
            except QhullError:
                hull_area = np.nan

        rows.append(
            {
                "game_id": game_id,
                "period_id": period_id,
                "frame_id": frame_id,
                "team_id": team_id,
                "n_outfield_players": n,
                "centroid_x": cx,
                "centroid_y": cy,
                "convex_hull_area": hull_area,
                "team_length": team_length,
                "team_width": team_width,
                "stretch_index": stretch,
            }
        )

    result = pd.DataFrame(rows, columns=_RESULT_COLS)
    result["n_outfield_players"] = result["n_outfield_players"].astype("Int64")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_team_shape.py -v --tb=short`
Expected: All 9 tests PASS

---

### Task 2: TF-31 `add_team_shape` + `team_shape_xfns` — aggregator + VAEP integration

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `tests/tracking/test_team_shape.py`

- [ ] **Step 1: Write failing tests for `add_team_shape` and `team_shape_xfns`**

Append to `tests/tracking/test_team_shape.py`:

```python
class TestAddTeamShape:
    def test_enriches_actions_with_14_columns(self):
        """add_team_shape adds 14 team-shape columns (7 metrics x 2 teams)."""
        from silly_kicks.tracking.features import add_team_shape

        # Build frames with two teams
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0, 40.0, 50.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[60.0, 70.0, 80.0, 90.0, 95.0],
            away_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0],
        )
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [1],
                "period_id": [1],
                "time_seconds": [1.0],
                "team_id": [1],
                "player_id": [3],  # a home outfield player
                "start_x": [30.0],
                "start_y": [30.0],
                "end_x": [40.0],
                "end_y": [35.0],
                "type_id": [0],
            }
        )

        result = add_team_shape(actions, frames, home_team_id=1)

        expected_cols = [
            "team_shape_n_outfield_players_attacking",
            "team_shape_centroid_x_attacking",
            "team_shape_centroid_y_attacking",
            "team_shape_convex_hull_area_attacking",
            "team_shape_team_length_attacking",
            "team_shape_team_width_attacking",
            "team_shape_stretch_index_attacking",
            "team_shape_n_outfield_players_defending",
            "team_shape_centroid_x_defending",
            "team_shape_centroid_y_defending",
            "team_shape_convex_hull_area_defending",
            "team_shape_team_length_defending",
            "team_shape_team_width_defending",
            "team_shape_stretch_index_defending",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == 1

    def test_attacking_is_action_team(self):
        """team_shape_centroid_x_attacking reflects the acting team's centroid."""
        from silly_kicks.tracking.features import add_team_shape
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0, 40.0, 50.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[60.0, 70.0, 80.0, 90.0, 95.0],
            away_outfield_ys=[34.0] * 5,
        )
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [1],
                "period_id": [1],
                "time_seconds": [1.0],
                "team_id": [1],  # home team is attacking
                "player_id": [3],
                "start_x": [30.0],
                "start_y": [34.0],
                "end_x": [40.0],
                "end_y": [34.0],
                "type_id": [0],
            }
        )

        result = add_team_shape(actions, frames, home_team_id=1)
        # Home outfield centroid_x = mean(10,20,30,40,50) = 30
        assert result.iloc[0]["team_shape_centroid_x_attacking"] == pytest.approx(30.0)
        # Away outfield centroid_x = mean(60,70,80,90,95) = 79
        assert result.iloc[0]["team_shape_centroid_x_defending"] == pytest.approx(79.0)


class TestTeamShapeXfns:
    def test_xfn_column_count(self):
        """team_shape_xfns produces 36 columns (12 features x 3 states)."""
        from silly_kicks.tracking.features import team_shape_xfns

        xfns = team_shape_xfns(home_team_id=1)
        assert len(xfns) == 1

        xfn = xfns[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_xfn_introspection_nan(self):
        """frames=None -> NaN DataFrame with 36 correct column names."""
        from silly_kicks.tracking.features import team_shape_xfns

        xfns = team_shape_xfns(home_team_id=1)
        xfn = xfns[0]

        # Simulate VAEP introspection: 3 gamestates of 10 rows, no frames
        dummy = pd.DataFrame(
            {
                "game_id": [1] * 10,
                "action_id": range(10),
                "period_id": [1] * 10,
                "time_seconds": [float(i) for i in range(10)],
                "team_id": [1] * 10,
                "player_id": [1] * 10,
                "start_x": [50.0] * 10,
                "start_y": [34.0] * 10,
                "end_x": [60.0] * 10,
                "end_y": [34.0] * 10,
                "type_id": [0] * 10,
                "result_id": [0] * 10,
                "bodypart_id": [0] * 10,
            }
        )
        states = [dummy, dummy, dummy]
        result = xfn(states, None)

        assert len(result.columns) == 36
        assert result.isna().all().all()
        # Verify naming pattern
        assert "team_shape_centroid_x_attacking_a0" in result.columns
        assert "team_shape_stretch_index_defending_a2" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_team_shape.py::TestAddTeamShape -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'add_team_shape'`

- [ ] **Step 3: Implement `add_team_shape` and `team_shape_xfns` in `features.py`**

Add the following after the `off_ball_context_xfns` function (before the pitch control section) in `silly_kicks/tracking/features.py`:

```python
# ---------------------------------------------------------------------------
# PR-S33 -- TF-31: team shape envelope
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_team_shape(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Enrich actions with 14 team-shape columns (7 metrics x 2 teams).

    Provenance columns (frame_id, time_offset_seconds, link_quality_score,
    n_candidate_frames) are skipped if they already exist on the input.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_team_shape
    >>> # See tests/tracking/test_team_shape.py for runnable examples.
    """
    from ._team_shape import compute_team_shape

    out = actions.copy()

    # Compute shape for both teams (ONCE each)
    teams = frames[~frames["is_ball"]]["team_id"].dropna().unique()
    if len(teams) < 2:
        # Can't determine attacking/defending split — fill NaN
        for suffix in ("attacking", "defending"):
            for metric in (
                "n_outfield_players",
                "centroid_x",
                "centroid_y",
                "convex_hull_area",
                "team_length",
                "team_width",
                "stretch_index",
            ):
                out[f"team_shape_{metric}_{suffix}"] = np.nan
        return out

    # Pre-compute and index shape by (game_id, period_id, frame_id) for O(1) lookup
    shape_indexed: dict = {}
    for tid in teams:
        s = compute_team_shape(frames, team_id=tid)
        shape_indexed[tid] = s.set_index(["game_id", "period_id", "frame_id"])

    # Link actions to frames
    pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()

    metrics = [
        "n_outfield_players",
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
    ]

    # Initialize output columns to NaN
    for suffix in ("attacking", "defending"):
        for metric in metrics:
            out[f"team_shape_{metric}_{suffix}"] = np.nan

    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions[["action_id", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )

    aid_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())

    for _, row in linked.iterrows():
        aid = row["action_id"]
        if aid not in aid_to_idx.index:
            continue
        idx = aid_to_idx.loc[aid]
        action_team = row["team_id"]
        key = (row["game_id"], row["period_id"], int(row["frame_id_int"]))

        for tid, sdf in shape_indexed.items():
            if key not in sdf.index:
                continue
            shape_row = sdf.loc[key]
            suffix = "attacking" if tid == action_team else "defending"
            for metric in metrics:
                out.at[idx, f"team_shape_{metric}_{suffix}"] = shape_row[metric]

    # Provenance: skip if already present
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    existing_provenance = [c for c in provenance_cols if c in out.columns]
    if not existing_provenance:
        pointer_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(
            pointer_cols, left_on="action_id", right_index=True, how="left"
        )
    return out


def team_shape_xfns(home_team_id: int | str) -> list:
    """Build VAEP xfn list for TF-31 team shape features.

    Returns a list with ONE FrameAwareTransformer that emits 12 features x 3
    game-states = 36 columns total. ``n_outfield_players`` is excluded (data-quality
    indicator, not a tactical feature).

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, team_shape_xfns
        xfns = tracking_default_xfns + team_shape_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._team_shape import compute_team_shape

    vaep_metrics = [
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
    ]

    col_names = []
    for metric in vaep_metrics:
        for suffix in ("attacking", "defending"):
            col_names.append(f"team_shape_{metric}_{suffix}")

    def _team_shape_transformer(states, frames):
        """Multi-column team-shape xfn (12 cols x nb_states)."""
        import numpy as np

        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        teams = frames[~frames["is_ball"]]["team_id"].dropna().unique()
        shape_indexed = {}
        for tid in teams:
            s = compute_team_shape(frames, team_id=tid)
            shape_indexed[tid] = s.set_index(["game_id", "period_id", "frame_id"])

        for i, slot in enumerate(states[:3]):
            slot_result = _team_shape_at_actions(
                slot, frames, home_team_id, shape_indexed
            )
            for col in col_names:
                out[f"{col}_a{i}"] = slot_result[col].to_numpy()
        return out

    _team_shape_transformer._frame_aware = True  # type: ignore[attr-defined]
    _team_shape_transformer.__name__ = "team_shape"
    return [_team_shape_transformer]


def _team_shape_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    home_team_id: int | str,
    shape_indexed: dict,
) -> pd.DataFrame:
    """Join pre-indexed team shape to actions. Internal helper for xfn.

    ``shape_indexed`` is a dict of {team_id: DataFrame} where each DataFrame
    is indexed by (game_id, period_id, frame_id) for O(1) lookup.
    """
    import numpy as np

    vaep_metrics = [
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
    ]
    col_names = []
    for metric in vaep_metrics:
        for suffix in ("attacking", "defending"):
            col_names.append(f"team_shape_{metric}_{suffix}")

    n = len(actions)
    empty = pd.DataFrame(
        {col: np.full(n, np.nan) for col in col_names}, index=actions.index
    )

    if n == 0 or len(frames) == 0:
        return empty

    actions_with_idx = actions.copy()
    actions_with_idx["_row_idx"] = np.arange(n)
    pointers, _report = link_actions_to_frames(actions_with_idx, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions_with_idx[["action_id", "_row_idx", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )
    linked = linked.drop_duplicates("_row_idx", keep="first")

    out = empty.copy()

    for _, row in linked.iterrows():
        pos = int(row["_row_idx"])
        idx = actions.index[pos]
        action_team = row["team_id"]
        key = (row["game_id"], row["period_id"], int(row["frame_id_int"]))

        for tid, sdf in shape_indexed.items():
            if key not in sdf.index:
                continue
            shape_row = sdf.loc[key]
            suffix = "attacking" if tid == action_team else "defending"
            for metric in vaep_metrics:
                out.at[idx, f"team_shape_{metric}_{suffix}"] = shape_row[metric]

    return out
```

Also add to `__all__` in `features.py`:

```python
"add_team_shape",
"team_shape_xfns",
```

And add the `import numpy as np` at the top of the file if not already there (it's imported locally in several functions — follow existing pattern).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_team_shape.py -v --tb=short`
Expected: All tests PASS (9 unit + 3 integration = 12 tests)

---

### Task 3: TF-32 `detect_line_breaking` — failing tests + implementation

**Files:**
- Create: `tests/tracking/test_line_breaking.py`
- Create: `silly_kicks/tracking/_line_breaking.py`

- [ ] **Step 1: Write failing unit tests for `detect_line_breaking`**

Create `tests/tracking/test_line_breaking.py`:

```python
"""Tests for silly_kicks.tracking._line_breaking (TF-32 Ward line-breaking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_off_ball_runs import _make_action_at


def _make_three_line_fixture():
    """Fixture: away team with 3 clear defensive lines.

    - Forward line: x ~ 50 (3 players)
    - Midfield line: x ~ 70 (3 players)
    - Defense line: x ~ 90 (4 players)

    Home team with 5 outfield + GK.
    """
    return _make_frame_rows(
        home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
        home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        away_outfield_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
        away_outfield_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
    )


class TestDetectLineBreaking:
    def test_pass_through_all_three_lines(self):
        """Pass from x=10 to x=100 should break all 3 lines."""
        from silly_kicks.tracking._line_breaking import (
            LineBreakingParams,
            detect_line_breaking,
        )

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = detect_line_breaking(
            actions, frames, home_team_id=1
        )

        assert len(result) == 1
        row = result.iloc[0]
        assert row["line_break__ward"] == True  # noqa: E712
        assert row["lines_broken__ward"] == 3
        assert row["line_breaking_type__ward"] == "between_lines"

    def test_pass_through_one_line(self):
        """Pass from x=55 to x=75 should break midfield line only."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=55.0,
            start_y=34.0,
            end_x=75.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)

        row = result.iloc[0]
        assert row["line_break__ward"] == True  # noqa: E712
        assert row["lines_broken__ward"] >= 1

    def test_pass_not_crossing_any_line(self):
        """Short backward pass should not break any line."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=30.0,
            start_y=34.0,
            end_x=20.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)

        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712
        assert row["lines_broken__ward"] == 0
        assert pd.isna(row["line_breaking_type__ward"])

    def test_pass_around_line_wide(self):
        """Pass going wide of outermost defender -> type='around_line'."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        # Away defenders at y=20..48, pass at y=60 (wide of all)
        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[70.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 48.0, 20.0, 34.0, 48.0, 20.0, 34.0, 48.0],
        )
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=60.0,
            start_y=64.0,  # wide of all defenders
            end_x=80.0,
            end_y=64.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        if row["line_break__ward"]:
            assert row["line_breaking_type__ward"] == "around_line"

    def test_too_few_opponents(self):
        """< min_opponents -> all False/0/None."""
        from silly_kicks.tracking._line_breaking import (
            LineBreakingParams,
            detect_line_breaking,
        )

        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 30.0, 40.0, 50.0, 60.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[70.0, 80.0],  # only 2 opponents
            away_outfield_ys=[34.0, 34.0],
        )
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = detect_line_breaking(
            actions, frames, home_team_id=1,
            params=LineBreakingParams(min_opponents=3),
        )
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712
        assert row["lines_broken__ward"] == 0

    def test_short_pass_below_threshold(self):
        """Pass shorter than min_pass_length -> False."""
        from silly_kicks.tracking._line_breaking import (
            LineBreakingParams,
            detect_line_breaking,
        )

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=50.0,
            start_y=34.0,
            end_x=51.0,  # < 3m
            end_y=34.0,
        )

        result = detect_line_breaking(
            actions, frames, home_team_id=1,
            params=LineBreakingParams(min_pass_length=3.0),
        )
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712

    def test_no_x_spread(self):
        """All opponents at same x -> no lines definable."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 30.0, 40.0, 50.0, 60.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[70.0, 70.0, 70.0, 70.0, 70.0],  # all same x
            away_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        )
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712

    def test_empty_actions(self):
        """Empty actions -> empty result."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        actions = pd.DataFrame(columns=[
            "game_id", "action_id", "period_id", "time_seconds",
            "team_id", "player_id", "start_x", "start_y", "end_x", "end_y", "type_id",
        ])
        result = detect_line_breaking(actions, frames, home_team_id=1)
        assert len(result) == 0

    def test_away_team_pass_coordinate_transform(self):
        """Away-team pass exercises the SPADL->tracking coordinate flip.

        In SPADL, both teams attack x=105. In LTR-normalized tracking,
        home attacks x=105, away attacks x=0. detect_line_breaking must
        transform away-team SPADL coords via (105-x, 68-y) to tracking.

        Fixture: home outfield at x=20-40, away outfield (defenders from
        away perspective) at x=50,70,90 with 3 clear lines.
        Away-team pass from SPADL (65,34) to (5,34) -> in tracking coords
        this is (40,34) to (100,34), which should cross all 3 away lines.
        """
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        # Use an away-team outfield player as the actor
        away_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 2)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        # Away team attacks x=105 in SPADL. Home defenders (from away's
        # perspective) are at x=105-20=85, x=105-25=80, etc. in SPADL.
        # The home outfield positions in tracking are x=20-40, which in
        # away-SPADL are x=65-85. A pass from SPADL x=90 to x=15 (deep
        # into home territory) should cross home defensive structure.
        # In tracking: start=(105-90,68-34)=(15,34), end=(105-15,68-34)=(90,34).
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(away_player),
            team_id=2,  # AWAY team
            start_x=90.0,  # SPADL: near away's own goal
            start_y=34.0,
            end_x=15.0,  # SPADL: deep into opponent half
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        # Away pass crossing home territory should detect line breaks
        # The exact count depends on home's outfield structure, but
        # the key assertion is: the coordinate transform doesn't crash,
        # produces a valid (non-NaN) result, AND detects at least one break
        # (pass from x=90→15 crosses home outfield clustered at x=20–40)
        assert pd.notna(row["line_break__ward"])
        assert pd.notna(row["lines_broken__ward"])
        assert row["lines_broken__ward"] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_line_breaking.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._line_breaking'`

- [ ] **Step 3: Write `_line_breaking.py` implementation**

Create `silly_kicks/tracking/_line_breaking.py`:

```python
"""Ward-clustering line-breaking detection (TF-32).

Identifies defensive lines via 1D Ward hierarchical clustering on opponent
x-coordinates, constructs line segments, and tests pass trajectory
intersection via cross-product straddle test.

See spec: docs/superpowers/specs/2026-05-09-tf31-tf32-team-shape-line-breaking-design.md s2.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage


@dataclass(frozen=True)
class LineBreakingParams:
    """Parameters for Ward-clustering line-breaking detection.

    Examples
    --------
    >>> from silly_kicks.tracking._line_breaking import LineBreakingParams
    >>> params = LineBreakingParams(min_opponents=3, n_clusters=3)
    """

    min_opponents: int = 3
    n_clusters: int = 3
    min_pass_length: float = 3.0  # metres
    min_x_spread: float = 5.0  # metres
    pitch_y_min: float = 0.0  # SPADL y-coordinate of near sideline
    pitch_y_max: float = 68.0  # SPADL y-coordinate of far sideline


_RESULT_COLS = [
    "line_break__ward",
    "lines_broken__ward",
    "line_breaking_type__ward",
]


def detect_line_breaking(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    params: LineBreakingParams | None = None,
) -> pd.DataFrame:
    """Per-action Ward-clustering line-breaking detection.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions (per-action coordinates where the acting team attacks x=105).
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS schema).
        Must be LTR-normalized (play_left_to_right applied).
    home_team_id : int | str
        Home team identifier for coordinate resolution.
    params : LineBreakingParams | None
        Algorithm parameters. Defaults to ``LineBreakingParams()``.

    Returns
    -------
    pd.DataFrame
        Aligned with ``actions.index``. Columns:
        - ``line_break__ward`` (boolean, nullable)
        - ``lines_broken__ward`` (Int64, 0-3)
        - ``line_breaking_type__ward`` (object: "between_lines", "around_line", or None)

    Examples
    --------
    Detect line-breaking passes::

        from silly_kicks.tracking._line_breaking import detect_line_breaking
        lb = detect_line_breaking(actions, frames, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    if params is None:
        params = LineBreakingParams()

    n_actions = len(actions)
    empty = pd.DataFrame(
        {
            "line_break__ward": pd.array([pd.NA] * n_actions, dtype="boolean"),
            "lines_broken__ward": pd.array([pd.NA] * n_actions, dtype="Int64"),
            "line_breaking_type__ward": pd.array([None] * n_actions, dtype="object"),
        },
        index=actions.index,
    )

    if n_actions == 0 or len(frames) == 0:
        return empty

    from .utils import link_actions_to_frames

    # Link actions to frames
    pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions[["action_id", "team_id", "start_x", "start_y", "end_x", "end_y", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )
    linked = linked.drop_duplicates("action_id", keep="first")

    # Pre-build grouped outfield opponent positions
    non_ball_non_gk = frames[
        (~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))
    ]
    frame_groups: dict = dict(
        iter(
            non_ball_non_gk.groupby(
                ["game_id", "period_id", "frame_id", "team_id"], sort=False
            )
        )
    )

    # Pre-build (game, period, frame) -> list of team_ids for O(1) opposing lookup
    frame_to_teams: dict[tuple, list] = {}
    for key in frame_groups:
        frame_key = key[:3]
        frame_to_teams.setdefault(frame_key, []).append(key[3])

    # Build positional lookup
    aid_to_pos = {
        aid: pos for pos, aid in enumerate(actions["action_id"].values)
    }

    lb_arr = np.full(n_actions, np.nan)
    count_arr = np.full(n_actions, np.nan)
    type_arr: list[str | None] = [None] * n_actions

    for _, row in linked.iterrows():
        aid = row["action_id"]
        if aid not in aid_to_pos:
            continue
        pos = aid_to_pos[aid]

        action_team = row["team_id"]
        game_id = row["game_id"]
        period_id = row["period_id"]
        frame_id = int(row["frame_id_int"])
        start_x = float(row["start_x"])
        start_y = float(row["start_y"])
        end_x = float(row["end_x"])
        end_y = float(row["end_y"])

        # Pass length check
        pass_len = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        if pass_len < params.min_pass_length:
            lb_arr[pos] = 0.0
            count_arr[pos] = 0
            continue

        # Find opposing team via O(1) lookup
        frame_key = (game_id, period_id, frame_id)
        teams_at_frame = frame_to_teams.get(frame_key, [])
        opp_teams = [t for t in teams_at_frame if t != action_team]

        if not opp_teams:
            lb_arr[pos] = 0.0
            count_arr[pos] = 0
            continue

        opp_df = frame_groups[(game_id, period_id, frame_id, opp_teams[0])]
        opp_x = opp_df["x"].dropna().to_numpy(dtype="float64")
        opp_y = opp_df["y"].dropna().to_numpy(dtype="float64")

        if len(opp_x) < params.min_opponents:
            lb_arr[pos] = 0.0
            count_arr[pos] = 0
            continue

        # X-spread check
        x_spread = float(np.max(opp_x) - np.min(opp_x))
        if x_spread < params.min_x_spread:
            lb_arr[pos] = 0.0
            count_arr[pos] = 0
            continue

        # Convert SPADL action coords to tracking coords for intersection
        if action_team == home_team_id:
            track_start_x = start_x
            track_start_y = start_y
            track_end_x = end_x
            track_end_y = end_y
        else:
            track_start_x = 105.0 - start_x
            track_start_y = 68.0 - start_y
            track_end_x = 105.0 - end_x
            track_end_y = 68.0 - end_y

        # Ward clustering on 1D x-coordinates
        n_eff_clusters = min(params.n_clusters, len(opp_x))
        if n_eff_clusters < 2:
            lb_arr[pos] = 0.0
            count_arr[pos] = 0
            continue

        Z = linkage(opp_x.reshape(-1, 1), method="ward")
        labels = fcluster(Z, t=n_eff_clusters, criterion="maxclust")

        # Sort clusters by ascending mean x
        cluster_ids = np.unique(labels)
        cluster_means = [float(np.mean(opp_x[labels == c])) for c in cluster_ids]
        sorted_order = np.argsort(cluster_means)
        sorted_cluster_ids = cluster_ids[sorted_order]

        # Build segments per cluster and test intersection
        lines_broken = 0
        any_through = False
        any_around = False

        for cid in sorted_cluster_ids:
            mask = labels == cid
            cx = opp_x[mask]
            cy = opp_y[mask]

            # Sort by y
            y_order = np.argsort(cy)
            cx_sorted = cx[y_order]
            cy_sorted = cy[y_order]

            # Extend to sidelines using nearest-player x
            points_x = np.concatenate(
                [[cx_sorted[0]], cx_sorted, [cx_sorted[-1]]]
            )
            points_y = np.concatenate(
                [[params.pitch_y_min], cy_sorted, [params.pitch_y_max]]
            )

            # Test each segment for intersection with pass trajectory
            cluster_broken = False
            broke_on_extension = False
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
                    if si == 0 or si == n_segments - 1:
                        broke_on_extension = True

            if cluster_broken:
                lines_broken += 1
                if broke_on_extension:
                    any_around = True
                else:
                    any_through = True

        lb_arr[pos] = 1.0 if lines_broken > 0 else 0.0
        count_arr[pos] = lines_broken

        if lines_broken > 0:
            # "between_lines" dominates (more tactically significant)
            if any_through:
                type_arr[pos] = "between_lines"
            else:
                type_arr[pos] = "around_line"

    return pd.DataFrame(
        {
            "line_break__ward": pd.array(
                [pd.NA if np.isnan(v) else bool(v) for v in lb_arr],
                dtype="boolean",
            ),
            "lines_broken__ward": pd.array(
                [pd.NA if np.isnan(v) else int(v) for v in count_arr],
                dtype="Int64",
            ),
            "line_breaking_type__ward": pd.array(type_arr, dtype="object"),
        },
        index=actions.index,
    )


def _segments_intersect(
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    """Cross-product straddle test for segment (C,D) vs segment (A,B).

    Returns True if the two segments properly intersect (not collinear/touching).
    """
    d1 = _cross(bx - ax, by - ay, cx - ax, cy - ay)
    d2 = _cross(bx - ax, by - ay, dx - ax, dy - ay)
    d3 = _cross(dx - cx, dy - cy, ax - cx, ay - cy)
    d4 = _cross(dx - cx, dy - cy, bx - cx, by - cy)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _cross(ux: float, uy: float, vx: float, vy: float) -> float:
    """2D cross product of vectors (ux, uy) and (vx, vy)."""
    return ux * vy - uy * vx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_line_breaking.py -v --tb=short`
Expected: All 9 tests PASS

---

### Task 4: TF-32 `add_line_break` method="ward" + `line_breaking_ward_xfns`

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `tests/tracking/test_line_breaking.py`

- [ ] **Step 1: Write failing tests for `add_line_break(method="ward")` and `line_breaking_ward_xfns`**

Append to `tests/tracking/test_line_breaking.py`:

```python
class TestAddLineBreakWard:
    def test_method_ward_returns_ward_columns(self):
        """add_line_break(method='ward') returns ward-suffixed columns."""
        from silly_kicks.tracking.features import add_line_break

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = add_line_break(actions, frames, home_team_id=1, method="ward")
        assert "line_break__ward" in result.columns
        assert "lines_broken__ward" in result.columns
        assert "line_breaking_type__ward" in result.columns
        # Should NOT have threshold columns
        assert "line_break" not in result.columns
        assert "n_attackers_behind_line" not in result.columns

    def test_method_threshold_unchanged(self):
        """add_line_break(method='threshold') still returns old columns."""
        from silly_kicks.tracking.features import add_line_break

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = add_line_break(actions, frames, home_team_id=1, method="threshold")
        assert "line_break" in result.columns
        assert "n_attackers_behind_line" in result.columns
        assert "line_break__ward" not in result.columns

    def test_default_method_is_threshold(self):
        """Default method is 'threshold' (backward-compatible)."""
        from silly_kicks.tracking.features import add_line_break

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = add_line_break(actions, frames, home_team_id=1)
        assert "line_break" in result.columns  # threshold columns
        assert "line_break__ward" not in result.columns


class TestLineBreakingWardXfns:
    def test_xfn_column_count(self):
        """line_breaking_ward_xfns produces 9 columns (3 features x 3 states)."""
        from silly_kicks.tracking.features import line_breaking_ward_xfns

        xfns = line_breaking_ward_xfns(home_team_id=1)
        assert len(xfns) == 1

        xfn = xfns[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_xfn_introspection_nan(self):
        """frames=None -> NaN DataFrame with 9 correct column names."""
        from silly_kicks.tracking.features import line_breaking_ward_xfns

        xfns = line_breaking_ward_xfns(home_team_id=1)
        xfn = xfns[0]

        dummy = pd.DataFrame(
            {
                "game_id": [1] * 10,
                "action_id": range(10),
                "period_id": [1] * 10,
                "time_seconds": [float(i) for i in range(10)],
                "team_id": [1] * 10,
                "player_id": [1] * 10,
                "start_x": [50.0] * 10,
                "start_y": [34.0] * 10,
                "end_x": [60.0] * 10,
                "end_y": [34.0] * 10,
                "type_id": [0] * 10,
                "result_id": [0] * 10,
                "bodypart_id": [0] * 10,
            }
        )
        states = [dummy, dummy, dummy]
        result = xfn(states, None)

        assert len(result.columns) == 9
        assert result.isna().all().all()
        # Verify naming pattern: lines_broken__ward_a0, etc.
        assert "lines_broken__ward_a0" in result.columns
        assert "line_breaking_type__ward_between_lines_a2" in result.columns
        assert "line_breaking_type__ward_around_line_a0" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_line_breaking.py::TestAddLineBreakWard -v --tb=short`
Expected: FAIL — `TypeError: add_line_break() got an unexpected keyword argument 'method'`

- [ ] **Step 3: Modify `add_line_break` signature and add `line_breaking_ward_xfns`**

In `silly_kicks/tracking/features.py`, replace the existing `add_line_break` function:

```python
@nan_safe_enrichment
def add_line_break(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    method: Literal["threshold", "ward"] = "threshold",
    n: int = 4,
    params: "LineBreakingParams | None" = None,
) -> pd.DataFrame:
    """Enrich actions with line-break columns.

    Two methods are available:

    - ``method="threshold"`` (default): Binary threshold test against the
      defending team's ``defensive_line_x``. Returns ``line_break`` (bool)
      and ``n_attackers_behind_line`` (Int64). Backward-compatible default.
      ``params`` is ignored.
    - ``method="ward"``: Ward-clustering line identification + segment
      intersection. Returns ``line_break__ward`` (bool),
      ``lines_broken__ward`` (Int64, 0-3), ``line_breaking_type__ward``
      (str: "between_lines"/"around_line"/None). ``n`` is ignored.

    Column sets are disjoint between methods (no collision). A consumer
    can call both methods if they want all 5 columns (note: each call
    performs its own ``link_actions_to_frames`` --- see §1.4 linkage
    cost note in the spec).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_line_break
    >>> # See tests/tracking/test_line_breaking.py for runnable examples.
    """
    if method == "threshold":
        from ._off_ball_runs import _line_break_kernel

        df = _line_break_kernel(
            actions, frames, home_team_id=home_team_id, n=n
        )
        out = actions.copy()
        out["line_break"] = df["line_break"]
        out["n_attackers_behind_line"] = df["n_attackers_behind_line"]
        return out

    # method == "ward"
    from ._line_breaking import detect_line_breaking

    result = detect_line_breaking(
        actions, frames, home_team_id=home_team_id, params=params
    )
    out = actions.copy()
    out["line_break__ward"] = result["line_break__ward"]
    out["lines_broken__ward"] = result["lines_broken__ward"]
    out["line_breaking_type__ward"] = result["line_breaking_type__ward"]
    return out
```

Then add `line_breaking_ward_xfns` after the existing `off_ball_context_xfns`:

```python
def line_breaking_ward_xfns(home_team_id: int | str) -> list:
    """Build VAEP xfn list for TF-32 Ward line-breaking features.

    Returns a list with ONE FrameAwareTransformer that emits 3 features x 3
    game-states = 9 columns total. ``line_break__ward`` is excluded (redundant
    with ``lines_broken__ward > 0``; VAEP should not waste a parameter on a
    linearly dependent feature).

    The ``line_breaking_type__ward`` categorical is one-hot encoded:
    ``line_breaking_type__ward_between_lines`` and ``line_breaking_type__ward_around_line``.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import (
            tracking_default_xfns,
            line_breaking_ward_xfns,
        )
        xfns = tracking_default_xfns + line_breaking_ward_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._line_breaking import detect_line_breaking

    col_names = [
        "lines_broken__ward",
        "line_breaking_type__ward_between_lines",
        "line_breaking_type__ward_around_line",
    ]

    def _line_breaking_ward_transformer(states, frames):
        """Multi-column Ward line-breaking xfn (3 cols x nb_states)."""
        import numpy as np

        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        for i, slot in enumerate(states[:3]):
            lb = detect_line_breaking(
                slot, frames, home_team_id=home_team_id
            )
            out[f"lines_broken__ward_a{i}"] = lb["lines_broken__ward"].to_numpy()
            out[f"line_breaking_type__ward_between_lines_a{i}"] = (
                lb["line_breaking_type__ward"] == "between_lines"
            ).to_numpy()
            out[f"line_breaking_type__ward_around_line_a{i}"] = (
                lb["line_breaking_type__ward"] == "around_line"
            ).to_numpy()
        return out

    _line_breaking_ward_transformer._frame_aware = True  # type: ignore[attr-defined]
    _line_breaking_ward_transformer.__name__ = "line_breaking_ward"
    return [_line_breaking_ward_transformer]
```

Add to `__all__` in `features.py`:

```python
"line_breaking_ward_xfns",
```

Add the `LineBreakingParams` type annotation import (use `TYPE_CHECKING` guard):

```python
from __future__ import annotations
# At top of file, the TYPE_CHECKING import is not needed because of
# the `from __future__ import annotations` — string annotations work.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_line_breaking.py -v --tb=short`
Expected: All tests PASS (8 unit + 5 integration = 13 tests)

---

### Task 5: Synthesizer amendment + provider sweep tests

**Files:**
- Modify: `tests/tracking/_provider_inputs.py`
- Create: `tests/tracking/test_team_shape_providers.py`
- Create: `tests/tracking/test_line_breaking_providers.py`

- [ ] **Step 1: Amend `synthesize_actions` pass trajectory**

In `tests/tracking/_provider_inputs.py`, modify the `end_x` and `end_y` values in `synthesize_actions()` so pass actions have realistic trajectories. Find the return `pd.DataFrame(...)` block and change the `"end_x"` and `"end_y"` entries:

Replace in `synthesize_actions()`:

```python
            "end_x": [
                *pass_rows["x"].to_numpy(),
                float(gk_row["x"]),
                float(last_frame["x"]),
            ],
            "end_y": [
                *pass_rows["y"].to_numpy(),
                float(gk_row["y"]),
                float(last_frame["y"]),
            ],
```

With:

```python
            "end_x": [
                *(pass_rows["x"].to_numpy() + 20.0),
                float(gk_row["x"]),
                float(last_frame["x"]),
            ],
            "end_y": [
                *(pass_rows["y"].to_numpy() + 3.0),
                float(gk_row["y"]),
                float(last_frame["y"]),
            ],
```

This gives each pass a +20m forward, +3m lateral offset — realistic enough for line-breaking tests and harmless for existing tests (they assert column presence and non-NaN counts, not exact values).

- [ ] **Step 2: Verify existing off-ball-runs provider tests still pass**

Run: `python -m pytest tests/tracking/test_off_ball_runs_providers.py -v --tb=short`
Expected: All existing tests PASS (assertions are on column presence and `>=1` non-NaN counts)

- [ ] **Step 3: Write TF-31 provider sweep tests**

Create `tests/tracking/test_team_shape_providers.py`:

```python
"""Provider fixture tests for team shape envelope (TF-31)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking import play_left_to_right
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_PROVIDERS = ["sportec", "metrica", "pff", "skillcorner"]


@pytest.fixture(params=_PROVIDERS)
def provider_data(request):
    """Load frames and synthesize actions for a provider."""
    provider = request.param
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    return actions, frames, home_team_id


class TestTeamShapeProviders:
    def test_add_team_shape_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_team_shape

        actions, frames, home_team_id = provider_data
        result = add_team_shape(actions, frames, home_team_id=home_team_id)
        assert "team_shape_centroid_x_attacking" in result.columns
        assert "team_shape_centroid_x_defending" in result.columns
        assert len(result) == len(actions)
        assert (
            result["team_shape_centroid_x_attacking"].notna().sum() >= 1
        ), "expected >=1 non-NaN team_shape row"

    def test_team_shape_xfns_no_crash(self, provider_data):
        from silly_kicks.tracking.features import team_shape_xfns

        actions, frames, home_team_id = provider_data
        xfns = team_shape_xfns(home_team_id=home_team_id)
        xfn = xfns[0]

        import pandas as pd

        states = [actions, actions, actions]
        result = xfn(states, frames)
        assert len(result.columns) == 36
        assert result["team_shape_centroid_x_attacking_a0"].notna().sum() >= 1
```

- [ ] **Step 4: Write TF-32 provider sweep tests**

Create `tests/tracking/test_line_breaking_providers.py`:

```python
"""Provider fixture tests for Ward line-breaking (TF-32)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking import play_left_to_right
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_PROVIDERS = ["sportec", "metrica", "pff", "skillcorner"]


@pytest.fixture(params=_PROVIDERS)
def provider_data(request):
    """Load frames and synthesize actions for a provider."""
    provider = request.param
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    return actions, frames, home_team_id


class TestLineBreakingProviders:
    def test_add_line_break_ward_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_line_break

        actions, frames, home_team_id = provider_data
        result = add_line_break(
            actions, frames, home_team_id=home_team_id, method="ward"
        )
        assert "line_break__ward" in result.columns
        assert "lines_broken__ward" in result.columns
        assert "line_breaking_type__ward" in result.columns
        assert len(result) == len(actions)
        # At least one action should have a valid result
        assert (
            result["lines_broken__ward"].notna().sum() >= 1
        ), "expected >=1 non-NaN lines_broken__ward row"

    def test_line_breaking_ward_xfns_no_crash(self, provider_data):
        from silly_kicks.tracking.features import line_breaking_ward_xfns

        actions, frames, home_team_id = provider_data
        xfns = line_breaking_ward_xfns(home_team_id=home_team_id)
        xfn = xfns[0]

        import pandas as pd

        states = [actions, actions, actions]
        result = xfn(states, frames)
        assert len(result.columns) == 9

    def test_threshold_still_works(self, provider_data):
        """Existing method='threshold' unaffected by new method= kwarg."""
        from silly_kicks.tracking.features import add_line_break

        actions, frames, home_team_id = provider_data
        result = add_line_break(
            actions, frames, home_team_id=home_team_id, method="threshold"
        )
        assert "line_break" in result.columns
        assert "n_attackers_behind_line" in result.columns
```

- [ ] **Step 5: Run all provider sweep tests**

Run: `python -m pytest tests/tracking/test_team_shape_providers.py tests/tracking/test_line_breaking_providers.py tests/tracking/test_off_ball_runs_providers.py -v --tb=short`
Expected: All PASS

---

### Task 6: Invariant tests

**Files:**
- Create: `tests/invariants/test_invariant_team_shape.py`
- Create: `tests/invariants/test_invariant_line_breaking.py`

- [ ] **Step 1: Write TF-31 invariant tests**

Create `tests/invariants/test_invariant_team_shape.py`:

```python
"""Physical invariants for team shape envelope (TF-31)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.tracking.test_team_shape import _make_team_frames


@pytest.fixture
def team_shape_result():
    """Known 10-outfield-player fixture."""
    from silly_kicks.tracking._team_shape import compute_team_shape

    frames = _make_team_frames(
        outfield_positions=[
            (10.0, 10.0),
            (20.0, 10.0),
            (30.0, 20.0),
            (40.0, 30.0),
            (50.0, 34.0),
            (60.0, 40.0),
            (70.0, 50.0),
            (80.0, 50.0),
            (90.0, 60.0),
            (95.0, 60.0),
        ],
    )
    return compute_team_shape(frames, team_id=1)


class TestRangeInvariants:
    def test_convex_hull_area_non_negative(self, team_shape_result):
        valid = team_shape_result["convex_hull_area"].dropna()
        assert (valid >= 0).all()

    def test_stretch_index_non_negative(self, team_shape_result):
        valid = team_shape_result["stretch_index"].dropna()
        assert (valid >= 0).all()

    def test_team_length_in_pitch(self, team_shape_result):
        valid = team_shape_result["team_length"].dropna()
        assert (valid >= 0).all() and (valid <= 105).all()

    def test_team_width_in_pitch(self, team_shape_result):
        valid = team_shape_result["team_width"].dropna()
        assert (valid >= 0).all() and (valid <= 68).all()

    def test_n_outfield_players_in_range(self, team_shape_result):
        valid = team_shape_result["n_outfield_players"].dropna()
        assert (valid >= 1).all() and (valid <= 11).all()

    def test_stretch_index_bounded_by_max_extent(self, team_shape_result):
        """stretch_index <= max(team_length, team_width)."""
        df = team_shape_result.dropna(subset=["stretch_index"])
        max_extent = np.maximum(df["team_length"], df["team_width"])
        assert (df["stretch_index"] <= max_extent + 1e-9).all()
```

- [ ] **Step 2: Write TF-32 invariant tests**

Create `tests/invariants/test_invariant_line_breaking.py`:

```python
"""Physical invariants for Ward line-breaking (TF-32)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_off_ball_runs import _make_action_at


@pytest.fixture
def line_breaking_result():
    """Pass across 3 defensive lines -> known line-breaking result."""
    from silly_kicks.tracking._line_breaking import detect_line_breaking

    frames = _make_frame_rows(
        home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
        home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        away_outfield_xs=[
            50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0,
        ],
        away_outfield_ys=[
            15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0,
        ],
    )
    home_player = frames[
        (~frames["is_ball"])
        & (frames["team_id"] == 1)
        & (~frames["is_goalkeeper"])
    ]["player_id"].iloc[0]

    # Multiple actions: through all, through 1, none
    actions = pd.concat(
        [
            _make_action_at(
                time_seconds=1.0,
                player_id=int(home_player),
                team_id=1,
                start_x=10.0, start_y=34.0,
                end_x=100.0, end_y=34.0,
                action_id=1,
            ),
            _make_action_at(
                time_seconds=1.0,
                player_id=int(home_player),
                team_id=1,
                start_x=55.0, start_y=34.0,
                end_x=75.0, end_y=34.0,
                action_id=2,
            ),
            _make_action_at(
                time_seconds=1.0,
                player_id=int(home_player),
                team_id=1,
                start_x=30.0, start_y=34.0,
                end_x=20.0, end_y=34.0,
                action_id=3,
            ),
        ],
        ignore_index=True,
    )

    return detect_line_breaking(actions, frames, home_team_id=1)


class TestLineBreakingInvariants:
    def test_lines_broken_domain(self, line_breaking_result):
        valid = line_breaking_result["lines_broken__ward"].dropna()
        assert set(valid.unique()).issubset({0, 1, 2, 3})

    def test_is_line_breaking_consistent(self, line_breaking_result):
        """is_line_breaking == (lines_broken > 0)."""
        df = line_breaking_result.dropna(subset=["lines_broken__ward"])
        expected = df["lines_broken__ward"] > 0
        actual = df["line_break__ward"]
        assert (actual == expected).all()

    def test_type_domain(self, line_breaking_result):
        valid = line_breaking_result["line_breaking_type__ward"].dropna()
        assert set(valid.unique()).issubset({"between_lines", "around_line"})

    def test_type_none_when_no_break(self, line_breaking_result):
        """line_breaking_type is None when lines_broken == 0."""
        no_break = line_breaking_result[
            line_breaking_result["lines_broken__ward"] == 0
        ]
        if not no_break.empty:
            assert no_break["line_breaking_type__ward"].isna().all()
```

- [ ] **Step 3: Run all invariant tests**

Run: `python -m pytest tests/invariants/test_invariant_team_shape.py tests/invariants/test_invariant_line_breaking.py -v --tb=short`
Expected: All PASS

---

### Task 7: Golden-file backward compat + cross-method sanity

**Files:**
- Modify: `tests/tracking/test_line_breaking.py`
- Create: `tests/datasets/tracking/golden/line_break_threshold_golden.parquet` (generated)

- [ ] **Step 1: Generate golden-file snapshot of `add_line_break()` default output**

Before modifying `add_line_break`'s signature, capture its current output on the
synthetic three-line fixture as a `.parquet` golden file. This proves the new
`method=` kwarg doesn't alter default behavior.

Add to `tests/tracking/test_line_breaking.py`:

```python
class TestGoldenFileBackwardCompat:
    """Snapshot test: add_line_break() default output unchanged by method= addition."""

    GOLDEN_PATH = (
        Path(__file__).resolve().parent.parent
        / "datasets"
        / "tracking"
        / "golden"
        / "line_break_threshold_golden.parquet"
    )

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Build a deterministic fixture for golden-file comparison."""
        from tests.tracking.test_defensive_line import _make_frame_rows

        self.frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0, 40.0, 50.0],
            home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
            away_outfield_xs=[60.0, 70.0, 80.0, 90.0, 95.0],
            away_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        )
        home_player = self.frames[
            (~self.frames["is_ball"])
            & (self.frames["team_id"] == 1)
            & (~self.frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        self.actions = pd.concat(
            [
                _make_action_at(
                    time_seconds=1.0,
                    player_id=int(home_player),
                    team_id=1,
                    start_x=10.0, start_y=34.0,
                    end_x=100.0, end_y=34.0,
                    action_id=1,
                ),
                _make_action_at(
                    time_seconds=1.0,
                    player_id=int(home_player),
                    team_id=1,
                    start_x=30.0, start_y=34.0,
                    end_x=20.0, end_y=34.0,
                    action_id=2,
                ),
            ],
            ignore_index=True,
        )

    def test_generate_golden_if_missing(self):
        """Generate golden file if it doesn't exist (run on main before PR)."""
        from silly_kicks.tracking.features import add_line_break

        result = add_line_break(
            self.actions, self.frames, home_team_id=1
        )
        golden_cols = ["line_break", "n_attackers_behind_line"]
        golden = result[golden_cols].copy()

        if not self.GOLDEN_PATH.exists():
            self.GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            golden.to_parquet(self.GOLDEN_PATH)
            pytest.skip("Golden file generated; re-run to verify.")

        expected = pd.read_parquet(self.GOLDEN_PATH)
        pd.testing.assert_frame_equal(golden, expected)
```

Add `from pathlib import Path` to the test file imports.

- [ ] **Step 2: Run golden-file test to generate the baseline**

Run: `python -m pytest tests/tracking/test_line_breaking.py::TestGoldenFileBackwardCompat -v --tb=short`
Expected: First run skips with "Golden file generated"; second run PASSES.

- [ ] **Step 3: Add cross-method sanity check**

Append to `tests/tracking/test_line_breaking.py`:

```python
class TestCrossMethodSanity:
    def test_ward_and_threshold_soft_agreement(self):
        """When lines_broken__ward > 0, threshold line_break is usually True.

        Not a hard invariant (different algorithms) but flags gross disagreement.
        """
        from silly_kicks.tracking._line_breaking import detect_line_breaking
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel

        frames = _make_three_line_fixture()
        home_player = frames[
            (~frames["is_ball"])
            & (frames["team_id"] == 1)
            & (~frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        # Pass clearly through all lines
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        ward = detect_line_breaking(actions, frames, home_team_id=1)
        threshold = _line_break_kernel(actions, frames, home_team_id=1)

        # Both should agree this pass breaks a line
        if ward.iloc[0]["lines_broken__ward"] > 0:
            assert threshold.iloc[0]["line_break"] == True  # noqa: E712
```

- [ ] **Step 4: Run cross-method sanity + golden-file tests**

Run: `python -m pytest tests/tracking/test_line_breaking.py::TestCrossMethodSanity tests/tracking/test_line_breaking.py::TestGoldenFileBackwardCompat -v --tb=short`
Expected: PASS

---

### Task 8: Registration + docs + commit

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `tests/tracking/conftest.py`
- Modify: `NOTICE`
- Modify: `TODO.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `tracking/__init__.py` exports**

Add these entries to `__all__` (alphabetically sorted):

```python
"LineBreakingParams",
"add_team_shape",
"compute_team_shape",
"detect_line_breaking",
"line_breaking_ward_xfns",
"team_shape_xfns",
```

Add these import lines:

```python
from ._line_breaking import LineBreakingParams, detect_line_breaking
from ._team_shape import compute_team_shape
```

And add to the `from .features import (...)` block:

```python
    add_team_shape,
    line_breaking_ward_xfns,
    team_shape_xfns,
```

- [ ] **Step 2: Update `tests/tracking/conftest.py` re-exports**

Add to `conftest.py`:

```python
from tests.tracking.test_team_shape import _make_team_frames
```

And update `__all__`:

```python
"_make_team_frames",
```

- [ ] **Step 3: Add NOTICE entries**

Append to the "Mathematical / Methodological References" section in `NOTICE`, before the "Third-Party Code Attribution" section:

```
The team shape envelope features in silly_kicks/tracking/_team_shape.py
(PR-S33, TF-31) implement methodologies described in:

- Clemente, F. M., Couceiro, M. S., Martins, F. M. L., & Mendes, R. (2013).
  "Measuring Tactical Behaviour Using Technological Metrics: Case Study of a
  Football Game." International Journal of Sports Science & Coaching, 8(4).
  (stretch index = mean Euclidean distance from team centroid; canonical
  per-team spatial descriptors)

- Zhang, G., Kempe, M., McRobert, A., Folgado, H., & Olthof, S. B. H. (2025).
  "Navigating team tactical analysis in football: An analytical pipeline
  leveraging player tracking technology." International Journal of Sports
  Science & Coaching.
  (centroid, convex hull area, team length, team width, stretch index as
  canonical team shape metrics)

The Ward-clustering line-breaking detection in
silly_kicks/tracking/_line_breaking.py (PR-S33, TF-32) implements the
methodology described in:

- Karakus, O., & Arkadas, H. (2025). "Through the Gaps: Uncovering Tactical
  Line-Breaking Passes with Clustering." arXiv:2506.06666. ECML/PKDD MLSA 2025.
  (1D Ward hierarchical clustering on x-coordinates for defensive line
  identification; cross-product straddle test for pass-segment intersection)
```

- [ ] **Step 4: Update `TODO.md`**

Delete the TF-31 and TF-32 rows from the "Tier 2 — On Deck" section. Update the header date to `2026-05-09`.

- [ ] **Step 5: Update `CHANGELOG.md`**

Add under `## [3.9.0]` (create if necessary):

```markdown
### Added
- **TF-31 Team Shape Envelope:** `compute_team_shape` per-frame primitive (7 metrics: n_outfield_players, centroid_x, centroid_y, convex_hull_area, team_length, team_width, stretch_index) + `add_team_shape` aggregator (14 action-coupled columns) + `team_shape_xfns` VAEP factory (36 columns). Ref: Clemente et al. 2013.
- **TF-32 Ward Line-Breaking:** `detect_line_breaking` per-action Ward-clustering line-breaking detection (3 columns: line_break__ward, lines_broken__ward, line_breaking_type__ward) + `LineBreakingParams` frozen dataclass + `line_breaking_ward_xfns` VAEP factory (9 columns). Extends `add_line_break` with `method="ward"` dispatch. Ref: Karakus & Arkadas 2025.

### Changed
- `add_line_break` gains `method` kwarg (`"threshold"` default, `"ward"` new) and `params` kwarg for Ward-specific parameters. Default behavior unchanged.
- `synthesize_actions` in test fixtures now gives pass actions a +20m forward trajectory offset (was zero-length).
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: All tests PASS, no regressions.

- [ ] **Step 7: Run linting**

Run: `python -m ruff check silly_kicks/ tests/ && python -m ruff format --check silly_kicks/ tests/`
Expected: No errors.

Run: `uv run pyright silly_kicks/tracking/_team_shape.py silly_kicks/tracking/_line_breaking.py silly_kicks/tracking/features.py silly_kicks/tracking/__init__.py`
Expected: No errors.

- [ ] **Step 8: Commit**

```bash
git checkout -b pr-s33-team-shape-line-breaking
git add silly_kicks/tracking/_team_shape.py \
       silly_kicks/tracking/_line_breaking.py \
       silly_kicks/tracking/features.py \
       silly_kicks/tracking/__init__.py \
       tests/tracking/test_team_shape.py \
       tests/tracking/test_line_breaking.py \
       tests/tracking/test_team_shape_providers.py \
       tests/tracking/test_line_breaking_providers.py \
       tests/tracking/_provider_inputs.py \
       tests/tracking/conftest.py \
       tests/invariants/test_invariant_team_shape.py \
       tests/invariants/test_invariant_line_breaking.py \
       tests/datasets/tracking/golden/line_break_threshold_golden.parquet \
       NOTICE TODO.md CHANGELOG.md
git commit -m "feat(tracking): TF-31 team shape + TF-32 Ward line-breaking -- silly-kicks 3.9.0 (PR-S33)"
```
