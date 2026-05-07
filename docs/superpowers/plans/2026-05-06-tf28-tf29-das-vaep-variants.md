# TF-28 + TF-29: DAS Adapter & VAEP Design-Space Variants — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `accessible-space` DAS adapter + VAEP windowing/goalscore-bias variants as silly-kicks 3.8.0.

**Architecture:** TF-28 is a thin adapter module (`tracking/_das.py`) wrapping the `accessible-space` PyPI package with coordinate transforms, input validation, and the standard action-coupled VAEP integration layer. TF-29 adds `window=` parameter to `scores()`/`concedes()` in `vaep/labels.py` (3 modes: action/possession/time) and publishes goalscore-free xfn variant lists. Both are independent — TF-28 touches `tracking/`, TF-29 touches `vaep/`.

**Tech Stack:** pandas, numpy, accessible-space (optional dep), pytest

**Spec:** `docs/superpowers/specs/2026-05-06-tf28-tf29-das-vaep-variants-design.md`

---

## File Map

### New files
| File | Responsibility |
|------|---------------|
| `tests/tracking/test_das.py` | DAS adapter unit tests |
| `tests/tracking/test_das_e2e.py` | DAS per-provider e2e tests |
| `tests/invariants/test_das_invariants.py` | DAS physical invariants |
| `tests/vaep/test_labels_windowing.py` | Windowing unit + hand-crafted fixture tests |
| `tests/vaep/test_labels_windowing_e2e.py` | Windowing real-data e2e tests |

### Modified files
| File | What changes |
|------|-------------|
| `silly_kicks/tracking/_ball_carrier.py` | Add `derive_team_in_possession()` |
| `silly_kicks/tracking/_das.py` | **New** — DAS adapter module |
| `silly_kicks/tracking/__init__.py` | Re-export new DAS + carrier symbols |
| `silly_kicks/tracking/features.py` | Add `das_at_action`, `add_das`, `das_xfns` |
| `silly_kicks/vaep/labels.py` | Add `window` + `window_seconds` params |
| `silly_kicks/vaep/base.py` | Add `xfns_default_no_goalscore` |
| `silly_kicks/vaep/hybrid.py` | Add `hybrid_xfns_default_no_goalscore` |
| `silly_kicks/vaep/__init__.py` | Re-export new symbols |
| `pyproject.toml` | Add `[das]` optional extra |
| `NOTICE` | Add Bischofberger & Baca 2026 + DTAI blog refs |
| `CHANGELOG.md` | 3.8.0 entry |
| `TODO.md` | Delete TF-28, TF-29 rows |

---

## Task 1: `derive_team_in_possession` — tests + implementation

**Files:**
- Modify: `silly_kicks/tracking/_ball_carrier.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Create: `tests/tracking/test_derive_team_in_possession.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_derive_team_in_possession.py`:

```python
"""Tests for derive_team_in_possession."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ball_carrier import derive_team_in_possession


def _make_frames(n_frames: int = 3) -> pd.DataFrame:
    """Minimal tracking frames: 2 players + ball, 2 frames per period."""
    rows = []
    for fid in range(n_frames):
        for pid, tid in [("P1", "TeamA"), ("P2", "TeamB")]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "x": 50.0,
                    "y": 34.0,
                    "is_ball": False,
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "x": 52.0,
                "y": 34.0,
                "is_ball": True,
            }
        )
    return pd.DataFrame(rows)


def _make_carrier(team_ids: list) -> pd.DataFrame:
    """Carrier df matching _make_frames."""
    return pd.DataFrame(
        {
            "game_id": [1] * len(team_ids),
            "period_id": [1] * len(team_ids),
            "frame_id": list(range(len(team_ids))),
            "ball_carrier_player_id": ["P1"] * len(team_ids),
            "ball_carrier_distance_m": [1.0] * len(team_ids),
            "ball_carrier_team_id": team_ids,
        }
    )


class TestDeriveTeamInPossession:
    def test_basic_merge(self) -> None:
        frames = _make_frames(3)
        carrier = _make_carrier(["TeamA", "TeamB", "TeamA"])
        result = derive_team_in_possession(frames, carrier)
        assert "team_in_possession" in result.columns
        # All rows in frame 0 should have TeamA
        f0 = result[result["frame_id"] == 0]
        assert (f0["team_in_possession"] == "TeamA").all()
        # All rows in frame 1 should have TeamB
        f1 = result[result["frame_id"] == 1]
        assert (f1["team_in_possession"] == "TeamB").all()

    def test_unmatched_frames_get_nan(self) -> None:
        frames = _make_frames(3)
        # Carrier only has frames 0 and 1
        carrier = _make_carrier(["TeamA", "TeamB"])
        result = derive_team_in_possession(frames, carrier)
        f2 = result[result["frame_id"] == 2]
        assert f2["team_in_possession"].isna().all()

    def test_does_not_mutate_input(self) -> None:
        frames = _make_frames(2)
        carrier = _make_carrier(["TeamA", "TeamB"])
        original_cols = set(frames.columns)
        _ = derive_team_in_possession(frames, carrier)
        assert set(frames.columns) == original_cols
        assert "team_in_possession" not in frames.columns

    def test_empty_carrier(self) -> None:
        frames = _make_frames(2)
        carrier = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "ball_carrier_player_id",
                "ball_carrier_distance_m",
                "ball_carrier_team_id",
            ]
        )
        result = derive_team_in_possession(frames, carrier)
        assert "team_in_possession" in result.columns
        assert result["team_in_possession"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_derive_team_in_possession.py -v --tb=short`
Expected: ImportError — `derive_team_in_possession` does not exist.

- [ ] **Step 3: Implement `derive_team_in_possession`**

Append to `silly_kicks/tracking/_ball_carrier.py`:

```python
def derive_team_in_possession(
    frames: pd.DataFrame,
    carrier: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ball-carrier team into tracking frames as ``team_in_possession``.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    carrier : pd.DataFrame
        Output of :func:`infer_ball_carrier`: must contain ``game_id``,
        ``period_id``, ``frame_id``, ``ball_carrier_team_id``.

    Returns
    -------
    pd.DataFrame
        Copy of ``frames`` with an additional ``team_in_possession`` column.
        Frames with no carrier match get ``NaN``.

    Examples
    --------
    Typical pipeline — infer carrier, then derive possession::

        from silly_kicks.tracking import infer_ball_carrier, derive_team_in_possession

        carrier = infer_ball_carrier(frames)
        frames_with_poss = derive_team_in_possession(frames, carrier)
    """
    merge_cols = ["game_id", "period_id", "frame_id"]
    carrier_slim = carrier[merge_cols + ["ball_carrier_team_id"]].copy()
    carrier_slim = carrier_slim.rename(columns={"ball_carrier_team_id": "team_in_possession"})
    return frames.merge(carrier_slim, on=merge_cols, how="left")
```

- [ ] **Step 4: Add re-export in `tracking/__init__.py`**

Add to `__all__` list (alphabetical):
```python
"derive_team_in_possession",
```

Replace the existing import:
```python
from ._ball_carrier import infer_ball_carrier
```
with:
```python
from ._ball_carrier import derive_team_in_possession, infer_ball_carrier
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_derive_team_in_possession.py -v --tb=short`
Expected: 4 PASSED.

---

## Task 2: DAS adapter core — `_das.py` coordinate transform + wrappers

**Files:**
- Create: `silly_kicks/tracking/_das.py`
- Create: `tests/tracking/test_das.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `[das]` extra to `pyproject.toml`**

Add after the `golden-master` line in `[project.optional-dependencies]`:
```toml
das = ["accessible-space>=2.0,<3"]
```

- [ ] **Step 2: Write failing unit tests for coordinate transform + validation**

Create `tests/tracking/test_das.py`:

```python
"""Unit tests for DAS adapter."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._das import (
    _X_OFFSET,
    _Y_OFFSET,
)


class TestTodasCoords:
    def test_origin_shifts(self) -> None:
        from silly_kicks.tracking._das import _to_das_coords

        df = pd.DataFrame(
            {
                "x": [0.0, 105.0, 52.5],
                "y": [0.0, 68.0, 34.0],
                "vx": [1.0, 2.0, 0.0],
                "vy": [0.5, -0.5, 0.0],
            }
        )
        result = _to_das_coords(df)
        np.testing.assert_allclose(result["x"].values, [-_X_OFFSET, _X_OFFSET, 0.0])
        np.testing.assert_allclose(result["y"].values, [-_Y_OFFSET, _Y_OFFSET, 0.0])
        # Velocities unchanged
        np.testing.assert_allclose(result["vx"].values, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(result["vy"].values, [0.5, -0.5, 0.0])

    def test_does_not_mutate_input(self) -> None:
        from silly_kicks.tracking._das import _to_das_coords

        df = pd.DataFrame({"x": [50.0], "y": [30.0], "vx": [1.0], "vy": [0.0]})
        _ = _to_das_coords(df)
        assert df["x"].iloc[0] == 50.0
        assert df["y"].iloc[0] == 30.0


class TestInputValidation:
    def _minimal_frames(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [1],
                "period_id": [1],
                "frame_id": [0],
                "player_id": ["P1"],
                "team_id": ["A"],
                "x": [50.0],
                "y": [34.0],
                "vx": [0.0],
                "vy": [0.0],
                "is_ball": [False],
                "team_in_possession": ["A"],
            }
        )

    def test_missing_vx_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vx"])
        with pytest.raises(ValueError, match="velocity columns"):
            _validate_das_inputs(df)

    def test_missing_vy_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vy"])
        with pytest.raises(ValueError, match="velocity columns"):
            _validate_das_inputs(df)

    def test_missing_team_in_possession_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["team_in_possession"])
        with pytest.raises(ValueError, match="team_in_possession"):
            _validate_das_inputs(df)

    def test_valid_frames_pass(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames()
        _validate_das_inputs(df)  # should not raise


class TestImportGuard:
    def test_missing_package_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate accessible-space not installed."""
        import silly_kicks.tracking._das as das_mod

        monkeypatch.setattr(
            das_mod,
            "_import_accessible_space",
            lambda: (_ for _ in ()).throw(
                ImportError(
                    "accessible-space is required for DAS features. "
                    "Install with: pip install 'silly-kicks[das]'"
                )
            ),
        )
        with pytest.raises(ImportError, match="silly-kicks\\[das\\]"):
            das_mod._import_accessible_space()


class TestGetDasShapeAlignment:
    """Verify ret.acc_space / ret.das align with input frame count."""

    def test_output_length_matches_input(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_das

        rng = np.random.default_rng(42)
        n_frames = 3
        rows = []
        for fid in range(n_frames):
            for pid, tid in [("P1", "Home"), ("P2", "Away")]:
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "player_id": pid,
                        "team_id": tid,
                        "x": rng.uniform(10, 95),
                        "y": rng.uniform(5, 63),
                        "vx": rng.normal(0, 2),
                        "vy": rng.normal(0, 2),
                        "is_ball": False,
                        "team_in_possession": "Home",
                    }
                )
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": "ball",
                    "team_id": None,
                    "x": rng.uniform(20, 80),
                    "y": rng.uniform(10, 58),
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": True,
                    "team_in_possession": "Home",
                }
            )
        frames = pd.DataFrame(rows)
        result = get_das(frames, use_progress_bar=False)
        assert len(result) == len(frames), (
            f"get_das output length {len(result)} != input length {len(frames)}"
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_das.py -v --tb=short`
Expected: ImportError — `_das` module does not exist.

- [ ] **Step 4: Create `silly_kicks/tracking/_das.py` with core infrastructure**

```python
"""Dangerous Accessible Space adapter (TF-28).

Thin wrapper over the ``accessible-space`` PyPI package (MIT), mapping
silly-kicks 20-column tracking schema to the library's API.

See docs/superpowers/specs/2026-05-06-tf28-tf29-das-vaep-variants-design.md
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Coordinate transform constants: silly-kicks [0,105]×[0,68] ↔ DAS [-52.5,52.5]×[-34,34]
_X_OFFSET = 52.5
_Y_OFFSET = 34.0
_X_PITCH_MIN = -52.5
_X_PITCH_MAX = 52.5
_Y_PITCH_MIN = -34.0
_Y_PITCH_MAX = 34.0

# Column-name mapping: silly-kicks schema → accessible-space parameter names.
# Module constant — never changes at runtime.
_COLUMN_MAP = {
    "x_col": "x",
    "y_col": "y",
    "vx_col": "vx",
    "vy_col": "vy",
    "player_col": "player_id",
    "team_col": "team_id",
    "frame_col": "frame_id",
    "period_col": "period_id",
    "team_in_possession_col": "team_in_possession",
}


def _import_accessible_space():  # type: ignore[return]
    """Lazy import guard for the optional accessible-space package."""
    try:
        import accessible_space

        return accessible_space
    except ImportError as e:
        raise ImportError(
            "accessible-space is required for DAS features. "
            "Install with: pip install 'silly-kicks[das]'"
        ) from e


def _to_das_coords(frames: pd.DataFrame) -> pd.DataFrame:
    """Shift silly-kicks [0,105]x[0,68] to DAS [-52.5,52.5]x[-34,34]."""
    out = frames.copy()
    out["x"] = out["x"] - _X_OFFSET
    out["y"] = out["y"] - _Y_OFFSET
    return out


def _validate_das_inputs(frames: pd.DataFrame) -> None:
    """Validate required columns, raising with actionable messages."""
    if "vx" not in frames.columns or "vy" not in frames.columns:
        raise ValueError(
            "DAS requires velocity columns ('vx', 'vy'). "
            "Call derive_velocities() or smooth_frames() first."
        )
    if "team_in_possession" not in frames.columns:
        raise ValueError(
            "DAS requires a 'team_in_possession' column. "
            "Call derive_team_in_possession(frames, carrier_df) to add it."
        )


def _prepare_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Validate, transform coordinates, normalise ball rows."""
    _validate_das_inputs(frames)
    out = _to_das_coords(frames)
    # Map ball rows: accessible-space expects ball_player_id="ball"
    ball_mask = out["is_ball"] == True  # noqa: E712
    out.loc[ball_mask, "player_id"] = "ball"
    return out


def get_das(
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Team-level Accessible Space and Dangerous Accessible Space per frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames. Must contain ``vx``, ``vy``, and
        ``team_in_possession`` columns.
    use_progress_bar : bool, default False
        Show progress bar during simulation.
    **kwargs
        Passthrough to ``accessible_space.get_dangerous_accessible_space``
        (e.g. ``n_angles``, ``n_v0``, ``chunk_size``).

    Returns
    -------
    pd.DataFrame
        Input frames with added ``AS`` and ``DAS`` columns (float64).

    Examples
    --------
    Full pipeline from raw tracking::

        from silly_kicks.tracking import (
            derive_velocities, infer_ball_carrier, derive_team_in_possession,
        )
        from silly_kicks.tracking._das import get_das

        frames = derive_velocities(raw_frames)
        carrier = infer_ball_carrier(frames)
        frames = derive_team_in_possession(frames, carrier)
        result = get_das(frames)

    See NOTICE for full bibliographic citations.
    """
    asmod = _import_accessible_space()
    prepared = _prepare_frames(frames)

    ret = asmod.get_dangerous_accessible_space(
        prepared,
        ball_player_id="ball",
        x_pitch_min=_X_PITCH_MIN,
        x_pitch_max=_X_PITCH_MAX,
        y_pitch_min=_Y_PITCH_MIN,
        y_pitch_max=_Y_PITCH_MAX,
        infer_attacking_direction=True,
        use_progress_bar=use_progress_bar,
        **_COLUMN_MAP,
        **kwargs,
    )

    # Shape assertion: ret.acc_space must align with prepared frame count
    if len(ret.acc_space) != len(prepared):
        warnings.warn(
            f"accessible-space returned {len(ret.acc_space)} values for "
            f"{len(prepared)} input rows; output may be misaligned",
            UserWarning,
            stacklevel=2,
        )

    result = frames.copy()
    result["AS"] = ret.acc_space
    result["DAS"] = ret.das
    return result


def get_individual_das(
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Per-player Accessible Space and Dangerous Accessible Space per frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames with ``vx``, ``vy``, ``team_in_possession``.
    use_progress_bar : bool, default False
        Show progress bar.
    **kwargs
        Passthrough to ``accessible_space.get_individual_dangerous_accessible_space``.

    Returns
    -------
    pd.DataFrame
        Input frames with added ``AS`` and ``DAS`` columns (float64, per-player).

    Examples
    --------
    Per-player DAS decomposition::

        from silly_kicks.tracking._das import get_individual_das
        result = get_individual_das(frames)
        # result["DAS"] is per-player dangerous accessible space

    See NOTICE for full bibliographic citations.
    """
    asmod = _import_accessible_space()
    prepared = _prepare_frames(frames)

    ret = asmod.get_individual_dangerous_accessible_space(
        prepared,
        ball_player_id="ball",
        x_pitch_min=_X_PITCH_MIN,
        x_pitch_max=_X_PITCH_MAX,
        y_pitch_min=_Y_PITCH_MIN,
        y_pitch_max=_Y_PITCH_MAX,
        infer_attacking_direction=True,
        use_progress_bar=use_progress_bar,
        **_COLUMN_MAP,
        **kwargs,
    )

    if len(ret.player_acc_space) != len(prepared):
        warnings.warn(
            f"accessible-space returned {len(ret.player_acc_space)} values for "
            f"{len(prepared)} input rows; output may be misaligned",
            UserWarning,
            stacklevel=2,
        )

    result = frames.copy()
    result["AS"] = ret.player_acc_space
    result["DAS"] = ret.player_das
    return result


def get_xc(
    passes: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    use_progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Expected pass completion (xC) for each pass using tracking context.

    Parameters
    ----------
    passes : pd.DataFrame
        SPADL actions filtered to passes. Must contain ``start_x``, ``start_y``,
        ``end_x``, ``end_y``, ``team_id``, ``player_id``.
    frames : pd.DataFrame
        Long-form tracking frames with ``vx``, ``vy``, ``team_in_possession``.
    use_progress_bar : bool, default False
        Show progress bar.
    **kwargs
        Passthrough to ``accessible_space.get_expected_pass_completion``.

    Returns
    -------
    pd.DataFrame
        Copy of ``passes`` with added ``xC`` column (float64, probability).

    Examples
    --------
    Compute xC for all passes in a match::

        from silly_kicks.tracking._das import get_xc
        pass_actions = actions[actions["type_name"] == "pass"]
        result = get_xc(pass_actions, frames)

    See NOTICE for full bibliographic citations.
    """
    asmod = _import_accessible_space()
    _validate_das_inputs(frames)

    # Transform coordinates
    prepared_frames = _to_das_coords(frames)
    ball_mask = prepared_frames["is_ball"] == True  # noqa: E712
    prepared_frames.loc[ball_mask, "player_id"] = "ball"

    # Transform pass coordinates
    prepared_passes = passes.copy()
    prepared_passes["start_x"] = prepared_passes["start_x"] - _X_OFFSET
    prepared_passes["start_y"] = prepared_passes["start_y"] - _Y_OFFSET
    prepared_passes["end_x"] = prepared_passes["end_x"] - _X_OFFSET
    prepared_passes["end_y"] = prepared_passes["end_y"] - _Y_OFFSET

    ret = asmod.get_expected_pass_completion(
        prepared_passes,
        prepared_frames,
        event_frame_col="frame_id",
        event_player_col="player_id",
        event_team_col="team_id",
        event_start_x_col="start_x",
        event_start_y_col="start_y",
        event_end_x_col="end_x",
        event_end_y_col="end_y",
        tracking_frame_col=_COLUMN_MAP["frame_col"],
        tracking_player_col=_COLUMN_MAP["player_col"],
        tracking_team_col=_COLUMN_MAP["team_col"],
        tracking_x_col=_COLUMN_MAP["x_col"],
        tracking_y_col=_COLUMN_MAP["y_col"],
        tracking_vx_col=_COLUMN_MAP["vx_col"],
        tracking_vy_col=_COLUMN_MAP["vy_col"],
        ball_tracking_player_id="ball",
        x_pitch_min=_X_PITCH_MIN,
        x_pitch_max=_X_PITCH_MAX,
        y_pitch_min=_Y_PITCH_MIN,
        y_pitch_max=_Y_PITCH_MAX,
        tracking_period_col=_COLUMN_MAP["period_col"],
        infer_attacking_direction=True,
        use_progress_bar=use_progress_bar,
        **kwargs,
    )

    result = passes.copy()
    result["xC"] = ret.xc
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_das.py -v --tb=short`
Expected: 6 PASSED (TestTodasCoords: 2, TestInputValidation: 4). TestGetDasShapeAlignment and TestImportGuard: 2 more = 8 total.

---

## Task 3: DAS action-coupled layer + VAEP integration

**Critical design note:** DAS computation costs ~28 ms/frame. The standard `lift_to_states` pattern would call the helper 3× (once per gamestate slot), and if we had 3 separate helpers lifted individually, that's 12n total `get_das` calls (where `_das_diff` double-counts by calling both team and opponent helpers). Instead, this task uses a **single-pass precomputation** architecture:

1. Call `get_das(frames)` **exactly once** on the full frames DataFrame.
2. Build a `(period_id, frame_id) → {team_id: DAS}` lookup dict.
3. A single custom `FrameAwareTransformer` maps actions to the lookup per gamestate slot.

Total: **1 `get_das` call + 3 `link_actions_to_frames` calls** instead of 12n.

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `silly_kicks/tracking/__init__.py`

- [ ] **Step 1: Write failing tests for `das_at_action` and `das_xfns`**

Append to `tests/tracking/test_das.py`:

```python
class TestDasXfns:
    def test_das_xfns_are_frame_aware(self) -> None:
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        for xfn in das_xfns:
            assert is_frame_aware(xfn), f"{xfn.__name__} is not frame_aware"

    def test_das_xfns_feature_column_names(self) -> None:
        """feature_column_names introspection with empty frames must not crash."""
        from silly_kicks.vaep.features import feature_column_names
        from silly_kicks.tracking.features import das_xfns

        cols = feature_column_names(das_xfns, nb_prev_actions=3)
        # 3 features × 3 gamestate slots = 9 columns
        expected_cols = {
            "das_team_a0", "das_team_a1", "das_team_a2",
            "das_opponent_a0", "das_opponent_a1", "das_opponent_a2",
            "das_diff_a0", "das_diff_a1", "das_diff_a2",
        }
        assert expected_cols == set(cols)

    def test_das_xfns_length(self) -> None:
        from silly_kicks.tracking.features import das_xfns

        # Single custom transformer produces all 9 columns
        assert len(das_xfns) == 1

    def test_das_at_action_introspection(self) -> None:
        """frames=None introspection must return NaN Series with correct name."""
        from silly_kicks.tracking.features import das_at_action

        dummy = pd.DataFrame({"action_id": [1, 2], "team_id": [1, 1]})
        result = das_at_action(dummy, None)
        assert result.name == "das_team"
        assert result.isna().all()
        assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_das.py::TestDasXfns -v --tb=short`
Expected: ImportError — `das_xfns` not in `features.py`.

- [ ] **Step 3: Implement `_precompute_das_lookup`, `_map_das_to_actions`, `das_at_action`, `add_das`, `das_xfns`**

Add at the bottom of `silly_kicks/tracking/features.py`:

```python
# ---------------------------------------------------------------------------
# DAS — Dangerous Accessible Space (TF-28)
#
# Architecture: single-pass precomputation. get_das() runs ONCE on the full
# frames DataFrame; a (period_id, frame_id) → {team_id: DAS} lookup dict is
# built from the result; action-coupled helpers and the VAEP transformer map
# into this lookup. This avoids the 12n redundant get_das calls that would
# result from 3 separate lift_to_states helpers × 3 gamestate slots.
# ---------------------------------------------------------------------------

import warnings as _warnings


def _precompute_das_lookup(
    frames: pd.DataFrame,
) -> dict[tuple, dict]:
    """Run get_das ONCE on all frames, build per-frame team-level DAS lookup.

    Returns a dict mapping ``(period_id, frame_id)`` to ``{team_id: DAS_value}``.
    """
    from ._das import get_das

    das_frames = get_das(frames, use_progress_bar=False)

    player_rows = das_frames[das_frames["is_ball"] != True]  # noqa: E712
    lookup: dict[tuple, dict] = {}
    for (pid, fid, tid), grp in player_rows.groupby(
        ["period_id", "frame_id", "team_id"]
    ):
        # Team-level DAS is identical for all players of the same team in the
        # same frame, so any row suffices.
        lookup.setdefault((pid, fid), {})[tid] = float(grp["DAS"].iloc[0])
    return lookup


def _map_das_to_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    das_lookup: dict[tuple, dict],
) -> pd.DataFrame:
    """Map precomputed DAS lookup to actions. Returns 3-column DataFrame."""
    import numpy as np

    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")

    team_vals = np.full(len(actions), np.nan)
    opp_vals = np.full(len(actions), np.nan)

    for i, (_idx, row) in enumerate(actions.iterrows()):
        aid = row["action_id"]
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        key = (row["period_id"], int(float(fid_raw)))
        if key not in das_lookup:
            continue

        team_id = row["team_id"]
        team_vals[i] = das_lookup[key].get(team_id, np.nan)
        # Football: exactly 2 teams per frame; take the sole opponent.
        opp = [v for k, v in das_lookup[key].items() if k != team_id]
        if opp:
            opp_vals[i] = opp[0]

    return pd.DataFrame(
        {
            "das_team": team_vals,
            "das_opponent": opp_vals,
            "das_diff": team_vals - opp_vals,
        },
        index=actions.index,
    )


def das_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    col_name: str = "das_team",
) -> pd.Series:
    """Team-level DAS at the linked frame for the acting team.

    Returns a Series with one value per action. NaN where action couldn't
    link to a frame or DAS computation failed.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import das_at_action
    >>> das = das_at_action(actions, frames)
    """
    import numpy as np

    # Introspection mode: VAEP fit-time calls with frames=None
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    try:
        lookup = _precompute_das_lookup(frames)
    except (ValueError, RuntimeError, ImportError) as exc:
        _warnings.warn(
            f"DAS computation failed ({type(exc).__name__}: {exc}); "
            f"returning NaN for all actions",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series(np.nan, index=actions.index, name=col_name)

    mapped = _map_das_to_actions(actions, frames, lookup)
    s = mapped["das_team"]
    s.name = col_name
    return s


@nan_safe_enrichment
def add_das(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich actions with ``das_team``, ``das_opponent``, ``das_diff`` columns.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_das
    >>> enriched = add_das(actions, frames)
    """
    import numpy as np

    out = actions.copy()

    try:
        lookup = _precompute_das_lookup(frames)
    except (ValueError, RuntimeError, ImportError) as exc:
        _warnings.warn(
            f"DAS computation failed ({type(exc).__name__}: {exc}); "
            f"returning NaN for all DAS columns",
            UserWarning,
            stacklevel=2,
        )
        out["das_team"] = np.nan
        out["das_opponent"] = np.nan
        out["das_diff"] = np.nan
        return out

    mapped = _map_das_to_actions(actions, frames, lookup)
    out["das_team"] = mapped["das_team"].values
    out["das_opponent"] = mapped["das_opponent"].values
    out["das_diff"] = mapped["das_diff"].values
    return out


def _make_das_transformer():
    """Build a single FrameAwareTransformer that emits all 9 DAS columns.

    Single-pass: calls get_das() ONCE on the full frames DataFrame, then
    looks up per-action across all 3 gamestate slots. Returns columns:
    das_team_a0..a2, das_opponent_a0..a2, das_diff_a0..a2.
    """
    import numpy as np

    _DAS_COLS = ("das_team", "das_opponent", "das_diff")

    def das_features(states, frames):
        nb = min(len(states), 3)
        out = pd.DataFrame(index=states[0].index)

        # Empty frames → column-name probing (feature_column_names)
        if len(frames) == 0:
            for i in range(nb):
                for col in _DAS_COLS:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Precompute DAS for ALL frames — single get_das call
        try:
            lookup = _precompute_das_lookup(frames)
        except (ValueError, RuntimeError, ImportError) as exc:
            _warnings.warn(
                f"DAS computation failed ({type(exc).__name__}: {exc}); "
                f"returning NaN for all DAS features",
                UserWarning,
                stacklevel=2,
            )
            for i in range(nb):
                for col in _DAS_COLS:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Map per gamestate slot
        for i, slot in enumerate(states[:nb]):
            mapped = _map_das_to_actions(slot, frames, lookup)
            for col in _DAS_COLS:
                out[f"{col}_a{i}"] = mapped[col].to_numpy()
        return out

    das_features._frame_aware = True  # type: ignore[attr-defined]
    das_features.__name__ = "das_features"
    das_features.__qualname__ = "das_features"
    return das_features


das_xfns = [_make_das_transformer()]
```

Also add to `__all__` in `features.py`:
```python
"add_das",
"das_at_action",
"das_xfns",
```

- [ ] **Step 4: Update `tracking/__init__.py` re-exports**

Add to `__all__`:
```python
"add_das",
"das_at_action",
"das_xfns",
"derive_team_in_possession",
"get_das",
"get_individual_das",
"get_xc",
```

Add imports:
```python
from ._das import get_das, get_individual_das, get_xc
```
Merge `add_das`, `das_at_action`, `das_xfns` into the existing `from .features import (...)` block.

Replace:
```python
from ._ball_carrier import infer_ball_carrier
```
with:
```python
from ._ball_carrier import derive_team_in_possession, infer_ball_carrier
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_das.py -v --tb=short`
Expected: 12 PASSED (Task 2: 8 + Task 3: 4).

---

## Task 4: VAEP windowing — `scores()` and `concedes()` parameter extension

**Files:**
- Create: `tests/vaep/test_labels_windowing.py`
- Modify: `silly_kicks/vaep/labels.py`

- [ ] **Step 1: Write failing unit tests for windowing**

Create `tests/vaep/test_labels_windowing.py`:

```python
"""Tests for VAEP label windowing variants (TF-29)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadl
import silly_kicks.vaep.labels as lab


def _make_actions(
    n: int = 10,
    *,
    goal_at: int | None = None,
    owngoal_at: int | None = None,
    possession_ids: list[int] | None = None,
    time_seconds: list[float] | None = None,
    period_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal SPADL-shaped actions DataFrame for windowing tests."""
    types = ["pass"] * n
    results = ["success"] * n
    if goal_at is not None:
        types[goal_at] = "shot"
        results[goal_at] = "success"
    if owngoal_at is not None:
        types[owngoal_at] = "shot"
        results[owngoal_at] = "owngoal"

    df = pd.DataFrame(
        {
            "game_id": [1] * n,
            "period_id": period_ids if period_ids else [1] * n,
            "action_id": list(range(n)),
            "type_id": [spadl.actiontype_id.get(t, 0) for t in types],
            "type_name": types,
            "result_id": [spadl.result_id.get(r, 0) for r in results],
            "result_name": results,
            "team_id": [1 if i % 2 == 0 else 2 for i in range(n)],
            "player_id": list(range(n)),
            "start_x": [50.0] * n,
            "start_y": [34.0] * n,
            "end_x": [60.0] * n,
            "end_y": [34.0] * n,
            "bodypart_id": [0] * n,
            "bodypart_name": ["foot"] * n,
            "time_seconds": time_seconds if time_seconds else [float(i * 3) for i in range(n)],
        }
    )
    if possession_ids is not None:
        df["possession_id"] = possession_ids
    return df


class TestWindowActionBackwardCompat:
    """window='action' must produce identical output to the original API."""

    def test_scores_action_default(self) -> None:
        actions = _make_actions(10, goal_at=5)
        old = lab.scores(actions, nr_actions=10)
        new = lab.scores(actions, nr_actions=10, window="action")
        pd.testing.assert_frame_equal(old, new)

    def test_concedes_action_default(self) -> None:
        actions = _make_actions(10, goal_at=5)
        old = lab.concedes(actions, nr_actions=10)
        new = lab.concedes(actions, nr_actions=10, window="action")
        pd.testing.assert_frame_equal(old, new)


class TestWindowPossession:
    def test_missing_possession_id_raises(self) -> None:
        actions = _make_actions(5)
        assert "possession_id" not in actions.columns
        with pytest.raises(ValueError, match="possession_id"):
            lab.scores(actions, window="possession")

    def test_scores_within_possession(self) -> None:
        # 3 possession chains: [0,1,2], [3,4,5], [6,7,8]
        # Goal at action 2 (chain 0), team_id=1
        # All even actions are team 1, odd are team 2
        actions = _make_actions(
            9,
            goal_at=2,
            possession_ids=[0, 0, 0, 1, 1, 1, 2, 2, 2],
        )
        scores = lab.scores(actions, window="possession")
        # Action 2 (team 1, goal) itself scores
        assert scores["scores"].iloc[2]
        # Action 0 (team 1) sees goal at action 2 (team 1, same team) → scores
        assert scores["scores"].iloc[0]
        # Action 1 (team 2) sees goal at action 2 (team 1, different team) → does NOT score
        assert not scores["scores"].iloc[1]
        # Actions in chains 1 and 2 should not score (no goal in their chain)
        assert not scores["scores"].iloc[3:].any()

    def test_concedes_within_possession(self) -> None:
        # Goal at action 2 (team 1); action 1 (team 2) should concede
        actions = _make_actions(
            6,
            goal_at=2,
            possession_ids=[0, 0, 0, 1, 1, 1],
        )
        concedes = lab.concedes(actions, window="possession")
        # Action 1 (team 2) sees goal at action 2 (team 1, different team) → concedes
        assert concedes["concedes"].iloc[1]
        # Action 0 (team 1) sees goal at action 2 (team 1, same team) → does NOT concede
        assert not concedes["concedes"].iloc[0]
        # Actions in chain 1: no goal → no concedes
        assert not concedes["concedes"].iloc[3:].any()


class TestWindowTime:
    def test_missing_time_seconds_raises(self) -> None:
        actions = _make_actions(5)
        actions = actions.drop(columns=["time_seconds"])
        with pytest.raises(ValueError, match="time_seconds"):
            lab.scores(actions, window="time")

    def test_strict_boundary(self) -> None:
        """goal_time - action_time < window_seconds (strict inequality)."""
        # Goal at action 2, t=10.0. window_seconds=5.0
        # Action at t=5.0: 10-5=5.0, NOT < 5.0, should NOT score
        # Action at t=5.01: 10-5.01=4.99, < 5.0, should score
        actions = _make_actions(
            4,
            goal_at=2,
            time_seconds=[0.0, 5.0, 10.0, 15.0],
        )
        # All same team for simplicity
        actions["team_id"] = 1
        scores = lab.scores(actions, window="time", window_seconds=5.0)
        assert not scores["scores"].iloc[0]  # t=0, 10-0=10 >= 5
        assert not scores["scores"].iloc[1]  # t=5, 10-5=5.0, NOT < 5.0
        assert scores["scores"].iloc[2]  # t=10, goal action itself
        assert not scores["scores"].iloc[3]  # t=15, after goal

    def test_cross_period_no_bleed(self) -> None:
        """Goal in period 2 must not bleed into period 1."""
        actions = _make_actions(
            4,
            goal_at=3,
            time_seconds=[80.0, 89.0, 1.0, 5.0],
            period_ids=[1, 1, 2, 2],
        )
        actions["team_id"] = 1
        scores = lab.scores(actions, window="time", window_seconds=15.0)
        assert not scores["scores"].iloc[0]  # period 1, no goal in period 1
        assert not scores["scores"].iloc[1]  # period 1
        assert scores["scores"].iloc[2]  # period 2, within window of goal at t=5
        assert scores["scores"].iloc[3]  # the goal itself

    def test_unsorted_raises(self) -> None:
        """time_seconds must be non-decreasing within each period."""
        actions = _make_actions(
            3,
            time_seconds=[10.0, 5.0, 15.0],
        )
        with pytest.raises(ValueError, match="non-decreasing"):
            lab.scores(actions, window="time")


class TestNrActionsWarning:
    def test_warns_when_non_default_with_possession(self) -> None:
        actions = _make_actions(5, possession_ids=[0, 0, 0, 1, 1])
        with pytest.warns(UserWarning, match="nr_actions.*ignored"):
            lab.scores(actions, nr_actions=5, window="possession")

    def test_warns_when_non_default_with_time(self) -> None:
        actions = _make_actions(5)
        with pytest.warns(UserWarning, match="nr_actions.*ignored"):
            lab.scores(actions, nr_actions=5, window="time")

    def test_no_warning_when_default_nr_actions(self) -> None:
        actions = _make_actions(5, possession_ids=[0, 0, 0, 1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lab.scores(actions, nr_actions=10, window="possession")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/vaep/test_labels_windowing.py -v --tb=short`
Expected: FAIL — `scores()` does not accept `window` kwarg.

- [ ] **Step 3: Implement windowing in `vaep/labels.py`**

Add imports at the top of `silly_kicks/vaep/labels.py` (after existing imports):

```python
import warnings
from typing import Literal

import numpy as np
```

Rename the existing `scores` function body to `_scores_action` and `concedes` body to `_concedes_action` (pure internal rename, identical logic).

Replace `scores` with:

```python
def scores(
    actions: pd.DataFrame,
    nr_actions: int = 10,
    xg_column: str | None = None,
    *,
    window: Literal["action", "possession", "time"] = "action",
    window_seconds: float = 15.0,
) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next window.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10
        Number of actions after the current action to consider.
        Only used when ``window="action"``.
    xg_column : str or None, default=None
        If provided, return xG-weighted scoring probability instead of boolean labels.
    window : {"action", "possession", "time"}, default="action"
        Lookahead window mode. ``"action"`` uses ``nr_actions`` (original VAEP).
        ``"possession"`` looks within the same ``possession_id`` chain (requires
        column; call ``add_possessions()`` first — use default params for DTAI-naive,
        or ``merge_brief_opposing_actions=2, brief_window_seconds=2.0,
        defensive_transition_types=("interception", "clearance")`` for DTAI-extended).
        ``"time"`` looks within ``window_seconds`` of the current action's
        ``time_seconds``, bounded by ``period_id``.
    window_seconds : float, default=15.0
        Time window in seconds. Only used when ``window="time"``.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores'.

    Examples
    --------
    Compute "scores" labels for VAEP training::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import scores

        actions_with_names = add_names(actions)
        y_scores = scores(actions_with_names, nr_actions=10)

    Possession-based windowing::

        from silly_kicks.spadl import add_possessions
        actions = add_possessions(actions)
        y_scores = scores(actions, window="possession")

    Time-based windowing (15-second lookahead)::

        y_scores = scores(actions, window="time", window_seconds=15.0)
    """
    _warn_if_nr_actions_ignored(nr_actions, window)

    if window == "action":
        if xg_column is not None:
            return _scores_xg(actions, nr_actions, xg_column)
        return _scores_action(actions, nr_actions)
    elif window == "possession":
        _require_column(actions, "possession_id", window)
        return _scores_possession(actions, xg_column)
    elif window == "time":
        _require_column(actions, "time_seconds", window)
        return _scores_time(actions, window_seconds, xg_column)
    else:
        raise ValueError(f"Unknown window mode: {window!r}")
```

Replace `concedes` with (same pattern):

```python
def concedes(
    actions: pd.DataFrame,
    nr_actions: int = 10,
    xg_column: str | None = None,
    *,
    window: Literal["action", "possession", "time"] = "action",
    window_seconds: float = 15.0,
) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next window.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10
        Number of actions after the current action to consider.
        Only used when ``window="action"``.
    xg_column : str or None, default=None
        If provided, return xG-weighted conceding probability.
    window : {"action", "possession", "time"}, default="action"
        Lookahead window mode. See :func:`scores` for details.
    window_seconds : float, default=15.0
        Time window in seconds. Only used when ``window="time"``.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes'.

    Examples
    --------
    Compute "concedes" labels (the dual of ``scores``) for VAEP training::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import concedes

        actions_with_names = add_names(actions)
        y_concedes = concedes(actions_with_names, nr_actions=10)
    """
    _warn_if_nr_actions_ignored(nr_actions, window)

    if window == "action":
        if xg_column is not None:
            return _concedes_xg(actions, nr_actions, xg_column)
        return _concedes_action(actions, nr_actions)
    elif window == "possession":
        _require_column(actions, "possession_id", window)
        return _concedes_possession(actions, xg_column)
    elif window == "time":
        _require_column(actions, "time_seconds", window)
        return _concedes_time(actions, window_seconds, xg_column)
    else:
        raise ValueError(f"Unknown window mode: {window!r}")
```

Add shared helpers (after imports, before `scores`):

```python
def _warn_if_nr_actions_ignored(nr_actions: int, window: str) -> None:
    if nr_actions != 10 and window != "action":
        warnings.warn(
            f"nr_actions={nr_actions} is ignored when window={window!r}; "
            f"only window='action' uses nr_actions",
            UserWarning,
            stacklevel=3,
        )


def _require_column(actions: pd.DataFrame, col: str, window: str) -> None:
    if col not in actions.columns:
        raise ValueError(
            f"window={window!r} requires a '{col}' column. "
            + (
                "Call add_possessions() first."
                if col == "possession_id"
                else f"Ensure '{col}' is present in the actions DataFrame."
            )
        )
```

Rename existing `scores` body to `_scores_action` (same logic, unchanged):

```python
def _scores_action(actions: pd.DataFrame, nr_actions: int) -> pd.DataFrame:
    """Original VAEP action-count windowed scoring labels."""
    goal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
    owngoal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["owngoal"])
    team_id = actions["team_id"]

    result = goal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        result = result | (shifted_goal & same_team) | (shifted_owngoal & ~same_team)

    return pd.DataFrame(result, columns=["scores"])
```

**Note:** `_scores_xg` and `_concedes_xg` already exist in `labels.py` — keep them unchanged. They handle the `xg_column is not None` branch for `window="action"`.

Same for `_concedes_action`:

```python
def _concedes_action(actions: pd.DataFrame, nr_actions: int) -> pd.DataFrame:
    """Original VAEP action-count windowed conceding labels."""
    goal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
    owngoal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["owngoal"])
    team_id = actions["team_id"]

    result = owngoal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        result = result | (shifted_goal & ~same_team) | (shifted_owngoal & same_team)

    return pd.DataFrame(result, columns=["concedes"])
```

Add possession-based implementations. **Note:** O(n²) within each possession chain — acceptable because possession chains are typically 5–15 actions long:

```python
def _scores_possession(actions: pd.DataFrame, xg_column: str | None) -> pd.DataFrame:
    goal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
    owngoal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["owngoal"])
    team_id = actions["team_id"]

    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]

    group_cols = ["game_id", "possession_id"] if "game_id" in actions.columns else ["possession_id"]
    result = pd.Series(False, index=actions.index) if xg_column is None else pd.Series(0.0, index=actions.index)

    for _key, grp in actions.groupby(group_cols):
        idx = grp.index
        for i, pos in enumerate(idx):
            for j_pos in idx[i + 1 :]:
                if goal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
                if owngoal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if not same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
        # The goal action itself scores
        for pos in idx:
            if goal.loc[pos]:
                if xg_column is not None:
                    result.loc[pos] = max(result.loc[pos], xg.loc[pos])
                else:
                    result.loc[pos] = True

    return pd.DataFrame(result, columns=["scores"])


def _concedes_possession(actions: pd.DataFrame, xg_column: str | None) -> pd.DataFrame:
    goal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
    owngoal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["owngoal"])
    team_id = actions["team_id"]

    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]

    group_cols = ["game_id", "possession_id"] if "game_id" in actions.columns else ["possession_id"]
    result = pd.Series(False, index=actions.index) if xg_column is None else pd.Series(0.0, index=actions.index)

    for _key, grp in actions.groupby(group_cols):
        idx = grp.index
        for i, pos in enumerate(idx):
            for j_pos in idx[i + 1 :]:
                if goal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if not same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
                if owngoal.loc[j_pos]:
                    same_team = team_id.loc[pos] == team_id.loc[j_pos]
                    if same_team:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
        for pos in idx:
            if owngoal.loc[pos]:
                if xg_column is not None:
                    result.loc[pos] = max(result.loc[pos], xg.loc[pos])
                else:
                    result.loc[pos] = True

    return pd.DataFrame(result, columns=["concedes"])
```

Add time-based implementations. Uses `np.searchsorted` with `side="left"` for strict inequality (`t_goal - t_action < window_seconds`):

```python
def _scores_time(actions: pd.DataFrame, window_seconds: float, xg_column: str | None) -> pd.DataFrame:
    goal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
    owngoal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["owngoal"])
    team_id = actions["team_id"].values
    time_s = actions["time_seconds"].values

    period_col = "period_id" if "period_id" in actions.columns else None
    game_col = "game_id" if "game_id" in actions.columns else None

    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).values  # type: ignore[reportOptionalMemberAccess]

    result = np.zeros(len(actions), dtype=float) if xg_column else np.zeros(len(actions), dtype=bool)

    # Group by (game_id, period_id) for boundary isolation
    group_keys = []
    if game_col:
        group_keys.append(game_col)
    if period_col:
        group_keys.append(period_col)

    if group_keys:
        groups = actions.groupby(group_keys)
    else:
        groups = [(None, actions)]

    for _key, grp in groups:
        idx = grp.index
        t = time_s[idx]
        g = goal.values[idx]
        og = owngoal.values[idx]
        tid = team_id[idx]

        # Precondition: time_seconds must be non-decreasing within each period
        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError(
                "time_seconds must be non-decreasing within each (game_id, period_id) group. "
                "Sort actions by (game_id, period_id, time_seconds) before calling."
            )

        # searchsorted with side="left": boundary = first index where t >= t[i] + window_seconds
        # This gives strict inequality: only actions j where t[j] < t[i] + window_seconds
        boundaries = np.searchsorted(t, t + window_seconds, side="left")

        for local_i in range(len(idx)):
            global_i = idx[local_i]
            end = boundaries[local_i]
            for local_j in range(local_i + 1, min(end, len(idx))):
                if g[local_j]:
                    if tid[local_i] == tid[local_j]:
                        if xg_column:
                            result[global_i] = max(result[global_i], xg[idx[local_j]])
                        else:
                            result[global_i] = True
                            break
                if og[local_j]:
                    if tid[local_i] != tid[local_j]:
                        if xg_column:
                            result[global_i] = max(result[global_i], xg[idx[local_j]])
                        else:
                            result[global_i] = True
                            break

            # The goal action itself
            if g[local_i]:
                if xg_column:
                    result[global_i] = max(result[global_i], xg[idx[local_i]])
                else:
                    result[global_i] = True

    return pd.DataFrame({"scores": result})


def _concedes_time(actions: pd.DataFrame, window_seconds: float, xg_column: str | None) -> pd.DataFrame:
    goal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["success"])
    owngoal = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadl.result_id["owngoal"])
    team_id = actions["team_id"].values
    time_s = actions["time_seconds"].values

    period_col = "period_id" if "period_id" in actions.columns else None
    game_col = "game_id" if "game_id" in actions.columns else None

    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).values  # type: ignore[reportOptionalMemberAccess]

    result = np.zeros(len(actions), dtype=float) if xg_column else np.zeros(len(actions), dtype=bool)

    group_keys = []
    if game_col:
        group_keys.append(game_col)
    if period_col:
        group_keys.append(period_col)

    if group_keys:
        groups = actions.groupby(group_keys)
    else:
        groups = [(None, actions)]

    for _key, grp in groups:
        idx = grp.index
        t = time_s[idx]
        g = goal.values[idx]
        og = owngoal.values[idx]
        tid = team_id[idx]

        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError(
                "time_seconds must be non-decreasing within each (game_id, period_id) group. "
                "Sort actions by (game_id, period_id, time_seconds) before calling."
            )

        boundaries = np.searchsorted(t, t + window_seconds, side="left")

        for local_i in range(len(idx)):
            global_i = idx[local_i]
            end = boundaries[local_i]
            for local_j in range(local_i + 1, min(end, len(idx))):
                if g[local_j]:
                    if tid[local_i] != tid[local_j]:
                        if xg_column:
                            result[global_i] = max(result[global_i], xg[idx[local_j]])
                        else:
                            result[global_i] = True
                            break
                if og[local_j]:
                    if tid[local_i] == tid[local_j]:
                        if xg_column:
                            result[global_i] = max(result[global_i], xg[idx[local_j]])
                        else:
                            result[global_i] = True
                            break

            if og[local_i]:
                if xg_column:
                    result[global_i] = max(result[global_i], xg[idx[local_i]])
                else:
                    result[global_i] = True

    return pd.DataFrame({"concedes": result})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/vaep/test_labels_windowing.py -v --tb=short`
Expected: All PASSED (12 tests: TestWindowActionBackwardCompat 2, TestWindowPossession 3, TestWindowTime 4, TestNrActionsWarning 3).

- [ ] **Step 5: Run existing label tests for regression**

Run: `python -m pytest tests/vaep/test_labels.py -v --tb=short`
Expected: All existing tests still PASS.

---

## Task 5: Goalscore-free xfn lists

**Files:**
- Modify: `silly_kicks/vaep/base.py`
- Modify: `silly_kicks/vaep/hybrid.py`
- Modify: `silly_kicks/vaep/__init__.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/vaep/test_labels_windowing.py`:

```python
class TestGoalscoreFreeXfns:
    def test_xfns_default_no_goalscore_length(self) -> None:
        from silly_kicks.vaep.base import xfns_default, xfns_default_no_goalscore

        assert len(xfns_default_no_goalscore) == len(xfns_default) - 1

    def test_xfns_default_no_goalscore_excludes(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.base import xfns_default_no_goalscore

        assert fs.goalscore not in xfns_default_no_goalscore

    def test_xfns_default_no_goalscore_order(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.base import xfns_default, xfns_default_no_goalscore

        expected = [x for x in xfns_default if x is not fs.goalscore]
        assert xfns_default_no_goalscore == expected

    def test_hybrid_no_goalscore_length(self) -> None:
        from silly_kicks.vaep.hybrid import hybrid_xfns_default, hybrid_xfns_default_no_goalscore

        assert len(hybrid_xfns_default_no_goalscore) == len(hybrid_xfns_default) - 1

    def test_hybrid_no_goalscore_excludes(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.hybrid import hybrid_xfns_default_no_goalscore

        assert fs.goalscore not in hybrid_xfns_default_no_goalscore

    def test_hybrid_no_goalscore_order(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.hybrid import hybrid_xfns_default, hybrid_xfns_default_no_goalscore

        expected = [x for x in hybrid_xfns_default if x is not fs.goalscore]
        assert hybrid_xfns_default_no_goalscore == expected

    def test_feature_column_names_no_goalscore(self) -> None:
        from silly_kicks.vaep.base import xfns_default, xfns_default_no_goalscore
        from silly_kicks.vaep.features import feature_column_names

        cols_full = feature_column_names(list(xfns_default), 3)
        cols_no_gs = feature_column_names(list(xfns_default_no_goalscore), 3)
        assert len(cols_no_gs) < len(cols_full)
        assert not any("goalscore" in c for c in cols_no_gs)

    def test_reexport_from_vaep(self) -> None:
        from silly_kicks.vaep import (  # noqa: F401
            hybrid_xfns_default_no_goalscore,
            xfns_default_no_goalscore,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/vaep/test_labels_windowing.py::TestGoalscoreFreeXfns -v --tb=short`
Expected: ImportError.

- [ ] **Step 3: Add lists to `base.py` and `hybrid.py`**

In `silly_kicks/vaep/base.py`, after the `xfns_default` list:
```python
xfns_default_no_goalscore = [x for x in xfns_default if x is not fs.goalscore]
```

In `silly_kicks/vaep/hybrid.py`, after the `hybrid_xfns_default` list:
```python
hybrid_xfns_default_no_goalscore = [x for x in hybrid_xfns_default if x is not fs.goalscore]
```

- [ ] **Step 4: Update `vaep/__init__.py` re-exports**

Show the complete final import block (keep the existing submodule re-exports):
```python
from . import features, formula, labels          # ← keep existing
from .base import VAEP, xfns_default_no_goalscore
from .hybrid import HybridVAEP, hybrid_xfns_default_no_goalscore

__all__ = [
    "VAEP",
    "HybridVAEP",
    "features",
    "formula",
    "labels",
    "xfns_default_no_goalscore",
    "hybrid_xfns_default_no_goalscore",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/vaep/test_labels_windowing.py::TestGoalscoreFreeXfns -v --tb=short`
Expected: 8 PASSED.

---

## Task 6: DAS invariant tests

**Files:**
- Create: `tests/invariants/test_das_invariants.py`

- [ ] **Step 1: Write invariant tests**

Create `tests/invariants/test_das_invariants.py`:

```python
"""Physical invariants for DAS adapter (TF-28)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("accessible_space")

from silly_kicks.tracking._das import get_das, get_individual_das


def _synthetic_frames(n_frames: int = 5) -> pd.DataFrame:
    """Minimal synthetic tracking with 4 players + ball, with team_in_possession."""
    rng = np.random.default_rng(42)
    rows = []
    for fid in range(n_frames):
        for pid, tid in [("P1", "Home"), ("P2", "Home"), ("P3", "Away"), ("P4", "Away")]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "x": rng.uniform(0, 105),
                    "y": rng.uniform(0, 68),
                    "vx": rng.normal(0, 2),
                    "vy": rng.normal(0, 2),
                    "is_ball": False,
                    "team_in_possession": "Home",
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "x": rng.uniform(20, 80),
                "y": rng.uniform(10, 58),
                "vx": rng.normal(0, 3),
                "vy": rng.normal(0, 3),
                "is_ball": True,
                "team_in_possession": "Home",
            }
        )
    return pd.DataFrame(rows)


class TestDasInvariants:
    @pytest.fixture
    def das_result(self) -> pd.DataFrame:
        return get_das(_synthetic_frames(5), use_progress_bar=False)

    @pytest.fixture
    def individual_result(self) -> pd.DataFrame:
        return get_individual_das(_synthetic_frames(5), use_progress_bar=False)

    def test_as_non_negative(self, das_result: pd.DataFrame) -> None:
        valid = das_result["AS"].dropna()
        assert (valid >= 0).all(), f"Negative AS found: {valid[valid < 0].values}"

    def test_das_non_negative(self, das_result: pd.DataFrame) -> None:
        valid = das_result["DAS"].dropna()
        assert (valid >= 0).all(), f"Negative DAS found: {valid[valid < 0].values}"

    def test_as_geq_das(self, das_result: pd.DataFrame) -> None:
        valid = das_result[["AS", "DAS"]].dropna()
        assert (valid["AS"] >= valid["DAS"] - 1e-9).all(), "AS < DAS found"

    def test_individual_as_non_negative(self, individual_result: pd.DataFrame) -> None:
        valid = individual_result["AS"].dropna()
        assert (valid >= 0).all()

    def test_individual_das_non_negative(self, individual_result: pd.DataFrame) -> None:
        valid = individual_result["DAS"].dropna()
        assert (valid >= 0).all()

    def test_output_length_matches_input(self, das_result: pd.DataFrame) -> None:
        expected_len = 5 * 5  # 5 frames × (4 players + 1 ball)
        assert len(das_result) == expected_len


class TestStationary:
    def test_stationary_players_valid(self) -> None:
        frames = _synthetic_frames(3)
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        result = get_das(frames, use_progress_bar=False)
        valid = result["DAS"].dropna()
        assert len(valid) > 0, "All-stationary frames produced all-NaN DAS"
```

- [ ] **Step 2: Run invariant tests**

Run: `python -m pytest tests/invariants/test_das_invariants.py -v --tb=short`
Expected: All PASSED (7 tests).

---

## Task 7: NOTICE, CHANGELOG, TODO updates

**Files:**
- Modify: `NOTICE`
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`

- [ ] **Step 1: Add references to NOTICE**

Append to the "Mathematical / Methodological References" section of `NOTICE`:

```
The DAS adapter (silly_kicks/tracking/_das.py) wraps the accessible-space
package implementing: Bischofberger, J., & Baca, A. (2026). "Dangerous
accessible space: a unified model of space and value in team sports."
Journal of Big Data, 13, 76. Package: accessible-space on PyPI (MIT).

The VAEP windowing variants (silly_kicks/vaep/labels.py) implement design
choices from the DTAI Sports blog series: Cascioli, Robberechts, Van Tente
& Davis (2024-2025). "Three Key Design Decisions for Possession State Value
Models: An Experimental Analysis." Parts 1-3. KU Leuven / DTAI Sports.
```

- [ ] **Step 2: Add CHANGELOG 3.8.0 entry**

Prepend to `CHANGELOG.md` (before the `[3.7.0]` entry):

```markdown
## [3.8.0] — 2026-05-06

### Added

#### TF-28: DAS adapter — Dangerous Accessible Space

- `silly_kicks.tracking._das` module — thin adapter over `accessible-space` PyPI package (MIT)
- `get_das(frames)` → team-level AS/DAS per frame
- `get_individual_das(frames)` → per-player AS/DAS per frame
- `get_xc(passes, frames)` → expected pass completion per pass
- `derive_team_in_possession(frames, carrier)` → general tracking helper (in `_ball_carrier.py`)
- `das_at_action(actions, frames)` → action-coupled DAS
- `add_das(actions, frames)` → enrichment aggregator (`das_team`, `das_opponent`, `das_diff`)
- `das_xfns` — VAEP-compatible xfn list (single-pass precomputation, 9 columns)
- `[das]` optional extra in pyproject.toml (`accessible-space>=2.0,<3`)

#### TF-29: VAEP design-space variants — windowing + goalscore bias control

- `window` parameter on `scores()` / `concedes()`: `"action"` (default), `"possession"`, `"time"`
- `window_seconds` parameter for time-based windowing (default 15.0s)
- `xfns_default_no_goalscore` in `vaep/base.py`
- `hybrid_xfns_default_no_goalscore` in `vaep/hybrid.py`

#### Academic references (NOTICE)

- Bischofberger & Baca 2026 (Dangerous Accessible Space)
- Cascioli, Robberechts, Van Tente & Davis 2024-2025 (DTAI VAEP design-space blog series)
```

- [ ] **Step 3: Update TODO.md**

Delete the TF-28 and TF-29 rows from the Tier 2 table. Update the header date and current release:
```
**Last updated**: 2026-05-06. **Current release**: silly-kicks 3.8.0.
```

- [ ] **Step 4: Update version in pyproject.toml**

Change `version = "3.7.0"` to `version = "3.8.0"`.

---

## Task 8: Lint + type-check + full test suite

- [ ] **Step 1: Run ruff format**

Run: `ruff format silly_kicks/ tests/`

- [ ] **Step 2: Run ruff check**

Run: `ruff check silly_kicks/ tests/ --fix`

- [ ] **Step 3: Run pyright**

Run: `uv run pyright silly_kicks/`

Fix any type errors.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`

All tests must pass. Fix any failures.

---

## Task 9: Per-provider e2e tests (TF-28) + real-data e2e (TF-29)

**Files:**
- Create: `tests/tracking/test_das_e2e.py`
- Create: `tests/vaep/test_labels_windowing_e2e.py`

Before writing the tests, verify fixture coverage:
1. Check each tracking provider fixture has velocity columns or can get them via `derive_velocities`.
2. Check the StatsBomb WC2018 H5 fixture has matches with goals.
3. Regenerate fixtures from lakehouse/local data if any gaps are found.

- [ ] **Step 1: Write DAS e2e tests**

Create `tests/tracking/test_das_e2e.py`:

```python
"""DAS per-provider e2e tests (TF-28).

Full pipeline: load frames → derive_velocities → infer_ball_carrier →
derive_team_in_possession → get_das / das_at_action.

Uses tests/tracking/_provider_inputs.py loader + synthesize_actions for
consistent action synthesis across providers.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("accessible_space")

from tests.tracking._provider_inputs import (
    load_provider_frames,
    synthesize_actions,
    SLIM_DIR,
    PFF_DIR,
)
from silly_kicks.tracking import (
    derive_team_in_possession,
    infer_ball_carrier,
)
from silly_kicks.tracking._das import get_das
from silly_kicks.tracking.features import das_at_action, add_das
from silly_kicks.tracking.preprocess import derive_velocities

# Available providers: slim-parquet providers + pff
_SLIM_PROVIDERS = sorted(
    p.stem.replace("_slim", "")
    for p in SLIM_DIR.glob("*_slim.parquet")
)
_PROVIDERS = _SLIM_PROVIDERS + (["pff"] if PFF_DIR.exists() else [])


def _prepare_provider(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frames, add velocities + team_in_possession, synthesize actions."""
    frames = load_provider_frames(provider)
    if "vx" not in frames.columns:
        frames = derive_velocities(frames)
    carrier = infer_ball_carrier(frames)
    frames_with_poss = derive_team_in_possession(frames, carrier)
    actions = synthesize_actions(frames)
    return frames_with_poss, actions


@pytest.fixture(params=_PROVIDERS)
def provider_data(request) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    provider = request.param
    frames, actions = _prepare_provider(provider)
    return provider, frames, actions


class TestGetDasE2E:
    def test_output_has_as_das_columns(self, provider_data) -> None:
        provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        assert "AS" in result.columns, f"{provider}: missing AS column"
        assert "DAS" in result.columns, f"{provider}: missing DAS column"

    def test_output_length_matches_input(self, provider_data) -> None:
        provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        assert len(result) == len(frames), (
            f"{provider}: output length {len(result)} != input {len(frames)}"
        )

    def test_das_dtype_float64(self, provider_data) -> None:
        _provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        assert result["DAS"].dtype == np.float64

    def test_not_all_nan(self, provider_data) -> None:
        provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        valid = result["DAS"].dropna()
        assert len(valid) > 0, f"{provider}: all DAS values are NaN"


class TestDasAtActionE2E:
    def test_das_at_action_runs(self, provider_data) -> None:
        provider, frames, actions = provider_data
        result = das_at_action(actions, frames)
        assert len(result) == len(actions)
        assert result.name == "das_team"
        valid = result.dropna()
        assert len(valid) >= 1, f"{provider}: all das_team NaN"

    def test_add_das_adds_three_columns(self, provider_data) -> None:
        provider, frames, actions = provider_data
        enriched = add_das(actions, frames)
        for col in ("das_team", "das_opponent", "das_diff"):
            assert col in enriched.columns, f"{provider}: missing {col}"
        assert len(enriched) == len(actions)
        # At least one action should have non-NaN DAS
        valid_team = enriched["das_team"].dropna()
        assert len(valid_team) >= 1, f"{provider}: all das_team NaN after add_das"
```

- [ ] **Step 2: Write windowing e2e tests**

Create `tests/vaep/test_labels_windowing_e2e.py`:

```python
"""VAEP windowing e2e tests using StatsBomb WC2018 H5 fixture (TF-29)."""

from pathlib import Path

import pandas as pd
import pytest

import silly_kicks.vaep.labels as lab
from silly_kicks.spadl import add_names
from silly_kicks.spadl.utils import add_possessions

_H5_PATH = Path(__file__).resolve().parent.parent / "datasets" / "spadl" / "spadl-WorldCup-2018.h5"


@pytest.fixture
def wc2018_actions() -> pd.DataFrame:
    """Load a single WC2018 match's SPADL actions from the H5 fixture."""
    if not _H5_PATH.exists():
        pytest.skip(f"H5 fixture not found: {_H5_PATH}")
    games = pd.read_hdf(_H5_PATH, "games")
    game_id = games.iloc[0]["game_id"]
    actions = pd.read_hdf(_H5_PATH, f"actions/game_{game_id}")
    return add_names(actions)


class TestWindowActionE2E:
    def test_scores_shape_and_dtype(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.scores(wc2018_actions, nr_actions=10, window="action")
        assert len(result) == len(wc2018_actions)
        assert result["scores"].dtype == bool

    def test_concedes_shape_and_dtype(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.concedes(wc2018_actions, nr_actions=10, window="action")
        assert len(result) == len(wc2018_actions)
        assert result["concedes"].dtype == bool


class TestWindowPossessionE2E:
    def test_scores_runs(self, wc2018_actions: pd.DataFrame) -> None:
        actions = add_possessions(wc2018_actions)
        result = lab.scores(actions, window="possession")
        assert len(result) == len(actions)
        # WC2018 matches have goals → at least one True
        assert result["scores"].sum() > 0

    def test_concedes_runs(self, wc2018_actions: pd.DataFrame) -> None:
        actions = add_possessions(wc2018_actions)
        result = lab.concedes(actions, window="possession")
        assert len(result) == len(actions)
        assert result["concedes"].sum() > 0


class TestWindowTimeE2E:
    def test_scores_runs(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.scores(wc2018_actions, window="time", window_seconds=15.0)
        assert len(result) == len(wc2018_actions)
        assert result["scores"].sum() > 0

    def test_concedes_runs(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.concedes(wc2018_actions, window="time", window_seconds=15.0)
        assert len(result) == len(wc2018_actions)
        assert result["concedes"].sum() > 0

    def test_wider_window_more_positives(self, wc2018_actions: pd.DataFrame) -> None:
        narrow = lab.scores(wc2018_actions, window="time", window_seconds=5.0)
        wide = lab.scores(wc2018_actions, window="time", window_seconds=30.0)
        # Wider window should find >= as many scoring situations
        assert wide["scores"].sum() >= narrow["scores"].sum()
```

- [ ] **Step 3: Run e2e tests**

Run: `python -m pytest tests/tracking/test_das_e2e.py tests/vaep/test_labels_windowing_e2e.py -v --tb=short`

---

## Task 10: Final review

- [ ] **Step 1: Run `/final-review`**

Invoke the `mad-scientist-skills:final-review` skill. Address any findings.

- [ ] **Step 2: Await explicit commit approval**

Do NOT commit without user approval. Present the summary and wait.
