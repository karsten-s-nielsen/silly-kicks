"""derive_velocities -- vx/vy/speed from smoothed positions.

PR-S24 / lakehouse-review S4: raises ValueError when smoothed columns are
absent (no hidden auto-invocation of smooth_frames). Principle of least
surprise -- protects applyInPandas StructType-declared UDFs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import (
    PreprocessConfig,
    derive_velocities,
    smooth_frames,
)


def _toy_frames(n: int = 30, hz: float = 25.0) -> pd.DataFrame:
    t = np.arange(n) / hz
    return pd.DataFrame(
        {
            "game_id": "G1",
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": t,
            "frame_rate": hz,
            "player_id": "P1",
            "team_id": "T1",
            "is_ball": False,
            "is_goalkeeper": False,
            "x": np.linspace(0.0, 5.0, n),
            "y": np.full(n, 30.0),
            "z": np.nan,
            "speed": np.nan,
            "speed_source": None,
            "ball_state": "alive",
            "team_attacking_direction": "ltr",
            "source_provider": "sportec",
        }
    )


def test_raises_when_smoothed_columns_missing():
    """S4: raise loud rather than auto-invoke smooth_frames."""
    f = _toy_frames()
    with pytest.raises(ValueError, match=r"silly_kicks\.tracking\.preprocess\.smooth_frames"):
        derive_velocities(f, config=PreprocessConfig.default())


def test_emits_only_velocity_columns():
    """S4 corollary: do NOT auto-add x_smoothed/y_smoothed/_preprocessed_with.

    ``vx`` and ``vy`` are new columns; ``speed`` is overwritten in place
    (the input frame schema already has a ``speed`` column with NaN values
    where the provider didn't supply one). Net effect: caller gets the three
    documented velocity fields populated, and the StructType-declared UDF
    schema sees no surprise extras.
    """
    f = _toy_frames()
    smoothed = smooth_frames(f, config=PreprocessConfig.default())
    pre_cols = set(smoothed.columns)
    out = derive_velocities(smoothed, config=PreprocessConfig.default())
    new_cols = set(out.columns) - pre_cols
    assert new_cols == {"vx", "vy"}, (
        f"derive_velocities must add only vx/vy as new columns; speed is overwritten in place. Got new_cols={new_cols}"
    )
    # speed is a known column from input schema; verify it is now populated where positions allowed.
    valid = out["x_smoothed"].notna() & out["y_smoothed"].notna()
    assert out.loc[valid, "speed"].notna().any(), "derive_velocities did not populate speed"


def test_velocity_dtypes():
    smoothed = smooth_frames(_toy_frames(), config=PreprocessConfig.default())
    out = derive_velocities(smoothed, config=PreprocessConfig.default())
    assert out["vx"].dtype == np.float64
    assert out["vy"].dtype == np.float64
    assert out["speed"].dtype == np.float64


def test_speed_equals_norm_of_vx_vy():
    smoothed = smooth_frames(_toy_frames(), config=PreprocessConfig.default())
    out = derive_velocities(smoothed, config=PreprocessConfig.default())
    expected = (out["vx"] ** 2 + out["vy"] ** 2) ** 0.5
    valid = out["speed"].notna()
    np.testing.assert_allclose(out.loc[valid, "speed"], expected.loc[valid], atol=1e-9)


def test_constant_position_yields_zero_velocity():
    f = _toy_frames()
    f["x"] = 50.0
    f["y"] = 30.0
    smoothed = smooth_frames(f, config=PreprocessConfig.default())
    out = derive_velocities(smoothed, config=PreprocessConfig.default())
    valid = out["speed"].notna()
    np.testing.assert_allclose(out.loc[valid, "speed"], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Single-frame and short-group edge cases (GS WC2022 match 3851 bug)
# ---------------------------------------------------------------------------


def _pre_smoothed_frames(
    n: int = 30,
    hz: float = 25.0,
    *,
    player_id: str = "P1",
    period_id: int = 1,
) -> pd.DataFrame:
    """Build a minimal pre-smoothed frame set (bypasses smooth_frames)."""
    t = np.arange(n) / hz
    return pd.DataFrame(
        {
            "game_id": "G1",
            "period_id": period_id,
            "frame_id": np.arange(n),
            "time_seconds": t,
            "frame_rate": hz,
            "player_id": player_id,
            "team_id": "T1",
            "is_ball": False,
            "is_goalkeeper": False,
            "x": np.linspace(0.0, 5.0, n),
            "y": np.full(n, 30.0),
            "z": np.nan,
            "speed": np.nan,
            "speed_source": None,
            "ball_state": "alive",
            "team_attacking_direction": "ltr",
            "source_provider": "sportec",
            "x_smoothed": np.linspace(0.0, 5.0, n),
            "y_smoothed": np.full(n, 30.0),
            "_preprocessed_with": "test",
        }
    )


def test_single_frame_group_no_crash():
    """Single-frame player group must not crash -- velocity is undefined.

    Real-world: GS WC2022 match 3851, away #10 has exactly 1 frame in
    period 2.  np.gradient requires >= 2 points.
    """
    f = _pre_smoothed_frames(n=1)
    result = derive_velocities(f)
    assert len(result) == 1
    assert "vx" in result.columns
    assert "vy" in result.columns
    assert "speed" in result.columns
    # Single frame has no meaningful velocity -- must be NaN.
    assert np.isnan(result["vx"].iloc[0])
    assert np.isnan(result["vy"].iloc[0])
    assert np.isnan(result["speed"].iloc[0])


def test_two_frame_group_produces_finite_velocity():
    """Two-frame group is the minimum for np.gradient -- must not crash and
    must produce finite velocity values."""
    f = _pre_smoothed_frames(n=2)
    result = derive_velocities(f)
    assert len(result) == 2
    assert result["vx"].notna().all()
    assert result["vy"].notna().all()
    assert result["speed"].notna().all()


def test_mixed_group_sizes_single_and_normal():
    """DataFrame with both a single-frame group and a normal-length group.

    Simulates the real-world scenario: most player-periods have hundreds of
    frames but one degenerate player-period has exactly 1 frame.
    """
    normal = _pre_smoothed_frames(n=30, player_id="P_normal", period_id=1)
    single = _pre_smoothed_frames(n=1, player_id="P_single", period_id=1)
    combined = pd.concat([normal, single], ignore_index=True)

    result = derive_velocities(combined)
    assert len(result) == 31

    # Normal group: finite velocities.
    normal_out = result[result["player_id"] == "P_normal"]
    assert normal_out["speed"].notna().all()

    # Single-frame group: NaN velocities.
    single_out = result[result["player_id"] == "P_single"]
    assert np.isnan(single_out["vx"].iloc[0])
    assert np.isnan(single_out["vy"].iloc[0])
    assert np.isnan(single_out["speed"].iloc[0])
