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
    expected = np.sqrt(out["vx"] ** 2 + out["vy"] ** 2)
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
