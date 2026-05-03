"""smooth_frames -- additive output schema; raw x/y preserved unchanged.

PR-S24 / lakehouse-review S4: smoothing emits x_smoothed / y_smoothed
(ADDITIVE), never mutates raw x / y (Hyrum's Law protection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames


def _toy_frames(n: int = 30, hz: float = 25.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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
            "x": 5.0 + 0.1 * t + rng.normal(0, 0.04, n),
            "y": 30.0 + 0.0 * t + rng.normal(0, 0.04, n),
            "z": np.nan,
            "speed": np.nan,
            "speed_source": None,
            "ball_state": "alive",
            "team_attacking_direction": "ltr",
            "source_provider": "sportec",
        }
    )


def test_raw_x_y_preserved():
    """Hyrum-Law protection: original x/y must not be mutated."""
    f = _toy_frames()
    raw_x = f["x"].copy()
    raw_y = f["y"].copy()
    out = smooth_frames(f, config=PreprocessConfig.default())
    pd.testing.assert_series_equal(out["x"], raw_x, check_names=False)
    pd.testing.assert_series_equal(out["y"], raw_y, check_names=False)


def test_smoothed_columns_added():
    out = smooth_frames(_toy_frames(), config=PreprocessConfig.default())
    assert "x_smoothed" in out.columns
    assert "y_smoothed" in out.columns
    assert out["x_smoothed"].dtype == np.float64
    assert out["y_smoothed"].dtype == np.float64


def test_provenance_column_added():
    out = smooth_frames(_toy_frames(), config=PreprocessConfig.default())
    assert "_preprocessed_with" in out.columns
    val = out["_preprocessed_with"].iloc[0]
    assert "savgol" in val
    assert "0.4" in val


def test_constant_signal_passes_through():
    f = _toy_frames()
    f["x"] = 50.0
    f["y"] = 30.0
    out = smooth_frames(f, config=PreprocessConfig.default())
    assert np.allclose(out["x_smoothed"], 50.0, atol=1e-9)
    assert np.allclose(out["y_smoothed"], 30.0, atol=1e-9)


def test_idempotence():
    f = _toy_frames()
    out1 = smooth_frames(f, config=PreprocessConfig.default())
    out2 = smooth_frames(out1, config=PreprocessConfig.default())
    pd.testing.assert_series_equal(out1["x_smoothed"], out2["x_smoothed"])
    pd.testing.assert_series_equal(out1["y_smoothed"], out2["y_smoothed"])


def test_ema_method_recorded_in_provenance():
    out = smooth_frames(_toy_frames(), method="ema", config=PreprocessConfig.default())
    val = out["_preprocessed_with"].iloc[0]
    assert "ema" in val


def test_nan_positions_stay_nan_in_smoothed():
    f = _toy_frames()
    f.loc[5, "x"] = np.nan
    out = smooth_frames(f, config=PreprocessConfig.default())
    assert pd.isna(out.loc[5, "x_smoothed"])


def test_unsupported_method_raises():
    import pytest

    with pytest.raises(ValueError, match="unsupported method"):
        smooth_frames(_toy_frames(), method="kalman")
