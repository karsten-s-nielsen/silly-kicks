"""interpolate_frames -- linear NaN gap-fill up to max_gap_seconds.

PR-S24 / lakehouse-review N3: cubic intentionally NOT in the API surface;
restrict Literal to "linear". Cubic ships as TF-9-cubic when requested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig, interpolate_frames


def _frames_with_gap(gap_indices: list[int], n: int = 20, hz: float = 25.0) -> pd.DataFrame:
    t = np.arange(n) / hz
    x = np.linspace(0.0, 19.0, n)
    y = np.linspace(30.0, 49.0, n)
    for i in gap_indices:
        x[i] = np.nan
        y[i] = np.nan
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
            "x": x,
            "y": y,
            "z": np.nan,
            "speed": np.nan,
            "speed_source": None,
            "ball_state": "alive",
            "team_attacking_direction": "ltr",
            "source_provider": "sportec",
        }
    )


def test_short_gap_filled():
    f = _frames_with_gap([5, 6])  # 2-frame gap = 0.08 s @ 25 Hz; well under 0.5 s default
    out = interpolate_frames(f, config=PreprocessConfig.default())
    assert not out["x"].isna().any()
    # Linearly interpolated between x[4]=4.0 and x[7]=7.0 -> x[5]=5.0, x[6]=6.0
    assert np.isclose(out.loc[5, "x"], 5.0, atol=1e-6)
    assert np.isclose(out.loc[6, "x"], 6.0, atol=1e-6)


def test_long_gap_stays_nan():
    f = _frames_with_gap(list(range(5, 19)))  # 14-frame gap = 0.56 s @ 25 Hz; > 0.5 default
    out = interpolate_frames(f, config=PreprocessConfig.default())
    assert out["x"].isna().any()


def test_observed_values_unchanged():
    f = _frames_with_gap([5])
    out = interpolate_frames(f, config=PreprocessConfig.default())
    raw = _frames_with_gap([5])
    valid = raw["x"].notna()
    pd.testing.assert_series_equal(
        out.loc[valid, "x"].reset_index(drop=True),
        raw.loc[valid, "x"].reset_index(drop=True),
        check_names=False,
    )


def test_idempotence():
    f = _frames_with_gap([5, 6])
    once = interpolate_frames(f, config=PreprocessConfig.default())
    twice = interpolate_frames(once, config=PreprocessConfig.default())
    pd.testing.assert_series_equal(once["x"], twice["x"])


def test_cubic_method_rejected():
    """N3 fix: only 'linear' is supported in PR-S24."""
    f = _frames_with_gap([5])
    with pytest.raises(ValueError, match="Only 'linear' is supported"):
        interpolate_frames(f, method="cubic")
