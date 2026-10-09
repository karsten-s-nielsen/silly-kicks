"""TF-65 Task 4: soft plausibility guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig
from silly_kicks.tracking.preprocess._guard import PlausibilityWarning, apply_plausibility_guard


def _frames_with_speed(speeds):
    n = len(speeds)
    return pd.DataFrame(
        {
            "vx": speeds,
            "vy": np.zeros(n),
            "speed": np.abs(speeds),
            "ax": np.zeros(n),
            "ay": np.zeros(n),
            "accel": np.zeros(n),
        }
    )


def test_guard_nans_implausible_soft_with_warning_not_raise():
    f = _frames_with_speed(np.array([5.0, 500.0, 10.0]))
    cfg = PreprocessConfig.default()
    with pytest.warns(PlausibilityWarning):
        out = apply_plausibility_guard(f, cfg)
    assert pd.isna(out.loc[1, "speed"]) and pd.isna(out.loc[1, "vx"])
    assert out.loc[0, "speed"] == 5.0 and out.loc[2, "speed"] == 10.0  # plausible untouched


def test_guard_does_not_raise_on_out_of_pitch_ball():
    f = _frames_with_speed(np.full(100, 60.0))  # ~6% out-of-pitch ball analogue, all > 40
    cfg = PreprocessConfig.default()
    out = apply_plausibility_guard(f, cfg)  # must return, not raise
    assert out["speed"].isna().all()
