import numpy as np
import pandas as pd

from silly_kicks.tracking._occurrence_labels import _build_occurrence_labels


def _frames_index():
    # frames-side team column is `team_in_possession` (matches xS, line 299).
    return pd.DataFrame(
        {
            "game_id": ["g"] * 6,
            "period_id": [1] * 6,
            "frame_id": [10, 11, 12, 13, 14, 15],
            "time_seconds": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            "team_in_possession": ["A"] * 6,
        }
    )


def test_occurrence_label_within_horizon():
    fidx = _frames_index()
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.0]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [1, 1, 1, 0, 0, 0]


def test_occurrence_label_robust_to_frame_id_gap():
    fidx = _frames_index()
    fidx.loc[3:, "frame_id"] = [99, 100, 101]  # non-contiguous ids, intact time
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.0]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [1, 1, 1, 0, 0, 0]  # unchanged -> no frame_id arithmetic


def test_occurrence_label_no_period_bleed():
    fidx = _frames_index()
    fidx.loc[3:, "period_id"] = 2
    events = pd.DataFrame({"game_id": ["g"], "period_id": [2], "team_id": ["A"], "time_seconds": [1.3]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [0, 0, 0, 1, 0, 0]


def test_occurrence_label_nan_team_event_dropna_false():
    """M2: groupby(dropna=False) -- a NaN-team event labels nothing but must not raise."""
    fidx = _frames_index()
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": [np.nan], "time_seconds": [0.5]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [0, 0, 0, 0, 0, 0]


def test_occurrence_label_dtype_is_int():
    """M2: preserve xS's platform-int dtype (np.zeros(dtype=int))."""
    fidx = _frames_index()
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.0]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert y.dtype == np.dtype(int)
