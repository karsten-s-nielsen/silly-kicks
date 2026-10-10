"""Unit tests for silly_kicks.spadl.diagnose_coordinates (coordinate-integrity tripwire)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import CoordinateDiagnosis, CoordinateDiagnosisParams, diagnose_coordinates


def _actions(x: float, y: float, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "start_x": rng.uniform(0, x, n),
            "start_y": rng.uniform(0, y, n),
            "end_x": rng.uniform(0, x, n),
            "end_y": rng.uniform(0, y, n),
        }
    )


def _frames(x: float, y: float, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "is_ball": False,
            "x": rng.uniform(0, x, n).astype("float32"),
            "y": rng.uniform(0, y, n).astype("float32"),
        }
    )


def test_spadl_meters_clean():
    d = diagnose_coordinates(_actions(105, 68, 100), _frames(105, 68, 100))
    assert d.actions is not None and d.frames is not None
    assert d.actions.inferred_scale == "spadl_meters"
    assert d.frames.inferred_scale == "spadl_meters"
    assert d.flags == []


def test_normalized_0_1_flags_scale():
    d = diagnose_coordinates(_actions(1, 1, 100), None)
    assert d.actions is not None
    assert d.actions.inferred_scale == "normalized_0_1"
    assert "coords_scale_suspect" in d.flags


def test_scale_0_100_flags_scale():
    d = diagnose_coordinates(_actions(100, 100, 100), None)
    assert d.actions is not None
    assert d.actions.inferred_scale == "scale_0_100"
    assert "coords_scale_suspect" in d.flags


def test_undetermined_below_min_n():
    d = diagnose_coordinates(_actions(105, 68, 5), None)
    assert d.actions is not None
    assert d.actions.inferred_scale == "undetermined"
    assert "coords_scale_suspect" not in d.flags


def test_actions_out_of_pitch_flags_but_frames_off_pitch_is_info():
    a = _actions(105, 68, 100)
    a.loc[0, "start_x"] = 130.0  # 25 m off; actions are clipped by contract -> defect
    f = _frames(105, 68, 100)
    f.loc[0, "x"] = np.float32(130.0)  # legit off-pitch for tracking
    d = diagnose_coordinates(a, f)
    assert d.frames is not None
    assert "actions_out_of_pitch" in d.flags
    assert d.frames.out_of_pitch_fraction > 0  # reported...
    assert "coords_gross_out_of_range" not in d.flags  # ...but NOT a defect flag for frames


def test_gross_out_of_range_both_tables():
    d = diagnose_coordinates(_actions(1050, 680, 100), _frames(1050, 680, 100))
    assert "coords_gross_out_of_range" in d.flags


def test_all_nan_and_actions_start_nan():
    a = _actions(105, 68, 100)
    a[["start_x", "start_y"]] = np.nan
    d = diagnose_coordinates(a, None)
    assert "actions_start_nan" in d.flags
    f = _frames(105, 68, 100)
    f[["x", "y"]] = np.nan
    d2 = diagnose_coordinates(None, f)
    assert "coords_all_nan" in d2.flags


def test_end_only_nan_does_not_flag_actions_start_nan():  # discriminating: start-NaN vs end-NaN (frozen token)
    a = _actions(105, 68, 100)
    a[["end_x", "end_y"]] = np.nan  # END only
    d = diagnose_coordinates(a, None)
    assert "actions_start_nan" not in d.flags  # start coords intact -> must NOT fire


def test_start_only_nan_does_not_overclaim_coords_all_nan():  # coords_all_nan = TRUE all-NaN, not any-per-row
    a = _actions(105, 68, 100)
    a[["start_x", "start_y"]] = np.nan  # start NaN, end finite
    d = diagnose_coordinates(a, None)
    assert "actions_start_nan" in d.flags
    assert "coords_all_nan" not in d.flags  # end coords finite -> NOT all-coords-NaN


def test_both_none_raises():
    with pytest.raises(ValueError):
        diagnose_coordinates(None, None)


def test_one_none_diagnoses_the_present_table():
    d = diagnose_coordinates(_actions(105, 68, 100), None)
    assert d.actions is not None and d.frames is None


def test_purity_and_determinism():
    a, f = _actions(105, 68, 100), _frames(105, 68, 100)
    a2, f2 = a.copy(), f.copy()
    d1 = diagnose_coordinates(a, f)
    d2 = diagnose_coordinates(a, f)
    pd.testing.assert_frame_equal(a, a2)  # unmutated
    pd.testing.assert_frame_equal(f, f2)
    assert d1 == d2  # deterministic


def test_notes_state_non_claims():
    d = diagnose_coordinates(_actions(105, 68, 100), _frames(105, 68, 100))
    joined = " ".join(d.notes).lower()
    assert "orientation" in joined and "off-pitch" in joined


def test_for_provider_is_neutral_v1():
    assert CoordinateDiagnosisParams.for_provider("skillcorner") == CoordinateDiagnosisParams()


def test_returns_the_diagnosis_dataclass():
    assert isinstance(diagnose_coordinates(_actions(105, 68, 100), None), CoordinateDiagnosis)
