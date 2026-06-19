"""Unit tests for silly_kicks.tracking.gradientsports.convert_to_frames."""

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking.gradientsports import convert_to_frames
from silly_kicks.tracking.schema import (
    GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
    TRACKING_CONSTRAINTS,
    TrackingConversionReport,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "datasets" / "tracking" / "gradientsports"
TINY = pd.read_parquet(FIXTURE_DIR / "tiny.parquet")
MEDIUM = pd.read_parquet(FIXTURE_DIR / "medium_halftime.parquet")


def test_tiny_output_columns_match_schema():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    assert set(frames.columns) == set(GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS)


def test_tiny_player_team_id_are_int64_nullable():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    assert str(frames["player_id"].dtype) == "Int64"
    assert str(frames["team_id"].dtype) == "Int64"


def test_tiny_ball_rows_have_nan_player_id():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    ball = frames[frames["is_ball"]]
    assert ball["player_id"].isna().all()
    assert ball["team_id"].isna().all()


def test_tiny_coordinate_bounds():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    lo_x, hi_x = TRACKING_CONSTRAINTS["x"]
    assert frames["x"].between(lo_x, hi_x).all()


def test_home_start_left_self_corrects_via_geometry():
    # TF-23b (ADR-035): the geometric backstop makes orientation data-driven, so a correct vs
    # flipped home_team_start_left converge to the SAME frames when a GK anchor is present (TINY
    # carries a native home GK) -- the flag no longer drives the final orientation. (Pre-4.34.0
    # this asserted x_t + x_f == 105, i.e. the flag flipped the output.)
    f_t, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    f_f, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=False)
    j = f_t.merge(
        f_f,
        on=["period_id", "frame_id", "player_id", "team_id", "is_ball"],
        suffixes=("_t", "_f"),
    )
    assert len(j) > 0
    assert (j["x_t"].round(6) == j["x_f"].round(6)).all()
    assert (j["y_t"].round(6) == j["y_f"].round(6)).all()


def test_extratime_flag_required_when_period3_present():
    medium_with_et = MEDIUM.copy()
    et_rows = medium_with_et[medium_with_et["period_id"] == 2].head(60).copy()
    et_rows["period_id"] = 3
    medium_with_et = pd.concat([medium_with_et, et_rows], ignore_index=True)
    with pytest.raises(ValueError, match="home_team_start_left_extratime"):
        convert_to_frames(
            medium_with_et,
            home_team_id=100,
            home_team_start_left=True,
        )


def test_extratime_parameter_propagates():
    medium_with_et = MEDIUM.copy()
    et_rows = medium_with_et[medium_with_et["period_id"] == 2].head(60).copy()
    et_rows["period_id"] = 3
    medium_with_et = pd.concat([medium_with_et, et_rows], ignore_index=True)
    f, _ = convert_to_frames(
        medium_with_et,
        home_team_id=100,
        home_team_start_left=True,
        home_team_start_left_extratime=False,
    )
    p3 = f[f["period_id"] == 3]
    assert len(p3) > 0
    assert p3["x"].between(0, 105).all()


def test_report_provider_is_gradientsports():
    _, report = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    assert isinstance(report, TrackingConversionReport)
    assert report.provider == "gradientsports"
