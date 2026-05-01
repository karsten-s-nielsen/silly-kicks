"""Unit tests for silly_kicks.tracking.sportec.convert_to_frames."""

from pathlib import Path

import pandas as pd

from silly_kicks.tracking.schema import (
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CONSTRAINTS,
    TrackingConversionReport,
)
from silly_kicks.tracking.sportec import convert_to_frames

FIXTURE_DIR = Path(__file__).resolve().parent / "datasets" / "tracking" / "sportec"
TINY = pd.read_parquet(FIXTURE_DIR / "tiny.parquet")
MEDIUM = pd.read_parquet(FIXTURE_DIR / "medium_halftime.parquet")


def test_tiny_output_shape_and_dtypes():
    frames, _ = convert_to_frames(
        TINY,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )
    assert set(frames.columns) == set(SPORTEC_TRACKING_FRAMES_COLUMNS)
    for col, dtype_str in SPORTEC_TRACKING_FRAMES_COLUMNS.items():
        actual = str(frames[col].dtype)
        if dtype_str == "object" and actual in {"object", "string"}:
            continue
        assert actual == dtype_str or (dtype_str == "bool" and actual == "boolean"), (
            f"{col}: expected {dtype_str}, got {actual}"
        )


def test_tiny_coordinate_bounds():
    frames, _ = convert_to_frames(
        TINY,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )
    lo_x, hi_x = TRACKING_CONSTRAINTS["x"]
    lo_y, hi_y = TRACKING_CONSTRAINTS["y"]
    assert frames["x"].between(lo_x, hi_x).all()
    assert frames["y"].between(lo_y, hi_y).all()


def test_tiny_ball_rows_have_nan_player_team():
    frames, _ = convert_to_frames(
        TINY,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )
    ball = frames[frames["is_ball"]]
    assert ball["player_id"].isna().all()
    assert ball["team_id"].isna().all()
    assert (ball["is_goalkeeper"] == False).all()  # noqa: E712
    assert ball["team_attacking_direction"].isna().all()


def test_tiny_conversion_report_total_input_frames():
    _, report = convert_to_frames(
        TINY,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )
    assert isinstance(report, TrackingConversionReport)
    assert report.provider == "sportec"
    assert report.total_input_frames == TINY["frame_id"].nunique()


def test_home_start_left_false_flips_x_versus_true():
    f_true, _ = convert_to_frames(
        TINY,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )
    f_false, _ = convert_to_frames(
        TINY,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=False,
    )
    merge_cols = ["period_id", "frame_id", "player_id", "team_id", "is_ball"]
    j = f_true.merge(f_false, on=merge_cols, suffixes=("_t", "_f"))
    assert ((j["x_t"] + j["x_f"]).round(6) == 105.0).all()


def test_medium_period_flip_consistency():
    frames, _ = convert_to_frames(
        MEDIUM,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
        home_team_start_left_extratime=None,
    )
    assert frames["x"].between(0, 105).all()
    assert frames["y"].between(0, 68).all()
    assert frames["time_seconds"].notna().all()


def test_unrecognized_player_id_field_is_set():
    """Sportec adapter does not currently validate against a roster --- the
    field is always an empty set in PR-S19. Pinning the type here so a future
    roster-validation feature can tighten the contract via test changes.
    """
    bad = TINY.copy()
    bad.loc[bad.index[0], "player_id"] = "DFL-OBJ-99999"
    _, report = convert_to_frames(
        bad,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )
    assert isinstance(report.unrecognized_player_ids, set)
