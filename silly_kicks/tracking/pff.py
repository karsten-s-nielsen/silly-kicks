"""PFF FC tracking DataFrame converter.

Mirrors silly_kicks.tracking.sportec but for PFF-shaped input. Reuses the
shared ``_direction`` helper extracted from silly_kicks.spadl.pff (PR-S18) for
``home_team_start_left[_extratime]`` direction normalization.

Input contract (EXPECTED_INPUT_COLUMNS):
  Same shape as sportec, except ``player_id`` / ``team_id`` are nullable
  Int64 (PFF integer identifiers) and ``game_id`` is int64.

Coordinate transformation: ``x = x_centered + 52.5``;
``y = y_centered + 34.0``. Per-period direction flip via the shared
``home_attacks_right_per_period`` helper.
"""

from __future__ import annotations

import pandas as pd

from . import _direction
from .schema import PFF_TRACKING_FRAMES_COLUMNS, TrackingConversionReport

EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "frame_rate",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x_centered",
        "y_centered",
        "z",
        "speed_native",
        "ball_state",
    }
)


def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert PFF-shaped raw tracking frames to canonical schema.

    Parameters
    ----------
    raw_frames : pd.DataFrame
        PFF input (see EXPECTED_INPUT_COLUMNS).
    home_team_id : int
        PFF homeTeam.id from the metadata JSON.
    home_team_start_left : bool
        From PFF metadata ``homeTeamStartLeft``.
    home_team_start_left_extratime : bool | None
        From PFF metadata ``homeTeamStartLeftExtraTime``; required when
        periods 3/4 are present.

    Returns
    -------
    frames : pd.DataFrame
        PFF_TRACKING_FRAMES_COLUMNS-shaped output, 105x68 m SPADL coordinates.
    report : TrackingConversionReport

    Examples
    --------
    Read PFF tracking JSONL.bz2, flatten to frames, then convert::

        import bz2, json, pandas as pd
        from silly_kicks.tracking.pff import convert_to_frames
        with bz2.open("10501.jsonl.bz2", "rt") as fh:
            rows = [json.loads(line) for line in fh]
        raw = pd.json_normalize(rows)  # caller-shaped flattening
        frames, report = convert_to_frames(
            raw, home_team_id=366, home_team_start_left=True,
        )
    """
    _ = preserve_native  # reserved for future PR
    missing = EXPECTED_INPUT_COLUMNS - set(raw_frames.columns)
    if missing:
        raise ValueError(f"pff convert_to_frames missing columns: {sorted(missing)}")

    if raw_frames["period_id"].isin([3, 4]).any() and home_team_start_left_extratime is None:
        raise ValueError(
            "pff convert_to_frames: frames contain ET periods (period_id in {3, 4}) "
            "but home_team_start_left_extratime was not provided. Set it from "
            "PFF metadata.homeTeamStartLeftExtraTime, or filter ET frames out."
        )

    out = raw_frames.copy()
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    home_attacks_right = _direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    home_rtl_periods = {p for p, attacks_right in home_attacks_right.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]

    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    is_home = (out["team_id"] == home_team_id).fillna(False).to_numpy(dtype=bool)
    is_known_period = out["period_id"].isin([1, 2, 3, 4]).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known_period, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known_period, "team_attacking_direction"] = "rtl"

    out["speed"] = out["speed_native"].astype("float64")
    speed_source: list[object] = ["native" if pd.notna(v) else None for v in out["speed"]]
    out["speed_source"] = pd.Series(speed_source, index=out.index, dtype="object")
    out["confidence"] = None
    out["visibility"] = None
    out["source_provider"] = "pff"

    final = pd.DataFrame({col: out[col] for col in PFF_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in PFF_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str == "Int64":
            final[col] = final[col].astype("Int64")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)  # type: ignore[arg-type]
        elif dtype_str == "object":
            final[col] = final[col].astype(object)

    n_input_frames = int(raw_frames["frame_id"].nunique())
    n_periods = int(raw_frames["period_id"].nunique())
    cov: dict[int, float] = {}
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        expected = int(g["frame_id"].max() - g["frame_id"].min() + 1)
        actual = int(g["frame_id"].nunique())
        cov[int(p)] = float(actual) / max(expected, 1)  # type: ignore[arg-type]
        ball_g = g[g["is_ball"]]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)  # type: ignore[arg-type]

    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider="pff",
        total_input_frames=n_input_frames,
        total_output_rows=len(final),
        n_periods=n_periods,
        frame_coverage_per_period=cov,
        ball_out_seconds_per_period=ball_out,
        nan_rate_per_column=nan_rate,
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
    )
    return final, report
