"""Shared helpers: lakehouse wide-form rows -> silly_kicks adapter inputs.

Used by both:
  - ``tests/test_tracking_real_data_sweep.py`` (e2e, queries the lakehouse
    on the developer machine and feeds the slim result through these
    helpers).
  - ``tests/test_tracking_realistic_fixtures.py`` (CI, loads the committed
    ``lakehouse_derived.parquet`` slices and feeds them through the same
    helpers --- so CI exercises the same code path against real-distribution
    data without needing Databricks connectivity).

The lakehouse mart is wide-form 120x80 yards, direction-normalized, with
``x`` / ``y`` per player and separate ``ball_x`` / ``ball_y`` per frame.
silly_kicks tracking adapters expect long-form (player rows + ball rows)
in pitch-centered meters (sportec) or kloppy ``TrackingDataset`` (gateway).
The helpers below do the bridge.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd

# Sportec-shaped output column set the native adapter consumes.
_SPORTEC_KEEP_COLS = [
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
]


def lakehouse_to_sportec_input(lakehouse_df: pd.DataFrame) -> pd.DataFrame:
    """Translate wide-form 120x80 lakehouse rows -> sportec adapter input.

    The lakehouse mart is direction-normalized; callers should pass
    ``home_team_start_left=True`` to make the adapter's period-1 flip a
    no-op. Period 2 will then be flipped, mismatching the lakehouse
    normalization, but the bounds + distribution audits still exercise the
    adapter's coordinate translation + dtype contract on real data.
    """
    src = lakehouse_df.copy()
    src["x_centered"] = (src["x"].astype(float) / 120.0) * 105.0 - 52.5
    src["y_centered"] = (src["y"].astype(float) / 80.0) * 68.0 - 34.0
    src["z"] = float("nan")
    src["speed_native"] = pd.to_numeric(src["speed_ms"], errors="coerce")
    src["ball_state"] = "alive"
    src = src.rename(
        columns={
            "period": "period_id",
            "frame": "frame_id",
            "timestamp_seconds": "time_seconds",
        }
    )
    src["is_ball"] = False
    src["frame_rate"] = pd.to_numeric(src["frame_rate"], errors="coerce").astype("float64")
    ball_rows = src.drop_duplicates(["match_id", "period_id", "frame_id"])[
        ["match_id", "period_id", "frame_id", "time_seconds", "frame_rate", "ball_x", "ball_y"]
    ].copy()
    ball_rows["x_centered"] = (ball_rows["ball_x"].astype(float) / 120.0) * 105.0 - 52.5
    ball_rows["y_centered"] = (ball_rows["ball_y"].astype(float) / 80.0) * 68.0 - 34.0
    ball_rows["player_id"] = None
    ball_rows["team_id"] = None
    ball_rows["is_ball"] = True
    ball_rows["is_goalkeeper"] = False
    ball_rows["z"] = float("nan")
    ball_rows["speed_native"] = float("nan")
    ball_rows["ball_state"] = "alive"
    ball_rows["game_id"] = ball_rows["match_id"]
    ball_rows = ball_rows.drop(columns=["ball_x", "ball_y"])
    src["game_id"] = src["match_id"]
    out = pd.concat(
        [src[_SPORTEC_KEEP_COLS], ball_rows[_SPORTEC_KEEP_COLS]],
        ignore_index=True,
    )
    return out.sort_values(["period_id", "frame_id", "is_ball"]).reset_index(drop=True)


def lakehouse_to_kloppy_dataset(lakehouse_df: pd.DataFrame, provider: Any):
    """Build a kloppy TrackingDataset from wide-form lakehouse rows.

    Uses ``CustomCoordinateSystem`` to declare the source coords as 120x80
    yards. The silly_kicks gateway transforms to 105x68 m as part of its
    normal pipeline.
    """
    from kloppy.domain import (
        CustomCoordinateSystem,
        DatasetFlag,
        Dimension,
        Frame,
        Ground,
        Metadata,
        MetricPitchDimensions,
        Orientation,
        Origin,
        Period,
        Player,
        PlayerData,
        Point,
        Point3D,
        PositionType,
        Team,
        TrackingDataset,
        VerticalOrientation,
    )

    df = lakehouse_df.copy()
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["period"])
    df["period"] = df["period"].astype(int)

    home_team = Team(team_id="home_team", name="Home", ground=Ground.HOME)
    away_team = Team(team_id="away_team", name="Away", ground=Ground.AWAY)

    home_players_dict: dict[str, Player] = {}
    away_players_dict: dict[str, Player] = {}
    for _, row in df.drop_duplicates(["team", "player_id"]).iterrows():
        team_label = row["team"]
        pid = str(row["player_id"])
        is_gk = bool(row.get("is_goalkeeper", False))
        if team_label == "home":
            jersey = len(home_players_dict)
            home_players_dict[pid] = Player(
                player_id=pid,
                team=home_team,
                jersey_no=jersey,
                first_name="H",
                last_name=pid,
                starting_position=PositionType.Goalkeeper if is_gk else PositionType.Unknown,
            )
        elif team_label == "away":
            jersey = len(away_players_dict)
            away_players_dict[pid] = Player(
                player_id=pid,
                team=away_team,
                jersey_no=jersey,
                first_name="A",
                last_name=pid,
                starting_position=PositionType.Goalkeeper if is_gk else PositionType.Unknown,
            )
    home_team.players = list(home_players_dict.values())
    away_team.players = list(away_players_dict.values())

    period_ids = sorted(df["period"].unique())
    periods = []
    for pid_int in period_ids:
        period_rows = df[df["period"] == pid_int]
        periods.append(
            Period(
                id=int(pid_int),
                start_timestamp=datetime.timedelta(seconds=float(period_rows["timestamp_seconds"].min())),
                end_timestamp=datetime.timedelta(seconds=float(period_rows["timestamp_seconds"].max())),
            )
        )

    pitch = MetricPitchDimensions(
        x_dim=Dimension(0, 120.0),
        y_dim=Dimension(0, 80.0),
        standardized=False,
        pitch_length=120.0,
        pitch_width=80.0,
    )
    coord_system = CustomCoordinateSystem(
        origin=Origin.BOTTOM_LEFT,
        vertical_orientation=VerticalOrientation.BOTTOM_TO_TOP,
        pitch_dimensions=pitch,
    )
    metadata = Metadata(
        teams=[home_team, away_team],
        periods=periods,
        pitch_dimensions=pitch,
        coordinate_system=coord_system,
        score=None,
        frame_rate=float(df["frame_rate"].iloc[0]),
        orientation=Orientation.HOME_AWAY,
        flags=DatasetFlag.BALL_OWNING_TEAM,
        provider=provider,
        game_id=str(df["match_id"].iloc[0]),
    )

    period_lookup = {p.id: p for p in periods}
    frames = []
    for (pid_int, fid_int), group in df.groupby(["period", "frame"], sort=True):
        period_obj = period_lookup[int(pid_int)]  # type: ignore[arg-type]  # groupby keys are Hashable but runtime gives ints
        ts = datetime.timedelta(seconds=float(group["timestamp_seconds"].iloc[0]))
        players_data = {}
        for _, row in group.iterrows():
            pid = str(row["player_id"])
            player = home_players_dict.get(pid) or away_players_dict.get(pid)
            if player is None or pd.isna(row["x"]) or pd.isna(row["y"]):
                continue
            speed_val = row.get("speed_ms")
            speed = float(speed_val) if pd.notna(speed_val) else None
            players_data[player] = PlayerData(
                coordinates=Point(x=float(row["x"]), y=float(row["y"])),
                distance=None,
                speed=speed,
            )
        ball_x = group["ball_x"].iloc[0]
        ball_y = group["ball_y"].iloc[0]
        # Real Metrica frames have ~77% NaN ball positions (per probe);
        # substitute a center-pitch placeholder rather than dropping the
        # frame so the gateway exercises full ranges of player rows.
        ball_coords = Point3D(
            x=float(ball_x) if pd.notna(ball_x) else 60.0,
            y=float(ball_y) if pd.notna(ball_y) else 40.0,
            z=0.0,
        )
        frames.append(
            Frame(
                period=period_obj,
                timestamp=ts,
                statistics=[],
                ball_owning_team=home_team,
                ball_state=None,
                frame_id=int(fid_int),  # type: ignore[arg-type]  # groupby key Hashable -> int at runtime
                players_data=players_data,
                other_data={},
                ball_coordinates=ball_coords,
                ball_speed=None,
            )
        )
    return TrackingDataset(metadata=metadata, records=frames)
