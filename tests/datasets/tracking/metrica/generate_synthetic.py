"""Synthetic in-memory Metrica TrackingDataset builder for kloppy-gateway tests.

The kloppy gateway accepts a ``TrackingDataset`` directly. We construct it
in memory rather than persisting a CSV and re-parsing, since the gateway's
contract is "kloppy-domain in, canonical-DataFrame out".
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
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
    Provider,
    Team,
    TrackingDataset,
    VerticalOrientation,
)

_TESTS_DIR = Path(__file__).resolve().parents[3]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from datasets.tracking._generator_common import (  # noqa: E402
    deterministic_uniform_motion,
    get_provider_baseline,
)

BASELINE = get_provider_baseline("metrica")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 25.0)


def _build_synthetic_tracking_dataset(
    *,
    provider: Provider,
    frame_rate: float,
    n_frames: int,
    inject_realistic_edge_cases: bool = False,
    edge_case_provider: str | None = None,
) -> TrackingDataset:
    """Build a kloppy TrackingDataset with deterministic uniform motion.

    When ``inject_realistic_edge_cases=True`` the underlying reference
    DataFrame is generated with empirical-baseline-calibrated edge cases
    (off-pitch player tail, ball-out interval, ball-x throw-in tail, NaN
    ball coordinates per the per-provider documented rate). Frames whose
    NaN ball-coordinate injection succeeded are dropped from the output
    TrackingDataset (kloppy ``Frame`` requires non-None
    ``ball_coordinates``); the gateway test focuses on player-row
    behaviour through realistic distributions.
    """
    home_team = Team(team_id="home_team", name="Home", ground=Ground.HOME)
    away_team = Team(team_id="away_team", name="Away", ground=Ground.AWAY)
    home_team.players = [
        Player(
            player_id=f"h_{i}",
            team=home_team,
            jersey_no=i,
            first_name="H",
            last_name=str(i),
            starting_position=PositionType.Goalkeeper if i == 0 else PositionType.Unknown,
        )
        for i in range(11)
    ]
    away_team.players = [
        Player(
            player_id=f"a_{i}",
            team=away_team,
            jersey_no=i,
            first_name="A",
            last_name=str(i),
            starting_position=PositionType.Goalkeeper if i == 0 else PositionType.Unknown,
        )
        for i in range(11)
    ]
    period = Period(
        id=1,
        start_timestamp=datetime.timedelta(seconds=0),
        end_timestamp=datetime.timedelta(seconds=n_frames / frame_rate),
    )
    pitch = MetricPitchDimensions(
        x_dim=Dimension(0, 105.0),
        y_dim=Dimension(0, 68.0),
        standardized=False,
        pitch_length=105.0,
        pitch_width=68.0,
    )
    # Custom CS lets us declare the source coords are already in 105x68 m,
    # so the gateway's transform() to 105x68 m is a no-op (plus the
    # orientation flip if needed).
    coord_system = CustomCoordinateSystem(
        origin=Origin.BOTTOM_LEFT,
        vertical_orientation=VerticalOrientation.BOTTOM_TO_TOP,
        pitch_dimensions=pitch,
    )
    metadata = Metadata(
        teams=[home_team, away_team],
        periods=[period],
        pitch_dimensions=pitch,
        coordinate_system=coord_system,
        score=None,
        frame_rate=frame_rate,
        orientation=Orientation.HOME_AWAY,
        flags=DatasetFlag.BALL_OWNING_TEAM,
        provider=provider,
        game_id="synthetic_match",
    )
    ref = deterministic_uniform_motion(
        n_frames=n_frames,
        frame_rate=frame_rate,
        inject_realistic_edge_cases=inject_realistic_edge_cases,
        edge_case_provider=edge_case_provider,
    )
    frames = []
    for fid in range(n_frames):
        ts = datetime.timedelta(seconds=fid / frame_rate)
        frame_rows = ref[ref["frame_id"] == fid]
        players_data = {}
        for _, row in frame_rows[~frame_rows["is_ball"]].iterrows():
            team = home_team if row["team_id"] == 100 else away_team
            jersey = int(row["jersey"])
            player = team.players[jersey]
            x_centered = row["x_centered"]
            y_centered = row["y_centered"]
            if pd.isna(x_centered) or pd.isna(y_centered):
                continue
            x = float(x_centered) + 52.5
            y = float(y_centered) + 34.0
            players_data[player] = PlayerData(
                coordinates=Point(x=x, y=y),
                distance=None,
                speed=float(row["speed_native"]),
            )
        ball_row = frame_rows[frame_rows["is_ball"]].iloc[0]
        ball_x_centered = ball_row["x_centered"]
        ball_y_centered = ball_row["y_centered"]
        if pd.isna(ball_x_centered) or pd.isna(ball_y_centered):
            # Realistic edge cases injected NaN ball coordinates --- per
            # the lakehouse Metrica baseline (~77% NaN), this is normal.
            # kloppy Frame requires non-None ball_coordinates; substitute
            # a center-pitch placeholder so the frame survives.
            ball_x_centered, ball_y_centered = 0.0, 0.0
        frames.append(
            Frame(
                period=period,
                timestamp=ts,
                statistics=[],
                ball_owning_team=home_team,
                ball_state=None,
                frame_id=fid,
                players_data=players_data,
                other_data={},
                ball_coordinates=Point3D(
                    x=float(ball_x_centered) + 52.5,
                    y=float(ball_y_centered) + 34.0,
                    z=float(ball_row["z"]) if str(ball_row["z"]) != "nan" else 0.0,
                ),
                ball_speed=float(ball_row["speed_native"]),
            )
        )
    return TrackingDataset(metadata=metadata, records=frames)


def build_metrica_tracking_dataset(
    n_frames: int = 75,
    *,
    inject_realistic_edge_cases: bool = False,
) -> TrackingDataset:
    """Build a synthetic Metrica TrackingDataset (default 75 frames at 25 Hz = 3 s).

    Set ``inject_realistic_edge_cases=True`` to draw structural edge
    cases (NaN ball coords at the per-provider documented rate, off-pitch
    player tail, ball-out interval) from the empirical baseline.
    """
    return _build_synthetic_tracking_dataset(
        provider=Provider.METRICA,
        frame_rate=FRAME_RATE,
        n_frames=n_frames,
        inject_realistic_edge_cases=inject_realistic_edge_cases,
        edge_case_provider="metrica",
    )


if __name__ == "__main__":
    ds = build_metrica_tracking_dataset()
    print(f"Built TrackingDataset with {len(ds.records)} frames, {len(ds.metadata.teams[0].players)} home players")
    ds_realistic = build_metrica_tracking_dataset(n_frames=200, inject_realistic_edge_cases=True)
    print(f"Realistic variant: {len(ds_realistic.records)} frames")
