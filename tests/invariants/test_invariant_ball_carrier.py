"""Physical invariants for ball-carrier inference (TF-5)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking.test_ball_carrier import _concat_frames, _make_carrier_frame


@pytest.fixture
def carrier_multi_frame():
    """Multi-frame fixture with varied carrier scenarios."""
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier

    f1 = _make_carrier_frame(
        frame_id=1,
        ball_x=50.0,
        ball_y=34.0,
        players=[
            dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            dict(pid=20, tid=2, x=52.0, y=35.0, vx=-1.0, vy=0.0),
            dict(pid=30, tid=1, x=48.0, y=34.0, vx=0.0, vy=0.0),
        ],
    )
    f2 = _make_carrier_frame(
        frame_id=2,
        ball_x=50.0,
        ball_y=34.0,
        players=[
            dict(pid=10, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
            dict(pid=20, tid=2, x=51.0, y=34.5, vx=-2.0, vy=0.0),
        ],
    )
    frames = _concat_frames(f1, f2)
    return infer_ball_carrier(frames, tolerance_m=3.0), frames


class TestCarrierInvariants:
    def test_distance_bounded_by_tolerance(self, carrier_multi_frame):
        result, _ = carrier_multi_frame
        valid = result["ball_carrier_distance_m"].dropna()
        assert (valid <= 3.0 + 1e-9).all()

    def test_carrier_is_never_ball_row(self, carrier_multi_frame):
        result, frames = carrier_multi_frame
        ball_pids = frames[frames["is_ball"] == True]["player_id"].unique()  # noqa: E712
        carrier_pids = result["ball_carrier_player_id"].dropna().unique()
        for cpid in carrier_pids:
            assert cpid not in ball_pids or pd.isna(cpid)

    def test_team_id_matches_carrier_player(self, carrier_multi_frame):
        result, frames = carrier_multi_frame
        valid = result[result["ball_carrier_player_id"].notna()]
        player_teams = (
            frames[~frames["is_ball"]]
            .drop_duplicates("player_id")[["player_id", "team_id"]]
            .set_index("player_id")["team_id"]
        )
        for _, row in valid.iterrows():
            cpid = row["ball_carrier_player_id"]
            expected_tid = player_teams.loc[cpid]
            assert row["ball_carrier_team_id"] == expected_tid
