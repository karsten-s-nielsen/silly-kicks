"""Tests for derive_team_in_possession."""

import pandas as pd

from silly_kicks.tracking._ball_carrier import derive_team_in_possession


def _make_frames(n_frames: int = 3) -> pd.DataFrame:
    """Minimal tracking frames: 2 players + ball, 2 frames per period."""
    rows = []
    for fid in range(n_frames):
        for pid, tid in [("P1", "TeamA"), ("P2", "TeamB")]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "x": 50.0,
                    "y": 34.0,
                    "is_ball": False,
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "x": 52.0,
                "y": 34.0,
                "is_ball": True,
            }
        )
    return pd.DataFrame(rows)


def _make_carrier(team_ids: list) -> pd.DataFrame:
    """Carrier df matching _make_frames."""
    return pd.DataFrame(
        {
            "game_id": [1] * len(team_ids),
            "period_id": [1] * len(team_ids),
            "frame_id": list(range(len(team_ids))),
            "ball_carrier_player_id": ["P1"] * len(team_ids),
            "ball_carrier_distance_m": [1.0] * len(team_ids),
            "ball_carrier_team_id": team_ids,
        }
    )


class TestDeriveTeamInPossession:
    def test_basic_merge(self) -> None:
        frames = _make_frames(3)
        carrier = _make_carrier(["TeamA", "TeamB", "TeamA"])
        result = derive_team_in_possession(frames, carrier)
        assert "team_in_possession" in result.columns
        f0 = result[result["frame_id"] == 0]
        assert (f0["team_in_possession"] == "TeamA").all()
        f1 = result[result["frame_id"] == 1]
        assert (f1["team_in_possession"] == "TeamB").all()

    def test_unmatched_frames_get_nan(self) -> None:
        frames = _make_frames(3)
        carrier = _make_carrier(["TeamA", "TeamB"])
        result = derive_team_in_possession(frames, carrier)
        f2 = result[result["frame_id"] == 2]
        assert f2["team_in_possession"].isna().all()

    def test_does_not_mutate_input(self) -> None:
        frames = _make_frames(2)
        carrier = _make_carrier(["TeamA", "TeamB"])
        original_cols = set(frames.columns)
        _ = derive_team_in_possession(frames, carrier)
        assert set(frames.columns) == original_cols
        assert "team_in_possession" not in frames.columns

    def test_empty_carrier(self) -> None:
        frames = _make_frames(2)
        carrier = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "ball_carrier_player_id",
                "ball_carrier_distance_m",
                "ball_carrier_team_id",
            ]
        )
        result = derive_team_in_possession(frames, carrier)
        assert "team_in_possession" in result.columns
        assert result["team_in_possession"].isna().all()
