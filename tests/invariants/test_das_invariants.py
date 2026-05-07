"""Physical invariants for DAS adapter (TF-28)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("accessible_space")

from silly_kicks.tracking._das import get_das, get_individual_das

pytestmark = pytest.mark.e2e


def _synthetic_frames(n_frames: int = 5) -> pd.DataFrame:
    """Minimal synthetic tracking with 4 players + ball, with team_in_possession."""
    rng = np.random.default_rng(42)
    rows = []
    for fid in range(n_frames):
        for pid, tid in [("P1", "Home"), ("P2", "Home"), ("P3", "Away"), ("P4", "Away")]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "x": rng.uniform(0, 105),
                    "y": rng.uniform(0, 68),
                    "vx": rng.normal(0, 2),
                    "vy": rng.normal(0, 2),
                    "is_ball": False,
                    "team_in_possession": "Home",
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "x": rng.uniform(20, 80),
                "y": rng.uniform(10, 58),
                "vx": rng.normal(0, 3),
                "vy": rng.normal(0, 3),
                "is_ball": True,
                "team_in_possession": "Home",
            }
        )
    return pd.DataFrame(rows)


class TestDasInvariants:
    @pytest.fixture
    def das_result(self) -> pd.DataFrame:
        return get_das(_synthetic_frames(5), use_progress_bar=False)

    @pytest.fixture
    def individual_result(self) -> pd.DataFrame:
        return get_individual_das(_synthetic_frames(5), use_progress_bar=False)

    def test_as_non_negative(self, das_result: pd.DataFrame) -> None:
        valid = das_result["AS"].dropna()
        assert (valid >= 0).all(), f"Negative AS found: {valid[valid < 0].values}"

    def test_das_non_negative(self, das_result: pd.DataFrame) -> None:
        valid = das_result["DAS"].dropna()
        assert (valid >= 0).all(), f"Negative DAS found: {valid[valid < 0].values}"

    def test_as_geq_das(self, das_result: pd.DataFrame) -> None:
        valid = das_result[["AS", "DAS"]].dropna()
        assert (valid["AS"] >= valid["DAS"] - 1e-9).all(), "AS < DAS found"

    def test_individual_as_non_negative(self, individual_result: pd.DataFrame) -> None:
        valid = individual_result["AS"].dropna()
        assert (valid >= 0).all()

    def test_individual_das_non_negative(self, individual_result: pd.DataFrame) -> None:
        valid = individual_result["DAS"].dropna()
        assert (valid >= 0).all()

    def test_output_length_matches_input(self, das_result: pd.DataFrame) -> None:
        expected_len = 5 * 5  # 5 frames x (4 players + 1 ball)
        assert len(das_result) == expected_len


class TestStationary:
    def test_stationary_players_valid(self) -> None:
        frames = _synthetic_frames(3)
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        result = get_das(frames, use_progress_bar=False)
        valid = result["DAS"].dropna()
        assert len(valid) > 0, "All-stationary frames produced all-NaN DAS"
