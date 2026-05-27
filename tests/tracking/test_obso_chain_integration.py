"""Integration test: OBSO -> Space Creation -> PAUSA chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._obso import ObsoSurface, compute_obso_surface
from silly_kicks.tracking._pausa import compute_pausa, compute_pausa_batch
from silly_kicks.tracking._space_creation import compute_space_created
from silly_kicks.tracking.pitch_control import compute_pitch_control


def _make_chain_frame(n_per_team: int = 4) -> pd.DataFrame:
    """Build a single tracking frame for the chain test."""
    rows = []
    rng = np.random.RandomState(7)
    for tid_idx, tid in enumerate([1, 2]):
        for j in range(n_per_team):
            x = 25.0 + tid_idx * 55 + rng.uniform(-5, 5)
            y = 10.0 + j * 14 + rng.uniform(-2, 2)
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": 0,
                    "time_seconds": 0.0,
                    "player_id": tid * 100 + j,
                    "team_id": tid,
                    "x": x,
                    "y": y,
                    "vx": 0.5 * (1 - 2 * tid_idx),
                    "vy": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": j == 0,
                }
            )
    rows.append(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 0,
            "time_seconds": 0.0,
            "player_id": None,
            "team_id": None,
            "x": 52.5,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


class TestObsoChain:
    def test_obso_feeds_space_creation(self):
        """OBSO surface can be computed from the same PC used in space creation."""
        frame = _make_chain_frame()
        ball_pos = (52.5, 34.0)
        atk_id = 1

        # Step 1: compute pitch control
        pc = compute_pitch_control(frame, atk_id, ball_position=ball_pos)
        assert pc.surface.shape[0] > 0

        # Step 2: compute OBSO
        obso = compute_obso_surface(pc, ball_pos)
        assert isinstance(obso, ObsoSurface)
        assert obso.values.shape == pc.surface.shape
        assert np.all(obso.values >= 0)
        assert np.all(obso.values <= 1.0)

        # Step 3: space creation uses the same PC machinery internally
        sc = compute_space_created(frame, atk_id, ball_position=ball_pos)
        assert len(sc) > 0
        assert "space_created_m2" in sc.columns

    def test_obso_triplet_feeds_pausa(self):
        """OBSO triplet (actual, peak, optimal) feeds directly into PAUSA."""
        actual = 0.4
        peak = 0.8
        optimal = 0.6

        pausa = compute_pausa(actual, peak, optimal)
        assert pausa["pausa_temporal"] == pytest.approx(0.5)
        assert pausa["pausa_spatial"] == pytest.approx(0.4 / 0.6)
        assert pausa["pausa_composite"] == pytest.approx(
            pausa["pausa_temporal"] * pausa["pausa_spatial"],
        )

    def test_full_chain_batch(self):
        """Simulated full chain: OBSO values -> PAUSA batch."""
        # Simulate 3 pass actions with OBSO triplets
        actions = pd.DataFrame(
            {
                "obso_actual": [0.3, 0.5, 0.1],
                "obso_peak": [0.6, 0.5, 0.4],
                "obso_optimal": [0.4, 0.7, 0.2],
            }
        )

        result = compute_pausa_batch(actions)
        assert len(result) == 3
        assert (result["pausa_temporal"] >= 0).all()
        assert (result["pausa_temporal"] <= 1).all()
        assert (result["pausa_spatial"] >= 0).all()
        assert (result["pausa_spatial"] <= 1).all()

        # Composite = temporal * spatial
        np.testing.assert_array_almost_equal(
            result["pausa_composite"].values,
            (result["pausa_temporal"] * result["pausa_spatial"]).values,
        )
