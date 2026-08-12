"""Tests for silly_kicks.tracking._pausa — PAUSA scoring (Lee et al. 2026)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._pausa import compute_pausa, compute_pausa_batch

# ADR-041 opt-out: deliberately exercises the synthetic placeholder EPV path (the default surface is this module's
# subject).
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")

# ---------------------------------------------------------------------------
# Tests — compute_pausa (scalar)
# ---------------------------------------------------------------------------


class TestComputePausa:
    def test_basic_decomposition(self):
        result = compute_pausa(0.5, 1.0, 0.8)
        assert result["pausa_temporal"] == pytest.approx(0.5)
        assert result["pausa_spatial"] == pytest.approx(0.625)
        assert result["pausa_composite"] == pytest.approx(0.3125)

    def test_perfect_pass(self):
        result = compute_pausa(1.0, 1.0, 1.0)
        assert result["pausa_temporal"] == pytest.approx(1.0)
        assert result["pausa_spatial"] == pytest.approx(1.0)
        assert result["pausa_composite"] == pytest.approx(1.0)

    def test_zero_peak_obso(self):
        """Division by zero on peak yields 0 temporal."""
        result = compute_pausa(0.5, 0.0, 0.8)
        assert result["pausa_temporal"] == 0.0
        assert result["pausa_composite"] == 0.0

    def test_zero_optimal_obso(self):
        """Division by zero on optimal yields 0 spatial."""
        result = compute_pausa(0.5, 1.0, 0.0)
        assert result["pausa_spatial"] == 0.0
        assert result["pausa_composite"] == 0.0

    def test_both_zero(self):
        result = compute_pausa(0.5, 0.0, 0.0)
        assert result["pausa_temporal"] == 0.0
        assert result["pausa_spatial"] == 0.0
        assert result["pausa_composite"] == 0.0

    def test_clamp_above_one(self):
        """actual > peak → temporal clamped to 1.0."""
        result = compute_pausa(1.5, 1.0, 1.0)
        assert result["pausa_temporal"] == pytest.approx(1.0)

    def test_keys(self):
        result = compute_pausa(0.5, 1.0, 0.8)
        assert set(result.keys()) == {
            "pausa_temporal",
            "pausa_spatial",
            "pausa_composite",
        }


# ---------------------------------------------------------------------------
# Tests — compute_pausa_batch (vectorized)
# ---------------------------------------------------------------------------


class TestComputePausaBatch:
    def _make_actions(self, n: int = 5) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "obso_actual": rng.uniform(0.0, 0.6, n),
                "obso_peak": rng.uniform(0.5, 1.0, n),
                "obso_optimal": rng.uniform(0.3, 1.0, n),
            }
        )

    def test_output_columns(self):
        df = self._make_actions()
        result = compute_pausa_batch(df)
        assert "pausa_temporal" in result.columns
        assert "pausa_spatial" in result.columns
        assert "pausa_composite" in result.columns

    def test_preserves_input_columns(self):
        df = self._make_actions()
        result = compute_pausa_batch(df)
        for col in ("obso_actual", "obso_peak", "obso_optimal"):
            assert col in result.columns

    def test_values_match_scalar(self):
        """Batch results should match scalar computation."""
        df = self._make_actions(3)
        batch = compute_pausa_batch(df)
        for i in range(len(df)):
            scalar = compute_pausa(
                df.iloc[i]["obso_actual"],
                df.iloc[i]["obso_peak"],
                df.iloc[i]["obso_optimal"],
            )
            assert batch.iloc[i]["pausa_temporal"] == pytest.approx(
                scalar["pausa_temporal"],
                abs=1e-10,
            )
            assert batch.iloc[i]["pausa_spatial"] == pytest.approx(
                scalar["pausa_spatial"],
                abs=1e-10,
            )
            assert batch.iloc[i]["pausa_composite"] == pytest.approx(
                scalar["pausa_composite"],
                abs=1e-10,
            )

    def test_composite_equals_product(self):
        df = self._make_actions()
        result = compute_pausa_batch(df)
        expected = result["pausa_temporal"] * result["pausa_spatial"]
        np.testing.assert_array_almost_equal(
            result["pausa_composite"].values,  # type: ignore[arg-type]
            expected.values,  # type: ignore[arg-type]
        )

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"obso_actual": [0.5], "obso_peak": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_pausa_batch(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            {
                "obso_actual": pd.Series(dtype=float),
                "obso_peak": pd.Series(dtype=float),
                "obso_optimal": pd.Series(dtype=float),
            }
        )
        result = compute_pausa_batch(df)
        assert len(result) == 0
        assert "pausa_composite" in result.columns

    def test_zero_peak_vectorized(self):
        df = pd.DataFrame(
            {
                "obso_actual": [0.5, 0.3],
                "obso_peak": [0.0, 1.0],
                "obso_optimal": [0.8, 0.0],
            }
        )
        result = compute_pausa_batch(df)
        assert result.iloc[0]["pausa_temporal"] == 0.0
        assert result.iloc[1]["pausa_spatial"] == 0.0

    def test_values_clamped_zero_to_one(self):
        df = self._make_actions(20)
        result = compute_pausa_batch(df)
        assert (result["pausa_temporal"] >= 0.0).all()
        assert (result["pausa_temporal"] <= 1.0).all()
        assert (result["pausa_spatial"] >= 0.0).all()
        assert (result["pausa_spatial"] <= 1.0).all()
        assert (result["pausa_composite"] >= 0.0).all()
        assert (result["pausa_composite"] <= 1.0).all()


# ---------------------------------------------------------------------------
# Aggregator + VAEP factory tests
# ---------------------------------------------------------------------------


def _make_pausa_actions_and_frames():
    """Minimal actions + frames for PAUSA aggregator tests."""
    rng = np.random.RandomState(42)
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [0.2, 0.6],
            "team_id": [1, 2],
            "player_id": [101, 201],
            "start_x": [30.0, 80.0],
            "start_y": [34.0, 34.0],
            "end_x": [50.0, 60.0],
            "end_y": [34.0, 34.0],
            "type_id": [0, 0],
            "type_name": ["pass", "pass"],
            "result_id": [1, 1],
            "result_name": ["success", "success"],
            "bodypart_id": [0, 0],
            "bodypart_name": ["foot", "foot"],
        }
    )
    # Frames are the canonical home-attacks-right convention: the callers below pass
    # ``home_team_id=1``, so team 1 attacks x=105 ("ltr") and team 2 attacks x=0 ("rtl").
    # Ball rows carry None -- that is what ``convert_to_frames`` emits, and
    # ``acting_team_attacks_rtl`` filters them out anyway.
    #
    # Before this was set the fixture had NO ``team_attacking_direction`` at all, so
    # ``acting_team_attacks_rtl`` returned an all-False flip and action_id=1 (team 2, the
    # AWAY team) was scored on the unoriented path. PAUSA delegates to ``add_obso``, so it
    # inherited the same defect; it now exercises the real ADR-028 re-projection.
    frame_rows = []
    for fid in range(50):
        t = fid / 25.0
        for tid in [1, 2]:
            for j in range(4):
                frame_rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "time_seconds": t,
                        "source_provider": "test",
                        "player_id": tid * 100 + j,
                        "team_id": tid,
                        "x": 20 + (tid - 1) * 60 + rng.uniform(-3, 3),
                        "y": 10 + j * 15 + rng.uniform(-2, 2),
                        "vx": 0.5 * (1 - 2 * (tid - 1)),
                        "vy": 0.0,
                        "is_ball": False,
                        "is_goalkeeper": j == 0,
                        "team_attacking_direction": "ltr" if tid == 1 else "rtl",
                    }
                )
        frame_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "source_provider": "test",
                "player_id": None,
                "team_id": None,
                "x": 52.5,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "is_goalkeeper": False,
                "team_attacking_direction": None,
            }
        )
    frames = pd.DataFrame(frame_rows)
    return actions, frames


class TestAddPausa:
    def test_enrichment_columns(self):
        """add_pausa adds 3 PAUSA columns (+ OBSO if missing)."""
        from silly_kicks.tracking.features import add_pausa

        actions, frames = _make_pausa_actions_and_frames()
        result = add_pausa(actions, frames)
        expected_cols = {"pausa_temporal", "pausa_spatial", "pausa_composite"}
        added = set(result.columns) - set(actions.columns)
        assert expected_cols.issubset(added)

    def test_pausa_bounded(self):
        """PAUSA columns are in [0, 1] or NaN."""
        from silly_kicks.tracking.features import add_pausa

        actions, frames = _make_pausa_actions_and_frames()
        result = add_pausa(actions, frames)
        for col in ["pausa_temporal", "pausa_spatial", "pausa_composite"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 0.0
                assert vals.max() <= 1.0

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_pausa

        actions, frames = _make_pausa_actions_and_frames()
        result = add_pausa(actions, frames)
        assert len(result) == len(actions)


class TestPausaXfns:
    def test_column_count(self):
        """pausa_xfns produces 3 lifted xfns (9 VAEP columns)."""
        from silly_kicks.tracking.features import pausa_xfns

        xfns = pausa_xfns()
        assert len(xfns) == 3

    def test_introspection_nan(self):
        """xfns produce NaN in introspection mode (frames=None)."""
        from silly_kicks.tracking.features import pausa_xfns

        xfns = pausa_xfns()
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [0],
                "period_id": [1],
                "time_seconds": [10.0],
                "team_id": [1],
                "player_id": [101],
                "start_x": [30.0],
                "start_y": [34.0],
                "end_x": [60.0],
                "end_y": [30.0],
                "type_id": [0],
                "type_name": ["pass"],
                "result_id": [1],
                "result_name": ["success"],
                "bodypart_id": [0],
                "bodypart_name": ["foot"],
            }
        )
        gamestates = [actions, actions, actions]
        for xfn in xfns:
            result = xfn(gamestates, None)
            assert result.isna().all().all() or result.isna().all()
