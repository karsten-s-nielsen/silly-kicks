"""Tests for silly_kicks.tracking._elastic_sync — ELASTIC sync (Kim et al. 2025)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._elastic_sync import (
    ElasticSyncParams,
    align_events_to_frames,
    extract_ball_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracking_frames(
    n_frames: int = 50,
    n_players_per_team: int = 3,
    frame_rate: int = 25,
) -> pd.DataFrame:
    """Build minimal tracking frames with ball + players."""
    rows = []
    rng = np.random.RandomState(42)
    for fid in range(n_frames):
        t = fid / frame_rate
        # Ball with some movement
        bx = 50.0 + 2.0 * np.sin(fid * 0.3) + rng.normal(0, 0.2)
        by = 34.0 + 1.0 * np.cos(fid * 0.2) + rng.normal(0, 0.1)
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "x": bx,
                "y": by,
                "is_ball": True,
            }
        )
        # Players
        for tid in [1, 2]:
            for pid_idx in range(n_players_per_team):
                px = 30.0 + tid * 25 + rng.normal(0, 3)
                py = 15.0 + pid_idx * 15 + rng.normal(0, 2)
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "time_seconds": t,
                        "player_id": f"p{tid}_{pid_idx}",
                        "team_id": tid,
                        "x": px,
                        "y": py,
                        "is_ball": False,
                    }
                )
    return pd.DataFrame(rows)


def _make_actions(n: int = 5) -> pd.DataFrame:
    """Build minimal SPADL-like actions for alignment."""
    return pd.DataFrame(
        {
            "action_id": range(n),
            "game_id": [1] * n,
            "period_id": [1] * n,
            "time_seconds": np.linspace(0.2, 1.6, n),
            "player_id": [f"p1_{i % 3}" for i in range(n)],
            "type_id": [0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Tests — ElasticSyncParams
# ---------------------------------------------------------------------------


class TestElasticSyncParams:
    def test_frozen(self):
        params = ElasticSyncParams()
        with pytest.raises(AttributeError):
            params.window_seconds = 2.0  # type: ignore[misc]

    def test_defaults(self):
        params = ElasticSyncParams()
        assert params.window_seconds == 1.0
        assert params.accel_weight == pytest.approx(0.6)
        assert params.proximity_weight == pytest.approx(0.4)
        assert params.min_confidence == pytest.approx(0.1)
        assert params.frame_rate == 25


# ---------------------------------------------------------------------------
# Tests — extract_ball_features
# ---------------------------------------------------------------------------


class TestExtractBallFeatures:
    def test_output_columns(self):
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert set(bf.columns) == {
            "game_id",
            "period_id",
            "frame_id",
            "ball_x",
            "ball_y",
            "ball_speed",
            "ball_accel",
        }

    def test_one_row_per_frame(self):
        frames = _make_tracking_frames(n_frames=20)
        bf = extract_ball_features(frames)
        assert len(bf) == 20

    def test_speed_nonnegative(self):
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert (bf["ball_speed"] >= 0).all()

    def test_accel_nonnegative(self):
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert (bf["ball_accel"] >= 0).all()

    def test_first_frame_zero_speed(self):
        """First frame in a period has zero speed/accel (no prior)."""
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert bf.iloc[0]["ball_speed"] == pytest.approx(0.0)
        assert bf.iloc[0]["ball_accel"] == pytest.approx(0.0)

    def test_empty_frames(self):
        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "x",
                "y",
                "is_ball",
            ]
        )
        bf = extract_ball_features(frames)
        assert len(bf) == 0

    def test_custom_frame_rate(self):
        frames = _make_tracking_frames(frame_rate=10)
        params = ElasticSyncParams(frame_rate=10)
        bf = extract_ball_features(frames, params=params)
        assert len(bf) > 0

    def test_multi_period(self):
        f1 = _make_tracking_frames(n_frames=10)
        f2 = _make_tracking_frames(n_frames=10)
        f2["period_id"] = 2
        f2["frame_id"] = f2["frame_id"] + 100
        frames = pd.concat([f1, f2], ignore_index=True)
        bf = extract_ball_features(frames)
        assert len(bf) == 20
        # Speed at period boundary should be 0
        p2_first = bf[bf["period_id"] == 2].iloc[0]
        assert p2_first["ball_speed"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests — align_events_to_frames
# ---------------------------------------------------------------------------


class TestAlignEventsToFrames:
    def test_output_columns(self):
        frames = _make_tracking_frames()
        actions = _make_actions()
        result = align_events_to_frames(actions, frames)
        assert set(result.columns) == {
            "action_id",
            "elastic_frame_id",
            "elastic_confidence",
            "elastic_error_seconds",
        }

    def test_confidence_range(self):
        frames = _make_tracking_frames()
        actions = _make_actions()
        result = align_events_to_frames(actions, frames)
        if len(result) > 0:
            assert (result["elastic_confidence"] >= 0.0).all()
            assert (result["elastic_confidence"] <= 1.0).all()

    def test_error_nonnegative(self):
        frames = _make_tracking_frames()
        actions = _make_actions()
        result = align_events_to_frames(actions, frames)
        if len(result) > 0:
            assert (result["elastic_error_seconds"] >= 0.0).all()

    def test_aligned_frame_within_window(self):
        """Aligned frame should be within window of nominal frame."""
        params = ElasticSyncParams(window_seconds=1.0, frame_rate=25)
        frames = _make_tracking_frames(n_frames=100)
        actions = _make_actions()
        result = align_events_to_frames(actions, frames, params=params)

        for _, row in result.iterrows():
            action = actions[actions["action_id"] == row["action_id"]].iloc[0]
            nominal = round(action["time_seconds"] * params.frame_rate)
            window = int(params.window_seconds * params.frame_rate)
            assert abs(row["elastic_frame_id"] - nominal) <= window

    def test_empty_actions(self):
        frames = _make_tracking_frames()
        actions = pd.DataFrame(
            columns=[
                "action_id",
                "game_id",
                "period_id",
                "time_seconds",
                "player_id",
                "type_id",
            ]
        )
        result = align_events_to_frames(actions, frames)
        assert len(result) == 0

    def test_empty_frames(self):
        actions = _make_actions()
        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "x",
                "y",
                "is_ball",
                "player_id",
                "team_id",
            ]
        )
        result = align_events_to_frames(actions, frames)
        assert len(result) == 0

    def test_min_confidence_filter(self):
        """High min_confidence should filter out low-scoring alignments."""
        frames = _make_tracking_frames()
        actions = _make_actions()
        params_loose = ElasticSyncParams(min_confidence=0.01)
        params_strict = ElasticSyncParams(min_confidence=0.99)
        result_loose = align_events_to_frames(actions, frames, params=params_loose)
        result_strict = align_events_to_frames(actions, frames, params=params_strict)
        assert len(result_strict) <= len(result_loose)

    def test_custom_weights(self):
        frames = _make_tracking_frames()
        actions = _make_actions()
        params = ElasticSyncParams(accel_weight=1.0, proximity_weight=0.0)
        result = align_events_to_frames(actions, frames, params=params)
        assert len(result) >= 0  # Just verifies it doesn't crash


# ---------------------------------------------------------------------------
# Aggregator + VAEP factory tests
# ---------------------------------------------------------------------------


def _make_spadl_actions() -> pd.DataFrame:
    """Minimal SPADL-like actions for aggregator tests."""
    return pd.DataFrame(
        {
            "action_id": [0, 1, 2],
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [0.2, 0.8, 1.4],
            "team_id": [1, 1, 2],
            "player_id": ["p1_0", "p1_1", "p2_0"],
            "start_x": [30.0, 40.0, 70.0],
            "start_y": [34.0, 34.0, 34.0],
            "end_x": [40.0, 50.0, 60.0],
            "end_y": [34.0, 34.0, 34.0],
            "type_id": [0, 0, 0],
            "type_name": ["pass", "pass", "pass"],
            "result_id": [1, 1, 1],
            "result_name": ["success", "success", "success"],
            "bodypart_id": [0, 0, 0],
            "bodypart_name": ["foot", "foot", "foot"],
        }
    )


class TestAddElasticSync:
    def test_enrichment_columns(self):
        """add_elastic_sync adds 3 ELASTIC columns."""
        from silly_kicks.tracking.features import add_elastic_sync

        actions = _make_spadl_actions()
        frames = _make_tracking_frames()
        result = add_elastic_sync(actions, frames)
        expected_cols = {
            "elastic_frame_id",
            "elastic_confidence",
            "elastic_error_seconds",
        }
        added = set(result.columns) - set(actions.columns)
        assert expected_cols.issubset(added)

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_elastic_sync

        actions = _make_spadl_actions()
        frames = _make_tracking_frames()
        result = add_elastic_sync(actions, frames)
        assert len(result) == len(actions)

    def test_confidence_bounded(self):
        """elastic_confidence is in [0, 1] or NaN."""
        from silly_kicks.tracking.features import add_elastic_sync

        actions = _make_spadl_actions()
        frames = _make_tracking_frames()
        result = add_elastic_sync(actions, frames)
        vals = result["elastic_confidence"].dropna()
        if len(vals) > 0:
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0


class TestElasticSyncXfns:
    def test_column_count(self):
        """elastic_sync_xfns produces 2 lifted xfns (6 VAEP columns)."""
        from silly_kicks.tracking.features import elastic_sync_xfns

        xfns = elastic_sync_xfns()
        assert len(xfns) == 2

    def test_introspection_nan(self):
        """xfns produce NaN in introspection mode (frames=None)."""
        from silly_kicks.tracking.features import elastic_sync_xfns

        xfns = elastic_sync_xfns()
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [0],
                "period_id": [1],
                "time_seconds": [10.0],
                "team_id": [1],
                "player_id": ["p1_0"],
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
