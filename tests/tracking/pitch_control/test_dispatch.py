"""Tests for pitch control dispatch layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import (
    VoronoiParams,
    compute_pitch_control,
    compute_pitch_control_at_points,
)


def _make_frame(with_velocity=True):
    """Standard 2v2 test frame."""
    rows = [
        {"player_id": 1, "team_id": 10, "x": 30, "y": 34, "is_ball": False, "is_goalkeeper": True},
        {"player_id": 2, "team_id": 10, "x": 50, "y": 50, "is_ball": False, "is_goalkeeper": False},
        {"player_id": 3, "team_id": 20, "x": 70, "y": 34, "is_ball": False, "is_goalkeeper": True},
        {"player_id": 4, "team_id": 20, "x": 80, "y": 20, "is_ball": False, "is_goalkeeper": False},
        {"player_id": np.nan, "team_id": np.nan, "x": 52.5, "y": 34, "is_ball": True, "is_goalkeeper": False},
    ]
    if with_velocity:
        for r in rows:
            r["vx"] = 0.0
            r["vy"] = 0.0
    return pd.DataFrame(rows)


class TestDispatchRouting:
    def test_spearman_default(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, attacking_team_id=10)
        assert s.method == "spearman"

    def test_fernandez_bornn(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, 10, method="fernandez_bornn")
        assert s.method == "fernandez_bornn"

    def test_voronoi(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, 10, method="voronoi")
        assert s.method == "voronoi"

    def test_wrong_params_type_raises(self):
        frame = _make_frame()
        with pytest.raises(TypeError):
            compute_pitch_control(frame, 10, method="spearman", params=VoronoiParams())


class TestVelocityRequirement:
    def test_spearman_without_velocity_raises(self):
        frame = _make_frame(with_velocity=False)
        with pytest.raises(ValueError, match="requires velocity"):
            compute_pitch_control(frame, 10, method="spearman")

    def test_fernandez_bornn_without_velocity_raises(self):
        frame = _make_frame(with_velocity=False)
        with pytest.raises(ValueError, match="requires velocity"):
            compute_pitch_control(frame, 10, method="fernandez_bornn")

    def test_voronoi_without_velocity_ok(self):
        frame = _make_frame(with_velocity=False)
        s = compute_pitch_control(frame, 10, method="voronoi")
        assert s.method == "voronoi"


class TestBallPositionInference:
    def test_infers_from_ball_row(self):
        frame = _make_frame()
        # Ball at (52.5, 34) in the frame
        s = compute_pitch_control(frame, 10, method="spearman")
        # Should not crash -- ball position auto-inferred
        assert s.surface.shape == (32, 50)

    def test_explicit_overrides_frame(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, 10, method="spearman", ball_position=(10, 10))
        assert s.surface.shape == (32, 50)


class TestComputeAtPoints:
    def test_batch_point_query(self):
        frame = _make_frame()
        targets = np.array([[30, 34], [70, 34], [52.5, 34]], dtype="float64")
        result = compute_pitch_control_at_points(frame, targets, 10)
        assert result.shape == (3,)
        assert (result >= 0).all() and (result <= 1).all()
        # Attacker near (30, 34), defender near (70, 34)
        assert result[0] > result[1]

    def test_empty_targets(self):
        frame = _make_frame()
        targets = np.empty((0, 2))
        result = compute_pitch_control_at_points(frame, targets, 10)
        assert result.shape == (0,)


class TestOffPitchBall:
    def test_ball_outside_bounds_treated_as_none(self):
        frame = _make_frame()
        # Ball at x=200 (off-pitch per TRACKING_CONSTRAINTS)
        s = compute_pitch_control(frame, 10, ball_position=(200, 34))
        # Should not crash; treated as no ball conditioning
        assert s.surface.shape == (32, 50)
