"""Atomic mirror of test_pre_window_features.py -- verify atomic public API works."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.tracking.features import (
    actor_arc_length_pre_window,
    actor_displacement_pre_window,
    add_actor_pre_window,
)


def test_atomic_arc_length_constant_velocity() -> None:
    """Same shape as standard test, atomic schema (x, y instead of start_x, start_y)."""
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [100.0],
            "player_id": [10],
            "x": [50.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
        }
    )
    n_frames = 13
    frames = pd.DataFrame(
        [
            {
                "frame_id": i,
                "period_id": 1,
                "time_seconds": 100.0 + dt,
                "player_id": 10,
                "is_ball": False,
                "x": 50.0 + 5.0 * (dt + 0.5),
                "y": 34.0,
            }
            for i, dt in enumerate(np.linspace(-0.5, 0.0, n_frames))
        ]
    )
    out = actor_arc_length_pre_window(actions, frames)
    assert out.iloc[0] == pytest.approx(2.5, rel=1e-6)


def test_atomic_displacement_circular() -> None:
    n_frames = 13
    angles = np.linspace(0, math.pi, n_frames)
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [100.0],
            "player_id": [10],
            "x": [50.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
        }
    )
    frames = pd.DataFrame(
        [
            {
                "frame_id": i,
                "period_id": 1,
                "time_seconds": 100.0 + dt,
                "player_id": 10,
                "is_ball": False,
                "x": 50.0 + 2.0 * math.cos(a),
                "y": 34.0 + 2.0 * math.sin(a),
            }
            for i, (a, dt) in enumerate(zip(angles, np.linspace(-0.5, 0.0, n_frames), strict=False))
        ]
    )
    out = actor_displacement_pre_window(actions, frames)
    assert out.iloc[0] == pytest.approx(4.0, rel=1e-6)


def test_atomic_aggregator_includes_provenance() -> None:
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [100.0],
            "player_id": [10],
            "x": [50.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
        }
    )
    frames = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "period_id": 1,
                "time_seconds": 99.5,
                "player_id": 10,
                "is_ball": False,
                "x": 50.0,
                "y": 34.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 100.0,
                "player_id": 10,
                "is_ball": False,
                "x": 51.0,
                "y": 34.0,
                "source_provider": "synthetic",
            },
        ]
    )
    out = add_actor_pre_window(actions, frames)
    assert "actor_arc_length_pre_window" in out.columns
    assert "actor_displacement_pre_window" in out.columns
    assert "frame_id" in out.columns
