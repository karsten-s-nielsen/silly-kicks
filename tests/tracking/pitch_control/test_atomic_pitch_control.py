"""Tests for atomic-SPADL pitch control integration."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from silly_kicks.atomic.tracking.features import (
    add_pitch_control,
    atomic_pitch_control_default_xfns,
    atomic_pitch_control_xfns,
    off_ball_run_value_xfns,
    pitch_control_at_target,
)
from silly_kicks.tracking.pitch_control import PitchControlCache


def _make_atomic_actions():
    """Minimal atomic actions (uses x, y not start_x, start_y)."""
    return pd.DataFrame(
        {
            "action_id": [1, 2],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "team_id": [10, 10],
            "player_id": [1, 2],
            "x": [30.0, 50.0],
            "y": [34.0, 34.0],
            "dx": [20.0, 20.0],
            "dy": [0.0, 0.0],
            "type_id": [0, 0],
        }
    )


def _make_frames():
    """Minimal frames DataFrame matching action timestamps."""
    rows = []
    for t in [10.0, 20.0]:
        for pid, tid, x, y in [(1, 10, 30, 34), (2, 10, 50, 50), (3, 20, 70, 34), (4, 20, 80, 20)]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": int(t * 25),
                    "time_seconds": t,
                    "frame_rate": 25.0,
                    "player_id": pid,
                    "team_id": tid,
                    "x": x,
                    "y": y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": pid in (1, 3),
                    "speed": 0.0,
                    "speed_source": "derived",
                    "z": np.nan,
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": np.nan,
                    "visibility": np.nan,
                    "source_provider": "sportec",
                    "is_goalkeeper_source": "native",
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": int(t * 25),
                "time_seconds": t,
                "frame_rate": 25.0,
                "player_id": np.nan,
                "team_id": np.nan,
                "x": 52.5,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "is_goalkeeper": False,
                "speed": 0.0,
                "speed_source": "derived",
                "z": np.nan,
                "ball_state": "alive",
                "team_attacking_direction": np.nan,
                "confidence": np.nan,
                "visibility": np.nan,
                "source_provider": "sportec",
                "is_goalkeeper_source": np.nan,
            }
        )
    return pd.DataFrame(rows)


class TestAtomicPitchControlAtAction:
    def test_returns_series_with_correct_name(self):
        actions = _make_atomic_actions()
        frames = _make_frames()
        result = pitch_control_at_target(actions, frames)
        assert isinstance(result, pd.Series)
        assert result.name == "pitch_control_at_target__spearman"
        assert len(result) == 2

    def test_values_in_bounds(self):
        actions = _make_atomic_actions()
        frames = _make_frames()
        result = pitch_control_at_target(actions, frames)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_introspection_mode(self):
        actions = _make_atomic_actions()
        result = pitch_control_at_target(actions, None)
        assert result.isna().all()

    def test_method_kwarg(self):
        actions = _make_atomic_actions()
        frames = _make_frames()
        result = pitch_control_at_target(actions, frames, method="voronoi")
        assert result.name == "pitch_control_at_target__voronoi"


class TestAtomicAddPitchControl:
    def test_adds_column(self):
        actions = _make_atomic_actions()
        frames = _make_frames()
        result = add_pitch_control(actions, frames)
        assert "pitch_control_at_target__spearman" in result.columns


class TestAtomicXfnFactory:
    def test_default_xfns_is_list(self):
        assert isinstance(atomic_pitch_control_default_xfns, list)
        assert len(atomic_pitch_control_default_xfns) == 1

    def test_xfn_has_frame_aware_marker(self):
        xfn = atomic_pitch_control_xfns("spearman")[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_all_methods_produce_xfns(self):
        for method in ("spearman", "fernandez_bornn", "voronoi"):
            xfns = atomic_pitch_control_xfns(method)
            assert len(xfns) == 1


class TestAtomicPitchControlCache:
    """TF-7 xfns cache -- atomic mirror threading (signature + byte-identity)."""

    def test_atomic_pitch_control_xfns_accepts_cache(self):
        assert "pitch_control_cache" in inspect.signature(atomic_pitch_control_xfns).parameters

    def test_atomic_pitch_control_at_target_accepts_cache(self):
        assert "pitch_control_cache" in inspect.signature(pitch_control_at_target).parameters

    def test_atomic_off_ball_run_value_xfns_accepts_cache(self):
        assert "pitch_control_cache" in inspect.signature(off_ball_run_value_xfns).parameters

    def test_atomic_pc_xfns_shared_cache_byte_identical_to_none(self):
        actions, frames = _make_atomic_actions(), _make_frames()
        gs = [actions]
        none_out = atomic_pitch_control_xfns("voronoi")[0](gs, frames)
        cached_out = atomic_pitch_control_xfns("voronoi", pitch_control_cache=PitchControlCache())[0](gs, frames)
        pd.testing.assert_frame_equal(none_out, cached_out)
        # Non-vacuity: the atomic PC path actually produced values, so the cache= threading was
        # genuinely exercised (not an all-NaN no-op).
        assert any(none_out[c].notna().any() for c in none_out.columns)
