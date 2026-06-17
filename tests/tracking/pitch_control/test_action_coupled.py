"""Tests for pitch control VAEP integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import (
    add_pitch_control,
    pitch_control_at_target,
    pitch_control_default_xfns,
    pitch_control_xfns,
)


def _make_actions():
    """Minimal actions DataFrame."""
    return pd.DataFrame(
        {
            "action_id": [1, 2],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "team_id": [10, 10],
            "player_id": [1, 2],
            "start_x": [30.0, 50.0],
            "start_y": [34.0, 34.0],
            "end_x": [50.0, 70.0],
            "end_y": [34.0, 34.0],
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
        # Ball row
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


class TestPitchControlAtAction:
    def test_returns_series(self):
        actions = _make_actions()
        frames = _make_frames()
        result = pitch_control_at_target(actions, frames)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_values_in_bounds(self):
        actions = _make_actions()
        frames = _make_frames()
        result = pitch_control_at_target(actions, frames)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


class TestAddPitchControl:
    def test_adds_column(self):
        actions = _make_actions()
        frames = _make_frames()
        result = add_pitch_control(actions, frames)
        assert "pitch_control_at_target__spearman" in result.columns


class TestXfnFactory:
    def test_default_xfns_is_list(self):
        assert isinstance(pitch_control_default_xfns, list)
        assert len(pitch_control_default_xfns) == 1

    def test_xfn_has_frame_aware_marker(self):
        xfn = pitch_control_xfns("spearman")[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_introspection_mode_no_crash(self):
        """VAEP fit-time introspection: 10-row dummy, frames=None."""
        xfn = pitch_control_xfns("spearman")[0]
        dummy = pd.DataFrame(
            {
                "game_id": range(10),
                "period_id": 1,
                "time_seconds": range(10),
                "team_id": 1,
                "player_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "type_id": 0,
                "result_id": 0,
                "bodypart_id": 0,
                "action_id": range(10),
                "original_event_id": range(10),
                "score_home": 0,
                "score_away": 0,
            }
        )
        states = [dummy] * 3
        result = xfn(states, None)
        assert result.shape[0] == 10
        assert result.isna().all().all()
