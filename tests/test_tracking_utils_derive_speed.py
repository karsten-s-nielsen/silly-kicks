"""Unit tests for silly_kicks.tracking.utils._derive_speed."""

import numpy as np
import pandas as pd

from silly_kicks.tracking.utils import _derive_speed


def _uniform_motion_frames(
    speed_mps: float,
    n_frames: int,
    frame_rate: float,
    period_id: int = 1,
) -> pd.DataFrame:
    """Build long-form frames where one player moves at uniform speed_mps along x."""
    dt = 1.0 / frame_rate
    rows = []
    for i in range(n_frames):
        rows.append(
            {
                "game_id": 1,
                "period_id": period_id,
                "frame_id": i,
                "time_seconds": i * dt,
                "frame_rate": frame_rate,
                "player_id": 7,
                "team_id": 100,
                "is_ball": False,
                "is_goalkeeper": False,
                "x": 10.0 + speed_mps * i * dt,
                "y": 34.0,
                "z": float("nan"),
                "speed": float("nan"),
                "speed_source": None,
                "ball_state": "alive",
                "team_attacking_direction": "ltr",
                "confidence": None,
                "visibility": None,
                "source_provider": "metrica",
            }
        )
    return pd.DataFrame(rows)


def test_derive_speed_uniform_motion_one_mps():
    frames = _uniform_motion_frames(speed_mps=1.0, n_frames=10, frame_rate=25.0)
    out = _derive_speed(frames)
    assert pd.isna(out.iloc[0]["speed"])
    np.testing.assert_allclose(out.iloc[1:]["speed"].to_numpy(), 1.0, atol=1e-6)
    assert (out.iloc[1:]["speed_source"] == "derived").all()
    assert pd.isna(out.iloc[0]["speed_source"]) or out.iloc[0]["speed_source"] is None


def test_derive_speed_period_boundary_no_leakage():
    p1 = _uniform_motion_frames(speed_mps=1.0, n_frames=5, frame_rate=25.0, period_id=1)
    p2 = _uniform_motion_frames(speed_mps=1.0, n_frames=5, frame_rate=25.0, period_id=2)
    frames = pd.concat([p1, p2], ignore_index=True)
    out = _derive_speed(frames)
    p1_first = out[(out.period_id == 1) & (out.frame_id == 0)]
    p2_first = out[(out.period_id == 2) & (out.frame_id == 0)]
    assert pd.isna(p1_first.iloc[0]["speed"])
    assert pd.isna(p2_first.iloc[0]["speed"])


def test_derive_speed_ball_treated_as_one_entity():
    rows = []
    for i in range(5):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": i,
                "time_seconds": i / 25.0,
                "frame_rate": 25.0,
                "player_id": pd.NA,
                "team_id": pd.NA,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": 50.0 + 2.0 * i / 25.0,
                "y": 34.0,
                "z": 0.5,
                "speed": float("nan"),
                "speed_source": None,
                "ball_state": "alive",
                "team_attacking_direction": None,
                "confidence": None,
                "visibility": None,
                "source_provider": "metrica",
            }
        )
    frames = pd.DataFrame(rows)
    out = _derive_speed(frames)
    assert pd.isna(out.iloc[0]["speed"])
    np.testing.assert_allclose(out.iloc[1:]["speed"].to_numpy(), 2.0, atol=1e-6)


def test_derive_speed_preserves_native_when_present():
    """If a row already has speed populated, _derive_speed should not overwrite it."""
    frames = _uniform_motion_frames(speed_mps=1.0, n_frames=5, frame_rate=25.0)
    frames.loc[2, "speed"] = 99.9
    frames.loc[2, "speed_source"] = "native"
    out = _derive_speed(frames).sort_values(["frame_id"]).reset_index(drop=True)
    assert out.iloc[2]["speed"] == 99.9
    assert out.iloc[2]["speed_source"] == "native"
