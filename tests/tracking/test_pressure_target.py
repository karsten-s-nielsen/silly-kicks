"""pressure_on_target: the frame-level scalar convenience over pressure_on_actor (TF-56).

Proves pressure_on_target REUSES pressure_on_actor (a synthesized one-row action) rather than
re-implementing the pressure math; add_pressure_on_actor / pressure_on_actor stay untouched (the
existing tests/tracking/test_pressure_*.py remain their guard).
"""

from __future__ import annotations

import math

import numpy as np

from silly_kicks.tracking import pressure_on_actor, pressure_on_target
from tests.tracking._gk_test_helpers import _make_two_team_frame


def _frame():
    # home player_id 10 at (50,34); away player_id 60 tightly at (52,34) -> real pressure on 10.
    frame = _make_two_team_frame(
        home_positions=[(50.0, 34.0)],
        away_positions=[(52.0, 34.0)],
        home_gk_pos=(3.0, 34.0),
        away_gk_pos=(100.0, 34.0),
        away_velocities=[(-3.0, 0.0)],  # presser actively closing on player 10 (bekkers active-press)
    )
    frame["speed"] = np.hypot(frame["vx"].astype(float), frame["vy"].astype(float))  # bekkers_pi reads speed
    return frame


def test_pressure_on_target_returns_a_nonneg_scalar():
    p = pressure_on_target(_frame(), 10, method="bekkers_pi")
    assert isinstance(p, float)
    assert p >= 0.0
    assert p > 0.0  # a defender is 2 m away -> real pressure


def test_pressure_on_target_matches_pressure_on_actor_via_synthesized_action():
    """Byte-identical reuse: the scalar equals pressure_on_actor on the same synthesized action."""
    import pandas as pd

    frame = _frame()
    r = frame[(~frame["is_ball"].astype(bool)) & (frame["player_id"] == 10)].iloc[0]
    synth = pd.DataFrame(
        {
            "game_id": [r["game_id"]],
            "period_id": [r["period_id"]],
            "action_id": [0],
            "time_seconds": [r["time_seconds"]],
            "team_id": [r["team_id"]],
            "player_id": [10],
            "start_x": [float(r["x"])],
            "start_y": [float(r["y"])],
        }
    )
    direct = float(pressure_on_actor(synth, frame, method="bekkers_pi").iloc[0])
    via = pressure_on_target(frame, 10, method="bekkers_pi")
    assert via == direct


def test_pressure_on_target_absent_player_is_nan():
    assert math.isnan(pressure_on_target(_frame(), 9999, method="bekkers_pi"))
