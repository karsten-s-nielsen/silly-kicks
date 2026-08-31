"""zero_velocity_if_unavailable + compute_gk_influence are public tracking seams (TF-60 PR2)."""

import pandas as pd
import pytest

import silly_kicks.tracking as tk


def test_zero_velocity_is_exported():
    assert "zero_velocity_if_unavailable" in tk.__all__
    assert callable(tk.zero_velocity_if_unavailable)


def test_compute_gk_influence_is_exported():
    assert "compute_gk_influence" in tk.__all__
    assert callable(tk.compute_gk_influence)
    from silly_kicks.tracking import compute_gk_influence  # must not raise ImportError

    assert compute_gk_influence is tk.compute_gk_influence


def test_present_velocity_is_a_no_op_same_object():
    frame = pd.DataFrame(
        {
            "is_ball": [False],
            "team_id": [1],
            "player_id": [1],
            "x": [10.0],
            "y": [34.0],
            "vx": [0.0],
            "vy": [0.0],
            "is_goalkeeper": [False],
        }
    )
    assert tk.zero_velocity_if_unavailable(frame, method="spearman") is frame


def test_undeclared_missing_velocity_raises():
    frame = pd.DataFrame(
        {
            "is_ball": [False],
            "team_id": [1],
            "player_id": [1],
            "x": [10.0],
            "y": [34.0],
            "is_goalkeeper": [False],  # no vx/vy, no velocity-unavailable marker
        }
    )
    with pytest.raises((ValueError, KeyError)):
        tk.zero_velocity_if_unavailable(frame, method="spearman")
