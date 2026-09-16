"""_orientation: opponent rows reflected into the acting team's frame (TF-52 Task 3, ADR-028/045)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.team_metrics._orientation import reflect_into_team_frame


def test_reflect_is_point_reflection_per_row():
    opp = pd.DataFrame(
        {
            "start_x": [10.0, 90.0],
            "start_y": [5.0, 60.0],
            "end_x": [20.0, 80.0],
            "end_y": [10.0, 50.0],
            "team_id": [20, 20],
        }
    )
    out = reflect_into_team_frame(opp)
    fl, fw = spadlconfig.field_length, spadlconfig.field_width
    # per-row point reflection, NOT an aggregate mean (ADR-045 guard shape)
    assert np.allclose(out["start_x"].to_numpy(), fl - opp["start_x"].to_numpy())
    assert np.allclose(out["start_y"].to_numpy(), fw - opp["start_y"].to_numpy())
    assert np.allclose(out["end_x"].to_numpy(), fl - opp["end_x"].to_numpy())
    assert np.allclose(out["end_y"].to_numpy(), fw - opp["end_y"].to_numpy())
    assert list(out["team_id"]) == [20, 20]  # non-position column untouched


def test_reflect_is_pure():
    opp = pd.DataFrame({"start_x": [10.0], "start_y": [5.0], "end_x": [20.0], "end_y": [10.0]})
    snap = opp.copy(deep=True)
    reflect_into_team_frame(opp)
    pd.testing.assert_frame_equal(opp, snap)
