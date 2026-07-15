"""The events transform CLAMPS; tracking must never inherit that (spec 3.4).

Measured on real data: routing tracking through the clamping transform snaps 11.31% of ball
rows and 0.71% of player rows, by up to 9.00 m -- and turns a ball nine metres behind the goal
into a ball on the goal line, erasing goal-vs-save.
"""

import numpy as np
import pandas as pd

from silly_kicks.spadl.skillcorner import _scale_to_spadl, _transform_coords


def test_scale_is_affine_and_never_clamps():
    # raw centre-origin metres on a 104 x 68 pitch, including legitimately OFF-PITCH points:
    # a ball 9 m behind the goal line, a keeper behind his line, a ball past the touchline.
    x = pd.Series([-52.0, 0.0, 52.0, 61.0, -55.0])
    y = pd.Series([-34.0, 0.0, 34.0, 10.0, 40.0])
    sx, sy = _scale_to_spadl(x, y, 104.0, 68.0)

    assert sx.iloc[0] == 0.0  # goal line
    assert sx.iloc[1] == 52.5  # centre spot
    assert sx.iloc[2] == 105.0  # far goal line
    assert sx.iloc[3] > 105.0  # 9 m BEYOND the goal -- must survive
    assert sx.iloc[4] < 0.0  # behind the other goal -- must survive
    assert sy.iloc[4] > 68.0  # past the touchline -- must survive


def test_transform_coords_still_clamps_for_events():
    """The events converter's behaviour is UNCHANGED -- an action is on-pitch by construction."""
    x = pd.Series([61.0, -55.0])
    y = pd.Series([10.0, 40.0])
    cx, cy = _transform_coords(x, y, 104.0, 68.0)
    assert cx.iloc[0] == 105.0  # clamped
    assert cx.iloc[1] == 0.0  # clamped
    assert cy.iloc[1] == 68.0  # clamped


def test_transform_coords_equals_scale_then_clamp():
    """_transform_coords must be exactly _scale_to_spadl + clamp -- one truth, not two."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(-60, 60, 500))
    y = pd.Series(rng.uniform(-40, 40, 500))
    sx, sy = _scale_to_spadl(x, y, 103.0, 67.0)
    cx, cy = _transform_coords(x, y, 103.0, 67.0)
    pd.testing.assert_series_equal(cx, sx.clip(0.0, 105.0))
    pd.testing.assert_series_equal(cy, sy.clip(0.0, 68.0))
