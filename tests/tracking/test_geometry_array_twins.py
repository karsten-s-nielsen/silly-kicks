"""Array goal-relative transforms == scalar, element-wise (TF-58 C2, Task 5)."""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking._geometry import (
    GEOMETRY_VERSION,
    to_goal_relative_x,
    to_goal_relative_x_array,
    to_goal_relative_y,
    to_goal_relative_y_array,
)


def test_array_twins_equal_scalar_elementwise():
    rng = np.random.default_rng(0)
    for g in (0.0, 105.0):
        x = np.concatenate([rng.uniform(0, 105, 20), [np.nan]])
        y = np.concatenate([rng.uniform(0, 68, 20), [np.nan]])
        np.testing.assert_array_equal(
            to_goal_relative_x_array(x, goal_x=g), np.array([to_goal_relative_x(v, goal_x=g) for v in x])
        )
        np.testing.assert_array_equal(
            to_goal_relative_y_array(y, goal_x=g), np.array([to_goal_relative_y(v, goal_x=g) for v in y])
        )


def test_array_twins_are_point_reflection():
    x = np.array([0.0, 30.0, 105.0])
    y = np.array([0.0, 20.0, 68.0])
    np.testing.assert_allclose(to_goal_relative_x_array(to_goal_relative_x_array(x, goal_x=105.0), goal_x=105.0), x)
    np.testing.assert_allclose(to_goal_relative_y_array(to_goal_relative_y_array(y, goal_x=105.0), goal_x=105.0), y)


def test_array_twins_do_not_mutate_input():
    x = np.array([1.0, 2.0, 3.0])
    before = x.copy()
    to_goal_relative_x_array(x, goal_x=105.0)
    to_goal_relative_x_array(x, goal_x=0.0)
    to_goal_relative_y_array(x, goal_x=105.0)
    to_goal_relative_y_array(x, goal_x=0.0)
    np.testing.assert_array_equal(x, before)


def test_geometry_version_unchanged():
    assert GEOMETRY_VERSION == "goal-relative-2"
