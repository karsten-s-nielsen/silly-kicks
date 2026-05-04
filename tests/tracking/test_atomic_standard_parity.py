"""Atomic <-> standard parity gate per spec section 8.8 / lakehouse review item 6."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.atomic.tracking.features import (
    actor_arc_length_pre_window as atomic_arc,
)
from silly_kicks.atomic.tracking.features import (
    actor_displacement_pre_window as atomic_disp,
)
from silly_kicks.atomic.tracking.features import (
    pressure_on_actor as atomic_pressure,
)
from silly_kicks.tracking.features import (
    actor_arc_length_pre_window as std_arc,
)
from silly_kicks.tracking.features import (
    actor_displacement_pre_window as std_disp,
)
from silly_kicks.tracking.features import (
    pressure_on_actor as std_pressure,
)

from .test_pressure_snapshot import _build_fixture


def _to_atomic(actions):
    """Map standard-shape actions (start_x, start_y) -> atomic-shape (x, y, dx, dy)."""
    out = actions.copy()
    out["x"] = out["start_x"]
    out["y"] = out["start_y"]
    out["dx"] = 0.0
    out["dy"] = 0.0
    return out.drop(columns=["start_x", "start_y"], errors="ignore")


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_atomic_standard_parity_pressure(method: str) -> None:
    actions, frames = _build_fixture()
    atomic = _to_atomic(actions)
    std_result = std_pressure(actions, frames, method=method)
    atomic_result = atomic_pressure(atomic, frames, method=method)
    np.testing.assert_array_equal(
        std_result.fillna(-99999).values,
        atomic_result.fillna(-99999).values,
    )


def test_atomic_standard_parity_arc_length() -> None:
    actions, frames = _build_fixture()
    atomic = _to_atomic(actions)
    std_result = std_arc(actions, frames)
    atomic_result = atomic_arc(atomic, frames)
    np.testing.assert_array_equal(
        std_result.fillna(-99999).values,
        atomic_result.fillna(-99999).values,
    )


def test_atomic_standard_parity_displacement() -> None:
    actions, frames = _build_fixture()
    atomic = _to_atomic(actions)
    std_result = std_disp(actions, frames)
    atomic_result = atomic_disp(atomic, frames)
    np.testing.assert_array_equal(
        std_result.fillna(-99999).values,
        atomic_result.fillna(-99999).values,
    )
