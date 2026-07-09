"""PossessionValue port + shared value types (ADR-036 §3, §7, §9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

PressureLevel = Literal[1, 2, 3]


@dataclass(frozen=True)
class State:
    zone: int
    pressure_level: PressureLevel


@dataclass(frozen=True)
class DeltaV:
    delta: float
    pressure_component: float
    position_component: float


@runtime_checkable
class PossessionValue(Protocol):
    def value(self, zone: int, p: PressureLevel) -> float: ...
    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]: ...
    def delta_v(self, s: State, s_next: State) -> DeltaV: ...


def zone_of(x: float, y: float, l: int = N, w: int = M) -> int:
    """Flat grid index for a coordinate, matching value_iteration's ravel()."""
    return int(_get_flat_indexes(pd.Series([float(x)]), pd.Series([float(y)]), l, w).iloc[0])
