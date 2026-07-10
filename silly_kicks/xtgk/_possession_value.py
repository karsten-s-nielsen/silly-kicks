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


def flat_zones(x, y, l: int = N, w: int = M) -> np.ndarray:
    """Vectorized flat grid indices for coordinate Series, NaN-coord-safe.

    Real cohorts carry NaN start/end coords (e.g. incomplete actions); ``xthreat._get_flat_indexes``
    casts binned coords to int and would raise on NaN. Such rows are dropped downstream (the solve
    ``dropna``s coords; their tercile band is a don't-care) so NaN -> 0 here is harmless.
    """
    xs = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)
    ys = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0.0)
    return _get_flat_indexes(xs, ys, l, w).to_numpy()


def mirror_zone(zone: int, l: int = N, w: int = M) -> int:
    """180-degree point reflection of a flat grid index (column reversal xi->l-1-xi AND row
    reversal yj->w-1-yj). Maps a losing team's origin zone (attack-LTR) to the winning team's
    zone in its OWN attack-LTR frame -- the V_opp mirror (ADR-036 §Part 2)."""
    z = int(zone)
    xi, yj = z % l, z // l
    return (w - 1 - yj) * l + (l - 1 - xi)
