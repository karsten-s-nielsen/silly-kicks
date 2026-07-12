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


COORD_COLUMNS = ("start_x", "start_y", "end_x", "end_y")


def flat_zones(x, y, l: int = N, w: int = M) -> np.ndarray:
    """Vectorized flat grid indices for coordinate Series. NaN coords map to ``(0, 0)`` -> zone **176**.

    .. warning::
       The NaN -> ``(0.0, 0.0)`` fallback is a **FIT-PATH contract, NOT a general one.** It is safe
       only because no NaN-coord row ever reaches a fitted surface: ``_moves.py``, ``_xg_reward.py``,
       ``_markov.py`` (support counts), ``_empirical.py`` and ``_turnover.py`` drop them *before*
       calling in, while ``_markov.py``/``_empirical.py``/``_diagnostics.py`` pass NaN rows *through*
       here to assign pressure terciles and drop them immediately afterwards.

       **SCORING callers MUST mask with** :func:`finite_coord_mask` **first.** A scoring caller that
       does not will silently fabricate a real value at zone 176 (the own-corner cell) for every
       NaN-coord row. That defect shipped in 4.40.0-4.45.0 and corrupted ~24% of the xT-GK v2
       GK-distribution domain. See ADR-036 and
       ``docs/superpowers/specs/2026-07-12-xtgk-v2-resolved-origin-design.md``.
    """
    xs = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)
    ys = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0.0)
    return _get_flat_indexes(xs, ys, l, w).to_numpy()


def finite_coord_mask(actions: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """True where ALL of ``start_x``/``start_y``/``end_x``/``end_y`` are finite.

    The blessed pre-filter for any caller that SCORES (as opposed to fits) on the grid -- it is what
    stops :func:`flat_zones` fabricating zone 176 out of a NaN coordinate. See ADR-036.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(
    ...     {"start_x": [5.0, float("nan")], "start_y": [34.0, 34.0],
    ...      "end_x": [40.0, 40.0], "end_y": [34.0, 34.0]}
    ... )
    >>> finite_coord_mask(a).tolist()
    [True, False]
    """
    mask = np.ones(len(actions), dtype=bool)
    for col in COORD_COLUMNS:
        mask &= np.isfinite(pd.to_numeric(actions[col], errors="coerce").to_numpy(dtype=float))
    return mask


def mirror_zone(zone: int, l: int = N, w: int = M) -> int:
    """180-degree point reflection of a flat grid index (column reversal xi->l-1-xi AND row
    reversal yj->w-1-yj). Maps a losing team's origin zone (attack-LTR) to the winning team's
    zone in its OWN attack-LTR frame -- the V_opp mirror (ADR-036 §Part 2)."""
    z = int(zone)
    xi, yj = z % l, z // l
    return (w - 1 - yj) * l + (l - 1 - xi)
