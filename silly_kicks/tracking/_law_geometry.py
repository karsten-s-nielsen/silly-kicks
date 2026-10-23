"""Law-of-the-Game locations against the SCALED tracking frame (TF-46 Arm 1 Part 2; spec sections 6.4, 12 and the
16.1 table, amended 2026-10-22, C2 r2 C-22 / C-25).

Tracking frames are scaled to the 105 x 68 SPADL frame from the provider's own pitch, but the Laws fix distances in
REAL metres: the penalty spot is 11 m from the goal line, the goal area is 5.5 m deep and 18.32 m wide, the corner arc
has a 1 m radius. A real-metre Law position therefore lands at a different scaled coordinate on a 100 m pitch than on
a 105 m one. This module is the ONE place that maps between the two: a point in the restarting team's action frame
(it attacks +x) is converted to real metres with the provider pitch dimensions, and every Law test runs there.

The pitch dimensions are declared inputs first; else the converter's ``frames.attrs`` stamp
(``pitch_length`` / ``pitch_width``; attrs do not survive every pandas op, so a caller that reshapes frames passes
them); else 105 x 68 (a provider already on the SPADL frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._restart_validation import RATIFIED

_L, _W = float(spadlconfig.field_length), float(spadlconfig.field_width)


@dataclass(frozen=True)
class PitchDims:
    """The provider pitch in real metres (default: the 105 x 68 SPADL frame itself).

    Examples
    --------
    >>> from silly_kicks.tracking._law_geometry import PitchDims
    >>> round(PitchDims(100.0, 64.0).spot_x, 3)  # the 11 m spot in the scaled 105 m action frame
    93.45
    """

    length: float = _L
    width: float = _W

    def to_real(self, ax, ay) -> tuple[np.ndarray, np.ndarray]:
        """Scaled action-frame coordinates -> real metres on this pitch."""
        return (
            np.asarray(ax, dtype=np.float64) * (self.length / _L),
            np.asarray(ay, dtype=np.float64) * (self.width / _W),
        )

    @property
    def spot_x(self) -> float:
        """The penalty spot x in the scaled action frame (11 real metres before the attacked goal line)."""
        return _L - 11.0 * (_L / self.length)


def resolve_pitch_dims(frames: pd.DataFrame | None, pitch_length=None, pitch_width=None) -> PitchDims:
    """Declared keyword > ``frames.attrs`` stamp > 105 x 68, per dimension.

    Examples
    --------
    ::

        dims = resolve_pitch_dims(frames)  # the SkillCorner converter stamps both dimensions
    """
    attrs = frames.attrs if frames is not None else {}
    length = float(pitch_length) if pitch_length is not None else float(attrs.get("pitch_length", _L))
    width = float(pitch_width) if pitch_width is not None else float(attrs.get("pitch_width", _W))
    return PitchDims(length, width)


def law_boundary_distance(base: str, ax: float, ay: float, dims: PitchDims) -> float:
    """The signed real-metre distance of a rest centroid (action frame) to its section 6.4 Law location; <= the
    tolerance means the location holds. Corner: the distance to the nearer attacked-end flag ARC; goal kick: the
    distance OUTSIDE the own goal area (negative inside); penalty: the distance to the spot. NaN for a free kick
    (rest anywhere). The run (1) boundary diagnostic reports exactly this.

    Examples
    --------
    ::

        d = law_boundary_distance("goal_kick", 5.6, 34.0, PitchDims())  # 0.1 m outside the goal area line
    """
    x, y = dims.to_real(ax, ay)
    x, y = float(x), float(y)
    if base == "corner":
        d = min(np.hypot(x - dims.length, y), np.hypot(x - dims.length, y - dims.width))
        return float(abs(d - spadlconfig.corner_arc_radius))
    if base == "goal_kick":
        dx = x - spadlconfig.goal_area_depth
        dy = abs(y - dims.width / 2.0) - spadlconfig.goal_area_half_width
        if dx <= 0.0 and dy <= 0.0:
            return float(max(dx, dy))  # inside: negative, the distance to the nearer line
        return float(np.hypot(max(dx, 0.0), max(dy, 0.0)))
    if base == "penalty":
        return float(np.hypot(x - (dims.length - 11.0), y - dims.width / 2.0))
    return float("nan")


def rest_location_ok(base: str, ax: float, ay: float, dims: PitchDims) -> bool:
    """The section 6.4 rest location in real metres (amended 2026-10-22): corner -- within ``corner_rest_m`` of an
    attacked-end flag's arc; goal kick -- within one ball radius of the own goal area, lines included (Law 1: the
    lines belong to the area; Law 2 radius); penalty -- within ``penalty_spot_r_m`` of the spot; free kick --
    anywhere.

    Examples
    --------
    >>> from silly_kicks.tracking._law_geometry import PitchDims, rest_location_ok
    >>> rest_location_ok("goal_kick", 5.6, 34.0, PitchDims()), rest_location_ok("goal_kick", 5.8, 34.0, PitchDims())
    (True, False)
    """
    d = law_boundary_distance(base, ax, ay, dims)
    if base == "corner":
        return bool(d <= RATIFIED["corner_rest_m"])
    if base == "goal_kick":
        return bool(d <= spadlconfig.ball_radius)
    if base == "penalty":
        return bool(d <= RATIFIED["penalty_spot_r_m"])
    return True
