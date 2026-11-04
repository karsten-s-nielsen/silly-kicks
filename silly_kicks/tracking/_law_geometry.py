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
