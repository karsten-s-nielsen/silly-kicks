"""StatsBomb 120x80 -> SPADL affine, WITHOUT the clip.

Split for the reason ADR-038 already split SkillCorner's: ``_scale_to_spadl`` is affine only and
``_transform_coords`` is scale + clamp, because a clamp that is safe for EVENTS (on-pitch by
construction) is destructive for anything else. A ``visible_area`` polygon legitimately extends
past the touchline -- the broadcast camera sees beyond it -- so clipping would silently shrink the
observed region, which is the entire quantity that column carries.

``_convert_locations`` in ``statsbomb.py`` remains the per-ROW wrapper over this, and keeps BOTH
event-only behaviours: the clip, and the 3-element (x, y, z) shot form whose ``y_offset`` is 0.05
rather than the cell-centre correction.

**Do NOT call the row wrapper on a flat polygon.** StatsBomb ships ``visible_area`` as a flat
``[x1, y1, x2, y2, ...]`` list, which satisfies ``_convert_locations``' ``len >= 2`` guard and
yields only the FIRST vertex -- measured, a 4-vertex polygon returns shape ``(1, 2)`` with no error
and no NaN. Reshape to ``(N, 2)`` and call :func:`sb_xy_to_spadl` directly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import config as spadlconfig

#: StatsBomb's pitch is a fixed 120x80 cell grid regardless of the true pitch dimensions.
SB_FIELD_LENGTH = 120.0
SB_FIELD_WIDTH = 80.0


def cell_side(fidelity_version: int) -> float:
    """Cell edge length for a ``xy_fidelity_version``.

    StatsBomb coordinates are CELL-based: at fidelity 1, ``(1, 1)`` is the top-left square yard.
    Fidelity 2 records a finer grid, so the cell -- and therefore the centre correction -- is a
    tenth the size.
    """
    return 0.1 if fidelity_version == 2 else 1.0


def sb_xy_to_spadl(
    xy: npt.NDArray[np.float64],
    *,
    fidelity_version: int,
    y_offset: npt.NDArray[np.float64] | float | None = None,
) -> npt.NDArray[np.float64]:
    """``(N, 2)`` StatsBomb cell coordinates -> ``(N, 2)`` SPADL. No clipping.

    Parameters
    ----------
    xy : ndarray, shape (N, 2)
        StatsBomb coordinates in the 120x80 grid.
    fidelity_version : int
        ``xy_fidelity_version`` from the match metadata; selects the cell size.
    y_offset : ndarray | float | None, optional
        Per-row y correction. Defaults to the cell-centre correction
        (``cell_side / 2``). ``_convert_locations`` passes 0.05 for 3-element shot
        locations; that is EVENT semantics and must not be applied to polygon vertices.

    Returns
    -------
    ndarray, shape (N, 2)
        SPADL coordinates, y inverted, NOT clipped to the pitch.

    Examples
    --------
    Convert a pair of freeze-frame player locations::

        import numpy as np
        from silly_kicks.spadl._sb_coordinates import sb_xy_to_spadl
        xy = np.array([[60.0, 40.0], [119.0, 79.0]])
        spadl_xy = sb_xy_to_spadl(xy, fidelity_version=1)
    """
    crc = cell_side(fidelity_version) / 2
    if y_offset is None:
        y_offset = crc
    out = np.empty_like(xy, dtype=float)
    out[:, 0] = (xy[:, 0] - crc) / SB_FIELD_LENGTH * spadlconfig.field_length
    out[:, 1] = spadlconfig.field_width - (xy[:, 1] - y_offset) / SB_FIELD_WIDTH * spadlconfig.field_width
    return out
