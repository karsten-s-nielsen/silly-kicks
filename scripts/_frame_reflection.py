"""The point reflection of a tracking frame set, for the W5.6 mirror checks (spec 2026-11-08 tracking-batch W5.6, W5.7).

One copy, shared by the ``compute_shape_graphs`` mirror test (plan Task 11) and the role-gate driver
``scripts/validate_shape_graph_roles.py`` (plan Task 11A). It lives under ``scripts/``, inside ``CODE_IDENTITY_PATHS``,
so the driver's code identity covers the mirror arm (TB-PLAN-37).

It does not use :func:`silly_kicks.reflection.reflect`, which stores the result back at the column's dtype: a
float32 ``x`` would round ``105 - x``. Here the reflected ``x`` and ``y`` are float64, and the reflection of a float32
value in float64 is exact, so reflecting twice gives the input values bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Pitch length and width (m) of the SPADL frame convention.
_LENGTH, _WIDTH = 105.0, 68.0

_SWAP = {"ltr": "rtl", "rtl": "ltr"}


def reflect_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """``frames`` with ``x -> 105 - x`` and ``y -> 68 - y`` in float64, and ``team_attacking_direction`` swapped
    (``"ltr"`` <-> ``"rtl"``; any other value, null included, is kept). The input is not mutated.

    Examples
    --------
    >>> import pandas as pd
    >>> from scripts._frame_reflection import reflect_frames
    >>> f = pd.DataFrame({"x": [10.0], "y": [20.0], "team_attacking_direction": ["ltr"]})
    >>> out = reflect_frames(f)
    >>> (float(out["x"].iloc[0]), float(out["y"].iloc[0]), out["team_attacking_direction"].iloc[0])
    (95.0, 48.0, 'rtl')
    """
    out = frames.copy()
    out["x"] = _LENGTH - frames["x"].to_numpy(dtype=np.float64)
    out["y"] = _WIDTH - frames["y"].to_numpy(dtype=np.float64)
    if "team_attacking_direction" in out.columns:
        direction = frames["team_attacking_direction"]
        out["team_attacking_direction"] = direction.map(lambda v: _SWAP.get(v, v) if isinstance(v, str) else v)
    return out
