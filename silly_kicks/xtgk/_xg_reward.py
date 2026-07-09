"""Per-cell immediate reward E[xG | shot in cell] from an injected xg_column (ADR-036 §4.1).

The xG analogue of xthreat._grid._scoring_prob: goal COUNTS -> an xG SUM over shots, / shot
counts. NOT goal-gated (that is the v1 degeneracy). Own goals never appear (not the
possessing team's shot xG). Same (w,l) layout as _scoring_prob; access via .ravel()[flat].
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import M, N, _count, _get_flat_indexes, _safe_divide


def _weighted_cell_sum(x: pd.Series, y: pd.Series, w: pd.Series, l: int, ww: int) -> npt.NDArray[np.float64]:
    # Layout MUST match xthreat._grid._count exactly: reshape the flat-indexed vector to
    # (ww, l) with NO extra flip (the y-flip already lives in _get_flat_indexes).
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(w))
    x, y, w = x[mask], y[mask], w[mask]
    flat = _get_flat_indexes(x, y, l, ww).to_numpy()
    out = np.zeros(ww * l, dtype=np.float64)
    np.add.at(out, flat, w.to_numpy(dtype=np.float64))
    return out.reshape((ww, l))


def xg_scoring_prob(actions: pd.DataFrame, *, xg_column: str, l: int = N, w: int = M) -> npt.NDArray[np.float64]:
    """E[xG|shot] per grid cell, shape (w, l), same layout as xthreat _scoring_prob."""
    if xg_column not in actions.columns:
        raise ValueError(
            f"xg_column {xg_column!r} not found. Supply a calibrated per-shot xG column "
            f"(silly-kicks ships no xG model; see ADR-036 §6)."
        )
    shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]].dropna(subset=["start_x", "start_y"])
    shotmatrix = _count(shots.start_x, shots.start_y, l, w)
    xgsum = _weighted_cell_sum(shots.start_x, shots.start_y, shots[xg_column], l, w)
    return _safe_divide(xgsum, shotmatrix)
