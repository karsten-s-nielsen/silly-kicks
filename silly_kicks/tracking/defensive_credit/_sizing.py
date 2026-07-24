"""Sizing ports: xG per shot (injected column) + extinguished xT (injected fitted surface)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def xg_of_shot(shot_action: pd.Series, *, xg_column: str) -> float:
    """Return the injected per-shot xG. Fail-loud if the column is absent (xtgk/_xg_reward idiom).

    xG is a *pre-block* quantity, so a blocked shot still carries a value (shot_block sizing needs it).
    A present-but-NaN xG passes through as NaN -> a fired-but-unsizable long-form row.
    """
    if xg_column not in shot_action.index:
        raise ValueError(
            f"xg_column {xg_column!r} not found on the shot action. Supply a calibrated per-shot xG "
            f"column (silly-kicks ships no xG model; see spec section 7)."
        )
    return float(shot_action[xg_column])


def extinguished_xt(points, xt) -> np.ndarray:
    """xT at each action-LTR (x, y) point on the injected fitted surface (the threat extinguished).

    ``points``: iterable of (x, y) in action-LTR metres (attacked goal at x=105).
    NaN coords -> NaN value (values_at_points is NaN-tolerant).
    """
    # Lazy import: a module-level `from silly_kicks.xthreat import ...` closes the
    # xthreat -> tracking -> defensive_credit -> xthreat import cycle (xthreat imports
    # tracking.direction, which runs tracking/__init__). See tests/test_no_import_cycles.py.
    from silly_kicks.xthreat import require_fitted_xt, values_at_points

    require_fitted_xt(xt, caller="defensive_credit.extinguished_xt")
    if len(points) == 0:
        return np.array([], dtype="float64")
    xs = np.asarray([p[0] for p in points], dtype="float64")
    ys = np.asarray([p[1] for p in points], dtype="float64")
    return values_at_points(xt, xs, ys)
