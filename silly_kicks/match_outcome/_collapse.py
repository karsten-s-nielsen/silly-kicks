"""Rung-3b same-possession collapse (TF-53 spec §4).

A save->rebound->goal is ONE opportunity, not several. ``collapse_team_xgs`` groups a team's shots by
``possession_id`` (added upstream by ``spadl.add_possessions``) and combines each possession's shots
into a single Bernoulli ``P(>=1 goal) = 1 - prod(1 - xg)``. On an all-distinct-possession shot set the
result is identical to the independent path (each possession has one shot). NaN-xg shots are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_team_xgs(shots: pd.DataFrame, *, xg_column: str) -> np.ndarray:
    """Per-possession combined scoring probabilities for one team's shots.

    Requires a ``possession_id`` column; without it each shot is treated as its own possession
    (i.e. the independent path). Returns one probability per possession that contains a finite-xg shot.
    """
    if xg_column not in shots.columns:
        return np.array([], dtype="float64")
    df = shots[shots[xg_column].notna()]
    if df.empty:
        return np.array([], dtype="float64")
    if "possession_id" not in df.columns:
        return df[xg_column].to_numpy(dtype="float64")
    combined = (
        df.groupby("possession_id", sort=False)[xg_column]
        .apply(lambda s: 1.0 - float(np.prod(1.0 - s.to_numpy(dtype="float64"))))
        .to_numpy(dtype="float64")
    )
    return combined
