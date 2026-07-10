"""Marts-native feature extractor for the rho retention model (ADR-036 §Part 3).

Sourced from the gold action marts (fct_action_values geometry/type + fct_action_context pressure),
NOT tracking frames (deprecated as an active source). 8 features: pass geometry + release pressure +
restart-type flags. The frames-only receiver-density feature is intentionally absent (unavailable in
the sanctioned marts). ONE code path at train and serve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]

RETENTION_FEATURE_NAMES = [
    "length",
    "forwardness",
    "dy_abs",
    "dest_x",
    "dest_y_off",
    "release_pressure",
    "is_goalkick",
    "is_throw_in",
]


def extract_retention_features(actions: pd.DataFrame, *, pressure_column: str = "pressure") -> pd.DataFrame:
    """8 mart-derived features from an attack-LTR SPADL action frame carrying start/end coords,
    ``type_id``, and a pressure column. NaN-coord-tolerant (geometry rows drop downstream)."""
    ox = pd.to_numeric(actions["start_x"], errors="coerce").to_numpy(float)
    oy = pd.to_numeric(actions["start_y"], errors="coerce").to_numpy(float)
    dxx = pd.to_numeric(actions["end_x"], errors="coerce").to_numpy(float)
    dyy = pd.to_numeric(actions["end_y"], errors="coerce").to_numpy(float)
    dx, dy = dxx - ox, dyy - oy
    length = np.hypot(dx, dy)
    tid = actions["type_id"].to_numpy()
    release = (
        pd.to_numeric(actions[pressure_column], errors="coerce").to_numpy(float)
        if pressure_column in actions.columns
        else np.full(len(actions), np.nan)
    )
    return pd.DataFrame(
        {
            "length": length,
            "forwardness": np.divide(dx, length, out=np.zeros_like(dx), where=length > 0),
            "dy_abs": np.abs(dy),
            "dest_x": dxx,
            "dest_y_off": np.abs(dyy - spadlconfig.field_width / 2),
            "release_pressure": release,
            "is_goalkick": (tid == _GOALKICK).astype(float),
            "is_throw_in": (tid == _THROW_IN).astype(float),
        },
        index=actions.index,
    )
