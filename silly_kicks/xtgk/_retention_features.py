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
from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

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


#: The subset of RETENTION_FEATURE_NAMES that is pure arithmetic on start/end coords. The metric's
#: coordinate-coherence check recomputes EXACTLY these (ADR-036 amendment) -- single-sourced here so
#: the check and the trainer can never drift.
COORD_DERIVED_NAMES = ["length", "forwardness", "dy_abs", "dest_x", "dest_y_off"]


def _coord_derived(actions: pd.DataFrame) -> pd.DataFrame:
    """The five coordinate-derived retention features. RAW arithmetic -- standardisation happens
    inside ``GkRetentionModel``, which is what makes them directly comparable to the coordinates."""
    ox = pd.to_numeric(actions["start_x"], errors="coerce").to_numpy(float)
    oy = pd.to_numeric(actions["start_y"], errors="coerce").to_numpy(float)
    dxx = pd.to_numeric(actions["end_x"], errors="coerce").to_numpy(float)
    dyy = pd.to_numeric(actions["end_y"], errors="coerce").to_numpy(float)
    dx, dy = dxx - ox, dyy - oy
    length = np.hypot(dx, dy)
    return pd.DataFrame(
        {
            "length": length,
            "forwardness": np.divide(dx, length, out=np.zeros_like(dx), where=length > 0),
            "dy_abs": np.abs(dy),
            "dest_x": dxx,
            "dest_y_off": np.abs(dyy - spadlconfig.field_width / 2),
        },
        index=actions.index,
    )


def extract_retention_features(actions: pd.DataFrame, *, pressure_column: str = "pressure") -> pd.DataFrame:
    """8 mart-derived features from an attack-LTR SPADL action frame carrying start/end coords,
    ``type_id``, and a pressure column. NaN-coord-tolerant (geometry rows drop downstream).

    When ``actions`` carries the ``gk_geometry_source`` stamp (i.e. it came through
    :func:`~silly_kicks.xtgk.apply_resolved_gk_geometry`) the stamp is **passed through** as a
    non-feature column. This is inert to the model -- ``GkRetentionModel.fit``/``predict_proba``
    both select ``features[self.feature_names]`` -- and lets ``compute_xt_gk_v2`` attest that the
    features and the actions came from the same resolved frame.
    """
    out = _coord_derived(actions)
    tid = actions["type_id"].to_numpy()
    out["release_pressure"] = (
        pd.to_numeric(actions[pressure_column], errors="coerce").to_numpy(float)
        if pressure_column in actions.columns
        else np.full(len(actions), np.nan)
    )
    out["is_goalkick"] = (tid == _GOALKICK).astype(float)
    out["is_throw_in"] = (tid == _THROW_IN).astype(float)
    if GK_GEOMETRY_SOURCE_COLUMN in actions.columns:
        out[GK_GEOMETRY_SOURCE_COLUMN] = actions[GK_GEOMETRY_SOURCE_COLUMN].to_numpy()
    return out
