"""identify_corners -- provider-aware corner anchoring (TF-46 Arm 1, spec §7).

Anchors every corner on its SPADL ``corner_crossed`` / ``corner_short`` action (present for every
provider -- the converters map a corner restart to one of these). ``corner_short`` -> a short corner;
``corner_crossed`` -> a direct corner. ``type_source`` travels from the action. A derived (events-only)
corner carries a NaN start, so its ``t_kick`` / ``kick_x`` / ``kick_y`` / ``corner_side`` are NaN (spec
section 9.4). The legacy ``gi=`` refinement is deprecated and ignored (decision B1): GI ``start_type`` is
the receiver's possession, not the taker's.

PURE -- never mutates ``actions``. Returns the per-``(game_id, corner_id)`` identification table.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._columns import CORNER_IDENTIFY_COLUMNS
from ._config import CornerWindowParams

_DEFAULT_WINDOW_PARAMS = CornerWindowParams()

_CORNER_CROSSED_ID = spadlconfig.actiontype_id["corner_crossed"]
_CORNER_SHORT_ID = spadlconfig.actiontype_id["corner_short"]
_CORNER_TYPE_IDS = frozenset({_CORNER_CROSSED_ID, _CORNER_SHORT_ID})

_REQUIRED = ("game_id", "period_id", "time_seconds", "team_id", "start_x", "start_y", "type_id")

_SOURCE_BY_PROVIDER = {
    "skillcorner": "skillcorner_marker",  # the interruption MARKER, not the GI reception (decision C7)
    "statsbomb": "sb360_event",
    "sb360": "sb360_event",
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in CORNER_IDENTIFY_COLUMNS.items()})


def identify_corners(
    actions: pd.DataFrame,
    *,
    provider: str,
    params: CornerWindowParams = _DEFAULT_WINDOW_PARAMS,
    gi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-corner identification table at grain ``(game_id, corner_id)``.

    Parameters
    ----------
    actions: SPADL actions (one or more games).
    provider: the source provider; sets ``corner_source``.
    params: window params (only ``short_corner_max_m`` consulted here; timing is in ``add_corner_windows``).
    gi: DEPRECATED and ignored (decision B1); passing it emits ``DeprecationWarning``. GI ``start_type`` is
        the receiver's possession, not the taker's, so it cannot refine a corner. Removed next major.

    Returns
    -------
    A new DataFrame; short/unclear corners are KEPT (``corner_kind``), never silently dropped (ADR-042).

    Examples
    --------
    Anchor the corners in a SPADL action stream::

        from silly_kicks.corners import identify_corners
        corners = identify_corners(actions, provider="sportec")
    """
    missing = [c for c in _REQUIRED if c not in actions.columns]
    if missing:
        raise KeyError(f"identify_corners: actions is missing required column(s) {missing}.")

    is_corner = actions["type_id"].isin(_CORNER_TYPE_IDS)
    if not bool(is_corner.any()):
        return _empty()

    if gi is not None:
        warnings.warn(
            "identify_corners(gi=) is deprecated and ignored: a SkillCorner corner now comes from the "
            "interruption marker, and GI start_type is the receiver's possession, not the taker's, so it "
            "cannot refine a corner (decision B1). The parameter is removed at the next major version.",
            DeprecationWarning,
            stacklevel=2,
        )

    sel = list(_REQUIRED) + (["type_source"] if "type_source" in actions.columns else [])
    c = actions.loc[is_corner, sel].copy()
    c = c.sort_values(["game_id", "period_id", "time_seconds"], kind="stable")

    start_x = c["start_x"].astype("float64").to_numpy()
    start_y = c["start_y"].astype("float64").to_numpy()
    # A derived (events-only) restart carries a NaN start; its kick geometry, side and time are unknown,
    # so t_kick is NaN -- never the ordering placeholder (spec section 9.4).
    derived = np.isnan(start_x)

    out = pd.DataFrame(index=c.index)
    out["game_id"] = c["game_id"].to_numpy()
    out["corner_id"] = c.groupby("game_id", sort=False).cumcount().to_numpy()
    out["period_id"] = c["period_id"].to_numpy()
    out["t_kick"] = np.where(derived, np.nan, c["time_seconds"].astype("float64").to_numpy())
    out["attacking_team_id"] = c["team_id"].to_numpy()

    midline = float(spadlconfig.field_width) / 2.0
    # NaN start_y -> NaN side on every provider (was "right" without warning at the old :91).
    side = np.where(start_y >= midline, "left", "right").astype(object)
    side[np.isnan(start_y)] = None
    out["corner_side"] = side
    out["kick_x"] = start_x
    out["kick_y"] = start_y

    kind = np.where(c["type_id"].to_numpy() == _CORNER_SHORT_ID, "short", "direct")
    out["corner_kind"] = kind
    out["corner_source"] = _SOURCE_BY_PROVIDER.get(provider, "spadl_corner")
    out["type_source"] = c["type_source"].to_numpy() if "type_source" in c.columns else None

    out = out.reset_index(drop=True)
    return out.reindex(columns=list(CORNER_IDENTIFY_COLUMNS)).astype(CORNER_IDENTIFY_COLUMNS)
