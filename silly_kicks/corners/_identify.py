"""identify_corners -- provider-aware corner anchoring (TF-46 Arm 1, spec §7).

Anchors every corner on its SPADL ``corner_crossed`` / ``corner_short`` action (present for every
provider -- the converters map a ``corner_for`` restart to one of these). ``corner_short`` -> a short
corner; ``corner_crossed`` -> a direct corner. For SkillCorner, an optional ``gi`` frame carrying the
first-possession ``start_type`` after the restart refines the classification: a ``corner_for`` whose
start type is not ``corner_reception`` / ``corner_interception`` is ``unclear`` (spec amendment (a)).

PURE -- never mutates ``actions``. Returns the per-``(game_id, corner_id)`` identification table.
"""

from __future__ import annotations

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

#: GI start types that confirm a direct corner reception (SkillCorner amendment (a)).
_CONFIRMING_START_TYPES = frozenset({"corner_reception", "corner_interception"})

_SOURCE_BY_PROVIDER = {
    "skillcorner": "skillcorner_gi",
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
    provider: the source provider; sets ``corner_source`` and whether ``gi`` refinement applies.
    params: window params (only ``short_corner_max_m`` consulted here; timing is in ``add_corner_windows``).
    gi: optional SkillCorner first-possession tokens with columns ``game_id``, ``period_id``,
        ``time_seconds`` (aligned to the corner action time), ``game_interruption_before``, ``start_type``.

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

    c = actions.loc[is_corner, list(_REQUIRED)].copy()
    c = c.sort_values(["game_id", "period_id", "time_seconds"], kind="stable")

    out = pd.DataFrame(index=c.index)
    out["game_id"] = c["game_id"].to_numpy()
    out["corner_id"] = c.groupby("game_id", sort=False).cumcount().to_numpy()
    out["period_id"] = c["period_id"].to_numpy()
    out["t_kick"] = c["time_seconds"].astype("float64").to_numpy()
    out["attacking_team_id"] = c["team_id"].to_numpy()

    midline = float(spadlconfig.field_width) / 2.0
    out["corner_side"] = np.where(c["start_y"].to_numpy() >= midline, "left", "right")
    out["kick_x"] = c["start_x"].astype("float64").to_numpy()
    out["kick_y"] = c["start_y"].astype("float64").to_numpy()

    kind = np.where(c["type_id"].to_numpy() == _CORNER_SHORT_ID, "short", "direct")
    out["corner_kind"] = kind
    out["corner_source"] = _SOURCE_BY_PROVIDER.get(provider, "spadl_corner")

    if provider == "skillcorner" and gi is not None and len(gi) > 0:
        out = _apply_gi_refinement(out, gi)

    out = out.reset_index(drop=True)
    return out.reindex(columns=list(CORNER_IDENTIFY_COLUMNS)).astype(CORNER_IDENTIFY_COLUMNS)


def _apply_gi_refinement(out: pd.DataFrame, gi: pd.DataFrame) -> pd.DataFrame:
    """Mark a corner ``unclear`` when its GI first-possession start type does not confirm a reception."""
    cols = ["game_id", "period_id", "time_seconds", "game_interruption_before", "start_type"]
    g = gi.loc[:, cols].rename(columns={"time_seconds": "t_kick"})
    merged = out.merge(g, on=["game_id", "period_id", "t_kick"], how="left")
    unclear = (
        (merged["game_interruption_before"] == "corner_for")
        & merged["start_type"].notna()
        & ~merged["start_type"].isin(_CONFIRMING_START_TYPES)
    )
    merged.loc[unclear, "corner_kind"] = "unclear"
    return merged.drop(columns=["game_interruption_before", "start_type"])
