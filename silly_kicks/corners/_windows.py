"""add_corner_windows -- per-corner window timing + first-contact winner (TF-46, spec §6.2).

Additive enricher on the identification table. For a tracking corner, the target snapshot is
``min(t_firstcontact + target_contact_lead_s, t_kick + target_cap_s)`` with the first contact (and its
winner) from the shipped TF-79 ``first_contact_in_window``; when TF-79 resolves no contact the
``t_kick + cap`` branch is the fallback (winner ``<NA>`` -- NO GI fallback). A freeze-frame corner
(``corner_source == "sb360_event"``) has no continuous tracking, so winner / first_contact are ``<NA>``
and the target-snapshot source is ``sb360_event`` (the first-on-ball-event freeze frame; its time is
supplied by the driver). ``first_possession_time`` is a GI diagnostic (NaN unless ``gi`` is given).

The contact row (TF-79) carries no team, so ``winner_team_id`` is resolved from ``frames`` at the
contact frame. PURE (ADR-033); NaN-tolerant (ADR-003); ``frames`` positional-or-keyword, kwargs
keyword-only (ADR-078).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._ball_contact import first_contact_in_window

from ._columns import CORNER_WINDOW_COLUMNS
from ._config import CornerWindowParams

_DEFAULT_WINDOW_PARAMS = CornerWindowParams()


def _team_of(frames: pd.DataFrame, frame_id, player_id) -> object:
    """Team of ``player_id`` at ``frame_id`` in ``frames`` (``<NA>`` if unresolved)."""
    if player_id is None or pd.isna(player_id) or frame_id is None or pd.isna(frame_id):
        return pd.NA
    sub = frames.loc[frames["frame_id"] == frame_id, ["player_id", "team_id"]].reset_index(drop=True)
    if sub.empty:
        return pd.NA
    # NA-safe id compare (ADR-019): the ball row carries a <NA> player_id, so a raw ``==`` yields an
    # NA-bearing mask that crashes under boolean indexing. ids_match maps <NA> -> False.
    match = ids_match(sub["player_id"], player_id).to_numpy()
    team = sub["team_id"].to_numpy()[match]
    return team[0] if len(team) else pd.NA


@nan_safe_enrichment
def add_corner_windows(
    corners: pd.DataFrame,
    frames: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    params: CornerWindowParams = _DEFAULT_WINDOW_PARAMS,
    gi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add the window + winner + timing-diagnostic columns to the corners table (new frame).

    Examples
    --------
    Enrich an identified corner table with TF-79 timing::

        from silly_kicks.corners import add_corner_windows
        windowed = add_corner_windows(corners, frames, contacts)
    """
    out = corners.copy()
    n = len(out)
    t_init = np.full(n, np.nan)
    t_target = np.full(n, np.nan)
    t_outcome = np.full(n, np.nan)
    first_contact_time = np.full(n, np.nan)
    first_possession_time = np.full(n, np.nan)
    winner_team: list[object] = [pd.NA] * n
    winner_player: list[object] = [pd.NA] * n
    source: list[object] = [None] * n

    gi_lookup: dict[tuple, float] = {}
    if gi is not None and len(gi) > 0 and "start_time" not in gi.columns:
        for _, g in gi.iterrows():
            gi_lookup[(g["game_id"], int(g["period_id"]), float(g["time_seconds"]))] = float(g["time_seconds"])

    src_col = out["corner_source"].astype(object)
    for i in range(n):
        t_kick = float(out["t_kick"].iloc[i])
        period = int(out["period_id"].iloc[i])
        t_init[i] = t_kick - params.init_lead_s
        t_outcome[i] = t_kick + params.outcome_window_s
        if gi_lookup:
            key = (out["game_id"].iloc[i], period, t_kick)
            if key in gi_lookup:
                first_possession_time[i] = gi_lookup[key]

        if src_col.iloc[i] == "sb360_event":
            # Freeze-frame: no continuous tracking -> no TF-79 contact; FF target time set by the driver.
            source[i] = "sb360_event"
            t_target[i] = t_kick + params.target_cap_s
            continue

        contact = first_contact_in_window(contacts, period, t_kick, t_kick + params.outcome_window_s)
        if contact is not None:
            t_fc = float(contact["time_seconds"])
            first_contact_time[i] = t_fc
            t_target[i] = min(t_fc + params.target_contact_lead_s, t_kick + params.target_cap_s)
            source[i] = "first_contact"
            winner_player[i] = contact["player_id"]
            winner_team[i] = _team_of(frames, contact["frame_id"], contact["player_id"])
        else:
            t_target[i] = t_kick + params.target_cap_s
            source[i] = "kick_plus_2s"

    out["t_init"] = t_init
    out["t_target"] = t_target
    out["t_outcome"] = t_outcome
    out["first_contact_time"] = first_contact_time
    out["first_possession_time"] = first_possession_time
    out["winner_team_id"] = pd.array(winner_team, dtype="object")
    out["winner_player_id"] = pd.array(winner_player, dtype="object")
    out["target_snapshot_source"] = pd.Series(source, index=out.index, dtype="object").astype("category")
    return out.astype({c: t for c, t in CORNER_WINDOW_COLUMNS.items() if c in out.columns})
