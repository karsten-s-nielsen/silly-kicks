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

    _opt = [
        "player_id",
        "type_source",
        "kick_time_source",
        "kick_time_evidence",
        "taker_player_id",
        "t_kick",
        "enriched_start_x",
        "enriched_start_y",
        "origin_evidence",
        "gi_next_possession_time",
    ]
    sel = list(_REQUIRED) + [col for col in _opt if col in actions.columns]
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

    # TF-46 Arm 1 Part 2 (spec section 9): the resolve_kicks kick-time + taker provenance, passthrough.
    out["kick_time_source"] = c["kick_time_source"].to_numpy() if "kick_time_source" in c.columns else pd.NA
    out["kick_time_evidence"] = c["kick_time_evidence"].to_numpy() if "kick_time_evidence" in c.columns else pd.NA
    out["taker_player_id"] = c["taker_player_id"].to_numpy() if "taker_player_id" in c.columns else pd.NA
    out["taker_known"] = out["taker_player_id"].notna()
    out["gi_next_possession_time"] = (
        c["gi_next_possession_time"].astype("float64").to_numpy() if "gi_next_possession_time" in c.columns else np.nan
    )

    # Gated geometry (spec section 9): where resolve_kicks shipped a t_kick (kick_time_source not
    # withheld/none) use it; where the origin gate shipped (origin_evidence set) use the enriched origin +
    # its y for corner_side. In commit 2 everything is withheld, so this is a no-op (native passthrough).
    if "kick_time_source" in c.columns and "t_kick" in c.columns:
        kts = c["kick_time_source"].to_numpy()
        shipped = np.array(
            [s not in ("withheld", "none", None) and not (isinstance(s, float) and np.isnan(s)) for s in kts]
        )
        out["t_kick"] = np.where(shipped & ~derived, c["t_kick"].astype("float64").to_numpy(), out["t_kick"].to_numpy())
    if "enriched_start_x" in c.columns and "origin_evidence" in c.columns:
        oe = c["origin_evidence"].to_numpy()
        has_origin = np.array([e is not None and not (isinstance(e, float) and np.isnan(e)) for e in oe])
        ky = np.where(has_origin, c["enriched_start_y"].astype("float64").to_numpy(), out["kick_y"].to_numpy())
        out["kick_x"] = np.where(
            has_origin, c["enriched_start_x"].astype("float64").to_numpy(), out["kick_x"].to_numpy()
        )
        out["kick_y"] = ky
        new_side = np.where(ky >= midline, "left", "right").astype(object)
        new_side[np.isnan(ky)] = None
        out["corner_side"] = np.where(has_origin, new_side, out["corner_side"].to_numpy())

    # Section 9.3: a provider whose EVENTS code the first touch (GS / PFF: the next event by a different player)
    # carries it as native passthrough; add_corner_windows ships it as winner_source="event_coded" (not gated).
    ev_t, ev_p, ev_team = _event_first_touch(actions, c, provider)
    out["event_first_touch_time"] = ev_t
    out["event_first_touch_player_id"] = pd.array(ev_p, dtype="object")
    out["event_first_touch_team_id"] = pd.array(ev_team, dtype="object")

    out = out.reset_index(drop=True)
    return out.reindex(columns=list(CORNER_IDENTIFY_COLUMNS)).astype(CORNER_IDENTIFY_COLUMNS)


#: Providers whose EVENTS code the corner's first touch (spec section 9.3): Gradient Sports, through the section
#: 5.6 PFF fields that its SPADL conversion carries as the next action (a challenge resolved to its winner).
_EVENT_CODED_FIRST_TOUCH: frozenset[str] = frozenset({"gradientsports"})
_FIRST_TOUCH_WINDOW_S = 6.0  # the R* window (spec section 16.1 rstar_window_s)


def _event_first_touch(actions: pd.DataFrame, corners: pd.DataFrame, provider: str):
    """Per corner row: the native first touch = the first action AFTER the kick, within the 6 s R* window, by a
    player other than the taker (section 9.3). NaN / ``<NA>`` off the event-coded providers or when none.

    The actions are grouped by period ONCE and sorted, so each corner is a ``searchsorted`` lookup (ADR-068)
    rather than a re-filter of the whole action table."""
    n = len(corners)
    ev_t = np.full(n, np.nan)
    ev_p: list[object] = [pd.NA] * n
    ev_team: list[object] = [pd.NA] * n
    if provider not in _EVENT_CODED_FIRST_TOUCH or n == 0 or "player_id" not in actions.columns:
        return ev_t, ev_p, ev_team
    from silly_kicks.id_compat import canonical_id, same_id

    by_period: dict[tuple, tuple] = {}
    for key, g in actions.groupby(["game_id", "period_id"], sort=False):
        g = g.sort_values("time_seconds", kind="stable")
        by_period[key] = (g["time_seconds"].to_numpy(float), g["player_id"].to_numpy(), g["team_id"].to_numpy())
    for i in range(n):
        key = (corners["game_id"].iloc[i], corners["period_id"].iloc[i])
        hit = by_period.get(key)
        if hit is None:
            continue
        times, players, teams = hit
        t0 = float(corners["time_seconds"].iloc[i])
        taker = corners["player_id"].iloc[i] if "player_id" in corners.columns else None
        taker_c = canonical_id(taker) if taker is not None and not pd.isna(taker) else None
        j = int(np.searchsorted(times, t0, side="right"))
        while j < len(times) and times[j] <= t0 + _FIRST_TOUCH_WINDOW_S:
            pj = players[j]
            if pj is not None and not pd.isna(pj) and not same_id(pj, taker_c):
                ev_t[i], ev_p[i], ev_team[i] = float(times[j]), pj, teams[j]
                break
            j += 1
    return ev_t, ev_p, ev_team
