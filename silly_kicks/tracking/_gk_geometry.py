"""Goal-kick geometry resolution for xT-GK (scoped; promotable -- see TODO general-enrichment).

Derives a goal-kick's origin/destination when the SPADL event omits them (real GS data:
~67% NaN origin), WITHOUT mutating the shared ``actions`` frame. Conditional origin in
confidence order (native -> in-area tracking-GK -> goal-area rule point); destination
native -> next-event -> unresolved. Emits per-row source + continuous confidence. All tiers
measured on owner data (spec 2026-06-08-xt-gk-goalkick-coverage-design.md). See NOTICE.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._id_compat import ids_match

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_GOAL_AREA_DEPTH = 16.5  # m from own goal line; tracking-GK beyond this is "off position" (measured 48%)
_RULE_POINT = (5.5, 34.0)  # 6-yard-box centre, LTR
# Effective tiers are 3 (native / in-area-tracking-GK / rule-point): the "empirical median" and
# "rule point" are not data-distinguishable, so they collapse to one positional prior.
_CONF = {"native": 1.0, "tracking_gk": 0.7, "goalkick_prior": 0.2, "unresolved": 0.0}

# --- General restart-coordinate enrichment (spec 2026-06-10-general-restart-coordinate-enrichment) ---
_PENALTY = spadlconfig.actiontype_id["shot_penalty"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
_CORNER_TYPES = (spadlconfig.actiontype_id["corner_crossed"], spadlconfig.actiontype_id["corner_short"])
_RESTART_PRIOR_TYPES = (_GOALKICK, _PENALTY, _THROW_IN, *_CORNER_TYPES)  # types that get a rule-point

_PENALTY_SPOT = (spadlconfig.field_length - 11.0, spadlconfig.field_width / 2.0)  # (94.0, 34.0) LTR
_CORNER_X = spadlconfig.field_length  # 105.0 (opponent goal line)
_TOUCHLINE_LO, _TOUCHLINE_HI = 0.0, spadlconfig.field_width  # 0.0 / 68.0
_MID_Y = spadlconfig.field_width / 2.0  # 34.0 (side split)

# Per-type restart-prior confidence (generic source label is always "restart_prior"; confidence varies
# by type). goalkick 0.2 is FROZEN (parity). Others provisional (spec section 4.4).
_PRIOR_CONF = {_GOALKICK: 0.2, _PENALTY: 0.5, _THROW_IN: 0.3, _CORNER_TYPES[0]: 0.4, _CORNER_TYPES[1]: 0.4}
_CONF_TRACKING_BALL = 0.8  # origin
_CONF_TRACKING_BALL_DEST = 0.5  # dest (provisional; spec section 9 may drop)
_CONF_NEXT_EVENT = 0.6
_CONF_TRACKING_GK = 0.7  # FROZEN (goalkick parity)

# Tripwire regions (LTR; imputed origin coords only). Tolerances provisional (spec section 6).
_TRIPWIRE = {
    "goalkick": lambda x, y: x <= _GOAL_AREA_DEPTH,
    "penalty": lambda x, y: abs(x - _PENALTY_SPOT[0]) <= 3.0 and abs(y - _PENALTY_SPOT[1]) <= 3.0,
    "corner": lambda x, y: x >= 100.0 and (y <= 5.0 or y >= 63.0),
    "throw_in": lambda x, y: y <= 3.0 or y >= 65.0,
}


def resolve_gk_geometry(
    actions: pd.DataFrame, *, frames: pd.DataFrame | None, links: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Goal-kick coordinate derivation (the frozen pre-promotion contract). Returns a frame indexed
    like ``actions`` with origin_x/origin_y/origin_source/origin_confidence + dest_x/dest_y/
    dest_source. Only goal-kicks (type 22) get origin imputation; other rows pass native coords
    through. ``actions`` is never mutated.

    Thin delegation to :func:`resolve_restart_geometry` with ``impute_types=(goalkick,)`` -- so the
    engine imputes GOAL-KICKS ONLY (non-goalkick rows are never imputed -> no revert needed) and runs
    no tripwire (pure engine). The shim then renames to the legacy columns, drops the dest-confidence
    column, and maps ``restart_prior`` -> ``goalkick_prior``. Public API; do NOT change the output
    contract (4 internal callers + the xT-GK completion path depend on it byte-for-byte)."""
    g = resolve_restart_geometry(actions, frames=frames, links=links, impute_types=(_GOALKICK,))

    # Whole-array numpy (no .loc-mask assignment -> index-independent, matches the original style).
    osrc = g["start_coord_source"].to_numpy().astype(object).copy()
    # label map: restart_prior -> goalkick_prior (goalkick rule-point). tracking_gk / native /
    # next_event / unresolved pass through unchanged. NO tracking_ball->tracking_gk mapping.
    osrc = np.where(osrc == "restart_prior", "goalkick_prior", osrc)

    out = pd.DataFrame(index=actions.index)
    out["origin_x"] = g["enriched_start_x"].to_numpy()
    out["origin_y"] = g["enriched_start_y"].to_numpy()
    out["origin_source"] = osrc
    out["origin_confidence"] = g["start_coord_confidence"].to_numpy()
    out["dest_x"] = g["enriched_end_x"].to_numpy()
    out["dest_y"] = g["enriched_end_y"].to_numpy()
    out["dest_source"] = g["end_coord_source"].to_numpy()
    # DROP end_coord_confidence -- the frozen contract has origin_confidence only.
    return out[["origin_x", "origin_y", "origin_source", "origin_confidence", "dest_x", "dest_y", "dest_source"]]


def _next_event_start(actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Next row's start coords (the receiver location), positionally. NaN'd across a
    (game_id, period_id) boundary (a period-/match-final goalkick must NOT take the next
    period's first action as its destination -> falls to dest_source == 'unresolved')."""
    nx = actions["start_x"].shift(-1).to_numpy(float)
    ny = actions["start_y"].shift(-1).to_numpy(float)
    same = np.ones(len(actions), dtype=bool)
    for col in ("game_id", "period_id"):
        if col in actions.columns:
            same &= actions[col].to_numpy() == actions[col].shift(-1).to_numpy()
    nx = np.where(same, nx, np.nan)
    ny = np.where(same, ny, np.nan)
    return nx, ny


def _tracking_gk_xy(actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame | None) -> np.ndarray:
    """Acting-team GK position at each goal-kick's linked frame, CLAMPED to the goal area
    (x <= _GOAL_AREA_DEPTH in LTR own-half coords); NaN where unavailable or off-position."""
    from ._kernels import resolve_frame_ids_by_position

    n = len(actions)
    res = np.full((n, 2), np.nan, dtype=float)
    fid = resolve_frame_ids_by_position(actions, frames, links=links)
    fg = frames.groupby("frame_id")
    team_ids = actions["team_id"].to_numpy()
    for i in range(n):
        if not np.isfinite(fid[i]):
            continue
        try:
            fr = fg.get_group(int(fid[i]))
        except KeyError:
            continue
        gk = fr[
            fr["is_goalkeeper"].astype(bool) & (~fr["is_ball"].astype(bool)) & ids_match(fr["team_id"], team_ids[i])
        ]
        if gk.empty:
            continue
        gx, gy = float(gk.iloc[0]["x"]), float(gk.iloc[0]["y"])
        if gx <= _GOAL_AREA_DEPTH:  # clamp: off-position GK falls through to the rule point
            res[i] = (gx, gy)
    return res


def _truthy_bool(s: pd.Series) -> np.ndarray:
    """Coerce a possibly object/string ``is_*`` column to a bool mask. NEVER ``.astype(bool)`` a
    provider string qualifier -- ``pd.Series(["False"]).astype(bool)`` is ``True`` (non-empty string),
    the exact trap behind the ADR-019 object-``is_ball`` ``~``-no-op bug. Real bool/numeric columns
    pass through sensibly; string ``"true"``/``"1"``/``"yes"`` (case-insensitive) -> ``True``."""
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy() != 0
    return s.fillna("").astype(str).str.strip().str.lower().isin(("true", "1", "yes")).to_numpy()


def _tracking_ball_xy(actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame | None) -> np.ndarray:
    """Ball position at each action's linked frame; NaN where unavailable. The ball IS the dead-ball
    restart spot. ADR-019: ``is_ball`` is coerced via :func:`_truthy_bool` (a bare ``.astype(bool)``
    on an object/string column mis-selects -- ``"False"`` is truthy)."""
    from ._kernels import resolve_frame_ids_by_position

    n = len(actions)
    res = np.full((n, 2), np.nan, dtype=float)
    fid = resolve_frame_ids_by_position(actions, frames, links=links)
    is_ball = _truthy_bool(frames["is_ball"])  # coerce (ADR-019; never .astype(bool) a string)
    ball_frames = frames[is_ball]
    fg = ball_frames.groupby("frame_id")
    for i in range(n):
        if not np.isfinite(fid[i]):
            continue
        try:
            fr = fg.get_group(int(fid[i]))
        except KeyError:
            continue
        if fr.empty:
            continue
        res[i] = (float(fr.iloc[0]["x"]), float(fr.iloc[0]["y"]))
    return res


def _side_y(actions: pd.DataFrame, frames, links) -> np.ndarray:
    """Y used to pick a corner/throw-in side: native end_y -> next-event start_y -> tracking-ball y.
    NaN where none resolves (caller leaves the row unresolved -- never guess a side)."""
    side = actions["end_y"].to_numpy(float).copy()
    _, ny = _next_event_start(actions)
    side = np.where(np.isfinite(side), side, ny)
    if frames is not None:
        ball = _tracking_ball_xy(actions, frames, links)
        side = np.where(np.isfinite(side), side, ball[:, 1])
    return side


def _throwin_x(actions: pd.DataFrame, frames, links) -> np.ndarray:
    """X along the touchline for a throw-in: next-event start_x -> tracking-ball x. NaN if none."""
    nx, _ = _next_event_start(actions)
    x = nx.copy()
    if frames is not None:
        ball = _tracking_ball_xy(actions, frames, links)
        x = np.where(np.isfinite(x), x, ball[:, 0])
    return x


def resolve_restart_geometry(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
    impute_types: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """General restart-coordinate enrichment. Returns an index-aligned frame with
    enriched_start_x/_y/enriched_end_x/_y + start_coord_source/start_coord_confidence +
    end_coord_source/end_coord_confidence. NEVER mutates ``actions``. PURE: emits no warnings and
    applies no tripwire (the tripwire is a feature-policy step applied at the
    ``add_restart_coordinates`` edge; spec section 6) -- so the ``resolve_gk_geometry`` shim that
    delegates here can never leak a warning onto the frozen ``compute_xt_gk`` path.

    PRECONDITION: ``actions`` is in chronological ``(game_id, period_id, action_id)`` order (the
    ``next_event`` ``shift(-1)`` is positional). The public ``add_restart_coordinates`` sorts first;
    callers passing pre-sorted SPADL streams (e.g. ``compute_xt_gk``) satisfy this.

    ``impute_types``: action-type ids eligible for imputation past ``native``. ``None`` = all types
    (the general default). The ``resolve_gk_geometry`` shim passes ``(goalkick,)`` so non-goalkick
    rows are NEVER imputed (parity: matches the frozen goalkick-only contract; perf: zero
    ``_tracking_ball_xy`` work on the frozen hot path).

    Origin tiers (confidence order): native -> [goalkick: in-area tracking_gk; non-goalkick:
    tracking_ball] -> restart_prior (goalkick/penalty/corner/throw_in only) -> unresolved.
    Destination tiers: native -> next_event (full-frame positional) -> tracking_ball (non-goalkick
    only) -> unresolved. tracking_ball is gated OFF for goal-kicks (origin AND dest) so
    ``resolve_gk_geometry`` stays byte-identical (spec section 4.1 invariant).

    See NOTICE; spec 2026-06-10-general-restart-coordinate-enrichment-design.md.
    """
    n = len(actions)
    out = pd.DataFrame(index=actions.index)
    sx = actions["start_x"].to_numpy(float)
    sy = actions["start_y"].to_numpy(float)
    ex = actions["end_x"].to_numpy(float)
    ey = actions["end_y"].to_numpy(float)
    tid = actions["type_id"].to_numpy()
    is_gk = tid == _GOALKICK
    is_corner = np.isin(tid, _CORNER_TYPES)
    is_throw = tid == _THROW_IN
    eligible = np.ones(n, dtype=bool) if impute_types is None else np.isin(tid, tuple(impute_types))

    # ---------- origin ----------
    ox, oy = sx.copy(), sy.copy()
    osrc = np.where(np.isfinite(sx) & np.isfinite(sy), "native", "unresolved").astype(object)
    oconf = np.where(osrc == "native", 1.0, 0.0).astype(float)

    need = (osrc == "unresolved") & eligible
    # tier 2a: goalkick in-area tracking-GK (goalkick ONLY; no tracking_ball for goalkick)
    if frames is not None and (need & is_gk).any():
        gk = _tracking_gk_xy(actions, frames, links)
        use = need & is_gk & np.isfinite(gk[:, 0])
        ox[use], oy[use] = gk[use, 0], gk[use, 1]
        osrc[use], oconf[use] = "tracking_gk", _CONF_TRACKING_GK
        need = (osrc == "unresolved") & eligible
    # tier 2b: tracking-ball (NON-goalkick eligible rows). Skipped entirely on the goalkick-only
    # (frozen) path -- (need & ~is_gk) is empty there, so _tracking_ball_xy is never called.
    if frames is not None and (need & ~is_gk).any():
        ball = _tracking_ball_xy(actions, frames, links)
        use = need & ~is_gk & np.isfinite(ball[:, 0])
        ox[use], oy[use] = ball[use, 0], ball[use, 1]
        osrc[use], oconf[use] = "tracking_ball", _CONF_TRACKING_BALL
        need = (osrc == "unresolved") & eligible
    # tier 3: restart rule-points (restart-prior types only). _side_y / _throwin_x computed ONLY
    # when a corner/throw-in actually needs them (avoids wasted _tracking_ball_xy on the frozen path).
    side = _side_y(actions, frames, links) if (need & (is_corner | is_throw)).any() else None
    twx = _throwin_x(actions, frames, links) if (need & is_throw).any() else None
    for i in np.where(need)[0]:
        t = int(tid[i])
        if t == _GOALKICK:
            ox[i], oy[i] = _RULE_POINT
        elif t == _PENALTY:
            ox[i], oy[i] = _PENALTY_SPOT
        elif t in _CORNER_TYPES:
            if side is None or not np.isfinite(side[i]):
                continue  # cannot determine side -> leave unresolved
            ox[i], oy[i] = _CORNER_X, (_TOUCHLINE_LO if side[i] < _MID_Y else _TOUCHLINE_HI)
        elif t == _THROW_IN:
            if side is None or twx is None or not (np.isfinite(side[i]) and np.isfinite(twx[i])):
                continue
            ox[i], oy[i] = twx[i], (_TOUCHLINE_LO if side[i] < _MID_Y else _TOUCHLINE_HI)
        else:
            continue  # open-play / freekick_short -> no rule-point
        osrc[i], oconf[i] = "restart_prior", _PRIOR_CONF[t]

    out["enriched_start_x"], out["enriched_start_y"] = ox, oy
    out["start_coord_source"], out["start_coord_confidence"] = osrc, oconf

    # ---------- destination ----------
    dx, dy = ex.copy(), ey.copy()
    dsrc = np.where(np.isfinite(ex) & np.isfinite(ey), "native", "unresolved").astype(object)
    dconf = np.where(dsrc == "native", 1.0, 0.0).astype(float)
    dneed = (dsrc == "unresolved") & eligible
    # tier 2: next_event (eligible rows; full-frame positional). On the goalkick-only path this fires
    # for goalkicks only -> matches the frozen contract's goalkick-gated next_event.
    if dneed.any():
        nx, ny = _next_event_start(actions)
        use = dneed & np.isfinite(nx) & np.isfinite(ny)
        dx[use], dy[use] = nx[use], ny[use]
        dsrc[use], dconf[use] = "next_event", _CONF_NEXT_EVENT
        dneed = (dsrc == "unresolved") & eligible
    # tier 3: tracking-ball dest (NON-goalkick eligible rows). Empty on the goalkick-only path.
    if frames is not None and (dneed & ~is_gk).any():
        ball = _tracking_ball_xy(actions, frames, links)
        use = dneed & ~is_gk & np.isfinite(ball[:, 0])
        dx[use], dy[use] = ball[use, 0], ball[use, 1]
        dsrc[use], dconf[use] = "tracking_ball", _CONF_TRACKING_BALL_DEST

    out["enriched_end_x"], out["enriched_end_y"] = dx, dy
    out["end_coord_source"], out["end_coord_confidence"] = dsrc, dconf
    return out  # NO tripwire here -- applied at the add_restart_coordinates edge


def _tripwire_key(t: int) -> str | None:
    if t == _GOALKICK:
        return "goalkick"
    if t == _PENALTY:
        return "penalty"
    if t in _CORNER_TYPES:
        return "corner"
    if t == _THROW_IN:
        return "throw_in"
    return None


def apply_restart_tripwire(out: pd.DataFrame) -> int:
    """Validate imputed restart ORIGIN coords against their Law region, IN PLACE on an enriched
    frame (as emitted by :func:`resolve_restart_geometry`, carrying a ``type_id`` column). Imputed
    (non-``native``) coords that violate -> reverted to NaN, source ``tripwire_reverted``, confidence
    0.0. Native violations warn only (provider truth, never reverted). Destinations are NOT guarded in
    Phase 1 (spec section 6). Returns the reversion count. PURE policy step -- called by
    ``add_restart_coordinates``, never by the engine (so the frozen ``resolve_gk_geometry`` path stays
    silent + revert-free)."""
    tid = out["type_id"].to_numpy()
    sx = out["enriched_start_x"].to_numpy().copy()
    sy = out["enriched_start_y"].to_numpy().copy()
    ssrc = out["start_coord_source"].to_numpy().astype(object).copy()
    sconf = out["start_coord_confidence"].to_numpy().astype(float).copy()
    reverts = 0
    for i in range(len(out)):
        key = _tripwire_key(int(tid[i]))
        if key is None or not np.isfinite(sx[i]):
            continue
        if _TRIPWIRE[key](sx[i], sy[i]):
            continue  # in-region
        if ssrc[i] == "native":
            warnings.warn(
                f"add_restart_coordinates: native {key} origin ({sx[i]:.1f},{sy[i]:.1f}) outside "
                f"its Law region (data-quality signal; not reverted).",
                stacklevel=2,
            )
            continue
        warnings.warn(
            f"add_restart_coordinates: imputed {key} origin ({sx[i]:.1f},{sy[i]:.1f}) outside its "
            f"Law region; reverted to unresolved.",
            stacklevel=2,
        )
        sx[i] = sy[i] = np.nan
        ssrc[i], sconf[i] = "tripwire_reverted", 0.0
        reverts += 1
    out["enriched_start_x"], out["enriched_start_y"] = sx, sy
    out["start_coord_source"], out["start_coord_confidence"] = ssrc, sconf
    return reverts
