"""TF-86: the tracking cross attempt and its height (ADR-133; spec 2026-11-13-cross-definition-design §6.3).

The event layer types a cross by one geometric rule (``spadl.base.classify_deliveries``). Event data cannot see
a blocked cross or a delivery that ends in a restart, and some providers' events miss defenders' in-box touches.
This enricher decides the delivery from the ball itself:

* **Release.** The action's own time when its ``time_source`` is ``"tracking"`` (the provider timed it on its own
  tracking clock; cross spec §6.6); else the ELASTIC aligned frame (TF-43) when the action matched confidently; else
  the event time. The nearest ball frame must lie within half a frame period of that time, and the ball must be
  within ``release_match_m`` of the action start, or the path is unanswerable.
* **Path.** From the release to the first resolved ball contact (ADR-124/125 detector) by a player other than the
  passer, at or after ``release + release_gap_s``, capped at ``release + path_cap_s``.
* **Attempt.** ``True`` when the path enters the penalty area. ``<NA>`` (cut out, cross spec §6.3) when it does not,
  the first contact is an opponent's within ``cut_out_window_s`` of the release, and the initial ball direction
  points into the penalty area or cannot be resolved: the detector cannot tell a block from an opponent's touch near
  the crosser, so the label claims neither. ``False`` otherwise. The initial direction is the mean ball movement over
  ``direction_window_s`` after the release, cut at the first contact when that comes earlier (the contact's own frame
  is out); fewer than two finite ball frames leave it unresolved.
* **Unknown, never guessed.** A row whose action start is unknown fails the release match (``<NA>``); an
  open-play pass is ``False`` only when its start is known to lie outside the cross zone.

One match per call: frames, contacts and ELASTIC frame times are keyed by period, so several ``game_id`` values
raise.
* **Height.** ``"aerial"`` when the peak ball z over the path is above ``BallContactParams.aerial_z_m``;
  honest ``<NA>`` unless the ball z is ``"measured"``.

Geometry is in the action-LTR frame (ADR-028): frame positions are flipped for an acting team that attacks
right-to-left (``acting_team_attacks_rtl``). Ships no ``*_xfns``: the label reads the delivery's own outcome
(ADR-030/047). Reference: Vidal-Codina et al. (2022), arXiv:2202.00804 (see NOTICE).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.base import _BOX_X, _BOX_Y_HI, _BOX_Y_LO, CROSS_START_X

from ._action_orientation import acting_team_attacks_rtl
from ._ball_contact import BallContactParams, detect_ball_contacts
from ._elastic_sync import ElasticSyncParams, align_events_to_frames

CROSS_HEIGHT_VALUES: tuple[str, ...] = ("aerial", "ground")

_L = spadlconfig.field_length
_W = spadlconfig.field_width
_ID = spadlconfig.actiontype_id
_OPEN_PLAY = (_ID["pass"], _ID["cross"])
_SET_PIECE = (_ID["corner_crossed"], _ID["corner_short"], _ID["freekick_crossed"], _ID["freekick_short"])


@dataclasses.dataclass(frozen=True)
class CrossDeliveryParams:
    """Parameters of :func:`add_cross_deliveries`. Values from the corpus calibration (spec §7.3).

    Examples
    --------
    A stricter release check::

        params = CrossDeliveryParams(release_match_m=2.0)
        out = add_cross_deliveries(actions, frames, params=params)
    """

    direction_window_s: float = 0.2
    release_gap_s: float = 0.12
    min_ball_coverage: float = 0.8
    release_match_m: float = 3.0
    #: Fixed, not calibrated (cross spec §6.3): an opponent's first contact within this time of the release, on a ball
    #: heading into the box that never enters it, cuts the delivery out (``cross_attempt`` <NA>).
    cut_out_window_s: float = 1.0
    path_cap_s: float = 4.0
    cross_start_x: float = CROSS_START_X
    contact_params: BallContactParams = dataclasses.field(default_factory=BallContactParams)


def _ray_hits_box(x0: float, y0: float, dx: float, dy: float) -> bool:
    """Slab test: does the ray from (x0, y0) along (dx, dy) hit the penalty area (x >= _BOX_X, y in the box)?"""
    if not (np.isfinite(dx) and np.isfinite(dy)) or (dx == 0.0 and dy == 0.0):
        return False
    tmin, tmax = 0.0, np.inf
    for p, d, lo, hi in ((x0, dx, _BOX_X, _L), (y0, dy, _BOX_Y_LO, _BOX_Y_HI)):
        if d == 0.0:
            if p < lo or p > hi:
                return False
            continue
        t1, t2 = (lo - p) / d, (hi - p) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


@nan_safe_enrichment
def add_cross_deliveries(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: CrossDeliveryParams | None = None,
) -> pd.DataFrame:
    """Add ``cross_attempt`` and ``cross_height`` to SPADL actions (TF-86, ADR-133).

    ``cross_attempt`` (nullable ``boolean``), by row class:

    * open-play ``pass``/``cross`` released in the cross zone (wide, at or beyond ``params.cross_start_x``) and
      corner / free-kick rows: True or False by the ball path; ``<NA>`` when the path is unanswerable (no aligned
      release, an unknown action start, too little ball coverage, or an unresolved attacking direction) or cut out
      by an opponent near the release (cross spec §6.3);
    * open-play ``pass``/``cross`` whose start is known to lie outside the cross zone: False;
    * every other row: ``<NA>`` (not applicable).

    ``cross_height`` (category ``"aerial"``/``"ground"``) is set where ``cross_attempt`` is True and the ball z is
    measured; ``<NA>`` elsewhere.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions (canonical LTR) of ONE match with ``action_id``, ``period_id``, ``time_seconds``,
        ``team_id``, ``player_id``, ``type_id``, ``start_x``, ``start_y``. Precomputed ``elastic_frame_id`` /
        ``elastic_confidence`` (``add_elastic_sync``) are read when present; else ELASTIC runs here.
    frames : pd.DataFrame
        Long-form tracking frames with ball rows (``x``, ``y``, ``z``, ``z_source``) and player rows carrying
        ``team_attacking_direction``.
    params : CrossDeliveryParams, optional
        Keyword-only parameters.

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` with the two columns added.

    Examples
    --------
    Label the deliveries of a tracked match (needs frames)::

        out = add_cross_deliveries(actions, frames)
        out.loc[out["cross_attempt"] == True, "cross_height"].value_counts()
    """
    params = params or CrossDeliveryParams()
    out = actions.copy()
    d = _cross_delivery_diagnostics(out, frames, params)
    out["cross_attempt"] = pd.array(d["attempt"].astype("boolean").to_numpy(dtype=object), dtype="boolean")
    out["cross_height"] = pd.Categorical(
        [pd.NA if pd.isna(v) else v for v in d["height"]], categories=CROSS_HEIGHT_VALUES
    )
    return out


def _cross_delivery_diagnostics(
    actions: pd.DataFrame, frames: pd.DataFrame, params: CrossDeliveryParams | None = None
) -> pd.DataFrame:
    """The enricher's per-row evidence, index-aligned to ``actions`` (spec §7.3 calibration; ADR-133).

    ``add_cross_deliveries`` reads ``attempt`` / ``height`` from here, so the calibration measures exactly the
    code path the label uses. Distances are metres in the action-LTR frame; times are seconds after the release.
    """
    params = params or CrossDeliveryParams()
    _validate_inputs(actions, frames)
    out = actions
    n = len(out)
    attempt = np.full(n, None, dtype=object)
    height = np.full(n, None, dtype=object)
    diag = {
        "release_source": np.full(n, None, dtype=object),
        "release_t": np.full(n, np.nan),
        "release_dist_m": np.full(n, np.nan),
        "passer_contact_dt": np.full(n, np.nan),
        "first_contact_dt": np.full(n, np.nan),
        "first_contact_is_opponent": np.full(n, None, dtype=object),
        "first_contact_dist_m": np.full(n, np.nan),
        "coverage": np.full(n, np.nan),
        "cut_out": np.zeros(n, dtype=bool),
        "direction_cut": np.zeros(n, dtype=bool),
        "direction_unresolved": np.zeros(n, dtype=bool),
    }
    applicable = np.zeros(n, dtype=bool)
    if n == 0:
        return _diag_frame(out, applicable, diag, attempt, height)

    type_id = out["type_id"].to_numpy()
    sx = out["start_x"].to_numpy(dtype=np.float64, copy=True)
    sy = out["start_y"].to_numpy(dtype=np.float64, copy=True)
    is_open = np.isin(type_id, _OPEN_PLAY)
    known_start = np.isfinite(sx) & np.isfinite(sy)
    in_zone = ((sy < _BOX_Y_LO) | (sy > _BOX_Y_HI)) & (sx >= params.cross_start_x)
    applicable = (is_open & in_zone) | np.isin(type_id, _SET_PIECE)
    # False only for a start KNOWN to lie outside the zone; an unknown start stays <NA> (ADR-055).
    attempt[is_open & known_start & ~in_zone] = False
    rows = np.flatnonzero(applicable)
    if len(rows) == 0 or len(frames) == 0 or "is_ball" not in frames.columns:
        return _diag_frame(out, applicable, diag, attempt, height)

    flip = acting_team_attacks_rtl(out, frames).to_numpy(dtype=object)
    release_t, used_native, used_elastic = _release_times(out, frames, rows)
    ball = _ball_by_period(frames)
    contacts = _contacts_by_period(detect_ball_contacts(frames, params=params.contact_params))
    team_of = _player_team(frames)

    period = out["period_id"].to_numpy()
    acting_team = out["team_id"].to_numpy(dtype=object)
    passer = out["player_id"].to_numpy(dtype=object)
    for i in rows:
        diag["release_source"][i] = "native" if used_native[i] else ("elastic" if used_elastic[i] else "event")
        diag["release_t"][i] = release_t[i]
        if pd.isna(flip[i]) or period[i] not in ball:
            continue
        t, bx, by, bz, zsrc, frame_period = ball[period[i]]
        if not np.isfinite(frame_period):
            continue  # no trajectory (a freeze frame per action): the path cannot be judged
        fl = bool(flip[i])
        t0 = release_t[i]
        if not np.isfinite(t0):
            continue  # no release time (NaN event time, no ELASTIC frame): nothing to match
        k = _nearest_frame(t, t0)
        if abs(t[k] - t0) > 0.5 * frame_period + 1e-6 or not np.isfinite(bx[k]):
            continue  # no ball frame within half a frame period of the release
        rx, ry = (_L - bx[k], _W - by[k]) if fl else (bx[k], by[k])
        diag["release_dist_m"][i] = float(np.hypot(rx - sx[i], ry - sy[i]))
        own = _first_contact(contacts.get(period[i]), t0 - 0.5, t0 + 1.0, passer[i], by_player=True)
        if own is not None:
            diag["passer_contact_dt"][i] = own[0] - t0
        if not diag["release_dist_m"][i] <= params.release_match_m:
            continue  # misaligned release, or an unknown action start (NaN distance fails closed)
        c = _first_contact(contacts.get(period[i]), t0 + params.release_gap_s, t0 + params.path_cap_s, passer[i])
        t_end = c[0] if c is not None else t0 + params.path_cap_s
        opponent, c_dist = False, np.nan
        if c is not None:
            ccx, ccy = (_L - c[1], _W - c[2]) if fl else (c[1], c[2])
            opponent = _is_opponent(team_of.get(canonical_id(c[3])), acting_team[i])
            c_dist = float(np.hypot(ccx - rx, ccy - ry))
            diag["first_contact_dt"][i] = c[0] - t0
            diag["first_contact_is_opponent"][i] = opponent
            diag["first_contact_dist_m"][i] = c_dist
        hi = int(np.searchsorted(t, t_end + 1e-9, side="right"))
        seg = slice(k, hi)
        px, py = bx[seg], by[seg]
        finite = np.isfinite(px) & np.isfinite(py)
        # Coverage is judged against the frames the path SHOULD hold at the ball's frame rate, so missing ball
        # rows count as missing, not only NaN ones.
        expected = round((t_end - t[k]) / frame_period) + 1
        diag["coverage"][i] = float(finite.sum()) / max(expected, 1)
        if finite.sum() < 2 or diag["coverage"][i] < params.min_ball_coverage:
            continue
        if fl:
            px, py = _L - px, _W - py
        enters = bool(np.any(finite & (px >= _BOX_X) & (py >= _BOX_Y_LO) & (py <= _BOX_Y_HI)))
        cut_out = False
        if not enters and c is not None and opponent and c[0] - t0 <= params.cut_out_window_s:
            # Cross spec §6.3 case 2: the initial direction over direction_window_s, cut at the first contact (its own
            # frame out); fewer than two finite frames leave it unresolved, which is unknown too.
            direction_cut = c[0] < t0 + params.direction_window_s
            if direction_cut:
                w = int(np.searchsorted(t, c[0], side="left"))
            else:
                w = int(np.searchsorted(t, t0 + params.direction_window_s + 1e-9, side="right"))
            wx, wy = bx[k:w], by[k:w]
            ok = np.isfinite(wx) & np.isfinite(wy)
            if ok.sum() >= 2:
                dx, dy = wx[ok][-1] - wx[ok][0], wy[ok][-1] - wy[ok][0]
                if fl:
                    dx, dy = -dx, -dy
                cut_out = _ray_hits_box(rx, ry, dx, dy)
            else:
                cut_out = True
                diag["direction_unresolved"][i] = True
            diag["direction_cut"][i] = direction_cut and cut_out
        diag["cut_out"][i] = cut_out
        attempt[i] = True if enters else (None if cut_out else False)
        if attempt[i] is True:
            zs, zz = zsrc[seg], bz[seg]
            if len(zz) and np.all(zs == "measured") and np.isfinite(zz).any():
                height[i] = "aerial" if np.nanmax(zz) > params.contact_params.aerial_z_m else "ground"
    return _diag_frame(out, applicable, diag, attempt, height)


def _validate_inputs(actions: pd.DataFrame, frames: pd.DataFrame) -> None:
    """The enricher's input contract, shared by the label and the corpus calibration.

    ``_cross_delivery_diagnostics`` calls it, so both paths refuse the same inputs.
    """
    required = {"action_id", "period_id", "time_seconds", "team_id", "player_id", "type_id", "start_x", "start_y"}
    missing = required - set(actions.columns)
    if missing:
        raise ValueError(f"add_cross_deliveries requires action columns {sorted(missing)}")
    # Ball frames, contacts and ELASTIC frame times are keyed by period, so two matches would mix silently.
    for name, df in (("actions", actions), ("frames", frames)):
        if "game_id" in df.columns and df["game_id"].dropna().nunique() > 1:
            raise ValueError(f"add_cross_deliveries takes one match at a time; {name} hold several game_id values")


def _diag_frame(actions, applicable, diag, attempt, height) -> pd.DataFrame:
    def _bool(values):
        return pd.array([pd.NA if v is None else bool(v) for v in values], dtype="boolean")

    return pd.DataFrame(
        {
            "applicable": applicable,
            **{k: v for k, v in diag.items() if k != "first_contact_is_opponent"},
            "first_contact_is_opponent": _bool(diag["first_contact_is_opponent"]),
            "attempt": _bool(attempt),
            "height": np.array([pd.NA if v is None else v for v in height], dtype=object),
        },
        index=actions.index,
    )


def _ensure_cross_attempts(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """The actions with ``cross_attempt``: as staged when the column is already present (a corpus staged with
    ``add_cross_deliveries``), else computed now. The one seam the xCross label callers share (ADR-133)."""
    if "cross_attempt" in actions.columns:
        return actions
    return add_cross_deliveries(actions, frames)


def _release_times(
    actions: pd.DataFrame, frames: pd.DataFrame, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§6.3 release steps 0-2: per row, the native time, else the confident ELASTIC frame's time, else the event time.

    Step 0 (cross spec 6.3): a row whose ``time_source`` is ``"tracking"`` already carries its provider's tracking-clock
    time, so ELASTIC is not consulted for it (it adds error there, spec §5.1). A missing ``time_source`` column means
    ``"event"`` for every row. ``rows`` are the positions whose release the caller reads; when all of them are native,
    ELASTIC is not run at all (a SkillCorner match's synthetic dribbles are "event" but never applicable). When it
    runs, it aligns the whole action stream. Returns the release times and two masks: the rows released at the native
    time, and the rows released at an ELASTIC frame.
    """
    t = actions["time_seconds"].to_numpy(dtype=np.float64, copy=True)
    native = (
        (actions["time_source"].astype(object) == "tracking").to_numpy(dtype=bool)
        if "time_source" in actions.columns
        else np.zeros(len(actions), dtype=bool)
    )
    used = np.zeros(len(actions), dtype=bool)
    if native[rows].all():
        return t, native, used
    if {"elastic_frame_id", "elastic_confidence"} <= set(actions.columns):
        fid = actions["elastic_frame_id"].to_numpy(dtype=np.float64, copy=True)
        conf = actions["elastic_confidence"].to_numpy(dtype=np.float64, copy=True)
    else:  # add_cross_deliveries requires action_id, the key ELASTIC returns its alignment on
        aligned = align_events_to_frames(actions, frames)
        lookup = aligned.set_index("action_id") if len(aligned) else None
        fid = np.full(len(actions), np.nan)
        conf = np.full(len(actions), np.nan)
        if lookup is not None:
            ids = actions["action_id"].to_numpy()
            hit = lookup.reindex(ids)
            fid = hit["elastic_frame_id"].to_numpy(dtype=np.float64, copy=True)
            conf = hit["elastic_confidence"].to_numpy(dtype=np.float64, copy=True)
    use = np.isfinite(fid) & np.isfinite(conf) & (conf >= ElasticSyncParams().min_confidence) & ~native
    if not use.any():
        return t, native, used
    ball = frames.loc[frames["is_ball"].to_numpy(dtype=bool), ["period_id", "frame_id", "time_seconds"]]
    ftime = dict(zip(zip(ball["period_id"], ball["frame_id"], strict=True), ball["time_seconds"], strict=True))
    period = actions["period_id"].to_numpy()
    for i in np.flatnonzero(use):
        ft = ftime.get((period[i], int(fid[i])))
        if ft is not None:
            t[i] = float(ft)
            used[i] = True
    return t, native, used


def _nearest_frame(t: np.ndarray, t0: float) -> int:
    """Index of the sample in the sorted, non-empty ``t`` nearest to ``t0`` (the earlier one on a tie)."""
    j = int(np.searchsorted(t, t0))
    if j == 0:
        return 0
    if j == len(t):
        return j - 1
    return j - 1 if t0 - t[j - 1] <= t[j] - t0 else j


def _ball_by_period(frames: pd.DataFrame) -> dict:
    ball = frames.loc[frames["is_ball"].to_numpy(dtype=bool)]
    out = {}
    for pid, g in ball.groupby("period_id", sort=False):
        g = g.sort_values("time_seconds", kind="stable")
        z = g["z"].to_numpy(dtype=np.float64, copy=True) if "z" in g else np.full(len(g), np.nan)
        zsrc = g["z_source"].astype(object).to_numpy() if "z_source" in g else np.full(len(g), "absent", dtype=object)
        tt = g["time_seconds"].to_numpy(dtype=np.float64, copy=True)
        dt = np.diff(tt)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        # The trajectory's frame period. No positive step (one sample, or freeze frames far apart) -> NaN.
        frame_period = float(np.median(dt)) if len(dt) and float(np.median(dt)) <= 0.5 else float("nan")
        out[pid] = (
            tt,
            g["x"].to_numpy(dtype=np.float64, copy=True),
            g["y"].to_numpy(dtype=np.float64, copy=True),
            z,
            zsrc,
            frame_period,
        )
    return out


def _contacts_by_period(contacts: pd.DataFrame) -> dict:
    if len(contacts) == 0:
        return {}
    src = contacts["contact_source"].astype(object)
    res = contacts.loc[(src != "unresolved") & contacts["player_id"].notna()]
    return {
        pid: (
            g["time_seconds"].to_numpy(dtype=np.float64, copy=True),
            g["contact_x"].to_numpy(dtype=np.float64, copy=True),
            g["contact_y"].to_numpy(dtype=np.float64, copy=True),
            g["player_id"].to_numpy(dtype=object),
        )
        for pid, g in res.sort_values("time_seconds", kind="stable").groupby("period_id", sort=False)
    }


def _first_contact(c, t_lo: float, t_hi: float, player, *, by_player: bool = False) -> tuple | None:
    """The first resolved contact in [t_lo, t_hi) by a player other than ``player`` (by ``player`` itself when
    ``by_player``: the passer's own kick), as (time, x, y, player_id)."""
    if c is None:
        return None
    ct, cx, cy, cp = c
    j = int(np.searchsorted(ct, t_lo, side="left"))
    while j < len(ct) and ct[j] < t_hi:
        if same_id(cp[j], player) == by_player:
            return float(ct[j]), float(cx[j]), float(cy[j]), cp[j]
        j += 1
    return None


def _player_team(frames: pd.DataFrame) -> dict:
    """player_id -> team_id, from the player rows (canonical id keys, ADR-019)."""
    players = frames.loc[~frames["is_ball"].to_numpy(dtype=bool), ["player_id", "team_id"]].dropna()
    players = players.drop_duplicates("player_id")
    return {canonical_id(p): t for p, t in zip(players["player_id"], players["team_id"], strict=True)}


def _is_opponent(contact_team, acting_team) -> bool:
    if contact_team is None or pd.isna(contact_team) or pd.isna(acting_team):
        return False
    return not same_id(contact_team, acting_team)
