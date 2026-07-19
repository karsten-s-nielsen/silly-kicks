"""Direction-of-play helpers shared between Gradient Sports events + tracking adapters.

Extracted from ``silly_kicks/spadl/gradientsports.py`` (PR-S18) into the
tracking package so events Gradient Sports, tracking Gradient Sports, and
tracking Sportec can share one implementation. ``home_attacks_right_per_period`` is the load-bearing
helper; ``compute_attacking_direction`` is a higher-level wrapper used
by tracking adapters to populate the per-row ``team_attacking_direction``
column.

Pure refactor: zero behaviour change in events.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.spadl import config as _spadlconfig

_PITCH_LENGTH_M: float = _spadlconfig.field_length  # 105.0
_PITCH_WIDTH_M: float = _spadlconfig.field_width  # 68.0
_PITCH_MID_X: float = _PITCH_LENGTH_M / 2.0  # 52.5
_LTR_KNOWN_PERIODS: tuple[int, ...] = (1, 2, 3, 4)  # period 5 (PSO) direction undefined


def require_et_direction(
    period_ids: pd.Series | np.ndarray | Sequence[int],
    home_team_start_left_extratime: bool | None,
    *,
    source: str,
) -> None:
    """Raise ``ValueError`` if ET periods are present but the ET direction is unset.

    Per-period-absolute converters (Sportec, Metrica, Gradient Sports --- tracking
    **and** events) flip coordinates per period by the home team's start direction.
    Extra time (periods 3/4) needs a separate ``home_team_start_left_extratime``
    flag; guessing it silently flips ET coordinates and corrupts every downstream
    geometric feature. This shared guard makes the failure loud and identical across
    all five converters. See ADR-010.

    Parameters
    ----------
    period_ids : pd.Series | np.ndarray | Sequence[int]
        The per-row / per-frame period identifiers of the data about to be converted.
    home_team_start_left_extratime : bool | None
        The ET start-direction flag from match metadata
        (e.g. ``homeTeamStartLeftExtraTime``). ``None`` means "not provided".
        ``False`` is a valid value and does **not** trigger the guard.
    source : str
        Human-readable converter identity for the error message, e.g.
        ``"sportec convert_to_frames"``.

    Raises
    ------
    ValueError
        If ``home_team_start_left_extratime is None`` and any period in
        ``period_ids`` is 3 or 4.

    Examples
    --------
    Validate a batch before converting::

        from silly_kicks.tracking import require_et_direction
        require_et_direction(frames["period_id"], meta_flag, source="sportec convert_to_frames")
    """
    if home_team_start_left_extratime is None and pd.Series(period_ids).isin([3, 4]).any():
        raise ValueError(
            f"{source}: data contains ET periods (period_id in {{3, 4}}) but "
            "home_team_start_left_extratime was not provided. Set it from the match "
            "metadata (e.g. homeTeamStartLeftExtraTime), or filter ET out before converting."
        )


def home_attacks_right_per_period(
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> dict[int, bool]:
    """Per-period flag: True iff the home team attacks RIGHT (LTR) in that period.

    Mirrors the original Gradient Sports events convention: in period 1, the home team
    attacks right when ``home_team_start_left=True`` (its goal is on the
    left, so it shoots toward the right goal). Period 2 inverts period 1.
    Period 3/4 (extra time) follow ``home_team_start_left_extratime``,
    falling back to True for period 4 when ET is absent. Period 5 (PSO)
    is a single-end shootout where direction is moot --- conventionally
    True.

    Parameters
    ----------
    home_team_start_left : bool
        From Gradient Sports metadata ``homeTeamStartLeft`` / DFL match-info equivalent.
    home_team_start_left_extratime : bool | None
        From Gradient Sports metadata ``homeTeamStartLeftExtraTime`` / DFL equivalent.
        Only required when ET periods (3/4) are present in the data.

    Returns
    -------
    dict[int, bool]
        ``{1: ..., 2: ..., 3: ..., 4: ..., 5: True}``.

    Examples
    --------
    Map a per-period flip lookup for the home team::

        from silly_kicks.tracking.direction import home_attacks_right_per_period
        flips = home_attacks_right_per_period(
            home_team_start_left=True, home_team_start_left_extratime=False,
        )
        assert flips[1] is True and flips[2] is False
    """
    return {
        1: bool(home_team_start_left),
        2: not bool(home_team_start_left),
        3: bool(home_team_start_left_extratime),
        4: (not bool(home_team_start_left_extratime) if home_team_start_left_extratime is not None else True),
        5: True,
    }


def compute_attacking_direction(
    *,
    team_id: pd.Series,
    period_id: pd.Series,
    is_ball: pd.Series,
    home_team_id: Any,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> pd.Series:
    """Per-row pre-flip attacking direction (``"ltr"`` / ``"rtl"``).

    Returns ``"ltr"`` for the team attacking left-to-right in this period
    (i.e., the side whose ``home_attacks_right_per_period`` flag is True
    iff that side is the home team), ``"rtl"`` otherwise. Ball rows always
    get ``None``. Period 5 (PSO) leaves direction undefined (``None``).

    This is the per-row analogue of ``home_attacks_right_per_period``. The
    tracking adapters use this to populate the
    ``team_attacking_direction`` schema column for the unflipped raw input;
    the adapter then per-period flips x/y so the final output is in
    home-team-attacks-LTR coordinates.
    """
    out = pd.Series([None] * len(team_id), dtype="object", index=team_id.index)
    flags = home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    for p in (1, 2, 3, 4):
        period_mask = (period_id == p) & ~is_ball
        if not period_mask.any():
            continue
        home_attacks_right = flags[p]
        is_home = ids_match(team_id, home_team_id)
        out.loc[period_mask & is_home] = "ltr" if home_attacks_right else "rtl"
        out.loc[period_mask & ~is_home] = "rtl" if home_attacks_right else "ltr"
    return out


def orient_frames_to_ltr_by_geometry(
    frames: pd.DataFrame,
    *,
    home_team_id: Any,
    source: str = "",
    game_id: Any = None,
    on_missing_home: Literal["raise", "warn"] = "raise",
    copy: bool = True,
) -> pd.DataFrame:
    """Flag-free geometric frame-LTR orientation: ensure home attacks +x every period.

    Per-period directional anchor = the home goalkeeper's median x. A GK sits deepest
    in its own half, so in the canonical home-attacks-right (LTR) frame the home GK
    must sit at LOW x (home defends x=0). Any period whose home-GK median x is on the
    attacking half (``> 52.5``) is mis-oriented; ALL its rows are point-reflected
    (``x->105-x``, ``y->68-y``, ``vx->-vx``, ``vy->-vy`` when present; ``speed`` is a
    magnitude, unchanged). ``team_attacking_direction`` is populated where null.

    Unlike :func:`silly_kicks.tracking.orient_frames_to_ltr` (flag-based), this reads
    orientation from the DATA, so it is robust to absent/defaulted
    ``home_team_start_left`` (no bronze field carries it) and to per-feed ET coordinate
    flips. **Idempotent** --- a no-op on already-correctly-oriented frames (home GK
    already at low x). Promoted from luxury-lakehouse ADR-053
    ``correct_frames_to_home_ltr``; see NOTICE.

    Orientation is the builder's owned, normal operation (every match flips ~half its
    periods), so normal flips are SILENT (unlike ADR-053's correctness-net logging);
    a period with no GK anchor warns; a ``home_team_id`` matching no player raises
    (ADR-019 --- mis-orienting is worse than failing).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form frames. Required: ``x``, ``y``, ``team_id``, ``period_id``,
        ``is_ball``, ``is_goalkeeper``. ``vx``/``vy`` flipped when present.
    home_team_id : Any
        Home-team id matching ``frames["team_id"]`` (compared via ADR-019 ``ids_match``).
    source, game_id : Any
        Diagnostic context only (warning messages).
    on_missing_home : {"raise", "warn"}, default "raise"
        Policy when ``home_team_id`` matches zero player rows (cannot anchor):
        ``"raise"`` (ADR-019 default --- mis-orienting is worse than failing; used by
        direct/lakehouse callers); ``"warn"`` emits a ``UserWarning`` and returns the frame
        un-oriented (the native adapters pass this via ``finalize_orientation`` so their
        established warn-don't-raise contract holds --- the flag-flip result stands).
    copy : bool, default True
        When True, operate on a defensive copy (input never mutated). Callers that already own
        a fresh frame (e.g. :func:`finalize_orientation`) pass ``False`` to avoid a redundant
        full-frame copy on tracking-scale data.

    Returns
    -------
    pd.DataFrame
        New DataFrame in home-attacks-right convention.

    Raises
    ------
    ValueError
        Missing required column, or ``home_team_id`` matches zero player rows.

    Examples
    --------
    Orient absolute metrica/skillcorner frames built from bronze::

        from silly_kicks.tracking.direction import orient_frames_to_ltr_by_geometry
        oriented = orient_frames_to_ltr_by_geometry(frames, home_team_id="Home")
    """
    required = {"x", "y", "team_id", "period_id", "is_ball", "is_goalkeeper"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"orient_frames_to_ltr_by_geometry: required column(s) missing: {sorted(missing)}")
    if len(frames) == 0:
        return frames.copy()

    out = frames.copy() if copy else frames
    is_ball = out["is_ball"].astype(bool)
    is_player = ~is_ball
    is_home = ids_match(out["team_id"], home_team_id).fillna(False)
    is_gk = out["is_goalkeeper"].astype(bool)

    if not bool((is_player & is_home).any()):
        msg = (
            f"orient_frames_to_ltr_by_geometry: home_team_id={home_team_id!r} matched ZERO "
            f"player rows ({source} game={game_id})"
        )
        if on_missing_home == "raise":
            raise ValueError(msg + " --- refusing to guess orientation.")
        warnings.warn(msg + " --- orientation left as-is.", stacklevel=2)
        return out

    x_arr = out["x"].to_numpy(dtype="float64")
    period_arr = out["period_id"].to_numpy()
    home_arr = is_home.to_numpy(dtype=bool)
    player_arr = is_player.to_numpy(dtype=bool)
    gk_arr = is_gk.to_numpy(dtype=bool)

    def _gk_median(mask: np.ndarray) -> float:
        vals = x_arr[mask]
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    has_vx, has_vy = "vx" in out.columns, "vy" in out.columns
    for period in pd.Series(period_arr[player_arr]).dropna().unique():
        if period not in _LTR_KNOWN_PERIODS:  # period 5 (PSO): orientation undefined --- never flip
            continue
        psel = player_arr & (period_arr == period)
        home_gk_x = _gk_median(psel & home_arr & gk_arr)
        if not np.isnan(home_gk_x):
            needs_flip = home_gk_x > _PITCH_MID_X
        else:
            away_gk_x = _gk_median(psel & ~home_arr & gk_arr)
            if np.isnan(away_gk_x):
                warnings.warn(
                    f"orient_frames_to_ltr_by_geometry: {source} game={game_id} period={period} "
                    "has no GK anchor (home or away) --- orientation left as-is for this period.",
                    stacklevel=2,
                )
                continue
            needs_flip = away_gk_x < _PITCH_MID_X
        if needs_flip:
            fmask = period_arr == period
            out.loc[fmask, "x"] = _PITCH_LENGTH_M - x_arr[fmask]
            out.loc[fmask, "y"] = _PITCH_WIDTH_M - out["y"].to_numpy(dtype="float64")[fmask]
            if has_vx:
                out.loc[fmask, "vx"] = -out["vx"].to_numpy(dtype="float64")[fmask]
            if has_vy:
                out.loc[fmask, "vy"] = -out["vy"].to_numpy(dtype="float64")[fmask]

    if "team_attacking_direction" in out.columns and out["team_attacking_direction"].isna().all():
        known = is_player & out["period_id"].isin(_LTR_KNOWN_PERIODS)
        out.loc[known & is_home, "team_attacking_direction"] = "ltr"
        out.loc[known & ~is_home, "team_attacking_direction"] = "rtl"
    return out


def finalize_orientation(
    out: pd.DataFrame,
    *,
    home_team_id: Any,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None,
    source: str,
    game_id: Any = None,
    on_missing_home: Literal["raise", "warn"] = "warn",
) -> pd.DataFrame:
    """Shared orientation tail for the sportec + gradientsports native tracking adapters.

    Single source of truth for the ET guard, the per-period flag flip, the post-flip
    period-gated ``team_attacking_direction`` label, and the TF-23b geometric backstop.
    Expects ``out`` to already carry canonical ``x``/``y`` (105x68 m) plus ``team_id``,
    ``period_id``, ``is_ball``, ``is_goalkeeper``. **Returns a NEW frame and does not mutate
    the input** (copy-at-entry). The output is in home-attacks-right (absolute) convention;
    the caller applies :func:`play_left_to_right` afterward for ``output_convention="ltr"``.

    The geometric backstop (:func:`orient_frames_to_ltr_by_geometry`) self-corrects any period
    whose home GK sits on the attacking half --- e.g. a wrong ``home_team_start_left_extratime``
    placeholder. It is idempotent, so on a correct-flag match it is a byte-identical no-op.
    ``on_missing_home="warn"`` (the adapter default) preserves the adapters' warn-don't-raise
    contract without re-implementing the net's zero-home condition.

    Parameters
    ----------
    out : pd.DataFrame
        Frames with canonical ``x``/``y`` already constructed.
    home_team_id : Any
        Home-team id matching ``out["team_id"]`` (ADR-019 ``ids_match``).
    home_team_start_left, home_team_start_left_extratime : bool, bool | None
        Per-period flip flags (see :func:`home_attacks_right_per_period`).
    source : str
        Converter identity for guard/warning messages, e.g. ``"sportec convert_to_frames"``.
    game_id : Any
        Diagnostic context for the backstop's warnings.
    on_missing_home : {"raise", "warn"}, default "warn"
        Backstop zero-home policy (see :func:`orient_frames_to_ltr_by_geometry`).

    Returns
    -------
    pd.DataFrame
        New frame in home-attacks-right convention.

    Examples
    --------
    Collapse a native adapter's orientation tail to one call::

        from silly_kicks.tracking import direction
        out = direction.finalize_orientation(
            out, home_team_id=home_team_id, home_team_start_left=True,
            home_team_start_left_extratime=None, source="sportec convert_to_frames",
        )
    """
    out = out.copy()  # clean value semantics --- never mutate the caller's frame
    require_et_direction(out["period_id"], home_team_start_left_extratime, source=source)

    flips = home_attacks_right_per_period(home_team_start_left, home_team_start_left_extratime)
    home_rtl_periods = {p for p, attacks_right in flips.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = _PITCH_LENGTH_M - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = _PITCH_WIDTH_M - out.loc[flip_mask, "y"]

    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    # ADR-019 dtype-safe is_home: a raw `==` silently matched ZERO players when home_team_id was
    # int and team_id object-string -> every player mislabeled -> play_left_to_right double-flip
    # -> mis-oriented frames (2026-06-09 fix). Do NOT "simplify" back to ==.
    is_home = ids_match(out["team_id"], home_team_id).fillna(False).to_numpy(dtype=bool)
    is_known = out["period_id"].isin(_LTR_KNOWN_PERIODS).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known, "team_attacking_direction"] = "rtl"

    if is_player.any():  # all-ball frame: nothing to anchor; skip the net entirely
        # copy=False: `out` is already this function's private copy (copy-at-entry), so the
        # net can mutate in place -- avoids a redundant third full-frame copy (review #1).
        out = orient_frames_to_ltr_by_geometry(
            out,
            home_team_id=home_team_id,
            source=source,
            game_id=game_id,
            on_missing_home=on_missing_home,
            copy=False,
        )
    return out
