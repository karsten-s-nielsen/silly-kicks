"""The kick resolver (TF-46 Arm 1 Part 2, spec sections 6, 7). Resolves, per restart row, the search bracket,
the kick-contact and ball-onset candidates, the arbitrated kick time `t_kick`, the taker, and (SkillCorner)
the per-match GI / tracking pairing.

Pure: returns a COPY of ``actions`` with the section 6.6 + 7 columns on restart rows (``<NA>`` elsewhere);
never mutates ``frames`` or ``actions``. The gate table is EMPTY in commit 2 (CL-3), so every ``kick_time``
cell resolves to ``withheld`` (candidates kept as evidence); an injected passing table ships the value.

Region tests run in the action frame (section 4.3) via ``_restart_frame``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from silly_kicks._frame_index import column_position, group_rows
from silly_kicks.spadl import config as spadlconfig

from ._ball_contact import detect_ball_contacts
from ._gk_resolve import GoalMap
from ._law_geometry import PitchDims, resolve_pitch_dims, rest_location_ok
from ._restart_frame import (
    FrameFlips,
    GiPairing,
    PeriodFrameRanges,
    assess_gi_pairing,
    gi_direction_check,
    resolve_frame_flips,
    resolve_period_frame_ranges,
    team_attacks_rtl,
    to_action_frame,
)
from ._restart_validation import GATE_TABLE_VERSION, RATIFIED, DataProfile, gate


def _i(v) -> int:
    """pandas Scalar -> int; the untyped arg sidesteps the Scalar stub friction."""
    return int(v)


# Restart action types the resolver brackets (penalties are detected in section 12 but still bracketed here).
_RESTART_TYPES: frozenset[int] = frozenset(
    spadlconfig.actiontype_id[n]
    for n in ("corner_crossed", "corner_short", "freekick_crossed", "freekick_short", "goalkick", "throw_in")
    if n in spadlconfig.actiontype_id
)
_DEAD_BALL = frozenset({"corner", "free_kick", "goal_kick", "penalty"})

#: string / id / boolean columns the resolver writes; all `<NA>` off restart rows.
_STR_COLS = (
    "kick_time_source",
    "kick_time_resolution",
    "kick_time_evidence",
    "kick_time_gate_version",
    "kick_time_cell_resolution",
    "taker_source",
    "taker_evidence",
    "taker_gate_version",
    "taker_cell_resolution",
    "gi_pairing",
    "gi_pairing_failed_test",
)
_FLOAT_COLS = (
    "kick_bracket_lo",
    "kick_bracket_hi",
    "gi_next_possession_time",
    "t_contact",
    "t_onset",
    "kick_time_delta",
    "t_kick",
    "kick_rest_x",
    "kick_rest_y",
)


def _speed(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-frame ball speed (m/s); speed[0] = 0."""
    s = np.zeros(len(t))
    dt = np.diff(t)
    dd = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        s[1:] = np.where(dt > 0, dd / dt, 0.0)
    return s


_L, _W = float(spadlconfig.field_length), float(spadlconfig.field_width)


@dataclass(frozen=True)
class RestEpisode:
    """One maximal ball rest window of a dead-ball trace (spec sections 5.2, 6.4, 16.1).

    ``start`` / ``end`` index the trace; ``centroid`` (raw frame) and ``sd`` are the window's rest centroid and
    position SD (the release displacement reference); ``rest_xy`` (raw frame) is the MEDIAN position over the last
    ``penalty_position_eval_s`` of the window (section 16.1); ``qualifies`` = the section 6.4 rest location for the
    restart type holds (action frame); ``release`` = the window's own section 5.2 release in the section 6.4
    direction, NaN when the ball leaves the rest some other way or rests again first; ``peak_speed`` = the highest
    ball speed SUSTAINED on ``k_release_frames`` consecutive deduplicated samples from the release to the next rest
    window (or the trace end), NaN without a release; ``strike`` = ``peak_speed >= v_kick`` (spec sections 5.2 /
    6.7 / 16.1, amended 2026-10-22 twice: a placement roll or nudge stays below ``v_kick``, and one jitter frame
    above it is not a strike).

    Examples
    --------
    Inspect the rest episodes of one corner trace (action frame: ``attacks_rtl=False``)::

        for e in rest_episodes(ball, restart_type="corner", attacks_rtl=False):
            print(e.start, e.end, e.qualifies, e.release, e.rest_xy)
    """

    start: int
    end: int
    centroid: tuple[float, float]
    sd: float
    rest_xy: tuple[float, float]
    qualifies: bool
    release: float
    peak_speed: float = float("nan")
    strike: bool = False


class KickVerdict(NamedTuple):
    """K from one episode list: ``release`` (NaN when none), ``rest_xy`` and the ``reason`` -- ``strike`` (the
    first strike), ``single_episode`` (no strike, one qualifying rest), ``no_strike_ambiguous`` (no strike, >= 2
    qualifying rests: ``none``, counted) or ``no_episode``.

    Examples
    --------
    Branch on the reason of one trace's verdict::

        v = kick_verdict(rest_episodes(ball_obs, restart_type="free_kick", attacks_rtl=rtl))
        if v.reason == "no_strike_ambiguous":
            ...  # `none`: no kick time ships for this restart
    """

    release: float
    rest_xy: tuple[float, float] | None
    reason: str


def _sustained_peak(speed: np.ndarray, need: int) -> float:
    """The highest speed SUSTAINED on ``need`` consecutive samples (the max over windows of the window minimum);
    NaN when fewer than ``need`` samples (spec sections 5.2 / 16.1, amended 2026-10-22: one jitter frame is never
    a strike)."""
    if need < 1 or len(speed) < need:
        return float("nan")
    return float(np.lib.stride_tricks.sliding_window_view(speed, need).min(axis=1).max())


def _maximal_rest_windows(t, x, y, rest_m: float, rest_s: float) -> list[tuple[int, int]]:
    """EVERY maximal rest window: a run of samples within ``rest_m`` of its centroid lasting >= ``rest_s`` s.
    Windows do not overlap; the scan resumes after each accepted window (section 16.1 rest condition).

    A NaN sample (a missed ball detection) neither extends nor ends a window: it is skipped, and the window's
    centroid and duration come from its finite samples, so a 1-2 frame tracking gap cannot cut a rest short.
    The centroid is kept as running sums; a new sample shifts it by ``shift``, which moves every earlier member's
    distance by at most ``shift``, so the exact all-members check runs only when that bound could cross ``rest_m``
    (the acceptance criterion is exactly "every finite member within ``rest_m`` of the new centroid")."""
    n = len(t)
    finite = np.isfinite(x) & np.isfinite(y)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        sx, sy, m = float(x[i]), float(y[i]), 1
        cx, cy = sx, sy
        bound = 0.0  # an upper bound on every member's distance to the centroid
        last = i
        k = i + 1
        while k < n:
            if not finite[k]:
                k += 1
                continue
            ncx, ncy = (sx + x[k]) / (m + 1), (sy + y[k]) / (m + 1)
            d_new = float(np.hypot(x[k] - ncx, y[k] - ncy))
            cand = bound + float(np.hypot(ncx - cx, ncy - cy))
            if d_new > rest_m:
                break
            if cand > rest_m:  # the cheap bound is inconclusive: the exact check over the finite members
                fm = finite[i : k + 1]
                cand = float(np.max(np.hypot(x[i : k + 1][fm] - ncx, y[i : k + 1][fm] - ncy)))
                if cand > rest_m:
                    break
            sx, sy, m = sx + float(x[k]), sy + float(y[k]), m + 1
            cx, cy, bound, last = float(ncx), float(ncy), max(cand, d_new), k
            k += 1
        if (t[last] - t[i]) >= rest_s:
            out.append((i, last))
            i = last + 1
        else:
            i += 1
    return out


def _release_direction_ok(base: str, ax0: float, ay0: float, dx: float, dy: float) -> bool:
    """The section 6.4 release direction (action frame): corner -- into the pitch (toward its centre); goal kick
    -- away from the own goal line (+x); penalty -- toward the goal (+x); free kick -- any."""
    if base == "corner":
        return bool(dx * (_L / 2.0 - ax0) + dy * (_W / 2.0 - ay0) > 0.0)
    if base in ("goal_kick", "penalty"):
        return bool(dx > 0.0)
    return True


def rest_episodes(
    ball: pd.DataFrame,
    *,
    restart_type: str,
    attacks_rtl: bool | None,
    pitch: PitchDims | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> list[RestEpisode]:
    """Every maximal rest window of one dead-ball trace, each with its section 6.4 location verdict and its own
    section 5.2 release (spec sections 5.2, 6.4, 6.7, 16.1).

    ``attacks_rtl`` = the restarting team attacks right-to-left in the trace's frame (GoalMap direction, ADR-055).
    ``None`` = unresolved: the location and direction conditions of a corner / goal kick / penalty cannot be
    evaluated, so no window qualifies (fail closed, never a guessed orientation); a free kick needs neither.
    ``pitch`` = the provider pitch in real metres (``PitchDims``, default 105 x 68): the section 6.4 Law locations
    are tested in real metres (``_law_geometry.rest_location_ok``; amended 2026-10-22). A window's release is
    searched from
    its end up to the next window's start, with its own centroid and SD; the first frame that meets the release
    thresholds must also move in the section 6.4 direction, else the window has no release.

    Examples
    --------
    Count the qualifying episodes of a penalty trace (the run (1) multi-episode diagnostic)::

        eps = rest_episodes(ball, restart_type="penalty", attacks_rtl=False, pitch=PitchDims(100.0, 64.0))
        n_kept = sum(e.qualifies for e in eps)
    """
    th = {**RATIFIED, **(thresholds or {})}
    t = ball["time_seconds"].to_numpy(dtype=np.float64)
    x = ball["x"].to_numpy(dtype=np.float64)
    y = ball["y"].to_numpy(dtype=np.float64)
    n = len(t)
    if n < 2:
        return []
    flip = bool(attacks_rtl) if attacks_rtl is not None and not pd.isna(attacks_rtl) else None
    dims = pitch if pitch is not None else PitchDims()
    needs_frame = restart_type in ("corner", "goal_kick", "penalty")
    windows = _maximal_rest_windows(t, x, y, th["k_rest_m"], th["k_rest_s"])
    # The release and strike tests read ONE deduplicated speed series (TF-79 `trajectory_dedup_mask`): a duplicate
    # frame gives speed 0 and its successor double speed, a sawtooth no "k consecutive frames" rule survives (the
    # 4.137.0 duplicate-frame finding). Consecutive frames are counted over deduplicated samples. Inside a REST
    # window a repeated position is the rest itself, never a duplicate artefact, so rest frames are all kept: the
    # first moving sample after a rest keeps its true speed. No duplicates -> the series is the raw one.
    from silly_kicks.tracking.preprocess._dedup import trajectory_dedup_mask

    keep = trajectory_dedup_mask(x, y)
    for i_w, j_w in windows:
        keep[i_w : j_w + 1] = True
        # ...and the still-resting samples right after the (greedy) window, within k_rest_m of its centroid
        cxw, cyw = float(np.nanmean(x[i_w : j_w + 1])), float(np.nanmean(y[i_w : j_w + 1]))
        k_w = j_w + 1
        while k_w < n and np.hypot(x[k_w] - cxw, y[k_w] - cyw) <= th["k_rest_m"]:
            keep[k_w] = True
            k_w += 1
    kidx = np.flatnonzero(keep)
    kspeed = _speed(t[kidx], x[kidx], y[kidx])
    kpos = np.full(n, -1, dtype=np.int64)
    kpos[kidx] = np.arange(len(kidx))
    need = int(th["k_release_frames"])
    out: list[RestEpisode] = []
    for w, (i, j) in enumerate(windows):
        cx = float(np.nanmean(x[i : j + 1]))
        cy = float(np.nanmean(y[i : j + 1]))
        sd = float(np.nanstd(np.hypot(x[i : j + 1] - cx, y[i : j + 1] - cy)))
        tail = (t[i : j + 1] >= t[j] - th["penalty_position_eval_s"]) & np.isfinite(x[i : j + 1])
        rest_xy = (float(np.median(x[i : j + 1][tail])), float(np.median(y[i : j + 1][tail])))
        if flip is None:
            qualifies = not needs_frame
            acx, acy = cx, cy
        else:
            acx, acy = (_L - cx, _W - cy) if flip else (cx, cy)
            qualifies = rest_location_ok(restart_type, acx, acy, dims)
        release = float("nan")
        peak = float("nan")
        stop = windows[w + 1][0] if w + 1 < len(windows) else n
        disp_thr = max(th["k_release_disp_m"], th["k_release_rest_sd_mult"] * sd)
        for k in range(j + 1, stop):
            q = int(kpos[k])
            if q < 0:
                continue  # a duplicate sample: no speed of its own
            disp = float(np.hypot(x[k] - cx, y[k] - cy))
            if (
                disp > disp_thr
                and q >= need - 1
                and bool(np.all(kspeed[q - need + 1 : q + 1] > th["k_release_speed_ms"]))
            ):
                dx, dy = x[k] - cx, y[k] - cy
                if flip:
                    dx, dy = -dx, -dy
                if flip is None and needs_frame:
                    break  # no direction -> no release verdict (fail closed)
                if _release_direction_ok(restart_type, acx, acy, dx, dy):
                    release = float(t[k])
                    q_stop = int(np.searchsorted(kidx, stop, side="left"))
                    peak = _sustained_peak(kspeed[q:q_stop], need)
                break  # the first release-like movement IS this window's departure
        strike = bool(np.isfinite(peak) and peak >= th["sc_freekick_vkick_ms"])
        out.append(RestEpisode(i, j, (cx, cy), sd, rest_xy, bool(qualifies), release, peak, strike))
    return out


def kick_verdict(episodes: list[RestEpisode]) -> KickVerdict:
    """K from an episode list (spec sections 5.2, 6.7, 16.1; amended 2026-10-22, owner approval of the DL
    strike rule). A strike is a release whose peak ball speed up to the next rest reaches ``v_kick``.

    K = the release of the last qualifying rest at or before the FIRST strike in the observation window -- the
    striking rest itself, since every rest before it ended in a sub-``v_kick`` placement. A rest after the first
    strike is post-kick and never anchors K, whatever the outcome (blocked, saved, cleared). With no strike: one
    qualifying rest gives K from that rest; two or more give ``none`` (``no_strike_ambiguous``, counted). The rest
    location is the chosen episode's median over its last 0.5 s.

    Examples
    --------
    K and its reason from one episode list::

        v = kick_verdict(rest_episodes(ball_obs, restart_type="free_kick", attacks_rtl=rtl))
        t_onset, rest_xy, reason = v.release, v.rest_xy, v.reason
    """
    kept = [e for e in episodes if e.qualifies]
    strikes = [e for e in kept if e.strike]
    if strikes:
        return KickVerdict(strikes[0].release, strikes[0].rest_xy, "strike")
    if len(kept) == 1:
        return KickVerdict(kept[0].release, kept[0].rest_xy, "single_episode")
    if len(kept) >= 2:
        return KickVerdict(float("nan"), None, "no_strike_ambiguous")
    return KickVerdict(float("nan"), None, "no_episode")


def choose_release(episodes: list[RestEpisode]) -> tuple[float, tuple[float, float] | None]:
    """``(release, rest_xy)`` of :func:`kick_verdict` (the K rule without its reason).

    Examples
    --------
    ::

        t_onset, rest_xy = choose_release(rest_episodes(ball_obs, restart_type="corner", attacks_rtl=rtl))
    """
    v = kick_verdict(episodes)
    return v.release, v.rest_xy


def kinematic_release(
    ball: pd.DataFrame,
    *,
    restart_type: str,
    thrower_xy: tuple[float, float] | None = None,
    thresholds: Mapping[str, float] | None = None,
    attacks_rtl: bool | None = None,
    pitch: PitchDims | None = None,
) -> tuple[float, tuple[float, float] | None]:
    """The section 5.2 / 16.1 release rule on one ball trace: ``(release_time, rest location or None)``.

    Dead-ball kicks: every maximal rest window (within ``k_rest_m`` of its centroid for >= ``k_rest_s``) is found;
    only windows at the section 6.4 rest location for the type are kept (action frame, from ``attacks_rtl``);
    each kept window's release is the first frame with displacement > max(``k_release_disp_m``,
    ``k_release_rest_sd_mult`` x that window's SD) AND speed > ``k_release_speed_ms`` on ``k_release_frames``
    consecutive frames, moving in the section 6.4 direction. K follows the strike rule of :func:`kick_verdict`
    (the first strike; without a strike a single kept rest, else ``none``); the rest location is the chosen
    window's median over its last 0.5 s. A
    corner / goal kick / penalty with ``attacks_rtl=None`` fails closed (NaN). Throw-in: the first free-flight
    frame (speed > ``throwin_release_speed_ms``, distance to the thrower rising > ``throwin_release_dist_m`` over
    ``throwin_release_window_s``). NaN release when the rule does not fire.

    ``thresholds`` overrides individual ``RATIFIED`` release thresholds (run (1) evaluates each at +/-50 %; the
    thresholds are never fitted to event times, spec section 5.2). Default ``None`` = the ratified values.

    Examples
    --------
    K on an undegraded corner trace, and its +50 % displacement-threshold sensitivity (run (1))::

        k, rest_xy = kinematic_release(ball, restart_type="corner", attacks_rtl=False)
        hi = {"k_release_disp_m": 0.75}
        k_hi, _ = kinematic_release(ball, restart_type="corner", attacks_rtl=False, thresholds=hi)
    """
    th = {**RATIFIED, **(thresholds or {})}
    t = ball["time_seconds"].to_numpy(dtype=np.float64)
    x = ball["x"].to_numpy(dtype=np.float64)
    y = ball["y"].to_numpy(dtype=np.float64)
    n = len(t)
    if n < 2:
        return float("nan"), None
    if restart_type == "throw_in":
        speed = _speed(t, x, y)
        thr_speed = th["throwin_release_speed_ms"]
        thr_dist = th["throwin_release_dist_m"]
        thr_win = th["throwin_release_window_s"]
        for k in range(1, n):
            if speed[k] <= thr_speed:
                continue
            if thrower_xy is None:
                return float(t[k]), None
            j = int(np.searchsorted(t, t[k] + thr_win, side="left"))
            if j >= n:
                continue
            d_now = float(np.hypot(x[k] - thrower_xy[0], y[k] - thrower_xy[1]))
            d_later = float(np.hypot(x[j] - thrower_xy[0], y[j] - thrower_xy[1]))
            if d_later - d_now > thr_dist:
                return float(t[k]), None
        return float("nan"), None
    eps = rest_episodes(ball, restart_type=restart_type, attacks_rtl=attacks_rtl, pitch=pitch, thresholds=thresholds)
    return choose_release(eps)


def _retake_rho(base: str) -> float | None:
    """The retake return-to-rest radius per restart type (section 16.1). A goal kick has no radius: its rule is
    "the same goal area", which every qualifying goal-kick episode already satisfies (its rest location IS the own
    goal area, ``_law_geometry.rest_location_ok``), so any two qualifying goal-kick episodes are within it
    (``inf``). None where retakes do not apply (throw-in)."""
    return {
        "penalty": RATIFIED["retake_rho_penalty_m"],
        "free_kick": RATIFIED["retake_rho_freekick_m"],
        "corner": RATIFIED["retake_rho_corner_m"],
        "goal_kick": float("inf"),
    }.get(base)


def detect_retake(episodes: list[RestEpisode], base: str, lo: float, hi: float) -> tuple[bool, float]:
    """A retake (section 6.7, amended 2026-10-22) from the SAME episode list K uses, all inside the strict bracket
    ``[lo, hi]``: a STRIKE, then a qualifying rest within ``rho_retake`` of the first rest, then a second STRIKE.
    A placement nudge stays below ``v_kick`` and never sets a retake. Returns ``(is_retake, last_onset)``: the last
    strike of the retake chain (``t_kick`` follows it; the resolver never silently takes the first onset).

    Examples
    --------
    The same episode list serves K and the retake rule::

        eps = rest_episodes(ball_obs, restart_type="corner", attacks_rtl=rtl)
        is_retake, last_onset = detect_retake(eps, "corner", lo, hi)
    """
    rho = _retake_rho(base)
    if rho is None:
        return False, float("nan")
    strikes = [e for e in episodes if e.qualifies and e.strike and lo <= e.release <= hi]
    if len(strikes) < 2:
        return False, float("nan")
    first = strikes[0]
    chain = [first]
    for e in strikes[1:]:
        # Section 6.7: each return to rest is measured against the FIRST rest centroid (C2 r2 C-27), so a chain of
        # small drifts can never walk a retake away from the original spot.
        if np.hypot(e.centroid[0] - first.centroid[0], e.centroid[1] - first.centroid[1]) > rho:
            break  # a strike from elsewhere is a later play, not a retake
        chain.append(e)
    if len(chain) >= 2:
        return True, chain[-1].release
    return False, float("nan")


def _frame_time(frame_id, frame_lookup: dict) -> float:
    """Period-relative time for a GI frame id via the period frame-id lookup; NaN when the id is missing or
    unmapped."""
    if frame_id is None or pd.isna(frame_id):
        return float("nan")
    return frame_lookup.get(int(frame_id), float("nan"))


def _ball_trace(ball_sorted: pd.DataFrame, ball_times: np.ndarray, lo: float, hi: float) -> pd.DataFrame:
    """The sorted period ball rows within ``[lo, hi]`` -- a ``searchsorted`` slice of the PRE-SORTED period
    ball trace (ADR-068: no per-restart re-filter of the whole trace)."""
    left = int(np.searchsorted(ball_times, lo, side="left"))
    right = int(np.searchsorted(ball_times, hi, side="right"))
    return ball_sorted.iloc[left:right]


def _arbitrate(t_contact, t_onset, lo, hi, tau, gated, winner=None):
    """The section 6.5 arbitration state machine -> (t_kick, kick_time_source, kick_time_resolution).

    ``winner`` = the gated ``kick_time`` entry's pre-registered agreement winner (``tf79_contact`` / ``ball_onset``,
    derived from run (2), spec 6.5 / 16.1 amended 2026-10-22). It is read from the gate entry, never hard-coded;
    an ungated row is withheld before the winner matters, and a gated entry without a winner raises."""
    c_obs = np.isfinite(t_contact)
    o_obs = np.isfinite(t_onset)
    c_in = bool(c_obs and lo <= t_contact <= hi)
    o_in = bool(o_obs and lo <= t_onset <= hi)
    n_in = int(c_in) + int(o_in)
    if not gated:
        return float("nan"), "withheld", pd.NA
    if n_in == 0:
        return float("nan"), "none", pd.NA
    delta = (t_contact - t_onset) if (c_obs and o_obs) else float("nan")
    if n_in == 1:
        src, tk = ("tf79_contact", t_contact) if c_in else ("ball_onset", t_onset)
        if c_obs and o_obs and abs(delta) > tau:
            return float(tk), f"conflict_resolved_{src}", "bracket"
        return float(tk), src, "single"
    # both inside
    if abs(delta) <= tau:
        if winner not in ("tf79_contact", "ball_onset"):
            raise ValueError(f"_arbitrate: a gated kick_time entry carries no agreement winner ({winner!r})")
        return float(t_contact if winner == "tf79_contact" else t_onset), "both", "agreement"
    return float("nan"), "conflict", pd.NA


def resolve_kicks(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    provider: str,
    goal_map: GoalMap,
    contacts: pd.DataFrame | None = None,
    data_profile: DataProfile | None = None,
    gi_pairing: Mapping[Any, GiPairing] | None = None,
    gate_table: Mapping | None = None,
    bracket_ab: Mapping[str, tuple[float, float]] | None = None,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
    flips: FrameFlips | None = None,
    period_ranges: PeriodFrameRanges | None = None,
) -> pd.DataFrame:
    """Resolve the kick columns (section 6.6 + 7) on every restart row; ``<NA>`` elsewhere. Pure.

    ``gate_table`` overrides the shipped ``RESTART_GATE_TABLE`` (default ``None`` = shipped). The pre-flight
    drivers inject a pass-all table to SCORE each rule regardless of the shipped gate; production never does.
    ``bracket_ab`` gives the native-anchor bracket ``(a, b)`` per restart base (run (1): the p0.5 / p99.5 of
    ``t_event - K``, clipped to 2.0 s, spec section 6.1); a base absent from it uses the 2.0 s clip.

    SkillCorner: when ``gi_pairing`` is None it is computed here (``assess_gi_pairing``, no provenance
    sources); a driver passes a precomputed mapping that carries the GI / tracking provenance. A SkillCorner
    match absent from a passed mapping raises ``ValueError`` naming the ``game_id`` (fail loud).

    ``pitch_length`` / ``pitch_width`` are the provider pitch in real metres (the section 6.4 Law locations scale
    with it; amended 2026-10-22): declared keywords first, else ``frames.attrs``, else 105 x 68.

    ``flips`` is the frames' recorded per-period orientation flip (``FrameFlips``, keyed ``(game_id, period_id)``;
    build it from the SkillCorner tracking report with ``skillcorner.frame_flips_from_report(report)`` -- the raw
    ``report.ltr_flip_by_period`` is keyed by period only and raises ``TypeError``), used by the computed GI
    direction check: the declared keyword first, else the adapter's ``frames.attrs`` stamp, else none -- every
    period direction-UNRESOLVED (its geometry withheld; GI is never compared raw against oriented frames). Frames
    persisted to disk lose the attrs stamp: store the flip record alongside them and pass it here.

    ``period_ranges`` is the PROVIDER period range per ``(game_id, period_id)`` (``PeriodFrameRanges``; build it with
    ``skillcorner.period_frame_ranges_from_report(report)``), read by the computed pairing's Range test (spec section
    6.1, amended 2026-10-25): the declared keyword first, else the ``frames.attrs`` stamp, else the retained range
    (stricter; ``GiPairing.range_basis = "retained"``). A GI frame inside the provider range but absent from the
    frames withholds only the rows that read it (``gi_frame_unmapped``).

    A multi-match input is resolved match by match (every frame-time structure is per match -- two matches share
    period ids and clocks); ``contacts`` must then carry ``game_id`` (else ``ValueError``: ambiguous).

    Examples
    --------
    Resolve the kicks of one IDSSE match (the native events anchor the bracket)::

        goal_map = resolve_defended_goals(frames)
        kicks = resolve_kicks(actions, frames, provider="idsse", goal_map=goal_map, data_profile=profile)
        shipped = kicks[kicks["t_kick"].notna()]  # gated kick times; withheld rows carry <NA>

    A SkillCorner match passes the frames' recorded flip (the GI direction check reads it)::

        frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
        kicks = resolve_kicks(
            actions, frames, provider="skillcorner", goal_map=resolve_defended_goals(frames),
            flips=skillcorner.frame_flips_from_report(report),
        )
    """
    # The provider pitch (real metres) for the section 6.4 Law locations: resolved ONCE from the declared keywords,
    # else the converter's frames.attrs stamp, else 105 x 68 -- before any per-match split (attrs need not survive it).
    dims = resolve_pitch_dims(frames, pitch_length, pitch_width)
    pitch_length, pitch_width = dims.length, dims.width
    flips = resolve_frame_flips(frames, flips)  # ONCE, before the per-match split (attrs need not survive it)
    period_ranges = resolve_period_frame_ranges(frames, period_ranges)  # ONCE, likewise (keyed (game, period))
    if "game_id" in actions.columns and "game_id" in frames.columns and actions["game_id"].nunique(dropna=True) > 1:
        return _resolve_per_match(
            actions,
            frames,
            provider=provider,
            goal_map=goal_map,
            contacts=contacts,
            data_profile=data_profile,
            gi_pairing=gi_pairing,
            gate_table=gate_table,
            bracket_ab=bracket_ab,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            flips=flips,
            period_ranges=period_ranges,
        )
    out = actions.copy()
    for c in _FLOAT_COLS:
        out[c] = np.nan
    for c in _STR_COLS:
        out[c] = pd.NA
    out["kick_retake"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["taker_player_id"] = pd.array([pd.NA] * len(out), dtype="object")

    is_restart = out["type_id"].isin(_RESTART_TYPES).to_numpy()
    report = _empty_report()
    if not is_restart.any():
        out.attrs["kick_resolver_report"] = report
        return out

    is_sc = provider == "skillcorner"
    if is_sc:
        if gi_pairing is None:
            direction = gi_direction_check(actions, goal_map, flips=flips)
            gi_pairing = assess_gi_pairing(
                actions, frames, provider=provider, direction=direction, period_ranges=period_ranges
            )
        else:
            for gid in out.loc[is_restart, "game_id"].unique():
                if gid not in gi_pairing:
                    raise ValueError(
                        f"resolve_kicks: SkillCorner match {gid!r} absent from the passed gi_pairing mapping"
                    )

    if contacts is None and len(frames):
        # the provider declares how visibility reads when no data profile is given (ADR-109; C2 r2 C-06)
        contacts = detect_ball_contacts(frames, data_profile=data_profile, provider=None if data_profile else provider)

    tau = RATIFIED["tau_fc_ceiling_s"]
    clip = float(RATIFIED["bracket_clip_s"])
    ab: dict[str, tuple[float, float]] = dict(bracket_ab or {})
    frames_by_period = {_i(p): g for p, g in frames.groupby("period_id")} if len(frames) else {}
    # per-(period) frame_id -> time_seconds (ball rows carry the canonical frame time)
    frame_time_lut: dict[int, dict] = {}
    for p, g in frames_by_period.items():
        frame_time_lut[p] = dict(zip(g["frame_id"].astype(int), g["time_seconds"].astype(float), strict=False))
    # ADR-068/073: group the contacts by period ONCE (O(1) per-restart lookup) and build each period's
    # player->team roster ONCE, so the per-restart loop never re-filters the whole contact table or rebuilds
    # the roster (the rescan-in-loop the Task 2.5 guard forbids). resolve_kicks is registered in SCALE_GUARDED.
    contacts_groups = group_rows(contacts, "period_id") if contacts is not None and len(contacts) else None
    roster_by_period: dict[int, dict] = {p: _build_roster(g) for p, g in frames_by_period.items()}
    # Fetch + sort each period's contacts ONCE (the group_rows .get is O(periods), not O(restarts)); the
    # per-restart window is a `searchsorted` slice of the pre-sorted times (O(log) + a few rows), never a
    # re-filter of the whole table. `contacts_by_period[p] = (sorted_frame, time_array)`.
    contacts_by_period: dict[int, tuple] = {}
    if contacts_groups is not None:
        for p in frames_by_period:
            cp = contacts_groups.get(p).sort_values("time_seconds").reset_index(drop=True)
            contacts_by_period[p] = (cp, cp["time_seconds"].to_numpy(dtype=float))
    # Each period's ball rows, sorted ONCE, so the per-restart observation window is a `searchsorted` slice
    # (O(log) + the window) rather than a full re-filter of the period ball trace (ADR-068; the Task 2.5 guard).
    ball_by_period: dict[int, tuple] = {}
    det_speed_by_period: dict[int, tuple] = {}
    for p, g in frames_by_period.items():
        bg = g[g["is_ball"].to_numpy(dtype=bool)].sort_values("time_seconds").reset_index(drop=True)
        ball_by_period[p] = (bg, bg["time_seconds"].to_numpy(dtype=float))
        det_speed_by_period[p] = _detected_ball_speed(bg, provider)

    chain_bounds = _chain_upper_bounds(out, is_restart, provider)
    for idx in np.where(is_restart)[0]:
        row = out.iloc[idx]
        gid = row["game_id"]
        period = int(row["period_id"])
        gp = frames_by_period.get(period)

        # --- gi_pairing (SkillCorner: every row of the match) ---
        pairing = gi_pairing.get(gid) if (is_sc and gi_pairing) else None
        if is_sc and pairing is not None:
            out.iat[int(idx), column_position(out, "gi_pairing")] = pairing.pairing
            out.iat[int(idx), column_position(out, "gi_pairing_failed_test")] = pairing.failed_test
        unpaired = bool(is_sc and pairing is not None and pairing.pairing == "unpaired")

        # --- bracket ---
        pair = ab.get(_restart_base(row))
        a_raw, b_raw = pair if pair is not None else (clip, clip)
        a, b = min(float(a_raw), clip), min(float(b_raw), clip)
        lo, hi, rest_xy = _bracket(row, provider, period, frame_time_lut, a, b, report, chain_bounds.get(int(idx)))
        out.iat[int(idx), column_position(out, "kick_bracket_lo")] = lo
        out.iat[int(idx), column_position(out, "kick_bracket_hi")] = hi
        if is_sc and not pd.isna(row.get("gi_next_possession_frame_start")):
            out.iat[int(idx), column_position(out, "gi_next_possession_time")] = _frame_time(
                row.get("gi_next_possession_frame_start"), frame_time_lut.get(period, {})
            )

        # A withheld-by-pairing SkillCorner match: bracket-dependent outputs are withheld for THAT match only.
        if unpaired or pd.isna(lo) or pd.isna(hi) or gp is None:
            out.iat[int(idx), column_position(out, "kick_time_source")] = (
                "none" if (pd.isna(lo) or pd.isna(hi)) else "withheld"
            )
            _taker_passthrough(out, idx, row, provider, gated=False)
            continue

        # --- candidates ---
        restart_base = _restart_base(row)
        obs_lo, obs_hi = lo - tau, hi + tau
        _ball_sorted, _ball_times = ball_by_period.get(period, (gp.iloc[:0], np.empty(0)))
        ball_obs = _ball_trace(_ball_sorted, _ball_times, obs_lo, obs_hi)
        # Section 6.4: the rest location / release direction are evaluated in the restarting team's action frame
        # (GoalMap direction, ADR-055; <NA> fails a corner / goal kick closed). One episode list serves K and the
        # retake rule (section 6.7).
        rtl_row = team_attacks_rtl(out.iloc[[int(idx)]], goal_map).iloc[0]
        rtl_flag = None if pd.isna(rtl_row) else bool(rtl_row)
        episodes = (
            rest_episodes(ball_obs, restart_type=restart_base, attacks_rtl=rtl_flag, pitch=dims)
            if len(ball_obs) and restart_base in _DEAD_BALL
            else []
        )
        if restart_base in _DEAD_BALL:
            verdict = kick_verdict(episodes)
            t_onset, rest_med = verdict.release, verdict.rest_xy
            # Counted by SPADL type (= restart type x subtype): >= 2 rests with no strike (`none`), and no qualifying
            # rest at all (a section 6.4 location miss is never silent; C2 r2 C-25).
            bucket = {"no_strike_ambiguous": "k_no_strike_ambiguous", "no_episode": "k_no_episode"}.get(verdict.reason)
            if bucket is not None:
                key = spadlconfig.actiontypes[int(row["type_id"])]
                report[bucket][key] = report[bucket].get(key, 0) + 1
        else:
            t_onset, rest_med = (
                kinematic_release(ball_obs, restart_type=restart_base) if len(ball_obs) else (float("nan"), None)
            )
        period_contacts = contacts_by_period.get(period)
        roster = roster_by_period.get(period)
        t_contact = _kick_contact(
            row,
            period_contacts,
            roster,
            lo,
            hi,
            restart_base,
            provider,
            goal_map,
            ball_obs=ball_obs,
            t_anchor=t_onset,
            rest_xy=rest_med,
            det_speed=det_speed_by_period.get(period),
        )
        if rest_xy is None:
            rest_xy = rest_med
        # Retake (section 6.7): a release, a return to rest within rho_retake, a second release -> t_kick follows
        # the arbitration on the LAST onset; cell_state "retake" (never pooled). Detected structurally (not gated).
        is_retake, last_onset = detect_retake(episodes, restart_base, lo, hi)
        out.iat[int(idx), column_position(out, "kick_retake")] = bool(is_retake)
        if is_retake:
            t_onset = last_onset
        out.iat[int(idx), column_position(out, "t_contact")] = t_contact
        out.iat[int(idx), column_position(out, "t_onset")] = t_onset
        if np.isfinite(t_contact) and np.isfinite(t_onset):
            out.iat[int(idx), column_position(out, "kick_time_delta")] = t_contact - t_onset
        if rest_xy is not None:
            # CL-11: kick_rest_x/_y are in the restarting team's ACTION frame (section 4.3; it attacks +x). The
            # rest median is measured on raw frame coords, so map it through the GoalMap direction; an
            # unresolved direction (<NA>) gives NaN (never a guessed orientation).
            ax, ay = to_action_frame(
                np.array([rest_xy[0]]), np.array([rest_xy[1]]), pd.Series([rtl_row], dtype="boolean")
            )
            out.iat[int(idx), column_position(out, "kick_rest_x")] = float(ax[0])
            out.iat[int(idx), column_position(out, "kick_rest_y")] = float(ay[0])

        # --- gate + arbitration ---
        cell_state = "retake" if is_retake else "default"
        entry = (
            gate(data_profile, restart_base, "kick_time", cell_state, table=gate_table)
            if data_profile is not None
            else None
        )
        t_kick, source, resolution = _arbitrate(
            t_contact,
            t_onset,
            lo,
            hi,
            tau,
            gated=entry is not None,
            winner=getattr(entry, "agreement_winner", None),
        )
        out.iat[int(idx), column_position(out, "t_kick")] = t_kick
        out.iat[int(idx), column_position(out, "kick_time_source")] = source
        out.iat[int(idx), column_position(out, "kick_time_resolution")] = resolution
        out.iat[int(idx), column_position(out, "kick_time_gate_version")] = GATE_TABLE_VERSION
        if entry is not None:
            out.iat[int(idx), column_position(out, "kick_time_evidence")] = entry.tier
            out.iat[int(idx), column_position(out, "kick_time_cell_resolution")] = entry.cell_resolution

        # The SkillCorner taker has its OWN gate (spec section 15.2: signal "taker", cell = restart type x
        # subtype), independent of the kick_time cell.
        taker_entry = (
            gate(data_profile, restart_base, "taker", taker_cell_state(row), table=gate_table)
            if data_profile is not None
            else None
        )
        _taker_passthrough(
            out,
            idx,
            row,
            provider,
            gated=taker_entry is not None,
            contact_player=_contact_player(row, period_contacts, t_contact, tau, provider),
        )
        if taker_entry is not None and out.iat[int(idx), column_position(out, "taker_source")] == "tracking":
            out.iat[int(idx), column_position(out, "taker_evidence")] = taker_entry.tier
            out.iat[int(idx), column_position(out, "taker_cell_resolution")] = taker_entry.cell_resolution
            out.iat[int(idx), column_position(out, "taker_gate_version")] = GATE_TABLE_VERSION

    out.attrs["kick_resolver_report"] = report
    return out


def _empty_report() -> dict:
    return {
        "gi_frame_unmapped": 0,
        "gi_unpaired": {},
        "k_no_strike_ambiguous": {},
        "k_no_episode": {},
    }


def _merge_report(into: dict, rep: dict) -> None:
    """Fold one match's ``kick_resolver_report`` into the combined one (counts add, per-key maps add / update)."""
    into["gi_frame_unmapped"] += int(rep.get("gi_frame_unmapped", 0))
    into["gi_unpaired"].update(rep.get("gi_unpaired", {}))
    for key in ("k_no_strike_ambiguous", "k_no_episode"):
        for k_, v_ in rep.get(key, {}).items():
            into[key][k_] = into[key].get(k_, 0) + int(v_)


def _resolve_per_match(actions, frames, *, contacts, **kw) -> pd.DataFrame:
    """``resolve_kicks`` on each match separately (its own frames and contacts), recombined in the input row order
    with one merged report. Actions, frames and contacts are partitioned by ``game_id`` ONCE (``group_rows``,
    ADR-068; ids canonical, ADR-019) -- never a full-table mask per game."""
    if contacts is not None and len(contacts) and "game_id" not in contacts.columns:
        raise ValueError("resolve_kicks: a multi-match input needs per-match contacts (a game_id column)")
    a_groups = group_rows(actions, "game_id")
    f_groups = group_rows(frames, "game_id")
    c_groups = group_rows(contacts, "game_id") if contacts is not None and len(contacts) else None
    parts, report = [], _empty_report()
    for gid in a_groups.keys():
        c = c_groups.get(gid) if c_groups is not None else None
        r = resolve_kicks(a_groups.get(gid), f_groups.get(gid), contacts=c, **kw)
        _merge_report(report, r.attrs.get("kick_resolver_report", {}))
        parts.append(r)
    out = pd.concat(parts).loc[actions.index]
    out.attrs["kick_resolver_report"] = report
    return out


def _base_of_name(name: str) -> str:
    if name.startswith("corner"):
        return "corner"
    if name.startswith("freekick"):
        return "free_kick"
    if name == "goalkick":
        return "goal_kick"
    if name == "throw_in":
        return "throw_in"
    if name == "shot_penalty":
        return "penalty"
    return name


#: SPADL ``type_id`` -> restart base, built once (not an English-name lookup per row).
_RESTART_BASE_BY_TYPE = {tid: _base_of_name(name) for tid, name in enumerate(spadlconfig.actiontypes)}


def _restart_base(row) -> str:
    return _RESTART_BASE_BY_TYPE.get(int(row["type_id"]), "")


def _chain_upper_bounds(out: pd.DataFrame, is_restart: np.ndarray, provider: str) -> dict[int, object]:
    """Spec section 6.1 chain rows: a SkillCorner derived restart whose next possession is itself a restart (Part-1
    ``restart_evidence`` ``chain`` / ``chain_ambiguous``) carries no next-possession frame id, so its kick lies
    BETWEEN the chain members' brackets: its upper bound is the NEXT chain member's lower bound (that member's
    marker ``frame_end``, which precedes the member's own kick). Returns ``{row position: frame id}``."""
    if provider != "skillcorner" or "restart_evidence" not in out.columns:
        return {}
    derived = is_restart & (out.get("action_provenance", pd.Series(index=out.index)) == "derived").to_numpy()
    pos = np.flatnonzero(derived)
    order = sorted(pos, key=lambda i: (int(out["period_id"].iloc[i]), float(out["time_seconds"].iloc[i])))
    bounds: dict[int, object] = {}
    for a, b in pairwise(order):
        ra, rb = out.iloc[a], out.iloc[b]
        if int(ra["period_id"]) != int(rb["period_id"]):
            continue
        if ra.get("restart_evidence") in ("chain", "chain_ambiguous") and pd.isna(
            ra.get("gi_next_possession_frame_start")
        ):
            nxt = rb.get("gi_marker_possession_frame_end")
            if nxt is not None and not pd.isna(nxt):
                bounds[int(a)] = nxt
    return bounds


def _bracket(row, provider, period, frame_time_lut, a, b, report, chain_hi=None):
    """``(lo, hi, rest_xy)`` for one restart row (section 6.1). rest_xy is None here (filled by onset).
    ``chain_hi`` = the frame id bounding a chain row from above (``_chain_upper_bounds``)."""
    if provider == "skillcorner" and row.get("action_provenance") == "derived":
        lut = frame_time_lut.get(period, {})
        fs_marker = row.get("gi_marker_possession_frame_end")
        fs_next = row.get("gi_next_possession_frame_start")
        if (fs_next is None or pd.isna(fs_next)) and chain_hi is not None:
            fs_next = chain_hi  # a chain row: bounded by the next chain member
        if fs_marker is None or pd.isna(fs_marker) or fs_next is None or pd.isna(fs_next):
            return float("nan"), float("nan"), None
        lo = lut.get(int(fs_marker), float("nan"))
        hi = lut.get(int(fs_next), float("nan"))
        if pd.isna(lo) or pd.isna(hi):
            report["gi_frame_unmapped"] += 1  # an in-range id absent from frames (a gap)
            return float("nan"), float("nan"), None
        return float(lo), float(hi), None
    # native anchor: [t_event - a, t_event + b]
    t_event = float(row["time_seconds"])
    return t_event - a, t_event + b, None


def _build_roster(gp) -> dict:
    """Per-period player -> team roster (canonical ids, ADR-019), built ONCE per period (ADR-068)."""
    from silly_kicks.id_compat import canonical_id_series

    if gp is None or not len(gp):
        return {}
    return dict(
        zip(
            canonical_id_series(gp["player_id"]).to_numpy(dtype=object),
            canonical_id_series(gp["team_id"]).to_numpy(dtype=object),
            strict=False,
        )
    )


def _detected_ball_speed(ball: pd.DataFrame, provider: str) -> tuple[np.ndarray, np.ndarray]:
    """``(times, speed)`` of the DETECTED ball frames of one period (sorted), built ONCE per period (ADR-068). The
    SkillCorner launch rule reads detected balls only (spec section 6.3; an extrapolated ball never counts,
    ADR-109). A provider without a detection flag keeps every ball frame."""
    from ._provider_visibility import detected_mask

    if ball.empty:
        return np.empty(0), np.empty(0)
    if "visibility" in ball.columns and ball["visibility"].notna().any():
        keep = detected_mask(ball["visibility"], provider=provider if provider == "skillcorner" else "skillcorner")
        ball = ball[keep]
    t = ball["time_seconds"].to_numpy(dtype=np.float64)
    if len(t) < 2:
        return t, np.zeros(len(t))
    return t, _speed(t, ball["x"].to_numpy(dtype=np.float64), ball["y"].to_numpy(dtype=np.float64))


_FIELD_L, _FIELD_W = float(spadlconfig.field_length), float(spadlconfig.field_width)


def _in_restart_region(base: str, ax: np.ndarray, ay: np.ndarray, rest_axy) -> np.ndarray:
    """The section 6.2 restart region in the restarting team's ACTION frame (it attacks +x): corner -- within
    ``corner_region_m`` of an attacked-end flag; goal kick -- its own penalty area; throw-in -- within
    ``throwin_region_m`` of a touchline; free kick (optical) -- within ``freekick_region_m`` of the stationary-
    window centroid (no centroid -> no region). Other restarts: everywhere."""
    ax = np.asarray(ax, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)
    if base == "corner":
        r = RATIFIED["corner_region_m"]
        return (np.hypot(ax - _FIELD_L, ay) <= r) | (np.hypot(ax - _FIELD_L, ay - _FIELD_W) <= r)
    if base == "goal_kick":
        return (ax <= 16.5) & (np.abs(ay - _FIELD_W / 2.0) <= spadlconfig.penalty_area_half_width)
    if base == "throw_in":
        r = RATIFIED["throwin_region_m"]
        return (ay <= r) | (ay >= _FIELD_W - r)
    if base == "free_kick":
        if rest_axy is None or not np.isfinite(rest_axy[0]):
            return np.zeros(len(ax), dtype=bool)
        return np.hypot(ax - rest_axy[0], ay - rest_axy[1]) <= RATIFIED["freekick_region_m"]
    return np.ones(len(ax), dtype=bool)


def _kick_contact(
    row,
    period_contacts,
    roster,
    lo,
    hi,
    restart_base,
    provider,
    goal_map,
    *,
    ball_obs: pd.DataFrame | None = None,
    t_anchor: float = float("nan"),
    rest_xy=None,
    det_speed: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """``t_contact`` (spec sections 6.2, 6.3; C-31, C-41).

    Dead-ball kicks: the LAST contact of the FIRST restarting-team player among the IN-REGION contacts after the
    ball's last entry into the restart region, before any other player's in-region contact. The region and every
    position are evaluated in the restarting team's action frame (section 4.3, GoalMap direction; an unresolved
    direction fails closed: NaN). The ball's last entry is the last outside -> inside transition at or before the
    release ``t_anchor`` (the kinematic onset; ``hi`` when there is none), so a retriever who brings the ball in
    from outside the region and a short-corner receiver (a different player) are never the kick. SkillCorner free
    kicks use the launch rule (``_launch_contact``).

    ``period_contacts`` is ``(sorted_frame, time_array)`` for the row's period and ``roster`` the period's
    prebuilt player->team map (ADR-068: the windows are ``searchsorted`` slices, never re-filters)."""
    if period_contacts is None:
        return float("nan")
    cp, times = period_contacts
    if len(cp) == 0:
        return float("nan")
    team = row.get("team_id")
    left = int(np.searchsorted(times, lo, side="left"))
    right = int(np.searchsorted(times, hi, side="right"))
    c = cp.iloc[left:right]
    if c.empty:
        return float("nan")
    if provider == "skillcorner" and restart_base == "free_kick":
        return _launch_contact(c, roster, team, det_speed)
    from silly_kicks.id_compat import canonical_id_series

    rtl = team_attacks_rtl(
        pd.DataFrame([{"game_id": row.get("game_id"), "period_id": row["period_id"], "team_id": team}]), goal_map
    )
    if pd.isna(rtl.iloc[0]):
        return float("nan")  # no direction -> no region test -> fail closed (never a guessed orientation)
    rest_axy = None
    if rest_xy is not None:
        rx, ry = to_action_frame(np.array([rest_xy[0]]), np.array([rest_xy[1]]), rtl)
        rest_axy = (float(rx[0]), float(ry[0]))
    # the ball's last entry into the region, at or before the release
    entry_t = float(lo)
    if ball_obs is not None and len(ball_obs):
        bt = ball_obs["time_seconds"].to_numpy(dtype=np.float64)
        bx, by = to_action_frame(
            ball_obs["x"].to_numpy(dtype=np.float64),
            ball_obs["y"].to_numpy(dtype=np.float64),
            pd.Series([rtl.iloc[0]] * len(ball_obs), dtype="boolean"),
        )
        anchor = t_anchor if np.isfinite(t_anchor) else float(hi)
        inside = _in_restart_region(restart_base, bx, by, rest_axy) & (bt <= anchor)
        entries = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
        if len(entries):
            entry_t = float(bt[entries[-1]])
    c = c[(c["time_seconds"].to_numpy(dtype=np.float64) >= entry_t) & c["player_id"].notna().to_numpy()]
    if c.empty:
        return float("nan")
    cx, cy = to_action_frame(
        c["contact_x"].to_numpy(dtype=np.float64),
        c["contact_y"].to_numpy(dtype=np.float64),
        pd.Series([rtl.iloc[0]] * len(c), dtype="boolean"),
    )
    c = c[_in_restart_region(restart_base, cx, cy, rest_axy)]
    if c.empty:
        return float("nan")
    tmask = _contact_team_mask(c, roster, team)
    if not tmask.any():
        return float("nan")
    pids = canonical_id_series(c["player_id"]).to_numpy(dtype=object)
    first = int(np.flatnonzero(tmask)[0])
    taker = pids[first]
    last_t = float("nan")
    for k in range(first, len(c)):
        if pids[k] != taker:
            break  # another player's in-region contact ends the taker's sequence (a receiver is never the kick)
        last_t = float(c["time_seconds"].iloc[k])
    return last_t


def _contact_team_mask(c, roster, team) -> np.ndarray:
    """Boolean mask: contacts whose attributed player belongs to ``team`` (via the PREBUILT period roster)."""
    from silly_kicks.id_compat import canonical_id, canonical_id_series

    if not roster or team is None or pd.isna(team):
        return np.ones(len(c), dtype=bool)
    target = canonical_id(team)
    cteam = [roster.get(p) for p in canonical_id_series(c["player_id"]).to_numpy(dtype=object)]
    # NA-safe: an unresolved / other-team contact is False, never pd.NA (a mask must not carry NA).
    return np.array([(v is not None and not pd.isna(v) and v == target) for v in cteam], dtype=bool)


def _launch_contact(c, roster, team, det_speed) -> float:
    """SkillCorner free-kick launch rule (spec section 6.3): the restarting-team contact that launches the peak
    DETECTED-ball speed inside the strict bracket, with launch speed >= ``v_kick`` (8 m/s); a second launch within
    15 % of the peak is ambiguous -> NaN. ``det_speed`` = the period's ``(times, speed)`` of detected balls, built
    once (``_detected_ball_speed``)."""
    tmask = _contact_team_mask(c, roster, team)
    cteam = c[tmask]
    if cteam.empty or det_speed is None or len(det_speed[0]) < 2:
        return float("nan")
    bt, speed = det_speed
    v_kick = RATIFIED["sc_freekick_vkick_ms"]
    ambig = RATIFIED["sc_freekick_ambiguous_share"]
    launches = []
    for ct in cteam["time_seconds"].to_numpy(dtype=np.float64):
        # the speed the contact LAUNCHES: the frames AFTER it (speed[j] is the motion arriving at frame j, i.e. the
        # previous launch -- crediting it would hand a re-touch the earlier kick's speed)
        j = int(np.searchsorted(bt, ct, side="right"))
        peak = float(np.max(speed[j : min(j + 5, len(speed))])) if j < len(speed) else 0.0
        if peak >= v_kick:
            launches.append((peak, float(ct)))
    if not launches:
        return float("nan")
    launches.sort(reverse=True)
    if len(launches) >= 2 and launches[1][0] >= (1 - ambig) * launches[0][0]:
        return float("nan")  # ambiguous launch
    return launches[0][1]


def taker_cell_state(row) -> str:
    """The ``taker`` gate cell state (spec section 15.2: restart type x subtype): ``short`` / ``crossed`` for a
    corner or free kick, ``default`` for a goal kick or a throw-in.

    Examples
    --------
    >>> from silly_kicks.spadl import config as spadlconfig
    >>> from silly_kicks.tracking._kick_resolver import taker_cell_state
    >>> taker_cell_state({"type_id": spadlconfig.actiontype_id["corner_crossed"]})
    'crossed'
    """
    tid = int(row["type_id"])
    name = spadlconfig.actiontypes[tid] if tid < len(spadlconfig.actiontypes) else ""
    if name.endswith("_short"):
        return "short"
    if name.endswith("_crossed"):
        return "crossed"
    return "default"


def _contact_player(row, period_contacts, t_contact, tau, provider):
    """The attributed player of the kick contact (for the SkillCorner taker); None off SkillCorner / unresolved.

    ``period_contacts`` is the period's ``(sorted contacts, time array)`` built ONCE (ADR-068): the window
    ``|t - t_contact| <= tau`` is a ``searchsorted`` slice, never a re-filter of the whole contact table."""
    if provider != "skillcorner" or period_contacts is None or not np.isfinite(t_contact):
        return None
    cp, times = period_contacts
    lo_i = int(np.searchsorted(times, t_contact - tau, side="left"))
    hi_i = int(np.searchsorted(times, t_contact + tau, side="right"))
    c = cp.iloc[lo_i:hi_i]
    c = c[c["player_id"].notna()]
    if c.empty:
        return None
    return c.iloc[int(np.argmin(np.abs(c["time_seconds"].to_numpy() - t_contact)))]["player_id"]


def _taker_passthrough(out, idx, row, provider, *, gated: bool, contact_player=None) -> None:
    """Section 7 taker. Event providers pass through ``player_id`` (ungated); SkillCorner takes the kick
    contact's player, gated (``withheld`` when the gate fails)."""
    if provider != "skillcorner":
        out.iat[int(idx), column_position(out, "taker_player_id")] = row.get("player_id")
        out.iat[int(idx), column_position(out, "taker_source")] = "native"
        return
    if not gated or contact_player is None or pd.isna(contact_player):
        out.iat[int(idx), column_position(out, "taker_source")] = "withheld"
        return
    out.iat[int(idx), column_position(out, "taker_player_id")] = contact_player
    out.iat[int(idx), column_position(out, "taker_source")] = "tracking"
