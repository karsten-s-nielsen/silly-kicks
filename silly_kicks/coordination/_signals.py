"""Signal preparation for coordination (spec 7.4): filtered, resampled, oriented team and player series.

``build_coordination_signals`` performs the refusals (spec 7.3), derives the effective rate, resolves
stoppages/detection/goal map, and per (game, period) produces filtered+resampled team signals (via the
collective kernel) and per-player phasor series. Everything positional is expressed in TEAM A's
goal-relative frame (C24); Task 17 applies per-window reference-team sign flips on top.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.coordination._catalog import PairSpec  # noqa: F401  (re-exported convenience for consumers)
from silly_kicks.coordination._columns import TEAM_SIGNALS
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._detection import DetectionCounts
from silly_kicks.coordination._kernels._phase import analytic_phase, pad_length, phase_advance_indicator, phasor
from silly_kicks.coordination._windows import resolve_stoppages, validate_windows
from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.tracking import GoalMap, collective_from_positions, resolve_defended_goals
from silly_kicks.tracking._collective import back_line_batch, compact_rows  # array kernels (not public seams)
from silly_kicks.tracking._geometry import to_goal_relative_x_array, to_goal_relative_y_array
from silly_kicks.tracking._provider_visibility import (
    _DETECTION_AWARE_PROVIDERS,
    detected_mask,
    validate_provider,
)
from silly_kicks.tracking.preprocess._butterworth import (
    butterworth_lowpass_rows,
    butterworth_min_length,
    grid_span,
)

_REQUIRED_COLUMNS = (
    "game_id",
    "period_id",
    "frame_id",
    "time_seconds",
    "frame_rate",
    "player_id",
    "team_id",
    "is_ball",
    "is_goalkeeper",
    "x",
    "y",
    "ball_state",
    "team_attacking_direction",
    "source_provider",
)
#: The columns the per-period build reads (``_prepare_period`` and the per-player slices), plus ``_detected``.
_PREP_COLUMNS = (
    "game_id",
    "period_id",
    "time_seconds",
    "frame_rate",
    "player_id",
    "team_id",
    "is_ball",
    "is_goalkeeper",
    "x",
    "y",
)


def _f(v) -> float:
    """pandas Scalar (itertuples/iloc) -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


def _is_na(v) -> bool:
    """``pd.isna`` on a Scalar/object without the stub's parameter-type friction."""
    return bool(pd.isna(v))


@dataclass(frozen=True)
class PlayerSeries:
    """One player's filtered/resampled/oriented series over a period.

    Examples
    --------
    Read one player's prepared arrays from a built signals object (needs a real match)::

        player = signals.periods[0].players[(team_id, player_id)]
        player.x           # oriented x on the effective grid; player.phasor_x is its analytic phasor
    """

    team_id: object
    player_id: object
    is_goalkeeper: bool
    x: np.ndarray
    y: np.ndarray
    runs: np.ndarray  # (R, 2) int [start, end)
    phasor_x: np.ndarray
    phasor_y: np.ndarray
    inst_freq_pos_x: np.ndarray
    inst_freq_pos_y: np.ndarray
    observed: np.ndarray
    on_pitch: np.ndarray
    detection: DetectionCounts  # raw-detected / on-pitch prefix counts: the player side's detection share (A-08)


@dataclass(frozen=True)
class PeriodSignals:
    """Team + player signals for one (game, period) on the effective grid ``t = arange(N)/fs``.

    Examples
    --------
    Read one (game, period)'s team series from a built signals object (needs a real match)::

        ps = signals.periods[0]
        ps.team_signal[(ps.team_ids[0], "centroid_x")]   # team A's centroid-x on ps.t
    """

    game_id: object
    period_id: int
    t: np.ndarray
    team_ids: tuple[object, object]
    goal_x: Mapping[object, float | None]
    reference_flip: bool
    segments: Mapping[object, np.ndarray]
    segment_id: Mapping[object, np.ndarray]
    team_signal: Mapping[tuple[object, str], np.ndarray]
    team_phasor: Mapping[tuple[object, str], np.ndarray]
    team_inst_freq_pos: Mapping[tuple[object, str], np.ndarray]
    observed_fraction: Mapping[object, np.ndarray]
    team_detection: Mapping[object, DetectionCounts]  # each team side's detection share (A-08)
    possession_team: np.ndarray
    window_ranges: np.ndarray
    players: Mapping[tuple[object, object], PlayerSeries]


@dataclass(frozen=True)
class CoordinationSignals:
    """All periods' prepared signals plus the run-level provenance a compute pass needs.

    Examples
    --------
    The object every family compute consumes (needs a real match)::

        signals = build_coordination_signals(frames, windows=period_windows(frames))
        pair, phase, report = compute_relative_phase(signals)
    """

    provider: str
    params: CoordinationParams
    native_hz: float
    fs: float
    rate_capped: bool
    windows: pd.DataFrame
    window_regime: str
    detection_source: str
    stoppage: Any
    periods: tuple[PeriodSignals, ...]
    counters: Mapping[str, int]


# --------------------------------------------------------------------------- refusals (spec 7.3 order)
def _refuse(frames: pd.DataFrame, windows: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in frames.columns]
    if missing:
        raise ValueError(f"frames missing required columns: {missing}")
    providers = set(frames["source_provider"].astype(object).unique())
    if "snapshot" in providers:
        raise ValueError("source_provider 'snapshot' (freeze frames) is refused at the coordination edge (spec 7.3)")
    if len(providers) > 1:
        raise ValueError(f"frames mix source_provider values {sorted(providers)}; one provider per call (spec 7.3)")
    direction = frames["team_attacking_direction"]
    if direction.isna().all():
        raise ValueError("frames are unoriented (team_attacking_direction all null); orient first (spec 7.3)")
    is_ball = frames["is_ball"].to_numpy(dtype=bool)
    # duplicate checks read only their key columns (a full-width boolean take of every column was the cost)
    if frames.loc[~is_ball, ["game_id", "period_id", "frame_id", "player_id"]].duplicated().any():
        raise ValueError("duplicate (game, period, frame, player_id) among non-ball rows (spec 7.3)")
    if frames.loc[is_ball, ["game_id", "period_id", "frame_id"]].duplicated().any():
        raise ValueError("more than one ball row per frame (spec 7.3)")
    if frames["frame_rate"].nunique() != 1:
        raise ValueError("frame_rate is not single-valued for the call (spec 7.3)")
    validate_windows(windows)


# --------------------------------------------------------------------------- per-player run processing
def _split_indices(present_t: np.ndarray, fs_native: float, gap_s: float) -> list[tuple[int, int]]:
    """Runs of consecutive present samples, split where the time gap exceeds ``gap_s``."""
    if present_t.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(present_t) > gap_s + 0.5 / fs_native)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [present_t.size]))
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=False)]


def _bridge_detection_gaps(seg_t: np.ndarray, seg_v: np.ndarray, fs_native: float) -> tuple[np.ndarray, np.ndarray]:
    """A run's detection gaps bridged by linear interpolation at the native rate (spec 7.4 step 2), so the filter that
    follows (step 3) sees evenly spaced samples -- filtering the detected samples as if they were evenly spaced
    distorted every bridged gap by up to metres (review A-07).

    ``seg_v`` holds one row per coordinate. An interval spanning ``m >= 2`` native steps gets ``m - 1`` evenly spaced
    samples, linear between its two detected endpoints; detected samples keep their exact times and values, so a
    gap-free run comes back unchanged (bit for bit)."""
    dt = np.diff(seg_t)
    steps = np.rint(dt * fs_native).astype(np.int64)
    if not (steps >= 2).any():
        return seg_t, seg_v
    counts = np.maximum(steps, 1)  # the samples each interval contributes, its left endpoint included
    left = np.repeat(np.arange(steps.size), counts)
    frac = (np.arange(left.size) - np.repeat(np.cumsum(counts) - counts, counts)) / np.repeat(counts, counts)
    t_new = np.concatenate((seg_t[left] + frac * dt[left], seg_t[-1:]))
    v_new = np.concatenate((seg_v[:, left] + frac * (seg_v[:, left + 1] - seg_v[:, left]), seg_v[:, -1:]), axis=1)
    return t_new, v_new


def _resample_runs(
    t_present: np.ndarray,
    columns: tuple[np.ndarray, ...],
    fs: float,
    fs_native: float,
    cutoff: float,
    order: int,
    grid_n: int,
    gap_s: float,
    counters: dict[str, int],
    *,
    apply_filter: bool = True,
) -> tuple[tuple[np.ndarray, ...], list[tuple[int, int]]]:
    """Bridge each present-run's detection gaps at the native rate (:func:`_bridge_detection_gaps`, spec 7.4 step 2),
    filter it (step 3; x, y share the run split) and resample it onto the ``arange(grid_n)/fs`` grid; NaN outside runs.
    A run whose BRIDGED length -- the array the filter sees -- is shorter than the filter's pad length is dropped and
    counted ONCE (spec 7.4 step 3).

    Resampling is ONE linear interpolation of the filtered run at the period-grid times it covers (spec 7.4 step
    4); :func:`grid_span` decides which grid points those are, with the float-tolerant edge rule shared with
    ``resample_uniform``. The former two-stage path (a run-local grid, then re-interpolated onto the period grid
    with NaN outside) lost edge samples to float rounding, which dropped the whole run's phase, and
    double-interpolated runs that start off the grid (TF-58 F7).

    ``apply_filter=False`` skips the zero-phase Butterworth and resamples the RAW run instead, keeping the
    SAME run split and ``butterworth_min_length`` admission so the raw and filtered builds align element-wise
    on the grid (the D1 Tier-B ``vc_epsilon`` seam, §8.2). The default preserves the filtered behaviour bit
    for bit.
    """
    outs = tuple(np.full(grid_n, np.nan) for _ in columns)
    run_bounds_grid: list[tuple[int, int]] = []
    min_len = butterworth_min_length(fs_native, cutoff, order)
    for s, e in _split_indices(t_present, fs_native, gap_s):
        seg_t, seg_v = _bridge_detection_gaps(t_present[s:e], np.stack([values[s:e] for values in columns]), fs_native)
        if seg_t.size < min_len:
            counters["n_runs_too_short"] += 1
            continue
        lo, hi = grid_span(float(seg_t[0]), float(seg_t[-1]), fs)
        lo, hi = max(lo, 0), min(hi, grid_n)
        if hi <= lo:
            continue
        grid_t = np.arange(lo, hi) / fs
        # the run's coordinates filtered in one pass: each row equals its own 1-D butterworth_lowpass bit for bit.
        # prewarp=True so the combined dual-pass -3 dB lands exactly at `cutoff` even at a high derived cutoff (A-48)
        filtered = butterworth_lowpass_rows(seg_v, fs_native, cutoff, order, prewarp=True) if apply_filter else seg_v
        for out, row in zip(outs, filtered, strict=True):
            out[lo:hi] = np.interp(grid_t, seg_t, row)
        run_bounds_grid.append((lo, hi))
    return outs, run_bounds_grid


def _phasors_over_runs(
    values: np.ndarray, runs: list[tuple[int, int]], fs: float, band_low_cpm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic phasor + positive-instantaneous-frequency indicator, computed once per run (D15)."""
    n = values.size
    ph = np.full(n, np.nan + 1j * np.nan, dtype=np.complex128)
    inst_pos = np.full(n, np.nan)  # float, NaN at each run's first sample (A-45): never counted non-advancing
    for lo, hi in runs:
        seg = values[lo:hi]
        if seg.size < 2 or not np.isfinite(seg).all():
            continue
        theta = analytic_phase(seg, pad_length(seg.size, fs, band_low_cpm))
        ph[lo:hi] = phasor(theta)
        inst_pos[lo:hi] = phase_advance_indicator(theta)
    return ph, inst_pos


# --------------------------------------------------------------------------- period build
def _possession_series(windows_gp: pd.DataFrame, t: np.ndarray, grid_n: int) -> np.ndarray:
    """Attacking team of the possession window covering each sample, held forward; NA before the first."""
    out = np.full(grid_n, pd.NA, dtype=object)
    poss = windows_gp[windows_gp["window_source"].isin(["possession_events", "possession_tracking"])]
    for w in poss.sort_values("start_time_s").itertuples(index=False):
        s = int(np.searchsorted(t, _f(w.start_time_s), side="left"))
        e = int(np.searchsorted(t, _f(w.end_time_s), side="left"))
        out[s:e] = w.attacking_team_id
    last: object = pd.NA
    for i in range(grid_n):  # forward-fill: hold the last value between windows (spec 7.8.4)
        if not _is_na(out[i]):
            last = out[i]
        elif not _is_na(last):
            out[i] = last
    return out


def _window_ranges(windows_gp: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    rows = [
        [
            int(idx),
            int(np.searchsorted(t, _f(w.start_time_s), side="left")),
            int(np.searchsorted(t, _f(w.end_time_s), side="left")),
        ]
        for idx, w in zip(windows_gp.index, windows_gp.itertuples(index=False), strict=False)
    ]
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def _prepare_period(
    gp: pd.DataFrame,
    *,
    windows_gp: pd.DataFrame,
    params: CoordinationParams,
    fs: float,
    gmap: GoalMap,
    provider: str,
    stoppage: Any,
    counters: dict[str, int],
    apply_filter: bool = True,
) -> PeriodSignals:
    game = gp["game_id"].iloc[0]
    period = int(gp["period_id"].iloc[0])
    fs_native = float(gp["frame_rate"].iloc[0])
    cutoff, order = params.butterworth_cutoff_hz, params.butterworth_order
    t_max = float(gp["time_seconds"].max())
    grid_n = grid_span(0.0, t_max, fs)[1]  # the grid covers t_max under the same float-tolerant edge rule (F7)
    t = np.arange(grid_n) / fs

    non_ball = gp[~gp["is_ball"].to_numpy(dtype=bool)]
    teams_raw = sorted(pd.unique(non_ball["team_id"].dropna()), key=lambda x: str(canonical_id(x)))
    team_a, team_b = teams_raw[0], teams_raw[1]
    goal_x = {tm: gmap.get(game, period, tm) for tm in (team_a, team_b)}
    ref_goal = goal_x[team_a]
    reference_flip = ref_goal == 105.0

    dead = _period_stoppages(stoppage, game, period)

    def xy_of(pg: pd.DataFrame, best_estimate: bool) -> _PlayerXY:
        return _player_xy(
            pg,
            fs,
            fs_native,
            cutoff,
            order,
            grid_n,
            ref_goal,
            params,
            counters,
            dead=dead,
            apply_filter=apply_filter,
            best_estimate=best_estimate,
        )

    # include_goalkeeper is per method (spec 7.14): the player series hold the keeper when the cluster OR the dyads
    # use it (each filters by its own flag downstream); the team signals take it per their own flag (spec 7.5).
    gk = params.include_goalkeeper
    players_with_gk = gk["cluster"] or gk["dyad"]
    # Each player's rows are filtered + resampled ONCE per distinct row set: a player without goalkeeper rows has the
    # same rows with or without the keeper filter (the same arrays), so the player series and the team signals share
    # them; only a player WITH goalkeeper rows gets its outfield rows resampled on their own when one side drops them.
    # Spec 7.11 (review A-02): the player series read the DETECTED samples, the team signals every present row's
    # best-estimate position; a player detected on every row has one series serving both.
    players: dict[tuple[object, object], PlayerSeries] = {}
    team_xy: dict[object, dict[object, _PlayerXY]] = {}
    for tm in (team_a, team_b):
        team_rows = non_ball[(non_ball["team_id"] == tm).to_numpy(dtype=bool, na_value=False)]  # NA team -> not tm
        outfield_rows = team_rows[~team_rows["is_goalkeeper"].to_numpy(dtype=bool)]
        resampled: dict[tuple[object, bool, bool], _PlayerXY] = {}

        def xy_for(
            pid: object,
            pg: pd.DataFrame,
            *,
            best: bool,
            cache: dict[tuple[object, bool, bool], _PlayerXY] = resampled,
        ) -> _PlayerXY:
            best = best and not bool(pg["_detected"].all())  # every row detected: one series serves both
            key = (pid, bool(pg["is_goalkeeper"].any()), best)  # (player, holds keeper rows, best-estimate rows)
            if key not in cache:
                cache[key] = xy_of(pg, best)
            return cache[key]

        roster = team_rows if players_with_gk else outfield_rows
        for pid, pg in roster.groupby("player_id", sort=True, observed=True):
            players[(tm, pid)] = _player_series(xy_for(pid, pg, best=False), tm, pid, fs, params)
        team_src = team_rows if gk["team_signals"] else outfield_rows
        team_xy[tm] = {
            pid: xy_for(pid, pg, best=True) for pid, pg in team_src.groupby("player_id", sort=True, observed=True)
        }

    stoppage_mask = _in_intervals(t, dead)
    team_signal, team_phasor, team_inst_freq_pos, observed_fraction, team_detection, segments = _team_signals(
        team_xy, team_a, team_b, goal_x, fs, grid_n, ref_goal, params, stoppage_mask, counters=counters
    )
    segment_id = {
        tm: _segment_id_array(segments.get(tm, np.empty((0, 2), dtype=np.int64)), grid_n) for tm in (team_a, team_b)
    }
    possession_team = _possession_series(windows_gp, t, grid_n)
    window_ranges = _window_ranges(windows_gp, t)
    return PeriodSignals(
        game_id=game,
        period_id=period,
        t=t,
        team_ids=(team_a, team_b),
        goal_x=goal_x,
        reference_flip=reference_flip,
        segments=segments,
        segment_id=segment_id,
        team_signal=team_signal,
        team_phasor=team_phasor,
        team_inst_freq_pos=team_inst_freq_pos,
        observed_fraction=observed_fraction,
        team_detection=team_detection,
        possession_team=possession_team,
        window_ranges=window_ranges,
        players=players,
    )


def _detected_rows(pg: pd.DataFrame) -> pd.DataFrame:
    """The rows a detection-aware provider actually detected; all rows for a fully-observed one.

    Reads the ``_detected`` mask that :func:`detected_mask` (ADR-109) computes once at the edge in
    :func:`build_coordination_signals`, so the §7.3 all-null trap fires only there -- never per
    player slice, where a fully-extrapolated player's all-null ``visibility`` would wrongly raise.
    """
    return pg[pg["_detected"].to_numpy()]


def _detected_grid(tp: np.ndarray, grid_n: int, fs: float) -> np.ndarray:
    """Raw detection mapped to the nearest grid index (for observed_fraction, NOT bridged)."""
    g = np.zeros(grid_n, dtype=bool)
    if tp.size:
        g[np.clip(np.round(tp * fs).astype(np.int64), 0, grid_n - 1)] = True
    return g


def _on_pitch_grid(present_t: np.ndarray, grid_n: int, fs: float) -> np.ndarray:
    t = np.arange(grid_n) / fs
    if present_t.size == 0:
        return np.zeros(grid_n, dtype=bool)
    return (t >= present_t.min() - 0.5 / fs) & (t <= present_t.max() + 0.5 / fs)


@dataclass(frozen=True)
class _PlayerXY:
    """One player's filtered, resampled, ``ref_goal``-oriented positions (x and y share the run split) with its
    on-pitch and raw-detection masks on the grid -- everything the player series and the team signals read."""

    x: np.ndarray
    y: np.ndarray
    runs: list[tuple[int, int]]
    on_pitch: np.ndarray
    observed: np.ndarray
    is_goalkeeper: bool  # the earliest row's flag


def _player_xy(
    pg,
    fs,
    fs_native,
    cutoff,
    order,
    grid_n,
    ref_goal,
    params,
    counters,
    *,
    dead: np.ndarray,
    apply_filter=True,
    best_estimate: bool = False,
) -> _PlayerXY:
    """One player's positions, filtered + resampled per run (spec 7.4 steps 2-4).

    ``best_estimate`` picks the rows (spec 7.11, review A-02): the PLAYER series use the detected samples only, the
    TEAM signals every present row's best-estimate position -- dropping an off-camera player would bias every team
    signal toward the ball side. On a fully observed provider both are every row.

    The run split happens on those samples OUTSIDE the period's long stoppages (``dead``, the same intervals that
    split the team segments): a stoppage longer than ``max_stoppage_s`` leaves a gap far longer than
    ``max_detection_gap_s``, so it ends one run and starts the next -- before the filter, which therefore never
    smooths across it. ``on_pitch`` reads every row (a stoppage is not an absence); ``observed`` the detected rows.
    """
    pg = pg.sort_values("time_seconds")
    present_t = pg["time_seconds"].to_numpy(dtype=np.float64)
    det = _detected_rows(pg)
    src = pg if best_estimate else det
    tp = src["time_seconds"].to_numpy(dtype=np.float64)
    live = ~_in_intervals(tp, dead)
    xr = src["x"].to_numpy(dtype=np.float64)[live]
    yr = src["y"].to_numpy(dtype=np.float64)[live]
    if ref_goal is not None:
        xr = to_goal_relative_x_array(xr, goal_x=ref_goal)
        yr = to_goal_relative_y_array(yr, goal_x=ref_goal)
    (x, y), runs = _resample_runs(
        tp[live],
        (xr, yr),
        fs,
        fs_native,
        cutoff,
        order,
        grid_n,
        params.max_detection_gap_s,
        counters,
        apply_filter=apply_filter,
    )
    on_pitch = _on_pitch_grid(present_t, grid_n, fs)
    observed = _detected_grid(det["time_seconds"].to_numpy(dtype=np.float64), grid_n, fs) & on_pitch
    return _PlayerXY(x, y, runs, on_pitch, observed, bool(pg["is_goalkeeper"].iloc[0]))


def _player_series(xy: _PlayerXY, tm, pid, fs: float, params: CoordinationParams) -> PlayerSeries:
    ph_x, ipx = _phasors_over_runs(xy.x, xy.runs, fs, params.band_low_cpm)
    ph_y, ipy = _phasors_over_runs(xy.y, xy.runs, fs, params.band_low_cpm)
    return PlayerSeries(
        team_id=tm,
        player_id=pid,
        is_goalkeeper=xy.is_goalkeeper,
        x=xy.x,
        y=xy.y,
        runs=np.array(xy.runs, dtype=np.int64).reshape(-1, 2),
        phasor_x=ph_x,
        phasor_y=ph_y,
        inst_freq_pos_x=ipx,
        inst_freq_pos_y=ipy,
        observed=xy.observed,
        on_pitch=xy.on_pitch,
        detection=DetectionCounts.from_masks(xy.on_pitch, xy.observed),
    )


def _team_signals(team_xy, team_a, team_b, goal_x, fs, grid_n, ref_goal, params, stoppage_mask, *, counters=None):
    team_signal: dict[tuple[object, str], np.ndarray] = {}
    team_phasor: dict[tuple[object, str], np.ndarray] = {}
    team_inst_freq_pos: dict[tuple[object, str], np.ndarray] = {}
    observed_fraction: dict[object, np.ndarray] = {}
    team_detection: dict[object, DetectionCounts] = {}
    segments: dict[object, np.ndarray] = {}
    for tm in (team_a, team_b):
        by_pid: dict[object, _PlayerXY] = team_xy.get(tm, {})
        pids = sorted(by_pid, key=lambda x: str(canonical_id(x)))
        if not pids:
            continue
        xs = np.full((grid_n, len(pids)), np.nan)
        ys = np.full((grid_n, len(pids)), np.nan)
        on_pitch = np.zeros((grid_n, len(pids)), dtype=bool)
        detected_raw = np.zeros((grid_n, len(pids)), dtype=bool)
        for j, pid in enumerate(pids):
            xy = by_pid[pid]
            xs[:, j] = xy.x
            ys[:, j] = xy.y
            on_pitch[:, j] = xy.on_pitch
            detected_raw[:, j] = xy.observed
        valid = np.isfinite(xs) & np.isfinite(ys)  # bridged positions
        on_count = on_pitch.sum(axis=1)
        # a team sample is scoreable iff every on-pitch outfield player has a (bridged) resampled position
        sample_valid = (on_count > 0) & ~(on_pitch & ~valid).any(axis=1)
        if counters is not None:  # A-19: team samples voided because an on-pitch player has no bridged position
            counters["samples_unobserved"] += int(((on_count > 0) & ~sample_valid).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            observed_fraction[tm] = np.where(
                on_count > 0, detected_raw.sum(axis=1) / np.where(on_count > 0, on_count, 1), np.nan
            )
        # the team side's detection share counts the RAW detections of the team-signal players (A-08)
        team_detection[tm] = DetectionCounts.from_masks(on_pitch, detected_raw)
        pos = np.stack([np.where(valid, xs, np.nan), np.where(valid, ys, np.nan)], axis=-1)  # (N, P, 2)
        compact, counts = compact_rows(pos, valid)
        coll = collective_from_positions(compact, counts)
        # `compact` is in the ref_goal-relative frame (line ~416), so the defended end must be named in THAT
        # frame: a team whose absolute goal == ref_goal defends x=0 there. Using the absolute `goal_x[tm] == 0.0`
        # here double-counts orientation and flips the back line under a pitch mirror (C24, ADR-051).
        defends_zero = (goal_x[tm] == 0.0) if ref_goal is None else (goal_x[tm] == ref_goal)
        defends0 = np.full(grid_n, defends_zero)
        back = back_line_batch(compact, counts, defends0, n=4, adaptive_max_n=5)
        for name in TEAM_SIGNALS:
            sig = coll.get(name)
            if sig is None:
                sig = back[name]
            team_signal[(tm, name)] = np.where(sample_valid, np.asarray(sig, dtype=np.float64), np.nan)
        handover = round(params.max_stoppage_s * fs)  # owner ruling 2026-10-03: the same bound as a short stoppage
        segments[tm] = _segments_with_count(sample_valid, on_count, stoppage_mask, handover_samples=handover)
        for name in TEAM_SIGNALS:
            ph, ipos = _team_phasor(team_signal[(tm, name)], segments[tm], fs, params.band_low_cpm)
            team_phasor[(tm, name)] = ph
            team_inst_freq_pos[(tm, name)] = ipos
    return team_signal, team_phasor, team_inst_freq_pos, observed_fraction, team_detection, segments


def _absorb_handovers(on_count: np.ndarray, handover_samples: int) -> np.ndarray:
    """``on_count`` with every handover excursion set back to its surrounding level.

    An inexact substitution handover moves the count off its level and back: the incoming player appears a little
    before the outgoing one disappears (+1), or a little after (-1). Spec 7.4 step 2 says a substitution keeps the
    count and does not split, so a run of samples whose count differs from the SAME count on both sides, and which
    lasts at most ``handover_samples`` (``max_stoppage_s`` at the grid rate: a change no longer than a short
    stoppage, owner ruling 2026-10-03, review M-13), takes that surrounding count. Nested excursions are absorbed
    from the inside out. A change that does not return to its prior level (a red card) or that touches the period
    edge (no prior level to return to) is kept. Only the segmentation reads this; the samples keep their values.
    """
    count = np.asarray(on_count).copy()
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(count)) + 1
        starts = np.concatenate(([0], edges))
        for k in range(1, len(starts)):  # every run but the first (which has no prior level to return to)
            lo = int(starts[k])
            level = count[lo - 1]  # the level the excursion left
            ahead = count[lo:]
            ret_rel = np.flatnonzero(ahead == level)  # where the count FIRST returns to that level (B m1 / R2-5)
            if ret_rel.size == 0:
                continue  # never returns (a red card, or the period edge): a real composition change -- kept
            ret = lo + int(ret_rel[0])
            if ret - lo <= handover_samples:  # a bounded excursion by ANY path, not just a mirror step
                count[lo:ret] = level
                changed = True
                break  # the run boundaries moved: rescan
    return count


def _segments_with_count(
    sample_valid: np.ndarray, on_count: np.ndarray, stoppage_mask: np.ndarray, *, handover_samples: int
) -> np.ndarray:
    """Maximal runs of valid, non-stoppage samples with a constant on-pitch count (a red card splits; a substitution
    does not -- not even an inexact handover lasting up to ``handover_samples``, see :func:`_absorb_handovers`)."""
    on_count = _absorb_handovers(on_count, handover_samples)
    active = sample_valid & ~stoppage_mask
    n = active.size
    segs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        c = on_count[i]
        while j < n and active[j] and on_count[j] == c:
            j += 1
        segs.append((i, j))
        i = j
    return np.array(segs, dtype=np.int64).reshape(-1, 2)


def _period_stoppages(stoppage: Any, game: object, period: int) -> np.ndarray:
    """This period's long-stoppage ``[start, end)`` intervals, ``(S, 2)`` -- the ONE source that splits both the player
    runs and the team segments (spec 7.4 step 2)."""
    parts = [
        np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        for (g, p), arr in stoppage.intervals.items()
        if same_id(g, game) and int(p) == period
    ]
    return np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.float64)


def _in_intervals(t: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """``t`` inside any ``[start, end)`` interval."""
    mask = np.zeros(t.size, dtype=bool)
    for lo, hi in intervals:
        mask |= (t >= lo) & (t < hi)
    return mask


def _team_phasor(
    values: np.ndarray, segments: np.ndarray, fs: float, band_low_cpm: float
) -> tuple[np.ndarray, np.ndarray]:
    ph = np.full(values.size, np.nan + 1j * np.nan, dtype=np.complex128)
    inst_pos = np.full(values.size, np.nan)  # float, NaN at each segment's first sample (A-45)
    for lo, hi in segments:
        seg = values[lo:hi]
        if seg.size < 2 or not np.isfinite(seg).all():
            continue
        theta = analytic_phase(seg, pad_length(seg.size, fs, band_low_cpm))
        ph[lo:hi] = phasor(theta)
        inst_pos[lo:hi] = phase_advance_indicator(theta)
    return ph, inst_pos


def _segment_id_array(segments: np.ndarray, grid_n: int) -> np.ndarray:
    sid = np.full(grid_n, -1, dtype=np.int64)
    for k, (lo, hi) in enumerate(segments):
        sid[lo:hi] = k
    return sid


def build_coordination_signals(
    frames: pd.DataFrame,
    *,
    windows: pd.DataFrame,
    params: CoordinationParams | None = None,
    actions: pd.DataFrame | None = None,
    goal_map: GoalMap | None = None,
    links: pd.DataFrame | None = None,
    stoppage_evidence: Literal["auto", "ball_state", "events", "none"] = "auto",
    detection: Literal["auto", "detection_aware", "fully_observed"] = "auto",
) -> CoordinationSignals:
    """Prepare filtered/resampled/oriented coordination signals for every (game, period) in ``frames``.

    Examples
    --------
    Prepare signals once, then run any family compute on them (needs a real match)::

        signals = build_coordination_signals(frames, windows=period_windows(frames), actions=actions)
        pair, phase, report = compute_relative_phase(signals)
    """
    return _build_coordination_signals(
        frames,
        windows=windows,
        params=params,
        actions=actions,
        goal_map=goal_map,
        links=links,
        stoppage_evidence=stoppage_evidence,
        detection=detection,
        apply_filter=True,
    )


def _build_coordination_signals(
    frames: pd.DataFrame,
    *,
    windows: pd.DataFrame,
    params: CoordinationParams | None = None,
    actions: pd.DataFrame | None = None,
    goal_map: GoalMap | None = None,
    links: pd.DataFrame | None = None,
    stoppage_evidence: Literal["auto", "ball_state", "events", "none"] = "auto",
    detection: Literal["auto", "detection_aware", "fully_observed"] = "auto",
    apply_filter: bool = True,
    detected_override: np.ndarray | None = None,
) -> CoordinationSignals:
    """Shared body of :func:`build_coordination_signals`.

    ``apply_filter=False`` yields RAW-resampled signals aligned element-wise to the filtered build (same run
    split, ``butterworth_min_length`` admission, orientation, resample grid, validity and stoppage segments) --
    the D1 Tier-B ``vc_epsilon`` seam (§8.2), consumed only by ``scripts/derive_coordination_params``. The
    default reproduces :func:`build_coordination_signals` bit for bit.

    ``detected_override`` (one bool per frames row) scores the build DETECTION-AWARE with that mask as the detection
    flag, whatever the provider: the D1 occlusion seam (owner ruling 2026-10-04, A-08), which scores an occluded fully
    observed match exactly the way production scores SkillCorner -- player series from the detected samples only,
    team signals from every row's best-estimate position, every row's ``coord_detected_share`` from the mask.
    """
    params = params or CoordinationParams()
    _refuse(frames, windows)
    native_hz = float(frames["frame_rate"].iloc[0])
    cutoff = params.butterworth_cutoff_hz
    # the signal-prep filter prewarps (A-48), so the combined -3 dB is at `cutoff`: it must be below the native
    # Nyquist (a prewarped design frequency is always representable, so the cutoff itself is the binding guard).
    if cutoff >= native_hz / 2.0:
        raise ValueError(
            f"butterworth cutoff {cutoff:.3f} Hz is at or above the native Nyquist "
            f"{native_hz / 2:.3f} Hz; lower butterworth_cutoff_hz or raise the sampling rate"
        )
    target = max(params.analysis_hz, 10.0 * cutoff)
    fs = min(native_hz, target)
    rate_capped = fs < target

    provider = str(frames["source_provider"].astype(object).iloc[0])
    validate_provider(provider)
    det_mode = (
        detection
        if detection != "auto"
        else ("detection_aware" if provider in _DETECTION_AWARE_PROVIDERS else "fully_observed")
    )
    # Single-source the detection verdict through the ADR-109 primitive: for a detection-aware
    # provider this runs `assert_detection_aware_visibility` (the §7.3 all-null edge refusal, same
    # scope as before) and yields the per-row mask; for fully-observed providers (assume_observed)
    # it is all-True after `validate_provider`. Computed once here so every per-player slice below
    # just indexes `_detected` (§7.4.1 "a detection mask of the same shape"), never re-raising the trap.
    if detected_override is None:
        if det_mode == "detection_aware" and "visibility" not in frames.columns:
            raise ValueError(  # spec 7.3: visibility is required for a detection-aware provider (review A-29)
                f"frames for detection-aware provider {provider!r} are missing the required 'visibility' column"
            )
        # a fully-observed provider needs no visibility column: assume_observed ignores the series' contents, so an
        # absent column is an all-NA placeholder of the right length rather than a KeyError (A-29).
        vis = frames["visibility"] if "visibility" in frames.columns else pd.Series(pd.NA, index=frames.index)
        detected = detected_mask(vis, provider=provider, assume_observed=det_mode != "detection_aware")
    else:
        detected = np.asarray(detected_override)
        if detected.dtype != np.bool_ or detected.shape != (len(frames),):
            raise ValueError(f"detected_override must be one bool per frames row ({len(frames)}); got {detected.shape}")
        det_mode = "detection_aware"
    stoppage = resolve_stoppages(
        frames, actions=actions, provider=provider, max_stoppage_s=params.max_stoppage_s, mode=stoppage_evidence
    )
    regime = validate_windows(windows)
    gmap = goal_map if goal_map is not None else resolve_defended_goals(frames)

    counters = {"n_segments": 0, "n_runs_too_short": 0, "samples_unobserved": 0}
    periods: list[PeriodSignals] = []
    # the period build reads only these columns: every per-period / per-player slice then moves 11 columns, not all
    work = frames[list(_PREP_COLUMNS)].assign(_detected=detected)
    groups = group_rows(work, ("game_id", "period_id"))
    win_groups = group_rows(windows, ("game_id", "period_id"))  # ADR-068: group once, .get in the loop (no rescan)
    key_df = work[["game_id", "period_id"]].drop_duplicates().sort_values(["game_id", "period_id"])
    for gp_key in key_df.itertuples(index=False):
        gp = groups.get(gp_key.game_id, gp_key.period_id)
        windows_gp = win_groups.get(gp_key.game_id, gp_key.period_id)
        ps = _prepare_period(
            gp,
            windows_gp=windows_gp,
            params=params,
            fs=fs,
            gmap=gmap,
            provider=provider,
            stoppage=stoppage,
            counters=counters,
            apply_filter=apply_filter,
        )
        counters["n_segments"] += sum(len(seg) for seg in ps.segments.values())
        periods.append(ps)
    return CoordinationSignals(
        provider=provider,
        params=params,
        native_hz=native_hz,
        fs=fs,
        rate_capped=rate_capped,
        windows=windows,
        window_regime=regime,
        detection_source=det_mode,
        stoppage=stoppage,
        periods=tuple(periods),
        counters=counters,
    )
