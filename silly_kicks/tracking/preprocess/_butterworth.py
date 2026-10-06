"""Zero-phase Butterworth low-pass, uniform resampling, and residual-analysis cutoff (TF-58 seam 1, spec 7.4).

Array kernels take numpy arrays; ``resample_frames`` is the frame-level wrapper. The cutoff is the -3 dB
point of the COMBINED dual pass: the design frequency is ``cutoff / winter_correction(order)`` (Winter 2009),
so ``sosfiltfilt`` (forward+backward) hits -3 dB at ``cutoff``.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi


def winter_correction(order: int) -> float:
    """Dual-pass cutoff correction ``(2**0.5 - 1) ** (1 / (2 * order))`` (Winter 2009): applying a
    Butterworth twice (``sosfiltfilt``) sharpens the response, so the single-pass design frequency is the
    target cutoff divided by this factor.

    Examples
    --------
    >>> round(winter_correction(3), 6)
    0.863384
    """
    return (2.0**0.5 - 1.0) ** (1.0 / (2.0 * order))


def exact_prewarped_cutoff(cutoff_hz: float, fs: float, order: int = 3) -> float:
    """The single-pass design frequency that puts the COMBINED dual-pass -3 dB EXACTLY at ``cutoff_hz`` (review A-48).

    The default path divides the cutoff by :func:`winter_correction` in LINEAR Hz, but ``scipy.signal.butter(fs=)``
    warps the frequency axis (the bilinear transform), so the combined -3 dB lands off-target — negligibly near DC
    (~0.18% at 0.4 Hz / 10 Hz) but ~9% at 3.0 Hz / 10 Hz. Applying the Winter sharpening in the WARPED domain removes
    the bias: ``design = (fs/π)·arctan(tan(π·cutoff/fs) / winter_correction(order))``. Because ``arctan`` is bounded by
    π/2, the result is always below the Nyquist, so ``cutoff_hz`` itself must be < fs/2 (its combined -3 dB cannot be
    represented at or above the Nyquist).

    This is the CORRECT design for EVERY dual-pass Butterworth; the default remains linear-Winter (byte-identity for
    the velocity/smoothing consumers), and this helper is the opt-in — coordination's signal preparation passes
    ``prewarp=True`` (spec 7.4 step 3). Flipping the default repo-wide is a separate owner-gated cycle (TODO.md).

    Examples
    --------
    >>> round(exact_prewarped_cutoff(0.4, 10.0, 3), 6)  # near DC: ~ the linear design
    0.462389
    """
    if not 0.0 < cutoff_hz < fs / 2.0:
        raise ValueError(f"exact_prewarped_cutoff: cutoff {cutoff_hz} Hz must be in (0, Nyquist={fs / 2.0}) Hz")
    return float(fs / np.pi * np.arctan(np.tan(np.pi * cutoff_hz / fs) / winter_correction(order)))


@functools.cache
def _design_sos(fs: float, cutoff_hz: float, order: int, prewarp: bool = False) -> np.ndarray:
    """The Butterworth SOS for one ``(fs, cutoff_hz, order)`` -- CACHED and READ-ONLY (ADR-111 performance).

    The design depends ONLY on these three scalars, but the filter is applied per player-series per contiguous
    run -- tens of thousands of applications per match -- so re-running ``scipy.signal.butter`` on every call
    dominated the coordination signal-prep cost (~36 s/match of redundant designs, measured). Memoization is
    deterministic, so the cached SOS is byte-identical to a fresh design.

    The array is returned READ-ONLY: it is a shared cached object, so a caller that mutates it in place would
    corrupt every other caller -- ``flags.writeable = False`` makes that fail loud instead. It is deliberately
    NOT copied per call (a copy would defeat the cache). The key space is a handful of designs (a few provider
    sampling rates x cutoffs x orders), so the unbounded cache stays small.
    """
    nyq = fs / 2.0
    design_freq = exact_prewarped_cutoff(cutoff_hz, fs, order) if prewarp else cutoff_hz / winter_correction(order)
    if design_freq >= nyq:
        raise ValueError(
            f"butterworth: design frequency {design_freq:.4f} Hz (cutoff {cutoff_hz} Hz / Winter correction) "
            f"is at or above the Nyquist frequency {nyq} Hz at fs={fs} Hz"
        )
    sos = np.asarray(butter(order, design_freq, btype="low", fs=fs, output="sos"), dtype=np.float64)
    sos.flags.writeable = False
    return sos


def _design(fs: float, cutoff_hz: float, order: int, prewarp: bool = False) -> np.ndarray:
    """Read-only cached SOS for ``(fs, cutoff_hz, order)``; args canonicalised to Python scalars for one key."""
    return _design_sos(float(fs), float(cutoff_hz), int(order), bool(prewarp))


@functools.cache
def _steady_state_zi(fs: float, cutoff_hz: float, order: int, prewarp: bool = False) -> np.ndarray:
    """``scipy.signal.sosfilt_zi`` of the cached design -- CACHED and READ-ONLY (ADR-111 performance).

    ``sosfiltfilt`` re-solves these per-section steady-state initial conditions (``lfilter_zi``: a linear solve
    per section) on EVERY call, although they depend only on the design; on fragmented tracking that is ~17k
    solves per match. Same bounded key space and read-only contract as :func:`_design_sos`.
    """
    zi = np.asarray(sosfilt_zi(_design_sos(fs, cutoff_hz, order, prewarp)), dtype=np.float64)
    zi.flags.writeable = False
    return zi


def _sosfiltfilt(sos_master: np.ndarray, zi_master: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``scipy.signal.sosfiltfilt(sos, x)`` (``padtype="odd"``, default ``padlen``, 1-D) step for step, with the
    steady-state ``zi`` precomputed: odd extension by ``3 * ntaps``, forward ``sosfilt`` seeded ``zi * ext[0]``,
    reverse pass seeded ``zi * y[-1]``, reverse, trim. Every numpy/scipy operation is scipy's own on the same values,
    so the output is bit-identical; the fence test compares it with the real ``sosfiltfilt`` on every run length.
    """
    n_sections = sos_master.shape[0]
    ntaps = 2 * n_sections + 1 - min(int((sos_master[:, 2] == 0).sum()), int((sos_master[:, 5] == 0).sum()))
    edge = 3 * ntaps
    # scipy.signal._arraytools.odd_ext(x, edge)
    ext = np.concatenate((2 * x[0:1] - x[edge:0:-1], x, 2 * x[-1:] - x[-2 : -(edge + 2) : -1]))
    sos = np.array(sos_master)  # scipy's sosfilt needs a writeable SOS buffer: a tiny per-call copy of the master
    y, _zf = sosfilt(sos, ext, zi=zi_master * ext[0:1])
    y, _zf = sosfilt(sos, y[::-1], zi=zi_master * y[-1:])
    return y[::-1][edge:-edge]


@functools.cache
def _min_length_cached(fs: float, cutoff_hz: float, order: int) -> int:
    sos = _design_sos(fs, cutoff_hz, order)
    ntaps = 2 * sos.shape[0] + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * ntaps + 1  # sosfiltfilt requires len > padlen = 3 * ntaps


def butterworth_min_length(fs: float, cutoff_hz: float, order: int = 3) -> int:
    """Minimum series length ``sosfiltfilt`` accepts (its default ``padlen`` + 1). Cached per design.

    Examples
    --------
    >>> butterworth_min_length(10.0, 0.4, 3) > 0
    True
    """
    return _min_length_cached(float(fs), float(cutoff_hz), int(order))


def butterworth_lowpass(
    values: np.ndarray, fs: float, cutoff_hz: float, order: int = 3, *, prewarp: bool = False
) -> np.ndarray:
    """Zero-phase Butterworth low-pass (forward-backward ``sosfiltfilt``) at the dual-pass ``cutoff_hz``.

    ``prewarp=True`` designs through :func:`exact_prewarped_cutoff` so the combined -3 dB lands exactly at
    ``cutoff_hz`` even at high cutoff (review A-48); the default stays linear-Winter (byte-identity).

    Raises ``ValueError`` when the design frequency reaches Nyquist, or the series is shorter than
    :func:`butterworth_min_length`.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(2000) / 10.0
    >>> x = np.sin(2 * np.pi * 0.05 * t)
    >>> y = butterworth_lowpass(x, 10.0, 0.4)
    >>> bool(np.max(np.abs(y[500:1500] - x[500:1500])) < 0.01)  # in-band, interior: near-identity
    True
    """
    values = np.asarray(values, dtype=np.float64)
    min_len = butterworth_min_length(fs, cutoff_hz, order)
    if values.shape[0] < min_len:
        raise ValueError(f"butterworth: series length {values.shape[0]} < minimum {min_len} for sosfiltfilt padding")
    key = (float(fs), float(cutoff_hz), int(order), bool(prewarp))
    return np.asarray(_sosfiltfilt(_design_sos(*key), _steady_state_zi(*key), values), dtype=np.float64)


GRID_TOLERANCE = 1e-9
"""Run-edge tolerance of the uniform grid, in grid samples. Time stamps are floats, so a grid point ``k / fs`` that
equals a run's first or last time in exact arithmetic can land an ulp outside it (``288.9 - 282.0 =
6.899999999999977``; ``2331 * 0.1 = 233.10000000000002``). Such a point belongs to the run: ``np.interp`` gives it
the endpoint value. Without the tolerance it was dropped or NaN (TF-58 F7)."""


def grid_span(start: float, end: float, fs: float) -> tuple[int, int]:
    """Half-open index range ``[k_lo, k_hi)`` of the uniform grid ``k / fs`` covered by the closed time span
    ``[start, end]``, with the :data:`GRID_TOLERANCE` edge rule. It is the ONE definition of which grid points a run
    covers, shared by :func:`resample_uniform`, :func:`resample_frames` and the coordination signal preparation.
    Callers clip the range to their grid.

    Examples
    --------
    >>> grid_span(282.0, 288.9, 10.0)  # (288.9 - 282.0) * 10 = 68.99999999999977 still covers 70 points
    (2820, 2890)
    >>> grid_span(2331 * 0.1, 240.0, 10.0)  # 233.10000000000002 still covers grid point 2331
    (2331, 2401)
    """
    return int(np.ceil(start * fs - GRID_TOLERANCE)), int(np.floor(end * fs + GRID_TOLERANCE)) + 1


def butterworth_lowpass_rows(
    rows: np.ndarray, fs: float, cutoff_hz: float, order: int = 3, *, prewarp: bool = False
) -> np.ndarray:
    """:func:`butterworth_lowpass` of every row of the 2-D ``rows`` (filtering along the last axis) in ONE pass.

    ``prewarp`` is forwarded to the design (review A-48); the default stays linear-Winter (byte-identity).

    The replica's steps on a leading row axis: odd extension per row, forward ``sosfilt`` seeded ``zi * ext[:, 0]``,
    reverse pass seeded ``zi * y[:, -1]``. scipy's ``sosfilt`` runs each row through the same recursion as a 1-D
    call, so every row equals its own 1-D :func:`butterworth_lowpass` bit for bit (fence-tested); the pass only
    saves the per-call overhead of filtering a run's coordinates one by one (ADR-111 ruling C).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(400) / 10.0
    >>> xy = np.stack([np.sin(2 * np.pi * 0.05 * t), np.cos(2 * np.pi * 0.05 * t)])
    >>> out = butterworth_lowpass_rows(xy, 10.0, 0.4)
    >>> bool(np.array_equal(out[1], butterworth_lowpass(xy[1], 10.0, 0.4)))
    True
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D (rows x samples); got shape {rows.shape}")
    min_len = butterworth_min_length(fs, cutoff_hz, order)
    if rows.shape[1] < min_len:
        raise ValueError(f"butterworth: series length {rows.shape[1]} < minimum {min_len} for sosfiltfilt padding")
    key = (float(fs), float(cutoff_hz), int(order), bool(prewarp))
    sos_master, zi_master = _design_sos(*key), _steady_state_zi(*key)
    n_sections = sos_master.shape[0]
    ntaps = 2 * n_sections + 1 - min(int((sos_master[:, 2] == 0).sum()), int((sos_master[:, 5] == 0).sum()))
    edge = 3 * ntaps
    x = rows
    ext = np.concatenate((2 * x[:, 0:1] - x[:, edge:0:-1], x, 2 * x[:, -1:] - x[:, -2 : -(edge + 2) : -1]), axis=1)
    sos = np.array(sos_master)  # scipy's sosfilt needs a writeable SOS buffer
    y, _zf = sosfilt(sos, ext, axis=-1, zi=zi_master[:, None, :] * ext[None, :, 0:1])
    y, _zf = sosfilt(sos, y[:, ::-1], axis=-1, zi=zi_master[:, None, :] * y[None, :, -1:])
    return np.asarray(y[:, ::-1][:, edge:-edge], dtype=np.float64)


def resample_uniform(
    t: np.ndarray,
    values: np.ndarray,
    fs_out: float,
    run_bounds: np.ndarray,
    *,
    n_out: int | None = None,
) -> np.ndarray:
    """Linear-interpolate ``values`` onto a uniform grid ``k / fs_out`` (k = 0..n_out-1), within runs only.
    Grid points outside every ``[start, end]`` in ``run_bounds`` are NaN; interpolation never crosses a run
    boundary (no rescan -- each run's native slice is found by ``np.searchsorted``). The grid points a run covers
    come from :func:`grid_span` (float-tolerant edges); the default ``n_out`` covers ``t.max()`` the same way.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.0, 1.0, 2.0])
    >>> resample_uniform(t, 2.0 * t, 2.0, np.array([[0.0, 2.0]]), n_out=5).tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0]
    """
    t = np.asarray(t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if n_out is None:
        n_out = grid_span(0.0, float(t.max()) if t.size else 0.0, fs_out)[1]
    out = np.full(n_out, np.nan)
    for start, end in np.asarray(run_bounds, dtype=np.float64).reshape(-1, 2):
        lo = int(np.searchsorted(t, start, side="left"))
        hi = int(np.searchsorted(t, end, side="right"))
        if hi - lo < 1:
            continue
        k_lo, k_hi = grid_span(float(start), float(end), fs_out)
        k_lo, k_hi = max(k_lo, 0), min(k_hi, n_out)
        if k_hi > k_lo:
            out[k_lo:k_hi] = np.interp(np.arange(k_lo, k_hi) / fs_out, t[lo:hi], values[lo:hi])
    return out


def _residual_curve(values: np.ndarray, fs: float, grid: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Winter's residual curve ``R(f) = RMS(values - butterworth_lowpass(values, f))`` over the evaluable grid (C18:
    only grid frequencies whose design frequency is below Nyquist)."""
    grid = np.sort(np.asarray(grid, dtype=np.float64))
    grid = grid[grid / winter_correction(order) < fs / 2.0]
    resid = np.array([np.sqrt(np.mean((values - butterworth_lowpass(values, fs, f, order)) ** 2)) for f in grid])
    return grid, resid


def _winter_intercept(grid: np.ndarray, resid: np.ndarray, tail_fraction: float) -> float:
    """The intercept of the line fit to the residual curve's linear high-frequency tail -- Winter's noise-floor RMS."""
    tail = grid >= grid.max() * tail_fraction
    _slope, intercept = np.polyfit(grid[tail], resid[tail], 1)
    return float(intercept)


def residual_analysis_noise_rms(
    values: np.ndarray, fs: float, grid: np.ndarray, *, order: int = 3, tail_fraction: float = 0.5
) -> float:
    """The Winter (2009) residual-analysis NOISE-FLOOR RMS: the intercept of the line fit to the residual curve's
    linear high-frequency tail (review A-14; spec 8.2 wants ``max_detection_gap_s`` compared against *this*, not the
    residual at a fixed interim cutoff). Same ill-posed guard as :func:`residual_analysis_cutoff`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(6000) / 25.0
    >>> noisy = np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.1, t.size)
    >>> bool(0.06 < residual_analysis_noise_rms(noisy, 25.0, np.arange(0.1, 5.0, 0.05)) < 0.14)
    True
    """
    grid, resid = _residual_curve(values, fs, grid, order)
    intercept = _winter_intercept(grid, resid, tail_fraction)
    if intercept <= 0.0 or intercept < 0.02 * float(resid.max()):
        raise ValueError("residual analysis: no noise floor detected -- the signal appears noise-free")
    return intercept


def residual_analysis_cutoff(
    values: np.ndarray, fs: float, grid: np.ndarray, *, order: int = 3, tail_fraction: float = 0.5
) -> float:
    """Winter (2009) residual analysis: the cutoff where the residual RMS ``R(f)`` meets the intercept of a
    line fit to its linear high-frequency tail (the noise floor). C18: only grid frequencies whose design
    frequency is below Nyquist are evaluated; the "tail" is the upper ``tail_fraction`` of that evaluable grid.
    Raises ``ValueError`` when the signal appears noise-free (no noise floor -- the residual method is
    ill-posed) or when ``R(f)`` never reaches the intercept on the grid.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(4000) / 25.0
    >>> clean = np.sin(2 * np.pi * 0.3 * t)
    >>> noisy = clean + rng.normal(0, 0.05, t.size)
    >>> c = residual_analysis_cutoff(noisy, 25.0, np.arange(0.1, 5.0, 0.05))
    >>> bool(0.3 < c < 1.2)
    True
    """
    grid, resid = _residual_curve(values, fs, grid, order)
    intercept = _winter_intercept(grid, resid, tail_fraction)
    if intercept <= 0.0 or intercept < 0.02 * float(resid.max()):
        # The high-frequency residual is ~0 relative to the peak: the signal appears noise-free, so there is
        # no noise floor to intersect and no cutoff is identifiable (residual analysis is ill-posed here).
        raise ValueError("residual analysis: no noise floor detected -- the signal appears noise-free")
    below = np.flatnonzero(resid <= intercept)
    if below.size == 0:
        raise ValueError("residual analysis: R(f) never reaches the noise-line intercept on this grid")
    i = int(below[0])
    if i == 0:
        return float(grid[0])
    f0, f1, r0, r1 = grid[i - 1], grid[i], resid[i - 1], resid[i]
    return float(f0 + (r0 - intercept) * (f1 - f0) / (r0 - r1))  # linear interpolation to the crossing


_RESAMPLE_ENTITY = ["game_id", "period_id", "is_ball", "player_id"]  # one entity never spans two games
_RESAMPLE_OVERWRITE = {"frame_id", "time_seconds", "frame_rate", "x", "y", "speed", "vx", "vy"}


def _cast_stored(arr: np.ndarray, dtype_str: str):
    """Cast a numpy float array to a column's stored dtype. A pandas NULLABLE extension dtype (``Float32``/``Float64``/
    ``Int64`` -- capitalised) cannot be named by ``np.ndarray.astype``, so route it through ``pd.array``; a plain numpy
    dtype uses ``astype`` (nit: ``resample_frames`` raised ``TypeError`` on nullable-Float input)."""
    if dtype_str[:1].isupper():
        # ``pd.array``'s overloads take only narrow per-family dtype args, so a dynamic dtype name
        # resolves none; Series construction accepts a dtype string and ``.array`` gives the same
        # nullable ExtensionArray (``pd.Series`` ``dtype`` is ``ExtensionDtype | NpDtype``, str-inclusive).
        return pd.Series(arr, dtype=dtype_str).array
    return arr.astype(dtype_str)


def resample_frames(frames: pd.DataFrame, target_hz: float, *, max_gap_seconds: float = 0.5) -> pd.DataFrame:
    """Resample tracking frames onto a uniform per-period ``target_hz`` grid. Per ``(game, period, is_ball,
    player_id)`` entity: ``x``/``y`` are linearly interpolated within runs split at gaps longer than
    ``max_gap_seconds``; every other column is step-held from the latest native row at or before the grid
    time; ``speed``/``vx``/``vy`` (if present) are set NaN (re-derive after resampling). ``frame_id`` is the
    grid index ``k`` and ``time_seconds`` is ``k / target_hz``.

    Examples
    --------
    >>> import pandas as pd
    >>> frames = pd.DataFrame(
    ...     {
    ...         "game_id": [1, 1, 1],
    ...         "period_id": [1, 1, 1],
    ...         "frame_id": [0, 1, 2],
    ...         "time_seconds": [0.0, 0.4, 0.8],
    ...         "frame_rate": [2.5, 2.5, 2.5],
    ...         "player_id": [7, 7, 7],
    ...         "team_id": [3, 3, 3],
    ...         "is_ball": [False, False, False],
    ...         "is_goalkeeper": [False, False, False],
    ...         "x": [0.0, 4.0, 8.0],
    ...         "y": [0.0, 0.0, 0.0],
    ...     }
    ... )
    >>> out = resample_frames(frames, 5.0)
    >>> out["time_seconds"].tolist()
    [0.0, 0.2, 0.4, 0.6, 0.8]
    >>> [round(v, 1) for v in out["x"]]
    [0.0, 2.0, 4.0, 6.0, 8.0]
    """
    if frames.empty:
        return frames.copy()
    # F1b (ADR-106): interpolate in float64, store back at the INPUT dtype (float32 storage), like interpolate/smooth.
    stored = {c: str(frames[c].dtype) for c in ("x", "y", "speed", "vx", "vy") if c in frames.columns}
    parts: list[pd.DataFrame] = []
    ordered = frames.sort_values([*_RESAMPLE_ENTITY, "time_seconds"], kind="mergesort")
    for _key, g in ordered.groupby(_RESAMPLE_ENTITY, dropna=False, observed=True, sort=False):
        t = g["time_seconds"].to_numpy(dtype="float64")
        if t.size == 0:
            continue
        split = np.flatnonzero(np.diff(t) > max_gap_seconds) + 1
        starts = np.concatenate(([0], split))
        ends = np.concatenate((split, [t.size]))
        run_bounds = np.array([[t[s], t[e - 1]] for s, e in zip(starts, ends, strict=True)])
        n_out = grid_span(0.0, float(t.max()), target_hz)[1]
        rx = resample_uniform(t, g["x"].to_numpy(dtype="float64"), target_hz, run_bounds, n_out=n_out)
        ry = resample_uniform(t, g["y"].to_numpy(dtype="float64"), target_hz, run_bounds, n_out=n_out)
        k = np.flatnonzero(~np.isnan(rx))
        if k.size == 0:
            continue
        grid_t = k / target_hz
        # "at or before the grid time" under the same edge tolerance as grid_span: a grid point an ulp before its
        # run's first sample holds THAT row, not the previous run's last one
        hold = np.clip(np.searchsorted(t * target_hz, k + GRID_TOLERANCE, side="right") - 1, 0, t.size - 1)
        sub = g.iloc[hold].reset_index(drop=True)
        sub["frame_id"] = k.astype("int64")
        sub["time_seconds"] = grid_t
        sub["frame_rate"] = float(target_hz)
        sub["x"] = _cast_stored(rx[k], stored["x"])
        sub["y"] = _cast_stored(ry[k], stored["y"])
        for c in ("speed", "vx", "vy"):
            if c in sub.columns:
                sub[c] = _cast_stored(np.full(len(sub), np.nan), stored[c])
        parts.append(sub)
    if not parts:
        return frames.iloc[:0].copy()
    return pd.concat(parts, ignore_index=True)[list(frames.columns)]
