"""Zero-phase Butterworth / resampling / residual analysis (TF-58 Task 4, spec 9.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import sosfiltfilt, sosfreqz

from silly_kicks.tracking.preprocess import (
    PreprocessConfig,
    butterworth_lowpass,
    resample_frames,
    resample_uniform,
    residual_analysis_cutoff,
    smooth_frames,
)
from silly_kicks.tracking.preprocess._butterworth import (
    _design,
    _design_sos,
    butterworth_min_length,
    winter_correction,
)

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def test_winter_correction_value():
    assert winter_correction(3) == (2.0**0.5 - 1.0) ** (1.0 / 6.0)


def test_zero_phase_lag_on_in_band_sinusoid():
    t = np.arange(6000) / 10.0  # 600 s at 10 Hz
    x = np.sin(2 * np.pi * 0.05 * t)
    y = butterworth_lowpass(x, 10.0, 0.4)
    xc = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    lag = xc.argmax() - (len(x) - 1)
    assert lag == 0  # zero phase
    assert np.max(np.abs(y[500:5500])) / np.max(np.abs(x[500:5500])) > 0.999  # in-band ~ unity


def test_dual_pass_minus_3db_at_cutoff():
    sos = _design(10.0, 0.4, 3)
    _, h = sosfreqz(sos, worN=[0.4], fs=10.0)
    dual_pass_gain = float(np.abs(np.asarray(h))[0] ** 2)  # sosfiltfilt applies |H| twice
    assert abs(dual_pass_gain - _INV_SQRT2) < 0.01
    t = np.arange(20000) / 10.0
    y = butterworth_lowpass(np.sin(2 * np.pi * 0.4 * t), 10.0, 0.4)
    assert abs(np.max(np.abs(y[5000:15000])) - _INV_SQRT2) < 0.02


def _dual_pass_gain(sos, cutoff: float, fs: float) -> float:
    """The COMBINED (forward+backward) power gain of ``sos`` at ``cutoff`` -- what sosfiltfilt actually applies."""
    _, h = sosfreqz(sos, worN=[cutoff], fs=fs)
    return float(np.abs(np.asarray(h))[0] ** 2)


@pytest.mark.parametrize("cutoff", [0.4, 1.0, 3.0])
def test_prewarp_puts_the_combined_minus_3db_exactly_at_cutoff(cutoff):
    # A-48: measure the PROPERTY (the real filtfilt -3 dB), not the call. The prewarped design lands the combined
    # power gain on 1/sqrt(2) at `cutoff` at every frequency; the linear-Winter default drifts with the bilinear warp
    # (fine near DC, ~9% off at 3.0 Hz / 10 Hz), so it is the non-vacuity counterfactual.
    fs = 10.0
    prewarped = _dual_pass_gain(_design(fs, cutoff, 3, True), cutoff, fs)
    assert abs(prewarped - _INV_SQRT2) < 1e-6, (cutoff, prewarped)
    default = _dual_pass_gain(_design(fs, cutoff, 3, False), cutoff, fs)
    if cutoff >= 3.0:  # the warp is large here: the default is visibly off, the prewarp is exact
        assert abs(default - _INV_SQRT2) > 0.02, (cutoff, default)


def test_prewarp_end_to_end_attenuates_a_high_tone_more_than_the_too_wide_default():
    # end to end: a 3.0 Hz tone at 10 Hz undersamples the crest, so the discrete max underestimates the true
    # envelope (both designs equally) -- the RELATIVE claim is robust. The prewarped filter lands its combined -3 dB
    # EXACTLY at 3.0 Hz (gain 1/sqrt(2)), while the linear-Winter default under-corrects the bilinear warp -> its
    # design frequency is too high -> its -3 dB sits ABOVE 3.0, so it passes MORE of the tone. prewarp attenuates more.
    fs, cutoff = 10.0, 3.0
    t = np.arange(20000) / fs
    tone = np.sin(2 * np.pi * cutoff * t)
    pre = float(np.max(np.abs(butterworth_lowpass(tone, fs, cutoff, prewarp=True)[5000:15000])))
    default = float(np.max(np.abs(butterworth_lowpass(tone, fs, cutoff)[5000:15000])))
    assert default > pre + 0.02, (pre, default)


def test_prewarp_default_is_byte_identical_to_linear_winter():
    # the default path is untouched: prewarp=False designs exactly as before (velocity/smoothing bit-identity)
    x = np.sin(2 * np.pi * 0.2 * (np.arange(2000) / 10.0))
    np.testing.assert_array_equal(butterworth_lowpass(x, 10.0, 0.4), butterworth_lowpass(x, 10.0, 0.4, prewarp=False))


def test_exact_prewarped_cutoff_rejects_cutoff_at_or_above_nyquist():
    from silly_kicks.tracking.preprocess._butterworth import exact_prewarped_cutoff

    for bad in (5.0, 6.0, 0.0):
        with pytest.raises(ValueError, match=r"Nyquist|must be in"):
            exact_prewarped_cutoff(bad, 10.0, 3)
    assert exact_prewarped_cutoff(0.4, 10.0, 3) < 5.0  # always below Nyquist by construction (arctan bound)


def test_cutoff_at_or_above_nyquist_raises():
    x = np.zeros(500)
    for cutoff in (5.0, 4.5):  # 4.5 / winter(3) = 5.18 > Nyquist 5.0
        with pytest.raises(ValueError, match="Nyquist"):
            butterworth_lowpass(x, 10.0, cutoff)


def test_short_series_raises_with_min_length():
    mn = butterworth_min_length(10.0, 0.4, 3)
    with pytest.raises(ValueError, match="minimum"):
        butterworth_lowpass(np.zeros(mn - 1), 10.0, 0.4)
    butterworth_lowpass(np.zeros(mn), 10.0, 0.4)  # exactly min: must not raise


def test_residual_analysis_recovers_planted_cutoff():
    # Winter's method places the cutoff just ABOVE the signal band (keep the signal, cut the broadband noise).
    # The signal is band-limited to 0.55 Hz, so the recovered cutoff sits between the band edge and ~1.5 Hz --
    # comfortably below Nyquist and clearly separating signal from noise.
    rng = np.random.default_rng(3)
    t = np.arange(6000) / 25.0
    clean = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.55 * t)  # band-limited < 0.6 Hz
    noisy = clean + rng.normal(0, 0.05, t.size)
    c = residual_analysis_cutoff(noisy, 25.0, np.arange(0.1, 5.0, 0.05))
    assert 0.55 < c < 1.5


@pytest.mark.parametrize("sigma", [0.05, 0.12])
def test_residual_analysis_noise_rms_recovers_the_planted_noise_floor(sigma):
    # A-14: Winter's noise-floor RMS is the intercept of the residual curve's high-frequency tail. A band-limited
    # signal plus white noise of a KNOWN sigma -> the recovered floor is near sigma (the broadband noise level).
    from silly_kicks.tracking.preprocess._butterworth import residual_analysis_noise_rms

    rng = np.random.default_rng(7)
    t = np.arange(8000) / 25.0
    clean = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # band-limited < 0.6 Hz
    floor = residual_analysis_noise_rms(clean + rng.normal(0, sigma, t.size), 25.0, np.arange(0.1, 5.0, 0.05))
    assert abs(floor - sigma) < 0.3 * sigma  # within 30% of the planted noise level


def test_residual_analysis_noise_rms_raises_on_noise_free_input():
    from silly_kicks.tracking.preprocess._butterworth import residual_analysis_noise_rms

    t = np.arange(4000) / 25.0
    with pytest.raises(ValueError, match="no noise floor"):
        residual_analysis_noise_rms(np.sin(2 * np.pi * 0.3 * t), 25.0, np.arange(0.1, 5.0, 0.05))


def test_residual_analysis_noise_free_input():
    # A noise-free signal has no noise floor, so the residual method is ill-posed and raises (documented).
    t = np.arange(4000) / 25.0
    clean = np.sin(2 * np.pi * 0.3 * t)
    with pytest.raises(ValueError, match="no noise floor"):
        residual_analysis_cutoff(clean, 25.0, np.arange(0.1, 5.0, 0.05))


def test_residual_analysis_skips_grid_points_above_nyquist():
    # At 10 Hz the grid includes 5.0, whose design freq (5.0/winter) > Nyquist -> would raise if not skipped (C18).
    rng = np.random.default_rng(4)
    t = np.arange(6000) / 10.0
    noisy = np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.05, t.size)
    c = residual_analysis_cutoff(noisy, 10.0, np.arange(0.1, 5.0, 0.05))
    assert np.isfinite(c) and c < 5.0 * winter_correction(3)


def test_resample_uniform_exact_on_linear():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    out = resample_uniform(t, 2.0 * t + 1.0, 2.0, np.array([[0.0, 3.0]]), n_out=7)
    np.testing.assert_allclose(out, 2.0 * (np.arange(7) / 2.0) + 1.0, rtol=1e-12)


def test_resample_never_crosses_split():
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    v = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    runs = np.array([[0.0, 0.2], [1.0, 1.2]])
    out = resample_uniform(t, v, 10.0, runs, n_out=13)  # grid 0.0..1.2
    assert np.isnan(out[3:10]).all()  # the gap (0.3..0.9) is NaN
    assert out[0] == 0.0 and out[2] == 2.0  # run 1 only
    assert out[10] == 100.0 and out[12] == 102.0  # run 2 only (no bleed from run 1)


# F7: time stamps are floats, so a run bound can land an ulp either side of the grid point it equals in exact
# arithmetic. That grid point belongs to the run (np.interp's endpoint value), never NaN / never dropped.
def test_resample_uniform_keeps_a_grid_point_an_ulp_outside_a_run_bound():
    t = np.array([3, 4, 5]) * 0.1  # 0.30000000000000004, 0.4, 0.5
    assert t[0] > 3 / 10  # fixture precondition: the run starts an ulp AFTER grid point 3
    out = resample_uniform(t, 2.0 * t, 10.0, np.array([[t[0], t[-1]]]), n_out=6)
    np.testing.assert_allclose(out[3:], [0.6, 0.8, 1.0], rtol=1e-12)
    assert np.isnan(out[:3]).all()


def test_resample_uniform_default_n_out_covers_a_float_span_that_rounds_down():
    t = np.arange(2820, 2890) / 10.0 - 282.0  # t.max() = 6.899999999999977
    assert np.floor(t.max() * 10.0) == 68  # fixture precondition: the float span rounds down
    out = resample_uniform(t, 2.0 * t, 10.0, np.array([[0.0, t.max()]]))
    assert out.size == 70  # grid 0.0 .. 6.9 -- the last sample is not dropped
    np.testing.assert_allclose(out, 2.0 * np.arange(70) / 10.0, rtol=1e-12)


def test_resample_frames_float_edge_grid_point_is_kept_and_holds_its_own_run():
    # run 2 starts at 12 * 0.1 = 1.2000000000000002, an ulp after grid point 12: the grid row is kept and its
    # step-held columns come from run 2's first row, not run 1's last.
    t = np.array([0, 1, 2, 12, 13, 14]) * 0.1
    assert t[3] > 12 / 10  # fixture precondition
    frames = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": np.arange(6),
            "time_seconds": t,
            "frame_rate": 10.0,
            "player_id": 7,
            "team_id": 3,
            "is_ball": False,
            "is_goalkeeper": False,
            "x": 2.0 * t,
            "y": 0.0,
            "ball_state": ["alive", "alive", "alive", "dead", "dead", "dead"],
        }
    )
    out = resample_frames(frames, 10.0, max_gap_seconds=0.5).set_index("frame_id")
    assert out.index.tolist() == [0, 1, 2, 12, 13, 14]
    assert out.loc[12, "ball_state"] == "dead"
    assert out.loc[12, "x"] == pytest.approx(2.4)


def _tiny_frames() -> pd.DataFrame:
    rows = []
    for pid, base in ((1, 0.0), (2, 50.0)):
        for tt, xx in [(0.0, base), (0.1, base + 1.0), (0.2, base + 2.0), (1.0, base + 9.0), (1.1, base + 9.5)]:
            rows.append((tt, pid, xx))
    df = pd.DataFrame(rows, columns=["time_seconds", "player_id", "x"])
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = np.arange(len(df))
    df["frame_rate"] = 10.0
    df["team_id"] = 7
    df["is_ball"] = False
    df["is_goalkeeper"] = False
    df["y"] = 30.0
    df["speed"] = 1.0
    return df


def test_resample_frames_contract():
    frames = _tiny_frames()
    out = resample_frames(frames, 10.0, max_gap_seconds=0.5)
    assert (out["frame_rate"] == 10.0).all()
    np.testing.assert_allclose(out["time_seconds"].to_numpy(), out["frame_id"].to_numpy() / 10.0)
    assert out["speed"].isna().all()  # re-derive after resampling
    assert (out["team_id"] == 7).all()  # step-held
    # x interpolated within a run: player 1 at k=1 (t=0.1) -> 1.0
    p1 = out[out["player_id"] == 1].set_index("frame_id")
    assert p1.loc[1, "x"] == pytest.approx(1.0)
    # the gap (frame 3..9) carries no rows for either player
    assert out[(out["frame_id"] >= 3) & (out["frame_id"] <= 9)].empty


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_resample_frames_stores_at_the_input_dtype(dtype):
    # ADR-106: frame coordinates are float32 STORAGE / float64 COMPUTE -- the resampler interpolates in float64 and
    # stores back at the input dtype (like interpolate/smooth), and the NaN'd kinematic column keeps its dtype too.
    # nit: a pandas NULLABLE Float32/Float64 input must not raise (np.ndarray.astype cannot name an extension dtype).
    frames = _tiny_frames().astype({"x": dtype, "y": dtype, "speed": dtype})
    out = resample_frames(frames, 10.0, max_gap_seconds=0.5)
    assert len(out)  # non-vacuity: rows were produced
    assert {c: str(out[c].dtype) for c in ("x", "y", "speed")} == dict.fromkeys(("x", "y", "speed"), dtype)


def test_resample_frames_keeps_every_game_apart():
    # The resampling entity includes the game: two games with the same player id resample separately -- neither
    # game's rows are dropped or interpolated into the other's (the entity once omitted game_id: 12 rows in, 6 out).
    rows = []
    for game, base in ((1, 0.0), (2, 100.0)):
        for k in range(6):
            rows.append(
                {
                    "game_id": game,
                    "period_id": 1,
                    "frame_id": k,
                    "time_seconds": k / 10.0,
                    "frame_rate": 10.0,
                    "player_id": 7,
                    "team_id": 3,
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": base + k,
                    "y": 0.0,
                }
            )
    out = resample_frames(pd.DataFrame(rows), 10.0)
    assert len(out) == 12
    for game, base in ((1, 0.0), (2, 100.0)):
        part = out[out["game_id"] == game].sort_values("frame_id")
        np.testing.assert_allclose(part["x"].to_numpy(), base + np.arange(6.0))


def test_smooth_frames_butterworth_additive_and_tagged():
    # a long single-player series that exceeds the sosfiltfilt minimum
    t = np.arange(400) / 10.0
    frames = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": np.arange(400),
            "time_seconds": t,
            "frame_rate": 10.0,
            "player_id": 1,
            "team_id": 7,
            "is_ball": False,
            "is_goalkeeper": False,
            "x": np.sin(2 * np.pi * 0.05 * t),
            "y": np.cos(2 * np.pi * 0.05 * t),
        }
    )
    cfg = PreprocessConfig(smoothing_method="butterworth", derive_velocity=False)
    out = smooth_frames(frames, config=cfg)
    np.testing.assert_array_equal(out["x"].to_numpy(), frames["x"].to_numpy())  # raw preserved
    assert "x_smoothed" in out.columns and "y_smoothed" in out.columns
    assert (out["_preprocessed_with"] == "method=butterworth|bw_cutoff_hz=0.4|bw_order=3").all()
    again = smooth_frames(out, config=cfg)  # idempotent
    np.testing.assert_array_equal(again["x_smoothed"].to_numpy(), out["x_smoothed"].to_numpy())


def test_smooth_frames_savgol_ema_tags_unchanged():
    frames = _tiny_frames()
    sg = smooth_frames(frames, config=PreprocessConfig(smoothing_method="savgol"))
    ema = smooth_frames(frames, config=PreprocessConfig(smoothing_method="ema"))
    assert (sg["_preprocessed_with"] == "method=savgol|sg_window_s=0.4|sg_poly=3|ema_alpha=0.3").all()
    assert (ema["_preprocessed_with"] == "method=ema|sg_window_s=0.4|sg_poly=3|ema_alpha=0.3").all()


def test_smooth_frames_short_group_passes_through():
    frames = _tiny_frames()  # 5 rows per player -- far below the butterworth minimum
    cfg = PreprocessConfig(smoothing_method="butterworth", derive_velocity=False)
    out = smooth_frames(frames, config=cfg)
    p1 = out[out["player_id"] == 1]
    np.testing.assert_array_equal(p1["x_smoothed"].to_numpy(), p1["x"].to_numpy())  # pass-through


# --------------------------------------------------------------------------- design cache (ADR-111 perf)
# The cache is a pure memoization: it must not change any output. The gate is byte-identity against a fresh,
# independently-computed scipy design over a battery spanning the provider rates + the residual-analysis grid.
from scipy.signal import butter as _butter_oracle  # noqa: E402


def _fresh_sos(fs, cutoff_hz, order):
    """A from-scratch SOS oracle bypassing the module cache (same math as _design_sos)."""
    return np.asarray(
        _butter_oracle(order, cutoff_hz / winter_correction(order), btype="low", fs=fs, output="sos"),
        dtype=np.float64,
    )


_DESIGN_BATTERY = [(10.0, 0.4, 3), (25.0, 0.4, 3), (10.0, 0.22, 3), (25.0, 1.35, 3), (10.0, 0.83, 2)]


@pytest.mark.parametrize(("fs", "cutoff", "order"), _DESIGN_BATTERY)
def test_design_cache_is_byte_identical_to_a_fresh_design(fs, cutoff, order):
    np.testing.assert_array_equal(_design(fs, cutoff, order), _fresh_sos(fs, cutoff, order))


@pytest.mark.parametrize(("fs", "cutoff", "order"), _DESIGN_BATTERY)
def test_butterworth_lowpass_byte_identical_pre_and_post_cache(fs, cutoff, order):
    # the pre/post-cache proof for the SHARED consumer path: filtered output == a from-scratch sosfiltfilt.
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 0.1 * (np.arange(4000) / fs)) + rng.normal(0, 0.02, 4000)
    got = butterworth_lowpass(x, fs, cutoff, order)
    want = np.asarray(sosfiltfilt(_fresh_sos(fs, cutoff, order), np.asarray(x, dtype=np.float64)), dtype=np.float64)
    np.testing.assert_array_equal(got, want)


def test_design_cache_returns_one_readonly_shared_object():
    a = _design(10.0, 0.4, 3)
    b = _design(10.0, 0.4, 3)
    assert a is b  # the cache serves one object, not a per-call copy
    assert a.flags.writeable is False
    with pytest.raises(ValueError, match=r"read-only|assignment destination"):
        a[0, 0] = 123.0  # a mutation must fail LOUD, never corrupt other callers


def test_design_cache_canonicalises_numpy_and_python_scalars_to_one_key():
    # deliberately pass numpy scalars (runtime-valid; the canonicalisation coerces them) -> one cache key.
    npy = _design(np.float64(10.0), np.float64(0.4), np.int64(3))  # pyright: ignore[reportArgumentType]
    assert npy is _design(10.0, 0.4, 3)


def test_min_length_cached_matches_a_fresh_computation():
    for fs, cutoff, order in _DESIGN_BATTERY:
        sos = _fresh_sos(fs, cutoff, order)
        ntaps = 2 * sos.shape[0] + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
        assert butterworth_min_length(fs, cutoff, order) == 3 * ntaps + 1


def test_readonly_master_reads_fine_and_lowpass_copies_for_sosfiltfilt():
    # sosfreqz + column reads take the read-only master directly; scipy's sosfilt writes to the SOS buffer, so
    # butterworth_lowpass hands it a copy -- the read-only master must never reach sosfiltfilt (would raise).
    sos = _design_sos(10.0, 0.4, 3)
    sosfreqz(sos, worN=[0.4], fs=10.0)  # a pure reader: read-only OK
    with pytest.raises(ValueError, match=r"read-only"):
        sosfiltfilt(sos, np.zeros(butterworth_min_length(10.0, 0.4, 3)))  # proves the copy in lowpass is load-bearing
    butterworth_lowpass(np.zeros(butterworth_min_length(10.0, 0.4, 3)), 10.0, 0.4, 3)  # copies internally -> works


# --------------------------------------------------------------------------- sosfiltfilt replica (ADR-111 perf, D3)
# butterworth_lowpass replicates scipy's sosfiltfilt step for step with the steady-state `zi` solved ONCE per
# design. FENCE: the replica must equal scipy's real sosfiltfilt bit for bit on every run length the coordination
# signal prep can hand it (the fragmented SkillCorner runs sit right at the minimum length), so a scipy upgrade that
# changes sosfiltfilt fails here loudly instead of silently diverging.
def _fence_lengths(fs, cutoff, order):
    mn = butterworth_min_length(fs, cutoff, order)
    return sorted({mn, mn + 1, mn + 2, mn + 7, 2 * mn, 2 * mn + 1, 97, 250, 1001, 4096} - set(range(mn)))


@pytest.mark.parametrize(("fs", "cutoff", "order"), _DESIGN_BATTERY)
def test_butterworth_lowpass_is_scipy_sosfiltfilt_bit_for_bit(fs, cutoff, order):
    rng = np.random.default_rng(int(fs * 100 + cutoff * 1000 + order))
    for n in _fence_lengths(fs, cutoff, order):
        for x in (
            rng.normal(0.0, 1.0, n),  # broadband
            50.0 + 20.0 * np.sin(2 * np.pi * 0.05 * np.arange(n) / fs) + rng.normal(0.0, 0.3, n),  # pitch-like
            np.full(n, 34.0),  # constant (the steady-state zi path)
        ):
            got = butterworth_lowpass(x, fs, cutoff, order)
            want = np.asarray(sosfiltfilt(_fresh_sos(fs, cutoff, order), x), dtype=np.float64)
            assert got.view(np.int64).tolist() == want.view(np.int64).tolist(), (n, fs, cutoff, order)


@pytest.mark.parametrize(("fs", "cutoff", "order"), _DESIGN_BATTERY)
def test_butterworth_lowpass_rows_equals_each_row_bit_for_bit(fs, cutoff, order):
    # The coordination signal prep filters a run's x and y in ONE two-row pass (ADR-111 ruling C). scipy's sosfilt
    # runs every row through the same recursion, so each row equals its own 1-D butterworth_lowpass bit for bit.
    from silly_kicks.tracking.preprocess._butterworth import butterworth_lowpass_rows

    rng = np.random.default_rng(int(fs * 10 + cutoff * 100 + order))
    for n in _fence_lengths(fs, cutoff, order):
        rows = np.stack([rng.normal(0.0, 1.0, n), 50.0 + np.cumsum(rng.normal(0.0, 0.3, n)), np.full(n, 34.0)])
        got = butterworth_lowpass_rows(rows, fs, cutoff, order)
        assert got.shape == rows.shape
        for i in range(rows.shape[0]):
            want = butterworth_lowpass(rows[i], fs, cutoff, order)
            assert got[i].view(np.int64).tolist() == want.view(np.int64).tolist(), (n, i)
    with pytest.raises(ValueError, match="minimum"):
        butterworth_lowpass_rows(np.zeros((2, butterworth_min_length(fs, cutoff, order) - 1)), fs, cutoff, order)


def test_butterworth_lowpass_solves_the_steady_state_once_per_design(monkeypatch):
    # R8 structural guard: scipy's sosfiltfilt re-solves sosfilt_zi (lfilter_zi -> a linear solve per section) on
    # EVERY call, which dominated the per-run filtering of fragmented tracking (17k calls per SkillCorner match).
    # The replica caches it per (fs, cutoff, order): 200 calls on one design solve it at most once.
    import scipy.signal._signaltools as scipy_signaltools

    from silly_kicks.tracking.preprocess import _butterworth
    from tests._perf_structural import call_counter

    _butterworth._steady_state_zi.cache_clear()
    solves = call_counter(monkeypatch, scipy_signaltools, "sosfilt_zi")
    replica_solves = call_counter(monkeypatch, _butterworth, "sosfilt_zi")
    x = np.random.default_rng(1).normal(size=500)
    for _ in range(200):
        butterworth_lowpass(x, 10.0, 0.4, 3)
    assert solves["n"] + replica_solves["n"] <= 1


def test_steady_state_zi_is_one_readonly_shared_object():
    from silly_kicks.tracking.preprocess._butterworth import _steady_state_zi

    zi = _steady_state_zi(10.0, 0.4, 3)
    assert zi is _steady_state_zi(10.0, 0.4, 3)
    assert zi.flags.writeable is False
    with pytest.raises(ValueError, match=r"read-only|assignment destination"):
        zi[0, 0] = 1.0


def test_preprocess_config_butterworth_fields():
    assert PreprocessConfig().butterworth_cutoff_hz == 0.4
    assert PreprocessConfig().butterworth_order == 3
    with pytest.raises(ValueError, match="butterworth_cutoff_hz"):
        PreprocessConfig(butterworth_cutoff_hz=0.0)
    with pytest.raises(ValueError, match="butterworth_order"):
        PreprocessConfig(butterworth_order=0)
    PreprocessConfig(smoothing_method="butterworth", derive_velocity=True)  # accepted
