"""TF-58 Task 16: signal preparation (refusals, effective rate, orientation, segments, phasors)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from silly_kicks.coordination import _signals
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows, possession_windows_from_actions
from silly_kicks.id_compat import same_id
from silly_kicks.tracking._collective import collective_from_positions, compact_rows
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore:vx/vy columns not found")


def _valid_frames(**kw):
    f = make_coordination_match(seconds=90.0, hz=10.0, provider="sportec", **kw)
    return f, period_windows(f)


def test_phasors_over_runs_does_not_count_each_runs_first_sample_as_non_advancing():
    # A-45: the instantaneous-advance indicator forced the FIRST sample of every run to non-advancing, biasing
    # coord_rp_phase_valid_fraction down by n_runs/n. A clean sinusoid advances monotonically, so a two-run window's
    # fraction must be ~1.0, with each run's first sample excluded (NaN), not counted as 0.
    fs, f = 10.0, 0.3
    x = np.sin(2 * np.pi * f * np.arange(0.0, 20.0, 1.0 / fs))  # 200 samples, 6 cycles
    runs = [(0, 100), (100, 200)]
    _ph, ipos = _signals._phasors_over_runs(x, runs, fs, 0.22)
    assert ipos.dtype == np.float64, "the indicator must be float (NaN-capable), not a bool that forces 0 at run start"
    assert np.isnan(ipos[0]) and np.isnan(ipos[100]), "each run's first sample has no predecessor -> NaN, not False"
    idx = np.r_[np.arange(0, 100), np.arange(100, 200)]
    # the 2 run-firsts drop out of the denominator: #adv/198 (new) vs #adv/200 (old, counting them as 0). A few Hilbert
    # edge transients give #adv ~= 194, so new ~= 0.980 > old ~= 0.970 -- the structural NaNs above are the real proof.
    assert np.nanmean(ipos[idx]) > 0.975


def test_apply_filter_seam_default_is_the_public_path_and_actually_filters():
    """The private ``apply_filter`` seam (D1 Tier-B): the PUBLIC ``build_coordination_signals`` is byte-for-byte
    the ``apply_filter=True`` path (the refactor did not fork the default), and the flag is load-bearing --
    ``apply_filter=False`` yields the RAW-resampled signal, aligned element-wise (same NaN mask) but un-smoothed."""
    f, w = _valid_frames()
    pub = build_coordination_signals(f, windows=w)
    default = _signals._build_coordination_signals(f, windows=w, apply_filter=True)
    raw = _signals._build_coordination_signals(f, windows=w, apply_filter=False)
    ps_pub, ps_raw = pub.periods[0], raw.periods[0]
    # 1) the public entry point IS the filtered implementation, byte-identical on every team signal.
    for key, arr in ps_pub.team_signal.items():
        assert np.array_equal(arr, default.periods[0].team_signal[key], equal_nan=True)
    # 2) raw is aligned to filtered (same NaN pattern), the flag changes the output, and it truly low-passes
    #    (the filtered centroid is smoother -- a smaller mean |first difference| -- than the raw-resampled one).
    tm = ps_pub.team_ids[0]
    filt = ps_pub.team_signal[(tm, "centroid_x")]
    rawx = ps_raw.team_signal[(tm, "centroid_x")]
    assert np.array_equal(np.isnan(filt), np.isnan(rawx))
    m = np.isfinite(filt)
    assert not np.allclose(filt[m], rawx[m])
    assert np.mean(np.abs(np.diff(filt[m]))) < np.mean(np.abs(np.diff(rawx[m])))


# --------------------------------------------------------------------------- refusals (spec 7.3)
def test_every_refusal():
    f, w = _valid_frames()
    with pytest.raises(ValueError, match="missing required columns"):
        build_coordination_signals(f.drop(columns=["x"]), windows=w)
    snap = f.copy()
    snap["source_provider"] = "snapshot"
    with pytest.raises(ValueError, match="snapshot"):
        build_coordination_signals(snap, windows=w)
    mixed = f.copy()
    mixed["source_provider"] = mixed["source_provider"].astype(object)
    mixed.loc[mixed.index[:5], "source_provider"] = "idsse"
    with pytest.raises(ValueError, match="mix source_provider"):
        build_coordination_signals(mixed, windows=w)
    unoriented = f.copy()
    unoriented["team_attacking_direction"] = None
    with pytest.raises(ValueError, match="unoriented"):
        build_coordination_signals(unoriented, windows=w)
    dup = pd.concat([f, f[~f["is_ball"]].iloc[[10]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_coordination_signals(dup, windows=w)
    two_balls = pd.concat([f, f[f["is_ball"]].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="ball row"):
        build_coordination_signals(two_balls, windows=w)
    two_rates = f.copy()
    two_rates.loc[two_rates.index[:5], "frame_rate"] = 25.0
    with pytest.raises(ValueError, match="frame_rate"):
        build_coordination_signals(two_rates, windows=w)


def test_ball_state_is_a_required_column():
    # spec 7.3: ball_state is required (the stoppage evidence reads it); a missing column must be the 7.3 ValueError
    # naming it, not a later KeyError (review A-29)
    f, w = _valid_frames()
    with pytest.raises(ValueError, match=r"missing required columns.*ball_state"):
        build_coordination_signals(f.drop(columns=["ball_state"]), windows=w)


def test_visibility_required_only_for_detection_aware_providers():
    # spec 7.3: visibility is required for detection-aware providers, optional otherwise (A-29). A fully-observed
    # provider without a visibility column must build; a detection-aware one without it must refuse, naming it.
    sportec = make_coordination_match(seconds=30.0, hz=10.0, provider="sportec").drop(columns=["visibility"])
    build_coordination_signals(sportec, windows=period_windows(sportec))  # fully observed: no visibility needed
    sk = make_coordination_match(seconds=30.0, hz=10.0, provider="skillcorner").drop(columns=["visibility"])
    with pytest.raises(ValueError, match="visibility"):
        build_coordination_signals(sk, windows=period_windows(sk))


def test_unclassified_provider_and_discarded_visibility_refused():
    f = make_coordination_match(seconds=30.0, hz=10.0, provider="skillcorner")
    w = period_windows(f)
    wyscout = f.copy()
    wyscout["source_provider"] = "wyscout"
    with pytest.raises(ValueError, match="unclassified provider"):
        build_coordination_signals(wyscout, windows=w)
    discarded = f.copy()
    discarded["visibility"] = None  # detection-aware provider with all-null visibility
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        build_coordination_signals(discarded, windows=w)


# --------------------------------------------------------------------------- effective rate
def test_effective_rate_default_raised_and_capped():
    f10, w10 = _valid_frames()
    cs = build_coordination_signals(f10, windows=w10)
    assert cs.fs == 10.0 and cs.rate_capped is False  # cutoff 0.4 -> target 10, native 10

    f25 = make_coordination_match(seconds=90.0, hz=25.0, provider="sportec")
    w25 = period_windows(f25)
    hi_cut = dataclasses.replace(CoordinationParams(), butterworth_cutoff_hz=1.5)
    raised = build_coordination_signals(f25, windows=w25, params=hi_cut)
    assert raised.fs == 15.0 and raised.rate_capped is False  # target max(10, 15)=15 <= native 25

    capped = build_coordination_signals(f10, windows=w10, params=hi_cut)
    assert capped.fs == 10.0 and capped.rate_capped is True  # target 15 > native 10 -> capped

    over_nyquist = dataclasses.replace(CoordinationParams(), butterworth_cutoff_hz=5.0)
    with pytest.raises(ValueError, match="Nyquist"):
        build_coordination_signals(f10, windows=w10, params=over_nyquist)


# --------------------------------------------------------------------------- runs / segments
def test_short_run_dropped_and_counted():
    f = make_coordination_match(seconds=1.0, hz=10.0, provider="sportec")  # 10 samples < butterworth min length 13
    cs = build_coordination_signals(f, windows=period_windows(f))
    assert cs.counters["n_runs_too_short"] > 0
    a = cs.periods[0].team_ids[0]
    assert np.isnan(cs.periods[0].team_signal[(a, "centroid_x")]).all()


def test_too_short_runs_are_counted_once_per_run():
    """Spec 7.4 step 3: a too-short run is dropped and counted ONCE -- not once per coordinate (x, y) and once more
    per pipeline path (player series, team signals), which counted every outfield run 4x and a goalkeeper's 2x."""
    f = make_coordination_match(seconds=1.0, hz=10.0, provider="sportec")  # every player: ONE 10-sample run < 13
    cs = build_coordination_signals(f, windows=period_windows(f))
    n_players = f.loc[~f["is_ball"], ["team_id", "player_id"]].drop_duplicates().shape[0]  # 11 per team incl. the GK
    assert n_players == 22
    assert cs.counters["n_runs_too_short"] == n_players


def test_red_card_splits_team_segment_and_steps_count():
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", red_card=(60.0, 3))
    cs = build_coordination_signals(f, windows=period_windows(f))
    a = cs.periods[0].team_ids[0]
    segs = cs.periods[0].segments[a]
    assert len(segs) == 2  # the on-pitch count steps down at t=60 -> split
    assert segs[0][1] == segs[1][0]


def _segments(counts, handover_samples, *, valid=None):
    counts = np.asarray(counts)
    valid = np.ones(counts.size, dtype=bool) if valid is None else valid
    stoppage = np.zeros(counts.size, dtype=bool)
    out = _signals._segments_with_count(valid, counts, stoppage, handover_samples=handover_samples)
    return out.tolist()


@pytest.mark.parametrize(("k", "split"), [(50, False), (51, True)], ids=["excursion-at-tolerance", "longer-splits"])
def test_count_excursion_within_the_handover_tolerance_does_not_split(k, split):
    # owner ruling 2026-10-03 (review M-13): an inexact substitution handover moves the count off its level and back
    # within max_stoppage_s -- that is no change of team composition (spec 7.4 step 2: "a substitution keeps the count
    # and does not split"); an excursion one sample longer is a real change and splits.
    counts = [10] * 100 + [11] * k + [10] * 100
    want = [[0, 100], [100, 100 + k], [100 + k, 200 + k]] if split else [[0, 200 + k]]
    assert _segments(counts, 50) == want


def test_a_count_change_that_does_not_return_always_splits():
    # a red card / a player who never comes back: the count does not return to its prior level, however short the
    # step -- and an excursion with no prior level (at the period start) cannot be judged a handover either.
    assert _segments([10] * 100 + [9] * 5 + [8] * 100, 50) == [[0, 100], [100, 105], [105, 205]]
    assert _segments([11] * 5 + [10] * 100, 50) == [[0, 5], [5, 105]]


def test_a_count_excursion_at_the_period_END_is_kept_not_absorbed():
    # A-43: `_absorb_handovers` only absorbs a run bounded by the SAME count on BOTH sides, so an excursion touching the
    # period END (no "after" level to return to) is kept and splits -- the symmetric case to the period-START test
    # above. A short excursion that DOES return before the end is still absorbed (the other side of the band).
    assert _segments([10] * 100 + [11] * 5, 50) == [[0, 100], [100, 105]]  # edge excursion: kept, however short
    assert _segments([10] * 100 + [11] * 5 + [10] * 100, 50) == [[0, 205]]  # the same step, now interior: absorbed


def test_nested_excursions_are_absorbed_from_the_inside_out():
    # two overlapping handovers (10 -> 11 -> 12 -> 11 -> 10) within the tolerance: one segment
    assert _segments([10] * 100 + [11] * 3 + [12] * 2 + [11] * 3 + [10] * 100, 50) == [[0, 208]]


def test_a_multi_step_excursion_that_returns_by_another_path_is_absorbed():
    # B m1 / R2-5: a double substitution with staggered exits (11 -> 10 -> 9 -> 11) leaves the level and FIRST returns
    # to it within the tolerance -- that is a handover, not a composition change, so it must not split (the ruling is
    # "returns to its prior count", by any path). The mirror 11 -> 12 -> 11 already absorbs; this is the same from the
    # other side. One sample longer than the tolerance still splits.
    counts = [11] * 100 + [10] * 20 + [9] * 20 + [11] * 150
    assert _segments(counts, 250) == [[0, 290]]  # the 40-sample excursion returns to 11 within tolerance -> absorbed
    # one sample longer than the tolerance -> the excursion is a real composition change and splits
    assert _segments(counts, 39) == [[0, 100], [100, 120], [120, 140], [140, 290]]


@pytest.mark.parametrize(("gap_s", "n_segments"), [(24.0, 1), (26.0, 3)], ids=["gap-24s-absorbed", "gap-26s-splits"])
def test_late_substitute_handover_both_sides_of_max_stoppage(gap_s, n_segments):
    # The fixture's substitution is exact; plant a late incoming player (no rows for gap_s after the outgoing one's
    # last row): team 1's count dips 10 -> 9 for gap_s. Under max_stoppage_s (25 s) it is a handover (one segment);
    # over it, a real count change (three segments).
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", substitution=(60.0, 3))
    late = (f["player_id"] == 1 * 100 + 900 + 3).to_numpy(dtype=bool, na_value=False) & (
        f["time_seconds"] < 60.0 + gap_s
    ).to_numpy()
    assert late.sum() == round(gap_s * 10)  # fixture precondition (ADR-032): the incoming player is gap_s late
    f = f[~late].reset_index(drop=True)
    p = build_coordination_signals(f, windows=period_windows(f)).periods[0]
    team_1 = next(tm for tm in p.team_ids if same_id(tm, 1))
    assert len(p.segments[team_1]) == n_segments


def test_substitution_no_splice():
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", substitution=(60.0, 3))
    cs = build_coordination_signals(f, windows=period_windows(f))
    p = cs.periods[0]
    a = p.team_ids[0]
    assert len(p.segments[a]) == 1  # a substitution keeps the on-pitch count constant -> no split
    a_players = {pid: ps for (tm, pid), ps in p.players.items() if tm == a and not ps.is_goalkeeper}
    assert len(a_players) == 11  # 10 slots + 1 replacement id
    # exactly one pair (the substituted player and the replacement) is disjoint on-pitch; the rest overlap.
    on = [ps.on_pitch for ps in a_players.values()]
    non_overlap = sum(1 for i, x in enumerate(on) for y in on[i + 1 :] if not (x & y).any())
    assert non_overlap == 1


@pytest.mark.parametrize(
    ("dead", "runs"),
    [([(60.0, 120.0)], [[0, 600], [1200, 1800]]), ([(60.0, 80.0)], [[0, 1800]])],
    ids=["longer-than-max-stoppage-splits", "shorter-does-not"],
)
def test_long_stoppage_splits_every_player_run(dead, runs):
    # spec 7.4 step 2: a dead-ball stoppage longer than max_stoppage_s (25 s) splits EVERY player's run -- before the
    # filter -- not only the team segments, so no player phase, dyad or cluster sample spans it. 20 s splits nothing.
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec", dead_intervals=dead)
    p = build_coordination_signals(f, windows=period_windows(f)).periods[0]
    assert len(p.players) == 22  # fixture precondition (ADR-032): every player, both keepers included
    for key, series in p.players.items():
        assert series.runs.tolist() == runs, key
    for tm in p.team_ids:
        assert p.segments[tm].tolist() == runs  # the team segments split at the same place


@pytest.mark.parametrize(("extra_missing", "n_runs"), [(0, 1), (1, 2)], ids=["gap-at-max-bridged", "longer-splits"])
def test_detection_gap_split_both_sides(extra_missing, n_runs):
    # spec 9.3: on a detection-aware provider, consecutive detections max_detection_gap_s apart are bridged (one
    # run); one more missing sample splits the player's run in two.
    params = CoordinationParams.for_provider("skillcorner")
    hz = 10.0
    gap_steps = round(params.max_detection_gap_s * hz)
    f = make_coordination_match(seconds=120.0, hz=hz, provider="skillcorner")
    pid, lo = 103, 600  # team 1, slot 3; the gap opens after frame 600 (t = 60 s)
    hidden = (
        (f["player_id"] == pid).to_numpy(dtype=bool, na_value=False)
        & (f["frame_id"] > lo).to_numpy()
        & (f["frame_id"] < lo + gap_steps + extra_missing).to_numpy()
    )
    assert hidden.sum() == gap_steps - 1 + extra_missing  # fixture precondition (ADR-032)
    f.loc[hidden, "visibility"] = False
    cs = build_coordination_signals(f, windows=period_windows(f), params=params)
    series = next(ps for (_tm, p), ps in cs.periods[0].players.items() if same_id(p, pid))
    assert len(series.runs) == n_runs


def test_possession_series_hold_rule():
    # spec 7.8.4: between possession windows the series holds the last possession; before the first it is NA.
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    poss = possession_windows_from_actions(make_coordination_actions(f), f, n_phases=3)
    poss = poss.sort_values("start_time_s").reset_index(drop=True)
    held, gap = poss.iloc[2], poss.iloc[3]
    kept = poss.drop(index=[0, 3])  # no window before the second possession; a hole where the fourth was
    cs = build_coordination_signals(f, windows=pd.concat([period_windows(f), kept], ignore_index=True))
    p = cs.periods[0]
    before = p.t < float(kept["start_time_s"].min())
    assert before.any() and all(pd.isna(v) for v in p.possession_team[before])
    in_gap = (p.t >= float(gap["start_time_s"])) & (p.t < float(gap["end_time_s"]))
    assert in_gap.any() and all(same_id(v, held["attacking_team_id"]) for v in p.possession_team[in_gap])
    # non-vacuity: the held team is NOT the team of the window that was removed
    assert not same_id(held["attacking_team_id"], gap["attacking_team_id"])


# --------------------------------------------------------------------------- resampling (spec 7.4 step 4; F7)
# Each filtered run is resampled by ONE linear interpolation at the period-grid times it covers. Time stamps are
# floats, so a run's first/last time can land an ulp either side of the grid point it equals in exact arithmetic
# (288.9 - 282.0 = 6.899999999999977; 2331 * 0.1 = 233.10000000000002). F7: the former two-stage resample (onto a
# run-local grid, then re-interpolated onto the period grid with NaN outside) turned those edge points into NaN --
# and `_phasors_over_runs` then dropped the WHOLE run's phase (80% of SkillCorner resample calls, measured).
def _resample(t, v, *, fs_native=10.0):
    counters = {"n_runs_too_short": 0}
    (out,), runs = _signals._resample_runs(t, (v,), 10.0, fs_native, 0.4, 3, 3000, 0.5, counters, apply_filter=False)
    return out, runs


@pytest.mark.parametrize(
    "t",
    [
        pytest.param(np.arange(2820, 2890) / 10.0, id="float-span-rounds-down"),
        pytest.param(np.arange(2331, 2401) * 0.1, id="start-an-ulp-after-its-grid-point"),
    ],
)
def test_resample_keeps_the_run_edge_samples(t):
    # fixture precondition (ADR-032): each case really exercises a float edge
    span_rounds_down = np.floor((t[-1] - t[0]) * 10.0) < round((t[-1] - t[0]) * 10.0)
    start_after_grid_point = t[0] > round(t[0] * 10.0) / 10.0
    assert span_rounds_down or start_after_grid_point
    v = 3.0 * t - 100.0  # linear: resampling is exact (spec 9.1)
    out, runs = _resample(t, v)
    lo, hi = round(t[0] * 10.0), round(t[-1] * 10.0) + 1
    assert runs == [(lo, hi)]
    grid = np.arange(lo, hi) / 10.0
    np.testing.assert_allclose(out[lo:hi], 3.0 * grid - 100.0, rtol=0.0, atol=1e-9)  # every sample, edges included
    assert np.isnan(out[:lo]).all() and np.isnan(out[hi:]).all()  # nothing outside the run


def test_resample_interpolates_an_off_grid_run_once():
    """Spec 7.4 step 4: ONE linear interpolation of the run at the period-grid times. A 25 Hz run starting off the
    10 Hz grid (t0 = 12.04 s) equals ``np.interp(grid, t, v)`` -- not an interpolation of an interpolation (the
    former two-stage path re-interpolated between points 0.1 s apart, a ~2e-3 error on this curved signal)."""
    t = np.arange(301, 401) / 25.0  # 12.04 .. 16.0 s
    v = (t - 14.0) ** 2
    out, runs = _resample(t, v, fs_native=25.0)
    assert runs == [(121, 161)]
    grid = np.arange(121, 161) / 10.0
    np.testing.assert_allclose(out[121:161], np.interp(grid, t, v), rtol=1e-13, atol=1e-13)


# --------------------------------------------------------------------------- bridge, THEN filter (spec 7.4 steps 2-3)
# Step 2 bridges a run's detection gaps (up to max_detection_gap_s) by linear interpolation; step 3 filters the run at
# the native rate. Review A-07: the run was filtered on its DETECTED samples as if they were evenly spaced and only
# then bridged on the grid -- a 0.2-s gap moved a ~4 m/s player by 0.57 m, a 0.5-s gap by 1.36 m.
_FS_NATIVE, _CUTOFF, _ORDER = 25.0, 0.4, 3


def _weaving_run(missing: int):
    t = np.arange(1500) / _FS_NATIVE  # 60 s at 25 Hz
    v = 4.0 * t + 3.0 * np.sin(2 * np.pi * t / 20.0)  # ~4 m/s with a slow weave
    keep = np.ones(t.size, dtype=bool)
    keep[701 : 701 + missing] = False  # one detection gap of (missing + 1) native steps
    return t, v, keep


def _resample_filtered(t, v):
    counters = {"n_runs_too_short": 0}
    (out,), runs = _signals._resample_runs(t, (v,), 10.0, _FS_NATIVE, _CUTOFF, _ORDER, 600, 0.5, counters)
    return out, runs, counters


@pytest.mark.parametrize("missing", [4, 11])  # a 0.2-s and a 0.48-s gap, both within max_detection_gap_s = 0.5
def test_a_detection_gap_is_bridged_before_the_run_is_filtered(missing):
    from silly_kicks.tracking.preprocess._butterworth import butterworth_lowpass

    t, v, keep = _weaving_run(missing)
    out, runs, _ = _resample_filtered(t[keep], v[keep])
    assert runs == [(0, 600)]  # one run: the gap is bridged, not split
    grid = np.arange(600) / 10.0
    bridged = np.interp(t, t[keep], v[keep])  # step 2 at the native timestamps
    # production prewarps the design (A-48), so the oracle must too
    oracle = np.interp(grid, t, butterworth_lowpass(bridged, _FS_NATIVE, _CUTOFF, _ORDER, prewarp=True))
    np.testing.assert_allclose(out, oracle, rtol=0.0, atol=1e-9)
    # non-vacuity: filtering the detected samples first really is a different answer on this fixture
    filter_first = np.interp(grid, t[keep], butterworth_lowpass(v[keep], _FS_NATIVE, _CUTOFF, _ORDER, prewarp=True))
    assert np.max(np.abs(filter_first - oracle)) > 0.1


def test_a_gap_free_run_is_filtered_exactly_as_before():
    # the other side: with every native sample detected nothing is inserted -- bit for bit the filter-then-resample path
    t, v, _ = _weaving_run(0)
    out, runs, _ = _resample_filtered(t, v)
    assert runs == [(0, 600)]
    want = np.interp(
        np.arange(600) / 10.0,
        t,
        _signals.butterworth_lowpass_rows(v[None, :], _FS_NATIVE, _CUTOFF, _ORDER, prewarp=True)[0],
    )
    np.testing.assert_array_equal(out, want)


def test_run_admission_counts_the_bridged_run():
    # sosfiltfilt needs the FILTERED array longer than its pad: that array is the bridged run, so a run with fewer
    # detected samples than the minimum but a long enough bridged span is filtered, not dropped (spec 7.4 step 3)
    from silly_kicks.tracking.preprocess._butterworth import butterworth_min_length

    min_len = butterworth_min_length(_FS_NATIVE, _CUTOFF, _ORDER)
    t = np.arange(min_len + 4) / _FS_NATIVE
    v = np.sin(t)
    keep = np.ones(t.size, dtype=bool)
    keep[10:16] = False  # 6 missing samples: detected count below the minimum, bridged span above it
    assert keep.sum() < min_len <= t.size  # fixture precondition (ADR-032)
    out, runs, counters = _resample_filtered(t[keep], v[keep])
    assert counters["n_runs_too_short"] == 0 and len(runs) == 1
    assert np.isfinite(out[runs[0][0] : runs[0][1]]).all()
    # the other side: the same run with its span cut below the minimum is still dropped and counted
    short = keep & (np.arange(t.size) < min_len - 1)
    out, runs, counters = _resample_filtered(t[short], v[short])
    assert counters["n_runs_too_short"] == 1 and runs == []


def test_player_run_keeps_its_phase_when_its_float_span_rounds_down():
    """F7 end to end: a detection run over [10.4, 89.8] s has a float span of 793.9999999999999 samples; it keeps a
    finite position AND phasor over every sample instead of losing the whole run's phase."""
    f = make_coordination_match(seconds=90.0, hz=10.0, provider="skillcorner")
    t = f["time_seconds"].to_numpy()
    pid = f.loc[~f["is_ball"] & ~f["is_goalkeeper"], "player_id"].iloc[0]
    rows = (f["player_id"] == pid).fillna(False).to_numpy()
    f["visibility"] = f["visibility"].astype(object)
    f.loc[rows, "visibility"] = pd.Series((t >= 10.4 - 1e-9) & (t <= 89.8 + 1e-9), index=f.index)[rows].astype(object)
    assert np.floor((898 / 10.0 - 104 / 10.0) * 10.0) == 793  # fixture precondition: the span rounds down
    cs = build_coordination_signals(f, windows=period_windows(f))
    (ps,) = [s for (_tm, p), s in cs.periods[0].players.items() if same_id(p, pid)]
    assert ps.runs.tolist() == [[104, 899]]
    assert np.isfinite(ps.x[104:899]).all() and np.isfinite(ps.y[104:899]).all()
    assert np.isfinite(ps.phasor_x[104:899]).all() and np.isfinite(ps.phasor_y[104:899]).all()


# --------------------------------------------------------------------------- include_goalkeeper (spec 7.5 / 7.14)
def test_include_goalkeeper_team_signals_puts_the_keeper_into_every_team_signal():
    """``include_goalkeeper["team_signals"]`` decides whether the keeper is one of the team's n players (spec 7.5:
    "outfield unless include_goalkeeper"). Default False keeps the outfield-only signals; True puts the keeper's
    resampled position into the collective kernel -- so the flag must MOVE the signal (non-vacuity)."""
    f, w = _valid_frames()
    base = CoordinationParams()
    on = dataclasses.replace(base, include_goalkeeper={**base.include_goalkeeper, "team_signals": True})
    p_off = build_coordination_signals(f, windows=w).periods[0]
    p_on = build_coordination_signals(f, windows=w, params=on).periods[0]
    a = p_on.team_ids[0]
    i = 400
    team_a = [ps for (tm, _pid), ps in p_on.players.items() if same_id(tm, a)]
    assert sum(ps.is_goalkeeper for ps in team_a) == 1  # fixture precondition: team A has one keeper series
    xs = [ps.x[i] for ps in team_a if np.isfinite(ps.x[i])]
    assert p_on.team_signal[(a, "centroid_x")][i] == pytest.approx(np.mean(xs))  # keeper included
    outfield = [ps.x[i] for ps in team_a if not ps.is_goalkeeper and np.isfinite(ps.x[i])]
    assert p_off.team_signal[(a, "centroid_x")][i] == pytest.approx(np.mean(outfield))  # default: outfield only
    assert abs(p_on.team_signal[(a, "centroid_x")][i] - p_off.team_signal[(a, "centroid_x")][i]) > 1.0


def test_include_goalkeeper_dyad_needs_the_keeper_series_even_without_the_cluster_flag():
    # The player series must hold the keeper when EITHER the cluster or the dyads use it; the cluster roster still
    # filters by its own flag.
    f, w = _valid_frames()
    base = CoordinationParams()
    params = dataclasses.replace(base, include_goalkeeper={"team_signals": False, "dyad": True, "cluster": False})
    players = build_coordination_signals(f, windows=w, params=params).periods[0].players
    assert sum(ps.is_goalkeeper for ps in players.values()) == 2  # one keeper per team


# --------------------------------------------------------------------------- team signals / observed fraction
def test_team_signals_equal_collective_kernel():
    f, w = _valid_frames()
    cs = build_coordination_signals(f, windows=w)
    p = cs.periods[0]
    a = p.team_ids[0]
    i = 400
    xs, ys = [], []
    for (tm, _pid), ps in p.players.items():
        if tm == a and not ps.is_goalkeeper and np.isfinite(ps.x[i]):
            xs.append(ps.x[i])
            ys.append(ps.y[i])
    pos = np.array([list(zip(xs, ys, strict=False))])
    compact, counts = compact_rows(pos, np.ones((1, len(xs)), dtype=bool))
    coll = collective_from_positions(compact, counts)
    assert coll["centroid_x"][0] == pytest.approx(p.team_signal[(a, "centroid_x")][i])
    assert coll["convex_hull_area"][0] == pytest.approx(p.team_signal[(a, "convex_hull_area")][i])


def test_observed_fraction():
    f, w = _valid_frames()
    cs = build_coordination_signals(f, windows=w)
    a = cs.periods[0].team_ids[0]
    assert np.nanmean(cs.periods[0].observed_fraction[a]) == pytest.approx(1.0)  # sportec fully observed

    sk = make_coordination_match(seconds=90.0, hz=10.0, provider="skillcorner", visibility_drop=0.3)
    cssk = build_coordination_signals(sk, windows=period_windows(sk))
    ask = cssk.periods[0].team_ids[0]
    n = 10  # outfield count
    assert abs(np.nanmean(cssk.periods[0].observed_fraction[ask]) - 0.7) < 2.0 / n


def _undetected_stretch(t0: float, t1: float, pid: int = 103):
    """A 120 s SkillCorner match where one outfield player stays on the pitch -- every row present, its best-estimate
    position extrapolated -- but is NOT detected (``visibility`` False) over ``(t0, t1)``."""
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="skillcorner")
    hidden = (
        (f["player_id"] == pid).to_numpy(dtype=bool, na_value=False)
        & (f["time_seconds"] > t0).to_numpy()
        & (f["time_seconds"] < t1).to_numpy()
    )
    f.loc[hidden, "visibility"] = False
    return f, int(hidden.sum())


def test_team_signals_use_best_estimate_positions_on_a_detection_aware_provider():
    # spec 7.11 (review A-02): team-level signals use every on-pitch player's best-estimate position -- dropping an
    # off-camera player biases every team signal toward the ball side. Only the PLAYER series use detected samples.
    f, n_hidden = _undetected_stretch(40.0, 50.0)
    assert n_hidden > 90  # precondition: ~10 s undetected, far beyond max_detection_gap_s
    cs = build_coordination_signals(f, windows=period_windows(f))
    ps = cs.periods[0]
    team = next(tm for tm in ps.team_ids if same_id(tm, 1))
    stretch = (ps.t > 41.0) & (ps.t < 49.0)
    assert np.isfinite(ps.team_signal[(team, "centroid_x")][stretch]).all()  # the team sample survives
    assert len(ps.segments[team]) == 1  # ... with no split at the undetected stretch
    assert ps.observed_fraction[team][stretch] == pytest.approx(0.9)  # 9 of 10 outfield players detected
    series = next(s for (_tm, p), s in ps.players.items() if same_id(p, 103))
    assert len(series.runs) == 2  # the player's OWN series splits there (observed samples only)


def test_team_signals_on_a_fully_observed_provider_are_unchanged():
    # the other side: with every row detected, the best-estimate team signal IS the detected one
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    cs = build_coordination_signals(f, windows=period_windows(f))
    ps = cs.periods[0]
    for tm in ps.team_ids:
        assert np.isfinite(ps.team_signal[(tm, "centroid_x")]).all()
        assert ps.observed_fraction[tm] == pytest.approx(1.0)


# --------------------------------------------------------------------------- orientation (C24)
def test_orientation_team_a_frame():
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", periods=2)
    cs = build_coordination_signals(f, windows=period_windows(f))
    p1, p2 = cs.periods
    a = p1.team_ids[0]
    assert p1.goal_x[a] == 0.0
    assert p1.reference_flip is False  # team A defends x=0 in period 1
    assert p2.goal_x[a] == 105.0
    assert p2.reference_flip is True  # team A defends x=105 in period 2 (frame reflected)
    # in team A's oriented frame the defended end is x=0, so its centroid stays on the pitch
    cx1 = p1.team_signal[(a, "centroid_x")]
    assert np.nanmin(cx1) >= -1.0 and np.nanmax(cx1) <= 106.0


# --------------------------------------------------------------------------- phasors once per segment (D15)
def test_phasors_computed_once_per_segment(monkeypatch):
    from tests._perf_structural import call_counter

    f, _ = _valid_frames()
    w_one = period_windows(f)
    w_many = period_windows(f, length_s=10.0, step_s=5.0)

    def count(windows):
        counter = call_counter(monkeypatch, _signals, "analytic_phase")
        build_coordination_signals(f, windows=windows)
        return counter["n"]

    assert count(w_one) == count(w_many)  # phasors are per (series, segment), not per window


def test_each_player_run_is_filtered_in_one_two_row_pass(monkeypatch):
    # Ruling C structural guard (ADR-111): a run's x AND y go through butterworth_lowpass_rows TOGETHER -- one call per
    # filtered run, two rows each. Filtering the coordinates one by one (the saving regressing) fails here.
    shapes: list[tuple[int, ...]] = []
    real = _signals.butterworth_lowpass_rows

    def spy(rows, *args, **kwargs):
        shapes.append(np.shape(rows))
        return real(rows, *args, **kwargs)

    monkeypatch.setattr(_signals, "butterworth_lowpass_rows", spy)
    f, w = _valid_frames()
    cs = build_coordination_signals(f, windows=w)
    n_runs = sum(len(ps.runs) for ps in cs.periods[0].players.values())
    assert n_runs > 0 and shapes  # non-vacuity
    assert all(shape[0] == 2 for shape in shapes)
    assert len(shapes) <= n_runs  # never more than one filter call per run (a shared run is filtered once)


def test_period_build_reads_only_the_pruned_columns(monkeypatch):
    # Ruling C structural guard (ADR-111): the per-period build moves only _PREP_COLUMNS (+ the detection mask), not
    # every frame column -- a wide extra column on the frames never reaches it.
    seen: list[set[str]] = []
    real = _signals._prepare_period

    def spy(gp, **kwargs):
        seen.append(set(gp.columns))
        return real(gp, **kwargs)

    monkeypatch.setattr(_signals, "_prepare_period", spy)
    f, w = _valid_frames()
    f = f.assign(wide_payload=np.zeros(len(f)))
    build_coordination_signals(f, windows=w)
    assert seen and all(cols == set(_signals._PREP_COLUMNS) | {"_detected"} for cols in seen)


def test_no_input_mutation():
    f, w = _valid_frames()
    before = f.copy(deep=True)
    build_coordination_signals(f, windows=w)
    pd.testing.assert_frame_equal(f, before)
