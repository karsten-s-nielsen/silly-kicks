"""TF-58 D2(a): the four pair-family surrogate nulls are batched over draws, byte-identical to the per-draw loop.

The oracle is the time-shift branch of ``_surrogate`` as first built -- per draw, copy the whole of series B, roll
every B segment the metric reads (plan Task 17's contributing segments: those holding a row of the metric's slices),
recompute the metric with the reference kernels -- kept here verbatim (the ADR-105 legacy-loop-oracle pattern). Every
`computed` (and `segment_too_short`) row of every family must match it bit for bit.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

import silly_kicks.coordination._compute as CC
from silly_kicks.coordination._catalog import METHODS_BY_LEVEL, resolve_pairs
from silly_kicks.coordination._columns import DEFAULT_LEVELS
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._kernels._spectral import pooled_coherence, welch_spectra
from silly_kicks.coordination._kernels._surrogates import draw_shifts, surrogate_rng, surrogate_triple
from silly_kicks.coordination._kernels._vector_coding import classify, coupling_angle_deg, stationary_mask
from silly_kicks.coordination._kernels._xcorr import fisher_pool, lagged_pearson, min_slice_samples, xcorr_summary
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows
from tests._perf_structural import call_counter
from tests.coordination._fixtures import make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore:vx/vy columns not found")

_NAN3 = (float("nan"), float("nan"), float("nan"))
_KEY_COLS = ("level", "signal_a", "signal_b", "team_a_id", "team_b_id", "player_a_id", "player_b_id")
_REFERENCE_NUMERICS_ENV = "SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS"


@pytest.fixture(scope="module")
def pair_signals():
    """600 s, one 40 s stoppage (two team segments), a substitution, and 240 s sliding windows next to the period
    window, so every family scores whole-segment and sub-segment windows. Welch segments of 120 s resolve the
    0.22-0.83 cpm coherence band (shorter ones leave no bin in it, so coherence would never score)."""
    f = make_coordination_match(
        seconds=600.0, hz=10.0, provider="sportec", dead_intervals=[(120.0, 160.0)], substitution=(7.0, 4), n_outfield=6
    )
    base = CoordinationParams()
    params = dataclasses.replace(
        base,
        n_surrogates=7,
        welch_segment_s=120.0,
        xcorr_max_lag_s=5.0,
        min_shift_s=dict.fromkeys(base.min_shift_s, 5.0),
    )
    windows = pd.concat([period_windows(f), period_windows(f, length_s=240.0, step_s=120.0)], ignore_index=True)
    return build_coordination_signals(f, windows=windows, params=params)


# --------------------------------------------------------------------------- the per-draw oracle
def _read_segments(ctx, read_slices):
    """The B segments holding a row of ``read_slices`` (the slices the metric reads), in segment order."""
    return [
        (int(lo), int(hi))
        for lo, hi in np.asarray(ctx.seg_b).reshape(-1, 2)
        if any(a < hi and lo < b for a, b in read_slices)
    ]


def _legacy_time_shift(ps, ctx, signal_b, params, fs, descriptor, family, metric_fns, obs_vals, read_slices):
    """``_surrogate``'s time-shift branch as first built: full-period copy + per-segment roll, per draw."""
    if all(np.isnan(v) for v in obs_vals.values()):
        return dict.fromkeys(metric_fns, _NAN3), "not_scored"
    segs = _read_segments(ctx, read_slices)
    tau = round(params.min_shift_s[signal_b if signal_b in params.min_shift_s else "cluster_amplitude"] * fs)
    shift_cols = []
    for lo, hi in segs:
        rng = surrogate_rng(params.surrogate_seed, (str(ps.game_id), int(ps.period_id), int(lo), *descriptor, family))
        sh = draw_shifts(rng, hi - lo, tau, params.n_surrogates)
        if sh is None:
            return dict.fromkeys(metric_fns, _NAN3), "segment_too_short"
        shift_cols.append(sh)
    dists: dict[str, list[float]] = {m: [] for m in metric_fns}
    for k in range(params.n_surrogates):
        phb_k = ctx.phb.copy()
        vb_k = ctx.vb.copy()
        for (lo, hi), sh in zip(segs, shift_cols, strict=True):
            phb_k[lo:hi] = np.roll(ctx.phb[lo:hi], int(sh[k]))
            vb_k[lo:hi] = np.roll(ctx.vb[lo:hi], int(sh[k]))
        for m, fn in metric_fns.items():
            dists[m].append(fn(phb_k, vb_k))
    return {m: surrogate_triple(obs_vals[m], np.array(dists[m])) for m in metric_fns}, "computed"


def _rp_metrics(ctx, params):
    idx, sa, sb, pha = ctx.idx, ctx.sa, ctx.sb, ctx.pha
    cos_thr = float(np.cos(np.radians(params.near_in_phase_deg)))

    def _r(phbk, _vbk):
        zz = (sa * sb) * pha[idx] * np.conj(phbk[idx])
        return float(abs(zz.sum()) / idx.size)

    def _near(phbk, _vbk):
        zz = (sa * sb) * pha[idx] * np.conj(phbk[idx])
        return float(np.mean(np.real(zz) >= cos_thr))

    return {"coord_rp_resultant_length": _r, "coord_rp_pct_near_in_phase": _near}


def _xc_metrics(ctx, params, fs):
    lag = round(params.xcorr_max_lag_s * fs)
    slices = [(lo, hi) for lo, hi in ctx.runs if hi - lo >= min_slice_samples(lag)]

    def _xc(_phbk, vbk):
        rr, nn = [], []
        for lo, hi in slices:
            r2, n2 = lagged_pearson(ctx.va[lo:hi], vbk[lo:hi], lag)
            rr.append((ctx.sa * ctx.sb) * r2)
            nn.append(n2)
        pooled2 = fisher_pool(np.array(rr), np.array(nn))
        return float("nan") if np.isnan(pooled2).all() else xcorr_summary(pooled2, fs)[0]

    return {"coord_xc_max_abs_r": _xc}


def _vc_metrics(ctx, spec, params):
    eps_a = params.vc_epsilon[spec.signal_a]
    eps_b = params.vc_epsilon[spec.signal_b]

    def _pct(vbk, cls_idx):
        das, dbs = [], []
        for lo, hi in ctx.runs:
            if hi - lo >= 2:
                das.append(ctx.sa * np.diff(ctx.va[lo:hi]))
                dbs.append(ctx.sb * np.diff(vbk[lo:hi]))
        if not das:
            return float("nan")
        da = np.concatenate(das)
        db = np.concatenate(dbs)
        keep = ~stationary_mask(da, db, eps_a, eps_b)
        if int(keep.sum()) < 3:
            return float("nan")
        return float(np.mean(classify(coupling_angle_deg(da[keep], db[keep])) == cls_idx))

    return {
        "coord_vc_pct_in_phase": lambda _phbk, vbk: _pct(vbk, 0),
        "coord_vc_pct_anti_phase": lambda _phbk, vbk: _pct(vbk, 1),
    }


def _coh_metrics(ctx, params, fs):
    nperseg = round(params.welch_segment_s * fs)
    slices = [(lo, hi) for lo, hi in ctx.runs if hi - lo >= nperseg]

    def _coh(_phbk, vbk):
        spec2 = [welch_spectra(ctx.va[lo:hi], vbk[lo:hi], fs, nperseg) for lo, hi in slices]
        return pooled_coherence(spec2, params.band_low_cpm, params.band_high_cpm)[0]

    return {"coord_coh_band_mean": _coh}


_FAMILIES = {
    "relative_phase": ("coord_rp_surrogate_source", lambda sig: CC.compute_relative_phase(sig)[0]),
    "cross_correlation": ("coord_xc_surrogate_source", lambda sig: CC.compute_cross_correlation(sig)[0]),
    "vector_coding": ("coord_vc_surrogate_source", lambda sig: CC.compute_vector_coding(sig)[0]),
    "coherence": ("coord_coh_surrogate_source", lambda sig: CC.compute_coherence(sig)[0]),
}


def _metrics_for(family, ctx, spec, params, fs):
    """The family's metric functions and the slices they read (restated here, independently of production)."""
    if family == "relative_phase":
        return _rp_metrics(ctx, params), list(ctx.runs)
    if family == "cross_correlation":
        lag = round(params.xcorr_max_lag_s * fs)
        return _xc_metrics(ctx, params, fs), [(lo, hi) for lo, hi in ctx.runs if hi - lo >= min_slice_samples(lag)]
    if family == "vector_coding":
        return _vc_metrics(ctx, spec, params), [(lo, hi) for lo, hi in ctx.runs if hi - lo >= 2]
    nperseg = round(params.welch_segment_s * fs)
    return _coh_metrics(ctx, params, fs), [(lo, hi) for lo, hi in ctx.runs if hi - lo >= nperseg]


def _bits_equal(have: float, want: float) -> bool:
    return (np.isnan(have) and np.isnan(want)) or np.float64(have).view(np.int64) == np.float64(want).view(np.int64)


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_pair_surrogate_equals_the_per_draw_loop(pair_signals, family, monkeypatch):
    # The DIRECT (reference) nulls stay the batched per-draw loop bit for bit; relative phase and cross-correlation
    # score through the spec 7.9 identities by default, which the D2(b) tests below hold against these direct nulls.
    monkeypatch.setenv(_REFERENCE_NUMERICS_ENV, "1")
    sig = pair_signals
    params, fs = sig.params, sig.fs
    source_col, compute = _FAMILIES[family]
    table = compute(sig)
    specs = [p for p in resolve_pairs(None, DEFAULT_LEVELS) if family in METHODS_BY_LEVEL[p.level]]
    checked: dict[str, int] = {}
    for ps, wrow, s, e in CC._iter_windows(sig):
        in_window = (table.window_kind == wrow["window_kind"]) & (table.window_id == wrow["window_id"])
        for spec in specs:
            if spec.level == "dyad" and wrow["window_kind"] != "period":
                continue
            binds = CC._bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                continue
            for b in binds:
                key = (spec.level, spec.signal_a, spec.signal_b, CC._cid(b.team_a_id), CC._cid(b.team_b_id))
                key += (CC._cid(b.player_a_id), CC._cid(b.player_b_id))
                sel = in_window.copy()
                for col, v in zip(_KEY_COLS, key, strict=True):
                    sel &= (table[col].isna()) if v is pd.NA else (table[col] == v)  # _cid maps missing to pd.NA
                rows = table[sel]
                assert len(rows) == 1, (family, key)
                row = rows.iloc[0]
                if row[source_col] not in ("computed", "segment_too_short"):
                    continue
                ctx = CC._context(ps, spec, b, s, e)
                metric_fns, read = _metrics_for(family, ctx, spec, params, fs)
                obs = {m: float(row[m]) for m in metric_fns}
                triples, src = _legacy_time_shift(
                    ps, ctx, spec.signal_b, params, fs, CC._descriptor(spec, b), family, metric_fns, obs, read
                )
                assert row[source_col] == src, (family, key)
                for m, (sm, pc, ex) in triples.items():
                    assert _bits_equal(float(row[f"{m}_surrogate_mean"]), sm), (family, key, m)
                    assert _bits_equal(float(row[f"{m}_percentile"]), pc), (family, key, m)
                    assert _bits_equal(float(row[f"{m}_excess"]), ex), (family, key, m)
                checked[src] = checked.get(src, 0) + 1
    assert checked.get("computed", 0) > 0, checked  # non-vacuity: real nulls were compared


@pytest.mark.parametrize("reference_nulls", [False, True])
def test_pair_surrogates_never_roll_a_series_per_draw(pair_signals, monkeypatch, reference_nulls):
    # R8 structural guard: the batched nulls gather only the rows each metric reads, for all draws at once -- no
    # per-draw full-period copy + np.roll of series B (the first build rolled every overlapping segment per draw).
    if reference_nulls:
        monkeypatch.setenv(_REFERENCE_NUMERICS_ENV, "1")
    rolls = call_counter(monkeypatch, np, "roll")
    for _source_col, compute in _FAMILIES.values():
        compute(pair_signals)
    assert rolls["n"] == 0


# --------------------------------------------------------------------------- D2(b): the spec 7.9 identities
# Relative phase (R: one circular cross-correlation per B segment; % near-in-phase: direct counts) and cross-correlation
# (circular FFT cross-correlation + exact edge correction + prefix sums) score every time-shift draw through the spec's
# algebraic identities. They are held against the DIRECT nulls above (bit-identical to the per-draw loop): no source
# token and no percentile may move, and the measured max deviation of every other surrogate column is reported.
_IDENTITY_FAMILIES = {
    "relative_phase": ("coord_rp_resultant_length", "coord_rp_pct_near_in_phase"),
    "cross_correlation": ("coord_xc_max_abs_r",),
}
_IDENTITY_FN = {"relative_phase": "_rp_identity", "cross_correlation": "_xc_identity"}


def _spy_identity(monkeypatch, name):
    """Count the identity calls that actually scored (a ``None`` return means it declined, the direct null ran)."""
    real = getattr(CC, name)
    counter = {"scored": 0, "declined": 0}

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        counter["declined" if out is None else "scored"] += 1
        return out

    monkeypatch.setattr(CC, name, spy)
    return counter


@pytest.mark.parametrize("family", list(_IDENTITY_FAMILIES))
def test_pair_identity_null_matches_the_direct_null(pair_signals, family, monkeypatch):
    source_col, compute = _FAMILIES[family]
    monkeypatch.setenv(_REFERENCE_NUMERICS_ENV, "1")
    direct = compute(pair_signals)
    monkeypatch.delenv(_REFERENCE_NUMERICS_ENV)
    used = _spy_identity(monkeypatch, _IDENTITY_FN[family])
    ident = compute(pair_signals)
    assert used["scored"] > 0, used  # non-vacuous: the identity scored real nulls
    pd.testing.assert_series_equal(ident[source_col], direct[source_col])  # no source token moves
    computed = (direct[source_col] == "computed").to_numpy()
    assert computed.sum() > 0
    for m in _IDENTITY_FAMILIES[family]:
        np.testing.assert_array_equal(  # no percentile moves
            ident.loc[computed, f"{m}_percentile"].to_numpy(float),
            direct.loc[computed, f"{m}_percentile"].to_numpy(float),
        )
        for suffix in ("_surrogate_mean", "_excess"):
            have = ident.loc[computed, m + suffix].to_numpy(float)
            want = direct.loc[computed, m + suffix].to_numpy(float)
            np.testing.assert_array_equal(np.isnan(have), np.isnan(want))
            dev = float(np.nanmax(np.abs(have - want)))
            print(f"{family} {m}{suffix}: max |identity - direct| = {dev:.3e} over {int(computed.sum())} rows")
            assert dev <= 1e-9
    triples = ("_surrogate_mean", "_percentile", "_excess")
    others = [c for c in direct.columns if not c.endswith(triples)]
    pd.testing.assert_frame_equal(ident[others], direct[others])  # the observed metrics are untouched


def _identity_cases(sig, family):
    """Every pair-window with time-shift draws, as ``(ctx, draws)`` -- the inputs ``_surrogate`` hands the null."""
    params = sig.params
    specs = [p for p in resolve_pairs(None, DEFAULT_LEVELS) if family in METHODS_BY_LEVEL[p.level]]
    for ps, wrow, s, e in CC._iter_windows(sig):
        for spec in specs:
            if spec.level == "dyad" and wrow["window_kind"] != "period":
                continue
            binds = CC._bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                continue
            for b in binds:
                ctx = CC._context(ps, spec, b, s, e)
                if len(ctx.idx) < 3:
                    continue
                read = list(ctx.runs) if family == "relative_phase" else _xc_slices(sig, ctx)[1]
                segs = _read_segments(ctx, read)  # the B segments the family's null draws from
                if not segs:
                    continue
                tau = round(params.min_shift_s[spec.signal_b] * sig.fs)
                key = (*CC._descriptor(spec, b), family)
                shifts = [
                    draw_shifts(
                        surrogate_rng(params.surrogate_seed, (str(ps.game_id), int(ps.period_id), int(lo), *key)),
                        hi - lo,
                        tau,
                        params.n_surrogates,
                    )
                    for lo, hi in segs
                ]
                drawn = [sh for sh in shifts if sh is not None]
                if len(drawn) != len(shifts):
                    continue  # a segment too short to shift: the null is `segment_too_short`, not scored
                yield ctx, CC._ShiftedB(ctx, segs, drawn, params.n_surrogates)


def _segment_holding(segs, row):
    return next((lo, hi) for lo, hi in segs if lo <= row < hi)


def _xc_slices(sig, ctx):
    lag = round(sig.params.xcorr_max_lag_s * sig.fs)
    return lag, [(lo, hi) for lo, hi in ctx.runs if hi - lo >= min_slice_samples(lag)]


def test_rp_identity_declines_a_non_finite_b_segment(pair_signals):
    # One NaN phasor in a touched B segment would poison EVERY shift under the FFT, while the direct null NaNs only the
    # draws that move it onto an idx row -- so the identity declines and the direct null keeps its NaN semantics.
    cos_thr = float(np.cos(np.radians(pair_signals.params.near_in_phase_deg)))
    for ctx, draws in _identity_cases(pair_signals, "relative_phase"):
        a = (ctx.sa * ctx.sb) * ctx.pha[ctx.idx]
        if CC._rp_identity(ctx, a, draws, cos_thr) is None:
            continue
        lo, hi = _segment_holding(draws.segs, int(ctx.idx[0]))  # a segment the null actually reads
        phb = ctx.phb.copy()
        phb[(lo + hi) // 2] = complex(np.nan, np.nan)
        tainted = dataclasses.replace(ctx, phb=phb)
        assert CC._rp_identity(tainted, a, dataclasses.replace(draws, ctx=tainted), cos_thr) is None
        return
    raise AssertionError("the identity scored no relative-phase null on the fixture")


def test_xc_identity_declines_a_non_finite_b_segment(pair_signals):
    for ctx, draws in _identity_cases(pair_signals, "cross_correlation"):
        lag, slices = _xc_slices(pair_signals, ctx)
        if not slices or CC._xc_identity(ctx, slices, lag, draws) is None:
            continue
        lo, hi = _segment_holding(draws.segs, slices[0][0])  # the B segment the first slice is rolled within
        vb = ctx.vb.copy()
        vb[(lo + hi) // 2] = np.nan
        tainted = dataclasses.replace(ctx, vb=vb)
        assert CC._xc_identity(tainted, slices, lag, dataclasses.replace(draws, ctx=tainted)) is None
        return
    raise AssertionError("the identity scored no cross-correlation null on the fixture")


def _split_b_segment_under_a_slice(pair_signals):
    """A scored cross-correlation case whose first slice's B segment is cut in two TOUCHING segments mid-slice -- what
    an on-pitch-count change (``_segments_with_count``: a red card) produces, as on GS 3828 -- with a valid draw set."""
    for ctx, draws in _identity_cases(pair_signals, "cross_correlation"):
        lag, slices = _xc_slices(pair_signals, ctx)
        if not slices or CC._xc_identity(ctx, slices, lag, draws) is None:
            continue
        lo, hi = slices[0]
        g = draws.segs.index(_segment_holding(draws.segs, lo))
        glo, ghi = draws.segs[g]
        cut = (lo + hi) // 2
        rng = np.random.default_rng(58)
        parts = [(glo, cut), (cut, ghi)]
        segs = [*draws.segs[:g], *parts, *draws.segs[g + 1 :]]
        part_shifts = [rng.integers(0, b - a, draws.n_draws) for a, b in parts]
        shifts = [*draws.shifts[:g], *part_shifts, *draws.shifts[g + 1 :]]
        split = dataclasses.replace(ctx, seg_b=np.array(segs, dtype=np.int64))
        return split, slices, lag, CC._ShiftedB(split, segs, shifts, draws.n_draws)
    raise AssertionError("the identity scored no cross-correlation null on the fixture")


def test_xc_identity_refuses_a_slice_spanning_touching_b_segments(pair_signals):
    # `_both_runs` cuts slices at every segment boundary, touching ones included (spec 7.4 step 2 + D15), so a slice
    # spanning two B segments can no longer reach the identity: if one does, the invariant is broken and it says so.
    ctx, slices, lag, draws = _split_b_segment_under_a_slice(pair_signals)
    lo, hi = slices[0]
    assert not any(glo <= lo and hi <= ghi for glo, ghi in draws.segs)  # non-vacuity: no single B segment holds it
    with pytest.raises(RuntimeError, match="a bug"):
        CC._xc_identity(ctx, slices, lag, draws)


def test_xc_identity_refuses_a_slice_outside_every_b_segment(pair_signals):
    # A slice reaching outside B's segments breaks the same invariant (a slice is a run of rows inside BOTH sides').
    ctx, _slices, lag, draws = _split_b_segment_under_a_slice(pair_signals)
    uncovered = [(lo, hi) for lo, hi in draws.segs if hi - lo > 2]
    lo, hi = uncovered[0]
    gapped = [s for s in draws.segs if s != (lo, hi)] + [(lo, hi - 1)]
    shifts = [sh for s, sh in zip(draws.segs, draws.shifts, strict=True) if s != (lo, hi)]
    shifts.append(np.zeros(draws.n_draws, dtype=np.int64))
    broken = CC._ShiftedB(ctx, gapped, shifts, draws.n_draws)
    with pytest.raises(RuntimeError, match="a bug"):
        CC._xc_identity(ctx, [(lo, hi)], lag, broken)


# --------------------------------------------------------------------------- Decision A: relative-phase dispatch
# Owner ruling A (2026-09-28): relative phase runs the identity only where it is cheaper than the direct null, by a
# FIXED size rule over n * ceil(log2 n) FFT units vs K * |idx| direct products. It reads sizes only (never values),
# and both branches are parity-equal (test_pair_identity_null_matches_the_direct_null), so it chooses speed only.
def test_rp_identity_crossover_is_the_ruled_value():
    # Ruling A fixes the crossover at 17/4 (a ruled value, not a tuning knob): the boundary tests below only bracket it
    # (about 3.96 <= c < 4.29), so a drift to e.g. 4 would survive them -- pin the exact rational.
    from fractions import Fraction

    assert Fraction(17, 4) == CC.RP_IDENTITY_CROSSOVER
    assert isinstance(CC.RP_IDENTITY_CROSSOVER, Fraction)  # exact rational arithmetic at the boundary, never a float


def test_rp_dispatch_is_a_fixed_size_rule_from_both_sides():
    # one touched segment of n = 1024 rows: 1024 * 10 = 10240 FFT units against 17/4 * K * |idx| with K = 199:
    # 17/4 * 199 * 12 = 10149.0 < 10240 (direct), 17/4 * 199 * 13 = 10994.75 >= 10240 (identity).
    segs = [(0, 1024)]
    assert CC._rp_identity_is_cheaper(np.arange(12), segs, 199) is False
    assert CC._rp_identity_is_cheaper(np.arange(13), segs, 199) is True
    # a segment the window's rows never touch costs the identity nothing (it is skipped, as in _rp_identity)
    assert CC._rp_identity_is_cheaper(np.arange(13), [*segs, (5000, 9000)], 199) is True
    # integer FFT units n * ceil(log2 n), compared exactly (no platform-dependent float log2 at the boundary):
    # n = 1025 -> 1025 * 11 = 11275 > 10994.75 (direct), although 1025 * log2(1025) = 10251.4 would pass
    assert CC._rp_identity_is_cheaper(np.arange(13), [(0, 1025)], 199) is False


def test_rp_dispatch_takes_both_branches_on_the_fixture(pair_signals, monkeypatch):
    # non-vacuity for the parity test above: on the fixture the default path scores some relative-phase nulls through
    # the identity and routes others to the direct null by the size rule.
    real = CC._rp_identity_is_cheaper
    seen = {True: 0, False: 0}

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen[out] += 1
        return out

    monkeypatch.setattr(CC, "_rp_identity_is_cheaper", spy)
    CC.compute_relative_phase(pair_signals)
    assert seen[True] > 0 and seen[False] > 0, seen


@pytest.mark.parametrize(
    ("family", "prep_name", "kernel_name"),
    [
        ("relative_phase", "phasor_spectrum", "shifted_phasor_sums"),
        ("cross_correlation", "lagged_pearson_b_side", "shifted_slice_lagged_pearson"),
    ],
)
def test_identity_b_side_is_prepared_once_per_segment(pair_signals, monkeypatch, family, prep_name, kernel_name):
    # R8 structural guard (ADR-111 ruling C): every window rolled within one B segment shares that segment's B side
    # (R: conj(fft(B)); cross-correlation: centred B, its rfft, its prefix sums) -- prepared once per (B series,
    # segment) per compute call, never once per window. The dispatch is pinned to the identity for the count.
    monkeypatch.setattr(CC, "RP_IDENTITY_CROSSOVER", math.inf)
    preps = call_counter(monkeypatch, CC, prep_name)
    uses = call_counter(monkeypatch, CC, kernel_name)
    _FAMILIES[family][1](pair_signals)
    assert 0 < preps["n"] < uses["n"], (preps, uses)


def test_pair_identity_ffts_do_not_scale_with_the_draw_count(pair_signals, monkeypatch):
    # Spec C11 structural gate: the identities take a FIXED number of FFTs per scored segment/slice, whatever K. The
    # relative-phase dispatch is pinned to the identity here, so the count isolates the identity's own FFT work.
    from silly_kicks.coordination._kernels import _surrogates

    monkeypatch.setattr(CC, "RP_IDENTITY_CROSSOVER", math.inf)
    counts = {}
    for k in (7, 29):
        sig = dataclasses.replace(pair_signals, params=dataclasses.replace(pair_signals.params, n_surrogates=k))
        counters = [call_counter(monkeypatch, _surrogates._fft, name) for name in ("fft", "ifft", "rfft", "irfft")]
        CC.compute_relative_phase(sig)
        CC.compute_cross_correlation(sig)
        counts[k] = sum(c["n"] for c in counters)
    assert counts[7] == counts[29] > 0, counts
