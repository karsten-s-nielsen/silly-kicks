"""Plan Task 17's surrogate-source rule: a time-shift null is ``segment_too_short`` when a CONTRIBUTING segment of the
shifted series -- one holding a row the row's statistic reads -- is too short to shift (``shift_bounds`` is None).

A segment that only overlaps the window's span, holding no row the statistic reads, takes no part: the null never reads
its rows, so it must not refuse the row either. Found 2026-10-03 (the pair families and team sync gated on every B
segment overlapping the span); the cluster family follows the same "contributing" rule (review B m3, owner-ratified
2026-10-04): it checks only a player's runs that hold a usable rho_group sample, not any run touching the window.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

import silly_kicks.coordination._compute as CC
from silly_kicks.coordination import CoordinationParams, build_coordination_signals
from silly_kicks.coordination._catalog import PairSpec
from silly_kicks.coordination._kernels._surrogates import shift_bounds
from silly_kicks.coordination._windows import period_windows
from silly_kicks.id_compat import same_id
from tests.coordination._fixtures import make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning")

_HZ = 10.0
_A, _B = 103, 203  # a cross-team dyad: A = team 1 (ps.team_ids[0]), B = team 2 -- B is the shifted series
_DYAD_X = [PairSpec("dyad", "player_x", "player_x", "canonical")]
_ISLAND = (330.0, 340.0)  # B's 10 s island between two detection gaps: far shorter than 2 tau + 1 (tau = 60 s)


# --------------------------------------------------------------------------- the helper, both sides
def test_a_segment_holding_no_scored_row_takes_no_part():
    segs = np.array([[0, 300], [330, 340], [400, 900]])
    rows = np.concatenate([np.arange(0, 300), np.arange(400, 900)])
    assert CC._segments_holding(segs, rows) == [(0, 300), (400, 900)]


def test_a_segment_holding_one_scored_row_contributes():
    segs = np.array([[0, 300], [330, 340], [400, 900]])
    rows = np.concatenate([np.arange(0, 300), [335], np.arange(400, 900)])
    assert CC._segments_holding(segs, rows) == [(0, 300), (330, 340), (400, 900)]


def test_a_scored_row_outside_every_segment_is_a_bug():
    with pytest.raises(RuntimeError, match="a bug"):
        CC._segments_holding(np.array([[0, 300], [400, 900]]), np.array([10, 350]))


# --------------------------------------------------------------------------- the pair families, end to end
def _hide(f: pd.DataFrame, pid: int, t0: float, t1: float) -> int:
    m = (
        (f["player_id"] == pid).to_numpy(dtype=bool, na_value=False)
        & (f["time_seconds"] > t0).to_numpy()
        & (f["time_seconds"] < t1).to_numpy()
    )
    f.loc[m, "visibility"] = False
    return int(m.sum())


def _dyad_signals(*, shared_island: bool):
    """900 s on a detection-aware provider. B is detected on a 10 s island between two detection gaps; A is either
    undetected over the island's whole neighbourhood (the island holds no slice of the dyad) or detected on the island
    too (the island is a 10 s slice of the dyad)."""
    # pin the interim-regime thresholds these fixtures were built for: mof 0.5 (the hidden dyad must still score --
    # the commit-2 derivation default 1.0 would mark the partially-observed window insufficient) and min_shift 60 s
    # (the 10 s island is the only run too short to shift; derivation's ~200 s per-signal min_shift would make every
    # run too short, erasing the contrast this test turns on).
    base = CoordinationParams.for_provider("skillcorner")
    params = dataclasses.replace(
        base,
        n_surrogates=9,
        min_observed_fraction={k: 0.5 for k in base.min_observed_fraction},
        min_shift_s={k: 60.0 for k in base.min_shift_s},
    )
    f = make_coordination_match(seconds=900.0, hz=_HZ, provider="skillcorner")
    assert _hide(f, _B, 300.0, _ISLAND[0]) and _hide(f, _B, _ISLAND[1], 400.0)
    if shared_island:
        assert _hide(f, _A, 300.0, _ISLAND[0]) and _hide(f, _A, _ISLAND[1], 400.0)
    else:
        assert _hide(f, _A, 300.0, 400.0)
    return build_coordination_signals(f, windows=period_windows(f), params=params)


def _runs(sig, pid) -> list[tuple[int, int]]:
    series = next(s for (_tm, p), s in sig.periods[0].players.items() if same_id(p, pid))
    return [(int(lo), int(hi)) for lo, hi in series.runs]


def _island_run(sig) -> tuple[int, int]:
    lo_i, hi_i = round(_ISLAND[0] * sig.fs), round(_ISLAND[1] * sig.fs)
    held = [(lo, hi) for lo, hi in _runs(sig, _B) if lo <= lo_i + 1 and hi_i - 1 <= hi]
    assert len(held) == 1, _runs(sig, _B)
    return held[0]


def _period_row(table: pd.DataFrame) -> pd.Series:
    sel = table[
        (table["level"] == "dyad")
        & (table["window_kind"] == "period")
        & (table["signal_a"] == "player_x")
        & table["player_a_id"].map(lambda v: same_id(v, _A))
        & table["player_b_id"].map(lambda v: same_id(v, _B))
    ]
    assert len(sel) == 1
    return sel.iloc[0]


@pytest.mark.parametrize("shared_island", [False, True])
def test_fixture_preconditions(shared_island):
    # ADR-032: B's island is its own run, too short to shift; A shares it only in the shared case
    sig = _dyad_signals(shared_island=shared_island)
    lo, hi = _island_run(sig)
    tau = round(sig.params.min_shift_s["player_x"] * sig.fs)
    assert shift_bounds(hi - lo, tau) is None
    a_on_island = any(alo < hi and lo < ahi for alo, ahi in _runs(sig, _A))
    assert a_on_island is shared_island
    # every other B run is long enough to shift, so the island alone decides
    assert all(shift_bounds(r1 - r0, tau) is not None for r0, r1 in _runs(sig, _B) if (r0, r1) != (lo, hi))


def test_a_b_segment_holding_no_slice_does_not_refuse_the_null():
    sig = _dyad_signals(shared_island=False)
    rp, _phase, _report = CC.compute_relative_phase(sig, levels=("dyad",), pairs=_DYAD_X)
    row = _period_row(rp)
    assert row["coord_rp_source"] == "scored"
    assert row["coord_rp_surrogate_source"] == "computed"
    assert np.isfinite(row["coord_rp_resultant_length_percentile"])
    xc, _report = CC.compute_cross_correlation(sig, levels=("dyad",), pairs=_DYAD_X)
    row = _period_row(xc)
    assert row["coord_xc_source"] == "scored"
    assert row["coord_xc_surrogate_source"] == "computed"


def test_a_contributing_short_b_segment_still_refuses_the_null():
    # the other side of the band: the shared island is a relative-phase slice, so its 10 s B segment must be shifted
    sig = _dyad_signals(shared_island=True)
    rp, _phase, _report = CC.compute_relative_phase(sig, levels=("dyad",), pairs=_DYAD_X)
    row = _period_row(rp)
    assert row["coord_rp_source"] == "scored"
    assert row["coord_rp_surrogate_source"] == "segment_too_short"


def test_cross_correlation_draws_only_from_the_slices_it_scores():
    # the shared island is a relative-phase slice but shorter than min_slice_samples(lag) (60 s), so cross-correlation
    # never reads it: its B segment takes no part in the cross-correlation null
    sig = _dyad_signals(shared_island=True)
    xc, _report = CC.compute_cross_correlation(sig, levels=("dyad",), pairs=_DYAD_X)
    row = _period_row(xc)
    assert row["coord_xc_source"] == "scored"
    assert row["coord_xc_surrogate_source"] == "computed"


# --------------------------------------------------------------------------- team sync
def _team_sync_case(*, b_reads_short_segment: bool):
    """Team B's amplitude has a 10 s segment between two long ones; the rows finite on BOTH sides (``both``, the rows
    the Pearson r reads) skip it, or read one of its rows."""
    params = dataclasses.replace(CoordinationParams(), n_surrogates=9)
    fs, s, e, tb = 10.0, 0, 5000, 2
    rng = np.random.default_rng(58)
    segments = np.array([[0, 2000], [2100, 2200], [2300, 5000]])
    amp_b = np.full(e, np.nan)
    for lo, hi in segments:
        amp_b[lo:hi] = rng.uniform(0.2, 1.0, hi - lo)
    amp_a = rng.uniform(0.2, 1.0, e)
    amp_a[2000:2300] = np.nan  # team A has no usable amplitude around B's short segment
    if b_reads_short_segment:
        amp_a[2150] = 0.5
    both = np.isfinite(amp_a) & np.isfinite(amp_b[s:e])
    kern = CC._cluster_kernels()
    obs_r = float(kern.pearson_rows(amp_a, amp_b[None, s:e], both, 4)[0])
    ps = SimpleNamespace(segments={tb: segments}, game_id=1, period_id=1)
    cb = cast("CC._ClusterCtx", SimpleNamespace(amp=amp_b))  # the surrogate reads only team B's amplitude
    return CC._team_sync_surrogate(ps, tb, "x", cb, amp_a, s, e, both, obs_r, params, fs, kern)


def test_team_sync_ignores_a_b_segment_it_never_reads():
    (_mean, percentile, _excess), src = _team_sync_case(b_reads_short_segment=False)
    assert src == "computed"
    assert np.isfinite(percentile)


def test_team_sync_refuses_when_it_reads_a_short_b_segment():
    _triple, src = _team_sync_case(b_reads_short_segment=True)
    assert src == "segment_too_short"


# --------------------------------------------------------------------------- B m3: cluster null -> contributing runs
def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask as [lo, hi)."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)]


def _cluster_ctx_for(valid_cols: list[np.ndarray], min_players: int) -> CC._ClusterCtx:
    """A minimal `_ClusterCtx`: `valid_cols[j]` is player j's validity over the grid; unit phasors where valid, NaN
    elsewhere; `usable` = (>= min_players valid per row); `runs[j]` = player j's contiguous valid runs."""
    from silly_kicks.coordination._kernels._cluster import phasor_parts

    grid_n, k = len(valid_cols[0]), len(valid_cols)
    z = np.full((grid_n, k), np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.zeros((grid_n, k), dtype=bool)
    for j, col in enumerate(valid_cols):
        col = np.asarray(col, dtype=bool)
        valid[:, j] = col
        z[col, j] = np.exp(1j * np.linspace(0.0, 2.0, int(col.sum())))  # non-degenerate phasors
    usable = valid.sum(axis=1) >= min_players
    runs = [_true_runs(valid[:, j]) for j in range(k)]
    return CC._ClusterCtx(
        z=z,
        parts=phasor_parts(z),
        valid=valid,
        usable=usable,
        rel=np.zeros_like(z),
        pids=list(range(k)),
        runs=runs,
        on_pitch=valid,
        amp=np.zeros(grid_n),
        detections=[],
    )


def _cluster_src(c, *, fs=10.0):
    # tau = 0.5 s * 10 Hz = 5 -> 2*tau+1 = 11: the 40/55-sample runs shift, the 5-sample run does not
    params = dataclasses.replace(
        CoordinationParams(),
        n_surrogates=19,
        min_players=2,
        min_shift_s={**CoordinationParams().min_shift_s, "player_x": 0.5, "player_y": 0.5},
    )
    ps = SimpleNamespace(game_id=1, period_id=1)
    (_m, pct, _ex), src = CC._cluster_surrogate(ps, 1, "x", c, 0, c.z.shape[0], params, fs, 0.5, CC._cluster_kernels())
    return pct, src


def test_cluster_null_skips_a_touching_but_non_contributing_short_run():
    # B m3 (owner-ratified): a player run that TOUCHES the window but holds no usable rho_group sample (the team is
    # below min_players there) takes no part -> it must NOT force segment_too_short. players 0,1 valid except [40,45);
    # player 2 valid ONLY [40,45) (5 samples, < 2tau+1), where only it is valid -> usable is False there.
    grid_n = 100
    p01 = np.ones(grid_n, dtype=bool)
    p01[40:45] = False
    p2 = np.zeros(grid_n, dtype=bool)
    p2[40:45] = True
    c = _cluster_ctx_for([p01, p01, p2], min_players=2)
    assert not c.usable[40:45].any() and c.usable[:40].all() and c.usable[45:].all()  # fixture precondition
    pct, src = _cluster_src(c)
    assert src == "computed"  # the non-contributing short run is skipped (mutant "any touching run" turns this RED)
    assert np.isfinite(pct)


def test_cluster_null_still_refuses_a_contributing_short_run():
    # the other side: the short run DOES hold usable samples (players 0,1 valid there too) -> it contributes and must
    # be shifted; shorter than 2tau+1 -> segment_too_short (the real too-short case).
    grid_n = 100
    full = np.ones(grid_n, dtype=bool)
    p2 = np.zeros(grid_n, dtype=bool)
    p2[40:45] = True
    c = _cluster_ctx_for([full, full, p2], min_players=2)
    assert c.usable[40:45].all()  # player 2's short run is contributing (usable there)
    _pct, src = _cluster_src(c)
    assert src == "segment_too_short"


# --------------------------------------------------------------------------- A-43: IAAFT path for VC, COH (and RP, XC)
_IAAFT_FAMILIES = [
    (CC.compute_relative_phase, "coord_rp_surrogate_source"),
    (CC.compute_cross_correlation, "coord_xc_surrogate_source"),
    (CC.compute_vector_coding, "coord_vc_surrogate_source"),
    (CC.compute_coherence, "coord_coh_surrogate_source"),
]


@pytest.mark.parametrize(("family_fn", "src_col"), _IAAFT_FAMILIES, ids=[c.split("_")[1] for _f, c in _IAAFT_FAMILIES])
@pytest.mark.parametrize("converged", [True, False], ids=["converged", "nonconverged"])
def test_pair_family_iaaft_null_marks_nonconvergence(family_fn, src_col, converged, monkeypatch):
    # A-43: the IAAFT opt-in and its `computed_nonconverged` token were untested for vector coding and coherence (and
    # the token was untested anywhere). All four pair families share `_surrogate`, so parametrize: with IAAFT selected
    # and the kernel forced to (not) converge, each scored row's surrogate source is computed / computed_nonconverged.
    # welch_segment_s=120 resolves the 0.22-0.83 cpm band so coherence actually SCORES (test_coherence_near_one_*);
    # the in-band oscillation + phase offset also feed RP/XC/VC. One config so all four share the IAAFT path.
    params = dataclasses.replace(
        CoordinationParams.for_provider("sportec"), n_surrogates=5, surrogate_method="iaaft", welch_segment_s=120.0
    )
    f = make_coordination_match(seconds=600.0, hz=_HZ, provider="sportec", phase_offset_deg=40.0)
    sig = build_coordination_signals(f, windows=period_windows(f), params=params)
    # force the convergence flag; the surrogate values are irrelevant to the source token
    monkeypatch.setattr(CC, "iaaft", lambda x, rng, _it: (np.asarray(x, dtype=np.float64).copy(), converged))
    table = family_fn(sig)[0]
    scored = table[table[src_col].isin(["computed", "computed_nonconverged"])]
    assert len(scored)  # non-vacuity: the family actually ran IAAFT nulls
    assert (scored[src_col] == ("computed" if converged else "computed_nonconverged")).all()
