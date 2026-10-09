"""TF-58 Task 17: coordination family computes + orchestrator."""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from silly_kicks.coordination import _compute
from silly_kicks.coordination._catalog import PairSpec
from silly_kicks.coordination._columns import COORDINATION_PAIR_KEYS
from silly_kicks.coordination._compute import (
    _orientation_signs,
    compute_cluster_phase,
    compute_coherence,
    compute_cross_correlation,
    compute_relative_phase,
    compute_relative_stretch,
    compute_spectral,
    compute_team_coordination,
    compute_vector_coding,
)
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._detection import DetectionCounts
from silly_kicks.coordination._report import CoordinationCoverageWarning
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows, phase_assignment, possession_windows_from_actions
from silly_kicks.tracking import GoalMap
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore:vx/vy columns not found")
_FAST = dataclasses.replace(CoordinationParams(), n_surrogates=5, welch_segment_s=20.0)
_FAST0 = dataclasses.replace(_FAST, n_surrogates=0)


@pytest.fixture(scope="module")
def signals():
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
    return build_coordination_signals(f, windows=period_windows(f), params=_FAST)


def _team_row(df, signal="centroid_x"):
    sub = df[(df.level == "team_team") & (df.signal_a == signal) & (df.window_kind == "period")]
    return sub.iloc[0]


def test_team_centroid_relative_phase_recovers_planted_offset(signals):
    pair, _phase, _rep = compute_relative_phase(signals)
    row = _team_row(pair)
    assert abs(float(row.coord_rp_mean_deg) - 40.0) < 3.0
    assert float(row.coord_rp_resultant_length) > 0.9


def test_positive_lag_means_a_leads():
    # phase offset 6 deg at 0.5 cpm (period 120 s) == team B lagging team A by 2 s.
    f = make_coordination_match(seconds=600.0, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=6.0)
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST)
    pair, _rep = compute_cross_correlation(sig)
    row = _team_row(pair)
    assert float(row.coord_xc_lag_s) == pytest.approx(2.0, abs=0.3)


def test_vector_coding_fractions_sum_to_one(signals):
    pair, _phase, _rep = compute_vector_coding(signals)
    row = _team_row(pair)
    total = (
        float(row.coord_vc_pct_in_phase)
        + float(row.coord_vc_pct_anti_phase)
        + float(row.coord_vc_pct_a_phase)
        + float(row.coord_vc_pct_b_phase)
    )
    assert total == pytest.approx(1.0)


def _stepped_match(dead_intervals) -> pd.DataFrame:
    """Team 1's whole side sits 20 m further up the pitch from t = 60 s: a planted step inside (40, 80)."""
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", dead_intervals=dead_intervals)
    step = (~f["is_ball"] & (f["team_id"] == 1) & (f["time_seconds"] >= 60.0)).to_numpy(dtype=bool, na_value=False)
    f.loc[step, "x"] += 20.0
    return f


def _vc_steps_seen(monkeypatch, f: pd.DataFrame) -> np.ndarray:
    """Every |delta_A| the vector coding classifies (window AND phase subdivisions) on the team centroid_x pair."""
    seen: list[np.ndarray] = []
    real = _compute._vc_stats_into

    def spy(row, da, db, eps_a, eps_b):
        seen.append(np.abs(np.asarray(da)))
        return real(row, da, db, eps_a, eps_b)

    monkeypatch.setattr(_compute, "_vc_stats_into", spy)
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST0)
    compute_vector_coding(sig, pairs=[PairSpec("team_team", "centroid_x", "centroid_x", "canonical")])
    return np.concatenate(seen)


def test_vector_coding_differences_never_span_a_split(monkeypatch):
    # The 20 m step happens inside a 40 s stoppage (> max_stoppage_s = 25 s), so the split keeps it out of every
    # difference. Without the stoppage the same step IS differenced (the filter spreads it over ~1 s) -- the
    # counterfactual that makes the first assertion non-vacuous.
    split = _vc_steps_seen(monkeypatch, _stepped_match([(40.0, 80.0)]))
    joined = _vc_steps_seen(monkeypatch, _stepped_match([]))
    assert split.max() < 0.5 < joined.max()


def _vc_phase_statuses(steps: np.ndarray) -> list[str]:
    """``_vc_pair_phase`` on a hand-built window: one run over ``steps.size + 1`` samples, both teams stepping alike."""
    m = steps.size + 1
    va = np.concatenate([[0.0], np.cumsum(steps)])
    empty = np.empty(0)
    det = DetectionCounts.from_masks(np.ones(m, dtype=bool), np.ones(m, dtype=bool))
    ctx = _compute._Ctx(
        np.arange(m), [(0, m)], va, va.copy(), empty, empty, empty, empty, det, det, empty, empty, 1.0, 1.0
    )
    ps = SimpleNamespace(game_id=1, period_id=1)
    signals = SimpleNamespace(detection_source="all_rows", stoppage=SimpleNamespace(source="none"))
    spec = PairSpec("team_team", "centroid_x", "centroid_x", "canonical")
    b = _compute._Binding(1, 2, pd.NA, pd.NA, False)
    wrow = {"window_kind": "possession", "window_id": 0, "n_phases": 3}  # the window's own n_phases (A-22)
    out = _compute._vc_pair_phase(ps, wrow, spec, b, ctx, 0, m, 10.0, 0.01, 0.01, _FAST0, signals)
    return [status for _row, status in out]


def test_vector_coding_min_samples_per_subdivision_both_sides():
    # spec 7.8.3: >= 3 non-stationary samples per phase subdivision. 12 samples -> 4 per third -> 3 steps each:
    # scored; 9 samples -> 3 per third -> 2 steps each: too_short.
    assert _vc_phase_statuses(np.full(11, 1.0)) == ["scored"] * 3
    assert _vc_phase_statuses(np.full(8, 1.0)) == ["too_short"] * 3
    # a stationary step (|delta| < epsilon on both sides) does not count: the middle third keeps only 2
    steps = np.full(11, 1.0)
    steps[5] = 0.0  # samples 4..7 are the middle third; steps 4, 5, 6 are its differences
    assert _vc_phase_statuses(steps) == ["scored", "too_short", "scored"]


def test_coherence_near_one_for_coupled_centroids():
    # welch_segment_s must resolve the 0.22-0.83 cpm band (nperseg >= ~1200 at 10 Hz) and the window must give K>=4.
    f = make_coordination_match(seconds=600.0, hz=10.0, provider="sportec", phase_offset_deg=40.0)
    p = dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=120.0)
    sig = build_coordination_signals(f, windows=period_windows(f), params=p)
    pair, _rep = compute_coherence(sig)
    row = _team_row(pair)
    assert row.coord_coh_source == "scored"
    assert float(row.coord_coh_band_mean) > 0.9


def test_spectral_row_per_team_signal_and_possession(signals):
    spec, _rep = compute_spectral(signals)
    per = spec[spec.window_kind == "period"]
    assert (per.signal == "possession").sum() >= 1
    assert set(per[per.signal != "possession"].signal.unique()) >= {"centroid_x", "convex_hull_area"}


def test_cluster_rho_high_for_synchronised_team(signals):
    ct, _cp, _ts, _rep = compute_cluster_phase(signals)
    scored = ct[(ct.window_kind == "period") & (ct.coord_cluster_source == "scored")]
    assert len(scored)
    assert float(scored.coord_rho_group_mean.max()) > 0.8


_REFERENCE_NUMERICS_ENV = "SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS"


def _numerics_module(numerics: str):
    """The cluster kernels of one numerics mode: ADR-111 D4's (production) or the as-built arithmetic (reference)."""
    from silly_kicks.coordination._kernels import _cluster, _cluster_reference

    return _cluster_reference if numerics == "reference" else _cluster


def _per_draw_cluster_surrogate(ps, tm, axis, s, e, params, fs, numerics="production"):
    """The cluster null computed the direct way, with plan R1's shift unit (each player's own phase runs): per draw,
    roll each contributing player's phasor within each of its runs over the WHOLE period, then re-run
    cluster_phase + window_cluster_stats (of the given numerics). The batched path must equal this bit for bit."""
    import silly_kicks.coordination._compute as CC
    from silly_kicks.coordination._kernels._surrogates import draw_shifts, surrogate_rng, surrogate_triple
    from silly_kicks.id_compat import canonical_id

    cluster_phase = _numerics_module(numerics).cluster_phase
    window_cluster_stats = _numerics_module(numerics).window_cluster_stats
    z, valid, pids, _on = CC._cluster_inputs(ps, tm, axis, params)
    _q, rel, usable = cluster_phase(z, valid, params.min_players)
    team = str(canonical_id(tm))
    series = {str(canonical_id(p)): pl for (t2, p), pl in ps.players.items() if str(canonical_id(t2)) == team}
    tau = round(params.min_shift_s["player_x" if axis == "x" else "player_y"] * fs)
    units = []
    for j, pid in enumerate(pids):
        for lo, hi in series[str(canonical_id(pid))].runs:
            lo, hi = int(lo), int(hi)
            if hi <= s or lo >= e or not valid[lo:hi, j].all():
                continue  # outside the window, or a run whose phase was never computed
            key = (str(canonical_id(ps.game_id)), int(ps.period_id), lo, team, axis, str(canonical_id(pid)), "cluster")
            sh = draw_shifts(surrogate_rng(params.surrogate_seed, key), hi - lo, tau, params.n_surrogates)
            if sh is None:
                return (float("nan"), float("nan"), float("nan")), "segment_too_short"
            units.append((j, lo, hi, sh))
    obs_mean = window_cluster_stats(rel, valid, usable, s, e).rho_group_mean
    dist = []
    for draw in range(params.n_surrogates):
        zk = z.copy()
        for j, lo, hi, sh in units:
            zk[lo:hi, j] = np.roll(z[lo:hi, j], int(sh[draw]))
        _qk, relk, usablek = cluster_phase(zk, valid, params.min_players)
        dist.append(window_cluster_stats(relk, valid, usablek, s, e).rho_group_mean)
    return surrogate_triple(obs_mean, np.array(dist)), "computed"


def _stoppage_sub_signals(dead_intervals=((100.0, 140.0), (150.0, 180.0)), substitution_s: float = 7.0):
    """240 s with stoppages (they split the team segments AND every player's run, spec 7.4 step 2), an early
    substitution whose OUTGOING player's run [0, 7) s = 70 samples is shorter than 2*tau + 1 = 101, and 20 s sliding
    windows. The default [150, 180) stoppage also leaves a 10 s gap between stoppages: a 10 s team segment and
    10 s player runs, too short to shift."""
    import pandas as pd

    f = make_coordination_match(
        seconds=240.0,
        hz=10.0,
        provider="sportec",
        dead_intervals=list(dead_intervals),
        substitution=(substitution_s, 3),
    )
    base = CoordinationParams()
    params = dataclasses.replace(
        base, n_surrogates=19, welch_segment_s=20.0, min_shift_s=dict.fromkeys(base.min_shift_s, 5.0)
    )
    windows = pd.concat([period_windows(f), period_windows(f, length_s=20.0, step_s=15.0)], ignore_index=True)
    return build_coordination_signals(f, windows=windows, params=params), params


def _pearson_one(a, b, numerics):
    """One draw's team-sync Pearson r: ``np.corrcoef`` (as-built) or the D4 two-pass formula (production)."""
    if numerics == "reference":
        return float(np.corrcoef(a, b)[0, 1])
    from silly_kicks.coordination._kernels._cluster import pearson_rows

    return float(pearson_rows(a, b[None, :], np.ones(a.size, dtype=bool), 4)[0])


def _legacy_team_sync_surrogate(ps, axis, s, e, params, fs, numerics="production"):
    """The team-sync surrogate as first built (per draw: copy team B's full amplitude series, roll each segment),
    from the same amplitude inputs ``_team_sync_row`` scores; returns ``None`` where the row is not scored."""
    import silly_kicks.coordination._compute as CC
    from silly_kicks.coordination._kernels._surrogates import draw_shifts, surrogate_rng, surrogate_triple
    from silly_kicks.id_compat import canonical_id

    mod = _numerics_module(numerics)
    amps = []
    for tm in ps.team_ids:
        z, valid, _pids, _on = CC._cluster_inputs(ps, tm, axis, params)
        if z.shape[1] == 0:
            return None
        _q, rel, usable = mod.cluster_phase(z, valid, params.min_players)
        amps.append(CC._amplitude_series(rel, valid, usable, ps.segments[tm], mod.window_cluster_stats))
    amp_a, amp_b_full = amps[0][s:e], amps[1]
    amp_b = amp_b_full[s:e]
    both = np.isfinite(amp_a) & np.isfinite(amp_b)
    if int(both.sum()) < 4 or np.std(amp_a[both]) == 0 or np.std(amp_b[both]) == 0:
        return None
    obs_r = _pearson_one(amp_a[both], amp_b[both], numerics)
    tb = ps.team_ids[1]
    read = s + np.flatnonzero(both)  # the rows the r reads: only B segments holding one are shifted (plan Task 17)
    segs = [
        (int(lo), int(hi))
        for lo, hi in np.asarray(ps.segments[tb]).reshape(-1, 2)
        if ((read >= lo) & (read < hi)).any()
    ]
    tau = round(params.min_shift_s["cluster_amplitude"] * fs)
    shift_cols = []
    for lo, hi in segs:
        key = (str(canonical_id(ps.game_id)), int(ps.period_id), int(lo), str(canonical_id(tb)), axis, "team_sync")
        sh = draw_shifts(surrogate_rng(params.surrogate_seed, key), hi - lo, tau, params.n_surrogates)
        if sh is None:
            return (float("nan"), float("nan"), float("nan")), "segment_too_short"
        shift_cols.append(sh)
    dist = []
    for draw in range(params.n_surrogates):
        bk = amp_b_full.copy()
        for (lo, hi), sh in zip(segs, shift_cols, strict=True):
            bk[lo:hi] = np.roll(amp_b_full[lo:hi], int(sh[draw]))
        bw = bk[s:e]
        mask = both & np.isfinite(bw)
        if int(mask.sum()) < 4:
            dist.append(float("nan"))
        elif numerics == "reference" and (np.std(amp_a[mask]) == 0 or np.std(bw[mask]) == 0):
            dist.append(float("nan"))
        else:  # the D4 two-pass formula returns NaN itself for a zero-variance side
            dist.append(_pearson_one(amp_a[mask], bw[mask], numerics))
    return surrogate_triple(obs_r, np.array(dist)), "computed"


def _bits_equal(have: float, want: float) -> bool:
    return (np.isnan(have) and np.isnan(want)) or np.float64(have).view(np.int64) == np.float64(want).view(np.int64)


@pytest.mark.parametrize("numerics", ["production", "reference"])
def test_cluster_surrogate_equals_the_per_draw_player_run_loop(monkeypatch, numerics):
    # Windows inside a stoppage (no team segment and no player run there: neither is scored), windows cut by the
    # late substitute's too-short run, and ordinary windows: every scored row's surrogate triple must equal the
    # direct per-draw player-run loop exactly -- and team sync (team segments, unchanged) its own direct loop -- in
    # BOTH numerics: ADR-111 D4's (production) and the as-built arithmetic (the reference mode).
    import silly_kicks.coordination._compute as CC

    if numerics == "reference":
        monkeypatch.setenv(_REFERENCE_NUMERICS_ENV, "1")
    sig, params = _stoppage_sub_signals()
    ct, _cp, ts, _rep = compute_cluster_phase(sig)
    seen: dict[str, int] = {}
    for ps in sig.periods:
        for row_idx, s, e in ps.window_ranges:
            wrow = sig.windows.loc[row_idx]
            in_window = (ct.window_kind == wrow["window_kind"]) & (ct.window_id == wrow["window_id"])
            for tm in ps.team_ids:
                for axis in ("x", "y"):
                    got = ct[in_window & (ct.team_id == CC._cid(tm)) & (ct.axis == axis)]
                    assert len(got) == 1
                    row = got.iloc[0]
                    tsegs = np.asarray(ps.segments[tm]).reshape(-1, 2)
                    if not ((tsegs[:, 1] > int(s)) & (tsegs[:, 0] < int(e))).any():
                        # inside a long stoppage the player runs are split too (spec 7.4 step 2): nothing to score
                        assert row.coord_cluster_source != "scored"
                        seen["no_team_segment_unscored"] = seen.get("no_team_segment_unscored", 0) + 1
                        continue
                    if row.coord_cluster_surrogate_source == "not_scored":
                        continue
                    (sm, pc, ex), src = _per_draw_cluster_surrogate(
                        ps, tm, axis, int(s), int(e), params, sig.fs, numerics
                    )
                    seen[src] = seen.get(src, 0) + 1
                    assert row.coord_cluster_surrogate_source == src
                    assert _bits_equal(float(row.coord_rho_group_mean_surrogate_mean), sm)
                    assert _bits_equal(float(row.coord_rho_group_mean_percentile), pc)
                    assert _bits_equal(float(row.coord_rho_group_mean_excess), ex)
            for axis in ("x", "y"):
                sync = ts[
                    (ts.window_kind == wrow["window_kind"]) & (ts.window_id == wrow["window_id"]) & (ts.axis == axis)
                ]
                assert len(sync) == 1
                srow = sync.iloc[0]
                legacy = _legacy_team_sync_surrogate(ps, axis, int(s), int(e), params, sig.fs, numerics)
                if legacy is None:
                    assert srow.coord_team_sync_surrogate_source == "not_scored"
                    continue
                (sm, pc, ex), src = legacy
                seen[f"team_sync_{src}"] = seen.get(f"team_sync_{src}", 0) + 1
                assert srow.coord_team_sync_surrogate_source == src
                assert _bits_equal(float(srow.coord_team_sync_pearson_r_surrogate_mean), sm)
                assert _bits_equal(float(srow.coord_team_sync_pearson_r_percentile), pc)
                assert _bits_equal(float(srow.coord_team_sync_pearson_r_excess), ex)
    # every branch is exercised: shifted draws, the too-short refusal, team sync, and windows inside a stoppage
    assert {"computed", "no_team_segment_unscored", "segment_too_short", "team_sync_computed"} <= set(seen), seen


def test_cluster_surrogate_never_emits_a_zero_variance_null(monkeypatch):
    # HARD GATE (F1): a `computed` cluster null must actually vary. The team-segment design reported windows with
    # no overlapping team segment as `computed` with the observed value x K (percentile 0.5, excess 0).
    import silly_kicks.coordination._compute as CC

    sig, _params = _stoppage_sub_signals()
    nulls: list[np.ndarray] = []
    active = {"on": False}
    real_surrogate, real_triple = CC._cluster_surrogate, CC.surrogate_triple

    def _cluster_surrogate(*args, **kwargs):
        active["on"] = True
        try:
            return real_surrogate(*args, **kwargs)
        finally:
            active["on"] = False

    def _triple(obs, surr):
        if active["on"]:
            nulls.append(np.asarray(surr, dtype=np.float64))
        return real_triple(obs, surr)

    monkeypatch.setattr(CC, "_cluster_surrogate", _cluster_surrogate)
    monkeypatch.setattr(CC, "surrogate_triple", _triple)
    ct, _cp, _ts, _rep = compute_cluster_phase(sig)
    computed = int((ct.coord_cluster_surrogate_source == "computed").sum())
    assert computed > 0 and len(nulls) == computed  # non-vacuity: every computed row's null was inspected
    degenerate = [n for n in nulls if np.isfinite(n).any() and float(np.nanvar(n)) == 0.0]
    assert not degenerate, f"{len(degenerate)} of {len(nulls)} computed nulls have zero variance"


def test_include_goalkeeper_dyad_adds_the_keeper_pairs():
    """``include_goalkeeper["dyad"]`` (spec 7.14, per method): default False keeps outfield dyads only; True adds every
    pair with a keeper -- scored like any other dyad (non-vacuity: the flag must change the dyad table)."""
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    w = period_windows(f)
    on = dataclasses.replace(_FAST, include_goalkeeper={**_FAST.include_goalkeeper, "dyad": True})
    off_rows = compute_relative_phase(build_coordination_signals(f, windows=w, params=_FAST), levels=("dyad",))[0]
    on_rows = compute_relative_phase(build_coordination_signals(f, windows=w, params=on), levels=("dyad",))[0]
    keepers = {100, 200}  # the fixture's goalkeeper ids (_pid(team, 0))

    def keeper_rows(rows):
        ids = rows[["player_a_id", "player_b_id"]].astype("Int64")
        return rows[ids.isin(keepers).any(axis=1).to_numpy()]

    assert keeper_rows(off_rows).empty  # default: no keeper dyads
    kept = keeper_rows(on_rows)
    # per axis: 10 new same-team pairs per team (keeper x 10 outfielders) + 21 new cross-team pairs (11 x 11 - 10 x 10)
    n_axes = on_rows["axis"].nunique()
    assert len(on_rows) - len(off_rows) == n_axes * (2 * 10 + 21) == len(kept)
    assert (kept["coord_rp_source"] == "scored").any()


def test_include_goalkeeper_cluster_false_drops_the_keeper():
    """``include_goalkeeper["cluster"]`` (spec 7.14, per method): default True puts the keeper in the cluster roster
    (Duarte 2013); False drops it -- the player table loses both keepers and the team synchrony moves (non-vacuity)."""
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    w = period_windows(f)
    off = dataclasses.replace(_FAST0, include_goalkeeper={**_FAST0.include_goalkeeper, "cluster": False})
    ct_on, cp_on = compute_cluster_phase(build_coordination_signals(f, windows=w, params=_FAST0))[:2]
    ct_off, cp_off = compute_cluster_phase(build_coordination_signals(f, windows=w, params=off))[:2]
    keepers = {100, 200}  # the fixture's goalkeeper ids (_pid(team, 0))
    assert keepers <= set(cp_on["player_id"].astype("Int64"))
    assert not keepers & set(cp_off["player_id"].astype("Int64"))
    on, off_ = ct_on["coord_rho_group_mean"].to_numpy(), ct_off["coord_rho_group_mean"].to_numpy()
    assert np.isfinite(on).all() and np.isfinite(off_).all() and not np.allclose(on, off_)


def test_include_goalkeeper_cluster_false_drops_the_keeper_even_when_dyads_keep_it():
    # B m16 / minor 16: with dyad=True the keeper IS in ps.players, so the `_cluster_roster` guard is the only thing
    # keeping it out of the cluster table. The prior test used dyad=False, where the keeper never reached the roster
    # anyway -- so the m11 mutation (guard removed) survived. Here the guard is load-bearing and this kills it.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    gk = {**_FAST0.include_goalkeeper, "cluster": False, "dyad": True}
    params = dataclasses.replace(_FAST0, include_goalkeeper=gk)
    cp = compute_cluster_phase(build_coordination_signals(f, windows=period_windows(f), params=params))[1]
    assert len(cp)  # non-vacuity: the cluster player table has rows
    assert not {100, 200} & set(cp["player_id"].astype("Int64"))  # no keeper, though dyad=True put it in ps.players


def _both_runs_oracle(seg_a, seg_b, s, e):
    """``_both_runs`` by definition (spec 7.4 step 2 + D15): every non-empty window ∩ A-segment ∩ B-segment slice, in
    order -- so two segments that TOUCH (an on-pitch-count change cuts a segment with no gap) stay two slices."""
    out = []
    for la, ha in seg_a:
        for lb, hb in seg_b:
            lo, hi = max(int(la), int(lb), s), min(int(ha), int(hb), e)
            if lo < hi:
                out.append((lo, hi))
    return sorted(out)


def _random_segments(rng, n, n_segs):
    """Sorted, non-overlapping segments; drawing cuts WITH replacement makes consecutive segments touch sometimes."""
    cuts = np.sort(rng.choice(np.arange(n + 1), size=2 * n_segs, replace=True)).reshape(-1, 2)
    return cuts[cuts[:, 0] < cuts[:, 1]].astype(np.int64).reshape(-1, 2)


def test_both_runs_split_at_touching_segment_boundaries():
    from silly_kicks.coordination._compute import _both_runs

    whole, cut = np.array([[0, 100]]), np.array([[0, 50], [50, 100]])
    assert _both_runs(whole, cut, 0, 100) == [(0, 50), (50, 100)]  # a touching cut on B splits the slice
    assert _both_runs(cut, whole, 10, 90) == [(10, 50), (50, 90)]  # ... and on A, inside the window
    # the other side of the rule: a real gap, a plain overlap and no overlap behave as before
    assert _both_runs(np.array([[0, 40], [60, 100]]), whole, 0, 100) == [(0, 40), (60, 100)]
    assert _both_runs(np.array([[0, 60]]), np.array([[30, 100]]), 0, 100) == [(30, 60)]
    assert _both_runs(whole, cut, 100, 120) == []


def test_both_runs_equals_every_segment_intersection():
    # `_both_runs` visits only the segments overlapping the window (SkillCorner: ~140 per side, 5k calls a match); its
    # slices are exactly the window ∩ A-segment ∩ B-segment intersections -- touching, empty and unsorted lists
    # included.
    from silly_kicks.coordination._compute import _both_runs

    rng = np.random.default_rng(3)
    checked = touching = 0
    for _ in range(300):
        n = int(rng.integers(20, 400))
        seg_a = _random_segments(rng, n, int(rng.integers(0, 9)))
        seg_b = _random_segments(rng, n, int(rng.integers(0, 9)))
        touching += any(seg_b[i, 1] == seg_b[i + 1, 0] for i in range(len(seg_b) - 1))
        if rng.random() < 0.2:
            seg_b = seg_b[::-1]  # order-agnostic
        s = int(rng.integers(0, n - 1))
        e = int(rng.integers(s + 1, n + 1))
        want = _both_runs_oracle(seg_a, seg_b, s, e)
        assert _both_runs(seg_a, seg_b, s, e) == want
        checked += bool(want)
    assert checked > 50 and touching > 10  # non-vacuity: many joint slices, and touching segments were drawn


def test_red_card_windowed_slices_never_cross_the_count_change():
    # The red card removes a player from each team at 120 s, so both teams' segments are cut there with NO gap (the
    # count step would otherwise leak into spectra and correlations, spec 7.4 step 2). Every windowed slice the pair
    # families and the RSI read must end at the cut, never run across it.
    from silly_kicks.coordination._compute import _both_runs

    f = make_coordination_match(seconds=240.0, hz=10.0, provider="sportec", red_card=(120.0, 3))
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST)
    ps = sig.periods[0]
    ta, tb = ps.team_ids
    cuts = {
        int(hi) for (lo, hi), (nlo, _nhi) in zip(ps.segments[ta][:-1], ps.segments[ta][1:], strict=True) if hi == nlo
    }
    assert cuts  # non-vacuity: the red card cut team A's segment with no gap
    runs = _both_runs(ps.segments[ta], ps.segments[tb], 0, len(ps.t))
    assert len(runs) >= 2
    assert not [(lo, hi) for lo, hi in runs for c in cuts if lo < c < hi]


def _merge_touching_both_runs(real):
    """A `_both_runs` that UN-does the count-cut: it merges adjacent touching slices back into one (the pre-M1
    behaviour). Threaded in via monkeypatch, a family whose output is unchanged is NOT reading the count-cut split."""

    def wrapped(seg_a, seg_b, s, e):
        runs = real(seg_a, seg_b, s, e)
        out: list[tuple[int, int]] = []
        for lo, hi in runs:
            if out and out[-1][1] == lo:
                out[-1] = (out[-1][0], hi)
            else:
                out.append((lo, hi))
        return out

    return wrapped


def _family_output(family, sig):
    fn = {
        "relative_phase": compute_relative_phase,
        "cross_correlation": compute_cross_correlation,
        "vector_coding": compute_vector_coding,
        "coherence": compute_coherence,
        "spectral": compute_spectral,
        "relative_stretch": compute_relative_stretch,
    }[family]
    return fn(sig)[0]


@pytest.mark.parametrize("family", ["relative_phase", "cross_correlation", "vector_coding", "coherence"])
def test_family_consumes_the_count_cut_split(family, monkeypatch):
    # A-18: M-13 (window ∩ A-seg ∩ B-seg) was pinned only at the `_both_runs` unit; making a FAMILY merge touching
    # slices back survived the whole suite. Each pair family here must read the split: on a red-card match (teams cut
    # with no gap at 120 s), replacing `_both_runs` with a merging variant must change the family's output -- or the
    # family rejects the spanning slice outright (XC's `_xc_identity` strict raise). Either proves the wiring; "no
    # change" fails the test, and the test also fails if `_both_runs` itself is ever made to merge (base would merge).
    # (RSI's switch events and the possession spectrum are pinned separately below -- their observed statistics are
    # sample-set invariant, so only their split-sensitive parts -- switch counting / slice segmentation -- move.)
    f = make_coordination_match(seconds=240.0, hz=10.0, provider="sportec", red_card=(120.0, 3))
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST)
    base = _family_output(family, sig)
    assert len(base)  # non-vacuity: the family scored rows on the red-card match

    monkeypatch.setattr(_compute, "_both_runs", _merge_touching_both_runs(_compute._both_runs))
    try:
        merged = _family_output(family, sig)
    except (ValueError, AssertionError):
        return  # the family refuses a slice spanning the count cut -> it depends on the split (both-sides)
    assert not base.reset_index(drop=True).equals(merged.reset_index(drop=True)), (
        f"{family}: merging touching slices left the output unchanged -> it ignores the count-cut split (M-13)"
    )


def test_rsi_switch_events_respect_run_boundaries():
    # A-18 (RSI incl. switch events): the sign-switch detector -- the single source behind coord_rsi_switch_rate_per_min
    # and rsi_switch_times -- counts sign changes only WITHIN a run, never across the count cut between two touching
    # runs. A flip straddling the boundary is NOT a switch; flips inside each run are.
    from silly_kicks.coordination._compute import _sign_change_local_indices

    runs = [(0, 2), (2, 4)]
    assert _sign_change_local_indices(np.array([1.0, 1.0, -1.0, -1.0]), runs) == []  # flip at the boundary: not counted
    assert _sign_change_local_indices(np.array([1.0, -1.0, -1.0, 1.0]), runs) == [1, 3]  # one flip inside each run


def test_possession_spectrum_consumes_the_count_cut_split(monkeypatch):
    # A-18 (possession spectrum): the possession-series spectrum slices per window AND both teams' segments via
    # `_both_runs`, so a count cut splits it. Two touching segments -> two median slices; merging them -> one. Built
    # directly because the possession slices rarely reach min_spectral_samples through the full fixture.
    from silly_kicks.coordination._compute import _possession_spectral_row

    n, cut, ta, tb = 100, 50, 1, 2
    seg = np.array([[0, cut], [cut, n]])  # touching at the cut, same for both teams
    pattern = np.empty(n)
    pattern[:cut] = np.tile([1.0, 0.0], cut // 2)  # fast alternation (period 2) in the first run
    pattern[cut:] = np.tile([1.0] * 5 + [0.0] * 5, (n - cut) // 10)  # slower alternation (period 10) in the second run
    possession = np.array([ta if p == 1.0 else tb for p in pattern], dtype=object)
    ps = SimpleNamespace(
        game_id=1, period_id=1, possession_team=possession, team_ids=(ta, tb), segments={ta: seg, tb: seg}
    )
    signals = SimpleNamespace(detection_source="fully_observed", stoppage=SimpleNamespace(source="ball_state"))
    wrow = pd.Series({"window_kind": "period", "window_id": 0})

    real = _compute._both_runs
    base = _possession_spectral_row(ps, wrow, 0, n, 10.0, 10, signals)
    monkeypatch.setattr(_compute, "_both_runs", _merge_touching_both_runs(real))
    merged = _possession_spectral_row(ps, wrow, 0, n, 10.0, 10, signals)
    assert base["coord_n_segments"] == 2 and merged["coord_n_segments"] == 1  # split -> 2 slices, merge -> 1


def test_cluster_null_never_resplits_the_phasors_per_window(monkeypatch):
    # R8 structural guard (ADR-111 ruling C): the numba null reads the context's phasors as real/imag parts. They are
    # split ONCE per (period, team, axis) context and handed over -- the kernel's own per-call split (a copy of the
    # whole period's phasors per window, ~3 s per full-tracking match) must never run on the compute path.
    from silly_kicks.coordination._kernels import _cluster
    from silly_kicks.coordination._kernels._numba import use_numba
    from tests._perf_structural import call_counter

    if not use_numba():
        pytest.skip("the phasor split feeds only the numba null")
    sig, _params = _stoppage_sub_signals()
    kernel_splits = call_counter(monkeypatch, _cluster, "phasor_parts")
    ct, _cp, _ts, _rep = compute_cluster_phase(sig)
    assert int((ct.coord_cluster_surrogate_source == "computed").sum()) > 1  # non-vacuity: several windows scored
    assert kernel_splits["n"] == 0


def test_cluster_window_below_two_usable_samples_is_too_short():
    # One usable sample: each player's mean relative phasor IS its one relative phasor, so rho_group is exactly 1 for
    # the data and for every shift draw -- no information, and its percentile is float rounding (the 12 no-flip
    # crossings, all on 0.1 s IDSSE possession windows, 2026-10-02). Below 2 usable samples the family is too_short.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", oscillation_cpm=0.5)
    # the cluster surrogate shifts each PLAYER run; pin player min_shift small so the 300 s segment shifts and the
    # 2-sample window's null is "computed" (the commit-2 derivation default player min_shift is ~200 s, longer than
    # this synthetic segment, which would make every window segment_too_short and erase the scored/too_short contrast).
    fast = dataclasses.replace(
        _FAST, min_shift_s={**CoordinationParams().min_shift_s, "player_x": 0.5, "player_y": 0.5}
    )

    def sliding_rows(length_s):
        sig = build_coordination_signals(f, windows=period_windows(f, length_s=length_s, step_s=60.0), params=fast)
        ct, cp = compute_cluster_phase(sig)[:2]
        return ct[ct.window_kind == "sliding"], cp[cp.window_kind == "sliding"]

    (one, one_players), (two, two_players) = sliding_rows(0.1), sliding_rows(0.2)
    assert len(one) and (one.coord_n_samples == 1).all()  # non-vacuity: the windows ARE single-sample
    assert (one.coord_cluster_source == "too_short").all()
    assert one.coord_rho_group_mean.isna().all() and one.coord_rho_group_mean_percentile.isna().all()
    assert one.coord_rho_group_sd.isna().all() and one.coord_rho_group_sampen.isna().all()
    # a too_short window now KEEPS its roster's player rows, tokenised (ADR-042; owner ruling 2026-10-04, A-21)
    assert len(one_players) and (one_players.coord_cluster_player_source == "too_short").all()
    assert one_players.coord_rho_k.isna().all() and one_players.coord_phi_sd_deg.isna().all()
    assert (one.coord_cluster_surrogate_source == "not_scored").all()
    assert len(two_players)  # the other side: a 2-sample window does carry its players
    assert len(two) and (two.coord_n_samples == 2).all()  # the other side of the bound still scores
    assert (two.coord_cluster_source != "too_short").all()
    assert two.coord_rho_group_mean.notna().all() and (two.coord_cluster_surrogate_source == "computed").all()


def test_cluster_surrogate_refuses_a_too_short_player_run():
    # Plan R1: the shift unit is each player's own run, so a window touching the substituted player's 7 s run
    # (70 < 2*tau + 1 = 101 samples at tau = 5 s) cannot be shifted -> `segment_too_short`. With ONE stoppage both
    # team segments ([0, 100) and [140, 240) s) are long, so only the player run can refuse here.
    sig, _params = _stoppage_sub_signals(dead_intervals=((100.0, 140.0),), substitution_s=7.0)
    assert all(int(hi) - int(lo) >= 101 for ps in sig.periods for tm in ps.team_ids for lo, hi in ps.segments[tm])
    ct, _cp, _ts, _rep = compute_cluster_phase(sig)
    period = ct[(ct.window_kind == "period") & (ct.coord_cluster_source != "insufficient_players")]
    assert len(period) == 4  # 2 teams x 2 axes
    assert (period.coord_cluster_surrogate_source == "segment_too_short").all()
    sliding = sig.windows[sig.windows.window_kind == "sliding"]
    last = sliding.loc[sliding.start_time_s.idxmax()]  # [210, 230) s: far from the too-short run
    late = ct[(ct.window_kind == "sliding") & (ct.window_id == last.window_id)]
    assert len(late) == 4
    assert (late.coord_cluster_surrogate_source == "computed").all()


def _irregular_rhythm_frames(seconds: float = 240.0, hz: float = 10.0, seed: int = 7):
    """The synthetic match with each team's outfield players locked to ONE common but irregular rhythm.

    A pure sinusoid shifted in time is only a constant per-player lag, which cluster phase removes by design (Frank &
    Richardson), so it cannot show a null moving. A random-walk frequency around 0.5 cpm keeps the players
    synchronised with each other while making every time shift a genuinely different phase history.
    """
    f = make_coordination_match(seconds=seconds, hz=hz, provider="sportec")
    rng = np.random.default_rng(seed)
    n = round(seconds * hz)
    out = f.copy()
    players = out[~out["is_ball"] & ~out["is_goalkeeper"]]
    for _team, team_rows in players.groupby("team_id", sort=True):
        freq_hz = (0.5 / 60.0) * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        common = 52.5 + 15.0 * np.sin(2.0 * np.pi * np.cumsum(freq_hz) / hz)
        for _pid, rows in team_rows.groupby("player_id", sort=True):
            pos = np.round(rows["time_seconds"].to_numpy(dtype=np.float64) * hz).astype(np.int64)
            x = common[pos] + rng.uniform(-12.0, 12.0) + rng.normal(0.0, 0.3, pos.size)
            out.loc[rows.index, "x"] = x.astype(out["x"].to_numpy().dtype)
    return out


def test_cluster_surrogate_null_moves_for_a_synchronised_team():
    # Non-vacuity (mandatory): players locked to one irregular rhythm are synchronised in x; shifting each player's
    # run independently must put the observation at the top of its null. A null that is not shifted -- or shifted
    # by one common lag -- sits at the observation instead (percentile ~0.5).
    f = _irregular_rhythm_frames()
    base = CoordinationParams()
    params = dataclasses.replace(
        base, n_surrogates=19, welch_segment_s=20.0, min_shift_s=dict.fromkeys(base.min_shift_s, 5.0)
    )
    ct, _cp, _ts, _rep = compute_cluster_phase(build_coordination_signals(f, windows=period_windows(f), params=params))
    x_rows = ct[(ct.window_kind == "period") & (ct.axis == "x")]
    assert len(x_rows) == 2
    assert (x_rows.coord_cluster_surrogate_source == "computed").all()
    assert (x_rows.coord_rho_group_mean >= 0.9).all()  # the precondition: the team really is synchronised
    assert (x_rows.coord_rho_group_mean_percentile >= 0.95).all()
    assert (x_rows.coord_rho_group_mean_excess > 0.0).all()


def test_cluster_numerics_redefinition_moves_no_token_and_no_percentile(monkeypatch):
    # ADR-111 D4 against the as-built arithmetic (the reference mode) on every cluster-family table: no source token and
    # no percentile moves; every other float column within 1e-12, the measured max reported.
    sig, _params = _stoppage_sub_signals()
    monkeypatch.setenv(_REFERENCE_NUMERICS_ENV, "1")
    reference = compute_cluster_phase(sig)[:3]
    monkeypatch.delenv(_REFERENCE_NUMERICS_ENV)
    production = compute_cluster_phase(sig)[:3]
    for name, ref, new in zip(("cluster_team", "cluster_player", "team_sync"), reference, production, strict=True):
        assert list(ref.columns) == list(new.columns) and len(ref) == len(new) > 0, name
        worst = 0.0
        for col in ref.columns:
            if ref[col].dtype != np.float64:
                assert ref[col].astype(object).fillna("<NA>").equals(new[col].astype(object).fillna("<NA>")), (
                    name,
                    col,
                )
                continue
            have, want = new[col].to_numpy(float), ref[col].to_numpy(float)
            np.testing.assert_array_equal(np.isnan(have), np.isnan(want), err_msg=f"{name}.{col}")
            if col.endswith("_percentile"):
                np.testing.assert_array_equal(have, want, err_msg=f"{name}.{col}")
            fin = ~np.isnan(want)
            if fin.any():
                worst = max(worst, float(np.max(np.abs(have[fin] - want[fin]))))
        print(f"{name}: max |D4 - as-built| over every float column = {worst:.3e}")
        assert worst <= 1e-12, name


def _no_goal_map() -> GoalMap:
    # a real GoalMap whose every defended end is unresolved (empty mappings; not a fake duck type)
    return GoalMap(resolved={}, guessed={}, unresolved=frozenset())


def test_unwrap_phase_runs_makes_a_pm_pi_crossing_continuous():
    # A-25: phi_k is a wrapped angle; unwrapping per run makes the SampEn metric see a continuous phase across the
    # +/-pi seam instead of a spurious 2pi jump. A run [3.0, -3.0] has a true step of +0.283 rad (< pi), so unwrap
    # lifts the -3.0 to -3.0 + 2pi.
    out = _compute._unwrap_phase_runs([np.array([3.0, -3.0])])
    np.testing.assert_allclose(out[0], [3.0, -3.0 + 2 * np.pi], atol=1e-12)


def test_unwrap_phase_runs_raises_when_a_step_reaches_pi():
    # the precondition (checked, not assumed): a consecutive wrapped step at/over pi means the analysis rate is too
    # low to unwrap reliably -> fail loud with the offending step, never a silently-wrong SampEn.
    with pytest.raises(ValueError, match=r"phase step.*>= pi"):
        _compute._unwrap_phase_runs([np.array([0.0, np.pi, 0.0])])


def test_sig_goal_unresolved_covers_positional_and_compactness_x():
    # A-24: a signal needs its team's defended end iff it is positional OR back-line-derived (compactness_x, a MAGNITUDE
    # signal built from the defended-end mask). spread/convex_hull_area do not; the check keys on goal_x being None.
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    ps = build_coordination_signals(f, windows=period_windows(f), params=_FAST0, goal_map=_no_goal_map()).periods[0]
    tm = ps.team_ids[0]
    for sig in ("centroid_x", "defensive_line_x", "back_line_high_x", "compactness_x"):
        assert _compute._sig_goal_unresolved(ps, sig, tm), sig  # end-dependent + unresolved
    for sig in ("spread", "convex_hull_area", "team_length"):
        assert not _compute._sig_goal_unresolved(ps, sig, tm), sig  # magnitude, end-independent


def test_spectral_flags_back_line_signals_when_the_goal_end_is_unresolved():
    # A-24: the spectral path had NO goal-end check, so a back-line-derived signal was scored against a guessed end.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", oscillation_cpm=0.5)
    unresolved = build_coordination_signals(f, windows=period_windows(f), params=_FAST0, goal_map=_no_goal_map())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        spec_u, _rep = compute_spectral(unresolved)
    for sig in ("defensive_line_x", "back_line_high_x", "compactness_x", "centroid_x"):
        rows = spec_u[spec_u.signal == sig]
        assert len(rows) > 0 and (rows.coord_spectral_source == "goal_end_unresolved").all(), sig
    assert (spec_u[spec_u.signal == "spread"].coord_spectral_source != "goal_end_unresolved").all()  # end-independent
    # the other side: with a resolved goal map the same back-line signals are scored
    resolved = build_coordination_signals(f, windows=period_windows(f), params=_FAST0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        spec_r, _rep = compute_spectral(resolved)
    assert (spec_r[spec_r.signal == "compactness_x"].coord_spectral_source != "goal_end_unresolved").all()


def test_pair_compactness_x_is_goal_end_unresolved():
    # A-24: compactness_x is magnitude but end-dependent; a pair reading it must flag goal_end_unresolved (it was not,
    # because the pair check looked at kind=="positional" only).
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", oscillation_cpm=0.5)
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST0, goal_map=_no_goal_map())
    pair, _rep = compute_cross_correlation(
        sig,
        levels=("cross_variable",),
        pairs=[PairSpec("cross_variable", "team_length", "compactness_x", "attacking_defending")],
    )
    scored_or_unresolved = pair[pair.coord_xc_source.isin(["goal_end_unresolved", "no_possession_role"])]
    assert len(scored_or_unresolved) == len(pair)  # period windows -> no_possession_role; but none SCORED on a guess
    # on a possession window the attacking_defending pair binds and must be goal_end_unresolved, not scored
    sig2 = build_coordination_signals(f, windows=_one_possession_window(f, 2), params=_FAST0, goal_map=_no_goal_map())
    pair2, _rep = compute_cross_correlation(
        sig2,
        levels=("cross_variable",),
        pairs=[PairSpec("cross_variable", "team_length", "compactness_x", "attacking_defending")],
    )
    assert len(pair2) > 0 and (pair2.coord_xc_source == "goal_end_unresolved").all()


def _one_possession_window(f, attacking_team_id) -> pd.DataFrame:
    """One possession window over the whole first period with a chosen attacker (spec 7.7 pair-order test input)."""
    w = period_windows(f).iloc[[0]].copy()
    w["window_kind"] = "possession"
    w["window_source"] = "possession_tracking"
    w["attacking_team_id"] = pd.array([attacking_team_id], dtype="object")
    w["n_phases"] = pd.array([3], dtype="Int64")
    return w.reset_index(drop=True)


@pytest.mark.parametrize(("attacking", "other"), [(2, 1), (1, 2)])
def test_rsi_pair_order_is_attacking_first_on_possession_windows(attacking, other):
    # spec 7.7 (A-23): on a possession window A = the attacking team, so RSI = SI_attacking - SI_defending. Team 2
    # sits twice as deep as team 1 (more stretched on x), so SI_2 > SI_1: with team 2 attacking the mean is positive,
    # with team 1 attacking it is the negative of that. The emitted team_a_id is the attacking team.
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    rows = (~f["is_ball"] & ~f["is_goalkeeper"] & (f["team_id"] == 2)).to_numpy(dtype=bool, na_value=False)
    slot = (f["player_id"].to_numpy(dtype="float64", na_value=np.nan)[rows] % 100 - 1).astype(int)
    depth = np.linspace(-12.0, 12.0, 10)
    depth -= depth.mean()
    f.loc[rows, "x"] += depth[slot] * 1.0  # team 2 twice as deep -> SI_2 = 2 * SI_1 on x
    sig = build_coordination_signals(f, windows=_one_possession_window(f, attacking), params=_FAST0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        rsi, _rep = compute_relative_stretch(sig)
    row = rsi[(rsi.window_kind == "possession") & (rsi.axis == "x") & (rsi.coord_rsi_source == "scored")].iloc[0]
    assert int(row.team_a_id) == attacking and int(row.team_b_id) == other
    assert (float(row.coord_rsi_mean_m) > 0.5) == (attacking == 2)  # SI_2 > SI_1: positive iff attacking is team 2


def test_cross_team_dyad_pair_order_is_attacking_first_on_possession_windows():
    # A-23: a cross-team dyad on a possession window (dyad_windows="all") orders team_a = attacking team.
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", oscillation_cpm=0.5)
    sig = build_coordination_signals(f, windows=_one_possession_window(f, 2), params=_FAST0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        pair = compute_cross_correlation(sig, levels=("dyad",), dyad_windows="all")[0]
    cross = pair[pair.team_a_id.notna() & pair.team_b_id.notna()]
    cross = cross[cross.team_a_id.map(int) != cross.team_b_id.map(int)]  # cross-team dyads only
    assert len(cross) > 0
    assert (cross.team_a_id.map(int) == 2).all() and (cross.team_b_id.map(int) == 1).all()


def _stretched_rsi_x(scale_of_t) -> pd.Series:
    """Team 2's formation depth scaled by ``1 + scale(t)`` while team 1 stays put, so on the x axis
    RSI = SI_1 - SI_2 = -scale(t) * SI_1, with SI_1 = mean |depth| = 20/3 m. Returns the period-window x row."""
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    rows = (~f["is_ball"] & ~f["is_goalkeeper"] & (f["team_id"] == 2)).to_numpy(dtype=bool, na_value=False)
    slot = (f["player_id"].to_numpy(dtype="float64", na_value=np.nan)[rows] % 100 - 1).astype(int)
    depth = np.linspace(-12.0, 12.0, 10)
    depth -= depth.mean()
    f.loc[rows, "x"] += depth[slot] * scale_of_t(f["time_seconds"].to_numpy()[rows])
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST0)
    rsi, _rep = compute_relative_stretch(sig)
    return rsi[(rsi.window_kind == "period") & (rsi.axis == "x")].iloc[0]


def test_rsi_sign_and_bimodality_on_alternating_stretch():
    # Team 2's depth breathes +-50% with a 60 s period: RSI = -(10/3) sin(2 pi t / 60) -- mean 0, positive half the
    # time, 9 sign switches in 5 minutes, and a sine's arcsine distribution is bimodal: BC = 1 / (3 - 1.5) = 2/3 > 5/9.
    alt = _stretched_rsi_x(lambda t: 0.5 * np.sin(2 * np.pi * t / 60.0))
    assert alt.coord_rsi_source == "scored"
    assert abs(float(alt.coord_rsi_mean_m)) < 0.05
    assert float(alt.coord_rsi_fraction_positive) == pytest.approx(0.5, abs=0.02)
    assert float(alt.coord_rsi_switch_rate_per_min) == pytest.approx(9 / 5.0, abs=0.05)
    assert float(alt.coord_rsi_bimodality_coefficient) == pytest.approx(2 / 3, abs=0.03)
    # the other side: team 2 permanently half as deep -> team 1 always the more stretched, by 10/3 m, unimodal
    const = _stretched_rsi_x(lambda t: np.full(t.shape, -0.5))
    assert float(const.coord_rsi_mean_m) == pytest.approx(10 / 3, abs=0.05)
    assert float(const.coord_rsi_fraction_positive) == 1.0
    assert float(const.coord_rsi_switch_rate_per_min) == 0.0
    assert float(const.coord_rsi_bimodality_coefficient) < 5 / 9


def test_a_window_with_no_row_is_counted_under_no_row_not_as_scored():
    # A-30: a possession window under levels=("dyad",) (dyads skip non-period windows, D17) emits NO row, so it was
    # silently counted as scored (n_scored by subtraction) and the coverage warning could never see it. It must be
    # counted under a reason, and conservation must hold.
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    w = pd.concat([period_windows(f), possession_windows_from_actions(a, f, n_phases=3)], ignore_index=True)
    sig = build_coordination_signals(f, windows=w, params=_FAST0, actions=a)
    n_possession = int((w.window_kind == "possession").sum())
    assert n_possession > 0  # fixture precondition (ADR-032)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        _pair, _phase, rep = compute_relative_phase(sig, levels=("dyad",))
    assert rep.windows_dropped.get("no_row", 0) == n_possession
    assert rep.n_windows_scored == len(w) - n_possession  # the possession windows are NOT scored
    assert rep.conservation_errors() == []  # windows_in == scored + sum(dropped)


@pytest.mark.parametrize("att_raw", ["1", 1, np.int64(1)])
def test_attacking_defending_maps_a_drifted_window_id_to_the_frame_team_objects(att_raw):
    # A-26: the windows table's attacking_team_id can differ in dtype from the frames' team_id (str vs int). The
    # binding's team id keys into the frame-keyed signal maps, so the raw window value would KeyError -- it must be
    # mapped through id_compat to the frame's own team object, whatever the drift.
    att, deff = _compute._attacking_defending(att_raw, ta=1, tb=2)  # frames use int team ids
    assert att == 1 and deff == 2 and type(att) is int  # the FRAME object, never the raw str/int64 window value
    att2, deff2 = _compute._attacking_defending("2", ta=1, tb=2)
    assert att2 == 2 and deff2 == 1  # the other side


def test_cross_variable_binding_survives_attacking_team_id_dtype_drift():
    # end to end: cross_variable pairs are attacking_defending on possession windows; a str attacking_team_id over int
    # frame team ids must NOT KeyError into the frame-keyed signal maps, and the emitted team_a_id is a frame id.
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
    a = make_coordination_actions(f)
    w = possession_windows_from_actions(a, f, n_phases=3).astype({"attacking_team_id": "object"})
    w["attacking_team_id"] = w["attacking_team_id"].map(lambda v: str(int(v)) if pd.notna(v) else v)
    assert w["attacking_team_id"].notna().any()  # fixture precondition: at least one possession window has an attacker
    sig = build_coordination_signals(f, windows=w, params=_FAST0, actions=a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        pair, _rep = compute_cross_correlation(sig, levels=("cross_variable",))  # no KeyError on the drifted id
    resolved = pair[pair.team_a_id.notna()]
    assert len(resolved) > 0  # non-vacuity: attacking/defending pairs were bound (not all no_possession_role)
    assert set(resolved.team_a_id.map(int)) <= {1, 2}  # the emitted ids are the FRAME team ids


def test_a_fully_scored_call_has_no_no_row_windows():
    # the other side: every window produces a scored row -> no "no_row" reason, every window scored
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        _pair, _phase, rep = compute_relative_phase(sig)
    assert "no_row" not in rep.windows_dropped
    assert rep.n_windows_scored == rep.n_windows_in and rep.conservation_errors() == []


def test_dyads_period_only_by_default_and_all_when_requested():
    # D17 with possession windows present, so the default is actually exercised: dyads score period windows only,
    # unless dyad_windows="all".
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    w = pd.concat([period_windows(f), possession_windows_from_actions(a, f, n_phases=3)], ignore_index=True)
    sig = build_coordination_signals(f, windows=w, params=_FAST0, actions=a)
    assert set(w.window_kind) == {"period", "possession"}  # fixture precondition (ADR-032)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        for compute in (compute_relative_phase, compute_cross_correlation):
            default = compute(sig, levels=("dyad",))[0]
            everything = compute(sig, levels=("dyad",), dyad_windows="all")[0]
            assert set(default.window_kind) == {"period"}, compute.__name__
            assert set(everything.window_kind) == {"period", "possession"}, compute.__name__


def test_dyad_rows_have_null_vc_and_coh():
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    res = compute_team_coordination(f, params=_FAST)
    dyad = res.pair[res.pair.level == "dyad"]
    assert len(dyad)
    assert dyad["coord_vc_pct_in_phase"].isna().all()
    assert dyad["coord_coh_band_mean"].isna().all()


def test_orchestrator_runs_and_conserves():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    res = compute_team_coordination(f, actions=a, params=_FAST)
    assert res.report.conservation_errors() == []
    for tbl in (res.pair, res.spectral, res.cluster_team, res.team_sync, res.rsi):
        assert len(tbl)


def _phase_dispersed_match(envelope_phase_b: float) -> pd.DataFrame:
    """Each team's players phase-modulated around a 2 cpm carrier by a zero-mean envelope (1.2 rad, 120 s period)
    scaled by their slot, so the team's instantaneous synchrony dips whenever its envelope is far from 0. Team 2's
    envelope is shifted by ``envelope_phase_b``."""
    f = make_coordination_match(seconds=600.0, hz=10.0, provider="sportec", oscillation_cpm=2.0, phase_offset_deg=0.0)
    rows = (~f["is_ball"] & ~f["is_goalkeeper"]).to_numpy(dtype=bool)
    t = f["time_seconds"].to_numpy()[rows]
    team = f["team_id"].to_numpy(dtype="float64", na_value=np.nan)[rows]
    slot = (f["player_id"].to_numpy(dtype="float64", na_value=np.nan)[rows] % 100 - 1).astype(int)
    depth = np.linspace(-12.0, 12.0, 10)
    depth -= depth.mean()
    env = 1.2 * np.sin(2 * np.pi * t / 120.0 + np.where(team == 2, envelope_phase_b, 0.0))
    f.loc[rows, "x"] = 52.5 + 15.0 * np.sin(2 * np.pi * (2.0 / 60.0) * t + (slot - 4.5) / 4.5 * env) + depth[slot]
    return f


def test_team_sync_pearson_positive_for_common_drive():
    # Duarte 2013's team-team r over the two teams' cluster amplitudes: a COMMON synchrony drive -> r > 0; the same
    # drive a quarter-cycle apart (one team dispersed while the other is locked) -> r < 0. Both sides: not vacuous.
    for phase_b, sign in ((0.0, 1.0), (np.pi / 2, -1.0)):
        f = _phase_dispersed_match(phase_b)
        sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST0)
        _ct, _cp, ts, _rep = compute_cluster_phase(sig)
        row = ts[(ts.window_kind == "period") & (ts.axis == "x")].iloc[0]
        assert row.coord_team_sync_source == "scored"
        assert sign * float(row.coord_team_sync_pearson_r) > 0.4, phase_b


# --------------------------------------------------------------------------- n_phases per window (A-22)
# A window's OWN n_phases decides its subdivision (spec 7.6, D3 "default 3 on possession windows"); NA = none. The
# compute ignored it and gave every window params.n_phases (3) phase rows -- thirds of a 45-minute half included,
# which are not Moura's possession thirds (owner ruling 2026-10-04).
def _caller_windows(frames, spans):
    """Caller windows over the fixture's period: one ``(window_kind, start_s, end_s, n_phases)`` per window."""
    base = period_windows(frames).iloc[[0]]
    rows, next_id = [], {}
    for kind, lo, hi, n in spans:
        r = base.copy()
        r["window_source"] = "caller"
        r["window_kind"] = kind
        r["window_id"] = pd.array([next_id.get(kind, 0)], dtype="Int64")
        next_id[kind] = next_id.get(kind, 0) + 1
        r["start_time_s"], r["end_time_s"] = float(lo), float(hi)
        r["n_phases"] = pd.array([n], dtype="Int64")
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def _phase_indices(phase, signal="centroid_x"):
    sub = phase[(phase.level == "team_team") & (phase.signal_a == signal)]
    return {
        (kind, int(wid)): sorted(g.phase_index.astype(int).tolist())
        for (kind, wid), g in sub.groupby(["window_kind", "window_id"], sort=True)
    }


def _osc_match(seconds):
    return make_coordination_match(
        seconds=seconds, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0
    )


def test_pair_phase_subdivisions_use_time_thirds():
    # C5 rank form: sample i of m is in phase k when (i+1)/m is in ((k-1)/3, k/3] -- the first sample is in phase 1.
    assert phase_assignment(10, 3).tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    f = _osc_match(300.0)
    sig = build_coordination_signals(f, windows=_caller_windows(f, [("sliding", 0.0, 300.0, 3)]), params=_FAST0)
    pair, phase, _rep = compute_relative_phase(sig)
    row = pair[(pair.level == "team_team") & (pair.signal_a == "centroid_x")].iloc[0]
    sub = phase[(phase.level == "team_team") & (phase.signal_a == "centroid_x")].sort_values("phase_index")
    m = int(row.coord_n_samples)
    assert sub.phase_index.tolist() == [1, 2, 3]
    assert sub.coord_n_samples.tolist() == np.bincount(phase_assignment(m, 3))[1:].tolist()
    assert float(sub.coord_duration_s.sum()) == pytest.approx(float(row.coord_duration_s))


def test_period_and_sliding_windows_emit_no_pair_phase_rows():
    f = _osc_match(120.0)
    w = pd.concat([period_windows(f), period_windows(f, length_s=40.0, step_s=40.0)], ignore_index=True)
    assert w["n_phases"].isna().all()  # fixture precondition (ADR-032): the builders stamp NA
    sig = build_coordination_signals(f, windows=w, params=_FAST0)
    for compute, src in ((compute_relative_phase, "coord_rp_source"), (compute_vector_coding, "coord_vc_source")):
        pair, phase, _rep = compute(sig)
        assert (pair[src] == "scored").any(), compute.__name__  # non-vacuity: the windows ARE scored
        assert len(phase) == 0, compute.__name__


def test_each_window_is_subdivided_by_its_own_n_phases():
    # one call, three windows: n = 2, n = 4 and NA -> 2 phases, 4 phases and none (each window read separately)
    f = _osc_match(180.0)
    w = _caller_windows(f, [("sliding", 0.0, 60.0, 2), ("sliding", 60.0, 120.0, 4), ("sliding", 120.0, 180.0, pd.NA)])
    sig = build_coordination_signals(f, windows=w, params=_FAST0)
    for compute in (compute_relative_phase, compute_vector_coding):
        _pair, phase, _rep = compute(sig)
        assert _phase_indices(phase) == {("sliding", 0): [1, 2], ("sliding", 1): [1, 2, 3, 4]}, compute.__name__


def test_possession_windows_default_to_three_phases():
    f = _osc_match(180.0)
    a = make_coordination_actions(f)
    w = possession_windows_from_actions(a, f)
    assert (w["n_phases"] == 3).all()
    sig = build_coordination_signals(f, windows=w, params=_FAST0, actions=a)
    pair, phase, _rep = compute_relative_phase(sig)
    scored = pair[(pair.level == "team_team") & (pair.signal_a == "centroid_x") & (pair.coord_rp_source == "scored")]
    assert len(scored) > 0  # non-vacuity
    got = _phase_indices(phase)
    assert all(got[("possession", int(wid))] == [1, 2, 3] for wid in scored.window_id)


def test_a_caller_possession_window_with_na_n_phases_is_not_subdivided():
    f = _osc_match(120.0)
    w = _caller_windows(f, [("possession", 0.0, 60.0, pd.NA), ("possession", 60.0, 120.0, 3)])
    sig = build_coordination_signals(f, windows=w, params=_FAST0)
    _pair, phase, _rep = compute_relative_phase(sig)
    assert _phase_indices(phase) == {("possession", 1): [1, 2, 3]}


def test_each_subdivision_needs_three_samples_both_sides():
    # 9 grid samples split 3/3/3 (all scored); 8 split 2/3/3 -> the first third is too_short
    f = _osc_match(60.0)
    w = _caller_windows(f, [("sliding", 30.0, 30.9, 3), ("sliding", 40.0, 40.8, 3)])
    sig = build_coordination_signals(f, windows=w, params=_FAST0)
    _pair, phase, _rep = compute_relative_phase(sig)
    sub = phase[(phase.level == "team_team") & (phase.signal_a == "centroid_x")].sort_values("phase_index")
    nine, eight = sub[sub.window_id == 0], sub[sub.window_id == 1]
    assert nine.coord_n_samples.tolist() == [3, 3, 3]
    assert (nine.coord_rp_source == "scored").all()
    assert eight.coord_n_samples.tolist() == [2, 3, 3]
    assert eight.coord_rp_source.tolist() == ["too_short", "scored", "scored"]


def test_n_phases_as_numpy_int_or_nullable_int64_gives_the_same_phase_rows():
    f = _osc_match(120.0)
    w = _caller_windows(f, [("sliding", 0.0, 60.0, 2), ("sliding", 60.0, 120.0, 3)])
    a = compute_relative_phase(build_coordination_signals(f, windows=w, params=_FAST0))[1]
    b = compute_relative_phase(build_coordination_signals(f, windows=w.astype({"n_phases": "int64"}), params=_FAST0))[1]
    one = a[(a.level == "team_team") & (a.signal_a == "centroid_x")]
    assert len(one) == 5  # non-vacuity: 2 + 3 phase rows for the pair
    pd.testing.assert_frame_equal(a, b)


def _family_calls(sig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        return {
            "relative_phase": compute_relative_phase(sig),
            "cross_correlation": compute_cross_correlation(sig),
            "vector_coding": compute_vector_coding(sig),
            "coherence": compute_coherence(sig),
            "spectral": compute_spectral(sig),
            "cluster_phase": compute_cluster_phase(sig),
            "relative_stretch": compute_relative_stretch(sig),
        }


def test_orchestrator_tables_match_family_computes():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    w = period_windows(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        res = compute_team_coordination(f, windows=w, params=_FAST)
    fam = _family_calls(build_coordination_signals(f, windows=w, params=_FAST))
    ct, cp, ts, _rep = fam["cluster_phase"]
    for have, want in (
        (res.spectral, fam["spectral"][0]),
        (res.cluster_team, ct),
        (res.cluster_player, cp),
        (res.team_sync, ts),
        (res.rsi, fam["relative_stretch"][0]),
    ):
        pd.testing.assert_frame_equal(have.reset_index(drop=True), want.reset_index(drop=True))
    keys = list(COORDINATION_PAIR_KEYS)
    for name in ("relative_phase", "cross_correlation", "vector_coding", "coherence"):
        table = fam[name][0]
        own = [c for c in table.columns if c not in keys and table[c].notna().any()]
        merged = table[keys + own].merge(res.pair[keys + own], on=keys, how="left", suffixes=("", "__orch"))
        assert len(merged) == len(table), name
        for c in own:
            pd.testing.assert_series_equal(merged[c], merged[f"{c}__orch"], check_names=False, obj=f"{name}.{c}")
    # the merged report carries EVERY family's rows (spec 7.13: per method, rows per source token)
    expected: dict[str, dict[str, int]] = {}
    for out in fam.values():
        for family, counts in out[-1].rows_by_source.items():
            bucket = expected.setdefault(family, {})
            for token, n in counts.items():
                bucket[token] = bucket.get(token, 0) + n
    assert {k: dict(v) for k, v in res.report.rows_by_source.items()} == expected


def test_orchestrator_default_windows_period_plus_one_possession_source():
    # C25 / spec 7.2: with no windows the orchestrator scores period windows plus ONE possession source -- the
    # events builder when actions are passed, the tracking builder otherwise; never both.
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        tracking_only = compute_team_coordination(f, params=_FAST0)
        with_events = compute_team_coordination(f, actions=make_coordination_actions(f), params=_FAST0)
    assert set(tracking_only.windows.window_source) == {"period", "possession_tracking"}
    assert set(with_events.windows.window_source) == {"period", "possession_events"}


def _stoppage_windows():
    """Possession windows every 12 s around a 60 s stoppage: the windows inside it have no samples, so every family
    drops some windows (spec 7.13's trigger has something to count) and the orchestrator keeps the rest."""
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec", dead_intervals=[(60.0, 120.0)])
    a = make_coordination_actions(f)
    w = pd.concat([period_windows(f), possession_windows_from_actions(a, f, n_phases=3)], ignore_index=True)
    return f, a, w


def _coverage_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = call()
    return out, [w for w in caught if issubclass(w.category, CoordinationCoverageWarning)]


def _at(params: CoordinationParams, fraction: float) -> CoordinationParams:
    return dataclasses.replace(params, coverage_warn_fraction=fraction)


def _dropped_share(report) -> float:
    return (report.n_windows_in - report.n_windows_scored) / report.n_windows_in


def test_single_warning_from_orchestrator_both_sides_of_threshold():
    # spec 7.13: ONE call-level warning when the call's dropped share of windows EXCEEDS coverage_warn_fraction,
    # attributed to the caller (stacklevel=2 at the public function) -- never one per family.
    f, a, w = _stoppage_windows()

    def run(fraction):
        return _coverage_warnings(
            lambda: compute_team_coordination(f, windows=w, actions=a, params=_at(_FAST0, fraction))
        )

    res, never = run(1.0)
    share = _dropped_share(res.report)
    assert never == [] and 0.0 < share < 1.0  # non-vacuity: some windows dropped, some scored
    assert run(share)[1] == []  # at the threshold: not "exceeds"
    above = run(share - 1e-9)[1]
    assert len(above) == 1
    assert above[0].filename == __file__


@pytest.mark.parametrize(
    "family",
    [
        "relative_phase",
        "cross_correlation",
        "vector_coding",
        "coherence",
        "spectral",
        "cluster_phase",
        "relative_stretch",
    ],
)
def test_each_family_compute_warns_once_at_its_caller_both_sides(family):
    f, a, w = _stoppage_windows()
    sig = build_coordination_signals(f, windows=w, params=_at(_FAST0, 1.0), actions=a)
    compute = getattr(_compute, f"compute_{family}")
    out, never = _coverage_warnings(lambda: compute(sig))
    share = _dropped_share(out[-1])
    assert never == [] and share > 0.0  # non-vacuity: this family drops some windows here
    at = dataclasses.replace(sig, params=_at(sig.params, share))
    above = dataclasses.replace(sig, params=_at(sig.params, share - 1e-9))
    assert _coverage_warnings(lambda: compute(at))[1] == []
    caught = _coverage_warnings(lambda: compute(above))[1]
    assert len(caught) == 1 and caught[0].filename == __file__


def test_surrogate_draws_are_keyed_on_the_canonical_game_id():
    # spec 7.9: every surrogate seed key is built from the CANONICAL id (ADR-019), so a game id stored as 1.0 draws
    # exactly the surrogates the same game stored as 1 does (str(1.0) != str(1) would re-key every draw).
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    g = f.assign(game_id=f["game_id"].astype("float64"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        a = compute_team_coordination(f, windows=period_windows(f), params=_FAST)
        b = compute_team_coordination(g, windows=period_windows(g), params=_FAST)
    compared = 0
    for name in ("pair", "cluster_team", "team_sync"):
        left, right = getattr(a, name), getattr(b, name)
        for col in [c for c in left.columns if c.endswith("_surrogate_mean")]:
            np.testing.assert_array_equal(left[col].to_numpy(), right[col].to_numpy(), err_msg=f"{name}.{col}")
            compared += int(np.isfinite(left[col].to_numpy(dtype="float64")).sum())
    assert compared > 0  # non-vacuity: real surrogate means were compared


def test_possession_spectrum_is_sliced_by_the_segments():
    # spec 7.8.4: the possession series gets the same window-and-segment slicing as the team signals, so a long
    # stoppage splits it (the series itself holds the last possession straight through the stoppage).
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec", dead_intervals=[(60.0, 120.0)])
    a = make_coordination_actions(f)
    w = pd.concat([period_windows(f), possession_windows_from_actions(a, f, n_phases=3)], ignore_index=True)
    params = dataclasses.replace(_FAST0, band_low_cpm=3.0, band_high_cpm=10.0)  # minimum slice = 2 periods = 40 s
    sig = build_coordination_signals(f, windows=w, params=params, actions=a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        spec, _rep = compute_spectral(sig)
    row = spec[(spec.window_kind == "period") & (spec.signal == "possession")].iloc[0]
    assert row.coord_spectral_source == "scored"
    assert int(row.coord_n_segments) == 2  # [0, 60) and [120, 180): never across the stoppage
    assert float(row.coord_duration_s) == pytest.approx(120.0, abs=0.2)


def test_orientation_signs_c24():
    pos_pos = PairSpec("intra_team", "defensive_line_x", "centroid_x", "same_team")
    assert _orientation_signs(pos_pos, False) == (1.0, 1.0)
    assert _orientation_signs(pos_pos, True) == (-1.0, -1.0)  # both positional -> product +1
    mag = PairSpec("team_team", "spread", "spread", "canonical")
    assert _orientation_signs(mag, True) == (1.0, 1.0)  # magnitude signals never flip
    mixed = PairSpec("cross_variable", "team_length", "compactness_x", "attacking_defending")  # magnitude both
    assert _orientation_signs(mixed, True) == (1.0, 1.0)


def test_determinism_same_seed_same_output():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    sig = build_coordination_signals(f, windows=period_windows(f), params=_FAST)
    a1, _p1, _r1 = compute_relative_phase(sig)
    a2, _p2, _r2 = compute_relative_phase(sig)
    np.testing.assert_array_equal(
        a1.coord_rp_resultant_length_percentile.to_numpy(), a2.coord_rp_resultant_length_percentile.to_numpy()
    )


def test_determinism_is_independent_of_window_order():
    # B m13 / plan Task 17 ("...across_calls_and_window_order"): a window's result does not depend on the ORDER the
    # windows are passed -- surrogate seeds are keyed by (game, period, window, ...), never by position. Reversing the
    # windows reproduces every window's percentile.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
    w = period_windows(f, length_s=90.0, step_s=45.0)  # several sliding windows, so order genuinely varies
    assert len(w) >= 3  # non-vacuity: more than one window to reorder
    key = ["game_id", "period_id", "window_kind", "window_id", "level", "signal_a", "signal_b", "axis", "team_a_id"]
    fwd = compute_relative_phase(build_coordination_signals(f, windows=w, params=_FAST))[0].sort_values(key)
    rev_w = w.iloc[::-1].reset_index(drop=True)
    rev = compute_relative_phase(build_coordination_signals(f, windows=rev_w, params=_FAST))[0].sort_values(key)
    np.testing.assert_array_equal(
        fwd["coord_rp_resultant_length_percentile"].to_numpy(), rev["coord_rp_resultant_length_percentile"].to_numpy()
    )


def test_surrogate_disabled_when_zero():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    p0 = dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=20.0)
    sig = build_coordination_signals(f, windows=period_windows(f), params=p0)
    pair, _phase, _rep = compute_relative_phase(sig)
    scored = pair[pair.coord_rp_source == "scored"]
    assert (scored.coord_rp_surrogate_source == "disabled").all()
