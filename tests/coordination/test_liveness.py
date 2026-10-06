"""TF-58 Task 18 Step 7 (ADR-032): every coordination metric column is live (non-NaN AND non-constant) over
the union of fixtures, plus the R2 idsse_half preconditions and the measured commit-1 limitations.

Owner ruling (2026-09-27): with the INTERIM params, ``coord_median_freq_cpm`` + the four ``coord_coh_*`` are
unreachable on real football (no in-play segment reaches the 545 s spectral minimum; coherence needs ~4x400 s
welch windows), and ``coord_vc_n_stationary`` is identically 0 (``vc_epsilon == 0`` -> strict ``<`` never
fires). Those columns are therefore exercised in the regime where the math is well-posed -- a contiguous
synthetic match for spectral/coherence, a positive-epsilon fixture for the stationary count -- and a
precondition test pins that the committed real fixtures / interim epsilon cannot reach them. Real-data liveness
of these columns moves to D3 (Task 23) under the D2-calibrated params. The interim base params are NOT changed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

import silly_kicks.coordination as C
from silly_kicks.coordination import CoordinationParams, build_coordination_signals, compute_team_coordination
from silly_kicks.coordination._kernels._spectral import min_spectral_samples
from silly_kicks.coordination._windows import period_windows
from tests.coordination._fixtures import make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]

_FIX = Path(__file__).resolve().parents[1] / "datasets" / "tracking" / "idsse_half"
_LIVENESS_SURROGATES = 19  # liveness needs values, not the K=199 chance resolution (spec §9.5)

_METRIC_COLUMNS = {
    "pair": C.COORDINATION_PAIR_METRIC_COLUMNS,
    "pair_phase": C.COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    "spectral": C.COORDINATION_SPECTRAL_METRIC_COLUMNS,
    "cluster_team": C.COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    "cluster_player": C.COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    "team_sync": C.COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
    "rsi": C.COORDINATION_RSI_METRIC_COLUMNS,
}
_ALL_METRIC_COLS = {f"{tbl}.{c}" for tbl, cols in _METRIC_COLUMNS.items() for c in cols}


def _params(**kw):
    return dataclasses.replace(CoordinationParams(), n_surrogates=_LIVENESS_SURROGATES, **kw)


# The union of well-posed regimes (owner ruling): contiguous long matches for spectral/coherence (two lengths so
# coord_coh_n_segments varies), a detection-aware match for the observed-fraction / player-count columns, and a
# positive-epsilon match for the stationary count. Every regime here is one the interim params support.
def _union_fixtures():
    osc = {"provider": "sportec", "oscillation_cpm": 0.5, "phase_offset_deg": 40.0}
    return [
        ("syn2700", make_coordination_match(seconds=2700.0, **osc), _params()),
        ("syn1500", make_coordination_match(seconds=1500.0, **osc), _params()),
        (
            "skillcorner_vis",
            make_coordination_match(seconds=300.0, provider="skillcorner", visibility_drop=0.25),
            _params(),
        ),
        (
            "vc_eps",
            make_coordination_match(seconds=300.0, **osc),
            _params(vc_epsilon={k: 0.1 for k in CoordinationParams().vc_epsilon}),
        ),
    ]


def _accumulate_values(res, values: dict[str, set]):
    for tbl, cols in _METRIC_COLUMNS.items():
        df = getattr(res, tbl)
        for c in cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                values.setdefault(f"{tbl}.{c}", set()).update(s.dropna().tolist())


@pytest.mark.slow
def test_every_metric_column_live_over_union():
    values: dict[str, set] = {}
    for _name, frames, params in _union_fixtures():
        _accumulate_values(compute_team_coordination(frames, params=params), values)
    dead = sorted(col for col in _ALL_METRIC_COLS if len(values.get(col, set())) <= 1)
    assert not dead, f"non-live (NaN or constant) metric columns over the union: {dead}"


# The metric columns the R2 real half CANNOT make live at the interim params, each with the measured reason its own
# precondition test below pins. Every other metric column must be live (non-NaN and non-constant) on it, and this map
# is exact both ways: a column dying OR coming alive on real data changes it, deliberately.
_IDSSE_HALF_NOT_LIVE: dict[str, str] = {
    **dict.fromkeys(
        (
            "pair.coord_xc_max_abs_r_surrogate_mean",
            "pair.coord_xc_max_abs_r_percentile",
            "pair.coord_xc_max_abs_r_excess",
        ),
        "every cross-correlation null is segment_too_short (test_idsse_half_xc_null_hinges_on_the_interim_min_shift)",
    ),
    **dict.fromkeys(
        (
            "pair.coord_coh_band_mean",
            "pair.coord_coh_peak_freq_cpm",
            "pair.coord_coh_n_segments",
            "pair.coord_coh_band_mean_surrogate_mean",
            "pair.coord_coh_band_mean_percentile",
            "pair.coord_coh_band_mean_excess",
            "spectral.coord_median_freq_cpm",
            "spectral.coord_duration_s",
            "spectral.coord_n_segments",
        ),
        "below the spectral / coherence minima (test_committed_real_fixture_below_spectral_and_coherence_minima)",
    ),
    **dict.fromkeys(
        ("pair.coord_vc_n_stationary", "pair_phase.coord_vc_n_stationary"),
        "vc_epsilon == 0 at the interim params (test_interim_vc_epsilon_zeroes_the_stationary_count)",
    ),
    **dict.fromkeys(
        (
            "pair.coord_observed_fraction_a",
            "pair.coord_observed_fraction_b",
            "cluster_team.coord_observed_fraction",
            # the A-08 gate input (owner ruling 2026-10-04): a fully observed provider's share is 1.0 by construction
            *(f"{tbl}.coord_detected_share" for tbl in _METRIC_COLUMNS),
        ),
        "IDSSE is fully observed: every observed fraction is 1 (test_idsse_half_is_fully_observed_with_full_teams)",
    ),
    "cluster_team.coord_n_players_mean": (
        "both teams field 11 players in every frame (test_idsse_half_is_fully_observed_with_full_teams)"
    ),
}


@pytest.mark.slow
def test_idsse_half_real_data_liveness_smoke():
    """The R2 real fixture runs end-to-end, and EXACTLY the columns its measured limitations exclude are not live."""
    frames = pd.read_parquet(_FIX / "frames.parquet")
    res = compute_team_coordination(frames, params=_params())
    not_live = {
        f"{tbl}.{c}"
        for tbl, cols in _METRIC_COLUMNS.items()
        for c in cols
        if c not in getattr(res, tbl).columns
        or pd.to_numeric(getattr(res, tbl)[c], errors="coerce").nunique(dropna=True) <= 1
    }
    assert not_live == set(_IDSSE_HALF_NOT_LIVE), (
        f"newly not live: {sorted(not_live - set(_IDSSE_HALF_NOT_LIVE))}; "
        f"now live (drop from _IDSSE_HALF_NOT_LIVE): {sorted(set(_IDSSE_HALF_NOT_LIVE) - not_live)}"
    )


# --------------------------------------------------------------------------- preconditions (ADR-032)
def _idsse_signals():
    frames = pd.read_parquet(_FIX / "frames.parquet")
    return build_coordination_signals(frames, windows=period_windows(frames), params=CoordinationParams())


def test_idsse_half_preconditions():
    frames = pd.read_parquet(_FIX / "frames.parquet")
    actions = pd.read_parquet(_FIX / "actions.parquet")
    import silly_kicks.spadl.config as spadlconfig

    ball = frames[frames["is_ball"]].sort_values("time_seconds")
    assert ball["time_seconds"].max() - ball["time_seconds"].min() >= 1100.0, "duration below 1,100 s"
    # a dead interval > 25 s
    state = ball["ball_state"].astype(str).to_numpy()
    t = ball["time_seconds"].to_numpy()
    longest_dead = 0.0
    i = 0
    while i < len(state):
        if state[i] == "dead":
            j = i
            while j < len(state) and state[j] == "dead":
                j += 1
            longest_dead = max(longest_dead, t[j - 1] - t[i])
            i = j
        else:
            i += 1
    assert longest_dead > 25.0, f"no dead interval > 25 s (longest {longest_dead:.1f})"
    # both teams field 10 outfield players in >= 90% of samples
    out = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"])]
    per = out.groupby(["frame_id", "team_id"]).size()
    for tm in [t for t in pd.unique(frames.loc[~frames["is_ball"], "team_id"].dropna())][:2]:
        assert float((per.xs(tm, level="team_id") == 10).mean()) >= 0.90, f"team {tm} not 10-outfield in >=90% samples"
    # actions cover the span with >= 1 goal and >= 5 restarts
    lo, hi = float(ball["time_seconds"].min()), float(ball["time_seconds"].max())
    assert actions["time_seconds"].between(lo, hi).any()
    tn = actions["type_id"].map(lambda i: spadlconfig.actiontypes[i])
    rn = actions["result_id"].map(lambda i: spadlconfig.results[i])
    assert int(((tn == "shot") & (rn == "success")).sum()) >= 1, "no goal in the span"
    restarts = {"corner_crossed", "corner_short", "freekick_crossed", "freekick_short", "throw_in", "goalkick"}
    assert int(tn.isin(restarts).sum()) >= 5, "fewer than 5 restart actions"


def test_committed_real_fixture_below_spectral_and_coherence_minima():
    """R2 measured limitation: idsse_half meets NEITHER the spectral (>=1 segment of the >=545 s minimum) NOR
    the coherence (>=4 welch segments) minimum at interim params -- so both are structurally uncoverable there."""
    sig = _idsse_signals()
    ps = sig.periods[0]
    min_spec = min_spectral_samples(sig.fs, sig.params.band_low_cpm)
    welch_n = round(sig.params.welch_segment_s * sig.fs)
    for tm in ps.team_ids:
        seg_lengths = [hi - lo for lo, hi in ps.segments[tm]]
        assert max(seg_lengths) < min_spec, f"a segment reached the spectral minimum ({max(seg_lengths)} >= {min_spec})"
        # coherence pools 50%-overlap welch windows across segments; K >= 4 is the minimum
        welch_windows = sum(max(0, (n - welch_n) // (welch_n // 2) + 1) for n in seg_lengths if n >= welch_n)
        assert welch_windows < 4, f"coherence reached >= 4 welch segments ({welch_windows})"


@pytest.mark.parametrize(("min_shift_s", "expected"), [(60.0, "segment_too_short"), (40.0, "computed")])
def test_idsse_half_xc_null_hinges_on_the_interim_min_shift(min_shift_s, expected):
    """Measured limitation (2026-10-03). Cross-correlation reads only slices of >= min_slice_samples(lag) = 60 s, which
    no possession window reaches, so it scores the period window alone; and the half's long stoppages leave a 100.4 s
    piece of play -- long enough to be read, too short to shift at the interim min_shift_s = 60 s (2 tau + 1 =
    120.1 s). So every cross-correlation null on this half is ``segment_too_short``; at a min_shift_s the piece can
    take (40 s), the same nulls are computed. The columns stay live over the union of fixtures."""
    from silly_kicks.coordination._compute import compute_cross_correlation
    from silly_kicks.coordination._kernels._xcorr import min_slice_samples
    from silly_kicks.coordination._windows import possession_windows_from_frames

    frames = pd.read_parquet(_FIX / "frames.parquet")
    sig = _idsse_signals()
    params, fs = sig.params, sig.fs
    min_slice = min_slice_samples(round(params.xcorr_max_lag_s * fs))
    poss = possession_windows_from_frames(frames, n_phases=params.n_phases, params=params)
    assert len(poss) and ((poss["end_time_s"] - poss["start_time_s"]) * fs < min_slice).all()
    assert set(params.min_shift_s.values()) == {60.0}  # the interim base (D1 derives the per-signal values)
    ps = sig.periods[0]
    for tm in ps.team_ids:
        lengths = [hi - lo for lo, hi in ps.segments[tm]]
        assert any(min_slice <= n < 2 * round(60.0 * fs) + 1 for n in lengths), lengths
    shifted = dataclasses.replace(
        params, n_surrogates=_LIVENESS_SURROGATES, min_shift_s=dict.fromkeys(params.min_shift_s, min_shift_s)
    )
    xc, _report = compute_cross_correlation(dataclasses.replace(sig, params=shifted), levels=("team_team",))
    scored = xc[xc["coord_xc_source"] == "scored"]
    assert len(scored)  # non-vacuity: the period window's team pairs are scored
    assert set(scored["coord_xc_surrogate_source"]) == {expected}


def test_idsse_half_is_fully_observed_with_full_teams():
    """Measured limitation: IDSSE is a fully observed provider (every observed fraction is 1), and both teams field
    11 players in every frame of the half (no red card, every substitution exact), so the cluster family's mean player
    count is 11 in every window."""
    frames = pd.read_parquet(_FIX / "frames.parquet")
    assert _idsse_signals().detection_source == "fully_observed"
    players = frames[~frames["is_ball"].to_numpy(dtype=bool)]
    per_frame = players.groupby(["frame_id", "team_id"], observed=True).size()
    assert set(per_frame.unique()) == {11}


def test_interim_vc_epsilon_zeroes_the_stationary_count():
    """R2-class measured limitation: at the interim ``vc_epsilon == 0`` the strict ``<`` makes
    coord_vc_n_stationary identically 0, so it is live only in the D2-calibrated ``vc_epsilon > 0`` regime."""
    assert all(v == 0.0 for v in CoordinationParams().vc_epsilon.values())
    f = make_coordination_match(seconds=300.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
    res = compute_team_coordination(f, params=dataclasses.replace(CoordinationParams(), n_surrogates=0))
    for tbl in ("pair", "pair_phase"):
        s = pd.to_numeric(getattr(res, tbl)["coord_vc_n_stationary"], errors="coerce").dropna()
        assert (s == 0).all(), f"{tbl}.coord_vc_n_stationary not identically 0 at vc_epsilon==0"


def test_synthetic_long_fixtures_meet_spectral_and_coherence_minima():
    """The contiguous synthetic fixtures DO clear both minima (so their liveness is not vacuous)."""
    for secs in (2700.0, 1500.0):
        f = make_coordination_match(seconds=secs, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
        sig = build_coordination_signals(f, windows=period_windows(f), params=CoordinationParams())
        ps = sig.periods[0]
        min_spec = min_spectral_samples(sig.fs, sig.params.band_low_cpm)
        assert max(hi - lo for lo, hi in ps.segments[ps.team_ids[0]]) >= min_spec, f"{secs}s below spectral minimum"
