"""A-20 (owner ruling 2026-10-04): every NaN carries its token; SampEn has its OWN source column.

Rows were ``scored`` with NaN metrics (a constant spectral slice, a coherence band with no bin, an undefined
Cross-SampEn), and a cluster team row was demoted to ``entropy_undefined`` although its rho_group_mean was valid. The
ruling: ``coord_cluster_sampen_source``, ``coord_cluster_player_sampen_source`` and ``coord_team_sync_sampen_source``
carry the SampEn token (``scored`` / ``entropy_undefined``; a degraded row's own token when it was not computed), so
the main token speaks only for the main metrics.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.coordination._catalog import PairSpec
from silly_kicks.coordination._columns import (
    COORDINATION_CLUSTER_PLAYER_COLUMNS,
    COORDINATION_CLUSTER_TEAM_COLUMNS,
    COORDINATION_TEAM_SYNC_COLUMNS,
)
from silly_kicks.coordination._compute import compute_cluster_phase, compute_coherence, compute_spectral
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._report import CoordinationCoverageWarning
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows
from tests.coordination._fixtures import make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]
_P = dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=20.0)


def _match(seconds: float = 120.0, **kw) -> pd.DataFrame:
    return make_coordination_match(
        seconds=seconds, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0, **kw
    )


def _caller(frames: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    w = period_windows(frames).iloc[[0]].copy()
    w["window_source"] = "caller"
    w["window_kind"] = "sliding"
    w["start_time_s"], w["end_time_s"] = lo, hi
    return w.reset_index(drop=True)


def _cluster(frames, windows, params=_P):
    sig = build_coordination_signals(frames, windows=windows, params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        return compute_cluster_phase(sig)


def test_the_sampen_source_columns_exist():
    assert "coord_cluster_sampen_source" in COORDINATION_CLUSTER_TEAM_COLUMNS
    assert "coord_cluster_player_sampen_source" in COORDINATION_CLUSTER_PLAYER_COLUMNS
    assert "coord_team_sync_sampen_source" in COORDINATION_TEAM_SYNC_COLUMNS


def test_an_undefined_sampen_no_longer_demotes_a_valid_cluster_row():
    # a 0.3-s window: 3 usable samples -> rho_group_mean is valid, but SampEn has no m + 1 template match
    f = _match()
    team, players, sync, _rep = _cluster(f, _caller(f, 50.0, 50.3))
    assert team["coord_rho_group_mean"].notna().all()
    assert (team["coord_cluster_source"] == "scored").all()
    assert team["coord_rho_group_sampen"].isna().all()
    assert (team["coord_cluster_sampen_source"] == "entropy_undefined").all()
    scored_players = players[players["coord_cluster_player_source"] == "scored"]
    assert len(scored_players) > 0 and scored_players["coord_rho_k"].notna().all()
    assert (scored_players["coord_cluster_player_sampen_source"] == "entropy_undefined").all()
    assert (sync["coord_team_sync_sampen_source"].isin(["entropy_undefined", "too_short"])).all()


def test_a_defined_sampen_is_scored_in_its_own_column():
    f = _match()
    team, players, sync, _rep = _cluster(f, period_windows(f))
    assert (team["coord_cluster_sampen_source"] == "scored").all() and team["coord_rho_group_sampen"].notna().all()
    assert (players["coord_cluster_player_sampen_source"] == "scored").any()
    scored = sync[sync["coord_team_sync_sampen_source"] == "scored"]
    assert len(scored) > 0 and scored["coord_team_sync_cross_sampen"].notna().all()


def test_a_degraded_row_carries_its_own_token_in_the_sampen_column():
    f = _match(n_outfield=3, with_gk=False)  # 3 < min_players: insufficient_players, nothing computed
    team, players, _sync, _rep = _cluster(f, period_windows(f))
    assert (team["coord_cluster_source"] == "insufficient_players").all()
    assert (team["coord_cluster_sampen_source"] == "insufficient_players").all()
    assert (players["coord_cluster_player_sampen_source"] == "insufficient_players").all()


def test_a_constant_spectral_slice_is_degenerate_not_scored():
    # spec 7.8.4: a zero-power (constant) slice has no median frequency -- NaN with degenerate_constant, never scored
    f = _match()
    params = dataclasses.replace(_P, band_low_cpm=2.0, band_high_cpm=6.0)  # 2-period minimum = 60 s at 10 Hz
    sig = build_coordination_signals(f, windows=period_windows(f), params=params)
    for tm in sig.periods[0].team_ids:
        sig.periods[0].team_signal[(tm, "spread")][:] = 5.0
    spec, _rep = compute_spectral(sig)
    spread = spec[spec["signal"] == "spread"]
    assert len(spread) == 2
    assert spread["coord_median_freq_cpm"].isna().all()
    assert (spread["coord_spectral_source"] == "degenerate_constant").all()
    other = spec[spec["signal"] == "centroid_x"]
    assert (other["coord_spectral_source"] == "scored").all()  # the other side: a live signal is scored


def test_a_coherence_band_without_a_bin_is_too_short_not_scored():
    # welch_segment_s = 20 s resolves 3 cycles/min: no bin falls in [0.22, 0.83] -- the segment is too short to
    # resolve the band, so there is no band mean to report
    f = _match(seconds=300.0)
    sig = build_coordination_signals(f, windows=period_windows(f), params=_P)
    pair, _rep = compute_coherence(sig, pairs=[PairSpec("team_team", "centroid_x", "centroid_x", "canonical")])
    assert (pair["coord_coh_source"] == "too_short").all()
    assert pair["coord_coh_band_mean"].isna().all()
    resolved = dataclasses.replace(_P, welch_segment_s=120.0)  # the other side: 0.5 cpm bins -> scored
    sig2 = build_coordination_signals(f, windows=period_windows(f), params=resolved)
    pair2, _rep = compute_coherence(sig2, pairs=[PairSpec("team_team", "centroid_x", "centroid_x", "canonical")])
    assert (pair2["coord_coh_source"] == "scored").all() and pair2["coord_coh_band_mean"].notna().all()


_TOKEN_METRICS = {
    "cluster_team": {
        "coord_cluster_source": ["coord_rho_group_mean", "coord_rho_group_sd", "coord_n_players_mean"],
        "coord_cluster_sampen_source": ["coord_rho_group_sampen"],
    },
    "cluster_player": {
        "coord_cluster_player_source": ["coord_phi_mean_deg", "coord_rho_k"],
        "coord_cluster_player_sampen_source": ["coord_phi_sampen"],
    },
    "team_sync": {
        "coord_team_sync_source": ["coord_team_sync_pearson_r"],
        "coord_team_sync_sampen_source": ["coord_team_sync_cross_sampen"],
    },
}


@pytest.mark.parametrize("lo_hi", [(0.0, 120.0), (50.0, 50.3), (50.0, 51.0)])
def test_a_scored_token_never_sits_on_a_nan_metric(lo_hi):
    f = _match()
    team, players, sync, _rep = _cluster(f, _caller(f, *lo_hi))
    for table, df in (("cluster_team", team), ("cluster_player", players), ("team_sync", sync)):
        for token_col, metrics in _TOKEN_METRICS[table].items():
            scored = df[df[token_col] == "scored"]
            assert np.isfinite(scored[metrics].to_numpy(dtype=np.float64)).all(), (table, token_col, lo_hi)
