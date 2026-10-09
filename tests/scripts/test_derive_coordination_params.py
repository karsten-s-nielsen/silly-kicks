"""TF-58 Task 20 (D1, §8.2): the pure Tier-B reducers of ``derive_coordination_params``.

Every rule tested from BOTH sides (the both-bands convention). The corpus passes / CLI / provenance wiring are
exercised by the driver integration tests (added with the driver itself); this file pins the reducers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import derive_coordination_params as d


@pytest.fixture(autouse=True)
def _private_manifest(monkeypatch):
    """ADR-038 label lookups never reach the network here: the stub manifest lists nothing, so every match is
    private (fail-closed)."""
    monkeypatch.setattr("scripts._loader_pining.match_visibility", lambda providers, *, token=None, base_url=None: {})


# --------------------------------------------------------------------------- residual cutoff (Pass A)
def test_residual_cutoff_for_run_recovers_signal_band():
    rng = np.random.default_rng(0)
    t = np.arange(4000) / 25.0
    x = np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.05, t.size)
    assert 0.3 < d.residual_cutoff_for_run(x, 25.0) < 1.2


def test_residual_cutoff_for_run_raises_on_noise_free_signal():
    t = np.arange(4000) / 25.0
    with pytest.raises(ValueError, match="noise"):
        d.residual_cutoff_for_run(np.sin(2 * np.pi * 0.3 * t), 25.0)


def test_position_noise_rms_is_the_residual_analysis_floor_not_the_interim_cutoff_residual():
    # A-14: _position_noise_rms must return Winter's residual-analysis noise floor (median over player-axis runs),
    # which recovers a planted white-noise sigma -- NOT the RMS of the residual at the fixed interim 0.4-Hz cutoff
    # (which, by passing most of the band-limited signal to the residual, is much larger than sigma).
    import pandas as pd

    rng = np.random.default_rng(11)
    sigma, native_hz, n = 0.1, 25.0, 8000
    t = np.arange(n) / native_hz
    clean = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # band-limited < 0.6 Hz
    rows = []
    for pid in (1, 2):  # two player runs, one game-period
        xy = clean + rng.normal(0, sigma, n)
        yy = clean + rng.normal(0, sigma, n)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": "g",
                    "period_id": 1,
                    "player_id": pid,
                    "is_ball": False,
                    "frame_rate": native_hz,
                    "time_seconds": t,
                    "x": xy,
                    "y": yy,
                }
            )
        )
    frames = pd.concat(rows, ignore_index=True)
    floor = d._position_noise_rms(frames)
    assert abs(floor - sigma) < 0.3 * sigma  # near the planted sigma
    # non-vacuity: the OLD interim-0.4-Hz-residual RMS is far larger (it keeps the 0.3/0.5 Hz signal in the residual)
    from silly_kicks.tracking.preprocess._butterworth import butterworth_lowpass

    interim_resid = clean + rng.normal(0, sigma, n) - butterworth_lowpass(clean, native_hz, 0.4, 3)
    assert float(np.sqrt(np.mean(interim_resid**2))) > 2 * floor  # the old interim residual over-reads the floor ~2x+


# --------------------------------------------------------------------------- weighting + quantiles
def test_weighted_quantile_equal_weight_parity_with_numpy():
    v = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
    for q in (0.05, 0.5, 0.95):
        assert d.weighted_quantile(v, q, np.ones(v.size)) == float(np.quantile(v, q, method="inverted_cdf"))


def test_weighted_quantile_weights_shift_the_result():
    v = np.array([0.0, 1.0])
    assert d.weighted_quantile(v, 0.5, np.array([9.0, 1.0])) == 0.0  # mass on 0.0
    assert d.weighted_quantile(v, 0.5, np.array([1.0, 9.0])) == 1.0  # mass on 1.0


def test_unit_weights_sum_to_one_and_each_match_sums_to_its_share():
    p = np.array(["a", "a", "a", "b"])
    m = np.array([1, 1, 2, 9])
    w = d.unit_weights(p, m, provider_neutral=False)
    assert abs(w.sum() - 1.0) < 1e-12
    # match-weighted within one/both providers: 3 matches -> each match total 1/3
    df = pd.DataFrame({"m": [f"{pp}-{mm}" for pp, mm in zip(p, m, strict=True)], "w": w})
    assert np.allclose(sorted(df.groupby("m")["w"].sum()), [1 / 3, 1 / 3, 1 / 3])


def test_unit_weights_provider_neutral_gives_each_provider_equal_total():
    # provider a: 2 matches; provider b: 1 match. provider_neutral -> each provider totals 1/2.
    p = np.array(["a", "a", "a", "b", "b"])
    m = np.array([1, 1, 2, 9, 9])
    w = d.unit_weights(p, m, provider_neutral=True)
    assert abs(w.sum() - 1.0) < 1e-12
    by_p = pd.DataFrame({"p": p, "w": w}).groupby("p")["w"].sum()
    assert abs(by_p["a"] - 0.5) < 1e-12 and abs(by_p["b"] - 0.5) < 1e-12


def test_provider_cutoff_is_weighted_median():
    assert d.provider_cutoff(np.array([0.3, 0.4, 0.5]), np.ones(3)) == 0.4


# --------------------------------------------------------------------------- Pass B reducers
def test_vc_epsilon_for_signal_scales_with_noise():
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 0.01, 500))  # smooth
    filt = base.copy()
    quiet = d.vc_epsilon_for_signal(base + rng.normal(0, 0.01, 500), filt)
    loud = d.vc_epsilon_for_signal(base + rng.normal(0, 0.1, 500), filt)
    assert 0.0 < quiet < loud


def test_first_acf_zero_crossing_is_quarter_period_on_a_sinusoid():
    t = np.arange(4000) / 25.0
    assert abs(d.first_acf_zero_crossing_s(np.sin(2 * np.pi * 0.5 * t), 25.0) - 0.5) < 0.02  # period 2s
    assert abs(d.first_acf_zero_crossing_s(np.sin(2 * np.pi * 1.0 * t), 25.0) - 0.25) < 0.02  # period 1s


def test_acf_zero_over_segments_is_per_segment_not_spliced():
    # A-39: the decorrelation estimate must be computed PER stationary segment, never across a gap/stoppage, so it
    # matches how the coordination families read the signal. Two adjacent segments with different periods: the
    # per-segment estimate is the duration-weighted mean of each segment's crossing; the old spliced ACF over the
    # concatenation gives something else (non-vacuity).
    fs, n1, n2 = 25.0, 2000, 2000
    s1 = np.sin(2 * np.pi * 0.5 * np.arange(n1) / fs)  # period 2 s -> crossing 0.5 s
    s2 = np.sin(2 * np.pi * 2.0 * np.arange(n2) / fs)  # period 0.5 s -> crossing 0.125 s
    values = np.concatenate([s1, s2])
    per_seg = d._acf_zero_over_segments(values, [(0, n1), (n1, n1 + n2)], fs)
    assert per_seg == pytest.approx((0.5 + 0.125) / 2, abs=0.03)
    assert abs(d.first_acf_zero_crossing_s(values, fs) - per_seg) > 0.05  # splicing really changes the estimate


def test_median_freq_over_segments_honours_the_metric_min_length():
    # A-39: a segment shorter than the metric's own minimum (min_spectral_samples) must not contribute a median, or a
    # tiny segment inflates band_high_cpm. A sub-minimum segment is excluded; adding it changes nothing.
    from silly_kicks.coordination._config import CoordinationParams
    from silly_kicks.coordination._kernels._spectral import min_spectral_samples

    fs = 10.0
    min_n = min_spectral_samples(fs, CoordinationParams().band_low_cpm)
    long_n, short_n = min_n + 500, 50
    vals = np.concatenate([np.sin(2 * np.pi * 0.25 * np.arange(long_n) / fs), np.zeros(short_n)])
    only_long = d._median_freq_over_segments(vals, [(0, long_n)], fs, min_n)[0]
    with_short = d._median_freq_over_segments(vals, [(0, long_n), (long_n, long_n + short_n)], fs, min_n)[0]
    assert np.isfinite(only_long) and only_long == with_short  # the sub-minimum segment is excluded
    assert np.isnan(d._median_freq_over_segments(vals, [(long_n, long_n + short_n)], fs, min_n)[0])


def test_min_shift_for_signal_is_weighted_p95():
    v = np.arange(1.0, 21.0)  # 1..20
    assert d.min_shift_for_signal(v, np.ones(v.size)) == float(np.quantile(v, 0.95, method="inverted_cdf"))


def test_band_from_median_frequencies_is_p5_p95():
    v = np.arange(1.0, 101.0)
    lo, hi = d.band_from_median_frequencies(v, np.ones(v.size))
    assert lo == float(np.quantile(v, 0.05, method="inverted_cdf"))
    assert hi == float(np.quantile(v, 0.95, method="inverted_cdf"))


def test_welch_segment_rule_r4_pin():
    assert d.welch_segment_rule(0.22, 0.83, 2700.0) == (400.0, True)  # 393.4 -> 400; 12 segments >= 8


def test_welch_segment_rule_narrow_band_cannot_hold_8_segments():
    seg, both = d.welch_segment_rule(0.50, 0.55, 2700.0)  # width 0.05 -> res needs 4800 s
    assert seg == 4800.0 and both is False  # resolution wins, segment-count shortfall flagged


def test_welch_segment_rule_is_the_shortest_segment_meeting_the_resolution(
    interim_band: tuple[float, float] = (0.22, 0.83),
):
    # A-52 (owner ruling 2026-10-04): the SHORTEST segment giving resolution <= band/4, not the longest. At the
    # interim band resolution needs >= 240/0.61 = 393.4 s, rounded up to the next 10 s = 400; the step below (390)
    # resolves 0.1538 cpm > band/4 = 0.1525, so it would NOT meet the requirement.
    seg, _both = d.welch_segment_rule(*interim_band, 2700.0)
    assert seg == 400.0
    assert 60.0 / 390.0 > (interim_band[1] - interim_band[0]) / 4.0  # 390 s under-resolves the band
    assert 60.0 / 400.0 <= (interim_band[1] - interim_band[0]) / 4.0  # 400 s is the first that meets it


def test_residual_grid_spans_the_spec_interval_inclusive():
    # A-52: spec 8.2 says the residual grid is [0.1, 5.0] Hz in 0.05 steps; it must include BOTH endpoints
    # (it stopped at 4.95). residual_analysis_cutoff clips to the evaluable sub-grid per native rate (C18).
    assert d.RESIDUAL_GRID[0] == pytest.approx(0.1)
    assert d.RESIDUAL_GRID[-1] == pytest.approx(5.0)
    assert np.allclose(np.diff(d.RESIDUAL_GRID), 0.05)


def test_possession_gap_argmax_prefers_max_then_smaller_gap():
    assert d.possession_gap_argmax({0.2: 0.5, 0.4: 0.9, 0.6: 0.7}) == 0.4  # clear max
    assert d.possession_gap_argmax({0.2: 0.5, 0.4: 0.8, 0.6: 0.8}) == 0.4  # tie -> smaller gap


# --------------------------------------------------------------------------- occlusion reducers
def test_min_observed_fraction_rule_picks_smallest_qualifying_bin():
    err = {0.5: 0.30, 0.6: 0.20, 0.7: 0.08, 0.8: 0.04}
    assert d.min_observed_fraction_rule(err, between_match_sd=0.20) == 0.7  # 0.5*0.20 = 0.10 -> first <= is 0.7


def test_min_observed_fraction_rule_raises_when_none_qualifies():
    with pytest.raises(ValueError, match="observed-fraction"):
        d.min_observed_fraction_rule({0.6: 0.9, 0.8: 0.8}, between_match_sd=0.2)


def test_max_detection_gap_rule_picks_longest_qualifying_gap():
    rmse = {0.2: 0.05, 0.4: 0.08, 0.6: 0.30}
    assert d.max_detection_gap_rule(rmse, noise_rms=0.05) == 0.4  # 2*0.05 = 0.10 -> 0.2,0.4 qualify -> longest 0.4


def test_max_detection_gap_rule_raises_when_none_qualifies():
    with pytest.raises(ValueError, match="shortest gap"):
        d.max_detection_gap_rule({0.2: 0.5}, noise_rms=0.05)


# --------------------------------------------------------------------------- thin-provider precision (TF58-PLAN-03)
def _match_median_reducer(u: pd.DataFrame) -> float:
    return float(np.median(u.groupby("match")["cutoff"].median()))


def test_provider_bootstrap_se_zero_when_match_medians_agree():
    # every match has median 0.5 (runs vary only WITHIN matches) -> resampling matches cannot move it.
    units = pd.DataFrame(
        {
            "match": [1, 1, 2, 2, 3, 3],
            "cutoff": [0.4, 0.6, 0.3, 0.7, 0.45, 0.55],  # all match medians == 0.5
        }
    )
    se = d.provider_bootstrap_se(units, _match_median_reducer, seed_key=("cutoff", "prov"), n_boot=200)
    assert se == 0.0


def test_provider_bootstrap_se_is_seeded_and_key_dependent():
    units = pd.DataFrame({"match": [1, 1, 2, 2, 3, 3], "cutoff": [0.3, 0.3, 0.5, 0.5, 0.7, 0.7]})
    a1 = d.provider_bootstrap_se(units, _match_median_reducer, seed_key=("q", "A"), n_boot=200)
    a2 = d.provider_bootstrap_se(units, _match_median_reducer, seed_key=("q", "A"), n_boot=200)
    b = d.provider_bootstrap_se(units, _match_median_reducer, seed_key=("q", "B"), n_boot=200)
    assert a1 == a2 and a1 > 0.0  # same key reproducible, non-trivial between-match variation
    assert a1 != b  # a different key draws a different resample set


def test_thin_provider_flags_both_sides():
    per_provider = {"a": 0.30, "b": 0.34, "c": 0.32}
    sd = float(np.std(list(per_provider.values()), ddof=1))
    se = {"a": sd * 0.1, "b": sd * 0.1, "c": sd * 5.0}  # c noisier than the spread
    flags = d.thin_provider_flags(per_provider, se)
    assert flags["c"] == "flagged" and flags["a"] == "ok" and flags["b"] == "ok"


def test_thin_provider_flags_not_assessable_with_one_provider():
    assert d.thin_provider_flags({"a": 0.3}, {"a": 0.01}) == {"a": "not_assessable"}


# --------------------------------------------------------------------------- input contract
def test_input_contract_declares_driver_and_digest():
    ic = d.input_contract()
    assert ic["driver"] == "derive_coordination_params"
    assert isinstance(ic["digest"], str) and len(ic["digest"]) == 64
    assert ic["geometry_version"] == d.GEOMETRY_VERSION


# ============================================================================ driver passes / reduce (integration)
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from _fake_corpus import SpyLoader, make_loaded, make_ref  # noqa: E402

import scripts._coordination_params_codegen as cg  # noqa: E402 -- package form so write_generated_params sees the patch
from tests.coordination._fixtures import make_coordination_match  # noqa: E402

_CLEAN = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}


def _args(tmp_path, **over):
    base = dict(
        out=str(tmp_path),
        providers=("sportec",),
        match_ids_json=None,
        max_matches=None,
        cache_dir=None,
        token=None,
        allow_dirty=True,
        list_matches=False,
        corpus_json=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _write_share(dest, name, df, *, tag="all", **extra):
    """One worker's share of a D1 pass, in the layout ``write_worker_partial`` writes (table + manifest)."""
    keys = sorted({f"{p}__{m}" for p, m in zip(df["provider"], df["match_id"], strict=True)}) if len(df) else []
    df.to_parquet(dest / f"{name}.{tag}.parquet", index=False)
    manifest = {
        "generation": "g1",
        "n_attempted": len(keys),
        "stage_seconds": {"corpus": 1.0},
        "listed": keys,
        "produced": keys,
        "excluded_keys": [],
        "failed_keys": [],
        "run_commit": _CLEAN["commit"],
        "run_tree_dirty": False,
        "run_tree_state": "clean",
        **extra,
    }
    (dest / f"manifest_{name}.{tag}.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_pass_a_resumes_before_load_and_excludes(tmp_path, monkeypatch):
    frames = make_coordination_match(seconds=60.0, provider="sportec")
    matches = {
        ("sportec", "m1"): make_loaded("sportec", "m1", frames=frames, actions=None),
        ("sportec", "m2"): make_loaded("sportec", "m2", frames=frames, actions=None),
    }
    refs = [make_ref("sportec", "m1"), make_ref("sportec", "m2"), make_ref("sportec", "bad")]
    loader = SpyLoader(matches, exclude={("sportec", "bad"): "planted S1 geometry gate"})
    monkeypatch.setattr(d, "corpus_source", lambda args: (refs, loader))

    res1 = d._pass_a(_args(tmp_path), _CLEAN)
    assert res1.excluded == 1 and res1.skipped == 0  # first run: the bad match is excluded, nothing skipped
    assert (res1.shard_dir / "sportec__bad.excluded.json").is_file()

    n_loads = len(loader.calls)  # m1, m2, bad were all loaded on the first pass
    res2 = d._pass_a(_args(tmp_path), _CLEAN)
    # resume-before-load: the two good shards replay as skips, the bad match replays from its marker, none re-loaded.
    assert res2.attempted == 0 and res2.skipped == 2 and res2.excluded == 1
    assert len(loader.calls) == n_loads
    assert (tmp_path / "a.all.parquet").is_file()


def test_pass_a_refuses_discarded_visibility():
    frames = make_coordination_match(seconds=60.0, provider="skillcorner")
    frames["visibility"] = None  # a detection-aware provider with its detection flag discarded (ADR-069 Layer 2)
    loaded = SimpleNamespace(provider="skillcorner", match_id="x", frames=frames, actions=None)
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        d.pass_a_match(loaded)


# --------------------------------------------------------------------------- reduce: pooling + codegen round-trip
def _a_row(provider, mid, run, axis, cutoff):
    return {
        "provider": provider,
        "match_id": mid,
        "period_id": 1,
        "team_id": 1,
        "player_id": run,
        "axis": axis,
        "run_index": run,
        "cutoff_hz": cutoff,
    }


def _b_row(provider, mid, period, team, pid, quantity, signal, value, *, gap_s=None, duration_s=None):
    return {
        "provider": provider,
        "match_id": mid,
        "period_id": period,
        "team_id": team,
        "player_id": pid,
        "quantity": quantity,
        "signal": signal,
        "value": value,
        "gap_s": gap_s,
        "duration_s": duration_s,
    }


def _o_row(provider, mid, quantity, family, obs_bin, gap_s, value, *, construct=None, column=None, kind=None):
    return {
        "provider": provider,
        "match_id": mid,
        "quantity": quantity,
        "family": family,
        "construct": construct,
        "column": column,
        "kind": kind,
        "obs_bin": obs_bin,
        "gap_s": gap_s,
        "value": value,
    }


#: A-09 / C.8.6: one representative construct per family for the planted occlusion fixture.
_FAMILY_FIXTURE_COLUMN = {
    "relative_phase": "coord_rp_resultant_length",
    "cross_correlation": "coord_xc_max_abs_r",
    "vector_coding": "coord_vc_pct_in_phase",
    "coherence": "coord_coh_band_mean",
    "spectral": "coord_median_freq_cpm",
    "cluster": "coord_rho_group_mean",
    "team_sync": "coord_team_sync_pearson_r",
    "rsi": "coord_rsi_mean_m",
}


def _complete_planted(providers=("gradientsports", "idsse"), n_matches=3):
    a_rows, b_rows, occ_rows = [], [], []
    for pi, prov in enumerate(providers):
        for m in range(n_matches):
            mid = f"{prov}-{m}"
            for run in range(2):
                for axis in ("x", "y"):
                    a_rows.append(_a_row(prov, mid, run, axis, 0.35 + 0.05 * pi))
            for period in (1, 2):
                for team in (1, 2):
                    for s in d.TEAM_SIGNALS:
                        b_rows.append(_b_row(prov, mid, period, team, None, "vc_epsilon", s, 0.05 + 0.01 * pi))
                        b_rows.append(_b_row(prov, mid, period, team, None, "acf_zero_s", s, 30.0 + pi))
                        b_rows.append(
                            _b_row(
                                prov, mid, period, team, None, "median_freq_cpm", s, 0.5 + 0.1 * pi, duration_s=400.0
                            )
                        )
                    b_rows.append(_b_row(prov, mid, period, team, None, "acf_zero_s", "cluster_amplitude", 25.0 + pi))
                    for pid in (10, 11):
                        b_rows.append(_b_row(prov, mid, period, team, pid, "acf_zero_s", "player_x", 20.0 + pi))
                        b_rows.append(_b_row(prov, mid, period, team, pid, "acf_zero_s", "player_y", 22.0 + pi))
            for g in d.POSSESSION_GAP_GRID:
                b_rows.append(
                    _b_row(
                        prov,
                        mid,
                        None,
                        None,
                        None,
                        "boundary_f1",
                        None,
                        0.9 if abs(g - 2.0) < 1e-9 else 0.6,
                        gap_s=float(g),
                    )
                )
            for fam in d.COORD_METHOD_FAMILIES:
                col = _FAMILY_FIXTURE_COLUMN[fam]
                ck = {"construct": col, "column": col, "kind": "linear"}
                occ_rows.append(_o_row(prov, mid, "occ_full", fam, None, None, 1.0 + 0.1 * pi + 0.01 * m, **ck))
                occ_rows.append(_o_row(prov, mid, "occ_err", fam, 0.6, None, 0.2, **ck))
                occ_rows.append(_o_row(prov, mid, "occ_err", fam, 0.8, None, 0.02, **ck))
            for g in d._DETECTION_GAP_GRID:
                occ_rows.append(_o_row(prov, mid, "bridge_rmse", None, None, float(g), 0.05 * g))
            occ_rows.append(_o_row(prov, mid, "noise_rms", None, None, None, 0.1))
            occ_rows.append(_o_row(prov, mid, "gk_rate", None, None, None, 0.2))
    return pd.DataFrame(a_rows), pd.DataFrame(b_rows), pd.DataFrame(occ_rows)


def test_reduce_writes_derivation_and_codegen_reproduces(tmp_path, monkeypatch):
    a_df, b_df, occ_df = _complete_planted()
    dest = tmp_path / "out"
    dest.mkdir()
    _write_share(dest, "a", a_df)
    _write_share(dest, "b", b_df, cutoffs={})
    _write_share(dest, "occlusion_cal", occ_df)  # the FOV-width histogram sub-pass (review m8: its timing is surfaced)
    _write_share(dest, "occlusion", occ_df, width_m=40.0, n_calibrated=71)
    committed = (Path(__file__).resolve().parents[2] / cg.GENERATED_PATH).read_bytes()
    monkeypatch.setattr(d, "THIN_PROVIDER_N_BOOT", 20)  # keep the report-only bootstrap fast in the test

    d._reduce(_args(dest), _CLEAN)

    derivation = json.loads((dest / "derivation.json").read_text(encoding="utf-8"))
    assert set(derivation["pooled"]) == set(cg.INTERIM_BASE)  # every base key derived
    # TF58-PLAN-03: the thin-provider block covers every scalar AND every per-signal / per-family map entry.
    thin = derivation["thin_providers"]
    assert {
        "butterworth_cutoff_hz",
        "possession_gap_s",
        "max_detection_gap_s",
        "vc_epsilon",
        "min_shift_s",
        "min_observed_fraction",
    } <= set(thin)
    assert set(thin["vc_epsilon"]) == set(d.TEAM_SIGNALS)  # one flag block per team signal
    assert set(thin["min_observed_fraction"]) == set(d.COORD_METHOD_FAMILIES)  # one per family
    assert all("flags" in thin["vc_epsilon"][s] for s in thin["vc_epsilon"])
    assert "input_contract" in derivation and derivation["run_commit"] == "0" * 40
    generated = (dest / d.GENERATED_ARTIFACT).read_text(encoding="utf-8")
    assert 'BASE_SOURCE: str = "derivation"' in generated
    # the committed generator reproduces the written file byte-for-byte from derivation.json.
    assert d.render_generated_params(derivation, None) == generated
    # M-5 artifact handoff: the package module is NOT touched (the DGX chain stays on the clean commit-1 tree).
    assert (Path(__file__).resolve().parents[2] / cg.GENERATED_PATH).read_bytes() == committed
    assert derivation["occlusion"]["width_m"] == 40.0
    assert set(derivation["stage_seconds"]) == {"a", "b", "occlusion-cal", "occlusion"}  # m8: occlusion-cal included
    assert derivation["occlusion"]["width_source"] == "calibrated"  # B m6: a non-empty histogram calibrated the width
    assert derivation["population"]["a"]["n_workers"] == 1
    assert derivation["corpus_visibility"] == "full"  # ADR-038: GS in the population, nothing public


def test_d1_reduce_hands_its_artifact_to_d2(tmp_path, monkeypatch):
    # M-5 artifact handoff, end to end on a clean tree: D2's layer a computes with exactly the params D1's reduce
    # wrote into its --out (what commit 2's module will hold) -- never the in-package module -- and records the
    # digest of that very file.
    import hashlib
    from types import SimpleNamespace

    import calibrate_coordination as d2
    from _fake_corpus import SpyLoader, make_loaded, make_ref

    from scripts._provenance import git_provenance
    from silly_kicks.coordination import CoordinationParams

    # review m8: exercise a REAL clean-tree provenance (the full git_provenance shape -- commit/platform/machine/
    # dirty_files -- forced clean), not a hand-built dict that could miss a key a consumer reads.
    prov = {**git_provenance(), "dirty": False, "tree_state": "clean"}
    a_df, b_df, occ_df = _complete_planted()
    d1_out = tmp_path / "d1"
    d1_out.mkdir()
    _write_share(d1_out, "a", a_df)
    _write_share(d1_out, "b", b_df, cutoffs={})
    _write_share(d1_out, "occlusion_cal", occ_df)  # m8: the FOV-width sub-pass share, so its timing is combined
    _write_share(d1_out, "occlusion", occ_df, width_m=40.0, n_calibrated=71)
    monkeypatch.setattr(d, "THIN_PROVIDER_N_BOOT", 20)
    d._reduce(_args(d1_out), prov)
    derivation_path = d1_out / "derivation.json"
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))

    provider = sorted(derivation["providers"])[0]
    loader = SpyLoader({(provider, "m1"): make_loaded(provider, "m1", frames=pd.DataFrame({"x": [1.0]}))})
    monkeypatch.setattr(d2, "corpus_source", lambda args: ([make_ref(provider, "m1")], loader))
    seen = []

    def spy(loaded, params, **kw):
        # Mirror match_tables(variants=...): the baseline layer-a pass emits ONE row per post-preparation variant,
        # each tagged with the `variant` column the per-variant share writer filters on (option C's per-variant
        # split of the stacked melt, D2-SPEC-05). A variant-less frame is the old stacked-pass shape.
        seen.append(params)
        keys = list(kw["variants"])
        return pd.DataFrame(
            {
                "provider": [provider] * len(keys),
                "match_id": ["m1"] * len(keys),
                "variant": keys,
                "table": ["pairs"] * len(keys),
            }
        )

    monkeypatch.setattr(d2, "match_tables", spy)
    d2_out = tmp_path / "d2"
    args = SimpleNamespace(
        out=str(d2_out),
        level=d2.BASELINE_LEVEL,
        derivation=str(derivation_path),
        providers=(provider,),
        match_ids_json=None,
        corpus_json=None,
    )
    d2._layer_a(args, prov)
    assert seen == [cg.params_from_artifacts(provider, derivation, None)]
    assert seen[0] != CoordinationParams.for_provider(provider)  # non-vacuity: the derived values differ
    name = d2.level_share_name(d2.BASELINE_LEVEL, "base")  # per-variant baseline layout (option C): the base share
    manifest = json.loads((d2_out / f"manifest_{name}.all.json").read_text(encoding="utf-8"))
    assert manifest["derivation_sha256"] == hashlib.sha256(derivation_path.read_bytes()).hexdigest()


def test_pass_b_takes_its_cutoffs_from_every_pass_a_worker(tmp_path, monkeypatch):
    # B-1: pass b's per-provider cutoff is corpus-wide -- it combines EVERY worker's pass-a share (and refuses when a
    # worker is missing), never just the slice the current worker happened to run.
    a_rows = [
        {
            "provider": "sportec",
            "match_id": m,
            "period_id": 1,
            "team_id": "A",
            "player_id": 1,
            "axis": "x",
            "run_index": 0,
            "cutoff_hz": c,
        }
        for m, c in (("m1", 0.5), ("m2", 0.7))
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"sportec": ["m1", "m2"]}), encoding="utf-8")
    _write_share(tmp_path, "a", pd.DataFrame(a_rows[:1]), tag="w0")
    with pytest.raises(SystemExit, match="miss 1 corpus key"):  # worker w1's pass a has not run yet
        d._pass_b(_args(tmp_path, corpus_json=str(corpus)), _CLEAN)
    _write_share(tmp_path, "a", pd.DataFrame(a_rows[1:]), tag="w1")
    seen = {}
    monkeypatch.setattr(d, "corpus_source", lambda args: ([], None))
    monkeypatch.setattr(d, "_provider_cutoffs_from_pass_a", lambda a_df: seen.setdefault("a", a_df) is not None and {})
    d._pass_b(_args(tmp_path, corpus_json=str(corpus)), _CLEAN)
    assert sorted(seen["a"]["match_id"]) == ["m1", "m2"]  # both workers' pass-a rows reached the cutoff rule


def test_derivation_records_whether_the_fov_width_was_calibrated_or_a_fallback():
    # B m6: an empty FOV histogram falls back to 40 m; derivation.json must SAY so (`width_source`), not leave the
    # fallback visible only in a worker manifest's n_calibrated.
    a_df, b_df, occ_df = _complete_planted()
    fb = d.build_derivation(a_df, b_df, occ_df, _CLEAN, n_boot=10, occlusion_width_m=40.0, occlusion_n_calibrated=0)
    assert fb["occlusion"]["width_source"] == "fallback_default" and fb["occlusion"]["n_calibrated"] == 0
    cal = d.build_derivation(a_df, b_df, occ_df, _CLEAN, n_boot=10, occlusion_width_m=38.5, occlusion_n_calibrated=71)
    assert cal["occlusion"]["width_source"] == "calibrated"


def test_pass_b_worker_with_a_provider_subset_combines_the_whole_corpus(tmp_path, monkeypatch):
    # B m5: pass a ran the whole corpus; a pass-b worker launched with a --providers SUBSET (run_params_token's premise,
    # how the MEDIA-PC no-flip was launched) must still combine EVERY provider's pass-a share -- not fail closed because
    # its own slice lists fewer providers than pass a produced.
    a_rows = [
        {
            "provider": p,
            "match_id": m,
            "period_id": 1,
            "team_id": "A",
            "player_id": 1,
            "axis": "x",
            "run_index": 0,
            "cutoff_hz": 0.6,
        }
        for p, m in (("sportec", "m1"), ("idsse", "d1"))
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"sportec": ["m1"], "idsse": ["d1"]}), encoding="utf-8")
    _write_share(tmp_path, "a", pd.DataFrame(a_rows))
    monkeypatch.setattr(d, "corpus_source", lambda args: ([], None))  # no pass-b compute; exercise only the combine
    # the sportec-slice worker must NOT raise on the full-corpus pass-a combine (it did, "add 1", before the fix)
    d._pass_b(_args(tmp_path, corpus_json=str(corpus), providers=("sportec",)), _CLEAN)


def test_reduce_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts._provenance.git_provenance",
        lambda: {"commit": "abc", "dirty": True, "tree_state": "dirty", "dirty_files": ["x.py"]},
    )
    monkeypatch.setattr("sys.argv", ["derive_coordination_params.py", "--pass", "reduce", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="DIRTY"):
        d.main()


# --------------------------------------------------------------------------- provider-neutral vs match-weighted (A3)
def test_pooled_cutoff_is_provider_neutral_not_match_or_run_weighted():
    # Three providers chosen so the three weightings give DISTINCT medians, off the fragile 0.5 boundary
    # (A3 / TF58-PLAN-02): A = 10 matches @0.30 (skews match-weighting low), B = 1 match @0.50, C = 1 match with
    # 100 runs @0.70 (skews run-weighting high). Provider-neutral (each provider 1/3) lands squarely on B = 0.50.
    a_rows = [_a_row("A", f"A-{m}", 0, "x", 0.30) for m in range(10)]
    a_rows += [_a_row("B", "B-0", 0, "x", 0.50)]
    a_rows += [_a_row("C", "C-0", r, "x", 0.70) for r in range(100)]
    a_df = pd.DataFrame(a_rows)
    empty_b = pd.DataFrame(columns=d._PASS_B_COLUMNS)
    empty_occ = pd.DataFrame(columns=d._OCC_COLUMNS)

    deriv = d.build_derivation(a_df, empty_b, empty_occ, _CLEAN, n_boot=50)
    pooled = deriv["pooled"]["butterworth_cutoff_hz"]

    provider_neutral = d.provider_cutoff(
        a_df["cutoff_hz"].to_numpy(float),
        d.unit_weights(a_df["provider"].to_numpy(), a_df["match_id"].to_numpy(), provider_neutral=True),
    )
    match_weighted = d.provider_cutoff(
        a_df["cutoff_hz"].to_numpy(float),
        d.unit_weights(a_df["provider"].to_numpy(), a_df["match_id"].to_numpy(), provider_neutral=False),
    )
    run_weighted = float(np.median(a_df["cutoff_hz"].to_numpy(float)))
    assert pooled == provider_neutral == 0.50
    assert match_weighted == 0.30  # non-vacuity: corpus/match-weighting is pulled low by A's 10 matches
    assert run_weighted == 0.70  # non-vacuity: run-weighting is dominated by C's 100 runs
    assert deriv["providers"]["C"]["butterworth_cutoff_hz"] == 0.70  # per-provider stays match-weighted within C


def test_per_provider_cutoff_is_match_weighted_not_run_weighted():
    # one provider, one heavy match (50 runs @0.7) + nine light matches (2 runs @0.3).
    rows = [_a_row("A", "heavy", r, "x", 0.7) for r in range(50)]
    rows += [_a_row("A", f"light-{m}", r, "x", 0.3) for m in range(9) for r in range(2)]
    a_df = pd.DataFrame(rows)
    deriv = d.build_derivation(
        a_df, pd.DataFrame(columns=d._PASS_B_COLUMNS), pd.DataFrame(columns=d._OCC_COLUMNS), _CLEAN, n_boot=50
    )
    assert deriv["providers"]["A"]["butterworth_cutoff_hz"] == 0.3  # 10 matches, the heavy one counts once
    assert float(np.median(a_df["cutoff_hz"].to_numpy(float))) == 0.7  # run-weighted would be 0.7 (rejected)


def test_thin_provider_flag_surfaces_on_a_map_entry():
    # TF58-PLAN-03: a provider too noisy on a per-signal MAP entry must be flagged, not silently shipped in the base.
    # vc_epsilon[centroid_x]: A, B well-sampled and stable (10 matches @0.05); C thin -- 2 matches, widely apart.
    b_rows = [
        _b_row(prov, f"{prov}-{m}", 1, 1, None, "vc_epsilon", "centroid_x", 0.05)
        for prov in ("A", "B")
        for m in range(10)
    ]
    b_rows += [
        _b_row("C", "C-0", 1, 1, None, "vc_epsilon", "centroid_x", 0.02),
        _b_row("C", "C-1", 1, 1, None, "vc_epsilon", "centroid_x", 0.30),
    ]
    b_df = pd.DataFrame(b_rows)
    a_df = pd.DataFrame(columns=d._PASS_A_COLUMNS)
    occ_df = pd.DataFrame(columns=d._OCC_COLUMNS)

    deriv = d.build_derivation(a_df, b_df, occ_df, _CLEAN, n_boot=200)
    block = deriv["thin_providers"]["vc_epsilon"]["centroid_x"]
    assert block["flags"]["C"] == "flagged"  # C's match-level SE exceeds the between-provider spread
    assert block["flags"]["A"] == "ok" and block["flags"]["B"] == "ok"  # stable providers are not flagged
    assert deriv["pooled"]["vc_epsilon"]["centroid_x"] == deriv["pooled"]["vc_epsilon"]["centroid_x"]  # report-only
    # non-vacuity: the flag NEVER altered the pooled value -- it equals the provider-neutral computation.
    pooled_neutral = d._weighted_by_signal(b_df, 0.5, provider_neutral=True, key="vc_epsilon", why={})["centroid_x"]
    assert deriv["pooled"]["vc_epsilon"]["centroid_x"] == pooled_neutral


# --------------------------------------------------------------------------- round 2: A-05, A-32, A-15
def _three_providers_without_skillcorner_occlusion():
    """GradientSports + IDSSE carry the occlusion leg (spec 8.2: it runs on the fully observed providers);
    SkillCorner -- the detection-aware provider the leg exists for -- has pass-a/b units and NO occlusion units."""
    a_df, b_df, occ_df = _complete_planted(providers=("gradientsports", "idsse", "skillcorner"))
    return a_df, b_df, occ_df[occ_df["provider"] != "skillcorner"].reset_index(drop=True)


def test_a_provider_without_occlusion_units_takes_the_occlusion_derived_values():
    # review A-05: the per-provider block of a provider with no occlusion units fell back to min_observed_fraction =
    # 1.0 for every family and the interim max_detection_gap_s -- overriding, for SkillCorner, exactly the values the
    # occlusion leg exists to derive. They are corpus-level (spec 8.2), so every provider block takes the pooled value.
    from scripts._coordination_params_codegen import params_from_artifacts

    a_df, b_df, occ_df = _three_providers_without_skillcorner_occlusion()
    deriv = d.build_derivation(a_df, b_df, occ_df, _CLEAN, n_boot=20)
    pooled = deriv["pooled"]
    assert any(v < 1.0 for v in pooled["min_observed_fraction"].values())  # non-vacuity: the leg derived something
    for provider in ("gradientsports", "idsse", "skillcorner"):
        block = deriv["providers"][provider]
        assert block["min_observed_fraction"] == pooled["min_observed_fraction"], provider
        assert block["max_detection_gap_s"] == pooled["max_detection_gap_s"], provider
    sc = params_from_artifacts("skillcorner", deriv, None)
    assert dict(sc.min_observed_fraction) == pooled["min_observed_fraction"]
    assert sc.max_detection_gap_s == pooled["max_detection_gap_s"]


def test_derivation_records_the_plan_required_diagnostics():
    # review A-32 / plan Task 20: beside `pooled`, each provider's total weight and match count, the corpus-
    # representative (match-weighted) value of every quantity, and every intermediate distribution summary
    a_df, b_df, occ_df = _three_providers_without_skillcorner_occlusion()
    deriv = d.build_derivation(a_df, b_df, occ_df, _CLEAN, n_boot=20)
    weights = deriv["provider_weights"]
    assert set(weights) == {"gradientsports", "idsse", "skillcorner"}
    for pass_name in ("a", "b", "occlusion"):  # each pass's provider-neutral weights sum to one
        assert sum(w["weight"][pass_name] for w in weights.values()) == pytest.approx(1.0), pass_name
    assert weights["skillcorner"]["weight"]["occlusion"] == 0.0  # no occlusion units -> no weight there
    assert weights["skillcorner"]["n_matches"] == {"a": 3, "b": 3, "occlusion": 0}
    assert set(deriv["representative"]) == set(deriv["pooled"])  # the match-weighted diagnostic of every quantity
    summaries = deriv["summaries"]
    for quantity in ("cutoff_hz", "median_freq_cpm", "vc_epsilon", "acf_zero_s", "boundary_f1", "bridge_rmse"):
        assert summaries[quantity]["n_units"] > 0, quantity
        assert {"p05", "p25", "p50", "p75", "p95"} <= set(summaries[quantity]), quantity


def test_every_fallback_records_its_reason():
    # review A-32: a value that falls back (no units, a rule with no qualifying bin/gap) says so in derivation.json --
    # never a silent default
    a_df, b_df, occ_df = _complete_planted()
    deriv = d.build_derivation(a_df, b_df, occ_df.iloc[0:0], _CLEAN, n_boot=20)
    reasons = deriv["fallbacks"]["pooled"]
    assert "max_detection_gap_s" in reasons and "no occlusion units" in reasons["max_detection_gap_s"]
    assert all(f"min_observed_fraction.{fam}" in reasons for fam in d.COORD_METHOD_FAMILIES)
    complete = d.build_derivation(*_complete_planted(), _CLEAN, n_boot=20)
    assert complete["fallbacks"]["pooled"] == {}  # the other side: nothing fell back


def test_min_observed_moves_clip_identically_in_d2_and_the_codegen():
    # review A-15: D2 clipped a min_observed_fraction level to [0, 1]; the codegen did not, so a +0.2 offset on a 0.9
    # base rendered 1.1, which CoordinationParams refuses -- one function now applies a move everywhere.
    import dataclasses

    from scripts._coordination_params_codegen import params_from_artifacts, render_generated_params
    from scripts.calibrate_coordination import apply_level
    from silly_kicks.coordination import CoordinationParams

    a_df, b_df, occ_df = _complete_planted()
    deriv = d.build_derivation(a_df, b_df, occ_df, _CLEAN, n_boot=20)
    for block in (deriv["pooled"], *deriv["providers"].values()):
        block["min_observed_fraction"] = dict.fromkeys(block["min_observed_fraction"], 0.9)
    moves = {"min_observed_fraction": {"multiplier": 1.0, "offset": 0.2}}
    calibration = {"confirmation": {"gate_cleared": True}, "moved_multipliers": moves}
    base = dataclasses.replace(
        CoordinationParams(), min_observed_fraction=dict(deriv["pooled"]["min_observed_fraction"])
    )
    assert set(apply_level(base, "min_observed_fraction", 0.2).min_observed_fraction.values()) == {1.0}
    params = params_from_artifacts("idsse", deriv, calibration)
    assert set(params.min_observed_fraction.values()) == {1.0}
    assert "1.1" not in render_generated_params(deriv, moves)
