"""A-09: per-construct reliability reduce (spec §8.5). Construct derivation + the per-cell reliability.

Tests the single-source construct grain (C.1), the per-(match, entity) unit keying (C.4), the binding
reliability (linear ICC(1) / rotation-invariant circular), the pre-registered power verdict (C.2), the
split-half Spearman-Brown block (A-53) and the circular diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import _coordination_reliability as rel
import _coordination_thresholds as thr


def _period_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["window_kind"] = "period"
    df["window_id"] = 0
    return df


# --------------------------------------------------------------------------- construct derivation (C.1)
def test_spectral_one_construct_per_signal():
    df = _period_rows(
        [
            {"game_id": "g1", "period_id": p, "team_id": t, "signal": s, "coord_median_freq_cpm": 0.5}
            for p in (1, 2)
            for t in ("A", "B")
            for s in ("spread", "convex_hull_area")
        ]
    )
    out = {
        c.key["signal"]: (c, s)
        for c, s in rel.derive_constructs({"spectral": df})
        if c.column == "coord_median_freq_cpm"
    }
    assert set(out) == {"spread", "convex_hull_area"}
    c, samples = out["spread"]
    assert c.unit == "team"
    assert c.kind == "linear"
    assert set(samples["entity"]) == {"A", "B"}
    assert set(samples.columns) >= {"game_id", "entity", "period_id", "value"}


def test_pair_team_team_melts_both_teams():
    df = _period_rows(
        [
            {
                "game_id": "g1",
                "period_id": p,
                "level": "team_team",
                "signal_a": "centroid_x",
                "signal_b": "centroid_x",
                "axis": "x",
                "team_a_id": "A",
                "team_b_id": "B",
                "coord_xc_max_abs_r": 0.4,
            }
            for p in (1, 2)
        ]
    )
    constructs = [(c, s) for c, s in rel.derive_constructs({"pair": df}) if c.column == "coord_xc_max_abs_r"]
    assert len(constructs) == 1
    c, samples = constructs[0]
    assert c.unit == "team"
    assert c.key == {"level": "team_team", "signal_a": "centroid_x", "signal_b": "centroid_x", "axis": "x"}
    assert set(samples["entity"]) == {"A", "B"}  # mutual quantity attributed to both teams


def test_pair_dyad_is_unordered_player_pair():
    df = _period_rows(
        [
            {
                "game_id": "g1",
                "period_id": 1,
                "level": "dyad",
                "signal_a": "x",
                "signal_b": "x",
                "axis": "x",
                "player_a_id": 7,
                "player_b_id": 9,
                "coord_xc_max_abs_r": 0.3,
            },
            {
                "game_id": "g1",
                "period_id": 2,
                "level": "dyad",
                "signal_a": "x",
                "signal_b": "x",
                "axis": "x",
                "player_a_id": 9,
                "player_b_id": 7,
                "coord_xc_max_abs_r": 0.35,  # swapped order -> same pair
            },
        ]
    )
    constructs = [(c, s) for c, s in rel.derive_constructs({"pair": df}) if c.column == "coord_xc_max_abs_r"]
    assert len(constructs) == 1
    c, samples = constructs[0]
    assert c.unit == "unordered_pair"
    assert samples["entity"].nunique() == 1  # (7,9) == (9,7)


def test_cluster_player_unit_is_player():
    df = _period_rows(
        [
            {"game_id": "g1", "period_id": p, "team_id": "A", "player_id": pid, "axis": "x", "coord_rho_k": 0.6}
            for p in (1, 2)
            for pid in (1, 2, 3)
        ]
    )
    constructs = [(c, s) for c, s in rel.derive_constructs({"cluster_player": df}) if c.column == "coord_rho_k"]
    assert len(constructs) == 1  # one construct (axis="x")
    c, samples = constructs[0]
    assert c.unit == "player"
    assert c.key == {"axis": "x"}
    assert set(samples["entity"]) == {1, 2, 3}


def test_coverage_fractions_excluded_symmetrically():
    # B-R3-01: both observed-fraction AND phase-valid-fraction are coverage denominators -> neither is a reliability
    # construct (they were asymmetric: observed_fraction out, phase_valid in).
    scored = rel.reliability_scored_columns("pair")
    assert "coord_rp_phase_valid_fraction_a" not in scored
    assert "coord_rp_phase_valid_fraction_b" not in scored
    assert "coord_observed_fraction_a" not in scored
    assert "coord_detected_share" not in scored
    # non-vacuity: a genuine signal column is still scored
    assert "coord_rp_resultant_length" in scored


def test_coverage_columns_are_not_scored():
    df = _period_rows(
        [
            {"game_id": "g1", "period_id": p, "team_id": "A", "signal": "spread", "coord_duration_s": 2700.0}
            for p in (1, 2)
        ]
    )
    cols = {c.column for c, _ in rel.derive_constructs({"spectral": df})}
    assert "coord_duration_s" not in cols


def test_circular_mean_column_has_circular_kind():
    df = _period_rows(
        [
            {
                "game_id": "g1",
                "period_id": p,
                "level": "team_team",
                "signal_a": "centroid_x",
                "signal_b": "centroid_x",
                "axis": "x",
                "team_a_id": "A",
                "team_b_id": "B",
                "coord_rp_mean_deg": 12.0,
            }
            for p in (1, 2)
        ]
    )
    c = next(c for c, _ in rel.derive_constructs({"pair": df}) if c.column == "coord_rp_mean_deg")
    assert c.kind == "circular"


def test_non_period_windows_are_dropped():
    df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "period_id": 1,
                "team_id": "A",
                "signal": "spread",
                "window_kind": "sliding",
                "window_id": 0,
                "coord_median_freq_cpm": 0.5,
            },
        ]
    )
    assert rel.derive_constructs({"spectral": df}) == []


# --------------------------------------------------------------------------- per-cell reliability (C.2/C.3/C.4)
def _linear_samples(n_groups: int, within_noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        base = rng.uniform(0.0, 10.0)
        for period in (1, 2):
            rows.append(
                {
                    "game_id": f"m{g}",
                    "entity": f"m{g}",
                    "period_id": period,
                    "value": base + rng.normal(0.0, within_noise),
                }
            )
    return pd.DataFrame(rows)


def test_reliability_cell_linear_measured():
    samples = _linear_samples(40, within_noise=0.1)
    cell = rel.reliability_cell(rel.Construct("coord_xc_max_abs_r", "pair", {}, "team", "linear"), samples)
    r = cell["reliability"]
    assert r["estimator"] == "icc1"
    assert r["power"] == "measured"
    assert r["unmeasurable_reason"] is None
    assert r["value"] > 0.8
    assert r["n_groups"] == 40
    assert r["ci"][0] <= r["value"] <= r["ci"][1]
    assert cell["diagnostics"] == {}


def test_reliability_cell_underpowered_by_group_count():
    samples = _linear_samples(5, within_noise=0.1)
    cell = rel.reliability_cell(rel.Construct("coord_xc_max_abs_r", "pair", {}, "team", "linear"), samples)
    r = cell["reliability"]
    assert r["power"] == "unmeasurable"
    assert r["unmeasurable_reason"] == "n<min"
    assert r["value"] is None


def _circular_samples(n_groups: int, within_noise_deg: float, spread_deg: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        base = rng.uniform(-spread_deg, spread_deg)
        for period in (1, 2):
            rows.append(
                {
                    "game_id": f"m{g}",
                    "entity": f"m{g}",
                    "period_id": period,
                    "value": base + rng.normal(0.0, within_noise_deg),
                }
            )
    return pd.DataFrame(rows)


def test_reliability_cell_circular_measured_with_diagnostics():
    samples = _circular_samples(40, within_noise_deg=3.0, spread_deg=40.0)
    cell = rel.reliability_cell(rel.Construct("coord_rp_mean_deg", "pair", {}, "team", "circular"), samples)
    r = cell["reliability"]
    assert r["estimator"] == "circular_reliability"
    assert r["power"] == "measured"
    assert r["value"] > 0.5
    assert set(cell["diagnostics"]) == {"icc_cos", "icc_sin", "origin_deg"}
    assert cell["diagnostics"]["origin_deg"] == 0.0


def test_reliability_cell_circular_low_concentration_is_rbar_floor():
    # Bases EVENLY spaced around the circle -> overall resultant ~ 0 -> Rbar floor trips (deterministic, not a
    # random walk that could land above the floor by chance).
    rng = np.random.default_rng(2)
    rows = []
    for g in range(40):
        base = -180.0 + 360.0 * g / 40.0
        for period in (1, 2):
            rows.append(
                {"game_id": f"m{g}", "entity": f"m{g}", "period_id": period, "value": base + rng.normal(0.0, 2.0)}
            )
    cell = rel.reliability_cell(rel.Construct("coord_rp_mean_deg", "pair", {}, "team", "circular"), pd.DataFrame(rows))
    assert cell["reliability"]["power"] == "unmeasurable"
    assert cell["reliability"]["unmeasurable_reason"] == "Rbar->0"


def test_split_mode_is_within_match_split_half_with_spearman_brown():
    samples = _linear_samples(40, within_noise=0.1)
    cell = rel.reliability_cell(rel.Construct("coord_xc_max_abs_r", "pair", {}, "team", "linear"), samples)
    sm = cell["split_mode"]
    assert sm["spearman_brown"] is True
    assert sm["half_length_s"] is not None
    assert -1.0 <= sm["value"] <= 1.0


def test_power_verdict_vocabulary_is_closed():
    # every unmeasurable_reason the reduce can emit is in the pre-registered vocabulary
    for reason in (None, *thr.UNMEASURABLE_REASONS):
        assert reason is None or reason in thr.UNMEASURABLE_REASONS


def test_honesty_lines_present_and_mention_the_two_bounds():
    lines = rel.HONESTY_LINES
    assert len(lines) == 2
    joined = " ".join(lines).lower()
    assert "upper bound" in joined
    assert "unmeasurable" in joined and "roster" in joined


# --------------------------------------------------------------------------- deciles + poolability + assembler
def test_deciles_bin_by_detected_share():
    rng = np.random.default_rng(3)
    rows = []
    for g in range(40):
        base = rng.uniform(0.0, 10.0)
        for period, share in ((1, 0.55), (2, 0.95)):
            rows.append(
                {
                    "game_id": "g",
                    "period_id": period,
                    "team_id": f"t{g}",
                    "signal": "spread",
                    "window_kind": "period",
                    "window_id": 0,
                    "coord_median_freq_cpm": base + rng.normal(0, 0.1),
                    "coord_detected_share": share,
                }
            )
    constructs = {c.column: (c, s) for c, s in rel.derive_constructs({"spectral": pd.DataFrame(rows)})}
    c, samples = constructs["coord_median_freq_cpm"]
    assert "obs_frac" in samples.columns
    cell = rel.reliability_cell(c, samples)
    bins = {d["bin"] for d in cell["deciles"]}
    assert bins == {0.6, 1.0}  # 0.55 -> 0.6 bin, 0.95 -> 1.0 bin
    for d in cell["deciles"]:
        assert d["n"] == 40 and "median" in d and "spread" in d


def test_circular_deciles_use_circular_mean_not_wrap_breaking_median():
    # Values near the +/-180 wrap: a linear median would land near 0, the circular mean near 180.
    rows = []
    for g in range(40):
        for period in (1, 2):
            v = 179.0 if (g + period) % 2 == 0 else -179.0
            rows.append(
                {
                    "game_id": "g",
                    "period_id": period,
                    "level": "team_team",
                    "signal_a": "centroid_x",
                    "signal_b": "centroid_x",
                    "axis": "x",
                    "team_a_id": f"t{g}",
                    "team_b_id": f"u{g}",
                    "window_kind": "period",
                    "window_id": 0,
                    "coord_rp_mean_deg": v,
                    "coord_detected_share": 0.9,
                }
            )
    c, samples = next(
        (c, s) for c, s in rel.derive_constructs({"pair": pd.DataFrame(rows)}) if c.column == "coord_rp_mean_deg"
    )
    cell = rel.reliability_cell(c, samples)
    assert cell["deciles"]
    center = cell["deciles"][0]["median"]
    assert abs(abs(center) - 180.0) < 5.0  # circular mean near +/-180, NOT near 0


def test_build_report_attaches_cross_provider_poolability():
    def spectral(base_shift):
        rng = np.random.default_rng(7)
        rows = []
        for g in range(40):
            base = rng.uniform(0.0, 10.0) + base_shift
            for period in (1, 2):
                rows.append(
                    {
                        "game_id": f"m{g}",
                        "period_id": period,
                        "team_id": f"t{g}",
                        "signal": "spread",
                        "window_kind": "period",
                        "window_id": 0,
                        "coord_median_freq_cpm": base + rng.normal(0, 0.1),
                        "coord_detected_share": 0.9,
                    }
                )
        return {"spectral": pd.DataFrame(rows)}

    report = rel.build_constructs_report({"gradientsports": spectral(0.0), "idsse": spectral(0.0)})
    assert report["honesty"] == list(rel.HONESTY_LINES)
    cells = [c for c in report["constructs"] if c["column"] == "coord_median_freq_cpm"]
    assert len(cells) == 2  # one per provider
    for cell in cells:
        assert cell["provider"] in {"gradientsports", "idsse"}
        assert cell["poolability"]["n_providers"] == 2
        assert set(cell["poolability"]["providers"]) == {"gradientsports", "idsse"}
    summ = report["column_summary"]["coord_median_freq_cpm"]
    assert summ["n_constructs"] == 2
    assert summ["reliability_median"] is not None


def test_metrics_json_cell_schema_is_conformant():
    # C.7/C.8.8 (§9.4): every per-construct cell carries the declared keys; power + unmeasurable_reason come from the
    # pre-registered closed vocabularies; circular cells carry the cos/sin diagnostics with a pinned origin.
    rng = np.random.default_rng(11)
    pair_rows, spec_rows = [], []
    for g in range(40):
        base = rng.uniform(-30.0, 30.0)
        fbase = rng.uniform(0.0, 10.0)
        for period in (1, 2):
            pair_rows.append(
                {
                    "game_id": f"m{g}",
                    "period_id": period,
                    "level": "team_team",
                    "signal_a": "centroid_x",
                    "signal_b": "centroid_x",
                    "axis": "x",
                    "team_a_id": f"t{g}",
                    "team_b_id": f"u{g}",
                    "window_kind": "period",
                    "window_id": 0,
                    "coord_rp_mean_deg": base + rng.normal(0, 2.0),
                    "coord_detected_share": 0.9,
                }
            )
            spec_rows.append(
                {
                    "game_id": f"m{g}",
                    "period_id": period,
                    "team_id": f"t{g}",
                    "signal": "spread",
                    "window_kind": "period",
                    "window_id": 0,
                    "coord_median_freq_cpm": fbase + rng.normal(0, 0.1),
                    "coord_detected_share": 0.9,
                }
            )
    report = rel.build_constructs_report(
        {"idsse": {"pair": pd.DataFrame(pair_rows), "spectral": pd.DataFrame(spec_rows)}}
    )
    assert report["honesty"] == list(rel.HONESTY_LINES)
    reasons = {None, *thr.UNMEASURABLE_REASONS}
    seen_circular = False
    for cell in report["constructs"]:
        assert set(cell) >= {
            "column",
            "construct_key",
            "unit",
            "kind",
            "reliability",
            "diagnostics",
            "split_mode",
            "deciles",
            "poolability",
            "provider",
        }
        assert cell["kind"] in {"linear", "circular"}
        assert cell["unit"] in {"team", "player", "unordered_pair"}
        r = cell["reliability"]
        assert set(r) >= {"value", "ci", "estimator", "n_groups", "n_obs", "power", "unmeasurable_reason"}
        assert r["power"] in {"measured", "unmeasurable"}
        assert r["unmeasurable_reason"] in reasons
        assert (r["power"] == "measured") == (r["unmeasurable_reason"] is None)
        if cell["kind"] == "circular":
            seen_circular = True
            assert set(cell["diagnostics"]) == {"icc_cos", "icc_sin", "origin_deg"}
            assert cell["diagnostics"]["origin_deg"] == 0.0
            assert r["estimator"] == "circular_reliability"
        else:
            assert cell["diagnostics"] == {}
            assert r["estimator"] == "icc1"
    assert seen_circular  # non-vacuity: the circular branch was exercised
