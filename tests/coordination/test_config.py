"""TF-58 Task 14: CoordinationParams defaults, validation, provider merge, A3 single-sourcing."""

from __future__ import annotations

import ast
import dataclasses
import pathlib
from typing import cast

import pytest

import silly_kicks.coordination._provider_params_generated as gen
from silly_kicks.coordination._columns import COORD_METHOD_FAMILIES, TEAM_SIGNALS
from silly_kicks.coordination._config import CoordinationParams

_TIER_B = {
    "butterworth_cutoff_hz",
    "max_detection_gap_s",
    "min_observed_fraction",
    "vc_epsilon",
    "min_shift_s",
    "band_low_cpm",
    "band_high_cpm",
    "welch_segment_s",
    "possession_gap_s",
}


def test_tier_a_and_convention_defaults():
    p = CoordinationParams()
    assert p.xcorr_max_lag_s == 15.0
    assert p.n_phases == 3
    assert p.near_in_phase_deg == 30.0
    assert p.max_stoppage_s == 25.0
    assert p.sampen_m == 1
    assert p.sampen_r_sd == 0.2
    assert p.butterworth_order == 3
    assert p.analysis_hz == 10.0
    assert p.min_players == 6
    assert p.n_surrogates == 199
    assert p.iaaft_max_iter == 100
    assert p.coverage_warn_fraction == 0.25
    assert p.surrogate_method == "time_shift"
    assert p.surrogate_seed == 0
    assert dict(p.include_goalkeeper) == {"team_signals": False, "dyad": False, "cluster": True}


def test_tier_b_defaults_come_from_generated_base():
    p = CoordinationParams()
    assert p.butterworth_cutoff_hz == gen.BASE_COORDINATION_PARAMS["butterworth_cutoff_hz"]
    assert p.max_detection_gap_s == gen.BASE_COORDINATION_PARAMS["max_detection_gap_s"]
    assert p.band_low_cpm == gen.BASE_COORDINATION_PARAMS["band_low_cpm"]
    assert p.band_high_cpm == gen.BASE_COORDINATION_PARAMS["band_high_cpm"]
    assert p.welch_segment_s == gen.BASE_COORDINATION_PARAMS["welch_segment_s"]
    assert p.possession_gap_s == gen.BASE_COORDINATION_PARAMS["possession_gap_s"]
    assert dict(p.min_observed_fraction) == gen.BASE_COORDINATION_PARAMS["min_observed_fraction"]
    assert dict(p.vc_epsilon) == gen.BASE_COORDINATION_PARAMS["vc_epsilon"]
    assert dict(p.min_shift_s) == gen.BASE_COORDINATION_PARAMS["min_shift_s"]


def test_no_tier_b_numeric_literal_in_config_source():
    src = (pathlib.Path(__file__).resolve().parents[2] / "silly_kicks/coordination/_config.py").read_text("utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CoordinationParams")
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in _TIER_B:
            numeric = [
                d
                for d in ast.walk(node.value)  # type: ignore[arg-type]
                if isinstance(d, ast.Constant) and isinstance(d.value, (int, float)) and not isinstance(d.value, bool)
            ]
            assert not numeric, f"{node.target.id} has a numeric literal default; must read from the generated base"


def test_base_from_derivation_in_commit_2():
    # Commit 2 regenerates the base from D1's pooled derivation: BASE_SOURCE flips interim -> derivation.
    # The KEY STRUCTURE is invariant across the flip; the VALUES are the DGX-derived pooled numbers (validated
    # by the authoritative run + CoordinationParams.__post_init__), so they are checked structurally here, not
    # pinned to brittle run-specific literals.
    assert gen.BASE_SOURCE == "derivation"
    base = gen.BASE_COORDINATION_PARAMS
    mof = cast("dict[str, float]", base["min_observed_fraction"])
    vce = cast("dict[str, float]", base["vc_epsilon"])
    mss = cast("dict[str, float]", base["min_shift_s"])
    assert set(mof) == set(COORD_METHOD_FAMILIES)
    assert set(vce) == set(TEAM_SIGNALS)
    assert set(mss) == set(TEAM_SIGNALS) | {"player_x", "player_y", "cluster_amplitude"}
    assert cast("float", base["band_low_cpm"]) < cast("float", base["band_high_cpm"])  # ordered bands survive deriv
    assert cast("float", base["butterworth_cutoff_hz"]) > 0 and cast("float", base["welch_segment_s"]) > 0
    CoordinationParams()  # the derived base constructs a valid params (every __post_init__ bound holds)


def test_generated_map_populated_in_commit_2():
    # Commit 2 fills the per-provider map from the derivation (empty only at commit 1); every provider's
    # partial override merges into a valid params.
    providers = gen.PROVIDER_COORDINATION_PARAMS
    assert set(providers) == {"gradientsports", "idsse", "skillcorner"}
    for prov in providers:
        CoordinationParams.for_provider(prov)


def test_maps_complete_and_frozen():
    p = CoordinationParams()
    with pytest.raises(TypeError):
        p.vc_epsilon["centroid_x"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="vc_epsilon keys"):
        CoordinationParams(vc_epsilon={"centroid_x": 0.0})  # incomplete key set


@pytest.mark.parametrize(
    ("field_name", "bad", "good"),
    [
        ("n_phases", 1, 2),  # >= 2: one phase only duplicates the window's own row (A-22, owner ruling 2026-10-04)
        ("xcorr_max_lag_s", 0.0, 0.1),
        ("near_in_phase_deg", 0.0, 0.1),
        ("near_in_phase_deg", 180.0, 179.0),
        ("max_stoppage_s", 0.0, 1.0),  # good > default max_detection_gap_s (0.5): the A-28 guard
        ("sampen_m", 0, 1),
        ("sampen_r_sd", 0.0, 0.1),
        ("butterworth_order", 0, 1),
        ("analysis_hz", 0.0, 0.1),
        ("butterworth_cutoff_hz", 0.0, 0.1),
        ("min_players", 1, 2),
        ("n_surrogates", -1, 0),
        ("iaaft_max_iter", 0, 1),
        ("coverage_warn_fraction", 1.1, 1.0),
        ("max_detection_gap_s", -0.1, 0.0),
        ("welch_segment_s", 0.0, 1.0),
        ("possession_gap_s", -0.1, 0.0),
        ("surrogate_seed", -1, 0),
        # a misspelt method must not silently run IAAFT (spec 7.14: __post_init__ rejects the impossible)
        ("surrogate_method", "iaaf", "iaaft"),
        ("surrogate_method", "timeshift", "time_shift"),
    ],
)
def test_every_rejection_both_sides(field_name, bad, good):
    with pytest.raises(ValueError):
        dataclasses.replace(CoordinationParams(), **{field_name: bad})
    dataclasses.replace(CoordinationParams(), **{field_name: good})  # must not raise


def test_band_ordering_rejection():
    with pytest.raises(ValueError, match="band_low_cpm < band_high_cpm"):
        dataclasses.replace(CoordinationParams(), band_low_cpm=0.9, band_high_cpm=0.83)


def test_max_detection_gap_must_be_below_max_stoppage():
    # A-28: a detection-gap tolerance at or above the stoppage threshold would let a player run bridge a long
    # stoppage the team segments split on.
    with pytest.raises(ValueError, match=r"max_detection_gap_s.*< max_stoppage_s"):
        dataclasses.replace(CoordinationParams(), max_detection_gap_s=25.0, max_stoppage_s=25.0)
    with pytest.raises(ValueError, match=r"max_detection_gap_s.*< max_stoppage_s"):
        dataclasses.replace(CoordinationParams(), max_detection_gap_s=30.0)
    dataclasses.replace(CoordinationParams(), max_detection_gap_s=2.0)  # 2 < 25: must not raise


def test_map_value_rejections_both_sides():
    base_mof = dict(CoordinationParams().min_observed_fraction)
    bad = {**base_mof, "spectral": 1.5}
    with pytest.raises(ValueError, match="min_observed_fraction"):
        dataclasses.replace(CoordinationParams(), min_observed_fraction=bad)
    ok = {**base_mof, "spectral": 1.0}
    dataclasses.replace(CoordinationParams(), min_observed_fraction=ok)  # 1.0 accepted


def test_for_provider_merges_keywise(monkeypatch):
    partial = {"vc_epsilon": {"centroid_x": 0.7}}  # only one key overridden
    monkeypatch.setattr(gen, "PROVIDER_COORDINATION_PARAMS", {"acme": partial})
    p = CoordinationParams.for_provider("acme")
    assert p.vc_epsilon["centroid_x"] == 0.7
    # untouched keys keep the base value (read from the generated base, robust to the commit-2 derivation flip)
    base_vce = cast("dict[str, float]", gen.BASE_COORDINATION_PARAMS["vc_epsilon"])
    assert p.vc_epsilon["centroid_y"] == base_vce["centroid_y"]


def test_for_provider_unlisted_is_base():
    assert CoordinationParams.for_provider("nope") == CoordinationParams()


def test_default_flag_idiom():
    assert CoordinationParams.default().is_default() is True
    assert CoordinationParams.default(force_universal=True).is_default() is False
    assert CoordinationParams().is_default() is False


def test_hashable_and_equal_hash_for_equal_params():
    assert hash(CoordinationParams()) == hash(CoordinationParams())
    assert CoordinationParams() == CoordinationParams()
    assert hash(CoordinationParams.default()) == hash(CoordinationParams())  # the flag is compare=False
