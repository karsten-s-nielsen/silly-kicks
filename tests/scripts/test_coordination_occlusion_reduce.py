"""A-09 / C.8.6: the per-construct occlusion reduce (owner ruling 2026-10-04 — consumed per family, MAX over the
family's constructs' qualifying shares; per-construct analysis + over-restriction diagnostic recorded).

The pure core: ``construct_qualifying_share`` (a construct's smallest qualifying observed-fraction bin + the
estimable verdict) and ``family_max_observed_fraction`` (the fail-closed family threshold + the over-restriction
diagnostic the owner reads to decide the per-construct-consumption follow-up).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import _coordination_thresholds as thr
from derive_coordination_params import construct_qualifying_share, family_max_observed_fraction

MIN = thr.OCCLUSION_MIN_MATCHES_PER_BIN


def _powered(bins):
    return {b: MIN for b in bins}


def test_estimable_clean_crossing_returns_the_smallest_qualifying_bin():
    # between_sd 0.4 -> threshold 0.2; error drops below at 0.6 and stays below (clean suffix).
    err = {0.1: 0.9, 0.3: 0.5, 0.6: 0.18, 0.8: 0.1, 1.0: 0.05}
    res = construct_qualifying_share(err, _powered(err), 0.4, min_matches=MIN)
    assert res["estimable"] is True
    assert res["share"] == 0.6
    assert res["reason"] is None


def test_never_crossing_is_one_and_a_finding():
    err = {0.1: 0.9, 0.5: 0.8, 1.0: 0.7}  # never <= 0.2
    res = construct_qualifying_share(err, _powered(err), 0.4, min_matches=MIN)
    assert res["share"] == 1.0
    assert res["estimable"] is False
    assert res["reason"] == "no_crossing"


def test_underpowered_bin_is_not_estimable():
    err = {0.1: 0.9, 0.6: 0.18, 1.0: 0.05}
    n = {0.1: MIN, 0.6: MIN - 1, 1.0: MIN}  # one bin below the floor
    res = construct_qualifying_share(err, n, 0.4, min_matches=MIN)
    assert res["estimable"] is False
    assert res["reason"] == "underpowered_bins"
    assert res["share"] == 1.0


def test_non_unique_crossing_is_not_estimable():
    # dips below the bar at 0.6 then pops back ABOVE at 0.8 -> not a clean single crossing.
    err = {0.1: 0.9, 0.6: 0.15, 0.8: 0.30, 1.0: 0.05}
    res = construct_qualifying_share(err, _powered(err), 0.4, min_matches=MIN)
    assert res["estimable"] is False
    assert res["reason"] == "non_unique_crossing"
    assert res["share"] == 1.0


def test_family_threshold_is_the_max_over_constructs_and_records_over_restriction():
    per_construct = {
        "cA": {"share": 0.6, "estimable": True, "reason": None},
        "cB": {"share": 0.8, "estimable": True, "reason": None},
        "cC": {"share": 0.3, "estimable": True, "reason": None},
    }
    fam = family_max_observed_fraction(per_construct)
    assert fam["threshold"] == 0.8  # fail-closed MAX
    assert fam["binding_construct"] == "cB"
    assert fam["finding"] is False
    # the over-restriction each non-binding construct suffers vs its own share
    assert fam["over_restriction"]["cA"] == pytest.approx(0.2)
    assert fam["over_restriction"]["cC"] == pytest.approx(0.5)
    assert "cB" not in fam["over_restriction"]  # the binding construct is not over-restricted


def test_per_construct_detail_records_ci_per_bin():
    # B-R3-02 (C.5 "curve + n/CI per bin"): occlusion_per_construct_detail records a per-bin CI, not just n.
    import pandas as pd
    from derive_coordination_params import occlusion_per_construct_detail

    rows = []
    for m in range(6):
        rows.append(
            {
                "provider": "gradientsports",
                "match_id": f"g{m}",
                "quantity": "occ_full",
                "family": "spectral",
                "construct": "column=coord_median_freq_cpm|signal=spread",
                "column": "coord_median_freq_cpm",
                "kind": "linear",
                "obs_bin": None,
                "gap_s": None,
                "value": 1.0 + 0.05 * m,
            }
        )
        for ob, e in ((0.6, 0.2 + 0.01 * m), (0.8, 0.02 + 0.005 * m)):
            rows.append(
                {
                    "provider": "gradientsports",
                    "match_id": f"g{m}",
                    "quantity": "occ_err",
                    "family": "spectral",
                    "construct": "column=coord_median_freq_cpm|signal=spread",
                    "column": "coord_median_freq_cpm",
                    "kind": "linear",
                    "obs_bin": ob,
                    "gap_s": None,
                    "value": e,
                }
            )
    detail = occlusion_per_construct_detail(pd.DataFrame(rows), provider_neutral=True)
    rec = detail["spectral"]["constructs"]["column=coord_median_freq_cpm|signal=spread"]
    assert set(rec["ci_by_bin"]) == {0.6, 0.8}
    for _bin, ci in rec["ci_by_bin"].items():
        assert len(ci) == 2 and ci[0] <= ci[1]


def test_family_driven_to_one_by_a_single_nonestimable_construct_is_a_finding():
    per_construct = {
        "cA": {"share": 0.6, "estimable": True, "reason": None},
        "cBad": {"share": 1.0, "estimable": False, "reason": "no_crossing"},
    }
    fam = family_max_observed_fraction(per_construct)
    assert fam["threshold"] == 1.0
    assert fam["binding_construct"] == "cBad"
    assert fam["finding"] is True
