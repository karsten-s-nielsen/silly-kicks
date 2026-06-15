"""TF-48 owner-gated acceptance e2e: GS WC2022 derived crossings vs StatsBomb
``end_location`` (spec section 10; ADR-030). Heavy; runs on the DGX against pining.

Floors are PRE-REGISTERED in ADR-030 after the pilot (spec section 10.4) and then
asserted here on the held-out matches. Until registration this test SKIPS LOUDLY
(explicit reason) -- never a silent pass.
"""

import os

import pytest

pytest.importorskip("statsbombpy")

pytestmark = pytest.mark.e2e

# Pre-registered accept floors (ADR-030, registered 2026-06-11 at pilot-v5 review, owner-approved
# -- spec section 10.4). Regression TRIPWIRES with ~2x headroom over the pilot-v5 measured values
# (dy 1.21 m / dz 0.52 m / agreement 0.61 / coverage 0.795); a floor failure is a hard STOP.
_FLOOR_GOALS_DY_MEDIAN_M: float | None = 2.5
_FLOOR_GOALS_DZ_MEDIAN_M: float | None = 1.25
_FLOOR_ON_TARGET_AGREEMENT: float | None = 0.45
_FLOOR_COVERAGE: float | None = 0.60


def _require_owner_token():
    if not os.environ.get("PINING_FOR_THE_DATA_TOKEN"):
        pytest.skip("owner pining token not configured (GS corpus is owner-tier)")


def test_pilot_protocol_runs(tmp_path):
    """The harness completes the full protocol on the pilot subset (smoke +
    handedness settlement + stratified report). Floor-free by design."""
    _require_owner_token()
    from scripts.validate_shot_goalmouth_sb import run

    result = run("pilot", str(tmp_path / "pilot.json"), None, None)
    assert result["n_matched"] > 0
    assert "goals" in result, "no matched goals -- protocol cannot settle handedness"
    assert result["handedness_sign"] in (1, -1)


def test_holdout_floors(tmp_path):
    """PRE-REGISTERED floors on the held-out matches (goals only -- spec H4)."""
    _require_owner_token()
    if _FLOOR_GOALS_DY_MEDIAN_M is None:
        pytest.skip("floors not yet pre-registered in ADR-030 -- pilot pending (spec 10.4)")
    from scripts.validate_shot_goalmouth_sb import run

    result = run("holdout", str(tmp_path / "holdout.json"), None, None)
    goals = result["goals"]
    assert goals["dy_median_m"] <= _FLOOR_GOALS_DY_MEDIAN_M
    assert goals["dz_median_m"] <= _FLOOR_GOALS_DZ_MEDIAN_M
    assert result["on_target_agreement"] >= _FLOOR_ON_TARGET_AGREEMENT
    assert result["coverage_on_target"] >= _FLOOR_COVERAGE
