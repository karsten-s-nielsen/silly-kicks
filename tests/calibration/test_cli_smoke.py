import math

import pytest

import silly_kicks
from scripts.calibrate_tracking_defaults import build_manifest, run_stage


def test_build_manifest_has_data_and_version_identity(frozen_xt):
    manifest = build_manifest(
        source="pining",
        seed=42,
        n_trials=2,
        match_ids={"skillcorner": ["m1", "m2"]},
        xt=frozen_xt,
        stage=1,
    )
    assert "silly_kicks_version" in manifest
    # Provenance must reflect the SOURCE version that actually ran, not the
    # installed-dist metadata (stale on an editable install bumped post-install,
    # which is how the maintainer dev-sweep runs).
    assert manifest["silly_kicks_version"] == silly_kicks.__version__
    assert "ruthless_version" in manifest
    assert "xgboost_version" in manifest
    assert manifest["match_ids"] == {"skillcorner": ["m1", "m2"]}
    assert manifest["xt_artifact"]["sha256"] == frozen_xt.sha256


def test_stage1_smoke_returns_result(stage1_fold, tmp_path):
    result, objective = run_stage(
        stage=1,
        fold=stage1_fold,
        n_trials=2,
        seed=42,
        store_path=str(tmp_path / "s1.db"),
        xt=None,
        carrier_params=None,
    )
    assert result.best is not None
    assert "carrier_accuracy" in result.best.metrics
    assert hasattr(objective, "diagnostics")  # surfaced into the manifest (M1/M8)


@pytest.mark.slow
def test_stage2_smoke_accepts_frozen_xt_artifact(stage2_fold, frozen_xt, tmp_path):
    # Regression guard for the CLI Stage-2 wiring that shipped "e2e-green" yet crashed: main() hands
    # run_stage the FrozenXt ARTIFACT (the same object build_manifest gets), and the Stage-2
    # objective must unwrap the inner ExpectedThreat itself. Passing the FrozenXt straight through
    # must NOT AttributeError in prepare() (compute_gk_influence calls xt.interpolator). The e2e
    # test exercises the real path but is network-gated; this is the CI-fast synthetic guard.
    result, objective = run_stage(
        stage=2,
        fold=stage2_fold,
        n_trials=2,
        seed=42,
        store_path=str(tmp_path / "s2.db"),
        xt=frozen_xt,  # FrozenXt artifact — exactly what main() passes (NOT frozen_xt.xt)
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
    )
    assert result.best is not None
    assert math.isfinite(result.best.metrics["brier"])
    assert hasattr(objective, "diagnostics")
