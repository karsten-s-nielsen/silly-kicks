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
