import numpy as np
import pandas as pd
import pytest

from scripts.hf_jobs import headtohead as h2h
from scripts.hf_jobs import registry as reg


def test_build_eval_accepts_feature_set_kw():
    # fail-loud when the corpus is absent, but the SIGNATURE must accept feature_set.
    with pytest.raises(h2h.EvalDataUnavailableError):
        h2h.build_eval("xshot_occurrence", corpus_dir=None, feature_set="position_only")


def test_position_only_variant_carries_its_feature_set():
    for name in ("xshot_occurrence", "xcross_attempt", "ghost_gk"):
        spec = reg.REGISTRY[name]
        po = next((b for b in spec.builds if b.variant == "position_only"), None)
        assert po is not None and po.feature_set == "position_only", name
        default = next(b for b in spec.builds if b.variant == "default")
        assert default.feature_set == "faithful", name


def test_owner_eval_classifier_threads_feature_set_to_prepare(tmp_path):
    # The vintage fix: the eval extractor runs at the VARIANT's feature_set so a 26-feat position_only
    # booster is scored on a 26-feat frame (xgboost predict_proba is strict). Prove the thread-through.
    frames = pd.DataFrame({"game_id": ["g1"], "x": [1.0]})
    fpath = tmp_path / "idsse__g1.parquet"
    frames.to_parquet(fpath)
    actions_dir = tmp_path / "_actions"
    actions_dir.mkdir()
    pd.DataFrame({"type_id": [21], "type_name": ["shot"]}).to_parquet(actions_dir / "idsse__g1.parquet")

    seen = {}

    def fake_prepare(frames_arg, actions_arg, *, home_team_id, feature_set):
        seen["feature_set"] = feature_set
        return pd.DataFrame({"f0": [0.1, 0.2]}), np.array([0, 1]), np.array(["g1", "g1"])

    bundle = h2h._owner_eval_classifier(
        [fpath], {"g1": "home1"}, actions_dir, fake_prepare, feature_set="position_only"
    )
    assert seen["feature_set"] == "position_only"
    assert len(bundle["features"]) == 2 and list(bundle["y"]) == [0, 1]
