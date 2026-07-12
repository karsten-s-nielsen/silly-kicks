import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_construct_validity_reports_all_baselines_with_finite_v2_auc():
    from scripts.validate_xtgk_v2 import construct_validity_scores
    from tests.xtgk.conftest import mixed_shot_and_shotless_cohort

    actions = mixed_shot_and_shotless_cohort()
    scores = construct_validity_scores(actions, xg_column="xg", pressure_column="pressure")
    for key in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_stored", "lift"):
        assert key in scores, f"missing baseline {key}"
    # both classes present in the out-of-sample test split -> finite AUC for the frame-free scorers
    assert np.isfinite(scores["xt_gk_v2"]["auc"])
    assert np.isfinite(scores["destination_xt"]["auc"])
