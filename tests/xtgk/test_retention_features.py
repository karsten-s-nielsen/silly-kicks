import numpy as np

from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES, extract_retention_features
from tests.xtgk.conftest import three_band_cohort


def test_features_have_expected_columns_and_length():
    actions = three_band_cohort()  # carries start/end coords + a 'pressure' column
    X = extract_retention_features(actions)  # marts-native: geometry + pressure + type, no frames
    assert list(X.columns) == RETENTION_FEATURE_NAMES
    assert len(RETENTION_FEATURE_NAMES) == 8  # frames-only density dropped (tracking deprecated)
    assert len(X) == len(actions)
    assert np.isfinite(X["length"].to_numpy()).all()


def test_release_pressure_absent_column_is_nan():
    actions = three_band_cohort().drop(columns=["pressure"])
    X = extract_retention_features(actions)
    assert X["release_pressure"].isna().all()
