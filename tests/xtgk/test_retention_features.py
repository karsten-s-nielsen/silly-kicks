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


def test_coord_derived_is_the_single_source_used_by_extract():
    """The metric's coherence check recomputes these; they MUST be the same code path."""
    import pandas as pd

    from silly_kicks.xtgk._retention_features import (
        COORD_DERIVED_NAMES,
        _coord_derived,
        extract_retention_features,
    )

    a = pd.DataFrame(
        {
            "start_x": [5.5, 30.0],
            "start_y": [34.0, 20.0],
            "end_x": [40.0, 55.0],
            "end_y": [34.0, 44.0],
            "type_id": [12, 0],
            "pressure": [0.1, 0.4],
        }
    )
    full = extract_retention_features(a)
    derived = _coord_derived(a)
    for c in COORD_DERIVED_NAMES:
        pd.testing.assert_series_equal(full[c], derived[c], check_names=False)


def test_gk_geometry_source_passes_through_when_present():
    import pandas as pd

    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN
    from silly_kicks.xtgk._retention_features import extract_retention_features

    a = pd.DataFrame(
        {
            "start_x": [5.5],
            "start_y": [34.0],
            "end_x": [40.0],
            "end_y": [34.0],
            "type_id": [12],
            "pressure": [0.1],
            GK_GEOMETRY_SOURCE_COLUMN: ["resolved_origin"],
        }
    )
    out = extract_retention_features(a)
    assert out[GK_GEOMETRY_SOURCE_COLUMN].tolist() == ["resolved_origin"]
