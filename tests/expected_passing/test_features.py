import numpy as np

from silly_kicks.expected_passing._features import (
    FEATURE_NAMES,
    feature_contract_block,
    pass_completion_features,
)


def test_feature_matrix_shape_and_names():
    X = pass_completion_features(np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([40.0]))
    assert X.shape == (1, len(FEATURE_NAMES))
    assert FEATURE_NAMES == [
        "distance",
        "angle",
        "forward",
        "lateral",
        "origin_x",
        "origin_y",
        "target_x",
        "target_y",
        "origin_third",
        "target_third",
    ]  # exact order pinned


def test_pitch_thirds_bucket_by_x():
    X = pass_completion_features(
        np.array([10.0, 90.0]),
        np.array([34.0, 34.0]),
        np.array([50.0, 100.0]),
        np.array([34.0, 34.0]),
    )
    ot = FEATURE_NAMES.index("origin_third")
    tt = FEATURE_NAMES.index("target_third")
    assert X[0, ot] == 0 and X[1, ot] == 2  # x=10 -> defensive third; x=90 -> attacking third
    assert X[0, tt] == 1 and X[1, tt] == 2  # x=50 -> middle; x=100 -> attacking


def test_distance_and_forward_are_correct():
    X = pass_completion_features(np.array([20.0]), np.array([34.0]), np.array([50.0]), np.array([34.0]))
    d = X[0, FEATURE_NAMES.index("distance")]
    fwd = X[0, FEATURE_NAMES.index("forward")]
    assert abs(d - 30.0) < 1e-9 and abs(fwd - 30.0) < 1e-9  # straight forward 30 m


def test_nan_coordinate_yields_nan_features_not_a_fabricated_value():
    X = pass_completion_features(np.array([np.nan]), np.array([34.0]), np.array([50.0]), np.array([40.0]))
    assert np.isnan(X[0]).all()


def test_feature_contract_block_is_stable_and_declares_constants():
    b = feature_contract_block()
    assert b["feature_names"] == FEATURE_NAMES
    assert np.isfinite(np.asarray(b["probe_features"])).all()


def test_scalar_origin_broadcasts_against_array_target():
    # IMPL-01 root cause: the counterfactual seam scores ONE origin against its k selected zone centres,
    # i.e. a SCALAR origin vs a length-k target array. This must broadcast, not raise (it did before the
    # fix -- np.column_stack could not concat the 0-d origin cols with the length-k target cols).
    X = pass_completion_features(20.0, 34.0, np.array([50.0, 60.0, 70.0]), np.array([40.0, 44.0, 30.0]))
    assert X.shape == (3, len(FEATURE_NAMES))
    d = X[:, FEATURE_NAMES.index("distance")]
    np.testing.assert_allclose(d, np.hypot([30.0, 40.0, 50.0], [6.0, 10.0, -4.0]), rtol=0, atol=1e-9)
    # the shared scalar origin is broadcast into every row's origin_x column
    np.testing.assert_allclose(X[:, FEATURE_NAMES.index("origin_x")], [20.0, 20.0, 20.0], rtol=0, atol=0)
