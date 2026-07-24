import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.defensive_credit._sizing import extinguished_xt, xg_of_shot


def test_xg_of_shot_reads_injected_column():
    actions = pd.DataFrame({"action_id": [7], "xg": [0.23]})
    assert xg_of_shot(actions.iloc[0], xg_column="xg") == pytest.approx(0.23)


def test_xg_of_shot_nan_passes_through():
    actions = pd.DataFrame({"action_id": [7], "xg": [np.nan]})
    assert np.isnan(xg_of_shot(actions.iloc[0], xg_column="xg"))


def test_xg_of_shot_fails_loud_when_column_absent():
    actions = pd.DataFrame({"action_id": [7]})
    with pytest.raises(ValueError, match="xg_column"):
        xg_of_shot(actions.iloc[0], xg_column="xg")


def test_extinguished_xt_reads_fitted_surface(fitted_xt):
    # a deep-in-attack point should have higher xT than a deep-own-half point
    vals = extinguished_xt([(95.0, 34.0), (10.0, 34.0)], fitted_xt)
    assert vals[0] > vals[1]


def test_extinguished_xt_requires_fitted(unfitted_xt):
    with pytest.raises((ValueError, RuntimeError)):  # require_fitted_xt raises NotFittedError family
        extinguished_xt([(50.0, 34.0)], unfitted_xt)
