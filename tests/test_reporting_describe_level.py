import numpy as np
import pandas as pd
import pytest

from silly_kicks.reporting import describe_level


@pytest.mark.parametrize(
    "z,label",
    [
        (2.0, "outstanding"),
        (1.5, "outstanding"),
        (1.49, "excellent"),
        (1.0, "excellent"),
        (0.99, "good"),
        (0.5, "good"),
        (0.49, "average"),
        (-0.5, "average"),
        (-0.51, "below average"),
        (-1.0, "below average"),
        (-1.01, "poor"),
        (-5.0, "poor"),
    ],
)
def test_bands_higher_is_better(z, label):
    assert describe_level(z) == label


def test_direction_flip():
    assert describe_level(2.0, higher_is_better=False) == "poor"
    assert describe_level(-2.0, higher_is_better=False) == "outstanding"
    # exactly-average is direction-invariant
    assert describe_level(0.0) == "average"
    assert describe_level(0.0, higher_is_better=False) == "average"


def test_nan_is_unknown():
    assert describe_level(float("nan")) == "unknown"


def test_vectorised_array_and_series():
    out = describe_level(np.array([2.0, 0.0, np.nan]))
    assert isinstance(out, np.ndarray)
    assert list(out) == ["outstanding", "average", "unknown"]
    s = pd.Series([2.0, -2.0], index=["a", "b"])
    r = describe_level(s)
    assert isinstance(r, pd.Series) and list(r.index) == ["a", "b"]
    assert list(r) == ["outstanding", "poor"]


def test_array_direction_flip():
    out = describe_level(np.array([2.0, -2.0]), higher_is_better=False)
    assert list(out) == ["poor", "outstanding"]


def test_scalar_returns_str_not_array():
    assert isinstance(describe_level(1.0), str)
