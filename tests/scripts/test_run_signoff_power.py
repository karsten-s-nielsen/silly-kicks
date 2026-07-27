"""TF-19 sign-off package: the power driver's design-matrix assembly (review R3 MEDIUM 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.run_signoff_power as mod  # bare import: tests/scripts/ has NO __init__.py


def test_all_nan_confounder_raises_naming_the_column():
    """MEASURED alternative: an all-NaN column reaches `fit_propensity` and dies inside sklearn as
    `ValueError: Input X contains NaN. LogisticRegression does not accept missing value` -- a
    message that names no culprit, surfacing deep in a corpus run."""
    spells = pd.DataFrame({"r": [1.0, 2.0], "theta": [0.1, 0.2], "score_differential": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="score_differential"):
        mod.build_design_matrix(spells, ("r", "theta", "score_differential"))


def test_absent_confounder_raises_naming_the_column():
    spells = pd.DataFrame({"r": [1.0, 2.0]})
    with pytest.raises(ValueError, match="theta"):
        mod.build_design_matrix(spells, ("r", "theta"))


def test_design_matrix_returns_columns_in_the_registered_order():
    spells = pd.DataFrame({"theta": [0.1, 0.2], "r": [1.0, 2.0]})
    X = mod.build_design_matrix(spells, ("r", "theta"))
    assert X.shape == (2, 2)
    assert X[0, 0] == 1.0 and X[0, 1] == 0.1  # registered order, not frame order


def test_a_partially_nan_confounder_is_allowed_through():
    """Only an ENTIRELY dead column is refused -- per-row missingness is the estimator's problem,
    and silently rejecting it here would drop usable spells."""
    spells = pd.DataFrame({"r": [1.0, np.nan, 3.0], "theta": [0.1, 0.2, 0.3]})
    X = mod.build_design_matrix(spells, ("r", "theta"))
    assert X.shape == (3, 2)
