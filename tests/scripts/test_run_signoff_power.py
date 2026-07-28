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


def test_a_partially_nan_confounder_is_ALSO_refused_naming_the_count():
    """SUPERSEDES the earlier "partial missingness is the estimator's problem" reading (R3 MEDIUM 2).

    That rationale rested on a factual claim about the estimator that does not hold. MEASURED
    directly against the shipped `fit_propensity`::

        X = [[1.0, 0.1], [nan, 0.2], [3.0, 0.3], [4.0, 0.4]]
        -> ValueError: Input X contains NaN.

    sklearn rejects a SINGLE NaN cell, and every spell belongs to some cluster, so a resample will
    eventually include the offending row. Letting it through therefore does not preserve usable
    spells -- it relocates the same fatal error to the middle of a long run, with a message that
    names no column. Dropping the rows here is rejected for the original reason: that silently
    redefines the estimation sample, which is a design change rather than error handling.
    """
    spells = pd.DataFrame({"r": [1.0, np.nan, 3.0], "theta": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match=r"r: 1/3 non-finite"):
        mod.build_design_matrix(spells, ("r", "theta"))


def test_a_fully_finite_design_matrix_is_accepted():
    """The other side: the guard must not reject a healthy matrix, or it is just an outage."""
    spells = pd.DataFrame({"r": [1.0, 2.0, 3.0], "theta": [0.1, 0.2, 0.3]})
    assert mod.build_design_matrix(spells, ("r", "theta")).shape == (3, 2)
