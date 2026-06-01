"""Physical-probability invariant: xshot_occurrence in [0, 1]."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking import _xshot_occurrence as xs


def test_predict_proba_in_unit_interval():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(50, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(50) < 0.1).astype(int))
    p = xs.XShotOccurrenceModel().fit(X, y).predict_proba(X)
    assert np.all((p >= 0.0) & (p <= 1.0))
