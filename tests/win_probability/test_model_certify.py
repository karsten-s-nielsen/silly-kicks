import numpy as np
import pytest

from silly_kicks.win_probability import (
    WinProbabilityIntegrityError,
    WinProbabilityModel,
    WinProbabilityParams,
)


def _inject(beta, intercept=0.0):
    m = WinProbabilityModel(params=WinProbabilityParams.default())
    m._beta = np.asarray(beta, dtype="float64")
    m._intercept = float(intercept)
    m._isotonic = None
    m._fitted = True
    return m


def test_certify_passes_monotone_model():
    # sane, monotone-ish hazard: certification must NOT raise.
    _inject([0.3, -0.005, 0.2, 0.05, 0.1], -3.0).certify_coherence(params=WinProbabilityParams.default())


def test_certify_rejects_negative_leverage_model():
    # a hugely-negative score_diff coefficient makes a lead suppress own scoring so hard that P(win) is
    # non-monotone in score_diff (negative leverage) -> certify must raise (TF63-SPEC-09 bundle-time gate).
    m = _inject([-5.0, 0.0, 0.0, 0.0, 0.0], 0.0)
    with pytest.raises(WinProbabilityIntegrityError):
        m.certify_coherence(params=WinProbabilityParams.default())
