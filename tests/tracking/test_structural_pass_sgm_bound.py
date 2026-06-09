"""SGM (structural_sgm) numeric-bound fix (BUG 3, 2026-06-09).

structural_sgm = 1/rho_r - 1/rho_p, rho = sum of Gaussian kernels over opponent defenders.
When the passer/receiver are far from ALL defenders, rho underflows toward 0 and 1/rho explodes
(lakehouse: GS min -88,955,384). The StructuralPassParams "sigma=15 -> intrinsically bounded,
no eps-floor" claim is falsified by real byline-cross / fast-break frames. An eps-floor on rho
(a defender at 3-sigma contributes exp(-4.5) ~= 0.0111, capping 1/rho ~= 90) bounds it.
"""

import numpy as np

from silly_kicks.tracking._structural_pass import _structural_pass_core


def test_sgm_bounded_when_defenders_far():
    # all defenders ~90 m from the passer/receiver (a byline cross with defenders upfield):
    # rho underflows; sgm must stay bounded by the eps-floor, NOT explode to ~1e7.
    defenders = np.array([[5.0, 34.0], [8.0, 30.0], [10.0, 38.0]])
    _lbs, sgm, _sdi = _structural_pass_core(defenders, (100.0, 34.0), (98.0, 30.0), sigma=15.0)
    assert np.isfinite(sgm)
    assert abs(sgm) <= 100.0, f"sgm should be bounded (~90) by the eps-floor, got {sgm}"


def test_sgm_sane_when_defenders_near_unaffected_by_floor():
    # defenders within range of passer/receiver -> rho is O(1); the floor must NOT bite.
    defenders = np.array([[50.0, 34.0], [55.0, 30.0], [47.0, 36.0]])
    _lbs, sgm, _sdi = _structural_pass_core(defenders, (48.0, 34.0), (60.0, 32.0), sigma=15.0)
    assert abs(sgm) < 5.0, f"normal-geometry sgm should be small, got {sgm}"
