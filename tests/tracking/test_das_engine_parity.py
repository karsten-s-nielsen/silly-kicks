"""Task 4 gates: the numpy engine reproduces accessible-space 2.0.15 (reference quadrature), Δ ~ 0."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from silly_kicks.tracking._das_params import DAS_PARAMS
from tests.tracking._das_golden import NORMAL_SCENES, load_golden
from tests.tracking._das_helpers import engine_arrays

_REF = dataclasses.replace(DAS_PARAMS, quadrature="reference")
_FIELDS = ("team_as", "team_das", "player_as", "player_das")


@pytest.mark.parametrize("scene", NORMAL_SCENES)
def test_numpy_engine_reproduces_reference(scene):
    exp = load_golden().reference_for(scene)
    got = engine_arrays(scene, _REF, engine="numpy")
    for col in _FIELDS:
        e, a = exp[col], got[col]
        assert a.shape == e.shape, f"{scene}/{col} shape {a.shape} != {e.shape}"
        assert np.array_equal(np.isfinite(a), np.isfinite(e)), f"{scene}/{col} finite-mask mismatch"
        m = np.isfinite(e)
        np.testing.assert_allclose(a[m], e[m], rtol=1e-12, atol=1e-12, err_msg=f"{scene}/{col}")


def test_reference_parity_max_abs_delta_recorded(capsys):
    """Report the measured max |Δ| across all normal scenes (evidence, not just pass/fail)."""
    worst = 0.0
    for scene in NORMAL_SCENES:
        exp = load_golden().reference_for(scene)
        got = engine_arrays(scene, _REF, engine="numpy")
        for col in _FIELDS:
            m = np.isfinite(exp[col])
            if m.any():
                worst = max(worst, float(np.max(np.abs(got[col][m] - exp[col][m]))))
    assert worst <= 1e-9, f"max |Δ| {worst:.3e} exceeds 1e-9"
    print(f"\nnumpy-vs-reference max |Δ| over normal scenes: {worst:.3e}")


def test_parity_gate_discriminates_b1_perturbation():
    bumped = dataclasses.replace(_REF, b1=_REF.b1 * (1 + 1e-9))
    exp = load_golden().reference_for("S05")["team_das"]
    got = engine_arrays("S05", bumped, engine="numpy")["team_das"]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_parity_gate_discriminates_periodic_quadrature():
    exp = load_golden().reference_for("S05")["team_das"]
    got = engine_arrays("S05", DAS_PARAMS, engine="numpy")["team_das"]  # periodic default
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


_HAS_NUMBA = pytest.importorskip("numba", reason="numba engine parity") is not None


@pytest.mark.parametrize("scene", NORMAL_SCENES)
def test_numba_engine_reproduces_reference(scene):
    exp = load_golden().reference_for(scene)
    got = engine_arrays(scene, _REF, engine="numba")
    for col in _FIELDS:
        e, a = exp[col], got[col]
        assert np.array_equal(np.isfinite(a), np.isfinite(e)), f"{scene}/{col} finite-mask mismatch"
        m = np.isfinite(e)
        np.testing.assert_allclose(a[m], e[m], rtol=1e-10, atol=1e-10, err_msg=f"{scene}/{col}")
