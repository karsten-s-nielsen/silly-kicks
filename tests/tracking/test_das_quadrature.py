"""Task 6 gates: the periodic-vs-reference quadrature is EXACTLY an end-ray weight change (ADR-108)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from silly_kicks.tracking._das_engine import _numpy_integrand, compute_das
from silly_kicks.tracking._das_pack import pack_frames
from silly_kicks.tracking._das_params import DAS_PARAMS, simulation_grids
from tests.tracking._das_golden import load_golden
from tests.tracking._das_helpers import engine_arrays

_REF = dataclasses.replace(DAS_PARAMS, quadrature="reference")


def _delta_weight():
    gp = simulation_grids(DAS_PARAMS)
    gr = simulation_grids(_REF)
    return gp.dr[None, :] * gp.d_area - gr.dr[None, :] * gr.d_area  # (PHI, T)


def test_das_difference_is_exactly_the_end_ray_weight_difference():
    packed = pack_frames(load_golden().frames_for("S01"), attacking_direction_col="dir")
    _g_as, g_das = _numpy_integrand(packed, DAS_PARAMS)  # quadrature-independent per-ray integrand
    per = compute_das(packed, DAS_PARAMS, engine="numpy")
    ref = compute_das(packed, _REF, engine="numpy")
    delta_w = _delta_weight()
    expected = np.sum(g_das * delta_w[None, :, :], axis=(1, 2))
    np.testing.assert_allclose(per.team_das - ref.team_das, expected, rtol=1e-12, atol=1e-12)


def test_delta_weight_is_nonzero_only_at_end_rays():
    dw = _delta_weight()
    assert np.any(dw[0] != 0) and np.any(dw[-1] != 0)
    assert not np.any(dw[1:-1] != 0)


def test_quadrature_change_is_non_vacuous():
    # if the relation used a zero delta (i.e. ignored the quadrature fix) it would be wrong:
    packed = pack_frames(load_golden().frames_for("S05"), attacking_direction_col="dir")
    per = compute_das(packed, DAS_PARAMS, engine="numpy")
    ref = compute_das(packed, _REF, engine="numpy")
    assert np.max(np.abs(per.team_das - ref.team_das)) > 1e-6  # the two quadratures really differ


def test_interior_weight_perturbation_breaks_the_relation(monkeypatch):
    packed = pack_frames(load_golden().frames_for("S01"), attacking_direction_col="dir")
    _g_as, g_das = _numpy_integrand(packed, DAS_PARAMS)
    # a delta that (wrongly) also perturbs an interior ray must NOT reconstruct per - ref.
    bad = _delta_weight()
    bad[5, :] += 1e-6
    per = compute_das(packed, DAS_PARAMS, engine="numpy")
    ref = compute_das(packed, _REF, engine="numpy")
    reconstructed = np.sum(g_das * bad[None, :, :], axis=(1, 2))
    assert not np.allclose(per.team_das - ref.team_das, reconstructed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scene,mirror", [("S01", "M01"), ("S05", "M02"), ("S10", "M03")])
def test_periodic_das_is_mirror_invariant(scene, mirror):
    # Mirror invariance is EXACT in real arithmetic; the numerical floor is set by the danger term's
    # arccos, which is ill-conditioned near the goal mouth (d/dx arccos -> inf as x -> 1), so the
    # reflected arithmetic differs by ~1e-7 relative. That is ~1e5x below the reference-quadrature
    # defect this test's sibling records, so rtol=1e-6 is a decisive invariance assertion.
    a = engine_arrays(scene, DAS_PARAMS, engine="numpy")["team_das"]
    b = engine_arrays(mirror, DAS_PARAMS, engine="numpy")["team_das"]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def test_reference_quadrature_is_not_mirror_invariant():
    # The recorded defect: reference-quadrature DAS differs from its point reflection by >1% (scene
    # dependent: ~1.3% (S10) to ~44% (S05)); periodic (above) is mirror-invariant to ~1e-7.
    g = load_golden()
    for scene, mirror in (("S01", "M01"), ("S05", "M02"), ("S10", "M03")):
        a = g.reference_for(scene)["team_das"]
        b = g.reference_for(mirror)["team_das"]
        relgap = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-9)))
        assert relgap > 1e-2, f"{scene}: reference relgap {relgap:.3e} not > 1e-2"
