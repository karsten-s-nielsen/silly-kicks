"""Task 8 gates: native get_xc reproduces accessible-space; D-XC-FRAME / D-XC-TEAM divergences."""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking._das_engine import compute_xc
from silly_kicks.tracking._das_params import XC_PARAMS
from tests.tracking._das_golden import load_golden


def _xc_passes_and_frames():
    g = load_golden()
    passes = g.passes_for("X01").sort_values("pass_index").reset_index(drop=True)
    frames = g.frames_for("S01")  # X01 tracking is the S01 frames (see the generator)
    return g, passes, frames


def test_xc_reproduces_reference():
    g, passes, frames = _xc_passes_and_frames()
    got = compute_xc(passes, frames, params=XC_PARAMS)
    exp = g.xc.sort_values("pass_index")["xC"].to_numpy()
    assert np.array_equal(np.isfinite(got), np.isfinite(exp))
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_xc_in_unit_interval():
    _g, passes, frames = _xc_passes_and_frames()
    got = compute_xc(passes, frames, params=XC_PARAMS)
    assert np.all((got[np.isfinite(got)] >= 0.0) & (got[np.isfinite(got)] <= 1.0))


def test_d_xc_frame_missing_is_nan_not_raise():
    g, passes, frames = _xc_passes_and_frames()
    bad = passes.iloc[[0]].copy()
    bad["frame_id"] = 999
    got = compute_xc(bad, frames, params=XC_PARAMS)
    assert np.isnan(got[0])
    # the reference RAISED here (golden recorded it)
    assert g.errors["V-XC-FRAME"]["type"] == "ValueError"


def test_d_xc_team_absent_is_nan_not_raise():
    g, passes, frames = _xc_passes_and_frames()
    bad = passes.iloc[[0]].copy()
    bad["team_id"] = 9  # team not in the tracking frame
    got = compute_xc(bad, frames, params=XC_PARAMS)
    assert np.isnan(got[0])
    assert g.errors["V-XC-TEAM"]["type"] == "ValueError"


def test_passer_absent_from_frame_is_computed():
    _g, passes, frames = _xc_passes_and_frames()
    bad = passes.iloc[[0]].copy()
    bad["player_id"] = 100000  # a passer not present -> nothing to exclude, still computed
    got = compute_xc(bad, frames, params=XC_PARAMS)
    assert np.isfinite(got[0])
