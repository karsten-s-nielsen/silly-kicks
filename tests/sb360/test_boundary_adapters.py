"""Non-vacuity guards for the boundary-entry adapters (spec Part 1/Part 2 + Non-vacuity section)."""

from __future__ import annotations

import numpy as np

from tests.sb360 import _fixture as F
from tests.sb360._registry import SB360_ENTRIES


def test_xt_gk_v2_columns_are_live_so_identical_is_a_real_comparison():
    """The `identical`->`works` verdict must be a real number comparison, not NaN==NaN. Every
    column is finite; the four terms the doubles drive are non-constant. `xt_gk_v2_pev` is EXEMPT:
    it is 0 by construction in the base metric (p'=p), documented in _metric.py."""
    entry = SB360_ENTRIES["xtgk.compute_xt_gk_v2"]
    actions, frames, links = F.build_leg_b()  # velocity-bearing leg; frame-blind fn ignores it
    out = entry.call(actions, frames, links, F.HOME_TEAM_ID)
    for col in ("xt_gk_v2_position", "xt_gk_v2_pev", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv", "xt_gk_v2"):
        vals = out[col].to_numpy(dtype=float)
        assert np.isfinite(vals).all(), f"{col} has non-finite values: {vals}"
    for col in ("xt_gk_v2_position", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv", "xt_gk_v2"):
        assert np.unique(out[col].to_numpy(dtype=float)).size > 1, f"{col} is constant -- doubles not live"
    assert np.allclose(out["xt_gk_v2_pev"].to_numpy(dtype=float), 0.0), "pev is 0 by construction (p'=p)"


def test_gkdv_build_ghost_frames_is_live_asymmetric_across_legs():
    """Both legs now serve a ghost -- Leg A (freeze-frame) via the position_only variant (ADR-067),
    Leg B via the velocity default -- and they DIFFER on >=1 scored action. That difference (a
    differs_by_design signal, not a NaN asymmetry) is what the substantive verdict rests on, and the
    `not allclose` clause guards against the two legs secretly being the same object (spec Part 2)."""
    entry = SB360_ENTRIES["gkdv.build_ghost_frames"]
    a = entry.call(*F.build_leg_a(), F.HOME_TEAM_ID)["displacement_m"].to_numpy(dtype=float)
    b = entry.call(*F.build_leg_b(), F.HOME_TEAM_ID)["displacement_m"].to_numpy(dtype=float)
    assert np.isfinite(a).any(), "Leg A must serve >=1 ghost via the position_only variant"
    assert np.isfinite(b).any(), "Leg B must score >=1 action"
    both = np.isfinite(a) & np.isfinite(b)
    assert both.any() and not np.allclose(a[both], b[both]), "the two legs' ghosts must DIFFER"


def test_gkdv_delta_das_is_honest_nan_on_the_velocity_less_leg():
    """delta_das STRUCTURALLY needs velocity (ADR-043), so it degrades to honest-NaN on the
    freeze-frame leg and scores on the velocity-bearing leg -- an honest all-NaN, not a masked 0.0."""
    das = SB360_ENTRIES["gkdv.delta_das"]
    a = das.call(*F.build_leg_a(), F.HOME_TEAM_ID)["delta_das"].to_numpy(dtype=float)
    b = das.call(*F.build_leg_b(), F.HOME_TEAM_ID)["delta_das"].to_numpy(dtype=float)
    assert not np.isfinite(a).any(), "delta_das: Leg A must be all-NaN (DAS needs velocity)"
    assert np.isfinite(b).any(), "delta_das: Leg B must score >=1 action"


def test_gkdv_delta_threat_is_live_and_non_vacuous_across_legs():
    """delta_threat computes on BOTH legs (pitch control has a valid zero-velocity positional model,
    ADR-063) and is LIVE: >=1 scored action is NON-ZERO and the two legs DIFFER there. This is the
    non-vacuity fix -- delta_threat was previously a masked 0.0 because no receiver was ahead of the
    ball; sb360-fixture-2 adds a striker so the keeper's threat-suppression is actually measured."""
    thr = SB360_ENTRIES["gkdv.delta_threat_suppression"]
    ta = thr.call(*F.build_leg_a(), F.HOME_TEAM_ID)["delta_threat_suppression"].to_numpy(dtype=float)
    tb = thr.call(*F.build_leg_b(), F.HOME_TEAM_ID)["delta_threat_suppression"].to_numpy(dtype=float)
    assert np.isfinite(ta).any() and np.isfinite(tb).any(), "delta_threat must score on both legs"
    assert np.nanmax(np.abs(ta)) > 0, "delta_threat must be NON-ZERO on >=1 action (not a masked 0.0)"
    both = np.isfinite(ta) & np.isfinite(tb)
    assert both.any() and not np.allclose(ta[both], tb[both]), "the two legs' delta_threat must DIFFER"
