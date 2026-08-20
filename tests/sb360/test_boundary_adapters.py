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
    """Leg A (freeze-frame) serves no ghost (ADR-054 refusal) -> all NaN; Leg B scores the
    in-domain actions -> finite. That asymmetry is the honest_nan signal (spec Part 2)."""
    entry = SB360_ENTRIES["gkdv.build_ghost_frames"]
    a_out = entry.call(*F.build_leg_a(), F.HOME_TEAM_ID)
    b_out = entry.call(*F.build_leg_b(), F.HOME_TEAM_ID)
    assert not np.isfinite(a_out["displacement_m"].to_numpy(dtype=float)).any(), "Leg A must be all-NaN"
    assert np.isfinite(b_out["displacement_m"].to_numpy(dtype=float)).any(), "Leg B must score >=1 action"


def test_gkdv_arms_are_live_asymmetric_across_legs():
    for name, col in (
        ("gkdv.delta_das", "delta_das"),
        ("gkdv.delta_threat_suppression", "delta_threat_suppression"),
    ):
        entry = SB360_ENTRIES[name]
        a = entry.call(*F.build_leg_a(), F.HOME_TEAM_ID)[col].to_numpy(dtype=float)
        b = entry.call(*F.build_leg_b(), F.HOME_TEAM_ID)[col].to_numpy(dtype=float)
        assert not np.isfinite(a).any(), f"{name}: Leg A must be all-NaN (ghost refusal)"
        assert np.isfinite(b).any(), f"{name}: Leg B must score >=1 action"
