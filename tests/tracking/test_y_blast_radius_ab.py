"""Task 4.3 / ADR-031: the §6 blast-radius SENSITIVITY e2e -- which features depend on frame-y.

Committed + OFFLINE (geometric features only; NO model weights / NO network -- rev-6 C). On a real
SkillCorner slice it flips frame-y (``y -> 68 - y``) with the ACTION anchors held fixed, then asserts
the CORRUPTED (action-anchor x frame-y) features change while the y-INVARIANT ones do not. This keeps
the §6 taxonomy honest after future refactors. It is NOT a fix-correctness e2e -- that is the manual
DGX Gates A/B recorded in ADR-031; this proves *which columns depend on y*.

Measured on this fixture (``synthesize_actions`` on the committed SkillCorner slim slice):
  CHANGES   : nearest_defender_distance, pre_shot_gk_distance_to_shot, pre_shot_gk_y (mirrored position)
  INVARIANT : actor_speed (a magnitude), pre_shot_gk_x (x-only),
              pre_shot_gk_distance_to_goal -- the goal is at CENTRE y=34, so the distance is
              y-symmetric; this REFINES §6, which had lumped it with the corrupted "distances".
  NOT EXERCISED on this slice (documented, not silently capped): receiver_zone_density and
              defenders_in_triangle_to_goal are structurally 0 here (the synthesized actions have no
              players in their receiver-zones / goal-triangles), so this fixture cannot demonstrate
              their y-sensitivity. They are corrupted in principle (§6); a denser fixture would be
              needed to exercise them.
"""

import numpy as np

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_action_context, add_pre_shot_gk_position
from tests.tracking._provider_inputs import (
    N_ACTIONS_PER_PROVIDER,
    load_provider_frames,
    synthesize_actions,
)

W = 68.0


def _maxd(a, b, col):
    x = a[col].to_numpy(dtype=float)
    y = b[col].to_numpy(dtype=float)
    m = ~(np.isnan(x) & np.isnan(y))
    return float(np.nanmax(np.abs(x[m] - y[m]))) if m.any() else 0.0


def test_y_flip_sensitivity_matches_blast_radius_taxonomy():
    frames = load_provider_frames("skillcorner")
    actions = add_pre_shot_gk_context(synthesize_actions(frames, n_actions=N_ACTIONS_PER_PROVIDER))
    fflip = frames.copy()
    fflip["y"] = W - fflip["y"]

    ac_a, ac_b = add_action_context(actions, frames), add_action_context(actions, fflip)
    gk_a, gk_b = add_pre_shot_gk_position(actions, frames), add_pre_shot_gk_position(actions, fflip)

    # CORRUPTED -- action anchor combined with frame-y; MUST depend on frame-y:
    assert _maxd(ac_a, ac_b, "nearest_defender_distance") > 1e-6
    assert _maxd(gk_a, gk_b, "pre_shot_gk_distance_to_shot") > 1e-6
    assert _maxd(gk_a, gk_b, "pre_shot_gk_y") > 1e-6  # absolute mirrored position

    # Y-INVARIANT -- MUST NOT change under a frame-y flip:
    assert _maxd(ac_a, ac_b, "actor_speed") < 1e-6  # a speed magnitude
    assert _maxd(gk_a, gk_b, "pre_shot_gk_x") < 1e-6  # x-only
    assert _maxd(gk_a, gk_b, "pre_shot_gk_distance_to_goal") < 1e-6  # goal at centre y=34 -> y-symmetric
