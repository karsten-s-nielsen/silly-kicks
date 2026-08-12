"""C1.1 -- the behavioural red test for the D3 re-key (ADR-051 D3, ADR-055).

THE PROPERTY. An aggregator's action-LTR geometry must be invariant under a PHYSICAL MIRROR
of the frames with ``home_team_id`` UNCHANGED. Attacking direction is a property of the
FRAMES -- which end each team's keeper stands at, which way the direction labels point -- not
of which team happens to carry the ``home`` label. Mirror the pitch and the same physical
scene is described from the other end; the acting team still attacks the goal it attacked,
so every action-LTR quantity must come back identical.

WHY THIS TEST AND NOT GATE A OR GATE B.

* **Gate A** mirrors the frames AND swaps ``home_team_id``, precisely because "after a
  physical mirror the team attacking +x really is the other one". That swap restores the
  very invariant identity-keying assumes, so Gate A passes whether the aggregator is safe or
  not -- it is structurally blind to this defect.
* **Gate B** varies ``home_team_id`` on fixed frames and DOES see it, which is why the D3
  entries carry ``defect_b``. But Gate B goes VACUOUS the moment the parameter is removed
  (it skips on ``role="unused"``), so it cannot witness the fix -- only the defect.

This test sees the defect AND survives the fix, because it never mentions ``home_team_id``
except to hold it constant. That is what makes it the cycle's primary evidence: the
ASSERTION is unchanged across the transition. The invocation is adapted (the re-key removes
``home_team_id`` from these signatures, so any hard break edits the call) -- red -> edited ->
green is the honest description, and the PR carries the diff so a reader can verify the
assertion was not rewritten to fit the new behaviour.

MEASURED RED before the re-key, on the coherent canonical scene:

    add_defensive_line   defensive_line_x 23.25, back_line_high_x 9.40,
                         compactness_x 3.84, lateral_width 9.40, max_lateral_gap 4.80
    add_packing          packing_made 6.0, packing_net 6.0, packing_goal_threat 4.0

Every one of those must be 0 after the re-key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._mirror_registry import canonical_scene, mirror_frames

#: The six D3 sites reach the surface through these aggregators. Keyed by aggregator name so
#: a reader can map a failure back to the site; the value is the call and the columns whose
#: action-LTR value must not move.
_CASES: dict[str, tuple] = {
    "add_defensive_line": (
        lambda a, f: __import__("silly_kicks.tracking.features", fromlist=["add_defensive_line"]).add_defensive_line(
            a, f
        ),
        ("defensive_line_x", "back_line_high_x", "compactness_x", "lateral_width", "max_lateral_gap"),
    ),
    "add_packing": (
        lambda a, f: __import__("silly_kicks.tracking.features", fromlist=["add_packing"]).add_packing(a, f),
        ("packing_made", "packing_net", "packing_goal_threat"),
    ),
    "add_structural_pass": (
        lambda a, f: __import__("silly_kicks.tracking.features", fromlist=["add_structural_pass"]).add_structural_pass(
            a, f
        ),
        ("structural_lbs", "structural_sgm", "structural_sdi"),
    ),
    "add_line_break": (
        lambda a, f: __import__("silly_kicks.tracking.features", fromlist=["add_line_break"]).add_line_break(a, f),
        ("line_break", "n_attackers_behind_line"),
    ),
    # The two D11 BOOL sites have no Gate C entry -- they take a resolved direction, never a
    # GoalMap, so swapping a map they never receive would move nothing and such an entry would
    # pass because its input is ignored. THIS test is their only detector, which is exactly why
    # leaving either out would repeat the `packing_goal_threat` near-miss one level up.
    "add_player_influence": (
        lambda a, f: __import__(
            "silly_kicks.tracking.features", fromlist=["add_player_influence"]
        ).add_player_influence(a, f, _gate_xt()),
        ("off_ball_xt_team", "off_ball_xt_opponent", "off_ball_xt_diff"),
    ),
}


def _gate_xt():
    """The registry's non-degenerate, y-ASYMMETRIC xT -- reused rather than re-rolled.

    A y-symmetric grid makes an x-only reprojection look exact, which is the incomplete repair
    ADR-041 shipped; borrowing the registry's fixture keeps this test honest about both axes.
    """
    from tests.tracking._mirror_registry import gate_xt

    return gate_xt()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_action_ltr_geometry_is_invariant_under_a_frame_mirror_with_home_id_held(name):
    """Mirror the FRAMES, hold ``home_team_id``: action-LTR geometry must not move."""
    call, cols = _CASES[name]
    actions, frames = canonical_scene()

    base = call(actions.copy(), frames.copy())
    mirrored = call(actions.copy(), mirror_frames(frames.copy()))

    moved = {}
    for col in cols:
        b = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        m = pd.to_numeric(mirrored[col], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(b) & np.isfinite(m)
        assert both.any(), f"{name}.{col}: no comparable rows -- the test would be vacuous"
        delta = float(np.abs(b[both] - m[both]).max())
        if delta > 1e-9:
            moved[col] = delta

    assert not moved, (
        f"{name}: action-LTR geometry moved when the FRAMES were mirrored and home_team_id "
        f"held constant: {moved}. Attacking direction is a property of the frames, not of "
        f"which team is labelled home -- this is the D3 identity-keyed direction defect."
    )


def test_the_mirror_actually_moves_the_scene():
    """Non-vacuity: the invariance assertion passes trivially if the mirror is a no-op.

    Cheap to get wrong -- a mirror helper that returned its input unchanged would make every
    case above pass while proving nothing.
    """
    _actions, frames = canonical_scene()
    mirrored = mirror_frames(frames.copy())
    dx = float(np.abs(frames["x"].to_numpy(float) - mirrored["x"].to_numpy(float)).max())
    dy = float(np.abs(frames["y"].to_numpy(float) - mirrored["y"].to_numpy(float)).max())
    assert dx > 1.0 and dy > 1.0, f"mirror moved almost nothing (dx={dx}, dy={dy})"
    labels_base = set(frames["team_attacking_direction"].dropna())
    labels_mirror = set(mirrored["team_attacking_direction"].dropna())
    assert labels_base == labels_mirror == {"ltr", "rtl"}, (
        "both legs must carry both direction labels, or the mirror did not swap them"
    )
