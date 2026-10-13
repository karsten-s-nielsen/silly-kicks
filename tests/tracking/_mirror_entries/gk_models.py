"""GK-model MirrorEntry registrations (ADR-028 section 6).

Group ``gk_models``: ``add_ghost_gk``.

The served value is a boosted-tree prediction (NOT pure geometry), so the entry carries a measured
tolerance rather than a bit-exact one -- see the ``basis`` note on the registration below.
"""

from __future__ import annotations

import functools


@functools.cache
def _ghost_model():
    """Load the bundled ghost model ONCE (Gate A + Gate B call an entry five times)."""
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    return GhostGkModel.from_variant("default")


def register() -> None:
    from silly_kicks.tracking.features import add_ghost_gk
    from tests.tracking._mirror_registry import _entry

    _entry(
        "add_ghost_gk",
        lambda a, f, h: add_ghost_gk(a, f, home_team_id=h, model=_ghost_model()),
        {"ghost_gk_x": "invariant", "ghost_gk_y": "invariant"},
        tol=3.0,
        basis=(
            "NOT pure geometry -- the served value is a boosted-tree prediction, and the bundled "
            "model is not exactly mirror-symmetric (its features include strict-inequality box "
            "counts and a convex hull, neither of which commutes with a reflection). Measured on "
            "this scene: 0.0755 m on x, 0.278 m on y. 3.0 m matches the shipped _GHOST_Y_TOL in "
            "test_action_ltr_mirror_invariance.py, whose lateralized probe measures 1.26 m of the "
            "same model asymmetry, and leaves headroom for the queued re-fit. CAVEAT, recorded so "
            "nobody reads more into a green than is there: this scene is not lateralized, so the "
            "model predicts a near-central keeper (y ~ 33.6-33.7) and a y-REPROJECTION FLIP would "
            "move y by only ~0.71 m -- under this tol. x is the discriminating axis here (a "
            "goal-side leak moves it ~90 m); the dedicated y-flip guard is the off-centre probe in "
            "test_action_ltr_mirror_invariance.py::test_ghost_gk_mirror_invariant."
        ),
        role="direction_only",
        non_vacuity=("ghost_gk_x", "ghost_gk_y"),
    )
