"""Mirror-registry entry for TF-86 ``add_cross_deliveries`` (ADR-133; ADR-028 section 6).

``add_cross_deliveries`` takes no ``home_team_id``: direction comes from the frames through
``acting_team_attacks_rtl``, so ``role="unused"`` and Gate B skips it. Gate A runs on a dedicated
scene, because ``canonical_scene`` has no wide delivery followed by a ball path into the box, so
both output columns would be all-<NA> on the away rows (vacuous exactly where an ADR-028 defect
lives).

Both outputs are action-level facts about the same physical delivery (did it head into the box,
was it aerial), so they are ``invariant`` under the point reflection.
"""

from __future__ import annotations


def _scene():
    """Two crosses from the action-LTR spot (80, 5): a HOME cross (attacks +x) into the box at 2.5 m, and an
    AWAY cross (attacks -x, so physically from (25, 63) towards the left box) along the ground. Each ends at a
    receiver in the box, so ``cross_attempt`` is True on both rows and ``cross_height`` takes both values."""
    from tests.tracking._mirror_registry import AWAY, HOME
    from tests.tracking.test_cross_delivery import _actions, _frames, _path

    ball = _path((1.0, 2.0, (80.0, 5.0), (95.0, 34.0)), z=2.5) + _path((11.0, 12.0, (25.0, 63.0), (10.0, 34.0)), z=0.3)
    players = {10: (HOME, 80.0, 5.0), 11: (HOME, 96.0, 34.0), 60: (AWAY, 25.0, 63.0), 61: (AWAY, 9.0, 34.0)}
    actions = _actions([("pass", 1.0, 80.0, 5.0, HOME, 10), ("pass", 11.0, 80.0, 5.0, AWAY, 60)])
    return actions, _frames(ball, players, ltr_team=HOME)


def register() -> None:
    from silly_kicks.tracking.features import add_cross_deliveries
    from tests.tracking._mirror_registry import _entry

    _entry(
        "add_cross_deliveries",
        lambda a, f, h: add_cross_deliveries(a, f),
        {"cross_attempt": "invariant", "cross_height": "invariant"},
        tol=1e-9,
        basis=(
            "Both columns are categorical facts about one physical delivery (heads into the box; aerial), so a "
            "point reflection of the frames with swapped direction labels must leave them identical. The scene's "
            "away row is a real cross (physically towards the left box), so the comparison is live."
        ),
        role="unused",
        non_vacuity=("cross_attempt", "cross_height"),
        scene=_scene,
    )
