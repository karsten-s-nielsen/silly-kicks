"""Mirror-registry entry for TF-79 ``add_ball_contacts`` (ADR-028 section 6).

``add_ball_contacts`` takes no ``home_team_id`` (it is a purely geometric ball-contact
detector), so ``role="unused"`` and Gate B skips it -- direction cannot be identity-keyed
in code that never sees the identity. Gate A runs on a dedicated scene: TF-79 v2 credits a contact
only at a real velocity change with a player in reach (detector spec section 4), and
``canonical_scene``'s ball merely follows the play, so its away rows would be all-<NA>.

The two geometric outputs ``first_contact_x``/``first_contact_y`` are PITCH-ABSOLUTE
(frame-LTR ball coordinates, not action-LTR), so under the registry's point reflection they
flip (x -> FIELD_LENGTH - x, y -> FIELD_WIDTH - y) and are ``mirrored_pitch_absolute``. The
remaining columns -- time, player id, confidence, aerial flag, source -- are the SAME physical
contact in both legs, hence ``invariant``.
"""

from __future__ import annotations


def _scene():
    """A HOME pass from (80, 5) received by player 11 at (95, 34) after 1 s, and an AWAY pass (physically from
    (25, 63)) received by player 61 at (10, 34). Each reception is a real velocity change (the ball arrives at about
    17 m/s and rolls on at 2 m/s) with the receiver at the ball, so both rows resolve a first contact."""
    from tests.tracking._mirror_registry import AWAY, HOME
    from tests.tracking.test_cross_delivery import _actions, _frames, _path

    ball = _path((1.0, 2.0, (80.0, 5.0), (95.0, 34.0)), (2.0, 3.0, (95.0, 34.0), (97.0, 34.0))) + _path(
        (11.0, 12.0, (25.0, 63.0), (10.0, 34.0)), (12.0, 13.0, (10.0, 34.0), (8.0, 34.0))
    )
    players = {10: (HOME, 80.0, 5.0), 11: (HOME, 95.0, 34.0), 60: (AWAY, 25.0, 63.0), 61: (AWAY, 10.0, 34.0)}
    actions = _actions([("pass", 1.0, 80.0, 5.0, HOME, 10), ("pass", 11.0, 80.0, 5.0, AWAY, 60)])
    return actions, _frames(ball, players, ltr_team=HOME)


def register() -> None:
    from silly_kicks.tracking.features import add_ball_contacts
    from tests.tracking._mirror_registry import _entry

    _entry(
        "add_ball_contacts",
        lambda a, f, h: add_ball_contacts(a, f),
        {
            "first_contact_x": "mirrored_pitch_absolute",
            "first_contact_y": "mirrored_pitch_absolute",
            "first_contact_time": "invariant",
            "first_contact_player_id": "invariant",
            "first_contact_velocity_change_ms": "invariant",
            "first_contact_confidence": "invariant",
            "first_contact_is_aerial": "invariant",
            "first_contact_source": "invariant",
        },
        tol=1e-9,
        basis=(
            "The contact coordinates are frame-LTR ball positions, so a point reflection is exact "
            "(FIELD_LENGTH - x / FIELD_WIDTH - y). Every other column is the SAME physical contact in "
            "both legs (same player, time, confidence, aerial flag, source), so it is invariant. "
            "The dedicated scene's away pass resolves its reception to player 61, so the comparison is live, "
            "not degenerate."
        ),
        role="unused",
        non_vacuity=("first_contact_x", "first_contact_player_id"),
        reflections={"first_contact_x": "x", "first_contact_y": "y"},
        scene=_scene,
        exempt={
            "first_contact_x": "pitch-absolute ball x; reflects as FIELD_LENGTH - x under a point mirror",
            "first_contact_y": "pitch-absolute ball y; reflects as FIELD_WIDTH - y under a point mirror",
        },
    )
