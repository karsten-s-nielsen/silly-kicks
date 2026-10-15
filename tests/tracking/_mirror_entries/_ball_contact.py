"""Mirror-registry entry for TF-79 ``add_ball_contacts`` (ADR-028 section 6).

``add_ball_contacts`` takes no ``home_team_id`` (it is a purely geometric ball-contact
detector), so ``role="unused"`` and Gate B skips it -- direction cannot be identity-keyed
in code that never sees the identity. Gate A runs on ``canonical_scene`` (the ball follows
the play, so a candidate contact lands in a home AND an away action window -- measured:
away actions 3 and 4 resolve to players 60 and 63).

The two geometric outputs ``first_contact_x``/``first_contact_y`` are PITCH-ABSOLUTE
(frame-LTR ball coordinates, not action-LTR), so under the registry's point reflection they
flip (x -> FIELD_LENGTH - x, y -> FIELD_WIDTH - y) and are ``mirrored_pitch_absolute``. The
remaining columns -- time, player id, confidence, aerial flag, source -- are the SAME physical
contact in both legs, hence ``invariant``.
"""

from __future__ import annotations


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
            "Measured max |base - reflected mirror| = 0.0 on canonical_scene, where away actions 3/4 "
            "resolve contacts to players 60/63 (so the comparison is live, not degenerate)."
        ),
        role="unused",
        non_vacuity=("first_contact_x", "first_contact_player_id"),
        reflections={"first_contact_x": "x", "first_contact_y": "y"},
        exempt={
            "first_contact_x": "pitch-absolute ball x; reflects as FIELD_LENGTH - x under a point mirror",
            "first_contact_y": "pitch-absolute ball y; reflects as FIELD_WIDTH - y under a point mirror",
        },
    )
