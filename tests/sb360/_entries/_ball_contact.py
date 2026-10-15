"""SB360 verdict -- TF-79 ball-contact family.

Observations are TRANSCRIBED FROM EXECUTION; only a human writes a rationale.

``add_ball_contacts`` is trajectory-only: candidate detection needs a dense ball trajectory, so a
single SB360 freeze frame resolves NO contact and every ``first_contact_*`` column is all-NaN by
design (honest-NaN, not a defect). Same data-regime boundary as ``add_elastic_sync``.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import AxisVerdict, _entry

_BALL_CONTACT_COLS = (
    "first_contact_time",
    "first_contact_player_id",
    "first_contact_x",
    "first_contact_y",
    "first_contact_velocity_change_ms",
    "first_contact_confidence",
    "first_contact_is_aerial",
    "first_contact_source",
)
_BALL_CONTACT_HONEST_NAN_RATIONALE = (
    "Ball-contact detection is continuous-only: candidate detection needs a dense ball trajectory "
    "(RDP structural breaks + velocity breaks). A single SB360 freeze frame has no trajectory, so "
    "detect_ball_contacts resolves no contact and every first_contact_* column is all-NaN -- it "
    "declines to fabricate a touch from one snapshot rather than guess (E7). The leg difference is a "
    "data-regime boundary, not a library defect."
)


def _bc_hn() -> AxisVerdict:
    return AxisVerdict("all_nan", "honest_nan", rationale=_BALL_CONTACT_HONEST_NAN_RATIONALE)


_entry(
    "add_ball_contacts",
    C.generic(T.add_ball_contacts),
    columns=_BALL_CONTACT_COLS,
    velocity={c: _bc_hn() for c in _BALL_CONTACT_COLS},
    visibility={
        "gk_absent": {c: _bc_hn() for c in _BALL_CONTACT_COLS},
        "defender_absent": {c: _bc_hn() for c in _BALL_CONTACT_COLS},
        "gk_one_end": {c: _bc_hn() for c in _BALL_CONTACT_COLS},
    },
    applicability={c: "no_support" for c in _BALL_CONTACT_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _BALL_CONTACT_COLS},
)
