"""SB360 verdict -- TF-79 ball-contact family.

Observations are TRANSCRIBED FROM EXECUTION; only a human writes a rationale.

``add_ball_contacts`` is trajectory-only: candidate detection needs a dense ball trajectory, so a
single SB360 freeze frame resolves NO contact and every ``first_contact_*`` column is all-NaN by
design (honest-NaN, not a defect). Same data-regime boundary as ``add_elastic_sync``.

Two of the nine columns -- ``first_contact_velocity_change_ms`` and ``first_contact_is_aerial`` --
are pure signal columns whose both-leg all-NaN the harness classes ``no_signal`` (which admits only
``not_exercised``); the other seven class ``all_nan`` -> ``honest_nan``. Same underlying honest-NaN,
different harness class (measured; ADR-125 folded ``is_aerial`` onto the z-provenance gate).
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
    "first_contact_aerial_source",
    "first_contact_source",
)
#: The two columns observed ``no_signal`` (both legs all-NaN) rather than ``all_nan`` -- a pure
#: signal (|Δv|) and the z-gated boolean, which carry no value at all when no contact resolves.
_BALL_CONTACT_NO_SIGNAL_COLS = (
    "first_contact_velocity_change_ms",
    "first_contact_is_aerial",
)
_BALL_CONTACT_HONEST_NAN_RATIONALE = (
    "Ball-contact detection is continuous-only: candidate detection needs a dense ball trajectory "
    "(RDP structural breaks + velocity breaks). A single SB360 freeze frame has no trajectory, so "
    "detect_ball_contacts resolves no contact and every first_contact_* column is all-NaN -- it "
    "declines to fabricate a touch from one snapshot rather than guess (E7). The leg difference is a "
    "data-regime boundary, not a library defect."
)
_BALL_CONTACT_NOT_EXERCISED_RATIONALE = (
    "On a single SB360 freeze frame detect_ball_contacts resolves no contact (no trajectory), so "
    "this signal column is all-NaN in BOTH legs -- the harness classes a both-NaN column no_signal, "
    "which admits only not_exercised. A continuous-only data-regime boundary, not a library "
    "property; multi-frame tracking would exercise it."
)


def _bc_hn() -> AxisVerdict:
    return AxisVerdict("all_nan", "honest_nan", rationale=_BALL_CONTACT_HONEST_NAN_RATIONALE)


def _bc_ne() -> AxisVerdict:
    return AxisVerdict("no_signal", "not_exercised", rationale=_BALL_CONTACT_NOT_EXERCISED_RATIONALE)


def _bc_block() -> dict[str, AxisVerdict]:
    return {c: (_bc_ne() if c in _BALL_CONTACT_NO_SIGNAL_COLS else _bc_hn()) for c in _BALL_CONTACT_COLS}


_entry(
    "add_ball_contacts",
    C.generic(T.add_ball_contacts),
    columns=_BALL_CONTACT_COLS,
    velocity=_bc_block(),
    visibility={
        "gk_absent": _bc_block(),
        "defender_absent": _bc_block(),
        "gk_one_end": _bc_block(),
    },
    applicability={c: "no_support" for c in _BALL_CONTACT_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _BALL_CONTACT_COLS},
)
