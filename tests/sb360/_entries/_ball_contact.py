"""SB360 verdict -- TF-79 ball-contact family.

Observations are TRANSCRIBED FROM EXECUTION; only a human writes a rationale.

``add_ball_contacts`` is trajectory-only: candidate detection needs a dense ball trajectory, so a
single SB360 freeze frame resolves NO contact and every ``first_contact_*`` column is all-NaN by
design (honest-NaN, not a defect). Same data-regime boundary as ``add_elastic_sync``.

``first_contact_is_aerial`` is the z-gated boolean (ADR-125): SB360 z is absent, so it is both-leg
all-NaN, which the harness classes ``no_signal`` (admits only ``not_exercised``). The other eight
columns class ``all_nan`` -> ``honest_nan``: the tracking leg resolves one contact and the freeze-frame
leg none (detector v2 reports the admitting |Δv|, ADR-134, so ``first_contact_velocity_change_ms`` joins
them). Under ``gk_absent`` that one contact does not resolve on either leg (v2 credits only a player in
reach, with no fallback radius), so all nine columns are ``no_signal`` there (measured).
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
#: The column observed ``no_signal`` (both legs all-NaN) on every roster: the z-gated boolean, which
#: carries no value on SB360 (z absent).
_BALL_CONTACT_NO_SIGNAL_COLS = ("first_contact_is_aerial",)
_BALL_CONTACT_HONEST_NAN_RATIONALE = (
    "Ball-contact detection is continuous-only: a candidate is a change of the ball's mean velocity "
    "over a window of samples (ADR-134). A single SB360 freeze frame has no trajectory, so "
    "detect_ball_contacts resolves no contact and every first_contact_* column is all-NaN on that leg "
    "-- it declines to fabricate a touch from one snapshot rather than guess (E7). The leg difference is "
    "a data-regime boundary, not a library defect."
)
_BALL_CONTACT_NOT_EXERCISED_RATIONALE = (
    "On a single SB360 freeze frame detect_ball_contacts resolves no contact (no trajectory), and this "
    "column carries nothing on the tracking leg either (the z-gated is_aerial: SB360 z is absent), so it "
    "is all-NaN in BOTH legs -- the harness classes a both-NaN column no_signal, which admits only "
    "not_exercised. Multi-frame tracking with measured z would exercise it."
)
_BALL_CONTACT_GK_ABSENT_RATIONALE = (
    "Measured: with the keeper removed, the one contact that the tracking leg resolves on the other "
    "rosters does not resolve (detector v2 credits only a player in reach, with no fallback radius, "
    "ADR-134), so every first_contact_* column is all-NaN in BOTH legs -> no_signal -> not_exercised. "
    "A fixture limit on this roster, not a library property; multi-frame tracking exercises it."
)


def _bc_hn() -> AxisVerdict:
    return AxisVerdict("all_nan", "honest_nan", rationale=_BALL_CONTACT_HONEST_NAN_RATIONALE)


def _bc_ne() -> AxisVerdict:
    return AxisVerdict("no_signal", "not_exercised", rationale=_BALL_CONTACT_NOT_EXERCISED_RATIONALE)


def _bc_block() -> dict[str, AxisVerdict]:
    return {c: (_bc_ne() if c in _BALL_CONTACT_NO_SIGNAL_COLS else _bc_hn()) for c in _BALL_CONTACT_COLS}


def _bc_gk_absent_block() -> dict[str, AxisVerdict]:
    return {
        c: (
            _bc_ne()
            if c in _BALL_CONTACT_NO_SIGNAL_COLS
            else AxisVerdict("no_signal", "not_exercised", rationale=_BALL_CONTACT_GK_ABSENT_RATIONALE)
        )
        for c in _BALL_CONTACT_COLS
    }


_entry(
    "add_ball_contacts",
    C.generic(T.add_ball_contacts),
    columns=_BALL_CONTACT_COLS,
    velocity=_bc_block(),
    visibility={
        "gk_absent": _bc_gk_absent_block(),
        "defender_absent": _bc_block(),
        "gk_one_end": _bc_block(),
    },
    applicability={c: "no_support" for c in _BALL_CONTACT_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _BALL_CONTACT_COLS},
)
