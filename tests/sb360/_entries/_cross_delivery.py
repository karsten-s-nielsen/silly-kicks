"""SB360 verdict -- TF-86 cross delivery (ADR-133).

Regenerated with the committed tools (``python tests/sb360/_regenerate.py`` then ``python tests/sb360/_adjudicate.py``):
the observations are measured, and the verdicts and rationales come from the ``_adjudicate`` rules. The block is
kept in its own family file, like the other per-feature entries.

``add_cross_deliveries`` judges a delivery from the ball trajectory. A single SB360 freeze frame per action has
none, so delivery rows are unknown (``<NA>``) on that leg and ``cross_attempt`` observes ``partial_nan``;
``cross_height`` needs an attempt, so it is both-NaN (``no_signal``).
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import AxisVerdict, _entry

_entry(
    "add_cross_deliveries",
    C.generic(T.add_cross_deliveries),
    columns=(
        "cross_attempt",
        "cross_height",
    ),
    velocity={
        "cross_attempt": AxisVerdict(
            "partial_nan",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so the "
                "metric is defined for some actions and not others; the NaNs are honest absences rather than "
                "fabricated values. [measured cause=frame_count]"
            ),
        ),
        "cross_height": AxisVerdict(
            "no_signal",
            "not_exercised",
            rationale=(
                "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                "occurrence context, or blocking defender to score). A fixture inadequacy, not a library property "
                "-- widening the fixture would move it. [measured cause=velocity+frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "cross_attempt": AxisVerdict(
                "partial_nan",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so "
                    "the metric is defined for some actions and not others; the NaNs are honest absences rather "
                    "than fabricated values. [measured cause=frame_count]"
                ),
            ),
            "cross_height": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "cross_attempt": AxisVerdict(
                "partial_nan",
                "differs_by_design",
                rationale=(
                    "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single sample, so "
                    "the metric is defined for some actions and not others; the NaNs are honest absences rather "
                    "than fabricated values. [measured cause=frame_count]"
                ),
            ),
            "cross_height": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "gk_one_end": {
            "cross_attempt": AxisVerdict("identical", "works"),
            "cross_height": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
    },
    applicability={
        "cross_attempt": "no_support",
        "cross_height": "no_support",
    },
    applicability_deltas={
        "cross_attempt": {"extreme": 0.0, "near": 0.0},
        "cross_height": {"extreme": 0.0, "near": 0.0},
    },
)
