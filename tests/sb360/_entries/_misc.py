"""SB360 verdicts -- misc family.

Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a
human writes an adjudication or a rationale.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry

_entry(
    "add_visible_area_coverage",
    ADAPTERS["add_visible_area_coverage"](T.add_visible_area_coverage),
    columns=(
        "visible_area_fraction",
        "visible_area_source",
    ),
    velocity={
        "visible_area_fraction": AxisVerdict("identical", "works"),
        "visible_area_source": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "visible_area_fraction": AxisVerdict("identical", "works"),
            "visible_area_source": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "visible_area_fraction": AxisVerdict("identical", "works"),
            "visible_area_source": AxisVerdict("identical", "works"),
        },
        "gk_one_end": {
            "visible_area_fraction": AxisVerdict("identical", "works"),
            "visible_area_source": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "visible_area_fraction": "no_support",
        "visible_area_source": "no_support",
    },
    applicability_deltas={
        "visible_area_fraction": {"extreme": 0.0, "near": 0.0},
        "visible_area_source": {"extreme": 0.0, "near": 0.0},
    },
)
