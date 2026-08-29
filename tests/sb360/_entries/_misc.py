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

# ADR-054 keeper-identity placement helper. It stamps the opponent keeper's id from a keeper_map
# built from the ACTIONS (identical on both legs), so `defending_gk_player_id` is frame-INDEPENDENT:
# it reads no frame coordinate and produces the same value on the freeze-frame leg and the
# velocity-bearing leg by construction. `works` because the honest verdict for a frame-blind id
# stamp is that neither velocity nor a roster ablation can move it -- observed `identical` on every
# axis (all six rows row_identical) and `no_support` applicability (deltas 0.0/0.0), both re-derived
# by the lock in test_axis_locks.py.
_entry(
    "add_defending_gk_player_id",
    ADAPTERS["add_defending_gk_player_id"](T.add_defending_gk_player_id),
    columns=("defending_gk_player_id",),
    velocity={
        "defending_gk_player_id": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {"defending_gk_player_id": AxisVerdict("identical", "works")},
        "defender_absent": {"defending_gk_player_id": AxisVerdict("identical", "works")},
        "gk_one_end": {"defending_gk_player_id": AxisVerdict("identical", "works")},
    },
    applicability={"defending_gk_player_id": "no_support"},
    applicability_deltas={"defending_gk_player_id": {"extreme": 0.0, "near": 0.0}},
)
