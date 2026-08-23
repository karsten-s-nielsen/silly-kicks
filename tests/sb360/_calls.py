"""Per-aggregator call adapters -- re-export shim.

The adapter bodies were MOVED to ``scripts/_sb_battery.py`` so the ADR-053 audit (this package) and
the licensed-corpus validation driver (``scripts/``) resolve ONE call convention (round-4 review).
This shim keeps ``tests.sb360._calls`` importable with the SAME names, so the committed
``tests/sb360/_entries/*.py`` (which the generator emits with ``from tests.sb360 import _calls as C``
and calls as ``C.generic(...)``) stay byte-identical across the move.

Layering: ``tests -> scripts``. See ``scripts/_sb_battery.py`` for the adapter bodies and the reason
they are single-sourced there.
"""

from __future__ import annotations

from scripts._sb_battery import (
    audit_xt,
    defensive_credit,
    generic,
    gradientsports_player_ids,
    pre_shot_gk_angle,
    pre_shot_gk_position,
    sync_score,
    visible_area_coverage,
    with_pressure_methods,
    with_xt,
    with_xt_keyword,
)

__all__ = [
    "audit_xt",
    "defensive_credit",
    "generic",
    "gradientsports_player_ids",
    "pre_shot_gk_angle",
    "pre_shot_gk_position",
    "sync_score",
    "visible_area_coverage",
    "with_pressure_methods",
    "with_xt",
    "with_xt_keyword",
]
