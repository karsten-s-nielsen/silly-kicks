"""Opt-in loud-guard for MarkovPossessionValue.fit inputs (ADR-036 §11, G5).

House style: one diagnosis object (cf. validate_time_base ADR-017, validate_id_dtypes ADR-019),
the natural home for the attack-orientation guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import silly_kicks.spadl.config as spadlconfig

_REQUIRED = [
    "game_id",
    "period_id",
    "action_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


@dataclass(frozen=True)
class PossessionValueInputDiagnosis:
    ok: bool
    problems: list[str] = field(default_factory=list)


def validate_possession_value_input(
    actions: pd.DataFrame,
    *,
    xg_column: str,
    pressure_column: str,
    require_possession_id: bool = False,
) -> PossessionValueInputDiagnosis:
    problems: list[str] = []
    for c in [*_REQUIRED, xg_column, pressure_column]:
        if c not in actions.columns:
            problems.append(f"missing required column: {c!r}")
    if require_possession_id and "possession_id" not in actions.columns:
        problems.append("missing 'possession_id' (call spadl.add_possessions first)")
    if "type_id" in actions.columns and "start_x" in actions.columns:
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        if len(shots) >= 10:
            frac_far = (shots["start_x"] > spadlconfig.field_length / 2).mean()
            if frac_far < 0.5:
                problems.append(
                    f"orientation: only {frac_far:.0%} of shots are in the attacking half; "
                    f"actions must be attack-LTR (attack toward x=105) — ADR-028/§M4"
                )
    return PossessionValueInputDiagnosis(ok=(len(problems) == 0), problems=problems)
