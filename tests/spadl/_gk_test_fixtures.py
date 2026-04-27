"""Shared SPADL action-row fixtures for GK helper tests.

Underscore-prefixed module name keeps pytest from collecting it as a test file.
Imported by ``test_add_gk_role.py``, ``test_add_gk_distribution_metrics.py``,
and ``test_add_pre_shot_gk_context.py``.

Builders return plain ``dict[str, object]`` rows. ``_df(rows)`` assembles them
into a SPADL-shaped ``pd.DataFrame``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_ACT = spadlconfig.actiontype_id
_RES = spadlconfig.result_id
_BP = spadlconfig.bodypart_id


def _make_action(
    *,
    action_id: int,
    game_id: int = 1,
    period_id: int = 1,
    time_seconds: float = 0.0,
    team_id: int = 100,
    player_id: int = 200,
    type_name: str = "pass",
    result_name: str = "success",
    bodypart_name: str = "foot",
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    original_event_id: object = None,
) -> dict[str, Any]:
    """Build a minimal valid SPADL action row.

    All keyword-only kwargs default to a "central pass by team 100, player 200"
    so tests only need to override the columns they care about.
    """
    return {
        "game_id": game_id,
        "original_event_id": str(action_id) if original_event_id is None else original_event_id,
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": time_seconds,
        "team_id": team_id,
        "player_id": player_id,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "type_id": _ACT[type_name],
        "result_id": _RES[result_name],
        "bodypart_id": _BP[bodypart_name],
    }


def _make_gk_action(
    *,
    action_id: int,
    keeper_action: str,
    player_id: int = 999,
    team_id: int = 100,
    start_x: float = 5.0,
    start_y: float = 34.0,
    end_x: float = 5.0,
    end_y: float = 34.0,
    result_name: str = "success",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a GK action row.

    ``keeper_action`` must be one of ``keeper_save``, ``keeper_claim``,
    ``keeper_punch``, ``keeper_pick_up``. Defaults place the GK near their
    own goal line (``start_x=5.0``, well inside the penalty area), with
    ``player_id=999`` to mark it as the canonical "the GK" of team 100.

    Any ``_make_action`` kwarg can be overridden via ``**overrides``.
    """
    base = _make_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=keeper_action,
        result_name=result_name,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        bodypart_name="other",
    )
    base.update(overrides)
    return base


def _make_shot_action(
    *,
    action_id: int,
    player_id: int = 700,
    team_id: int = 200,
    start_x: float = 95.0,
    start_y: float = 34.0,
    end_x: float = 105.0,
    end_y: float = 34.0,
    result_name: str = "fail",
    shot_type: str = "shot",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a shot action row.

    ``shot_type`` must be one of ``shot``, ``shot_freekick``, ``shot_penalty``.
    Defaults place an attacker (team 200) shooting at the goal line of team 100.
    ``result_name="fail"`` because most shots miss; override to ``success`` for goal.
    """
    base = _make_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=shot_type,
        result_name=result_name,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        bodypart_name="foot",
    )
    base.update(overrides)
    return base


def _make_pass_action(
    *,
    action_id: int,
    player_id: int = 200,
    team_id: int = 100,
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    pass_type: str = "pass",
    result_name: str = "success",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a pass action row.

    ``pass_type`` may be ``pass``, ``cross``, ``goalkick``, ``freekick_short``,
    ``freekick_crossed``, ``corner_short``, ``corner_crossed``, ``throw_in``.
    """
    base = _make_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=pass_type,
        result_name=result_name,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
    )
    base.update(overrides)
    return base


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a SPADL DataFrame from a list of action dicts."""
    return pd.DataFrame(rows)
