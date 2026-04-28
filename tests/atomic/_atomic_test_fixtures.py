"""Shared Atomic-SPADL action-row fixtures for atomic helper tests.

Underscore-prefixed module name keeps pytest from collecting it as a test file.
Imported by ``test_atomic_add_possessions.py``, ``test_atomic_add_gk_role.py``,
``test_atomic_add_gk_distribution_metrics.py``, ``test_atomic_add_pre_shot_gk_context.py``,
``test_atomic_validate_spadl.py``, and ``test_atomic_cross_validation.py``.

Builders return plain ``dict[str, object]`` rows. ``_df(rows)`` assembles them
into an Atomic-SPADL-shaped ``pd.DataFrame``.

Atomic-SPADL schema differs from standard SPADL on three axes:

- Coordinate columns are ``x, y, dx, dy`` (NOT ``start_x, start_y, end_x, end_y``).
- No ``result_id`` column — outcomes are encoded as follow-up atomic actions
  (``receival`` = pass success, ``interception`` / ``out`` / ``offside`` =
  pass failure variants; ``goal`` / ``owngoal`` for shots; ``yellow_card`` /
  ``red_card`` for fouls).
- 33 action types: 23 standard + 10 atomic-only. Atomic ``_simplify`` collapses
  ``corner_short`` / ``corner_crossed`` → ``corner`` and
  ``freekick_short`` / ``freekick_crossed`` / ``shot_freekick`` → ``freekick``,
  so atomic streams use the collapsed type names.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from silly_kicks.atomic.spadl import config as atomicspadlconfig

_ACT = atomicspadlconfig.actiontype_id
_BP = atomicspadlconfig.bodypart_id


def _make_atomic_action(
    *,
    action_id: int,
    game_id: int = 1,
    period_id: int = 1,
    time_seconds: float = 0.0,
    team_id: int = 100,
    player_id: int = 200,
    type_name: str = "pass",
    bodypart_name: str = "foot",
    x: float = 50.0,
    y: float = 34.0,
    dx: float = 10.0,
    dy: float = 0.0,
    original_event_id: object = None,
) -> dict[str, Any]:
    """Build a minimal valid Atomic-SPADL action row.

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
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "type_id": _ACT[type_name],
        "bodypart_id": _BP[bodypart_name],
    }


def _make_atomic_gk_action(
    *,
    action_id: int,
    keeper_action: str,
    player_id: int = 999,
    team_id: int = 100,
    x: float = 5.0,
    y: float = 34.0,
    dx: float = 0.0,
    dy: float = 0.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build an atomic GK action row.

    ``keeper_action`` must be one of ``keeper_save``, ``keeper_claim``,
    ``keeper_punch``, ``keeper_pick_up``. Defaults place the GK near their
    own goal line (``x=5.0``, well inside the penalty area), with
    ``player_id=999`` to mark it as the canonical "the GK" of team 100.

    Any ``_make_atomic_action`` kwarg can be overridden via ``**overrides``.
    """
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=keeper_action,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        bodypart_name="other",
    )
    base.update(overrides)
    return base


def _make_atomic_shot_action(
    *,
    action_id: int,
    player_id: int = 700,
    team_id: int = 200,
    x: float = 95.0,
    y: float = 34.0,
    dx: float = 10.0,
    dy: float = 0.0,
    shot_type: str = "shot",
    **overrides: Any,
) -> dict[str, Any]:
    """Build an atomic shot action row.

    ``shot_type`` should be ``shot`` or ``shot_penalty``. (Atomic ``_simplify``
    collapses ``shot_freekick`` into ``freekick``; tests use ``shot``,
    ``shot_penalty``, or ``freekick`` directly.)

    Defaults place an attacker (team 200) shooting at the goal line of team 100.
    Outcome is encoded by appending a ``goal`` / ``owngoal`` / ``out`` follow-up
    action — the shot row itself has no ``result_id`` (atomic schema has no
    such column).
    """
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=shot_type,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        bodypart_name="foot",
    )
    base.update(overrides)
    return base


def _make_atomic_pass_action(
    *,
    action_id: int,
    player_id: int = 200,
    team_id: int = 100,
    x: float = 50.0,
    y: float = 34.0,
    dx: float = 10.0,
    dy: float = 0.0,
    pass_type: str = "pass",
    **overrides: Any,
) -> dict[str, Any]:
    """Build an atomic pass action row.

    ``pass_type`` may be ``pass``, ``cross``, ``goalkick``, ``throw_in``, or
    the atomic-collapsed set-piece names ``corner`` / ``freekick``. (Standard
    SPADL ``corner_short`` / ``corner_crossed`` / ``freekick_short`` /
    ``freekick_crossed`` collapse to ``corner`` / ``freekick`` in atomic.)
    """
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=pass_type,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
    )
    base.update(overrides)
    return base


def _make_atomic_receival(
    *,
    action_id: int,
    player_id: int = 201,
    team_id: int = 100,
    x: float = 60.0,
    y: float = 34.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a receival follow-up action — encodes "preceding pass succeeded".

    Defaults place the receiver as a teammate of the passer (team 100,
    player 201 — different from the canonical passer player 200).
    Receival actions have ``dx=dy=0`` (the ball arrived; this row marks the
    arrival, not motion).
    """
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name="receival",
        x=x,
        y=y,
        dx=0.0,
        dy=0.0,
    )
    base.update(overrides)
    return base


def _make_atomic_interception(
    *,
    action_id: int,
    player_id: int = 300,
    team_id: int = 200,
    x: float = 60.0,
    y: float = 34.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build an interception follow-up — encodes "preceding pass intercepted".

    Defaults place the intercepting player on the opposing team (team 200).
    """
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name="interception",
        x=x,
        y=y,
        dx=0.0,
        dy=0.0,
    )
    base.update(overrides)
    return base


def _make_atomic_out(
    *,
    action_id: int,
    player_id: int = 200,
    team_id: int = 100,
    x: float = 80.0,
    y: float = 0.5,
    **overrides: Any,
) -> dict[str, Any]:
    """Build an out follow-up — encodes "preceding pass went out of play"."""
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name="out",
        x=x,
        y=y,
        dx=0.0,
        dy=0.0,
    )
    base.update(overrides)
    return base


def _make_atomic_offside(
    *,
    action_id: int,
    player_id: int = 200,
    team_id: int = 100,
    x: float = 70.0,
    y: float = 34.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build an offside follow-up — encodes "preceding pass was offside"."""
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name="offside",
        x=x,
        y=y,
        dx=0.0,
        dy=0.0,
    )
    base.update(overrides)
    return base


def _make_atomic_card(
    *,
    action_id: int,
    card: str,
    player_id: int = 400,
    team_id: int = 100,
    x: float = 50.0,
    y: float = 34.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a yellow_card / red_card follow-up action.

    ``card`` must be ``yellow_card`` or ``red_card``. Cards are emitted as
    follow-up rows after a foul; in atomic-SPADL they don't affect possession
    boundaries by themselves.
    """
    base = _make_atomic_action(
        action_id=action_id,
        team_id=team_id,
        player_id=player_id,
        type_name=card,
        x=x,
        y=y,
        dx=0.0,
        dy=0.0,
    )
    base.update(overrides)
    return base


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an Atomic-SPADL DataFrame from a list of action dicts."""
    return pd.DataFrame(rows)
