"""Synthetic SkillCorner dynamic-event scenes for the derived restart and penalty rows (spec §4.2, §4.17).

Coordinates are SkillCorner's centred metres on a 105 x 68 pitch, in the possession team's attacking frame (the
converter's POSSESSION_PERSPECTIVE). Times are seconds of period 1 unless a row sets ``period``.
"""

from __future__ import annotations

import pandas as pd

META = {
    "id": 4242,
    "pitch_length": 105,
    "pitch_width": 68,
    "home_team": {"id": "H", "name": "Home"},
    "away_team": {"id": "A", "name": "Away"},
}

_COLUMNS = [
    "event_type",
    "event_id",
    "period",
    "time_start",
    "time_end",
    "team_id",
    "player_id",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "start_type",
    "end_type",
    "game_interruption_before",
    "game_interruption_after",
    "is_header",
    "hand_pass",
    "player_targeted_x_reception",
    "player_targeted_y_reception",
]


def _clock(seconds: float, period: int) -> str:
    total = seconds + {1: 0.0, 2: 2700.0, 3: 5400.0, 4: 6300.0}[period]
    return f"{int(total // 60):02d}:{total % 60:04.1f}"


def pp(
    eid: str,
    t_start: float,
    t_end: float,
    team: str,
    player: str,
    start: tuple[float, float] = (0.0, 0.0),
    end: tuple[float, float] = (5.0, 0.0),
    *,
    end_type: str = "pass",
    start_type: str = "pass_reception",
    gib: str | None = None,
    gia: str | None = None,
    period: int = 1,
    **extra,
) -> dict:
    """One ``player_possession`` row."""
    row = {
        "event_type": "player_possession",
        "event_id": eid,
        "period": period,
        "time_start": _clock(t_start, period),
        "time_end": _clock(t_end, period),
        "team_id": team,
        "player_id": player,
        "x_start": start[0],
        "y_start": start[1],
        "x_end": end[0],
        "y_end": end[1],
        "start_type": start_type,
        "end_type": end_type,
        "game_interruption_before": gib,
        "game_interruption_after": gia,
        "is_header": False,
        "hand_pass": False,
        "player_targeted_x_reception": None,
        "player_targeted_y_reception": None,
    }
    row.update(extra)
    return row


def events(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for c in _COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def convert(*rows: dict, meta: dict | None = None, **kw) -> pd.DataFrame:
    from silly_kicks.spadl.skillcorner import convert_to_actions

    actions, _ = convert_to_actions(events(*rows), meta or META, **kw)
    return actions


def derived_restarts(actions: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl.schema import restart_kind

    kinds = actions["type_id"].map(restart_kind)
    mask = (actions["action_provenance"] == "derived") & kinds.isin(
        ["corner", "free_kick", "goal_kick", "throw_in", "penalty"]
    )
    return actions[mask].reset_index(drop=True)
