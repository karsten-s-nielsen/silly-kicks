"""Shared helpers for TF-52 team_metrics tests (real SPADL type_id/result_id).

A uniquely-named module (NOT ``conftest``) imported by full path -- a bare ``from conftest import``
resolves to whichever sibling ``conftest.py`` is first on sys.path when the whole suite runs together
(``tests/scripts/conftest.py`` shadowed this one). Mirrors ``tests/restdefense/_fixtures.py``.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig

PASS = spadlconfig.actiontype_id["pass"]
CROSS = spadlconfig.actiontype_id["cross"]
TACKLE = spadlconfig.actiontype_id["tackle"]
INTERCEPTION = spadlconfig.actiontype_id["interception"]
FOUL = spadlconfig.actiontype_id["foul"]
SHOT = spadlconfig.actiontype_id["shot"]
SHOT_PENALTY = spadlconfig.actiontype_id["shot_penalty"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
DRIBBLE = spadlconfig.actiontype_id["dribble"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]

FL = spadlconfig.field_length
FW = spadlconfig.field_width


def make_actions(records: list[dict], *, game_id=1) -> pd.DataFrame:
    """Build a full SPADL ``actions`` frame from partial records, filling column defaults.

    Each record needs at least ``team_id``, ``type_id``, ``time_seconds``, ``start_x``. Optional:
    ``period_id`` (default 1), ``result_id`` (default success), ``start_y`` (34), ``end_x`` (=start_x),
    ``end_y`` (=start_y), ``player_id`` (row index).
    """
    df = pd.DataFrame(records).copy()
    df["game_id"] = game_id

    def _fill(col, default):
        # A record may specify a column for only SOME rows; fill the rest per-row (never skip).
        df[col] = df[col].fillna(default) if col in df.columns else default

    _fill("period_id", 1)
    _fill("result_id", SUCCESS)
    _fill("start_y", 34.0)  # must precede end_y (which defaults to start_y)
    df["end_x"] = df["end_x"].fillna(df["start_x"]) if "end_x" in df.columns else df["start_x"]
    df["end_y"] = df["end_y"].fillna(df["start_y"]) if "end_y" in df.columns else df["start_y"]
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].fillna(pd.Series(range(len(df)), index=df.index))
    else:
        df["player_id"] = range(len(df))
    df["action_id"] = range(len(df))
    return df
