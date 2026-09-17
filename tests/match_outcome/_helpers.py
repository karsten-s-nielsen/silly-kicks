"""Shared helpers for TF-53 match_outcome tests (real SPADL type_id/result_id + injected xg).

A uniquely-named module (NOT ``conftest``) imported by full path -- a bare ``from conftest import``
resolves to whichever sibling ``conftest.py`` is first on sys.path when the whole suite runs together.
Mirrors ``tests/team_metrics/_helpers.py``.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig

SHOT = spadlconfig.actiontype_id["shot"]
SHOT_PENALTY = spadlconfig.actiontype_id["shot_penalty"]
SHOT_FREEKICK = spadlconfig.actiontype_id["shot_freekick"]
PASS = spadlconfig.actiontype_id["pass"]
DRIBBLE = spadlconfig.actiontype_id["dribble"]
BAD_TOUCH = spadlconfig.actiontype_id["bad_touch"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]
OWNGOAL = spadlconfig.result_id["owngoal"]


def make_actions(records: list[dict], *, game_id=1) -> pd.DataFrame:
    """Build a SPADL ``actions`` frame (with an ``xg`` column) from partial records.

    Each record needs at least ``team_id``, ``type_id``. Optional: ``xg`` (NaN), ``result_id``
    (success), ``period_id`` (1), ``time_seconds`` (row index), ``start_x`` (50), ``start_y`` (34),
    ``end_x`` (=start_x), ``end_y`` (=start_y), ``possession_id`` (row index), ``player_id`` (row idx).
    """
    df = pd.DataFrame(records).copy()
    df["game_id"] = game_id

    def _fill(col, default):
        df[col] = df[col].fillna(default) if col in df.columns else default

    _fill("period_id", 1)
    _fill("result_id", SUCCESS)
    _fill("xg", float("nan"))
    df["time_seconds"] = df["time_seconds"] if "time_seconds" in df.columns else range(len(df))
    _fill("start_x", 50.0)
    _fill("start_y", 34.0)
    df["end_x"] = df["end_x"].fillna(df["start_x"]) if "end_x" in df.columns else df["start_x"]
    df["end_y"] = df["end_y"].fillna(df["start_y"]) if "end_y" in df.columns else df["start_y"]
    df["player_id"] = df["player_id"] if "player_id" in df.columns else range(len(df))
    df["action_id"] = range(len(df))
    return df


def rows_by(samples: pd.DataFrame, key: str = "team_id") -> dict:
    """Row-dicts keyed by ``key`` -- ``Any``-typed scalar access (avoids pyright's ``Series`` union on
    ``DataFrame.loc[label]`` row indexing, which otherwise makes every ``abs(...) < eps`` a
    ``Series[bool]`` conditional)."""
    return samples.set_index(key).to_dict("index")
