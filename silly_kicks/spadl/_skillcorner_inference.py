"""SkillCorner derived action inference.

Produces SPADL actions that are not directly mapped from ``player_possession``
dispatch but inferred from cross-referencing event types:

- Defensive actions (interceptions, tackles) from ``start_type`` + OBE
- Keeper saves from shot -> opponent-possession sequences

All returned DataFrames have partial SPADL columns (at minimum:
``period_id``, ``time_seconds``, ``team_id``, ``player_id``, ``start_x``,
``start_y``, ``type_id``, ``result_id``, ``bodypart_id``,
``action_provenance``). The caller merges them into the main action stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as spadlconfig

_OBE_TEMPORAL_WINDOW: float = 2.0  # seconds


def _is_defensive_start_type(start_type: pd.Series) -> pd.Series:
    """True for start_type values that indicate a possession win."""
    return start_type.str.endswith("_interception", na=False) | (start_type == "recovery")


def infer_defensive_actions(
    pp: pd.DataFrame,
    obe: pd.DataFrame,
) -> pd.DataFrame:
    """Infer interceptions and tackles from start_type + OBE cross-referencing.

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows, sorted chronologically.
        Required columns: ``event_id``, ``period``, ``time_seconds``,
        ``team_id``, ``player_id``, ``start_type``, ``x_start``, ``y_start``.
    obe : pd.DataFrame
        ``on_ball_engagement`` rows.
        Required columns: ``period``, ``time_seconds``, ``team_id``,
        ``player_id``, ``end_type``, ``x_start``, ``y_start``.

    Returns
    -------
    pd.DataFrame
        Derived defensive actions with partial SPADL columns.
    """
    defensive_mask = _is_defensive_start_type(pp["start_type"])
    if not defensive_mask.any():
        return pd.DataFrame()

    rows: list[dict] = []
    obe_regains = obe[obe["end_type"] == "direct_regain"] if len(obe) > 0 else pd.DataFrame()

    for idx in pp.index[defensive_mask]:
        row = pp.loc[idx]
        period = row["period"]
        t = row["time_seconds"]

        # Default: interception from start_type
        action_type = spadlconfig.actiontype_id["interception"]
        player = row["player_id"]
        team = row["team_id"]
        x = row["x_start"]
        y = row["y_start"]

        # OBE upgrade: check for direct_regain within temporal window + same team
        if len(obe_regains) > 0:
            candidates = obe_regains[
                (obe_regains["period"] == period)
                & (obe_regains["team_id"] == row["team_id"])
                & ((obe_regains["time_seconds"] - t).abs() <= _OBE_TEMPORAL_WINDOW)
            ]
            if len(candidates) > 0:
                # Order-insensitive pick (ADR-065): nearest in time, ties broken by CONTENT columns
                # that are always present on an OBE row (``time_seconds`` then ``player_id``), NOT by
                # positional ``argmin()`` (first-on-tie in obe's INPUT row order, so two equidistant
                # same-team ``direct_regain`` rows would flip the tackle's attributed
                # player/team/coords when the input is permuted). ``event_id`` is deliberately NOT
                # used: it is not a guaranteed OBE column at this seam.
                best = (
                    candidates.assign(_dt=(candidates["time_seconds"] - t).abs())
                    .sort_values(["_dt", "time_seconds", "player_id"], kind="mergesort")
                    .iloc[0]
                )
                action_type = spadlconfig.actiontype_id["tackle"]
                player = best["player_id"]
                team = best["team_id"]
                x = best["x_start"]
                y = best["y_start"]

        rows.append(
            {
                "event_id": row["event_id"],
                "period_id": int(period),
                "time_seconds": float(t) - 0.01,  # just before the native action
                "team_id": team,
                "player_id": player,
                "start_x": float(x),
                "start_y": float(y),
                "end_x": float(x),
                "end_y": float(y),
                "type_id": action_type,
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": spadlconfig.bodypart_id["foot"],
                "action_provenance": "derived",
            }
        )

    return pd.DataFrame(rows)


def infer_keeper_saves(pp: pd.DataFrame) -> pd.DataFrame:
    """Infer keeper saves from shot -> opponent-possession sequences.

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows, sorted chronologically.
        Required columns: ``period``, ``time_seconds``, ``team_id``,
        ``player_id``, ``end_type``, ``x_start``, ``y_start``.
        Optional: ``game_interruption_after``.

    Returns
    -------
    pd.DataFrame
        Derived keeper_save actions with partial SPADL columns.

    Notes
    -----
    Keeper save is attributed to whoever starts the next possession, which is
    typically an outfield player (e.g. CB taking the goal kick), NOT the actual
    GK. This is a data limitation -- SkillCorner does not tag saves natively.
    """
    rows: list[dict] = []
    end_types = pp["end_type"].to_numpy()
    team_ids = pp["team_id"].to_numpy()
    gia = (
        pp["game_interruption_after"].to_numpy() if "game_interruption_after" in pp.columns else np.full(len(pp), None)
    )

    for i in range(len(pp) - 1):
        if end_types[i] != "shot":
            continue
        # Only infer save for on-target shots: gi_after is NaN (plausible save)
        # or corner_for (deflected behind goal line by GK/defender).
        # Skip: goal_for (scored), goal_kick_against (missed wide/over bar),
        # free_kick_against (foul), throw_in_* (unusual -- not a save).
        if gia[i] == "goal_for":
            continue
        if not (pd.isna(gia[i]) or gia[i] == "corner_for"):
            continue
        if team_ids[i] == team_ids[i + 1]:
            continue

        next_row = pp.iloc[i + 1]
        rows.append(
            {
                "period_id": int(next_row["period"]),
                "time_seconds": float(next_row["time_seconds"]) - 0.01,
                "team_id": next_row["team_id"],
                "player_id": next_row["player_id"],
                "start_x": float(next_row["x_start"]),
                "start_y": float(next_row["y_start"]),
                "end_x": float(next_row["x_start"]),
                "end_y": float(next_row["y_start"]),
                "type_id": spadlconfig.actiontype_id["keeper_save"],
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": spadlconfig.bodypart_id["foot"],
                "action_provenance": "derived",
            }
        )

    return pd.DataFrame(rows)
