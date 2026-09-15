"""Possession-foundation layer for TF-52 team KPIs -- derived ONCE over ``spadl.add_possessions``.

Reuses the single validated possession primitive (never forks one; Chesterton's Fence) and derives the
shared layer every KPI family consumes: possession spells, transition events (recovery = regain,
turnover = loss), team-in-possession, and possession-minutes per team.

Definitions (spec Section 3.3):
- A possession's ``duration_s`` = ``time_seconds(last) - time_seconds(first)`` within it (period-relative).
- ``team_in_possession`` = the team owning the majority of the possession's non-NA ``team_id`` actions
  (ties -> the first action's team; an all-NA possession -> ``pd.NA``).
- ``is_open_play`` = the possession's first action is not a set-piece restart.
- ``is_recovery`` = ``team_in_possession`` differs (``id_compat.ids_differ``) from the previous
  possession's team, within the same game+period (the first possession of a period is not a recovery).
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import canonical_id_series, ids_differ
from silly_kicks.spadl import add_possessions
from silly_kicks.spadl import config as spadlconfig

#: Possession-start action types that mark a NON-open-play (set-piece) possession.
_SET_PIECE_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[name]
    for name in (
        "throw_in",
        "freekick_crossed",
        "freekick_short",
        "corner_crossed",
        "corner_short",
        "goalkick",
        "shot_freekick",
        "shot_penalty",
    )
)


def add_possession_context(actions: pd.DataFrame, *, params) -> pd.DataFrame:
    """Return a chronologically-sorted COPY of ``actions`` + ``possession_id`` + ``team_in_possession``.

    ``possession_id`` comes from :func:`spadl.add_possessions` (the params map to its real kwargs);
    ``team_in_possession`` is the per-possession owner (majority team, raw id).
    """
    out = add_possessions(
        actions,
        max_gap_seconds=params.possession_max_gap_seconds,
        retain_on_set_pieces=params.possession_retain_on_set_pieces,
    )
    out["team_in_possession"] = _owner_team_by_possession(out)
    return out


def _owner_team_by_possession(actions_ctx: pd.DataFrame) -> pd.Series:
    """Broadcast the per-``(game_id, possession_id)`` owner team (majority, raw id) to every row."""

    def _owner(team_series: pd.Series):
        vals = team_series.dropna()
        if vals.empty:
            return pd.NA
        canon = canonical_id_series(vals).to_numpy()
        raw = vals.to_numpy()
        order = list(dict.fromkeys(canon.tolist()))  # first-appearance order of canonical ids
        best = max(order, key=lambda c: int((canon == c).sum()))  # max ties -> first-appearing
        return raw[canon == best][0]

    owner = actions_ctx.groupby(["game_id", "possession_id"], sort=False)["team_id"].apply(_owner)
    mi = pd.MultiIndex.from_frame(actions_ctx[["game_id", "possession_id"]])
    return pd.Series(owner.reindex(mi).to_numpy(), index=actions_ctx.index)


def build_spells(actions_ctx: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(game_id, possession_id)`` with the derived spell fields (see module docstring)."""
    agg = (
        actions_ctx.groupby(["game_id", "possession_id"], sort=True)
        .agg(
            team_id=("team_in_possession", "first"),
            period_id=("period_id", "first"),
            start_time=("time_seconds", "min"),
            end_time=("time_seconds", "max"),
            n_actions=("time_seconds", "size"),
            first_type=("type_id", "first"),
            start_x_ltr=("start_x", "first"),
            last_x_ltr=("start_x", "last"),  # last action's x (turnover / loss location)
        )
        .reset_index()
    )
    agg["duration_s"] = agg["end_time"] - agg["start_time"]
    agg["is_open_play"] = ~agg["first_type"].isin(_SET_PIECE_TYPE_IDS)

    spells = agg.sort_values(["game_id", "period_id", "start_time", "possession_id"]).reset_index(drop=True)
    spells["prev_team_id"] = spells.groupby(["game_id", "period_id"], sort=False)["team_id"].shift(1)
    # is_recovery: team differs from the previous possession's; the first possession of a period
    # has prev = NA -> ids_differ = NA -> fillna(False) (not a recovery). .to_numpy() dodges the
    # fresh-RangeIndex alignment trap of ids_differ.
    spells["is_recovery"] = ids_differ(spells["team_id"], spells["prev_team_id"]).fillna(False).to_numpy()
    return spells.drop(columns=["first_type"])


def possession_minutes(spells: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(game_id, team_id)``: ``in_possession_min`` / ``out_of_possession_min`` (minutes).

    ``out_of_possession`` for team A = the sum of the game's OTHER possessions' durations (in a
    two-team game, the opponent's possession time).
    """
    by_team = spells.groupby(["game_id", "team_id"], sort=False)["duration_s"].sum().reset_index(name="in_s")
    game_total = spells.groupby("game_id", sort=False)["duration_s"].sum()
    by_team["_game_total"] = by_team["game_id"].map(game_total)
    by_team["in_possession_min"] = by_team["in_s"] / 60.0
    by_team["out_of_possession_min"] = (by_team["_game_total"] - by_team["in_s"]) / 60.0
    return by_team[["game_id", "team_id", "in_possession_min", "out_of_possession_min"]]
