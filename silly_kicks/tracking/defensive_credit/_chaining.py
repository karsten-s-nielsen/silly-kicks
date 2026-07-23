"""Possession-scoped resulting-shot + recovery resolvers."""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_possessions

_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick"))


def with_possessions(actions: pd.DataFrame) -> pd.DataFrame:
    """Attach possession_id (int64), sorted (game_id, period_id, action_id). Pure -- returns a copy."""
    return add_possessions(actions)


def resulting_shot_in_possession(actions, anchor_idx, *, attacking_team_id, max_actions):
    """First shot by attacking_team_id in the anchor's possession, within max_actions forward rows."""
    anchor = actions.iloc[anchor_idx]
    same_poss = (
        (actions["game_id"] == anchor["game_id"])
        & (actions["period_id"] == anchor["period_id"])
        & (actions["possession_id"] == anchor["possession_id"])
    )
    fwd = actions[same_poss & (actions.index > anchor_idx)].head(max_actions)
    for _, r in fwd.iterrows():
        if r["type_id"] in _SHOT_TYPE_IDS and same_id(r["team_id"], attacking_team_id):
            return r
    return None


def recovery_after_pass(actions, pass_idx, *, max_actions):
    """First OPPONENT ball-regain within max_actions rows of the failed pass. NaN-team skipped.

    The defending team is inferred as the first team != the pass's acting team (two-team match) --
    the SINGLE recovery resolver (P-3: no duplicate in _rules). Returns the recovery row or None.
    """
    passer_team = actions.iloc[pass_idx]["team_id"]
    fwd = actions.iloc[pass_idx + 1 : pass_idx + 1 + max_actions]
    for _, r in fwd.iterrows():
        if pd.isna(r["team_id"]):
            continue  # ADR-027: NaN-team rows never decide
        if not same_id(r["team_id"], passer_team):  # first opponent regain
            return r
    return None
