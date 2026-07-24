"""Event-only, per-team bravery -- % of opponent final actions blocked, per-type breakdown (R2-2).

See NOTICE for full bibliographic citations (Tigres Femenil "bravery" metric, Sumpter module 16.3).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_CROSS = spadlconfig.actiontype_id["cross"]  # open-play cross
_CORNER_CROSSED = spadlconfig.actiontype_id["corner_crossed"]
_FREEKICK_CROSSED = spadlconfig.actiontype_id["freekick_crossed"]
_SET_PIECE_CROSSES = frozenset({_CORNER_CROSSED, _FREEKICK_CROSSED})

_COLS = [
    "game_id",
    "team_id",
    "bravery_shots",
    "bravery_open_play_crosses",
    "bravery_set_piece_crosses",
    "bravery_pct_known_domain",
    "n_shots_faced",
    "n_open_play_crosses_faced",
    "n_set_piece_crosses_faced",
    "n_blocks_known",
]


def compute_bravery(
    actions: pd.DataFrame,
    *,
    shot_blocked_column: str = "shot_blocked",
    cross_blocked_column: str = "cross_blocked",
) -> pd.DataFrame:
    """Per (game_id, defending team) bravery. The defending team is the OPPONENT of the actor.

    Examples
    --------
    Event-only per-team bravery (% of opponent final actions blocked; no frames/xt needed)::

        bravery = compute_bravery(actions)
        bravery[["team_id", "bravery_pct_known_domain", "n_set_piece_crosses_faced"]]
    """
    a = actions
    is_shot = a["type_id"] == _SHOT
    is_open_cross = a["type_id"] == _CROSS
    is_set_cross = a["type_id"].isin(_SET_PIECE_CROSSES)
    final = a[is_shot | is_open_cross | is_set_cross]
    if final.empty:
        return pd.DataFrame({c: pd.Series([], dtype="float64" if "bravery" in c else "object") for c in _COLS})

    out_rows = []
    for (game_id, actor_team), g in final.groupby(["game_id", "team_id"], dropna=True):
        # the DEFENDING team faced these actor_team final actions (two-team match assumption)
        defending_team = _opponent_team(a, game_id, actor_team)
        n_shots = int((g["type_id"] == _SHOT).sum())
        n_open = int((g["type_id"] == _CROSS).sum())
        n_set = int(g["type_id"].isin(_SET_PIECE_CROSSES).sum())

        b_shots = _rate(g, _SHOT, shot_blocked_column)
        b_open = _rate(g, _CROSS, cross_blocked_column)
        b_set = np.nan  # v1 column limitation -- always unknown

        known_blocked, known_faced = _known_domain(g, shot_blocked_column, cross_blocked_column)
        headline = (known_blocked / known_faced) if known_faced > 0 and not np.isnan(known_blocked) else np.nan
        if np.isnan(headline):
            warnings.warn(
                f"bravery: game {game_id} team {defending_team} has no known-domain block signal "
                f"(both shot and cross blocked columns unknown) -> bravery_pct_known_domain=NaN.",
                stacklevel=2,
            )
        out_rows.append(
            {
                "game_id": game_id,
                "team_id": defending_team,
                "bravery_shots": b_shots,
                "bravery_open_play_crosses": b_open,
                "bravery_set_piece_crosses": b_set,
                "bravery_pct_known_domain": headline,
                "n_shots_faced": n_shots,
                "n_open_play_crosses_faced": n_open,
                "n_set_piece_crosses_faced": n_set,
                "n_blocks_known": int(known_blocked) if not np.isnan(known_blocked) else pd.NA,
            }
        )
    df = pd.DataFrame(out_rows, columns=_COLS)
    for c in ("n_shots_faced", "n_open_play_crosses_faced", "n_set_piece_crosses_faced", "n_blocks_known"):
        df[c] = df[c].astype("Int64")
    return df


def _rate(g, type_id, blocked_col):
    sub = g[g["type_id"] == type_id]
    if len(sub) == 0:
        return np.nan
    if blocked_col not in sub.columns or sub[blocked_col].isna().all():
        return np.nan  # R2-2: unknown -> NaN, never 0
    return float((sub[blocked_col] == True).sum()) / float(len(sub))  # noqa: E712


def _known_domain(g, shot_col, cross_col):
    """Blocked-count + faced-count over shots + open-play crosses whose block-status is known."""
    known_blocked = 0.0
    known_faced = 0
    any_known = False
    for type_id, col in ((_SHOT, shot_col), (_CROSS, cross_col)):
        sub = g[g["type_id"] == type_id]
        if len(sub) == 0:
            continue
        if col in sub.columns and not sub[col].isna().all():
            any_known = True
            known_blocked += float((sub[col] == True).sum())  # noqa: E712
            known_faced += len(sub)
    return (known_blocked if any_known else np.nan), known_faced


def _opponent_team(actions, game_id, actor_team):
    """The single opponent team id in this game (two-team assumption)."""
    teams = [
        t for t in actions[actions["game_id"] == game_id]["team_id"].dropna().unique() if not same_id(t, actor_team)
    ]
    return teams[0] if teams else pd.NA
