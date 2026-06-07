"""Shared SPADL factories + the WC2018 actions builder for xthreat tests.

A plain module (not conftest) because these are parametrized factories, not fixtures.
"""

from typing import cast

import numpy as np
import pandas as pd

import silly_kicks.spadl as spadl
import silly_kicks.spadl.config as cfg

_PASS = cfg.actiontype_id["pass"]
_SUCCESS = cfg.result_id["success"]


def _worldcup_ltr(sb_worldcup_data) -> pd.DataFrame:
    """All WC2018 games concatenated as left-to-right SPADL actions."""
    games = cast(pd.DataFrame, sb_worldcup_data["games"]).set_index("game_id")
    return cast(
        pd.DataFrame,
        pd.concat(
            [
                spadl.play_left_to_right(cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{gid}"]), g.home_team_id)
                for gid, g in games.iterrows()
            ]
        ),
    )


def _moves(n_per_zone: int = 20, seed: int = 0) -> pd.DataFrame:
    """Successful passes from 3 source x-bands with gaussian-jittered destinations."""
    rng = np.random.default_rng(seed)
    rows, aid = [], 0
    for sx in (20.0, 50.0, 80.0):
        for _ in range(n_per_zone):
            rows.append(
                dict(
                    game_id=1,
                    action_id=aid,
                    period_id=1,
                    time_seconds=float(aid),
                    team_id=1,
                    player_id=1,
                    bodypart_id=0,
                    type_id=_PASS,
                    result_id=_SUCCESS,
                    start_x=sx,
                    start_y=34.0,
                    end_x=float(np.clip(sx + rng.normal(10, 3), 0, cfg.field_length)),
                    end_y=float(np.clip(34 + rng.normal(0, 3), 0, cfg.field_width)),
                )
            )
            aid += 1
    return pd.DataFrame(rows)


def _sparse_overfit_corpus(seed: int = 0, n_games: int = 20) -> pd.DataFrame:
    """Sparse, wide-jitter passes from 4 centres across many games — Singh overfits (spiky rows),
    KDE smooths. Used by the KDE-beats-Singh hard gate (Task 10).

    game_id is seed-offset (seed*1000 + g) so different seeds vary BOTH the destinations AND the
    sha256 holdout split. n_games=20 keeps the 25% split non-degenerate (~5 holdout games).
    """
    rng = np.random.default_rng(seed)
    centres = [(15.0, 20.0), (40.0, 50.0), (70.0, 30.0), (90.0, 60.0)]
    rows, aid = [], 0
    for g in range(n_games):
        game_id = seed * 1000 + g
        for sx, sy in centres:
            for _ in range(2):  # only 2 obs per (game, centre) -> spiky Singh rows
                rows.append(
                    dict(
                        game_id=game_id,
                        action_id=aid,
                        period_id=1,
                        time_seconds=float(aid),
                        team_id=1,
                        player_id=1,
                        bodypart_id=0,
                        type_id=_PASS,
                        result_id=_SUCCESS,
                        start_x=sx,
                        start_y=sy,
                        end_x=float(np.clip(sx + 12 + rng.normal(0, 6), 0, cfg.field_length)),
                        end_y=float(np.clip(sy + rng.normal(0, 6), 0, cfg.field_width)),
                    )
                )
                aid += 1
    return pd.DataFrame(rows)
