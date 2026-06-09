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


def _corpus_with_shots(n_per_zone: int = 40, seed: int = 0) -> pd.DataFrame:
    """``_moves`` plus a cluster of shots near goal (some scored) so the fitted xT grid is
    non-zero. A passes-only corpus yields P(score)=0 everywhere -> an all-zero ``.xT`` ->
    ``ExpectedThreat.rate`` (and the xt_xfns fitted-check) raise NotFittedError. Used by the
    xt-VAEP-feature tests, which need a *rateable* fitted model."""
    moves = _moves(n_per_zone=n_per_zone, seed=seed)
    shot_id = cfg.actiontype_id["shot"]
    rng = np.random.default_rng(seed + 99)
    rows, aid = [], int(moves.action_id.max()) + 1
    for _ in range(60):
        scored = rng.random() < 0.2
        rows.append(
            dict(
                game_id=1,
                action_id=aid,
                period_id=1,
                time_seconds=float(aid),
                team_id=1,
                player_id=1,
                bodypart_id=0,
                type_id=shot_id,
                result_id=cfg.result_id["success"] if scored else cfg.result_id["fail"],
                start_x=float(np.clip(95.0 + rng.normal(0, 3), 0, cfg.field_length)),
                start_y=float(np.clip(34.0 + rng.normal(0, 8), 0, cfg.field_width)),
                end_x=cfg.field_length,
                end_y=cfg.field_width / 2.0,
            )
        )
        aid += 1
    return pd.concat([moves, pd.DataFrame(rows)], ignore_index=True)


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


def nll_relative_win(baseline_nll: float, candidate_nll: float) -> float:
    """Relative held-out-NLL improvement of ``candidate`` over ``baseline``: ``(b - c) / b``.

    Positive == candidate is better (lower NLL). Returns ``nan`` if ``baseline`` is non-finite or
    zero (e.g. an empty-corpus ``compute_holdout_nll``), or if ``candidate`` is ``nan``.
    """
    if not np.isfinite(baseline_nll) or baseline_nll == 0:
        return float("nan")
    return (baseline_nll - candidate_nll) / baseline_nll


def kde_clears_tripwire(singh_nll: float, kde_nll: float, *, floor: float) -> bool:
    """The owner-gated tripwire predicate: KDE strictly beats Singh AND clears the relative floor.

    Pure + NaN-safe (a non-finite relative win -> ``False``). Unit-tested so a flipped comparison or
    wrong-direction floor is caught in CI, not only on the owner's mart.
    """
    rel = nll_relative_win(singh_nll, kde_nll)
    if not np.isfinite(rel):
        return False
    return bool(kde_nll < singh_nll and rel >= floor)
