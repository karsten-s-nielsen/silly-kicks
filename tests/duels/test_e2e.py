"""Owner/network-gated e2e: Glicko-2 duel ratings run end-to-end on real StatsBomb open data.

Product-appropriate triangulation on public data via statsbombpy (the xT-e2e substrate). Loads a few
FIFA World Cup 2022 open matches, converts each to SPADL, and runs ``compute_duel_ratings`` over the
pooled matches -- exercising the DERIVED tackle/take_on adjacency path (StatsBomb carries no native
tackle winner/loser) AND the multi-match rating-period carry. Marked e2e: deselected in the normal
suite (network + slow); skips cleanly if statsbombpy / the network is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.duels import DUEL_COLUMNS, DUEL_WINNER_SOURCE_VALUES, compute_duel_ratings
from silly_kicks.spadl import statsbomb

_COMPETITION_ID = 43  # FIFA World Cup 2022
_SEASON_ID = 106


@pytest.mark.e2e
def test_duel_ratings_run_on_statsbomb_open():
    pytest.importorskip("statsbombpy")
    from statsbombpy import sb  # type: ignore[import-not-found]

    from scripts._sb_raw import flatten_events

    try:
        matches = sb.matches(competition_id=_COMPETITION_ID, season_id=_SEASON_ID, fmt="dict")
    except Exception as exc:  # network / availability
        pytest.skip(f"StatsBomb open-data unavailable: {exc}")
    if not matches:
        pytest.skip("no matches returned for the competition")

    per_match: list[pd.DataFrame] = []
    for match_id, m in list(matches.items())[:5]:
        try:
            home_team_id = int(m["home_team"]["home_team_id"])
            events = list(sb.events(match_id=int(match_id), fmt="dict").values())
            actions, _ = statsbomb.convert_to_actions(flatten_events(events, int(match_id)), home_team_id=home_team_id)
            per_match.append(actions)
        except Exception as exc:  # one bad live match must not sink the e2e
            print(f"skip {match_id}: {exc!r}")
    if len(per_match) < 3:
        pytest.skip(f"only {len(per_match)} matches converted; too few for the carry")

    actions = pd.concat(per_match, ignore_index=True)
    samples, report = compute_duel_ratings(actions)

    # schema conforms
    assert list(samples.columns) == list(DUEL_COLUMNS)
    for c, t in DUEL_COLUMNS.items():
        assert str(samples[c].dtype) == t, f"{c}: {samples[c].dtype} != {t}"

    # StatsBomb has no native winner/loser -> the derived adjacency path, and it MUST find duels
    assert report.labeling_strategy == "derived"
    assert report.n_matches == len(per_match)
    assert report.n_duels > 0, "the tackle/take_on adjacency found no duels on real StatsBomb data"
    assert set(samples["duel_winner_source"]) <= DUEL_WINNER_SOURCE_VALUES == {"native", "derived"}
    assert set(samples["duel_winner_source"]) == {"derived"}

    # each scored duel contributes to two tallies, deduped per (player, match)
    assert report.n_player_match_rows == len(samples) > 0
    assert int(samples["duels_won"].sum()) == report.n_duels  # exactly one winner per scored duel
    assert int(samples["duels_lost"].sum()) == report.n_duels  # exactly one loser per scored duel

    # ratings are finite, positive, and MOVED off the 1500 seed (the metric actually rated somebody)
    for c in ("duel_rating", "duel_rating_deviation", "duel_volatility"):
        v = pd.to_numeric(samples[c])
        assert np.isfinite(v).all() and (v > 0).all()
    assert (pd.to_numeric(samples["duel_rating"]) != 1500.0).any()
    # winners of every duel this match sit above losers on average is not guaranteed per-row, but the
    # rating spread must be non-trivial once duels are processed.
    assert pd.to_numeric(samples["duel_rating"]).std() > 0.0

    # resume-equivalence on the REAL trajectory (plan B7.1): one batch of N matches == two batches
    # threaded via initial_ratings. Split by the SAME sorted game order the orchestrator processes in.
    game_ids = sorted(pd.unique(actions["game_id"]))
    assert len(game_ids) >= 2
    mid = len(game_ids) // 2
    b1 = actions[actions["game_id"].isin(game_ids[:mid])]
    b2 = actions[actions["game_id"].isin(game_ids[mid:])]
    _, r1 = compute_duel_ratings(b1)
    _, r2 = compute_duel_ratings(b2, initial_ratings=r1.final_ratings)
    assert r2.final_ratings == report.final_ratings  # byte-equal to the single-batch final state
