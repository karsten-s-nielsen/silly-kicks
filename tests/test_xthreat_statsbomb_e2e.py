"""Owner/network-gated e2e: KDE beats Singh on held-out NLL on a full StatsBomb OPEN competition.

Product-appropriate triangulation on open data (NOT the lakehouse mart — that would couple the
product's test suite to consumer infrastructure; see SK-xT-1 spec review #1). Pulls a full open
competition via statsbombpy (fmt="dict" -> raw event dicts the silly-kicks converter consumes),
converts to SPADL, fits Singh + KDE at the default 16x12, and asserts KDE strictly beats Singh on
held-out (game_id-split) pass NLL. Marked e2e: deselected in the normal suite (network + slow).
"""

from __future__ import annotations

import pandas as pd
import pytest

import silly_kicks.spadl as spadl
from scripts._sb_raw import flatten_events
from silly_kicks.spadl import statsbomb
from silly_kicks.xthreat import (
    GridSpec,
    KDEParams,
    compute_holdout_nll,
    holdout_split,
    kde_smoothed_transition_matrix,
    singh_transition_matrix,
)

# FIFA World Cup 2022 — a full open-data tournament (64 matches), distinct from the committed
# WC2018 fixture so this is genuinely independent.
_COMPETITION_ID = 43
_SEASON_ID = 106


@pytest.mark.e2e
def test_kde_beats_singh_on_statsbomb_open_competition():
    pytest.importorskip("statsbombpy")
    from statsbombpy import sb  # type: ignore[import-not-found]

    try:
        matches = sb.matches(competition_id=_COMPETITION_ID, season_id=_SEASON_ID, fmt="dict")
    except Exception as exc:  # network / availability
        pytest.skip(f"StatsBomb open-data unavailable: {exc}")
    if not matches:
        pytest.skip("no matches returned for the competition")

    frames: list[pd.DataFrame] = []
    converted = 0
    failures: list[str] = []
    for match_id, m in matches.items():
        try:
            home_team_id = int(m["home_team"]["home_team_id"])
            events = list(sb.events(match_id=int(match_id), fmt="dict").values())
            adapted = flatten_events(events, int(match_id))
            actions, _ = statsbomb.convert_to_actions(adapted, home_team_id=home_team_id)
            frames.append(spadl.play_left_to_right(actions, home_team_id))
            converted += 1
        except Exception as exc:  # one bad live match must not sink the e2e; record + move on
            failures.append(f"{match_id}: {exc!r}")
    if failures:
        print(f"\n{len(failures)} match(es) skipped:\n  " + "\n  ".join(failures))

    assert converted >= 30, f"only {converted} matches converted; corpus too small to be meaningful"
    all_actions = pd.concat(frames, ignore_index=True)

    grid = GridSpec(n_zones_x=16, n_zones_y=12)
    train, holdout = holdout_split(all_actions, holdout_fraction=0.2, key_cols=("game_id",))
    holdout_passes = holdout[holdout["type_id"] == spadl.config.actiontype_id["pass"]]
    assert len(train) > 0 and len(holdout_passes) > 0

    nll_singh = compute_holdout_nll(singh_transition_matrix(train, grid), holdout_passes, grid=grid)
    nll_kde = compute_holdout_nll(kde_smoothed_transition_matrix(train, grid, KDEParams()), holdout_passes, grid=grid)
    print(
        f"\n[xT StatsBomb open e2e comp={_COMPETITION_ID} season={_SEASON_ID} "
        f"matches={converted}] Singh NLL={nll_singh:.5f} KDE NLL={nll_kde:.5f} "
        f"delta={nll_singh - nll_kde:+.5f}"
    )
    assert nll_kde < nll_singh, f"KDE {nll_kde} should beat Singh {nll_singh} on open data"
