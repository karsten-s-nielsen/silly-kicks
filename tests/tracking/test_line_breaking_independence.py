"""W6: ``detect_line_breaking`` as an array pass with row independence (spec 2026-11-08 tracking-batch section 4.6;
plan Task 12).

The two-game detection (``links=None`` and caller-supplied ``links=``) is the Task 3 game-split property; these
tests pin the chunk property and per-row results for duplicate action ids. The bit identity to the pre-change code
on single-game input was proven against a frozen oracle, which ended with ruling C5 (spec section 7).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import detect_line_breaking
from tests._prod_fixtures import GAME_A, make_gamestate_slots, make_production_frames

_PASS = spadlconfig.actiontype_id["pass"]
_CROSS = spadlconfig.actiontype_id["cross"]
_DRIBBLE = spadlconfig.actiontype_id["dribble"]
_TEAMS = (5, 6)


def _inputs(seed: int = 20261111, n_actions: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded single-game actions over production frames, with every NaN route of the function:

    - a NaN actor (``team_id`` <NA>);
    - a non-pass action (a dribble);
    - an unresolved direction (one team's direction label is null in period 2);
    - no opponents (the other team's outfield rows are removed from some frames).
    """
    frames = make_production_frames(seed=seed, n_frames=300, n_periods=2)
    rng = np.random.default_rng(seed)
    period = rng.integers(1, 3, n_actions)
    frame_id = rng.integers(0, 300, n_actions)
    team = pd.array(rng.choice(_TEAMS, n_actions), dtype="Int64")
    team[rng.random(n_actions) < 0.06] = pd.NA
    type_id = rng.choice([_PASS, _PASS, _PASS, _CROSS, _DRIBBLE], n_actions)
    start = np.column_stack([rng.uniform(15.0, 70.0, n_actions), rng.uniform(5.0, 63.0, n_actions)])
    end = start + np.column_stack([rng.uniform(5.0, 40.0, n_actions), rng.normal(0.0, 12.0, n_actions)])
    actions = pd.DataFrame(
        {
            "game_id": GAME_A,
            "action_id": np.arange(n_actions),
            "period_id": period,
            "time_seconds": frame_id / 25.0,
            "team_id": team,
            "player_id": 1,
            "type_id": type_id,
            "start_x": start[:, 0],
            "start_y": start[:, 1],
            "end_x": np.clip(end[:, 0], 0.0, 105.0),
            "end_y": np.clip(end[:, 1], 0.0, 68.0),
        }
    ).sort_values(["period_id", "time_seconds"], kind="stable")
    actions["action_id"] = np.arange(n_actions)
    actions = actions.reset_index(drop=True)
    # an unresolved direction: team 6's label is null in period 2
    unresolved = (frames["period_id"] == 2) & (frames["team_id"].astype(object) == 6)
    frames.loc[unresolved.to_numpy(), "team_attacking_direction"] = None
    # no opponents: team 6's outfield rows are gone from every 7th frame
    lonely = (frames["frame_id"] % 7 == 0) & (frames["team_id"].astype(object) == 6)
    frames = frames[~lonely.to_numpy()].reset_index(drop=True)
    return actions, frames


def _routes(actions: pd.DataFrame, out: pd.DataFrame) -> dict[str, int]:
    computed = out["line_break__ward"].notna().to_numpy()
    return {
        "nan_actor": int(actions["team_id"].isna().sum()),
        "non_pass": int((actions["type_id"] == _DRIBBLE).sum()),
        "computed": int(computed.sum()),
        "breaks": int(out["line_break__ward"].fillna(False).sum()),
        "zero_no_opponent": int(((out["lines_broken__ward"] == 0) & out["line_breaking_type__ward"].isna()).sum()),
    }


def test_inputs_reach_every_route():
    # precondition of the fixture the other tests share (ADR-056)
    actions, frames = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = detect_line_breaking(actions, frames)
    routes = _routes(actions, out)
    assert all(v > 0 for v in routes.values()), routes  # non-vacuity: every route and a real break


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_chunk_property(seed):
    actions, frames = _inputs(seed=20261111 + seed)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = detect_line_breaking(actions, frames)
        cuts = sorted({0, len(actions), *rng.integers(1, len(actions), 3).tolist()})
        parts = [detect_line_breaking(actions.iloc[lo:hi], frames) for lo, hi in itertools.pairwise(cuts)]
    assert_bits_equal(pd.concat(parts), whole)


def test_chunk_property_game_split():
    # chunks that split two games apart: each game alone equals its rows of the two-game call
    a1, f1 = _inputs(seed=11)
    a2, f2 = _inputs(seed=12)
    a2, f2 = a2.assign(game_id=8), f2.assign(game_id=8)
    actions = pd.concat([a1, a2], ignore_index=True)
    frames = pd.concat([f1, f2], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = detect_line_breaking(actions, frames)
        parts = [
            detect_line_breaking(actions.iloc[: len(a1)], frames),
            detect_line_breaking(actions.iloc[len(a1) :], frames),
        ]
    assert_bits_equal(pd.concat(parts), whole)


def test_gamestate_duplicates_each_row_own_result():
    actions, frames = _inputs()
    slots = make_gamestate_slots(actions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = detect_line_breaking(actions, frames)
        for slot in slots:
            assert slot["action_id"].duplicated().any() or slot is slots[0]  # precondition: duplicates exist
            out = detect_line_breaking(slot, frames)
            # each row equals the result of its own action (the row it was copied from)
            src = slot["action_id"].to_numpy()
            assert_bits_equal(out.reset_index(drop=True), base.iloc[src].reset_index(drop=True))
