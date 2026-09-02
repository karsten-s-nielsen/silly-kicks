from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.id_compat import canonical_id
from silly_kicks.keeper_identity import (
    KEEPER_ID_SOURCE_EVENT,
    KEEPER_ID_SOURCE_ROSTER,
    KEEPER_ID_SOURCE_UNRESOLVED,
    resolve_keeper_identities,
)

# NOTE: replace GOALKICK with the real SPADL type_name for a goal kick, read from spadlconfig.
GOALKICK = "goalkick"


def _actions(*, goalkick_taker=None, goalkick_team=None):
    """One shot by team 10 (defended by team 20's keeper) + optionally a goal kick by `goalkick_team`."""
    rows = [
        {
            "action_id": 0,
            "game_id": 1,
            "period_id": 1,
            "time_seconds": 5.0,
            "team_id": 10,
            "player_id": 101,
            "type_name": "shot",
        },
    ]
    if goalkick_taker is not None:
        rows.append(
            {
                "action_id": 1,
                "game_id": 1,
                "period_id": 1,
                "time_seconds": 60.0,
                "team_id": goalkick_team,
                "player_id": goalkick_taker,
                "type_name": GOALKICK,
            }
        )
    return pd.DataFrame(rows)


def _frames(team_ids=(10, 20)):
    """Minimal frames carrying the two real team ids on non-ball rows + a ball row."""
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "frame_id": [0, 0, 0],
            "team_id": [team_ids[0], team_ids[1], pd.NA],
            "player_id": [1, 2, pd.NA],
            "is_ball": [False, False, True],
            "is_goalkeeper": [True, True, False],
        }
    ).astype({"team_id": "Int64", "player_id": "Int64"})


def test_defending_keeper_resolves_from_roster():
    m, rep = resolve_keeper_identities(_actions(), _frames(), identity="roster", roster={10: 901, 20: 902})
    assert m[(canonical_id(1), 1, canonical_id(20))].gk_id == 902
    assert m[(canonical_id(1), 1, canonical_id(20))].source == KEEPER_ID_SOURCE_ROSTER
    assert rep.n_resolved == 2 and rep.n_unresolved == 0


def test_goalkick_event_overrides_a_wrong_roster_starter():
    # Roster says team 20's keeper is 902, but a goal kick by 999 (team 20) says otherwise -> event wins, conflict.
    m, rep = resolve_keeper_identities(
        _actions(goalkick_taker=999, goalkick_team=20),
        _frames(),
        identity="roster",
        roster={10: 901, 20: 902},
    )
    entry = m[(canonical_id(1), 1, canonical_id(20))]
    assert entry.gk_id == 999
    assert entry.source == KEEPER_ID_SOURCE_EVENT
    assert entry.conflict is True
    assert rep.n_conflict == 1


def test_unresolved_team_is_NA_and_counted_not_fabricated():
    m, rep = resolve_keeper_identities(
        _actions(),
        _frames(),
        identity="roster",
        roster={10: 901},  # no entry for team 20
    )
    entry = m[(canonical_id(1), 1, canonical_id(20))]
    assert entry.gk_id is pd.NA  # unresolved -> the NA singleton (never fabricated), counted below
    assert entry.source == KEEPER_ID_SOURCE_UNRESOLVED
    assert rep.n_unresolved == 1


def test_synthetic_0_1_team_pair_raises_under_roster_identity():
    # Frames carry the synthetic {0,1} fallback pair; roster keys (10,20) intersect none of them.
    synthetic = _frames(team_ids=(0, 1))
    with pytest.raises(ValueError, match=r"synthetic|roster|team"):
        resolve_keeper_identities(_actions(), synthetic, identity="roster", roster={10: 901, 20: 902})


def test_mid_period_substitution_later_goalkick_taker_wins_and_flags_conflict():
    # Two DIFFERENT goal-kick takers for team 20 in the SAME period (a keeper sub): the LATER-time
    # taker wins and the event-vs-event disagreement is flagged conflict=True. No roster entry for
    # team 20, so the conflict is purely the mid-period sub (isolated from any roster-vs-event one).
    actions = pd.DataFrame(
        [
            {
                "action_id": 0,
                "game_id": 1,
                "period_id": 1,
                "time_seconds": 5.0,
                "team_id": 10,
                "player_id": 101,
                "type_name": "shot",
            },
            {
                "action_id": 1,
                "game_id": 1,
                "period_id": 1,
                "time_seconds": 60.0,
                "team_id": 20,
                "player_id": 999,
                "type_name": GOALKICK,
            },
            {
                "action_id": 2,
                "game_id": 1,
                "period_id": 1,
                "time_seconds": 120.0,
                "team_id": 20,
                "player_id": 888,
                "type_name": GOALKICK,
            },
        ]
    )
    m, rep = resolve_keeper_identities(actions, _frames(), identity="roster", roster={10: 901})
    entry = m[(canonical_id(1), 1, canonical_id(20))]
    assert entry.gk_id == 888  # the later-time taker wins
    assert entry.source == KEEPER_ID_SOURCE_EVENT
    assert entry.conflict is True  # two distinct takers in one period -> conflict
    assert rep.n_conflict == 1


def test_goalkick_that_agrees_with_roster_still_records_event_as_the_winning_source():
    # event > roster is UNCONDITIONAL: a goal-kick taker that AGREES with the roster keeper still
    # records source="event" (the more authoritative rung), with conflict=False (no disagreement).
    m, rep = resolve_keeper_identities(
        _actions(goalkick_taker=902, goalkick_team=20),
        _frames(),
        identity="roster",
        roster={10: 901, 20: 902},
    )
    entry = m[(canonical_id(1), 1, canonical_id(20))]
    assert entry.gk_id == 902
    assert entry.source == KEEPER_ID_SOURCE_EVENT  # event wins even on agreement with the roster
    assert entry.conflict is False  # roster and event agree -> no conflict
    assert rep.n_conflict == 0
