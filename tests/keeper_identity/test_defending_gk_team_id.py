"""add_defending_gk_player_id stamps the AUTHORITATIVE defending team (ADR-085; TF-59 PR2 Task 1).

The keeper's team is the opponent the resolver already derives to pick the keeper -- not an inference
from the shot actions. It is read from the ``keeper_map`` VALUE (``KeeperIdentity.team_id``, the
resolver's authoritative raw team; ADR-085 gold-standard) and emitted on BOTH paths as
``defending_gk_team_id``, NA only where the opponent team is unidentifiable -- INDEPENDENT of the keeper.
Because the team comes from the map value (not a from-actions recovery), it resolves even for an
opponent that never appears in the actions -- a frame-seeded map (see the non-vacuity test below).

Resolver-fixture rule (empirically pinned against shipped PR1 4.106.0): a fixture that calls
``resolve_keeper_identities`` MUST include ``type_name`` (the roster resolver reads it for the
goal-kick override, keeper_identity.py:760) AND have every keeper's team ACT per (game, period) so the
event-only roster path SEEDS it into the map.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.keeper_identity import add_defending_gk_player_id, resolve_keeper_identities
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_FAIL = spadlconfig.result_id["fail"]


def _actions() -> pd.DataFrame:
    # Team 10 shoots (defended by team 20's keeper 88); team 20 shoots (defended by team 10's keeper 99).
    # BOTH teams act, so the event-only roster resolver SEEDS both into keeper_map. type_name is REQUIRED
    # (the roster resolver reads actions["type_name"] for the goal-kick override; keeper_identity.py:760).
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 100.0,
                "team_id": 10,
                "player_id": 1,
                "type_id": _SHOT,
                "type_name": "shot",
                "result_id": _FAIL,
            },
            {
                "game_id": 1,
                "action_id": 1,
                "period_id": 1,
                "time_seconds": 200.0,
                "team_id": 20,
                "player_id": 2,
                "type_id": _SHOT,
                "type_name": "shot",
                "result_id": _FAIL,
            },
        ]
    )


def _keeper_map():
    kmap, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 99, 20: 88})
    return kmap


def test_defending_gk_team_id_is_the_opponent_team_coarse_path():
    out = add_defending_gk_player_id(_actions(), _keeper_map())
    assert "defending_gk_team_id" in out.columns
    t10 = out[out["team_id"] == 10].iloc[0]  # team-10 shot -> defended by team 20 (keeper 88)
    t20 = out[out["team_id"] == 20].iloc[0]  # team-20 shot -> defended by team 10 (keeper 99)
    assert same_id(t10["defending_gk_player_id"], 88) and same_id(t10["defending_gk_team_id"], 20)
    assert same_id(t20["defending_gk_player_id"], 99) and same_id(t20["defending_gk_team_id"], 10)


def test_defending_gk_team_id_matches_keeper_on_appearance_path():
    # The appearance path must ALSO carry defending_gk_team_id.
    from silly_kicks.keeper_identity import KeeperSegment, build_keeper_appearances_from_segments

    ap = pd.concat(
        [
            build_keeper_appearances_from_segments(
                [KeeperSegment(20, 88, "starting_xi", 1, 0.0, 1, float("inf"))], [1], game_id=1
            ),
            build_keeper_appearances_from_segments(
                [KeeperSegment(10, 99, "starting_xi", 1, 0.0, 1, float("inf"))], [1], game_id=1
            ),
        ],
        ignore_index=True,
    )
    out = add_defending_gk_player_id(_actions(), _keeper_map(), appearances=ap)
    t10 = out[out["team_id"] == 10].iloc[0]
    assert same_id(t10["defending_gk_player_id"], 88) and same_id(t10["defending_gk_team_id"], 20)


def test_team_known_but_keeper_unresolved():
    # roster names team 10's keeper but NOT team 20's. Team 20 is still IDENTIFIABLE (present in the
    # actions), so a team-10 shot -> defending_gk_team_id = 20 (KNOWN) while defending_gk_player_id = NA
    # (team 20's keeper unresolved). The team's NA-ness is NOT tied to the keeper's.
    kmap, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 99})
    out = add_defending_gk_player_id(_actions(), kmap)
    t10 = out[out["team_id"] == 10].iloc[0]  # defended by team 20, whose keeper is unresolved
    assert same_id(t10["defending_gk_team_id"], 20)
    assert pd.isna(t10["defending_gk_player_id"])


def test_defending_team_read_from_map_value_when_opponent_absent_from_actions():
    # GOLD STANDARD (ADR-085) + NON-VACUITY. The defending team is read from the keeper_map VALUE
    # (KeeperIdentity.team_id -- the resolver's authoritative raw team), so it resolves even for an
    # opponent that never appears in the actions -- a frame-seeded map where a team is in the frames but
    # never acts. A from-actions recovery returned NA here; asserting team 20 (which does NOT appear in
    # `actions`) proves the map-value read measurably moved the result off NA.
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.keeper_identity import KeeperIdentity, KeeperIdentityMap

    actions = pd.DataFrame(  # ONLY team 10 acts; team 20 never appears in the actions' team_id
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 100.0,
                "team_id": 10,
                "player_id": 1,
                "type_id": _SHOT,
                "type_name": "shot",
                "result_id": _FAIL,
            }
        ]
    )
    # A native/frame-seeded map: BOTH teams carry a raw team_id in the VALUE, though team 20 has no action.
    kmap: KeeperIdentityMap = {
        (canonical_id(1), 1, canonical_id(10)): KeeperIdentity(gk_id=99, source="native", conflict=False, team_id=10),
        (canonical_id(1), 1, canonical_id(20)): KeeperIdentity(gk_id=88, source="native", conflict=False, team_id=20),
    }
    r = add_defending_gk_player_id(actions, kmap).iloc[0]
    assert same_id(r["defending_gk_player_id"], 88)  # team 20's keeper (the opponent from the map)
    assert same_id(r["defending_gk_team_id"], 20)  # RESOLVED from the map value, though team 20 never acts


def test_team_id_na_when_no_opponent_in_period():
    # A (game, period) with only ONE team in the actions -> no distinct opponent -> BOTH
    # defending_gk_team_id and defending_gk_player_id are pd.NA (never fabricated).
    one_team = pd.DataFrame(  # only team 30 acts -> no opponent seeded (the intended NA case)
        [
            {
                "game_id": 2,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 10.0,
                "team_id": 30,
                "player_id": 5,
                "type_id": _SHOT,
                "type_name": "shot",
                "result_id": _FAIL,
            }
        ]
    )
    kmap, _ = resolve_keeper_identities(one_team, identity="roster", roster={30: 7})
    r = add_defending_gk_player_id(one_team, kmap).iloc[0]
    assert pd.isna(r["defending_gk_team_id"]) and pd.isna(r["defending_gk_player_id"])
