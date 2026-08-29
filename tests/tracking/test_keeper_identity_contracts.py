from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._keeper_identity import resolve_keeper_identities


def _actions():
    return pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [5.0],
            "team_id": [10],
            "player_id": [101],
            "type_name": ["shot"],
        }
    )


def _frames():
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "frame_id": [0, 0, 0],
            "team_id": [10, 20, pd.NA],
            "player_id": [1, 2, pd.NA],
            "is_ball": [False, False, True],
            "is_goalkeeper": [True, True, False],
        }
    ).astype({"team_id": "Int64", "player_id": "Int64"})


def test_resolver_does_not_mutate_its_inputs():
    a, f = _actions(), _frames()
    a_snap, f_snap = a.copy(deep=True), f.copy(deep=True)
    resolve_keeper_identities(a, f, identity="roster", roster={10: 901, 20: 902})
    pd.testing.assert_frame_equal(a, a_snap)
    pd.testing.assert_frame_equal(f, f_snap)


def test_roster_keys_match_across_id_dtypes_via_id_compat():
    # Frames carry Int64 team ids; roster keys are python ints AND strings -- both must resolve (ADR-019).
    m_int, _ = resolve_keeper_identities(_actions(), _frames(), identity="roster", roster={10: 901, 20: 902})
    m_str, _ = resolve_keeper_identities(_actions(), _frames(), identity="roster", roster={"10": 901, "20": 902})
    assert m_int[(canonical_id(1), 1, canonical_id(20))].gk_id == 902
    assert m_str[(canonical_id(1), 1, canonical_id(20))].gk_id == 902
