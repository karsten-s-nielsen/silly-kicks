from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.id_compat import canonical_id
from silly_kicks.keeper_identity import KEEPER_ID_SOURCE_ROSTER, resolve_keeper_identities


def _actions():
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "period_id": [1, 1, 2, 2],
            "team_id": [10, 20, 10, 20],
            "player_id": [901, 902, 901, 902],
            "type_name": ["pass"] * 4,
            "time_seconds": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_roster_path_runs_without_frames():
    m, rep = resolve_keeper_identities(_actions(), identity="roster", roster={10: 901, 20: 902})
    assert m[(canonical_id(1), 1, canonical_id(10))].source == KEEPER_ID_SOURCE_ROSTER
    assert rep.n_resolved >= 1


def test_native_without_frames_raises():
    with pytest.raises(ValueError, match=r"(?i)native.*frames"):
        resolve_keeper_identities(_actions(), identity="native")
