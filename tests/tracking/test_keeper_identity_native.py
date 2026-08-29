from __future__ import annotations

from unittest import mock

import pandas as pd

import silly_kicks.tracking as T
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._keeper_identity import (
    KEEPER_ID_SOURCE_DERIVED,
    KEEPER_ID_SOURCE_NATIVE,
    resolve_keeper_identities,
)


def _actions():
    return pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [5.0, 6.0],
            "team_id": [10, 20],
            "player_id": [101, 201],
            "type_name": ["pass", "pass"],
        }
    )


def _frames(gk_source_team20="native"):
    # Two teams, each with a keeper carrying a REAL player_id (native-provider shape).
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1],
            "frame_id": [0, 0, 0, 1, 1],
            "time_seconds": [5.0, 5.0, 5.0, 6.0, 6.0],
            "team_id": [10, 20, pd.NA, 10, 20],
            "player_id": [910, 920, pd.NA, 910, 920],
            "is_ball": [False, False, True, False, False],
            "is_goalkeeper": [True, True, False, True, True],
            "is_goalkeeper_source": ["native", gk_source_team20, "native", "native", gk_source_team20],
        }
    ).astype({"team_id": "Int64", "player_id": "Int64"})


def test_native_path_resolves_keeper_ids_from_the_frame():
    m, rep = resolve_keeper_identities(_actions(), _frames(), identity="native")
    assert m[(canonical_id(1), 1, canonical_id(10))].gk_id == 910
    assert m[(canonical_id(1), 1, canonical_id(20))].gk_id == 920
    assert m[(canonical_id(1), 1, canonical_id(10))].source == KEEPER_ID_SOURCE_NATIVE
    assert rep.n_resolved == 2


def test_native_path_source_reflects_is_goalkeeper_source():
    m, _ = resolve_keeper_identities(_actions(), _frames(gk_source_team20="derived"), identity="native")
    assert m[(canonical_id(1), 1, canonical_id(20))].source == KEEPER_ID_SOURCE_DERIVED


def test_native_path_delegates_and_does_not_reimplement():
    """Single-source (ADR-055): the native path CALLS the TF-13 resolvers."""
    real_def = T.defending_gk_from_frames
    real_act = T.acting_gk_from_frames
    with (
        mock.patch("silly_kicks.tracking._keeper_identity.defending_gk_from_frames", wraps=real_def) as md,
        mock.patch("silly_kicks.tracking._keeper_identity.acting_gk_from_frames", wraps=real_act) as ma,
    ):
        resolve_keeper_identities(_actions(), _frames(), identity="native")
    assert md.called, "native path must delegate to defending_gk_from_frames, not reimplement it"
    assert ma.called, "native path must delegate to acting_gk_from_frames, not reimplement it"


def test_resolver_is_exported_from_tracking():
    assert hasattr(T, "resolve_keeper_identities")
    assert T.resolve_keeper_identities is resolve_keeper_identities
    assert set(T.KEEPER_ID_SOURCE_VALUES) == {"event", "roster", "native", "derived", "unresolved"}
