from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from silly_kicks.tracking._keeper_identity import (
    KEEPER_ID_SOURCE_DERIVED,
    KEEPER_ID_SOURCE_EVENT,
    KEEPER_ID_SOURCE_NATIVE,
    KEEPER_ID_SOURCE_ROSTER,
    KEEPER_ID_SOURCE_UNRESOLVED,
    KEEPER_ID_SOURCE_VALUES,
    KeeperIdentity,
    KeeperIdentityReport,
    resolve_keeper_identities,
)


def test_source_vocabulary_is_exactly_the_five_values():
    assert KEEPER_ID_SOURCE_VALUES == (
        KEEPER_ID_SOURCE_EVENT,
        KEEPER_ID_SOURCE_ROSTER,
        KEEPER_ID_SOURCE_NATIVE,
        KEEPER_ID_SOURCE_DERIVED,
        KEEPER_ID_SOURCE_UNRESOLVED,
    )
    assert set(KEEPER_ID_SOURCE_VALUES) == {"event", "roster", "native", "derived", "unresolved"}


def test_keeper_identity_is_a_three_field_named_tuple():
    ki = KeeperIdentity(gk_id=7, source=KEEPER_ID_SOURCE_ROSTER, conflict=False)
    assert (ki.gk_id, ki.source, ki.conflict) == (7, "roster", False)


def test_report_is_frozen_and_conserves():
    rep = KeeperIdentityReport(n_teams_in=2, n_resolved=2, n_unresolved=0, n_conflict=0, source_counts={"roster": 2})
    assert rep.n_resolved + rep.n_unresolved == rep.n_teams_in
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.n_resolved = 1  # type: ignore[misc]


def test_unknown_identity_mode_raises_value_error():
    # Task 3 implements the native path (covered by test_keeper_identity_native.py); the dispatch's
    # final branch rejects any mode outside {"native", "roster"}.
    with pytest.raises(ValueError, match=r"unknown identity mode"):
        resolve_keeper_identities(pd.DataFrame(), pd.DataFrame(), identity="bogus")  # type: ignore[arg-type]
