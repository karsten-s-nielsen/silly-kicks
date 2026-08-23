"""Team-from-qualifier resolution for DIRECT DFL set-pieces (parse port).

A DFL set-piece with no nested ``<Play>`` -- a *direct* free kick / goal kick / corner /
penalty -- arrives from ``_build_event_row`` with ``team='unknown'``: the acting team is not
on the event, only its executor CLU id in the ``{type}_team`` qualifier column (the
first-child ``Team`` attribute). ``_resolve_idsse_team_from_qualifiers`` fills ``team`` from
those columns by matching the CLU against ``home_team_id_native`` / ``away_team_id_native``.

RED before the fix: ``_TEAM_QUALIFIER_PRIORITY`` listed only ``play_team`` / ``throwin_team`` /
``foul_team_fouler``, so ``freekick_team`` / ``goalkick_team`` / ``corner_team`` /
``penalty_team`` were never consulted and ``team`` stayed ``'unknown'`` -- which crashed the
downstream opponent guards (lakehouse: 8 idsse AC-drain units on direct free kicks). All four
are the set-piece EXECUTOR, so filling ``team`` from them is correct (inert where the qualifier
is absent, correct where present).
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.providers.sportec.parse import _resolve_idsse_team_from_qualifiers

_HOME = "DFL-CLU-000008"
_AWAY = "DFL-CLU-00000G"


def _direct_setpiece(event_type: str, qual_col: str, clu: str) -> pd.DataFrame:
    """One DIRECT set-piece row: ``team='unknown'``, the executor CLU only in ``qual_col``, and
    deliberately NO ``play_team`` column, so resolution must fall through to the set-piece
    executor columns rather than short-circuiting on the generic ``play_team``."""
    return pd.DataFrame(
        {
            "event_type": [event_type],
            "team": ["unknown"],
            qual_col: [clu],
            "home_team_id_native": [_HOME],
            "away_team_id_native": [_AWAY],
        }
    )


@pytest.mark.parametrize(
    ("event_type", "qual_col", "clu", "expected"),
    [
        ("FreeKick", "freekick_team", _AWAY, "away"),
        ("GoalKick", "goalkick_team", _HOME, "home"),
        ("CornerKick", "corner_team", _AWAY, "away"),
        ("Penalty", "penalty_team", _HOME, "home"),
    ],
)
def test_direct_setpiece_team_resolves_from_executor_qualifier(event_type, qual_col, clu, expected):
    df = _direct_setpiece(event_type, qual_col, clu)
    _resolve_idsse_team_from_qualifiers(df)
    assert df["team"].iloc[0] == expected


def test_setpiece_qualifier_never_overrides_an_already_resolved_team():
    # Non-vacuity for the fill above: the fill is gated on ``team=='unknown'``, so a row already
    # resolved (e.g. via a nested Play / play_team) is never clobbered by a conflicting executor
    # qualifier. Proves the fix WIDENS the source columns without weakening the resolved gate.
    df = _direct_setpiece("FreeKick", "freekick_team", _AWAY)
    df.loc[0, "team"] = "home"
    _resolve_idsse_team_from_qualifiers(df)
    assert df["team"].iloc[0] == "home"
