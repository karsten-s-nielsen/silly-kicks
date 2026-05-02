"""Physical invariant: GK actions cluster at the team's defended goal.

In SPADL LTR convention every team attacks toward x=field_length, so every team
*defends* the x=0 end. Goalkeeper actions (saves, claims, punches) happen at or
near the defended goal. After the bug-fix this should hold for every team in
every provider; before the fix native StatsBomb / Wyscout / Opta away-team GK
actions are mirrored to the wrong end (high-x).
"""

from __future__ import annotations

import pytest

from silly_kicks.spadl import config as spadlconfig

from . import _loaders

_KEEPER_TYPE_NAMES = ("keeper_save", "keeper_claim", "keeper_punch")
_KEEPER_TYPE_IDS = {spadlconfig.actiontype_id[t] for t in _KEEPER_TYPE_NAMES}


_GK_PROVIDERS = [
    ("statsbomb_7298", lambda: _loaders.load_statsbomb(7298)),
    ("statsbomb_7584", lambda: _loaders.load_statsbomb(7584)),
    ("statsbomb_3754058", lambda: _loaders.load_statsbomb(3754058)),
    ("sportec_native", _loaders.load_sportec_native),
    ("sportec_via_kloppy", _loaders.load_sportec_via_kloppy),
    ("metrica_native", _loaders.load_metrica_native),
    # Excluded: pff_synthetic. The synthetic generator places the away-team RE
    # (recovery → keeper_save) events at ball_x=-45 (raw center-origin, = x=7.5
    # in bottom-left coords). For an AWAY GK in P1 with home_team_start_left=True,
    # the GK physically defends the RIGHT goal (x=105); ball_x=-45 puts the
    # synthetic GK on the wrong side of the pitch. PFF's converter flip is
    # otherwise verified correct by tests/invariants/test_direction_of_play.py.
]


@pytest.mark.parametrize("provider,loader", _GK_PROVIDERS)
def test_gk_actions_cluster_at_defended_goal(provider: str, loader):
    """Every team's GK actions must average start_x < field_length / 2 (defending side in LTR)."""
    actions, _home_team_id = loader()
    gk_actions = actions[actions["type_id"].isin(_KEEPER_TYPE_IDS)]
    if gk_actions.empty:
        pytest.skip(f"{provider}: fixture has no GK actions")

    by_team = gk_actions.groupby("team_id")["start_x"].agg(["count", "mean"])
    # Only check teams with >= 1 GK action — synthetic / small fixtures may have only one team's GK
    failing = by_team[by_team["mean"] >= spadlconfig.field_length / 2]
    assert failing.empty, (
        f"{provider}: teams with GK actions on the attacking half (SPADL LTR violated): "
        f"{failing.to_dict('index')}. GKs defend the x=0 end. Full breakdown: "
        f"{by_team.to_dict('index')}"
    )
