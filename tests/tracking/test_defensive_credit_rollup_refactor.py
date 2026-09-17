"""TF-51 Item 4: the defending-aggregate rollup is extracted into a shared, representation-agnostic
helper (`_rollup_defending_aggregate`) so the std and atomic aggregates share ONE implementation.

The strong parity guarantee is the existing std defensive-credit suite passing byte-identical after
the extraction (run in the plan's Task 3 Step 4); this module pins the extracted symbol + its
long-form -> aggregate contract on a hand-built long-form.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.tracking.defensive_credit._orchestration import _rollup_defending_aggregate


def test_rollup_symbol_exists():
    assert callable(_rollup_defending_aggregate)


def test_rollup_splits_defending_from_acting_on_caller_frame():
    """The rollup credits only the DEFENDING team (credited team != acting team) and assembles on the
    caller's OWN actions frame -- never a synthesized one."""
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "team_id": [1, 1],  # acting team = 1 on both actions
            "x": [10.0, 20.0],  # a caller-native column that must survive untouched
        }
    )
    long = pd.DataFrame(
        {
            "action_id": [0, 0, 1],
            "team_id": [2, 1, 2],  # defender(2) + own-team(1) on action 0; defender(2) on action 1
            "signed_value": [0.30, -0.10, -0.05],
            "resolution": ["nearest", "anchor_actor", "nearest"],
            "origin_x": [float("nan")] * 3,
            "origin_y": [float("nan")] * 3,
            "region_radius": [float("nan")] * 3,
        }
    )
    out = _rollup_defending_aggregate(actions, long, params=None, visible_area=None, links=None)
    # caller column survives; only defending (team 2) credits counted
    assert list(out["x"]) == [10.0, 20.0]
    assert out.loc[out["action_id"] == 0, "defensive_credit_net"].iloc[0] == 0.30  # own-team -0.10 excluded
    assert out.loc[out["action_id"] == 0, "n_defensive_credits"].iloc[0] == 1
    assert out.loc[out["action_id"] == 1, "defensive_credit_minus"].iloc[0] == -0.05
