import numpy as np
import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.keeper_identity import (
    add_defending_gk_player_id,
    resolve_keeper_identities,
    validate_keeper_appearances,
)


def _actions():
    # shots by team 10 in period 1 at t=100 (before sub) and t=3000 (after sub); defending team 20
    return pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "period_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [500, 500],
            "type_name": ["shot", "shot"],
            "time_seconds": [100.0, 3000.0],
        }
    )


def _resolver_actions():
    # The event-only resolver seeds ONLY the teams it WITNESSES in `actions` (a real match witnesses
    # both -- cf. tests/keeper_identity/test_event_only_path.py). The 2-shot single-team `_actions()`
    # above witnesses only team 10, so resolving on it alone would starve the coarse `keeper_map` of
    # team 20 and NO opponent (the defending team) would ever be resolvable. Resolve on a two-team
    # action set so the map carries both teams -- exactly as the WITH-frames placement oracle
    # (tests/tracking/test_keeper_placement_helpers.py, whose freeze-frame witnesses both teams)
    # already does; the stamped `_actions()` stays team-10-only, so the defending keeper is always
    # team 20's.
    return pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "period_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [500, 700],
            "type_name": ["pass", "pass"],
            "time_seconds": [10.0, 20.0],
        }
    )


def _appearances_with_gk_sub():
    # team 20 keeper 902 until t=2700, then keeper 999 from 2700
    return validate_keeper_appearances(
        pd.DataFrame(
            {
                "game_id": ["g", "g", "g"],
                "team_id": pd.array([10, 20, 20], dtype="Int64"),
                "player_id": pd.array([901, 902, 999], dtype="Int64"),
                "period_id": [1, 1, 1],
                "start_time_seconds": [0.0, 0.0, 2700.0],
                "end_time_seconds": [np.inf, 2700.0, np.inf],
                "source": ["starting_xi", "starting_xi", "sub_events"],
            }
        )
    )


def test_attribution_flips_at_the_sub_minute():
    m, _ = resolve_keeper_identities(_resolver_actions(), identity="roster", roster={10: 901, 20: 902})
    out = add_defending_gk_player_id(_actions(), m, appearances=_appearances_with_gk_sub())
    ids = list(out["defending_gk_player_id"])
    assert same_id(ids[0], 902), "pre-sub shot -> starter keeper"
    assert same_id(ids[1], 999), "post-sub shot -> replacement keeper"


def test_omitting_appearances_is_byte_identical():
    m, _ = resolve_keeper_identities(_resolver_actions(), identity="roster", roster={10: 901, 20: 902})
    base = add_defending_gk_player_id(_actions(), m)
    also = add_defending_gk_player_id(_actions(), m, appearances=None)
    pd.testing.assert_frame_equal(base, also)
    assert "defending_gk_source" not in base.columns  # provenance is appearance-path only
    assert all(same_id(v, 902) for v in base["defending_gk_player_id"])


def test_conflict_flagged_when_interval_disagrees_with_map():
    # coarse map says team-20 keeper is 902 all period; the interval says 999 after the sub (spec §5.4 cross-check).
    m, _ = resolve_keeper_identities(_resolver_actions(), identity="roster", roster={10: 901, 20: 902})
    out = add_defending_gk_player_id(_actions(), m, appearances=_appearances_with_gk_sub())
    src = list(out["defending_gk_source"])
    assert src[0] == "appearance"  # pre-sub: interval agrees with the map (902)
    assert src[1] == "appearance_map_conflict"  # post-sub: interval 999 disagrees with map 902
