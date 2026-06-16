"""Goal-end map extraction (TF-48 prerequisite; spec section 5.1).

``defended_goal_x`` moved byte-identically from ``_xshot_occurrence.py`` into
``_gk_resolve.py`` so TF-48 shares it without duplication; xS re-imports via shim.
"""

import pandas as pd

from silly_kicks.tracking._gk_resolve import defended_goal_x


def _frames(gk_x_a=5.0, gk_x_b=100.0, with_gk=True):
    rows = []
    for pid, team, gk, x in [
        (1, "A", True, gk_x_a),
        (2, "A", False, 40.0),
        (3, "B", True, gk_x_b),
        (4, "B", False, 60.0),
    ]:
        if not with_gk and gk:
            continue
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=0,
                time_seconds=0.0,
                player_id=pid,
                team_id=team,
                is_ball=False,
                is_goalkeeper=gk,
                x=x,
                y=34.0,
                z=0.0,
            )
        )
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=0,
            time_seconds=0.0,
            player_id=None,
            team_id=None,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            z=0.0,
        )
    )
    return pd.DataFrame(rows)


def test_gk_based_resolution():
    m = defended_goal_x(_frames())
    assert m[(1, 1, "A")] == 0.0 and m[(1, 1, "B")] == 105.0


def test_outfield_fallback_when_no_gk():
    # N1 fallback: a (game, period, team) with no GK rows resolves from team mean x
    m = defended_goal_x(_frames(with_gk=False))
    assert m[(1, 1, "A")] == 0.0 and m[(1, 1, "B")] == 105.0


def test_xs_shim_is_same_object():
    from silly_kicks.tracking._xshot_occurrence import _defended_goal_x

    assert _defended_goal_x is defended_goal_x
