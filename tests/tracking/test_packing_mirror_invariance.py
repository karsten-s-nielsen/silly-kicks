"""TF-49 mirror-invariance gate (ADR-028 pattern): identical add_packing output when
the same physical situation is expressed in both frame conventions, PLUS the
asymmetric ground-truth pin (symmetry alone is insufficient -- house discipline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import add_packing

HOME, AWAY = 1, 2
PASS = spadlconfig.actiontype_id["pass"]
FL, FW = 105.0, 68.0

_NUMERIC_COLS = ["packing_made", "packing_net", "packing_goal_threat"]


def _scenario():
    """Home attacks right. Away completed pass 60->70 (attack-positive) bypasses the
    HOME defenders at frame x=40 (-> 65) and x=42 (-> 63); home completed pass 10->25
    bypasses the AWAY defenders at frame x=15 and x=18."""
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=100,
        time_seconds=4.0,
        frame_rate=25.0,
        z=0.0,
        speed=0.0,
        speed_source="native",
        ball_state="alive",
        confidence=None,
        visibility=None,
        source_provider="synthetic",
        is_goalkeeper_source="native",
    )
    rows = [
        dict(
            player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=4.0, y=31.0, team_attacking_direction="ltr"
        ),
        dict(
            player_id=11,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=40.0,
            y=30.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=12,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=42.0,
            y=44.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=13,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=50.0,
            y=20.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=14,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=20.0,
            y=50.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=50,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=True,
            x=101.0,
            y=37.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=61,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=15.0,
            y=30.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=62,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=18.0,
            y=40.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=63,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=45.0,
            y=25.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=45.0,
            y=34.0,
            team_attacking_direction=None,
        ),
    ]
    frames = pd.DataFrame([{**base, **r} for r in rows])
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=AWAY,
                player_id=61.0,
                type_id=PASS,
                result_id=1,
                start_x=60.0,
                start_y=34.0,
                end_x=70.0,
                end_y=34.0,
                time_seconds=3.9,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=HOME,
                player_id=11.0,
                type_id=PASS,
                result_id=1,
                start_x=10.0,
                start_y=30.0,
                end_x=25.0,
                end_y=34.0,
                time_seconds=4.0,
            ),
        ]
    )
    return actions, frames


def _mirror(actions, frames):
    """Physical left/right mirror: flip frame x/y, swap directions. SPADL action
    coords are already attack-positive and stay unchanged (ADR-028 pattern)."""
    f = frames.copy()
    f["x"] = FL - f["x"]
    f["y"] = FW - f["y"]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return actions.copy(), f


def test_add_packing_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_packing(a, f)
    # After the mirror, the team attacking right is AWAY (the file-pattern note in
    # test_action_ltr_mirror_invariance.py) -- the packing kernel keys its defender
    # mirror on home_team_id, exactly like structural_pass.
    mir = add_packing(am, fm)
    b = base.set_index("action_id")
    m = mir.set_index("action_id")
    for col in _NUMERIC_COLS:
        # NaN==NaN and exact-value equality across both action rows in one shot.
        pd.testing.assert_series_equal(b[col], m[col], check_names=False)


def test_asymmetric_ground_truth_pin():
    """Absolute expected counts (the B2 away-actor fixture at aggregator level):
    mirror-invariance alone would pass a double-flip bug that corrupts BOTH
    conventions identically -- the pin catches it."""
    a, f = _scenario()
    out = add_packing(a, f).set_index("action_id")
    # Away pass 60->70: HOME defenders at frame x=40 -> 105-40=65 and x=42 -> 63 are
    # inside (60, 70]; x=50 -> 55 and x=20 -> 85 are not. Forward pass -> net == made.
    assert out.loc[0, "packing_made"] == 2
    assert out.loc[0, "packing_net"] == 2.0
    assert out.loc[0, "packing_goal_threat"] == 2  # back-4 == all 4 HOME outfielders
    # Home pass 10->25: AWAY defenders at x=15, 18 inside (10, 25]; x=45 not.
    assert out.loc[1, "packing_made"] == 2
