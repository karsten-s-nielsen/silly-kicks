"""Integration: orient_frames_to_ltr closes the metrica/skillcorner GK bimodality.

Builds absolute frames + a home-team shot (P1) and an away-team shot (P2). After
orient_frames_to_ltr + the ADR-028 reprojection in add_pre_shot_gk_position, the
defending GK lands near the attacked goal (x>=95) for BOTH shots. The control on the
un-oriented absolute frames reproduces the bimodality (one GK ~near goal, one ~far).

Scope note (review concern B): defending_gk_player_id is hardcoded on the actions, so
this covers position reprojection under orientation, NOT add_pre_shot_gk_context GK
resolution. Pure synthetic -> intentionally NOT marked @pytest.mark.e2e (concern D).
"""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import orient_frames_to_ltr
from silly_kicks.tracking.features import add_pre_shot_gk_position

SHOT = spadlconfig.actiontype_id["shot"]


def _frame_row(period, frame, pid, tid, isball, isgk, x, y):
    return {
        "game_id": 1,
        "period_id": period,
        "frame_id": frame,
        "time_seconds": frame / 25.0,
        "frame_rate": 25.0,
        "player_id": pid,
        "team_id": tid,
        "is_ball": isball,
        "is_goalkeeper": isgk,
        "x": x,
        "y": y,
        "z": float("nan"),
        "speed": 0.0,
        "speed_source": "native",
        "ball_state": "alive",
        "team_attacking_direction": None,
        "confidence": None,
        "visibility": None,
        "source_provider": "metrica",
    }


def _abs_frames():
    """ABSOLUTE (physical) P2 frame. home_team_start_left=True => in P2 the home team
    attacks LEFT (toward x=0) and the away team attacks RIGHT (toward x=105):

      HOME-GK defends home's own goal at x=105  -> x=100
      AWAY-GK defends away's own goal at x=0     -> x=5
      HOME-ATT (home shot, attacks x=0)          -> x=15
      AWAY-ATT (away shot, attacks x=105)        -> x=90

    Both shots live in P2 so the control is genuinely bimodal: the home shot needs the
    P2 frame flip (orient), the away shot additionally needs the per-acting-team
    reprojection flip (ADR-028).
    """
    rows = [
        _frame_row(2, 100, "HOME-GK", 100, False, True, 100.0, 34.0),
        _frame_row(2, 100, "AWAY-GK", 200, False, True, 5.0, 34.0),
        _frame_row(2, 100, "HOME-ATT", 100, False, False, 15.0, 34.0),
        _frame_row(2, 100, "AWAY-ATT", 200, False, False, 90.0, 34.0),
        _frame_row(2, 100, None, None, True, False, 52.5, 34.0),
    ]
    return pd.DataFrame(rows)


def _actions():
    """Both shots in P2, per-acting-team LTR (start_x=90, acting team attacks x=105):
    action 1 = HOME shot (defending AWAY-GK)  -> exercises orient fixing the P2 frame
    action 2 = AWAY shot (defending HOME-GK)  -> exercises the away reprojection flip
    """
    return pd.DataFrame(
        [
            {
                "action_id": 1,
                "game_id": 1,
                "period_id": 2,
                "time_seconds": 4.0,
                "team_id": 100,
                "player_id": "HOME-ATT",
                "type_id": SHOT,
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 105.0,
                "end_y": 34.0,
                "defending_gk_player_id": "AWAY-GK",
            },
            {
                "action_id": 2,
                "game_id": 1,
                "period_id": 2,
                "time_seconds": 4.0,
                "team_id": 200,
                "player_id": "AWAY-ATT",
                "type_id": SHOT,
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 105.0,
                "end_y": 34.0,
                "defending_gk_player_id": "HOME-GK",
            },
        ]
    )


def test_oriented_gk_clusters_at_attacked_goal():
    oriented = orient_frames_to_ltr(_abs_frames(), home_team_id=100, home_team_start_left=True)
    enriched = add_pre_shot_gk_position(_actions(), oriented)
    gk_x = enriched.set_index("action_id")["pre_shot_gk_x"]
    assert gk_x[1] >= 95.0, f"home shot defending GK x={gk_x[1]}"
    assert gk_x[2] >= 95.0, f"away shot defending GK x={gk_x[2]}"


def test_unoriented_control_is_REFUSED_not_bimodal():
    """WITHOUT orient, the geometry is now REFUSED rather than silently mis-projected.

    This is the NEGATIVE control for the orientation helper, so it deliberately feeds absolute
    (unoriented) frames. Its assertion changed in 4.80.0, and the direction of the change is the
    point: it used to assert the DEFECT -- one GK near the attacked goal and one far, i.e. a
    bimodal, confidently wrong column -- because ADR-028 D2 could only make that audible, not
    prevent it. ADR-051 D3's nullable contract prevents it: an unresolvable direction is ``<NA>``,
    the shared action-context nulls the sampled positions rather than leaving them in the frame
    convention, and every downstream kernel produces NaN by ordinary arithmetic.

    The warning is still asserted, because a refusal a caller cannot explain is only half an
    answer -- the message is what names the frames, not the consumer, as the thing to fix.
    """
    from silly_kicks.tracking import OrientationUnresolvedWarning

    with pytest.warns(OrientationUnresolvedWarning):
        enriched = add_pre_shot_gk_position(_actions(), _abs_frames())
    gk_x = enriched.set_index("action_id")["pre_shot_gk_x"]
    assert gk_x[1:3].isna().all(), f"unoriented frames must yield NaN geometry, got {gk_x.to_dict()}"

    # NON-VACUITY. All-NaN would also be what a broken fixture produces, so the control is only
    # a control if the SAME actions and the SAME frames, merely oriented, do yield values.
    oriented = orient_frames_to_ltr(_abs_frames(), home_team_id=100, home_team_start_left=True)
    assert add_pre_shot_gk_position(_actions(), oriented).set_index("action_id")["pre_shot_gk_x"][1:3].notna().all()
