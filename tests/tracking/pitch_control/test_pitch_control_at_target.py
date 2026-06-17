"""Ground-truth correctness for pitch_control_at_target (ADR-032).

Mirror-symmetry is necessary but NOT sufficient (a symmetric-wrong / cancelling-double-flip projection passes
a symmetry-only test). So we pin an ABSOLUTE, hand-computable value on an ASYMMETRIC frame: the action's
destination cell is acting-team-controlled (~1.0) while its 180-degree absolute-frame reflection is
opponent-controlled (~0.0). For the AWAY action the correct projection lands on the ~1.0 cell and a
wrong-direction flip lands on the ~0.0 reflection -> RED on a broken projection. EXTREME separation (acting
players adjacent to the destination, opponents ~a pitch-length away) keeps the asymptote robust against the
Spearman sigmoid/reaction-time params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import pitch_control_at_target

_HOME, _AWAY = "H", "A"


def _frame(frame_id, *, controllers_xy, opponents_xy, controller_team, opp_team):
    """One canonical (home-attacks-right) frame: ``controller_team`` clustered adjacent to controllers_xy,
    ``opp_team`` clustered adjacent to opponents_xy, plus a ball off in midfield. Players static (vx=vy=0)."""

    def _dir(team):
        return "ltr" if team == _HOME else "rtl"

    def _player(k, team, x, y) -> dict[str, object]:
        return dict(
            game_id=1,
            period_id=1,
            frame_id=frame_id,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=f"{team}{k}",
            team_id=team,
            is_ball=False,
            is_goalkeeper=False,
            x=float(x),
            y=float(y),
            vx=0.0,
            vy=0.0,
            z=np.nan,
            speed=0.0,
            speed_source="derived",
            ball_state="alive",
            team_attacking_direction=_dir(team),
            confidence=np.nan,
            visibility=np.nan,
            source_provider="sportec",
            is_goalkeeper_source="native",
        )

    rows: list[dict[str, object]] = [_player(k, controller_team, cx, cy) for k, (cx, cy) in enumerate(controllers_xy)]
    rows += [_player(k, opp_team, ox, oy) for k, (ox, oy) in enumerate(opponents_xy)]
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=frame_id,
            time_seconds=10.0,
            frame_rate=25.0,
            player_id=None,
            team_id=None,
            is_ball=True,
            is_goalkeeper=False,
            x=52.5,
            y=34.0,
            vx=0.0,
            vy=0.0,
            z=np.nan,
            speed=0.0,
            speed_source="derived",
            ball_state="alive",
            team_attacking_direction=None,
            confidence=np.nan,
            visibility=np.nan,
            source_provider="sportec",
            is_goalkeeper_source=np.nan,
        )
    )
    return pd.DataFrame(rows)


def _action(team_id, end_x, end_y):
    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                time_seconds=10.0,
                team_id=team_id,
                player_id="X",
                start_x=52.5,
                start_y=34.0,
                end_x=float(end_x),
                end_y=float(end_y),
            )
        ]
    )


def test_home_action_destination_controlled_by_acting_team_reads_high():
    # HOME attacks right (ltr). Destination (80,20) is action-LTR == absolute (no flip). Home adjacent; away
    # at the reflection (25,48). PPCF(home)@(80,20) ~ 1.0.
    fr = _frame(
        100,
        controllers_xy=[(80, 20), (79, 21), (81, 19)],
        opponents_xy=[(25, 48), (26, 47), (24, 49)],
        controller_team=_HOME,
        opp_team=_AWAY,
    )
    s = pitch_control_at_target(_action(_HOME, 80.0, 20.0), fr, method="spearman")
    assert s.iloc[0] > 0.9, f"home at_target should be ~1.0 (acting team controls destination), got {s.iloc[0]}"


def test_away_action_pins_reprojection_direction():
    # AWAY action-LTR destination (80,20) -> absolute (105-80,68-20)=(25,48). Correct projection samples (25,48)
    # where AWAY is clustered -> ~1.0. A wrong-direction (no-flip) projection samples (80,20)=HOME -> ~0.0 -> RED.
    fr = _frame(
        200,
        controllers_xy=[(25, 48), (26, 47), (24, 49)],
        opponents_xy=[(80, 20), (79, 21), (81, 19)],
        controller_team=_AWAY,
        opp_team=_HOME,
    )
    s = pitch_control_at_target(_action(_AWAY, 80.0, 20.0), fr, method="spearman")
    assert s.iloc[0] > 0.9, (
        f"away at_target should be ~1.0 after the ADR-028 reprojection lands on absolute cell (25,48); "
        f"got {s.iloc[0]} (a ~0.0 value means the projection direction is wrong / not applied)"
    )


def test_mirror_invariance_home_vs_away():
    # Physically identical: acting team controls its destination in both. Necessary-not-sufficient symmetry guard.
    fh = _frame(300, controllers_xy=[(80, 20)], opponents_xy=[(25, 48)], controller_team=_HOME, opp_team=_AWAY)
    fa = _frame(301, controllers_xy=[(25, 48)], opponents_xy=[(80, 20)], controller_team=_AWAY, opp_team=_HOME)
    sh = pitch_control_at_target(_action(_HOME, 80.0, 20.0), fh, method="spearman").iloc[0]
    sa = pitch_control_at_target(_action(_AWAY, 80.0, 20.0), fa, method="spearman").iloc[0]
    assert abs(sh - sa) < 0.05, f"mirrored home/away at_target must agree: {sh} vs {sa}"


def test_multi_action_mixed_home_away_each_row_correct():
    # ONE frame, ONE call, TWO actions: HOME (dest (80,20), no flip) + AWAY (action-LTR (80,20) -> abs (25,48),
    # flips). The base frame already has HOME@(80,20) + AWAY@(25,48) + one ball. A flip mis-aligned across rows
    # reads ~0.0 for one -> RED. Only test exercising the per-action flip VECTORIZATION the production path uses.
    fr = _frame(
        400,
        controllers_xy=[(80, 20), (79, 21), (81, 19)],
        opponents_xy=[(25, 48), (26, 47), (24, 49)],
        controller_team=_HOME,
        opp_team=_AWAY,
    )
    actions = pd.concat([_action(_HOME, 80.0, 20.0), _action(_AWAY, 80.0, 20.0)], ignore_index=True)
    actions["action_id"] = [1, 2]
    s = pitch_control_at_target(actions, fr, method="spearman")
    assert s.iloc[0] > 0.9, f"home row (no flip) should read ~1.0 @ (80,20); got {s.iloc[0]}"
    assert s.iloc[1] > 0.9, f"away row (flipped) should read ~1.0 @ (25,48); got {s.iloc[1]} (flip mis-aligned?)"


def test_atomic_mirror_parity_matches_standard():
    """Atomic at_target (samples x+dx, y+dy) equals the standard column on geometry-matched actions."""
    from silly_kicks.atomic.tracking.features import pitch_control_at_target as atomic_pc

    fr = _frame(
        500,
        controllers_xy=[(80, 20), (79, 21), (81, 19)],
        opponents_xy=[(25, 48), (26, 47), (24, 49)],
        controller_team=_HOME,
        opp_team=_AWAY,
    )
    std = pitch_control_at_target(_action(_HOME, 80.0, 20.0), fr, method="spearman").iloc[0]
    # atomic action: x,y = start; dx,dy chosen so x+dx, y+dy == the standard end (80, 20).
    atomic_actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                time_seconds=10.0,
                team_id=_HOME,
                player_id="X",
                x=52.5,
                y=34.0,
                dx=80.0 - 52.5,
                dy=20.0 - 34.0,
            )
        ]
    )
    atom = atomic_pc(atomic_actions, fr, method="spearman").iloc[0]
    assert abs(std - atom) < 1e-9, f"atomic at_target must match standard: std={std} atom={atom}"
