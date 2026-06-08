"""Shared geometry-correct spell-fixture builders for the causal tests (single source -- R3-L2).

GEOMETRY (R2-H1): team 5 ATTACKS x=0 -> team-5 GK at HIGH x (101, defends 105); team-6 GK at LOW x
(4, defends 0). _build_goal_map then yields goal_x=0 for team-5 possession, so a low-x wide ball
(x<=35 advanced, y<14 or y>54) is in the advanced wide area.
"""

import pandas as pd

META = {"cross_types": ["cross"], "carrier_params": {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}}
WIDE = (12.0, 6.0)  # advanced (x<=35 from goal 0) + wide (y=6<14)
CENTRAL = (12.0, 34.0)  # advanced but central (14<y<54) -> NOT wide area
NEAR0 = (8.0, 6.0)


def frow(pid, team, gk, x, y, t, *, is_ball=False, period=1):
    return dict(
        game_id=1,
        period_id=period,
        frame_id=round(t * 25),
        time_seconds=round(t, 3),
        frame_rate=25.0,
        player_id=pid,
        team_id=team,
        is_ball=is_ball,
        is_goalkeeper=gk,
        x=float(x),
        y=float(y),
        z=0.0,
        speed=2.0,
        vx=2.0,
        vy=0.0,
        speed_source="native",
        ball_state="alive",
        team_attacking_direction="ltr",
        source_provider="test",
    )


def frames(possession_by_time, ball_xy_by_time, *, period=1):
    rows = []
    for t, pt in possession_by_time.items():
        bx, by = ball_xy_by_time[t]
        rows.append(frow(10 if pt == 5 else 20, pt, False, bx, by, t, period=period))  # carrier on ball
        rows += [
            frow(11, 5, False, 18.0, 40.0, t, period=period),
            frow(12, 5, False, 15.0, 30.0, t, period=period),
            frow(21, 6, False, 8.0, 40.0, t, period=period),
            frow(22, 6, False, 10.0, 30.0, t, period=period),
            frow(1, 5, True, 101.0, 34.0, t, period=period),  # team-5 GK high x -> attacks 0
            frow(2, 6, True, 4.0, 34.0, t, period=period),  # team-6 GK low x -> defends 0
        ]
        rows.append(frow(pd.NA, pd.NA, False, bx, by, t, is_ball=True, period=period))
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


def spell(team=5, t0=10.0, t1=10.4, dt=0.2, ball=WIDE, *, period=1):
    """A continuous in-domain possession spell for ``team`` over [t0, t1] (inclusive)."""
    poss, xy, t = {}, {}, t0
    while t <= t1 + 1e-9:
        key = round(t, 3)
        poss[key], xy[key] = team, ball
        t += dt
    return frames(poss, xy, period=period)


def actions(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "action_id",
            "period_id",
            "team_id",
            "time_seconds",
            "type_id",
            "result_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ],
    )
