"""TF-49 packing geometry core. Frame convention: home attacks +x (LTR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._packing import PackingParams, compute_packing_metrics


def _frame(players):
    """players: list of (team_id, x, y, is_gk). Ball row appended automatically."""
    rows = [
        {
            "team_id": t,
            "x": x,
            "y": y,
            "is_goalkeeper": gk,
            "is_ball": False,
            "player_id": i + 1,
            "game_id": 1,
            "period_id": 1,
            "frame_id": 100,
        }
        for i, (t, x, y, gk) in enumerate(players)
    ]
    rows.append(
        {
            "team_id": None,
            "x": 50.0,
            "y": 34.0,
            "is_goalkeeper": False,
            "is_ball": True,
            "player_id": None,
            "game_id": 1,
            "period_id": 1,
            "frame_id": 100,
        }
    )
    return pd.DataFrame(rows)


HOME = 1
AWAY = 2
# 5 away outfielders at x = 40, 47, 55, 60, 90; away GK at x = 100. The defender at 47
# is LOAD-BEARING (review blocker 2): it puts a nonzero count inside every theta-probe
# and backward-pass interval, so the side/back multipliers are never multiplied by 0.
DEFENDERS = [
    (AWAY, 40.0, 30.0, False),
    (AWAY, 47.0, 40.0, False),
    (AWAY, 55.0, 20.0, False),
    (AWAY, 60.0, 50.0, False),
    (AWAY, 90.0, 34.0, False),
    (AWAY, 100.0, 34.0, True),
]


def test_packing_made_forward_count():
    m = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(50.0, 34.0),
        receiver_xy=(70.0, 34.0),
    )
    assert m["packing_made"] == 2  # 55 and 60 bypassed; 40 behind; 90 beyond


def test_include_gk_flag():
    m = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(85.0, 34.0),
        receiver_xy=(102.0, 34.0),
        params=PackingParams(include_gk=True),
    )
    assert m["packing_made"] == 2  # outfielder at 90 + GK at 100
    m2 = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(85.0, 34.0),
        receiver_xy=(102.0, 34.0),
    )
    assert m2["packing_made"] == 1  # GK excluded by default


def test_line_x_is_max_bypassed_defender_x():
    m = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(50.0, 34.0),
        receiver_xy=(70.0, 34.0),
    )
    assert m["line_x"] == pytest.approx(60.0)


def test_goal_threat_restricted_to_back_line():
    # back_line_n=2 -> the two deepest away outfielders (90, 60); pass 50->95 bypasses
    # 55,60,90 overall but only 60,90 of the back-2
    m = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(50.0, 34.0),
        receiver_xy=(95.0, 34.0),
        params=PackingParams(back_line_n=2),
    )
    assert m["packing_made"] == 3
    assert m["packing_goal_threat"] == 2


@pytest.mark.parametrize(
    ("end", "expected_mult"),
    [
        ((70.0, 34.0), 1.0),  # theta 0 -> forward
        ((70.0, 60.0), 0.5),  # theta ~52deg -> side
        ((45.0, 36.0), -1.0),
    ],  # backward
)
def test_net_direction_bands(end, expected_mult):
    m = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(50.0, 34.0),
        receiver_xy=end,
    )
    lo, hi = min(50.0, end[0]), max(50.0, end[0])
    interval = sum(1 for (_, x, _, gk) in DEFENDERS if not gk and lo < x <= hi)
    assert interval >= 1, "vacuous band probe -- fixture regression (review blocker 2)"
    assert m["packing_net"] == pytest.approx(expected_mult * interval)


def test_theta_band_boundaries():
    p = PackingParams()
    for theta_deg, mult in [(44.0, 1.0), (46.0, 0.5), (134.0, 0.5), (136.0, -1.0)]:
        dx = np.cos(np.radians(theta_deg)) * 10
        dy = np.sin(np.radians(theta_deg)) * 10
        m = compute_packing_metrics(
            _frame(DEFENDERS),
            attacking_team_id=HOME,
            home_team_id=HOME,
            passer_xy=(50.0, 30.0),
            receiver_xy=(50.0 + dx, 30.0 + dy),
            params=p,
        )
        lo, hi = min(50.0, 50.0 + dx), max(50.0, 50.0 + dx)
        n = sum(1 for (_, x, _, gk) in DEFENDERS if not gk and lo < x <= hi)
        assert n >= 1, f"vacuous probe at {theta_deg} deg -- fixture regression"
        assert m["packing_net"] == pytest.approx(mult * n), theta_deg


def test_backward_x_tie_far_end_inclusive():
    """Spec s7 x-tie probe: interval is (min, max] regardless of travel direction, so a
    defender exactly at the BACKWARD pass's ORIGIN x is counted (accepted, documented
    boundary asymmetry -- review minor)."""
    m = compute_packing_metrics(
        _frame(DEFENDERS),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(60.0, 34.0),
        receiver_xy=(50.0, 33.0),
    )
    # interval (50, 60]: defenders at 55 and 60 (60 == max, inclusive); backward mult -1
    assert m["packing_net"] == pytest.approx(-2.0)


def test_away_actor_mirrors_defenders():
    """Ground-truth asymmetric: away team attacks -x in frame coords; action coords are
    attack-positive. Defender at frame x=40 is at attack-positive 65 for the away team."""
    m = compute_packing_metrics(
        _frame([(HOME, 40.0, 30.0, False)]),
        attacking_team_id=AWAY,
        home_team_id=HOME,
        passer_xy=(60.0, 34.0),
        receiver_xy=(70.0, 34.0),
    )
    assert m["packing_made"] == 1  # 105-40=65 in (60, 70]


def test_no_defenders_nan():
    m = compute_packing_metrics(
        _frame([(HOME, 40.0, 30.0, False)]),
        attacking_team_id=HOME,
        home_team_id=HOME,
        passer_xy=(50.0, 34.0),
        receiver_xy=(70.0, 34.0),
    )
    assert np.isnan(m["packing_made"])


def test_params_validation():
    with pytest.raises(ValueError):
        PackingParams(side_multiplier=-0.5)
    with pytest.raises(ValueError):
        PackingParams(back_multiplier=0.5)
    with pytest.raises(ValueError):
        PackingParams(forward_max_deg=140.0, back_min_deg=135.0)
    with pytest.raises(ValueError):
        PackingParams(action_types=("passe",))
