"""Byte-identity oracle, CAPTURED from the pre-change tree.

The vectorized seam must reproduce the pinned loop on well-formed frames. ``GOLDEN`` below
was captured before any call site changed, by running against ``defended_goal_x`` at
``12f77f9``:

    python -c "
    from silly_kicks.tracking import defended_goal_x
    from tests.tracking.test_goal_map_oracle import well_formed_frames
    print(repr(defended_goal_x(well_formed_frames())))
    "

Regenerating it after the change would make it circular -- an oracle authored after its own
repair arrives green and is never observed failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking import resolve_defended_goals

PITCH = 105.0
HOME, AWAY = 1, 2

# Captured 2026-08-07 from `defended_goal_x` at 12f77f9, keys canonicalized for comparison.
GOLDEN: dict[tuple, float] = {
    ("1", "1", "1"): 0.0,
    ("1", "1", "2"): 105.0,
    ("1", "2", "1"): 105.0,
    ("1", "2", "2"): 0.0,
}


def well_formed_frames(n: int = 200, seed: int = 20260807) -> pd.DataFrame:
    """Two periods, ends swapped at half-time, every team with a keeper in every frame."""
    rng = np.random.default_rng(seed)
    recs = []
    for period in (1, 2):
        home_goal = 0.0 if period == 1 else PITCH
        for f in range(n):
            fid = (period - 1) * n + f
            for team in (HOME, AWAY):
                goal = home_goal if team == HOME else PITCH - home_goal
                gx = goal + (6.0 if goal < 50 else -6.0) + rng.normal(0, 1.0)
                recs.append((1, period, fid, 100 + team, team, gx, 34.0, False, True))
                lo, hi = (20.0, 95.0) if goal < 50 else (10.0, 85.0)
                for j in range(10):
                    recs.append(
                        (1, period, fid, team * 20 + j, team, rng.uniform(lo, hi), rng.uniform(2, 66), False, False)
                    )
            recs.append((1, period, fid, -1, pd.NA, 52.5, 34.0, True, False))
    df = pd.DataFrame(
        recs,
        columns=["game_id", "period_id", "frame_id", "player_id", "team_id", "x", "y", "is_ball", "is_goalkeeper"],
    )
    df["team_id"] = df["team_id"].astype("Int64")
    return df


def test_seam_reproduces_the_pre_change_golden() -> None:
    gm = resolve_defended_goals(well_formed_frames())
    merged = {**dict(gm.guessed), **dict(gm.resolved)}
    assert merged == GOLDEN
    assert gm.n_guessed == 0, "a well-formed fixture must resolve entirely from keepers"
    assert not gm.unresolved


def test_the_golden_is_not_vacuous() -> None:
    """A golden that would match an empty map proves nothing."""
    assert len(GOLDEN) == 4
    assert set(GOLDEN.values()) == {0.0, PITCH}
