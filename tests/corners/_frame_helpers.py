"""Shared synthetic-frame helper for the corner encoding tests (TF-46 Arm 1 Part 2, F0, spec section 13.3).

The run encoding orients each snapshot through the ``GoalMap`` (ADR-055), which ``resolve_defended_goals``
builds from the keepers in the frames. A minimal synthetic scene carries no keepers, so the GoalMap cannot
resolve the period and the encoding withholds. ``with_keepers`` adds one keeper per team to every frame so the
attacking team attacks +x (the geometry the minimal scenes are written in).
"""

from __future__ import annotations

import pandas as pd


def with_keepers(
    frames: pd.DataFrame, *, attacker: str = "tA", defender: str = "tB", game_id: str = "g1"
) -> pd.DataFrame:
    """``frames`` plus, per frame, the attacker's keeper at x = 2 (it defends x = 0) and the defender's keeper
    at x = 103 (it defends x = 105), so ``GoalMap.attacked_goal(attacker) == 105``. Adds ``game_id`` and
    ``is_goalkeeper`` (False on the original rows) when absent."""
    base = frames.copy()
    if "game_id" not in base.columns:
        base["game_id"] = game_id
    if "is_goalkeeper" not in base.columns:
        base["is_goalkeeper"] = False
    keys = base[["game_id", "frame_id", "period_id", "time_seconds"]].drop_duplicates()
    rows = []
    for _, k in keys.iterrows():
        for team, pid, x in ((attacker, "gk_att", 2.0), (defender, "gk_def", 103.0)):
            rows.append(
                {
                    "game_id": k["game_id"],
                    "frame_id": k["frame_id"],
                    "period_id": k["period_id"],
                    "time_seconds": k["time_seconds"],
                    "team_id": team,
                    "player_id": pid,
                    "x": x,
                    "y": 34.0,
                    "is_ball": False,
                    "is_goalkeeper": True,
                }
            )
    out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    for c in ("team_id", "player_id"):
        if c in frames.columns:
            out[c] = out[c].astype(frames[c].dtype if frames[c].dtype == object else object)
    return out
