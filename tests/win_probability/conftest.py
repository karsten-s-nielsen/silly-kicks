import pandas as pd
import pytest

from silly_kicks.spadl import config as C

SHOT = C.actiontype_id["shot"]
PASS = C.actiontype_id["pass"]
SUCC = C.result_id["success"]
FAIL = C.result_id["fail"]


def _match(game_id: str, a_goal_min_s: int, b_goal_min_s_p2: int) -> pd.DataFrame:
    # teams A(home) & B; a scored shot each + filler + a late action to set final minute ~92.5.
    rows = [
        (1, "A", 300, PASS, SUCC),
        (1, "A", a_goal_min_s, SHOT, SUCC),  # A goal in period 1
        (1, "B", 1500, SHOT, FAIL),
        (2, "B", b_goal_min_s_p2, SHOT, SUCC),  # B goal in period 2
        (2, "A", 2850, PASS, SUCC),  # minute ~92.5 -> final ~92.5
    ]
    return pd.DataFrame(
        {
            "game_id": [game_id] * len(rows),
            "action_id": range(len(rows)),
            "period_id": [r[0] for r in rows],
            "team_id": [r[1] for r in rows],
            "time_seconds": [r[2] for r in rows],
            "type_id": [r[3] for r in rows],
            "result_id": [r[4] for r in rows],
        }
    )


@pytest.fixture
def sample_two_match_actions() -> pd.DataFrame:
    m1 = _match("g1", a_goal_min_s=1800, b_goal_min_s_p2=900)
    m2 = _match("g2", a_goal_min_s=600, b_goal_min_s_p2=1800)
    return pd.concat([m1, m2], ignore_index=True)


@pytest.fixture
def sample_games() -> pd.DataFrame:
    return pd.DataFrame({"game_id": ["g1", "g2"], "home_team_id": ["A", "A"]})
