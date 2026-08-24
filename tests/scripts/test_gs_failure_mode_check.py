"""Task 9 (R6): GS failure-mode tagging reliability check."""

from __future__ import annotations

import pandas as pd

from scripts._gs_failure_mode_check import failure_mode_reliability
from scripts._receiver_validation import _R, _T

_COLS = [
    "action_id",
    "game_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _clean() -> pd.DataFrame:
    P, TI, F, S = _T["pass"], _T["throw_in"], _R["fail"], _R["success"]
    rows = [
        (1, 1, 1, 10, 1, 9, P, F, 50, 34, 60, 34),  # failed
        (2, 1, 1, 11, 2, 20, P, S, 60, 34, 40, 34),  # opponent -> intercepted
        (3, 1, 1, 20, 1, 9, P, F, 40, 60, 40, 68),  # failed
        (4, 1, 1, 21, 2, 22, TI, S, 40, 68, 45, 60),  # throw_in -> out
    ]
    return pd.DataFrame(rows, columns=_COLS)


def _noisy() -> pd.DataFrame:
    P, F, S = _T["pass"], _R["fail"], _R["success"]
    rows = [
        (1, 1, 1, 10, 1, 9, P, F, 50, 34, 60, 34),  # failed
        (2, 1, 1, 11, 1, 10, P, S, 60, 34, 40, 34),  # SAME team next -> other (un-classifiable)
        (3, 1, 1, 20, 1, 9, P, F, 40, 60, 40, 68),  # failed
        (4, 1, 1, 21, 1, 11, P, S, 40, 68, 45, 60),  # SAME team next -> other
    ]
    return pd.DataFrame(rows, columns=_COLS)


def test_clean_tagging_is_reliable():
    r = failure_mode_reliability(_clean())
    assert r["n_failed"] == 2 and r["ambiguous_rate"] == 0.0 and r["reliable"] is True


def test_noisy_tagging_is_unreliable():
    r = failure_mode_reliability(_noisy())
    assert r["n_failed"] == 2 and r["ambiguous_rate"] == 1.0 and r["reliable"] is False
