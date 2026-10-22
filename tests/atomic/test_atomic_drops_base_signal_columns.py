"""Atomic SPADL drops the four TF-46 Arm 1 base signal columns (as ADR-046 drops the block columns).

``ATOMIC_SPADL_COLUMNS`` is an independent dict (it does not spread ``SPADL_COLUMNS``) and
``convert_to_atomic`` projects to it via ``_finalize_output``, so the base signal columns are dropped.
"""

import pandas as pd

from silly_kicks.atomic.spadl.base import convert_to_atomic
from silly_kicks.atomic.spadl.schema import ATOMIC_SPADL_COLUMNS
from silly_kicks.spadl import config as spadlcfg

_SIGNAL = ("result_source", "restart_evidence", "type_source", "repair_kind")


def _spadl_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "original_event_id": ["e1", "e2", "e3"][:n],
            "action_id": list(range(n)),
            "period_id": [1] * n,
            "time_seconds": [0.0, 1.0, 2.0][:n],
            "team_id": [100] * n,
            "player_id": [200] * n,
            "start_x": [50.0] * n,
            "start_y": [34.0] * n,
            "end_x": [60.0] * n,
            "end_y": [34.0] * n,
            "type_id": [spadlcfg.actiontype_id["pass"]] * n,
            "result_id": [spadlcfg.result_id["success"]] * n,
            "bodypart_id": [spadlcfg.bodypart_id["foot"]] * n,
            "shot_blocked": pd.array([pd.NA] * n, dtype="boolean"),
            "cross_blocked": pd.array([pd.NA] * n, dtype="boolean"),
            "result_source": pd.array(["native", "inferred", pd.NA][:n], dtype="object"),
            "restart_evidence": pd.array([pd.NA] * n, dtype="object"),
            "type_source": pd.array([pd.NA] * n, dtype="object"),
            # non-NA repair_kind so test_convert_to_atomic_drops_base_signal_columns proves the DROP, not
            # just an all-NA no-op (TF-46 Arm 1 Part 2 decision D1: repair_kind is a base signal column).
            "repair_kind": pd.array(["missing_goal", pd.NA, pd.NA][:n], dtype="object"),
        }
    )


def test_atomic_schema_omits_base_signal_columns():
    for col in _SIGNAL:
        assert col not in ATOMIC_SPADL_COLUMNS


def test_convert_to_atomic_drops_base_signal_columns():
    atomic = convert_to_atomic(_spadl_df())
    for col in _SIGNAL:
        assert col not in atomic.columns
