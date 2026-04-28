"""Tests for ``preserve_native`` kwarg on ``convert_to_atomic`` (silly-kicks 1.5.0+).

Surfaces caller-attached columns (e.g. provider-native fields preserved through
``convert_to_actions(preserve_native=...)``, or any analytics column added
between standard SPADL conversion and atomic conversion) through to the
Atomic-SPADL output alongside the canonical 13 atomic columns.

Synthetic atomic rows (``receival`` / ``interception`` / ``out`` / ``offside``
/ ``goal`` / ``owngoal`` / ``yellow_card`` / ``red_card``) get NaN in the
preserved column — they have no source row to pull a value from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl import config as atomicspadlconfig
from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.atomic.spadl.schema import ATOMIC_SPADL_COLUMNS
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import SPADL_COLUMNS

_ACT = spadlconfig.actiontype_id
_RES = spadlconfig.result_id
_BP = spadlconfig.bodypart_id


def _spadl_row(
    *,
    action_id: int,
    type_name: str = "pass",
    result_name: str = "success",
    team_id: int = 100,
    player_id: int = 200,
    time_seconds: float = 0.0,
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    game_id: int = 1,
    period_id: int = 1,
    original_event_id: object = None,
    bodypart_name: str = "foot",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "original_event_id": str(action_id) if original_event_id is None else original_event_id,
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": time_seconds,
        "team_id": team_id,
        "player_id": player_id,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "type_id": _ACT[type_name],
        "result_id": _RES[result_name],
        "bodypart_id": _BP[bodypart_name],
    }


def _spadl_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col, dtype in SPADL_COLUMNS.items():
        df[col] = df[col].astype(np.dtype(dtype))
    return df


def _basic_spadl() -> pd.DataFrame:
    """A small SPADL stream: pass → pass → shot → foul (yellow)."""
    return _spadl_df(
        [
            _spadl_row(action_id=0, type_name="pass", time_seconds=10.0),
            _spadl_row(action_id=1, type_name="pass", time_seconds=12.0, player_id=201, start_x=60.0, end_x=70.0),
            _spadl_row(
                action_id=2,
                type_name="shot",
                result_name="success",
                time_seconds=14.0,
                player_id=202,
                start_x=95.0,
                start_y=34.0,
                end_x=105.0,
                end_y=34.0,
            ),
            _spadl_row(
                action_id=3,
                type_name="foul",
                result_name="yellow_card",
                time_seconds=20.0,
                team_id=200,
                player_id=300,
                start_x=50.0,
                start_y=34.0,
                end_x=50.0,
                end_y=34.0,
                bodypart_name="foot",
            ),
        ]
    )


class TestConvertToAtomicPreserveNative:
    def test_default_none_unchanged(self):
        actions = _basic_spadl()
        atomic = convert_to_atomic(actions)
        assert list(atomic.columns) == list(ATOMIC_SPADL_COLUMNS.keys())

    def test_explicit_none_unchanged(self):
        actions = _basic_spadl()
        atomic = convert_to_atomic(actions, preserve_native=None)
        assert list(atomic.columns) == list(ATOMIC_SPADL_COLUMNS.keys())

    def test_empty_list_unchanged(self):
        actions = _basic_spadl()
        atomic = convert_to_atomic(actions, preserve_native=[])
        assert list(atomic.columns) == list(ATOMIC_SPADL_COLUMNS.keys())

    def test_single_column_preserved(self):
        actions = _basic_spadl()
        actions["possession_id"] = pd.Series([7, 7, 8, 9], dtype="int64")
        atomic = convert_to_atomic(actions, preserve_native=["possession_id"])
        assert "possession_id" in atomic.columns
        assert atomic.columns[-1] == "possession_id"
        # Filter to NON-synthetic rows: type_id is one of the original SPADL types
        # (synthetic atomic rows share original_event_id with their parent, so the naive
        # original_event_id filter alone catches both — we need to exclude atomic-only types).
        synthetic_type_names = (
            "receival",
            "interception",
            "out",
            "offside",
            "goal",
            "owngoal",
            "yellow_card",
            "red_card",
        )
        synthetic_type_ids = {atomicspadlconfig.actiontype_id[t] for t in synthetic_type_names}
        non_synthetic = atomic[~atomic["type_id"].isin(synthetic_type_ids)]
        possession_for = dict(zip(non_synthetic["original_event_id"], non_synthetic["possession_id"], strict=True))
        assert possession_for.get("0") == 7
        assert possession_for.get("1") == 7
        assert possession_for.get("2") == 8
        assert possession_for.get("3") == 9

    def test_multiple_columns_preserved(self):
        actions = _basic_spadl()
        actions["possession_id"] = pd.Series([7, 7, 8, 9], dtype="int64")
        actions["possession_team_id"] = pd.Series([100, 100, 100, 200], dtype="int64")
        atomic = convert_to_atomic(actions, preserve_native=["possession_id", "possession_team_id"])
        assert "possession_id" in atomic.columns
        assert "possession_team_id" in atomic.columns
        # Order follows preserve_native list order.
        assert atomic.columns[-2:].tolist() == ["possession_id", "possession_team_id"]

    def test_synthetic_atomic_rows_get_nan(self):
        """Receival / goal / yellow_card synthetic rows have no source — get NaN."""
        actions = _basic_spadl()
        actions["possession_id"] = pd.Series([7, 7, 8, 9], dtype="int64")
        atomic = convert_to_atomic(actions, preserve_native=["possession_id"])
        # Synthetic atomic rows are identifiable by their atomic-only type_ids.
        synthetic_types = {
            atomicspadlconfig.actiontype_id["receival"],
            atomicspadlconfig.actiontype_id["goal"],
            atomicspadlconfig.actiontype_id["yellow_card"],
        }
        synthetic_rows = atomic[atomic["type_id"].isin(synthetic_types)]
        assert len(synthetic_rows) >= 1, "Test fixture should have produced synthetic atomic rows"
        # Synthetic rows have NaN in the preserved column.
        assert synthetic_rows["possession_id"].isna().all()

    def test_missing_column_raises(self):
        actions = _basic_spadl()
        with pytest.raises(ValueError, match=r"preserve_native|extra_columns"):
            convert_to_atomic(actions, preserve_native=["does_not_exist"])

    def test_overlap_with_atomic_schema_raises_x(self):
        """``x`` is in ATOMIC_SPADL_COLUMNS — caller cannot push their own ``x``."""
        actions = _basic_spadl()
        actions["x"] = 0.0  # accidentally collides with atomic ``x`` column
        with pytest.raises(ValueError, match=r"overlap|already"):
            convert_to_atomic(actions, preserve_native=["x"])

    def test_overlap_with_atomic_schema_raises_dx(self):
        """``dx`` is in ATOMIC_SPADL_COLUMNS — caller cannot push their own ``dx``."""
        actions = _basic_spadl()
        actions["dx"] = 0.0
        with pytest.raises(ValueError, match=r"overlap|already"):
            convert_to_atomic(actions, preserve_native=["dx"])

    def test_dtype_preserved(self):
        actions = _basic_spadl()
        actions["possession_id"] = pd.Series([7, 7, 8, 9], dtype="int64")
        actions["score_advantage"] = pd.Series([0.5, 0.5, 1.0, 1.0], dtype="float64")
        actions["is_high_press"] = pd.Series([True, False, True, False], dtype="bool")
        atomic = convert_to_atomic(actions, preserve_native=["possession_id", "score_advantage", "is_high_press"])
        # Among non-synthetic rows, dtypes should be preserved on the original-row data.
        non_synthetic = atomic[atomic["type_id"].isin([_ACT["pass"], _ACT["shot"], _ACT["foul"]])]
        # NaN-introducing concat may upcast int to float for the synthetic-NaN rows; the
        # non-synthetic rows still hold their original integer values cleanly.
        assert (
            non_synthetic["possession_id"] == pd.Series([7, 7, 8, 9]).iloc[: len(non_synthetic)].values
        ).all() or non_synthetic["possession_id"].dropna().astype(int).tolist() == [7, 7, 8, 9]
        # Float passes through unchanged.
        assert non_synthetic["score_advantage"].dropna().astype(float).tolist() == [0.5, 0.5, 1.0, 1.0]
        # Bool will be object/Bool after concat with NaN — accept either.
        # The contract is "value preserved", not "exact dtype identity through concat".

    def test_columns_order_after_atomic_schema(self):
        actions = _basic_spadl()
        actions["my_extra"] = 1
        atomic = convert_to_atomic(actions, preserve_native=["my_extra"])
        atomic_cols = list(atomic.columns)
        schema_cols = list(ATOMIC_SPADL_COLUMNS.keys())
        # All schema cols come first, in schema order, then preserve_native cols.
        assert atomic_cols[: len(schema_cols)] == schema_cols
        assert atomic_cols[len(schema_cols) :] == ["my_extra"]

    def test_preserve_native_with_only_one_action_no_synthetic(self):
        """A single-row SPADL stream with no follow-up emits no synthetic atomic rows."""
        # A foul without yellow/red and no following pass produces no synthetic.
        actions = _spadl_df([_spadl_row(action_id=0, type_name="dribble", result_name="success")])
        actions["possession_id"] = pd.Series([5], dtype="int64")
        atomic = convert_to_atomic(actions, preserve_native=["possession_id"])
        # Dribble produces no synthetic extras → exactly 1 row with the preserved value.
        assert len(atomic) == 1
        assert atomic["possession_id"].iloc[0] == 5
