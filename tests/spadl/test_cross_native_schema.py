"""The ``cross_native`` SPADL column (spec 2026-11-13-cross-definition-design §6.2, ADR-133)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.reflection import SPADL_REFLECTION_KINDS
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import schema
from silly_kicks.spadl.utils import _finalize_output

_SCHEMAS = {
    "SPADL_COLUMNS": schema.SPADL_COLUMNS,
    "KLOPPY_SPADL_COLUMNS": schema.KLOPPY_SPADL_COLUMNS,
    "SPORTEC_SPADL_COLUMNS": schema.SPORTEC_SPADL_COLUMNS,
    "GRADIENTSPORTS_SPADL_COLUMNS": schema.GRADIENTSPORTS_SPADL_COLUMNS,
    "SKILLCORNER_SPADL_COLUMNS": schema.SKILLCORNER_SPADL_COLUMNS,
}


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_cross_native_in_every_schema(name):
    cols = _SCHEMAS[name]
    assert cols.get("cross_native") == "boolean"
    keys = list(cols)
    assert keys.index("cross_native") == keys.index("cross_blocked") + 1


def test_cross_native_declared_non_geometric():
    assert SPADL_REFLECTION_KINDS["cross_native"] == "invariant"


def _frame(n: int = 3) -> pd.DataFrame:
    pid = spadlconfig.actiontype_id
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "original_event_id": [str(i) for i in range(n)],
            "action_id": list(range(n)),
            "period_id": [1] * n,
            "time_seconds": [float(i) for i in range(n)],
            "team_id": [7] * n,
            "player_id": [100 + i for i in range(n)],
            "start_x": [80.0, 95.0, 50.0][:n],
            "start_y": [5.0, 34.0, 34.0][:n],
            "end_x": [95.0, 100.0, 60.0][:n],
            "end_y": [34.0, 34.0, 34.0][:n],
            "type_id": [pid["cross"], pid["shot"], pid["pass"]][:n],
            "result_id": [1] * n,
            "bodypart_id": [0] * n,
            "shot_blocked": pd.array([pd.NA, False, pd.NA][:n], dtype="boolean"),
            "cross_blocked": pd.array([False, pd.NA, pd.NA][:n], dtype="boolean"),
            "cross_native": pd.array([True, pd.NA, False][:n], dtype="boolean"),
        }
    )


def test_finalize_keeps_nullable_cross_native():
    out = _finalize_output(_frame())
    assert str(out["cross_native"].dtype) == "boolean"
    assert out["cross_native"].tolist() == [True, pd.NA, False]


def test_finalize_casts_object_nan_to_boolean_na():
    f = _frame()
    f["cross_native"] = pd.Series([True, float("nan"), False], dtype=object)
    out = _finalize_output(f)
    assert out["cross_native"].tolist() == [True, pd.NA, False]


def test_atomic_preserves_cross_native_opt_in():
    actions = _finalize_output(_frame())
    with_col = convert_to_atomic(actions, preserve_native=["cross_native"])
    assert "cross_native" in with_col.columns
    assert str(with_col["cross_native"].dtype) == "boolean"
    # The cross's own atomic row keeps the value; a derived atomic row (the receival) carries <NA>, as
    # cross_blocked does.
    first = with_col.loc[with_col["original_event_id"] == "0", "cross_native"]
    assert first.iloc[0] == True  # noqa: E712 -- a nullable-boolean scalar
    assert first.iloc[1:].isna().all()
    assert "cross_native" not in convert_to_atomic(actions).columns
