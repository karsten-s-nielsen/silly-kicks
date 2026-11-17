"""The ``time_source`` SPADL column and its predicates (action-time repair spec §4.1, §4.18)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.reflection import SPADL_REFLECTION_KINDS
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import schema
from silly_kicks.spadl.base import _add_dribbles
from silly_kicks.spadl.schema import TIME_SOURCE_VALUES, is_order_only, is_time_unmeasured, restart_kind
from silly_kicks.spadl.utils import _finalize_output
from tests.spadl._converter_cases import CONVERTER_CASES

_SCHEMAS = {
    "SPADL_COLUMNS": schema.SPADL_COLUMNS,
    "KLOPPY_SPADL_COLUMNS": schema.KLOPPY_SPADL_COLUMNS,
    "SPORTEC_SPADL_COLUMNS": schema.SPORTEC_SPADL_COLUMNS,
    "GRADIENTSPORTS_SPADL_COLUMNS": schema.GRADIENTSPORTS_SPADL_COLUMNS,
    "SKILLCORNER_SPADL_COLUMNS": schema.SKILLCORNER_SPADL_COLUMNS,
}


def _frame(n: int = 3, time_source: list[str | None] | None = None) -> pd.DataFrame:
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
            "start_x": [30.0, 50.0, 70.0][:n],
            "start_y": [34.0] * n,
            "end_x": [40.0, 60.0, 80.0][:n],
            "end_y": [34.0] * n,
            "type_id": [pid["pass"]] * n,
            "result_id": [1] * n,
            "bodypart_id": [0] * n,
            "shot_blocked": pd.array([pd.NA] * n, dtype="boolean"),
            "cross_blocked": pd.array([pd.NA] * n, dtype="boolean"),
            "cross_native": pd.array([pd.NA] * n, dtype="boolean"),
            "time_source": time_source if time_source is not None else ["event"] * n,
        }
    )


def test_vocabulary_is_exact():
    assert TIME_SOURCE_VALUES == ("event", "tracking", "elastic", "kick", "placeholder", "order_only", "derived")


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_time_source_in_every_schema(name):
    cols = _SCHEMAS[name]
    assert cols.get("time_source") == "object"
    keys = list(cols)
    assert keys.index("time_source") == keys.index("cross_native") + 1


def test_finalize_refuses_unknown_value():
    with pytest.raises(ValueError, match="time_source"):
        _finalize_output(_frame(time_source=["event", "bogus", "event"]))


def test_finalize_refuses_missing_value():
    with pytest.raises(ValueError, match="time_source"):
        _finalize_output(_frame(time_source=["event", None, "event"]))


def test_finalize_keeps_valid_values():
    out = _finalize_output(_frame(time_source=["tracking", "kick", "derived"]))
    assert out["time_source"].tolist() == ["tracking", "kick", "derived"]
    assert out["time_source"].dtype == object


def test_is_order_only_absent_column_all_false():
    f = _frame().drop(columns=["time_source"])
    assert is_order_only(f).tolist() == [False, False, False]
    assert is_time_unmeasured(f).tolist() == [False, False, False]


def test_is_order_only_marks_only_order_only():
    f = _frame(time_source=["placeholder", "order_only", "event"])
    assert is_order_only(f).tolist() == [False, True, False]
    assert isinstance(is_order_only(f), np.ndarray)


def test_is_time_unmeasured_both_values():
    f = _frame(time_source=["placeholder", "order_only", "kick"])
    assert is_time_unmeasured(f).tolist() == [True, True, False]


def test_restart_kind_map():
    tid = spadlconfig.actiontype_id
    assert restart_kind(tid["corner_crossed"]) == "corner"
    assert restart_kind(tid["corner_short"]) == "corner"
    assert restart_kind(tid["freekick_crossed"]) == "free_kick"
    assert restart_kind(tid["freekick_short"]) == "free_kick"
    assert restart_kind(tid["goalkick"]) == "goal_kick"
    assert restart_kind(tid["throw_in"]) == "throw_in"
    assert restart_kind(tid["shot_penalty"]) == "penalty"
    assert restart_kind(tid["pass"]) == "pass"
    assert restart_kind(tid["interception"]) == "interception"


def test_dribble_is_derived():
    f = _frame(n=2, time_source=["event", "event"])
    f.loc[1, "start_x"] = 55.0  # a gap of 15 m between the end of row 0 and the start of row 1
    out = _add_dribbles(f)
    dribble = out[out["type_id"] == spadlconfig.actiontype_id["dribble"]]
    assert len(dribble) == 1
    assert dribble["time_source"].tolist() == ["derived"]
    assert out.loc[out["type_id"] != spadlconfig.actiontype_id["dribble"], "time_source"].tolist() == ["event", "event"]


def test_atomic_carries_time_source_opt_in():
    actions = _finalize_output(_frame(time_source=["elastic", "event", "tracking"]))
    with_col = convert_to_atomic(actions, preserve_native=["time_source"])
    assert "time_source" in with_col.columns
    first = with_col.loc[with_col["original_event_id"] == "0", "time_source"]
    assert first.iloc[0] == "elastic"
    assert "time_source" not in convert_to_atomic(actions).columns


def test_reflection_declares_time_source():
    assert SPADL_REFLECTION_KINDS["time_source"] == "invariant"


_EXPECTED_NATIVE = {
    "statsbomb": "event",
    "opta": "event",
    "wyscout": "event",
    "sportec": "event",
    "metrica": "event",
    "kloppy": "event",
    "gradientsports": "tracking",
    "skillcorner": "tracking",
}


@pytest.mark.parametrize("name", sorted(CONVERTER_CASES))
def test_time_source_per_converter(name):
    case = CONVERTER_CASES[name]
    actions = case.run(case.build_input())
    assert "time_source" in actions.columns
    # The synthetic dribbles of _add_dribbles carry no provider event id; a provider's own dribble keeps its clock.
    is_dribble = (
        (actions["type_id"] == spadlconfig.actiontype_id["dribble"]) & actions["original_event_id"].isna()
    ).to_numpy()
    values = actions["time_source"].to_numpy()
    assert set(values[is_dribble]) <= {"derived"}
    other = set(values[~is_dribble])
    allowed = {_EXPECTED_NATIVE[name]}
    if name == "skillcorner":
        allowed |= {"placeholder", "order_only"}  # the derived restart and penalty rows (spec §4.2, §4.17)
    assert other <= allowed, (name, other)
    assert _EXPECTED_NATIVE[name] in other
