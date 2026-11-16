"""The ``time_source`` SPADL provenance column (ADR-133, spec 2026-11-13 r8 §6.6 C).

``"tracking"`` where the provider times its events from its own tracking clock (Gradient Sports, SkillCorner,
Metrica -- measured in the plan's Task 12), ``"event"`` otherwise. Synthetic dribbles are ``"event"`` (their time is
interpolated). The column joins every SPADL schema after ``cross_native``, dtype ``object``, closed vocabulary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.reflection import SPADL_REFLECTION_KINDS
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import schema
from silly_kicks.spadl.utils import _finalize_output
from tests.spadl._converter_cases import CONVERTER_CASES

_DRIBBLE = spadlconfig.actiontype_id["dribble"]
_TRACKING_TIMED = {"gradientsports", "skillcorner", "metrica"}
_SCHEMAS = {
    "SPADL_COLUMNS": schema.SPADL_COLUMNS,
    "KLOPPY_SPADL_COLUMNS": schema.KLOPPY_SPADL_COLUMNS,
    "SPORTEC_SPADL_COLUMNS": schema.SPORTEC_SPADL_COLUMNS,
    "GRADIENTSPORTS_SPADL_COLUMNS": schema.GRADIENTSPORTS_SPADL_COLUMNS,
    "SKILLCORNER_SPADL_COLUMNS": schema.SKILLCORNER_SPADL_COLUMNS,
}


def test_vocabulary():
    assert schema.TIME_SOURCE_VALUES == ("tracking", "event")


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_time_source_in_every_schema_after_cross_native(name):
    cols = _SCHEMAS[name]
    assert cols.get("time_source") == "object"
    keys = list(cols)
    assert keys.index("time_source") == keys.index("cross_native") + 1


def test_declared_non_geometric():
    assert SPADL_REFLECTION_KINDS["time_source"] == "invariant"


def _run(name: str) -> pd.DataFrame:
    if name == "kloppy":
        pytest.importorskip("kloppy")
    case = CONVERTER_CASES[name]
    return case.run(case.build_input())


@pytest.mark.parametrize("name", sorted(CONVERTER_CASES))
def test_values_per_converter(name):
    out = _run(name)
    assert str(out["time_source"].dtype) == "object"
    assert set(out["time_source"]) <= set(schema.TIME_SOURCE_VALUES)
    synthetic_dribble = (out["type_id"] == _DRIBBLE) & out["original_event_id"].isna()
    if name == "skillcorner":
        synthetic_dribble = (out["type_id"] == _DRIBBLE) & (out["action_provenance"] == "derived")
    if name == "gradientsports":
        synthetic_dribble = pd.Series(False, index=out.index)  # GS never calls _add_dribbles
    expected_real = "tracking" if name in _TRACKING_TIMED else "event"
    assert (out.loc[~synthetic_dribble, "time_source"] == expected_real).all(), out["time_source"].value_counts()
    assert (out.loc[synthetic_dribble, "time_source"] == "event").all()


def test_non_vacuous_synthetic_dribbles():
    # The SkillCorner case produces at least one synthetic dribble, so the "event" branch above is exercised.
    out = _run("skillcorner")
    assert ((out["type_id"] == _DRIBBLE) & (out["action_provenance"] == "derived")).any()


def _frame(time_source) -> pd.DataFrame:
    pid = spadlconfig.actiontype_id
    f = pd.DataFrame(
        {
            "game_id": [1, 1],
            "original_event_id": ["0", "1"],
            "action_id": [0, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 2.0],
            "team_id": [7, 7],
            "player_id": [100, 101],
            "start_x": [50.0, 60.0],
            "start_y": [34.0, 34.0],
            "end_x": [60.0, 70.0],
            "end_y": [34.0, 34.0],
            "type_id": [pid["pass"], pid["dribble"]],
            "result_id": [1, 1],
            "bodypart_id": [0, 0],
            "shot_blocked": pd.array([pd.NA, pd.NA], dtype="boolean"),
            "cross_blocked": pd.array([pd.NA, pd.NA], dtype="boolean"),
            "cross_native": pd.array([False, pd.NA], dtype="boolean"),
        }
    )
    if time_source is not None:
        f["time_source"] = time_source
    return f


def test_finalize_defaults_a_missing_column_to_event():
    assert _finalize_output(_frame(None))["time_source"].tolist() == ["event", "event"]


def test_finalize_fills_nan_with_event():
    out = _finalize_output(_frame(["tracking", None]))
    assert out["time_source"].tolist() == ["tracking", "event"]


def test_finalize_rejects_a_value_outside_the_vocabulary():
    with pytest.raises(ValueError, match="time_source"):
        _finalize_output(_frame(["tracking", "video"]))


def test_finalize_does_not_mutate_its_input():
    f = _frame(["tracking", None])
    before = f.copy(deep=True)
    _finalize_output(f)
    pd.testing.assert_frame_equal(f, before)


def test_atomic_carries_time_source_opt_in():
    actions = _finalize_output(_frame(["tracking", "event"]))
    atomic = convert_to_atomic(actions, preserve_native=["time_source"])
    assert set(atomic["time_source"].dropna()) <= {"tracking", "event"}
    assert "time_source" not in convert_to_atomic(actions).columns
