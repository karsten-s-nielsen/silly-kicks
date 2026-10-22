"""The ONE combine of unit frames into a stage output (ADR-128): `_stages.combine_frames`.

Real HF, Cycle 1: a multi-provider corpus mixes int ids (gradientsports: game_id int64, player_id Int64,
team_id float64) with str ids (idsse / skillcorner: str, team_id category). Each unit stores fine; the
COMBINED stage output failed its parquet write in the reduce, after every unit ran (ArrowInvalid on
`game_id`, then `team_id` 'DFL-CLU-00000G'). The combine makes every declared id column storable and refuses
any other column that mixes types, by name, before parquet does."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import _stages as s
from scripts._toy_corpora import staged_frames


def _units() -> list[pd.DataFrame]:
    base = pd.DataFrame(
        {
            "game_id": ["100", "100", "100"],
            "player_id": ["ball", "p1", "a10"],
            "team_id": [None, 1, 2],
            "x": [1.0, 2.0, 3.0],
        }
    )
    gs = staged_frames(base, "int")  # the measured gradientsports dtypes
    sc = staged_frames(base.assign(game_id="101"), "str")  # the measured skillcorner dtypes
    return [gs, sc]


def _roundtrip(tmp_path, frame):
    store = s.LocalStore(tmp_path)
    s.write_result(store, "st/_output", frame)
    return s.read_result(store, "st/_output")


def test_the_raw_concat_of_real_provider_dtypes_cannot_be_stored(tmp_path):
    """The precondition (the Cycle 1 failure, reproduced locally): without the combine, the reduce fails."""
    raw = pd.concat(_units(), ignore_index=True)
    with pytest.raises(TypeError, match="game_id"):  # the contract guard names the column, not ArrowInvalid
        _roundtrip(tmp_path, raw)


def test_the_combine_stores_every_declared_id_column_as_the_legacy_strings(tmp_path):
    out = s.combine_frames(_units(), ids=("game_id", "player_id", "team_id"))
    back = _roundtrip(tmp_path, out)
    assert back["game_id"].tolist() == ["100", "100", "100", "101", "101", "101"]
    assert back["player_id"].tolist()[1:3] == ["1001", "2010"] and back["player_id"].tolist()[4:] == ["p1", "a10"]
    assert back["team_id"].tolist()[1:3] == ["1", "2"] and back["team_id"].tolist()[4:] == ["1", "2"]
    assert back["player_id"].isna().tolist() == [True, False, False, True, False, False]  # the ball stays missing
    assert back["team_id"].isna().tolist() == [True, False, False, True, False, False]
    assert back["x"].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_a_single_provider_combine_leaves_the_ids_untouched():
    gs = _units()[0]
    out = s.combine_frames([gs, gs], ids=("game_id", "player_id", "team_id"))
    assert str(out["game_id"].dtype) == "int64" and str(out["player_id"].dtype) == "Int64"
    assert str(out["team_id"].dtype) == "float64"


def test_an_undeclared_column_that_mixes_types_refuses_by_name():
    a = pd.DataFrame({"game_id": [1], "note": [3]})
    b = pd.DataFrame({"game_id": [2], "note": ["x"]})
    with pytest.raises(TypeError, match="note"):
        s.combine_frames([a, b], ids=("game_id",))


def test_int_and_float_values_are_not_a_mix():
    a = pd.DataFrame({"v": np.array([1, 2], dtype=object)})
    b = pd.DataFrame({"v": np.array([1.5], dtype=object)})
    out = s.combine_frames([a, b], ids=())
    assert out["v"].tolist() == [1, 2, 1.5]


def test_empty_parts_combine_to_an_empty_frame():
    assert s.combine_frames([], ids=("game_id",)).empty


def test_a_nested_column_parquet_can_store_is_not_a_mix(tmp_path):
    """Reviewer B CCHC-IMPL-14: `infer_dtype` calls list / array / dict columns "mixed" too; only a column
    pyarrow cannot convert is a mix."""
    frame = pd.DataFrame(
        {
            "xy": [[1.0, 2.0], [3.0, 4.0]],
            "arr": [np.array([1.0, 2.0]), np.array([3.0])],
            "meta": [{"a": 1}, {"a": 2}],
            "game_id": [1, 2],
        }
    )
    s.assert_storable(frame)  # no refusal
    back = _roundtrip(tmp_path, frame)
    assert back["xy"].map(list).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(TypeError, match="pid"):
        s.assert_storable(pd.DataFrame({"pid": np.array([1, "a"], dtype=object)}))
