"""The xShot / xCross extract reduces store a multi-provider corpus (ADR-128, Reviewer B CCHC-IMPL-10).

`_group` mixes kloppy str and Gradient Sports int game ids (the study groups on `astype(str)`). Parquet
cannot store such a column: the reduce stores the strings the study groups on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._stages import LocalStore, read_result, write_result


@pytest.mark.parametrize("module", ["scripts.train_xshot_occurrence", "scripts.train_xcross_attempt"])
def test_the_extract_reduce_stores_a_mixed_provider_group_column(tmp_path, module):
    import importlib

    reduce = importlib.import_module(module)._extract_reduce
    a = pd.DataFrame({"_group": np.array([10502, 10502], dtype=object), "f": [1.0, 2.0]})
    b = pd.DataFrame({"_group": np.array(["J03WMX"], dtype=object), "f": [3.0]})
    store = LocalStore(tmp_path)
    write_result(store, "s/_output", reduce({"a": a, "b": b}, None))
    back = read_result(store, "s/_output")
    legacy = np.asarray(pd.concat([a, b], ignore_index=True)["_group"].to_numpy()).astype(str)
    assert back["_group"].tolist() == legacy.tolist() == ["10502", "10502", "J03WMX"]
    one = reduce({"a": a}, None)
    assert one["_group"].tolist() == [10502, 10502]  # one type: unchanged


def test_the_receiver_rows_combine_stores_a_mixed_provider_pool(tmp_path):
    """Reviewer A T26-IMPL-01: the receiver pools a second dataset (`--pool-dataset`, another provider), so
    its rows combine may mix int and str ids even though the campaign runs it on sb360 alone today."""
    from scripts.train_receiver_model import _concat_non_empty

    a = pd.DataFrame({"game_id": [3829], "player_id": [11094], "team_id": [366.0], "label": [1]})
    b = pd.DataFrame(
        {"game_id": ["J03WMX"], "player_id": ["DFL-OBJ-0002DR"], "team_id": ["DFL-CLU-00000G"], "label": [0]}
    )
    store = LocalStore(tmp_path)
    write_result(store, "s/_output", _concat_non_empty({"a": a, "b": b}, None))
    back = read_result(store, "s/_output")
    assert back["game_id"].tolist() == ["3829", "J03WMX"] and back["team_id"].tolist() == ["366", "DFL-CLU-00000G"]
    assert back["label"].tolist() == [1, 0]
