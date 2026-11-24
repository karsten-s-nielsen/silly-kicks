"""R2: the shape-graph joins key on the canonical game id, never on an ``astype(str)`` re-key.

Spec 2026-11-08 tracking-batch §4.9 W9.4 ("the R2 ``astype(str)`` game re-keys are replaced by ``align_join_keys``")
and plan Task 5 step 3. An ``int`` ``game_id`` on one side and a ``str`` on the other is the same game (ADR-019): the
result must equal the result for aligned ids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.tracking.features import add_shape_graph, shape_graph_xfns
from tests._prod_fixtures import make_single_game_inputs

_COLUMNS = [
    f"shape_graph_{m}_{s}" for m in ("density", "n_edges", "mean_stability") for s in ("attacking", "defending")
]


def _str_game(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(game_id=df["game_id"].astype(str).astype(object))


@pytest.mark.parametrize("side", ["actions", "frames"])
def test_shape_graph_game_id_rekey_detection(side):
    actions, frames = make_single_game_inputs()
    aligned = add_shape_graph(actions, frames)[_COLUMNS]
    assert aligned.notna().to_numpy().any()  # non-vacuity: the aligned call links and scores frames
    if side == "actions":
        mixed = add_shape_graph(_str_game(actions), frames)
    else:
        mixed = add_shape_graph(actions, _str_game(frames))
    assert_bits_equal(mixed[_COLUMNS].reset_index(drop=True), aligned.reset_index(drop=True))


@pytest.mark.parametrize("side", ["actions", "frames"])
def test_shape_graph_xfn_game_id_rekey_detection(side):
    (transformer,) = shape_graph_xfns()
    actions, frames = make_single_game_inputs()
    aligned = transformer([actions] * 3, frames)
    assert np.isfinite(aligned.to_numpy(dtype="float64")).any()  # non-vacuity
    if side == "actions":
        mixed = transformer([_str_game(actions)] * 3, frames)
    else:
        mixed = transformer([actions] * 3, _str_game(frames))
    assert_bits_equal(mixed.reset_index(drop=True), aligned.reset_index(drop=True))
