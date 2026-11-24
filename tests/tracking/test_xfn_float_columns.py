"""Nullable kernel columns leave a VAEP xfn as float64, whatever the data (owner ruling 2026-11-12).

``Series.to_numpy()`` on an ``Int64`` / ``boolean`` column gives int64 / bool when the column has no missing value,
and float64 / object when it has one. The xfn dtype then depends on the data: a two-game call and the one-game calls
differ (spec W9.6, measured on whole gradientsports matches), and an object column breaks the GBM learners. Every
other xfn emits float64, so these do too: NaN for missing, ``line_break`` as 1.0 / 0.0 / NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.invariants._link_population import resolve_call, split_by_game
from tests.test_multigame_linking import make_two_game_inputs

_MEMBERS = [
    "silly_kicks.tracking.features:defensive_line_xfns",
    "silly_kicks.tracking.features:off_ball_context_xfns",
]


@pytest.mark.parametrize("member", _MEMBERS)
def test_every_xfn_column_is_float64(member):
    actions, frames = make_two_game_inputs()
    for game_actions, game_frames in split_by_game(actions, frames):
        out = resolve_call(member)(game_actions, game_frames, None)
        assert isinstance(out, pd.DataFrame)
        assert {c: str(t) for c, t in out.dtypes.items()} == {c: "float64" for c in out.columns}
        mixed = out.isna().any() & out.notna().any()
        assert mixed.any(), "non-vacuity: some column must hold both a value and a missing value (the drift case)"


def test_line_break_is_one_zero_or_nan():
    actions, frames = make_two_game_inputs()
    (game_actions, game_frames), _ = split_by_game(actions, frames)
    out = resolve_call("silly_kicks.tracking.features:off_ball_context_xfns")(game_actions, game_frames, None)
    assert isinstance(out, pd.DataFrame)
    cells = out[["line_break_a0", "line_break_a1", "line_break_a2"]].to_numpy(dtype="float64").ravel()
    values = set(cells[~np.isnan(cells)].tolist())
    assert values <= {0.0, 1.0} and values, values
