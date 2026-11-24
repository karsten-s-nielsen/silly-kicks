"""An action that links to no frame reads a NaN frame id on every supported pandas (``pandas>=2.1.1``).

``pointers_by_action_row`` gives an unlinked action an ``Int64`` ``<NA>`` frame id. pandas 2.1 raises on
``to_numpy(dtype="float64")`` of a masked array that holds a missing value unless ``na_value`` is given; later
versions fill NaN. The Part Deux cycle found the same trap in the pre-change ``resolve_frame_ids_by_position``
(2026-11-23). These tests reach both conversion sites of ``tracking/_kernels.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking import add_defensive_line
from silly_kicks.tracking._kernels import resolve_frame_ids_by_position
from tests._prod_fixtures import make_single_game_inputs


def _with_unlinked_action() -> tuple[pd.DataFrame, pd.DataFrame]:
    actions, frames = make_single_game_inputs()
    late = actions.iloc[[-1]].copy()
    late["action_id"] = actions["action_id"].max() + 1
    late["time_seconds"] = float(frames["time_seconds"].max()) + 600.0  # ten minutes after the last frame
    return pd.concat([actions, late], ignore_index=True), frames


def test_resolve_frame_ids_by_position_unlinked_action_is_nan():
    actions, frames = _with_unlinked_action()
    out = resolve_frame_ids_by_position(actions, frames)
    assert out.dtype == np.float64
    assert len(out) == len(actions)
    assert np.isnan(out[-1])
    assert np.isfinite(out[:-1]).any()  # non-vacuity: the other actions link


def test_add_defensive_line_unlinked_action_is_nan():
    actions, frames = _with_unlinked_action()
    out = add_defensive_line(actions, frames)
    assert len(out) == len(actions)
    assert pd.isna(out["frame_id"].iloc[-1])
    assert pd.isna(out["defensive_line_x"].iloc[-1])
    assert out["frame_id"].iloc[:-1].notna().any()  # non-vacuity: the other actions link
