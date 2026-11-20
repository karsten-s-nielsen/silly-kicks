"""The declared pandas floor (``pandas>=2.1.1``; owner ruling 2026-11-19: fix the code, keep 2.1.1).

On pandas 2.1, ``Int64`` with a missing value raises on ``.to_numpy(dtype=float)`` ("cannot convert to 'float64'-dtype
NumPy array with missing values"); later pandas returns NaN. The linked ``frame_id`` is ``Int64`` with ``<NA>`` for an
unlinked action, so every family that resolves frames through ``resolve_frame_ids_by_position`` raised on the floor.
These tests pass on every pandas this library declares; their red shows on the floor (the CI floor leg runs them).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._kernels import resolve_frame_ids_by_position


def _actions() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [1, 1], "action_id": [0, 1], "period_id": [1, 1], "time_seconds": [1.0, 99.0]})


def test_supplied_links_with_an_unlinked_action_give_nan():
    links = pd.DataFrame({"action_id": [0, 1], "frame_id": pd.array([10, pd.NA], dtype="Int64")})
    out = resolve_frame_ids_by_position(_actions(), pd.DataFrame(), links=links)
    assert out.dtype == np.float64
    assert out[0] == 10.0 and np.isnan(out[1])


def test_internal_links_with_an_unlinked_action_give_nan():
    frames = pd.DataFrame(
        {"game_id": [1, 1], "period_id": [1, 1], "frame_id": [10, 11], "time_seconds": [1.0, 1.1],
         "is_ball": [True, True], "x": [50.0, 50.0], "y": [34.0, 34.0], "source_provider": "gradientsports"}
    )  # fmt: skip
    out = resolve_frame_ids_by_position(_actions(), frames)
    assert out.dtype == np.float64
    assert out[0] == 10.0 and np.isnan(out[1])  # the action at 99 s has no frame near it
