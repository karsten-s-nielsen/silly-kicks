"""Tracking-frame columns as arrays, each read once (spec r4 D20).

The extract paths of ghost-GK, xShot and xCross walked their frames as one pandas DataFrame per frame. They now
walk row positions (:func:`silly_kicks._frame_index.ordered_group_rows`) and read every column through
:class:`FrameArrays`, which turns a column into an array once, on first use. A column the per-frame code never
read stays unread, so a frame set without it fails where it failed before.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FrameArrays:
    """The columns of ``frames`` as arrays, each built once, on first use.

    ``frames[name]`` is the column's own storage (``to_numpy()``, no upcast), for the numeric columns: element ``i``
    is the value ``frames[name].iloc[i]`` read, and float32 storage reaches the scalar code as before (ADR-106).
    ``cells(name)`` is the column's backing array (``Series.array``), for the cells a loop hands on: its element ``i``
    IS ``frames[name].iloc[i]``, the same object type on every dtype (a nullable or categorical id included).

    Both are READ-ONLY by contract: no copy is made, so a caller reads them through fancy indexing (which copies)
    and never writes into them or into a view of them. The flag ``writeable = False`` is NOT set: under pandas 2
    the array can be the frame's own cached column view, and the flag would then reach the caller's frame.
    """

    def __init__(self, frames: pd.DataFrame) -> None:
        self._frames = frames
        self._arrays: dict[str, np.ndarray] = {}
        self._cells: dict[str, Any] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        array = self._arrays.get(name)
        if array is None:
            array = self._arrays[name] = self._frames[name].to_numpy()
        return array

    def cells(self, name: str) -> Any:
        """``frames[name].array``: index it with a position for the cell ``Series.iloc`` returned."""
        cells = self._cells.get(name)
        if cells is None:
            cells = self._cells[name] = self._frames[name].array
        return cells

    def has(self, name: str) -> bool:
        return name in self._frames.columns


#: :func:`first_other`'s answer when no row qualifies (a team id can be any value, ``None`` included).
NONE_OTHER = object()


def first_other(cells: Any, rows: np.ndarray, value: Any) -> Any:
    """The first cell among ``rows`` (in row order) that is ``!=`` ``value``, else :data:`NONE_OTHER`.

    It is the head of the per-frame ``[t for t in column.iloc[rows].unique() if t != value]`` (spec r4 D20):
    ``unique`` lists each value at its first row, so the first entry that differs from ``value`` is the value of
    the first row that differs. ``rows`` holds no missing cell (the old ``dropna``).
    """
    for row in rows:
        cell = cells[row]
        # Raw `!=` on purpose (not `id_compat`): the `2474b73` body compared with `!=`, and D20 reproduces it byte
        # for byte. A dtype-safe rewrite here would change which cell is "other" for mixed-dtype ids.
        if cell != value:
            return cell
    return NONE_OTHER
