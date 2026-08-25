"""O(1) dtype-safe row-group lookup — replaces the rescan-in-loop anti-pattern (ADR-068).

The recurring defect the 2026-08-24 optimization audit found: a full-table filter
(``df[df["frame_id"] == fid]`` / ``df[(df["a"]==x) & (df["b"]==y)]``) executed inside a
per-item Python loop, which is O(n*m) where O(n) is available. :class:`RowGroups` builds the
grouping ONCE (O(n)) and serves each key in O(1).

Backed by ``groupby().indices`` (key -> positional int array), so NO group frames are copied at
construction — the memory cost is one int array of length ``n``, not a partition of the whole
table. ``df.take(pos)`` materialises only the requested group.

Keys are canonicalised via :func:`silly_kicks.id_compat.canonical_id` (ADR-019), so an
``Int64``-typed group column is still found by a Python ``int`` or ``str`` lookup key. A missing
key returns an EMPTY frame carrying ``df``'s columns and dtypes (never ``KeyError``) — matching
the ``df[df[k]==v]`` semantics it replaces, so downstream ``.empty`` / column access is unchanged.

This is a private leaf utility (the ``_geometry`` / ``_polygon`` position), consumed in-repo by
``causal/``, ``tracking/`` and ``spadl/``; see ``docs/PRIVATE_CONSUMERS.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id


class RowGroups:
    """O(1) dtype-safe row-group lookup over ``df``, grouped by ``by`` (ADR-068).

    See the module docstring. Build once, look up per item.
    """

    def __init__(self, df: pd.DataFrame, by: str | tuple[str, ...]) -> None:
        self._df = df
        self._by = (by,) if isinstance(by, str) else tuple(by)
        gb = df.groupby(list(self._by), sort=False)
        self._indices = {self._canon(k): v for k, v in gb.indices.items()}
        # Collision guard: canonicalisation collapses 366/366.0/"366" -> "366", so a mixed-dtype
        # key column would silently overwrite a group and lose its rows (the raw `df[df[k]==v]`
        # rescan kept them separate). Refuse loud rather than lose rows.
        if len(self._indices) != len(gb.indices):
            raise ValueError(
                f"group_rows: {len(gb.indices) - len(self._indices)} group key(s) collapsed under "
                f"ADR-019 canonicalisation on columns {self._by} -- the key column mixes dtypes "
                f"(e.g. int 366 and str '366'). Clean the key dtype before grouping."
            )

    def _canon(self, key):
        if len(self._by) == 1:
            return canonical_id(key)
        return tuple(canonical_id(k) for k in key)  # multi-key: `key` is a tuple

    def get(self, *key) -> pd.DataFrame:
        """Rows for ``key`` (O(1)); an empty frame (``df``'s columns/dtypes) on a miss.

        Single-key: ``get(fid)``. Multi-key: ``get(period_id, frame_id)`` (positional).
        """
        k = key[0] if (len(key) == 1 and len(self._by) == 1) else tuple(key)
        pos = self._indices.get(self._canon(k))
        if pos is None:
            return self._df.iloc[:0]
        return self._df.take(np.asarray(pos, dtype=np.intp))

    def __contains__(self, key) -> bool:
        # single-key: pass the scalar; multi-key: pass a tuple, e.g. `(2, 10) in groups`.
        return self._canon(key) in self._indices


def group_rows(df: pd.DataFrame, by: str | tuple[str, ...]) -> RowGroups:
    """Build a :class:`RowGroups` O(1) lookup over ``df`` grouped by ``by``. See the module docstring."""
    return RowGroups(df, by)
