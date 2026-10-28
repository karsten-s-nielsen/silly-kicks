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

from typing import cast

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
        # observed=True (ADR-105 Task 5/6): on a categorical key, the pandas-2 default (observed=False)
        # returns EVERY declared category as a group -- including zero-row ones -- and pandas 3 flips the
        # default to True (ADR-057), so an un-parameterised groupby is pandas-major-DEPENDENT + injects
        # phantom empty groups. Pinning True is deterministic across majors and byte-identical for the
        # current (non-categorical) keys. Defensive: ADR-103 introduced category frame columns.
        gb = df.groupby(list(self._by), sort=False, observed=True)
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

    def keys(self) -> list:
        """The CANONICAL group keys (ADR-019), each usable directly in :meth:`get` (splat a tuple).

        Single-key -> a list of canonical scalars; multi-key -> a list of canonical tuples. Lets an
        iterate-and-match caller (e.g. a batched two-frame arm) enumerate + look up groups dtype-safely
        instead of a raw ``dict(tuple(df.groupby(...)))``, whose raw-tuple keys mis-compare across a
        dtype skew between two frame sets.
        """
        return list(self._indices.keys())


def group_rows(df: pd.DataFrame, by: str | tuple[str, ...]) -> RowGroups:
    """Build a :class:`RowGroups` O(1) lookup over ``df`` grouped by ``by``. See the module docstring."""
    return RowGroups(df, by)


def ordered_group_rows(
    df: pd.DataFrame, by: list[str], *, sort: bool = True, dropna: bool = True
) -> list[tuple[tuple, np.ndarray]]:
    """``[(key, rows) for key, group in df.groupby(by, sort=sort, dropna=dropna)]`` without a frame per group.

    The per-frame loops of the extract paths (spec r4 D20) walk ``df.groupby(...)`` in its iteration
    order and pass each key on. This returns the same walk as row POSITIONS: ``key`` is the tuple the
    iteration yields, ``rows`` holds the group's positions in row order, and the list follows the
    iteration's group order, which ``ngroup()`` numbers by definition. The keys are the iteration's own
    objects. ``groupby().indices`` holds them in pandas 2.3 and 3.1, but not everywhere: pandas 3.0 holds
    numpy scalars there and iterates Python ones (A review R4IMPL-C3), and a missing key part
    (``dropna=False``) is spelled differently (pandas 3 spells a nullable column's NA ``nan`` there and
    ``<NA>`` in the iteration). So a few groups are iterated for real -- every missing-key group and one
    group per (key part, value type), `_type_anchors` -- in one small regroup of their first rows: the
    missing-key groups take their iterated keys, and the other groups take the conversion the probed ones
    show (`_item_parts`). Unlike :class:`RowGroups`, the keys are NOT canonicalised: the loops hand them on
    as they were.

    Two or more key columns only: a one-column groupby iterates its keys as Python scalars, which
    ``indices`` does not hold. A categorical key with an unobserved category is outside the domain (the
    iteration yields an empty group there); no tracking-frame key is categorical.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"period_id": [2, 1, 2, 1], "frame_id": [7, 9, 7, 3]})
    >>> groups = ordered_group_rows(df, ["period_id", "frame_id"], sort=False)
    >>> [([int(k) for k in key], rows.tolist()) for key, rows in groups]
    [([2, 7], [0, 2]), ([1, 9], [1]), ([1, 3], [3])]
    """
    if len(by) < 2:
        raise ValueError(f"ordered_group_rows: give two or more key columns, not {by!r}")
    groups = df.groupby(by, sort=sort, dropna=dropna)
    indices = groups.indices
    if not indices:
        return []
    number = groups.ngroup().to_numpy()
    walk = cast("list[tuple[tuple, np.ndarray]]", sorted(indices.items(), key=lambda group: number[group[1][0]]))
    missing = {i for i, (key, _) in enumerate(walk) if any(pd.isna(part) for part in key)}
    probe = sorted(missing | _type_anchors(walk, missing))
    firsts = df.iloc[[int(walk[i][1][0]) for i in probe]][by].assign(_walk_position=probe)
    iterated: dict[int, tuple] = {}
    for key, row in firsts.groupby(by, sort=sort, dropna=dropna):
        (i,) = row["_walk_position"].tolist()  # each probed group is its own walk group
        iterated[i] = cast("tuple", key)
    item_parts = _item_parts(walk, iterated, missing)
    for i, (key, rows) in enumerate(walk):
        if i in missing:
            walk[i] = (iterated[i], rows)
        elif item_parts:
            walk[i] = (tuple(p.item() if (j, type(p)) in item_parts else p for j, p in enumerate(key)), rows)
    return walk


def _type_anchors(walk: list[tuple[tuple, np.ndarray]], missing: set[int]) -> set[int]:
    """The walk position of the first group without a missing part that holds each (key part, value type).

    These groups are iterated for real (with the missing-key groups) for two reasons. Their keys show how this
    pandas iterates each part type (`_item_parts`). And they keep every key column's value types in the
    regroup: a regroup of the missing-key rows alone can type a column differently from the full frame (in
    pandas 2 an object column that holds only missing values becomes a float64 level, so its missing part comes
    back as ``np.float64(nan)`` where the full iteration yields the ``float`` ``nan`` of an object level; B
    review R4I-04)."""
    seen: set[tuple[int, type]] = set()
    out: set[int] = set()
    for i, (key, _) in enumerate(walk):
        if i in missing:
            continue
        for j, part in enumerate(key):
            if (j, type(part)) not in seen:
                seen.add((j, type(part)))
                out.add(i)
    return out


def _item_parts(walk: list[tuple[tuple, np.ndarray]], iterated: dict[int, tuple], missing: set[int]) -> set:
    """The (key part, ``indices`` value type) pairs that the iteration yields as the value's ``.item()``.

    pandas 3.0 holds numpy scalars in ``indices`` and iterates Python ones (A review R4IMPL-C3); pandas 2.3
    and 3.1 hold the iterated objects, so this is empty there. Learned on the probed groups without a missing
    part; any other difference between the two is refused, so a pandas that types its keys another way fails
    loud instead of handing the loops a different object."""
    same: set[tuple[int, type]] = set()
    item: set[tuple[int, type]] = set()
    for i, want in iterated.items():
        if i in missing:
            continue
        for j, (have, got) in enumerate(zip(walk[i][0], want, strict=True)):
            pair = (j, type(have))
            if type(got) is type(have) and got == have:
                same.add(pair)
            elif hasattr(have, "item") and type(have.item()) is type(got) and have.item() == got:
                item.add(pair)
            else:
                raise ValueError(
                    f"ordered_group_rows: pandas {pd.__version__} iterates key part {j} as "
                    f"{type(got).__name__} {got!r} where groupby().indices holds {type(have).__name__} {have!r}"
                )
    if same & item:
        raise ValueError(f"ordered_group_rows: pandas {pd.__version__} iterates one key part type two ways")
    return item
