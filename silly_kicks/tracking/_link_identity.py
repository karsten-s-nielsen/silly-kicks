"""Pointer identity for the W9 game-aware linker: action fingerprints and the positional binding (owner ruling R9).

The linker's pointers carry ``action_row`` (the 0-based row of the action in the ``actions`` frame passed to the
linker) and ``action_fingerprint`` (a frozen BLAKE2b-64 of the action's identity). A consumer finds each pointer's
action by position through :func:`resolve_link_positions`, which verifies every position: ``links`` must be built from
exactly this ``actions`` frame, in this row order. A function that re-sorts, filters, or subsets ``actions`` before it
passes ``links`` on maps the pointers through its permutation with :func:`remap_links`.

**The fingerprint is a contract** (spec 2026-11-08 tracking-batch §4.9 W9.2). Each action row gives the UTF-8 record
``FINGERPRINT_RECORD_FORMAT``: ``game``, ``period``, and ``action`` are the ``id_compat.canonical_id`` strings, ``time``
is ``float(time_seconds).hex()`` after an exact upcast to float64, and a missing value (an NA id or a NaN time) is the
token ``"<NA>"``. A frame with no ``game_id`` column is one game, so its ``game`` is ``"<NA>"`` too. The fingerprint
is ``int.from_bytes(blake2b(record, digest_size=8).digest(), "little")``. It does not depend on pandas or the
platform. A change to it is a contract change: ``tests/tracking/test_link_identity.py`` freezes the records and the
vector.

The public path is :mod:`silly_kicks.tracking.utils` (``action_fingerprints``, ``FINGERPRINT_RECORD_FORMAT``).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
from collections.abc import Iterator

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id, canonical_id_series

__all__ = [
    "FINGERPRINT_RECORD_FORMAT",
    "LinkedFrameRows",
    "action_fingerprints",
    "append_columns_by_action_row",
    "append_pointer_columns",
    "assemble_by_game",
    "fingerprint_records",
    "frame_join_keys",
    "game_keys",
    "has_game_key",
    "iter_games",
    "link_identity_scope",
    "linked_frame_rows",
    "pointers_by_action_row",
    "pointers_with_action_columns",
    "remap_links",
    "require_links_bound",
    "require_unambiguous_action_ids",
    "resolve_link_positions",
]

FINGERPRINT_RECORD_FORMAT = "{game}|{period}|{action}|{time}"

#: The record token for a missing value, and the canonical key of an NA game (spec W9.2, Review Focus 2).
NA_TOKEN = "<NA>"  # noqa: S105 -- the missing-value record token, not a secret ("TOKEN" in the name)

_MISMATCH = "links must be built from exactly this actions frame, in this row order"

#: The temporary positional merge key of :func:`append_pointer_columns` (never a column of the result).
_ROW_KEY = "__silly_kicks_action_row__"

_LEGACY_AMBIGUOUS = (
    "links has no action_row (pointers built before the game-aware linker), and an action_id join is ambiguous "
    "here: actions span more than one game, or action_id repeats in links or in actions (VAEP gamestate slots). "
    "re-link with link_actions_to_frames on exactly this actions frame"
)


def _canonical_strings(values: pd.Series) -> np.ndarray:
    """The ``canonical_id`` string of each value, with ``"<NA>"`` for a missing one (object array).

    Canonicalises each DISTINCT value once (``pd.factorize``), so a frame-sized column costs one hash pass plus one
    ``canonical_id`` call per distinct value. Distinct raw values that canonicalise alike (``7`` and ``"7"``) map to
    the same string, which is the ADR-019 alignment.
    """
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    canon = canonical_id_series(pd.Series(uniques)).astype(object)
    table = np.empty(len(canon) + 1, dtype=object)
    table[: len(canon)] = canon.where(canon.notna(), NA_TOKEN).to_numpy(dtype=object)
    table[len(canon)] = NA_TOKEN  # code -1 (a missing value) reads the last slot
    return table[codes]


def game_keys(game_ids: pd.Series) -> np.ndarray:
    """The linker's game key per row: the canonical ``game_id`` string, ``"<NA>"`` for an NA game (object array)."""
    return _canonical_strings(game_ids)


def has_game_key(actions: pd.DataFrame, frames: pd.DataFrame) -> bool:
    """True when both ``actions`` and ``frames`` carry ``game_id``. When either side has none, the call is treated
    as one game, as before W9 (the one rule of the linker and of every pointer-to-frame join)."""
    return "game_id" in actions.columns and "game_id" in frames.columns


def frame_join_keys(actions: pd.DataFrame, frames: pd.DataFrame) -> list[str]:
    """The key of a pointer-to-frame join (spec W9.4): ``["game_id", "period_id", "frame_id"]``, with the action's
    own ``game_id`` and ``period_id`` read by position.

    A key column takes part when BOTH frames carry it. Without a game key (:func:`has_game_key`) the call is one
    game, so the key is ``["period_id", "frame_id"]``. A minimal input without ``period_id`` on one side joins on
    the rest, as it did before W9. ``frame_id`` is always last.
    """
    shared = [c for c in ("game_id", "period_id") if c in actions.columns and c in frames.columns]
    return [*shared, "frame_id"]


def iter_games(actions: pd.DataFrame, frames: pd.DataFrame) -> Iterator[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(rows, game_actions, game_frames)`` for each game of ``actions``, in first-appearance order.

    ``rows`` are the positions of that game's actions in ``actions``, ascending, and ``game_actions`` is
    ``actions.iloc[rows]``. ``game_frames`` are the frames of the same game (canonical game keys, ADR-019). A
    function whose frame join is not a pointer join (a time window from ``slice_around_event``) runs once per game
    and writes its result at ``rows``, so a window never reads another game's frames.

    When the call is one game (no game key, or one game on both sides), or ``actions`` is empty, it yields the
    inputs themselves, once: no copy. Put per-game ``add_*`` results back in action order with
    :func:`assemble_by_game`.
    """
    if not has_game_key(actions, frames):
        yield np.arange(len(actions)), actions, frames
        return
    action_games, frame_games = game_keys(actions["game_id"]), game_keys(frames["game_id"])
    order = list(dict.fromkeys(action_games.tolist()))
    if not order or (len(order) == 1 and bool((frame_games == order[0]).all())):
        yield np.arange(len(actions)), actions, frames
        return
    for game in order:
        rows = np.flatnonzero(action_games == game)
        yield rows, actions.iloc[rows], frames[frame_games == game]


def assemble_by_game(parts: list[tuple[np.ndarray, pd.DataFrame]]) -> pd.DataFrame:
    """The per-game results of an ``add_*`` function as one frame, in the row order of ``actions``.

    ``parts`` holds ``(rows, result)`` for each game of :func:`iter_games`: ``result`` has one row per action of
    that game, and ``rows`` are their positions in ``actions``. One part is returned as it is.
    """
    if len(parts) == 1:
        return parts[0][1]
    order = np.argsort(np.concatenate([rows for rows, _ in parts]), kind="stable")
    out = pd.concat([result for _, result in parts])
    # pandas concat turns an all-NA object column (pd.NA / None) into nan: rebuild each object column from
    # the parts' own values, so a game keeps the missing-value token its result had.
    for column in out.columns:
        if all(result[column].dtype == object for _, result in parts):
            values = np.concatenate([result[column].to_numpy(dtype=object) for _, result in parts])
            out[column] = pd.Series(values, index=out.index, dtype=object)  # pandas 3 infers `str` from a raw array
    return out.iloc[order]


class LinkedFrameRows:
    """The frame rows of an action's linked frame, looked up by action row (spec W9.4).

    A frame is found on ``(game, period_id, frame_id)`` with the action's OWN game and period, read by position.
    ``frame_id`` alone is not unique across periods, and ``(period_id, frame_id)`` is not unique across games. The
    game key is the linker's: canonical strings, so an int ``game_id`` on one side and a str on the other match
    (ADR-019), and an NA game reads only NA-game frames. Without a game key (:func:`has_game_key`) the call is one
    game. Build it once per call (``group_rows``, ADR-068), then look up per action.

    :meth:`get` reads the game and period of a row of ``actions``. :meth:`at` takes an explicit key, for a row
    that is not a row of ``actions`` (a VAEP gamestate slot). :meth:`key` is that key in hashable form, for a
    per-frame cache: a cache keyed on ``(period_id, frame_id)`` alone serves one game's frame to another game.
    """

    def __init__(self, actions: pd.DataFrame, frames: pd.DataFrame) -> None:
        self._frames = frames
        join_keys = frame_join_keys(actions, frames)  # the key columns that both sides carry
        keys: dict = {}
        self._game: np.ndarray | None = None
        if "game_id" in join_keys:
            self._game = game_keys(actions["game_id"])
            keys["game"] = game_keys(frames["game_id"])
        self._period: np.ndarray | None = None
        if "period_id" in join_keys:
            self._period = actions["period_id"].to_numpy()
            keys["period_id"] = frames["period_id"].array
        keys["frame_id"] = frames["frame_id"].array
        self._groups = group_rows(pd.DataFrame(keys), tuple(keys))

    def group_key(self, game_id, period_id) -> tuple:
        """The hashable key of ``period_id`` in ``game_id``, for a per-period table (a time window of frames).
        A part that is not in the join key (no game key, or no ``period_id`` on one side) is left out."""
        key: tuple = ()
        if self._game is not None:
            game = canonical_id(game_id)
            key = (NA_TOKEN if pd.isna(game) else game,)
        if self._period is not None:
            key = (*key, canonical_id(period_id))
        return key

    def key(self, game_id, period_id, frame_id) -> tuple:
        """The hashable key of frame ``frame_id`` in ``game_id`` and ``period_id``: :meth:`group_key` plus the
        frame."""
        return (*self.group_key(game_id, period_id), canonical_id(frame_id))

    def at(self, game_id, period_id, frame_id) -> pd.DataFrame:
        """The rows of frame ``frame_id`` in ``game_id`` and ``period_id``; an empty frame (the columns and dtypes
        of ``frames``) when ``frame_id`` is missing or names no frame there."""
        if pd.isna(frame_id):
            return self._frames.iloc[:0]
        positions = self._groups.positions(*self.key(game_id, period_id, frame_id))
        return self._frames.take(positions) if len(positions) else self._frames.iloc[:0]

    def get(self, action_row: int, frame_id) -> pd.DataFrame:
        """The rows of frame ``frame_id`` for the action at ``action_row`` of ``actions``: its own game and period,
        read by position. Empty as :meth:`at`."""
        game = None if self._game is None else self._game[action_row]
        period = None if self._period is None else self._period[action_row]
        return self.at(game, period, frame_id)


def linked_frame_rows(actions: pd.DataFrame, frames: pd.DataFrame, pointers: pd.DataFrame) -> list[pd.DataFrame]:
    """The frame rows of each action's linked frame, one entry per ``actions`` row, in row order (spec W9.4).

    ``pointers`` are read by position (:func:`pointers_by_action_row`, verified against ``actions``) and each frame
    is found on the action's own game and period (:class:`LinkedFrameRows`). An action without a linked frame gets
    an empty frame with the columns of ``frames``. It replaces the ``action_id -> frame_id -> frames.groupby(...)``
    lookup of a per-action loop.
    """
    frame_ids = pointers_by_action_row(actions, pointers)["frame_id"].to_numpy(dtype=object)
    rows = LinkedFrameRows(actions, frames)
    return [rows.get(i, fid) for i, fid in enumerate(frame_ids)]


def fingerprint_records(actions: pd.DataFrame) -> list[str]:
    """The canonical record string of each action row (``FINGERPRINT_RECORD_FORMAT``), built once per column."""
    if len(actions) == 0:
        return []
    # no game_id column: one game with the key "<NA>", the linker's one-game rule (has_game_key)
    games = (
        _canonical_strings(actions["game_id"])
        if "game_id" in actions.columns
        else np.full(len(actions), NA_TOKEN, dtype=object)
    )
    periods = _canonical_strings(actions["period_id"])
    action_ids = _canonical_strings(actions["action_id"])
    times = actions["time_seconds"].to_numpy(dtype=np.float64, na_value=np.nan)  # exact upcast (float32 too)
    return [
        f"{g}|{p}|{a}|{NA_TOKEN if t != t else float(t).hex()}"
        for g, p, a, t in zip(games, periods, action_ids, times, strict=True)
    ]


def action_fingerprints(actions: pd.DataFrame) -> np.ndarray:
    """The frozen uint64 fingerprint of each action row: BLAKE2b-64 of its record, read little-endian.

    Examples
    --------
    One fingerprint per action row, stable across platforms and pandas versions::

        from silly_kicks.tracking.utils import action_fingerprints
        fp = action_fingerprints(actions)  # np.ndarray, dtype uint64, len(actions)
    """
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(rec.encode("utf-8"), digest_size=8).digest(), "little")
            for rec in fingerprint_records(actions)
        ),
        dtype=np.uint64,
        count=len(actions),
    )


# The per-call cache: a list of (frame, fingerprints) entries, or None outside any scope. An entry holds a STRONG
# reference to its frame, and a lookup hits only on `entry frame is actions` -- never a bare id(), which CPython reuses
# after a frame is freed (spec W9.2 cost note).
_CACHE: contextvars.ContextVar[list[tuple[pd.DataFrame, np.ndarray]] | None] = contextvars.ContextVar(
    "silly_kicks_link_identity_cache", default=None
)


@contextlib.contextmanager
def link_identity_scope() -> Iterator[None]:
    """Open the fingerprint cache for one top-level call; it is cleared when the outermost scope exits.

    Nested scopes (a public entry point that calls another) share the outer cache.

    Examples
    --------
    Verify one pointer frame against one ``actions`` frame several times, and hash the actions once::

        from silly_kicks.tracking.utils import link_identity_scope, resolve_link_positions
        with link_identity_scope():
            rows = resolve_link_positions(actions, links)
            rows_again = resolve_link_positions(actions, links)  # served from the scope's cache
    """
    if _CACHE.get() is not None:
        yield
        return
    token = _CACHE.set([])
    try:
        yield
    finally:
        _CACHE.reset(token)


def _cached_fingerprints(actions: pd.DataFrame) -> np.ndarray:
    """``action_fingerprints(actions)``, served from the open scope's cache when the same frame object is in it."""
    entries = _CACHE.get()
    if entries is None:
        return action_fingerprints(actions)
    for frame, fingerprints in entries:
        if frame is actions:
            return fingerprints
    fingerprints = action_fingerprints(actions)
    entries.append((actions, fingerprints))
    return fingerprints


def _cache_size() -> int:
    """The number of frames in the open scope's cache (0 outside a scope)."""
    entries = _CACHE.get()
    return 0 if entries is None else len(entries)


def _verify_positions(actions: pd.DataFrame, links: pd.DataFrame, rows: np.ndarray) -> None:
    """Raise ``ValueError`` naming the first ``links`` row whose ``action_row`` is out of range for ``actions``, or
    whose ``action_fingerprint`` differs from the fingerprint of ``actions.iloc[action_row]`` (spec W9.5)."""
    if "action_fingerprint" not in links.columns:
        raise ValueError(f"{_MISMATCH}; links carries action_row but no action_fingerprint")
    n = len(actions)
    in_range = (rows >= 0) & (rows < n)
    ok = in_range.copy()
    if n and in_range.any():
        fingerprints = _cached_fingerprints(actions)
        want = links["action_fingerprint"].to_numpy(dtype=np.uint64)
        ok[in_range] = fingerprints[rows[in_range]] == want[in_range]
    if not ok.all():
        raise ValueError(f"{_MISMATCH}; first mismatch at links row {int(np.flatnonzero(~ok)[0])}")


def require_unambiguous_action_ids(actions: pd.DataFrame) -> None:
    """Raise ``ValueError`` when ``action_id`` does not name one row of ``actions``: the frame spans more than one
    game, or an ``action_id`` repeats (VAEP gamestate slots). It is the ``actions`` half of the W9.5 rule for
    legacy pointers, which carry no ``action_row`` and can only join on ``action_id``."""
    single_game = "game_id" not in actions.columns or len(set(game_keys(actions["game_id"]))) <= 1
    if not single_game or pd.Series(_canonical_strings(actions["action_id"])).duplicated().any():
        raise ValueError(_LEGACY_AMBIGUOUS)


def _legacy_positions(actions: pd.DataFrame, links: pd.DataFrame) -> np.ndarray:
    """The W9.5 rule for pointers without ``action_row``: an ``action_id`` join, only when it is unambiguous."""
    link_ids = pd.Series(_canonical_strings(links["action_id"]))
    action_ids = pd.Series(_canonical_strings(actions["action_id"]))
    if link_ids.duplicated().any():
        raise ValueError(_LEGACY_AMBIGUOUS)
    require_unambiguous_action_ids(actions)
    position_of = dict(zip(action_ids.tolist(), range(len(action_ids)), strict=True))
    positions = np.fromiter((position_of.get(k, -1) for k in link_ids), dtype=np.int64, count=len(link_ids))
    if (positions < 0).any():
        first = int(np.flatnonzero(positions < 0)[0])
        raise ValueError(f"{_MISMATCH}; first mismatch at links row {first} (its action_id is not in actions)")
    return positions


def resolve_link_positions(actions: pd.DataFrame, links: pd.DataFrame) -> np.ndarray:
    """The int64 position in ``actions`` of each ``links`` row (spec W9.5).

    - ``links`` with ``action_row``: each row is range-checked and its ``action_fingerprint`` is compared with the
      fingerprint of ``actions.iloc[action_row]``. A mismatch raises ``ValueError("links must be built from exactly
      this actions frame, in this row order; first mismatch at links row {i}")``.
    - legacy ``links`` (no ``action_row``): joined on ``action_id`` when ``actions`` span one game and ``action_id`` is
      unique in both frames; otherwise ``ValueError`` (re-link). An ambiguous join is never guessed.

    Examples
    --------
    Read each action's linked ``frame_id`` by position, never through an ``action_id`` join::

        import numpy as np
        from silly_kicks.tracking.utils import link_actions_to_frames, resolve_link_positions
        links, _report = link_actions_to_frames(actions, frames)
        rows = resolve_link_positions(actions, links)  # links row i points at actions.iloc[rows[i]]
        frame_id = np.full(len(actions), np.nan)
        frame_id[rows] = links["frame_id"].to_numpy(dtype="float64", na_value=np.nan)
    """
    if "action_row" not in links.columns:
        return _legacy_positions(actions, links)
    rows = links["action_row"].to_numpy(dtype=np.int64)
    _verify_positions(actions, links, rows)
    return rows


#: Nullable dtypes that keep the identity columns exact when an action has no pointer (a float64 NaN fill would
#: round a uint64 fingerprint above 2**53).
_IDENTITY_NULLABLE = {"action_row": "Int64", "action_fingerprint": "UInt64"}


def pointers_by_action_row(actions: pd.DataFrame, pointers: pd.DataFrame) -> pd.DataFrame:
    """``pointers`` with one row per ``actions`` row, in ``actions`` row order (a ``RangeIndex``).

    Row ``i`` is the pointer of ``actions.iloc[i]``, found by position through :func:`resolve_link_positions`, so
    ``pointers`` are verified against ``actions`` (spec W9.4, W9.5). It is the one pointer-to-action join: a
    consumer reads a pointer column at an action row, and never joins on ``action_id``.

    - When two pointers name one action, the first wins.
    - An action with no pointer gets an all-NA row. The other columns then follow the ``reindex`` dtype rules, and
      the identity columns become ``Int64`` / ``UInt64``.
    - A pointer frame with exactly one pointer per action keeps every column dtype.
    """
    rows = resolve_link_positions(actions, pointers)
    table = pointers.set_axis(rows, axis=0)
    if not table.index.is_unique:
        table = table[~table.index.duplicated(keep="first")]
    if len(table) != len(actions):
        table = table.astype({c: t for c, t in _IDENTITY_NULLABLE.items() if c in table.columns})
    return table.reindex(pd.RangeIndex(len(actions)))


def append_pointer_columns(
    out: pd.DataFrame, actions: pd.DataFrame, pointers: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """``out`` with the ``columns`` of each action's pointer appended, joined by position (spec W9.4).

    ``out`` has one row per ``actions`` row, in the same order (an ``add_*`` result under construction). It
    replaces ``out.merge(pointers.set_index("action_id")[columns], left_on="action_id", right_index=True,
    how="left")``: the values come from :func:`pointers_by_action_row`, so no ``action_id`` join takes part. On one
    game with unique ids the result is bit-identical to that merge, with the same index and the same suffixes for
    a column that ``out`` already has. ``out`` is not written to.
    """
    return append_columns_by_action_row(out, pointers_by_action_row(actions, pointers)[list(columns)])


def append_columns_by_action_row(out: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """``out`` with the columns of ``table`` appended, where the index of ``table`` is the action row.

    ``out`` has one row per action, in action order. It is a left merge on the row position, so it replaces a
    merge on ``action_id`` (``out.merge(table, left_on="action_id", right_index=True, how="left")``) and keeps
    its result on one game with unique ids: the same index, NaN for an action with no row in ``table``, and the
    same suffixes for a column that ``out`` already has. ``out`` is not written to.
    """
    keyed = out.assign(**{_ROW_KEY: np.arange(len(out), dtype=np.int64)})
    return keyed.merge(table, left_on=_ROW_KEY, right_index=True, how="left").drop(columns=_ROW_KEY)


def pointers_with_action_columns(actions: pd.DataFrame, pointers: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each ``pointers`` row, in pointer order, with ``action_row`` and the named columns of its action.

    The action is found by position through :func:`resolve_link_positions`, never by ``action_id`` (spec W9.4), so
    a pointer reads the ``game_id``, ``period_id``, or ``team_id`` of exactly its own action. The result has a
    ``RangeIndex``. The pointer columns come first and keep their dtypes, then ``action_row`` when ``pointers`` are
    legacy links without it, then ``columns`` with the dtypes they have in ``actions``. When ``pointers`` already
    has a column named in ``columns``, the value of the action replaces it.
    """
    rows = resolve_link_positions(actions, pointers)
    out = pointers.drop(columns=[c for c in columns if c in pointers.columns]).reset_index(drop=True)
    if "action_row" not in out.columns:
        out["action_row"] = rows
    return pd.concat([out, actions[columns].iloc[rows].reset_index(drop=True)], axis=1)


def require_links_bound(actions: pd.DataFrame, links: pd.DataFrame | None) -> None:
    """Raise ``ValueError`` unless caller ``links`` are built from exactly this ``actions`` frame (spec W9.5).

    Every public ``links=`` acceptor calls it as its first statement, so it raises on mismatched pointers even in a
    call that never joins them, and before it re-sorts or subsets ``actions`` (``remap_links`` cannot tell a dropped
    action from a row of a larger frame). ``links=None`` passes. It is a plain call, not a wrapper decorator: a
    wrapper frame would make every ``warnings.warn(..., stacklevel=2)`` in the acceptor name the wrapper, not the
    caller (the ``nan_safe_enrichment`` marker returns the function itself for the same reason).
    """
    if links is not None:
        resolve_link_positions(actions, links)


def remap_links(
    links: pd.DataFrame | None,
    permutation: np.ndarray,
    new_actions: pd.DataFrame,
    *,
    actions: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Map pointers through ``new_actions = actions.iloc[permutation]`` (a re-sort, filter, or subset).

    Each pointer's ``action_row`` becomes its action's row in ``new_actions``; a pointer whose action was dropped is
    dropped. The remapped pointers are verified against ``new_actions``, so a caller's ``links`` that were not built on
    the original frame raise here. Legacy ``links`` (no ``action_row``) join on ``action_id`` and are returned as is.
    ``links=None`` (the caller supplied no pointers) stays ``None``.

    Pass ``actions``, the frame that ``permutation`` indexes, to verify ``links`` against it first. Without it, a
    pointer of a row beyond that frame looks like the pointer of a dropped action, so ``links`` built on a LARGER
    frame with the same first rows pass. A function that receives caller ``links`` and has not verified them yet
    (``require_links_bound``) passes ``actions``.

    Examples
    --------
    Keep pointers valid for a subset of the actions (here: the shots)::

        import numpy as np
        from silly_kicks.tracking.utils import link_actions_to_frames, remap_links
        links, _report = link_actions_to_frames(actions, frames)
        is_shot = (actions["type_id"] == 11).to_numpy()
        shots = actions[is_shot]
        shot_links = remap_links(links, np.flatnonzero(is_shot), shots)  # bound to `shots`, in its row order
    """
    if links is None:
        return None
    if actions is not None:
        resolve_link_positions(actions, links)
    return _remap(links, permutation, new_actions)


def _remap(links: pd.DataFrame, permutation: np.ndarray, new_actions: pd.DataFrame) -> pd.DataFrame:
    """The body of :func:`remap_links`, looked up at call time (the planted-mutation seam of the binding fixtures)."""
    if "action_row" not in links.columns:
        return links
    perm = np.asarray(permutation, dtype=np.int64)
    rows = links["action_row"].to_numpy(dtype=np.int64)
    size = int(max(perm.max(initial=-1), rows.max(initial=-1))) + 1
    new_row_of = np.full(size, -1, dtype=np.int64)
    new_row_of[perm] = np.arange(len(perm), dtype=np.int64)
    new_rows = np.where(rows >= 0, new_row_of[np.clip(rows, 0, None)], -1) if len(rows) else rows
    keep = new_rows >= 0
    out = links.loc[keep].reset_index(drop=True)
    out["action_row"] = new_rows[keep].astype(np.int64)
    _verify_positions(new_actions, out, out["action_row"].to_numpy(dtype=np.int64))
    return out
