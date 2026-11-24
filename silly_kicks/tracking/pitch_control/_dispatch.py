"""Pitch control dispatch — routes method kwarg to the correct model.

Public API:
- compute_pitch_control(frame, attacking_team_id, ...) -> PitchControlSurface
- compute_pitch_control_at_points(frame, targets, ...) -> np.ndarray

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 7.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pandas as pd

from silly_kicks._frame_index import RowGroups, group_rows

from ._fernandez_bornn import compute_fernandez_bornn
from ._index import _SpearmanFrameIndex
from ._params import (
    FernandezBornnParams,
    Method,
    PitchControlParams,
    SpearmanParams,
    VoronoiParams,
    validate_params_for_method,
)
from ._spearman import compute_spearman
from ._spearman_batch import _spearman_batch_core
from ._surface import PitchControlSurface
from ._voronoi import compute_voronoi

_VELOCITY_REQUIRED_METHODS: set[Method] = {"spearman", "fernandez_bornn"}


def compute_pitch_control(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
    """Compute pitch control surface for a single tracking frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data with columns: player_id, team_id, x, y,
        vx, vy (for Spearman/F&B), is_ball, is_goalkeeper.
    attacking_team_id : int | str
        Team whose control maps to 1.0 in the output surface.
    method : {"spearman", "fernandez_bornn", "voronoi"}
        Pitch control model to use. Default: "spearman".
    params : PitchControlParams | None
        Model-specific parameters. None uses defaults.
    decompose : bool
        If True, compute per-player influence decomposition.
    ball_position : tuple[float, float] | None
        Explicit ball position (x, y) in meters. If None, inferred from
        the ball row in the frame.

    Returns
    -------
    PitchControlSurface

    Examples
    --------
    Compute the pitch-control surface for a single frame and query it::

        from silly_kicks.tracking.pitch_control import compute_pitch_control
        surface = compute_pitch_control(frame, attacking_team_id=1)
        surface.at_point(52.5, 34.0)  # -> 0.55
    """
    # Validate params for method
    validate_params_for_method(method, params)

    # Check velocity columns for methods that require them
    if method in _VELOCITY_REQUIRED_METHODS:
        if "vx" not in frame.columns or "vy" not in frame.columns:
            raise ValueError(
                f"method='{method}' requires velocity columns ('vx', 'vy') "
                f"in the tracking frame. Use derive_velocities() or "
                f"smooth_frames() to add them, or use method='voronoi' "
                f"for position-only pitch control."
            )

    # Resolve ball position
    bp = _resolve_ball_position(frame, ball_position)

    # Dispatch to model
    if method == "spearman":
        sp = params if isinstance(params, SpearmanParams) else SpearmanParams()
        return compute_spearman(
            frame,
            attacking_team_id,
            sp,
            decompose=decompose,
            ball_position=bp,
        )
    elif method == "fernandez_bornn":
        fp = params if isinstance(params, FernandezBornnParams) else FernandezBornnParams()
        return compute_fernandez_bornn(
            frame,
            attacking_team_id,
            fp,
            decompose=decompose,
            ball_position=bp,
        )
    else:  # voronoi
        vp = params if isinstance(params, VoronoiParams) else VoronoiParams()
        return compute_voronoi(
            frame,
            attacking_team_id,
            vp,
            decompose=decompose,
            ball_position=bp,
        )


def compute_pitch_control_at_points(
    frame: pd.DataFrame,
    targets: np.ndarray,
    attacking_team_id: int | str,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    ball_position: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute pitch control at specific (x, y) points.

    Computes the full surface then interpolates at the given points.
    For repeated queries on the same frame, prefer computing the surface
    once and calling surface.at_points().

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data.
    targets : np.ndarray
        (N, 2) array of (x, y) positions to query.
    attacking_team_id : int | str
        Team whose control maps to 1.0.
    method : Method
        Pitch control model to use.
    params : PitchControlParams | None
        Model-specific parameters.
    ball_position : tuple[float, float] | None
        Explicit ball position override.

    Returns
    -------
    np.ndarray of shape (N,) with control values in [0, 1].

    Examples
    --------
    Compute pitch control at a set of target points in one call::

        targets = np.array([[50, 34], [80, 20]])
        compute_pitch_control_at_points(frame, targets, 1)  # -> array([0.52, 0.35])
    """
    if targets.shape[0] == 0:
        return np.empty(0, dtype="float64")

    surface = compute_pitch_control(
        frame,
        attacking_team_id,
        method=method,
        params=params,
        decompose=False,
        ball_position=ball_position,
    )
    return surface.at_points(targets)


#: The frame key of a batched request.
_FRAME_KEYS = ("game_id", "period_id", "frame_id")

#: One batched request: ``((game_id, period_id, frame_id), attacking_team_id, decompose)``.
PitchControlRequest = tuple[tuple, "int | str", bool]


@dataclasses.dataclass(frozen=True)
class PitchControlFrameIndex:
    """A match's frames grouped and indexed ONCE, for several request waves (spec 2026-11-08 tracking-batch §4.3).

    Build it with :func:`build_pitch_control_index` and pass it to :func:`compute_pitch_control_batch_from_index`.
    ``groups`` is the ``(game_id, period_id, frame_id)`` grouping (every method takes its frame slices from it);
    ``_spearman`` is the columnar spearman index of every frame (private).

    Examples
    --------
    Score one match in two request waves and pay the grouping once::

        from silly_kicks.tracking.pitch_control import (
            build_pitch_control_index,
            compute_pitch_control_batch_from_index,
        )
        index = build_pitch_control_index(frames)
        first = compute_pitch_control_batch_from_index(index, [((1, 1, 10), 1, False)])
        second = compute_pitch_control_batch_from_index(index, [((1, 1, 11), 2, True)])
    """

    groups: RowGroups
    _spearman: _SpearmanFrameIndex


def build_pitch_control_index(frames: pd.DataFrame) -> PitchControlFrameIndex:
    """Group and index ``frames`` once, by ``(game_id, period_id, frame_id)``.

    Examples
    --------
    Build the index of a match's frames::

        from silly_kicks.tracking.pitch_control import build_pitch_control_index
        index = build_pitch_control_index(frames)
    """
    groups = group_rows(frames, _FRAME_KEYS)
    return PitchControlFrameIndex(groups=groups, _spearman=_SpearmanFrameIndex.from_groups(groups))


def compute_pitch_control_batch(
    frames: pd.DataFrame,
    requests: Sequence[PitchControlRequest],
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    max_frames_per_chunk: int | None = 2048,
) -> list[PitchControlSurface]:
    """Compute N pitch-control surfaces in one call (ADR-103 F2; spec 2026-11-08 tracking-batch §4.2-4.3).

    ``frames`` is a whole-unit long-form frame set; ``requests`` names the
    ``((game_id, period_id, frame_id), attacking_team_id, decompose)`` surfaces to compute. The frame set is
    grouped and indexed ONCE, and DUPLICATE requests are computed once. **Bit-identical** to calling
    :func:`compute_pitch_control` per request over the same per-frame slice (parity-gated). Returns surfaces
    aligned 1:1 with ``requests``.

    The kernel runs over the distinct frames the requests name, in chunks of at most ``max_frames_per_chunk``
    frames (``None``: one chunk). A chunk holds every request on its frames. The values do not depend on the
    chunk size: each surface depends only on its own frame. Memory: the spearman TTI intermediate is per distinct
    frame, at most ``max_frames_per_chunk`` x 22 players x 1,600 targets (the default 50 x 32 grid) x 8 bytes,
    about 0.58 GB at the default 2048. The measured ``tracemalloc`` peak is 7.1 GB with numba and 9.9 GB with
    numpy (the default chunk size, one request per frame of the whole gradientsports match 10517: 254,242
    requests, of which the returned surface arrays alone are about 3.3 GB). A memory-tight caller lowers the chunk
    size. The chunk size does not bound the
    result: the returned list holds every surface (1,600 targets x 8 bytes, about 12.8 kB each at the default
    grid), so a caller with very many requests sends them in waves.

    Examples
    --------
    Compute a decomposed + aggregate surface for two frames in one call::

        reqs = [((1, 1, 10), 1, True), ((1, 1, 11), 1, False)]
        surfaces = compute_pitch_control_batch(frames, reqs, method="spearman")
    """
    _validate_chunk_size(max_frames_per_chunk)
    if not requests:
        return []
    return compute_pitch_control_batch_from_index(
        build_pitch_control_index(frames),
        requests,
        method=method,
        params=params,
        max_frames_per_chunk=max_frames_per_chunk,
    )


def compute_pitch_control_batch_from_index(
    index: PitchControlFrameIndex,
    requests: Sequence[PitchControlRequest],
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    max_frames_per_chunk: int | None = 2048,
) -> list[PitchControlSurface]:
    """:func:`compute_pitch_control_batch` over a prebuilt :class:`PitchControlFrameIndex` (no re-grouping).

    Examples
    --------
    Several request waves over one index::

        index = build_pitch_control_index(frames)
        surfaces = compute_pitch_control_batch_from_index(index, reqs, max_frames_per_chunk=512)
    """
    _validate_chunk_size(max_frames_per_chunk)
    if not requests:
        return []
    return _batch_from_index(index, requests, method=method, params=params, max_frames_per_chunk=max_frames_per_chunk)


def _validate_chunk_size(max_frames_per_chunk: object) -> None:
    ok = max_frames_per_chunk is None or (
        isinstance(max_frames_per_chunk, int)
        and not isinstance(max_frames_per_chunk, bool)
        and max_frames_per_chunk > 0
    )
    if not ok:
        raise ValueError(f"max_frames_per_chunk must be a positive int or None, got {max_frames_per_chunk!r}")


def _batch_from_groups(
    groups,
    requests: Sequence[PitchControlRequest],
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
) -> list[PitchControlSurface]:
    """The batched kernel over a PRE-BUILT ``group_rows`` grouping (ADR-105 Task 5): one chunk.

    A caller that grouped ``frames`` itself (``PitchControlCache.warm``) does not re-group.
    """
    if not requests:
        return []
    keys = list(dict.fromkeys(tuple(k) for k, _team, _dec in requests))
    index = PitchControlFrameIndex(groups=groups, _spearman=_SpearmanFrameIndex.from_groups(groups, keys))
    return _batch_from_index(index, requests, method=method, params=params, max_frames_per_chunk=None, keys=keys)


def _batch_from_index(
    index: PitchControlFrameIndex,
    requests: Sequence[PitchControlRequest],
    *,
    method: Method,
    params: PitchControlParams | None,
    max_frames_per_chunk: int | None,
    keys: list[tuple] | None = None,
) -> list[PitchControlSurface]:
    """Dedup, chunk over distinct frames, run each chunk. ``keys`` are the frame keys of ``index._spearman`` when it
    was built for exactly those keys (``_batch_from_groups``); otherwise it covers every group of ``index.groups``."""
    # Dedup DISTINCT (frame_key, team, decompose); preserve request order for the return. The team is keyed by its
    # exact type AND value (R2): team 1 and team "1" are two requests, each surface reporting its own team.
    order: list[tuple] = []
    distinct: dict[tuple, tuple] = {}
    for frame_key, attacking_team_id, decompose in requests:
        dedup_key = (tuple(frame_key), type(attacking_team_id), attacking_team_id, bool(decompose))
        order.append(dedup_key)
        if dedup_key not in distinct:
            distinct[dedup_key] = (tuple(frame_key), attacking_team_id, bool(decompose))

    groups = index.groups
    computed: dict[tuple, PitchControlSurface] = {}
    if method == "spearman":
        validate_params_for_method(method, params)
        sp = params if isinstance(params, SpearmanParams) else SpearmanParams()
        columns = groups.frame.columns  # every frame slice carries the grouped frame's columns
        if "vx" not in columns or "vy" not in columns:  # preserve compute_pitch_control's velocity-required raise
            raise ValueError(
                "method='spearman' requires velocity columns ('vx', 'vy') in the tracking frame. "
                "Use derive_velocities() or smooth_frames() to add them, or use method='voronoi' "
                "for position-only pitch control."
            )
        position = _frame_positions(index, keys)
        distinct_keys = list(distinct)
        request_frames = [distinct[k][0] for k in distinct_keys]
        chunks = list(_chunks(list(dict.fromkeys(request_frames)), max_frames_per_chunk))
        for chunk, rows in zip(chunks, _requests_per_chunk(request_frames, chunks), strict=True):
            chunk_requests = [distinct_keys[i] for i in rows]
            local = {fk: i for i, fk in enumerate(chunk)}
            sub = index._spearman.select([position(fk) for fk in chunk])
            core_requests = [
                (local[distinct[k][0]], distinct[k][1], distinct[k][2], sub.ball(local[distinct[k][0]]))
                for k in chunk_requests
            ]
            computed.update(zip(chunk_requests, _spearman_batch_core(sub, core_requests, sp), strict=True))
    else:
        for dedup_key, (frame_key, attacking_team_id, decompose) in distinct.items():
            frame = groups.get(*frame_key)
            computed[dedup_key] = compute_pitch_control(
                frame, attacking_team_id, method=method, params=params, decompose=decompose
            )

    return [computed[k] for k in order]


def _frame_positions(index: PitchControlFrameIndex, keys: list[tuple] | None):
    """A function: frame key -> its frame number in ``index._spearman``, or ``-1`` for a key that names no frame
    (``select`` makes it an empty frame: the per-request path scores an empty slice as the uniform 0.5 surface)."""
    if keys is not None:
        where = {tuple(k): i for i, k in enumerate(keys)}
        return lambda fk: where[tuple(fk)]
    where_canon = {k: i for i, k in enumerate(index.groups.keys())}
    return lambda fk: where_canon.get(index.groups.canonical_key(*fk), -1)


def _requests_per_chunk(request_frames: list, chunks: list[list]) -> list[list[int]]:
    """For each chunk, the positions in ``request_frames`` whose frame key is in that chunk, in request order.

    One pass over the chunks and one over the requests: never one scan of every request per chunk, which grew with
    requests x chunks (reviewer C TBC-IMPL-10). Each frame key is in exactly one chunk."""
    chunk_of = {fk: c for c, chunk in enumerate(chunks) for fk in chunk}
    out: list[list[int]] = [[] for _ in chunks]
    for i, fk in enumerate(request_frames):
        out[chunk_of[fk]].append(i)
    return out


def _chunks(items: list, size: int | None):
    if size is None or size >= len(items):
        yield items
        return
    for lo in range(0, len(items), size):
        yield items[lo : lo + size]


def _resolve_ball_position(
    frame: pd.DataFrame,
    explicit: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Resolve ball position: explicit > frame ball row > None."""
    if explicit is not None:
        # Validate on-pitch (loose bounds)
        x, y = explicit
        if x < -5 or x > 110 or y < -5 or y > 73:
            return None  # off-pitch, treat as no conditioning
        return explicit

    # Try to extract from ball row
    ball_rows = frame[frame["is_ball"].astype(bool)]
    if ball_rows.empty:
        return None
    bx = float(ball_rows["x"].iloc[0])
    by = float(ball_rows["y"].iloc[0])
    if np.isnan(bx) or np.isnan(by):
        return None
    return (bx, by)
