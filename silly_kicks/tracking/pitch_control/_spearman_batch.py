"""The batched spearman pitch-control core (ADR-105 Task 1; spec 2026-11-08 tracking-batch §4.2 W2).

`compute_spearman_batch` computes N spearman surfaces in one call, BIT-IDENTICAL to the per-request loop
(parity-gated against the per-frame public ``compute_pitch_control`` in the same test run, on both backends).

- The frames go into one columnar :class:`~._index._SpearmanFrameIndex` (one vectorised filter pass), so extraction,
  ball resolution, and TTI run once per distinct frame slice. TTI is per (player, target) and independent of every
  other player, so one :func:`compute_tti` call over all index rows is exact.
- ``is_attacking`` comes from ONE ``ids_match`` per (index part, team) over that team's requested rows, when the
  part's team ids take one path for every frame (``id_compat._ids_match_path_is_content_free``: a non-object
  column, or object holding only strings). ``ids_match`` picks its raw or canonical path from the content it sees,
  so an object column with non-string ids keeps one call per request on that request's frame view (spec W2.4 as
  changed by owner ruling 2026-11-18, TBC-PERF-03). The decomposed ``player_team_ids`` follow the same rule.
- The combine runs over the concatenated rows of all requests: the opponent minimum TTI per (request, side) (``min``
  is exact and order-free), one elementwise :func:`_influence_rows` call, the ball-travel ``too_slow`` mask, and the
  GK weighting.

**The sums stay sequential.** Each request's rows are laid out contiguously as [attackers in frame-row order, then
defenders in frame-row order] in one C-contiguous array, and each side is summed with a C-order axis-0 ``.sum`` over
its contiguous row slice -- sequential at any row count, exactly as the per-frame sum over a standalone C-contiguous
array. ``np.add.reduceat`` and transposed sums are NOT sequential and must not be used here. An empty side sums to
``zeros(n_targets)``, as the per-frame ``np.zeros((0, n_targets)).sum(axis=0)``.

MEMORY: this kernel materializes one ``compute_tti`` intermediate over the whole batch it is handed. Callers that
must bound peak memory (rest_defense, off_ball) hand it BOUNDED chunks (Tasks 2/4), never a whole unit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from silly_kicks.id_compat import _ids_match_path_is_content_free, ids_match

from ._grids import pitch_grid
from ._index import _SpearmanFrameIndex
from ._params import SpearmanParams
from ._spearman import _empty_spearman_surface, compute_tti
from ._surface import PitchControlSurface

#: One request of the core: ``(frame, attacking_team_id, decompose, ball_position)``, ``frame`` an index entry.
SpearmanRequest = tuple[int, "int | str", bool, "tuple[float, float] | None"]


def _batch_kernels():
    """The batch-core numba kernels when the PITCH_CONTROL family selects numba, else None (HFU spec 13.2: numba is
    core; only ``SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY=1`` selects the numpy reference, and a missing numba raises)."""
    from silly_kicks import _numba_dispatch

    if not _numba_dispatch.use_numba("PITCH_CONTROL"):
        return None
    return _numba_dispatch.import_kernels("silly_kicks.tracking.pitch_control._batch_numba_kernels")


def _influence_rows(tti_rows: np.ndarray, opp_min_rows: np.ndarray, sigma: float) -> np.ndarray:
    """The logistic influence of each row, elementwise: ``1 / (1 + exp(-k * (opp_min_rows - tti_rows)))`` with
    ``k = pi / (sqrt(3) * sigma)``. Each row carries its own request's opponent minimum TTI.

    The numba kernel runs unless ``SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY=1`` selects this numpy reference; both are
    parity-gated against the per-frame ``_compute_influence``.
    """
    kernels = _batch_kernels()
    if kernels is not None:
        return kernels.influence_rows_numba(tti_rows, opp_min_rows, sigma)
    k = np.pi / (np.sqrt(3.0) * sigma)
    return 1.0 / (1.0 + np.exp(-k * (opp_min_rows - tti_rows)))


def _part_rows(index: _SpearmanFrameIndex, frames: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """The source-part row positions of ``frames`` (all in one part), concatenated, and each frame's bounds in them."""
    sel = np.asarray(frames, dtype=np.int64)
    lengths = index.offsets[sel + 1] - index.offsets[sel]
    bounds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    positions = np.repeat(index.src_lo[sel] - bounds[:-1], lengths) + np.arange(bounds[-1], dtype=np.int64)
    return positions, bounds


def _attacking_masks(
    index: _SpearmanFrameIndex, requests: Sequence[SpearmanRequest], live: Sequence[int]
) -> dict[int, np.ndarray]:
    """``is_attacking`` of each live request: which rows of its frame belong to the request's team.

    One ``ids_match`` per (index part, team) reads that team's requested rows when no slice can change the path
    (owner ruling 2026-11-18, TBC-PERF-03). Otherwise each request keeps its own call on its frame's view (W2.4).
    """
    groups: dict[tuple, list[int]] = {}
    for r in live:
        frame, team = requests[r][0], requests[r][1]
        groups.setdefault((int(index.part[frame]), type(team), team), []).append(r)
    masks: dict[int, np.ndarray] = {}
    for (part, _kind, team), members in groups.items():
        positions, bounds = _part_rows(index, [requests[r][0] for r in members])
        # a fresh index: one frame can be requested twice for one team (decompose on and off), and the canonical
        # path of ``ids_match`` assigns by label
        team_ids = index.team_parts[part].iloc[positions].reset_index(drop=True)
        if not _ids_match_path_is_content_free(team_ids):
            for r in members:
                masks[r] = ids_match(index.team_ids(requests[r][0]), team).to_numpy()
            continue
        flat = ids_match(team_ids, team).to_numpy()
        for k, r in enumerate(members):
            masks[r] = flat[bounds[k] : bounds[k + 1]]
    return masks


def _player_team_ids(
    index: _SpearmanFrameIndex, requests: Sequence[SpearmanRequest], live: Sequence[int]
) -> dict[int, np.ndarray]:
    """``index.team_ids(frame).to_numpy()`` of each decomposed live request.

    One ``to_numpy`` per index part, when no slice can change its dtype: a numpy dtype, or no missing value among
    the rows (a nullable or categorical column with a missing value converts to another dtype). Otherwise each
    request keeps its own call. Every request gets its own array, as before.
    """
    by_part: dict[int, list[int]] = {}
    for r in live:
        if requests[r][2]:
            by_part.setdefault(int(index.part[requests[r][0]]), []).append(r)
    arrays: dict[int, np.ndarray] = {}
    for part, members in by_part.items():
        positions, bounds = _part_rows(index, [requests[r][0] for r in members])
        team_ids = index.team_parts[part].iloc[positions]
        if not (isinstance(team_ids.dtype, np.dtype) or not team_ids.isna().any()):
            for r in members:
                arrays[r] = index.team_ids(requests[r][0]).to_numpy()
            continue
        flat = team_ids.to_numpy()
        for k, r in enumerate(members):
            arrays[r] = flat[bounds[k] : bounds[k + 1]].copy()
    return arrays


def _spearman_batch_core(
    index: _SpearmanFrameIndex, requests: Sequence[SpearmanRequest], params: SpearmanParams
) -> list[PitchControlSurface]:
    """One surface per request over the frames of ``index`` (see the module docstring for the order rules)."""
    grid_x, grid_y, targets = pitch_grid(params.grid_cells_x, params.grid_cells_y)
    n_targets = targets.shape[0]
    results: list[PitchControlSurface | None] = [None] * len(requests)

    live = [r for r, (frame, *_rest) in enumerate(requests) if index.n_rows(frame) > 0]
    for r, (frame, team, _decompose, _ball) in enumerate(requests):
        if index.n_rows(frame) == 0:
            results[r] = _empty_spearman_surface(grid_x, grid_y, params, team)
    if not live:
        return results  # type: ignore[return-value]

    tti_all = compute_tti(index.pos, index.vel, targets, params.reaction_time, params.max_acceleration)

    # Per request: the attacker rows and the defender rows (frame-row order), and each side's minimum TTI.
    order: list[np.ndarray] = []
    att_counts: list[int] = []
    def_counts: list[int] = []
    opp_min: list[np.ndarray] = []
    travel: list[np.ndarray] = []
    masks = _attacking_masks(index, requests, live)
    team_arrays = _player_team_ids(index, requests, live)
    for r in live:
        frame, _team, _decompose, ball = requests[r]
        sl = index.rows(frame)
        is_attacking = masks[r]
        att_rows = sl.start + np.flatnonzero(is_attacking)
        def_rows = sl.start + np.flatnonzero(~is_attacking)
        att_min = tti_all[att_rows].min(axis=0) if len(att_rows) else np.full(n_targets, np.inf)
        def_min = tti_all[def_rows].min(axis=0) if len(def_rows) else np.full(n_targets, np.inf)
        order += [att_rows, def_rows]
        att_counts.append(len(att_rows))
        def_counts.append(len(def_rows))
        opp_min += [def_min, att_min]  # an attacker meets the defenders' minimum, a defender the attackers'
        if ball is not None:
            ball_pos = np.array(ball, dtype="float64")
            ball_dist = np.sqrt(((targets - ball_pos[np.newaxis, :]) ** 2).sum(axis=1))
            travel.append(ball_dist / params.average_ball_speed)
        else:
            travel.append(np.full(n_targets, np.inf))  # no ball: no row is ever too slow

    rows = np.concatenate(order)
    side_counts = np.array([c for pair in zip(att_counts, def_counts, strict=True) for c in pair], dtype=np.int64)
    tti_rows = tti_all[rows]
    opp_min_rows = np.stack(opp_min)[np.repeat(np.arange(len(opp_min)), side_counts)]
    infl = _influence_rows(tti_rows, opp_min_rows, params.sigma)
    request_counts = side_counts.reshape(-1, 2).sum(axis=1)
    too_slow = tti_rows > np.stack(travel)[np.repeat(np.arange(len(live)), request_counts)]
    infl[too_slow] = 0.0
    gk = index.is_gk[rows]
    if gk.any():
        infl[gk] *= params.lambda_gk

    lo = 0
    for k, r in enumerate(live):
        frame, team, decompose, _ball = requests[r]
        n_att, n_def = att_counts[k], def_counts[k]
        att_block = infl[lo : lo + n_att]
        def_block = infl[lo + n_att : lo + n_att + n_def]
        lo += n_att + n_def
        att_sum = att_block.sum(axis=0)
        def_sum = def_block.sum(axis=0)
        total = att_sum + def_sum
        safe_total = np.maximum(total, 1e-10)
        surface = np.where(total > 1e-10, att_sum / safe_total, 0.5).reshape(params.grid_cells_y, params.grid_cells_x)
        per_player = p_ids = p_team_ids = None
        if decompose:
            is_attacking = masks[r]
            per_player_flat = np.zeros((len(is_attacking), n_targets))
            per_player_flat[np.flatnonzero(is_attacking)] = att_block
            per_player_flat[np.flatnonzero(~is_attacking)] = def_block
            per_player = per_player_flat.reshape(len(is_attacking), params.grid_cells_y, params.grid_cells_x)
            p_ids = index.player_ids(frame)
            p_team_ids = team_arrays[r]
        results[r] = PitchControlSurface(
            grid_x=grid_x,
            grid_y=grid_y,
            surface=surface,
            method="spearman",
            attacking_team_id=team,
            per_player_influence=per_player,
            player_ids=p_ids,
            player_team_ids=p_team_ids,
        )
    return results  # type: ignore[return-value]


def compute_spearman_batch(
    frame_slices: Sequence[pd.DataFrame],
    teams: Sequence[int | str],
    decomposes: Sequence[bool],
    ball_positions: Sequence[tuple[float, float] | None],
    *,
    params: SpearmanParams,
) -> list[PitchControlSurface]:
    """N spearman surfaces, one per request, bit-identical to ``compute_spearman`` per request.

    ``frame_slices[i]`` / ``teams[i]`` / ``decomposes[i]`` / ``ball_positions[i]`` are the i-th request (a single-frame
    slice, its attacking team, whether to decompose, and its resolved ball position). Every slice is its own frame:
    two slices with one frame key (a frame and its counterfactual) are never deduplicated.

    Examples
    --------
    Compute two frames' surfaces in one call::

        surfaces = compute_spearman_batch([frame_a, frame_b], [1, 1], [False, True],
                                          [(50.0, 34.0), None], params=SpearmanParams())
    """
    n = len(frame_slices)
    if not (n == len(teams) == len(decomposes) == len(ball_positions)):
        raise ValueError("compute_spearman_batch: frame_slices/teams/decomposes/ball_positions length mismatch")
    index = _SpearmanFrameIndex.from_slices(frame_slices)
    requests = [(i, teams[i], bool(decomposes[i]), ball_positions[i]) for i in range(n)]
    return _spearman_batch_core(index, requests, params)
