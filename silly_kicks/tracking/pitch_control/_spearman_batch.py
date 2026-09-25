"""Vectorized cross-frame spearman pitch-control kernel (ADR-105 Task 1).

`compute_spearman_batch` computes N spearman surfaces in one call, BYTE-IDENTICAL to calling
`compute_spearman` per request (parity-gated, `tests/tracking/pitch_control/test_spearman_batch.py`).

The win: TTI is computed per-(player, target) INDEPENDENTLY of any other player, so the valid players of
ALL requested frames are concatenated and passed through the existing `compute_tti` in ONE call (same
numba/numpy path the per-frame code uses); slicing the result back per frame is bit-identical. The
"after-TTI" combine is the single-sourced `_spearman._spearman_combine`, so no math is duplicated.

MEMORY: this kernel materializes one `compute_tti` intermediate over the WHOLE batch it is handed. Callers
that must bound peak memory (rest_defense, off_ball) hand it BOUNDED chunks (Tasks 2/4), never a whole unit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

from ._grids import pitch_grid
from ._params import SpearmanParams
from ._spearman import (
    _empty_spearman_surface,
    _extract_frame_players,
    _spearman_combine,
    compute_tti,
)
from ._surface import PitchControlSurface


def compute_spearman_batch(
    frame_slices: Sequence[pd.DataFrame],
    teams: Sequence[int | str],
    decomposes: Sequence[bool],
    ball_positions: Sequence[tuple[float, float] | None],
    *,
    params: SpearmanParams,
) -> list[PitchControlSurface]:
    """N spearman surfaces, one per request, byte-identical to `compute_spearman` per request.

    ``frame_slices[i]`` / ``teams[i]`` / ``decomposes[i]`` / ``ball_positions[i]`` are the i-th request
    (a single-frame slice, its attacking team, whether to decompose, and its resolved ball position).

    Examples
    --------
    Compute two frames' surfaces in one call::

        surfaces = compute_spearman_batch([frame_a, frame_b], [1, 1], [False, True],
                                          [(50.0, 34.0), None], params=SpearmanParams())
    """
    n = len(frame_slices)
    if not (n == len(teams) == len(decomposes) == len(ball_positions)):
        raise ValueError("compute_spearman_batch: frame_slices/teams/decomposes/ball_positions length mismatch")

    grid_x, grid_y, targets = pitch_grid(params.grid_cells_x, params.grid_cells_y)

    extracted = [_extract_frame_players(fr) for fr in frame_slices]
    valid = [(i, e) for i, e in enumerate(extracted) if e is not None]

    tti_by_request: dict[int, np.ndarray] = {}
    if valid:
        # ONE compute_tti over ALL valid players concatenated -- per-(player, target) so slicing back is
        # bit-identical to a per-frame call, and it uses the same numba/numpy path.
        pos_all = np.concatenate([e[0] for _i, e in valid], axis=0)
        vel_all = np.concatenate([e[1] for _i, e in valid], axis=0)
        tti_flat = compute_tti(pos_all, vel_all, targets, params.reaction_time, params.max_acceleration)
        lengths = [e[0].shape[0] for _i, e in valid]
        bounds = np.cumsum([0, *lengths])
        for k, (i, _e) in enumerate(valid):
            tti_by_request[i] = tti_flat[bounds[k] : bounds[k + 1]]

    results: list[PitchControlSurface] = []
    for i in range(n):
        e = extracted[i]
        if e is None:
            results.append(_empty_spearman_surface(grid_x, grid_y, params, teams[i]))
            continue
        _pos, _vel, is_gk, player_ids_arr, team_id_series = e
        is_attacking = ids_match(team_id_series, teams[i]).to_numpy()
        results.append(
            _spearman_combine(
                tti_by_request[i],
                is_attacking,
                is_gk,
                player_ids_arr,
                team_id_series.to_numpy(),
                ball_positions[i],
                params,
                grid_x,
                grid_y,
                targets,
                teams[i],
                decomposes[i],
            )
        )
    return results
