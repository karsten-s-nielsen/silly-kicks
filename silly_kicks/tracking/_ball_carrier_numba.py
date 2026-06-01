"""Optional numba-accelerated kernel for ball-carrier inference.

Mirrors the Python _carrier_loop_numpy in _ball_carrier.py but uses
@numba.njit for ~30-50x speedup on large tracking datasets.

Import pattern:
    try:
        from ._ball_carrier_numba import _carrier_loop_numba
        _HAS_NUMBA = True
    except ImportError:
        _HAS_NUMBA = False
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError("numba is required for _ball_carrier_numba. Install with: pip install silly-kicks[numba]") from e

# numba's on-disk cache (cache=True) requires a writable cache *locator* to be
# resolved AT IMPORT TIME (a writable __pycache__ beside the source, a writable
# user-wide cache dir, or NUMBA_CACHE_DIR set). On read-only / ephemeral installs
# (e.g. Databricks serverless: wheel on a read-only ephemeral NFS path) all
# locators fail and numba raises RuntimeError from inside a successful import —
# taking down all of silly_kicks.tracking, not just the cached function (the
# consumer try/except ImportError guards do NOT catch a RuntimeError). Default the
# on-disk cache OFF so JIT works everywhere (cache=False keeps full native speed;
# it only drops cross-process persistence → a one-time per-process recompile).
# Opt back in via SILLY_KICKS_NUMBA_CACHE=1 (explicit, stable env / local dev) or
# numba's own NUMBA_CACHE_DIR pointing at a writable dir (guaranteed-writable).
_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE", "0") == "1" or bool(os.environ.get("NUMBA_CACHE_DIR"))


@njit(cache=_NUMBA_CACHE)
def _carrier_loop_numba(
    bx: np.ndarray,
    by: np.ndarray,
    ball_dead: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pvx: np.ndarray,
    pvy: np.ndarray,
    player_slots: np.ndarray,
    n_valid: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    tolerance_m: float,
    beta: float,
    gamma: float,
    has_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated carrier loop — identical logic to _carrier_loop_numpy.

    Parameters
    ----------
    bx, by : (n_frames,) float64 — ball position per frame
    ball_dead : (n_frames,) bool — True if dead ball or NaN ball position
    px, py : (n_frames, max_players) float64 — player positions, NaN-padded
    pvx, pvy : (n_frames, max_players) float64 — player velocities, NaN-padded
    player_slots : (n_frames, max_players) int64 — player slot indices, -1 empty
    n_valid : (n_frames,) int64 — valid player count per frame
    seg_starts, seg_ends : (n_segments,) int64 — half-open segment ranges
    tolerance_m, beta, gamma : float64 — algorithm parameters
    has_velocity : bool — whether to use velocity scoring

    Returns
    -------
    winner_slot : (n_frames,) int64 — winning player slot (-1 = no carrier)
    winner_dist : (n_frames,) float64 — distance to ball (NaN = no carrier)
    """
    n_frames = len(bx)
    winner_slot = np.full(n_frames, -1, dtype=np.int64)
    winner_dist = np.full(n_frames, np.nan)
    n_segments = len(seg_starts)

    for s in range(n_segments):
        incumbent = -1
        for f in range(seg_starts[s], seg_ends[s]):
            if ball_dead[f]:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            nv = n_valid[f]
            if nv == 0:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            # Single-pass: find candidates within tolerance and select best
            best_score = 1e30
            best_dist = np.nan
            best_slot = -1
            n_within = 0

            for i in range(nv):
                dx = px[f, i] - bx[f]
                dy = py[f, i] - by[f]
                d = np.sqrt(dx * dx + dy * dy)
                if d > tolerance_m:
                    continue

                score = d
                if has_velocity:
                    if d > 0:
                        ux = -dx / d
                        uy = -dy / d
                    else:
                        ux = 0.0
                        uy = 0.0
                    vx_val = pvx[f, i]
                    vy_val = pvy[f, i]
                    if np.isnan(vx_val) or np.isnan(vy_val):
                        v_toward = 0.0
                    else:
                        v_toward = vx_val * ux + vy_val * uy
                        if v_toward < 0.0:
                            v_toward = 0.0
                    score = d - beta * v_toward

                # Hysteresis
                if incumbent >= 0 and gamma > 0.0 and player_slots[f, i] == incumbent:
                    score -= gamma

                # Select best: lowest score, tiebreak by lowest slot
                slot_i = player_slots[f, i]
                if n_within == 0:
                    best_score = score
                    best_dist = d
                    best_slot = slot_i
                elif score < best_score - 1e-12:
                    best_score = score
                    best_dist = d
                    best_slot = slot_i
                elif abs(score - best_score) < 1e-12 and slot_i < best_slot:
                    best_score = score
                    best_dist = d
                    best_slot = slot_i

                n_within += 1

            if n_within == 0:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
            else:
                winner_slot[f] = best_slot
                winner_dist[f] = best_dist
                incumbent = best_slot

    return winner_slot, winner_dist
