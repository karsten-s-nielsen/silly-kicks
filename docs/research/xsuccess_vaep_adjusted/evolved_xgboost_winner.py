"""EVOLVE TARGET: xsuccess_features -- END-BLIND, event-only feature builder for xSuccess (TF-61).

The evaluator scores each candidate on held-out CALIBRATED log-loss (StratifiedGroupKFold by match)
and REJECTS any candidate that reads the realized action end (end_x / end_y) -- that is target
leakage, because for a FAILED action the SPADL end IS the outcome. Improve the feature
representation WITHIN the END-FREE input allowlist; never read end_x, end_y, or result_id.

Allowed END-FREE input columns:
    game_id, period_id, type_id, bodypart_id, start_x, start_y, time_seconds

SPADL actions are canonical action-LTR: the attacked goal centre is at (field_length, field_width/2)
= (105.0, 34.0). A non-finite start coordinate / time / period must yield an all-NaN feature row
(never a fabricated value) so a downstream scorer NaN-propagates.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/silly-kicks-tf61"))
import silly_kicks.spadl.config as cfg  # noqa: E402

_GOAL = (float(cfg.field_length), float(cfg.field_width) / 2.0)
_TYPES = list(cfg.actiontypes)
_BODY = list(cfg.bodyparts)


# EVOLVE-BLOCK-START
def xsuccess_features(actions) -> np.ndarray:
    """SPADL actions -> feature matrix (END-BLIND, event-only)."""
    sx = np.asarray(actions["start_x"], dtype=float)
    sy = np.asarray(actions["start_y"], dtype=float)
    secs = np.asarray(actions["time_seconds"], dtype=float)
    period = np.asarray(actions["period_id"], dtype=float)
    tid = np.asarray(actions["type_id"])
    bid = np.asarray(actions["bodypart_id"])

    L, W = _GOAL[0], _GOAL[1] * 2.0
    dx = _GOAL[0] - sx
    dy = _GOAL[1] - sy
    ady = np.abs(dy)                      # y-symmetric lateral offset from centre
    dist = np.hypot(dx, dy)              # distance from START to goal centre
    ang = np.arctan2(dy, dx)             # bearing from START to goal centre

    # Nonlinear location transforms
    logdist = np.log1p(dist)
    invdist = 1.0 / (1.0 + dist)
    dx2 = dx * dx
    ady2 = ady * ady

    # Distance to nearest pitch boundary (crowding near touchline / goal line)
    edge_x = np.minimum(sx, L - sx)
    edge_y = np.minimum(sy, W - sy)

    # Shot/pass angle subtended by the goal mouth (near/far post, symmetric)
    gh = 7.32 / 2.0
    a_near = np.arctan2(_GOAL[1] - gh - sy, dx)
    a_far = np.arctan2(_GOAL[1] + gh - sy, dx)
    goal_angle = np.abs(a_far - a_near)  # visible goal width from start

    # Normalized game time (calibration of late-game / fatigue effects)
    tnorm = secs / 2700.0
    late = np.clip(secs - 2400.0, 0.0, None) / 300.0  # last ~5 min emphasis

    # Forward progress geometry: how far up the pitch the action starts.
    # Distance to own goal (defensive risk) is asymmetric vs attacking goal.
    own_dist = np.hypot(sx, dy)
    # Polar directionality to goal centre (smooth, y-symmetric via |sin|).
    cos_ang = dx / (dist + 1e-6)
    sin_ang = ady / (dist + 1e-6)

    # Forward-progress fraction up the pitch (0 at own goal, 1 at attacking).
    xfrac = sx / L
    # "Shot-quality" style proxy: being both close AND central is what makes
    # goalward actions succeed; the product interacts distance with lateral
    # offset in a way a single distance term cannot capture.
    central_close = invdist * (1.0 / (1.0 + ady))

    # Type x geometry interactions. Give EVERY action type its own distance
    # slope (sparse one-hot * logdist), since a dribble, cross, long-pass,
    # take-on and clearance each have a very different completion-vs-distance
    # profile. The salient goalward types (pass, cross, shot) additionally get
    # a directionality slope.
    inter = []
    for t in range(len(_TYPES)):
        m = (tid == t).astype(float)
        inter.append(m * logdist)         # per-type distance slope
    for t in (0, 1, 11):
        if t < len(_TYPES):
            m = (tid == t).astype(float)
            inter.append(m * cos_ang)     # directionality interaction
            inter.append(m * central_close)  # close+central quality slope

    cols = [secs, sx, sy, dist, ang, period,
            ady, logdist, invdist, dx2, ady2,
            edge_x, edge_y, goal_angle, tnorm,
            late, cos_ang, sin_ang, own_dist,
            xfrac, central_close]
    cols.extend(inter)
    # Bodypart x distance: headed actions decay with distance differently
    # from footed ones -- a cheap, orthogonal interaction.
    cols.extend((bid == b).astype(float) * logdist for b in range(len(_BODY)))
    cols.extend((tid == t).astype(float) for t in range(len(_TYPES)))
    cols.extend((bid == b).astype(float) for b in range(len(_BODY)))
    X = np.column_stack(cols)  # freshly allocated -> writable under pandas-3 copy-on-write

    bad = ~np.isfinite(np.column_stack([sx, sy, secs, period])).all(axis=1)
    X[bad] = np.nan  # NaN start/time/period -> NaN row (never fabricated)
    return X
# EVOLVE-BLOCK-END
