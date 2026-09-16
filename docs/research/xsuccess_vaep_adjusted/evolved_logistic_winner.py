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

    dx = _GOAL[0] - sx
    dy = _GOAL[1] - sy
    ady = np.abs(dy)  # symmetry about pitch centre (y=34)
    dist = np.hypot(dx, dy)  # distance from START to goal centre
    ang = np.arctan2(ady, dx)  # y-symmetric bearing to goal centre

    # goal-mouth angle subtended from start (near/far post geometry)
    gh = 3.66  # half goal width (~7.32m mouth)
    a1 = np.arctan2(dy + gh, dx)
    a2 = np.arctan2(dy - gh, dx)
    goal_ang = np.abs(a1 - a2)  # visible goal angle

    # nonlinear location transforms
    ldist = np.log1p(dist)
    xn = sx / _GOAL[0]           # normalized progress toward goal
    yn = ady / _GOAL[1]          # normalized lateral offset (symmetric)

    # exponential distance decay -- smoother probability shape near goal
    ddec = np.exp(-dist / 20.0)
    # central proximity: close AND in front of goal (small lateral offset).
    # This is the strongest shot-completion shape and is y-symmetric.
    central = ddec * np.exp(-ady / 12.0)
    ay2 = ady * ady  # squared lateral offset -- convex penalty for wide angles
    inv_dist = 1.0 / (dist + 1.0)  # bounded inverse-distance term

    # smooth bearing encoding: avoids +-pi/2 discontinuity for a GLM calibrator
    sin_ang = np.sin(ang)
    cos_ang = np.cos(ang)
    is_second_half = (period >= 2).astype(float)

    type_oh = np.column_stack([(tid == t).astype(float) for t in range(len(_TYPES))])
    bid_oh = np.column_stack([(bid == b).astype(float) for b in range(len(_BODY))])

    # systematic per-type geometry interactions: every action type gets its own
    # smooth distance/angle/lateral-offset/proximity-completion slope, instead
    # of a small hand-picked subset -- avoids redundant, overfit-prone terms.
    dist_x_type = dist[:, None] * type_oh
    ang_x_type = goal_ang[:, None] * type_oh
    ay_x_type = ady[:, None] * type_oh
    near_x_type = ddec[:, None] * type_oh
    dist_x_bid = dist[:, None] * bid_oh
    ang_x_bid = goal_ang[:, None] * bid_oh

    cols = [secs, secs / 2700.0, sx, sy, ady, ay2, dist, dist * dist, ldist,
            sin_ang, cos_ang, goal_ang, xn, yn, inv_dist, ddec, central,
            is_second_half, period]
    cols.extend(type_oh[:, t] for t in range(len(_TYPES)))
    cols.extend(bid_oh[:, b] for b in range(len(_BODY)))
    cols.extend(dist_x_type[:, t] for t in range(len(_TYPES)))
    cols.extend(ang_x_type[:, t] for t in range(len(_TYPES)))
    cols.extend(ay_x_type[:, t] for t in range(len(_TYPES)))
    cols.extend(near_x_type[:, t] for t in range(len(_TYPES)))
    cols.extend(dist_x_bid[:, b] for b in range(len(_BODY)))
    cols.extend(ang_x_bid[:, b] for b in range(len(_BODY)))
    X = np.column_stack(cols)  # freshly allocated -> writable under pandas-3 copy-on-write

    bad = ~np.isfinite(np.column_stack([sx, sy, secs, period])).all(axis=1)
    X[bad] = np.nan  # NaN start/time/period -> NaN row (never fabricated)
    return X
# EVOLVE-BLOCK-END
