"""Event-only, END-BLIND xSuccess features (TF-61).

Start-anchored geometry only — the realized action end is NEVER read (for a failed action the SPADL
end is the outcome, a target leak; spec §5.2). SPADL actions are canonical action-LTR, so geometry is
computed directly (goal centre at ``(field_length, field_width/2)``) with no orientation/goal_map
handling. A non-finite start coordinate / time / period yields an all-NaN feature row (never a
fabricated value), so a downstream scorer NaN-propagates.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np

import silly_kicks.spadl.config as cfg

_GOAL = (float(cfg.field_length), float(cfg.field_width) / 2.0)
_TYPES = list(cfg.actiontypes)
_BODY = list(cfg.bodyparts)

FEATURE_NAMES: list[str] = (
    ["seconds", "start_x", "start_y", "distance_to_goal", "angle_to_goal", "period_id"]
    + [f"type_{t}" for t in _TYPES]
    + [f"bodypart_{b}" for b in _BODY]
)


def xsuccess_features(actions) -> np.ndarray:
    """SPADL actions -> the ``FEATURE_NAMES`` feature matrix (END-BLIND, event-only).

    Parameters
    ----------
    actions : pandas.DataFrame
        SPADL actions with ``type_id`` / ``bodypart_id`` / ``start_x`` / ``start_y`` /
        ``time_seconds`` / ``period_id``. ``end_x`` / ``end_y`` / ``result_id`` are deliberately
        NOT read (end-blindness is guarded by ``tests/xsuccess/test_leakage_guard.py``).

    Returns
    -------
    numpy.ndarray
        ``(n, len(FEATURE_NAMES))`` float64 matrix, column order pinned by ``FEATURE_NAMES``.
    """
    sx = np.asarray(actions["start_x"], dtype=float)
    sy = np.asarray(actions["start_y"], dtype=float)
    secs = np.asarray(actions["time_seconds"], dtype=float)
    period = np.asarray(actions["period_id"], dtype=float)
    tid = np.asarray(actions["type_id"])
    bid = np.asarray(actions["bodypart_id"])

    dx = _GOAL[0] - sx
    dy = _GOAL[1] - sy
    dist = np.hypot(dx, dy)  # distance from START to goal centre
    ang = np.arctan2(dy, dx)  # bearing from START to goal centre

    cols = [secs, sx, sy, dist, ang, period]
    cols.extend((tid == t).astype(float) for t in range(len(_TYPES)))
    cols.extend((bid == b).astype(float) for b in range(len(_BODY)))
    X = np.column_stack(cols)  # freshly allocated -> writable under pandas-3 copy-on-write

    bad = ~np.isfinite(np.column_stack([sx, sy, secs, period])).all(axis=1)
    X[bad] = np.nan  # NaN start/time/period -> NaN row (never fabricated)
    return X


def _probe_actions():
    import pandas as pd

    return pd.DataFrame(
        dict(
            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]],
            bodypart_id=[cfg.bodypart_id["foot"], cfg.bodypart_id["head"]],
            start_x=[20.0, 88.0],
            start_y=[34.0, 30.0],
            time_seconds=[12.0, 2600.0],
            period_id=[1, 2],
        )
    )


def feature_contract_block() -> dict:
    """Ordered feature names, declared geometry constants, and a fixed-probe feature vector.

    Recorded in the artifact so ``XSuccessModel.load`` can fail-closed on a feature-name or
    geometry-constant drift (ADR-050 discipline).
    """
    return {
        "feature_names": list(FEATURE_NAMES),
        "geometry": {"field_length": float(cfg.field_length), "field_width": float(cfg.field_width)},
        "probe": {"feature_vector": xsuccess_features(_probe_actions()).tolist()},
    }
