"""Event-only pass-completion features (TF-54b). Origin/target geometry only -- no tracking.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np

import silly_kicks.spadl.config as spadlconfig

FEATURE_NAMES = [
    "distance",
    "angle",
    "forward",
    "lateral",
    "origin_x",
    "origin_y",
    "target_x",
    "target_y",
    "origin_third",
    "target_third",
]
_GOAL = (float(spadlconfig.field_length), float(spadlconfig.field_width) / 2.0)
_THIRD = float(spadlconfig.field_length) / 3.0

# NOTE (spec §5b): pitch-third IS included (event-only derivable from x). Score-differential /
# match-minute game-state are a RESERVED extension: SPADL event-only does not guarantee those columns,
# so adding them would fail-closed on providers lacking them. Deferred, not dropped -- a later opt-in
# feature-set behind a bumped feature contract.


def pass_completion_features(origin_x, origin_y, target_x, target_y) -> np.ndarray:
    """Origin/target pass geometry -> the ``FEATURE_NAMES`` feature matrix (event-only).

    A row whose any coordinate is non-finite yields an all-NaN feature row (never a fabricated
    value), so a downstream scorer NaN-propagates rather than inventing a probability.

    Parameters
    ----------
    origin_x, origin_y, target_x, target_y : array-like
        Pass origin and target coordinates in the SPADL action-LTR frame (metres).

    Returns
    -------
    numpy.ndarray
        ``(n, len(FEATURE_NAMES))`` float64 matrix, column order pinned by ``FEATURE_NAMES``.

    Examples
    --------
    A straight 30 m forward pass -- ``distance`` and ``forward`` both equal 30:

    >>> import numpy as np
    >>> from silly_kicks.expected_passing._features import FEATURE_NAMES, pass_completion_features
    >>> X = pass_completion_features(np.array([20.0]), np.array([34.0]),
    ...                              np.array([50.0]), np.array([34.0]))
    >>> round(float(X[0, FEATURE_NAMES.index("distance")]), 1)
    30.0
    """
    # Broadcast to a common shape (>= 1-d) so a SCALAR origin against a length-k target array works --
    # the failed-pass counterfactual seam evaluates one origin against its k selected zone centres
    # (IMPL-01). Byte-identical for equal-length array inputs (incl. the feature-contract probe).
    ox, oy, tx, ty = (np.atleast_1d(np.asarray(v, float)) for v in (origin_x, origin_y, target_x, target_y))
    ox, oy, tx, ty = np.broadcast_arrays(ox, oy, tx, ty)
    dx = tx - ox
    dy = ty - oy
    distance = np.hypot(dx, dy)
    angle = np.arctan2(_GOAL[1] - ty, _GOAL[0] - tx)  # angle from target to goal centre
    origin_third = np.clip(np.floor(ox / _THIRD), 0.0, 2.0)  # 0 def / 1 mid / 2 att (bucketed, PLAN-08)
    target_third = np.clip(np.floor(tx / _THIRD), 0.0, 2.0)
    X = np.column_stack([distance, angle, dx, dy, ox, oy, tx, ty, origin_third, target_third])
    bad = ~np.isfinite(np.column_stack([ox, oy, tx, ty])).all(axis=1)
    X[bad] = np.nan  # NaN in -> NaN features (never fabricated); see Global Constraints
    return X


def feature_contract_block() -> dict:
    """The feature-contract block: ordered names, a fixed-probe feature vector, geometry constants.

    Recorded into the model artifact so ``PassCompletionModel.load`` can fail-closed on a
    feature-name change or a declared geometry-constant drift (ADR-050).

    Returns
    -------
    dict
        ``{"feature_names": [...], "probe_features": [[...]], "geometry": {...}}``.

    Examples
    --------
    >>> from silly_kicks.expected_passing._features import FEATURE_NAMES, feature_contract_block
    >>> block = feature_contract_block()
    >>> block["feature_names"] == FEATURE_NAMES
    True
    """
    probe = pass_completion_features(np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([40.0]))
    return {
        "feature_names": list(FEATURE_NAMES),
        "probe_features": probe.tolist(),
        "geometry": {"field_length": _GOAL[0], "field_width_half": _GOAL[1]},
    }
