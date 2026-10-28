"""The `2474b73` xcross_attempt `_prepare`, `_extract_work` and `_probe_work`, VERBATIM -- the test oracle for spec r4
D18 (b), (d).

`scripts/train_xcross_attempt.py` at `2474b73` (git blob `097a2e10ddc27a9b6d42e9e023c454d1a36ff503`): copied
unchanged below. The extraction then ran with the variant's feature set;
`tests/scripts/test_position_only_projection.py` checks that the faithful extraction, projected by its one consumer,
is the old `position_only` output. Never edit the bodies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_xcross_attempt import _SIDE_COLS, _capture_record, _csv


def _prepare(item, ctx):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    collision = set(_SIDE_COLS) & set(XCROSS_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")
    _prov, _mid, actions, frames, home = item
    return prepare_xcross_training_data(
        frames,
        actions,
        home_team_id=home,
        feature_set=ctx.params["feature_set"],
        horizon_seconds=ctx.params["horizon_seconds"],
        wide_area_only=True,
        carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
    )


def _extract_work(item, ctx) -> pd.DataFrame | None:
    """One match -> its tidy feature rows + the four side columns (the legacy `_extract._work`), or None."""
    prov, mid, *_ = item
    X, y, groups = _prepare(item, ctx)
    if not len(X):
        return None  # "ran, produced no usable row"
    return X.assign(
        _y=np.asarray(y, int),
        _group=np.asarray(groups),
        _provider=str(prov),
        _match_id=str(mid),  # per-row match id (the arm key)
    )


def _probe_work(item, ctx) -> pd.DataFrame | None:
    """One probe match: its capture record when its provider is GATED (``--probe-providers``) and the
    extraction yields rows -- the legacy probe pass's capture rule -- else None."""
    prov, mid, *_rest = item
    if prov not in _csv(ctx.params["probe_providers"]):
        return None
    X, _y, groups = _prepare(item, ctx)
    if not len(X):
        return None
    return _capture_record(prov, mid, groups)
