"""The `2474b73` xshot_occurrence `_extract_work`, VERBATIM -- the test oracle for spec r4 D18 (b).

`scripts/train_xshot_occurrence.py` at `2474b73` (git blob `f266ebe73c68ff60e713750f7acd3e11725b41e6`): copied
unchanged below. The extraction then ran with the variant's feature set;
`tests/scripts/test_position_only_projection.py` checks that the faithful extraction, projected by its one consumer,
is the old `position_only` output. Never edit the bodies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_xshot_occurrence import _SIDE_COLS


def _extract_work(item, ctx) -> pd.DataFrame | None:
    """One match -> its tidy feature rows + the four side columns (the legacy `_extract._work`), or None
    ("ran, produced no usable row")."""
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

    collision = set(_SIDE_COLS) & set(XSHOT_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")
    prov, mid, shots, frames, home = item
    X, y, groups = prepare_xshot_training_data(
        frames,
        shots,
        home_team_id=home,
        feature_set=ctx.params["feature_set"],
        horizon_seconds=ctx.params["horizon_seconds"],
        attacking_third_only=True,
        carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
    )
    if not len(X):
        return None
    return X.assign(
        _y=np.asarray(y, int),
        _group=np.asarray(groups),
        _provider=str(prov),
        _match_id=str(mid),  # per-row match id (the arm key)
    )
