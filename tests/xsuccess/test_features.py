"""TF-61 xSuccess features — END-BLIND, start-anchored (spec §5.2/§5.3)."""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as cfg
from silly_kicks.xsuccess._features import FEATURE_NAMES, xsuccess_features


def _actions(**over):
    base = dict(
        type_id=cfg.actiontype_id["pass"],
        bodypart_id=cfg.bodypart_id["foot"],
        start_x=[20.0],
        start_y=[34.0],
        end_x=[60.0],
        end_y=[40.0],
        result_id=[cfg.result_id["success"]],
        time_seconds=[12.0],
        period_id=[1],
    )
    base.update(over)
    return pd.DataFrame({k: (v if isinstance(v, list) else [v]) for k, v in base.items()})


def test_shape_and_names():
    X = xsuccess_features(_actions())
    assert X.shape == (1, len(FEATURE_NAMES))


def test_distance_is_start_anchored():
    # goal centre (105, 34); start (20, 34) -> distance 85.0, independent of end
    X = xsuccess_features(_actions(start_x=[20.0], start_y=[34.0]))
    di = FEATURE_NAMES.index("distance_to_goal")
    assert round(float(X[0, di]), 1) == 85.0


def test_nan_start_gives_all_nan_row():
    X = xsuccess_features(_actions(start_x=[np.nan]))
    assert np.isnan(X[0]).all()


def test_onehot_type_and_bodypart_present():
    assert any(n.startswith("type_") for n in FEATURE_NAMES)
    assert any(n.startswith("bodypart_") for n in FEATURE_NAMES)


def test_type_onehot_is_hot_for_the_action_type():
    X = xsuccess_features(_actions(type_id=cfg.actiontype_id["shot"]))
    assert float(X[0, FEATURE_NAMES.index("type_shot")]) == 1.0
    assert float(X[0, FEATURE_NAMES.index("type_pass")]) == 0.0
