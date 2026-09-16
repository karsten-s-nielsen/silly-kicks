"""TF-61 xSuccess END-LOCATION leakage guard (spec §6.4, the ReceiverModel bar).

For a FAILED action the SPADL end IS the outcome (interception/death point), so an end-using feature
is a postdiction (target leakage). This guard proves xsuccess_features is END-BLIND: perturbing
end_x/end_y (and result_id) must leave the feature matrix byte-identical.
"""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as cfg
from silly_kicks.xsuccess._features import xsuccess_features


def _multi():
    return pd.DataFrame(
        dict(
            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["take_on"]],
            bodypart_id=[cfg.bodypart_id["foot"], cfg.bodypart_id["foot"]],
            start_x=[20.0, 55.0],
            start_y=[34.0, 20.0],
            end_x=[60.0, 70.0],
            end_y=[40.0, 25.0],
            result_id=[cfg.result_id["success"], cfg.result_id["fail"]],
            time_seconds=[12.0, 40.0],
            period_id=[1, 1],
        )
    )


def test_features_invariant_to_end_and_result():
    a = _multi()
    base = xsuccess_features(a)
    b = a.copy()
    b["end_x"] = b["end_x"] + 30.0
    b["end_y"] = 5.0
    b["result_id"] = cfg.result_id["success"]
    perturbed = xsuccess_features(b)
    assert np.array_equal(base, perturbed, equal_nan=True), "xSuccess features must be END-BLIND (and result-blind)"
