import dataclasses

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gk_decision import GkDecisionParams
from silly_kicks.gk_decision._value import option_value


def _rows(comp, byp):
    return pd.DataFrame({"completion": comp, "opponents_bypassed": byp})


def test_option_value_magnitude_and_boundaries():
    ev = option_value(_rows([0.9, 0.9, 0.9, 0.9, 0.5], [0, 3, -1, -5, 2]), params=GkDecisionParams())
    # bypassed=0 -> EV=completion; =3 -> completion*4; negative clipped to 0 -> EV=completion
    assert ev.tolist() == pytest.approx([0.9, 3.6, 0.9, 0.9, 1.5])


def test_option_value_nan_propagates():
    ev = option_value(_rows([np.nan, 0.9], [2, np.nan]), params=GkDecisionParams())
    assert np.isnan(ev.iloc[0]) and np.isnan(ev.iloc[1])


def test_unknown_value_fn_raises():
    p = dataclasses.replace(GkDecisionParams(), value_fn="xt_based")
    with pytest.raises(NotImplementedError):
        option_value(_rows([0.9], [1]), params=p)
