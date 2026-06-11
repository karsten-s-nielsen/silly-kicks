from typing import cast

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat
from tests._xthreat_helpers import _worldcup_ltr


def test_variable_resolution_24x16(sb_worldcup_data):
    actions = _worldcup_ltr(sb_worldcup_data)
    m = ExpectedThreat(l=24, w=16).fit(actions)
    assert m.xT.shape == (16, 24)
    assert m.transition_matrix.shape == (384, 384)  # type: ignore[union-attr]
    last = cast(pd.DataFrame, sb_worldcup_data["games"]).iloc[-1]
    acts = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{last.game_id}"])
    ratings = m.rate(acts)
    assert len(ratings) == len(acts)
    from silly_kicks.xthreat._grid import _get_successful_move_actions

    idx = _get_successful_move_actions(acts.reset_index()).index
    assert np.isfinite(ratings[idx]).all()
    m.interpolator()  # must construct without error at 24x16
