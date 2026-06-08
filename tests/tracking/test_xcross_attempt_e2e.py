"""Token-gated e2e for TF-17 xCrossAttempt (PR-B). Real pining data; skip when the token is unset.

These assert the extractor + GK validations RUN and EMIT their outputs on real data -- never that
GK "wins" (an inert result is a reported finding, not a CI failure). The tf19_ready gate's true
value is exercised only by the box full-corpus run (Task 11), not a single public match here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

_NO_TOKEN = not os.environ.get("PINING_FOR_THE_DATA_TOKEN")


def _one_public_match():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from _loader_pining import load_matches

    for prov, mid, actions, frames, home in load_matches(providers=["skillcorner"], max_per_provider=1):
        return prov, mid, actions, frames, home
    pytest.skip("no skillcorner match available")
    raise AssertionError("unreachable")


@pytest.mark.skipif(_NO_TOKEN, reason="PINING_FOR_THE_DATA_TOKEN unset")
def test_xcross_cross_provider_extract_runs():
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    _, _, actions, frames, home = _one_public_match()
    X, y, _groups = prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X):
        assert list(X.columns) == XCROSS_FEATURE_NAMES_FAITHFUL
        assert set(np.unique(y)).issubset({0, 1})


@pytest.mark.skipif(_NO_TOKEN, reason="PINING_FOR_THE_DATA_TOKEN unset")
def test_surface_gk_block_ablation_runs():
    from silly_kicks.tracking import _xcross_eval as ev
    from silly_kicks.tracking._xcross_attempt import prepare_xcross_training_data

    _, _, actions, frames, home = _one_public_match()
    X, y, groups = prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X) < 20 or len(np.unique(groups)) < 2:
        pytest.skip("insufficient single-match data for a 2-group CV ablation")
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}
    out = ev.gk_block_ablation(X, y, groups, params)
    assert "delta_pr_auc" in out and "delta_log_loss" in out  # EMITS both, regardless of sign


@pytest.mark.skipif(_NO_TOKEN, reason="PINING_FOR_THE_DATA_TOKEN unset")
def test_gk_substitution_sensitivity_runs():
    from silly_kicks.tracking import _xcross_eval as ev
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel, prepare_xcross_training_data

    _, _, actions, frames, home = _one_public_match()
    X, y, _groups = prepare_xcross_training_data(frames, actions, home_team_id=home)
    if not len(X):
        pytest.skip("no wide-area rows in this match")
    m = XCrossAttemptModel().fit(X, pd.Series(y))
    out = ev.gk_substitution_probe(m, frames, actions=actions, home_team_id=home, n_frames=50)
    # EMITS the distributions + flag; does NOT assert GK wins (inert -> reported, not a failure)
    for k in ("gk_median_abs_delta", "nearest_def_median_abs_delta", "random_band_median_abs_delta", "tf19_ready"):
        assert k in out
