"""id-dtype invariance for restdefense (ADR-019): numeric x string ids yield identical metrics.

There is no ``home_team_id`` parameter -- direction comes from the ``GoalMap`` -- so the only axes
are the actions' and the frames' id dtypes. Both directions (numeric actions x string frames and the
reverse) must reproduce the all-numeric metrics exactly."""

import pandas as pd

from silly_kicks.restdefense import RD_METRIC_COLUMNS
from silly_kicks.restdefense._compute import compute_rest_defense
from tests.restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture

_ID_COLS = ["game_id", "team_id", "player_id"]
_NUMERIC = [c for c in RD_METRIC_COLUMNS if c != "rd_shape_2_3_vs_3_2"]


def _stringify(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def _metrics_by_action(samples):
    resolved = samples[samples["rd_geometry_source"] == "resolved"].copy()
    resolved["action_id"] = resolved["action_id"].astype("int64")
    return resolved.set_index("action_id").sort_index()


def _assert_metrics_equal(a, b):
    ma, mb = _metrics_by_action(a), _metrics_by_action(b)
    assert list(ma.index) == list(mb.index)
    for c in _NUMERIC:
        pd.testing.assert_series_equal(ma[c].reset_index(drop=True), mb[c].reset_index(drop=True), check_names=False)
    assert list(ma["rd_shape_2_3_vs_3_2"]) == list(mb["rd_shape_2_3_vs_3_2"])


def test_string_frame_ids_match_numeric():
    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    base, _ = compute_rest_defense(actions, frames, xt=xt)
    out, _ = compute_rest_defense(actions, _stringify(frames, _ID_COLS), xt=xt)
    _assert_metrics_equal(base, out)


def test_string_action_ids_match_numeric():
    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    base, _ = compute_rest_defense(actions, frames, xt=xt)
    out, _ = compute_rest_defense(_stringify(actions, _ID_COLS), frames, xt=xt)
    _assert_metrics_equal(base, out)
