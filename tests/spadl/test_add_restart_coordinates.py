import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_restart_coordinates

_GK = 22
_CORNER_C = 5


def _actions():
    return pd.DataFrame(
        dict(
            game_id=[9, 9],
            period_id=[1, 1],
            action_id=[1, 0],
            team_id=[1, 1],
            player_id=[10, 10],
            type_id=[_GK, 0],
            time_seconds=[6.0, 5.0],
            start_x=[np.nan, 50.0],
            start_y=[np.nan, 30.0],
            end_x=[60.0, 55.0],
            end_y=[30.0, 30.0],
        )
    )


def test_emits_enriched_columns_and_does_not_mutate_canonical():
    a = _actions()
    before = a["start_x"].copy()
    out = add_restart_coordinates(a, frames=None)
    assert {
        "enriched_start_x",
        "start_coord_source",
        "start_coord_confidence",
        "enriched_end_x",
        "end_coord_source",
        "end_coord_confidence",
    } <= set(out.columns)
    # canonical start_x untouched on the original frame
    pd.testing.assert_series_equal(a["start_x"], before)


def test_sorts_by_game_period_action():
    out = add_restart_coordinates(_actions(), frames=None)
    assert list(out["action_id"]) == [0, 1]  # sorted


def test_goalkick_origin_imputed_events_only():
    out = add_restart_coordinates(_actions(), frames=None)
    gk = out[out["type_id"] == _GK].iloc[0]
    assert gk["start_coord_source"] == "restart_prior"
    assert gk["enriched_start_x"] == pytest.approx(5.5)


def test_nan_identifier_safe():
    a = _actions()
    a.loc[0, "player_id"] = np.nan
    out = add_restart_coordinates(a, frames=None)  # must not raise
    assert len(out) == 2


def test_tripwire_reverts_bad_imputed_corner_at_edge():
    # corner with NaN origin; force an out-of-region imputed coord via frames ball at midfield;
    # tripwire (edge) reverts + tags.
    a = pd.DataFrame(
        dict(
            game_id=[9],
            period_id=[1],
            action_id=[0],
            team_id=[1],
            player_id=[10],
            type_id=[_CORNER_C],
            time_seconds=[5.0],
            start_x=[np.nan],
            start_y=[np.nan],
            end_x=[np.nan],
            end_y=[np.nan],
        )
    )
    frames = pd.DataFrame(
        dict(
            game_id=[9],
            period_id=[1],
            frame_id=[1250],
            time_seconds=[5.0],
            team_id=[0],
            player_id=[-1],
            is_goalkeeper=[False],
            is_ball=[True],
            x=[50.0],
            y=[20.0],
            source_provider=["gradientsports"],
        )
    )
    with pytest.warns(UserWarning):
        out = add_restart_coordinates(a, frames=frames)
    assert out.loc[0, "start_coord_source"] == "tripwire_reverted"
    assert np.isnan(out.loc[0, "enriched_start_x"])  # type: ignore[arg-type]
