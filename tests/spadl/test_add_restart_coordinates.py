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
            game_id=[9, 9],
            period_id=[1, 1],
            frame_id=[1250, 1250],
            time_seconds=[5.0, 5.0],
            team_id=[0, 1],
            player_id=[-1, 10],
            is_goalkeeper=[False, False],
            is_ball=[True, False],
            # ADR-051 D3 (4.80.0): an unresolvable direction is <NA> and the tracking tier
            # refuses, so `start_coord_source` would be "unresolved" and the tripwire under test
            # would never run. The label has to sit on a NON-BALL row -- the resolver drops ball
            # rows before building its lookup, so a ball-only frame set can never resolve.
            team_attacking_direction=["ltr", "ltr"],
            x=[50.0, 40.0],
            y=[20.0, 30.0],
            source_provider=["gradientsports", "gradientsports"],
        )
    )
    with pytest.warns(UserWarning):
        out = add_restart_coordinates(a, frames=frames)
    assert out.loc[0, "start_coord_source"] == "tripwire_reverted"
    assert np.isnan(out.loc[0, "enriched_start_x"])  # type: ignore[arg-type]


# ---- derived SkillCorner restart rows (action-time repair spec section 4.7) --------------------------------


def _derived(type_name: str, start, src: str, prov: str = "derived"):
    import pandas as pd

    from silly_kicks.spadl import config as _c

    return pd.DataFrame(
        {
            "game_id": ["g"], "period_id": [1], "action_id": [0], "time_seconds": [10.0], "team_id": ["H"],
            "player_id": [pd.NA], "start_x": [start[0]], "start_y": [start[1]], "end_x": [100.0], "end_y": [30.0],
            "type_id": [_c.actiontype_id[type_name]], "result_id": [1], "bodypart_id": [0],
            "action_provenance": [prov], "time_source": [src],
        }
    )  # fmt: skip


import pytest as _pytest  # noqa: E402


@_pytest.mark.parametrize("src", ["placeholder", "order_only"])
@_pytest.mark.parametrize(
    ("type_name", "start"),
    [("corner_crossed", (105.0, 68.0)), ("goalkick", (5.5, 34.0)), ("throw_in", (40.0, 0.0)),
     ("shot_penalty", (94.0, 34.0))],
)  # fmt: skip
def test_derived_restart_with_an_unmeasured_time_reads_as_its_rule_point(type_name, start, src):
    from silly_kicks.spadl import config as _c
    from silly_kicks.tracking._gk_geometry import _PRIOR_CONF, resolve_restart_geometry

    out = resolve_restart_geometry(_derived(type_name, start, src))
    assert out["start_coord_source"].tolist() == ["restart_prior"]
    assert (out["enriched_start_x"].iloc[0], out["enriched_start_y"].iloc[0]) == start
    assert out["start_coord_confidence"].tolist() == [_PRIOR_CONF[_c.actiontype_id[type_name]]]


def test_derived_free_kick_with_an_unmeasured_time_is_unresolved():
    import math

    from silly_kicks.tracking._gk_geometry import resolve_restart_geometry

    out = resolve_restart_geometry(_derived("freekick_crossed", (60.0, 30.0), "placeholder"))
    assert out["start_coord_source"].tolist() == ["unresolved"]
    assert math.isnan(out["enriched_start_x"].iloc[0])


def test_derived_restart_with_a_nan_start_is_unresolved():
    import math

    from silly_kicks.tracking._gk_geometry import resolve_restart_geometry

    out = resolve_restart_geometry(_derived("corner_crossed", (105.0, float("nan")), "placeholder"))
    assert out["start_coord_source"].tolist() == ["unresolved"]
    assert math.isnan(out["enriched_start_y"].iloc[0])


def test_derived_kick_row_reads_as_tracking_ball():
    from silly_kicks.tracking._gk_geometry import _CONF_TRACKING_BALL, resolve_restart_geometry

    out = resolve_restart_geometry(_derived("corner_short", (104.8, 0.3), "kick"))
    assert out["start_coord_source"].tolist() == ["tracking_ball"]
    assert out["start_coord_confidence"].tolist() == [_CONF_TRACKING_BALL]
    assert (out["enriched_start_x"].iloc[0], out["enriched_start_y"].iloc[0]) == (104.8, 0.3)


def test_a_native_corner_is_unchanged_by_the_derived_rule():
    from silly_kicks.tracking._gk_geometry import resolve_restart_geometry

    out = resolve_restart_geometry(_derived("corner_crossed", (105.0, 68.0), "tracking", prov="native"))
    assert out["start_coord_source"].tolist() == ["native"]
