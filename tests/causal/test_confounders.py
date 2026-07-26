"""TF-19 sign-off package: Layer 2's confounder join + its registered provenance (spec §5.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.causal._confounders import CONFOUNDER_SOURCE, _time_remaining, join_layer2_confounders


def test_provenance_is_declared_as_frames_computed():
    assert CONFOUNDER_SOURCE == "frames_computed"


def test_mart_sourced_join_is_refused_by_name():
    """A mart-sourced join would hand Layer 2 pre-ADR-045 away-team pressure, and nothing else in
    this package would notice."""
    spells = pd.DataFrame({"game_id": [1], "period_id": [1], "entry_frame_id": [10], "possessing_team": [5]})
    with pytest.raises(ValueError, match="fct_action_context"):
        join_layer2_confounders(spells, frames=None, actions=None, home_team_id=5, source="fct_action_context")


def test_absent_score_differential_raises_rather_than_nan_filling():
    """MEASURED consequence of the alternative: `fit_propensity` on an X with one all-NaN column
    raises `ValueError: Input X contains NaN. LogisticRegression does not accept missing value` --
    naming no column, deep inside the run. Fail earlier and say which."""
    from silly_kicks.causal._confounders import join_layer2_confounders as fn

    with pytest.raises(ValueError, match="score_differential"):
        # source is valid, so the guard under test is the score_differential one
        fn(
            pd.DataFrame({"game_id": [], "period_id": [], "entry_frame_id": [], "possessing_team": []}),
            frames=None,
            actions=None,
            home_team_id=5,
            source=CONFOUNDER_SOURCE,
        )


def test_time_remaining_is_measured_from_the_periods_own_maximum():
    spells = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 2],
            "entry_time": [10.0, 40.0, 5.0],
            "end_time": [20.0, 50.0, 30.0],
        }
    )
    got = _time_remaining(spells)
    assert got == pytest.approx([40.0, 10.0, 25.0])


def test_defending_team_is_resolved_dtype_safely():
    """ADR-019: the frame `team_id` is nullable Int64 while the spell's `possessing_team` may be a
    plain int -- a raw `!=` mis-resolves across dtypes."""
    from silly_kicks.causal._confounders import _defending_team_id

    frames = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": pd.array([5, 6, pd.NA], dtype="Int64"),
            "is_ball": [False, False, True],
        }
    )
    spells = pd.DataFrame({"game_id": [1], "possessing_team": [5]})
    assert int(_defending_team_id(spells, frames).iloc[0]) == 6


@pytest.fixture
def layer2_fixture():
    """Real Layer 2 spells + their frames/actions -- the join's actual input."""
    from causal._fixtures import layer2_frames, layer2_shot_actions

    from silly_kicks.causal import build_opportunities, layer2_config

    frames = layer2_frames(gk_x=4.0)
    actions = layer2_shot_actions((10.0,))
    spells = build_opportunities(frames, actions, home_team_id=5, model_metadata={}, config=layer2_config({}))
    assert len(spells) > 0, "fixture produced no spells -- downstream tests would be vacuous"
    return {"spells": spells, "frames": frames, "actions": actions, "home_team_id": 5}


def test_every_layer2_confounder_column_is_present_after_the_join(layer2_fixture):
    from silly_kicks.causal import LAYER2_CONFOUNDERS

    out = join_layer2_confounders(**layer2_fixture)
    for col in LAYER2_CONFOUNDERS:
        assert col in out.columns, f"missing confounder {col}"


def test_the_pressure_confounder_is_not_structurally_all_nan(layer2_fixture):
    """`add_pressure_on_actor` answers a per-ACTION question while a spell row is FRAME-anchored,
    so an `entry_action_id` join would leave this column dead for every row -- and
    `build_design_matrix` aborts the run on an entirely non-finite confounder. The synthesized
    entry-action path is what makes it real; this is the assertion that proves it."""
    import numpy as np

    out = join_layer2_confounders(**layer2_fixture)
    assert np.isfinite(out["pressure_on_actor__bekkers_pi"].to_numpy(dtype=float)).any()


def test_the_join_does_not_mutate_the_input_spells(layer2_fixture):
    before = layer2_fixture["spells"].copy(deep=True)
    join_layer2_confounders(**layer2_fixture)
    pd.testing.assert_frame_equal(layer2_fixture["spells"], before)
