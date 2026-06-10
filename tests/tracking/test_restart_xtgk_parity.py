"""Goal-kick parity: the resolve_restart_geometry promotion must not change any
resolve_gk_geometry consumer's output (xT-GK + completion). Spec section 7.

The byte-identical guard is the Task-1 committed golden (tests/tracking/test_gk_geometry.py
TestGoldenSnapshot, captured on unmodified code). These tests are supplementary -- they assert the
shim-vs-engine type-gating contract directly (throw-in not imputed in the shim; goal-kick legacy
labels/columns)."""

import numpy as np
import pandas as pd

from silly_kicks.tracking._gk_geometry import resolve_gk_geometry, resolve_restart_geometry

_GK, _THROW = 22, 2


def _mixed():
    # row 0: goalkick (NaN origin). row 1: throw_in (NaN origin; native end_y gives the side, and the
    # next row's finite start_x gives the along-line x -> general CAN impute its origin). row 2:
    # goalkick (native origin 30,40; NaN end -> last row -> dest unresolved).
    return pd.DataFrame(
        dict(
            game_id=[9, 9, 9],
            period_id=[1, 1, 1],
            action_id=[0, 1, 2],
            team_id=[1, 1, 1],
            player_id=[10, 11, 10],
            type_id=[_GK, _THROW, _GK],
            time_seconds=[5.0, 6.0, 70.0],
            start_x=[np.nan, np.nan, 30.0],
            start_y=[np.nan, np.nan, 40.0],
            end_x=[60.0, 40.0, np.nan],
            end_y=[30.0, 20.0, np.nan],
        )
    )


def test_throwin_not_imputed_in_shim_but_imputed_in_general():
    a = _mixed()
    legacy = resolve_gk_geometry(a, frames=None)  # engine runs impute_types=(goalkick,)
    general = resolve_restart_geometry(a, frames=None)  # default impute_types -> all
    # row 1 is a throw_in: the shim's goalkick-only impute_types means its origin is NEVER imputed
    # (NaN -> unresolved), so no revert step is needed.
    assert legacy.loc[1, "origin_source"] == "unresolved"
    assert np.isnan(legacy.loc[1, "origin_x"])
    # general DOES impute it (side from native end_y=20 -> touchline 0; along-line x from next_event
    # start_x=30) -> restart_prior at (30, 0).
    assert general.loc[1, "start_coord_source"] == "restart_prior"
    assert general.loc[1, "enriched_start_x"] == 30.0
    assert general.loc[1, "enriched_start_y"] == 0.0


def test_goalkick_legacy_labels_and_columns():
    legacy = resolve_gk_geometry(_mixed(), frames=None)
    assert set(legacy.columns) == {
        "origin_x",
        "origin_y",
        "origin_source",
        "origin_confidence",
        "dest_x",
        "dest_y",
        "dest_source",
    }
    # goalkick row 0 -> rule-point labeled goalkick_prior (not restart_prior)
    assert legacy.loc[0, "origin_source"] == "goalkick_prior"
    # goalkick row 2 -> NaN end, no in-period next event (last row) -> unresolved (Major-2b)
    assert legacy.loc[2, "dest_source"] == "unresolved"
