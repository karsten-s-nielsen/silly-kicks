"""NaN-safety contract enforcement (ADR-003).

Auto-discovers every helper decorated with @nan_safe_enrichment and runs
it against a synthetic NaN-laced fixture. Fails fast if a helper crashes
on NaN-input rows in caller-supplied identifier columns.

Catches: future contributor adds a public enrichment helper without writing
a NaN-safety test. Auto-discovery here covers them automatically when they
opt in via the @nan_safe_enrichment decorator.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import silly_kicks.atomic.spadl.config as atomic_spadlcfg
import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.config as spadlcfg
import silly_kicks.spadl.utils as std_utils


def _discover(module) -> tuple:
    """Return all functions in `module` whose `_nan_safe` attribute is True."""
    return tuple(fn for _, fn in inspect.getmembers(module, inspect.isfunction) if getattr(fn, "_nan_safe", False))


STD_ENRICHMENTS = _discover(std_utils)
ATOMIC_ENRICHMENTS = _discover(atomic_utils)


# ---------------------------------------------------------------------------
# Registry-floor sanity — bulletproofs the auto-discovery mechanism itself.
# If a future refactor accidentally renames `_nan_safe` or breaks the
# decoration on every helper at once, these tests fail explicitly rather
# than silently running zero parametrize cases.
# ---------------------------------------------------------------------------


def test_registry_nonempty_std() -> None:
    """At least 5 @nan_safe_enrichment helpers in silly_kicks.spadl.utils."""
    assert len(STD_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.spadl.utils; "
        f"found {len(STD_ENRICHMENTS)}: {[fn.__name__ for fn in STD_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )


def test_registry_nonempty_atomic() -> None:
    """At least 5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl.utils."""
    assert len(ATOMIC_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl.utils; "
        f"found {len(ATOMIC_ENRICHMENTS)}: {[fn.__name__ for fn in ATOMIC_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )


# ---------------------------------------------------------------------------
# Fixtures: synthetic NaN-laced SPADL frames covering boundary cases:
# - First/middle/last row NaN player_id (positional boundaries)
# - The IDSSE-failure pattern: keeper-action with NaN player_id preceding
#   a shot by the other team.
# - A distribution-eligible row with NaN coordinates (latent crash pattern
#   in add_gk_distribution_metrics).
# ---------------------------------------------------------------------------


@pytest.fixture
def std_nan_laced_actions() -> pd.DataFrame:
    """10-row synthetic standard-SPADL fixture with strategic NaN placements."""
    pass_id = spadlcfg.actiontype_id["pass"]
    keeper_save_id = spadlcfg.actiontype_id["keeper_save"]
    shot_id = spadlcfg.actiontype_id["shot"]
    success_id = spadlcfg.result_id["success"]
    foot_id = spadlcfg.bodypart_id["foot"]

    return pd.DataFrame(
        {
            "game_id": [1] * 10,
            "period_id": [1] * 10,
            "action_id": list(range(10)),
            "team_id": [10, 20, 20, 10, 20, 10, 10, 20, 20, 10],
            "player_id": pd.array(
                [
                    np.nan,  # row 0: NaN at first position (boundary)
                    201.0,
                    np.nan,  # row 2: KEEPER ACTION with NaN player_id (IDSSE pattern)
                    101.0,
                    202.0,
                    np.nan,  # row 5: NaN mid-stream
                    102.0,  # row 6: SHOT (preceded by NaN-keeper at row 2)
                    201.0,
                    202.0,
                    np.nan,  # row 9: NaN at last position (boundary)
                ],
                dtype="float64",
            ),
            "type_id": [
                pass_id,
                pass_id,
                keeper_save_id,  # row 2: defending keeper
                pass_id,
                pass_id,
                pass_id,
                shot_id,  # row 6: shot — triggers add_pre_shot_gk_context
                pass_id,
                pass_id,
                pass_id,
            ],
            "result_id": [success_id] * 10,
            "result_name": ["success"] * 10,
            "bodypart_id": [foot_id] * 10,
            "bodypart_name": ["foot"] * 10,
            "type_name": ["pass"] * 10,  # placeholder
            "time_seconds": [float(i) for i in range(10)],
            "start_x": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                np.nan,  # row 5: NaN coordinate (latent crash pattern)
                60.0,
                70.0,
                80.0,
                90.0,
            ],
            "start_y": [10.0] * 10,
            "end_x": [20.0, 30.0, 40.0, 50.0, 60.0, np.nan, 70.0, 80.0, 90.0, 100.0],
            "end_y": [10.0] * 10,
        }
    )


@pytest.fixture
def atomic_nan_laced_actions() -> pd.DataFrame:
    """10-row synthetic atomic-SPADL fixture (same NaN positions; atomic schema)."""
    pass_id = atomic_spadlcfg.actiontype_id["pass"]
    keeper_save_id = atomic_spadlcfg.actiontype_id["keeper_save"]
    shot_id = atomic_spadlcfg.actiontype_id["shot"]

    return pd.DataFrame(
        {
            "game_id": [1] * 10,
            "period_id": [1] * 10,
            "action_id": list(range(10)),
            "team_id": [10, 20, 20, 10, 20, 10, 10, 20, 20, 10],
            "player_id": pd.array(
                [np.nan, 201.0, np.nan, 101.0, 202.0, np.nan, 102.0, 201.0, 202.0, np.nan],
                dtype="float64",
            ),
            "type_id": [
                pass_id,
                pass_id,
                keeper_save_id,
                pass_id,
                pass_id,
                pass_id,
                shot_id,
                pass_id,
                pass_id,
                pass_id,
            ],
            "type_name": ["pass"] * 10,
            "bodypart_id": [0] * 10,
            "bodypart_name": ["foot"] * 10,
            "time_seconds": [float(i) for i in range(10)],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, 60.0, 70.0, 80.0, 90.0],
            "y": [10.0] * 10,
            "dx": [10.0] * 10,
            "dy": [0.0] * 10,
        }
    )


# ---------------------------------------------------------------------------
# Auto-discovered fuzz: every decorated helper x NaN-laced fixture.
# Failure mode: any decorated helper that crashes on NaN-laced input fails
# its parametrized case here. Adding a new @nan_safe_enrichment-decorated
# helper auto-extends this test (no test-author work needed).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("helper", STD_ENRICHMENTS, ids=lambda h: h.__name__)
def test_standard_helper_nan_safe(helper, std_nan_laced_actions) -> None:
    """Every @nan_safe_enrichment standard helper survives NaN-laced input
    with default kwargs.
    """
    out = helper(std_nan_laced_actions)
    assert isinstance(out, pd.DataFrame), f"{helper.__name__} returned {type(out).__name__}, expected pd.DataFrame"
    assert len(out) == len(std_nan_laced_actions), (
        f"{helper.__name__} changed row count on NaN-laced input ({len(std_nan_laced_actions)} -> {len(out)})"
    )


@pytest.mark.parametrize("helper", ATOMIC_ENRICHMENTS, ids=lambda h: h.__name__)
def test_atomic_helper_nan_safe(helper, atomic_nan_laced_actions) -> None:
    """Every @nan_safe_enrichment atomic helper survives NaN-laced input."""
    out = helper(atomic_nan_laced_actions)
    assert isinstance(out, pd.DataFrame), f"{helper.__name__} returned {type(out).__name__}, expected pd.DataFrame"
    assert len(out) == len(atomic_nan_laced_actions), (
        f"{helper.__name__} changed row count on NaN-laced input ({len(atomic_nan_laced_actions)} -> {len(out)})"
    )


# ---------------------------------------------------------------------------
# Per-helper specific assertions — exact behavior on the bug-triggering rows.
# ---------------------------------------------------------------------------


def test_pre_shot_gk_context_preserves_nan_on_unidentifiable_shot(std_nan_laced_actions) -> None:
    """When the most-recent defending keeper-action has NaN player_id, the
    shot's defending_gk_player_id is NaN — not raises, not 0, not a sentinel.
    """
    out = std_utils.add_pre_shot_gk_context(std_nan_laced_actions)
    # Row 6 is the shot (type_id=shot, team=10); row 2 is the defending team's
    # keeper_save with NaN player_id. The helper must skip the int(NaN) cast.
    shot_row = out[out["action_id"] == 6].iloc[0]
    assert pd.isna(shot_row["defending_gk_player_id"]), (
        f"Expected NaN defending_gk_player_id for shot following NaN-keeper; got {shot_row['defending_gk_player_id']!r}"
    )


def test_atomic_pre_shot_gk_context_preserves_nan(atomic_nan_laced_actions) -> None:
    """Atomic counterpart of test_pre_shot_gk_context_preserves_nan."""
    out = atomic_utils.add_pre_shot_gk_context(atomic_nan_laced_actions)
    shot_row = out[out["action_id"] == 6].iloc[0]
    assert pd.isna(shot_row["defending_gk_player_id"])


def test_gk_distribution_metrics_excludes_nan_coords(std_nan_laced_actions) -> None:
    """When a distribution-eligible row has NaN coords, gk_xt_delta is NaN
    for that row (not raises, not arbitrary integer from int(NaN)).
    """
    xt_grid = np.zeros((12, 8), dtype=np.float64)
    out = std_utils.add_gk_distribution_metrics(std_nan_laced_actions, xt_grid=xt_grid)
    nan_coord_row = out.iloc[5]
    # gk_xt_delta should be NaN at NaN-coord rows (not crash, not arbitrary int).
    assert pd.isna(nan_coord_row["gk_xt_delta"]) or nan_coord_row["gk_xt_delta"] == 0.0, (
        f"Expected NaN/0.0 gk_xt_delta on NaN-coord row; got {nan_coord_row['gk_xt_delta']!r}"
    )
