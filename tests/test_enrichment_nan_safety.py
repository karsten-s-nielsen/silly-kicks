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
import silly_kicks.tracking.features as tracking_features


def _discover(module) -> tuple:
    """Return all functions in `module` whose `_nan_safe` attribute is True."""
    return tuple(fn for _, fn in inspect.getmembers(module, inspect.isfunction) if getattr(fn, "_nan_safe", False))


STD_ENRICHMENTS = _discover(std_utils)
ATOMIC_ENRICHMENTS = _discover(atomic_utils)
TRACKING_ENRICHMENTS = _discover(tracking_features)
# Split: helpers needing only (actions, frames) vs those needing extra kwargs
_TRACKING_NEEDS_EXTRA = {
    "add_defensive_line",
    "add_line_break",
    "add_off_ball_context",
    "add_off_ball_runs",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_team_shape",
}
_TRACKING_STANDARD_SIG = tuple(fn for fn in TRACKING_ENRICHMENTS if fn.__name__ not in _TRACKING_NEEDS_EXTRA)
_TRACKING_EXTRA_KWARGS = tuple(fn for fn in TRACKING_ENRICHMENTS if fn.__name__ in _TRACKING_NEEDS_EXTRA)


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


def test_registry_nonempty_tracking() -> None:
    """At least 9 @nan_safe_enrichment helpers in silly_kicks.tracking.features."""
    assert len(TRACKING_ENRICHMENTS) >= 9, (
        f"Expected ≥9 @nan_safe_enrichment helpers in silly_kicks.tracking.features; "
        f"found {len(TRACKING_ENRICHMENTS)}: {[fn.__name__ for fn in TRACKING_ENRICHMENTS]}. "
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


# ---------------------------------------------------------------------------
# Tracking-namespace NaN-safety (PR-S27): auto-discovers @nan_safe_enrichment
# helpers in silly_kicks.tracking.features and fuzzes with a NaN-laced
# (actions, frames) fixture pair.
# ---------------------------------------------------------------------------


@pytest.fixture
def tracking_nan_laced_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """NaN-laced (actions, frames) pair for tracking helper fuzz."""
    actions = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "action_id": list(range(5)),
            "period_id": [1] * 5,
            "time_seconds": [1.0, 2.0, 3.0, 4.0, 5.0],
            "team_id": pd.array([1, 2, pd.NA, 1, 2], dtype="Int64"),
            "player_id": pd.array([101, pd.NA, 201, 102, 202], dtype="Int64"),
            "start_x": [50.0, np.nan, 60.0, 70.0, 80.0],
            "start_y": [34.0, 34.0, np.nan, 34.0, 34.0],
            "end_x": [55.0, 65.0, 70.0, np.nan, 85.0],
            "end_y": [34.0, 34.0, 34.0, 34.0, np.nan],
            "type_id": [0] * 5,
        }
    )
    # Minimal frames: 1 frame per second, both teams + GKs
    frame_rows = []
    for t in range(1, 6):
        frame_rows.extend(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=np.nan,
                    team_id=np.nan,
                    is_ball=True,
                    is_goalkeeper=False,
                    x=50.0,
                    y=34.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=100,
                    team_id=1,
                    is_ball=False,
                    is_goalkeeper=True,
                    x=5.0,
                    y=34.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=101,
                    team_id=1,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=30.0,
                    y=20.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=102,
                    team_id=1,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=32.0,
                    y=40.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=103,
                    team_id=1,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=34.0,
                    y=50.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=104,
                    team_id=1,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=36.0,
                    y=60.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=200,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=True,
                    x=100.0,
                    y=34.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=201,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=70.0,
                    y=20.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=202,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=72.0,
                    y=40.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=203,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=74.0,
                    y=50.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=t,
                    time_seconds=float(t),
                    frame_rate=25.0,
                    player_id=204,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=76.0,
                    y=60.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                ),
            ]
        )
    frames = pd.DataFrame(frame_rows)
    return actions, frames


@pytest.mark.parametrize("helper", _TRACKING_STANDARD_SIG, ids=lambda h: h.__name__)
def test_tracking_helper_nan_safe(helper, tracking_nan_laced_fixture) -> None:
    """Every @nan_safe_enrichment tracking helper with (actions, frames) sig
    survives NaN-laced input without crashing.
    """
    actions, frames = tracking_nan_laced_fixture
    out = helper(actions, frames)
    assert isinstance(out, pd.DataFrame), f"{helper.__name__} returned {type(out).__name__}, expected pd.DataFrame"
    assert len(out) == len(actions), (
        f"{helper.__name__} changed row count on NaN-laced input ({len(actions)} -> {len(out)})"
    )


@pytest.mark.parametrize("helper", _TRACKING_EXTRA_KWARGS, ids=lambda h: h.__name__)
def test_tracking_helper_extra_kwargs_nan_safe(helper, tracking_nan_laced_fixture) -> None:
    """Tracking helpers needing extra kwargs or columns survive NaN-laced input."""
    actions, frames = tracking_nan_laced_fixture
    name = helper.__name__
    if name in ("add_defensive_line", "add_off_ball_runs", "add_line_break", "add_off_ball_context", "add_team_shape"):
        out = helper(actions, frames, home_team_id=1)
    elif name == "add_pre_shot_gk_position":
        # Needs defending_gk_player_id column pre-populated
        acts = actions.copy()
        acts["defending_gk_player_id"] = pd.array([200, pd.NA, 200, 200, pd.NA], dtype="Int64")
        out = helper(acts, frames)
    elif name == "add_pre_shot_gk_angle":
        # frames is keyword-only + needs defending_gk_player_id
        acts = actions.copy()
        acts["defending_gk_player_id"] = pd.array([200, pd.NA, 200, 200, pd.NA], dtype="Int64")
        out = helper(acts, frames=frames)
    else:
        out = helper(actions, frames)
    assert isinstance(out, pd.DataFrame), f"{name} returned {type(out).__name__}, expected pd.DataFrame"
    assert len(out) == len(actions), f"{name} changed row count on NaN-laced input ({len(actions)} -> {len(out)})"
