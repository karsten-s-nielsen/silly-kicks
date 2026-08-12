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

import silly_kicks.atomic.spadl as atomic_spadl
import silly_kicks.atomic.spadl.config as atomic_spadlcfg
import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl as std_spadl
import silly_kicks.spadl.config as spadlcfg
import silly_kicks.spadl.utils as std_utils
import silly_kicks.tracking as tracking_pkg

# ADR-041 opt-out: auto-enumerating gate: sweeps EVERY registered aggregator on defaults, so the OBSO family's
# synthetic-EPV notice is expected and irrelevant here.
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")


def _discover(module) -> tuple:
    """Return all functions in `module` whose `_nan_safe` attribute is True."""
    return tuple(fn for _, fn in inspect.getmembers(module, inspect.isfunction) if getattr(fn, "_nan_safe", False))


# Discovered from the PACKAGE, not the defining module (Cycle B).
#
# `TRACKING_ENRICHMENTS` used to scan `silly_kicks.tracking.features` alone, while the contract it
# guards covers the whole public `silly_kicks.tracking` surface. Three decorated helpers therefore
# sat outside the registry entirely -- `add_sync_score` (defined in `tracking/utils.py`),
# `add_xshot_occurrence` and `add_xcross_attempt` (defined in their own private modules). Two of the
# three were ALREADY decorated and still never exercised: the claim was made and nothing checked it.
#
# The discovery SCOPE was the defect, not the decorations. Measured: package discovery adds exactly
# those three and drops nothing (29 -> 32); `spadl` and `atomic.spadl` are unchanged at 7 and 5,
# because their public `add_*` all happen to live in `utils`.
STD_ENRICHMENTS = _discover(std_spadl)
ATOMIC_ENRICHMENTS = _discover(atomic_spadl)
TRACKING_ENRICHMENTS = _discover(tracking_pkg)
# Split: helpers needing only (actions, frames) vs those needing extra kwargs
_TRACKING_NEEDS_EXTRA = {
    "add_cover_shadows",
    "add_das",
    "add_defensive_credit",
    "add_defensive_line",
    "add_ghost_gk",
    "add_gk_influence",
    "add_line_break",
    "add_off_ball_context",
    "add_off_ball_runs",
    "add_packing",
    "add_player_influence",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_press_commitment",
    "add_shape_graph",
    "add_space_creation",
    "add_structural_pass",
    "add_team_shape",
    # ADR-055: not "needs extra kwargs" in the usual sense -- it takes NO frames at all
    # (`(actions, *, visible_area, links)`), so the standard-signature test cannot call it.
    "add_visible_area_coverage",
    "add_xt_gk",
    "add_off_ball_run_values",
    # Cycle B: newly VISIBLE to the registry once discovery widened to the package. All three
    # were outside `tracking.features` and therefore outside the old scan; two of them were
    # already decorated, so the claim existed while nothing exercised it.
    "add_sync_score",
    "add_xshot_occurrence",
    "add_xcross_attempt",
}
_TRACKING_STANDARD_SIG = tuple(fn for fn in TRACKING_ENRICHMENTS if fn.__name__ not in _TRACKING_NEEDS_EXTRA)
_TRACKING_EXTRA_KWARGS = tuple(fn for fn in TRACKING_ENRICHMENTS if fn.__name__ in _TRACKING_NEEDS_EXTRA)


# ---------------------------------------------------------------------------
# Registry-floor sanity — bulletproofs the auto-discovery mechanism itself.
# If a future refactor accidentally renames `_nan_safe` or breaks the
# decoration on every helper at once, these tests fail explicitly rather
# than silently running zero parametrize cases.
# ---------------------------------------------------------------------------


#: Public `add_*` helpers deliberately NOT @nan_safe_enrichment, each with a stated reason.
#: An entry is a decision on the record; an omission is a helper whose NaN-safety is never tested.
_NOT_NAN_SAFE: dict[str, str] = {
    "add_gradientsports_player_ids": (
        "not an action enricher and structurally outside ADR-003's contract, which is about NaN "
        "identifiers in a caller-supplied ACTIONS frame. Its signature is "
        "(jersey_frames, roster, *, home_team_id, away_team_id) -- it takes no actions frame at "
        "all -- and it returns a (DataFrame, GradientsportsRosterReport) TUPLE, so the harness's "
        "`isinstance(out, pd.DataFrame)` and row-count assertions do not apply. Unresolved "
        "jerseys are surfaced through its own report rather than by NaN-routing."
    ),
}

_PIN = (
    ("spadl", std_spadl, STD_ENRICHMENTS),
    ("atomic.spadl", atomic_spadl, ATOMIC_ENRICHMENTS),
    ("tracking", tracking_pkg, TRACKING_ENRICHMENTS),
)


@pytest.mark.parametrize("label,pkg,registry", _PIN, ids=[p[0] for p in _PIN])
def test_every_public_add_star_is_enrolled_or_exempted(label, pkg, registry) -> None:
    """ADR-003's registry is auto-discovered from the decorator, so it is complete over DECORATED
    helpers -- but decoration is the human-maintained opt-in and nothing tied it to the public
    surface. The three floors below pass identically whether or not a new public `add_*` was
    decorated.

    ADR-033 and ADR-051 both pin their surface to the public export in BOTH directions; this is
    ADR-003 catching up (Cycle B).

    Pinned to the PACKAGE export, not the module: `silly_kicks.spadl.utils` has no `__all__` at
    all, so a module-level pin would assert nothing on two of the three registries.
    """
    exported = {n for n in pkg.__all__ if n.startswith("add_")}
    decorated = {fn.__name__ for fn in registry}
    unenrolled = sorted(exported - decorated - set(_NOT_NAN_SAFE))
    assert not unenrolled, (
        f"public add_* in {label}.__all__ with no @nan_safe_enrichment and no exemption: "
        f"{unenrolled}. ADR-003 makes NaN-tolerance a contract for the whole public enrichment "
        f"family; an undecorated helper is never exercised against NaN identifiers."
    )


def test_nan_safe_exemptions_are_real_public_helpers() -> None:
    """Self-burning-down."""
    public: set[str] = set()
    for _, pkg, _ in _PIN:
        public |= set(pkg.__all__)
    stale = sorted(set(_NOT_NAN_SAFE) - public)
    assert not stale, f"_NOT_NAN_SAFE names helpers that are not public: {stale}"


def test_registry_nonempty_std() -> None:
    """At least 5 @nan_safe_enrichment helpers in silly_kicks.spadl."""
    assert len(STD_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.spadl; "
        f"found {len(STD_ENRICHMENTS)}: {[fn.__name__ for fn in STD_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )


def test_registry_nonempty_tracking() -> None:
    """At least 10 @nan_safe_enrichment helpers in silly_kicks.tracking."""
    assert len(TRACKING_ENRICHMENTS) >= 10, (
        f"Expected ≥10 @nan_safe_enrichment helpers in silly_kicks.tracking; "
        f"found {len(TRACKING_ENRICHMENTS)}: {[fn.__name__ for fn in TRACKING_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )


def test_registry_nonempty_atomic() -> None:
    """At least 5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl."""
    assert len(ATOMIC_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl; "
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
    if name in (
        # ADR-051 D3 (4.80.0): these five no longer take `home_team_id` -- direction comes
        # from the goal map or from `acting_team_attacks_rtl`, both derived from `frames`.
        "add_defensive_line",
        "add_line_break",
        "add_off_ball_context",
        "add_structural_pass",
        # 4.80.0 dead-parameter removal (these two were already direction-correct; the
        # argument they carried was never read).
        "add_team_shape",
        "add_shape_graph",
    ):
        out = helper(actions, frames)
    elif name in (
        "add_off_ball_runs",
        "add_space_creation",
    ):
        out = helper(actions, frames, home_team_id=1)
    elif name == "add_sync_score":
        # Takes (actions, LINKS) rather than (actions, frames) -- the only helper in the
        # registry with that shape, which is why it needs its own branch rather than a
        # column top-up.
        from silly_kicks.tracking import link_actions_to_frames

        out = helper(actions, link_actions_to_frames(actions, frames)[0])
    elif name == "add_xshot_occurrence":
        # `ball_state` is a documented contract column for this extractor and is NOT in
        # TRACKING_FRAMES_COLUMNS. Same supply-the-contract-columns precedent as add_das
        # below: without it the helper raises before reaching the NaN-IDENTIFIER surface
        # this gate exists to fuzz, and would pass vacuously.
        #
        # Deliberately NOT given vx/vy: this extractor WARNS and falls back to distance-only
        # carrier inference, so supplying them would silently change which path the gate
        # exercises. `add_xcross_attempt` used to share this branch and now has its own,
        # because it needs the vector and this one must not be given it.
        fr = frames.copy()
        if "ball_state" not in fr.columns:
            fr["ball_state"] = "alive"
        out = helper(actions, fr)
    elif name == "add_xcross_attempt":
        # `ball_state` as above, PLUS vx/vy -- the same supply-the-contract-columns precedent as
        # add_das below, and for the identical reason. This extractor now honours the ADR-054
        # velocity contract: velocity DECLARED with vx/vy ABSENT is the "forgot
        # derive_velocities()" case and RAISES. The shared fixture supplies neither, so without
        # the vector the gate would measure that refusal instead of the NaN-IDENTIFIER surface.
        #
        # It previously "passed" here only because the aggregator crashed with a bare
        # `KeyError: 'vx'` -- the same vacuous pass ADR-043 records for add_das before its catch
        # was narrowed.
        fr = frames.copy()
        if "ball_state" not in fr.columns:
            fr["ball_state"] = "alive"
        fr["vx"] = 0.0
        fr["vy"] = 0.0
        out = helper(actions, fr)
        # Non-vacuity, mirroring add_das below: the scorer must actually have run rather than the
        # whole call degrading, or the NaN-identifier assertions downstream mean nothing.
        assert "xcross_attempt" in out.columns
    elif name == "add_das":
        # Same supply-the-contract-columns precedent as add_packing / add_ghost_gk: vx, vy
        # and team_in_possession are NOT in TRACKING_FRAMES_COLUMNS (derive_velocities /
        # derive_team_in_possession produce them) and are add_das's documented contract.
        #
        # This branch is new in ADR-043 and it is NOT a relaxation -- it is the gate finally
        # getting teeth. Before the catch was narrowed, add_das swallowed the missing-vx
        # ValueError and returned an all-NaN column, so this helper passed the NaN-safety
        # gate VACUOUSLY: it never got past _validate_das_inputs and never touched the
        # NaN-IDENTIFIER surface (NaN team_id / player_id on the ball rows) the gate exists
        # to fuzz. Supplying the contract columns is what makes it actually run.
        frames = frames.copy()
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        frames["team_in_possession"] = 1
        out = helper(actions, frames)
        # Non-vacuity: prove the simulation really ran (some row scored) AND that the
        # NaN-identifier row took the documented per-row default (NaN out, named cause)
        # rather than the whole call degrading.
        assert (out["das_source"] == "computed").any(), "add_das degraded wholesale; gate is vacuous again"
        nan_team = actions["team_id"].isna().to_numpy()
        assert out.loc[nan_team, "das_team"].isna().all()
        assert (out.loc[nan_team, "das_source"] == "team_unresolved").all()
    elif name == "add_packing":
        # The shared fixture carries no result_id (it is not an identifier column);
        # add_packing's completion gate requires it. Supply a constant success so the
        # NaN-IDENTIFIER surface (NaN team/player/coords) is what gets fuzzed -- the
        # add_xt_gk branch's supply-the-contract-columns precedent.
        acts = actions.copy()
        acts["result_id"] = 1
        # ADR-051 D3 (4.80.0): no `home_team_id`; the goal map is built from `frames`. This
        # branch stays SEPARATE from the no-kwarg group above because of the `result_id`
        # top-up, not because of the signature.
        out = helper(acts, frames)
    elif name == "add_xt_gk":
        import numpy as np

        from silly_kicks.tracking._xt_gk import _gk_distribution_mask
        from silly_kicks.xthreat import ExpectedThreat

        # V1 (review #2): VERIFY (don't assume) this shared fixture has zero in-scope GK
        # distributions, so get_xc is never reached. If the fixture ever gains a goalkick or
        # a GK-actor pass, get_xc runs and the frame ball+player co-occurrence contract
        # applies (a hard-crash path the DAS columns below do NOT cover) -- fail loudly here.
        assert _gk_distribution_mask(actions, frames).sum() == 0, (
            "add_xt_gk nan-safety branch assumes no in-scope GK distributions in the shared "
            "fixture; it changed -- give this branch frames satisfying the full get_xc contract."
        )
        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        frames = frames.copy()  # defensive DAS columns (belt-and-suspenders; mask is empty)
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        frames["team_in_possession"] = 1
        out = helper(actions, frames, xt)
    elif name == "add_off_ball_run_values":
        import numpy as np

        from silly_kicks.xthreat import ExpectedThreat

        # Same supply-the-contract-columns precedent as add_packing: the shared fixture
        # carries no result_id, and TF-35's domain is "completed pass/cross", so without
        # one every row is off-domain and the NaN-IDENTIFIER surface (NaN team/player/
        # coords) -- the thing this gate actually fuzzes -- is never reached.
        acts = actions.copy()
        acts["result_id"] = 1
        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        out = helper(acts, frames, xt)
    elif name in ("add_gk_influence", "add_cover_shadows"):
        # ADR-055: these two take an optional `goal_map` and no `home_team_id`. Omitted, so
        # the aggregator derives the map from the SAME NaN-bearing frames this gate feeds
        # it -- which is the behaviour under test: a NaN team id must route to the
        # documented per-row default, including through goal-end resolution.
        import numpy as np

        from silly_kicks.xthreat import ExpectedThreat

        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        out = helper(actions, frames, xt)
    elif name == "add_player_influence":
        import numpy as np

        from silly_kicks.xthreat import ExpectedThreat

        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        # ADR-051 D3 (4.80.0): direction comes from `acting_team_attacks_rtl`, not an argument.
        out = helper(actions, frames, xt)
    elif name == "add_ghost_gk":
        import numpy as np

        from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

        rng = np.random.default_rng(42)
        X_train = pd.DataFrame(rng.standard_normal((50, 26)), columns=GHOST_GK_FEATURE_NAMES)
        X_train["phase"] = rng.integers(0, 3, 50).astype(float)
        X_train["team_in_possession"] = rng.integers(0, 2, 50).astype(float)
        X_train["ball_in_own_half"] = rng.integers(0, 2, 50).astype(float)
        labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 50), "gk_y": rng.uniform(25, 45, 50)})
        m = GhostGkModel(n_estimators=5)
        m.fit(X_train, labels)
        # PR-S81: the serve-carrier fix makes add_ghost_gk run infer_ball_carrier, which
        # needs ball_state (a TRACKING_FRAMES_COLUMNS field the minimal fixture omits) +
        # vx/vy. Provide them so the carrier resolves; NaN-laced identifiers stay fuzzed.
        frames = frames.copy()
        frames["ball_state"] = "alive"
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        out = helper(actions, frames, model=m, home_team_id=1)
    elif name == "add_defensive_credit":
        import numpy as np

        from silly_kicks.xthreat import ExpectedThreat

        # Supply the xg/block/on-target contract columns (+ a result_id the shared fixture lacks,
        # like add_packing) so the NaN-IDENTIFIER surface (NaN team/player) is what gets fuzzed.
        acts = actions.copy()
        acts["result_id"] = 1
        acts["xg"] = 0.2
        acts["shot_blocked"] = pd.array([pd.NA] * len(acts), dtype="boolean")
        acts["cross_blocked"] = pd.array([pd.NA] * len(acts), dtype="boolean")
        acts["shot_on_target_derived"] = pd.array([pd.NA] * len(acts), dtype="boolean")
        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        out = helper(acts, frames, xg_column="xg", xt=xt)
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
    elif name == "add_visible_area_coverage":
        # ADR-055: takes NO frames -- `(actions, *, visible_area, links)`. The NaN-IDENTIFIER
        # surface this gate fuzzes is `actions.action_id`, which is the join key against
        # `visible_area`, so a NaN there must route to the documented `no_polygon` default rather
        # than raise or drop the row. The polygon itself is a valid half-pitch: fuzzing IT would
        # test `as_polygon`, which `tests/tracking/test_visibility.py` already covers.
        import numpy as np

        half = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
        visible = pd.DataFrame({"action_id": list(actions["action_id"]), "polygon": [half] * len(actions)})
        out = helper(actions, visible_area=visible)
    elif name == "add_press_commitment":
        # The velocity contract raises loud on missing vx/vy (correct) -- supply them so the
        # NaN-IDENTIFIER surface (NaN team/player) is what gets fuzzed, not the velocity guard.
        frames = frames.copy()
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        out = helper(actions, frames)
    else:
        out = helper(actions, frames)
    assert isinstance(out, pd.DataFrame), f"{name} returned {type(out).__name__}, expected pd.DataFrame"
    assert len(out) == len(actions), f"{name} changed row count on NaN-laced input ({len(actions)} -> {len(out)})"
