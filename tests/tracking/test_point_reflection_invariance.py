"""Per-row point-reflection invariance across every reflection site (ADR-045).

PER-ROW, never aggregate: D2's mean bias is -1.1% and D3's is -0.002 because rows
over- and under-state in near-equal measure. A mean-comparison gate passes cleanly
on broken code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.reflection import GEOMETRIC_NAME as _GEOMETRY_COL
from silly_kicks.tracking.features import add_pressure_on_actor

FL, FW = 105.0, 68.0


def _scenario():
    actions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 10.0,
                "team_id": "H",
                "player_id": "h1",
                "start_x": 60.0,
                "start_y": 40.0,
                "end_x": 70.0,
                "end_y": 40.0,
                "type_name": "pass",
                "result_name": "success",
            }
        ]
    )
    rows = [
        {"player_id": "h1", "team_id": "H", "x": 60.0, "y": 40.0, "vx": 2.0, "vy": 0.0},
        {"player_id": "a1", "team_id": "A", "x": 63.0, "y": 41.0, "vx": -5.0, "vy": -1.0},
        {"player_id": "a2", "team_id": "A", "x": 66.0, "y": 38.0, "vx": -4.0, "vy": 1.5},
        {"player_id": "hgk", "team_id": "H", "x": 5.0, "y": 34.0, "vx": 0.0, "vy": 0.0},
        {"player_id": "agk", "team_id": "A", "x": 100.0, "y": 34.0, "vx": 0.0, "vy": 0.0},
    ]
    frames = pd.DataFrame(rows)
    frames["game_id"] = 1
    frames["period_id"] = 1
    frames["frame_id"] = 250
    frames["time_seconds"] = 10.0
    frames["is_ball"] = False
    frames["source_provider"] = "snapshot"
    frames["is_goalkeeper"] = frames["player_id"].isin(["hgk", "agk"])
    frames["speed"] = np.hypot(frames["vx"], frames["vy"])
    frames["team_attacking_direction"] = np.where(frames["team_id"] == "H", "ltr", "rtl")
    ball = {
        "player_id": None,
        "team_id": None,
        "x": 60.0,
        "y": 40.0,
        "vx": 2.0,
        "vy": 0.0,
        "game_id": 1,
        "period_id": 1,
        "frame_id": 250,
        "time_seconds": 10.0,
        "is_ball": True,
        "is_goalkeeper": False,
        "speed": 2.0,
        "team_attacking_direction": None,
        "source_provider": "snapshot",
    }
    frames = pd.concat([frames, pd.DataFrame([ball])], ignore_index=True)
    return actions, frames


def _mirror(actions, frames, *, complete: bool):
    """Physically mirror the FRAME. Actions are already LTR so they do not change.

    complete=True  -> positions AND velocities AND labels (the true physical mirror)
    complete=False -> positions only (the historical, incomplete mirror)
    """
    f = frames.copy()
    f["x"] = FL - f["x"]
    f["y"] = FW - f["y"]
    if complete:
        f["vx"] = -f["vx"]
        f["vy"] = -f["vy"]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return actions.copy(), f


def _pressure(a, f):
    out = add_pressure_on_actor(a, frames=f, methods=("bekkers_pi",))
    col = next(c for c in out.columns if c.startswith("pressure_on_actor__bekkers"))
    return float(out.iloc[0][col])


def _dir_labels(f: pd.DataFrame) -> pd.Series:
    """Home 'rtl' / away 'ltr' for player rows, None for ball rows (null direction).

    Built without ``np.where(cond, None, ...)`` -- None is not an ArrayLike branch and
    pyright rejects it. Assign the string labels first, then null the ball rows.
    """
    d = pd.Series(np.where(f["team_id"] == "H", "rtl", "ltr"), index=f.index, dtype=object)
    d[f["is_ball"].astype(bool)] = None
    return d


# --------------------------------------------------------------------------------------
# Task 5: D1 -- bekkers_pi velocity re-projection
# --------------------------------------------------------------------------------------
def test_bekkers_pressure_is_invariant_under_a_complete_physical_mirror():
    a, f = _scenario()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=True)
    assert _pressure(am, fm) == pytest.approx(base, abs=1e-6)


def test_nonvacuity_an_incomplete_mirror_would_be_caught():
    """The guard's discriminating power: a positions-only mirror MUST differ.

    Without this, the invariance test above would pass just as happily on code that
    reflects nothing at all.
    """
    a, f = _scenario()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=False)
    assert abs(_pressure(am, fm) - base) > 1e-3


# --------------------------------------------------------------------------------------
# Task 6: D2 -- re-project the ball row
# --------------------------------------------------------------------------------------
def test_ball_row_is_reprojected_into_action_ltr():
    """The ball must land near the actor for an away action, not at its mirror image."""
    from silly_kicks.tracking.features import _build_ball_xy_v_per_action
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    a, f = _scenario()
    # Make the acting team the AWAY side so the action requires re-projection.
    a["team_id"] = "A"
    a["player_id"] = "a1"
    a["start_x"] = FL - 63.0
    a["start_y"] = FW - 41.0

    ctx = _resolve_action_frame_context(a, f)
    ball = _build_ball_xy_v_per_action(a, f, ctx)

    # Ball sits at (60, 40) in frame coords -> (45, 28) in the away action's LTR frame.
    assert float(ball.iloc[0]["x"]) == pytest.approx(FL - 60.0, abs=1e-6)
    assert float(ball.iloc[0]["y"]) == pytest.approx(FW - 40.0, abs=1e-6)
    assert float(ball.iloc[0]["vx"]) == pytest.approx(-2.0, abs=1e-6)


def _scenario_ball_leads():
    """Ball near the defenders, defenders FAR from the actor.

    Moving the ball alone is not enough: at Task 5's separation (defenders x=63/66 vs actor
    x=60) p_to_actor SATURATES, so np.maximum(p_to_actor, p_to_ball) is won by the actor leg
    at EVERY ball position (measured: identical value at every ball placement). Moving the
    DEFENDERS out to x=92/95 leaves p_to_actor unsaturated so the ball leg can win the max.
    Measured: ball-on-actor 0.0011294381 vs ball-near-defenders 0.9504238642.
    """
    a, f = _scenario()
    f = f.copy()
    f.loc[f["player_id"] == "a1", ["x", "y"]] = [92.0, 40.0]
    f.loc[f["player_id"] == "a2", ["x", "y"]] = [95.0, 38.0]
    is_ball = f["is_ball"].astype(bool)
    f.loc[is_ball, "x"] = 90.0
    f.loc[is_ball, "y"] = 39.0
    return a, f


def _scenario_ball_on_actor():
    """_scenario_ball_leads with the ball moved back onto the actor. Same defenders."""
    a, f = _scenario_ball_leads()
    f = f.copy()
    is_ball = f["is_ball"].astype(bool)
    f.loc[is_ball, "x"] = 60.0
    f.loc[is_ball, "y"] = 40.0
    return a, f


def test_end_to_end_invariance_covers_the_ball_leg():
    a, f = _scenario_ball_leads()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=True)
    assert _pressure(am, fm) == pytest.approx(base, abs=1e-6)


def test_nonvacuity_the_ball_leg_actually_drives_this_fixture():
    """Proves the fixture discriminates: the ball leg must WIN the max here, otherwise
    test_end_to_end_invariance_covers_the_ball_leg is just Task 5's test again.

    Measured: 0.9504238642 (ball near defenders) vs 0.0011294381 (ball on actor).
    """
    ball_ahead = _pressure(*_scenario_ball_leads())
    on_actor = _pressure(*_scenario_ball_on_actor())
    assert abs(ball_ahead - on_actor) > 0.1, (
        "moving the ball did not change pressure -- the ball leg is not driving this fixture, so it cannot guard D2"
    )


def test_nonvacuity_an_incomplete_mirror_is_caught_on_the_ball_fixture():
    """Both-sides partner for the ball fixture specifically.

    Measured POST-fix: complete mirror 0.9504238642 (== base), incomplete mirror
    0.0000000981. PRE-fix the incomplete arm reads 0.0011294381 -- non-zero either way.
    """
    a, f = _scenario_ball_leads()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=False)
    assert abs(_pressure(am, fm) - base) > 1e-3


# --------------------------------------------------------------------------------------
# Task 7: D3/D3b -- play_left_to_right via the registry
# --------------------------------------------------------------------------------------
def test_play_left_to_right_negates_velocity_and_reflects_smoothed_positions():
    from silly_kicks.tracking.utils import play_left_to_right

    _a, f = _scenario()
    f = f.copy()
    f["team_attacking_direction"] = _dir_labels(f)
    f["x_smoothed"] = f["x"] + 0.5
    f["y_smoothed"] = f["y"] + 0.5

    out = play_left_to_right(f, "H")

    assert out.loc[0, "x"] == pytest.approx(FL - 60.0)
    assert out.loc[0, "vx"] == pytest.approx(-2.0)  # D3
    assert out.loc[0, "x_smoothed"] == pytest.approx(FL - 60.5)  # D3b
    assert out.loc[0, "y_smoothed"] == pytest.approx(FW - 40.5)
    # speed is a magnitude -- must NOT change
    assert out.loc[0, "speed"] == pytest.approx(f.loc[0, "speed"])


# --------------------------------------------------------------------------------------
# Task 8: D4 -- finalize_orientation flag leg negates velocity too
# --------------------------------------------------------------------------------------
def test_finalize_orientation_flag_leg_negates_velocity():
    """direction.py:284-289 already negates; the flag leg at :359-360 did not."""
    import silly_kicks.tracking.direction as D

    df = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "frame_id": [1],
            "time_seconds": [0.0],
            "player_id": ["p1"],
            "team_id": ["H"],
            "is_ball": [False],
            "is_goalkeeper": [False],
            "x": [20.0],
            "y": [10.0],
            "vx": [3.0],
            "vy": [-2.0],
            "speed": [np.hypot(3.0, 2.0)],
            "team_attacking_direction": ["ltr"],
        }
    )
    out = D._flip_frames_by_flag(df, np.array([True]))
    assert out.loc[0, "x"] == pytest.approx(FL - 20.0)
    assert out.loc[0, "vx"] == pytest.approx(-3.0)
    assert out.loc[0, "speed"] == pytest.approx(df.loc[0, "speed"])


def test_adapter_schemas_exclude_velocity_so_D4_stays_unreachable():
    """D4 is latent ONLY because the adapter schema projection drops vx/vy.

    The day velocity is added to a *_TRACKING_FRAMES_COLUMNS, D4 goes LIVE and the
    finalize_orientation flag leg starts mattering. Fail loudly then.
    """
    from silly_kicks.tracking.schema import (
        GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
        KLOPPY_TRACKING_FRAMES_COLUMNS,
        SPORTEC_TRACKING_FRAMES_COLUMNS,
        TRACKING_FRAMES_COLUMNS,
    )

    for name, cols in [
        ("TRACKING", TRACKING_FRAMES_COLUMNS),
        ("KLOPPY", KLOPPY_TRACKING_FRAMES_COLUMNS),
        ("SPORTEC", SPORTEC_TRACKING_FRAMES_COLUMNS),
        ("GRADIENTSPORTS", GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS),
    ]:
        assert not ({"vx", "vy"} & set(cols)), (
            f"{name}_TRACKING_FRAMES_COLUMNS now carries velocity. ADR-045 D4 is no longer "
            f"latent: finalize_orientation's flag leg re-projects real velocity data."
        )


# --------------------------------------------------------------------------------------
# Task 12: the two orienters must agree on vector semantics (and be pinned to physics)
# --------------------------------------------------------------------------------------
def test_the_two_orienters_agree_on_vector_semantics():
    """play_left_to_right and orient_frames_to_ltr_by_geometry must transform
    identically. They diverged for the whole life of D3: one negated velocity, the other
    did not."""
    from silly_kicks.tracking.direction import orient_frames_to_ltr_by_geometry
    from silly_kicks.tracking.utils import play_left_to_right

    _a, f = _scenario()
    f = f.copy()
    # Home keeper at HIGH x -> the geometric anchor says this period is mis-oriented,
    # and the label says the same, so both orienters must flip the same rows.
    f.loc[f["player_id"] == "hgk", "x"] = 100.0
    f.loc[f["player_id"] == "agk", "x"] = 5.0
    f["team_attacking_direction"] = _dir_labels(f)

    by_flag = play_left_to_right(f, "H")
    by_geom = orient_frames_to_ltr_by_geometry(f, home_team_id="H")

    for col in ("x", "y", "vx", "vy", "speed"):
        pd.testing.assert_series_equal(
            by_flag[col].reset_index(drop=True),
            by_geom[col].reset_index(drop=True),
            check_names=False,
            obj=f"orienter divergence on {col!r}",
        )


def test_orienters_are_pinned_to_PHYSICS_not_merely_to_each_other():
    """Non-vacuity partner: pins ONE orienter to the physical answer. Together with the
    agreement test above, both are pinned. The fixture is deliberately y-ASYMMETRIC."""
    from silly_kicks.tracking.utils import play_left_to_right

    _a, f = _scenario()
    f = f.copy()
    f.loc[f["player_id"] == "hgk", "x"] = 100.0
    f.loc[f["player_id"] == "agk", "x"] = 5.0
    # y-asymmetric by construction: every row sits off the centre line, and vy is non-zero.
    f["y"] = f["y"] + 12.0
    f.loc[~f["is_ball"].astype(bool), "vy"] = 3.0
    f["speed"] = np.hypot(f["vx"], f["vy"])
    f["team_attacking_direction"] = _dir_labels(f)

    out = play_left_to_right(f, "H")

    for i in range(len(f)):
        assert out.iloc[i]["x"] == pytest.approx(FL - f.iloc[i]["x"])
        assert out.iloc[i]["y"] == pytest.approx(FW - f.iloc[i]["y"])
        assert out.iloc[i]["vx"] == pytest.approx(-f.iloc[i]["vx"])
        assert out.iloc[i]["vy"] == pytest.approx(-f.iloc[i]["vy"])
        assert out.iloc[i]["speed"] == pytest.approx(f.iloc[i]["speed"])  # magnitude
    # Non-vacuity: the fixture must actually be y-asymmetric.
    assert (f["y"] - FW / 2).abs().min() > 1.0, "fixture is y-symmetric -- assertions are vacuous"


# ------------------------------------------------------------------------------------
# Task 12b: call-site conformance guards (sites 1-3, name-based)
# ------------------------------------------------------------------------------------


def test_every_geometry_column_on_the_context_is_enumerated_for_reprojection():
    """A geometry column that reaches the context but is not enumerated at the
    re-projection call site is silently left in frame coordinates -- ADR-045 D1/D2 exactly."""
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    a, f = _scenario()
    # RUN THE REAL PREPROCESS. Without it the fixture carries no x_smoothed/y_smoothed and
    # this guard is blind to exactly the columns it exists to catch -- and its non-vacuity
    # assertion below would be satisfied by the very enumeration it is checking, which is
    # how a guard reports green while covering nothing. derive_velocities REQUIRES the
    # smoothed pair (preprocess/_velocity.py:41), so this ordering is forced.
    f = derive_velocities(smooth_frames(f))

    # defending_gk_rows is EMPTY without this: _resolve_action_frame_context (utils.py:851-855)
    # returns an empty frame when the column is absent, and an `if rows.empty: continue` would
    # then skip a third of the surface this guard claims to cover -- silently.
    a = a.assign(defending_gk_player_id="agk")
    ctx = _resolve_action_frame_context(a, f)

    # What _reproject_rows actually enumerates (keep in sync with utils.py:874).
    enumerated = {"x", "y", "vx", "vy", "x_smoothed", "y_smoothed"}

    for name in ("actor_rows", "opposite_rows_per_action", "defending_gk_rows"):
        rows = getattr(ctx, name)
        assert not rows.empty, (
            f"{name} is empty -- this guard would silently cover only part of its surface. "
            f"Fix the fixture; do not skip the surface."
        )
        geometry = {c for c in rows.columns if _GEOMETRY_COL.match(c)}
        missed = geometry - enumerated
        assert not missed, (
            f"{name} carries geometry column(s) {sorted(missed)} that _reproject_rows does not "
            f"re-project. Either enumerate them at utils.py:874 or declare why they are exempt."
        )
        # Non-vacuity: the fixture must actually exercise the columns we claim to cover.
        # x_smoothed/y_smoothed are the load-bearing half -- they are the ones an earlier
        # draft's fixture lacked, which made this assertion satisfiable by the enumeration
        # it was supposed to be checking.
        assert {"x", "y", "vx", "vy", "x_smoothed", "y_smoothed"} <= geometry, (
            f"{name} fixture does not carry x/y/vx/vy/x_smoothed/y_smoothed -- this guard "
            f"would pass vacuously. Did preprocess actually run on the fixture?"
        )


def test_defensive_line_reprojection_enumerates_every_geometry_column():
    """Site 2 of 4. _kernels.py:879 passes y_cols=[] -- a live assumption that nothing
    lateral is ever added to the defensive-line output. Gate it."""
    from silly_kicks.tracking.features import add_defensive_line

    a, f = _scenario()
    # home_team_id is KEYWORD-ONLY AND REQUIRED (features.py:1186). An earlier draft called
    # add_defensive_line(a, frames=f) and would have died with TypeError before asserting
    # anything.
    out = add_defensive_line(a, frames=f, home_team_id="H")
    added = set(out.columns) - set(a.columns)

    # What _kernels.py:879 enumerates. Keep in sync.
    enumerated = {"defensive_line_x", "back_line_high_x"}
    # compactness_x is a SPAN (a difference of two x values), so it is flip-invariant --
    # documented at _kernels.py:876-877. Derived by measuring the real add_defensive_line
    # output against the pattern, not guessed: the geometry-matching columns are exactly
    # {defensive_line_x, back_line_high_x, compactness_x}.
    exempt = {"compactness_x": "span (difference of x values) -- flip-invariant"}

    geometry = {c for c in added if _GEOMETRY_COL.match(c)} - set(exempt)
    missed = geometry - enumerated
    assert not missed, (
        f"add_defensive_line emits geometry column(s) {sorted(missed)} that _kernels.py:879 "
        f"does not re-project (it passes y_cols=[]). Enumerate them, or add a documented "
        f"exemption."
    )
    # BOTH-SIDES: this guard was unconditionally passing for a full revision cycle because the
    # pattern matched nothing. Prove it sees the real columns.
    assert enumerated <= {c for c in added if _GEOMETRY_COL.match(c)}, (
        "the guard cannot see defensive_line_x / back_line_high_x -- it is vacuous"
    )


def test_finalize_orientation_enumerates_every_geometry_column_it_owns():
    """Site 3 of 4, found in review. finalize_orientation runs on a PRE-canonical frame that
    carries live geometric columns it does not reflect.

    gradientsports.py:121-123 does `out = raw_frames.copy()` then derives x/y from
    x_centered/y_centered, so x_centered/y_centered reach the flip UNREFLECTED. That is benign
    today only because they are dead after the projection at gradientsports.py:147 -- and
    nothing asserts they stay dead. The fix's own boundary would otherwise ship carrying the
    original defect shape.
    """
    import silly_kicks.tracking.direction as D

    df = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "frame_id": [1],
            "time_seconds": [0.0],
            "player_id": ["p1"],
            "team_id": ["H"],
            "is_ball": [False],
            "is_goalkeeper": [False],
            "x": [20.0],
            "y": [10.0],
            "vx": [3.0],
            "vy": [-2.0],
            "speed": [np.hypot(3.0, 2.0)],
            "x_centered": [-32.5],
            "y_centered": [-24.0],  # the adapter scratch columns
            "team_attacking_direction": ["ltr"],
        }
    )
    out = D._flip_frames_by_flag(df, np.array([True]))

    enumerated = {"x", "y", "vx", "vy"}
    # Adapter scratch, exempt ONLY because both adapters project them away before the frame
    # becomes canonical (sportec.py:157, gradientsports.py:147).
    scratch = {"x_centered", "y_centered"}

    # BOTH-SIDES first: this guard existed for a revision cycle while matching NOTHING (the
    # pattern was .match()-anchored, and x_centered is a PREFIX form that the round-two
    # proposed fix would still have missed). Prove it sees them before trusting the result.
    assert {c for c in df.columns if _GEOMETRY_COL.match(c)} >= scratch | enumerated, (
        "the guard cannot see the adapter scratch columns -- it is vacuous"
    )

    unreflected = {c for c in df.columns if _GEOMETRY_COL.match(c) and c not in enumerated and out[c].equals(df[c])}
    assert unreflected <= scratch, (
        f"geometry column(s) {sorted(unreflected - scratch)} pass through "
        f"finalize_orientation unreflected and are not documented as adapter scratch."
    )
    assert scratch <= unreflected, (
        "x_centered/y_centered were reflected -- if that is now intended, update this guard "
        "and confirm the adapters still project them away"
    )
    # The rule this encodes: any frame reaching a reflect_columns call must have every
    # geometry-named column either enumerated or explicitly exempted with a reason.
    # x_centered/y_centered are exempt ONLY because both adapters project them away
    # (sportec.py:157, gradientsports.py:147) before the frame becomes canonical.


# --------------------------------------------------------------------------------------
# Task 14: home-acting actions never enter re-projection (home byte-identical)
# --------------------------------------------------------------------------------------
def test_home_acting_actions_never_enter_reprojection():
    """The fix re-projects only rows whose acting team attacks rtl. Home rows must be
    byte-identical -- expressible today, no golden diff required."""
    a, f = _scenario()
    base = _pressure(a, f)  # home acting -> flip False -> early return

    # Mutate ONLY away-row velocities; the home action's value must not move.
    f2 = f.copy()
    away = f2["team_id"] == "A"
    f2.loc[away, "vx"] = f2.loc[away, "vx"] * 3.0
    assert _pressure(a, f2) != pytest.approx(base, abs=1e-9), (
        "away velocities do not affect this action -- fixture cannot discriminate"
    )

    # And the home-acting path never enters re-projection at all.
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

    assert not bool(acting_team_attacks_rtl(a, f).iloc[0])
