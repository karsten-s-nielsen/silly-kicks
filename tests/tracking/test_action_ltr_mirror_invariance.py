"""ADR-028 durable guard: emitted action-LTR geometry is invariant under a physical
left/right mirror of the frame + action.

A situation that physically happens near the absolute-left goal must yield the SAME
action-LTR feature values as its mirror near the absolute-right goal. Any seam that
leaks frame (home-attacks-right) orientation into a per-action position output breaks
this invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_defensive_line, add_ghost_gk, add_obso, add_team_shape

# ADR-041 opt-out: the OBSO mirror guard below exercises the synthetic placeholder EPV
# path deliberately -- it asserts ORIENTATION invariance, not EPV provenance.
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
FL, FW = 105.0, 68.0
# ghost_gk_y mirror-invariance tolerance: sized to the corrected 179-match model's inherent lateral
# asymmetry on the off-centre probe (measured 1.26 m), with headroom. See test_ghost_gk_mirror_invariant.
_GHOST_Y_TOL = 3.0
# OBSO mirror tolerance: bounded by PITCH CONTROL's own mirror asymmetry, NOT by orientation
# and NOT (any longer) by grid discretization.
#
# History, because two earlier attributions were wrong and the measurements are the point:
#   1. First attributed to the int()/FLOOR target indexing in compute_pass_obso (x=15 -> 14 while
#      its mirror x=90 -> 88, and the mirror of 14 is 89). That off-by-one was real and IS now
#      fixed (floor -> round, ADR-041), together with a half-cell registration error on the xt=
#      path -- but fixing both did NOT collapse this tolerance, which falsified that attribution.
#   2. Measured cause: pitch control is itself not mirror-symmetric at these query points.
#      compute_pitch_control(attacking_team_id=2) returns EXACTLY 0.5000000000 at frame (90, 34)
#      -- the documented Spearman degenerate/no-information fallback -- and 1.0000000000 at the
#      mirrored (15, 34), a clean 2x. The surface is not equal to its own mirror under either an
#      x-flip or a point reflection.
# So this bound is a property of the pitch-control model, is PRE-EXISTING, and is out of scope
# for an OBSO-orientation PR. Max measured base-vs-mirror difference on this fixture: 1.30e-2.
# 0.02 leaves headroom while a genuine orientation leak moves the value by >= 0.1 (5x the tol);
# the real orientation guards are the dedicated RED-verified tests in test_obso_orientation.py.
_OBSO_MIRROR_TOL = 0.02


def _scenario():
    """Away team shoots toward x=105 (away action, frame is home-attacks-right)."""
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=100,
        time_seconds=4.0,
        frame_rate=25.0,
        z=0.0,
        speed=2.0616,
        speed_source="native",
        # vx/vy are REQUIRED alongside speed_source="native": declaring velocity available while
        # omitting the columns is the "forgot derive_velocities()" case, which now RAISES. Before
        # the ghost velocity refusal these fixtures reached the model with 5 of 26 features NaN
        # and asserted a geometric property of the HGBR's IMPUTED output.
        #
        # NON-ZERO deliberately, and `speed` matches the vector. A fully stationary 22-player frame
        # is out-of-domain for a model fit on real matches: measured, vx=vy=0 inflates the ghost
        # mirror asymmetry to 3.73 m (vs 1.26 m recorded) and trips _GHOST_Y_TOL, while any
        # realistic velocity passes. Zeroing vx/vy is a NAMED fixture defect here -- CLAUDE.md
        # records it as one of two that made the xS liveness gate score noise for three cycles.
        vx=2.0,
        vy=0.5,
        ball_state="alive",
        confidence=None,
        visibility=None,
        source_provider="synthetic",
        is_goalkeeper_source="native",
    )
    rows = [
        dict(
            player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=4.0, y=31.0, team_attacking_direction="ltr"
        ),
        dict(
            player_id=50,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=True,
            x=101.0,
            y=37.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=11,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=20.0,
            y=30.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=12,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=24.0,
            y=44.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=13,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=16.0,
            y=20.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=61,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=15.0,
            y=30.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=62,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=18.0,
            y=40.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=13.0,
            y=34.0,
            team_attacking_direction=None,
        ),
    ]
    frames = pd.DataFrame([{**base, **r} for r in rows])
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=HOME,
                player_id=1.0,
                type_id=GOALKICK,
                result_id=1,
                start_x=5.0,
                start_y=31.0,
                end_x=40.0,
                end_y=34.0,
                time_seconds=3.6,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=AWAY,
                player_id=61.0,
                type_id=SHOT,
                result_id=1,
                start_x=92.0,
                start_y=38.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=4.0,
            ),
        ]
    )
    return actions, frames


def _mirror(actions, frames):
    """Physical left/right mirror: flip all frame x/y and swap team_attacking_direction.

    The action coordinates are LEFT UNCHANGED on purpose: SPADL actions are already
    LTR-normalized (the acting team attacks x=105 regardless of physical orientation),
    so they are invariant under a physical mirror. Only the frame coordinate system and
    the per-team attacking direction flip. The emitted action-LTR geometry must therefore
    be identical between the two.
    """
    f = frames.copy()
    f["x"] = FL - f["x"]
    f["y"] = FW - f["y"]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return actions.copy(), f


def _assert_invariant(base, mir, aid, cols, *, tol=1e-6):
    base = base.set_index("action_id")
    mir = mir.set_index("action_id")
    for col in cols:
        b, m = base.loc[aid, col], mir.loc[aid, col]
        assert (pd.isna(b) and pd.isna(m)) or abs(b - m) < tol, f"{col}: {b} vs {m}"


def test_pre_shot_gk_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_pre_shot_gk_context(a, frames=f)
    mir = add_pre_shot_gk_context(am, frames=fm)
    _assert_invariant(
        base,
        mir,
        1,
        [
            "pre_shot_gk_x",
            "pre_shot_gk_y",
            "pre_shot_gk_distance_to_goal",
            "pre_shot_gk_distance_to_shot",
        ],
    )


# NOTE on home_team_id in the mirrored call: the orientation-aware compute functions
# (compute_defensive_line) take home_team_id meaning "the team that attacks RIGHT in
# these frames" (= home for real convert_to_frames output, which is always
# home-attacks-right). After _mirror, the team attacking right is AWAY, so the mirrored
# call passes home_team_id=AWAY. The per-action re-projection itself reads
# team_attacking_direction and needs no home_team_id.


def test_defensive_line_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_defensive_line(a, f)
    mir = add_defensive_line(am, fm)
    _assert_invariant(base, mir, 1, ["defensive_line_x", "back_line_high_x", "compactness_x"])


def test_team_shape_centroids_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_team_shape(a, f)
    mir = add_team_shape(am, fm)
    # ADR-028: compute_team_shape is now orientation-aware (deepest line nearest the
    # defended goal), so defensive_line_height is the team's true defensive line and is
    # mirror-invariant for BOTH teams alongside the centroids/spans.
    _assert_invariant(
        base,
        mir,
        1,
        [
            "team_shape_centroid_x_attacking",
            "team_shape_centroid_y_attacking",
            "team_shape_centroid_x_defending",
            "team_shape_centroid_y_defending",
            "team_shape_team_length_attacking",
            "team_shape_convex_hull_area_attacking",
            "team_shape_defensive_line_height_attacking",
            "team_shape_defensive_line_height_defending",
        ],
    )


def _ghost_scenario():
    """A deliberately OFF-CENTRE variant of _scenario() for the ghost mirror test.

    The shared _scenario() puts the ball and the attack on the y=34 centre line, where the
    ghost model correctly predicts a near-centre keeper for BOTH a config and its mirror --
    so the y-axis carries almost no signal and a y-reprojection flip would move ghost_gk_y by
    only ~0.1 m (invisible under any sane tolerance). This scenario lateralizes the attack low
    (ball + away attackers around y=14-20, still inside the model's trained goal-relative box
    y in [18,50]) so the model predicts a CLEARLY off-centre ghost_gk_y. That turns the y-axis
    into a real guard: a flip now moves y by ~7 m (the non-vacuity assertion below), which no
    tolerance sized to the model's genuine asymmetry could mask.
    """
    a, f = _scenario()
    f = f.copy()
    # Lateralize the ball and the away attackers (players 61/62) low; nudge the home GK + line low
    # too. All goal-relative y stay >= 14 (attack) / keeper prediction stays inside [18,50].
    y_moves = {1: 20.0, 11: 19.0, 12: 24.0, 13: 14.0, 61: 14.0, 62: 20.0}  # player_id -> new frame y
    for pid, new_y in y_moves.items():
        f.loc[f["player_id"] == pid, "y"] = new_y
    f.loc[f["is_ball"], "y"] = 16.0
    a = a.copy()
    a.loc[a["action_id"] == 0, "start_y"] = 20.0  # goalkick from the (now low) keeper
    a.loc[a["action_id"] == 1, "start_y"] = 15.0  # the shot develops from the low flank
    return a, f


def test_ghost_gk_mirror_invariant():
    # Pre-load the model ONCE and share it across both calls (avoids a double ~18s load)
    # and locks the asymmetric ghost transform (x uniform, y per-action flip).
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    model = GhostGkModel.from_variant("default")
    a, f = _ghost_scenario()  # OFF-CENTRE (see _ghost_scenario) so the y-axis is a real guard
    am, fm = _mirror(a, f)
    base = add_ghost_gk(a, f, home_team_id=HOME, model=model).set_index("action_id")
    mir = add_ghost_gk(am, fm, home_team_id=AWAY, model=model).set_index("action_id")

    def _g(df: pd.DataFrame, col: str) -> float:
        return float(df.loc[1, col])  # type: ignore[arg-type]

    bx, by = _g(base, "ghost_gk_x"), _g(base, "ghost_gk_y")
    mx, my = _g(mir, "ghost_gk_x"), _g(mir, "ghost_gk_y")

    # (1) ORIENTATION GUARD -- the DURABLE, model-independent correctness check. Both mirrors emit
    # ghost_gk_x at the ATTACKED goal (x=105); a gross orientation leak puts x ~90 m away
    # (goal-relative ~13 vs action-LTR ~101). x is the axis that actually catches an orientation bug.
    assert bx > 95.0 and mx > 95.0, (bx, mx)
    assert abs(bx - mx) < 0.5, (bx, mx)

    # (2) NON-VACUITY -- the strengthening this fixture exists for. Off-centre, the model predicts a
    # clearly non-central keeper, so a y-reprojection FLIP (emitting 68-my instead of my) would move
    # y by ~7 m. The old central probe could NOT catch such a flip (it moved y by only ~0.1 m < the
    # 0.5 m tol -- the y check was vacuous). The 5.0 m floor is comfortably above _GHOST_Y_TOL (3.0 m),
    # so a flip would provably trip assertion (3); the floor is absolute (decoupled from the tol) so a
    # future tol bump can't silently re-vacuate this guard.
    flip_dy = abs(by - (FW - my))
    assert flip_dy > 5.0, f"probe not discriminating on y: a y-flip would move only {flip_dy:.2f} m"

    # (3) MIRROR INVARIANCE (y) -- base ~ mir within the CORRECTED 179-match model's inherent lateral
    # asymmetry on a lateralized probe. Measured 1.26 m (old 81-match: 0.20 m on the central probe;
    # the y-response, and its asymmetry, both scale with the off-centre signal). This is a soft
    # model-symmetry bound with churn headroom -- the real orientation guards are (1) and (2). If a
    # future refit pushes this past the tol while (1)+(2) still hold, re-measure and bump the tol; it
    # is model asymmetry, not an orientation leak.
    assert abs(by - my) < _GHOST_Y_TOL, f"ghost_gk_y asymmetry {abs(by - my):.3f} m exceeds {_GHOST_Y_TOL} m"


def test_obso_mirror_invariant():
    """OBSO joins this gate as of ADR-041 (DEFECT A).

    ADR-028 had classified obso as "self-reconciling". It was not: it read the raw
    action-LTR target against home-attacks-right pitch-control surfaces and applied an
    always-+x EPV grid, so away actions were sampled at the reflected point AND valued
    toward their own goal. With the per-action re-projection in place the emitted values
    must be identical under a physical mirror.

    Uses the away-team OBSO fixture (a pass action with a real frame window) rather than
    this module's single-frame shot scenario, which carries no pass for OBSO to value.
    """
    from tests.tracking.test_obso_orientation import _away_actions, _away_control_at_low_x

    # The low-x variant: the away team holds control at frame x~15, which is where action
    # 10's action-LTR target (90, 34) re-projects to. On the spread-out variant that cell
    # is empty and obso_actual is 0.0 -- the invariance would hold vacuously.
    a, f = _away_actions(), _away_control_at_low_x()
    am, fm = _mirror(a, f)
    base = add_obso(a, f)
    # After _mirror the team attacking RIGHT is AWAY (see the NOTE above).
    mir = add_obso(am, fm)

    cols = ["obso_actual", "obso_peak", "obso_optimal"]
    # Non-vacuity: the fixture must actually produce values, or the invariance is trivial.
    signal = float(base.set_index("action_id").iloc[0]["obso_actual"])
    assert signal > 0.05, f"OBSO signal {signal:.4g} too small for the tolerance to mean anything"
    for aid in (10, 11):
        _assert_invariant(base, mir, aid, cols, tol=_OBSO_MIRROR_TOL)


# Task 12b site 4: _reproject_team_shape behavioural gate (ADR-045)
def test_team_shape_reprojection_is_mirror_invariant_over_ALL_columns():
    """Site 4 of 4. _reproject_team_shape (features.py:2026) hand-enumerates
    _TEAM_SHAPE_X_COLS / _TEAM_SHAPE_Y_COLS. GEOMETRIC_NAME CANNOT SEE these infix names
    (`team_shape_centroid_x_attacking` -> .match() is False -- measured), so unlike sites
    1-3 this gate must be BEHAVIOURAL, not name-based: under a physical mirror every emitted
    team-shape column must be invariant in action-LTR. Auto-discovering `added` (not a hand
    list) is the anti-rot half -- a FUTURE lateral column that _reproject_team_shape forgets
    to enumerate would break this without any name signal.

    Uses _ghost_scenario (attack lateralised low), NOT _scenario. Measured 2026-07-20: on
    _scenario the acting-team centroid_y sits ~1 m off the centre line, so 68-y is a near
    identity and the y-reflection is UNTESTED -- the pre-existing
    test_team_shape_centroids_mirror_invariant is vacuous on the y-axis (disabling
    _TEAM_SHAPE_Y_COLS leaves its assertions green). _ghost_scenario's action 1 is an AWAY
    action (flip=True) with centroid_y ~= 51 (17 m off centre), so the y-axis carries real
    signal. The both-sides partner below proves it.
    """
    from silly_kicks.tracking.features import add_team_shape

    a, f = _ghost_scenario()
    am, fm = _mirror(a, f)
    base = add_team_shape(a, f)  # action 1: away, flip=True (reprojects)
    mir = add_team_shape(am, fm)  # action 1: home, flip=False (raw)

    # NON-VACUITY: the acting-team centroid must be genuinely off the centre line, or the
    # y-axis is untested exactly as the pre-existing test is. Measured base value ~= 51.
    b1 = base[base["action_id"] == 1].iloc[0]
    assert abs(float(b1["team_shape_centroid_y_attacking"]) - FW / 2) > 3.0, (
        "acting centroid_y is within 3 m of the centre line -- the y-reflection is not "
        "exercised (this is the vacuity measured in test_team_shape_centroids_mirror_invariant)"
    )

    # ANTI-ROT: EVERY added column invariant under the mirror. No name pattern. A lateral
    # column riding through _reproject_team_shape unreflected differs between the flip=True
    # (base) and flip=False (mir) representations of the same physical scene.
    #
    # NA-SAFE by necessity: team-shape emits nullable columns (a degenerate hull / absent
    # second inter-line gap on this fixture is pd.NA, not np.nan). `pd.isna` covers BOTH; a
    # bare `np.isnan` raises TypeError on pd.NA, and `pd.NA == pytest.approx(...)` raises
    # "boolean value of NA is ambiguous" -- measured 2026-07-20, this is a real crash, not a
    # style note. Two shared-NA columns are skipped; 22 numeric columns are checked.
    m1 = mir[mir["action_id"] == 1].iloc[0]
    added = sorted(set(base.columns) - set(a.columns))
    checked = 0
    for col in added:
        bv, mv = b1[col], m1[col]
        if pd.isna(bv) or pd.isna(mv):
            assert pd.isna(bv) and pd.isna(mv), (
                f"team-shape column {col!r} is NA on one side only (base={bv}, mir={mv}) -- "
                f"a mirror should not create or destroy a value"
            )
            continue
        assert float(bv) == pytest.approx(float(mv), abs=1e-6), (
            f"team-shape column {col!r} is not mirror-invariant (base={bv}, mir={mv}) -- a "
            f"lateral quantity is riding through _reproject_team_shape unreflected"
        )
        checked += 1
    assert checked >= 20, f"only {checked} columns actually compared -- fixture may be degenerate"


def test_team_shape_gate_fails_when_the_y_reprojection_is_disabled():
    """BOTH-SIDES partner for site 4, and the executable record of the vacuity finding.
    Disabling _TEAM_SHAPE_Y_COLS must BREAK the mirror-invariance above; if it does not, the
    scenario is not y-asymmetric enough and the gate is vacuous. Measured: the ON delta is 0
    and the OFF delta is ~34 on _ghost_scenario action 1."""
    import silly_kicks.tracking.features as _F
    from silly_kicks.tracking.features import add_team_shape

    a, f = _ghost_scenario()
    am, fm = _mirror(a, f)
    orig = _F._TEAM_SHAPE_Y_COLS
    try:
        _F._TEAM_SHAPE_Y_COLS = []  # a y re-projection that never happens
        base = add_team_shape(a, f)
        mir = add_team_shape(am, fm)
    finally:
        _F._TEAM_SHAPE_Y_COLS = orig
    b = float(base[base["action_id"] == 1].iloc[0]["team_shape_centroid_y_attacking"])
    m = float(mir[mir["action_id"] == 1].iloc[0]["team_shape_centroid_y_attacking"])
    assert abs(b - m) > 1.0, (
        "disabling the y re-projection did not break mirror-invariance -- the fixture is not "
        "y-asymmetric enough, so the site-4 invariance gate is vacuous on the y-axis"
    )
