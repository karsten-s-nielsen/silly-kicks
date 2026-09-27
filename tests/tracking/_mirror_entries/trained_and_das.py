"""Trained-model (xS / xCross) + DAS MirrorEntry registrations (ADR-028 section 6).

Three aggregators, three DIFFERENT reasons the mirror does or does not hold -- recorded
per entry rather than smoothed into one tolerance:

``add_das``
    Now silly-kicks' OWN native engine (ADR-107), not a third-party dependency. Its angular
    quadrature is PERIODIC (ADR-108), so a point reflection permutes rays without changing the
    integral and DAS is mirror-invariant in real arithmetic. The only residual is the danger
    term's ``arccos`` opening angle, whose derivative blows up at the goal mouth, so the entry
    uses a RELATIVE ``rtol`` (spec 6.4 F1) rather than the old absolute delta that sized around
    ``accessible-space``'s non-equivariant polar grid. It also joins Gate C now that direction
    comes from a ``GoalMap`` (ADR-055) the entry can vary.
``add_xshot_occurrence`` / ``add_xcross_attempt``
    Mirror-invariant as of PR 5, at the exact tolerance. They were NOT, and the cause was
    silly-kicks' own goal-relative transform -- see the resolved note below.

RESOLVED IN PR 5 (was: "finding, deliberately NOT xfail-ed"; then a strict xfail on both
entries through PRs 1-4). ``_geometry.py`` had ``to_goal_relative_x``/``_vx`` and no
``to_goal_relative_y``, so ``goal_x == 105`` mapped ``(x, y) -> (105 - x, y)`` -- determinant
-1 -- while ``goal_x == 0`` was the identity: the two ends used frames of OPPOSITE handedness.
Composed with ADR-028's point reflection that left every radial byte-identical and NEGATED
every bearing:

* xS  -- 12 of 27 features flipped sign (``theta``, ``GK_theta``, ``OffAngle_0..4``,
  ``DefAngle_0..4``); model output 0.01113238 -> 0.01293222 (+16.2%).
* xCross -- 3 of 16 flipped (``gk_theta``, ``ball_theta``, ``gk_lateral_offset``);
  model output 0.00168395 -> 0.00112845 (-33.0%).

Those are exactly the "12/27 and 3/16 sign-inconsistent" counts ADR-037 records, reached here
from the ADR-028 side: the property that made the pre-4.51.0 weights chirality-mis-served ALSO
meant one physical scene scored differently depending which end the acting team attacked.

PR 5 made the transform the 180-degree point reflection, so both entries now hold WITHOUT a
defect marker and the markers were deleted with the fix (strict xfail: an XPASS fails the
build, so they could not have survived it). The live invariant is gated directly, per feature
rather than per aggregate, in ``tests/tracking/test_pr5_chirality_gates.py``.
"""

from __future__ import annotations

# RELATIVE rtol (spec 6.4 F1), NOT an absolute delta: the native periodic quadrature (ADR-108)
# is mirror-invariant to the arccos float floor, which scales WITH the DAS magnitude. See the
# per-entry tolerance_basis for the derivation and the measured residuals.
_DAS_MIRROR_TOL = 1e-6


def _with_possession(frames):
    """``frames`` + ``team_in_possession`` -- the documented DAS caller prerequisite.

    ``add_das`` raises without it ("Call derive_team_in_possession(frames, carrier_df)"),
    and ``canonical_scene()`` is raw converter shape. Derived INSIDE the entry so each leg
    derives from its own frames; verified to resolve the same carrier (player 11, team 1)
    and the same possessing team in both legs, so the mirror comparison is not confounded
    by a possession flip.
    """
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    return derive_team_in_possession(frames, infer_ball_carrier(frames))


def _matched_team_id_dtype(actions, frames):
    """Cast ``actions.team_id`` to the frames' dtype before calling xS.

    NOT cosmetic, and NOT a fix for the aggregator: ``add_xshot_occurrence`` joins its
    scored frames on ``team_id`` and, when the two dtypes differ, coerces BOTH sides with a
    raw ``.astype(str)`` (``_xshot_occurrence.py:955-957``). ``canonical_scene()`` has int64
    action ids against float64 frame ids (the ball row's NaN team upcasts the column), so
    that coercion yields ``"1"`` vs ``"1.0"`` -- the ADR-019 failure mode -- and EVERY row
    comes back NaN. Its structural twin ``add_xcross_attempt`` routes the same join through
    ``align_join_keys`` and is unaffected.

    Handing xS matched dtypes restores the same-provider assumption its own score-lookup
    docstring records ("Assumes actions and frames share team ID type from the same
    provider"), so the mirror gate measures the mirror property instead of re-measuring a
    dtype defect that belongs to the ADR-019 gate. The defect is REPORTED, not papered over.

    Applied to xS ONLY. xCross must not get it: ``_build_score_lookup`` compares team ids
    with ``str()``, so a float64 ``team_id`` would make ``str(1.0) != str(1)`` and silently
    destroy its score attribution -- the thing Gate B exists to observe.
    """
    if actions["team_id"].dtype != frames["team_id"].dtype:
        actions = actions.assign(team_id=actions["team_id"].astype(frames["team_id"].dtype))
    return actions


def register() -> None:
    from silly_kicks.tracking import add_das, add_xcross_attempt, add_xshot_occurrence
    from tests.tracking._mirror_registry import _entry

    # ------------------------------------------------------------------
    # add_das -- native periodic quadrature (ADR-107/108), mirror-invariant
    # ------------------------------------------------------------------
    _entry(
        "add_das",
        lambda a, f, _h: add_das(a, _with_possession(f)),
        {
            "das_team": "invariant",
            "das_opponent": "invariant",
            "das_diff": "invariant",
            "das_source": "exempt",
        },
        tol=_DAS_MIRROR_TOL,
        relative_tolerance=True,  # rtol, not absolute delta -- the arccos floor scales with the value
        basis=(
            "The native DAS engine (ADR-107) reflects EXACTLY in real arithmetic: its angular "
            "quadrature is PERIODIC (ADR-108), so ray k and ray (k + n/2) carry equal wedges and a "
            "point reflection permutes rays without changing the integral. The one residual is the "
            "danger term's arccos opening angle, whose derivative -> inf at the goal mouth, so a tiny "
            "float difference in its argument amplifies to ~1e-7 RELATIVE (spec 6.4 F1; measured max "
            "1.4e-7 rel on golden scene S10). This is a RELATIVE rtol, not the old absolute delta: the "
            "floor scales with the DAS magnitude, so a fixed absolute number would be tight on a large "
            "scene and slack on a small one. rtol=1e-6 is ~7x the measured floor. On canonical_scene() "
            "the residual is 8.5e-13 absolute / 1.5e-15 relative (machine epsilon -- this scene carries "
            "no near-goal-mouth danger term), ~5e5x below the ~550 team-attribution swap the gate exists "
            "to catch (a map flip moves das_team/das_opponent/das_diff by 504/553/552 = ~1.0 relative). "
            "The arccos floor is INHERENT to accessible-space reference parity -- a stable atan2 opening "
            "angle would drop it to ~1e-12 but DIVERGE from the reference, breaking the 1e-12 parity gate "
            "-- so it is a floor, not a defect to fix. Discrimination is retained at engine level in "
            "test_das_quadrature.py (an interior perturbation above the tol fails)."
        ),
        role="unused",  # signature takes no home_team_id at all; direction now comes from the GoalMap
        non_vacuity=("das_team", "das_diff"),
        exempt=(
            {
                "das_source": (
                    "closed provenance vocabulary (ADR-043 DAS_SOURCE_VALUES), a string token "
                    "rather than geometry; 'computed' on every row of both legs"
                )
            }
        ),
        # Gate C (spec 6.9): direction is a GoalMap (ADR-055), so swapping the map must move DAS.
        # All three numeric columns move (measured 504/553/552 on canonical_scene) -- declared EXACTLY,
        # the completeness gate rejects a hand-picked subset.
        call_with_map=lambda a, f, gm: add_das(a, _with_possession(f), goal_map=gm),
        gate_c_must_move=("das_team", "das_opponent", "das_diff"),
    )

    # ------------------------------------------------------------------
    # add_xshot_occurrence -- FINDING: not mirror-invariant (goal-relative chirality)
    # ------------------------------------------------------------------
    # role: the signature DOES take home_team_id, but every occurrence in
    # _xshot_occurrence.py is a pass-through annotated "unused (goal resolved GK-based);
    # kept for call symmetry" and nothing reads it. Declared "attribution" rather than
    # "unused" so Gate B still RUNS -- it then serves as the D3 dead-parameter proof by
    # output identity across {HOME, AWAY, 999999}, which "unused" would only skip.
    _entry(
        "add_xshot_occurrence",
        lambda a, f, h: add_xshot_occurrence(_matched_team_id_dtype(a, f), f, home_team_id=h),
        {"xshot_occurrence": "invariant"},
        tol=1e-9,
        basis=(
            "A shot probability is a scalar with no orientation of its own, and every "
            "input feature is documented goal-relative, so the exact-arithmetic "
            "expectation is bit-equality. Deliberately NOT loosened to cover the measured "
            "1.7998e-3 (0.01113238 -> 0.01293222, +16.2%): that gap is caused by "
            "silly-kicks' own x-only goal-relative transform negating all 12 bearing "
            "features, which is a finding for this cycle, not a numerical artifact to "
            "absorb. See the module docstring."
        ),
        role="attribution",
        # EMPTY BY MEASUREMENT, not by omission: xS scores only the in-possession team (S1),
        # and canonical_scene()'s carrier is player 11 (team HOME, 2.83 m from the ball)
        # in every frame -- the nearest AWAY player is player 63 at 19.72 m -- so
        # xshot_occurrence is NaN on the away rows structurally, for any home_team_id and
        # in both legs. Gate A therefore compares the HOME rows only; the away-row leg of
        # this gate is UNTESTABLE for xS on this fixture and is reported rather than faked.
        non_vacuity=(),
        # DEFERRED TO PR 5 (spec section 8b), not fixed here. `_geometry.py` has no
        # `to_goal_relative_y`, so `goal_x=105` is an x-only MIRROR (det -1) while `goal_x=0` is
        # the identity (det +1): opposite handedness, so every BEARING negates while every RADIAL
        # feature is byte-identical. 12 of 27 xS features flip sign; output 0.01113 -> 0.01293.
        # Cannot ride in this cycle: the artifact carries chirality AND feature_contract stamps,
        # both fail-closed, so the fix, the retrain and the re-stamp are ATOMIC.
    )

    # ------------------------------------------------------------------
    # add_xcross_attempt -- FINDING on BOTH gates, two distinct causes
    # ------------------------------------------------------------------
    _entry(
        "add_xcross_attempt",
        lambda a, f, h: add_xcross_attempt(a, f, home_team_id=h),
        {"xcross_attempt": "invariant"},
        tol=1e-9,
        basis=(
            "Same reasoning as xS: a cross-propensity probability carries no orientation "
            "and its features are documented goal-relative, so bit-equality is the exact "
            "expectation. Gate A measures 5.5550e-4 (0.00168395 -> 0.00112845, -33.0%), "
            "isolated to the 3 sign-flipping bearing features plus space_controlled "
            "(328.17 -> 310.43) -- reproduced with home_team_id HELD at HOME in both legs, "
            "so it is chirality, not the gate's home_team_id swap. Gate B measures a "
            "SEPARATE 3.1619e-4 on the nonsense id alone. Neither is loosened away."
        ),
        # TRUE, and load-bearing: _xcross_attempt.py:297 records "home_team_id is USED to
        # sign score_differential (PA-H1)" and :349 applies the sign. With HOME or AWAY the
        # canonical scene is 1-1 so the differential is 0 and the sign is invisible; the
        # NONSENSE id attributes both goals to "away", moving the model input 0 -> -2 and
        # the output 0.00168395 -> 0.00200014. That is attribution, not direction.
        role="attribution",
        # Same structural reason as xS: xCross scores the in-possession team only, and
        # possession is HOME throughout canonical_scene().
        non_vacuity=(),
        # Gate B does not apply to this column. Its dependence on home_team_id is genuine
        # ATTRIBUTION -- the score is conditioned on score_differential, whose SIGN is a match
        # fact, not geometry (_xcross_attempt.py:297 / :349). Gate B's contract is that
        # action-LTR GEOMETRY cannot depend on which team is home; a model score conditioned on
        # match state legitimately can.
        #
        # This is the entry that exposed a gap in the gate itself: xcross_attempt is its ONLY
        # numeric column, so before `gate_b_exempt` existed there was no way to express this --
        # exempting the column tripped Gate B's own `assert checked > 0`. The vocabulary had
        # conflated two axes, treating the `invariant` MIRROR class as also naming Gate B's
        # surface. A column can be mirror-invariant AND legitimately identity-dependent.
        gate_b_exempt={
            "xcross_attempt": (
                "home_team_id signs score_differential (a match fact, not geometry): the nonsense "
                "id attributes both goals to 'away', moving the model input 0 -> -2 and the "
                "output 0.00168395 -> 0.00200014. Attribution, not direction-keying."
            )
        },
        # DEFERRED TO PR 5 (spec section 8b), same root cause as xS plus one of its own: 3 of 16
        # xCross features flip sign (gk_theta, ball_theta, gk_lateral_offset), and
        # `_dominant_region_area`'s y grid `arange(1.5, 68.0, 3.0)` centres on 34.5 rather than
        # 34.0 (x is fine -- 105/3 tiles exactly), so a y-mirror maps cells off-grid and
        # space_controlled moves 328.17 -> 310.43. Both ride in PR 5 together because
        # space_controlled is xCross model feature #3: splitting them retrains the same model twice.
    )
