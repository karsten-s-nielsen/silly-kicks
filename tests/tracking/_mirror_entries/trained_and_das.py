"""Trained-model (xS / xCross) + DAS MirrorEntry registrations (ADR-028 section 6).

Three aggregators, three DIFFERENT reasons the mirror does or does not hold -- recorded
per entry rather than smoothed into one tolerance:

``add_das``
    Not silly-kicks geometry at all. The value comes from ``accessible-space``'s polar
    quadrature, which is measurably non-equivariant under a reflection at the shipped
    ``n_angles=30``. Tolerance sized above the measurement, still far below the swap the
    gate exists to catch.
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

# Measured on canonical_scene(); see the per-entry tolerance_basis for the derivation.
_DAS_MIRROR_TOL = 15.0


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
    # add_das -- third-party quadrature, not silly-kicks geometry
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
        basis=(
            "accessible-space integrates over a POLAR grid whose angular quadrature is "
            "NOT periodic: core.py:626/630 set the first ray's lower bound and the last "
            "ray's upper bound to the ray itself, so rays 0 and n-1 each receive a "
            "half-width wedge. A point reflection maps ray k -> ray (k+15) mod 30 at the "
            "shipped n_angles=30, landing the two deficient wedges on different rays. "
            "Measured on canonical_scene(): 12.0349 (das_team/das_opponent) and 11.9761 "
            "(das_diff), bit-reproducible across repeat runs. Diagnosis confirmed by "
            "refinement -- the base-vs-mirror gap collapses 12.03 -> 3.43 -> 1.13 at "
            "n_angles 30 -> 120 -> 480 while the value itself converges 47.5 -> 91.1 -> "
            "94.8, the signature of a quadrature artifact rather than a convention defect. "
            "15.0 keeps ~25% headroom over the measurement and stays 3.1x below the ~46.9 "
            "a team-attribution swap produces on this fixture, so the defect class this "
            "gate exists to catch is still detectable."
        ),
        role="unused",  # signature takes no home_team_id at all
        non_vacuity=("das_team", "das_diff"),
        exempt=(
            {
                "das_source": (
                    "closed provenance vocabulary (ADR-043 DAS_SOURCE_VALUES), a string token "
                    "rather than geometry; 'computed' on every row of both legs"
                )
            }
        ),
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
