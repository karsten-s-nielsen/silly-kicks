"""Context-and-window ``MirrorEntry`` registrations (ADR-028 section 6).

Four aggregators, all sharing the ``_resolve_action_frame_context`` seam that ADR-028 repaired:
``add_action_context``, ``add_actor_pre_window``, ``add_pressure_on_actor``,
``add_press_commitment``. NONE of the four takes ``home_team_id`` (verified by
``inspect.signature``), so every entry is ``role="unused"`` and Gate B skips them -- direction
cannot be identity-keyed in code that never sees the identity.

**Why several entries pass NON-DEFAULT parameters.** On ``canonical_scene()`` the shipped defaults
drive four of these columns to a constant (``receiver_zone_density`` 0 at ``receiver_zone_radius=5``
because the nearest opponent to any pass destination is >5 m away; ``link_zones`` 0 because its
widest zone is 4 m and the nearest defender is 4.24 m; ``bekkers_pi`` 0 because its
``speed_threshold=2.0`` active-pressing filter rejects every player in a fixture where all speeds
are 1.0 m/s; ``press_commitment`` all-NaN because ``press_max_distance_m=3.0`` and the nearest
defender is 4.24 m). A column that is constant-0 or all-NaN in BOTH legs agrees trivially, so the
defaults would have registered four green comparisons that assert nothing. Each parameter below is
widened to the smallest value measured to reach the players the fixture actually contains, and the
scene itself is untouched -- the fixture author's constant positions are a recorded decision
("A positional drift would desynchronise the action anchors ... for no gain"), not an oversight,
and an entry has no business overriding it.

Instrument non-vacuity was checked with a plant rather than assumed: a deliberately
convention-mixing nearest-defender (actor read from the action in action-LTR, defenders read from
the frame un-reprojected -- the RC defect class) moves **24.72 m** between the two legs on this
fixture. The zero deltas recorded below are therefore measurements, not a degenerate scene.
"""

from __future__ import annotations


def register() -> None:
    from silly_kicks.tracking._press_commitment import PressCommitmentParams
    from silly_kicks.tracking.features import (
        add_action_context,
        add_actor_pre_window,
        add_press_commitment,
        add_pressure_on_actor,
    )
    from silly_kicks.tracking.pressure import BekkersParams, LinkParams
    from tests.tracking._mirror_registry import _entry

    # Linkage provenance is emitted by all four; the reason is identical, so it is written once.
    _PROVENANCE = ("frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score")
    _PROVENANCE_CLASSES = dict.fromkeys(_PROVENANCE, "exempt")
    _PROVENANCE_REASONS = dict.fromkeys(_PROVENANCE, "linkage provenance, not geometry")

    # ------------------------------------------------------------------ add_action_context
    # receiver_zone_radius 5.0 -> 15.0: at the default the density is 0 on ALL four rows, so the
    # one column here that samples FRAME positions around an ACTION-LTR destination -- i.e. the
    # column most exposed to convention mixing -- would compare 0 against 0. At 15.0 it reads
    # [0, 1, 3, 1], live on both away rows.
    _entry(
        "add_action_context",
        lambda a, f, h: add_action_context(a, f, receiver_zone_radius=15.0),
        {
            "nearest_defender_distance": "invariant",
            "actor_speed": "invariant",
            "receiver_zone_density": "invariant",
            "defenders_in_triangle_to_goal": "invariant",
            **_PROVENANCE_CLASSES,
        },
        tol=1e-9,
        basis=(
            "pure geometry; exact under a point reflection -- a distance between two frame-sourced "
            "points, a speed MAGNITUDE, and two counts over a reprojected frame. Measured max "
            "|base-mirror| = 0.0 on every column (the plant in the module docstring moves 24.72 m, "
            "so 0.0 is a measurement rather than a degenerate scene)."
        ),
        role="unused",
        non_vacuity=(
            "nearest_defender_distance",
            "receiver_zone_density",
            "defenders_in_triangle_to_goal",
        ),
        exempt=_PROVENANCE_REASONS,
    )

    # ------------------------------------------------------------------ add_actor_pre_window
    # Defaults kept: NO parameter can make this entry live on canonical_scene(), because the
    # fixture holds every player position CONSTANT across its three frames by design. Both metric
    # columns are therefore identically 0.0 -- non-null (so the vacuity guard is satisfied) but
    # substantively uninformative. Recorded as a fixture limitation, not papered over with a
    # locally drifted scene.
    _entry(
        "add_actor_pre_window",
        lambda a, f, h: add_actor_pre_window(a, f),
        {
            "actor_arc_length_pre_window": "invariant",
            "actor_displacement_pre_window": "invariant",
            **_PROVENANCE_CLASSES,
        },
        tol=1e-9,
        basis=(
            "path LENGTH and point-to-point DISPLACEMENT are both magnitudes over frame-sourced "
            "positions, so a point reflection is exact. Measured max |base-mirror| = 0.0 -- but "
            "note the fixture holds positions constant across its three frames, so both columns "
            "are identically 0.0 and this entry's Gate A comparison is weak by construction."
        ),
        role="unused",
        non_vacuity=("actor_arc_length_pre_window", "actor_displacement_pre_window"),
        exempt=_PROVENANCE_REASONS,
    )

    # ------------------------------------------------------------------ add_pressure_on_actor
    # All THREE methods, not just the shipped andrienko_oval default: bekkers_pi is the ADR-045
    # D1/D2 site (the away-action velocity reprojection in _reproject_rows and the ball
    # reprojection in _build_ball_xy_v_per_action), and a mirror gate that does not run it cannot
    # see a regression of the live defect this repo already shipped once.
    #   link_zones  r 4/3/2 -> 8/7/6 (ordering preserved): the nearest defender sits at 4.24 m,
    #               just outside the widest default zone, so the default reads 0 on all four rows.
    #   bekkers_pi  speed_threshold 2.0 -> 0.5: the fixture states speed = 1.0 m/s for every
    #               player, so the default active-pressing filter rejects the entire scene.
    _entry(
        "add_pressure_on_actor",
        lambda a, f, h: add_pressure_on_actor(
            a,
            f,
            methods=("andrienko_oval", "link_zones", "bekkers_pi"),
            params_per_method={
                "link_zones": LinkParams(r_hoz=8.0, r_lz=7.0, r_hz=6.0),
                "bekkers_pi": BekkersParams(speed_threshold=0.5),
            },
        ),
        {
            "pressure_on_actor__andrienko_oval": "invariant",
            "pressure_on_actor__link_zones": "invariant",
            "pressure_on_actor__bekkers_pi": "invariant",
            **_PROVENANCE_CLASSES,
        },
        tol=1e-9,
        basis=(
            "each method reduces to RELATIVE defender-to-actor geometry (plus, for bekkers_pi, a "
            "deterministic TTI normal-CDF over velocities that NEGATE together with the "
            "separation vector), so a point reflection is exact -- no pitch-control surface is "
            "involved and no fallback constant is reachable. Measured max |base-mirror| = 0.0 on "
            "all three methods, with bekkers_pi live on all four rows [0.0034, 0.4863, 0.8298, "
            "0.9372]."
        ),
        role="unused",
        non_vacuity=("pressure_on_actor__andrienko_oval", "pressure_on_actor__bekkers_pi"),
        exempt=_PROVENANCE_REASONS,
    )

    # ------------------------------------------------------------------ add_press_commitment
    # press_max_distance_m 3.0 -> 12.0: at the default NO defender is within range on ANY row, so
    # every output is NaN with source "no_pressing_defender" -- the vacuity guard would fail
    # outright. 12.0 is the smallest round value that reaches both away rows (nearest opponents at
    # 10.44 m and 4.24 m), making the away population -- the only rows an ADR-028 defect touches --
    # a real one rather than a single token action.
    _entry(
        "add_press_commitment",
        lambda a, f, h: add_press_commitment(a, f, params=PressCommitmentParams(press_max_distance_m=12.0)),
        {
            "press_commitment": "invariant",
            "press_commitment_closing_speed": "invariant",
            "press_commitment_source": "exempt",
            **_PROVENANCE_CLASSES,
        },
        tol=1e-9,
        basis=(
            "closing speed is the dot product of two quantities that BOTH negate under a point "
            "reflection -- the velocity (ADR-045: a vector negates, it does not reflect) and the "
            "defender->actor axis -- so the projection, and the least-squares slope of it, are "
            "exact. Measured max |base-mirror| = 0.0; closing speed is live and SIGNED on the away "
            "rows [+0.9099, +0.2121], so a leg that reprojected positions without negating "
            "velocities would flip its sign rather than agree."
        ),
        role="unused",
        non_vacuity=("press_commitment_closing_speed", "press_commitment"),
        exempt={
            "press_commitment_source": (
                "closed-vocabulary string provenance (PRESS_COMMITMENT_SOURCE_VALUES), not a "
                "geometric quantity; verified identical in both legs "
                "['no_pressing_defender', 'computed', 'computed', 'computed']"
            ),
            **_PROVENANCE_REASONS,
        },
    )
