"""GK-model MirrorEntry registrations (ADR-028 section 6).

Group ``gk_models``: ``add_ghost_gk``, ``add_gk_completion``, ``add_xt_gk``.

Two of the three need INPUT CONSTRUCTION beyond the bare canonical scene, and both adjustments
are recorded here rather than buried in a lambda, because each one is load-bearing for whether
the gate can see anything at all:

1. **A GK-distribution domain.** ``add_gk_completion`` and ``add_xt_gk`` score only the
   ``gk_distribution_mask`` domain (goal-kick, or a pass/throw-in by the acting keeper). The
   canonical scene carries two passes and two shots, so BOTH aggregators emit an all-NaN metric
   column on it and every comparison below would be vacuous -- the exact death-green the
   ``non_vacuity`` guard exists to catch. ``_gk_distribution_scene`` re-types the two PASS rows
   (one per team) into goal-kicks taken by that team's keeper. The transform is deterministic and
   reads neither ``home_team_id`` nor the frames, so both gate legs receive identical actions.

2. **A goal-kick whose native origin is absent, under a native-origin-TRUSTED provider.** This is
   the combination RC2 (spec 3.2) lived on, and it was measured, not assumed. RC2 was FIXED in
   4.71.0, so the bolded figures below are the PRE-FIX signature -- retained because they are what a
   regression must reproduce, and because they are what sets this group's tolerance ceiling:

   =========================  ==================  =====================  ====================
   goal-kick native origin    frames provider     ``add_gk_completion``  ``add_xt_gk``
   =========================  ==================  =====================  ====================
   present (5.0, 27.0)        synthetic           0.0                    0.0
   present (5.0, 27.0)        sportec             0.0                    0.0
   ABSENT (NaN)               synthetic           **0.125**              0.0
   ABSENT (NaN)               sportec / GS        **0.125**              **7.0 m** (origin_y)
   =========================  ==================  =====================  ====================

   (max base-vs-mirror difference over the emitted columns.)

   With a native origin present nothing was imputed, so ``_gk_geometry._tracking_gk_xy`` -- the
   then-defective sibling -- was never called and both aggregators were exactly mirror-invariant
   even before the fix. That is why the ABSENT row is the one that discriminates. Real
   Gradient Sports data is ~67% NaN native goal-kick origin (spec 3.2), which is why the tracking
   tier runs at all, so the ABSENT row is the honest one to register.

   The provider matters for ``add_xt_gk`` ONLY: ``compute_xt_gk`` gates the ADR-024
   ``distrust_native_origin`` flag on ``native_origin_is_trusted(source_provider)``, and
   ``"synthetic"`` is not on the allowlist -> distrust -> goal-kicks route through
   ``_tracking_gk_xy_detected``, the sibling that DOES re-project (its docstring says so). So on an
   untrusted provider ``add_xt_gk`` is mirror-clean and RC2 is unreachable. ``add_gk_completion``
   calls ``resolve_gk_geometry`` with no such flag, so it hits the defect on every provider.
   ``"gradientsports"`` is used because that is the provider the spec measured RC2's 19.0% away-row
   rate on.

The 0.0 readings in the table above are also what makes ``tol=1e-9`` defensible for these two
entries: the post-fix residual is not a guess, it is the measured native-origin row.
"""

from __future__ import annotations

import functools


@functools.cache
def _ghost_model():
    """Load the bundled ghost model ONCE (Gate A + Gate B call an entry five times)."""
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    return GhostGkModel.from_variant("default")


def _gk_distribution_scene(actions):
    """Re-type the canonical scene's two PASS rows into goal-kicks by each team's keeper.

    Row shape is preserved (4 actions, same order, same ``team_id``) because Gate A compares
    positionally against ``canonical_scene()``'s away mask. The keeper positions are
    (5.0, 27.0) in action-LTR for BOTH teams by construction of the fixture -- the away keeper's
    frame position (100.0, 41.0) is the exact point reflection of the home keeper's (5.0, 27.0) --
    so any asymmetry the gate reports comes from the aggregator, not from the scene.

    Native origin is left NaN on purpose (see the module docstring): a present native origin
    bypasses the imputation ladder entirely and the gate goes quietly, uselessly green.
    """
    import numpy as np

    a = actions.copy()
    from silly_kicks.spadl import config as spadlconfig

    goalkick = spadlconfig.actiontype_id["goalkick"]
    sel = a["action_id"].isin([1, 3])
    a.loc[sel, "type_id"] = goalkick
    a.loc[sel, ["start_x", "start_y"]] = np.nan
    a.loc[a["action_id"] == 1, "player_id"] = 1.0  # home keeper
    a.loc[a["action_id"] == 3, "player_id"] = 50.0  # away keeper
    # Destinations stay NATIVE and deliberately differ per team (and are y-asymmetric), so the two
    # goal-kicks are not accidental duplicates of one another.
    a.loc[a["action_id"] == 1, ["end_x", "end_y"]] = [60.0, 20.0]
    a.loc[a["action_id"] == 3, ["end_x", "end_y"]] = [55.0, 47.0]
    return a


def _trusted_provider_frames(frames):
    """Frames relabelled to a native-origin-TRUSTED provider (ADR-024 allowlist).

    ``assign`` rather than in-place: the gate hands the mirror leg the output of
    ``mirror_frames``, and an entry must not mutate what it is given.
    """
    return frames.assign(source_provider="gradientsports")


_LINKAGE = "linkage provenance, not geometry"


def register() -> None:
    from silly_kicks.tracking.features import add_ghost_gk, add_gk_completion, add_xt_gk
    from tests.tracking._mirror_registry import _entry, gate_xt

    _entry(
        "add_ghost_gk",
        lambda a, f, h: add_ghost_gk(a, f, home_team_id=h, model=_ghost_model()),
        {"ghost_gk_x": "invariant", "ghost_gk_y": "invariant"},
        tol=3.0,
        basis=(
            "NOT pure geometry -- the served value is a boosted-tree prediction, and the bundled "
            "model is not exactly mirror-symmetric (its features include strict-inequality box "
            "counts and a convex hull, neither of which commutes with a reflection). Measured on "
            "this scene: 0.0755 m on x, 0.278 m on y. 3.0 m matches the shipped _GHOST_Y_TOL in "
            "test_action_ltr_mirror_invariance.py, whose lateralized probe measures 1.26 m of the "
            "same model asymmetry, and leaves headroom for the queued re-fit. CAVEAT, recorded so "
            "nobody reads more into a green than is there: this scene is not lateralized, so the "
            "model predicts a near-central keeper (y ~ 33.6-33.7) and a y-REPROJECTION FLIP would "
            "move y by only ~0.71 m -- under this tol. x is the discriminating axis here (a "
            "goal-side leak moves it ~90 m); the dedicated y-flip guard is the off-centre probe in "
            "test_action_ltr_mirror_invariance.py::test_ghost_gk_mirror_invariant."
        ),
        role="direction_only",
        non_vacuity=("ghost_gk_x", "ghost_gk_y"),
    )

    _entry(
        "add_gk_completion",
        lambda a, f, h: add_gk_completion(_gk_distribution_scene(a), _trusted_provider_frames(f)),
        {
            "gk_completion": "invariant",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-9,
        basis=(
            "A deterministic logistic over exactly-reflected geometry; every fixture coordinate is "
            "an integer, and 105-n / 68-n are exact in binary, so the mirrored scene is bit-exact. "
            "The residual is MEASURED, not assumed: on the same scene with a NATIVE goal-kick "
            "origin (which bypasses RC2's imputation ladder) base-vs-mirror is exactly 0.0. Under "
            "RC2, with the origin imputed, it was 0.125; 4.71.0 reprojects the tracking tier so the "
            "imputed path now matches the native one. That 0.125 is retained deliberately -- it is "
            "the magnitude a REGRESSION would have to reproduce, and it sets this tolerance's "
            "ceiling."
        ),
        role="unused",
        non_vacuity=("gk_completion",),
        exempt={
            "frame_id": _LINKAGE,
            "time_offset_seconds": _LINKAGE,
            "n_candidate_frames": _LINKAGE,
            "link_quality_score": _LINKAGE,
        },
    )

    _entry(
        "add_xt_gk",
        lambda a, f, h: add_xt_gk(_gk_distribution_scene(a), _trusted_provider_frames(f), gate_xt(), home_team_id=h),
        {
            "xt_gk_base": "invariant",
            "xt_gk_pev": "invariant",
            "xt_gk_rav": "invariant",
            "xt_gk_dzv": "invariant",
            "xt_gk_pressure": "invariant",
            "xt_gk": "invariant",
            # PR-S101 audit coords: the EXACT action-LTR coords the grid lookups used.
            "xt_gk_origin_x": "invariant",
            "xt_gk_origin_y": "invariant",
            "xt_gk_dest_x": "invariant",
            "xt_gk_dest_y": "invariant",
            # Numeric, deterministic, and a function of WHICH resolution tier fired -- a tier that
            # flips between the two legs is precisely the RC2 symptom, so this is held invariant
            # rather than waved through as provenance.
            "xt_gk_origin_confidence": "invariant",
            "xt_gk_native_goalkick_out_of_region": "invariant",
            "xt_gk_origin_source": "exempt",
            "xt_gk_dest_source": "exempt",
            "xt_gk_completion_variant": "exempt",
            "xt_gk_completion_source": "exempt",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-9,
        basis=(
            "Pure grid arithmetic over exactly-reflected geometry: the xT lookup, the phi grid and "
            "the completion logistic are all deterministic, and the fixture's integer coordinates "
            "reflect bit-exactly. MEASURED, not assumed: on the same scene with a NATIVE goal-kick "
            "origin every emitted column is base-vs-mirror 0.0 on both a trusted and an untrusted "
            "provider. Under RC2, with the origin imputed, the away goal-kick's origin moved 7.0 m; "
            "4.71.0 reprojects the tracking tier so the imputed path now matches the native one. "
            "That 7.0 m is retained deliberately -- it is what a REGRESSION would have to reproduce."
        ),
        role="direction_only",
        non_vacuity=("xt_gk", "xt_gk_rav", "xt_gk_origin_y"),
        exempt={
            "xt_gk_origin_source": "string resolution-tier provenance, not a numeric quantity",
            "xt_gk_dest_source": "string resolution-tier provenance, not a numeric quantity",
            "xt_gk_completion_variant": "string model-variant provenance, not a numeric quantity",
            "xt_gk_completion_source": "string serve-mode provenance, not a numeric quantity",
            "frame_id": _LINKAGE,
            "time_offset_seconds": _LINKAGE,
            "n_candidate_frames": _LINKAGE,
            "link_quality_score": _LINKAGE,
        },
    )
