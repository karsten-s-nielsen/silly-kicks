"""Pitch-control family MirrorEntry registrations (ADR-028 section 6).

Covers ``add_pitch_control``, ``add_obso``, ``add_pausa``, ``add_space_creation`` -- the four
aggregators whose value is read off a ``PitchControlSurface``.

**Never pass ``pitch_control_cache``.** The cache keys on frame IDENTITY and excludes player
positions (ADR-043), so a mirrored frame carrying its twin's identity would be served the base
leg's surface and every entry here would pass at exactly zero difference. Leaving the kwarg at its
``None`` default gives each call its own local cache, which is the only safe arrangement for a
mirror gate.

``xt=gate_xt()`` is supplied to the three EPV consumers deliberately. Their fallback is the
synthetic ``linspace`` x-ramp, which is y-SYMMETRIC -- an x-only reprojection is exact on it, so the
gate would be blind to precisely the incomplete-repair class ADR-041 shipped. ``gate_xt()`` is
y-asymmetric by construction.

Measurements behind the declarations below, taken on ``canonical_scene()``:

===================  ================================  ===========  ===========
aggregator           column                            Gate A max   Gate B max
===================  ================================  ===========  ===========
add_pitch_control    pitch_control_at_target__spearman  7.45e-20     n/a (unused)
add_obso             obso_actual / _peak                5.55e-17     0
add_obso             obso_optimal                       4.996e-16    0
add_pausa            pausa_spatial / _composite         1.22e-15     0
add_pausa            pausa_temporal                     0            0
add_space_creation   space_created_m2                   4.44e-16     0
add_space_creation   space_denied_m2_opponent           4.44e-16     0
===================  ================================  ===========  ===========

The two space-creation rows read **1.20688** until 4.71.0, when RC3 was fixed. That pre-fix value is
the magnitude a regression must reproduce and is retained in the entry comment below.
"""

from __future__ import annotations

# Sized to the measurements in the module docstring (worst case 1.22e-15) with ~9 orders of
# headroom, because these values are grid quadratures over a pitch-control surface rather than
# closed-form geometry: they carry float64 accumulation noise and can differ by an ulp-scale amount
# between the numpy and numba kernels and across interpreter/BLAS builds.
#
# It stays 5 orders BELOW a real orientation leak. The two calibration points are both measured:
# tests/tracking/test_action_ltr_mirror_invariance.py:46 records that a genuine OBSO orientation
# leak moves the value by >= 0.1, and the RC3 swap in add_space_creation below measures 1.21.
#
# KNOWN mechanism that could break this tolerance WITHOUT an orientation defect: the Spearman
# degenerate/no-information fallback returns EXACTLY 0.5, and the surface is not equal to its own
# mirror across it (the same file, :39-43, measured 0.5 at frame (90, 34) against 1.0 at the
# mirrored (15, 34) -- a clean 2x). canonical_scene() does not straddle that boundary today
# (measured 7.45e-20), but a fixture drift that put a query point inside the fallback in only one
# leg would trip this gate as a FIXTURE problem, not as a convention-mixing defect. Re-measure
# before bumping.
_PC_FAMILY_TOL = 1e-6

_PC_FAMILY_BASIS = (
    "pitch-control-derived, not closed-form geometry: a grid quadrature over a surface that is not "
    "exactly mirror-symmetric (its Spearman degenerate/no-information fallback returns exactly 0.5 "
    "and does not equal its own mirror). Measured base-vs-mirror on canonical_scene(): 7.45e-20 "
    "(pitch control), 4.996e-16 (obso), 1.22e-15 (pausa). 1e-6 leaves ~9 orders of headroom over "
    "float64/numba accumulation noise while staying 5 orders below a real orientation leak "
    "(>= 0.1 measured, test_action_ltr_mirror_invariance.py:46; the RC3 swap in "
    "add_space_creation measures 1.21)."
)

# Emitted by every aggregator here that links actions to frames.
_LINK_PROVENANCE = ("frame_id", "time_offset_seconds", "link_quality_score", "n_candidate_frames")
_LINK_PROVENANCE_REASON = "linkage provenance, not geometry"

_EPV_SOURCE_REASON = (
    "string EPV-provenance token (ADR-041 'xt'/'synthetic'/'injected'), not a numeric geometric "
    "quantity; it records WHICH surface was sampled, never where"
)


def _provenance_exempt() -> dict[str, str]:
    return dict.fromkeys(_LINK_PROVENANCE, _LINK_PROVENANCE_REASON)


def register() -> None:
    from silly_kicks.tracking import add_obso, add_pausa, add_pitch_control, add_space_creation
    from tests.tracking._mirror_registry import _entry, gate_xt

    # ------------------------------------------------------------------
    # add_pitch_control -- the only member with NO home_team_id parameter.
    #
    # LIMITATION, recorded because a green result here means less than it looks like. The four
    # values on canonical_scene() are, at full precision:
    #     action 1 (HOME) 0.5                     <- exactly the documented Spearman
    #                                                degenerate/no-information fallback
    #     action 2 (HOME) 5.084081315560864e-05   <- the ONLY informative comparison (d=7.45e-20)
    #     action 3 (AWAY) 1.0                     <- saturated
    #     action 4 (AWAY) 0.0                     <- saturated
    # So Gate A's whole discriminating power for this entry sits on ONE HOME row, while both AWAY
    # rows -- the only rows an ADR-028 defect touches -- are pinned at the ends of [0, 1] and would
    # not move if a convention mix displaced the query point within the fully-controlled region.
    # `non_vacuity` cannot see this: it checks non-null, and 1.0/0.0 are non-null.
    # Closing it needs an away target in a CONTESTED part of the surface, which is a change to the
    # shared canonical_scene() fixture and therefore out of scope for this group module.
    # ------------------------------------------------------------------
    _entry(
        "add_pitch_control",
        lambda a, f, _h: add_pitch_control(a, f),
        {"pitch_control_at_target__spearman": "invariant"},
        tol=_PC_FAMILY_TOL,
        basis=_PC_FAMILY_BASIS,
        # `inspect.signature` has no home_team_id at all -- direction comes from the frames via
        # the shared action-frame context, so Gate B skips this entry by construction.
        role="unused",
        non_vacuity=("pitch_control_at_target__spearman",),
    )

    # ------------------------------------------------------------------
    # add_obso / add_pausa -- home_team_id is in the signature as the orientation slot, but the
    # docstring (features.py:5363-5367) states orientation is keyed on the frames'
    # `team_attacking_direction` via `acting_team_attacks_rtl`, NOT on this parameter. role is
    # "direction_only" rather than "unused" precisely so Gate B RUNS and proves that claim by
    # output-identity instead of taking the docstring's word for it.
    # ------------------------------------------------------------------
    _entry(
        "add_obso",
        lambda a, f, h: add_obso(a, f, home_team_id=h, xt=gate_xt()),
        {
            "obso_actual": "invariant",
            "obso_peak": "invariant",
            "obso_optimal": "invariant",
            "obso_epv_source": "exempt",
            **dict.fromkeys(_LINK_PROVENANCE, "exempt"),
        },
        tol=_PC_FAMILY_TOL,
        basis=_PC_FAMILY_BASIS,
        role="direction_only",
        # OBSO's domain is passes, so the away SHOT (action_id 4) is legitimately NaN; away
        # action_id 3 carries obso_actual 0.2796 / obso_optimal 0.3543.
        non_vacuity=("obso_actual", "obso_optimal"),
        exempt={"obso_epv_source": _EPV_SOURCE_REASON, **_provenance_exempt()},
    )

    _entry(
        "add_pausa",
        lambda a, f, h: add_pausa(a, f, home_team_id=h, xt=gate_xt()),
        {
            # add_pausa emits the OBSO block as well as its own three columns.
            "obso_actual": "invariant",
            "obso_peak": "invariant",
            "obso_optimal": "invariant",
            "pausa_temporal": "invariant",
            "pausa_spatial": "invariant",
            "pausa_composite": "invariant",
            "obso_epv_source": "exempt",
            **dict.fromkeys(_LINK_PROVENANCE, "exempt"),
        },
        tol=_PC_FAMILY_TOL,
        basis=_PC_FAMILY_BASIS,
        role="direction_only",
        # pausa_temporal saturates at exactly 1.0 on this fixture, so it is a poor non-vacuity
        # anchor even though it is non-null; pausa_spatial is 0.7893 on away action_id 3.
        non_vacuity=("obso_actual", "pausa_spatial"),
        exempt={"obso_epv_source": _EPV_SOURCE_REASON, **_provenance_exempt()},
    )

    # ------------------------------------------------------------------
    # add_space_creation -- RC3, FIXED in 4.71.0. Gate B still PASSES with delta 0, which remains the
    # measured evidence for spec 3.3's "the `home_team_id` parameter is dead" claim: the fix threads
    # a frames-derived `attacks_rtl` rather than reading that parameter, so it stays dead by design.
    # ------------------------------------------------------------------
    _entry(
        "add_space_creation",
        lambda a, f, h: add_space_creation(a, f, home_team_id=h, xt=gate_xt()),
        {
            "space_created_m2": "invariant",
            "space_denied_m2_opponent": "invariant",
            "obso_epv_source": "exempt",
            **dict.fromkeys(_LINK_PROVENANCE, "exempt"),
        },
        tol=_PC_FAMILY_TOL,
        basis=_PC_FAMILY_BASIS,
        role="direction_only",
        # Both columns are non-null on BOTH away rows (1.685 / 0.906 created, 0.478 / 0.777
        # denied), so this is a real comparison, not an empty one.
        non_vacuity=("space_created_m2", "space_denied_m2_opponent"),
        exempt={"obso_epv_source": _EPV_SOURCE_REASON, **_provenance_exempt()},
        # Pre-fix RC3 signature, retained because it is what a REGRESSION must reproduce: the two
        # columns were EXCHANGED between the legs, exactly spec 3.3's "the two emitted columns are
        # exchanged for away actions".
        #   max |base.created - mir.denied|  = 4.44e-16   <- the SWAPPED pair agreed to float noise
        #   max |base.denied  - mir.created| = 2.22e-16
        #   max |base.created - mir.created| = 1.20688    <- while like-for-like did not
        # 4.71.0 reflects the attack-LTR transition/EPV grids into frame coords, so like-for-like now
        # agrees and the swap is gone. A re-introduced RC3 would flip those two magnitudes back.
    )
