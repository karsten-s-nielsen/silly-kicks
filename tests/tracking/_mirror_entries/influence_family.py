"""Influence-family MirrorEntry registrations (ADR-028 section 6).

Four pitch-control-derived, xT-weighted aggregators: ``add_gk_influence``,
``add_player_influence``, ``add_cover_shadows`` and ``add_off_ball_run_values``.

All four take the fitted xT POSITIONALLY as the third argument and ``home_team_id`` KEYWORD-ONLY
(verified with ``inspect.signature``), so every ``call`` below is the same shape.

Every tolerance here is sized ABOVE a measurement recorded in its ``basis`` string, per section 6's
"record each entry's tolerance and its measured basis separately". None of them is inherited from a
sibling entry.
"""

from __future__ import annotations

_PROVENANCE_REASON = "linkage provenance, not geometry"

#: Shared across the three entries whose Gate A residual is float-reduction noise rather than a
#: geometric asymmetry. Spelled out per entry in ``basis`` with that entry's own measurement --
#: this constant is only the shared PREFIX of the argument, never the evidence.
_PC_SYMMETRY_NOTE = (
    "pitch control is NOT exactly mirror-symmetric in general, but on the canonical scene the "
    "180-degree point reflection maps the sampling grid one-to-one, so the surviving residual is "
    "float-reduction (re-association) noise"
)


def register() -> None:
    from silly_kicks.tracking.features import (
        add_cover_shadows,
        add_gk_influence,
        add_off_ball_run_values,
        add_player_influence,
    )
    from tests.tracking._mirror_registry import _entry, gate_xt

    # ------------------------------------------------------------------
    # add_gk_influence
    # ------------------------------------------------------------------
    # GATE A measured max residual 4.44e-16 (closing times, seconds); the share column reads
    # 1.39e-17 and the reachable area exactly 0.0.
    #
    # GATE B FAILS AND IS NOT XFAILED -- deliberately. This is a FINDING, not an oversight: the
    # spec's D3 re-key list does not name add_gk_influence, yet _gk_influence.py:318 resolves the
    # defended goal as `same_id(defending_team_id, home_team_id) -> 0.0 else 105.0` and :371
    # reflects the threat grid on `not same_id(attacking_team_id, home_team_id)`. Both are
    # identity-keyed direction, which is exactly the D1 class Gate B exists to find. Measured
    # movement when home_team_id -> away/nonsense: share 0.109, closing-time min 4.38 s, mean
    # 4.02 s. Manufacturing an xfail here would launder an unrecorded defect into a documented
    # one; the honest state is RED until the spec's D3 list is corrected or the re-key lands.
    _entry(
        "add_gk_influence",
        lambda a, f, h: add_gk_influence(a, f, gate_xt(), home_team_id=h),
        {
            "gk_pitch_control_share_weighted": "invariant",
            "gk_reachable_area_m2": "invariant",
            "gk_closing_time_min_s__six_yard_box": "invariant",
            "gk_closing_time_mean_s__six_yard_box": "invariant",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-9,
        basis=(
            f"{_PC_SYMMETRY_NOTE}. Measured Gate A max 4.44e-16 across all four columns "
            "(share ~0.12, area 9.40 m^2, closing times ~1.4 s). 1e-9 sits ~6 orders above that "
            "measurement and ~8 orders below the O(1) movement an orientation defect produces "
            "here (the identity-keyed goal-end resolution moves the closing times by 4.38 s)."
        ),
        role="direction_only",
        # EIGHTH D3 member -- found by Gate B, not by the audit that produced the spec's list of
        # seven (spec 4.3 predicted the enumeration would return MORE than seven; this is it).
        # Two identity-keyed sites:
        #   _gk_influence.py:318     goal-end resolution `if same_id(defending_team_id, home_team_id)`
        #   _gk_influence.py:371-372 the ADR-028 point reflection ITSELF, gated on
        #                            `same_id(attacking_team_id, home_team_id)` -- the reflection is
        #                            correct (both axes), the KEY is wrong.
        # Measured: gk_pitch_control_share_weighted +0.1085, gk_closing_time_min_s__six_yard_box
        # +4.38 s, IDENTICAL under ->AWAY and ->999999 (the `same_id(x, home) else ...` signature).
        # Gate A reads 4.44e-16 for the same aggregator: blind to this class exactly as designed.
        defect_b="D3 re-key pending: identity-keyed direction (spec 4.3); 8th member, found by Gate B",
        non_vacuity=("gk_pitch_control_share_weighted", "gk_closing_time_min_s__six_yard_box"),
        exempt={
            "frame_id": _PROVENANCE_REASON,
            "time_offset_seconds": _PROVENANCE_REASON,
            "n_candidate_frames": _PROVENANCE_REASON,
            "link_quality_score": _PROVENANCE_REASON,
        },
    )

    # ------------------------------------------------------------------
    # add_player_influence
    # ------------------------------------------------------------------
    # GATE A measured max residual 2.27e-13, on off_ball_xt_team / off_ball_xt_diff, whose
    # magnitudes are ~1.7e3 and ~7.9e3 -- i.e. ~3e-17 RELATIVE. The reachable-area columns and
    # off_ball_xt_opponent read exactly 0.0.
    #
    # GATE B is the D3 re-key target the spec names: measured 6.93e3 movement on off_ball_xt_diff
    # under both the away and the nonsense home id.
    _entry(
        "add_player_influence",
        lambda a, f, h: add_player_influence(a, f, gate_xt(), home_team_id=h),
        {
            "actor_reachable_area_m2": "invariant",
            "off_ball_xt_team": "invariant",
            "off_ball_xt_opponent": "invariant",
            "off_ball_xt_diff": "invariant",
            "reachable_area_team": "invariant",
            "reachable_area_opponent": "invariant",
            "reachable_area_diff": "invariant",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-6,
        basis=(
            f"{_PC_SYMMETRY_NOTE}. Measured Gate A max 2.27e-13, on the xT-weighted columns whose "
            "magnitude is ~8e3 -- an ABSOLUTE residual, because the value is a sum of pitch-control "
            "share x xT over every grid cell and float noise there scales with the accumulated "
            "magnitude, not with 1.0. 1e-6 is ~7 orders above the measurement (headroom for a "
            "re-ordered reduction on a numba vs numpy leg) and still ~10 orders below the 6.93e3 "
            "movement the identity-keyed grid reflection produces."
        ),
        role="direction_only",
        non_vacuity=("off_ball_xt_team", "off_ball_xt_diff"),
        exempt={
            "frame_id": _PROVENANCE_REASON,
            "time_offset_seconds": _PROVENANCE_REASON,
            "n_candidate_frames": _PROVENANCE_REASON,
            "link_quality_score": _PROVENANCE_REASON,
        },
        defect_b="D3 re-key pending: identity-keyed direction (spec 4.3)",
    )

    # ------------------------------------------------------------------
    # add_cover_shadows
    # ------------------------------------------------------------------
    # GATE A PASSES as of 4.70.0 (PR-S138). It FAILED under RC1, where the raw action-LTR passer
    # moved n_blocked_receivers by 3 and max_single_defender_blocking_score by 0.304, while the
    # passer-independent columns stayed clean -- blocking_score 2.84e-14, blocked_threat_fraction
    # 1.11e-16, n_potential_receivers exactly 0 -- matching spec 3.1's per-column scope exactly.
    # Those pre-fix magnitudes are retained deliberately: they are what a REGRESSION would have to
    # reproduce, and they set the tolerance ceiling below.
    #
    # GATE B STILL FAILS (D3) and is xfailed strictly: blocking_score moves 148.83.
    #
    # max_single_defender_player_id is EXEMPT: it is a player identity, and PR-S136 gated it to
    # detailed=True, so on the default cheap path it is all-NaN (verified: 0/4 non-null).
    _entry(
        "add_cover_shadows",
        lambda a, f, h: add_cover_shadows(a, f, gate_xt(), home_team_id=h),
        {
            "n_blocked_receivers": "invariant",
            "n_potential_receivers": "invariant",
            "blocking_score": "invariant",
            "blocked_threat_fraction": "invariant",
            "max_single_defender_blocking_score": "invariant",
            "max_single_defender_player_id": "exempt",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-6,
        basis=(
            f"{_PC_SYMMETRY_NOTE}. Measured Gate A residual on the two passer-INDEPENDENT float "
            "columns -- blocking_score 2.84e-14 (magnitude 148.8) and blocked_threat_fraction "
            "1.11e-16 -- so 1e-6 is ~8 orders of headroom there. It must NOT be loosened past the "
            "0.304 that max_single_defender_blocking_score moved under RC1: Gate A now PASSES, so "
            "that ceiling is what keeps an RC1 REGRESSION detectable rather than silently absorbed."
        ),
        role="direction_only",
        non_vacuity=("blocking_score", "n_potential_receivers"),
        exempt={
            "max_single_defender_player_id": (
                "player identity, not geometry; PR-S136 gates it to detailed=True so the default "
                "cheap path emits all-NaN (measured 0/4 non-null on the canonical scene)"
            ),
            "frame_id": _PROVENANCE_REASON,
            "time_offset_seconds": _PROVENANCE_REASON,
            "n_candidate_frames": _PROVENANCE_REASON,
            "link_quality_score": _PROVENANCE_REASON,
        },
        # RC1 FIXED in PR 2 (spec 3.1): the passer is now reprojected into frame coords at both
        # seams, so Gate A passes and its marker is GONE -- strict xfail makes that deletion
        # mandatory rather than optional.
        #
        # Gate B's marker STAYS: `_cover_shadows.py:1030` still keys `attacking_toward_high_x` on
        # `same_id(attacking_team_id, home_team_id)`. That is D3, which spec section 9 keeps out
        # of this cycle (byte-identical on converter output), NOT part of RC1.
        defect_b="D3 re-key pending: identity-keyed direction (spec 4.3)",
    )

    # ------------------------------------------------------------------
    # add_off_ball_run_values
    # ------------------------------------------------------------------
    # Both gates measured at EXACTLY 0.0 on every emitted column. TF-35 (ADR-042) keys its
    # direction on acting_team_attacks_rtl, which reads the frame's team_attacking_direction and
    # never home_team_id -- so the Gate B zero is the D3 property holding, not a vacuous check
    # (Gate B asserts checked > 0, and run_value_enabled_pass is finite and DISTINCT between the
    # home and away pass rows: 0.0508 vs 0.0978).
    #
    # The shot rows (action_id 2 and 4) are NaN by domain: value_off_ball_runs is scoped to
    # completed pass/cross with a resolvable receiver. The away population is therefore one row,
    # which is why the non-vacuity anchor is the column that is genuinely non-zero there.
    _entry(
        "add_off_ball_run_values",
        lambda a, f, h: add_off_ball_run_values(a, f, gate_xt(), home_team_id=h),
        {
            "run_value_target": "invariant",
            "run_value_disruptive_sum": "invariant",
            "run_value_enabled_pass": "invariant",
            "n_disruptive_runs": "invariant",
            "n_valued_disruptive_runs": "invariant",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-9,
        basis=(
            "run_value is a MAX over the cells a runner controls above region_influence_floor, so "
            "unlike a sum it never accumulates reduction noise: the point reflection maps that cell "
            "set one-to-one and the max is taken over the same multiset. Measured Gate A and Gate B "
            "residuals are EXACTLY 0.0 on all five columns; 1e-9 is headroom against a re-ordered "
            "threshold comparison on a numba leg, not a slack the current implementation needs."
        ),
        role="direction_only",
        non_vacuity=("run_value_enabled_pass", "run_value_target"),
        exempt={
            "frame_id": _PROVENANCE_REASON,
            "time_offset_seconds": _PROVENANCE_REASON,
            "n_candidate_frames": _PROVENANCE_REASON,
            "link_quality_score": _PROVENANCE_REASON,
        },
    )
