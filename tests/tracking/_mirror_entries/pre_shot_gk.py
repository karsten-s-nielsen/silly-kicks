"""Pre-shot-GK / shot-goalmouth ``MirrorEntry`` registrations (ADR-028 section 6).

All three aggregators in this group take NO ``home_team_id`` (verified by ``inspect.signature``),
so every entry declares ``role="unused"`` and Gate B skips them: there is no identity key to vary.
Orientation reaches them only from the FRAMES -- the two pre-shot-GK aggregators via
``_resolve_action_frame_context``'s ADR-028 reprojection (the ``team_attacking_direction`` LABEL),
``add_shot_goalmouth`` via ``_gk_resolve.defended_goal_x`` (pure GEOMETRY, ``_shot_goalmouth.py:746``,
no label read at all). That is D1's precedence already satisfied, so this group has nothing for
Gate B to catch and its skips are structural rather than a coverage hole.
"""

from __future__ import annotations


def _with_defending_gk(actions, frames):
    """Supply the ``defending_gk_player_id`` both pre-shot-GK aggregators REQUIRE.

    ``canonical_scene()`` carries no ``keeper_*`` actions, so
    ``spadl.utils.add_pre_shot_gk_context`` -- which resolves the keeper from the most recent
    defending-team keeper ACTION -- would return NaN for every row and make the gate vacuous.
    Resolve from the frames' own roster instead: the defending keeper is the OTHER team's
    ``is_goalkeeper`` row.

    Deliberately direction-FREE and identity-FREE: it reads ``team_id`` / ``player_id`` /
    ``is_goalkeeper``, none of which ``mirror_frames`` touches, so both Gate A legs receive the
    identical mapping and the gate measures the aggregator rather than this helper.
    """
    keepers = frames[frames["is_goalkeeper"].astype(bool) & frames["team_id"].notna()]
    # One keeper per team, FIRST wins: the fixture repeats each roster row once per frame, so a
    # bare dict comprehension would be last-wins over three identical rows. Pinning it makes the
    # helper's output independent of frame ordering rather than accidentally so.
    keepers = keepers.drop_duplicates("team_id", keep="first")
    by_team = {float(t): float(p) for t, p in zip(keepers["team_id"], keepers["player_id"], strict=True)}
    out = actions.copy()
    out["defending_gk_player_id"] = [
        next((pid for team, pid in by_team.items() if team != float(t)), float("nan")) for t in actions["team_id"]
    ]
    return out


_PROVENANCE = ("frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score")
_PROVENANCE_REASON = "linkage provenance, not geometry"


def register() -> None:
    from silly_kicks.tracking.features import (
        add_pre_shot_gk_angle,
        add_pre_shot_gk_position,
        add_shot_goalmouth,
    )
    from tests.tracking._mirror_registry import _entry

    # -- TF-11 / PR-S21 -------------------------------------------------------------------
    # Measured base-vs-mirror delta 0.0 on every column. Non-vacuity is real AND the gate is
    # discriminating: monkeypatching `acting_team_attacks_rtl` to the pre-ADR-028 all-False flip
    # moves pre_shot_gk_x by 95.0 m, _y by 14.0 m, distance_to_goal by 91.6 m and
    # distance_to_shot by 4.65 m -- so a green reading here is a measurement, not an absence.
    _entry(
        "add_pre_shot_gk_position",
        lambda a, f, h: add_pre_shot_gk_position(_with_defending_gk(a, f), f),
        {
            "pre_shot_gk_x": "invariant",
            "pre_shot_gk_y": "invariant",
            "pre_shot_gk_distance_to_goal": "invariant",
            "pre_shot_gk_distance_to_shot": "invariant",
            **dict.fromkeys(_PROVENANCE, "exempt"),
        },
        tol=1e-9,
        basis=(
            "pure geometry; exact under a point reflection. The kernel copies the reprojected GK "
            "x/y and takes two Euclidean distances, so base and mirror agree bit-for-bit "
            "(measured 0.0 on all four columns)."
        ),
        role="unused",
        non_vacuity=("pre_shot_gk_x", "pre_shot_gk_distance_to_shot"),
        exempt=dict.fromkeys(_PROVENANCE, _PROVENANCE_REASON),
    )

    # -- TF-12 / PR-S24 -------------------------------------------------------------------
    # `frames` is KEYWORD-ONLY here (it is positional on the other two) -- verified by signature.
    # Same plant: the no-flip plant moves the trajectory angle by 3.085 rad and the goal-line
    # angle by 1.020 rad, so both declared columns are genuinely exercised.
    _entry(
        "add_pre_shot_gk_angle",
        lambda a, f, h: add_pre_shot_gk_angle(_with_defending_gk(a, f), frames=f),
        {
            "pre_shot_gk_angle_to_shot_trajectory": "invariant",
            "pre_shot_gk_angle_off_goal_line": "invariant",
        },
        tol=1e-9,
        basis=(
            "pure geometry; exact under a point reflection. Both columns are an arctan2 over "
            "vectors built from the reprojected GK and the action anchor (measured 0.0)."
        ),
        role="unused",
        non_vacuity=("pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"),
    )

    # -- TF-48 / PR-S93 -------------------------------------------------------------------
    # PARTIALLY VACUOUS ON THIS FIXTURE, stated rather than papered over. `canonical_scene()` holds
    # three frames (t=7.6/7.8/8.0) with the ball STATIONARY at (38, 23), and both shots are stamped
    # at t=8.0 -- so the post-contact window contains ONE ball sample against `min_fit_frames=3`,
    # and the stationary ball never comes contactably near either shot stamp (15.6 m away). Both
    # shot rows resolve `shot_crossing_source == "insufficient_frames"` and every metric column is
    # NaN in BOTH legs. They are still declared: the classification is the contract, and Gate A
    # skips a column with nothing comparable rather than passing it falsely. The only column with
    # live values on the away rows is `shot_crossing_confidence` (0.0), which is what
    # `non_vacuity` therefore anchors on -- an honest floor, not evidence of coverage.
    _entry(
        "add_shot_goalmouth",
        lambda a, f, h: add_shot_goalmouth(a, f),
        {
            "shot_crossing_y": "invariant",
            "shot_crossing_z": "invariant",
            "shot_speed": "invariant",
            "shot_time_to_goal_line": "invariant",
            "shot_on_target_derived": "invariant",
            "shot_crossing_confidence": "invariant",
            "shot_fit_n_frames": "invariant",
            "shot_fit_rmse": "invariant",
            "shot_crossing_source": "exempt",
            "shot_fit_end_reason": "exempt",
            "shot_z_profile": "exempt",
            **dict.fromkeys(_PROVENANCE, "exempt"),
        },
        tol=1e-9,
        basis=(
            "pure geometry -- a least-squares trajectory fit plus a goal-plane intersection, exact "
            "under a point reflection in exact arithmetic; the engine is orientation-AGNOSTIC "
            "(goal ends come from the GK map, never from a direction label). NOT A MEASUREMENT: "
            "this fixture yields `insufficient_frames`, so only `shot_crossing_confidence` is "
            "compared (delta 0.0). 1e-9 is a first-principles bound and is the number to revisit "
            "-- not to widen silently -- once a real shot trajectory reaches the fixture and the "
            "LS solve's floating-point asymmetry becomes observable."
        ),
        role="unused",
        non_vacuity=("shot_crossing_confidence",),
        exempt={
            "shot_crossing_source": "string provenance vocabulary, not geometry",
            "shot_fit_end_reason": "string provenance vocabulary, not geometry",
            "shot_z_profile": "string provenance vocabulary (rolling/airborne/bounced), not geometry",
            **dict.fromkeys(_PROVENANCE, _PROVENANCE_REASON),
        },
    )
