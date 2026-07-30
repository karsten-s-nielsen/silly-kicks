"""Misc MirrorEntry registrations (ADR-028 section 6).

Four aggregators that share one property: **none of them takes ``home_team_id`` as a frame-
orientation input**, so every entry here declares ``role="unused"`` and Gate B skips. That is a
finding about the group, not an evasion -- ``inspect.signature`` shows three of the four have no
such parameter at all, and the fourth's is the OUTPUT TEAM-ID SPACE of a pre-conversion roster
join (see ``add_gradientsports_player_ids`` below).

Two of the entries need a fixture accommodation, both recorded at their call site:

* ``add_defensive_credit`` requires an injected per-shot xG column and a fitted ``ExpectedThreat``
  (silly-kicks ships no xG model), so the callable injects a constant ``xg`` and passes
  ``gate_xt()``.
* ``add_elastic_sync`` needs ``min_confidence=0.0``, because the canonical scene holds ball
  positions CONSTANT across its three frames -- which zeroes ball acceleration, the term carrying
  60% of the ELASTIC score -- so at the 0.1 default every action falls below threshold and all
  three columns come back all-NaN. That is the vacuity the gate's own non-vacuity guard exists to
  reject.
"""

from __future__ import annotations


def register() -> None:
    import pandas as pd

    from silly_kicks.tracking import (
        add_defensive_credit,
        add_elastic_sync,
        add_gradientsports_player_ids,
        add_sync_score,
        link_actions_to_frames,
    )
    from tests.tracking._mirror_registry import AWAY, HOME, _entry, gate_xt

    _LINK_PROVENANCE = "linkage provenance, not geometry"

    # ------------------------------------------------------------------
    # add_defensive_credit
    # ------------------------------------------------------------------
    def _call_defensive_credit(actions, frames, _home):
        # xg is an INJECTED per-shot column (ADR-036: silly-kicks ships no xG model). A constant
        # keeps the shot rules' sizing identical on both legs, so any base-vs-mirror difference is
        # geometry, never the injected value.
        acts = actions.copy()
        acts["xg"] = 0.25
        return add_defensive_credit(acts, frames, xg_column="xg", xt=gate_xt())

    _entry(
        "add_defensive_credit",
        _call_defensive_credit,
        {
            "defensive_credit_net": "invariant",
            "defensive_credit_plus": "invariant",
            "defensive_credit_minus": "invariant",
            "n_defensive_credits": "invariant",
            "frame_id": "exempt",
            "time_offset_seconds": "exempt",
            "n_candidate_frames": "exempt",
            "link_quality_score": "exempt",
        },
        tol=1e-9,
        basis=(
            "pure geometry -- proximity-gated defender resolution sized by xG / xT(origin); exact "
            "under a point reflection. Measured 0.0 base-vs-mirror on the canonical scene, and the "
            "gate has TEETH there: an x-ONLY mirror (the ADR-041 incomplete-repair shape) moves "
            "defensive_credit_net/_minus by 0.25 and n_defensive_credits by 1, and shoving the "
            "players +9 m in y drops the credits entirely."
        ),
        role="unused",
        non_vacuity=("defensive_credit_net", "n_defensive_credits"),
        exempt={
            "frame_id": _LINK_PROVENANCE,
            "time_offset_seconds": _LINK_PROVENANCE,
            "n_candidate_frames": _LINK_PROVENANCE,
            "link_quality_score": _LINK_PROVENANCE,
        },
    )

    # ------------------------------------------------------------------
    # add_elastic_sync
    # ------------------------------------------------------------------
    def _call_elastic_sync(actions, frames, _home):
        # min_confidence=0.0 disables ONLY the drop threshold; the ELASTIC scoring and frame
        # selection under test are untouched. Required because the canonical scene's ball is
        # stationary across all three frames -> ball_accel == 0 -> the 0.6-weighted acceleration
        # term vanishes and every action scores below the 0.1 default (measured: all four rows
        # all-NaN at 0.1 and at 0.05; at 0.02 only the HOME rows survive, which would leave the
        # away population -- the only rows an ADR-028 defect touches -- empty).
        return add_elastic_sync(actions, frames, min_confidence=0.0)

    _entry(
        "add_elastic_sync",
        _call_elastic_sync,
        {
            "elastic_frame_id": "invariant",
            "elastic_confidence": "invariant",
            "elastic_error_seconds": "invariant",
        },
        tol=1e-9,
        basis=(
            "the ELASTIC score is built from two MAGNITUDES -- ball acceleration and player-ball "
            "distance -- and a point reflection is a rigid isometry, so both are preserved "
            "exactly; measured 0.0 base-vs-mirror. Discriminating power is genuine but PARTIAL: "
            "moving the players +9 m in y moves elastic_confidence by 0.0105, yet an x-ONLY "
            "mirror also leaves it at 0.0, because ball and players reflect TOGETHER and any "
            "isometry preserves the distances. This entry therefore proves the columns are "
            "geometry-derived and mirror-stable; it cannot by itself distinguish a full point "
            "reflection from a partial one."
        ),
        role="unused",
        non_vacuity=("elastic_confidence", "elastic_frame_id"),
    )

    # ------------------------------------------------------------------
    # add_sync_score
    # ------------------------------------------------------------------
    def _call_sync_score(actions, frames, _home):
        # Second positional is `links`, NOT frames: build the pointers from whichever frames the
        # gate hands us, so the mirror leg is scored off the mirrored frames' own linkage.
        links = link_actions_to_frames(actions, frames)[0]
        return add_sync_score(actions, links)

    _entry(
        "add_sync_score",
        _call_sync_score,
        {
            "sync_score_min": "invariant",
            "sync_score_mean": "invariant",
            "sync_score_high_quality_frac": "invariant",
        },
        tol=1e-9,
        basis=(
            "exact BY CONSTRUCTION: sync_score consumes only the link pointers' time offsets and "
            "never reads a coordinate, so no spatial transform can move it (verified -- an x-only "
            "mirror AND a +9 m player shove both leave all three columns at delta 0.0). Kept "
            "`invariant` rather than `exempt` deliberately: an exempt column is never compared, so "
            "a future change that made these columns geometry-dependent would pass unnoticed."
        ),
        role="unused",
        non_vacuity=("sync_score_min", "sync_score_mean"),
    )

    # ------------------------------------------------------------------
    # add_gradientsports_player_ids
    # ------------------------------------------------------------------
    # NOT an action-coupled aggregator: it is a PRE-conversion roster join, run BEFORE
    # `gradientsports.convert_to_frames`, over `(jersey_frames, roster)` -- it never sees actions,
    # never sees a coordinate, and returns a `(frames, report)` tuple. Neither gate's premise
    # applies to it, so both are answered honestly rather than simulated:
    #
    #   * Gate A's swap rationale is "after a physical mirror the team attacking +x really is the
    #     other one" -- a statement about ORIENTATION. This function's `home_team_id` is the output
    #     team-id SPACE (which integer to stamp on "home" rows), so swapping it would relabel the
    #     teams for a reason that has nothing to do with mirroring. The callable therefore pins
    #     both team ids as part of the fixture, alongside the roster.
    #   * Gate B's premise is "action-LTR geometry cannot depend on which team is home". This
    #     function emits no geometry, so it has no `invariant` column to hold fixed. Threading the
    #     gate's ids through it changes every emitted column BY DESIGN -- measured on the fixture
    #     below, home rows only, over home_team_id in {HOME, AWAY, 999999}:
    #         team_id        1  -> 2        -> 999999
    #         player_id   1,10  -> 50,<NA>  -> <NA>,<NA>   (re-joins to the other team's shirt,
    #                                                        then misses the roster entirely)
    #         is_goalkeeper True,False -> True,False -> False,False
    #     Every column is therefore `exempt`, which leaves Gate B nothing to compare -- and a Gate B
    #     that compares nothing fails its own vacuity assertion. `role="unused"` records that the
    #     axis is inapplicable, not that the signature lacks the parameter.
    #
    # What is left is a smoke check: the call runs on both legs. That is the honest ceiling here.
    _JERSEY_FRAMES = pd.DataFrame(
        {
            "team_side": ["home", "home", "away", "away", None],
            "jersey_number": ["1", "7", "1", "9", None],
            "is_ball": [False, False, False, False, True],
        }
    )
    _ROSTER = pd.DataFrame(
        {
            "team_id": [HOME, HOME, AWAY, AWAY],
            "shirt_number": ["1", "7", "1", "9"],
            "player_id": [1, 10, 50, 60],
            "position_group_type": ["GK", "MF", "GK", "FW"],
        }
    )

    def _call_gs_player_ids(_actions, _frames, _home):
        frames, _report = add_gradientsports_player_ids(
            _JERSEY_FRAMES.copy(),
            _ROSTER.copy(),
            home_team_id=HOME,
            away_team_id=AWAY,
        )
        return frames

    _ROSTER_JOIN = (
        "pre-conversion roster join output, not geometry: this helper reads only team_side / "
        "jersey_number / is_ball and never a coordinate, so the column is undefined under a frame "
        "mirror rather than invariant under one"
    )
    _entry(
        "add_gradientsports_player_ids",
        _call_gs_player_ids,
        {
            "team_id": "exempt",
            "player_id": "exempt",
            "is_goalkeeper": "exempt",
        },
        tol=1e-9,
        basis=(
            "no tolerance is exercised -- every emitted column is exempt (see the block comment "
            "above). The entry's whole value is the anti-rot registration plus a smoke check that "
            "the call runs on both legs; 1e-9 is recorded only so the field is not a bare number "
            "nobody can revisit."
        ),
        role="unused",
        non_vacuity=(),
        exempt={
            "team_id": _ROSTER_JOIN,
            "player_id": _ROSTER_JOIN,
            "is_goalkeeper": _ROSTER_JOIN,
        },
    )
