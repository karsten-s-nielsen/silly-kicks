"""SB360 verdicts -- space family.

Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a
human writes an adjudication or a rationale.
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry

#: Shared by the five ``add_cover_shadows`` columns that ADR-055 moved from ``all_nan`` to
#: ``no_signal`` on the ``gk_absent`` roster. One constant because it is ONE finding about one
#: mechanism -- five hand-copied paragraphs would drift.
_GK_ABSENT_DEGENERATE_MAP = (
    "ADR-055. Dropping BOTH keepers sends resolve_defended_goals to its outfield rung, which on "
    "this fixture guesses BOTH teams at x=105 (measured outfield mean x: team 1 = 56.9, team 2 = "
    "76.5, both above the 52.5 midline). That map is DEGENERATE, so attacked_goal returns None by "
    "its documented same-end guard and add_cover_shadows emits a NaN row. Both legs go NaN for "
    "the same reason -- the cause is the ROSTER, not the kinematics -- so no informative rows "
    "remain and the observation collapses to no_signal. Recorded as unexercised because the "
    "vocabulary admits nothing else from no_signal; the collapse itself IS the finding, and it "
    "is a CHANGE: before the re-key, direction came from home_team_id, so both legs confidently "
    "produced a number these frames cannot support. add_cover_shadows is now keeper-dependent on "
    "freeze-frames, which for SB360 is exactly the coverage question."
)

_entry(
    "add_cover_shadows",
    ADAPTERS["add_cover_shadows"](T.add_cover_shadows),
    columns=(
        "n_blocked_receivers",
        "n_potential_receivers",
        "blocking_score",
        "blocked_threat_fraction",
        "max_single_defender_blocking_score",
        "max_single_defender_player_id",
    ),
    velocity={
        "n_blocked_receivers": AxisVerdict("all_nan", "honest_nan"),
        "n_potential_receivers": AxisVerdict("all_nan", "honest_nan"),
        "blocking_score": AxisVerdict("all_nan", "honest_nan"),
        "blocked_threat_fraction": AxisVerdict("all_nan", "honest_nan"),
        "max_single_defender_blocking_score": AxisVerdict("all_nan", "honest_nan"),
        "max_single_defender_player_id": AxisVerdict(
            "no_signal",
            "not_exercised",
            rationale=(
                "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                "occurrence context, or blocking defender to score). A fixture inadequacy, not a library property "
                "-- widening the fixture would move it. [measured cause=velocity+frame_count]"
            ),
        ),
    },
    visibility={
        # ADR-055 MOVED ALL FIVE of these from `all_nan` to `no_signal`, and the shared rationale
        # below is written by hand rather than taken from `_adjudicate.py`'s `fixture_domain`
        # rule, whose generic text ("no pressing sequence, shot-occurrence context, or blocking
        # defender to score") names a cause that is NOT what happens here.
        #
        # MEASURED mechanism. `gk_absent` drops BOTH keepers, so `resolve_defended_goals` falls to
        # the ladder's outfield rung -- and on this fixture that rung puts BOTH teams at the same
        # end: team 1 outfield mean x = 56.9 and team 2 = 76.5, both > 52.5, so both are guessed
        # to defend x=105. That map is DEGENERATE, `attacked_goal` returns None for both teams by
        # its documented second guard, and `_compute_cover_shadow_dict` raises
        # `GoalEndUnresolvedError`, which `add_cover_shadows` turns into a NaN row. Leg B goes NaN
        # for the same reason as leg A -- the cause is the roster, not the kinematics -- so the
        # comparison loses its informative rows and collapses to `no_signal`.
        #
        # This is the cycle working, not a regression: BEFORE the re-key, direction came from
        # `home_team_id`, which is always available and always confident, so both legs produced a
        # number from an assumption these frames cannot support. The NaN is the honest answer.
        #
        # It is also a real consequence worth naming: `add_cover_shadows` is now KEEPER-DEPENDENT
        # on freeze-frames. For SB360 that matters, because keeper presence is exactly what the
        # coverage question is about.
        "gk_absent": {
            "n_blocked_receivers": AxisVerdict("no_signal", "not_exercised", rationale=_GK_ABSENT_DEGENERATE_MAP),
            "n_potential_receivers": AxisVerdict("no_signal", "not_exercised", rationale=_GK_ABSENT_DEGENERATE_MAP),
            "blocking_score": AxisVerdict("no_signal", "not_exercised", rationale=_GK_ABSENT_DEGENERATE_MAP),
            "blocked_threat_fraction": AxisVerdict("no_signal", "not_exercised", rationale=_GK_ABSENT_DEGENERATE_MAP),
            "max_single_defender_blocking_score": AxisVerdict(
                "no_signal", "not_exercised", rationale=_GK_ABSENT_DEGENERATE_MAP
            ),
            "max_single_defender_player_id": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "n_blocked_receivers": AxisVerdict("all_nan", "honest_nan"),
            "n_potential_receivers": AxisVerdict("all_nan", "honest_nan"),
            "blocking_score": AxisVerdict("all_nan", "honest_nan"),
            "blocked_threat_fraction": AxisVerdict("all_nan", "honest_nan"),
            "max_single_defender_blocking_score": AxisVerdict("all_nan", "honest_nan"),
            "max_single_defender_player_id": AxisVerdict(
                "no_signal",
                "not_exercised",
                rationale=(
                    "The fixture does not produce this column's domain on either leg (no pressing sequence, shot- "
                    "occurrence context, or blocking defender to score). A fixture inadequacy, not a library "
                    "property -- widening the fixture would move it. [measured cause=velocity+frame_count]"
                ),
            ),
        },
    },
    applicability={
        "n_blocked_receivers": "no_support",
        "n_potential_receivers": "no_support",
        "blocking_score": "no_support",
        "blocked_threat_fraction": "no_support",
        "max_single_defender_blocking_score": "no_support",
        "max_single_defender_player_id": "no_support",
    },
    applicability_deltas={
        "n_blocked_receivers": {"extreme": 0.0, "near": 0.0},
        "n_potential_receivers": {"extreme": 0.0, "near": 0.0},
        "blocking_score": {"extreme": 0.0, "near": 0.0},
        "blocked_threat_fraction": {"extreme": 0.0, "near": 0.0},
        "max_single_defender_blocking_score": {"extreme": 0.0, "near": 0.0},
        "max_single_defender_player_id": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_das",
    C.generic(T.add_das),
    columns=(
        "das_team",
        "das_opponent",
        "das_diff",
        "das_source",
    ),
    velocity={
        "das_team": AxisVerdict("all_nan", "honest_nan"),
        "das_opponent": AxisVerdict("all_nan", "honest_nan"),
        "das_diff": AxisVerdict("all_nan", "honest_nan"),
        "das_source": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                "designed das_source to do exactly this. [measured cause=velocity]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "das_team": AxisVerdict("all_nan", "honest_nan"),
            "das_opponent": AxisVerdict("all_nan", "honest_nan"),
            "das_diff": AxisVerdict("all_nan", "honest_nan"),
            "das_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity]"
                ),
            ),
        },
        "defender_absent": {
            "das_team": AxisVerdict("all_nan", "honest_nan"),
            "das_opponent": AxisVerdict("all_nan", "honest_nan"),
            "das_diff": AxisVerdict("all_nan", "honest_nan"),
            "das_source": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "A provenance column: its job is to report WHICH path produced the value, so reporting a "
                    "different path on a freeze-frame leg than on a tracking leg is correct behaviour. ADR-043 "
                    "designed das_source to do exactly this. [measured cause=velocity]"
                ),
            ),
        },
    },
    applicability={
        "das_team": "no_support",
        "das_opponent": "no_support",
        "das_diff": "no_support",
        "das_source": "no_support",
    },
    applicability_deltas={
        "das_team": {"extreme": 0.0, "near": 0.0},
        "das_opponent": {"extreme": 0.0, "near": 0.0},
        "das_diff": {"extreme": 0.0, "near": 0.0},
        "das_source": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_obso",
    ADAPTERS["add_obso"](T.add_obso),
    columns=(
        "obso_actual",
        "obso_peak",
        "obso_optimal",
        "obso_epv_source",
    ),
    velocity={
        "obso_actual": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_peak": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured "
                "cause=velocity+frame_count]"
            ),
        ),
        "obso_optimal": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_epv_source": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict("identical", "works"),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "obso_actual": "no_support",
        "obso_peak": "no_support",
        "obso_optimal": "no_support",
        "obso_epv_source": "no_support",
    },
    applicability_deltas={
        "obso_actual": {"extreme": 0.0, "near": 0.0},
        "obso_peak": {"extreme": 0.0, "near": 0.0},
        "obso_optimal": {"extreme": 0.0, "near": 0.0},
        "obso_epv_source": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pausa",
    ADAPTERS["add_pausa"](T.add_pausa),
    columns=(
        "obso_actual",
        "obso_peak",
        "obso_optimal",
        "obso_epv_source",
        "pausa_temporal",
        "pausa_spatial",
        "pausa_composite",
    ),
    velocity={
        "obso_actual": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_peak": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured "
                "cause=velocity+frame_count]"
            ),
        ),
        "obso_optimal": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_epv_source": AxisVerdict("identical", "works"),
        "pausa_temporal": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Cause isolated as frame_count, not velocity: the feature needs a temporal window and a single "
                "freeze-frame legitimately yields a different, single-sample answer. Nothing is fabricated from "
                "absent kinematics. [measured cause=frame_count]"
            ),
        ),
        "pausa_spatial": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "pausa_composite": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured "
                "cause=velocity+frame_count]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict("identical", "works"),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
            "pausa_temporal": AxisVerdict("identical", "works"),
            "pausa_spatial": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "pausa_composite": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
        },
        "defender_absent": {
            "obso_actual": AxisVerdict("identical", "works"),
            "obso_peak": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
            "obso_optimal": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
            "pausa_temporal": AxisVerdict("identical", "works"),
            "pausa_spatial": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "pausa_composite": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured "
                    "cause=velocity+frame_count]"
                ),
            ),
        },
    },
    applicability={
        "obso_actual": "no_support",
        "obso_peak": "no_support",
        "obso_optimal": "no_support",
        "obso_epv_source": "no_support",
        "pausa_temporal": "no_support",
        "pausa_spatial": "no_support",
        "pausa_composite": "no_support",
    },
    applicability_deltas={
        "obso_actual": {"extreme": 0.0, "near": 0.0},
        "obso_peak": {"extreme": 0.0, "near": 0.0},
        "obso_optimal": {"extreme": 0.0, "near": 0.0},
        "obso_epv_source": {"extreme": 0.0, "near": 0.0},
        "pausa_temporal": {"extreme": 0.0, "near": 0.0},
        "pausa_spatial": {"extreme": 0.0, "near": 0.0},
        "pausa_composite": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_pitch_control",
    C.generic(T.add_pitch_control),
    columns=("pitch_control_at_target__spearman",),
    velocity={
        "pitch_control_at_target__spearman": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
    },
    visibility={
        "gk_absent": {
            "pitch_control_at_target__spearman": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "pitch_control_at_target__spearman": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "pitch_control_at_target__spearman": "no_support",
    },
    applicability_deltas={
        "pitch_control_at_target__spearman": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_player_influence",
    ADAPTERS["add_player_influence"](T.add_player_influence),
    columns=(
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "off_ball_xt_diff",
        "reachable_area_team",
        "reachable_area_opponent",
        "reachable_area_diff",
    ),
    velocity={
        "actor_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
        "off_ball_xt_team": AxisVerdict("all_nan", "honest_nan"),
        "off_ball_xt_opponent": AxisVerdict("all_nan", "honest_nan"),
        "off_ball_xt_diff": AxisVerdict("all_nan", "honest_nan"),
        "reachable_area_team": AxisVerdict("all_nan", "honest_nan"),
        "reachable_area_opponent": AxisVerdict("all_nan", "honest_nan"),
        "reachable_area_diff": AxisVerdict("all_nan", "honest_nan"),
    },
    visibility={
        "gk_absent": {
            "actor_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_team": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_opponent": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_diff": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_team": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_opponent": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_diff": AxisVerdict("all_nan", "honest_nan"),
        },
        "defender_absent": {
            "actor_reachable_area_m2": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_team": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_opponent": AxisVerdict("all_nan", "honest_nan"),
            "off_ball_xt_diff": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_team": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_opponent": AxisVerdict("all_nan", "honest_nan"),
            "reachable_area_diff": AxisVerdict("all_nan", "honest_nan"),
        },
    },
    applicability={
        "actor_reachable_area_m2": "no_support",
        "off_ball_xt_team": "no_support",
        "off_ball_xt_opponent": "no_support",
        "off_ball_xt_diff": "no_support",
        "reachable_area_team": "no_support",
        "reachable_area_opponent": "no_support",
        "reachable_area_diff": "no_support",
    },
    applicability_deltas={
        "actor_reachable_area_m2": {"extreme": 0.0, "near": 0.0},
        "off_ball_xt_team": {"extreme": 0.0, "near": 0.0},
        "off_ball_xt_opponent": {"extreme": 0.0, "near": 0.0},
        "off_ball_xt_diff": {"extreme": 0.0, "near": 0.0},
        "reachable_area_team": {"extreme": 0.0, "near": 0.0},
        "reachable_area_opponent": {"extreme": 0.0, "near": 0.0},
        "reachable_area_diff": {"extreme": 0.0, "near": 0.0},
    },
)

_entry(
    "add_space_creation",
    ADAPTERS["add_space_creation"](T.add_space_creation),
    columns=(
        "space_created_m2",
        "space_denied_m2_opponent",
        "obso_epv_source",
    ),
    velocity={
        "space_created_m2": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "space_denied_m2_opponent": AxisVerdict(
            "differs",
            "differs_by_design",
            rationale=(
                "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than the "
                "velocity-informed one, but a coherent quantity rather than an invented one. A consumer should "
                "know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
            ),
        ),
        "obso_epv_source": AxisVerdict("identical", "works"),
    },
    visibility={
        "gk_absent": {
            "space_created_m2": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "space_denied_m2_opponent": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
        "defender_absent": {
            "space_created_m2": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "space_denied_m2_opponent": AxisVerdict(
                "differs",
                "differs_by_design",
                rationale=(
                    "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker than "
                    "the velocity-informed one, but a coherent quantity rather than an invented one. A consumer "
                    "should know the value is positional-only; it is not a fabrication. [measured cause=velocity]"
                ),
            ),
            "obso_epv_source": AxisVerdict("identical", "works"),
        },
    },
    applicability={
        "space_created_m2": "support_data_defined",
        "space_denied_m2_opponent": "support_data_defined",
        "obso_epv_source": "no_support",
    },
    applicability_deltas={
        "space_created_m2": {"extreme": 0.0003106691724212851, "near": 0.0007322852434015203},
        "space_denied_m2_opponent": {"extreme": 0.0002771696771546317, "near": 0.003391238345130887},
        "obso_epv_source": {"extreme": 0.0, "near": 0.0},
    },
)


# ADR-055. Every axis reads ``identical`` and every applicability probe reads ``no_support``,
# and BOTH are structural rather than lucky: ``add_visible_area_coverage`` takes no frames at
# all (``inspect.signature`` is ``(actions, *, visible_area, links)``), so neither leg's
# kinematics, nor a roster ablation, nor moving a player can reach it. Recording it is still
# worth doing -- an aggregator that answers "how much did we OBSERVE" is exactly the one a
# reader of this audit will look for, and "it does not depend on the freeze-frame's contents"
# is the finding.
#
# Measured with the synthesized fixed half-pitch polygon that ``C.visible_area_coverage``
# supplies (the SB360 fixture carries no ``visible_area`` payload), so the emitted fraction is
# 0.5 on all six rows and the comparison is over live, non-NaN values rather than a vacuous
# all-NaN pair -- ``row_identical: 6, row_nan_both: 0`` on every axis.
_VISIBLE_AREA_FRAMES_FREE = (
    "Structural, not incidental: the aggregator reads no frame, so the freeze-frame leg and the "
    "velocity-bearing leg are given identical inputs and must produce identical output. The "
    "columns describe the PROVIDER's observed region, not the scene inside it."
)

_entry(
    "add_visible_area_coverage",
    C.visible_area_coverage(T.add_visible_area_coverage),
    columns=("visible_area_fraction", "visible_area_source"),
    velocity={
        "visible_area_fraction": AxisVerdict("identical", "works", rationale=_VISIBLE_AREA_FRAMES_FREE),
        "visible_area_source": AxisVerdict("identical", "works", rationale=_VISIBLE_AREA_FRAMES_FREE),
    },
    visibility={
        "gk_absent": {
            "visible_area_fraction": AxisVerdict("identical", "works", rationale=_VISIBLE_AREA_FRAMES_FREE),
            "visible_area_source": AxisVerdict("identical", "works", rationale=_VISIBLE_AREA_FRAMES_FREE),
        },
        "defender_absent": {
            "visible_area_fraction": AxisVerdict("identical", "works", rationale=_VISIBLE_AREA_FRAMES_FREE),
            "visible_area_source": AxisVerdict("identical", "works", rationale=_VISIBLE_AREA_FRAMES_FREE),
        },
    },
    applicability={
        "visible_area_fraction": "no_support",
        "visible_area_source": "no_support",
    },
    applicability_deltas={
        "visible_area_fraction": {"extreme": 0.0, "near": 0.0},
        "visible_area_source": {"extreme": 0.0, "near": 0.0},
    },
)
