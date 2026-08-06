"""Machine-readable glossary of every derived feature column silly-kicks emits.

A pure data registry: :class:`FeatureColumn` records keyed by exact base column name. It documents
the *derived* surface (``add_*`` / ``*_xfns`` outputs, atomic mirrors, spadl enrichers, vaep features),
NOT the base schema columns (``SPADL_COLUMNS`` etc.). ``describe_level`` lives in
:mod:`silly_kicks.reporting` (a generic transform, not metadata).

Completeness + attribution are CI-gated: ``tests/test_feature_glossary_coverage.py`` (every emitted
column has an entry, discovered by inspection) and ``tests/test_feature_glossary_notice_linkage.py``
(every non-None ``attribution`` token appears verbatim in ``NOTICE``). See NOTICE for citations.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Closed unit vocabulary -- a typo or ad-hoc unit fails type-check, not just review (drift guard).
Unit = Literal[
    "metres",
    "m^2",
    "m/s",
    "m/s^2",
    "seconds",
    "degrees",
    "radians",
    "probability",
    "count",
    "xT",
    "xG",
    "ratio",
    "dimensionless",
]

# Frozen once ``dump_glossary`` output has external consumers (Hyrum): the JSON shape + column names.
GLOSSARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureColumn:
    """One derived feature column's documentation.

    Attributes
    ----------
    name :
        Exact base/semantic emitted column name (the glossary key).
    definition :
        One sentence, interpretable by a first-time reader.
    unit :
        A value from the closed :data:`Unit` vocabulary.
    emitting_module :
        The metric's home/computation module (where the logic lives), e.g.
        ``"silly_kicks.tracking._packing"`` -- NOT the monolithic ``features.py`` where the public
        producer is defined. Gate-checked for importability + not-``.features``; beyond that it is
        documentation, not gate-verified.
    attribution :
        ``None`` for house-original; else a citation TOKEN present verbatim in ``NOTICE``.
    higher_is_better :
        Direction for ``reporting.describe_level``; ``None`` when perspective-dependent / not asserted.

    Examples
    --------
    >>> fc = FeatureColumn(name="packing_made", definition="Defenders bypassed.", unit="count",
    ...                    emitting_module="silly_kicks.tracking._packing")
    >>> fc.higher_is_better is None
    True
    """

    name: str
    definition: str
    unit: Unit
    emitting_module: str
    attribution: str | None = None
    higher_is_better: bool | None = None


def _register(*entries: FeatureColumn) -> dict[str, FeatureColumn]:
    out: dict[str, FeatureColumn] = {}
    for e in entries:
        if e.name in out:
            raise ValueError(f"duplicate glossary entry: {e.name}")
        out[e.name] = e
    return out


# ---------------------------------------------------------------------------
# Home-module constants (the metric's compute home -- NEVER the features.py monolith).
# ---------------------------------------------------------------------------
_M_ACTIONTYPE = "silly_kicks.vaep.features.actiontype"
_M_RESULT = "silly_kicks.vaep.features.result"
_M_BODYPART = "silly_kicks.vaep.features.bodypart"
_M_SPATIAL = "silly_kicks.vaep.features.spatial"
_M_TEMPORAL = "silly_kicks.vaep.features.temporal"
_M_CONTEXT = "silly_kicks.vaep.features.context"
_M_SPADL_UTILS = "silly_kicks.spadl.utils"
_M_KERNELS = "silly_kicks.tracking._kernels"
_M_PLAYER_INFLUENCE = "silly_kicks.tracking._player_influence"
_M_OFF_BALL_RUNS = "silly_kicks.tracking._off_ball_runs"
_M_RUN_VALUES = "silly_kicks.tracking._run_values"
_M_STRUCTURAL = "silly_kicks.tracking._structural_pass"
_M_PACKING = "silly_kicks.tracking._packing"
_M_OBSO = "silly_kicks.tracking._obso"
_M_PAUSA = "silly_kicks.tracking._pausa"
_M_SPACE_CREATION = "silly_kicks.tracking._space_creation"
_M_TEAM_SHAPE = "silly_kicks.tracking._team_shape"
_M_SHAPE_GRAPH = "silly_kicks.tracking._shape_graph"
_M_DEFENSIVE_LINE = "silly_kicks.tracking._defensive_line"
_M_DAS = "silly_kicks.tracking._das"
_M_PRESSURE = "silly_kicks.tracking.pressure"
_M_PITCH_CONTROL = "silly_kicks.tracking.pitch_control"
_M_GK_INFLUENCE = "silly_kicks.tracking._gk_influence"
_M_COVER_SHADOWS = "silly_kicks.tracking._cover_shadows"
_M_SHOT_GOALMOUTH = "silly_kicks.tracking._shot_goalmouth"
_M_TRACKING_UTILS = "silly_kicks.tracking.utils"
_M_ELASTIC = "silly_kicks.tracking._elastic_sync"
_M_GK_GEOMETRY = "silly_kicks.tracking._gk_geometry"
_M_GK_RESOLVE = "silly_kicks.tracking._gk_resolve"
_M_GK_COMPLETION = "silly_kicks.tracking._gk_completion"
_M_GHOST_GK = "silly_kicks.tracking._ghost_gk"
_M_XT_GK = "silly_kicks.tracking._xt_gk"
_M_DEFENSIVE_CREDIT = "silly_kicks.tracking.defensive_credit._orchestration"
_M_PRESS_COMMITMENT = "silly_kicks.tracking._press_commitment"
_M_XSHOT = "silly_kicks.tracking._xshot_occurrence"
_M_XCROSS = "silly_kicks.tracking._xcross_attempt"

# Attribution TOKENS -- each MUST appear verbatim in NOTICE (notice-linkage gate, ADR-005).
_A_STRUCTURAL = "arXiv:2603.28916"  # Karakus & Arkadas 2026 (LBS/SGM/SDI; shared by packing's LBS inequality)
_A_SPEARMAN_2018 = "Spearman, W. (2018)"  # Beyond Expected Goals (OBSO / pitch-control foundation)
_A_PAUSA = "arXiv:2506.09349"  # Lee 2026 PAUSA
_A_FERNANDEZ_BORNN = "Fernandez, J., & Bornn, L. (2018)"  # Wide Open Spaces (space creation)
_A_SOTUDEH = "Sotudeh, H. (2026)"  # shape graph
_A_DEFENSIVE_LINE = "arXiv:2511.06191"  # Herold 2022 (defensive-line discriminators)
_A_TEAM_SHAPE = "Zhang, G., Kempe, M."  # Zhang 2025 (canonical team-shape metrics)
_A_DAS = "Bischofberger, J., & Baca, A. (2026)"  # Dangerous Accessible Space
_A_ANDRIENKO = "Andrienko, G."  # Andrienko 2017 oval pressure
_A_SPEARMAN_2017 = "Spearman, W., Basye, A."  # Physics-Based pass probabilities (pitch control)
_A_CASCIOLI = "Cascioli, L., Wang, A."  # cover shadows
_A_ANZER_BAUER = "Anzer, G., & Bauer, P. (2021)"  # shot xG / GK-position / xGOT lineage
_A_LUCEY = "Lucey, P., Bialkowski, A."  # defenders-in-triangle / nearest-defender
_A_POWER = "Power, P., Ruiz, H."  # receiver-zone risk/reward
_A_ELASTIC = "arXiv:2508.09238"  # Kim 2025 ELASTIC
_A_GK_GEOMETRY = "Eyestone, J. (2025)"  # xT-GK
_A_GHOST_GK = "arXiv:2406.17220"  # Dutta 2024 NFL Ghosts (RFCDE density ghosting)
_A_DEFENSIVE_CREDIT = "arXiv:2606.19931"  # Bischofberger 2026 xDT turnover sizing
_A_PRESS_COMMITMENT = "TF-51 v2 pressure-commitment cue"  # practitioner concept (PSG/Luis Enrique; Sumpter)
_A_XSHOT = "arXiv:2512.00203"  # Pipping 2026 xShotOccurrence
_A_XCROSS = "arXiv:2505.11841"  # Cao 2025 xCrossAttempt


def _onehot_entries() -> list[FeatureColumn]:
    """Generate the 171 one-hot indicator columns by looping the spadl-config vocabularies.

    Emits, in the exact order the vaep one-hot transformers produce them:
    ``actiontype_<type>`` (23), ``result_<result>`` (6), ``actiontype_<type>_result_<result>``
    (23x6=138), and ``bodypart_<bp>`` (4 -- ``foot_left``/``foot_right`` are folded into ``foot``,
    matching ``bodypart_onehot``). All are dimensionless {0, 1} indicators with no attribution.
    """
    from silly_kicks.spadl import config as spadlcfg

    entries: list[FeatureColumn] = []
    for t in spadlcfg.actiontypes:
        entries.append(
            FeatureColumn(
                name=f"actiontype_{t}",
                definition=f"One-hot indicator: 1 if the action type is '{t}', else 0.",
                unit="dimensionless",
                emitting_module=_M_ACTIONTYPE,
            )
        )
    for r in spadlcfg.results:
        entries.append(
            FeatureColumn(
                name=f"result_{r}",
                definition=f"One-hot indicator: 1 if the action result is '{r}', else 0.",
                unit="dimensionless",
                emitting_module=_M_RESULT,
            )
        )
    for t in spadlcfg.actiontypes:
        for r in spadlcfg.results:
            entries.append(
                FeatureColumn(
                    name=f"actiontype_{t}_result_{r}",
                    definition=f"One-hot indicator: 1 if the action is type '{t}' AND result '{r}', else 0.",
                    unit="dimensionless",
                    emitting_module=_M_RESULT,
                )
            )
    for b in spadlcfg.bodyparts:
        if b in ("foot_left", "foot_right"):  # folded into "foot" by bodypart_onehot
            continue
        entries.append(
            FeatureColumn(
                name=f"bodypart_{b}",
                definition=f"One-hot indicator: 1 if the action body part is '{b}', else 0.",
                unit="dimensionless",
                emitting_module=_M_BODYPART,
            )
        )
    return entries


# Authored in per-module batches; the coverage gate drives completeness (author until green).
FEATURE_GLOSSARY: dict[str, FeatureColumn] = _register(
    *_onehot_entries(),
    # -- VAEP spatial (start/end polar + movement) ---------------------------------------------
    FeatureColumn(
        name="start_dist_to_goal",
        definition="Straight-line distance (m) from the action's start location to the centre of the opponent goal.",
        unit="metres",
        emitting_module=_M_SPATIAL,
    ),
    FeatureColumn(
        name="start_angle_to_goal",
        definition="Angle (radians) of the action's start location relative to the opponent goal centre.",
        unit="radians",
        emitting_module=_M_SPATIAL,
    ),
    FeatureColumn(
        name="end_dist_to_goal",
        definition="Straight-line distance (m) from the action's end location to the centre of the opponent goal.",
        unit="metres",
        emitting_module=_M_SPATIAL,
    ),
    FeatureColumn(
        name="end_angle_to_goal",
        definition="Angle (radians) of the action's end location relative to the opponent goal centre.",
        unit="radians",
        emitting_module=_M_SPATIAL,
    ),
    FeatureColumn(
        name="movement",
        definition="Total straight-line distance (m) the ball travelled during the action (start to end).",
        unit="metres",
        emitting_module=_M_SPATIAL,
    ),
    FeatureColumn(
        name="mov",
        definition=(
            "Straight-line distance (m) between a previous action's end and the current action's start, per gamestate "
            "slot."
        ),
        unit="metres",
        emitting_module=_M_SPATIAL,
    ),
    # -- VAEP context (goalscore + team) --------------------------------------------------------
    FeatureColumn(
        name="goalscore_team",
        definition="Goals scored so far by the team performing the current action (before this action).",
        unit="count",
        emitting_module=_M_CONTEXT,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="goalscore_opponent",
        definition="Goals scored so far by the opponent of the team performing the current action.",
        unit="count",
        emitting_module=_M_CONTEXT,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="goalscore_diff",
        definition="Goal difference (own minus opponent) for the acting team before the current action.",
        unit="count",
        emitting_module=_M_CONTEXT,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="team_1",
        definition=(
            "1 if the team that performed previous action a1 is the same as the current action's team (possession "
            "kept), else 0."
        ),
        unit="dimensionless",
        emitting_module=_M_CONTEXT,
    ),
    FeatureColumn(
        name="team_2",
        definition=(
            "1 if the team that performed previous action a2 is the same as the current action's team (possession "
            "kept), else 0."
        ),
        unit="dimensionless",
        emitting_module=_M_CONTEXT,
    ),
    # -- VAEP temporal --------------------------------------------------------------------------
    FeatureColumn(
        name="time_seconds_overall",
        definition="Seconds elapsed since kickoff (period-start offsets added; previous-period stoppage ignored).",
        unit="seconds",
        emitting_module=_M_TEMPORAL,
    ),
    FeatureColumn(
        name="time_delta_1",
        definition="Seconds between the current action a0 and the previous action a1.",
        unit="seconds",
        emitting_module=_M_TEMPORAL,
    ),
    FeatureColumn(
        name="time_delta_2",
        definition="Seconds between the current action a0 and the action a2 (two actions earlier).",
        unit="seconds",
        emitting_module=_M_TEMPORAL,
    ),
    # -- SPADL enrichers (launch + game state) --------------------------------------------------
    FeatureColumn(
        name="is_launch",
        definition=(
            "True if the action is a deliberate long distribution pass (length above the long threshold), else False."
        ),
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="game_state",
        definition="Scoreline context for the acting team at the action: 'winning', 'losing', or 'drawing'.",
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    # -- Tracking action-context (TF-20) --------------------------------------------------------
    FeatureColumn(
        name="nearest_defender_distance",
        definition="Distance (m) from the acting player to the nearest opponent at the linked tracking frame.",
        unit="metres",
        emitting_module=_M_KERNELS,
        attribution=_A_LUCEY,
    ),
    FeatureColumn(
        name="actor_speed",
        definition="Speed (m/s) of the acting player at the linked tracking frame.",
        unit="m/s",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="receiver_zone_density",
        definition="Number of opponents within a fixed radius of the action's end location at the linked frame.",
        unit="count",
        emitting_module=_M_KERNELS,
        attribution=_A_POWER,
    ),
    FeatureColumn(
        name="defenders_in_triangle_to_goal",
        definition=(
            "Number of opponents inside the triangle from the action's start location to the two goalposts "
            "(shot-blocking defenders)."
        ),
        unit="count",
        emitting_module=_M_KERNELS,
        attribution=_A_LUCEY,
    ),
    # -- Pre-shot GK position + angle (TF-21 / TF-24) -------------------------------------------
    FeatureColumn(
        name="pre_shot_gk_x",
        definition="Defending goalkeeper's x-coordinate (m, LTR-normalised) at the frame before a shot.",
        unit="metres",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="pre_shot_gk_y",
        definition="Defending goalkeeper's y-coordinate (m, LTR-normalised) at the frame before a shot.",
        unit="metres",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="pre_shot_gk_distance_to_goal",
        definition="Distance (m) from the defending goalkeeper to the centre of their own goal before a shot.",
        unit="metres",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="pre_shot_gk_distance_to_shot",
        definition="Distance (m) from the defending goalkeeper to the shot location before a shot.",
        unit="metres",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="pre_shot_gk_angle_off_goal_line",
        definition=(
            "Signed angle (radians) of the defending goalkeeper off the goal line, seen from the goal centre, before a "
            "shot."
        ),
        unit="radians",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="pre_shot_gk_angle_to_shot_trajectory",
        definition=(
            "Signed angle (radians) between the defending goalkeeper and the shooter-to-goal trajectory before a shot."
        ),
        unit="radians",
        emitting_module=_M_KERNELS,
        attribution=_A_ANZER_BAUER,
    ),
    # -- Actor pre-window motion (TF-3) ---------------------------------------------------------
    FeatureColumn(
        name="actor_arc_length_pre_window",
        definition="Path length (m) travelled by the acting player over the tracking window just before the action.",
        unit="metres",
        emitting_module=_M_KERNELS,
    ),
    FeatureColumn(
        name="actor_displacement_pre_window",
        definition=(
            "Straight-line displacement (m) of the acting player over the tracking window just before the action."
        ),
        unit="metres",
        emitting_module=_M_KERNELS,
    ),
    # -- Player influence + actor reachable area (TF-33 / TF-36) --------------------------------
    FeatureColumn(
        name="actor_reachable_area_m2",
        definition="Pitch area (m^2) the acting player uniquely reaches before any opponent, at the linked frame.",
        unit="m^2",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
    ),
    FeatureColumn(
        name="reachable_area_team",
        definition=(
            "Total pitch area (m^2) uniquely reached first by the acting team's outfield players at the linked frame."
        ),
        unit="m^2",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="reachable_area_opponent",
        definition=(
            "Total pitch area (m^2) uniquely reached first by the opponent's outfield players at the linked frame."
        ),
        unit="m^2",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="reachable_area_diff",
        definition="Acting team minus opponent uniquely-reachable area (m^2) at the linked frame.",
        unit="m^2",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="off_ball_xt_team",
        definition="Acting team's summed off-ball threat (pitch-control share x expected threat) at the linked frame.",
        unit="xT",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="off_ball_xt_opponent",
        definition="Opponent's summed off-ball threat (pitch-control share x expected threat) at the linked frame.",
        unit="xT",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="off_ball_xt_diff",
        definition="Acting team minus opponent off-ball threat (xT) at the linked frame.",
        unit="xT",
        emitting_module=_M_PLAYER_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    # -- Off-ball runs + line break (TF-4) ------------------------------------------------------
    FeatureColumn(
        name="n_off_ball_runners_pre_window",
        definition="Number of attacking teammates making a qualifying off-ball run in the window before the action.",
        unit="count",
        emitting_module=_M_OFF_BALL_RUNS,
    ),
    FeatureColumn(
        name="max_off_ball_run_displacement_pre_window",
        definition="Largest displacement (m) among the attacking teammates' off-ball runs before the action.",
        unit="metres",
        emitting_module=_M_OFF_BALL_RUNS,
    ),
    FeatureColumn(
        name="mean_off_ball_run_speed_pre_window",
        definition="Mean speed (m/s) of the attacking teammates' off-ball runs before the action.",
        unit="m/s",
        emitting_module=_M_OFF_BALL_RUNS,
    ),
    FeatureColumn(
        name="n_off_ball_runners_toward_goal_pre_window",
        definition="Number of off-ball runners moving toward the attacked goal before the action.",
        unit="count",
        emitting_module=_M_OFF_BALL_RUNS,
    ),
    FeatureColumn(
        name="line_break",
        definition="True if the action's trajectory breaks through the opponent's defensive line, else False.",
        unit="dimensionless",
        emitting_module=_M_OFF_BALL_RUNS,
    ),
    FeatureColumn(
        name="n_attackers_behind_line",
        definition="Number of attacking players positioned beyond the opponent's defensive line at the action.",
        unit="count",
        emitting_module=_M_OFF_BALL_RUNS,
    ),
    # -- Off-ball run valuation (TF-35) ---------------------------------------------------------
    FeatureColumn(
        name="run_value_target",
        definition=(
            "Value (pitch control x threat) of the dangerous space controlled by the off-ball run that receives the "
            "pass."
        ),
        unit="xT",
        emitting_module=_M_RUN_VALUES,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="run_value_enabled_pass",
        definition="Threat of the space the pass exploits, attributed to the enabling off-ball run.",
        unit="xT",
        emitting_module=_M_RUN_VALUES,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="run_value_disruptive_sum",
        definition="Summed value of the disruptive off-ball runs that pulled defenders away on this action.",
        unit="xT",
        emitting_module=_M_RUN_VALUES,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="n_disruptive_runs",
        definition="Number of off-ball runs classified as disruptive (not the pass target) on this action.",
        unit="count",
        emitting_module=_M_RUN_VALUES,
    ),
    FeatureColumn(
        name="n_valued_disruptive_runs",
        definition="Number of disruptive runs whose value was resolvable (runner present in the linked frame).",
        unit="count",
        emitting_module=_M_RUN_VALUES,
    ),
    # -- Structural pass primitives (TF-45) -----------------------------------------------------
    FeatureColumn(
        name="structural_lbs",
        definition=(
            "Line Bypass Score: number of opponents removed from play along the attacking axis by a completed "
            "pass/carry."
        ),
        unit="count",
        emitting_module=_M_STRUCTURAL,
        attribution=_A_STRUCTURAL,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="structural_sgm",
        definition=(
            "Space Gain Metric: increase in open space (inverse defender density) from the passer to the receiver "
            "location."
        ),
        unit="dimensionless",
        emitting_module=_M_STRUCTURAL,
        attribution=_A_STRUCTURAL,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="structural_sdi",
        definition=(
            "Structural Disruption Index: change (m) in the receiver's distance from the opponents' positional "
            "centroid due to the pass."
        ),
        unit="metres",
        emitting_module=_M_STRUCTURAL,
        attribution=_A_STRUCTURAL,
        higher_is_better=True,
    ),
    # -- Packing (TF-49) ------------------------------------------------------------------------
    FeatureColumn(
        name="packing_made",
        definition=(
            "Number of opponents bypassed (taken out of the defensive phase) by a completed pass, cross, or carry."
        ),
        unit="count",
        emitting_module=_M_PACKING,
        attribution=_A_STRUCTURAL,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="packing_net",
        definition="Directional packing: bypassed opponents weighted +1 forward / +0.5 sideways / -1 backward.",
        unit="count",
        emitting_module=_M_PACKING,
        attribution=_A_STRUCTURAL,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="packing_goal_threat",
        definition=(
            "Number of bypassed opponents who were among the last defensive-line players (goal-threatening packing)."
        ),
        unit="count",
        emitting_module=_M_PACKING,
        attribution=_A_STRUCTURAL,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="packing_secured",
        definition="True if the packed pass was received and secured by a teammate, else False.",
        unit="dimensionless",
        emitting_module=_M_PACKING,
    ),
    FeatureColumn(
        name="packing_receiver_player_id",
        definition="Identifier of the teammate who receives the packed pass (NaN if unresolved).",
        unit="dimensionless",
        emitting_module=_M_PACKING,
    ),
    # -- OBSO (TF-40) ---------------------------------------------------------------------------
    FeatureColumn(
        name="obso_actual",
        definition=(
            "Off-Ball Scoring Opportunity at the actual pass target (pitch control x ball transition x expected "
            "threat)."
        ),
        unit="probability",
        emitting_module=_M_OBSO,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="obso_optimal",
        definition="Best Off-Ball Scoring Opportunity available across all teammate positions at the action frame.",
        unit="probability",
        emitting_module=_M_OBSO,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="obso_peak",
        definition="Maximum Off-Ball Scoring Opportunity at the pass target across the frame window.",
        unit="probability",
        emitting_module=_M_OBSO,
        attribution=_A_SPEARMAN_2018,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="obso_epv_source",
        definition=(
            "Provenance of the expected-possession-value surface used for OBSO: 'xt', 'synthetic', or 'injected'."
        ),
        unit="dimensionless",
        emitting_module=_M_OBSO,
    ),
    # -- PAUSA (TF-42) --------------------------------------------------------------------------
    FeatureColumn(
        name="pausa_temporal",
        definition="Temporal pass judgment: actual OBSO / peak OBSO in the window (1.0 = released at the best moment).",
        unit="ratio",
        emitting_module=_M_PAUSA,
        attribution=_A_PAUSA,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="pausa_spatial",
        definition="Spatial pass selection: actual OBSO / best available OBSO (1.0 = the best receiver was chosen).",
        unit="ratio",
        emitting_module=_M_PAUSA,
        attribution=_A_PAUSA,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="pausa_composite",
        definition="PAUSA pass quality: temporal x spatial (1.0 = perfectly timed and targeted).",
        unit="ratio",
        emitting_module=_M_PAUSA,
        attribution=_A_PAUSA,
        higher_is_better=True,
    ),
    # -- Space creation (TF-41) -----------------------------------------------------------------
    FeatureColumn(
        name="space_created_m2",
        definition=(
            "Off-ball scoring space (m^2 equivalent) the acting player generates for the attack (leave-one-out OBSO)."
        ),
        unit="m^2",
        emitting_module=_M_SPACE_CREATION,
        attribution=_A_FERNANDEZ_BORNN,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="space_denied_m2_opponent",
        definition="Off-ball scoring space (m^2 equivalent) the acting player denies the opponent in rest-defence.",
        unit="m^2",
        emitting_module=_M_SPACE_CREATION,
        attribution=_A_FERNANDEZ_BORNN,
        higher_is_better=True,
    ),
    # -- Team shape envelope (TF-31 / TF-44) ----------------------------------------------------
    FeatureColumn(
        name="team_shape_centroid_x_attacking",
        definition="Attacking team's positional centroid x-coordinate (m) at the linked frame.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_centroid_x_defending",
        definition="Defending team's positional centroid x-coordinate (m) at the linked frame.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_centroid_y_attacking",
        definition="Attacking team's positional centroid y-coordinate (m) at the linked frame.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_centroid_y_defending",
        definition="Defending team's positional centroid y-coordinate (m) at the linked frame.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_convex_hull_area_attacking",
        definition="Area (m^2) of the convex hull enclosing the attacking team's outfield players.",
        unit="m^2",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_convex_hull_area_defending",
        definition="Area (m^2) of the convex hull enclosing the defending team's outfield players.",
        unit="m^2",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_team_length_attacking",
        definition="Longitudinal spread (m, x-extent) of the attacking team's outfield players.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_team_length_defending",
        definition="Longitudinal spread (m, x-extent) of the defending team's outfield players.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_team_width_attacking",
        definition="Lateral spread (m, y-extent) of the attacking team's outfield players.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_team_width_defending",
        definition="Lateral spread (m, y-extent) of the defending team's outfield players.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_stretch_index_attacking",
        definition="Mean distance (m) of the attacking team's players from their centroid (dispersion).",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_stretch_index_defending",
        definition="Mean distance (m) of the defending team's players from their centroid (dispersion).",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_defensive_line_height_attacking",
        definition="Height (m up the pitch) of the attacking team's deepest defensive line.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_defensive_line_height_defending",
        definition="Height (m up the pitch) of the defending team's deepest defensive line.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_inter_line_gap_1_attacking",
        definition="Gap (m) between the first and second defensive lines of the attacking team.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_inter_line_gap_1_defending",
        definition="Gap (m) between the first and second defensive lines of the defending team.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_inter_line_gap_2_attacking",
        definition="Gap (m) between the second and third defensive lines of the attacking team.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_inter_line_gap_2_defending",
        definition="Gap (m) between the second and third defensive lines of the defending team.",
        unit="metres",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_n_outfield_players_attacking",
        definition="Number of the attacking team's outfield players visible in the frame.",
        unit="count",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    FeatureColumn(
        name="team_shape_n_outfield_players_defending",
        definition="Number of the defending team's outfield players visible in the frame.",
        unit="count",
        emitting_module=_M_TEAM_SHAPE,
        attribution=_A_TEAM_SHAPE,
    ),
    # -- Shape graph (TF-39) --------------------------------------------------------------------
    FeatureColumn(
        name="shape_graph_density_attacking",
        definition="Edge density of the attacking team's shape graph (realised vs possible proximity connections).",
        unit="ratio",
        emitting_module=_M_SHAPE_GRAPH,
        attribution=_A_SOTUDEH,
    ),
    FeatureColumn(
        name="shape_graph_density_defending",
        definition="Edge density of the defending team's shape graph (realised vs possible proximity connections).",
        unit="ratio",
        emitting_module=_M_SHAPE_GRAPH,
        attribution=_A_SOTUDEH,
    ),
    FeatureColumn(
        name="shape_graph_mean_stability_attacking",
        definition="Mean angular stability of the attacking team's shape-graph edges (formation rigidity).",
        unit="dimensionless",
        emitting_module=_M_SHAPE_GRAPH,
        attribution=_A_SOTUDEH,
    ),
    FeatureColumn(
        name="shape_graph_mean_stability_defending",
        definition="Mean angular stability of the defending team's shape-graph edges (formation rigidity).",
        unit="dimensionless",
        emitting_module=_M_SHAPE_GRAPH,
        attribution=_A_SOTUDEH,
    ),
    FeatureColumn(
        name="shape_graph_n_edges_attacking",
        definition="Number of edges in the attacking team's stable shape graph.",
        unit="count",
        emitting_module=_M_SHAPE_GRAPH,
        attribution=_A_SOTUDEH,
    ),
    FeatureColumn(
        name="shape_graph_n_edges_defending",
        definition="Number of edges in the defending team's stable shape graph.",
        unit="count",
        emitting_module=_M_SHAPE_GRAPH,
        attribution=_A_SOTUDEH,
    ),
    # -- Defensive line geometry (TF-14) --------------------------------------------------------
    FeatureColumn(
        name="defensive_line_x",
        definition="Mean x-coordinate (m) of the opponent's back line at the linked frame.",
        unit="metres",
        emitting_module=_M_DEFENSIVE_LINE,
        attribution=_A_DEFENSIVE_LINE,
    ),
    FeatureColumn(
        name="back_line_high_x",
        definition="x-coordinate (m) of the highest (most advanced) back-line defender.",
        unit="metres",
        emitting_module=_M_DEFENSIVE_LINE,
        attribution=_A_DEFENSIVE_LINE,
    ),
    FeatureColumn(
        name="back_n_count",
        definition="Number of players identified in the opponent's back line.",
        unit="count",
        emitting_module=_M_DEFENSIVE_LINE,
        attribution=_A_DEFENSIVE_LINE,
    ),
    FeatureColumn(
        name="compactness_x",
        definition="Longitudinal spread (m) of the opponent's back line (front-to-back compactness).",
        unit="metres",
        emitting_module=_M_DEFENSIVE_LINE,
        attribution=_A_DEFENSIVE_LINE,
    ),
    FeatureColumn(
        name="lateral_width",
        definition="Lateral spread (m) of the opponent's back line across the pitch.",
        unit="metres",
        emitting_module=_M_DEFENSIVE_LINE,
        attribution=_A_DEFENSIVE_LINE,
    ),
    FeatureColumn(
        name="max_lateral_gap",
        definition="Largest lateral gap (m) between adjacent back-line defenders.",
        unit="metres",
        emitting_module=_M_DEFENSIVE_LINE,
        attribution=_A_DEFENSIVE_LINE,
    ),
    # -- Dangerous Accessible Space (TF-28) -----------------------------------------------------
    FeatureColumn(
        name="das_team",
        definition="Dangerous Accessible Space (m^2) controlled by the acting team at the linked frame.",
        unit="m^2",
        emitting_module=_M_DAS,
        attribution=_A_DAS,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="das_opponent",
        definition="Dangerous Accessible Space (m^2) controlled by the opponent at the linked frame.",
        unit="m^2",
        emitting_module=_M_DAS,
        attribution=_A_DAS,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="das_diff",
        definition="Acting team minus opponent Dangerous Accessible Space (m^2).",
        unit="m^2",
        emitting_module=_M_DAS,
        attribution=_A_DAS,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="das_source",
        definition=(
            "Provenance of the DAS value: computed / unlinked / unscoreable_frame / team_unresolved / unscoreable_call."
        ),
        unit="dimensionless",
        emitting_module=_M_DAS,
        attribution=_A_DAS,
    ),
    # -- Pressure on actor (TF-2) ---------------------------------------------------------------
    FeatureColumn(
        name="pressure_on_actor",
        definition="Pressure exerted on the ball carrier by nearby opponents (default model); higher = more pressure.",
        unit="dimensionless",
        emitting_module=_M_PRESSURE,
        attribution=_A_ANDRIENKO,
    ),
    FeatureColumn(
        name="pressure_on_actor__andrienko_oval",
        definition="Pressure on the ball carrier from the Andrienko 2017 directional-oval model.",
        unit="dimensionless",
        emitting_module=_M_PRESSURE,
        attribution=_A_ANDRIENKO,
    ),
    # -- Pitch control at target (TF-7) ---------------------------------------------------------
    FeatureColumn(
        name="pitch_control_at_target__spearman",
        definition=(
            "Probability the acting team controls the action's target location (Spearman 2017 pitch-control model)."
        ),
        unit="probability",
        emitting_module=_M_PITCH_CONTROL,
        attribution=_A_SPEARMAN_2017,
    ),
    # -- GK influence (TF-15) -------------------------------------------------------------------
    FeatureColumn(
        name="gk_pitch_control_share_weighted",
        definition=(
            "Defending goalkeeper's threat-weighted share of pitch control at the frame (dominance where it matters)."
        ),
        unit="ratio",
        emitting_module=_M_GK_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
    ),
    FeatureColumn(
        name="gk_reachable_area_m2",
        definition="Pitch area (m^2) the defending goalkeeper uniquely reaches first at the frame.",
        unit="m^2",
        emitting_module=_M_GK_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
    ),
    FeatureColumn(
        name="gk_closing_time_min_s__six_yard_box",
        definition="Minimum time (s) for the defending goalkeeper to reach the six-yard-box zone.",
        unit="seconds",
        emitting_module=_M_GK_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
    ),
    FeatureColumn(
        name="gk_closing_time_mean_s__six_yard_box",
        definition="Mean time (s) for the defending goalkeeper to reach points across the six-yard-box zone.",
        unit="seconds",
        emitting_module=_M_GK_INFLUENCE,
        attribution=_A_SPEARMAN_2018,
    ),
    # -- Cover shadows (TF-30) ------------------------------------------------------------------
    FeatureColumn(
        name="blocking_score",
        definition=(
            "Threat reduction from opponents' cover shadows blocking passing lanes to potential receivers. "
            "Non-negative BY CONSTRUCTION (clamped at zero), so it cannot express a defender whose "
            "positioning made things WORSE -- unlike the paper's SoccerMap-CNN counterfactual, which is "
            "signed. The underlying unclamped difference is argued non-negative structurally for the "
            "spearman and voronoi pitch-control methods, and verified empirically for fernandez_bornn."
        ),
        unit="xT",
        emitting_module=_M_COVER_SHADOWS,
        attribution=_A_CASCIOLI,
    ),
    FeatureColumn(
        name="n_blocked_receivers",
        definition="Number of potential receivers whose passing lane is blocked by a defender's cover shadow.",
        unit="count",
        emitting_module=_M_COVER_SHADOWS,
        attribution=_A_CASCIOLI,
    ),
    FeatureColumn(
        name="max_single_defender_blocking_score",
        definition=(
            "Largest blocking-score contribution from any single defender on this action. "
            "Non-negative BY CONSTRUCTION (clamped at zero), so it cannot express a defender whose "
            "positioning made things WORSE -- unlike the paper's SoccerMap-CNN counterfactual, which is "
            "signed. The underlying unclamped difference is argued non-negative structurally for the "
            "spearman and voronoi pitch-control methods, and verified empirically for fernandez_bornn."
        ),
        unit="xT",
        emitting_module=_M_COVER_SHADOWS,
        attribution=_A_CASCIOLI,
    ),
    FeatureColumn(
        name="blocked_threat_fraction",
        definition=(
            "Fraction of the total receiver threat blocked by opponents' cover shadows. "
            "Non-negative BY CONSTRUCTION (its numerator is the clamped blocking_score), so it cannot "
            "express a defender whose positioning made things WORSE -- unlike the paper's SoccerMap-CNN "
            "counterfactual, which is signed. The underlying unclamped difference is argued non-negative "
            "structurally for the spearman and voronoi pitch-control methods, and verified empirically "
            "for fernandez_bornn."
        ),
        unit="ratio",
        emitting_module=_M_COVER_SHADOWS,
        attribution=_A_CASCIOLI,
    ),
    FeatureColumn(
        name="max_single_defender_player_id",
        definition=(
            "Identity of the defender producing max_single_defender_blocking_score. "
            "POPULATED ONLY when add_cover_shadows is called with detailed=True; NA on the default "
            "detailed=False path, and NA wherever no defender earned an attribution. The cheap path "
            "can compute an identity but deliberately does not serve one: measured against the exact "
            "path on 970 qualifying actions it agreed only 0.157 of the time (95% CI [0.135, 0.181]) "
            "versus ~0.10 by chance, and the disagreements are not near-ties -- the median names a "
            "defender worth 1.6% of the true winner, and at the 90th percentile the named defender's "
            "exact contribution is zero. This is not a defect: the cheap path is faithful to a "
            "lane-based notion of 'blocks most' and the exact path to a pitch-control counterfactual, "
            "and the two rank the top of the list differently. Evidence: "
            "docs/research/cover_shadow_identity/."
        ),
        # No identifier token exists in the closed Unit vocabulary; "dimensionless" is the
        # least-wrong fit. Widening the Literal for one column would be a public contract change
        # (the speed_source -> "unavailable" precedent) that does not earn its cost here.
        unit="dimensionless",
        emitting_module=_M_COVER_SHADOWS,
        attribution=_A_CASCIOLI,
        # An identity has no direction -- "higher is better" is meaningless for a player id. Decided,
        # not forgotten; the five sibling cover-shadow columns are None for a different reason
        # (direction flips by perspective).
        higher_is_better=None,
    ),
    FeatureColumn(
        name="n_potential_receivers",
        definition=(
            "Number of teammates considered as potential pass receivers for the cover-shadow blocking computation."
        ),
        unit="count",
        emitting_module=_M_COVER_SHADOWS,
        attribution=_A_CASCIOLI,
    ),
    # -- Post-shot goalmouth geometry (TF-48) ---------------------------------------------------
    FeatureColumn(
        name="shot_crossing_y",
        definition=(
            "y-coordinate (m) where the shot's fitted trajectory crosses the goal plane (canonical attacked goal at "
            "x=105)."
        ),
        unit="metres",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_crossing_z",
        definition="Height (m) at which the shot crosses the goal plane.",
        unit="metres",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_crossing_confidence",
        definition="Confidence (0-1) of the fitted goal-plane crossing point.",
        unit="ratio",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_crossing_source",
        definition="Provenance of the crossing estimate (trajectory-fit method or fallback).",
        unit="dimensionless",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_fit_end_reason",
        definition="Why the trajectory-fit window ended: goal-plane straddle, trajectory break, or window cap.",
        unit="dimensionless",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_fit_n_frames",
        definition="Number of tracking frames used to fit the post-contact shot trajectory.",
        unit="count",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_fit_rmse",
        definition="Root-mean-square residual (m) of the shot-trajectory fit (lower = cleaner fit).",
        unit="metres",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="shot_on_target_derived",
        definition="True if the fitted trajectory crosses within the goal frame (on target), else False.",
        unit="dimensionless",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_speed",
        definition="Ball speed (m/s) over the contact segment of the shot.",
        unit="m/s",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_time_to_goal_line",
        definition="Time (s) for the shot to travel from contact to the goal plane.",
        unit="seconds",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    FeatureColumn(
        name="shot_z_profile",
        definition="Flight-height profile class of the shot: rolling, airborne, or bounced.",
        unit="dimensionless",
        emitting_module=_M_SHOT_GOALMOUTH,
        attribution=_A_ANZER_BAUER,
    ),
    # -- Event/tracking sync quality (TF-6) -----------------------------------------------------
    FeatureColumn(
        name="sync_score_mean",
        definition="Mean event-to-tracking synchronization quality over the linked window (1.0 = perfectly aligned).",
        unit="ratio",
        emitting_module=_M_TRACKING_UTILS,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="sync_score_min",
        definition="Worst-case event-to-tracking synchronization quality over the linked window.",
        unit="ratio",
        emitting_module=_M_TRACKING_UTILS,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="sync_score_high_quality_frac",
        definition="Fraction of the linked window with high-quality event-to-tracking synchronization.",
        unit="ratio",
        emitting_module=_M_TRACKING_UTILS,
        higher_is_better=True,
    ),
    # -- ELASTIC sync (TF-43) -------------------------------------------------------------------
    FeatureColumn(
        name="elastic_frame_id",
        definition="Tracking frame the ELASTIC refinement aligns this action to.",
        unit="dimensionless",
        emitting_module=_M_ELASTIC,
        attribution=_A_ELASTIC,
    ),
    FeatureColumn(
        name="elastic_confidence",
        definition="Confidence (0-1) of the ELASTIC event-to-frame alignment.",
        unit="ratio",
        emitting_module=_M_ELASTIC,
        attribution=_A_ELASTIC,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="elastic_error_seconds",
        definition="Estimated time misalignment (s) between the action and its ELASTIC-aligned frame.",
        unit="seconds",
        emitting_module=_M_ELASTIC,
        attribution=_A_ELASTIC,
        higher_is_better=False,
    ),
    # -- Restart-coordinate enrichment (ADR-025) ------------------------------------------------
    FeatureColumn(
        name="enriched_start_x",
        definition=(
            "Imputed action start x-coordinate (m) for restarts with a missing native coordinate (canonical start_x is "
            "never overwritten)."
        ),
        unit="metres",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="enriched_start_y",
        definition="Imputed action start y-coordinate (m) for restarts with a missing native coordinate.",
        unit="metres",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="enriched_end_x",
        definition="Imputed action end x-coordinate (m) for restarts with a missing native coordinate.",
        unit="metres",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="enriched_end_y",
        definition="Imputed action end y-coordinate (m) for restarts with a missing native coordinate.",
        unit="metres",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="start_coord_source",
        definition=(
            "Provenance tier of the imputed start coordinate (native / tracking_ball / tracking_gk / restart_prior / "
            "next_event / unresolved / tripwire_reverted)."
        ),
        unit="dimensionless",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="start_coord_confidence",
        definition="Confidence (0-1) of the imputed start coordinate.",
        unit="ratio",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="end_coord_source",
        definition=(
            "Provenance tier of the imputed end coordinate (native / tracking_ball / next_event / unresolved / etc.)."
        ),
        unit="dimensionless",
        emitting_module=_M_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="end_coord_confidence",
        definition="Confidence (0-1) of the imputed end coordinate.",
        unit="ratio",
        emitting_module=_M_GK_GEOMETRY,
    ),
    # -- Possession segmentation ----------------------------------------------------------------
    FeatureColumn(
        name="possession_id",
        definition="Sequential identifier grouping consecutive actions into a single team possession.",
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    # -- Defending-GK resolution (TF-13) --------------------------------------------------------
    FeatureColumn(
        name="defending_gk_player_id",
        definition=(
            "Identifier of the goalkeeper defending against the action (resolved from frames/roster; NaN before "
            "engagement)."
        ),
        unit="dimensionless",
        emitting_module=_M_GK_RESOLVE,
    ),
    # -- GK role + pre-shot GK context (spadl enrichers) ----------------------------------------
    FeatureColumn(
        name="gk_role",
        definition="Goalkeeper's role for the action, e.g. 'distribution' (NaN for non-keeper actions).",
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="gk_was_distributing",
        definition="True if the defending goalkeeper was distributing the ball in the possession preceding the shot.",
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="gk_was_engaged",
        definition="True if the defending goalkeeper was actively engaged in play before the shot.",
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="gk_actions_in_possession",
        definition="Number of goalkeeper actions in the possession preceding the shot.",
        unit="count",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="gk_pass_length_m",
        definition="Length (m) of the goalkeeper distribution pass (NaN on non-distribution rows).",
        unit="metres",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="gk_pass_length_class",
        definition="Goalkeeper distribution length class: short, medium, or long.",
        unit="dimensionless",
        emitting_module=_M_SPADL_UTILS,
    ),
    FeatureColumn(
        name="gk_xt_delta",
        definition="Expected-threat gain of a successful goalkeeper distribution (xT of end zone minus start zone).",
        unit="xT",
        emitting_module=_M_SPADL_UTILS,
    ),
    # -- GK distribution completion model -------------------------------------------------------
    FeatureColumn(
        name="gk_completion",
        definition="Modelled probability that the goalkeeper's distribution is completed to a teammate.",
        unit="probability",
        emitting_module=_M_GK_COMPLETION,
        higher_is_better=True,
    ),
    # -- Ghost-GK positioning model (TF-18) -----------------------------------------------------
    FeatureColumn(
        name="ghost_gk_x",
        definition=(
            "League-average 'ghost' goalkeeper's expected x-coordinate (m) for the same game state (positioning "
            "baseline)."
        ),
        unit="metres",
        emitting_module=_M_GHOST_GK,
        attribution=_A_GHOST_GK,
    ),
    FeatureColumn(
        name="ghost_gk_y",
        definition=(
            "League-average 'ghost' goalkeeper's expected y-coordinate (m) for the same game state (positioning "
            "baseline)."
        ),
        unit="metres",
        emitting_module=_M_GHOST_GK,
        attribution=_A_GHOST_GK,
    ),
    FeatureColumn(
        name="ghost_gk_source",
        definition=(
            "Provenance of the ghost-GK position: computed / velocity_unavailable (the frame source "
            "structurally cannot carry kinematics, so the model is refused rather than served "
            "imputed features) / no_keeper (the action reached a frame carrying no defending "
            "goalkeeper) / unlinked (the action reached no frame)."
        ),
        unit="dimensionless",
        emitting_module=_M_GHOST_GK,
        attribution=_A_GHOST_GK,
    ),
    # -- xT-GK v1 (Eyestone) --------------------------------------------------------------------
    FeatureColumn(
        name="xt_gk",
        definition="Expected Threat for Goalkeepers: composite value of a goalkeeper distribution action.",
        unit="xT",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_base",
        definition="Origin term of xT-GK: negative expected threat of the ball's starting zone.",
        unit="xT",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_pev",
        definition="Pressure-Escape Value: threat gained by moving the ball to a safer zone under pressure.",
        unit="xT",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_rav",
        definition=(
            "Risk-Adjusted Value: completion-probability-weighted expected threat of the distribution's destination."
        ),
        unit="xT",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_dzv",
        definition="Defensive-Zone Value: revaluation increment applied to distributions from deep defensive zones.",
        unit="xT",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_pressure",
        definition="Pressure on the goalkeeper at the moment of distribution (feeds the pressure-escape term).",
        unit="dimensionless",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_origin_x",
        definition=(
            "Resolved x-coordinate (m) of the distribution's origin used for the grid lookup (often imputed for "
            "goal-kicks)."
        ),
        unit="metres",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_origin_y",
        definition="Resolved y-coordinate (m) of the distribution's origin used for the grid lookup.",
        unit="metres",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_dest_x",
        definition="Resolved x-coordinate (m) of the distribution's destination used for the grid lookup.",
        unit="metres",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_dest_y",
        definition="Resolved y-coordinate (m) of the distribution's destination used for the grid lookup.",
        unit="metres",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_origin_source",
        definition=(
            "Provenance tier of the resolved origin (native / tracking_gk / goalkick_prior / next_event / unresolved)."
        ),
        unit="dimensionless",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_dest_source",
        definition="Provenance tier of the resolved destination (native / next_event / unresolved).",
        unit="dimensionless",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_origin_confidence",
        definition="Confidence (0-1) of the resolved origin coordinate.",
        unit="ratio",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_completion_variant",
        definition="Which completion-model variant scored the distribution (e.g. gs / skillcorner).",
        unit="dimensionless",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_completion_source",
        definition="Whether the completion probability came from the model or a per-type base rate.",
        unit="dimensionless",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    FeatureColumn(
        name="xt_gk_native_goalkick_out_of_region",
        definition="Data-quality flag: True if the provider's native goal-kick origin fell outside the goal area.",
        unit="dimensionless",
        emitting_module=_M_XT_GK,
        attribution=_A_GK_GEOMETRY,
    ),
    # -- Per-event defensive credit (TF-51) -----------------------------------------------------
    FeatureColumn(
        name="defensive_credit_net",
        definition="Net signed defensive credit for the defending team on this action (credit minus debit).",
        unit="xT",
        emitting_module=_M_DEFENSIVE_CREDIT,
        attribution=_A_DEFENSIVE_CREDIT,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="defensive_credit_plus",
        definition="Sum of positive defensive credit awarded to the defending team on this action.",
        unit="xT",
        emitting_module=_M_DEFENSIVE_CREDIT,
        attribution=_A_DEFENSIVE_CREDIT,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="defensive_credit_minus",
        definition="Sum of negative defensive credit (debit) charged to the defending team on this action.",
        unit="xT",
        emitting_module=_M_DEFENSIVE_CREDIT,
        attribution=_A_DEFENSIVE_CREDIT,
    ),
    FeatureColumn(
        name="n_defensive_credits",
        definition="Number of defensive credit/debit rows attributed on this action.",
        unit="count",
        emitting_module=_M_DEFENSIVE_CREDIT,
        attribution=_A_DEFENSIVE_CREDIT,
    ),
    # -- Pressure-commitment cue (TF-51 v2 Item 5) ----------------------------------------------
    FeatureColumn(
        name="press_commitment",
        definition=(
            "Least-squares slope of the pressing defender's closing-speed over the pre-action window; "
            "positive = COMMITS (drives in), negative = CONTAINS (jockeys/brakes)."
        ),
        unit="m/s^2",
        emitting_module=_M_PRESS_COMMITMENT,
        attribution=_A_PRESS_COMMITMENT,
        higher_is_better=None,  # style descriptor -- neither committing nor containing is universally better
    ),
    FeatureColumn(
        name="press_commitment_closing_speed",
        definition="Pressing defender's closing speed toward the actor at the action frame (context).",
        unit="m/s",
        emitting_module=_M_PRESS_COMMITMENT,
        attribution=_A_PRESS_COMMITMENT,
        higher_is_better=None,
    ),
    FeatureColumn(
        name="press_commitment_source",
        definition=(
            "Provenance of the cue: computed / no_pressing_defender / velocity_unavailable / "
            "window_too_short / degenerate_axis / unlinked."
        ),
        unit="dimensionless",
        emitting_module=_M_PRESS_COMMITMENT,
        attribution=_A_PRESS_COMMITMENT,
    ),
    # -- Trained state-anchored models (TF-16 / TF-17) ------------------------------------------
    FeatureColumn(
        name="xshot_occurrence",
        definition="Modelled probability that the in-possession team attempts a shot within ~1 s of the frame.",
        unit="probability",
        emitting_module=_M_XSHOT,
        attribution=_A_XSHOT,
    ),
    FeatureColumn(
        name="xcross_attempt",
        definition="Modelled probability that the in-possession team attempts a cross within ~1 s of the frame.",
        unit="probability",
        emitting_module=_M_XCROSS,
        attribution=_A_XCROSS,
    ),
)


def glossary_entry(name: str) -> FeatureColumn:
    """Return the :class:`FeatureColumn` for ``name`` (raises ``KeyError`` if undocumented).

    Examples
    --------
    Look up a documented column::

        from silly_kicks.feature_glossary import glossary_entry
        entry = glossary_entry("packing_made")
        entry.unit  # -> "count"
    """
    return FEATURE_GLOSSARY[name]


def undocumented_columns(cols: Iterable[str]) -> set[str]:
    """Return the subset of ``cols`` that have no glossary entry.

    Examples
    --------
    >>> undocumented_columns(["not_a_real_column"])
    {'not_a_real_column'}
    """
    return {c for c in cols if c not in FEATURE_GLOSSARY}


def glossary_to_json() -> str:
    """Serialise the glossary to JSON (pure; no I/O).

    Shape: ``{"schema_version": ..., "columns": {name: {definition, unit, emitting_module,
    attribution, higher_is_better}}}``.

    Examples
    --------
    >>> import json
    >>> json.loads(glossary_to_json())["schema_version"]
    '1.0'
    """
    columns = {
        fc.name: {
            "definition": fc.definition,
            "unit": fc.unit,
            "emitting_module": fc.emitting_module,
            "attribution": fc.attribution,
            "higher_is_better": fc.higher_is_better,
        }
        for fc in FEATURE_GLOSSARY.values()
    }
    return json.dumps(
        {"schema_version": GLOSSARY_SCHEMA_VERSION, "columns": columns},
        indent=2,
        sort_keys=True,
    )


def dump_glossary(path) -> None:
    """Write :func:`glossary_to_json` to ``path`` (the only impure symbol here).

    Examples
    --------
    Export for a language-agnostic consumer::

        from silly_kicks.feature_glossary import dump_glossary
        dump_glossary("feature_glossary.json")
    """
    Path(path).write_text(glossary_to_json(), encoding="utf-8")


def emitting_module_is_importable(name: str) -> bool:
    """True iff the dotted module ``name`` imports cleanly (no dead references).

    Examples
    --------
    >>> emitting_module_is_importable("silly_kicks.tracking._packing")
    True
    """
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False
