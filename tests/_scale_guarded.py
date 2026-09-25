"""Registry of scale-guarded primitives + AST discovery of group_rows callers (ADR-073, spec 4.3).

Every function that calls ``group_rows`` must appear in ``SCALE_GUARDED`` (the forcing function --
a new caller with no growth guard fails CI). ``DEGENERATE_OK`` marks entries whose counter is zero
by design (e.g. a fully-vectorized primitive); each MUST carry a discriminating companion.
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: All guard tests live in this module (values below are function names within it).
SCALE_GUARD_MODULE = "tests.test_scale_guards"

#: guarded primitive qualname -> the guard test's function name (growth or constant).
SCALE_GUARDED: dict[str, str] = {
    "silly_kicks.causal._confounders._pressure_at_entry": "test__pressure_at_entry_is_subquadratic",
    "silly_kicks.causal.opportunities.build_opportunities": "test_build_opportunities_is_subquadratic",
    "silly_kicks.tracking.defensive_credit._orchestration.compute_defensive_credits": (
        "test_compute_defensive_credits_is_subquadratic"
    ),
    "silly_kicks.spadl._skillcorner_inference.infer_defensive_actions": "test_infer_defensive_actions_is_subquadratic",
    "silly_kicks.tracking._off_ball_runs._off_ball_runs_kernel": "test__off_ball_runs_kernel_is_subquadratic",
    "silly_kicks.tracking._gk_identification.derive_goalkeepers": "test_derive_goalkeepers_is_subquadratic",
    "silly_kicks.tracking._run_values.detect_off_ball_runs": "test_detect_off_ball_runs_is_subquadratic",
    "scripts._loader_databricks.load_matches": "test_load_matches_query_count_is_constant_in_match_count",
    "silly_kicks.xtgk._turnover._opp_first_shot_scan": "test__opp_first_shot_scan_is_subquadratic",
    "silly_kicks.vaep.labels._possession_labels": "test__possession_labels_loc_is_subquadratic",
    "silly_kicks.spadl.utils.add_possessions": "test_add_possessions_is_subquadratic",
    "silly_kicks.atomic.spadl.utils.add_possessions": "test_atomic_add_possessions_is_subquadratic",
    "silly_kicks.restdefense._counting.count_goalside_by_sample": ("test_count_goalside_by_sample_is_subquadratic"),
    "silly_kicks.restdefense._compute._score_samples": "test_score_samples_is_subquadratic",
    "silly_kicks.gkdv._probe.paired_vector_controls": "test_paired_vector_controls_is_subquadratic",
    "silly_kicks.restdefense._probe.paired_vector_controls": "test_paired_vector_controls_rd_is_subquadratic",
    "silly_kicks.territorial_defense._engine.classify_arm_a_domain": "test_classify_arm_a_domain_is_subquadratic",
    "silly_kicks.territorial_defense._compute._score_arm_a": "test_score_arm_a_is_subquadratic",
    "silly_kicks.territorial_defense._compute._score_arm_b": "test_score_arm_b_is_subquadratic",
    # arm_a_threat_suppressed_batch groups BOTH legs via group_rows (L9); scales the frame-group dim.
    "silly_kicks.territorial_defense._arms.arm_a_threat_suppressed_batch": "test_arm_a_batch_is_subquadratic",
    # The public entry builds group_rows ONCE (L7) and threads it into classify + both arms.
    "silly_kicks.territorial_defense._compute.compute_territorial_defense": (
        "test_compute_territorial_defense_is_subquadratic"
    ),
    # _distinct_defenders groups the defensive actions once (ADR-068); it is driven (transitively) by
    # the _score_arm_b guard, which scales GAMES so its per-defender group lookup stays linear.
    "silly_kicks.territorial_defense._compute._distinct_defenders": "test_score_arm_b_is_subquadratic",
    # The validation driver's per-scored-frame dose-battery loop: group_rows over the frames ONCE, one
    # .get per scored Arm-A frame (scales the scored-frame dimension within a single match).
    "scripts.validate_territorial_defense._measure_match": "test_td_measure_match_is_subquadratic",
    # ELASTIC-NW (TF-57): both group_rows callers run INSIDE align_events_to_frames -- candidate
    # detection groups the feasible (frame,player) rows ONCE; _build_frame_lookups groups the merged
    # player-ball distances ONCE. The shared guard scales the episode + frame dimensions together so a
    # rescan-in-loop (O(episodes*frames)) would go quadratic; the real code uses searchsorted.
    "silly_kicks.tracking._elastic_sync._detect_candidate_frames": "test_elastic_align_is_subquadratic",
    "silly_kicks.tracking._elastic_sync._build_frame_lookups": "test_elastic_align_is_subquadratic",
    # TF-52 team KPIs: every family compute_* builds group_rows ONCE over (game_id[, team_id]) and
    # .get per (game, team) in the loop. The growth fixtures scale the GAME dimension (2 teams each),
    # so a per-(game, team) rescan of the full actions/spells would be O(games^2).
    "silly_kicks.team_metrics._pressing.compute_pressing_kpis": "test_compute_pressing_kpis_is_subquadratic",
    "silly_kicks.team_metrics._progression.compute_progression_kpis": ("test_compute_progression_kpis_is_subquadratic"),
    "silly_kicks.team_metrics._buildup.compute_buildup_kpis": "test_compute_buildup_kpis_is_subquadratic",
    # TF-53 match-outcome: compute_match_outcome builds group_rows ONCE over game_id and .get per game;
    # the growth fixture scales the GAME dimension so a per-game full-table rescan would go O(games^2).
    "silly_kicks.match_outcome._compute.compute_match_outcome": "test_compute_match_outcome_is_subquadratic",
    # ADR-103 F2: compute_pitch_control_batch groups the frames ONCE over (game_id,period_id,frame_id)
    # and .get per request; the growth fixture scales the DISTINCT-frame (loop-iteration) dimension so a
    # per-request `frames[frames.frame_id==fid]` rescan would be O(requests*frames) == quadratic.
    "silly_kicks.tracking.pitch_control._dispatch.compute_pitch_control_batch": (
        "test_compute_pitch_control_batch_is_subquadratic"
    ),
    # PitchControlCache.warm groups the frames ONCE (its own group_rows, on top of the batch it calls)
    # and .get per request; same linear pattern, scaled on the distinct-frame (loop) dimension.
    "silly_kicks.tracking.pitch_control._cache.warm": "test_cache_warm_is_subquadratic",
}

#: entries degenerate-by-design (zero counted work IS the guarantee) -> their MANDATORY companion.
DEGENERATE_OK: dict[str, str] = {
    "silly_kicks.vaep.labels._possession_labels": "test_possession_labels_ref_loop_is_superlinear",
}


def group_rows_callers() -> set[str]:
    """Every function that CALLS group_rows in silly_kicks/ + scripts/ (AST; excludes the def site)."""
    out: set[str] = set()
    for base in ("silly_kicks", "scripts"):
        for py in (_ROOT / base).rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            mod = str(py.relative_to(_ROOT).with_suffix("")).replace("/", ".").replace("\\", ".")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "group_rows":
                            if not (mod.endswith("_frame_index") and node.name == "group_rows"):
                                out.add(f"{mod}.{node.name}")
    return out


if __name__ == "__main__":  # pragma: no cover -- ad-hoc: print the derived caller set
    import json

    callers = group_rows_callers()
    print(json.dumps(sorted(callers), indent=2))
    print("MISSING from SCALE_GUARDED:", sorted(callers - set(SCALE_GUARDED)))
