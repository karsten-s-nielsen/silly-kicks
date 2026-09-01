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
