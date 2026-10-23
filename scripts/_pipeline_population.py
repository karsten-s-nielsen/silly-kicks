"""The compute-pipeline population, single-sourced in PRODUCTION code (ADR-129, S5.9).

Both ``scripts.hf_jobs.closures.record`` (a job-side module) and ``tests/scripts/test_compute_contract.py``
need the set of compliant pipeline modules. A job never imports ``tests/``, so the ONE definition lives here.
The AST helpers (``all_script_trees``, ``call_name``, ``calls``) live here too; ``tests/scripts/_corpus_load_rules.py``
imports them from here (one definition; its other users keep their imports).

A member is a ``train_*`` module, a corpus walker (``for_each`` / ``ItemStage``), OR a module that declares a
``PIPELINE = Pipeline(...)``. The last clause keeps a driver whose ``ItemStage`` units were extracted to a stage
module (ADR-129) -- e.g. ``build_territory_ranking_census`` -- in the population: it has no inline ``ItemStage`` any
more, but it still declares and runs a pipeline.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import re

_SCRIPTS = pathlib.Path(__file__).resolve().parent

_SEED = "ADR-128 landing seed"
_RULING = re.compile(r"^owner ruling \d{4}-\d{2}-\d{2}: .+")

#: Members that are legitimately not compute pipelines (the ADR-056 _NOT_A_DRIVER pattern). Adding one is
#: a reviewed decision.
NOT_A_PIPELINE: dict[str, str] = {
    "_loader_pining_to_cache": (
        "the pining CACHE WARMER: it downloads raw artifacts into a local cache and computes nothing a stage "
        "could own; it runs only where the pining token lives (DGX), never on HF."
    ),
    "materialize_tc3_frames": (
        "the DGX STAGING tool: it materialises the neutral flat corpus that `datasets record` then measures; "
        "it is the input side of every staged dataset, not a compute pipeline."
    ),
    "_xt_corpus": "a LIBRARY helper (the xT corpus pass) called by drivers; its pipeline is its caller's.",
    "_stages": (
        "the stage-contract ENGINE itself (ADR-128): it calls for_each only to run one ItemStage unit; every "
        "pipeline is built on it, it is not one."
    ),
    "_ghost_gk_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable validate/extract stages of train_ghost_gk), "
        "not a PIPELINE; the trainer imports and threads them in, so its parity/toy coverage is the trainer's."
    ),
    "_ghost_outfield_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable extract stage of train_ghost_outfield), not a "
        "PIPELINE; the trainer imports and threads it in, so its parity/toy coverage is the trainer's."
    ),
    "_receiver_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable extract/pool stages of train_receiver_model), "
        "not a PIPELINE; the trainer imports and threads them in, so its parity/toy coverage is the trainer's."
    ),
    "_win_probability_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable refs/extract stages of train_win_probability), "
        "not a PIPELINE; the trainer imports and threads them in, so its parity/toy coverage is the trainer's."
    ),
    "_xcross_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable extract/probe stages of train_xcross_attempt), "
        "not a PIPELINE; the trainer imports and threads them in, so its parity/toy coverage is the trainer's."
    ),
    "_xshot_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable extract stage of train_xshot_occurrence), not a "
        "PIPELINE; the trainer imports and threads it in, so its parity/toy coverage is the trainer's."
    ),
    "_census_stages": (
        "ADR-129 STAGE MODULE: it declares STAGES (the reusable refs/fit_prepass/prefit/score stages of "
        "build_territory_ranking_census), not a PIPELINE; the driver imports and threads them in, so its "
        "parity/toy coverage is the driver's."
    ),
}

#: Non-compliant members on the day the gate lands. Every entry carries a reason and an approval; a later
#: addition must cite a dated owner ruling ("owner ruling YYYY-MM-DD: ..."). The PR diff is the review
#: point for growth (no in-repo test can see history).
LEGACY_UNMIGRATED: dict[str, dict[str, str]] = {
    **{
        n: {"reason": "Cycle 3+ driver burn-down (ADR-128 section 11)", "approval": _SEED}
        for n in (
            "build_gkdv_arm_values",
            "build_layer2_spells",
            "build_rq_pass_scores",
            "build_sb360_coverage",
            "build_skillcorner_s1_event_validity",
            "build_tf19_instrument_responsiveness",
            "build_tf56_positioning_validity",
            "build_tf60_layer3_arm_values",
            "calibrate_coordination",
            "calibrate_tracking_defaults",
            "calibrate_xt_bandwidth",
            "check_stage1_argmax",
            "derive_coordination_params",
            "derive_opengoal_range",
            "evolve_xsuccess_features",
            "measure_ball_contact_validation",
            "measure_box_constant_delta",
            "measure_cover_shadow_argmax_agreement",
            "measure_f1b_feature_delta",
            "measure_gs_shot_distribution",
            "measure_rc4_orientation",
            "run_signoff_power",
            "tune_structural_pass_sigma",
            "validate_coordination_numerics",
            "validate_das_native_parity",
            "validate_gk_decision",
            "validate_match_outcome_calibration",
            "validate_sb360_licensed_corpus",
            "validate_shot_goalmouth_sb",
            "validate_team_coordination",
            "validate_team_kpi_reliability",
            "validate_territorial_defense",
            "validate_territory_counterfactual",
            "validate_xcross_causal",
            "validate_xs_probe",
            "validate_xshot_causal",
        )
    },
    **{
        n: {
            "reason": "Cycle 2: real pipeline; runs via the one-unit adapter stage (ADR-128 section 5.6)",
            "approval": _SEED,
        }
        for n in ("train_match_outcome_dependence", "train_pass_completion", "train_xsuccess")
    },
    "train_gk_retention": {"reason": "blocked: no non-Databricks data path (registry BLOCKED)", "approval": _SEED},
    # TF-46 merged to main before this gate landed (spec 12 R8): legacy, and the FIRST burn-down item
    # (owner ruling 10).
    **{
        n: {
            "reason": "TF-46 corners, merged before the gate (spec 12 R8): first Cycle 3+ burn-down item (ruling 10)",
            "approval": _SEED,
        }
        for n in ("build_corner_routines", "validate_corners")
    },
}


@functools.cache
def all_script_trees() -> dict[str, ast.AST]:
    """Every ``scripts/*.py`` parsed once, PRIVATE INCLUDED."""
    return {p.stem: ast.parse(p.read_text(encoding="utf-8")) for p in sorted(_SCRIPTS.glob("*.py"))}


def call_name(call: ast.Call) -> str:
    return getattr(call.func, "id", "") or getattr(call.func, "attr", "")


def calls(tree) -> set[str]:
    return {call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}


def declares_pipeline(tree) -> bool:
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PIPELINE" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and call_name(n.value) == "Pipeline"
        for n in tree.body
    )


def population() -> dict[str, ast.AST]:
    return {
        name: tree
        for name, tree in all_script_trees().items()
        if name.startswith("train_") or ({"for_each", "ItemStage"} & calls(tree)) or declares_pipeline(tree)
    }


def compliant() -> list[str]:
    return sorted(set(population()) - set(LEGACY_UNMIGRATED) - set(NOT_A_PIPELINE))
