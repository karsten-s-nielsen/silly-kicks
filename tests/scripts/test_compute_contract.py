"""Born-ready compute gate (ADR-128, ADR-056 derive-and-assert). Every train_*.py and every corpus walker
(a module that calls for_each or declares an ItemStage) is a compute-contract pipeline, or is on the legacy
list with an explicit approval, or is named in _NOT_A_PIPELINE with a reason."""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

import tests.scripts._corpus_load_rules as clr

_SEED = "ADR-128 landing seed"
_RULING = re.compile(r"^owner ruling \d{4}-\d{2}-\d{2}: .+")

#: Members that are legitimately not compute pipelines (the ADR-056 _NOT_A_DRIVER pattern). Adding one is
#: a reviewed decision.
_NOT_A_PIPELINE: dict[str, str] = {
    "_loader_pining_to_cache": (
        "the pining CACHE WARMER: it downloads raw artifacts into a local cache and computes nothing a stage "
        "could own; it runs only where the pining token lives (DGX), never on HF."
    ),
    "regen_tf46_fixtures": (
        "the TF-46 fixture REGENERATOR: it sources committed test fixtures from the pining mock-provider API "
        "and writes them with SHA256SUMS; it runs only where the pining token lives (DGX), never in CI or on "
        "HF, and computes no trained or staged artifact a stage could own."
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
}

#: Non-compliant members on the day the gate lands. Every entry carries a reason and an approval; a later
#: addition must cite a dated owner ruling ("owner ruling YYYY-MM-DD: ..."). The PR diff is the review
#: point for growth (no in-repo test can see history).
_LEGACY_UNMIGRATED: dict[str, dict[str, str]] = {
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


def _calls(tree) -> set[str]:
    return {clr._call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}


def _population() -> dict[str, ast.AST]:
    return {
        name: tree
        for name, tree in clr.all_script_trees().items()
        if name.startswith("train_") or ({"for_each", "ItemStage"} & _calls(tree))
    }


def _declares_pipeline(tree) -> bool:
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PIPELINE" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and clr._call_name(n.value) == "Pipeline"
        for n in tree.body
    )


def _compliant() -> list[str]:
    return sorted(set(_population()) - set(_LEGACY_UNMIGRATED) - set(_NOT_A_PIPELINE))


def test_lists_are_subsets_of_the_derived_population_and_disjoint():
    pop = set(_population())
    assert set(_LEGACY_UNMIGRATED) <= pop, sorted(set(_LEGACY_UNMIGRATED) - pop)
    assert set(_NOT_A_PIPELINE) <= pop, sorted(set(_NOT_A_PIPELINE) - pop)
    assert not set(_LEGACY_UNMIGRATED) & set(_NOT_A_PIPELINE)


@pytest.mark.parametrize("name", sorted(_LEGACY_UNMIGRATED))
def test_every_legacy_entry_carries_reason_and_approval(name):
    e = _LEGACY_UNMIGRATED[name]
    assert set(e) == {"reason", "approval"} and e["reason"].strip()
    assert e["approval"] == _SEED or _RULING.match(e["approval"]), e


@pytest.mark.parametrize("name", sorted(_NOT_A_PIPELINE))
def test_not_a_pipeline_names_an_existing_file_with_a_reason(name):
    assert pathlib.Path(f"scripts/{name}.py").is_file() and len(_NOT_A_PIPELINE[name]) > 40


@pytest.mark.parametrize("name", _compliant())
def test_member_declares_a_valid_pipeline(name):
    assert _declares_pipeline(clr.all_script_trees()[name]), (
        f"{name}.py must declare PIPELINE = Pipeline(...) (ADR-128)"
    )
    mod = importlib.import_module(f"scripts.{name}")
    mod.PIPELINE.validate()
    assert mod.PIPELINE.module == f"scripts.{name}"


@pytest.mark.parametrize("name", _compliant())
def test_member_plans_on_its_toy_input(name, tmp_path):
    from scripts._stages import LocalStore, build_parser, make_context, plan
    from scripts._toy_corpora import TOY_PARAMS, build_toy_for

    mod = importlib.import_module(f"scripts.{name}")
    assert mod.PIPELINE.module in TOY_PARAMS, f"{name}: register a toy input in scripts/_toy_corpora.TOY_PARAMS"
    data = build_toy_for(mod.PIPELINE.module, tmp_path / "data")
    ns = build_parser(mod.PIPELINE).parse_args(
        [*TOY_PARAMS[mod.PIPELINE.module], "--plan", "--store", str(tmp_path / "s")]
    )
    params = {p.name: getattr(ns, p.name) for p in mod.PIPELINE.params}
    ctx = make_context(
        mod.PIPELINE,
        params,
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=data,
    )
    out = plan(ctx)
    assert [s["name"] for s in out["stages"]] == [s.name for s in mod.PIPELINE.stages]


@pytest.mark.parametrize("name", _compliant())
def test_member_registers_a_parity_fixture_and_a_job_entry(name):
    from scripts.hf_jobs.registry import DRIVERS, REGISTRY

    mod = importlib.import_module(f"scripts.{name}")
    tests = list(pathlib.Path("tests/scripts").glob("test_parity_*.py"))
    assert any(f'"{mod.PIPELINE.module}"' in t.read_text(encoding="utf-8") for t in tests), (
        f"{name}: no W1N1-vs-W2N2 parity test"
    )
    assert any(s.pipeline == mod.PIPELINE.module for s in REGISTRY.values()) or any(
        d.module == mod.PIPELINE.module for d in DRIVERS.values()
    )


def test_the_gate_FLAGS_a_member_without_PIPELINE():
    """Red fixture (spec 5.8): a toy corpus walker without PIPELINE is a member and is NOT compliant."""
    tree = ast.parse(
        "from scripts._driver import for_each\n\ndef main():\n"
        "    for_each([], key=str, work=print, shard_root='x', token_inputs={'a': 1})\n"
    )
    assert {"for_each"} & _calls(tree)
    assert not _declares_pipeline(tree)


def test_the_compliant_set_is_the_cycle1_migration():
    assert set(_compliant()) == {
        "train_ghost_gk",
        "train_ghost_outfield",
        "train_xshot_occurrence",
        "train_xcross_attempt",
        "train_receiver_model",
        "train_win_probability",
        "build_territory_ranking_census",
        # TF-46 Arm 1 Part 2 pre-flight walkers: born-ready (owner ruling 2026-10-22; ADR-128 ruling 11, the DGX is
        # not a second path -- never carved into _NOT_A_PIPELINE).
        "tf46p2_truth",
        "tf46p2_kick_cells",
        "tf46p2_subtype_merge",
        "tf46p2_penalty",
        "tf46p2_origin",
    }
