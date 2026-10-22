"""Born-ready compute gate (ADR-128, ADR-056 derive-and-assert). Every train_*.py and every corpus walker
(a module that calls for_each or declares an ItemStage) is a compute-contract pipeline, or is on the legacy
list with an explicit approval, or is named in _NOT_A_PIPELINE with a reason.

The population and its AST helpers moved to PRODUCTION code (`scripts/_pipeline_population.py`, ADR-129 S5.9:
a job-side closure module needs them and must never import `tests/`); this gate imports them and keeps every
assertion."""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from scripts._pipeline_population import (
    _RULING,
    _SEED,
    all_script_trees,
)
from scripts._pipeline_population import (
    LEGACY_UNMIGRATED as _LEGACY_UNMIGRATED,
)
from scripts._pipeline_population import (
    NOT_A_PIPELINE as _NOT_A_PIPELINE,
)
from scripts._pipeline_population import (
    calls as _calls,
)
from scripts._pipeline_population import (
    compliant as _compliant,
)
from scripts._pipeline_population import (
    declares_pipeline as _declares_pipeline,
)
from scripts._pipeline_population import (
    population as _population,
)


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
    assert _declares_pipeline(all_script_trees()[name]), f"{name}.py must declare PIPELINE = Pipeline(...) (ADR-128)"
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
    }
