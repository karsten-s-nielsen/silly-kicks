"""The committed Cycle 1 campaign file resolves against the registries and the dataset lock."""

from __future__ import annotations

import importlib
import pathlib

import pytest

from scripts._stages import build_parser
from scripts.hf_jobs import campaign as C
from scripts.hf_jobs.datasets import declared
from scripts.hf_jobs.registry import TrainerSpec, spec_for

CYCLE1 = pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml")


def test_cycle1_names_known_jobs_variants_and_datasets():
    c = C.load_campaign(CYCLE1)
    for j in c.jobs:
        spec = spec_for(j.job)
        assert j.dataset in declared()
        if isinstance(spec, TrainerSpec):
            assert set(j.variants) <= {b.variant for b in spec.builds}, j.job


def test_cycle1_covers_the_spec_job_set():
    jobs = {(j.job, j.tier) for j in C.load_campaign(CYCLE1).jobs}
    assert {
        ("ghost_gk", "owner"),
        ("ghost_outfield", "owner"),
        ("xshot_occurrence", "public"),
        ("xshot_occurrence", "sc_extended"),
        ("xcross_attempt", "public"),
        ("xcross_attempt", "sc_extended"),
        ("receiver_model", "owner"),
        ("win_probability", "public"),
        ("build_territory_ranking_census", "public"),
        ("match_outcome_dependence", "public"),
        ("pass_completion", "public"),
        ("xsuccess", "public"),
    } == jobs


def test_instances_carry_dataset_args_and_derived_run_keys():
    c = C.load_campaign(CYCLE1)
    insts = C.instances(c, "abcdef1" + "0" * 33)
    assert len(insts) == 21
    xc = next(i for i in insts if i.job == "xcross_attempt" and i.tier == "sc_extended" and i.variant == "default")
    assert ["--dataset", "xshot115"] == list(xc.args[xc.args.index("--dataset") : xc.args.index("--dataset") + 2])
    assert "--probe-dataset" in xc.args and "--arms-json" in xc.args
    assert xc.run_key == "cycle1-abcdef1-xcross_attempt-xshot115"  # spec r4 D18 (c): one store per (job, dataset)
    assert xc.id == "cycle1-abcdef1-xcross_attempt-xshot115/sc_extended/default"


def _arg(inst, flag: str) -> str:
    return inst.args[inst.args.index(flag) + 1]


@pytest.mark.parametrize("job", ["xshot_occurrence", "xcross_attempt"])
def test_both_tiers_read_xshot115_into_one_store_each_with_only_its_arms(job):
    """Spec r4 D18 (c): the public tier reads `xshot115`, as `sc_extended` does, so both tiers share one store and one
    extraction; each tier receives only the arms it ships at most (public -> public; sc_extended -> public,
    sc_extended), computed off-HF. The public arm is exactly `xshot17`'s corpus (17 matches)."""
    import json

    from scripts.hf_jobs import arm_policy
    from scripts.hf_jobs.datasets import staged_units

    c = C.load_campaign(CYCLE1)
    insts = [i for i in C.instances(c, "abcdef1" + "0" * 33) if i.job == job]
    assert {(i.tier, i.variant) for i in insts} == {
        (t, v) for t in ("public", "sc_extended") for v in ("default", "position_only")
    }
    assert {i.run_key for i in insts} == {f"cycle1-abcdef1-{job}-xshot115"}
    assert {_arg(i, "--dataset") for i in insts} == {"xshot115"}
    arms = {i.tier: json.loads(_arg(i, "--arms-json")) for i in insts}
    assert set(arms["public"]["arms"]) == {"public"} and arms["public"]["ship_order"] == ["public"]
    assert set(arms["sc_extended"]["arms"]) == {"public", "sc_extended"}
    public17 = arm_policy.compute_arms(staged_units("xshot17"))["arms"]["public"]
    assert sorted(arms["public"]["arms"]["public"]) == sorted(public17) and len(public17) == 17


def test_a_staged_dataset_instance_fails_loud_without_its_dataset():
    """The legacy launcher refused an owner trainer without ``corpus_repo`` and xcross without
    ``probe_repo``. On the stage CLI each staged (non-open) ``--dataset`` / ``--probe-dataset`` is a
    REQUIRED param: the full argv parses, and the argv without that pair exits at parse time."""
    c = C.load_campaign(CYCLE1)
    checked: set[tuple[str, str]] = set()
    for inst in C.instances(c, "abcdef1" + "0" * 33):
        ap = build_parser(importlib.import_module(inst.module).PIPELINE)
        tail = ["--store", "s", "--plan"]
        ap.parse_args([*inst.args, *tail])
        access = {d.name: d.access for d in inst.datasets}
        for flag in ("--dataset", "--probe-dataset"):
            if flag not in inst.args:
                continue
            i = inst.args.index(flag)
            if access[inst.args[i + 1]] == "open":
                continue  # public open data: the revision-pinned default, never a staged corpus
            with pytest.raises(SystemExit):
                ap.parse_args([*inst.args[:i], *inst.args[i + 2 :], *tail])
            checked.add((inst.job, flag))
    assert checked == {
        ("ghost_gk", "--dataset"),
        ("ghost_outfield", "--dataset"),
        ("receiver_model", "--dataset"),
        ("xshot_occurrence", "--dataset"),
        ("xcross_attempt", "--dataset"),
        ("xcross_attempt", "--probe-dataset"),
    }
