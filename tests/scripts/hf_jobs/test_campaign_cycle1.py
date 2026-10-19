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
    assert xc.run_key == "cycle1-abcdef1-xcross_attempt-sc_extended"


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
