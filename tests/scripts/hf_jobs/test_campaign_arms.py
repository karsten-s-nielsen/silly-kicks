"""Spec 5.5 / 7.2 (corpus widening, ADR-132): a visibility-gated campaign job declares its `arms`.

`campaign.instances` passes `compute_arms(staged_units(dataset))` restricted to the job's arms; a gated job without
`arms`, with an arm the dataset lacks, or with ONE arm that is a strict subset of the dataset's units is refused
before any launch (`cmd_preflight` builds the instances first).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from scripts.hf_jobs import arm_policy
from scripts.hf_jobs import campaign as C
from scripts.hf_jobs.datasets import staged_units

SHA = "abcdef1" + "0" * 33
_CAMPAIGNS = pathlib.Path("scripts/hf_jobs/campaigns")

_HEAD = """[campaign]
name = "fx"
bucket = "silly-kicks/hf-refit-scratch"
concurrency_cap = 4
attempts = 3
"""


def _write(tmp_path, job_toml: str) -> C.Campaign:
    p = tmp_path / "fx.toml"
    p.write_text(_HEAD + "\n[[jobs]]\n" + job_toml, encoding="utf-8")
    return C.load_campaign(p)


def _arms_payload(inst: C.Instance) -> dict:
    return json.loads(inst.args[inst.args.index("--arms-json") + 1])


def test_gated_job_without_arms_refused(tmp_path):
    c = _write(tmp_path, 'job = "xshot_occurrence"\ntier = "public"\ndataset = "xshot17"\nvariants = ["default"]\n')
    with pytest.raises(ValueError, match="arms"):
        C.instances(c, SHA)


def test_job_gets_only_its_arms(tmp_path):
    c = _write(
        tmp_path,
        'job = "xcross_attempt"\ntier = "sc_extended"\ndataset = "xshot115"\nprobe_dataset = "xcross_probe"\n'
        'variants = ["default"]\narms = ["sc_extended"]\n',
    )
    (inst,) = C.instances(c, SHA)
    payload = _arms_payload(inst)
    assert list(payload["arms"]) == ["sc_extended"] and payload["ship_order"] == ["sc_extended"]
    assert c.jobs[0].arms == ("sc_extended",)


def test_an_arm_the_dataset_lacks_is_refused(tmp_path):
    c = _write(
        tmp_path,
        'job = "xshot_occurrence"\ntier = "public"\ndataset = "xshot17"\nvariants = ["default"]\n'
        'arms = ["public", "sc_extended"]\n',
    )
    with pytest.raises(ValueError, match="not produced by the dataset"):
        C.instances(c, SHA)


def test_an_unknown_arm_name_is_refused_at_load(tmp_path):
    with pytest.raises(ValueError, match="full"):
        _write(
            tmp_path,
            'job = "xshot_occurrence"\ntier = "public"\ndataset = "xshot17"\nvariants = ["default"]\narms = ["full"]\n',
        )


def test_preflight_refuses_a_strict_subset_job(tmp_path):
    # tc3 holds licensed SkillCorner units: a one-arm `public` job on it would study and gate on every row but fit
    # on the public rows only (spec 5.5). Refused before any check runs or any report is uploaded.
    c = _write(
        tmp_path,
        'job = "xshot_occurrence"\ntier = "public"\ndataset = "tc3"\nvariants = ["default"]\narms = ["public"]\n',
    )

    class _NoApi:
        def __getattr__(self, name):
            raise AssertionError(f"preflight touched the API ({name}) before refusing")

    with pytest.raises(ValueError, match="strict subset"):
        C.cmd_preflight(c, SHA, api=_NoApi())


def test_a_paired_job_gets_both_arms(tmp_path):
    c = _write(
        tmp_path,
        'job = "xshot_occurrence"\ntier = "public"\ndataset = "tc3"\nvariants = ["default"]\n'
        'arms = ["public", "sc_extended"]\n',
    )
    (inst,) = C.instances(c, SHA)
    assert _arms_payload(inst) == arm_policy.compute_arms(staged_units("tc3"))


def test_an_ungated_job_ignores_arms(tmp_path):
    with pytest.raises(ValueError, match="not visibility-gated"):
        _write(
            tmp_path,
            'job = "receiver_model"\ntier = "public"\ndataset = "sb_open"\nvariants = ["default"]\narms = ["public"]\n',
        )


@pytest.mark.parametrize("name", ["tf46_refit.toml", "cycle1.toml"])
def test_committed_campaigns_keep_todays_arms(name):
    c = C.load_campaign(_CAMPAIGNS / name)
    gated = [i for i in C.instances(c, SHA) if "--arms-json" in i.args]
    assert gated
    for inst in gated:
        job = next(j for j in c.jobs if j.job == inst.job and j.tier == inst.tier)
        today = arm_policy.compute_arms(staged_units(job.dataset))
        if job.dataset == "xshot17":  # the wheel jobs: the only arm xshot17 produces
            assert job.arms == ("public",) and _arms_payload(inst) == today
        else:  # the Hub jobs on xshot115: the arm today's run ships (ship_order[-1])
            assert job.dataset == "xshot115" and job.arms == ("sc_extended",)
            assert today["ship_order"][-1] == "sc_extended"
            assert _arms_payload(inst) == {
                "arms": {"sc_extended": today["arms"]["sc_extended"]},
                "ship_order": ["sc_extended"],
            }
