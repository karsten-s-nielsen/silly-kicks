"""Task 1: the campaign `mode` field + the committed `tf46_refit.toml` (adopt-for-refit seam)."""

from __future__ import annotations

import pathlib

import pytest

from scripts.hf_jobs.campaign import load_campaign

_REFIT_TOML = pathlib.Path("scripts/hf_jobs/campaigns/tf46_refit.toml")

_BASE = """
[campaign]
name = "x"
bucket = "silly-kicks/hf-refit-scratch"
target_minutes = 60
concurrency_cap = 10
attempts = 3
{mode}

[[jobs]]
job = "xshot_occurrence"
tier = "public"
dataset = "xshot17"
variants = ["default"]
"""


def _write(tmp_path: pathlib.Path, mode_line: str) -> pathlib.Path:
    p = tmp_path / "c.toml"
    p.write_text(_BASE.format(mode=mode_line), encoding="utf-8")
    return p


def test_mode_defaults_to_rematerialize(tmp_path):
    assert load_campaign(_write(tmp_path, "")).mode == "rematerialize"


def test_mode_refit_is_read(tmp_path):
    assert load_campaign(_write(tmp_path, 'mode = "refit"')).mode == "refit"


def test_unknown_mode_refused(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        load_campaign(_write(tmp_path, 'mode = "bogus"'))


def test_tf46_refit_campaign_is_committed_and_refit():
    c = load_campaign(_REFIT_TOML)
    assert c.name == "tf46_refit"
    assert c.mode == "refit"
    pairs = {(j.job, j.tier) for j in c.jobs}
    assert pairs == {
        ("xshot_occurrence", "public"),
        ("xshot_occurrence", "sc_extended"),
        ("xcross_attempt", "public"),
        ("xcross_attempt", "sc_extended"),
    }
    for j in c.jobs:
        assert tuple(j.variants) == ("default", "position_only")
    probes = {(j.job, j.probe_dataset) for j in c.jobs if j.job == "xcross_attempt"}
    assert probes == {("xcross_attempt", "xcross_probe")}
