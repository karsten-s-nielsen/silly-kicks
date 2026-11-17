"""ADR-128 W1N1-vs-W2N2 parity for `build_cross_definition` (ADR-133; scaffolded, toy = the xCross toy corpus)."""

from __future__ import annotations

from scripts._toy_corpora import TOY_BUILDERS
from tests.scripts._parity import artifact, fanned_jobs, outputs, plan, run_fanned, run_serial

MODULE = "scripts.build_cross_definition"
ARGS = ["--dataset", "toyxc"]


def test_w1n1_equals_w2n2(tmp_path):
    data = TOY_BUILDERS[MODULE](tmp_path / "data")
    run_serial(MODULE, tmp_path / "s1", data, ARGS, tmp_path / "o1")
    run_fanned(MODULE, tmp_path / "s2", data, ARGS, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, ARGS)["stages"]}
    assert fanned_jobs(tmp_path / "s2", gens["flat"]) == 2
