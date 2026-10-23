"""ADR-128 W1N1-vs-W2N2 parity for the five TF-46 Arm 1 Part 2 pre-flight walkers (owner ruling 2026-10-22: born-
ready stage-contract pipelines). Each runs on the toy staged Part 2 corpus (``build_part2_corpus``: two Sportec,
one Gradient Sports, two SkillCorner units with restart scenes, native events and a camera footprint).

Registered modules (the compute-contract gate reads these literals): "scripts.tf46p2_truth",
"scripts.tf46p2_kick_cells", "scripts.tf46p2_subtype_merge", "scripts.tf46p2_penalty", "scripts.tf46p2_origin".
"""

from __future__ import annotations

import pytest

from scripts._toy_corpora import build_part2_corpus
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULES = (
    "scripts.tf46p2_truth",
    "scripts.tf46p2_kick_cells",
    "scripts.tf46p2_subtype_merge",
    "scripts.tf46p2_penalty",
    "scripts.tf46p2_origin",
)
ARGS = ["--dataset", "toytf46"]


@pytest.mark.slow  # slow: five pipelines x (serial + fanned) subprocess runs on the toy corpus (ADR-023)
@pytest.mark.parametrize("module", MODULES)
def test_w1n1_equals_w2n2(module, tmp_path):
    data = build_part2_corpus(tmp_path / "data")
    run_serial(module, tmp_path / "s1", data, ARGS, tmp_path / "o1")
    run_fanned(module, tmp_path / "s2", data, ARGS, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    run = module.split(".")[-1].removeprefix("tf46p2_")
    assert f"{run}.json" in artifact(tmp_path / "o1")  # the provenance sidecar ships with the aggregate
    gens = {s["name"]: s["generation"] for s in plan(module, tmp_path / "s2", data, ARGS)["stages"]}
    assert completion_out_of_order(tmp_path / "s2", gens["score"])  # non-vacuity: the fan-out interleaved
