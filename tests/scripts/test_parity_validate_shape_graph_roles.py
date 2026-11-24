"""ADR-128 W1N1-vs-W2N2 parity for `scripts.validate_shape_graph_roles` (the W5.7 role gate; a local-only pipeline,
owner ruling 2026-11-11) on its offline pining toy: four public match ids, the first one heavy, so a job of the
fanned run (2 jobs x 2 workers) completes its units out of key order."""

from __future__ import annotations

import os

import pytest

from scripts._toy_corpora import TOY_PARAMS, toy_session
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULE = "scripts.validate_shape_graph_roles"


@pytest.mark.slow  # slow: measured 94 s on the Windows dev box; four toy matches, three label arms, twice (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    args = list(TOY_PARAMS[MODULE])
    with toy_session(MODULE, tmp_path / "data") as (data, extra_env):
        env = {**os.environ, **extra_env}
        run_serial(MODULE, tmp_path / "s1", data, args, tmp_path / "o1", env=env)
        run_fanned(MODULE, tmp_path / "s2", data, args, tmp_path / "o2", env=env)
        gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, args, env=env)["stages"]}
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {"metrics.json"}
    assert completion_out_of_order(tmp_path / "s2", gens["match"])
