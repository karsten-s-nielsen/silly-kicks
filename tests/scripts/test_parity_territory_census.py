"""ADR-128 W1N1-vs-W2N2 parity for `build_territory_ranking_census` on the served toy open-data tree.

Every stage output and census.json are byte-identical. Non-vacuity: the score stage completes out of key
order inside a job (the toy's heavy extra-time match). The bootstrap's own non-vacuity (replicates fitted in
any order, combined by replicate index) is tests/scripts/test_bootstrap_replicate_map.py: the toy corpus
cannot clear Tier-1 (one team per defender), so its Tier-2 bootstrap never runs here.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")  # the census refs unit reads the served tree through statsbombpy

from scripts._toy_corpora import TOY_PARAMS, toy_session
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULE = "scripts.build_territory_ranking_census"


@pytest.mark.slow  # slow: measured 239 s on the Windows dev box (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    args = list(TOY_PARAMS[MODULE])
    with toy_session(MODULE, tmp_path / "data") as (data, extra_env):
        env = {**os.environ, **extra_env}
        run_serial(MODULE, tmp_path / "s1", data, args, tmp_path / "o1", env=env)
        run_fanned(MODULE, tmp_path / "s2", data, args, tmp_path / "o2", env=env)
        gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, args, env=env)["stages"]}
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert "census.json" in artifact(tmp_path / "o1")
    assert completion_out_of_order(tmp_path / "s2", gens["score"])
