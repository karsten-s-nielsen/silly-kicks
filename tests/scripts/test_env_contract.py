"""S5.9: every environment read in silly_kicks/ and scripts/ is registered; the AST gate resolves constants."""

from __future__ import annotations

import ast
import pathlib

from scripts.hf_jobs import env_contract as ec


def _reads(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ec.env_reads(tree, module_path=path)  # names, or the sentinel UNRESOLVED


def test_every_env_read_is_registered():
    bad = []
    for root in ("silly_kicks", "scripts"):
        for p in pathlib.Path(root).rglob("*.py"):
            for name in _reads(p):
                if name is ec.UNRESOLVED or not (ec.registered(name) or name in ec.PLATFORM_NAMES):
                    bad.append((str(p), name))
    assert not bad, bad


def test_the_gate_resolves_an_imported_constant(tmp_path):
    (tmp_path / "a.py").write_text("ENV = 'SILLY_KICKS_NEW_THING'\n")
    (tmp_path / "b.py").write_text("import os\nfrom a import ENV\nv = os.environ.get(ENV)\n")
    got = ec.env_reads(ast.parse((tmp_path / "b.py").read_text()), module_path=tmp_path / "b.py", search=[tmp_path])
    assert got == ["SILLY_KICKS_NEW_THING"]


def test_a_forbidden_variable_fails_a_job():
    import pytest

    from scripts.hf_jobs.closures import ClosureViolationError

    with pytest.raises(ClosureViolationError, match="SILLY_KICKS_GHOST_GK_PATH"):
        ec.check_job_env({"SILLY_KICKS_IN_HF_JOB": "1", "SILLY_KICKS_GHOST_GK_PATH": "/x"})
    with pytest.raises(ClosureViolationError, match="SILLY_KICKS_UNKNOWN"):
        ec.check_job_env({"SILLY_KICKS_IN_HF_JOB": "1", "SILLY_KICKS_UNKNOWN": "1"})


def test_controller_plan_env_strips_output_switches_keeping_neutral_and_revision():
    """CCTP-IMPL-26: the controller's `--plan` env drops every output switch (so the operator's shell cannot
    move the token) but keeps OPEN_DATA_REVISION (excluded from the token; the launcher sets it in the job) and
    every neutral/platform variable (the subprocess still needs PATH, caches, the run key)."""
    env = {
        "PATH": "/x",
        "SILLY_KICKS_ASSERT_INVARIANTS": "1",
        "SILLY_KICKS_GHOST_FORCE_NUMPY": "1",
        "SILLY_KICKS_OPEN_DATA_REVISION": "rev",
        "SILLY_KICKS_RUN_KEY": "rk",
    }
    out = ec.controller_plan_env(env)
    assert "SILLY_KICKS_ASSERT_INVARIANTS" not in out and "SILLY_KICKS_GHOST_FORCE_NUMPY" not in out
    assert out["SILLY_KICKS_OPEN_DATA_REVISION"] == "rev"  # excluded from the token; kept for the subprocess
    assert out["PATH"] == "/x" and out["SILLY_KICKS_RUN_KEY"] == "rk"  # neutral/platform vars untouched
