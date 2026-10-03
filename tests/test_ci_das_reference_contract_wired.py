"""Structural guard: CI runs the DAS reference-leg contract against the REAL pinned oracle (combined-cycle
Phase B, F1/F5/F6).

``tests/scripts/test_das_reference_contract.py`` is the only test that runs the real accessible-space
2.0.15 through ``scripts/_das_reference_leg.py``. Everywhere else it skips (the library is the dev-only
``das-reference`` extra), so if the dedicated job were dropped, mis-installed, or lost its
``SK_REQUIRE_DAS_REFERENCE`` flag, every leg would stay green having checked nothing. Parsed-YAML topology,
the way ``test_ci_slow_reconcile_wired.py`` pins its job.
"""

from __future__ import annotations

import pathlib
import re

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
_JOB = "das-reference-contract"
_CONTRACT = "tests/scripts/test_das_reference_contract.py"


def _runs(job: dict) -> str:
    return "\n".join(str(s.get("run", "")) for s in job.get("steps", []))


def test_the_contract_job_installs_the_pinned_oracle_and_runs_the_contract():
    job = _CI["jobs"].get(_JOB)
    assert job is not None, f"the {_JOB!r} job is missing from ci.yml"
    runs = _runs(job)
    assert re.search(r'pip install -e "\.\[[^\]]*\bdas-reference\b[^\]]*\]"', runs), runs
    assert re.search(rf"pytest\s+{re.escape(_CONTRACT)}", runs), runs
    assert (_REPO / _CONTRACT).is_file()


def test_a_missing_library_fails_the_job_instead_of_skipping():
    job = _CI["jobs"][_JOB]
    pytest_step = next(s for s in job["steps"] if _CONTRACT in str(s.get("run", "")))
    env = {**job.get("env", {}), **pytest_step.get("env", {})}
    assert str(env.get("SK_REQUIRE_DAS_REFERENCE")) == "1"
    # ...and the contract module really honours that flag (a skip there would make the job vacuous)
    assert 'os.environ.get("SK_REQUIRE_DAS_REFERENCE") == "1"' in (_REPO / _CONTRACT).read_text(encoding="utf-8")


def test_the_oracle_stays_out_of_every_other_job():
    # The native engine needs no accessible-space (ADR-107); the matrix must keep proving that.
    for name, job in _CI["jobs"].items():
        if name != _JOB:
            assert "das-reference" not in _runs(job), f"job {name!r} installs the das-reference oracle"


def test_the_extra_is_the_frozen_oracle():
    # Plain text on purpose: `tomllib` is py3.11+ and absent on the CI 3.10 leg (as test_ci_shard_wiring.py).
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^das-reference = \["accessible-space==2\.0\.15", "pandas<3"\]$', text, re.MULTILINE), (
        "the das-reference extra must stay the frozen oracle: accessible-space==2.0.15 with pandas<3"
    )
