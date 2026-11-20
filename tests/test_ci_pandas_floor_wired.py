"""The declared pandas floor is tested in CI (owner ruling 2026-11-19: "Fix the code, keep 2.1.1").

``pyproject.toml`` promises ``pandas>=<floor>``, but every ``test`` matrix leg resolves the latest pandas, so a
floor-only failure shipped unseen: on pandas 2.1 an ``Int64`` column with ``<NA>`` raises on
``.to_numpy(dtype=float)``, and 8 tracking families failed at the floor for many releases. The ``pandas-floor`` job runs
the matrix's selection on the floor version. This guard reads ``ci.yml`` and ``pyproject.toml`` and pins that wiring.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CI = yaml.safe_load((_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
# Plain text on purpose: `tomllib` is py3.11+ and absent on the CI 3.10 leg (as test_ci_shard_wiring.py).
_PYPROJECT = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _declared_floor() -> str:
    m = re.search(r'^\s*"pandas>=\s*([0-9][0-9.]*)[^"]*",?\s*$', _PYPROJECT, flags=re.M)
    assert m, "pyproject declares no pandas floor in [project] dependencies"
    return m.group(1)


def _job() -> dict:
    assert "pandas-floor" in _CI["jobs"], "ci.yml has no pandas-floor job: the declared pandas floor is untested"
    return _CI["jobs"]["pandas-floor"]


def _runs(job: dict) -> list[str]:
    return [s["run"] for s in job["steps"] if "run" in s]


def test_the_floor_job_installs_exactly_the_declared_floor():
    floor = _declared_floor()
    installs = [r for r in _runs(_job()) if "pip install" in r]
    assert any(f'"pandas=={floor}"' in r for r in installs), (
        f"the pandas-floor job must install pandas=={floor} (the pyproject floor): {installs}"
    )
    assert any('"numpy<2"' in r for r in installs), "pandas 2.1 wheels predate numpy 2: pin numpy<2 on the floor job"
    # pyarrow >= 26 refuses numpy 1.x ("pyarrow requires NumPy 2.0 or newer"); the first CI floor run resolved
    # pyarrow 26.0.1 and every shard failed at collection. Pin the parquet engine to a numpy-1-capable release.
    assert any(re.search(r'"pyarrow<\d+', r) for r in installs), (
        "pin pyarrow below the first numpy-2-only release on the floor job, or parquet cannot load under numpy<2"
    )


def test_the_floor_job_asserts_the_resolved_version():
    floor = _declared_floor()
    assert any("pandas.__version__" in r and floor in r for r in _runs(_job())), (
        "the floor job must assert the pandas it resolved, so a resolver bump cannot silently test another version"
    )


def test_the_floor_job_runs_the_matrix_selection_sharded():
    matrix_cmds = [r for r in _runs(_CI["jobs"]["test"]) if r.startswith("pytest tests/") and "--splits" in r]
    floor_cmds = [r for r in _runs(_job()) if r.startswith("pytest tests/") and "--splits" in r]
    assert floor_cmds, "the pandas-floor job runs no sharded pytest"
    marker = re.search(r'-m "([^"]+)"', matrix_cmds[0]).group(1)  # type: ignore[union-attr]
    shards = _job()["strategy"]["matrix"]["floor-shard"]
    assert shards == list(range(1, len(shards) + 1)), f"floor-shard axis must be 1..N contiguous, got {shards}"
    for cmd in floor_cmds:
        assert f'-m "{marker}"' in cmd, f"the floor job must run the matrix selection -m {marker!r}: {cmd}"
        m = re.search(r"--splits\s+(\d+)", cmd)
        assert m and int(m.group(1)) == len(shards), f"--splits must equal the floor shard count {len(shards)}: {cmd}"
        assert "--group ${{ matrix.floor-shard }}" in cmd, f"missing per-shard --group: {cmd}"
        assert "-p no:randomly" in cmd, f"collection order must be pinned for the split: {cmd}"
