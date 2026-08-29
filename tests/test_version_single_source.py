"""Guard: the package version has ONE editable source — silly_kicks/_version.py.

Deleting the pyproject/__init__ duplication is what fixes drift (ADR-079); this test guards the
one thing deletion cannot — *reintroduction*. It fails the moment someone puts a static
`version = "..."` back into `[project]` (which would silently disagree with `_version.py`, since a
static-only pyproject still builds), or lets `__init__` hard-code a literal instead of re-exporting.

py3.10-safe: `tomllib` is 3.11+ and `tomli` is not a test dep, so the pyproject tables are read with
a scoped regex, mirroring tests/test_ci_publish_guard_wired.py.
"""

from __future__ import annotations

import pathlib
import re

import silly_kicks

_REPO = pathlib.Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO / "pyproject.toml"
_VERSION_MODULE = _REPO / "silly_kicks" / "_version.py"


def _project_table() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r"^\[project\]\s*$(.*?)(?=^\[)", text, re.MULTILINE | re.DOTALL)
    assert m is not None, "could not locate the [project] table in pyproject.toml"
    return m.group(1)


def test_project_declares_version_dynamic_and_carries_no_static_literal() -> None:
    table = _project_table()
    # No `version = "x.y.z"` static literal in [project].
    assert re.search(r"^\s*version\s*=", table, re.MULTILINE) is None, (
        "pyproject [project] carries a static `version` — reintroduces the second source that "
        "ADR-079 deleted. The version lives ONLY in silly_kicks/_version.py."
    )
    # `dynamic` lists "version".
    dyn = re.search(r"^\s*dynamic\s*=\s*\[(.*?)\]", table, re.MULTILINE | re.DOTALL)
    assert dyn is not None, "pyproject [project] must declare `dynamic = [...]`"
    assert "version" in re.findall(r"['\"]([^'\"]+)['\"]", dyn.group(1)), (
        "pyproject [project].dynamic must contain 'version' so hatchling derives it from _version.py"
    )


def test_hatch_version_source_points_at_the_version_module() -> None:
    text = _PYPROJECT.read_text(encoding="utf-8")
    tbl = re.search(r"^\[tool\.hatch\.version\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    assert tbl is not None, "pyproject must declare [tool.hatch.version]"
    path = re.search(r"^\s*path\s*=\s*['\"]([^'\"]+)['\"]", tbl.group(1), re.MULTILINE)
    assert path is not None and path.group(1) == "silly_kicks/_version.py", (
        "[tool.hatch.version].path must be 'silly_kicks/_version.py' (the single source)"
    )


def test_version_module_literal_is_what_the_runtime_exposes() -> None:
    # The literal in _version.py IS silly_kicks.__version__ (guards __init__ hard-coding a divergent
    # value instead of re-exporting). Reads the module text; does not import install metadata.
    src = _VERSION_MODULE.read_text(encoding="utf-8")
    literal = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", src, re.MULTILINE)
    assert literal is not None, 'silly_kicks/_version.py must define a `__version__ = "..."` literal'
    assert literal.group(1) == silly_kicks.__version__, (
        "silly_kicks.__version__ diverges from silly_kicks/_version.py — __init__ must re-export it, "
        "not hard-code its own literal"
    )
