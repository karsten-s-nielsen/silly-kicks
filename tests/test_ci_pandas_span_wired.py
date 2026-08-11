"""Structural guard: CI's leg set must still span both pandas majors.

``pyproject.toml`` pins ``pandas>=2.1.1,!=3.0.4`` with NO upper bound, so pip resolves the newest
compatible pandas per interpreter -- and pandas 3 requires Python >= 3.11 (verified against the
PyPI index: 3.0.5 declares ``requires_python >=3.11``, 2.3.3 declares ``>=3.9``). Measured on CI run
31316804815: ubuntu-3.10 -> pandas 2.3.3, every other leg -> 3.0.5.

That differential coverage is REAL but was ACCIDENTAL -- nothing declared it, so it could vanish
with no diff and no signal. This guard declares it. The repo already has one measured instance of a
silent pandas-3 behaviour change (DAS going all-NaN), which is the class this coverage exists to
expose.

**This asserts over the RESOLVED LEG SET, never the ``python-version`` axis.** GitHub computes legs
as os x python-version MINUS ``exclude`` PLUS ``include``, and ``exclude`` is already the pruning
mechanism in use here (two windows legs). Adding ``{os: ubuntu-latest, python-version: "3.10"}`` to
``exclude`` collapses the pandas-2 span while leaving ``"3.10"`` in the axis -- an axis-based
assertion would pass. ``tests/test_ci_slow_gating_wired.py`` already reads the matrix rather than
trusting the axes; this follows it.

**What this guard CANNOT see:** a span collapse caused by a dependency constraint rather than a
matrix edit -- e.g. adding ``pandas<3`` to ``pyproject.toml``, after which every leg resolves
pandas 2 while ``ci.yml`` is untouched. That hazard is real (this repo already pins ``!=3.0.4`` for
a segfault) and is covered by the ``pandas-span`` aggregation job in ``ci.yml``, which observes what
each leg actually installed. The two are complementary; neither subsumes the other.
"""

from __future__ import annotations

import itertools
import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"

#: pandas 3 requires Python >= 3.11, so a leg below it resolves pandas 2 and a leg at or above it
#: resolves pandas 3. This is the ASSUMPTION that makes a structural check a valid proxy for the
#: span. If pandas changes its minimum Python, THIS CONSTANT is what moves -- not the assertion,
#: and never by redefining the boundary to match a matrix that lost its leg.
_PANDAS3_MIN_PY = (3, 11)


def _pyver(leg: dict) -> tuple[int, ...]:
    return tuple(int(p) for p in str(leg["python-version"]).split("."))


def resolved_legs(matrix: dict) -> list[dict]:
    """``os`` x ``python-version``, MINUS ``exclude``, PLUS ``include`` -- GitHub's own order."""
    base = [{"os": os_, "python-version": py} for os_, py in itertools.product(matrix["os"], matrix["python-version"])]
    for ex in matrix.get("exclude", []):
        base = [leg for leg in base if not all(leg.get(k) == v for k, v in ex.items())]
    for inc in matrix.get("include", []):
        for leg in base:
            if all(leg.get(k) == v for k, v in inc.items() if k in leg):
                leg.update(inc)
    return base


def _matrix() -> dict:
    return yaml.safe_load(_CI.read_text(encoding="utf-8"))["jobs"]["test"]["strategy"]["matrix"]


def test_ci_leg_set_spans_both_pandas_majors() -> None:
    legs = resolved_legs(_matrix())
    below = [leg for leg in legs if _pyver(leg) < _PANDAS3_MIN_PY]
    at_or_above = [leg for leg in legs if _pyver(leg) >= _PANDAS3_MIN_PY]

    assert below and at_or_above, (
        f"CI's resolved leg set no longer straddles Python "
        f"{_PANDAS3_MIN_PY[0]}.{_PANDAS3_MIN_PY[1]}, so every leg resolves the SAME pandas major "
        f"and the differential coverage this repo relies on is gone. "
        f"legs={[(leg['os'], leg['python-version']) for leg in legs]}. "
        f"ASSUMPTION: pandas 3 requires Python >= {_PANDAS3_MIN_PY[0]}.{_PANDAS3_MIN_PY[1]}. If "
        f"pandas changed that, fix _PANDAS3_MIN_PY -- do NOT delete this assertion, and do not "
        f"'fix' it by moving the boundary to match a matrix that lost its old leg."
    )


def test_resolved_legs_honours_exclude_not_just_the_axis() -> None:
    """Non-vacuity for the resolver: ``exclude`` must actually remove a leg.

    Without this, ``resolved_legs`` could ignore ``exclude`` entirely and the guard above would
    still pass on today's matrix -- while missing the likeliest way the span gets destroyed, since
    ``exclude`` is the pruning mechanism this workflow already uses.
    """
    matrix = {
        "os": ["ubuntu-latest"],
        "python-version": ["3.10", "3.12"],
        "exclude": [{"os": "ubuntu-latest", "python-version": "3.10"}],
    }
    assert [leg["python-version"] for leg in resolved_legs(matrix)] == ["3.12"]


def test_resolved_legs_applies_include_without_inventing_legs() -> None:
    """``include`` decorates matching legs (the repo uses it to flag the primary leg); it must not
    silently add or drop one, or the span could be miscounted in either direction."""
    matrix = {
        "os": ["ubuntu-latest"],
        "python-version": ["3.10", "3.12"],
        "include": [{"os": "ubuntu-latest", "python-version": "3.12", "primary": True}],
    }
    legs = resolved_legs(matrix)
    assert len(legs) == 2
    assert [leg.get("primary") for leg in legs] == [None, True]


def test_the_aggregation_job_exists_and_needs_test() -> None:
    """Without ``needs: test`` the job runs before the artifacts exist and passes vacuously.

    The job's own script guards that too (it exits non-zero on zero artifacts), but the dependency
    is what makes the artifacts exist at all -- losing it turns a real gate into a no-op that still
    reports success.
    """
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    job = wf["jobs"].get("pandas-span")
    assert job is not None, (
        "the pandas-span aggregation job is gone. The structural guard above reads ci.yml only, so "
        "without this job a pandas upper bound in pyproject.toml collapses the span invisibly."
    )
    assert job["needs"] == "test"


def test_every_test_leg_records_its_pandas_major() -> None:
    """The aggregation asserts over a UNION; a leg that records nothing shrinks it silently."""
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    steps = wf["jobs"]["test"]["steps"]
    assert any("Record resolved pandas major" in str(s.get("name", "")) for s in steps), (
        "no leg records its resolved pandas major, so the aggregation job has nothing to union"
    )
    uploads = [
        s
        for s in steps
        if "upload-artifact" in str(s.get("uses", "")) and "pandas-major" in str(s.get("with", {}).get("name", ""))
    ]
    assert uploads, "the recorded pandas major is never uploaded, so no other job can see it"
    # The artifact name must vary per leg, or every leg overwrites one artifact and the union
    # collapses to a single entry -- which would read as a lost span rather than a naming bug.
    name = str(uploads[0]["with"]["name"])
    assert "matrix.os" in name and "matrix.python-version" in name, (
        f"artifact name {name!r} is not per-leg; every leg would collide on one artifact"
    )
