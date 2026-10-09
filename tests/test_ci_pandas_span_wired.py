"""Structural guard: CI's leg set must still span both pandas majors.

``pyproject.toml`` pins ``pandas>=2.1.1,!=3.0.4`` with NO upper bound, so pip resolves the newest compatible
pandas per interpreter -- and pandas 3 requires Python >= 3.11 (3.0.5 declares ``requires_python >=3.11``,
2.3.3 declares ``>=3.9``). So ubuntu-3.10 -> pandas 2, every other leg -> pandas 3.

That differential coverage is REAL but was ACCIDENTAL -- nothing declared it, so it could vanish with no
diff. This guard declares it. The repo already has one measured instance of a silent pandas-3 behaviour
change (DAS going all-NaN), which is the class this coverage exists to expose.

Since the asymmetric-sharding refactor the matrix is split into ``test-ubuntu`` (os=ubuntu-latest x
python-version axis) and ``test-windows`` (os=windows-latest, python 3.12 fixed). The leg set is built from
BOTH. **This asserts over the RESOLVED LEG SET, never a python-version axis alone** -- removing the 3.10 leg
collapses the pandas-2 span, and that must fail here.

**What this guard CANNOT see:** a span collapse caused by a dependency constraint rather than a matrix edit
(e.g. adding ``pandas<3`` to ``pyproject.toml``). That hazard is covered by the ``pandas-span`` aggregation
job in ``ci.yml``, which observes what each leg actually installed. The two are complementary.
"""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"

#: the sharded matrix jobs whose legs resolve pandas.
_MATRIX_JOBS = ("test-ubuntu", "test-windows")

#: pandas 3 requires Python >= 3.11, so a leg below it resolves pandas 2 and a leg at or above it resolves
#: pandas 3. This is the ASSUMPTION that makes a structural check a valid proxy for the span. If pandas
#: changes its minimum Python, THIS CONSTANT is what moves -- not the assertion, and never by redefining the
#: boundary to match a matrix that lost its leg.
_PANDAS3_MIN_PY = (3, 11)


def _pyver(py: str) -> tuple[int, ...]:
    return tuple(int(p) for p in str(py).split("."))


def _legs(wf: dict) -> list[tuple[str, str]]:
    """``(os, python)`` legs across both matrix jobs. python comes from the ``python-version`` matrix axis
    when present (ubuntu), else the job's ``setup-python`` literal (windows)."""
    legs: list[tuple[str, str]] = []
    for job_name in _MATRIX_JOBS:
        job = wf["jobs"][job_name]
        os_ = str(job["runs-on"])
        matrix = job["strategy"]["matrix"]
        if "python-version" in matrix:
            pys = [str(p) for p in matrix["python-version"]]
        else:
            setup = [s for s in job["steps"] if "setup-python" in str(s.get("uses", ""))]
            assert setup, f"{job_name}: no python-version axis and no setup-python step to read it from"
            pys = [str(setup[0]["with"]["python-version"])]
        legs += [(os_, py) for py in pys]
    return legs


def test_ci_leg_set_spans_both_pandas_majors() -> None:
    legs = _legs(yaml.safe_load(_CI.read_text(encoding="utf-8")))
    below = [leg for leg in legs if _pyver(leg[1]) < _PANDAS3_MIN_PY]
    at_or_above = [leg for leg in legs if _pyver(leg[1]) >= _PANDAS3_MIN_PY]
    assert below and at_or_above, (
        f"CI's resolved leg set no longer straddles Python {_PANDAS3_MIN_PY[0]}.{_PANDAS3_MIN_PY[1]}, so "
        f"every leg resolves the SAME pandas major and the differential coverage this repo relies on is "
        f"gone. legs={legs}. ASSUMPTION: pandas 3 requires Python >= {_PANDAS3_MIN_PY[0]}.{_PANDAS3_MIN_PY[1]}. "
        f"If pandas changed that, fix _PANDAS3_MIN_PY -- do NOT delete this assertion, and do not 'fix' it by "
        f"moving the boundary to match a matrix that lost its old leg."
    )


def test_leg_builder_reads_the_pandas2_leg_not_just_the_axis() -> None:
    """Non-vacuity: the span must rest on a REAL resolved leg below 3.11 (ubuntu-3.10), so dropping that
    python from the test-ubuntu axis would flip ``test_ci_leg_set_spans_both_pandas_majors`` red."""
    legs = _legs(yaml.safe_load(_CI.read_text(encoding="utf-8")))
    assert ("ubuntu-latest", "3.10") in legs, (
        f"the pandas-2 span rests on ubuntu-latest/3.10; it is not in the resolved legs {legs}"
    )


def test_the_aggregation_job_exists_and_needs_the_matrix_jobs() -> None:
    """Without ``needs`` on the matrix jobs the aggregation runs before the artifacts exist and passes
    vacuously. The job's own script also exits non-zero on zero artifacts, but the dependency is what makes
    the artifacts exist at all."""
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    job = wf["jobs"].get("pandas-span")
    assert job is not None, (
        "the pandas-span aggregation job is gone. The structural guard above reads ci.yml only, so without "
        "this job a pandas upper bound in pyproject.toml collapses the span invisibly."
    )
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    for required in _MATRIX_JOBS:
        assert required in needs, f"pandas-span must need {required} (else it runs before artifacts exist), got {needs}"


def test_every_matrix_job_records_its_pandas_major_per_leg() -> None:
    """The aggregation asserts over a UNION; a leg that records nothing shrinks it silently, and two legs
    sharing one artifact name collide (upload-artifact@v4 hard-fails on a duplicate)."""
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    names: list[str] = []
    for job_name in _MATRIX_JOBS:
        steps = wf["jobs"][job_name]["steps"]
        assert any("Record resolved pandas major" in str(s.get("name", "")) for s in steps), (
            f"{job_name}: no leg records its resolved pandas major, so the aggregation has nothing to union"
        )
        uploads = [
            s
            for s in steps
            if "upload-artifact" in str(s.get("uses", "")) and "pandas-major" in str(s.get("with", {}).get("name", ""))
        ]
        assert uploads, f"{job_name}: the recorded pandas major is never uploaded, so no other job can see it"
        names.append(str(uploads[0]["with"]["name"]))
    # test-ubuntu's name must vary per python (its axis); test-windows is a single literal leg. Either way the
    # two jobs' names must not collide, or legs would overwrite one artifact and the union would collapse.
    ub_name = next(n for n in names if "ubuntu" in n)
    assert "matrix.python-version" in ub_name, f"test-ubuntu pandas-major name {ub_name!r} is not per-python leg"
    win_name = next(n for n in names if "windows" in n)
    assert win_name != ub_name and "matrix.python-version" not in win_name, (
        f"test-windows pandas-major name {win_name!r} must be a distinct single-leg literal"
    )


def test_pandas_major_record_and_upload_are_shard_1_gated() -> None:
    """Under sharding, record + upload run on shard 1 ONLY. The artifact name has NO shard component, so if
    all N shards of a leg recorded it they would collide and upload-artifact@v4 hard-fails. Pins the
    ``matrix.shard == 1`` gate on both steps, in both matrix jobs."""

    def guard(s: dict) -> str:
        g = "".join(str(s.get("if", "")).split())
        g = g[3:-2] if g.startswith("${{") and g.endswith("}}") else g
        return g.replace("'", "").replace('"', "")

    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    for job_name in _MATRIX_JOBS:
        steps = wf["jobs"][job_name]["steps"]
        record = [s for s in steps if "Record resolved pandas major" in str(s.get("name", ""))]
        uploads = [
            s
            for s in steps
            if "upload-artifact" in str(s.get("uses", "")) and "pandas-major" in str(s.get("with", {}).get("name", ""))
        ]
        assert record and uploads, f"{job_name}: pandas-major record/upload steps missing"
        for s in record + uploads:
            assert "matrix.shard==1" in guard(s), (
                f"{job_name}: pandas-major step must be shard-1-gated (else N shards collide), got {guard(s)!r}"
            )
