"""Structural guard: the publish workflow refuses a tag that disagrees with the built version.

The tag name is a LABEL -- ``publish.yml`` checks out the tagged commit and builds from ITS
``pyproject.toml``, so a tag on the wrong commit uploads a version other than the one it names.
Nothing compared the two until 4.77.1, and **PyPI uploads are irreversible**: the only recovery from
a wrong publish is burning another version number.

Asserted STRUCTURALLY, not by substring: the guard must live in the job that builds, and must run
BEFORE the artifact upload. A guard that ran after upload -- or in the ``publish`` job, after the
artifact is already built and handed over -- would still let a mismatched wheel reach the point of
no return. The step ORDER is the property, so the order is what is asserted.

The guard's own behaviour (which tag/wheel pairs pass and fail) is verified by executing it; see the
PR for the case table. This file pins that it is WIRED, which execution cannot show.
"""

from __future__ import annotations

import pathlib
import re

import yaml
from packaging.requirements import Requirement

_REPO = pathlib.Path(__file__).resolve().parent.parent
_PUBLISH = _REPO / ".github" / "workflows" / "publish.yml"
_PYPROJECT = _REPO / "pyproject.toml"

# An upper bound is any operator that can REFUSE a future release. `>=` and `!=` cannot.
_BOUNDING_OPERATORS = frozenset({"<", "<=", "==", "===", "~="})


def _build_system_requires() -> list[str]:
    """Extract `[build-system] requires` WITHOUT a TOML parser.

    `tomllib` is 3.11+, this package supports 3.10, and `tomli` is not in the `[test]` extra --
    importing it would pass locally and error on the 3.10 CI leg. The target is one array in a
    file this repo owns, so a scoped text read is honest here; callers assert non-vacuity.
    """
    text = _PYPROJECT.read_text(encoding="utf-8")
    table = re.search(r"^\[build-system\](.*?)(?=^\[)", text, re.MULTILINE | re.DOTALL)
    if table is None:
        return []
    array = re.search(r"^requires\s*=\s*\[(.*?)\]", table.group(1), re.MULTILINE | re.DOTALL)
    if array is None:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", array.group(1))


def _is_bounded(requirement: str) -> bool:
    return any(spec.operator in _BOUNDING_OPERATORS for spec in Requirement(requirement).specifier)


def _build_steps() -> list[dict]:
    wf = yaml.safe_load(_PUBLISH.read_text(encoding="utf-8"))
    return wf["jobs"]["build"]["steps"]


def _index_of(steps: list[dict], predicate) -> int:
    for i, step in enumerate(steps):
        if predicate(step):
            return i
    return -1


def test_publish_workflow_triggers_only_on_version_tags() -> None:
    """If the trigger widened, the guard's premise (`GITHUB_REF_NAME` is a version tag) dissolves."""
    wf = yaml.safe_load(_PUBLISH.read_text(encoding="utf-8"))
    # PyYAML parses a bare `on:` key as the boolean True (the Norway problem's cousin).
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"push"}, f"publish must trigger on push only, got {sorted(triggers)}"
    assert triggers["push"] == {"tags": ["v*"]}, triggers["push"]


def test_the_version_guard_runs_in_build_BEFORE_the_artifact_upload() -> None:
    """Order is the property. After the upload, the mismatched wheel is already handed over."""
    steps = _build_steps()

    guard = _index_of(steps, lambda s: "Tag must match the built version" in str(s.get("name", "")))
    assert guard >= 0, (
        "no tag-vs-built-version guard in the `build` job. publish.yml builds from the TAGGED "
        "commit's pyproject.toml and PyPI uploads are irreversible -- restore it."
    )

    build = _index_of(steps, lambda s: "python -m build" in str(s.get("run", "")))
    upload = _index_of(steps, lambda s: "upload-artifact" in str(s.get("uses", "")))
    assert build >= 0 and upload >= 0, "publish.yml no longer builds and uploads as expected"

    assert build < guard < upload, (
        f"the guard must sit AFTER the build (so it can read dist/) and BEFORE the upload "
        f"(so a mismatch never reaches publish): build={build} guard={guard} upload={upload}"
    )


def test_the_guard_compares_the_BUILT_artifact_not_pyproject() -> None:
    """Reading pyproject would re-assert the input; the wheel is what actually uploads.

    A guard on pyproject passes whenever pyproject and the wheel agree -- which is always, since
    hatchling builds one from the other. It would prove nothing about what reaches PyPI.
    """
    steps = _build_steps()
    guard = steps[_index_of(steps, lambda s: "Tag must match the built version" in str(s.get("name", "")))]
    run = str(guard.get("run", ""))
    assert "dist" in run and ".whl" in run, "the guard must inspect the built wheel in dist/"
    assert "GITHUB_REF_NAME" in run, "the guard must read the tag from GITHUB_REF_NAME"
    assert "pyproject" not in run, (
        "the guard reads pyproject.toml -- that re-asserts the build INPUT rather than the artifact "
        "that gets uploaded, and the two agree by construction"
    )


def test_the_publish_job_still_depends_on_build() -> None:
    """The guard only gates publishing while `publish` needs `build`; drop that and it is bypassed."""
    wf = yaml.safe_load(_PUBLISH.read_text(encoding="utf-8"))
    assert wf["jobs"]["publish"]["needs"] == "build"


def test_the_build_backend_is_BOUNDED_so_the_artifact_is_not_a_function_of_WALL_CLOCK_TIME() -> None:
    """Every Action here is SHA-pinned; leaving the BACKEND unbounded made the artifact drift.

    Measured 2026-08-11, and the window was under four hours: `requires = ["hatchling"]` resolved
    1.31.0 at 01:39Z (`Metadata-Version: 2.4` -- v4.78.0 published) and 1.32.0 at 12:56Z (2.5),
    which the publish action's then-pinned `packaging==25.0` refused outright, so v4.79.0 built
    green and failed to upload with NO diff between the two runs.

    The bound is half the fix and cannot stop a repeat alone -- the publish action's own packaging
    pin is the other half, and nothing static can compare a producer to a validator baked into a
    container image. What it buys is that a backend release shows up in a DIFF instead of in a
    failed publish, which is the difference between a decision and an accident.
    """
    requires = _build_system_requires()
    assert requires, (
        "could not read [build-system] requires from pyproject.toml -- the extractor broke, so "
        "this guard was passing without checking anything"
    )
    assert any(Requirement(r).name == "hatchling" for r in requires), (
        f"hatchling is no longer the declared build backend: {requires}. If the backend changed "
        "deliberately, retarget this guard rather than deleting it."
    )
    unbounded = [r for r in requires if not _is_bounded(r)]
    assert not unbounded, (
        f"build-system requirement(s) with no upper bound: {unbounded}. An unbounded backend makes "
        "the published artifact a function of when it was built."
    )


def test_the_bound_predicate_would_actually_reject_the_form_that_broke_publishing() -> None:
    """Non-vacuity: the guard above passes trivially if `_is_bounded` says yes to everything.

    `"hatchling"` is the EXACT string that was in pyproject.toml when v4.79.0 failed, so this
    pins that the predicate rejects the real defect and not merely a hypothetical one.
    """
    assert not _is_bounded("hatchling"), "the unbounded form must be rejected"
    assert not _is_bounded("hatchling>=1.27"), "a floor alone bounds nothing above"
    assert _is_bounded("hatchling>=1.27,<2")
    assert _is_bounded("hatchling==1.31.0")
    assert _is_bounded("hatchling~=1.31")
