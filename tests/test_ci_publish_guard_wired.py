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

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_PUBLISH = _REPO / ".github" / "workflows" / "publish.yml"


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
