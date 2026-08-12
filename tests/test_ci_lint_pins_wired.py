"""Structural guard: every TYPING input to the lint job is pinned, numpy included.

pyright's verdict is a function of the type information available to it, and that comes from four
places in this job -- the checker itself, ruff, `pandas-stubs`, and **numpy**, which ships inline
types rather than a stubs package. Three of the four were exact-pinned; numpy was not, and the
omission was an inconsistency rather than a decision.

Measured, on two runs a day apart: main's lint job resolved numpy 2.5.2 and then DOWNGRADED to
2.4.6 (last install wins), while PR-S150's stayed on 2.5.2. The seven pyright errors that
difference produced sat in three files the PR's diff could not reach -- byte-identical to main, and
importing nothing from any changed module. A gate whose verdict moves with resolution luck blocks
unrelated work and, worse, goes green again on a re-run, which teaches everyone to re-run it.

WHY THE PLACEMENT IS THE PROPERTY, not merely the presence of a pin: the job installs the pinned
tools first and THEN runs `pip install -e ".[test]"`, and it is that second resolve which moves
numpy. A `numpy==` added to the first line would be overridden by the second -- the pin would be
visibly present and bind nothing. So this asserts the pin lives in the LAST install step, which is
the only position that survives the re-resolve.

The TEST jobs are deliberately NOT covered here. There, an unpinned numpy/pandas is COVERAGE
(ADR-057's span), and it earned its keep in this same release by catching a pandas-3
copy-on-write defect that the 3.10 leg structurally could not see. Typing inputs and behavioural
inputs want opposite policies; conflating them is how you get either a flaky gate or a blind one.
"""

from __future__ import annotations

import pathlib
import re

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"

#: Every package whose version can change what pyright REPORTS. numpy belongs here for the same
#: reason pandas-stubs does: it is where the types come from.
_TYPING_INPUTS = ("ruff", "pyright", "pandas-stubs", "numpy")


def _lint_steps() -> list[dict]:
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    return wf["jobs"]["lint"]["steps"]


def _install_steps(steps: list[dict]) -> list[tuple[int, str]]:
    """(index, command) for every step that runs a pip install."""
    out = []
    for i, step in enumerate(steps):
        run = step.get("run", "")
        if re.search(r"\bpip\s+install\b", run):
            out.append((i, run))
    return out


def _pins_in(command: str) -> set[str]:
    """Packages exact-pinned (``==``) in one pip command."""
    return {m.group(1) for m in re.finditer(r"[\"']?([A-Za-z0-9_.-]+)==[0-9][^\s\"']*", command)}


def test_every_typing_input_is_exact_pinned() -> None:
    steps = _lint_steps()
    installs = _install_steps(steps)
    assert installs, "lint job runs no pip install -- discovery is broken, not the pins"

    pinned: set[str] = set()
    for _, cmd in installs:
        pinned |= _pins_in(cmd)

    missing = [pkg for pkg in _TYPING_INPUTS if pkg not in pinned]
    assert not missing, (
        f"lint job does not exact-pin {missing}. Every input that can change what pyright REPORTS "
        f"must be pinned, or the gate's verdict moves with dependency resolution and an unrelated "
        f"PR goes red with no diff. Pinned: {sorted(pinned)}"
    )


def test_numpy_is_pinned_in_the_LAST_install_that_can_move_it() -> None:
    """Presence is not enough -- an earlier pin is overridden by the later re-resolve."""
    installs = _install_steps(_lint_steps())
    last_index, last_cmd = installs[-1]

    assert "numpy" in _pins_in(last_cmd), (
        "numpy is not pinned in the LAST pip install of the lint job "
        f"(step {last_index}: {last_cmd!r}). An earlier pin does not bind: that final install "
        "re-resolves the dependency graph and is exactly what moved numpy 2.5.2 -> 2.4.6 on main. "
        "The pin must sit on the last install, or it is decorative."
    )


def test_the_guard_would_CATCH_a_pin_that_only_sits_on_the_first_install() -> None:
    """Non-vacuity, against the specific broken arrangement this guard exists to reject.

    Without this, `test_numpy_is_pinned_in_the_LAST_install...` could pass for the wrong reason --
    e.g. if the job ever collapsed to a single install step, the "last" step would trivially be the
    only step and the ordering property would stop being tested at all.
    """
    broken = [
        {"run": "pip install ruff==0.15.7 pyright==1.1.409 pandas-stubs==2.3.3.260113 numpy==2.5.2"},
        {"run": "ruff check silly_kicks/"},
        {"run": 'pip install -e ".[test]"'},  # re-resolves, silently undoing the pin above
        {"run": "pyright"},
    ]
    installs = _install_steps(broken)
    assert len(installs) == 2, "the planted case must have two installs, or it is not the shape under test"
    _, last_cmd = installs[-1]
    assert "numpy" not in _pins_in(last_cmd), (
        "the planted broken arrangement is not actually broken -- this non-vacuity check is asserting nothing"
    )


def test_test_jobs_are_deliberately_NOT_pinned() -> None:
    """The other half of the policy, pinned so it cannot be 'tidied' into consistency.

    Someone applying the lint job's rule uniformly would pin the test matrix too, and that would
    silently delete ADR-057's span -- the thing that caught this release's pandas-3 defect. The
    asymmetry is intentional and is therefore asserted.
    """
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    for name, job in wf["jobs"].items():
        if name == "lint":
            continue
        for step in job.get("steps", []):
            run = step.get("run", "")
            if not re.search(r"\bpip\s+install\b", run):
                continue
            offenders = _pins_in(run) & {"numpy", "pandas"}
            assert not offenders, (
                f"job {name!r} exact-pins {sorted(offenders)}. The test matrix must resolve these "
                f"FREELY: ADR-057's pandas-major span is coverage, and pinning it would have hidden "
                f"the pandas-3 copy-on-write defect fixed in 4.80.0."
            )
