"""Unit tests for the paired-leg harness.

The harness OBSERVES; it never adjudicates. That separation is the whole design, so it is
pinned here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.sb360 import _fixture as F
from tests.sb360 import _vocabulary as V
from tests.sb360._harness import CallOutcomeError, run_axis
from tests.sb360._registry import Sb360Entry


def _stub(**cols):
    def call(actions, frames, links, home_team_id):
        return actions.assign(**cols)

    return call


def _entry(call, columns=("m",)):
    return Sb360Entry(name="stub", call=call, columns=tuple(columns))


def test_observation_is_produced_by_execution_and_not_adjudicated():
    got = run_axis(_entry(_stub(m=1.0)), axis="velocity")
    assert got["m"].observation in V.OBSERVATIONS
    assert got["m"].adjudication == "", "the harness observes; it must not adjudicate"
    assert got["m"].counts is not None, "the row-class tally is recorded for interpretation"


def test_identical_output_on_both_legs_observes_identical():
    got = run_axis(_entry(_stub(m=1.0)), axis="velocity")
    assert got["m"].observation == "identical"


def test_a_column_reading_velocity_observes_differs():
    """The load-bearing case: a feature that consumes vx/vy differs between legs."""

    def call(actions, frames, links, home_team_id):
        players = frames[~frames["is_ball"].astype(bool)]
        mean_speed = float(players["speed"].mean()) if "speed" in players else float("nan")
        return actions.assign(m=mean_speed)

    got = run_axis(_entry(call), axis="velocity")
    assert got["m"].observation in {"differs", "all_nan", "partial_nan"}, (
        f"a velocity-reading feature must be distinguishable across legs, got "
        f"{got['m'].observation!r} with counts {got['m'].counts}"
    )


def test_leg_a_raise_is_recorded_as_an_observation():
    def call(actions, frames, links, home_team_id):
        raise ValueError("nope")

    got = run_axis(_entry(call), axis="velocity")
    assert got["m"].observation == "raises_a"
    assert got["m"].counts is None, "a raise produces no frame, so there are no rows to tally"


def test_leg_a_raise_records_the_exception_so_a_harness_mis_call_is_visible():
    """A wrong call signature ALSO lands in `raises_a`. Without the detail it is
    indistinguishable from a library that genuinely refuses freeze-frame input, and would be
    adjudicated as one."""

    def bad_signature(actions, frames, links, home_team_id):
        raise TypeError("add_thing() got an unexpected keyword argument 'links'")

    got = run_axis(_entry(bad_signature), axis="velocity")
    assert got["m"].observation == "raises_a"
    assert got["m"].detail is not None
    assert "TypeError" in got["m"].detail
    assert "unexpected keyword argument" in got["m"].detail


def test_leg_b_raise_is_a_fixture_failure_not_a_library_property():
    calls = {"n": 0}

    def call(actions, frames, links, home_team_id):
        calls["n"] += 1
        if calls["n"] == 2:  # Leg B
            raise RuntimeError("leg B blew up")
        return actions.assign(m=1.0)

    with pytest.raises(CallOutcomeError, match=r"raises_b"):
        run_axis(_entry(call), axis="velocity")


def test_leg_b_declining_where_leg_a_succeeds_is_a_fixture_failure():
    """The richer leg yielding LESS means the comparison is broken for those rows, not that
    the library has a property."""
    calls = {"n": 0}

    def call(actions, frames, links, home_team_id):
        calls["n"] += 1
        if calls["n"] == 2:  # Leg B
            return actions.assign(m=[np.nan] * len(actions))
        return actions.assign(m=1.0)

    with pytest.raises(CallOutcomeError, match=r"leg_b_declined"):
        run_axis(_entry(call), axis="velocity")


def test_fixture_version_appears_in_every_fixture_failure_message():
    """The lock pins the fixture as well as the library, so a fixture change and a library
    regression must be distinguishable at the point of failure."""

    def call(actions, frames, links, home_team_id):
        raise_on = getattr(call, "_n", 0) + 1
        call._n = raise_on  # type: ignore[attr-defined]
        if raise_on == 2:
            raise RuntimeError("boom")
        return actions.assign(m=1.0)

    with pytest.raises(CallOutcomeError, match=F.FIXTURE_VERSION):
        run_axis(_entry(call), axis="velocity")


@pytest.mark.parametrize("roster", ["gk_absent", "defender_absent"])
def test_visibility_axis_passes_the_roster_to_both_legs(roster):
    """Both legs take the SAME roster. Passing different rosters would vary roster AND
    velocity at once, making every verdict unattributable -- the confound Layer B's 2x2 was
    built to avoid."""
    seen: list[int] = []

    def call(actions, frames, links, home_team_id):
        seen.append(int((~frames["is_ball"].astype(bool)).sum()))
        return actions.assign(m=1.0)

    run_axis(_entry(call), axis="visibility", roster=roster)
    assert len(seen) == 2
    per_frame_a = seen[0] / len(F._ACTIONS)
    n_frames_b = len(set(F.build_leg_b(roster=roster)[1]["frame_id"]))
    per_frame_b = seen[1] / n_frames_b
    assert per_frame_a == per_frame_b, (
        f"legs carry different rosters ({per_frame_a} vs {per_frame_b} players per frame)"
    )


def test_velocity_axis_ignores_the_roster_argument():
    """The velocity axis holds roster FIXED at the full complement whatever is passed."""
    seen: list[int] = []

    def call(actions, frames, links, home_team_id):
        seen.append(int((~frames["is_ball"].astype(bool)).sum()))
        return actions.assign(m=1.0)

    run_axis(_entry(call), axis="velocity", roster="gk_absent")
    full_a = len(F.build_leg_a(roster="full")[1])
    assert seen[0] == full_a - len(F._ACTIONS), "velocity axis must use the full roster"


def test_per_column_tolerance_override_is_honoured():
    """A loosened tolerance converts `differs` into `identical`, so the harness must actually
    read the registry's per-column value."""

    def call(actions, frames, links, home_team_id):
        has_v = "vx" in frames.columns
        return actions.assign(m=1.0 if has_v else 1.0 + 1e-6)

    strict = run_axis(_entry(call), axis="velocity")
    assert strict["m"].observation == "differs"

    loose = Sb360Entry(
        name="stub",
        call=call,
        columns=("m",),
        tolerances={"m": (1e-3, 1e-3)},
        tolerance_basis={"m": "probe"},
    )
    assert run_axis(loose, axis="velocity")["m"].observation == "identical"
