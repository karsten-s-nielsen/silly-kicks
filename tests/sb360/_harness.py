"""Runs one registry entry on both legs and returns OBSERVATIONS only.

The harness never adjudicates. That separation is the whole design: the machine half is
re-derived and locked on every CI run, the human half carries the judgement and its rationale.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

from tests.sb360 import _fixture as F
from tests.sb360._compare import DEFAULT_ATOL, DEFAULT_RTOL, compare_column
from tests.sb360._registry import AxisVerdict, Sb360Entry


class CallOutcomeError(AssertionError):
    """A fixture-integrity failure -- ``raises_b`` or ``leg_b_declined``.

    Never a library property: both mean the comparison itself is broken, so recording them as
    verdicts would attribute a fixture defect to the library under audit.
    """


def run_axis(entry: Sb360Entry, *, axis: str, roster: str = "full") -> dict[str, AxisVerdict]:
    """Execute ``entry`` on both legs and return one observation per emitted column.

    ``axis`` selects which factor varies:

    * ``"velocity"`` -- roster held FIXED at the full complement (the ``roster`` argument is
      deliberately ignored); only kinematics differ between legs.
    * ``"visibility"`` -- ``roster`` applied to BOTH legs, so kinematics are held fixed and
      only the player set differs.

    Passing different rosters to the two legs would vary roster and velocity together and make
    every verdict unattributable -- the confound the Layer B 2x2 was built to avoid.
    """
    effective_roster = roster if axis == "visibility" else "full"
    actions_a, frames_a, links_a = F.build_leg_a(roster=effective_roster)
    actions_b, frames_b, links_b = F.build_leg_b(roster=effective_roster)

    try:
        out_a = entry.call(actions_a, frames_a, links_a, F.HOME_TEAM_ID)
    except Exception as exc:
        # `raises_a` is normally a genuine library property: the feature refuses freeze-frame
        # input. No output frame exists, so there are no rows to classify and `counts` stays
        # None.
        #
        # The exception is RECORDED rather than discarded. A harness mis-call -- a wrong
        # signature producing `TypeError: unexpected keyword argument` -- also lands here, and
        # without the detail it is indistinguishable from a library property and would be
        # adjudicated as one. TypeError is not narrowed out, because a library may legitimately
        # raise it (the 2026-05-27 spec pins exactly that for a missing `home_team_id`); only a
        # human reading the message can tell the two apart.
        detail = f"{type(exc).__name__}: {exc}"
        return {c: AxisVerdict(observation="raises_a", adjudication="", detail=detail) for c in entry.columns}

    try:
        out_b = entry.call(actions_b, frames_b, links_b, F.HOME_TEAM_ID)
    except Exception as exc:
        raise CallOutcomeError(
            f"{entry.name}: call outcome `raises_b` on fixture {F.FIXTURE_VERSION} -- Leg B "
            f"raised where Leg A succeeded ({exc!r}). Leg B is a synthetic full-tracking "
            f"construction, so this is a FIXTURE defect and is never recorded as a library "
            f"property."
        ) from exc

    result: dict[str, AxisVerdict] = {}
    for col in entry.columns:
        rtol, atol = entry.tolerances.get(col, (DEFAULT_RTOL, DEFAULT_ATOL))
        obs, counts = compare_column(out_a[col], out_b[col], rtol=rtol, atol=atol)
        if obs == "leg_b_declined":
            raise CallOutcomeError(
                f"{entry.name}.{col}: observation `leg_b_declined` on fixture "
                f"{F.FIXTURE_VERSION} -- {counts['row_nan_b']} of {sum(counts.values())} rows "
                f"are NaN on Leg B where Leg A is finite. The richer leg yielded less, so the "
                f"comparison is broken for those rows. Row classes: {counts}."
            )
        result[col] = AxisVerdict(observation=obs, adjudication="", counts=counts)
    return result


#: What made a column's legs disagree. Written into the rationale, never into the lock.
CAUSE_VELOCITY = "velocity"
CAUSE_FRAME_COUNT = "frame_count"
CAUSE_BOTH = "velocity+frame_count"
CAUSE_NEITHER = "neither"


def diagnose_cause(entry: Sb360Entry, column: str, *, roster: str = "full") -> str:
    """Isolate WHY a column's legs disagree: velocity, frame count, both, or neither.

    ``differs`` and ``all_nan`` are each reachable two ways, and only one is a finding about
    freeze-frames. Comparing against the anchor-only diagnostic leg separates them:

    * Leg A vs anchor-only (1 frame both sides) -- any disagreement is **velocity**.
    * anchor-only vs full Leg B (velocity both sides) -- any disagreement is **frame count**.

    An adjudicator writing ``silent_degrade`` from the un-isolated reading would be attributing
    a temporal-window requirement to fabricated kinematics.
    """
    effective_roster = roster if roster != "full" else "full"
    a_actions, a_frames, a_links = F.build_leg_a(roster=effective_roster)
    n_actions, n_frames, n_links = F.build_leg_b_anchor_only(roster=effective_roster)
    b_actions, b_frames, b_links = F.build_leg_b(roster=effective_roster)

    rtol, atol = entry.tolerances.get(column, (DEFAULT_RTOL, DEFAULT_ATOL))

    def _obs(x_actions, x_frames, x_links, y_actions, y_frames, y_links) -> str:
        out_x = entry.call(x_actions, x_frames, x_links, F.HOME_TEAM_ID)
        out_y = entry.call(y_actions, y_frames, y_links, F.HOME_TEAM_ID)
        obs, _ = compare_column(out_x[column], out_y[column], rtol=rtol, atol=atol)
        return obs

    velocity_moved = _obs(a_actions, a_frames, a_links, n_actions, n_frames, n_links) != "identical"
    frames_moved = _obs(n_actions, n_frames, n_links, b_actions, b_frames, b_links) != "identical"

    if velocity_moved and frames_moved:
        return CAUSE_BOTH
    if velocity_moved:
        return CAUSE_VELOCITY
    if frames_moved:
        return CAUSE_FRAME_COUNT
    return CAUSE_NEITHER
