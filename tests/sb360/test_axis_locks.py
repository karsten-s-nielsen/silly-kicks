"""The lock. Re-derives every observation and asserts it against the registry.

Repair a function and its observation changes, CI fails, and the adjudication is forced to be
revisited. Locking the KEY SET alone would let a repaired function keep a stale verdict while
CI stayed green -- the defect the 2026-05-27 compatibility table demonstrated.

The lock covers the MACHINE half only. Adjudications and rationales are human judgement and no
gate can check them; what the gate guarantees is that a stale one cannot hide.
"""

from __future__ import annotations

import pytest

from tests.sb360 import _fixture as F
from tests.sb360._adjudicate import classify
from tests.sb360._harness import run_axis
from tests.sb360._probes import derive_applicability
from tests.sb360._registry import SB360_ENTRIES, iter_verdicts

#: (axis, roster). Velocity holds roster fixed; visibility varies it at fixed kinematics. Both
#: visibility rosters are swept because a feature can survive a missing outfielder and collapse
#: on a missing keeper -- which is why each roster has its own verdict slot.
_AXES = [
    ("velocity", "full"),
    ("visibility", "gk_absent"),
    ("visibility", "defender_absent"),
    ("visibility", "gk_one_end"),
]


@pytest.mark.slow
@pytest.mark.parametrize(("axis", "roster"), _AXES)
@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_observations_match_the_registry(name, axis, roster):
    entry = SB360_ENTRIES[name]
    observed = run_axis(entry, axis=axis, roster=roster)
    recorded = entry.velocity if axis == "velocity" else entry.visibility[roster]
    for col, got in observed.items():
        assert col in recorded, f"{name}.{col} ({axis}/{roster}) is emitted but unrecorded"
        expected = recorded[col].observation
        assert got.observation == expected, (
            f"{name}.{col} ({axis}/{roster}): observed {got.observation!r}, registry says "
            f"{expected!r}. Fixture {F.FIXTURE_VERSION}. If the FIXTURE changed, bump "
            f"FIXTURE_VERSION and re-record; if the LIBRARY changed, re-adjudicate. "
            f"Row classes: {got.counts}."
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_applicability_class_matches_the_registry(name):
    entry = SB360_ENTRIES[name]
    for col, expected in entry.applicability.items():
        got, deltas = derive_applicability(entry, col)
        assert got == expected, (
            f"{name}.{col}: probes derived {got!r}, registry says {expected!r}. "
            f"Measured deltas: {deltas}. Fixture {F.FIXTURE_VERSION}."
        )
        # A zero-delta `no_support` is indistinguishable from a probe that never ran. Any OTHER
        # class is a positive claim and must be backed by measurable movement.
        if expected != "no_support":
            probe = "extreme" if expected == "support_data_defined" else "near"
            assert deltas[probe] > 0.0, (
                f"{name}.{col}: class {expected!r} recorded but the {probe} probe moved "
                f"nothing ({deltas}). The classification would be vacuous."
            )


@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_no_signal_is_acknowledged_on_every_axis(name):
    """Per-column liveness, per AXIS. Set equality guarantees stability, not meaningfulness:
    a dead column would otherwise lock as `identical` and match forever."""
    entry = SB360_ENTRIES[name]
    for axis, roster, col, v in iter_verdicts(entry):
        if v.observation == "no_signal":
            assert v.adjudication == "not_exercised", (
                f"{name}.{col} ({axis}/{roster}): observed no_signal but is not adjudicated "
                f"not_exercised -- an unexercised column must be acknowledged, not absorbed"
            )


def test_the_canary_proves_the_legs_are_distinguishable():
    """Non-vacuity. NOT a `differs` canary -- naming a silently-degrading column in advance
    would assert the audit's OUTPUT as its input. `actor_speed` is the MODEL CITIZEN: NaN on
    Leg A (_snapshot.py sets speed=NaN; _kernels.py fills only where notna) against finite on
    Leg B. Anything other than `identical` proves the legs are distinguishable."""
    entry = SB360_ENTRIES["add_action_context"]
    observed = run_axis(entry, axis="velocity")
    got = observed["actor_speed"].observation
    assert got != "identical", (
        f"actor_speed observed {got!r} -- the legs are not distinguishable, so every "
        f"`identical` verdict in this audit is vacuous. Row classes: "
        f"{observed['actor_speed'].counts}."
    )


def test_the_machinery_can_still_express_a_fabrication():
    """The audit must be capable of returning its headline finding.

    Not a claim that fabrication EXISTS -- a claim that the machinery can express it. If every
    verdict were `works` or `honest_nan`, an audit that found nothing would be indistinguishable
    from an audit that could not find anything.

    **This test used to search the real registry for a `silent_degrade` and its failure message
    said "establish which before deleting this test". That question is now ANSWERED: the library's
    one fabrication (`add_ghost_gk` serving an HGBR imputed-feature vector on freeze-frames) was
    REPAIRED, the ghost positions re-derive to `all_nan` -> `honest_nan`, and the audit reports
    zero fabrications.**

    So the assertion moved to where it always belonged: a PLANTED case against the rule engine.
    The old form coupled a property of the HARNESS to the continued existence of a DEFECT in the
    library -- fixing the defect broke the guard, which is exactly backwards.
    """
    # A fitted-model column observed as `differs` with velocity isolated as the cause is the
    # fabrication shape. `add_ghost_gk` is still in FITTED_MODEL; only its POSITIONS stopped
    # observing `differs`, because they now decline.
    adjudication, rationale = classify(
        fn="add_ghost_gk",
        col="ghost_gk_x",
        roster="full",
        obs="differs",
        cause="velocity",
    )
    assert adjudication == "silent_degrade", (
        f"the rule engine can no longer express a fabrication: a FITTED_MODEL column observed "
        f"`differs` with cause=velocity classified as {adjudication!r}. The audit's headline "
        f"finding has become unreachable."
    )
    assert rationale, "a silent_degrade verdict must carry a rationale"


def test_no_column_is_currently_adjudicated_a_fabrication():
    """The repaired state, asserted so a regression is loud.

    Companion to the test above: that one proves the machinery CAN say `silent_degrade`, this one
    records that today nothing DOES. A new fabrication -- or a regression of the ghost refusal --
    fails here by name.
    """
    fabrications = [
        f"{e.name}.{col} ({axis}/{roster})"
        for e in SB360_ENTRIES.values()
        for axis, roster, col, v in iter_verdicts(e)
        if v.adjudication == "silent_degrade"
    ]
    assert not fabrications, (
        f"a column is adjudicated `silent_degrade`: {fabrications}. Either a new fabrication "
        f"landed, or the ghost-GK velocity refusal regressed. Both are real findings."
    )
