"""The gate that stops the design acquiring another hole.

Five revisions of the design spec acquired the SAME defect -- a state introduced at one level
and not propagated into every table that claims completeness. Reviewing again finds the next
instance; it does not stop the next instance. This does.

Decision: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

from tests.sb360 import _vocabulary as V


def test_every_call_outcome_is_placed():
    """Each call outcome is a precedence rule OR the declared row-classification precondition."""
    in_precedence = {name for _, name in V.PRECEDENCE}
    placed = in_precedence | {V.ROW_CLASSIFICATION_PRECONDITION}
    missing = V.CALL_OUTCOMES - placed
    assert not missing, (
        f"call_outcome(s) {sorted(missing)} appear in no precedence rule and are not the "
        f"declared row-classification precondition. This is the rev-4 defect: raises_b was "
        f"defined at Level 1 and absent from the table claiming to be the complete procedure."
    )


def test_every_row_class_is_consumed():
    assert set(V.ROW_CLASS_CONSUMERS) == set(V.ROW_CLASSES), (
        f"ROW_CLASS_CONSUMERS keys {sorted(V.ROW_CLASS_CONSUMERS)} != ROW_CLASSES {sorted(V.ROW_CLASSES)}"
    )
    consumed: set[str] = set().union(*V.ROW_CLASS_CONSUMERS.values())
    assert consumed <= V.OBSERVATIONS, f"unknown observations referenced: {sorted(consumed - V.OBSERVATIONS)}"
    unconsumed = [rc for rc, obs in V.ROW_CLASS_CONSUMERS.items() if not obs]
    assert not unconsumed, f"row_class(es) {sorted(unconsumed)} are consumed by no precedence rule"


def test_every_observation_carries_a_kind():
    producible = {name for _, name in V.PRECEDENCE}
    missing = producible - set(V.OBSERVATION_KIND)
    assert not missing, f"observation(s) {sorted(missing)} carry no kind"
    bad = {o: k for o, k in V.OBSERVATION_KIND.items() if k not in V.KINDS}
    assert not bad, f"unknown kind(s): {bad}"


def test_adjudicated_and_budgeted_observations_are_admissible_somewhere():
    reachable: set[str] = set().union(*V.ADMISSIBLE_FROM.values())
    for obs, kind in V.OBSERVATION_KIND.items():
        if kind in {"adjudicated", "budgeted"}:
            assert obs in reachable, (
                f"observation {obs!r} has kind {kind!r} so it reaches the registry, but no "
                f"adjudication admits it -- it would be unadjudicatable"
            )


def test_terminal_observations_are_absent_from_admissibility():
    reachable: set[str] = set().union(*V.ADMISSIBLE_FROM.values())
    for obs, kind in V.OBSERVATION_KIND.items():
        if kind == "terminal_fixture_failure":
            assert obs not in reachable, (
                f"observation {obs!r} is a terminal fixture failure and must never reach the "
                f"registry, but an adjudication admits it"
            )


def test_every_adjudication_is_reachable():
    assert set(V.ADMISSIBLE_FROM) == set(V.ADJUDICATIONS), (
        f"ADMISSIBLE_FROM keys != ADJUDICATIONS: {sorted(set(V.ADMISSIBLE_FROM) ^ set(V.ADJUDICATIONS))}"
    )
    orphans = [adj for adj, obs in V.ADMISSIBLE_FROM.items() if not obs]
    assert not orphans, f"adjudication(s) {sorted(orphans)} are reachable from no observation"


def test_shared_names_are_declared():
    """A name in two vocabularies is deliberate or it is a bug. Nothing in between."""
    overlap = V.CALL_OUTCOMES & V.OBSERVATIONS
    assert overlap == V.SHARED_NAMES, (
        f"call_outcome/observation name overlap {sorted(overlap)} != declared "
        f"SHARED_NAMES {sorted(V.SHARED_NAMES)}. Rev 4 had observation 'raises' and "
        f"adjudication 'raises' meaning different things; namespacing plus this "
        f"assertion is what makes reuse expressible rather than accidental."
    )


def test_rationale_rules_reference_declared_adjudications():
    unknown_always = V.RATIONALE_ALWAYS - V.ADJUDICATIONS
    unknown_cond = set(V.RATIONALE_CONDITIONAL) - V.ADJUDICATIONS
    assert not unknown_always, f"RATIONALE_ALWAYS names non-adjudications: {sorted(unknown_always)}"
    assert not unknown_cond, f"RATIONALE_CONDITIONAL names non-adjudications: {sorted(unknown_cond)}"
    both = V.RATIONALE_ALWAYS & set(V.RATIONALE_CONDITIONAL)
    assert not both, (
        f"adjudication(s) {sorted(both)} are both always-required and conditionally-required; "
        f"the condition would be dead"
    )
