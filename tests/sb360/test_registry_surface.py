"""Both-directions pin of the SB360 registry to the public surface.

Lands RED: the registry is empty, so every export is unregistered. A gate written after its
own repair arrives green and is never observed to work (ADR-051).
"""

from __future__ import annotations

import pytest

from tests.sb360 import _vocabulary as V
from tests.sb360._registry import (
    NOT_EXERCISED_BUDGET,
    SB360_ENTRIES,
    VISIBILITY_ROSTERS,
    audited_surface,
    iter_verdicts,
    public_add_star,
)


def test_every_public_add_star_is_registered():
    """Landed RED as a strict xfail and observed failing at 33/33 unregistered (ADR-051). The
    marker was deleted when the audit populated the registry -- which is exactly what a strict
    xfail forces, since it would otherwise XPASS and fail."""
    missing = public_add_star() - set(SB360_ENTRIES)
    assert not missing, (
        f"{len(missing)} public add_* export(s) carry no SB360 verdict: {sorted(missing)}. "
        f"Register them or CI stays red -- this is the anti-rot property the 2026-05-27 "
        f"compatibility table lacked."
    )


#: Boundary entry points the action-paired fixture structurally cannot exercise, each with its
#: reason. NOT a convenience list: every name here is one the audit does not cover, and saying
#: so explicitly is the difference between a SCOPED gap and a silent one.
UNAUDITABLE_BOUNDARY: dict[str, str] = {
    "gkdv.build_ghost_frames": (
        "Takes FRAMES only and returns (frames, provenance, report) -- not action-coupled, so "
        "the per-action paired comparison has no shape to compare. Also needs a fitted "
        "GhostGkModel."
    ),
    "gkdv.delta_das": (
        "Operates on a FACTUAL/GHOST frame PAIR rather than on actions, and consumes the output of build_ghost_frames."
    ),
    "gkdv.delta_threat_suppression": ("Same frame-pair shape as delta_das, plus a fitted ExpectedThreat."),
    "xtgk.compute_xt_gk_v2": (
        "Requires three injected ports (possession_value, retention, turnover_cost). "
        "MarkovPossessionValue needs an xG-calibrated fit and silly-kicks ships NO xG model, so "
        "any port the audit supplied would be auditing the stub rather than the library."
    ),
}


@pytest.mark.xfail(
    reason="Four boundary entry points are structurally out of reach of an action-paired "
    "fixture -- see UNAUDITABLE_BOUNDARY for the per-name reason. Kept STRICT so that covering "
    "any of them forces this marker to be revisited rather than quietly passing.",
    strict=True,
)
def test_every_boundary_entry_point_is_registered():
    from tests.sb360._registry import BOUNDARY_ENTRY_POINTS

    missing = set(BOUNDARY_ENTRY_POINTS) - set(SB360_ENTRIES)
    assert not missing, f"frame-consuming boundary entry point(s) carry no SB360 verdict: {sorted(missing)}"


def test_uncovered_boundary_points_each_carry_a_reason():
    """The gap is SCOPED, not silent: every unregistered boundary name is enumerated with why,
    and an excuse that outlives its need fails too."""
    from tests.sb360._registry import BOUNDARY_ENTRY_POINTS

    missing = set(BOUNDARY_ENTRY_POINTS) - set(SB360_ENTRIES)
    undocumented = missing - set(UNAUDITABLE_BOUNDARY)
    assert not undocumented, f"boundary entry point(s) {sorted(undocumented)} are unregistered AND unexplained"
    stale = set(UNAUDITABLE_BOUNDARY) & set(SB360_ENTRIES)
    assert not stale, f"{sorted(stale)} are documented as unauditable but ARE registered -- delete the excuse"


def test_no_registry_entry_names_a_missing_export():
    """Reverse direction. Meaningful from the first entry onward, so it is NOT xfailed: a
    registry naming something the library does not export is always wrong."""
    extra = set(SB360_ENTRIES) - audited_surface()
    assert not extra, f"registry names non-exported function(s): {sorted(extra)}"


def test_every_visibility_roster_has_its_own_slot():
    for name, entry in SB360_ENTRIES.items():
        assert set(entry.visibility) == set(VISIBILITY_ROSTERS), (
            f"{name}: visibility keys {sorted(entry.visibility)} != "
            f"{sorted(VISIBILITY_ROSTERS)}. Each roster needs its own verdict -- a shared "
            f"slot cannot represent a feature that survives a missing outfielder and "
            f"collapses on a missing keeper."
        )


def test_every_verdict_is_admissible_from_its_observation():
    for name, entry in SB360_ENTRIES.items():
        for axis, roster, col, verdict in iter_verdicts(entry):
            admissible = V.ADMISSIBLE_FROM[verdict.adjudication]
            assert verdict.observation in admissible, (
                f"{name}.{col} ({axis}/{roster}): adjudication {verdict.adjudication!r} is "
                f"not admissible from observation {verdict.observation!r}"
            )


def test_rationales_are_present_where_required():
    for name, entry in SB360_ENTRIES.items():
        for axis, roster, col, v in iter_verdicts(entry):
            needs = v.adjudication in V.RATIONALE_ALWAYS
            if v.adjudication == "honest_nan" and v.observation == "partial_nan":
                needs = True
            if v.adjudication == "works" and col in entry.tolerances:
                needs = True
            if needs:
                assert v.rationale, (
                    f"{name}.{col} ({axis}/{roster}): adjudication {v.adjudication!r} "
                    f"requires a written rationale and has none"
                )


def test_tolerance_overrides_carry_a_basis():
    for name, entry in SB360_ENTRIES.items():
        for col in entry.tolerances:
            assert entry.tolerance_basis.get(col), (
                f"{name}.{col}: tolerance override with no basis. Loosening a tolerance "
                f"converts `differs` into `identical` and manufactures a `works` verdict."
            )


def test_structural_impossibility_co_occurs_with_all_nan_or_raises():
    """Checked on EVERY axis. A structurally impossible feature cannot become possible
    because a defender left the frame."""
    for name, entry in SB360_ENTRIES.items():
        for axis, roster, col, v in iter_verdicts(entry):
            if col not in entry.structurally_impossible:
                continue
            assert v.observation in {"all_nan", "raises_a"}, (
                f"{name}.{col} ({axis}/{roster}) is annotated structurally_impossible but "
                f"observes {v.observation!r}. The annotation is falsifiable by construction "
                f"-- this is the contradiction."
            )


def test_applicability_classes_are_declared():
    for name, entry in SB360_ENTRIES.items():
        for col, cls in entry.applicability.items():
            assert cls in V.APPLICABILITY, f"{name}.{col}: undeclared applicability class {cls!r}"


def test_not_exercised_count_is_within_its_locked_budget():
    actual = sum(
        1
        for e in SB360_ENTRIES.values()
        for _axis, _roster, _col, v in iter_verdicts(e)
        if v.adjudication == "not_exercised"
    )
    assert actual == NOT_EXERCISED_BUDGET, (
        f"{actual} not_exercised verdict(s) against a locked budget of "
        f"{NOT_EXERCISED_BUDGET}. A fixture inadequacy must be acknowledged deliberately, "
        f"never allowed to grow quietly (ADR-052: a bounded pass logs what it dropped)."
    )
