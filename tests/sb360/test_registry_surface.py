"""Both-directions pin of the SB360 registry to the public surface.

Lands RED: the registry is empty, so every export is unregistered. A gate written after its
own repair arrives green and is never observed to work (ADR-051).
"""

from __future__ import annotations

from tests.sb360 import _vocabulary as V
from tests.sb360._registry import (
    NOT_EXERCISED_BUDGET,
    SB360_ENTRIES,
    VISIBILITY_ROSTERS,
    audited_surface,
    columns_exercised_on_no_roster,
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
#:
#: NOW EMPTY: every BOUNDARY_ENTRY_POINT is registered (the last two -- `gkdv.delta_das` and
#: `gkdv.delta_threat_suppression` -- landed via the inline frame-pair adapters in
#: `_entries/_boundary.py`, projecting each per-frame arm to its action ANCHOR frame). The symbol
#: is retained so `test_uncovered_boundary_points_each_carry_a_reason` still guards a FUTURE
#: unregistered boundary entry.
UNAUDITABLE_BOUNDARY: dict[str, str] = {}


def test_every_boundary_entry_point_is_registered():
    """Every frame-consuming boundary entry point outside ``tracking.__all__`` now carries an
    SB360 verdict -- the strict xfail retired once the last two (`gkdv.delta_das`,
    `gkdv.delta_threat_suppression`) landed and `UNAUDITABLE_BOUNDARY` emptied. A plain
    completeness assertion from here on: a NEW boundary entry must register or CI fails."""
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


def test_every_aggregator_emits_at_least_one_column() -> None:
    """An entry with NO columns has no verdicts, so the audit silently does not cover it.

    This is the gate for a defect class that shipped TWICE. `_regenerate.py` probes each aggregator
    to discover its columns and, on any exception, falls back to `cols = ()`. The committed verdicts
    stay correct until someone regenerates, so CI never noticed:

    * `add_visible_area_coverage` (4.77.0) -- its adapter was written in `_calls.py` but never
      registered in `ADAPTERS`, so the `C.generic` fallback raised `TypeError`;
    * `add_xcross_attempt` -- read `frames["vx"]` unguarded and raised `KeyError` on a freeze-frame
      that DECLARES `speed_source="unavailable"`.

    Both are repaired, so this gate cannot be landed red against a live instance. It was instead
    MUTATION-VERIFIED, which ADR-051 permits explicitly ("landed-red where practical,
    mutation-verified otherwise"): re-running the regenerator with `visible_area_coverage` deleted
    from `ADAPTERS` reproduces the empty-column state, and this assertion fails naming it.

    ADR-053's contract is that EVERY registered `add_*` carries an SB360 freeze-frame verdict. An
    empty entry satisfies the meta-assertion that pins the registry to `tracking.__all__` -- the
    entry EXISTS -- while carrying no information at all. That gap is what this closes.
    """
    empty = sorted(name for name, entry in SB360_ENTRIES.items() if not entry.columns)
    assert not empty, (
        f"{empty} carry ZERO columns, so every roster block for them is empty and the audit covers "
        f"them in name only. This is almost always a PROBE FAILURE in `_regenerate.py` -- run it "
        f"and read the '!! PROBE FAILED' report, which names the exception. The usual causes are an "
        f"adapter defined in `_calls.py` but never registered in `ADAPTERS`, or an aggregator that "
        f"raises on the freeze-frame fixture instead of honouring the ADR-054 velocity contract."
    )


#: ``(entry, column)`` pairs adjudicated ``not_exercised`` under EVERY visibility roster --
#: exercised NOWHERE in the sweep. A STANDING PIN, not a cycle deliverable: adding a roster cannot
#: shrink it for a column already exercised on a sibling roster, because "unexercised everywhere"
#: is a strictly stronger predicate than "unexercised on the roster in question". The `gk_one_end`
#: cycle deliberately left this set UNCHANGED while reclaiming five columns under its own roster.
#:
#: It gained ONE member when `add_xcross_attempt` was repaired -- see the entry below. A member
#: arriving because an aggregator became VISIBLE is not the regression this pin hunts; the
#: regression is a column that was exercised somewhere and stopped being. The failure message
#: distinguishes them, so read WHICH direction moved before rebaselining.
_EXPECTED_DARK_COLUMNS = {
    # The fixture has no pressing sequence, so the argmax-defender identity never has a domain.
    ("add_cover_shadows", "max_single_defender_player_id"),
    # `no_signal` on all three rosters; TF-51 needs a press the fixture does not stage.
    ("add_press_commitment", "press_commitment"),
    ("add_press_commitment", "press_commitment_closing_speed"),
    # A fitted model over a freeze-frame domain the fixture does not produce.
    ("add_xshot_occurrence", "xshot_occurrence"),
    # NEWLY VISIBLE, not newly dark. `add_xcross_attempt` used to raise `KeyError: 'vx'` on the
    # freeze-frame probe, `_regenerate.py` swallowed that into `cols = ()`, and the aggregator
    # therefore had NO columns and NO verdicts -- so it could not appear here however dark it was.
    # With the ADR-054 velocity contract honoured it probes cleanly, and both legs score NaN
    # because the fixture stages no in-possession wide-area cross. Same class as the
    # `add_xshot_occurrence` entry above: a fitted model over a domain this fixture does not
    # produce.
    ("add_xcross_attempt", "xcross_attempt"),
}


def test_no_column_is_unexercised_on_every_roster_except_the_recorded_ones() -> None:
    """A column ``not_exercised`` under EVERY visibility roster is exercised NOWHERE.

    Distinct from ``NOT_EXERCISED_BUDGET``, which counts per-ROSTER tuples and therefore RISES
    whenever a roster is added. This set is the per-COLUMN question, and it is the only one that
    can answer "is this column covered anywhere at all".
    """
    dark = columns_exercised_on_no_roster()
    assert dark == _EXPECTED_DARK_COLUMNS, (
        f"columns exercised on NO roster changed.\n"
        f"  newly dark: {sorted(dark - _EXPECTED_DARK_COLUMNS)}\n"
        f"  newly lit : {sorted(_EXPECTED_DARK_COLUMNS - dark)}\n"
        f"A column going dark is a coverage REGRESSION -- find which roster stopped exercising it. "
        f"A column lighting up is a gain: update the expectation and record which roster covered it."
    )


#: The five columns ADR-055 sent dark on ``gk_absent`` by making ``add_cover_shadows``
#: keeper-dependent. Reclaiming them is the entire point of the ``gk_one_end`` roster, so the claim
#: is ASSERTED rather than noted. ``max_single_defender_player_id`` is deliberately absent: it is
#: `not_exercised` for an UNRELATED reason (no pressing sequence) on every roster.
_RECLAIMED_BY_GK_ONE_END = (
    "n_blocked_receivers",
    "n_potential_receivers",
    "blocking_score",
    "blocked_threat_fraction",
    "max_single_defender_blocking_score",
)


def test_gk_one_end_reclaims_the_cover_shadow_columns() -> None:
    """The ``gk_one_end`` roster exists to make these five exercisable again.

    Under ``gk_absent`` BOTH keepers are gone, ``resolve_defended_goals`` guesses both teams at
    x=105, ``attacked_goal`` refuses the degenerate map, and every leg goes NaN for a roster-driven
    reason. With ONE keeper visible the ends differ (measured: team 1 resolves to 0.0, team 2
    guesses 105.0), so the columns carry real observations.

    This is the cycle's success criterion. Neither aggregate can express it -- the budget counts
    per-roster tuples and can only rise, and the no-roster set never contained these columns
    because they are `honest_nan` under `defender_absent`.
    """
    entry = SB360_ENTRIES["add_cover_shadows"]
    verdicts = entry.visibility.get("gk_one_end", {})
    assert verdicts, "add_cover_shadows has no gk_one_end block -- the roster was not regenerated"

    unexercised = sorted(
        col
        for col in _RECLAIMED_BY_GK_ONE_END
        if verdicts.get(col) is None or verdicts[col].adjudication == "not_exercised"
    )
    assert not unexercised, (
        f"gk_one_end did not reclaim {unexercised}. The roster's whole purpose is a NON-degenerate "
        f"goal map -- one keeper visible so the two ends differ. If these are still unexercised, "
        f"check that `_player_layout` drops only the AWAY keeper and that `resolve_defended_goals` "
        f"returns two DIFFERENT ends on this roster."
    )


def test_verdict_provenance_vocabulary_and_restart_declaration():
    from tests.sb360._vocabulary import VERDICT_PROVENANCE

    assert VERDICT_PROVENANCE == frozenset({"substantive", "structural"})
    entry = SB360_ENTRIES["spadl.add_restart_coordinates"]
    assert entry.verdict_provenance == "structural"
    assert entry.provenance_rationale, "a structural boundary entry needs a stated reason"


def test_boundary_entries_declare_admissible_provenance():
    """Every REGISTERED boundary entry declares substantive/structural, admissibly from its
    observation, so an empty UNAUDITABLE_BOUNDARY cannot be misread as end-to-end degradation
    coverage (ADR-053 Part 4). Population derived from BOUNDARY_ENTRY_POINTS -- a new boundary
    entry without a provenance fails here.

    KNOWN LIMIT (spec Part 4): this gate locks HALF the distinction. `works`=>`structural` is tight
    (a value that cannot move was not substantively handled). `differs_by_design`/`silent_degrade`
    =>`substantive` is enforceable but inert this cycle. But `honest_nan` is OBSERVATIONALLY
    AMBIGUOUS -- self-refusal (substantive) and inherited-refusal (structural, gkdv) both produce
    `all_nan`, so the gate CANNOT check gkdv's `structural` choice; it is author-asserted, forced
    only to carry a rationale. This is the machine-checkability ceiling, named deliberately.

    Cannot be landed red against the correct Task-1 state (add_restart_coordinates is already
    `structural`+`works`), so it was MUTATION-VERIFIED (ADR-051), both admissibility branches: see
    Step 2.
    """
    from tests.sb360._registry import BOUNDARY_ENTRY_POINTS
    from tests.sb360._vocabulary import VERDICT_PROVENANCE

    for name in sorted(set(BOUNDARY_ENTRY_POINTS) & set(SB360_ENTRIES)):
        entry = SB360_ENTRIES[name]
        prov = entry.verdict_provenance
        assert prov in VERDICT_PROVENANCE, (
            f"{name}: registered boundary entry carries verdict_provenance {prov!r}, not in "
            f"{sorted(VERDICT_PROVENANCE)}. Declare substantive/structural (spec Part 4)."
        )
        for _axis, _roster, col, v in iter_verdicts(entry):
            if v.adjudication == "works":
                assert prov == "structural", (
                    f"{name}.{col}: `works` (from `identical`) forces `structural` -- a value that "
                    f"cannot move across the velocity legs was not substantively handled. Got {prov!r}."
                )
            if v.adjudication in {"differs_by_design", "silent_degrade"}:
                assert prov == "substantive", (
                    f"{name}.{col}: {v.adjudication!r} forces `substantive` -- the value moved "
                    f"because of the function. Got {prov!r}."
                )
        if prov == "structural":
            assert entry.provenance_rationale, (
                f"{name}: `structural` requires a non-empty provenance_rationale naming WHY "
                f"(frame-blind / inherited-from-refusal)."
            )
