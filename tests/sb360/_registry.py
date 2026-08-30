"""The SB360 verdict registry.

Each entry carries observation/adjudication pairs on TWO independent axes -- velocity and
visibility -- because a feature can be sound on one and fabricated on the other. The
visibility axis is keyed BY ROSTER: a feature can survive a missing outfielder and collapse on
a missing keeper, which is the entire reason both are swept.

The CI gate locks the OBSERVATION, never the adjudication. Repair a function and the
observation changes, CI fails, and the human judgement is forced to be revisited. Locking the
adjudication instead would pretend a machine can adjudicate; locking neither is the rot the
2026-05-27 compatibility table demonstrated.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import silly_kicks.tracking as _T

#: Roster variants the visibility axis sweeps. Each gets its OWN verdict slot.
VISIBILITY_ROSTERS: tuple[str, ...] = ("gk_absent", "defender_absent", "gk_one_end")

#: Linkage-provenance columns, merged in by 11 of the 33 aggregators (the idempotent-merge
#: contract in CLAUDE.md). They are NOT feature columns and carry no verdict.
#:
#: Auditing them would be worse than noise, it would be meaningless: Leg A's ``frame_id`` is
#: its ``action_id`` by construction while Leg B numbers a 10 Hz stream, so ``frame_id`` and
#: ``n_candidate_frames`` differ between two legs that agree about everything that matters.
#: 41 of the first probe's 44 `differs` readings were these columns.
PROVENANCE_COLUMNS: frozenset[str] = frozenset(
    {"frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"}
)


def feature_columns(emitted) -> tuple[str, ...]:
    """Emitted columns minus linkage provenance, order preserved."""
    return tuple(c for c in emitted if c not in PROVENANCE_COLUMNS)


def _adapters() -> dict:
    """The single-sourced per-aggregator call adapters (moved to ``scripts/_sb_battery.py``).

    Returns a COPY of ``scripts._sb_battery.ADAPTER_MAP`` so the audit cannot mutate the shared
    map. The map and the adapter bodies live in ``scripts/`` (not here) so the licensed-corpus
    validation driver resolves the EXACT same call convention -- the layer that silently
    empty-blocked once (``add_visible_area_coverage`` unregistered -> ``generic`` ``TypeError``
    swallowed to ``cols=()``) must not be forked. Layering is ``tests -> scripts`` (round-4 review).
    """
    from scripts._sb_battery import ADAPTER_MAP

    return dict(ADAPTER_MAP)


#: Populated lazily on first access to avoid an import cycle (``_calls`` imports this module).
ADAPTERS: dict = {}


def _init_adapters() -> None:
    if not ADAPTERS:
        ADAPTERS.update(_adapters())


#: Frame-consuming public entry points that do not live in ``tracking.__all__``.
#: ``gkdv`` and ``xtgk`` v2 expose no ``add_*``; ``spadl.add_restart_coordinates`` accepts
#: ``frames=`` despite living outside ``tracking``.
BOUNDARY_ENTRY_POINTS: frozenset[str] = frozenset(
    {
        "gkdv.build_ghost_frames",
        "gkdv.delta_das",
        "gkdv.delta_threat_suppression",
        "xtgk.compute_xt_gk_v2",
        "spadl.add_restart_coordinates",
        "restdefense.compute_rest_defense",
    }
)


@dataclass(frozen=True)
class AxisVerdict:
    observation: str
    adjudication: str
    rationale: str | None = None
    counts: dict[str, int] | None = None
    #: For ``raises_a`` only: the exception that was raised, so an adjudicator can tell a
    #: library REFUSING freeze-frame input from a harness mis-call. Without it a
    #: ``TypeError: unexpected keyword argument`` -- a defect in how the audit calls the
    #: function -- is indistinguishable from a genuine library property, and would be
    #: recorded as one.
    detail: str | None = None


@dataclass(frozen=True)
class Sb360Entry:
    name: str
    call: Callable  # (actions, frames, links, home_team_id) -> pd.DataFrame
    columns: tuple[str, ...]
    #: column -> verdict
    velocity: dict[str, AxisVerdict] = field(default_factory=dict)
    #: roster -> column -> verdict
    visibility: dict[str, dict[str, AxisVerdict]] = field(default_factory=dict)
    applicability: dict[str, str] = field(default_factory=dict)
    #: col -> {"extreme": delta, "near": delta}. Recorded so a zero-movement classification is
    #: VISIBLE: a `no_support` derived from two zero deltas is indistinguishable from a probe
    #: that silently failed to perturb anything.
    applicability_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    #: col -> (rtol, atol). Absent means the default; presence means an override.
    tolerances: dict[str, tuple[float, float]] = field(default_factory=dict)
    tolerance_basis: dict[str, str] = field(default_factory=dict)
    structurally_impossible: dict[str, str] = field(default_factory=dict)
    #: Boundary-entry provenance (substantive/structural). None on the add_* surface. See Part 4.
    verdict_provenance: str | None = None
    provenance_rationale: str | None = None


SB360_ENTRIES: dict[str, Sb360Entry] = {}

#: Pre-registered count of (entry, axis, roster, column) verdicts adjudicated
#: ``not_exercised``. Raised only with a recorded reason; it is a budget, not a tally.
#:
#: 26 -> 31 (ADR-055). The five are ``add_cover_shadows``' emitted columns on the
#: ``visibility/gk_absent`` roster, which moved ``all_nan`` -> ``no_signal``; ``no_signal``
#: admits no adjudication but ``not_exercised``, so the rise is forced by the vocabulary rather
#: than chosen.
#:
#: The reason this budget wants, stated as the mechanism and not as "the fixture is thin":
#: ``gk_absent`` removes BOTH keepers, so ``resolve_defended_goals`` falls to its outfield rung
#: and guesses both teams at x=105 (measured outfield mean x 56.9 and 76.5, both past the 52.5
#: midline). A both-teams-same-end map is DEGENERATE, ``attacked_goal`` refuses it by its
#: documented same-end guard, and the aggregator emits NaN. Both legs go NaN for the same
#: roster-driven reason, so no informative row survives.
#:
#: It was a real widening of the audit's blind spot and is recorded as such: before the re-key
#: these five columns produced numbers on a keeper-less freeze-frame, because direction came
#: from ``home_team_id`` rather than from the frames. Those numbers were not evidence. The
#: honest consequence is that ``add_cover_shadows`` is keeper-dependent on SB360 input, and
#: ``gk_absent`` cannot exercise it. **This has since been RESOLVED by ADDING a roster rather
#: than widening this one**: ``gk_one_end`` keeps ONE keeper visible, which is exactly the
#: "keeper at ONE end" this note predicted would break the degeneracy, and all five columns are
#: reclaimed there. ``gk_absent`` is deliberately left alone -- it is a real visibility axis and
#: the only case exercising the both-absent refusal path, so widening it would have traded one
#: coverage loss for another.
#:
#: RAISED 31 -> 41 by the `gk_one_end` roster (ADR-055 follow-up). The budget counts
#: (entry, axis, roster, column) tuples, so a THIRD roster can only ADD to it -- a rise is
#: structural here, not a regression, and the reclaim this cycle delivers is asserted separately
#: by `test_gk_one_end_reclaims_the_cover_shadow_columns`. All 10 new tuples are under
#: `gk_one_end` and enumerate to three causes:
#:
#:   6  add_pre_shot_gk_{position,angle}.* geometry -- the roster drops ONE keeper, so the
#:      shot-facing GK geometry has no informative rows for the actions whose defending keeper
#:      went off-frame. Column-specific, not blanket: `defending_gk_player_id` still observes
#:      `identical`, so the aggregator runs and only the geometry loses its comparison.
#:   3  add_press_commitment.{press_commitment,press_commitment_closing_speed} and
#:      add_xshot_occurrence.xshot_occurrence -- `no_signal` on ALL THREE rosters. Pre-existing
#:      and unrelated to this roster; they are the same members `columns_exercised_on_no_roster`
#:      already reports.
#:   1  add_cover_shadows.max_single_defender_player_id -- pre-existing (the fixture has no
#:      pressing sequence), and deliberately NOT among the five columns this roster reclaims.
#:
#: RAISED 41 -> 45 by REPAIRING `add_xcross_attempt`, and this rise is the opposite of a
#: regression: the 4 new tuples are coverage that was ALREADY missing and is now VISIBLE.
#:
#: That aggregator used to raise a bare `KeyError: 'vx'` on the freeze-frame probe -- it read
#: `frames["vx"]` unguarded on input that DECLARES `speed_source="unavailable"` -- and
#: `_regenerate.py`'s handler swallowed the crash into `cols = ()`. So it was the only entry in
#: this registry with NO columns and FOUR empty verdict blocks, i.e. ADR-053's "every add_* carries
#: an SB360 freeze-frame verdict" was quietly untrue of it. It now honours the ADR-054 velocity
#: contract (declared-unavailable -> NaN; undeclared-missing -> an informative raise), so it probes
#: cleanly and carries real verdicts.
#:
#:   4  add_xcross_attempt.xcross_attempt -- `no_signal` on ALL FOUR axes (velocity/full plus the
#:      three visibility rosters). BOTH legs are NaN, and not because of the repair: the
#:      velocity-bearing leg scores NaN too, because the fixture never produces an in-possession
#:      wide-area cross context for the model to score. A fixture inadequacy, now recorded instead
#:      of hidden behind an empty block. It is consequently a NEW member of
#:      `columns_exercised_on_no_roster` -- the only one this cycle adds.
#:
#: RAISED 45 -> 49 by ADR-051 D3 (4.80.0), and this rise is a REAL loss of comparison, honestly
#: recorded rather than engineered away.
#:
#:   4  add_packing.{packing_made,packing_net,packing_goal_threat,packing_secured} under
#:      `gk_absent` ONLY. The re-key took packing's direction from team IDENTITY (which always
#:      answers) to the `GoalMap` (which can decline), and `gk_absent` is the one roster with no
#:      keeper at EITHER end, so no team's defended goal resolves and ADR-055's edge policy emits
#:      NaN. Previously these read `identical` -- a number produced by guessing a side, which is
#:      precisely the defect the re-key removes, so the old reading was worth LESS than this one.
#:      Narrowly scoped, and that is checked rather than assumed: `defender_absent` and
#:      `gk_one_end` still observe `identical` on all five columns, because one keeper anywhere
#:      is enough to resolve the map. `packing_receiver_player_id` also stays `identical` under
#:      `gk_absent` -- it is event-derived and never consults the map, which is the internal
#:      consistency check that this is a GEOMETRY refusal and not the aggregator falling over.
#:
#:      Surfaced by the audit only after a genuine bug was fixed: the refusal used to escape as
#:      `KeyError: 'line_x'` (observation `raises_a`), because `add_packing`'s
#:      `GoalEndUnresolvedError` fallback built the three EMITTED columns and not the internal
#:      one the event-only assembly reads immediately afterwards.
#:
#: RAISED 49 -> 52 by registering `gkdv.build_ghost_frames` as a boundary entry (spec Part 2).
#: The 3 new tuples are its emitted columns (`ghost_x`, `ghost_y`, `displacement_m`) under
#: `gk_absent` ONLY. `gk_absent` removes BOTH keepers, and gkdv's spec-§4.1 domain requires a
#: defending-GK row present, so no frame is eligible on EITHER leg -- both legs score zero and
#: every column observes `no_signal` (both-NaN), which admits only `not_exercised`. On the other
#: two rosters ONE keeper remains, so the defending keeper resolves and these columns observe
#: `all_nan` instead (Leg A refuses the velocity-less freeze-frame -- ADR-054 -- while Leg B
#: scores the in-domain actions), which is `honest_nan`, NOT `not_exercised`, and does not count
#: here.
#:
#: RAISED 52 -> 54 by registering `gkdv.delta_das` + `gkdv.delta_threat_suppression` as the last
#: two boundary entries (spec Part 2/Task 5). Same shape as `build_ghost_frames`: one emitted
#: column each (`delta_das`, `delta_threat_suppression`), `no_signal` -> `not_exercised` under
#: `gk_absent` ONLY (both keepers gone -> no eligible frame on either leg), and `honest_nan` on
#: the other two rosters. That was the final boundary bump -- every BOUNDARY_ENTRY_POINT is
#: registered and `UNAUDITABLE_BOUNDARY` is empty.
#:
#: LOWERED 54 -> 42 by sb360-fixture-2 (the ADR-067 position-only cycle): the realistic striker
#: ahead of the ball made `add_xshot_occurrence` and `add_xcross_attempt` probe cleanly and produce
#: real observations (differs / honest-NaN) across all four axes -- eight tuples -- and made
#: `add_press_commitment`'s two columns run to honest-NaN on defender_absent + gk_one_end -- four
#: more, twelve total, all previously `no_signal` -> `not_exercised`. A budget can only FALL when the
#: fixture exercises MORE, a coverage gain rather than the fixture-inadequacy this bound hunts.
#:
#: RAISED 42 -> 44 by registering `restdefense.compute_rest_defense` as a boundary entry (TF-60,
#: ADR-080). The 2 new tuples are its GK-position columns (`rd_gk_line_height`,
#: `rd_gk_to_line_distance`) under `gk_absent` ONLY: that roster removes BOTH keepers, so the GK
#: metrics have no keeper to read in EITHER leg -> both all-NaN -> `no_signal`, which admits only
#: `not_exercised`. The other nine Layer-1 columns stay `identical`/`works` (positional, both legs
#: agree), and the two GK columns are exercised (`identical`) on `defender_absent` + `gk_one_end`.
NOT_EXERCISED_BUDGET = 44


def _entry(
    name,
    call,
    columns,
    *,
    velocity=None,
    visibility=None,
    applicability=None,
    applicability_deltas=None,
    tolerances=None,
    tolerance_basis=None,
    structurally_impossible=None,
    verdict_provenance=None,
    provenance_rationale=None,
) -> None:
    SB360_ENTRIES[name] = Sb360Entry(
        name=name,
        call=call,
        columns=tuple(columns),
        velocity=velocity or {},
        visibility=visibility or {},
        applicability=applicability or {},
        applicability_deltas=applicability_deltas or {},
        tolerances=tolerances or {},
        tolerance_basis=tolerance_basis or {},
        structurally_impossible=structurally_impossible or {},
        verdict_provenance=verdict_provenance,
        provenance_rationale=provenance_rationale,
    )


def iter_verdicts(entry: Sb360Entry) -> Iterator[tuple[str, str, str, AxisVerdict]]:
    """Yield ``(axis, roster, column, verdict)`` for every recorded verdict.

    THE single iteration seam. Every gate walks the registry through this, so a gate cannot
    silently disagree with the schema about how verdicts are keyed -- which is exactly how a
    one-visibility-dict-for-two-rosters defect arose during plan review.
    """
    for col, v in entry.velocity.items():
        yield ("velocity", "full", col, v)
    for roster, cols in entry.visibility.items():
        for col, v in cols.items():
            yield ("visibility", roster, col, v)


def columns_exercised_on_no_roster() -> set[tuple[str, str]]:
    """``(entry, column)`` pairs adjudicated ``not_exercised`` under EVERY visibility roster.

    A column here is exercised NOWHERE in the visibility sweep, whatever the per-roster budget
    says. Columns absent from a roster's dict count as unexercised for that roster -- an absent
    verdict is not evidence of coverage.

    SCOPE, because two numbers over one registry WILL be compared by someone: this walks the
    VISIBILITY rosters only and ignores the velocity axis, while ``NOT_EXERCISED_BUDGET`` counts
    every ``(entry, axis, roster, column)`` tuple INCLUDING velocity. They are not comparable and
    neither is a subset of the other.

    This is a standing regression pin, NOT a cycle deliverable. Adding a roster cannot shrink it
    for a column already exercised on a sibling roster -- "unexercised everywhere" is a strictly
    stronger predicate than "unexercised on the roster in question". The ``gk_one_end`` cycle
    (ADR-055 follow-up) deliberately registered ZERO change here while reclaiming five columns
    under its own roster; the reclaim is asserted separately.
    """
    dark: set[tuple[str, str]] = set()
    for name, entry in SB360_ENTRIES.items():
        for col in entry.columns:
            verdicts = [entry.visibility.get(r, {}).get(col) for r in VISIBILITY_ROSTERS]
            if all(v is None or v.adjudication == "not_exercised" for v in verdicts):
                dark.add((name, col))
    return dark


def public_add_star() -> set[str]:
    """Every ``add_*`` exported from ``tracking.__all__``."""
    return {n for n in _T.__all__ if n.startswith("add_")}


def audited_surface() -> set[str]:
    """Every name this audit is allowed to register.

    SCOPE NOTE -- the SB360 visibility-companion columns are DELIBERATELY outside this surface.
    ``add_action_context(..., visible_area=...)`` emits six opt-in companion columns
    (``<feature>_observed_fraction`` / ``_observed_source`` for the three region-based counts). This
    audit runs every ``add_*`` through :mod:`tests.sb360._calls`'s ``generic`` adapter, which forwards
    ``links``/``home_team_id`` only -- never ``visible_area`` -- so the companions never appear in a
    verdict block, and their absence is CORRECT, not an omission.

    The ADR-077 FOV-observability cycle EXTENDS the exact same opt-in companion pattern to seven more
    aggregators (``add_pressure_on_actor`` / ``add_packing`` / ``add_defensive_line`` /
    ``add_team_shape`` / ``add_player_influence`` / ``add_xt_gk`` / ``add_defensive_credit`` -- eight
    companioned total, joining ``add_action_context``'s ADR-062 companions), routed
    through the single ``tracking._fov_registry`` engine, plus the ``validate_fov`` / ``FovDiagnosis``
    frame-set diagnostic. Every one of those companions is gated on the SAME ``visible_area is not
    None`` opt-in the ``generic`` adapter never supplies, so they are outside this surface for the
    IDENTICAL reason -- a two-leg full-coverage fixture makes a visibility companion vacuous
    (``identical -> works`` on a polygon both legs share), the "coverage denominator masquerading as a
    signal" trap. ``validate_fov`` is a diagnostic, not an ``add_*``, and emits no feature column at
    all. The FOV companions ARE verified, from both sides, in ``tests/tracking/test_fov_*.py`` (the
    registry engine, the per-metric tight-ROI and aggregate-zone both-sides tests, the completeness
    gate) and on the real licensed corpus by ``tests/tracking/test_fov_companions_licensed_e2e.py`` +
    ``scripts/validate_sb360_licensed_corpus.py``. This scope decision is recorded in ADR-077.

    The tempting "fix" is a third, observed-region axis. It was tried and REJECTED. The ADR-053 audit
    is a TWO-LEG (Leg A vs Leg B) fabrication detector, and the companions depend on the polygon +
    action geometry, not on kinematics or roster -- so on the full-coverage fixture polygon both legs
    are byte-identical and every verdict would be ``identical -> works``. A gate that records ``works``
    without ever exercising partial visibility is exactly the "coverage denominator masquerading as a
    signal" / "a gate that certifies the failure it catches is worse than none" trap this codebase
    names elsewhere: negative value, not zero.

    The companions ARE verified, from both sides, where verification is meaningful:
    ``tests/tracking/test_visibility.py`` (``classify_region_observation`` across every source token),
    ``tests/tracking/test_add_action_context.py`` (companion tokens, NaN policy, additive byte-identity
    gate), and ``scripts/validate_sb360_licensed_corpus.py`` (all five degradation tokens observed on
    the real licensed corpus). This reinterprets the design spec's §9 (which proposed the axis); the
    reinterpretation is recorded in ADR-062 and routed back to review. See the plan's Task 6.

    SCOPE NOTE (SB360 first-class-provider cycle) -- ``run_tracking_features`` and
    ``resolve_keeper_identities`` are DELIBERATELY outside this surface. ``run_tracking_features`` is an
    ORCHESTRATOR (a ``run_*``, not an ``add_*``): it adds no feature column of its own, it runs the
    already-audited ``add_*`` family, and its correctness is proven by composition-equivalence in
    ``tests/tracking/test_run_tracking_features.py`` -- a per-family SB360 verdict on it would only
    re-audit the members this registry already covers. ``resolve_keeper_identities`` returns an identity
    MAPPING (a ``resolve_*``), not action-grain feature columns, so it has no per-column verdict to
    record. Neither is an ``add_*`` nor a ``BOUNDARY_ENTRY_POINTS`` member, so ``audited_surface`` picks
    up neither automatically; recording the exclusion here keeps it documented, not silent.
    """
    return public_add_star() | set(BOUNDARY_ENTRY_POINTS)


def _load_entry_modules() -> None:
    """Import every per-family entry module for its registration side effects."""
    import importlib
    import pkgutil

    from tests.sb360 import _entries

    for mod in pkgutil.iter_modules(_entries.__path__):
        importlib.import_module(f"tests.sb360._entries.{mod.name}")


_init_adapters()
_load_entry_modules()
