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
VISIBILITY_ROSTERS: tuple[str, ...] = ("gk_absent", "defender_absent")

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
    """Aggregators whose signature does not fit the generic adapter.

    Hand-written in ``_calls`` rather than guessed: an adapter that supplies a default for a
    required argument turns a wrong call into a recorded verdict about the library.
    """
    from tests.sb360 import _calls as C

    return {
        # Require a fitted ExpectedThreat.
        "add_cover_shadows": C.with_xt,
        "add_gk_influence": C.with_xt,
        "add_off_ball_run_values": C.with_xt,
        "add_player_influence": C.with_xt,
        "add_xt_gk": C.with_xt,
        # Requires an xg_column too -- silly-kicks ships no xG model.
        "add_defensive_credit": C.defensive_credit,
        # xt is KEYWORD-only with a None default; left unset they take the SYNTHETIC EPV path
        # and emit SyntheticEPVWarning, which CI escalates -- recording `raises_a` for a
        # function that works fine when handed the xT a real consumer supplies.
        "add_obso": C.with_xt_keyword,
        "add_pausa": C.with_xt_keyword,
        "add_space_creation": C.with_xt_keyword,
        # frames is keyword-only here, positional in its sibling; both need the GK prerequisite.
        "add_pre_shot_gk_angle": C.pre_shot_gk_angle,
        "add_pre_shot_gk_position": C.pre_shot_gk_position,
        # Takes `links` as its second POSITIONAL argument and no frames at all.
        "add_sync_score": C.sync_score,
        # A jersey/roster helper over different inputs, returning a tuple of frames.
        "add_gradientsports_player_ids": C.gradientsports_player_ids,
    }


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
#: It is a real widening of the audit's blind spot and is recorded as such: before the re-key
#: these five columns produced numbers on a keeper-less freeze-frame, because direction came
#: from ``home_team_id`` rather than from the frames. Those numbers were not evidence. The
#: honest consequence is that ``add_cover_shadows`` is keeper-dependent on SB360 input, and the
#: fixture cannot exercise it without a keeper -- widening the fixture (a keeper at ONE end
#: would suffice to break the degeneracy) is what would reclaim these five.
NOT_EXERCISED_BUDGET = 31


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


def public_add_star() -> set[str]:
    """Every ``add_*`` exported from ``tracking.__all__``."""
    return {n for n in _T.__all__ if n.startswith("add_")}


def audited_surface() -> set[str]:
    """Every name this audit is allowed to register."""
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
