"""Public warning categories for tracking features (ADR-041).

These are the package's FIRST public ``Warning`` subclasses (before this, the only
warning-ish classes were two module-private ``IntegrityError`` exceptions), so this module
sets the convention: one module, every category re-exported via ``tracking.__all__``, so a
consumer's ``filterwarnings`` line has a single stable import path across releases.

Categories are deliberately SEPARATE rather than one umbrella: silencing the routine
synthetic-surface notice must not also silence a genuine misuse signal.
"""

from __future__ import annotations

__all__ = [
    "IgnoredSurfaceInputsWarning",
    "MissingFeatureContractWarning",
    "OrientationUnresolvedWarning",
    "RunValueCoverageWarning",
    "SyntheticEPVWarning",
    "UnverifiableFeatureContractWarning",
]


class SyntheticEPVWarning(UserWarning):
    """OBSO / space-creation / PAUSA is serving the synthetic placeholder EPV surface.

    The synthetic grids (a ``linspace(0.01, 0.3)`` x-ramp for EPV, a centred Gaussian for
    reachability) are unit-scale demo proxies, not production surfaces. Pass a fitted
    ``ExpectedThreat`` via ``xt=`` (or an explicit ``epv_grid=``) for real values.

    Examples
    --------
    Escalate to an error in a production pipeline::

        import warnings
        from silly_kicks.tracking import SyntheticEPVWarning

        warnings.filterwarnings("error", category=SyntheticEPVWarning)
    """


class IgnoredSurfaceInputsWarning(UserWarning):
    """Supplied ``xt=`` / ``epv_grid=`` / ``transition_grid=`` were ignored.

    Raised (as a warning) when a helper reuses already-present columns and therefore never
    consults the surface inputs it was handed. Distinct from
    :class:`SyntheticEPVWarning` on purpose: a caller who silences the routine synthetic
    notice must still hear about genuine misuse.

    Examples
    --------
    Keep misuse loud while muting the routine notice::

        import warnings
        from silly_kicks.tracking import IgnoredSurfaceInputsWarning, SyntheticEPVWarning

        warnings.filterwarnings("ignore", category=SyntheticEPVWarning)
        warnings.filterwarnings("error", category=IgnoredSurfaceInputsWarning)
    """


class MissingFeatureContractWarning(UserWarning):
    """A trained-model artifact carries NO feature contract, so it cannot be verified at all.

    Additive by design: pre-contract artifacts still load, because an artifact predating the
    contract is undeclared rather than known-bad. This is the category meant to be ESCALATED by a
    consumer that wants fail-closed semantics -- and it is escalated in this repo's own CI, where
    the opt-out list is the inventory of contract-less artifacts.

    Deliberately distinct from :class:`UnverifiableFeatureContractWarning`, which covers a contract
    that exists but could not be fully checked. Escalating this one must not turn a probe change
    into a hard failure.

    Examples
    --------
    Fail closed on any artifact that cannot be verified::

        import warnings
        from silly_kicks.tracking import MissingFeatureContractWarning

        warnings.filterwarnings("error", category=MissingFeatureContractWarning)
    """


class UnverifiableFeatureContractWarning(UserWarning):
    """A contract exists but part of it could not be checked on this load.

    Emitted when the probe itself changed (so the recorded fingerprint is not comparable), when a
    recorded constant is no longer declared by the library, or when a real mismatch was waved
    through by ``legacy_override``.

    SEPARATE from :class:`MissingFeatureContractWarning` on purpose. Adding a declared constant
    requires extending the probe, which changes the probe hash for every previously-saved
    artifact; those loads must keep working. If one umbrella category covered both, a consumer
    escalating the missing-contract case would silently turn every probe extension into a hard
    failure across every artifact not yet re-saved.

    Examples
    --------
    Fail closed on a missing contract while a probe change stays a notice::

        import warnings
        from silly_kicks.tracking import (
            MissingFeatureContractWarning,
            UnverifiableFeatureContractWarning,
        )

        warnings.filterwarnings("error", category=MissingFeatureContractWarning)
        warnings.filterwarnings("default", category=UnverifiableFeatureContractWarning)
    """


class OrientationUnresolvedWarning(UserWarning):
    """``acting_team_attacks_rtl`` could not resolve a direction and returned an all-False flip.

    An all-False flip means NO ADR-028 re-projection is applied, so every away-team action's
    geometry silently mixes coordinate conventions: an action-LTR anchor against frame-LTR
    positions. Measured on one canonical away action, labelled frames vs the same frames with the
    direction column dropped -- ``nearest_defender_distance`` 7.6158 -> 19.6977,
    ``receiver_zone_density`` 1 -> 0.

    Not hypothetical: the pining loader shipped SkillCorner frames with
    ``team_attacking_direction`` null on 100% of rows, so an entire provider's action-coupled
    geometry was computed unoriented with no signal of any kind.

    WARN rather than raise, deliberately. Consumers legitimately hold absolute/unlabelled frames
    (ADR-029) and a raise has no reachable remedy inside a converter; fail-closed belongs in CI.
    The one silent case is "there were no actions to flip" -- deliberately narrower than "either
    input was empty", because empty frames with live actions is a caller error.

    Examples
    --------
    Treat an unresolved orientation as fatal in a production pipeline::

        import warnings
        from silly_kicks.tracking import OrientationUnresolvedWarning

        warnings.filterwarnings("error", category=OrientationUnresolvedWarning)
    """


class RunValueCoverageWarning(UserWarning):
    """Some detected off-ball runs could not be valued.

    Emitted once per call with the count of runs whose runner was absent from the linked
    event frame (a tracking-visibility gap, common on broadcast providers). Those runs
    survive with a NaN value rather than a fabricated zero.

    Examples
    --------
    Treat incomplete coverage as fatal in a batch job::

        import warnings
        from silly_kicks.tracking import RunValueCoverageWarning

        warnings.filterwarnings("error", category=RunValueCoverageWarning)
    """
