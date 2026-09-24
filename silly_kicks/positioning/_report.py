"""Conservation census for compute_positioning_gap (spec section 6; ADR-042)."""

from __future__ import annotations

from dataclasses import dataclass

from ._config import PositioningParams

# The closed set of per-frame outcomes. Every evaluated frame carries exactly one of these in its
# ``positioning_gap_source`` column, and the report conserves over them.
POSITIONING_GAP_SOURCE_VALUES = (
    "scored",
    "excluded_out_of_domain",
    "excluded_not_two_teams",
    "velocity_unavailable",
    "unresolved_geometry",
    "degenerate_no_movable",
)


@dataclass(frozen=True)
class PositioningReport:
    """Run-level audit for :func:`compute_positioning_gap`.

    Conserves exactly: ``n_frames_scored + sum(drop_reasons.values()) == n_frames_in`` over the
    frames the metric evaluated (after 1 fps down-sampling). Echoes the params actually used --
    registration without traceability is not registration.

    Examples
    --------
    >>> r = PositioningReport(params=PositioningParams.default(), n_frames_in=3, n_frames_scored=1,
    ...                       drop_reasons={"excluded_out_of_domain": 2})
    >>> r.conserves()
    True
    """

    params: PositioningParams
    n_frames_in: int
    n_frames_scored: int
    drop_reasons: dict

    def conserves(self) -> bool:
        """True iff every input frame is accounted for (scored + dropped == in; ADR-042).

        >>> PositioningReport(params=PositioningParams.default(), n_frames_in=2, n_frames_scored=1,
        ...                   drop_reasons={"velocity_unavailable": 1}).conserves()
        True
        """
        return self.n_frames_scored + sum(self.drop_reasons.values()) == self.n_frames_in
