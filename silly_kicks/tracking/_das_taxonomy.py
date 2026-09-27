"""DAS degradation taxonomy — the ONLY degradable exception + the ``das_source`` vocabulary (ADR-043).

A leaf module (no heavy deps) so the pack/engine/facade layers can all import it without a cycle. The
public names are re-exported from ``silly_kicks.tracking._das`` and ``silly_kicks.tracking`` unchanged.
"""

from __future__ import annotations

#: Closed vocabulary for the ``das_source`` provenance column emitted by ``add_das`` (ADR-043). Makes
#: "DAS could not be computed" distinguishable from "DAS is genuinely absent for this action".
DAS_SOURCE_COMPUTED = "computed"
#: The action resolved to no frame (no link pointer, or a NaN ``frame_id``).
DAS_SOURCE_UNLINKED = "unlinked"
#: The FRAMES, not the computation, are why DAS is absent -- per-action (the linked frame carries no
#: DAS) or whole-call (the frame source structurally cannot carry velocity).
DAS_SOURCE_UNSCOREABLE_FRAME = "unscoreable_frame"
#: The linked frame carries DAS, but none of its teams matched the acting team.
DAS_SOURCE_TEAM_UNRESOLVED = "team_unresolved"
#: The DAS COMPUTATION degraded (:class:`DasUnscoreableError`) on frames that could in principle have
#: carried DAS -- a dead-ball window, NaN coordinates.
DAS_SOURCE_UNSCOREABLE_CALL = "unscoreable_call"

DAS_SOURCE_VALUES = (
    DAS_SOURCE_COMPUTED,
    DAS_SOURCE_UNLINKED,
    DAS_SOURCE_UNSCOREABLE_FRAME,
    DAS_SOURCE_TEAM_UNRESOLVED,
    DAS_SOURCE_UNSCOREABLE_CALL,
)


class DasUnscoreableError(ValueError):
    """DAS is genuinely undefined for these frames -- callers degrade to NaN + provenance.

    This is the ONLY exception ``add_das`` / ``das_at_action`` / ``das_xfns`` degrade on (ADR-043). It
    subclasses ``ValueError`` so any consumer that predates the taxonomy and catches the broad
    ``ValueError`` keeps working unchanged. ``das_source`` carries the provenance token the degrading
    caller stamps, so the reason survives the raise rather than being re-derived from the message text.
    """

    def __init__(self, *args, das_source: str = DAS_SOURCE_UNSCOREABLE_CALL) -> None:
        if das_source not in DAS_SOURCE_VALUES:
            raise ValueError(f"das_source must be one of {DAS_SOURCE_VALUES}, got {das_source!r}")
        super().__init__(*args)
        self.das_source = das_source
