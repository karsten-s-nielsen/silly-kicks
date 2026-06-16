"""Shared kloppy->SPADL coordinate system.

Used by BOTH the event gateway (``silly_kicks.spadl.kloppy``) and the tracking gateway
(``silly_kicks.tracking.kloppy``) so events and frames cannot drift in their vertical
orientation. The event gateway pinned this coordinate system from the start; the tracking
gateway did not, which inverted the tracking y-axis relative to the SPADL action y-axis for
every kloppy-based provider. Single-sourcing the coordinate system here fixes that and keeps
the two gateways aligned. See ADR-031.
"""

from __future__ import annotations

from kloppy.domain import (  # type: ignore[reportMissingImports]
    CoordinateSystem,
    Dimension,
    MetricPitchDimensions,
    Origin,
    PitchDimensions,
    Provider,
    VerticalOrientation,
)

from . import config as spadlconfig


class _SoccerActionCoordinateSystem(CoordinateSystem):
    def __init__(self, *, pitch_length: float, pitch_width: float) -> None:
        self._pitch_length = pitch_length
        self._pitch_width = pitch_width

    @property
    def provider(self) -> Provider:
        return "SoccerAction"  # type: ignore[reportReturnType]  # kloppy API varies by version

    @property
    def origin(self) -> Origin:
        return Origin.BOTTOM_LEFT

    @property
    def vertical_orientation(self) -> VerticalOrientation:
        return VerticalOrientation.BOTTOM_TO_TOP

    @property
    def pitch_length(self) -> float:  # type: ignore[override]
        return self._pitch_length

    @property
    def pitch_width(self) -> float:  # type: ignore[override]
        return self._pitch_width

    @property
    def pitch_dimensions(self) -> PitchDimensions:
        return MetricPitchDimensions(
            x_dim=Dimension(0, spadlconfig.field_length),
            y_dim=Dimension(0, spadlconfig.field_width),
            pitch_length=self._pitch_length,
            pitch_width=self._pitch_width,
            standardized=True,
        )


def socceraction_coordinate_system(metadata) -> _SoccerActionCoordinateSystem:
    """Build the canonical SPADL coordinate system from a kloppy dataset's metadata.

    Reads ``metadata.coordinate_system.pitch_length`` / ``.pitch_width`` -- identical to the
    inline construction the event gateway used before this extraction (ADR-031), so callers get
    byte-identical behaviour.

    Examples
    --------
    >>> from silly_kicks.spadl._kloppy_coordinates import socceraction_coordinate_system
    >>> cs = socceraction_coordinate_system(dataset.metadata)  # doctest: +SKIP
    >>> dataset.transform(to_coordinate_system=cs)  # doctest: +SKIP
    """
    src = metadata.coordinate_system
    return _SoccerActionCoordinateSystem(pitch_length=src.pitch_length, pitch_width=src.pitch_width)
