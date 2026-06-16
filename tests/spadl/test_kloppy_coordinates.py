"""Unit guards for the shared kloppy->SPADL coordinate system (ADR-031, PR-S94 Task 1.2 / rev-6 D).

These pin the extracted ``_kloppy_coordinates`` module so a future edit can't silently change the
canonical orientation or the metadata-sourcing of the pitch dimensions (the byte-equivalence concern):
the helper must read ``pitch_length``/``pitch_width`` FROM the dataset metadata (non-default values
flow through), and the coordinate system must stay BOTTOM_LEFT / BOTTOM_TO_TOP. The end-to-end event
byte-equivalence is additionally covered by ``tests/spadl/test_kloppy.py`` (25 tests, unchanged).
"""

from types import SimpleNamespace

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl._kloppy_coordinates import (
    _SoccerActionCoordinateSystem,
    socceraction_coordinate_system,
)


def test_helper_sources_pitch_dims_from_metadata_non_default():
    """Non-default pitch dims must flow through from metadata (not hardcoded) -- non-vacuous (rev-6 D)."""
    meta = SimpleNamespace(coordinate_system=SimpleNamespace(pitch_length=105.3, pitch_width=68.5))
    cs = socceraction_coordinate_system(meta)
    assert cs.pitch_length == 105.3
    assert cs.pitch_width == 68.5


def test_canonical_orientation_is_bottom_left_bottom_to_top():
    """The pinned SPADL convention -- a regression here is the whole bug class (ADR-031)."""
    from kloppy.domain import Origin, VerticalOrientation  # type: ignore[reportMissingImports]

    cs = _SoccerActionCoordinateSystem(pitch_length=105.0, pitch_width=68.0)
    assert cs.origin == Origin.BOTTOM_LEFT
    assert cs.vertical_orientation == VerticalOrientation.BOTTOM_TO_TOP


def test_pitch_dimensions_use_spadl_field_size():
    """pitch_dimensions are the standardized SPADL field (105x68 from spadlconfig)."""
    cs = _SoccerActionCoordinateSystem(pitch_length=105.0, pitch_width=68.0)
    dims = cs.pitch_dimensions
    assert dims.x_dim.max == spadlconfig.field_length
    assert dims.y_dim.max == spadlconfig.field_width
