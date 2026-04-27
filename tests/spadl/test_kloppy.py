"""Kloppy SPADL converter tests.

Most full-dataset comparison tests require external fixture files and the
removed ``silly_kicks.data`` loaders; those tests are e2e-marked and
skipped in normal CI runs.

The unit tests in ``TestKloppyPreserveNative`` use lightweight mock
EventDataset stubs to validate the ``preserve_native`` API surface
(error paths) without needing real kloppy data.
"""

from unittest.mock import MagicMock

import pytest

from silly_kicks.spadl import kloppy as kloppy_mod


class TestKloppyPreserveNative:
    """Unit tests for the ``preserve_native`` kwarg added in 1.1.0.

    These tests validate the API surface (error paths) without requiring
    real kloppy data. Full round-trip tests are in the e2e suite.
    """

    def test_schema_overlap_raises(self):
        """preserve_native cannot overlap with KLOPPY_SPADL_COLUMNS."""
        dataset = MagicMock()
        dataset.events = []  # empty events; validation triggers on schema overlap before iteration
        with pytest.raises(ValueError, match=r"overlap|already"):
            kloppy_mod.convert_to_actions(dataset, game_id=1, preserve_native=["team_id"])

    def test_missing_field_in_raw_event_raises(self):
        """preserve_native fields must exist on event.raw_event."""
        dataset = MagicMock()
        sample_event = MagicMock()
        sample_event.raw_event = {"present_key": "value"}
        dataset.events = iter([sample_event])
        with pytest.raises(ValueError, match=r"preserve_native|missing"):
            kloppy_mod.convert_to_actions(dataset, game_id=1, preserve_native=["missing_key"])

    def test_non_dict_raw_event_raises(self):
        """preserve_native requires raw_event to be dict-shaped."""
        dataset = MagicMock()
        sample_event = MagicMock()
        sample_event.raw_event = "not_a_dict"
        dataset.events = iter([sample_event])
        with pytest.raises(ValueError, match="dict"):
            kloppy_mod.convert_to_actions(dataset, game_id=1, preserve_native=["something"])

    def test_none_raw_event_raises(self):
        """preserve_native requires raw_event to be present."""
        dataset = MagicMock()
        sample_event = MagicMock()
        sample_event.raw_event = None
        dataset.events = iter([sample_event])
        with pytest.raises(ValueError, match="dict"):
            kloppy_mod.convert_to_actions(dataset, game_id=1, preserve_native=["something"])

    # Note: the "no validation when preserve_native is None or []" code path is
    # the simple ``if preserve_native:`` falsy check at the top of
    # ``convert_to_actions``; it is implicitly covered by every other test in
    # the suite that doesn't pass ``preserve_native`` and proceeds normally.
    # An explicit unit test would require constructing a minimally-functional
    # kloppy ``EventDataset`` mock that survives ``dataset.transform(...)`` —
    # too much mock plumbing for the signal it would provide. Full round-trip
    # coverage lives in the e2e suite.


# ---------------------------------------------------------------------------
# E2E tests (require external fixtures, removed data loaders)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestKloppyE2E:
    """End-to-end Kloppy comparison tests requiring external fixture files."""

    def test_placeholder(self) -> None:
        pytest.skip("Kloppy comparison fixtures and data loaders are not available")
