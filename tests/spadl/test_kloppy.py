"""Kloppy SPADL converter tests.

Most full-dataset comparison tests require external fixture files and the
removed ``silly_kicks.data`` loaders; those tests are e2e-marked and
skipped in normal CI runs.

The unit tests in ``TestKloppyPreserveNative`` use lightweight mock
EventDataset stubs to validate the ``preserve_native`` API surface
(error paths) without needing real kloppy data.
"""

import warnings as warnings_mod
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import kloppy as kloppy_mod
from silly_kicks.spadl.schema import KLOPPY_SPADL_COLUMNS


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
# Real-fixture tests (vendored kloppy fixtures under tests/datasets/kloppy/)
# ---------------------------------------------------------------------------

_KLOPPY_FIXTURES_DIR = Path(__file__).parent.parent / "datasets" / "kloppy"


class TestKloppyCoordinateSystemFix:
    """Regression test for the latent _SoccerActionCoordinateSystem bug.

    Pre-fix, ``convert_to_actions`` raised ``TypeError: _SoccerActionCoordinateSystem()
    takes no arguments`` on line 168 the moment a real EventDataset reached
    ``dataset.transform(...)``. The pre-existing mock-based tests never exercised
    this path. This test exists to ensure the bug stays fixed.
    """

    def test_real_dataset_transform_does_not_typeerror(self):
        """convert_to_actions on a real Sportec dataset must not raise TypeError."""
        from kloppy import sportec
        from kloppy.domain import Provider
        from packaging import version

        ds = sportec.load_event(
            event_data=str(_KLOPPY_FIXTURES_DIR / "sportec_events.xml"),
            meta_data=str(_KLOPPY_FIXTURES_DIR / "sportec_meta.xml"),
        )

        # Whitelist Sportec for this test only (the whitelist add lands in a later task);
        # this isolates the bug-fix coverage from the whitelist add.
        original = dict(kloppy_mod._SUPPORTED_PROVIDERS)
        kloppy_mod._SUPPORTED_PROVIDERS[Provider.SPORTEC] = version.parse("3.15.0")
        try:
            # Must not raise TypeError. The conversion may emit warnings; that's fine.
            actions, _report = kloppy_mod.convert_to_actions(ds, game_id="bug_fix_smoke")
        finally:
            kloppy_mod._SUPPORTED_PROVIDERS.clear()
            kloppy_mod._SUPPORTED_PROVIDERS.update(original)

        assert isinstance(actions, pd.DataFrame)
        assert len(actions) > 0


@pytest.fixture(scope="module")
def sportec_dataset():
    """Module-scoped: parse the Sportec fixture once per test module."""
    from kloppy import sportec

    return sportec.load_event(
        event_data=str(_KLOPPY_FIXTURES_DIR / "sportec_events.xml"),
        meta_data=str(_KLOPPY_FIXTURES_DIR / "sportec_meta.xml"),
    )


@pytest.fixture(scope="module")
def metrica_dataset():
    """Module-scoped: parse the Metrica fixture once per test module."""
    from kloppy import metrica

    return metrica.load_event(
        event_data=str(_KLOPPY_FIXTURES_DIR / "metrica_events.json"),
        meta_data=str(_KLOPPY_FIXTURES_DIR / "epts_metrica_metadata.xml"),
    )


class TestKloppySportec:
    """End-to-end conversion tests for the Sportec (IDSSE) provider."""

    def test_convert_to_actions_basic(self, sportec_dataset):
        """convert_to_actions returns (DataFrame, ConversionReport) with the canonical
        SPADL schema and at least one row from the Sportec fixture (29 events)."""
        actions, report = kloppy_mod.convert_to_actions(sportec_dataset, game_id="sportec_smoke")

        assert isinstance(actions, pd.DataFrame)
        # Schema check: every canonical column present, in declared order.
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())
        # Dtype check per schema.
        for col, expected_dtype in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected_dtype, f"{col}: got {actions[col].dtype}, want {expected_dtype}"
        # Non-empty: the Sportec fixture has 29 events; some are excluded (CARD, GENERIC,
        # BALL_OUT, RECOVERY) so action count is < 29 but > 0.
        assert len(actions) > 0
        # Report shape.
        assert report.provider == "Kloppy"
        assert report.total_events > 0
        assert report.total_actions == len(actions)

    def test_no_unrecognized_event_types(self, sportec_dataset):
        """All Sportec event types are covered by either _MAPPED_EVENT_TYPES or _EXCLUDED_EVENT_TYPES.

        Catches future kloppy versions that introduce new event types we don't handle.
        """
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _, report = kloppy_mod.convert_to_actions(sportec_dataset, game_id="sportec_unrec")

        assert report.unrecognized_counts == {}
        unrec_warnings = [str(w.message) for w in caught if "unrecognized event types" in str(w.message)]
        assert unrec_warnings == [], f"Got unrecognized-event warnings: {unrec_warnings}"

    def test_action_id_unique_and_zero_indexed(self, sportec_dataset):
        """action_id must be range(len(actions)) per game_id."""
        actions, _ = kloppy_mod.convert_to_actions(sportec_dataset, game_id="sportec_aid")
        assert actions["action_id"].tolist() == list(range(len(actions)))

    def test_preserve_native_with_dict_raw_event(self, sportec_dataset):
        """preserve_native surfaces a raw_event field as an extra column."""
        first_event = next(iter(sportec_dataset.events))
        raw_keys = list(first_event.raw_event.keys())
        # Pick a key that doesn't collide with KLOPPY_SPADL_COLUMNS.
        candidate = next((k for k in raw_keys if k not in KLOPPY_SPADL_COLUMNS), None)
        assert candidate is not None, f"Sportec raw_event has no non-schema-overlapping key. Keys: {raw_keys}"

        actions, _ = kloppy_mod.convert_to_actions(sportec_dataset, game_id="sportec_pn", preserve_native=[candidate])
        assert candidate in actions.columns
        # At least the first row should have a non-NaN value for the preserved column
        # (synthetic dribble rows get NaN; the very first action originates from a real event).
        assert actions[candidate].notna().any()

    def test_input_dataset_not_mutated(self, sportec_dataset):
        """convert_to_actions must not mutate the input dataset's metadata or event count."""
        before_provider = sportec_dataset.metadata.provider
        before_coord_sys_id = id(sportec_dataset.metadata.coordinate_system)
        before_event_count = len(sportec_dataset.events)

        _, _ = kloppy_mod.convert_to_actions(sportec_dataset, game_id="sportec_mut")

        assert sportec_dataset.metadata.provider == before_provider
        assert id(sportec_dataset.metadata.coordinate_system) == before_coord_sys_id
        assert len(sportec_dataset.events) == before_event_count


class TestKloppyMetrica:
    """End-to-end conversion tests for the Metrica Sports provider."""

    def test_convert_to_actions_basic(self, metrica_dataset):
        """convert_to_actions returns (DataFrame, ConversionReport) with the canonical
        SPADL schema for the Metrica fixture (3594 events)."""
        actions, report = kloppy_mod.convert_to_actions(metrica_dataset, game_id="metrica_smoke")

        assert isinstance(actions, pd.DataFrame)
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())
        for col, expected_dtype in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected_dtype, f"{col}: got {actions[col].dtype}, want {expected_dtype}"
        assert len(actions) > 0
        assert report.provider == "Kloppy"
        assert report.total_events > 0
        assert report.total_actions == len(actions)

    def test_no_unrecognized_event_types(self, metrica_dataset):
        """All Metrica event types are covered by either _MAPPED_EVENT_TYPES or _EXCLUDED_EVENT_TYPES."""
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _, report = kloppy_mod.convert_to_actions(metrica_dataset, game_id="metrica_unrec")

        assert report.unrecognized_counts == {}
        unrec_warnings = [str(w.message) for w in caught if "unrecognized event types" in str(w.message)]
        assert unrec_warnings == [], f"Got unrecognized-event warnings: {unrec_warnings}"

    def test_action_id_unique_and_zero_indexed(self, metrica_dataset):
        """action_id must be range(len(actions)) per game_id."""
        actions, _ = kloppy_mod.convert_to_actions(metrica_dataset, game_id="metrica_aid")
        assert actions["action_id"].tolist() == list(range(len(actions)))

    def test_preserve_native_with_dict_raw_event(self, metrica_dataset):
        """preserve_native surfaces a raw_event field as an extra column."""
        first_event = next(iter(metrica_dataset.events))
        raw_keys = list(first_event.raw_event.keys())
        candidate = next((k for k in raw_keys if k not in KLOPPY_SPADL_COLUMNS), None)
        assert candidate is not None, f"Metrica raw_event has no non-schema-overlapping key. Keys: {raw_keys}"

        actions, _ = kloppy_mod.convert_to_actions(metrica_dataset, game_id="metrica_pn", preserve_native=[candidate])
        assert candidate in actions.columns
        assert actions[candidate].notna().any()

    def test_input_dataset_not_mutated(self, metrica_dataset):
        """convert_to_actions must not mutate the input dataset's metadata or event count."""
        before_provider = metrica_dataset.metadata.provider
        before_coord_sys_id = id(metrica_dataset.metadata.coordinate_system)
        before_event_count = len(metrica_dataset.events)

        _, _ = kloppy_mod.convert_to_actions(metrica_dataset, game_id="metrica_mut")

        assert metrica_dataset.metadata.provider == before_provider
        assert id(metrica_dataset.metadata.coordinate_system) == before_coord_sys_id
        assert len(metrica_dataset.events) == before_event_count


class TestKloppyCoordinateClamping:
    """Output coords must be clamped to [0, field_length] x [0, field_width]
    (105 x 68 m), matching the convention established by StatsBomb / Wyscout / Opta
    converters."""

    @pytest.mark.parametrize("fixture_name", ["sportec_dataset", "metrica_dataset"])
    def test_clamps_to_pitch_bounds(self, request, fixture_name):
        """All start_x, start_y, end_x, end_y values must lie in the SPADL frame."""
        dataset = request.getfixturevalue(fixture_name)
        actions, _ = kloppy_mod.convert_to_actions(dataset, game_id="clamp_test")

        L = spadlconfig.field_length  # 105.0
        W = spadlconfig.field_width  # 68.0

        for col, upper in [("start_x", L), ("end_x", L), ("start_y", W), ("end_y", W)]:
            col_min = actions[col].min()
            col_max = actions[col].max()
            assert col_min >= 0, f"{col}.min()={col_min} (must be >= 0)"
            assert col_max <= upper, f"{col}.max()={col_max} (must be <= {upper})"


class TestKloppyConversionReport:
    """ConversionReport shape and provider-whitelist warning behaviour."""

    @pytest.mark.parametrize(
        "provider_name,fixture_name",
        [
            ("sportec", "sportec_dataset"),
            ("metrica", "metrica_dataset"),
        ],
    )
    def test_no_unrecognized_provider_warning(self, request, provider_name, fixture_name):
        """A whitelisted provider must NOT trigger the 'not yet supported' warning."""
        dataset = request.getfixturevalue(fixture_name)
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _, _ = kloppy_mod.convert_to_actions(dataset, game_id=f"{provider_name}_warn")

        provider_warnings = [str(w.message) for w in caught if "not yet supported" in str(w.message)]
        assert provider_warnings == [], f"{provider_name} should be whitelisted; got warning(s): {provider_warnings}"

    @pytest.mark.parametrize("fixture_name", ["sportec_dataset", "metrica_dataset"])
    def test_report_count_dicts_disjoint(self, request, fixture_name):
        """ConversionReport mapped/excluded/unrecognized count dicts must be pairwise disjoint."""
        dataset = request.getfixturevalue(fixture_name)
        _, report = kloppy_mod.convert_to_actions(dataset, game_id="report_disjoint")

        mapped = set(report.mapped_counts.keys())
        excluded = set(report.excluded_counts.keys())
        unrecognized = set(report.unrecognized_counts.keys())
        assert mapped & excluded == set()
        assert mapped & unrecognized == set()
        assert excluded & unrecognized == set()


# ---------------------------------------------------------------------------
# E2E tests (require external fixtures, removed data loaders)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestKloppyE2E:
    """End-to-end Kloppy comparison tests requiring external fixture files."""

    def test_placeholder(self) -> None:
        pytest.skip("Kloppy comparison fixtures and data loaders are not available")
