"""StatsBomb SPADL converter tests."""

import inspect

import pytest

from silly_kicks.spadl import statsbomb


# ---------------------------------------------------------------------------
# Tests that use inline data (no external fixtures required)
# ---------------------------------------------------------------------------


def test_statsbomb_no_inplace_fillna() -> None:
    """Bug #946: fillna must not use inplace=True (pandas 3.0 compat)."""
    source = inspect.getsource(statsbomb.convert_to_actions)
    assert "inplace=True" not in source, "inplace=True is deprecated in pandas 2.x"


# ---------------------------------------------------------------------------
# Tests below require StatsBomb event fixtures and are marked e2e.
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSpadlConvertor:
    """End-to-end SPADL converter tests that need StatsBomb fixture files.

    These tests require:
    - tests/datasets/statsbomb/raw/events/7584.json  (Japan vs Belgium)
    - tests/datasets/statsbomb/raw/events/7577.json  (Morocco game)
    - tests/datasets/statsbomb/raw/events/9912.json  (high-fidelity coords)
    - tests/datasets/statsbomb/raw/lineups/7584.json
    - silly_kicks.data.statsbomb.StatsBombLoader (removed)

    Skipped unless ``-m e2e`` is passed to pytest.
    """

    def test_placeholder(self) -> None:
        pytest.skip("StatsBomb fixture data and data loaders are not available")
