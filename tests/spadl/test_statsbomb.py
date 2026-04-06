"""StatsBomb SPADL converter tests.

Every test in this module requires StatsBomb event fixtures loaded via
the removed ``silly_kicks.data.statsbomb.StatsBombLoader``.  The fixture
files (``tests/datasets/statsbomb/raw/events/*.json``) are not committed
to the repository.

The entire module is marked ``e2e`` and skipped in normal CI runs.
"""

import pytest

pytestmark = pytest.mark.e2e


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
