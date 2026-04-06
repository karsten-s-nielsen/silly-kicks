"""Kloppy-vs-socceraction SPADL converter comparison tests.

Every test and fixture in this module requires external fixture files and
the removed ``silly_kicks.data`` loaders (OptaLoader, StatsBombLoader,
PublicWyscoutLoader, WyscoutLoader).  The fixture directories
(``tests/datasets/statsbomb/``, ``tests/datasets/opta/``,
``tests/datasets/wyscout_api/``, ``tests/datasets/wyscout_public/``)
are not committed to the repository.

The entire module is marked ``e2e`` and skipped in normal CI runs.
"""

import pytest

pytestmark = pytest.mark.e2e


def test_placeholder() -> None:
    pytest.skip("Kloppy comparison fixtures and data loaders are not available")
