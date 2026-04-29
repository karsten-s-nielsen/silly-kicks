"""Shared pytest fixtures for spadl tests.

Module-scoped kloppy dataset loaders for the vendored Sportec + Metrica
fixtures under ``tests/datasets/kloppy/``. Reused across ``test_kloppy.py``,
``test_sportec.py``, and ``test_metrica.py`` for cross-path consistency tests.
"""

from pathlib import Path

import pytest

_KLOPPY_FIXTURES_DIR = Path(__file__).parent.parent / "datasets" / "kloppy"


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
