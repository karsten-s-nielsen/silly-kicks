"""Configuration for pytest."""

import os
from collections.abc import Iterator

import pandas as pd
import pytest
from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    """Pytest configuration hook."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")


@pytest.fixture(scope="session")
def sb_worldcup_data() -> Iterator[pd.HDFStore]:
    hdf_file = os.path.join(os.path.dirname(__file__), "datasets", "statsbomb", "spadl-WorldCup-2018.h5")
    if not os.path.exists(hdf_file):
        pytest.fail(
            f"WorldCup-2018 SPADL fixture not found at {hdf_file!r}. "
            f"This fixture is committed to the repo as of silly-kicks 1.9.0. "
            f"If absent, regenerate via: python scripts/build_worldcup_fixture.py "
            f"(or check for accidental .gitignore exclusion)."
        )
    store = pd.HDFStore(hdf_file, mode="r")
    yield store
    store.close()


@pytest.fixture(scope="session")
def spadl_actions() -> pd.DataFrame:
    json_file = os.path.join(os.path.dirname(__file__), "datasets", "spadl", "spadl.json")
    return pd.read_json(json_file, orient="records")


@pytest.fixture(scope="session")
def atomic_spadl_actions() -> pd.DataFrame:
    json_file = os.path.join(os.path.dirname(__file__), "datasets", "spadl", "atomic_spadl.json")
    return pd.read_json(json_file, orient="records")
