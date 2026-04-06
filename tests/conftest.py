"""Configuration for pytest."""

import os
from collections.abc import Iterator

import pandera as pa

# pandera >=0.20 renamed SchemaModel to DataFrameModel; shim for compat
if not hasattr(pa, "SchemaModel") and hasattr(pa, "DataFrameModel"):
    pa.SchemaModel = pa.DataFrameModel  # type: ignore[attr-defined]

import pandas as pd
import pytest
from _pytest.config import Config
from pandera.typing import DataFrame
from silly_kicks.atomic.spadl import AtomicSPADLSchema
from silly_kicks.spadl import SPADLSchema


def pytest_configure(config: Config) -> None:
    """Pytest configuration hook."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")


@pytest.fixture(scope="session")
def sb_worldcup_data() -> Iterator[pd.HDFStore]:
    hdf_file = os.path.join(
        os.path.dirname(__file__), "datasets", "statsbomb", "spadl-WorldCup-2018.h5"
    )
    store = pd.HDFStore(hdf_file, mode="r")
    yield store
    store.close()


@pytest.fixture(scope="session")
def spadl_actions() -> DataFrame[SPADLSchema]:
    json_file = os.path.join(os.path.dirname(__file__), "datasets", "spadl", "spadl.json")
    return pd.read_json(json_file, orient="records")


@pytest.fixture(scope="session")
def atomic_spadl_actions() -> DataFrame[AtomicSPADLSchema]:
    json_file = os.path.join(os.path.dirname(__file__), "datasets", "spadl", "atomic_spadl.json")
    return pd.read_json(json_file, orient="records")
