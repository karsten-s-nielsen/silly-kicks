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


@pytest.fixture
def fitted_xt():
    """Shared fitted ExpectedThreat fixture for all test subdirectories."""
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@pytest.fixture
def shadow_pre_index_oracle(monkeypatch):
    """Run the pre-index ORACLE beside every ``_pre_index_frames`` call of a test (HFU fit-units spec 4.3).

    It replaces ``silly_kicks.tracking._ball_carrier._pre_index_frames`` with a wrapper that calls the oracle on a
    copy of the frames and the real function on the frames, and asserts that the two results are equal (or, where
    the oracle raises, that the real function raises the same type). So every frame table that a test builds
    inline becomes an oracle input. It lives here because test modules in ``tests/tracking`` and in
    ``tests/scripts`` use it. Returns the call counter (``{"n": int}``).
    """
    from silly_kicks.tracking import _ball_carrier
    from tests.tracking._pre_index_oracle import assert_pre_index_equal, pre_index_frames_oracle

    real = _ball_carrier._pre_index_frames
    calls = {"n": 0}

    def shadow(frames):
        calls["n"] += 1
        try:
            want = pre_index_frames_oracle(frames.copy(deep=True))
        except Exception as exc:
            with pytest.raises(type(exc)):
                real(frames)
            raise
        got = real(frames)
        assert_pre_index_equal(want, got)
        return got

    monkeypatch.setattr(_ball_carrier, "_pre_index_frames", shadow)
    return calls
