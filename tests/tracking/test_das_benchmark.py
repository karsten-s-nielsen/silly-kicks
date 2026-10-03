"""pytest-benchmark ms/frame per DAS engine (das-native plan Task 5; spec section 8).

Trend data only -- no timing assertion. The corpus figures that gate the spec section 4.2 targets are
docs/research/das_native_parity/performance.json (test_das_parity_artifact.py).
"""

import pandas as pd
import pytest

from silly_kicks.tracking._das_engine import compute_das
from silly_kicks.tracking._das_pack import pack_frames
from silly_kicks.tracking._das_params import DAS_PARAMS
from tests.tracking._das_helpers import single_frame

_N_FRAMES = 200


@pytest.fixture(scope="module")
def packed():
    frames = pd.concat([single_frame(frame=f, seed=f) for f in range(_N_FRAMES)], ignore_index=True)
    return pack_frames(frames, attacking_direction_col="dir")


def test_numpy_engine(benchmark, packed):
    benchmark(compute_das, packed, DAS_PARAMS, engine="numpy")


def test_numba_serial(benchmark, packed):
    pytest.importorskip("numba")
    compute_das(packed, DAS_PARAMS, engine="numba")  # JIT warm-up outside the timed region
    benchmark(compute_das, packed, DAS_PARAMS, engine="numba")


@pytest.mark.parametrize("n_threads", [2, 4])
def test_numba_prange(benchmark, packed, n_threads):
    pytest.importorskip("numba")
    compute_das(packed, DAS_PARAMS, engine="numba", n_threads=n_threads)
    benchmark(compute_das, packed, DAS_PARAMS, engine="numba", n_threads=n_threads)
