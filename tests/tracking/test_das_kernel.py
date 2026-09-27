"""Task 5 gates: numba kernel -- float32 rejection, lazy import, serial==parallel, thread restore."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from silly_kicks.tracking import _das_numba  # noqa: E402  (imported after the numba importorskip)
from silly_kicks.tracking._das_engine import compute_das  # noqa: E402
from silly_kicks.tracking._das_pack import pack_frames  # noqa: E402
from silly_kicks.tracking._das_params import DAS_PARAMS  # noqa: E402
from tests.tracking._das_helpers import golden_frames, kernel_args  # noqa: E402


def test_float32_arrays_rejected_by_kernel():
    with pytest.raises(TypeError, match="float64"):
        _das_numba.das_frames_serial(*kernel_args(dtype=np.float32))


def test_float64_arrays_accepted_by_kernel():
    _das_numba.das_frames_serial(*kernel_args(dtype=np.float64))  # no raise


def test_engine_modules_do_not_eagerly_import_the_numba_kernel():
    code = (
        "import sys, silly_kicks.tracking._das_engine, silly_kicks.tracking._das_pack; "
        "assert 'silly_kicks.tracking._das_numba' not in sys.modules, "
        "'the numba kernel must be imported lazily, not at module load'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)  # noqa: S603 -- fixed argv, sys.executable


def test_serial_and_parallel_kernels_byte_identical():
    packed = pack_frames(golden_frames("S10"), attacking_direction_col="dir")
    serial = _das_numba.compute_das_numba(packed, DAS_PARAMS, n_threads=None)
    parallel = _das_numba.compute_das_numba(packed, DAS_PARAMS, n_threads=4)
    assert np.array_equal(serial.team_das, parallel.team_das, equal_nan=True)
    assert np.array_equal(serial.team_as, parallel.team_as, equal_nan=True)
    assert np.array_equal(serial.player_das, parallel.player_das, equal_nan=True)
    assert np.array_equal(serial.player_as, parallel.player_as, equal_nan=True)


def test_n_threads_restores_numba_thread_count():
    before = numba.get_num_threads()
    packed = pack_frames(golden_frames("S01"), attacking_direction_col="dir")
    _das_numba.compute_das_numba(packed, DAS_PARAMS, n_threads=2)
    assert numba.get_num_threads() == before


def test_force_numpy_env_selects_numpy(monkeypatch):
    monkeypatch.setenv("SILLY_KICKS_DAS_FORCE_NUMPY", "1")
    from silly_kicks.tracking import _das_engine

    assert _das_engine._numba_available() is False
    packed = pack_frames(golden_frames("S01"), attacking_direction_col="dir")
    res = compute_das(packed, DAS_PARAMS, engine="auto")  # must not raise, uses numpy
    assert np.isfinite(res.team_das).any()
