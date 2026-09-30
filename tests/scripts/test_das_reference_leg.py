"""Offline unit tests for scripts/_das_reference_leg.py (no accessible-space, no network).

The accessible-space-calling paths (``reference_leg_arrays`` / ``__main__``) run only on the DGX under a
pandas-2 interpreter and are covered by the owner corpus smoke (spec 8); here we test the sk-free pure
helpers, the fail-loud env checker, the collision-free frame key, and the lazy-import contract.
"""

from __future__ import annotations

import builtins
import importlib
import warnings

import pandas as pd
import pytest

from scripts import _das_reference_leg as R


def test_module_imports_without_accessible_space(monkeypatch):
    # The lazy-import contract: importing the module and using its pure helpers must NOT require
    # accessible_space (absent in CI). Simulate absence and re-import.
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "accessible_space" or name.startswith("accessible_space."):
            raise ImportError("accessible_space blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    mod = importlib.reload(R)
    assert callable(mod._reference_lib_frames)
    assert mod._REFERENCE_COMMON["frame_col"] == "frame_id"
    importlib.reload(R)  # restore with the real import


def test_reference_common_matches_golden_generator():
    # NEW gate (no such test existed): the recipe must equal the frozen golden generator's _COMMON,
    # modulo attacking_direction_col (the generator uses "dir"; the parity pins the shared dir column).
    gen = importlib.import_module("tests.tracking._fixtures.das_golden._generate")
    common = dict(R._REFERENCE_COMMON)
    expected = dict(gen._COMMON)
    common.pop("attacking_direction_col")
    expected.pop("attacking_direction_col", None)
    assert common == expected


def test_unique_frame_col_distinguishes_reused_frame_id_across_periods():
    lib = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "period_id": [1, 1, 2, 2],
            "frame_id": [5, 5, 5, 5],  # reused across periods
            "player_id": ["a", "b", "a", "b"],
        }
    )
    out = R._add_unique_frame_col(lib)
    # rows within one (game,period,frame) share a code; the two periods differ.
    assert out.loc[0, "_uframe"] == out.loc[1, "_uframe"]
    assert out.loc[2, "_uframe"] == out.loc[3, "_uframe"]
    assert out.loc[0, "_uframe"] != out.loc[2, "_uframe"]


@pytest.mark.parametrize(
    "pv,av,ok",
    [("2.3.3", "2.0.15", True), ("3.0.6", "2.0.15", False), ("2.3.3", "2.1.0", False)],
)
def test_check_reference_env(pv, av, ok):
    if ok:
        R._check_reference_env(pv, av)
    else:
        with pytest.raises(RuntimeError):
            R._check_reference_env(pv, av)


def test_offside_warning_is_an_error_under_the_filter():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Offside not properly detectable")
        with pytest.raises(UserWarning):
            warnings.warn("Offside not properly detectable, maybe too few defenders. Ignoring offside.", stacklevel=2)
