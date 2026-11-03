"""numba is a core runtime dependency (HFU spec §13.2, owner 2026-11-06). The optional extra hid a 12x slowdown in
HF training jobs twice: ``[train]`` never carried numba, only ``[test]`` and ``[calibration]`` did."""

from __future__ import annotations

import pathlib
import re

from scripts._toml import loads_toml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = loads_toml((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _names(reqs: list[str]) -> list[str]:
    return [m.group(1).lower() for r in reqs if (m := _NAME.match(r))]


def test_numba_is_a_core_dependency():
    assert "numba>=0.59.0" in PYPROJECT["project"]["dependencies"]


def test_numba_extra_is_an_empty_alias():
    """``pip install silly-kicks[numba]`` keeps resolving for existing pins (Hyrum)."""
    assert PYPROJECT["project"]["optional-dependencies"]["numba"] == []


def test_no_duplicate_numba_in_extras():
    extras = PYPROJECT["project"]["optional-dependencies"]
    for name in ("test", "calibration", "train"):
        assert "numba" not in _names(extras[name]), name


def test_lock_lists_numba_as_a_root_dependency():
    lock = loads_toml((ROOT / "uv.lock").read_text(encoding="utf-8"))
    root = next(p for p in lock["package"] if p["name"] == "silly-kicks")
    assert "numba" in {d["name"] for d in root.get("dependencies", [])}
