"""C6/C1: Stage-2 must refuse a dirty or unprovenanced carrier selection (ADR-060).

The guard already ships in ``scripts/calibrate_tracking_defaults.py::_load_carrier_selection``;
this test LOCKS that contract so a future edit cannot silently drop it.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("ruthless")  # the CLI module imports ruthless at top; skip if the extra is absent

from scripts.calibrate_tracking_defaults import _load_carrier_selection


def _write(tmp_path, payload):
    p = tmp_path / "carrier_selected.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_refuses_dirty_carrier(tmp_path):
    path = _write(tmp_path, {"beta": 0.0, "gamma": 0.25, "run_commit": "abc123", "run_tree_dirty": True})
    with pytest.raises(ValueError, match="dirty"):
        _load_carrier_selection(path)


def test_refuses_unprovenanced_carrier(tmp_path):
    path = _write(tmp_path, {"beta": 0.0, "gamma": 0.25})  # no run_commit
    with pytest.raises(ValueError, match="provenance"):
        _load_carrier_selection(path)


def test_refuses_missing_keys(tmp_path):
    path = _write(tmp_path, {"run_commit": "abc123", "run_tree_dirty": False})  # no beta/gamma
    with pytest.raises(ValueError, match="missing keys"):
        _load_carrier_selection(path)


def test_accepts_clean_provenanced_and_sources_tolerance_from_default(tmp_path):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    path = _write(tmp_path, {"beta": 0.5, "gamma": 1.0, "run_commit": "abc123", "run_tree_dirty": False})
    out = _load_carrier_selection(path)
    assert out["beta"] == 0.5
    assert out["gamma"] == 1.0
    # tolerance_m is the HELD constant, NOT read from the file (ADR-060)
    assert out["tolerance_m"] == DEFAULT_CARRIER_PARAMS["tolerance_m"]
