"""Every ``docs/examples`` walkthrough runs end to end on a committed fixture.

The examples are documentation and need a user's data directory, so nothing ran them. The Gradient Sports
walkthrough broke twice without notice: its ``coverage_metrics`` / ``boundary_metrics`` calls stopped matching their
keyword-only signatures, and it never passed the ``start_time`` (ADR-065) or ``event_time`` (ADR-133) the converter
requires. This gate lays a committed fixture out in the directory shape each walkthrough expects and runs its
``main()``. The population is DERIVED from the directory (ADR-056): a new walkthrough without a runner here fails.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "docs" / "examples"
_GS_FIXTURE = _ROOT / "tests" / "datasets" / "gradientsports" / "synthetic_match.json"


def _gradientsports_layout(tmp: Path) -> list[str]:
    """``Event Data`` / ``Metadata`` / ``Rosters`` for the synthetic match 99999 (teams 100 and 200, players 1-22)."""
    for sub in ("Event Data", "Metadata", "Rosters"):
        (tmp / sub).mkdir()
    shutil.copy(_GS_FIXTURE, tmp / "Event Data" / "99999.json")
    meta = [
        {
            "id": 99999,
            "homeTeam": {"id": 100},
            "awayTeam": {"id": 200},
            "homeTeamStartLeft": True,
            "homeTeamStartLeftExtraTime": True,
        }
    ]
    (tmp / "Metadata" / "99999.json").write_text(json.dumps(meta), encoding="utf-8")
    roster = [{"player": {"id": p}, "team": {"id": 100 if p <= 11 else 200}} for p in range(1, 23)]
    (tmp / "Rosters" / "99999.json").write_text(json.dumps(roster), encoding="utf-8")
    return [str(tmp), "--match-id", "99999"]


# Walkthrough file name -> builds its data directory in tmp and returns its argv.
_RUNNERS: dict[str, Callable[[Path], list[str]]] = {
    "gradientsports_wc2022_walkthrough.py": _gradientsports_layout,
}


def _walkthroughs() -> list[str]:
    return sorted(p.name for p in _EXAMPLES.glob("*_walkthrough.py"))


def test_every_walkthrough_has_a_runner():
    assert _walkthroughs(), "the population is empty: docs/examples moved or the glob is wrong"
    assert set(_walkthroughs()) == set(_RUNNERS), "add a runner (fixture layout) for each docs/examples walkthrough"


@pytest.mark.parametrize("name", _walkthroughs())
def test_walkthrough_runs(name, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(f"_walkthrough_{name[:-3]}", _EXAMPLES / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", [name, *_RUNNERS[name](tmp_path)])
    mod.main()
    out = capsys.readouterr().out
    # main() returned, so every step ran; the first and last progress lines prove it printed its way through.
    assert "Converted to" in out
    assert "VAEP scores label" in out
